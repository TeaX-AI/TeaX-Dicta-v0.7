import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gguf
from transformers import AutoTokenizer


class CausalMix(nn.Module):
    def __init__(self, d, kernel_size=5):
        super().__init__()
        self.conv = nn.Conv1d(d, d, kernel_size, groups=d, padding=kernel_size - 1)
        self.gate = nn.Linear(d, d)
        self.kernel_size = kernel_size

    def forward(self, x):
        y = x.transpose(1, 2)
        y = self.conv(y)
        y = y[:, :, :x.size(1)]
        y = y.transpose(1, 2)
        return x + torch.sigmoid(self.gate(x)) * y


class SharedTrunk(nn.Module):
    def __init__(self, d, n_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([CausalMix(d) for _ in range(n_blocks)])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


class HierarchicalDictLayer(nn.Module):
    def __init__(self, d, n_experts, n_parents=None, children_per_parent=2,
                 hidden_mult=3, top_k=2, trunk_blocks=2):
        super().__init__()
        self.d = d
        self.n_experts = n_experts
        self.n_parents = n_parents
        self.children_per_parent = children_per_parent
        self.top_k = top_k
        hidden = d * hidden_mult

        self.router = nn.Linear(d, n_experts)
        self.W1 = nn.Parameter(torch.randn(n_experts, d, hidden) * 0.02)
        self.b1 = nn.Parameter(torch.zeros(n_experts, hidden))
        self.W2 = nn.Parameter(torch.randn(n_experts, hidden, d) * 0.02)
        self.b2 = nn.Parameter(torch.zeros(n_experts, d))
        self.trunk = SharedTrunk(d, trunk_blocks)

        if n_parents is not None:
            child_to_parent = torch.arange(n_experts) // children_per_parent
            self.register_buffer("child_to_parent", child_to_parent)
        else:
            self.child_to_parent = None

    def forward(self, x, parent_probs=None):
        B, L, D = x.shape
        N = B * L
        x_flat = x.reshape(N, D)

        logits = self.router(x_flat)

        if parent_probs is not None and self.child_to_parent is not None:
            pp = parent_probs.reshape(N, -1)
            if self.training:
                gating = pp
            else:
                gating = F.one_hot(pp.argmax(dim=-1), num_classes=self.n_parents).float()
            parent_prob_per_expert = gating[:, self.child_to_parent]
            logits = logits + torch.log(parent_prob_per_expert + 1e-6)

        probs = F.softmax(logits, dim=-1)
        topk_weights, topk_indices = torch.topk(probs, k=self.top_k, dim=-1)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)

        flat_experts = topk_indices.reshape(-1)
        flat_weights = topk_weights.reshape(-1)
        token_ids = torch.arange(N, device=x.device).repeat_interleave(self.top_k)
        total = N * self.top_k

        sorted_experts, sort_order = torch.sort(flat_experts)
        sorted_token_ids = token_ids[sort_order]
        sorted_weights = flat_weights[sort_order]
        sorted_x = x_flat[sorted_token_ids]

        expert_counts = torch.bincount(sorted_experts, minlength=self.n_experts)
        counts_list = expert_counts.tolist()

        offsets = [0] * self.n_experts
        for i in range(1, self.n_experts):
            offsets[i] = offsets[i - 1] + counts_list[i - 1]
        offsets_t = torch.tensor(offsets, device=x.device, dtype=torch.long)

        arange_t = torch.arange(total, device=x.device)
        within_slot = arange_t - offsets_t[sorted_experts]

        max_count = max(max(counts_list), 1)
        avg = max(total // max(self.n_experts, 1), 1)
        capacity = min(max_count, avg * 4 + 1)

        valid = within_slot < capacity
        valid_pos = valid.nonzero(as_tuple=True)[0]

        valid_experts = sorted_experts[valid_pos]
        valid_slots = within_slot[valid_pos]
        valid_weights = sorted_weights[valid_pos]
        valid_token_ids = sorted_token_ids[valid_pos]
        valid_x = sorted_x[valid_pos]

        padded_x = torch.zeros(self.n_experts, capacity, D, device=x.device, dtype=x.dtype)
        padded_weights = torch.zeros(self.n_experts, capacity, device=x.device, dtype=x.dtype)
        padded_token_ids = torch.zeros(self.n_experts, capacity, dtype=torch.long, device=x.device)
        padded_valid = torch.zeros(self.n_experts, capacity, dtype=torch.bool, device=x.device)

        flat_idx = valid_experts * capacity + valid_slots
        padded_x.view(-1, D).index_copy_(0, flat_idx, valid_x)
        padded_weights.view(-1).index_copy_(0, flat_idx, valid_weights)
        padded_token_ids.view(-1).index_copy_(0, flat_idx, valid_token_ids)
        padded_valid.view(-1).index_fill_(0, flat_idx, True)

        h = torch.bmm(padded_x, self.W1) + self.b1.unsqueeze(1)
        h = F.gelu(h)
        out = torch.bmm(h, self.W2) + self.b2.unsqueeze(1)

        out = out * padded_weights.unsqueeze(-1)
        out = out * padded_valid.unsqueeze(-1).to(out.dtype)

        out_flat = out.reshape(-1, D)
        token_ids_flat = padded_token_ids.reshape(-1)
        valid_flat = padded_valid.reshape(-1)

        out_valid = out_flat[valid_flat]
        token_ids_valid = token_ids_flat[valid_flat]

        result = torch.zeros(N, D, device=x.device, dtype=x.dtype)
        result.index_add_(0, token_ids_valid, out_valid)
        result = result.reshape(B, L, D)

        x = self.trunk(x + result)
        probs = probs.reshape(B, L, self.n_experts)
        return x, probs


class TeaXDictaV07(nn.Module):
    def __init__(self, vocab_size, d=640, layer_experts=(12, 24, 48),
                 children_per_parent=2, hidden_mult=3, top_k=2,
                 max_seq_len=192, trunk_blocks=2):
        super().__init__()
        self.layer_experts = list(layer_experts)
        self.children_per_parent = children_per_parent

        self.token_emb = nn.Embedding(vocab_size, d)
        self.pos_emb = nn.Embedding(max_seq_len, d)

        layers = []
        prev_n = None
        for n in self.layer_experts:
            layers.append(HierarchicalDictLayer(
                d, n,
                n_parents=prev_n,
                children_per_parent=children_per_parent,
                hidden_mult=hidden_mult,
                top_k=top_k,
                trunk_blocks=trunk_blocks,
            ))
            prev_n = n
        self.layers = nn.ModuleList(layers)

        self.norm = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, input_ids):
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(pos)

        parent_probs = None
        for layer in self.layers:
            x, probs = layer(x, parent_probs=parent_probs)
            parent_probs = probs

        logits = self.lm_head(self.norm(x))
        return logits

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=256, temperature=0.7,
                 top_k=50, top_p=0.9, eos_token_id=50256, repetition_penalty=1.0):
        self.eval()
        generated = input_ids

        for _ in range(max_new_tokens):
            logits = self(generated)
            next_logits = logits[:, -1, :].clone()

            if repetition_penalty != 1.0:
                seen = set(generated[0].tolist())
                for token_id in seen:
                    if next_logits[0, token_id] > 0:
                        next_logits[0, token_id] /= repetition_penalty
                    else:
                        next_logits[0, token_id] *= repetition_penalty

            next_logits = next_logits / max(temperature, 1e-5)

            if top_k > 0:
                k = min(top_k, next_logits.size(-1))
                topk_vals, _ = torch.topk(next_logits, k, dim=-1)
                threshold = topk_vals[:, -1].unsqueeze(-1)
                next_logits = torch.where(
                    next_logits < threshold,
                    torch.full_like(next_logits, -float('inf')),
                    next_logits,
                )

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cumulative_probs > top_p
                sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
                sorted_mask[:, 0] = False
                indices_to_remove = sorted_mask.scatter(1, sorted_indices, sorted_mask)
                next_logits = next_logits.masked_fill(indices_to_remove, -float('inf'))

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=-1)

            if next_token.item() == eos_token_id:
                break

        return generated


def read_gguf_metadata(reader):
    meta = {}
    for key, field in reader.fields.items():
        if key.startswith("GGUF."):
            continue
        if key.startswith("general."):
            continue
        if key.startswith("tokenizer."):
            continue
        try:
            meta[key] = field.contents()
        except Exception:
            pass
    return meta


def ggml_tensor_to_torch(tensor):
    data = tensor.data
    if isinstance(data, np.ndarray):
        arr = data
    else:
        arr = np.array(data)

    ttype = tensor.tensor_type

    if ttype == gguf.GGMLQuantizationType.F32:
        return torch.from_numpy(arr.astype(np.float32))
    if ttype == gguf.GGMLQuantizationType.F16:
        return torch.from_numpy(arr.astype(np.float32))
    if ttype == gguf.GGMLQuantizationType.BF16:
        u = arr.view(np.uint16).astype(np.uint32)
        f = (u << 16).view(np.float32)
        return torch.from_numpy(f)
    if ttype == gguf.GGMLQuantizationType.I32:
        return torch.from_numpy(arr.astype(np.int64))
    if ttype == gguf.GGMLQuantizationType.BOOL:
        return torch.from_numpy(arr.astype(bool))

    return torch.from_numpy(arr.astype(np.float32))


def load_model_from_gguf(gguf_path, device="cpu"):
    if not os.path.isfile(gguf_path):
        raise FileNotFoundError(f"GGUF file not found: {gguf_path}")

    reader = gguf.GGUFReader(gguf_path)
    meta = read_gguf_metadata(reader)

    d_model = int(meta.get("teax.d_model", 0))
    vocab_size = int(meta.get("teax.vocab_size", 0))
    seq_len = int(meta.get("teax.seq_len", 0))
    hidden_mult = int(meta.get("teax.hidden_mult", 3))
    children_per_parent = int(meta.get("teax.children_per_parent", 2))
    top_k = int(meta.get("teax.top_k", 2))
    layer_experts_raw = meta.get("teax.layer_experts", [])

    if isinstance(layer_experts_raw, np.ndarray):
        layer_experts = tuple(int(x) for x in layer_experts_raw.tolist())
    elif isinstance(layer_experts_raw, (list, tuple)):
        layer_experts = tuple(int(x) for x in layer_experts_raw)
    else:
        layer_experts = (12, 24, 48)

    if not d_model or not vocab_size:
        raise RuntimeError(f"GGUF missing arch metadata: d_model={d_model} vocab_size={vocab_size}")

    if not seq_len:
        seq_len = 192

    print(f"[loader] arch: d={d_model} vocab={vocab_size} seq={seq_len}")
    print(f"[loader] layer_experts={list(layer_experts)} hidden_mult={hidden_mult} top_k={top_k}")

    stage = meta.get("teax.source_stage", "unknown")
    global_step = int(meta.get("teax.global_step", 0))
    epochs = int(meta.get("teax.completed_epochs", 0))
    print(f"[loader] source_stage={stage} global_step={global_step} epochs={epochs}")

    model = TeaXDictaV07(
        vocab_size=vocab_size,
        d=d_model,
        layer_experts=layer_experts,
        children_per_parent=children_per_parent,
        hidden_mult=hidden_mult,
        top_k=top_k,
        max_seq_len=seq_len,
    )

    state_dict = {}
    for tensor in reader.tensors:
        name = tensor.name
        if not name:
            continue
        state_dict[name] = ggml_tensor_to_torch(tensor)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[loader] missing keys: {missing[:6]}{' ...' if len(missing) > 6 else ''} ({len(missing)} total)")
    if unexpected:
        print(f"[loader] unexpected keys: {unexpected[:6]}{' ...' if len(unexpected) > 6 else ''} ({len(unexpected)} total)")

    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[loader] params={n_params/1e6:.2f}M device={device}")

    return model


def load_tokenizer(tokenizer_name):
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


SYSTEM_WRAPPER = "<|system|>{content}"
USER_WRAPPER = "<|user|>{content}"
ASSISTANT_WRAPPER = "<|assistant|>{content}"


def build_prompt(system_prompt, history, user_input):
    if not history:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]
    else:
        messages = history + [{"role": "user", "content": user_input}]

    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            parts.append(SYSTEM_WRAPPER.format(content=content))
        elif role == "user":
            parts.append(USER_WRAPPER.format(content=content))
        elif role == "assistant":
            parts.append(ASSISTANT_WRAPPER.format(content=content))

    parts.append("<|assistant|>")
    return "".join(parts)


def strip_special(text):
    for marker in ["<|user|>", "<|system|>", "<|assistant|>", "<|endoftext|>"]:
        if marker in text:
            text = text.split(marker)[0]
    return text.strip()


def generate_response(model, tokenizer, prompt, args):
    enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                    max_length=model.pos_emb.num_embeddings)
    input_ids = enc["input_ids"].to(args.device)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=args.repetition_penalty,
    )

    new_ids = output_ids[0, input_ids.shape[1]:]
    text = tokenizer.decode(new_ids, skip_special_tokens=False)
    return strip_special(text)


def load_system_prompt(path):
    if not path:
        return "You are TeaX-Dicta, a helpful assistant."
    if not os.path.isfile(path):
        raise FileNotFoundError(f"system prompt file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def parse_args():
    p = argparse.ArgumentParser(
        description="DictionariRun — universal runner for Dictionari family models",
    )
    p.add_argument("--gguf-model", required=True, help="path to .gguf model file")
    p.add_argument("--systemprompt", default="", help="path to system prompt file")
    p.add_argument("--chat", default="", help="single-turn chat input, prints result and exits")
    p.add_argument("--cli", action="store_true", help="interactive CLI mode")
    p.add_argument("--device", default="cpu", help="cpu, cuda, or mps")
    p.add_argument("--tokenizer", default="gpt2", help="HF tokenizer name")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    return p.parse_args()


def run_chat_once(model, tokenizer, system_prompt, user_input, args):
    prompt = build_prompt(system_prompt, [], user_input)
    print(f"[inject] system prompt injected, prompt length={len(prompt)} chars")
    print()
    response = generate_response(model, tokenizer, prompt, args)
    print(response)


def run_cli(model, tokenizer, system_prompt, args):
    print("=" * 60)
    print("  Dictionari CLI")
    print("  /exit quit    /clear reset    /history show turns")
    print("=" * 60)
    print()

    history = []
    turn = 0

    while True:
        try:
            user_input = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[exit]")
            break

        if not user_input:
            continue

        if user_input == "/exit":
            print("[exit]")
            break

        if user_input == "/clear":
            history = []
            turn = 0
            print("[system] conversation cleared")
            continue

        if user_input == "/history":
            if not history:
                print("[system] no history yet")
            for i, msg in enumerate(history):
                preview = msg["content"][:80]
                print(f"  [{i}] {msg['role']}: {preview}")
            continue

        turn += 1
        prompt = build_prompt(system_prompt, history, user_input)

        if turn == 1:
            print(f"[inject] system prompt injected as <|system|> role ({len(prompt)} chars)")
            print()

        print("Dictionari > ", end="", flush=True)
        try:
            response = generate_response(model, tokenizer, prompt, args)
        except Exception as e:
            print(f"\n[error] {e}")
            continue

        print(response)
        print()

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})


def main():
    args = parse_args()

    if not args.chat and not args.cli:
        print("[error] specify either --chat \"...\" or --cli")
        sys.exit(1)

    if args.chat and args.cli:
        print("[error] --chat and --cli are mutually exclusive")
        sys.exit(1)

    system_prompt = load_system_prompt(args.systemprompt)
    print(f"[system] prompt loaded ({len(system_prompt)} chars)")

    tokenizer = load_tokenizer(args.tokenizer)
    print(f"[tokenizer] {args.tokenizer} vocab={tokenizer.vocab_size}")

    model = load_model_from_gguf(args.gguf_model, device=args.device)

    if args.chat:
        run_chat_once(model, tokenizer, system_prompt, args.chat, args)
    else:
        run_cli(model, tokenizer, system_prompt, args)


if __name__ == "__main__":
    main()