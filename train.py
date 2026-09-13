import argparse
import json
import os
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_NAME = "TeaX-Dicta-v0.7"


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
                 hidden_mult=4, top_k=2, trunk_blocks=2):
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
            assert n_experts == n_parents * children_per_parent
            child_to_parent = torch.arange(n_experts) // children_per_parent
            self.register_buffer("child_to_parent", child_to_parent)
        else:
            self.child_to_parent = None

    def balance_loss(self, probs):
        B, L, E = probs.shape
        flat = probs.reshape(-1, E)
        p = flat.mean(dim=0)
        f = torch.bincount(flat.argmax(dim=-1), minlength=E).float() / flat.size(0)
        return E * (f * p).sum()

    def forward(self, x, parent_probs=None):
        B, L, D = x.shape
        x_flat = x.reshape(B * L, D)

        logits = self.router(x_flat)

        if parent_probs is not None and self.child_to_parent is not None:
            pp = parent_probs.reshape(B * L, -1)
            if self.training:
                gating = pp
            else:
                gating = F.one_hot(pp.argmax(dim=-1), num_classes=self.n_parents).float()
            parent_prob_per_expert = gating[:, self.child_to_parent]
            logits = logits + torch.log(parent_prob_per_expert + 1e-6)

        probs = F.softmax(logits, dim=-1)
        topk_weights, topk_indices = torch.topk(probs, k=self.top_k, dim=-1)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)

        flat_indices = topk_indices.reshape(-1)
        flat_weights = topk_weights.reshape(-1)
        token_ids = torch.arange(B * L, device=x.device).repeat_interleave(self.top_k)

        sorted_experts, sort_order = torch.sort(flat_indices)
        sorted_token_ids = token_ids[sort_order]
        sorted_weights = flat_weights[sort_order]
        sorted_x = x_flat[sorted_token_ids]

        expert_counts = torch.bincount(sorted_experts, minlength=self.n_experts)

        sorted_output = torch.zeros_like(sorted_x)
        offset = 0
        for e in range(self.n_experts):
            count = int(expert_counts[e].item())
            if count == 0:
                continue
            expert_tokens = sorted_x[offset:offset + count]
            h = expert_tokens @ self.W1[e] + self.b1[e]
            h = F.gelu(h)
            out = h @ self.W2[e] + self.b2[e]
            w = sorted_weights[offset:offset + count].unsqueeze(-1)
            sorted_output[offset:offset + count] = out * w
            offset += count

        output = torch.zeros(B * L, D, device=x.device)
        output.index_add_(0, sorted_token_ids, sorted_output)
        output = output.reshape(B, L, D)

        x = self.trunk(x + output)
        probs = probs.reshape(B, L, self.n_experts)
        return x, probs


class TeaXDictaV07(nn.Module):
    def __init__(self, vocab_size, d=512, layer_experts=(16, 32, 64),
                 children_per_parent=2, hidden_mult=4, top_k=2,
                 max_seq_len=256, trunk_blocks=2):
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

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.Conv1d):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids, labels=None):
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(pos)

        aux_total = 0.0
        parent_probs = None
        for layer in self.layers:
            x, probs = layer(x, parent_probs=parent_probs)
            aux_total = aux_total + layer.balance_loss(probs)
            parent_probs = probs

        logits = self.lm_head(self.norm(x))

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss, aux_total / max(len(self.layers), 1)


def extract_text(example):
    if "text" in example and isinstance(example["text"], str) and example["text"]:
        return example["text"]
    if "conversations" in example and example["conversations"]:
        return "".join(
            f"<|{t.get('from', t.get('role', 'user'))}|>{t.get('value', t.get('content', ''))}"
            for t in example["conversations"]
        )
    if "query" in example and "answer" in example:
        return f"<|user|>{example['query']}<|assistant|>{example['answer']}"
    if "instruction" in example and "output" in example:
        return f"<|user|>{example['instruction']}<|assistant|>{example['output']}"
    if "prompt" in example and "response" in example:
        return f"<|user|>{example['prompt']}<|assistant|>{example['response']}"
    for k in ["content", "input", "question"]:
        if k in example and isinstance(example[k], str) and example[k]:
            return example[k]
    return None


def build_dataset(dataset_name, tokenizer, seq_len, max_samples=0):
    print(f"[data] loading {dataset_name} ...", flush=True)
    ds = load_dataset(dataset_name, split="train")
    if max_samples and max_samples > 0 and len(ds) > max_samples:
        ds = ds.select(range(max_samples))
        print(f"[data] truncated to {max_samples}", flush=True)
    print(f"[data] rows = {len(ds)}", flush=True)

    pad_id = tokenizer.pad_token_id

    def tok_fn(batch):
        n = len(next(iter(batch.values())))
        texts = []
        for i in range(n):
            t = extract_text({k: v[i] for k, v in batch.items()})
            texts.append(t or "")
        enc = tokenizer(texts, truncation=True, max_length=seq_len, padding="max_length")
        enc["labels"] = [
            [(-100 if t == pad_id else t) for t in ids]
            for ids in enc["input_ids"]
        ]
        return enc

    ds = ds.map(tok_fn, batched=True, remove_columns=ds.column_names)
    ds.set_format("torch")
    return ds


def parse_int_tuple(s):
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--config", default="")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--resume_from", default="")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--bsz", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--aux_weight", type=float, default=0.5)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--max_minutes", type=int, default=0)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--layer_experts", type=str, default="16,32,64")
    p.add_argument("--children_per_parent", type=int, default=2)
    p.add_argument("--hidden_mult", type=int, default=4)
    p.add_argument("--top_k", type=int, default=2)
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_config_overrides(args):
    if not args.config or not os.path.isfile(args.config):
        return args
    try:
        import yaml
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)
        print(f"[config] loaded overrides from {args.config}", flush=True)
    except Exception as e:
        print(f"[config] skip {args.config}: {e}", flush=True)
    return args


def lr_at(step, base_lr, warmup, total):
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    progress = min(max((step - warmup) / max(total - warmup, 1), 0.0), 1.0)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def make_loader(ds, args, epoch):
    g = torch.Generator()
    g.manual_seed(args.seed + epoch)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=args.bsz,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        generator=g,
    )


def build_optimizer(model, args):
    try:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        print("[opt] using 8-bit AdamW", flush=True)
        return opt
    except Exception as e:
        print(f"[opt] bitsandbytes unavailable ({e}), falling back to fp32 AdamW", flush=True)
        return torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )


def save_ckpt(model, opt, args, completed_epochs, epoch_step, global_step):
    path = os.path.join(args.output_dir, "model.pt")
    torch.save({
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "completed_epochs": completed_epochs,
        "epoch_step": epoch_step,
        "global_step": global_step,
        "stage": args.stage,
    }, path)
    meta = {
        "model": MODEL_NAME,
        "stage": args.stage,
        "completed_epochs": completed_epochs,
        "epoch_step": epoch_step,
        "global_step": global_step,
        "epochs": args.epochs,
        "complete": completed_epochs >= args.epochs,
        "d_model": args.d_model,
        "layer_experts": list(model.layer_experts),
        "children_per_parent": model.children_per_parent,
        "seq_len": args.seq_len,
        "vocab_size": model.token_emb.num_embeddings,
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[ckpt] stage={args.stage} epoch={completed_epochs} epoch_step={epoch_step} global={global_step}", flush=True)


def try_resume(model, args):
    src = args.resume_from
    if not src or not os.path.isfile(src):
        return 0, 0, 0, False

    state = torch.load(src, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        missing, unexpected = model.load_state_dict(state["model"], strict=False)
        print(f"[resume] missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        gs = int(state.get("global_step", 0))
        ep = int(state.get("completed_epochs", 0))
        es = int(state.get("epoch_step", 0))
    else:
        model.load_state_dict(state, strict=False)
        gs, ep, es = 0, 0, 0

    src_meta = os.path.join(os.path.dirname(src), "meta.json")
    src_stage = None
    if os.path.isfile(src_meta):
        try:
            meta = json.load(open(src_meta))
            src_stage = meta.get("stage")
            gs = int(meta.get("global_step", gs))
        except Exception:
            pass

    if src_stage == args.stage:
        print(f"[resume-same] epoch={ep} epoch_step={es} global={gs}", flush=True)
        return ep, es, gs, ep >= args.epochs
    else:
        print(f"[resume-cross] from={src_stage} global={gs}", flush=True)
        return 0, 0, gs, False


def main():
    args = get_args()
    args = load_config_overrides(args)

    torch.manual_seed(args.seed)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[env] {MODEL_NAME} torch={torch.__version__} threads={torch.get_num_threads()}", flush=True)
    print(f"[args] {vars(args)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = build_dataset(args.dataset, tokenizer, args.seq_len, args.max_samples)

    probe = make_loader(ds, args, epoch=0)
    steps_per_epoch = len(probe)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(int(total_steps * args.warmup_ratio), 1)
    print(f"[plan] steps_per_epoch={steps_per_epoch} epochs={args.epochs} total_steps={total_steps} warmup={warmup_steps}", flush=True)

    layer_experts = parse_int_tuple(args.layer_experts)
    model = TeaXDictaV07(
        vocab_size=len(tokenizer),
        d=args.d_model,
        layer_experts=layer_experts,
        children_per_parent=args.children_per_parent,
        hidden_mult=args.hidden_mult,
        top_k=args.top_k,
        max_seq_len=args.seq_len,
    )
    print(f"[model] layer_experts={list(layer_experts)} total_experts={sum(layer_experts)} top_k={args.top_k}", flush=True)
    print(f"[model] params = {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)

    start_epoch, start_epoch_step, global_step, done = try_resume(model, args)
    if done:
        print(f"[skip] {args.stage} already complete (epoch={start_epoch}/{args.epochs})", flush=True)
        return

    model.train()
    opt = build_optimizer(model, args)
    micro = 0
    t0 = time.time()
    train_start = time.time()
    running_loss = 0.0
    running_aux = 0.0

    for epoch in range(start_epoch, args.epochs):
        loader = make_loader(ds, args, epoch)
        skip = start_epoch_step if epoch == start_epoch else 0
        epoch_step = skip

        for batch in loader:
            if skip > 0:
                skip -= 1
                continue

            if args.max_minutes > 0 and (time.time() - train_start) > args.max_minutes * 60:
                print(f"[soft-stop] {args.max_minutes} minutes elapsed, saving checkpoint", flush=True)
                save_ckpt(model, opt, args, epoch, epoch_step, global_step)
                return

            _, loss, aux = model(batch["input_ids"], labels=batch["labels"])
            total = loss + args.aux_weight * aux
            (total / args.grad_accum).backward()
            micro += 1
            running_loss += loss.item()
            running_aux += aux.item()

            if micro % args.grad_accum == 0:
                lr = lr_at(global_step, args.lr, warmup_steps, total_steps)
                for g in opt.param_groups:
                    g["lr"] = lr

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                epoch_step += 1
                global_step += 1

                if epoch_step % args.log_every == 0:
                    elapsed = time.time() - t0
                    tok_per_s = (args.log_every * args.grad_accum * args.bsz * args.seq_len) / max(elapsed, 1e-6)
                    print(
                        f"[{MODEL_NAME}][{args.stage}] "
                        f"epoch={epoch+1}/{args.epochs} "
                        f"step={epoch_step}/{steps_per_epoch} "
                        f"global={global_step}/{total_steps} "
                        f"loss={running_loss/args.log_every:.4f} "
                        f"aux={running_aux/args.log_every:.4f} "
                        f"lr={lr:.2e} tok/s={tok_per_s:.0f}",
                        flush=True,
                    )
                    running_loss = 0.0
                    running_aux = 0.0
                    t0 = time.time()

                if epoch_step % args.save_every == 0:
                    save_ckpt(model, opt, args, epoch, epoch_step, global_step)

        save_ckpt(model, opt, args, epoch + 1, 0, global_step)
        start_epoch_step = 0

    print(f"[done] {MODEL_NAME} stage={args.stage} epochs={args.epochs} global={global_step}", flush=True)


if __name__ == "__main__":
    main()