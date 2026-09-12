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
        y = y[:, :, :x.size(1)]  # 裁掉因果 padding 多出的部分
        y = y.transpose(1, 2)
        return x + torch.sigmoid(self.gate(x)) * y


class DictionaryScheduler(nn.Module):
    def __init__(self, d_in, n_experts, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_experts),
        )

    def forward(self, x):
        return self.net(x)


class SharedTrunk(nn.Module):
    def __init__(self, d, n_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([CausalMix(d) for _ in range(n_blocks)])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


class ExpertHead(nn.Module):
    def __init__(self, d, hidden_mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d * hidden_mult),
            nn.GELU(),
            nn.Linear(d * hidden_mult, d),
        )

    def forward(self, x):
        return self.net(x)


class TeaXDictaV07(nn.Module):
    def __init__(self, vocab_size, d=256, n_layers=2, n_experts=8, max_seq_len=256, trunk_blocks=2):
        super().__init__()
        self.n_layers = n_layers
        self.n_experts = n_experts

        self.token_emb = nn.Embedding(vocab_size, d)
        self.pos_emb = nn.Embedding(max_seq_len, d)
        self.schedulers = nn.ModuleList([DictionaryScheduler(d, n_experts) for _ in range(n_layers)])
        self.trunks = nn.ModuleList([SharedTrunk(d, trunk_blocks) for _ in range(n_layers)])
        self.experts = nn.ModuleList([ExpertHead(d) for _ in range(n_experts)])
        self.norm = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def _balance_loss(self, probs):
        B, L, E = probs.shape
        flat = probs.reshape(-1, E)
        p = flat.mean(dim=0)
        idx = flat.argmax(dim=-1)
        f = torch.bincount(idx, minlength=E).float() / idx.numel()
        return E * (f * p).sum()

    def forward(self, input_ids, labels=None):
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(pos)

        aux_total = 0.0
        for sched, trunk in zip(self.schedulers, self.trunks):
            probs = F.softmax(sched(x), dim=-1)
            expert_out = 0.0
            for i, expert in enumerate(self.experts):
                expert_out = expert_out + probs[..., i:i + 1] * expert(x)
            x = trunk(x + expert_out)
            aux_total = aux_total + self._balance_loss(probs)

        logits = self.lm_head(self.norm(x))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss, aux_total / max(len(self.schedulers), 1)


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


def build_dataset(dataset_name, tokenizer, seq_len, max_samples=None):
    print(f"[data] loading {dataset_name} ...", flush=True)
    ds = load_dataset(dataset_name, split="train")
    if max_samples is not None and len(ds) > max_samples:
        ds = ds.select(range(max_samples))
    print(f"[data] rows = {len(ds)}", flush=True)

    def tok_fn(batch):
        n = len(next(iter(batch.values())))
        texts = []
        for i in range(n):
            t = extract_text({k: v[i] for k, v in batch.items()})
            texts.append(t or "")
        enc = tokenizer(texts, truncation=True, max_length=seq_len, padding="max_length")
        enc["labels"] = [ids[:] for ids in enc["input_ids"]]
        return enc

    ds = ds.map(tok_fn, batched=True, remove_columns=ds.column_names)
    ds.set_format("torch")
    return ds


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--config", default="")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--resume_from", default="")
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--bsz", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--aux_weight", type=float, default=0.01)
    p.add_argument("--max_samples", type=int, default=200000)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--n_experts", type=int, default=8)
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--log_every", type=int, default=20)
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
    progress = (step - warmup) / max(total - warmup, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def save_ckpt(model, args, step):
    path = os.path.join(args.output_dir, "model.pt")
    torch.save(model.state_dict(), path)
    meta = {
        "model": MODEL_NAME,
        "stage": args.stage,
        "steps": step,
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "n_experts": args.n_experts,
        "seq_len": args.seq_len,
        "vocab_size": model.token_emb.num_embeddings,
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[ckpt] saved {path} at step {step}", flush=True)


def main():
    args = get_args()
    args = load_config_overrides(args)

    torch.manual_seed(args.seed)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[env] {MODEL_NAME} | torch={torch.__version__} threads={torch.get_num_threads()}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = build_dataset(args.dataset, tokenizer, args.seq_len, args.max_samples)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.bsz, shuffle=True, drop_last=True, num_workers=0
    )

    model = TeaXDictaV07(
        vocab_size=len(tokenizer),
        d=args.d_model,
        n_layers=args.n_layers,
        n_experts=args.n_experts,
        max_seq_len=args.seq_len,
    )
    print(f"[model] params = {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)

    start_step = 0
    if args.resume_from and os.path.isfile(args.resume_from):
        state = torch.load(args.resume_from, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[resume] missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        meta_path = os.path.join(os.path.dirname(args.resume_from), "meta.json")
        if os.path.isfile(meta_path):
            try:
                start_step = int(json.load(open(meta_path)).get("steps", 0))
                print(f"[resume] start_step={start_step}", flush=True)
            except Exception:
                pass
    elif args.resume_from:
        print(f"[resume] file not found: {args.resume_from}, training from scratch", flush=True)

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    step, micro = start_step, 0
    t0 = time.time()
    running_loss = 0.0
    running_aux = 0.0

    while step < args.max_steps:
        for batch in loader:
            _, loss, aux = model(batch["input_ids"], labels=batch["labels"])
            total = loss + args.aux_weight * aux
            (total / args.grad_accum).backward()
            micro += 1
            running_loss += loss.item()
            running_aux += aux.item()

            if micro % args.grad_accum == 0:
                lr = lr_at(step, args.lr, args.warmup_steps, args.max_steps)
                for g in opt.param_groups:
                    g["lr"] = lr

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                step += 1

                if step % args.log_every == 0:
                    elapsed = time.time() - t0
                    tok_per_s = (args.log_every * args.grad_accum * args.bsz * args.seq_len) / max(elapsed, 1e-6)
                    print(
                        f"[{MODEL_NAME}][{args.stage}] step={step}/{args.max_steps} "
                        f"loss={running_loss/args.log_every:.4f} "
                        f"aux={running_aux/args.log_every:.4f} "
                        f"lr={lr:.2e} tok/s={tok_per_s:.0f}",
                        flush=True,
                    )
                    running_loss = 0.0
                    running_aux = 0.0
                    t0 = time.time()

                if step % args.save_every == 0:
                    save_ckpt(model, args, step)
                if step >= args.max_steps:
                    break

    save_ckpt(model, args, step)
    print(f"[done] {MODEL_NAME} stage={args.stage} steps={step}", flush=True)


if __name__ == "__main__":
    main()