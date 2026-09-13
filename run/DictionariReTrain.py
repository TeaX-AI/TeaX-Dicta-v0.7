import argparse
import copy
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer


DIM_FORMAT = "teax-dicta-dim"
ROLES_FORMAT = "teax-dicta-model-roles"


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

    def forward(self, input_ids, labels=None):
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(pos)

        aux_total = 0.0
        parent_probs = None
        for layer in self.layers:
            x, probs = layer(x, parent_probs=parent_probs)
            aux_total = aux_total + self._balance_loss(probs)
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

        aux = aux_total / max(len(self.layers), 1)
        return logits, loss, aux

    def _balance_loss(self, probs):
        B, L, E = probs.shape
        flat = probs.reshape(-1, E)
        p = flat.mean(dim=0)
        f = torch.bincount(flat.argmax(dim=-1), minlength=E).float() / flat.size(0)
        return E * (f * p).sum()


def classify_param(name):
    if ".router." in name:
        return "dict"
    if ".trunk." in name:
        return "trunk"
    if name.endswith(".W1") or name.endswith(".b1"):
        return "expert"
    if name.endswith(".W2") or name.endswith(".b2"):
        return "expert"
    return "model"


def build_roles(model):
    roles = {}
    dict_layers = set()
    expert_layers = set()
    trunk_layers = set()

    for name, _ in model.named_parameters():
        role = classify_param(name)
        roles[name] = role

        parts = name.split(".")
        if len(parts) >= 2 and parts[0] == "layers":
            try:
                layer_idx = int(parts[1])
            except ValueError:
                continue
            if role == "dict":
                dict_layers.add(layer_idx)
            elif role == "expert":
                expert_layers.add(layer_idx)
            elif role == "trunk":
                trunk_layers.add(layer_idx)

    return {
        "format": ROLES_FORMAT,
        "version": 1,
        "roles": roles,
        "dict_layers": sorted(dict_layers),
        "expert_layers": sorted(expert_layers),
        "trunk_layers": sorted(trunk_layers),
    }


def layer_index_of(name):
    parts = name.split(".")
    if len(parts) >= 2 and parts[0] == "layers":
        try:
            return int(parts[1])
        except ValueError:
            return -1
    return -1


def expert_index_of(name):
    parts = name.split(".")
    if len(parts) >= 3 and parts[0] == "layers":
        try:
            return int(parts[2])
        except ValueError:
            return -1
    return -1


def configure_trainable(model, args, roles):
    for p in model.parameters():
        p.requires_grad = False

    total = sum(1 for _ in model.parameters())
    trainable_names = []

    if args.unfreeze_mode == "full":
        for name, p in model.named_parameters():
            p.requires_grad = True
            trainable_names.append(name)
    else:
        for name, p in model.named_parameters():
            role = roles["roles"].get(name, "model")

            if role == "dict":
                p.requires_grad = True
                trainable_names.append(name)
                continue

            if args.unfreeze_mode == "dict+last-experts":
                if role == "expert":
                    li = layer_index_of(name)
                    if li == len(model.layers) - 1:
                        p.requires_grad = True
                        trainable_names.append(name)
                continue

            if args.unfreeze_mode == "dict+experts":
                if role == "expert":
                    p.requires_grad = True
                    trainable_names.append(name)
                continue

    trainable_count = sum(1 for _ in trainable_names)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[freeze] unfreeze_mode={args.unfreeze_mode}")
    print(f"[freeze] trainable tensors {trainable_count}/{total}")
    print(f"[freeze] trainable params {trainable_params/1e6:.2f}M")

    return trainable_names


def reinit_dict_layers(model, args, roles):
    if not args.reinit_dict:
        print("[dict] keeping existing router weights")
        return

    if args.reinit_dict_seed is not None:
        torch.manual_seed(args.reinit_dict_seed)

    targets = []
    for name, p in model.named_parameters():
        if roles["roles"].get(name) == "dict":
            targets.append((name, p))

    print(f"[dict] reinitializing {len(targets)} router tensors")
    for name, p in targets:
        if p.dim() == 1:
            nn.init.zeros_(p)
        else:
            nn.init.normal_(p, std=args.reinit_dict_std)

    if args.reinit_dict_perturb > 0.0:
        print(f"[dict] applying perturbation std={args.reinit_dict_perturb}")
        for _, p in targets:
            noise = torch.randn_like(p) * args.reinit_dict_perturb
            p.data.add_(noise)


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


def build_dataset(dataset_name, tokenizer, seq_len, max_samples):
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


def load_dim(dim_path):
    if not os.path.isfile(dim_path):
        raise FileNotFoundError(f"dim not found: {dim_path}")

    dim = torch.load(dim_path, map_location="cpu", weights_only=False)

    if not isinstance(dim, dict):
        raise RuntimeError("dim file is not a dict")
    if dim.get("format") != DIM_FORMAT:
        raise RuntimeError(f"not a dim file: format={dim.get('format')}")
    if "model" not in dim:
        raise RuntimeError("dim file has no 'model' key")
    if "arch" not in dim:
        raise RuntimeError("dim file has no 'arch' key")

    return dim


def save_dim(path, model, meta, roles, optim_state=None, keep_optimizer=False):
    arch = meta.get("arch", {})
    payload = {
        "format": DIM_FORMAT,
        "version": 1,
        "arch": arch,
        "model": {k: v.cpu() for k, v in model.state_dict().items()},
        "meta": meta,
        "dictionari-model": roles,
    }
    if keep_optimizer and optim_state is not None:
        payload["optimizer"] = optim_state
    torch.save(payload, path)
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"[dim] saved {path} ({size_mb:.1f} MB)")


def lr_at(step, base_lr, warmup, total):
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    progress = min(max((step - warmup) / max(total - warmup, 1), 0.0), 1.0)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def parse_args():
    p = argparse.ArgumentParser(
        description="DictionariReTrain — resource-efficient fine-tuning for Dictionari models",
    )
    p.add_argument("--dim", required=True, help="input .dim model file")
    p.add_argument("--dataset", required=True, help="fine-tune dataset name")
    p.add_argument("--output", required=True, help="output .dim file path")
    p.add_argument("--unfreeze-mode",
                   choices=["dict-only", "dict+last-experts", "dict+experts", "full"],
                   default="dict+last-experts",
                   help="which parameters to unfreeze")
    p.add_argument("--reinit-dict", action="store_true",
                   help="reinitialize router weights before training")
    p.add_argument("--reinit-dict-std", type=float, default=0.02)
    p.add_argument("--reinit-dict-seed", type=int, default=None)
    p.add_argument("--reinit-dict-perturb", type=float, default=0.0,
                   help="perturb router weights by this std after reinit")

    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--bsz", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--warmup-max", type=int, default=200)
    p.add_argument("--aux-weight", type=float, default=0.5)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--max-minutes", type=int, default=0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--device", default="cpu")
    p.add_argument("--keep-optimizer", action="store_true",
                   help="save optimizer state into output dim")
    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    cpu_count = os.cpu_count() or 4
    torch.set_num_threads(cpu_count)
    print(f"[env] cpus={cpu_count} device={args.device}")

    print(f"[dim] loading {args.dim}")
    dim = load_dim(args.dim)

    arch = dim["arch"]
    meta_in = dim.get("meta", {})
    roles_in = dim.get("dictionari-model", None)

    vocab_size = int(arch["vocab_size"])
    d_model = int(arch["d_model"])
    layer_experts = tuple(int(x) for x in arch["layer_experts"])
    children_per_parent = int(arch.get("children_per_parent", 2))
    hidden_mult = int(arch.get("hidden_mult", 3))
    top_k = int(arch.get("top_k", 2))
    seq_len = int(arch.get("seq_len", 192))

    print(f"[dim] arch d={d_model} vocab={vocab_size} seq={seq_len} "
          f"layer_experts={list(layer_experts)} hidden_mult={hidden_mult} top_k={top_k}")

    model = TeaXDictaV07(
        vocab_size=vocab_size,
        d=d_model,
        layer_experts=layer_experts,
        children_per_parent=children_per_parent,
        hidden_mult=hidden_mult,
        top_k=top_k,
        max_seq_len=seq_len,
    )

    missing, unexpected = model.load_state_dict(dim["model"], strict=False)
    if missing:
        print(f"[dim] missing={len(missing)}")
    if unexpected:
        print(f"[dim] unexpected={len(unexpected)}")

    model.to(args.device)

    roles = roles_in if roles_in is not None else build_roles(model)
    if roles_in is None:
        print("[dim] no dictionari-model roles found, using auto-detected roles")

    if roles["format"] != ROLES_FORMAT:
        raise RuntimeError(f"bad roles format: {roles['format']}")

    trainable_names = configure_trainable(model, args, roles)
    reinit_dict_layers(model, args, roles)

    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("no trainable parameters after freeze configuration")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = build_dataset(args.dataset, tokenizer, seq_len, args.max_samples)
    probe = make_loader(ds, args, epoch=0)
    steps_per_epoch = len(probe)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(int(total_steps * args.warmup_ratio), 1)
    warmup_steps = min(warmup_steps, args.warmup_max)

    print(f"[plan] steps_per_epoch={steps_per_epoch} epochs={args.epochs} "
          f"total_steps={total_steps} warmup={warmup_steps}")

    global_step = 0
    micro = 0
    t0 = time.time()
    train_start = time.time()
    running_loss = 0.0
    running_aux = 0.0

    for epoch in range(args.epochs):
        loader = make_loader(ds, args, epoch)

        for batch in loader:
            if args.max_minutes > 0 and (time.time() - train_start) > args.max_minutes * 60:
                print(f"[soft-stop] {args.max_minutes} minutes elapsed")
                break

            input_ids = batch["input_ids"].to(args.device)
            labels = batch["labels"].to(args.device)

            _, loss, aux = model(input_ids, labels=labels)
            total = loss + args.aux_weight * aux
            (total / args.grad_accum).backward()
            micro += 1
            running_loss += loss.item()
            running_aux += aux.item()

            if micro % args.grad_accum == 0:
                lr = lr_at(global_step, args.lr, warmup_steps, total_steps)
                for g in optimizer.param_groups:
                    g["lr"] = lr

                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.log_every == 0:
                    elapsed = time.time() - t0
                    tok_per_s = (args.log_every * args.grad_accum * args.bsz * seq_len) / max(elapsed, 1e-6)
                    print(
                        f"[ReTrain] epoch={epoch+1}/{args.epochs} "
                        f"step={global_step}/{total_steps} "
                        f"loss={running_loss/args.log_every:.4f} "
                        f"aux={running_aux/args.log_every:.4f} "
                        f"lr={lr:.2e} tok/s={tok_per_s:.0f}"
                    )
                    running_loss = 0.0
                    running_aux = 0.0
                    t0 = time.time()

                if global_step % args.save_every == 0:
                    meta_out = dict(meta_in)
                    meta_out.update({
                        "arch": arch,
                        "stage": meta_in.get("stage", "finetune"),
                        "finetune_mode": args.unfreeze_mode,
                        "finetune_dataset": args.dataset,
                        "finetune_step": global_step,
                        "reinit_dict": args.reinit_dict,
                    })
                    save_dim(args.output, model, meta_out, roles)

        if args.max_minutes > 0 and (time.time() - train_start) > args.max_minutes * 60:
            break

    meta_out = dict(meta_in)
    meta_out.update({
        "arch": arch,
        "stage": meta_in.get("stage", "finetune"),
        "finetune_mode": args.unfreeze_mode,
        "finetune_dataset": args.dataset,
        "finetune_step": global_step,
        "reinit_dict": args.reinit_dict,
        "complete": True,
    })

    optim_state = None
    if args.keep_optimizer:
        optim_state = optimizer.state_dict()

    save_dim(args.output, model, meta_out, roles,
             optim_state=optim_state, keep_optimizer=args.keep_optimizer)

    print(f"[done] finetuned {global_step} steps, saved to {args.output}")


if __name__ == "__main__":
    main()