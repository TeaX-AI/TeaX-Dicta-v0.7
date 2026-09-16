import argparse
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


class DictionariModel(nn.Module):
    def __init__(self, vocab_size, d=640, layer_experts=(12, 24, 48),
                 children_per_parent=2, hidden_mult=3, top_k=2,
                 max_seq_len=192, trunk_blocks=2):
        super().__init__()
        self.layer_experts = list(layer_experts)
        self.children_per_parent = children_per_parent
        self.hidden_mult = hidden_mult
        self.top_k = top_k
        self.trunk_blocks = trunk_blocks

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

    def arch_dict(self):
        return {
            "d_model": self.token_emb.embedding_dim,
            "vocab_size": self.token_emb.num_embeddings,
            "seq_len": self.pos_emb.num_embeddings,
            "layer_experts": list(self.layer_experts),
            "children_per_parent": self.children_per_parent,
            "hidden_mult": self.hidden_mult,
            "top_k": self.top_k,
        }

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

        aux = aux_total / max(len(self.layers), 1)
        return logits, loss, aux


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


def configure_freeze(model, unfreeze_mode, roles=None):
    if unfreeze_mode == "full":
        return

    for p in model.parameters():
        p.requires_grad = False

    n_layers = len(model.layers)

    for name, p in model.named_parameters():
        role = roles["roles"].get(name, classify_param(name)) if roles else classify_param(name)

        if unfreeze_mode == "dict-only":
            if role == "dict":
                p.requires_grad = True

        elif unfreeze_mode == "dict+last-experts":
            if role == "dict":
                p.requires_grad = True
            elif role == "expert" and layer_index_of(name) == n_layers - 1:
                p.requires_grad = True

        elif unfreeze_mode == "dict+experts":
            if role == "dict" or role == "expert":
                p.requires_grad = True

        elif unfreeze_mode == "all-experts":
            if role == "expert":
                p.requires_grad = True


def reinit_dict_layers(model, args):
    if not args.reinit_dict:
        return

    if args.reinit_dict_seed is not None:
        torch.manual_seed(args.reinit_dict_seed)

    targets = []
    for name, p in model.named_parameters():
        if classify_param(name) == "dict":
            targets.append((name, p))

    print(f"[dict] reinitializing {len(targets)} router tensors", flush=True)
    for _, p in targets:
        if p.dim() == 1:
            nn.init.zeros_(p)
        else:
            nn.init.normal_(p, std=args.reinit_dict_std)

    if args.reinit_dict_perturb > 0.0:
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


def build_optimizer(model, args):
    try:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        print("[opt] using 8-bit AdamW", flush=True)
        return opt
    except Exception as e:
        print(f"[opt] bitsandbytes unavailable ({e}), falling back to fp32 AdamW", flush=True)
        return torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )


def lr_at(step, base_lr, warmup, total):
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    progress = min(max((step - warmup) / max(total - warmup, 1), 0.0), 1.0)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def load_dim(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"dim not found: {path}")

    dim = torch.load(path, map_location="cpu", weights_only=False)

    if not isinstance(dim, dict):
        raise RuntimeError("dim file is not a dict")
    if dim.get("format") != DIM_FORMAT:
        raise RuntimeError(f"not a dim file: format={dim.get('format')}")
    if "arch" not in dim:
        raise RuntimeError("dim file has no 'arch' key")
    if "model" not in dim:
        raise RuntimeError("dim file has no 'model' key")

    return dim


def save_dim(path, model, arch, meta, roles, optimizer=None, keep_optimizer=False):
    payload = {
        "format": DIM_FORMAT,
        "version": 1,
        "arch": arch,
        "model": {k: v.cpu() for k, v in model.state_dict().items()},
        "meta": meta,
        "dictionari-model": roles,
    }
    if keep_optimizer and optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()

    torch.save(payload, path)
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"[dim] saved {path} ({size_mb:.1f} MB)", flush=True)


def save_ckpt(model, arch, meta, roles, optimizer, args, completed_epochs, epoch_step, global_step):
    path = os.path.join(args.output_dir, "model.pt")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "arch": arch,
        "completed_epochs": completed_epochs,
        "epoch_step": epoch_step,
        "global_step": global_step,
        "stage": args.stage,
    }, path)

    meta_out = dict(meta)
    meta_out.update({
        "stage": args.stage,
        "completed_epochs": completed_epochs,
        "epoch_step": epoch_step,
        "global_step": global_step,
        "epochs": args.epochs,
        "complete": completed_epochs >= args.epochs,
        "arch": arch,
    })

    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump(meta_out, f, indent=2)

    roles_path = os.path.join(args.output_dir, "dictionari-model.json")
    with open(roles_path, "w") as f:
        json.dump(roles, f, indent=2)

    print(f"[ckpt] stage={args.stage} epoch={completed_epochs} epoch_step={epoch_step} global={global_step}", flush=True)


def try_resume(model, args):
    own_ckpt = os.path.join(args.output_dir, "model.pt")
    own_meta = os.path.join(args.output_dir, "meta.json")

    if os.path.isfile(own_ckpt) and os.path.isfile(own_meta):
        try:
            meta = json.load(open(own_meta))
            if meta.get("stage") == args.stage:
                state = torch.load(own_ckpt, map_location="cpu", weights_only=False)
                model.load_state_dict(state["model"], strict=False)
                gs = int(state.get("global_step", 0))
                ep = int(state.get("completed_epochs", 0))
                es = int(state.get("epoch_step", 0))
                print(f"[resume-self] epoch={ep} epoch_step={es} global={gs}", flush=True)
                return ep, es, gs, ep >= args.epochs
            else:
                print(f"[resume-self] stage mismatch: ckpt={meta.get('stage')} current={args.stage}, ignoring", flush=True)
        except Exception as e:
            print(f"[resume-self] failed: {e}", flush=True)

    if args.resume_from and os.path.isfile(args.resume_from):
        state = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model" in state:
            model.load_state_dict(state["model"], strict=False)
        else:
            model.load_state_dict(state, strict=False)

        src_meta_path = os.path.join(os.path.dirname(args.resume_from), "meta.json")
        src_stage = None
        src_gs = 0
        if os.path.isfile(src_meta_path):
            try:
                src_meta = json.load(open(src_meta_path))
                src_stage = src_meta.get("stage")
                src_gs = int(src_meta.get("global_step", 0))
            except Exception:
                pass

        if src_stage == args.stage:
            print(f"[resume-from] same stage {src_stage}, inheriting global={src_gs}", flush=True)
            return 0, 0, src_gs, False
        else:
            print(f"[resume-from] stage changed: {src_stage} -> {args.stage}, weights loaded, step reset to 0", flush=True)
            return 0, 0, 0, False

    return 0, 0, 0, False


def parse_int_tuple(s):
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def get_args():
    p = argparse.ArgumentParser(
        description="Dictionari Train — universal training template for Dictionari family",
    )

    p.add_argument("--stage", required=True, help="stage name for checkpoint tracking")
    p.add_argument("--dataset", required=True, help="HuggingFace dataset name")
    p.add_argument("--output_dir", required=True, help="output directory")
    p.add_argument("--resume_from", default="", help="path to model.pt or dim to resume from")
    p.add_argument("--init_from_dim", default="", help="initialize model from a dim file")

    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--seq_len", type=int, default=192)
    p.add_argument("--bsz", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--warmup_max", type=int, default=2000)
    p.add_argument("--aux_weight", type=float, default=3.0)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--max_minutes", type=int, default=0)

    p.add_argument("--lr_drop_target", type=float, default=3.0,
                   help="loss must reach this low to arm the auto-drop mechanism")
    p.add_argument("--lr_drop_ceiling", type=float, default=4.0,
                   help="loss >= this value resets the stabilization counter")
    p.add_argument("--lr_drop_window", type=int, default=20,
                   help="number of consecutive steps below ceiling to trigger a drop")
    p.add_argument("--lr_drop_factor", type=float, default=0.5,
                   help="lr multiplier on each drop")
    p.add_argument("--lr_drop_min_ratio", type=float, default=0.1,
                   help="lr_scale floor, cannot drop below base_lr * this")

    p.add_argument("--d_model", type=int, default=640)
    p.add_argument("--layer_experts", type=str, default="12,24,48")
    p.add_argument("--children_per_parent", type=int, default=2)
    p.add_argument("--hidden_mult", type=int, default=3)
    p.add_argument("--top_k", type=int, default=2)
    p.add_argument("--trunk_blocks", type=int, default=2)

    p.add_argument("--unfreeze_mode",
                   choices=["full", "dict-only", "dict+last-experts", "dict+experts", "all-experts"],
                   default="full")
    p.add_argument("--reinit_dict", action="store_true")
    p.add_argument("--reinit_dict_std", type=float, default=0.02)
    p.add_argument("--reinit_dict_seed", type=int, default=None)
    p.add_argument("--reinit_dict_perturb", type=float, default=0.0)

    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--save_dim_every", type=int, default=0)
    p.add_argument("--keep_optimizer_in_dim", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main():
    args = get_args()

    torch.manual_seed(args.seed)
    cpu_count = os.cpu_count() or 4
    torch.set_num_threads(cpu_count)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[env] Dictionari Train torch={torch.__version__} cpus={cpu_count} device={args.device}", flush=True)
    print(f"[args] {vars(args)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    roles = None
    meta_in = {}
    arch = None
    dim = None

    if args.init_from_dim:
        print(f"[init] loading dim: {args.init_from_dim}", flush=True)
        dim = load_dim(args.init_from_dim)
        arch = dim["arch"]
        meta_in = dim.get("meta", {})
        roles = dim.get("dictionari-model", None)
        d_model = int(arch["d_model"])
        vocab_size = int(arch["vocab_size"])
        layer_experts = tuple(int(x) for x in arch["layer_experts"])
        children_per_parent = int(arch.get("children_per_parent", 2))
        hidden_mult = int(arch.get("hidden_mult", 3))
        top_k = int(arch.get("top_k", 2))
        seq_len = int(arch.get("seq_len", args.seq_len))
    else:
        d_model = args.d_model
        vocab_size = len(tokenizer)
        layer_experts = parse_int_tuple(args.layer_experts)
        children_per_parent = args.children_per_parent
        hidden_mult = args.hidden_mult
        top_k = args.top_k
        seq_len = args.seq_len

    if args.resume_from and not args.init_from_dim:
        if args.resume_from.endswith(".dim"):
            dim = load_dim(args.resume_from)
            arch = dim["arch"]
            meta_in = dim.get("meta", {})
            roles = dim.get("dictionari-model", None)
            d_model = int(arch["d_model"])
            vocab_size = int(arch["vocab_size"])
            layer_experts = tuple(int(x) for x in arch["layer_experts"])
            children_per_parent = int(arch.get("children_per_parent", 2))
            hidden_mult = int(arch.get("hidden_mult", 3))
            top_k = int(arch.get("top_k", 2))
            seq_len = int(arch.get("seq_len", args.seq_len))

    model = DictionariModel(
        vocab_size=vocab_size,
        d=d_model,
        layer_experts=layer_experts,
        children_per_parent=children_per_parent,
        hidden_mult=hidden_mult,
        top_k=top_k,
        max_seq_len=seq_len,
        trunk_blocks=args.trunk_blocks,
    )

    if args.init_from_dim:
        missing, unexpected = model.load_state_dict(dim["model"], strict=False)
        print(f"[init] missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    elif args.resume_from and args.resume_from.endswith(".dim"):
        missing, unexpected = model.load_state_dict(dim["model"], strict=False)
        print(f"[init] missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    if arch is None:
        arch = model.arch_dict()

    if roles is None:
        roles = build_roles(model)
        print("[roles] auto-detected from model parameters", flush=True)
    else:
        print(f"[roles] loaded from dim, dict_layers={roles.get('dict_layers')} "
              f"expert_layers={roles.get('expert_layers')}", flush=True)

    configure_freeze(model, args.unfreeze_mode, roles)
    reinit_dict_layers(model, args)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] params={n_params/1e6:.2f}M trainable={n_trainable/1e6:.2f}M "
          f"mode={args.unfreeze_mode}", flush=True)

    model.to(args.device)
    model.train()

    ds = build_dataset(args.dataset, tokenizer, seq_len, args.max_samples)

    probe = make_loader(ds, args, epoch=0)
    steps_per_epoch = len(probe)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(int(total_steps * args.warmup_ratio), 1)
    warmup_steps = min(warmup_steps, args.warmup_max)
    print(f"[plan] steps_per_epoch={steps_per_epoch} epochs={args.epochs} "
          f"total_steps={total_steps} warmup={warmup_steps}", flush=True)

    start_epoch, start_epoch_step, global_step, done = try_resume(model, args)
    if done:
        print(f"[skip] {args.stage} already complete", flush=True)
        return

    opt = build_optimizer(model, args)

    lr_scale = 1.0
    hit_target = False
    smooth_steps = 0
    lr_drops = 0
    lr_drop_floor = args.lr * args.lr_drop_min_ratio

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
                print(f"[soft-stop] {args.max_minutes} minutes elapsed", flush=True)
                save_ckpt(model, arch, meta_in, roles, opt, args, epoch, epoch_step, global_step)
                return

            input_ids = batch["input_ids"].to(args.device)
            labels = batch["labels"].to(args.device)

            _, loss, aux = model(input_ids, labels=labels)
            loss_val = loss.item()

            if loss_val <= args.lr_drop_target:
                hit_target = True

            if hit_target:
                if loss_val >= args.lr_drop_ceiling:
                    smooth_steps = 0
                else:
                    smooth_steps += 1

                if smooth_steps >= args.lr_drop_window:
                    new_scale = lr_scale * args.lr_drop_factor
                    if new_scale * args.lr < lr_drop_floor:
                        new_scale = args.lr_drop_min_ratio
                    if new_scale < lr_scale:
                        lr_scale = new_scale
                        lr_drops += 1
                        print(
                            f"[lr-drop #{lr_drops}] loss stabilized for "
                            f"{args.lr_drop_window} steps below {args.lr_drop_ceiling}, "
                            f"lr_scale -> {lr_scale:.4f}",
                            flush=True,
                        )
                    smooth_steps = 0

            total = loss + args.aux_weight * aux
            (total / args.grad_accum).backward()
            micro += 1
            running_loss += loss_val
            running_aux += aux.item()

            if micro % args.grad_accum == 0:
                base_lr = lr_at(global_step, args.lr, warmup_steps, total_steps)
                lr = base_lr * lr_scale
                for g in opt.param_groups:
                    g["lr"] = lr

                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                opt.step()
                opt.zero_grad()
                epoch_step += 1
                global_step += 1

                if epoch_step % args.log_every == 0:
                    elapsed = time.time() - t0
                    tok_per_s = (args.log_every * args.grad_accum * args.bsz * seq_len) / max(elapsed, 1e-6)
                    print(
                        f"[Dictionari][{args.stage}] "
                        f"epoch={epoch+1}/{args.epochs} "
                        f"step={epoch_step}/{steps_per_epoch} "
                        f"global={global_step}/{total_steps} "
                        f"loss={running_loss/args.log_every:.4f} "
                        f"aux={running_aux/args.log_every:.4f} "
                        f"lr={lr:.2e} scale={lr_scale:.3f} tok/s={tok_per_s:.0f}",
                        flush=True,
                    )
                    running_loss = 0.0
                    running_aux = 0.0
                    t0 = time.time()

                if epoch_step % args.save_every == 0:
                    save_ckpt(model, arch, meta_in, roles, opt, args, epoch, epoch_step, global_step)

                if args.save_dim_every > 0 and epoch_step % args.save_dim_every == 0:
                    dim_path = os.path.join(args.output_dir, f"step-{global_step}.dim")
                    meta_out = dict(meta_in)
                    meta_out.update({
                        "stage": args.stage,
                        "completed_epochs": epoch,
                        "epoch_step": epoch_step,
                        "global_step": global_step,
                        "arch": arch,
                    })
                    save_dim(dim_path, model, arch, meta_out, roles)

        save_ckpt(model, arch, meta_in, roles, opt, args, epoch + 1, 0, global_step)
        start_epoch_step = 0

    meta_out = dict(meta_in)
    meta_out.update({
        "stage": args.stage,
        "completed_epochs": args.epochs,
        "global_step": global_step,
        "epochs": args.epochs,
        "complete": True,
        "arch": arch,
        "lr_drops": lr_drops,
        "final_lr_scale": lr_scale,
    })

    final_dim = os.path.join(args.output_dir, "final.dim")
    save_dim(final_dim, model, arch, meta_out, roles,
             optimizer=opt, keep_optimizer=args.keep_optimizer_in_dim)

    print(f"[done] stage={args.stage} global_step={global_step} lr_drops={lr_drops}", flush=True)


if __name__ == "__main__":
    main()