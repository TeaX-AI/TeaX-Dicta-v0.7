import argparse
import json
import os

import torch


DIM_FORMAT = "teax-dicta-dim"
ROLES_FORMAT = "teax-dicta-model-roles"
DIM_VERSION = 1


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


def build_dictionari_model_json(model):
    roles = {}
    dict_layers = set()
    expert_layers = set()
    trunk_layers = set()

    for name, _ in model.named_parameters():
        role = classify_param(name)
        roles[name] = role

        if role == "dict" or role == "expert" or role == "trunk":
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
        "version": DIM_VERSION,
        "roles": roles,
        "dict_layers": sorted(dict_layers),
        "expert_layers": sorted(expert_layers),
        "trunk_layers": sorted(trunk_layers),
    }


def load_meta(args, state):
    meta = {}
    if args.meta and os.path.isfile(args.meta):
        try:
            with open(args.meta) as f:
                meta = json.load(f)
        except Exception:
            meta = {}

    for k in [
        "stage", "completed_epochs", "epoch_step", "global_step",
        "d_model", "layer_experts", "hidden_mult", "children_per_parent",
        "top_k", "seq_len", "vocab_size",
    ]:
        if k in state and k not in meta:
            meta[k] = state[k]

    return meta


def build_arch(meta):
    return {
        "d_model": int(meta.get("d_model", 0)),
        "layer_experts": list(meta.get("layer_experts", [])),
        "hidden_mult": int(meta.get("hidden_mult", 3)),
        "children_per_parent": int(meta.get("children_per_parent", 2)),
        "top_k": int(meta.get("top_k", 2)),
        "seq_len": int(meta.get("seq_len", 192)),
        "vocab_size": int(meta.get("vocab_size", 0)),
    }


def import_model_class():
    from DictionariReTrain import TeaXDictaV07
    return TeaXDictaV07


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="checkpoint .pt path")
    p.add_argument("--output", required=True, help="output .dim path")
    p.add_argument("--meta", default="", help="optional meta.json path")
    p.add_argument("--with-model-roles", action="store_true",
                   help="embed dictionari-model.json roles into the dim file")
    args = p.parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"input not found: {args.input}")

    state = torch.load(args.input, map_location="cpu", weights_only=False)

    if not isinstance(state, dict):
        raise RuntimeError("checkpoint is not a dict")
    if "model" not in state:
        raise RuntimeError("checkpoint has no 'model' key")

    meta = load_meta(args, state)
    arch = build_arch(meta)

    if not arch["d_model"] or not arch["vocab_size"] or not arch["layer_experts"]:
        raise RuntimeError(f"arch incomplete: {arch}")

    dim = {
        "format": DIM_FORMAT,
        "version": DIM_VERSION,
        "arch": arch,
        "model": state["model"],
        "meta": meta,
    }

    if args.with_model_roles:
        TeaXDictaV07 = import_model_class()
        model = TeaXDictaV07(
            vocab_size=arch["vocab_size"],
            d=arch["d_model"],
            layer_experts=tuple(arch["layer_experts"]),
            children_per_parent=arch["children_per_parent"],
            hidden_mult=arch["hidden_mult"],
            top_k=arch["top_k"],
            max_seq_len=arch["seq_len"],
        )
        model.load_state_dict(state["model"], strict=False)
        dim["dictionari-model"] = build_dictionari_model_json(model)
        print("[dim] dictionari-model roles embedded")

    torch.save(dim, args.output)

    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"[dim] arch: d={arch['d_model']} layer_experts={arch['layer_experts']} "
          f"hidden_mult={arch['hidden_mult']} top_k={arch['top_k']}")
    print(f"[dim] wrote {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()