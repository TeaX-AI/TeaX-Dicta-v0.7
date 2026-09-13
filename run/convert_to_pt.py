import argparse
import json
import os

import torch


DIM_FORMAT = "teax-dicta-dim"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="dim file path")
    p.add_argument("--output-dir", required=True,
                   help="directory to write model.pt and meta.json into")
    p.add_argument("--with-roles", action="store_true",
                   help="also write dictionari-model.json")
    args = p.parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"dim not found: {args.input}")

    dim = torch.load(args.input, map_location="cpu", weights_only=False)

    if not isinstance(dim, dict):
        raise RuntimeError("dim file is not a dict")
    if dim.get("format") != DIM_FORMAT:
        raise RuntimeError(f"not a dim file: format={dim.get('format')}")
    if "model" not in dim:
        raise RuntimeError("dim file has no 'model' key")

    os.makedirs(args.output_dir, exist_ok=True)

    state = {"model": dim["model"]}
    meta = dict(dim.get("meta", {}))
    arch = dim.get("arch", {})

    for k in [
        "d_model", "layer_experts", "hidden_mult", "children_per_parent",
        "top_k", "seq_len", "vocab_size",
    ]:
        if k in arch and k not in meta:
            meta[k] = arch[k]

    for k in ["stage", "completed_epochs", "epoch_step", "global_step"]:
        if k in dim and k not in meta:
            meta[k] = dim[k]

    model_path = os.path.join(args.output_dir, "model.pt")
    meta_path = os.path.join(args.output_dir, "meta.json")

    torch.save(state, model_path)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[pt] wrote {model_path}")
    print(f"[pt] wrote {meta_path}")

    if args.with_roles and "dictionari-model" in dim:
        roles_path = os.path.join(args.output_dir, "dictionari-model.json")
        with open(roles_path, "w") as f:
            json.dump(dim["dictionari-model"], f, indent=2)
        print(f"[pt] wrote {roles_path}")


if __name__ == "__main__":
    main()