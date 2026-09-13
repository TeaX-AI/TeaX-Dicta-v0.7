import argparse
import json
import os

import numpy as np
import torch
from gguf import GGUFWriter, GGMLQuantizationType


ARCH_NAME = "teax-dicta"
DIM_FORMAT = "teax-dicta-dim"


def tensor_dtype_to_ggml(tensor):
    if tensor.dtype == torch.float32:
        return GGMLQuantizationType.F32
    if tensor.dtype == torch.float16:
        return GGMLQuantizationType.F16
    if tensor.dtype == torch.bfloat16:
        return GGMLQuantizationType.BF16
    if tensor.dtype == torch.int32:
        return GGMLQuantizationType.I32
    if tensor.dtype == torch.int64:
        return GGMLQuantizationType.I32
    if tensor.dtype == torch.bool:
        return GGMLQuantizationType.BOOL
    return GGMLQuantizationType.F32


def should_cast_f16(name, tensor):
    if tensor.dtype != torch.float32:
        return False
    if "token_emb" in name:
        return False
    if "lm_head" in name:
        return False
    if "norm" in name:
        return False
    if tensor.numel() < 4096:
        return False
    return True


def cast_tensor(name, tensor, dtype_mode):
    if dtype_mode == "f16":
        if tensor.dtype == torch.float32:
            return tensor.half()
        return tensor
    if dtype_mode == "f32":
        if tensor.dtype != torch.float32:
            return tensor.float()
        return tensor
    if should_cast_f16(name, tensor):
        return tensor.half()
    return tensor


def load_from_dim(dim_path):
    dim = torch.load(dim_path, map_location="cpu", weights_only=False)
    if not isinstance(dim, dict):
        raise RuntimeError("dim file is not a dict")
    if dim.get("format") != DIM_FORMAT:
        raise RuntimeError(f"not a dim file: format={dim.get('format')}")
    if "model" not in dim:
        raise RuntimeError("dim file has no 'model' key")

    arch = dim.get("arch", {})
    meta = dim.get("meta", {})
    roles = dim.get("dictionari-model", None)

    return {
        "model": dim["model"],
        "arch": arch,
        "meta": meta,
        "roles": roles,
        "stage": meta.get("stage", "unknown"),
        "global_step": int(meta.get("global_step", 0)),
        "completed_epochs": int(meta.get("completed_epochs", 0)),
    }


def load_from_pt(pt_path, meta_path):
    state = torch.load(pt_path, map_location="cpu", mmap=True, weights_only=False)
    if not isinstance(state, dict):
        raise RuntimeError("pt file is not a dict")
    if "model" not in state:
        raise RuntimeError("pt file has no 'model' key")

    meta = {}
    if meta_path and os.path.isfile(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception:
            meta = {}

    return {
        "model": state["model"],
        "arch": meta,
        "meta": meta,
        "roles": None,
        "stage": state.get("stage", meta.get("stage", "unknown")),
        "global_step": int(state.get("global_step", meta.get("global_step", 0))),
        "completed_epochs": int(state.get("completed_epochs", meta.get("completed_epochs", 0))),
    }


def write_metadata(writer, arch, payload):
    writer.add_name("TeaX-Dicta-v0.7")
    writer.add_description("TeaX-Dicta v0.7 hierarchical dictionary MoE")

    writer.add_uint32("teax.d_model", int(arch.get("d_model", 0)))
    writer.add_uint32("teax.vocab_size", int(arch.get("vocab_size", 0)))
    writer.add_uint32("teax.seq_len", int(arch.get("seq_len", 0)))
    writer.add_uint32("teax.hidden_mult", int(arch.get("hidden_mult", 0)))
    writer.add_uint32("teax.children_per_parent", int(arch.get("children_per_parent", 2)))
    writer.add_uint32("teax.top_k", int(arch.get("top_k", 2)))

    layer_experts = list(arch.get("layer_experts", []))
    writer.add_uint32("teax.n_layers", len(layer_experts))
    writer.add_uint32("teax.total_experts", int(sum(layer_experts)))
    writer.add_array("teax.layer_experts", [str(int(x)) for x in layer_experts])

    writer.add_string("teax.source_stage", str(payload["stage"]))
    writer.add_uint32("teax.global_step", int(payload["global_step"]))
    writer.add_uint32("teax.completed_epochs", int(payload["completed_epochs"]))


def write_tensors(writer, ckpt_model, dtype_mode):
    names = list(ckpt_model.keys())
    total = len(names)
    print(f"[gguf] writing {total} tensors", flush=True)

    for i, name in enumerate(names):
        tensor = ckpt_model[name]
        if not isinstance(tensor, torch.Tensor):
            continue

        tensor = tensor.detach().cpu()
        tensor = cast_tensor(name, tensor, dtype_mode)

        arr = np.ascontiguousarray(tensor.numpy())
        qtype = tensor_dtype_to_ggml(tensor)
        writer.add_tensor(name, arr, raw_dtype=qtype)

        del arr
        del tensor

        if (i + 1) % 25 == 0:
            print(f"[gguf] {i + 1}/{total}", flush=True)


def write_config_json(output_path, arch, payload, tensor_names, roles):
    config_path = os.path.splitext(output_path)[0] + ".config.json"

    config = {
        "model": "TeaX-Dicta-v0.7",
        "arch": ARCH_NAME,
        "format": "gguf-custom",
        "note": "custom hierarchical MoE arch; llama.cpp cannot load this directly",
        "d_model": arch.get("d_model", 0),
        "vocab_size": arch.get("vocab_size", 0),
        "seq_len": arch.get("seq_len", 0),
        "hidden_mult": arch.get("hidden_mult", 0),
        "layer_experts": arch.get("layer_experts", []),
        "children_per_parent": arch.get("children_per_parent", 2),
        "top_k": arch.get("top_k", 2),
        "n_layers": len(arch.get("layer_experts", [])),
        "total_experts": int(sum(arch.get("layer_experts", []))),
        "source_stage": payload["stage"],
        "global_step": int(payload["global_step"]),
        "completed_epochs": int(payload["completed_epochs"]),
        "tensor_names": tensor_names,
    }

    if roles is not None:
        config["dictionari-model"] = roles

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"[gguf] config written {config_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="checkpoint .pt path")
    src.add_argument("--dim", help="dim file path")

    parser.add_argument("--meta", default="", help="meta.json path (only with --input)")
    parser.add_argument("--output", required=True, help="output .gguf path")
    parser.add_argument("--dtype", choices=["auto", "f32", "f16"], default="auto")
    args = parser.parse_args()

    if args.dim:
        print(f"[gguf] loading dim: {args.dim}", flush=True)
        payload = load_from_dim(args.dim)
    else:
        print(f"[gguf] loading pt: {args.input}", flush=True)
        payload = load_from_pt(args.input, args.meta)

    arch = payload["arch"]
    ckpt_model = payload["model"]

    print(f"[gguf] arch: d={arch.get('d_model')} vocab={arch.get('vocab_size')} "
          f"layer_experts={arch.get('layer_experts')} "
          f"hidden_mult={arch.get('hidden_mult')} top_k={arch.get('top_k')}")

    writer = GGUFWriter(args.output, arch=ARCH_NAME)
    write_metadata(writer, arch, payload)
    write_tensors(writer, ckpt_model, args.dtype)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"[gguf] gguf written {args.output}", flush=True)

    tensor_names = [k for k in ckpt_model.keys() if isinstance(ckpt_model[k], torch.Tensor)]
    write_config_json(args.output, arch, payload, tensor_names, payload["roles"])


if __name__ == "__main__":
    main()