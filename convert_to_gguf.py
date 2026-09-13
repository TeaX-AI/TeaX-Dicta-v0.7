import argparse
import json
import os

import numpy as np
import torch
from gguf import GGUFWriter, GGMLQuantizationType


ARCH_NAME = "teax-dicta"


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


def infer_d_model(ckpt_model, meta_in):
    if meta_in.get("d_model"):
        return int(meta_in["d_model"])
    if "token_emb.weight" in ckpt_model:
        return int(ckpt_model["token_emb.weight"].shape[1])
    if "layers.0.router.weight" in ckpt_model:
        return int(ckpt_model["layers.0.router.weight"].shape[1])
    return 0


def infer_vocab_size(ckpt_model, meta_in):
    if meta_in.get("vocab_size"):
        return int(meta_in["vocab_size"])
    if "token_emb.weight" in ckpt_model:
        return int(ckpt_model["token_emb.weight"].shape[0])
    return 0


def infer_seq_len(ckpt_model, meta_in):
    if meta_in.get("seq_len"):
        return int(meta_in["seq_len"])
    if "pos_emb.weight" in ckpt_model:
        return int(ckpt_model["pos_emb.weight"].shape[0])
    return 0


def infer_hidden_mult(ckpt_model, meta_in, d_model):
    if meta_in.get("hidden_mult"):
        return int(meta_in["hidden_mult"])
    if "layers.0.W1" in ckpt_model and d_model > 0:
        hidden_dim = int(ckpt_model["layers.0.W1"].shape[-1])
        return hidden_dim // d_model
    return 0


def infer_layer_experts(ckpt_model, meta_in):
    if meta_in.get("layer_experts"):
        return [int(x) for x in meta_in["layer_experts"]]
    result = []
    i = 0
    while True:
        key = "layers.{}.W1".format(i)
        if key not in ckpt_model:
            break
        result.append(int(ckpt_model[key].shape[0]))
        i += 1
    return result


def infer_children_per_parent(meta_in):
    if meta_in.get("children_per_parent"):
        return int(meta_in["children_per_parent"])
    return 2


def infer_top_k(meta_in):
    if meta_in.get("top_k"):
        return int(meta_in["top_k"])
    return 2


def build_arch(ckpt_model, meta_in):
    d_model = infer_d_model(ckpt_model, meta_in)
    vocab_size = infer_vocab_size(ckpt_model, meta_in)
    seq_len = infer_seq_len(ckpt_model, meta_in)
    hidden_mult = infer_hidden_mult(ckpt_model, meta_in, d_model)
    layer_experts = infer_layer_experts(ckpt_model, meta_in)
    children_per_parent = infer_children_per_parent(meta_in)
    top_k = infer_top_k(meta_in)

    return {
        "d_model": d_model,
        "vocab_size": vocab_size,
        "seq_len": seq_len,
        "hidden_mult": hidden_mult,
        "layer_experts": layer_experts,
        "children_per_parent": children_per_parent,
        "top_k": top_k,
        "n_layers": len(layer_experts),
        "total_experts": int(sum(layer_experts)),
    }


def write_metadata(writer, arch, state, meta_in):
    writer.add_name("TeaX-Dicta-v0.7")
    writer.add_description("TeaX-Dicta v0.7 hierarchical dictionary MoE")

    writer.add_uint32("teax.d_model", arch["d_model"])
    writer.add_uint32("teax.vocab_size", arch["vocab_size"])
    writer.add_uint32("teax.seq_len", arch["seq_len"])
    writer.add_uint32("teax.hidden_mult", arch["hidden_mult"])
    writer.add_uint32("teax.children_per_parent", arch["children_per_parent"])
    writer.add_uint32("teax.top_k", arch["top_k"])
    writer.add_uint32("teax.n_layers", arch["n_layers"])
    writer.add_uint32("teax.total_experts", arch["total_experts"])

    stage = state.get("stage", meta_in.get("stage", "unknown"))
    global_step = state.get("global_step", meta_in.get("global_step", 0))
    completed_epochs = state.get("completed_epochs", meta_in.get("completed_epochs", 0))

    writer.add_string("teax.source_stage", str(stage))
    writer.add_uint32("teax.global_step", int(global_step))
    writer.add_uint32("teax.completed_epochs", int(completed_epochs))

    experts_as_strings = [str(int(x)) for x in arch["layer_experts"]]
    writer.add_array("teax.layer_experts", experts_as_strings)


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


def write_tensors(writer, ckpt_model, dtype_mode):
    names = list(ckpt_model.keys())
    total = len(names)
    print("[gguf] writing {} tensors".format(total), flush=True)

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
            print("[gguf] {}/{}".format(i + 1, total), flush=True)


def write_config_json(output_path, arch, state, meta_in, tensor_names):
    config_path = os.path.splitext(output_path)[0] + ".config.json"

    stage = state.get("stage", meta_in.get("stage", "unknown"))
    global_step = state.get("global_step", meta_in.get("global_step", 0))
    completed_epochs = state.get("completed_epochs", meta_in.get("completed_epochs", 0))

    config = {
        "model": "TeaX-Dicta-v0.7",
        "arch": ARCH_NAME,
        "format": "gguf-custom",
        "note": "custom hierarchical MoE arch; llama.cpp cannot load this directly",
        "d_model": arch["d_model"],
        "vocab_size": arch["vocab_size"],
        "seq_len": arch["seq_len"],
        "hidden_mult": arch["hidden_mult"],
        "layer_experts": arch["layer_experts"],
        "children_per_parent": arch["children_per_parent"],
        "top_k": arch["top_k"],
        "n_layers": arch["n_layers"],
        "total_experts": arch["total_experts"],
        "source_stage": stage,
        "global_step": int(global_step),
        "completed_epochs": int(completed_epochs),
        "tensor_names": tensor_names,
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print("[gguf] config written {}".format(config_path), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--meta", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", choices=["auto", "f32", "f16"], default="auto")
    args = parser.parse_args()

    state = torch.load(
        args.input,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )

    if not isinstance(state, dict):
        raise RuntimeError("checkpoint is not a dict")

    if "model" not in state:
        raise RuntimeError("checkpoint has no 'model' key")

    ckpt_model = state["model"]

    meta_in = {}
    if args.meta and os.path.isfile(args.meta):
        try:
            with open(args.meta) as f:
                meta_in = json.load(f)
        except Exception:
            meta_in = {}

    arch = build_arch(ckpt_model, meta_in)
    print("[gguf] arch detected: {}".format(arch), flush=True)

    writer = GGUFWriter(args.output, arch=ARCH_NAME)
    write_metadata(writer, arch, state, meta_in)
    write_tensors(writer, ckpt_model, args.dtype)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print("[gguf] gguf written {}".format(args.output), flush=True)

    tensor_names = [k for k in ckpt_model.keys() if isinstance(ckpt_model[k], torch.Tensor)]
    write_config_json(args.output, arch, state, meta_in, tensor_names)


if __name__ == "__main__":
    main()