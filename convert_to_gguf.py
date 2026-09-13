import argparse
import json
import numpy as np
import torch
from gguf import GGUFWriter, GGMLQuantizationType


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--meta", default="")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    state = torch.load(args.input, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    meta = {}
    if args.meta:
        try:
            meta = json.load(open(args.meta))
        except Exception:
            meta = {}

    writer = GGUFWriter(args.output, arch="teax-dicta")
    writer.add_name("TeaX-Dicta-v0.7")
    writer.add_description("TeaX-Dicta v0.7 converted from PyTorch")
    if meta:
        writer.add_uint32("teax.d_model", int(meta.get("d_model", 0)))
        writer.add_uint32("teax.n_layers", int(meta.get("n_layers", 0)))
        writer.add_uint32("teax.n_experts", int(meta.get("n_experts", 0)))
        writer.add_uint32("teax.seq_len", int(meta.get("seq_len", 0)))
        writer.add_uint32("teax.vocab_size", int(meta.get("vocab_size", 0)))
        writer.add_uint32("teax.global_step", int(meta.get("global_step", 0)))
        writer.add_uint32("teax.completed_epochs", int(meta.get("completed_epochs", 0)))

    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        arr = tensor.detach().cpu().numpy()
        if arr.dtype == np.float32:
            qtype = GGMLQuantizationType.F32
        elif arr.dtype == np.float16:
            qtype = GGMLQuantizationType.F16
        elif arr.dtype in (np.int32, np.int64):
            arr = arr.astype(np.int32)
            qtype = GGMLQuantizationType.I32
        else:
            arr = arr.astype(np.float32)
            qtype = GGMLQuantizationType.F32
        writer.add_tensor(name, arr, raw_dtype=qtype)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"[gguf] written {args.output}")


if __name__ == "__main__":
    main()