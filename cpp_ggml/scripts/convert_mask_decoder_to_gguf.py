#!/usr/bin/env python3
"""Convert the released WeVisOne-VL mask decoder sidecar to GGUF.

Input:  mask_decoder.pt (state_dict + cfg, released with the checkpoints)
Output: wevisone-mask-decoder-<precision>.gguf for the C++ pixel decoder.

The converter bakes the layout decisions documented in
cpp_ggml/src/mask_decoder_graph.cpp:
  * torch (Cout, Cin, Kh, Kw) conv weights  -> ggml ne = (Kw, Kh, Cin, Cout)
  * depthwise (C, 1, 3, 3)                  -> (9, C) tap table (k = ky*3+kx)
  * torch Linear (out, in)                  -> ggml ne = (in, out) (2-D GEMM)
  * fuse 1x1 convs                          -> 2-D (out, in) GEMM weights
  * PixelShuffle / cascade weights          -> stored verbatim in F32

Tiers: f32 (the checkpoint dtype and the only parity-gated tier), f16, q8_0.
q4_K is intentionally NOT offered for the decoder: the Python forward runs
in fp32 and the GroupNorm/MSDA statistics are sensitive to 4-bit weights
(see models/MODEL_CARD.md).
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np
import torch


def ggml_dtype_tag(name: str) -> int:
    return {"f32": 0, "f16": 1, "q8_0": 8}[name]


def to_gguf_f32(name: str, t: torch.Tensor, precision: str) -> tuple[str, list[int], int, bytes]:
    """Returns (name, ne (ggml order), dtype tag, payload bytes)."""
    arr = t.detach().to(torch.float32).cpu().numpy()
    if precision == "f16":
        out = arr.astype(np.float16)
        payload = out.tobytes()
        tag = 1
    elif precision == "q8_0":
        flat = arr.reshape(-1)
        blocks = flat.reshape(-1, 32)
        scale = np.abs(blocks).max(axis=1, keepdims=True) / 127.0
        scale = np.where(scale == 0, 1.0, scale).astype(np.float16)
        q = np.rint(blocks / scale.astype(np.float32)).astype(np.int8)
        payload = b"".join(
            scale[i].tobytes() + q[i].tobytes() for i in range(q.shape[0])
        )
        tag = 8
    else:
        payload = arr.tobytes()
        tag = 0
    return name, list(arr.shape), tag, payload


def to_ggml_layout(t: torch.Tensor, name: str = "") -> tuple[torch.Tensor, list[int]]:
    """Standard GGUF layout contract: torch row-major bytes verbatim,
    ggml ne = reversed(torch shape) (ne[0] is the fastest axis).

    Covers every weight kind the decoder graph consumes without permutes:
      * Linear (out, in)                 -> ne = (in, out)  GEMM
      * 1x1 conv (out, in, 1, 1)         -> ne = (in, out)  GEMM
      * depthwise (C, 1, 3, 3)           -> reshape (C, 9)  -> ne = (9, C)
      * conv (Cout, Cin, Kh, Kw)         -> ne = (Kw, Kh, Cin, Cout)
      * norms / biases (C,)              -> ne = (C,)"""
    if t.dim() == 4 and t.shape[1] == 1 and t.shape[2] == 3 and t.shape[3] == 3 \
            and "depthwise" in name:
        # depthwise (C, 1, 3, 3) -> (C, 9) taps, k = ky*3 + kx (NAME-keyed:
        # the cascade rgb/coarse/previous convs share the (C, 1, 3, 3) shape
        # with in_channels=1 but are true convolutions).
        t = t.reshape(t.shape[0], 9)
    elif t.dim() == 4 and t.shape[2] == 1 and t.shape[3] == 1:
        # 1x1 conv -> 2-D GEMM (out, in); gguf ne = (in, out)
        t = t.reshape(t.shape[0], t.shape[1])
    # 4-D conv/convT weights keep the torch byte order (kw fastest):
    # ggml ne = (Kw, Kh, Cin, Cout) for conv and (Kw, Kh, Cout, Cin) for
    # ConvT — the reversed-shape rule produces both, for square kernels
    # byte-identically to the torch layout. No permutes anywhere.
    return t.contiguous(), list(reversed(t.shape))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidecar", required=True, help="mask_decoder.pt")
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--precision",
        choices=["f32", "f16", "q8_0"],
        default="f32",
        help="f32 is the checkpoint dtype and the parity-gated tier",
    )
    args = ap.parse_args()

    sidecar = torch.load(args.sidecar, map_location="cpu", weights_only=True)
    state = sidecar["state_dict"]
    cfg = sidecar.get("cfg", {})
    arch = cfg.get("architecture_version", "wevisone_vl_v1")
    if arch != "wevisone_vl_v1":
        print(f"unsupported decoder architecture: {arch}", file=sys.stderr)
        return 1

    tensors: list[tuple[str, list[int], int, bytes]] = []
    for name, t in sorted(state.items()):
        t, ne = to_ggml_layout(t, name)
        name_out, _torch_ne, tag, payload = to_gguf_f32(name, t, args.precision)
        tensors.append((name_out, ne, tag, payload))

    import sys
    sys.path.insert(0, "/home/asher/develop/code/dl/WeVisOne-VL-GGML/"
                       "cpp_ggml/third_party/llama.cpp/ggml/src/gguf-py")
    import gguf

    writer = gguf.GGUFWriter(args.output, "wevisone_mask")
    writer.add_string("general.architecture", "wevisone_mask")
    writer.add_string("general.quantization", args.precision)
    for k, v in cfg.items():
        if k == "vision_feature_layers":
            writer.add_array(f"wevisone.mask_decoder.{k}",
                             [int(x) for x in v])
        elif isinstance(v, str):
            writer.add_string(f"wevisone.mask_decoder.{k}", v)
        else:
            writer.add_uint32(f"wevisone.mask_decoder.{k}", int(v))
    for name, ne, tag, payload in tensors:
        # np shape = torch shape = reversed(ggml ne); ne[0] is the fastest
        np_shape = tuple(reversed(ne))
        if tag == 8:  # Q8_0 blocks: 2-byte fp16 scale + 32 int8
            n_blocks = len(payload) // 34
            raw = np.frombuffer(payload, dtype=np.uint8).reshape(n_blocks, 34)
            scales = raw[:, :2].copy().view(np.float16).astype(np.float32)
            qs = raw[:, 2:].copy().view(np.int8).astype(np.float32)
            arr = (qs * scales[:, None]).reshape(-1)
            writer.add_tensor(name, arr.reshape(np_shape),
                              raw_dtype=gguf.GGMLQuantizationType.Q8_0)
        elif tag == 1:
            arr = np.frombuffer(payload, dtype=np.float16)
            writer.add_tensor(name, arr.reshape(np_shape),
                              raw_dtype=gguf.GGMLQuantizationType.F16)
        else:
            arr = np.frombuffer(payload, dtype=np.float32)
            writer.add_tensor(name, arr.reshape(np_shape),
                              raw_dtype=gguf.GGMLQuantizationType.F32)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()
    total = sum(len(p_) // 4 for _, _, t_, p_ in tensors if t_ != 8)
    print(f"wrote {args.output}: {len(tensors)} tensors, tier={args.precision}")
    return 0


def check_name_contract(gguf_path: str) -> int:
    """GGUF names vs the C++ runtime mapping (wevisone_runtime.cpp): both
    directions must match exactly — a typo on either side is a silent
    missing-weight bug."""
    import sys
    sys.path.insert(0, "/home/asher/develop/code/dl/WeVisOne-VL-GGML/"
                       "cpp_ggml/third_party/llama.cpp/ggml/src/gguf-py")
    import gguf
    r = gguf.GGUFReader(gguf_path)
    names = {t.name for t in r.tensors}
    cfg = dict(cnn=1024, hidden=192, md=64, freqs=32, enc=3, lv=4, pts=4,
               tfm=6, heads=8, cc=48, blocks=2)
    expect = set()
    for l in range(cfg["lv"]):
        expect.add(f"visual_encoder.input_proj.{l}.0.weight")
        expect.add(f"visual_encoder.input_proj.{l}.1.weight")
        expect.add(f"visual_encoder.input_proj.{l}.1.bias")
    expect.add("visual_encoder.level_embed")
    for li in range(cfg["enc"]):
        p_ = f"visual_encoder.encoder.{li}."
        for k in ("depthwise.weight", "pointwise.weight", "norm1.weight",
                  "norm1.bias", "linear1.weight", "linear1.bias",
                  "linear2.weight", "linear2.bias", "norm2.weight",
                  "norm2.bias"):
            expect.add(p_ + k)
    qb = "query_builder."
    for k in ("content_proj.weight", "content_proj.bias",
              "content_norm.weight", "content_norm.bias",
              "box_position.mlp.0.weight", "box_position.mlp.0.bias",
              "box_position.mlp.2.weight", "box_position.mlp.2.bias",
              "position_norm.weight", "position_norm.bias"):
        expect.add(qb + k)
    for li in range(cfg["tfm"]):
        p_ = f"transformer.layers.{li}."
        for k in ("self_attn.in_proj_weight", "self_attn.in_proj_bias",
                  "self_attn.out_proj.weight", "self_attn.out_proj.bias",
                  "norm1.weight", "norm1.bias",
                  "cross_attn.value_proj.weight", "cross_attn.value_proj.bias",
                  "cross_attn.sampling_offsets.weight",
                  "cross_attn.sampling_offsets.bias",
                  "cross_attn.attention_weights.weight",
                  "cross_attn.attention_weights.bias",
                  "cross_attn.output_proj.weight", "cross_attn.output_proj.bias",
                  "norm2.weight", "norm2.bias",
                  "linear1.weight", "linear1.bias",
                  "linear2.weight", "linear2.bias", "norm3.weight", "norm3.bias"):
            expect.add(p_ + k)
    for l in range(cfg["lv"]):
        expect.add(f"pixel_fusion.fpn.lateral.{l}.weight")
    for i in range(cfg["lv"] - 1):
        p_ = f"pixel_fusion.fpn.output.{i}."
        expect.update({p_ + "0.weight", p_ + "1.weight", p_ + "1.bias"})
    pf = "pixel_fusion."
    for k in ("upsample.0.conv.weight", "upsample.0.conv.bias",
              "upsample.1.conv.weight", "upsample.1.conv.bias",
              "s2_upsample.conv.weight", "s2_upsample.conv.bias",
              "cnn.net.0.weight", "cnn.net.1.weight", "cnn.net.1.bias",
              "cnn.net.3.weight", "cnn.net.4.weight", "cnn.net.4.bias",
              "fpn_proj.weight", "cnn_proj.weight",
              "smooth.0.weight", "smooth.0.bias", "smooth.2.weight"):
        expect.add(pf + k)
    mp = "mask_predictor."
    for k in ("kernel.0.weight", "kernel.0.bias", "kernel.2.weight",
              "kernel.2.bias"):
        expect.add(mp + k)
    hr = "highres_refine."
    for nm in ("semantic", "rgb_s4", "rgb_s2", "rgb_s1", "coarse_s4",
               "coarse_s2", "previous_s2", "previous_s1"):
        expect.add(f"{hr}{nm}.0.weight")
        expect.add(f"{hr}{nm}.1.weight")
        expect.add(f"{hr}{nm}.1.bias")
    expect.update({hr + "fuse_s4.weight", hr + "fuse_s2.weight",
                   hr + "fuse_s1.weight", hr + "up_s4_to_s2.weight",
                   hr + "up_s2_to_s1.weight"})
    for s_ in ("s4", "s2", "s1"):
        expect.add(f"{hr}query_{s_}.weight")
        expect.add(f"{hr}query_{s_}.bias")
        expect.add(f"{hr}output_{s_}.weight")
        expect.add(f"{hr}output_{s_}.bias")
    for st in ("blocks_s4", "blocks_s2", "blocks_s1"):
        for b in range(cfg["blocks"]):
            p_ = f"{hr}{st}.{b}.net."
            for k in ("0.weight", "0.bias", "2.weight", "3.weight", "3.bias",
                      "5.weight"):
                expect.add(p_ + k)
    missing_in_gguf = sorted(expect - names)
    unexpected = sorted(names - expect)
    print(f"runtime expects {len(expect)}; gguf has {len(names)}")
    for n in missing_in_gguf:
        print("  MISSING in gguf:", n)
    for n in unexpected:
        print("  UNMAPPED in runtime:", n)
    print("NAME CONTRACT:", "PASS" if not missing_in_gguf and not unexpected
          else "FAIL")
    return 0 if not missing_in_gguf and not unexpected else 1


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) > 2 and _sys.argv[1] == "--check-names":
        raise SystemExit(check_name_contract(_sys.argv[2]))
    raise SystemExit(main())
