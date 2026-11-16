#!/usr/bin/env python3
"""Synthetic structurally-complete decoder GGUF (pure numpy, no torch).

Purpose: exercise the C++ decoder graph end-to-end (shapes, allocator,
compute) without the released checkpoint. Reduced cfg:
  S=128, hidden=32, mask_dim=16, cc=8, blocks=1, enc_layers=2, tfm_layers=2,
  heads=4, ffn=64, levels=4, points=4, gn=8, vit_channels=64, llm_dim=64.
Tensor names byte-match the runtime mapping (state_dict names).
"""
from __future__ import annotations

import argparse
import struct

import numpy as np

CFG = dict(cnn_image_size=128, hidden_dim=32, mask_feature_dim=16,
           num_fourier_freqs=8, msda_encoder_layers=2, msda_num_levels=4,
           msda_num_points=4, transformer_layers=2, transformer_heads=4,
           transformer_ffn_dim=64, group_norm_groups=8, stem_mid_dim=16,
           alpha_cascade_dim=8, alpha_cascade_blocks=1,
           vision_feature_layers=[5, 11, 17, 23],
           architecture_version="wevisone_vl_v1")
D, MD, CC, HD, LLM, VIT, GN, FFN = (32, 16, 8, 64, 64, 64, 8, 64)


def rng(shape, scale=0.02):
    return (np.random.default_rng(7).standard_normal(shape) * scale).astype(np.float32)


def tensors():
    t = {}

    def add(name, arr):
        t[name] = arr

    for l in range(4):
        add(f"visual_encoder.input_proj.{l}.0.weight", rng((D, VIT)))
        add(f"visual_encoder.input_proj.{l}.1.weight", np.ones(D, np.float32))
        add(f"visual_encoder.input_proj.{l}.1.bias", np.zeros(D, np.float32))
    add("visual_encoder.level_embed", rng((4, D)))
    for li in range(CFG["msda_encoder_layers"]):
        p = f"visual_encoder.encoder.{li}."
        add(p + "depthwise.weight", rng((D, 1, 3, 3)))
        add(p + "pointwise.weight", rng((D, D)))
        add(p + "norm1.weight", np.ones(D, np.float32))
        add(p + "norm1.bias", np.zeros(D, np.float32))
        add(p + "linear1.weight", rng((FFN, D)))
        add(p + "linear1.bias", np.zeros(FFN, np.float32))
        add(p + "linear2.weight", rng((D, FFN)))
        add(p + "linear2.bias", np.zeros(D, np.float32))
        add(p + "norm2.weight", np.ones(D, np.float32))
        add(p + "norm2.bias", np.zeros(D, np.float32))
    add("query_builder.content_proj.weight", rng((D, LLM)))
    add("query_builder.content_proj.bias", np.zeros(D, np.float32))
    add("query_builder.content_norm.weight", np.ones(D, np.float32))
    add("query_builder.content_norm.bias", np.zeros(D, np.float32))
    add("query_builder.box_position.mlp.0.weight", rng((D, 4 * 2 * CFG["num_fourier_freqs"])))
    add("query_builder.box_position.mlp.0.bias", np.zeros(D, np.float32))
    add("query_builder.box_position.mlp.2.weight", rng((D, D)))
    add("query_builder.box_position.mlp.2.bias", np.zeros(D, np.float32))
    add("query_builder.position_norm.weight", np.ones(D, np.float32))
    add("query_builder.position_norm.bias", np.zeros(D, np.float32))
    for li in range(CFG["transformer_layers"]):
        p = f"transformer.layers.{li}."
        add(p + "self_attn.in_proj_weight", rng((3 * D, D)) * 0.1)
        add(p + "self_attn.in_proj_bias", np.zeros(3 * D, np.float32))
        add(p + "self_attn.out_proj.weight", rng((D, D)))
        add(p + "self_attn.out_proj.bias", np.zeros(D, np.float32))
        add(p + "norm1.weight", np.ones(D, np.float32))
        add(p + "norm1.bias", np.zeros(D, np.float32))
        add(p + "cross_attn.value_proj.weight", rng((D, D)))
        add(p + "cross_attn.value_proj.bias", np.zeros(D, np.float32))
        add(p + "cross_attn.sampling_offsets.weight", rng((4 * 4 * 4 * 2, D)) * 0.01)
        add(p + "cross_attn.sampling_offsets.bias", np.zeros(4 * 4 * 4 * 2, np.float32))
        add(p + "cross_attn.attention_weights.weight", rng((4 * 4 * 4, D)) * 0.01)
        add(p + "cross_attn.attention_weights.bias", np.zeros(4 * 4 * 4, np.float32))
        add(p + "cross_attn.output_proj.weight", rng((D, D)))
        add(p + "cross_attn.output_proj.bias", np.zeros(D, np.float32))
        add(p + "norm2.weight", np.ones(D, np.float32))
        add(p + "norm2.bias", np.zeros(D, np.float32))
        add(p + "linear1.weight", rng((FFN, D)))
        add(p + "linear1.bias", np.zeros(FFN, np.float32))
        add(p + "linear2.weight", rng((D, FFN)))
        add(p + "linear2.bias", np.zeros(D, np.float32))
        add(p + "norm3.weight", np.ones(D, np.float32))
        add(p + "norm3.bias", np.zeros(D, np.float32))
    for l in range(4):
        add(f"pixel_fusion.fpn.lateral.{l}.weight", rng((D, D)))
    for i in range(3):
        p = f"pixel_fusion.fpn.output.{i}."
        add(p + "0.weight", rng((D, D, 3, 3)))
        add(p + "1.weight", np.ones(D, np.float32))
        add(p + "1.bias", np.zeros(D, np.float32))
    add("pixel_fusion.upsample.0.conv.weight", rng((D * 4, D, 3, 3)))
    add("pixel_fusion.upsample.0.conv.bias", np.zeros(D * 4, np.float32))
    add("pixel_fusion.upsample.1.conv.weight", rng((MD * 4, D, 3, 3)))
    add("pixel_fusion.upsample.1.conv.bias", np.zeros(MD * 4, np.float32))
    add("pixel_fusion.s2_upsample.conv.weight", rng((MD * 4, MD, 3, 3)))
    add("pixel_fusion.s2_upsample.conv.bias", np.zeros(MD * 4, np.float32))
    add("pixel_fusion.cnn.net.0.weight", rng((CFG["stem_mid_dim"], 3, 3, 3)))
    add("pixel_fusion.cnn.net.1.weight", np.ones(CFG["stem_mid_dim"], np.float32))
    add("pixel_fusion.cnn.net.1.bias", np.zeros(CFG["stem_mid_dim"], np.float32))
    add("pixel_fusion.cnn.net.3.weight", rng((MD, CFG["stem_mid_dim"], 3, 3)))
    add("pixel_fusion.cnn.net.4.weight", np.ones(MD, np.float32))
    add("pixel_fusion.cnn.net.4.bias", np.zeros(MD, np.float32))
    add("pixel_fusion.fpn_proj.weight", rng((MD, MD)))
    add("pixel_fusion.cnn_proj.weight", rng((MD, MD)))
    add("pixel_fusion.smooth.0.weight", np.ones(MD, np.float32))
    add("pixel_fusion.smooth.0.bias", np.zeros(MD, np.float32))
    add("pixel_fusion.smooth.2.weight", rng((MD, MD, 3, 3)))
    add("mask_predictor.kernel.0.weight", rng((D, D)))
    add("mask_predictor.kernel.0.bias", np.zeros(D, np.float32))
    add("mask_predictor.kernel.2.weight", rng((MD + 1, D)))
    add("mask_predictor.kernel.2.bias", np.zeros(MD + 1, np.float32))
    hr = "highres_refine."
    for name, cin in [("semantic", MD), ("rgb_s4", 3), ("rgb_s2", 3),
                      ("rgb_s1", 3), ("coarse_s4", 1), ("coarse_s2", 1),
                      ("previous_s2", 1), ("previous_s1", 1)]:
        add(f"{hr}{name}.0.weight", rng((CC, cin, 3, 3)))
        add(f"{hr}{name}.1.weight", np.ones(CC, np.float32))
        add(f"{hr}{name}.1.bias", np.zeros(CC, np.float32))
    add(hr + "fuse_s4.weight", rng((CC, 3 * CC)))
    add(hr + "fuse_s2.weight", rng((CC, 5 * CC)))
    add(hr + "fuse_s1.weight", rng((CC, 4 * CC)))
    add(hr + "up_s4_to_s2.weight", rng((CC, CC, 4, 4)))
    add(hr + "up_s2_to_s1.weight", rng((CC, CC, 4, 4)))
    for s in ("s4", "s2", "s1"):
        add(f"{hr}query_{s}.weight", rng((2 * CC, D)))
        add(f"{hr}query_{s}.bias", np.zeros(2 * CC, np.float32))
        add(f"{hr}output_{s}.weight", rng((1, CC)))
        add(f"{hr}output_{s}.bias", np.zeros(1, np.float32))
    for st, nm in enumerate(["blocks_s4", "blocks_s2", "blocks_s1"]):
        for b in range(CFG["alpha_cascade_blocks"]):
            p = f"{hr}{nm}.{b}.net."
            for k, gn_ch in ((0, CC), (3, CC)):
                add(f"{p}{k}.weight", np.ones(CC, np.float32))
                add(f"{p}{k}.bias", np.zeros(CC, np.float32))
            add(f"{p}2.weight", rng((CC, CC, 3, 3)))
            add(f"{p}5.weight", rng((CC, CC, 3, 3)))
    return t


def main() -> int:
    import sys
    sys.path.insert(0, "/home/asher/develop/code/dl/WeVisOne-VL-GGML/"
                       "cpp_ggml/third_party/llama.cpp/ggml/src/gguf-py")
    import gguf  # the vendored reference implementation — no hand-rolled format

    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    t = tensors()
    w = gguf.GGUFWriter(args.output, "wevisone_mask")
    for k, v in CFG.items():
        if k in ("vision_feature_layers",):
            continue  # the C++ graph uses the cfg default [5,11,17,23]
        elif isinstance(v, str):
            w.add_string(f"wevisone.mask_decoder.{k}", v)
        else:
            w.add_uint32(f"wevisone.mask_decoder.{k}", int(v))
    w.add_string("general.quantization", "f32")
    for name, arr in sorted(t.items()):
        if name.endswith("depthwise.weight"):
            # converter contract: (C,1,3,3) -> (C,9) taps -> gguf ne (9,C)
            arr = arr.reshape(arr.shape[0], 9)
        w.add_tensor(name, arr)  # torch-style shape; writer handles layout
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=False)
    w.close()
    print(f"synthetic decoder: {args.output} ({len(t)} tensors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
