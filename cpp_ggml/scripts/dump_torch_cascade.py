#!/usr/bin/env python3
"""Dump torch reference tensors for EVERY FullCanvasAlphaCascadeHead stage.

Reuses the stage dumps in --ref and replays the decoder with the cascade
forward manually expanded (mirrors mask_decoder.py lines 606-682 step by
step). Names match the C++ cas_* taps.

Usage:
  python3 cpp_ggml/scripts/dump_torch_cascade.py \
      --model tencent/WeVisOne-VL-2B --ref /tmp/wwv-ref --out /tmp/wwv-casref
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tencent/WeVisOne-VL-2B")
    ap.add_argument("--ref", default="/tmp/wwv-ref")
    ap.add_argument("--out", default="/tmp/wwv-casref")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from wevisone_vl.engine import WeVisOneVL

    ref = Path(args.ref)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def load(name):
        shape = tuple(int(x) for x in (ref / f"{name}.shape").read_text().split(","))
        return torch.from_numpy(
            np.frombuffer((ref / f"{name}.torch.bin").read_bytes(),
                          dtype=np.float32).reshape(shape).copy())

    def save(name, t):
        arr = t.detach().cpu().float().numpy()
        (out / f"{name}.torch.bin").write_bytes(
            np.ascontiguousarray(arr).tobytes())
        (out / f"{name}.shape").write_text(",".join(map(str, arr.shape)))
        print(f"  {name}: {tuple(arr.shape)}")

    engine = WeVisOneVL(args.model, device="cpu", dtype="float32")
    engine.cnn_size = int(load("cnn_image").shape[-1])

    cnn_image = load("cnn_image")[None]
    visual_maps = [load(f"level{i}")[None] for i in range(4)]
    t_seg = load("t_seg")
    boxes01 = load("boxes01")
    grid_hw = tuple(int(v) for v in load("grid_hw").reshape(-1))

    dec = engine.mask_decoder
    dec.to(device=engine.device, dtype=torch.float32).eval()
    grabbed = {}

    def grab_inputs(mod, inp):
        grabbed["args"] = [t.clone() for t in inp]

    handle = dec.highres_refine.register_forward_pre_hook(grab_inputs)
    with torch.inference_mode():
        engine.mask_decoder(image=cnn_image,
                            visual_features_2d=visual_maps, t_seg=t_seg,
                            boxes01=boxes01, grid_hw=grid_hw)
    handle.remove()

    image, pixel_feature, queries = grabbed["args"][0:3]
    coarse_logits = grabbed["args"][4]
    head = dec.highres_refine
    nq = int(queries.shape[0])
    s2_hw = tuple(int(x) for x in coarse_logits.shape[-2:])
    s1_hw = tuple(int(x) for x in image.shape[-2:])
    s4_hw = (max(1, s2_hw[0] // 2), max(1, s2_hw[1] // 2))

    def resize(value, size):
        # FullCanvasAlphaCascadeHead._resize: bilinear for ANY size change
        if tuple(value.shape[-2:]) == tuple(size):
            return value
        return F.interpolate(value.float(), size=size, mode="bilinear",
                             align_corners=False).to(dtype=value.dtype)

    def condition(feat, affine_out):
        scale, bias = affine_out.chunk(2, dim=-1)
        return feat * (1.0 + 0.1 * scale.tanh()[:, :, None, None]) \
            + 0.1 * bias[:, :, None, None]

    with torch.inference_mode():
        semantic_s2 = head.semantic(pixel_feature)
        save("cas_semantic_s2", semantic_s2)
        semantic_s4 = resize(semantic_s2, s4_hw)
        save("cas_semantic_s4", semantic_s4)
        rgb4 = head.rgb_s4(resize(image, s4_hw))
        save("cas_rgb4", rgb4)
        rgb2 = head.rgb_s2(resize(image, s2_hw))
        save("cas_rgb2", rgb2)
        rgb1 = head.rgb_s1(image)
        save("cas_rgb1", rgb1)
        coarse4_in = resize(coarse_logits[:, None], s4_hw)[:, 0]
        save("cas_coarse4_in", coarse4_in)
        coarse4_p = head.coarse_s4(coarse4_in[:, None].to(semantic_s4.dtype))
        save("cas_coarse4_p", coarse4_p)
        coarse2_p = head.coarse_s2(coarse_logits[:, None].to(semantic_s2.dtype))
        save("cas_coarse2_p", coarse2_p)

        fuse4 = head.fuse_s4(torch.cat([
            semantic_s4.expand(nq, -1, -1, -1),
            rgb4.expand(nq, -1, -1, -1),
            coarse4_p,
        ], dim=1))
        save("cas_fuse4", fuse4)
        sb4 = head.query_s4(queries)
        save("cas_q4_sb", sb4)
        fuse4c = condition(fuse4, sb4)
        save("cas_fuse4c", fuse4c)
        feat4 = head.blocks_s4(fuse4c)
        save("cas_feat4", feat4)
        delta4 = head.output_s4(feat4)[:, 0].float()
        save("cas_delta4", delta4)
        logits_s4 = coarse4_in.float() + delta4
        save("cas_logits_s4", logits_s4)

        up42 = head.up_s4_to_s2(feat4)
        save("cas_up42", up42)
        prev2_raw = resize(logits_s4[:, None], s2_hw)
        save("cas_prev2_raw", prev2_raw)
        prev2 = head.previous_s2(prev2_raw.sigmoid().to(semantic_s2.dtype))
        save("cas_prev2", prev2)
        fuse2 = head.fuse_s2(torch.cat([
            up42,
            semantic_s2.expand(nq, -1, -1, -1),
            rgb2.expand(nq, -1, -1, -1),
            coarse2_p,
            prev2,
        ], dim=1))
        save("cas_fuse2", fuse2)
        sb2 = head.query_s2(queries)
        save("cas_q2_sb", sb2)
        fuse2c = condition(fuse2, sb2)
        save("cas_fuse2c", fuse2c)
        f2 = head.blocks_s2(fuse2c)
        save("cas_f2", f2)
        delta2 = head.output_s2(f2)[:, 0].float()
        save("cas_delta2", delta2)
        logits_s2 = coarse_logits.float() + delta2
        save("cas_logits_s2", logits_s2)

        up21 = head.up_s2_to_s1(f2)
        save("cas_up21", up21)
        semantic_s1 = resize(semantic_s2, s1_hw)
        save("cas_semantic_s1", semantic_s1)
        prev1_raw = resize(logits_s2[:, None], s1_hw)
        save("cas_prev1_raw", prev1_raw)
        prev1 = head.previous_s1(prev1_raw.sigmoid().to(semantic_s1.dtype))
        save("cas_prev1", prev1)
        fuse1 = head.fuse_s1(torch.cat([
            up21,
            semantic_s1.expand(nq, -1, -1, -1),
            rgb1.expand(nq, -1, -1, -1),
            prev1,
        ], dim=1))
        save("cas_fuse1", fuse1)
        sb1 = head.query_s1(queries)
        save("cas_q1_sb", sb1)
        fuse1c = condition(fuse1, sb1)
        save("cas_fuse1c", fuse1c)
        f1 = head.blocks_s1(fuse1c)
        save("cas_f1", f1)
        delta1 = head.output_s1(f1)[:, 0].float()
        save("cas_delta1", delta1)
        save("cas_logits_s1", prev1_raw[:, 0].float() + delta1)
    print("dump complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
