#!/usr/bin/env python3
"""Dump torch reference tensors for EVERY AdditivePixelFusion submodule.

Reuses the stage dumps in --ref (cnn_image / level0-3 / t_seg / boxes01 /
grid_hw) so the LLM pass is skipped entirely, then replays the decoder with
forward hooks on the pixel-fusion submodules. Names match the C++ pix_*
taps.

Usage:
  python3 cpp_ggml/scripts/dump_torch_pixel.py \
      --model tencent/WeVisOne-VL-2B --ref /tmp/wwv-ref --out /tmp/wwv-pixref
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tencent/WeVisOne-VL-2B")
    ap.add_argument("--ref", default="/tmp/wwv-ref")
    ap.add_argument("--out", default="/tmp/wwv-pixref")
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
    tap_out = {}

    def hook(name):
        def fn(mod, inp, outp):
            tap_out[name] = outp
        return fn

    pf = dec.pixel_fusion
    handles = [
        pf.fpn.register_forward_hook(hook("pix_fpn")),
        pf.upsample.register_forward_hook(hook("pix_up")),
        pf.fpn_proj.register_forward_hook(hook("pix_fpn_proj")),
        pf.smooth.register_forward_hook(hook("pix_smooth")),
        pf.s2_upsample.register_forward_hook(hook("pix_s2up")),
        pf.cnn.register_forward_hook(hook("pix_cnn")),
        pf.cnn_proj.register_forward_hook(hook("pix_cnn_proj")),
        pf.register_forward_hook(hook("pixel_fusion")),
    ]
    def in_hook(name):
        def fn(mod, inp, outp):
            tap_out[name] = inp[0]
        return fn

    for i, lat in enumerate(pf.fpn.lateral):
        handles.append(lat.register_forward_hook(hook(f"pix_lat{i}")))
        handles.append(lat.register_forward_hook(in_hook(f"pix_lat{i}_in")))
    for i, outp in enumerate(pf.fpn.output):
        handles.append(outp.register_forward_hook(hook(f"pix_out{i}")))
        handles.append(outp.register_forward_hook(in_hook(f"pix_out{i}_in")))
    with torch.inference_mode():
        engine.mask_decoder(image=cnn_image,
                            visual_features_2d=visual_maps, t_seg=t_seg,
                            boxes01=boxes01, grid_hw=grid_hw)
    for h in handles:
        h.remove()
    for k in ("pix_fpn", "pix_up", "pix_fpn_proj", "pix_smooth", "pix_s2up",
              "pix_cnn", "pix_cnn_proj", "pixel_fusion",
              "pix_lat0", "pix_lat1", "pix_lat2", "pix_lat3",
              "pix_lat0_in", "pix_lat1_in", "pix_lat2_in", "pix_lat3_in",
              "pix_out0", "pix_out1", "pix_out2"):
        v = tap_out[k]
        save(f"tap_{k}", v[0] if isinstance(v, tuple) else v)
    print("dump complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
