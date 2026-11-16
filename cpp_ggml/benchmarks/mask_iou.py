#!/usr/bin/env python3
"""Mask-quality gate: the cpp matte vs the official pytorch matte.

Both stacks dump the raw first-matte alpha plane as identical f32 layouts
(cpp CLI --matte-out / python adapter --out-matte, original-image
resolution, row-major h*w + an "h,w" shape sidecar). This script loads
each (size, case) pair, aligns shapes, and reports:
  - binary IoU at threshold 0.5
  - soft agreement: mean |a - b| and max |a - b|
The band follows the sibling d10 mask gate: binary IoU >= 0.9 (the two
stacks share weights and inputs; residual comes from the bilinear
upsample and f16 VLM feature noise feeding the f32 decoder).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CASES = ("seabreeze_matting",)


def load_matte(p: Path) -> np.ndarray:
    h, w = p.with_suffix(p.suffix + ".shape").read_text().split(",")
    a = np.fromfile(p, dtype=np.float32)
    return a.reshape(int(h), int(w))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--sizes", default="2B,4B")
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--iou-band", type=float, default=0.9)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    results = Path(args.results)
    ok_all = True
    rows = []
    for size in args.sizes.split(","):
        for case in args.cases.split(","):
            ref = (results / "fixed" / size / "python-cuda"
                   / f"{case}.matte.bin")
            if not ref.exists():
                print(f"skip {size}/{case}: python matte missing")
                continue
            a = load_matte(ref)
            for backend in ("cpp-cuda", "cpp-vulkan"):
                got = results / "fixed" / size / backend / f"{case}.matte.bin"
                if not got.exists():
                    rows.append({"size": size, "case": case,
                                 "backend": backend, "verdict": "MISSING"})
                    ok_all = False
                    continue
                b = load_matte(got)
                if b.shape != a.shape:
                    rows.append({"size": size, "case": case,
                                 "backend": backend,
                                 "verdict": "SHAPE_MISMATCH"})
                    ok_all = False
                    continue
                am, bm = a >= args.threshold, b >= args.threshold
                inter = np.logical_and(am, bm).sum()
                union = np.logical_or(am, bm).sum()
                iou = float(inter / union) if union else 1.0
                mad = float(np.abs(a - b).mean())
                mxd = float(np.abs(a - b).max())
                verdict = "PASS" if iou >= args.iou_band else "FAIL"
                if verdict == "FAIL":
                    ok_all = False
                rows.append({"size": size, "case": case, "backend": backend,
                             "mask_iou": round(iou, 4),
                             "mean_abs_diff": round(mad, 5),
                             "max_abs_diff": round(mxd, 5),
                             "fg_frac_ref": round(float(am.mean()), 4),
                             "fg_frac_got": round(float(bm.mean()), 4),
                             "verdict": verdict})
                print(f"[{verdict}] {size}/{backend:<12} {case:<22} "
                      f"IoU {iou:.4f}  mean|a-b| {mad:.5f}  "
                      f"max|a-b| {mxd:.4f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"rows": rows, "band": args.iou_band,
         "threshold": args.threshold}, indent=2))
    print("MASK-QUALITY:", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
