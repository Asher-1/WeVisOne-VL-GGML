#!/usr/bin/env python3
"""Elementwise parity gate for the C++ port (sibling-integration band).

Pass criterion: |cpp - torch| <= max_abs + max_rel * |torch|. f32 tier
defaults 0.05/0.05 (hand-tightened per stage); f16 tier 0.5/0.05.
Input layout: <dir>/<stage>.torch.bin, <dir>/<stage>.cpp.bin, <dir>/<stage>.shape
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", help="dump directory with .torch/.cpp/.shape files")
    ap.add_argument("--max-abs", type=float, default=0.05)
    ap.add_argument("--max-rel", type=float, default=0.05)
    ap.add_argument(
        "--band-override", default="",
        help="per-stage band overrides, e.g. 'tap_cas_prev1=0.2:0.05'")
    args = ap.parse_args()
    bands = {}
    for spec in args.band_override.split(","):
        if not spec:
            continue
        name, band = spec.split("=")
        bands[name] = tuple(float(v) for v in band.split(":"))
    d = Path(args.dir)
    ok_all = True
    for shape_file in sorted(d.glob("*.shape")):
        stage = shape_file.name[: -len(".shape")]
        torch_bin = d / f"{stage}.torch.bin"
        cpp_bin = d / f"{stage}.cpp.bin"
        if torch_bin.exists() and not cpp_bin.exists():
            # the torch dump list is the authoritative gate roster: a missing
            # C++ side must fail loudly (a silent skip once masked an entire
            # prefix-mismatched tap family as all-green)
            print(f"{stage}: MISSING cpp side")
            ok_all = False
            continue
        if not torch_bin.exists() or not cpp_bin.exists():
            continue
        shape = tuple(int(x) for x in shape_file.read_text().split(","))
        expected = int(np.prod(shape)) if shape else 0
        a_raw = np.frombuffer(torch_bin.read_bytes(), dtype=np.float32)
        b_raw = np.frombuffer(cpp_bin.read_bytes(), dtype=np.float32)
        if expected != a_raw.size or expected != b_raw.size:
            # stale shape file vs newer bins: report instead of crashing so
            # one mismatched stage cannot hide the rest of the gate
            print(f"{stage}: SHAPE/BIN MISMATCH shape={shape} "
                  f"torch={a_raw.size} cpp={b_raw.size}")
            ok_all = False
            continue
        a = a_raw.reshape(shape)
        b = b_raw.reshape(shape)
        if a.size != b.size:
            print(f"{stage}: SIZE MISMATCH {a.size} vs {b.size}")
            ok_all = False
            continue
        max_abs, max_rel = bands.get(stage, (args.max_abs, args.max_rel))
        diff = np.abs(a - b)
        band = max_abs + max_rel * np.abs(a)
        violations = int((diff > band).sum())
        print(f"{stage}: mean_abs={diff.mean():.3e} max_abs={diff.max():.3e} "
              f"violations={violations}/{a.size}")
        ok_all = ok_all and violations == 0
    print("GATE:", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
