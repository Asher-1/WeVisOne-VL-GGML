#!/usr/bin/env python3
"""Side-by-side effect panels per case (PNG triple).

Layout: results/fixed/<size>/<backend>/<case>.png written by
run_benchmark.sh — one panel per (size, backend) with the input photo on
the left. Backends present in the tree are picked up automatically
(cpp-cuda / cpp-vulkan / python-cuda).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.image as mpimg  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    results, images, out = Path(args.results), Path(args.images), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # authoritative case -> image map (stems cannot be guessed: e.g.
    # coffee_sign_ocr comes from coffee_sign.jpg)
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from cases import CASES
    case_image = {f"{Path(img).stem}_{task}": img
                  for img, task, _q, _l in CASES}
    fixed = results / "fixed"
    for size_dir in sorted(p for p in fixed.iterdir() if p.is_dir()):
        size = size_dir.name
        backends = sorted(p.name for p in size_dir.iterdir() if p.is_dir())
        if not backends:
            continue
        cases = sorted(p.stem for p in (size_dir / backends[0]).glob("*.json"))
        for case in cases:
            panels = [(images / case_image[case], "input")]
            panels += [(size_dir / b / f"{case}.png", b.replace("cpp-", "cpp "))
                       for b in backends
                       if (size_dir / b / f"{case}.png").exists()]
            if len(panels) < 2:
                continue
            fig, axes = plt.subplots(
                1, len(panels), figsize=(4 * len(panels), 4),
                squeeze=False)
            for ax, (p, name) in zip(axes.flat, panels):
                ax.imshow(mpimg.imread(p))
                ax.set_title(name, fontsize=9)
                ax.axis("off")
            fig.suptitle(f"{case} ({size})", fontsize=11)
            fig.tight_layout()
            fig.savefig(out / f"case_{size}_{case}.png", dpi=150)
            plt.close(fig)
            print(f"case_{size}_{case}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
