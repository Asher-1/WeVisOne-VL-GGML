#!/usr/bin/env python3
"""Side-by-side effect panels: input | python | cpp per case (PNG triple)."""
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
    for js in sorted((results / "fixed" / "cpp").glob("*.json")):
        stem = js.stem
        panels = [images / f"{stem.split('_')[0]}.jpg",
                  results / "fixed" / "python" / f"{stem}.png",
                  results / "fixed" / "cpp" / f"{stem}.png"]
        panels = [p for p in panels if p.exists()]
        if len(panels) < 2:
            continue
        fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
        axes = axes if isinstance(axes, list) else [axes]
        names = ["input", "python", "cpp"]
        for ax, p, name in zip(axes, panels, names):
            ax.imshow(mpimg.imread(p))
            ax.set_title(name, fontsize=10)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out / f"case_{stem}_cpp_py.png", dpi=150)
        plt.close(fig)
        print(f"case_{stem}_cpp_py.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
