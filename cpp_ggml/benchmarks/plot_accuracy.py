#!/usr/bin/env python3
"""Renders the accuracy comparison chart from results/accuracy_summary.json
(written by accuracy_summary.py): matched-IoU bars per case, one panel per
(size, backend), with the point cases shown as mean token distance.
Output: charts/accuracy_iou.png.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data = json.loads(Path(args.json).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # the f16 parity tier is the accuracy figure of merit; the quantized
    # tiers (rows carry `precision` since the FIXED_PRECS sweep) are
    # reported in the README table
    def is_f16(r) -> bool:
        return r.get("precision", "f16") == "f16"

    backends = sorted({r["backend"] for rows in data["sizes"].values()
                       for r in rows if is_f16(r)
                       and ("matched_iou" in r
                            or "mean_dist" in r)})
    sizes = sorted(data["sizes"])
    # unified metric: matched IoU for box tasks, 1 - dist/band for point
    band = data.get("bands", {}).get("point", 0.02)

    def metric(r) -> float:
        # a FAIL verdict wins over its sub-metric: e.g. a point-count
        # mismatch stores the placeholder mean_dist=0, which would
        # otherwise render as a perfect closeness score
        if r.get("verdict") == "FAIL":
            return 0.0
        if "matched_iou" in r:
            return r["matched_iou"]
        if "mean_dist" in r:
            return max(0.0, 1.0 - r["mean_dist"] / band)
        return 0.0

    cases = sorted({r["case"] for rows in data["sizes"].values()
                    for r in rows if is_f16(r)})
    fig, axes = plt.subplots(
        1, len(sizes) * len(backends) // max(1, len(backends)),
        figsize=(1.1 * len(cases) + 2, 4), squeeze=False)
    # one panel per size; bars = backends
    for ax, size in zip(axes[0], sizes):
        x = np.arange(len(cases))
        width = 0.8 / max(1, len(backends))
        for k, backend in enumerate(backends):
            vals = []
            for case in cases:
                m = next((r for r in data["sizes"][size]
                          if r["backend"] == backend and r["case"] == case
                          and is_f16(r)),
                         None)
                vals.append(metric(m) if m else np.nan)
            ax.bar(x + (k - (len(backends) - 1) / 2) * width, vals, width,
                   label=backend.replace("cpp-", "cpp "))
        ax.axhline(0.9, color="#e8590c", lw=1, ls="--", label="band 0.9")
        ax.set_xticks(x, [c.replace("_", "\n", 1) for c in cases],
                      fontsize=6, rotation=45, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("matched IoU / point closeness")
        ax.set_title(size)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("GGUF port vs official pytorch (bf16) — fixed-mode agreement")
    fig.tight_layout()
    fig.savefig(out / "accuracy_iou.png", dpi=150)
    print(f"written {out / 'accuracy_iou.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
