#!/usr/bin/env python3
"""Renders the precision x backend x size latency charts from
results/latency_matrix.json (written by run_latency_matrix.sh).

Two apertures live in the data: wall_ms (process spawn -> exit) and
e2e_ms (CLI-internal pre+infer+post, load-free). The heatmap aggregates
per case with the median-of-medians on wall_ms (user-visible latency);
the grouped-bar figure keeps every case separate on e2e_ms (pure
inference throughput, load-free).
Outputs: charts/latency_matrix.png, charts/latency_cases.png.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BACKEND_ORDER = ("cuda", "vulkan", "cpu")
COLORS = {"cuda": "#2b8a3e", "vulkan": "#1c7ed6", "cpu": "#868e96"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data = json.loads(Path(args.json).read_text())
    rows = data["rows"]
    precisions = data["meta"]["precisions"]
    sizes = data["meta"].get("sizes") or sorted({r["size"] for r in rows})
    backends = [b for b in BACKEND_ORDER
                if b in {r["backend"] for r in rows}]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- heatmap: wall_ms median-of-medians, backend x precision ----
    fig, axes = plt.subplots(
        1, len(sizes), figsize=(1.8 * len(precisions) + 3.4, 1.5 * len(backends) + 1.2),
        squeeze=False)
    for ax, size in zip(axes[0], sizes):
        matrix = np.full((len(backends), len(precisions)), np.nan)
        for i, b in enumerate(backends):
            for j, p in enumerate(precisions):
                vals = [r["wall_ms"] for r in rows
                        if r["backend"] == b and r["precision"] == p
                        and r["size"] == size and r.get("wall_ms") is not None]
                if vals:
                    matrix[i, j] = statistics.median(vals)
        im = ax.imshow(matrix, cmap="viridis_r", aspect="auto")
        ax.set_xticks(range(len(precisions)), precisions, fontsize=8)
        ax.set_yticks(range(len(backends)), backends, fontsize=8)
        for i in range(len(backends)):
            for j in range(len(precisions)):
                if not np.isnan(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.0f}", ha="center",
                            va="center", color="white", fontsize=8)
        ax.set_title(f"GGUF VLM latency (ms, wall), {size}")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out / "latency_matrix.png", dpi=150)
    plt.close(fig)

    # ---- grouped bars: per-case e2e_ms, one panel per (size, precision) ----
    cases = sorted({r["case"] for r in rows})
    panels = [(s, p) for s in sizes for p in precisions
              if any(r["size"] == s and r["precision"] == p for r in rows)]
    ncols = min(len(panels), 3)
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows), squeeze=False)
    x = np.arange(len(cases))
    width = 0.8 / max(1, len(backends))
    for ax, (size, prec) in zip(axes.flat, panels):
        for k, b in enumerate(backends):
            vals = [next((r["e2e_ms"] for r in rows
                          if r["backend"] == b and r["precision"] == prec
                          and r["size"] == size and r["case"] == c
                          and r.get("e2e_ms") is not None), np.nan)
                    for c in cases]
            ax.bar(x + (k - (len(backends) - 1) / 2) * width, vals, width,
                   label=b, color=COLORS.get(b))
        ax.set_xticks(x, [c.replace("_", "\n", 1) for c in cases],
                      fontsize=6, rotation=45, ha="right")
        # the cpu reference is 1-2 orders of magnitude slower than the GPU
        # paths — a linear scale flattens the interesting cuda-vs-vulkan
        # comparison into invisibility
        ax.set_yscale("symlog" if "cpu" in backends else "linear")
        ax.set_ylabel("e2e inference (ms)" + (", symlog" if "cpu" in backends else ""),
                      fontsize=8)
        ax.set_title(f"{size} / {prec}", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=7)
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "latency_cases.png", dpi=150)
    print(f"written {out / 'latency_matrix.png'}, {out / 'latency_cases.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
