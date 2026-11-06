#!/usr/bin/env python3
"""Renders the precision x backend latency matrix heatmap, per size."""
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
    rows = data["rows"]
    precisions = data["meta"]["precisions"]
    sizes = data["meta"].get("sizes") or sorted({r["size"] for r in rows})
    backends = sorted({r["backend"] for r in rows})
    fig, axes = plt.subplots(
        1, len(sizes), figsize=(1.6 * len(precisions) + 3.2, 4),
        squeeze=False)
    for ax, size in zip(axes[0], sizes):
        matrix = np.full((len(backends), len(precisions)), np.nan)
        for r in rows:
            if r.get("size", size) != size:
                continue
            i = backends.index(r["backend"])
            if r["precision"] in precisions:
                j = precisions.index(r["precision"])
                m = matrix[i, j]
                matrix[i, j] = r["median_ms"] if np.isnan(m) else min(m, r["median_ms"])
        im = ax.imshow(matrix, cmap="viridis_r", aspect="auto")
        ax.set_xticks(range(len(precisions)), precisions, fontsize=8)
        ax.set_yticks(range(len(backends)), backends, fontsize=8)
        for i in range(len(backends)):
            for j in range(len(precisions)):
                if not np.isnan(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.0f}", ha="center",
                            va="center", color="white", fontsize=8)
        ax.set_title(f"median latency (ms), {size}")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "latency_matrix.png", dpi=150)
    print(f"written {out / 'latency_matrix.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
