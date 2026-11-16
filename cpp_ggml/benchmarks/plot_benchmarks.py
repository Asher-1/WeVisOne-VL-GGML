#!/usr/bin/env python3
"""Renders the benchmark charts: per-size latency bars across backends.

Layout: results/<mode>/<size>/<backend>/<case>.json written by
run_benchmark.sh. The python-cuda backend is the official stack (bf16);
cpp-* are the GGUF port tiers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BACKENDS = ("cpp-cuda", "cpp-vulkan", "python-cuda")
COLORS = {"cpp-cuda": "#2b8a3e", "cpp-vulkan": "#1c7ed6",
          "python-cuda": "#e8590c"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    results = Path(args.results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # collect: lat[mode][size][backend][case] = latency_s
    lat: dict = {}
    for js in results.glob("*/*/*/*.json"):
        mode, size, backend = js.parts[-4], js.parts[-3], js.parts[-2]
        try:
            d = json.loads(js.read_text())
        except json.JSONDecodeError:
            continue
        lat.setdefault(mode, {}).setdefault(size, {}).setdefault(
            backend, {})[js.stem] = d

    for mode in ("fixed", "closed"):
        if mode not in lat:
            continue
        sizes = sorted(lat[mode])
        fig, axes = plt.subplots(
            1, len(sizes), figsize=(7 * max(1, len(sizes)), 4.5),
            squeeze=False)
        for ax, size in zip(axes[0], sizes):
            backends = [b for b in BACKENDS if b in lat[mode][size]]
            labels = sorted(set().union(
                *[set(lat[mode][size][b]) for b in backends]))
            x = range(len(labels))
            width = 0.8 / max(1, len(backends))
            for i, backend in enumerate(backends):
                vals = [lat[mode][size].get(backend, {}).get(
                            k, {}).get("latency_s",
                                       lat[mode][size].get(backend, {})
                                       .get(k, {}).get("timings", {})
                                       .get("e2e_ms", 0) / 1000.0)
                        for k in labels]
                ax.bar([v + (i - len(backends) / 2 + 0.5) * width
                        for v in x], vals, width,
                       label=backend, color=COLORS.get(backend))
            ax.set_xticks(list(x))
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
            ax.set_ylabel("e2e latency (s)")
            ax.set_title(f"{mode} mode, {size}")
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"latency_ttft_bar_{mode}.png", dpi=150)
        print(f"written {out / f'latency_ttft_bar_{mode}.png'}")
    print(f"charts written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
