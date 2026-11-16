#!/usr/bin/env python3
"""End-to-end latency comparison across GGML quantizations, backends and
the official pytorch stack — the chart that was missing from charts/.

Data sources (both already collected, no GPU re-run needed):
  results/latency_matrix.json           GGML rows: size x precision x backend
                                        x case, wall_ms + e2e_ms (written by
                                        run_latency_matrix.sh)
  results/fixed/<size>/python-cuda/     official-stack rows: <case>.json with
                                        timings.e2e_ms (written by the
                                        benchmark harness python adapter)

The common aperture is e2e_ms — CLI-internal pre+infer+post on the GGML
side and engine.predict on the pytorch side, both load-free, so the bars
compare pure inference throughput rather than process startup. wall_ms
(including model load) lives only in latency_matrix.json and stays there.

Outputs:
  charts/latency_e2e_backends.png    every series on a log scale (the CPU
                                     f16 tier spans ~2 decades above CUDA)
  charts/latency_e2e_vs_pytorch.png  GPU-only, linear scale — the head-to-
                                     head quantized-GGML vs pytorch-cuda view
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

# series key -> (label, color, matrix selector / pytorch special case)
SERIES = [
    ("cuda:f16", "GGML CUDA f16", "#2b8a3e"),
    ("cuda:q8_0", "GGML CUDA q8_0", "#74b816"),
    ("cuda:q4_K_M", "GGML CUDA q4_K_M", "#a9e34b"),
    ("vulkan:f16", "GGML Vulkan f16", "#1c7ed6"),
    ("cpu:f16", "GGML CPU f16", "#868e96"),
    ("pytorch:cuda", "PyTorch CUDA (official)", "#e8590c"),
]


def load_series(matrix_rows: list[dict], results_root: Path,
                size: str) -> dict[str, dict[str, float]]:
    """series key -> case -> median e2e_ms for one size panel."""
    cases = sorted({r["case"] for r in matrix_rows})
    out: dict[str, dict[str, float]] = {}
    for key, _, _ in SERIES:
        backend, _, precision = key.partition(":")
        if backend == "pytorch":
            vals: dict[str, float] = {}
            for case in cases:
                # the harness stages the official-stack runs under the
                # fixed/ mode directory (closed/ rows have no timings)
                row = next((p for p in (
                    results_root / "fixed" / size / "python-cuda" / f"{case}.json",
                    results_root / "closed" / size / "python-cuda" / f"{case}.json",
                ) if p.is_file()), None)
                if row is None:
                    continue
                data = json.loads(row.read_text())
                e2e = (data.get("timings") or {}).get("e2e_ms")
                if e2e is not None:
                    vals[case] = float(e2e)
            out[key] = vals
            continue
        per_case: dict[str, list[float]] = {}
        for r in matrix_rows:
            if r["size"] == size and r["backend"] == backend \
                    and r["precision"] == precision \
                    and r.get("e2e_ms") is not None:
                per_case.setdefault(r["case"], []).append(float(r["e2e_ms"]))
        out[key] = {c: statistics.median(v) for c, v in per_case.items()}
    return out


def draw(vals: dict[str, dict[str, float]], size: str, cases: list[str],
         series_keys: list[str], out_path: Path, log_scale: bool) -> None:
    active = [s for s in SERIES
              if s[0] in series_keys and vals.get(s[0])]
    x = np.arange(len(cases))
    width = 0.8 / max(1, len(active))
    fig, ax = plt.subplots(figsize=(max(9.0, 0.55 * len(cases) * len(active)),
                                    4.8))
    for k, (key, label, color) in enumerate(active):
        ys = [vals[key].get(c, np.nan) for c in cases]
        ax.bar(x + (k - (len(active) - 1) / 2) * width, ys, width * 0.92,
               label=label, color=color)
        for xi, yi in zip(x, ys):
            if not np.isnan(yi):
                txt = f"{yi:.0f}" if log_scale and yi < 1e4 else (
                    f"{yi / 1000:.1f}s" if yi >= 1e4 else f"{yi:.0f}")
                ax.annotate(txt, (xi + (k - (len(active) - 1) / 2) * width,
                                  yi), ha="center", va="bottom",
                            fontsize=6, rotation=90)
    ax.set_xticks(x, [c.replace("_", "\n") for c in cases], fontsize=8)
    ax.set_ylabel("end-to-end latency (ms, load-free)")
    if log_scale:
        ax.set_yscale("log")
    ax.set_title(f"End-to-end latency per case — GGML quantizations/backends "
                 f"vs official pytorch ({size})")
    ax.legend(fontsize=8, ncol=min(len(active), 3))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="results/latency_matrix.json")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="charts")
    args = ap.parse_args()
    matrix = json.loads(Path(args.matrix).read_text())
    rows = matrix["rows"]
    results_root = Path(args.results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sizes = sorted({r["size"] for r in rows})
    all_keys = [s[0] for s in SERIES]
    gpu_keys = [k for k in all_keys if not k.startswith("cpu:")]
    for size in sizes:
        vals = load_series(rows, results_root, size)
        cases = sorted({r["case"] for r in rows if r["size"] == size})
        # per-size filenames: a single name would have the 4B panel
        # overwrite the 2B one
        draw(vals, size, cases, all_keys,
             out / f"latency_e2e_backends_{size.lower()}.png", log_scale=True)
        draw(vals, size, cases, gpu_keys,
             out / f"latency_e2e_vs_pytorch_{size.lower()}.png",
             log_scale=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
