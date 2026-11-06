#!/usr/bin/env bash
# Precision x backend x size latency matrix:
#   {f16, q8_0, q4_K_M} x {cuda, vulkan, cpu} x {2B, 4B} over ALL cases.
# The mask decoder runs f16 on every tier (its f32 gate is the parity
# contract, not the latency subject). Output:
#   results/latency_matrix.json + charts/latency_matrix_<size>.png
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
GGUF="$ROOT/cpp_ggml/models/gguf"
cd "$ROOT"
PYBIN="${PYBIN:-/home/asher/anaconda3/envs/python3.13/bin/python}"
PRECISIONS="${PRECISIONS:-f16 q8_0 q4_K_M}"
SIZES="${SIZES:-2B 4B}"
BUILDS="${BUILDS:-cpp_ggml/build-cuda cpp_ggml/build-vulkan cpp_ggml/build-cpu}"
WARMUP="${WARMUP:-2}"
RUNS="${RUNS:-5}"
"$PYBIN" - "$HERE" "$PRECISIONS" "$SIZES" "$BUILDS" "$WARMUP" "$RUNS" <<'PY'
import json, statistics, subprocess, sys, time
from pathlib import Path

here, precisions, sizes, builds, warmup, runs = (
    sys.argv[1], sys.argv[2].split(), sys.argv[3].split(),
    sys.argv[4].split(), int(sys.argv[5]), int(sys.argv[6]))
sys.path.insert(0, here)
from cases import CASES
out = {"meta": {"precisions": precisions, "sizes": sizes, "builds": builds,
                "warmup": warmup, "runs": runs}, "rows": []}
gguf = Path(here).parent / "models" / "gguf"
for build in builds:
    cli = Path(build) / "wevisone-cli"
    backend = Path(build).name.replace("build-", "")
    for size in sizes:
        lower = size.lower()
        for prec in precisions:
            vlm = gguf / f"wevisone-vlm-{lower}-{prec}.gguf"
            mm = gguf / f"mmproj-wevisone-vlm-{lower}-{prec}.gguf"
            dec = gguf / f"wevisone-mask-decoder-{lower}-f16.gguf"
            if not vlm.exists():
                print(f"skip {backend}/{size}/{prec}: {vlm.name} missing")
                continue
            for image, task, query, _lang in CASES:
                if isinstance(query, list):
                    query = " ".join(query)
                cmd = [str(cli), "run",
                       "--image", f"{here}/images/{image}",
                       "--task", task, "--query", query,
                       "--precision", prec, "--decoder-precision", "f16",
                       "--vlm-gguf", str(vlm), "--mmproj-gguf", str(mm),
                       "--decoder-gguf", str(dec)]
                times = []
                ok = True
                for i in range(warmup + runs):
                    t0 = time.perf_counter()
                    try:
                        subprocess.run(cmd, check=True, capture_output=True)
                    except subprocess.CalledProcessError as e:
                        print(f"FAIL {backend}/{size}/{prec}/{image}_{task}:",
                              e.stderr.decode()[-200:] if e.stderr else e)
                        ok = False
                        break
                    times.append((time.perf_counter() - t0) * 1000)
                if not ok:
                    continue
                times = times[warmup:]
                row = {"backend": backend, "size": size, "precision": prec,
                       "case": f"{Path(image).stem}_{task}",
                       "median_ms": statistics.median(times),
                       "min_ms": min(times), "max_ms": max(times)}
                out["rows"].append(row)
                print(f"{backend}/{size}/{prec}/{row['case']}: "
                      f"{row['median_ms']:.0f} ms")
Path(f"{here}/results").mkdir(exist_ok=True)
Path(f"{here}/results/latency_matrix.json").write_text(json.dumps(out, indent=2))
PY
"$PYBIN" "$HERE/plot_latency_matrix.py" --json "$HERE/results/latency_matrix.json" --out "$HERE/charts"
