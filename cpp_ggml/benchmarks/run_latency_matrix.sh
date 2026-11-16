#!/usr/bin/env bash
# Precision x backend x size latency matrix:
#   f16 x {cuda, vulkan, cpu} x {2B, 4B} x ALL cases, plus the quantized
#   tiers {q8_0, q4_K_M} on cuda only (the quantization variable is
#   evaluated against the production GPU path; vulkan/cpu stay f16 so the
#   matrix stays tractable — the full cross would be 1512 process spawns
#   with the cpu backend alone contributing hours).
# The mmproj and the mask decoder stay f16 on every tier (the quantization
# variable under test is the VLM text stack; the decoder f32 gate is the
# parity contract, not the latency subject). Each run records BOTH timing
# apertures: wall_ms (process spawn -> exit, the user-visible cold-call
# latency) and e2e_ms (the CLI's internal pre+infer+post, load-free).
# Output: results/latency_matrix.json + charts/latency_matrix_<size>.png
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
GGUF="$ROOT/cpp_ggml/models/gguf"
cd "$ROOT"
PYBIN="${PYBIN:-/home/asher/anaconda3/envs/python3.13/bin/python}"
PRECISIONS="${PRECISIONS:-f16 q8_0 q4_K_M}"
SIZES="${SIZES:-2B 4B}"
BUILDS="${BUILDS:-cpp_ggml/build-cuda cpp_ggml/build-vulkan cpp_ggml/build-cpu}"
# Which backends sweep the quantized tiers. The production-GPU default is
# cuda; pass a space list (e.g. "cuda vulkan") to widen the sweep.
QUANT_BACKENDS="${QUANT_BACKENDS:-cuda}"
WARMUP="${WARMUP:-1}"
RUNS="${RUNS:-3}"
"$PYBIN" - "$HERE" "$PRECISIONS" "$SIZES" "$BUILDS" "$WARMUP" "$RUNS" "$QUANT_BACKENDS" <<'PY'
import json, statistics, subprocess, sys, tempfile, time
from pathlib import Path

here, precisions, sizes, builds, warmup, runs, quant_backends = (
    sys.argv[1], sys.argv[2].split(), sys.argv[3].split(),
    sys.argv[4].split(), int(sys.argv[5]), int(sys.argv[6]),
    set(sys.argv[7].split()))
sys.path.insert(0, here)
from cases import CASES

# MERGE semantics: a re-run with a narrower sweep (e.g. adding the vulkan
# quantized tiers) updates only its own (backend, size, precision, case)
# rows and keeps the rest — a full re-run of the cpu cells would cost
# hours for unchanged numbers.
row_key = lambda r: (r["backend"], r["size"], r["precision"], r["case"])
merged = {}
matrix_path = Path(f"{here}/results/latency_matrix.json")
if matrix_path.exists():
    try:
        for r in json.loads(matrix_path.read_text())["rows"]:
            merged[row_key(r)] = r
    except Exception:
        merged = {}

out = {"meta": {"precisions": precisions, "sizes": sizes, "builds": builds,
                "warmup": warmup, "runs": runs,
                "apertures": ["wall_ms", "e2e_ms"],
                "scope": "quantized tiers run on cuda only; f16 on all"},
       "rows": []}
gguf = Path(here).parent / "models" / "gguf"
for build in builds:
    cli = Path(build) / "wevisone-cli"
    backend = Path(build).name.replace("build-", "")
    # see QUANT_BACKENDS above: the quantized sweep is a per-backend knob
    tier_preps = precisions if backend in quant_backends else ["f16"]
    for size in sizes:
        lower = size.lower()
        for prec in tier_preps:
            vlm = gguf / f"wevisone-vlm-{lower}-{prec}.gguf"
            mm = gguf / f"mmproj-wevisone-vlm-{lower}-f16.gguf"
            dec = gguf / f"wevisone-mask-decoder-{lower}-f16.gguf"
            if not vlm.exists():
                print(f"skip {backend}/{size}/{prec}: {vlm.name} missing")
                continue
            for image, task, query, _lang in CASES:
                # labels travel newline-separated (see run_benchmark.sh)
                if isinstance(query, list):
                    query = "\n".join(query)
                cmd = [str(cli), "run",
                       "--image", f"{here}/images/{image}",
                       "--task", task, "--query", query,
                       "--device", backend,
                       "--precision", prec, "--decoder-precision", "f16",
                       "--vlm-gguf", str(vlm), "--mmproj-gguf", str(mm),
                       "--decoder-gguf", str(dec)]
                walls, e2es = [], []
                ok = True
                with tempfile.TemporaryDirectory() as td:
                    jpath = Path(td) / "r.json"
                    cmd += ["--json", str(jpath)]
                    for i in range(warmup + runs):
                        t0 = time.perf_counter()
                        try:
                            subprocess.run(cmd, check=True, capture_output=True,
                                           env={"LD_LIBRARY_PATH": str(Path(build) / "bin"),
                                                "PATH": "/usr/bin:/bin",
                                                "HOME": str(Path.home())})
                        except subprocess.CalledProcessError as e:
                            print(f"FAIL {backend}/{size}/{prec}/{image}_{task}:",
                                  e.stderr.decode()[-200:] if e.stderr else e)
                            ok = False
                            break
                        walls.append((time.perf_counter() - t0) * 1000)
                        if jpath.exists():
                            e2es.append(json.loads(jpath.read_text())
                                        ["timings"]["e2e_ms"])
                        jpath.unlink(missing_ok=True)
                if not ok:
                    continue
                walls, e2es = walls[warmup:], e2es[warmup:]
                row = {"backend": backend, "size": size, "precision": prec,
                       "case": f"{Path(image).stem}_{task}",
                       "wall_ms": statistics.median(walls),
                       "e2e_ms": statistics.median(e2es) if e2es else None,
                       "wall_min_ms": min(walls), "wall_max_ms": max(walls)}
                out["rows"].append(row)
                merged[row_key(row)] = row
                print(f"{backend}/{size}/{prec}/{row['case']}: "
                      f"wall {row['wall_ms']:.0f} ms"
                      + (f", e2e {row['e2e_ms']:.0f} ms"
                         if row["e2e_ms"] is not None else ""))
if merged:
    out["rows"] = sorted(merged.values(),
                         key=lambda r: (r["size"], r["precision"],
                                        r["backend"], r["case"]))
    # the merged meta describes the UNION of everything ever swept
    out["meta"]["precisions"] = sorted(set(out["meta"]["precisions"]))
    out["meta"]["sizes"] = sorted(set(out["meta"]["sizes"]))
Path(f"{here}/results").mkdir(exist_ok=True)
matrix_path.write_text(json.dumps(out, indent=2))
PY
"$PYBIN" "$HERE/plot_latency_matrix.py" --json "$HERE/results/latency_matrix.json" --out "$HERE/charts"
