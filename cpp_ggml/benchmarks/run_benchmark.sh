#!/usr/bin/env bash
# One-click C++/python full-chain benchmark, TWO modes (VLX-Seek contract):
#   fixed  — the SAME reference boxes are fed to both backends (parity
#            criterion; isolates detector/path noise from backend noise).
#   closed — each backend runs its own full path (real user latency).
# Sizes: {2B, 4B}. C++ backends: cpp-cuda / cpp-vulkan (one build each;
# the CMake contract allows exactly one GPU backend per binary). The
# python backend is the official stack on CUDA (conda env python3.13,
# see python_infer.py for the offline environment adapter).
# Outputs: results/<mode>/<size>/<backend>/<case>.json, charts/*.png.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
GGUF="$ROOT/cpp_ggml/models/gguf"
cd "$ROOT"

SIZES="${SIZES:-2B 4B}"
MODE="${MODE:-both}"
PYBIN="${PYBIN:-/home/asher/anaconda3/envs/python3.13/bin/python}"
BUILD_CUDA="${BUILD_CUDA:-cpp_ggml/build-cuda}"
BUILD_VULKAN="${BUILD_VULKAN:-cpp_ggml/build-vulkan}"
BUILD_CPU="${BUILD_CPU:-cpp_ggml/build-cpu}"
# The offline host has no HF access: the official checkpoints live in
# checkpoints/hf-<size>, staged to /tmp with the single tokenizer_config
# field transformers 4.57 cannot read (see python_infer.py).
stage_bench_model () {
    local size=$1 src="$ROOT/checkpoints/hf-$size" dst="/tmp/wwv-bench-hf-$size"
    [ -d "$dst" ] && return 0
    "$PYBIN" - "$src" "$dst" <<'PY'
import json, os, shutil, sys
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
for f in src.iterdir():
    if f.name.startswith('.'):
        continue
    t = dst / f.name
    if f.name == "tokenizer_config.json":
        cfg = json.loads(f.read_text())
        if isinstance(cfg.get("extra_special_tokens"), list):
            cfg["extra_special_tokens"] = {}  # 4.57 loader requires a mapping
        if cfg.get("tokenizer_class") == "Qwen2Tokenizer":
            # tokenizer.json is fast-format; 4.57 lacks the 5.x slow->fast
            # auto upgrade, so the slow class would need a missing vocab.json
            cfg["tokenizer_class"] = "Qwen2TokenizerFast"
        t.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
    else:
        try:
            os.link(f, t)
        except OSError:
            shutil.copy2(f, t)
print("staged", dst)
PY
}

case_json () {  # size backend prec image task query mode
    local size=$1 backend=$2 prec=$3 image=$4 task=$5 query=$6 mode=$7
    local key="${image%.*}_$task"
    local dir="$HERE/results/$mode/$size/$backend"
    mkdir -p "$dir"
    local out="$dir/$key"
    if [[ "$backend" == cpp-* ]]; then
        local build="$BUILD_CUDA" dev="cuda"
        [ "$backend" = "cpp-vulkan" ] && { build="$BUILD_VULKAN"; dev="vulkan"; }
        [ "$backend" = "cpp-cpu" ] && { build="$BUILD_CPU"; dev="cpu"; }
        local lower=$(echo "$size" | tr 'A-Z' 'a-z')
        LD_LIBRARY_PATH="$ROOT/$build/bin" "$ROOT/$build/wevisone-cli" run \
            --image "$HERE/images/$image" --task "$task" --query "$query" \
            --device "$dev" --precision "$prec" --decoder-precision f16 \
            --vlm-gguf "$GGUF/wevisone-vlm-$lower-$prec.gguf" \
            --mmproj-gguf "$GGUF/mmproj-wevisone-vlm-$lower-$prec.gguf" \
            --decoder-gguf "$GGUF/wevisone-mask-decoder-$lower-f16.gguf" \
            --out "${out}.png" --json "${out}.json" > /dev/null
    else
        stage_bench_model "$size"
        "$PYBIN" "$HERE/python_infer.py" \
            --image "$HERE/images/$image" --task "$task" --query "$query" \
            --model-dir "/tmp/wwv-bench-hf-$size" \
            --out-png "${out}.png" --out-json "${out}.json" > /dev/null
    fi
    echo "  [$mode/$size/$backend] $key"
}

run_case () {  # mode size prec image task query
    local mode=$1 size=$2 prec=$3 image=$4 task=$5; shift 5
    local query="$*"
    for backend in cpp-cuda cpp-vulkan python-cuda; do
        case_json "$size" "$backend" "$prec" "$image" "$task" "$query" "$mode"
    done
}

SIZES_LOWER=$(echo "$SIZES" | tr 'A-Z' 'a-z')
for SIZE in $SIZES; do
    SIZE_LOWER=$(echo "$SIZE" | tr 'A-Z' 'a-z')
    for CASE in $("$PYBIN" -c "import sys; sys.path.insert(0,'$HERE'); from cases import CASES; print('\n'.join(f'{c[0]}|{c[1]}|{\" \".join(c[2]) if isinstance(c[2],list) else c[2]}' for c in CASES))"); do
        IFS='|' read -r image task query <<< "$CASE"
        if [ "$MODE" != "closed" ]; then
            run_case fixed "$SIZE" f16 "$image" "$task" "$query"
        fi
        if [ "$MODE" != "fixed" ]; then
            run_case closed "$SIZE" f16 "$image" "$task" "$query"
        fi
    done
done

echo "== charts =="
"$PYBIN" "$HERE/plot_benchmarks.py" --results "$HERE/results" --out "$HERE/charts"
