#!/usr/bin/env bash
# One-click launcher for the WeVisOne-VL dual-backend verification GUI.
#
#   bash run_gui.sh                 # pytorch + ggml, port 8017
#   bash run_gui.sh --backend cpp   # ggml side only (no torch download)
#   bash run_gui.sh --cpp-size 4B   # ggml 4B weights
#
# The pytorch side needs an offline-compatible checkpoint staging (the
# released checkpoints pin transformers 5.12 while the offline host runs
# 4.57): if /tmp/wwv-bench-hf-2B is missing it is recreated here exactly
# as the benchmark harness does (one tokenizer_config field rewritten).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYBIN="${PYBIN:-/home/asher/anaconda3/envs/python3.13/bin/python}"
BACKEND="${BACKEND:-both}"
SIZE="${SIZE:-2B}"

# extra args pass through to the GUI (after our own simple env knobs)
EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend) BACKEND="$2"; shift 2;;
        --cpp-size) SIZE="$2"; shift 2;;
        --py-model) EXTRA+=("$1" "$2"); shift 2;;
        --port) EXTRA+=("$1" "$2"); shift 2;;
        *) EXTRA+=("$1"); shift;;
    esac
done

STAGING="/tmp/wwv-bench-hf-$SIZE"
if [[ "$BACKEND" != "cpp" && ! -d "$STAGING" ]]; then
    SRC="$ROOT/checkpoints/hf-$SIZE"
    if [[ -d "$SRC" ]]; then
        echo "== staging offline-compatible checkpoint: $SRC -> $STAGING"
        mkdir -p "$STAGING"
        cp -r "$SRC"/. "$STAGING"/
        "$PYBIN" - "$STAGING" <<'PYEOF'
import json, sys
from pathlib import Path

path = Path(sys.argv[1]) / "tokenizer_config.json"
config = json.loads(path.read_text())
# transformers 4.57 requires a mapping; the release ships a list
if isinstance(config.get("extra_special_tokens"), list):
    config["extra_special_tokens"] = {}
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False))
    print("patched extra_special_tokens -> {}")
PYEOF
    fi
fi

cd "$ROOT"
exec "$PYBIN" gui/wevisone_gui.py --backend "$BACKEND" --cpp-size "$SIZE" \
    "${EXTRA[@]}"
