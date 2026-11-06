#!/usr/bin/env bash
# Export the full GGUF variant matrix for the 2B/4B checkpoints.
#
#   VLM (llama.cpp qwen3vl converter + llama-quantize):
#     f32 / f16 / q8_0 / q4_K_M  (f32 for parity gates, q4_K for the
#     cost/performance tier documented in MODEL_CARD)
#   Pixel decoder (convert_mask_decoder_to_gguf.py): f32 / f16 / q8_0
#
#   bash cpp_ggml/scripts/export_gguf_variants.sh --size 4B --out cpp_ggml/models/gguf
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIZE="${SIZE:-4B}"
OUT="${OUT:-$HERE/../models/gguf}"
SIDECAR="${SIDECAR:-$HERE/../../checkpoints/mask_decoder_${SIZE}.pt}"
mkdir -p "$OUT"

echo "== 1. VLM fp16 (llama.cpp converter, HF layout in /tmp/wwv-${SIZE}) =="
HF_SOURCE="${HF_SOURCE:-tencent/WeVisOne-VL-${SIZE}}"
python3 "$HERE/../third_party/llama.cpp/convert_hf_to_gguf.py" \
    "$HF_SOURCE" --outfile "$OUT/wevisone-vlm-f16.gguf" --outtype f16
for TIER in f32 q8_0 q4_K_M; do
    echo "== quantize -> $TIER =="
    python3 "$HERE/../third_party/llama.cpp/llama-quantize" \
        "$OUT/wevisone-vlm-f16.gguf" "$OUT/wevisone-vlm-${TIER%.gguf}.gguf" "$TIER"
done

echo "== 2. pixel decoder tiers =="
for TIER in f32 f16 q8_0; do
    python3 "$HERE/convert_mask_decoder_to_gguf.py" \
        --sidecar "$SIDECAR" --precision "$TIER" \
        --output "$OUT/wevisone-mask-decoder-$TIER.gguf"
done

echo "== 3. digest =="
( cd "$OUT" && sha256sum wevisone-*.gguf > SHA256SUMS )
cat "$OUT/SHA256SUMS"
