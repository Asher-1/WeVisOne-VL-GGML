#!/usr/bin/env bash
# Export the full GGUF variant matrix for the 2B/4B checkpoints, named by
# size exactly as the benchmark harness and runtime defaults expect:
#
#   wevisone-vlm-<size>-{f16,q8_0,q4_K_M}.gguf        (LLM text stack)
#   mmproj-wevisone-vlm-<size>-f16.gguf               (vision tower + projector)
#   wevisone-mask-decoder-<size>-{f32,f16}.gguf       (pixel decoder)
#
# Why the three families are separate files:
#   VLM / mmproj — the llama.cpp qwen3vl converter emits the LLM and the
#     multimodal projector as two GGUFs, and the mtmd runtime contract
#     (mtmd_init_from_file) loads the projector separately; quantizing them
#     independently is the FEATURE (vision tower stays f16 while the text
#     stack varies).
#   mask decoder — it is NOT a llama.cpp architecture model: its weights
#     come from the training sidecar mask_decoder.pt (not the HF
#     safetensors), the ggml graph is built by hand in
#     src/mask_decoder_graph.cpp, and its f32 file IS the parity gate.
#     Decoder quantization is a rejected non-goal (48-element tensors sit
#     below the q8_0 block granularity).
#
#   SIZE=4B bash cpp_ggml/scripts/export_gguf_variants.sh
#   (HF_SOURCE defaults to the offline local checkpoint checkpoints/hf-SIZE)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIZE="${SIZE:-4B}"
OUT="${OUT:-$HERE/../models/gguf}"
HF_SOURCE="${HF_SOURCE:-$HERE/../../checkpoints/hf-$SIZE}"
# llama-quantize is the COMPILED binary — `python3 .../llama-quantize` was
# the old py-converters alias and no longer exists.
QUANTIZE="${QUANTIZE:-$HERE/../build-cuda/quantize-sub/llama-quantize}"
SIDECAR="${SIDECAR:-$HERE/../../checkpoints/hf-$SIZE/mask_decoder.pt}"
SELFTEST="${SELFTEST:-$HERE/../build-cpu/wevisone-cli}"
# decoder structural-gate dims, read from the checkpoint config (both
# releases ship vit hidden 1024; llm differs: 2B 2048 / 4B 2560)
GRID="${GRID:-32}"
LOWER=$(echo "$SIZE" | tr 'A-Z' 'a-z')
mkdir -p "$OUT"

echo "== 1. VLM f16 (llama.cpp qwen3vl converter, source: $HF_SOURCE) =="
python3 "$HERE/../third_party/llama.cpp/convert_hf_to_gguf.py" \
    "$HF_SOURCE" --outfile "$OUT/wevisone-vlm-$LOWER-f16.gguf" --outtype f16

echo "== 1b. mmproj f16 (vision tower + projector, mtmd contract) =="
python3 "$HERE/../third_party/llama.cpp/convert_hf_to_gguf.py" \
    "$HF_SOURCE" --mmproj \
    --outfile "$OUT/mmproj-wevisone-vlm-$LOWER-f16.gguf" --outtype f16

echo "== 2. VLM quantized tiers (llama-quantize binary) =="
for TIER in q8_0 q4_K_M; do
    echo "-- $TIER"
    "$QUANTIZE" "$OUT/wevisone-vlm-$LOWER-f16.gguf" \
        "$OUT/wevisone-vlm-$LOWER-$TIER.gguf" "$TIER"
done

echo "== 3. mask decoder tiers (sidecar -> gguf) =="
for TIER in f32 f16; do
    python3 "$HERE/convert_mask_decoder_to_gguf.py" \
        --sidecar "$SIDECAR" --precision "$TIER" \
        --output "$OUT/wevisone-mask-decoder-$LOWER-$TIER.gguf"
done

echo "== 4. decoder structural gate (selftest, dims from config.json) =="
read -r VIT LLM <<< "$(python3 -c "
import json
c = json.load(open('$HF_SOURCE/config.json'))
print(c['vision_config']['hidden_size'], c['text_config']['hidden_size'])
")"
"$SELFTEST" "$OUT/wevisone-mask-decoder-$LOWER-f32.gguf" \
    --vit "$VIT" --llm "$LLM" --grid "$GRID" --q 3 --cnn 256

echo "== 5. digest =="
( cd "$OUT" && sha256sum wevisone-*.gguf mmproj-*.gguf > SHA256SUMS )
cat "$OUT/SHA256SUMS"
