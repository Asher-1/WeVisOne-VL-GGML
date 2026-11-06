# WeVisOne-VL GGUF Model Cards

GGUF exports of the released checkpoints (`tencent/WeVisOne-VL-2B`,
`tencent/WeVisOne-VL-4B`) plus the mask-decoder sidecar, produced by
`cpp_ggml/scripts/export_gguf_variants.sh` and published through
`prepare_hf_models.py` to `tencent/WeVisOne-VL-GGML-<SIZE>` on the Hugging
Face hub. Local layout mirrors the hub (this directory is gitignored except
this card).

## VLM variants (merged mmproj + LLM, llama.cpp `qwen3vl` arch)

| File | Tier | Intended use | Expected parity |
|---|---|---|---|
| `wevisone-vlm-f32.gguf` | fp32 | parity-gate reference; debugging | token-identical to torch fp32 (with the strict-GEMM patch) |
| `wevisone-vlm-f16.gguf` | fp16 | **default serving tier** | token-identical in fixed-mode benchmarks (sibling d10 criterion) |
| `wevisone-vlm-q8_0.gguf` | 8-bit | memory-constrained serving (≈2x smaller) | separate quantized tier, `ratio <= 2.5` noise band |
| `wevisone-vlm-q4_K_M.gguf` | 4-bit K-quant | highest cost/performance tier; batch/edge serving | quality tier only — verify on the parity suite before shipping |

Acceptance for every tier: greedy decoding must reproduce the structured
answer grammar (`<box>..</box>`, `<OBJ>`, `None` rejection); task-level
metrics (F1/MAD) are reported per tier in the benchmarks charts.

## Pixel-decoder variants (arch `wevisone_mask`)

| File | Tier | Intended use |
|---|---|---|
| `wevisone-mask-decoder-f32.gguf` | fp32 | **the only parity-gated tier** — the released decoder executes in fp32 by construction (`self.float()` + autocast off in the Python forward) |
| `wevisone-mask-decoder-f16.gguf` | fp16 | memory-tight edge deployments; MAD may drift, gate before use |
| `wevisone-mask-decoder-q8_0.gguf` | 8-bit | smallest footprint; the MSDA statistics are sensitive — treat as experimental |

q4_K is deliberately NOT offered for the decoder: GroupNorm/MSDA statistics
and the alpha-cascade residual path degrade beyond the published MAD
thresholds under 4-bit weights.

## Notes

- All variants ship `SHA256SUMS`; verify after download.
- The C++ binary selects a tier explicitly (`--precision`,
  `--decoder-precision`); there are no environment overrides.
- Parity data (fixed/closed benchmark tables and effect panels) lands in
  `cpp_ggml/benchmarks/charts` after running the benchmark suite.
