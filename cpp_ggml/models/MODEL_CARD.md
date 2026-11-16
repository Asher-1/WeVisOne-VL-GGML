# WeVisOne-VL GGUF Model Card

GGUF exports of the released checkpoints
([tencent/WeVisOne-VL-2B](https://huggingface.co/tencent/WeVisOne-VL-2B),
[tencent/WeVisOne-VL-4B](https://huggingface.co/tencent/WeVisOne-VL-4B))
for the C++ ggml inference runtime in this repository.

**Hub repository**: <https://huggingface.co/Asher-1/WeVisOne-VL-GGML>

Produced by `cpp_ggml/scripts/export_gguf_variants.sh` and published via
`cpp_ggml/models/upload_to_hf.py`.

---

## Architecture: three separate file families

A full WeVisOne-VL inference pipeline requires **three GGUF files** loaded
together (a "triplet"). They are separate because:

| Family | Role | Why separate |
|--------|------|--------------|
| **VLM** (`wevisone-vlm-*`) | LLM text stack (Qwen3-VL arch) | Quantized independently; the vision tower stays f16 while the text stack varies |
| **mmproj** (`mmproj-wevisone-vlm-*`) | Vision tower + multimodal projector | The llama.cpp `mtmd` contract (`mtmd_init_from_file`) loads the projector separately; always f16 |
| **Mask decoder** (`wevisone-mask-decoder-*`) | 9.6M-param pixel decoder (masks / alpha mattes) | Not a llama.cpp architecture; weights come from the training sidecar `mask_decoder.pt`, graph built by hand in `src/mask_decoder_graph.cpp` |

The VLM + mmproj pair handles all **object-level** tasks (detection,
referring, counting, pointing, OCR, freeform). Adding the mask decoder
enables **pixel-level** tasks (segmentation, matting, visual_prompt).

> **Size pairing rule**: always match 2B with 2B, 4B with 4B.
> The decoder's LLM-hidden-state dimension differs (2B: 2048, 4B: 2560);
> cross-size pairing will fail at load time.

---

## File inventory

### 2B triplet

| File | Precision | Size | Use case |
|------|-----------|------|----------|
| `wevisone-vlm-2b-f16.gguf` | FP16 | 3.80 GB | **Default serving tier** — token-identical to the PyTorch reference in fixed-mode benchmarks |
| `wevisone-vlm-2b-q8_0.gguf` | Q8_0 | 2.02 GB | Memory-constrained serving (~2x smaller); structured-answer grammar preserved, minor logit noise |
| `wevisone-vlm-2b-q4_K_M.gguf` | Q4_K_M | 1.20 GB | Edge / batch serving; quality tier — verify on the parity suite before shipping |
| `mmproj-wevisone-vlm-2b-f16.gguf` | FP16 | 0.76 GB | Vision tower + projector (always f16, not quantized) |
| `wevisone-mask-decoder-2b-f32.gguf` | FP32 | 36 MB | **Parity-gated decoder** — the released Python decoder executes in fp32 by construction |
| `wevisone-mask-decoder-2b-f16.gguf` | FP16 | 18 MB | Memory-tight edge deployments; MAD may drift slightly, gate before production use |

### 4B triplet

| File | Precision | Size | Use case |
|------|-----------|------|----------|
| `wevisone-vlm-4b-f16.gguf` | FP16 | 8.23 GB | **Default serving tier** — highest quality, token-identical in fixed-mode benchmarks |
| `wevisone-vlm-4b-q8_0.gguf` | Q8_0 | 4.37 GB | Memory-constrained serving (~2x smaller) |
| `wevisone-vlm-4b-q4_K_M.gguf` | Q4_K_M | 2.53 GB | Edge / batch serving; quality tier — verify on the parity suite |
| `mmproj-wevisone-vlm-4b-f16.gguf` | FP16 | 0.78 GB | Vision tower + projector (always f16) |
| `wevisone-mask-decoder-4b-f32.gguf` | FP32 | 37 MB | **Parity-gated decoder** |
| `wevisone-mask-decoder-4b-f16.gguf` | FP16 | 18 MB | Memory-tight edge deployments |

### Why no other tiers?

- **No f32 VLM**: the f16 VLM is already token-identical to the PyTorch
  bf16 reference on all structured tasks; an f32 export would double the
  file size with zero measurable gain.
- **No quantized mmproj**: the vision tower is shared across all VLM
  tiers and stays f16 by design — quantizing it would degrade visual
  features for every tier simultaneously.
- **No quantized decoder**: the 9.6M-param decoder contains 48-element
  tensors that sit below the q8_0 block granularity (32 elements).
  GroupNorm / MSDA statistics and the alpha-cascade residual path
  degrade beyond published MAD thresholds under quantized weights.
  Decoder quantization is a **rejected non-goal**.

---

## Pairing guide & CLI usage

### Recommended combinations

| Scenario | VLM | mmproj | Decoder | Total VRAM (approx) |
|----------|-----|--------|---------|---------------------|
| Default CUDA serving (2B) | 2b-f16 | 2b-f16 | 2b-f32 | ~5.5 GB |
| Default CUDA serving (4B) | 4b-f16 | 4b-f16 | 4b-f32 | ~10 GB |
| Memory-tight (2B, 6 GB card) | 2b-q8_0 | 2b-f16 | 2b-f16 | ~3.5 GB |
| Edge / CPU (2B) | 2b-q4_K_M | 2b-f16 | 2b-f16 | ~2.5 GB |
| Maximum quality (4B) | 4b-f16 | 4b-f16 | 4b-f32 | ~10 GB |
| Object-level only (no pixel tasks) | any | matching | *omit* | saves decoder load |

### Running with wevisone-cli

```bash
# Build (CUDA example)
git submodule update --init --recursive
cmake -S cpp_ggml -B cpp_ggml/build-cuda -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CUDA=ON
cmake --build cpp_ggml/build-cuda --parallel 8

# Full pipeline: matting with the 2B f16 triplet
cpp_ggml/build-cuda/wevisone-cli run \
    --image demo/examples/images/seabreeze.jpg \
    --task matting --query person \
    --models-dir cpp_ggml/models/gguf \
    --vlm-gguf cpp_ggml/models/gguf/wevisone-vlm-2b-f16.gguf \
    --mmproj-gguf cpp_ggml/models/gguf/mmproj-wevisone-vlm-2b-f16.gguf \
    --decoder-gguf cpp_ggml/models/gguf/wevisone-mask-decoder-2b-f32.gguf \
    --device cuda --out result.png

# Object-level only: detection (no decoder needed)
cpp_ggml/build-cuda/wevisone-cli run \
    --image demo/examples/images/livingroom.jpg \
    --task detection --query "sofa,pillow,lamp" \
    --vlm-gguf cpp_ggml/models/gguf/wevisone-vlm-2b-f16.gguf \
    --mmproj-gguf cpp_ggml/models/gguf/mmproj-wevisone-vlm-2b-f16.gguf \
    --device cuda

# Quantized tier: q8_0 VLM + f16 decoder
cpp_ggml/build-cuda/wevisone-cli run \
    --image demo/examples/images/seabreeze.jpg \
    --task matting --query person \
    --vlm-gguf cpp_ggml/models/gguf/wevisone-vlm-2b-q8_0.gguf \
    --mmproj-gguf cpp_ggml/models/gguf/mmproj-wevisone-vlm-2b-f16.gguf \
    --decoder-gguf cpp_ggml/models/gguf/wevisone-mask-decoder-2b-f16.gguf \
    --device cuda
```

> **Default path convention**: when `--vlm-gguf` / `--mmproj-gguf` /
> `--decoder-gguf` are omitted, the runtime resolves
> `{models_dir}/wevisone-vlm-{precision}.gguf` etc. (no size suffix).
> Either pass explicit paths as above, or create symlinks:
> ```bash
> cd cpp_ggml/models/gguf
> ln -s wevisone-vlm-2b-f16.gguf wevisone-vlm-f16.gguf
> ln -s mmproj-wevisone-vlm-2b-f16.gguf mmproj-wevisone-vlm-f16.gguf
> ln -s wevisone-mask-decoder-2b-f32.gguf wevisone-mask-decoder-f32.gguf
> ```

### Running with llama-cli (upstream compatibility)

The VLM + mmproj pair is a standard llama.cpp `qwen3vl` model and works
with upstream `llama-cli` / `llama-server` for freeform chat:

```bash
llama-cli -m wevisone-vlm-2b-f16.gguf \
    --mmproj mmproj-wevisone-vlm-2b-f16.gguf \
    --image your_image.jpg -p "Describe this image."
```

Note: upstream llama-cli does not load the mask decoder or apply the
WeVisOne task prompts — pixel-level tasks require `wevisone-cli`.

---

## Test data

A self-contained test-data archive is published alongside the models:

**Download**: <https://huggingface.co/Asher-1/WeVisOne-VL-GGML/blob/main/wevisone_vl_test_data.zip>

Contents (25 images + manifest + README):
- 7 official demo photos (detection, referring, counting, pointing, OCR,
  segmentation, matting, visual_prompt, freeform)
- 17 clean input photos extracted from the README case-study figures
  (matting, segmentation, OCR, counting)
- 1 alternate-resolution matting input (seabreeze 1400x933)
- `manifest.json`: 36 entries mapping each image to its task, query, and
  optional pixel-space reference boxes
- `README.md`: task-to-sample mapping and usage notes

```bash
# Download and unzip
wget https://huggingface.co/Asher-1/WeVisOne-VL-GGML/resolve/main/wevisone_vl_test_data.zip
unzip wevisone_vl_test_data.zip

# Run a sample from the manifest
cpp_ggml/build-cuda/wevisone-cli run \
    --image wevisone_vl_test_data/chicken.jpg \
    --task matting --query chicken \
    --vlm-gguf cpp_ggml/models/gguf/wevisone-vlm-2b-f16.gguf \
    --mmproj-gguf cpp_ggml/models/gguf/mmproj-wevisone-vlm-2b-f16.gguf \
    --decoder-gguf cpp_ggml/models/gguf/wevisone-mask-decoder-2b-f32.gguf \
    --device cuda --out chicken_matte.png
```

The `sheep_refseg_none` entry is a deliberate **negative** case (the query
describes a deer absent from the photo); a `none` / empty answer is the
correct outcome, not a failure.

---

## Integrity verification

All model files ship with `SHA256SUMS` in the hub repository:

```bash
# After downloading, verify:
sha256sum -c SHA256SUMS
```

---

## Parity acceptance criteria

| Tier | Criterion |
|------|-----------|
| VLM f16 | Token-identical to PyTorch bf16 reference on all structured tasks (fixed-mode greedy decoding) |
| VLM q8_0 | Structured-answer grammar preserved (`<box>..</box>`, `<OBJ>`, `None`); box IoU >= 0.9 vs f16 |
| VLM q4_K_M | Quality tier — run the full parity suite before production deployment |
| Decoder f32 | MAD <= published thresholds (the only parity-gated decoder tier) |
| Decoder f16 | MAD may drift; gate on your target domain before use |

Benchmark harness: `cpp_ggml/benchmarks/` (fixed/closed modes, IoU gate,
accuracy summary, latency matrix). Charts land in
`cpp_ggml/benchmarks/charts/` after running the suite.

---

## ACloudViewer integration notes

This repository is designed for future integration as a vision module:

- **C API**: `cpp_ggml/include/wevisone/wevisone_capi.h` — stable ABI,
  pure-C smoke test in `cpp_ggml/tests/wevisone_capi_smoke.c`
- **Independent build**: `cmake -S cpp_ggml -B build` with the llama.cpp
  submodule; patch replay is automatic (`cmake/apply_patches.cmake`)
- **Model layout**: point `--models-dir` (or the C API equivalent) at a
  directory containing the triplet; no environment overrides
- **No Python dependency at runtime**: the C++ path is self-contained;
  Python is only needed for the export/conversion scripts
- **Backends**: CUDA, Vulkan, CPU — selected at compile time
  (`-DGGML_CUDA=ON` / `-DGGML_VULKAN=ON`)

---

## Original models & license

These GGUF exports are based on:
- [tencent/WeVisOne-VL-2B](https://huggingface.co/tencent/WeVisOne-VL-2B)
- [tencent/WeVisOne-VL-4B](https://huggingface.co/tencent/WeVisOne-VL-4B)

Released under the same license as the original WeVisOne-VL model
(Apache 2.0 for Tencent-authored code; see the upstream model cards for
weight licensing terms).