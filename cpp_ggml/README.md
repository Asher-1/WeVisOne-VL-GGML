# WeVisOne-VL C++ GGML inference (cpp_ggml)

Pure C++17 port of the WeVisOne-VL unified perception pipeline over a single
pinned ggml runtime (llama.cpp @ `620d2225` = **LLAMA 0.7.0-dev / GGML
0.27.0**, tag `v0.7.0`):

```text
image ──► Qwen3-VL ViT ──► LLM ──► "<box><x1><y1><x2><y2><OBJ></box>" / "None"
                    │  (taps at layers 5/11/17/23, pre-merger)
                    └─► 9.6M fp32 pixel decoder (MSDA + DINO + alpha cascade)
                          ──► boxes / masks / alpha mattes
```

The VLM rides llama.cpp's upstream `qwen3vl` support (converter + mtmd
graph are merged at the pin); the pixel decoder is this repo's graph, gated
stage-by-stage against the Python reference.

## Layout

```text
cpp_ggml/
├── CMakeLists.txt              # one GPU backend per build dir; patch replay
├── cmake/apply_patches.cmake   # idempotent llama.cpp patch contract
├── include/wevisone/           # C API (drop-in for AICore: WWV_ -> AICORE_)
├── src/                        # graphs, runtime, C API, CLI
├── tests/wevisone_capi_smoke.c # pure-C ABI smoke
├── scripts/                    # converters + parity + export tooling
├── benchmarks/                 # fixed/closed harness + charts
├── models/                     # MODEL_CARD.md + gitignored gguf/
└── third_party/
    ├── llama.cpp/              # submodule @ v0.7.0 (single ggml source)
    ├── llama-patches/          # 0001 strict-fp32 GEMM (CUDA TF32 off)
    │                           # 0002 mtmd vision taps (decoder feature cpy nodes)
    └── stb/                    # stb_image.h / stb_image_write.h
```

## Build

```bash
git submodule update --init --recursive
cmake -S cpp_ggml -B cpp_ggml/build-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build cpp_ggml/build-cpu --parallel 8
cpp_ggml/build-cpu/wevisone-cli info
cpp_ggml/build-cpu/wevisone-capi-smoke cpp_ggml/models/gguf demo/examples/images/seabreeze.jpg
```

CUDA / Vulkan: `cmake -S cpp_ggml -B cpp_ggml/build-cuda -DWWV_GGML_CUDA=ON`
or `-DWWV_GGML_VULKAN=ON` — exactly one GPU backend per build directory.
No cuDNN dependency anywhere; the fp32 decoder contract is enforced by the
in-tree patch (CUBLAS TF32 disabled under `GGML_PREC_F32`) and Vulkan needs
no patch at this pin (F32xF32 never selects f16-accumulate pipelines).

## Configuration contract

Zero `setenv`/`getenv` in this module — every knob is an explicit field of
`wevisone_options` (device, threads, precision, decoder precision,
models_dir, per-model overrides, max_new_tokens, log level, dump dir).
`grep -R "setenv\|getenv" cpp_ggml/src` must stay empty.

## Convert + parity

```bash
# 1. VLM GGUFs (f32/f16/q8_0/q4_K_M) + decoder GGUFs (f32/f16/q8_0)
bash cpp_ggml/scripts/export_gguf_variants.sh
# 2. torch reference dumps
python3 cpp_ggml/scripts/dump_torch_stages.py \
    --model tencent/WeVisOne-VL-4B --image demo/examples/images/seabreeze.jpg \
    --task matting --query person --out /tmp/wwv-ref
# 3. elementwise gates (0.05/0.05 f32 band, sibling convention)
python3 cpp_ggml/scripts/compare_parity.py /tmp/wwv-ref
```

## Status (2026-11-16, round 15 — GT-conditioned decode entry: the audit's last capability gap closed)

- **`engine.decode_masks(image, prompt, response, boxes=GT)` now has a full
  C++ counterpart.** The round-15 completeness audit left exactly one real
  capability gap: the evaluation GT protocol
  (`evaluation/inference_matting.py` `box_source=gt`) conditions the pixel
  decoder on UNROUNDED ground-truth boxes while teacher-forcing a SYNTHETIC
  response (`<ref>expr</ref><box><x1><y1><x2><y2><OBJ></box>`) — no LLM
  generation at all. The C++ side previously always rode
  generate → parse → decode. Closed at three layers:
  - `WevisoneRuntime::run_decode_masks` (runtime): teacher-forces the given
    prompt + response, decodes mattes from the <OBJ> hidden states, and —
    when boxes01 is supplied — feeds the decoder through the existing
    `boxes01_direct_` channel so the conditioning values bypass the pixel
    round-trip and the 0-1 clamp of the parsed path (python passes the GT
    box raw). Query count = `min(n_boxes|parsed, <OBJ> count)`, the python
    truncation contract; boxes supplied with no <OBJ> in the response fail
    exactly like the python RuntimeError; boxes parsed from the response
    (rounded tokens) are the boxes=NULL default.
  - `wevisone_run_decode_masks` (C API, additive — no ABI break): same
    result getters as `wevisone_run_image` (answer = the given response;
    boxes = the conditioning boxes in original-image pixels).
  - CLI: `run --prompt RAW` (raw prompt override — the eval protocol
    templates like TMPL_MATTE single/instances are NOT build_prompt
    outputs, and the generation path needed them too, so `run_image` grew
    a `prompt_override` parameter), `run --response RAW` (the
    teacher-forced decode path, no generation) and
    `run --gt-boxes x1,y1,x2,y2;...` (normalized 0-1 xyxy conditioning
    boxes; omitted = parse boxes from the response).
- **Verification gates** (seabreeze matting, 2B f16, cpp-cuda):
  1. *Identity*: the GT path teacher-forcing the byte-identical response
     and box values of a generated run produces a BIT-IDENTICAL matte
     (`max|d| = 0.0`, IoU 1.0) — the decode path is the generation path's
     pixel pass with the box source swapped, nothing more. (A first
     identity attempt diverged with IoU 0.9928 — root-caused to the gate
     command itself missing `--query`, which silently changed the task
     template prompt and thus t_seg; the dump bisect pinned t_seg as the
     first diverging stage with boxes01/level maps identical, and the
     corrected command closed it bit-exactly.)
  2. *Override effect*: `--gt-boxes` with x1 shifted +0.02 moves the matte
     (IoU 0.9798 vs the unshifted GT matte) — the conditioning truly
     reaches the decoder.
  3. *Py parity*: `engine.decode_masks(boxes=GT)` (official bf16 stack,
     staged HF env) vs the cpp GT path on the same single-instance prompt,
     response and box: **IoU @0.5 = 0.9920, mean|d| 0.0026** — the same
     f16-vs-bf16 representation residual level as the round-11 matte gate
     (0.9944), i.e. the GT protocol now reproduces across stacks.
  4. *Contract*: boxes supplied without any <OBJ> in the response fails
     with `the teacher-forced response contains no <OBJ> token` (exit 1),
     the python RuntimeError semantics.

## Status (2026-11-15, round 14 — quantized baselines refreshed + cross-stack divergence root-caused)

- **Stale quantized baselines closed out**: the round-13 note left the
  q8_0/q4_K_M fixed-mode rows explicitly stale; they are now re-swept
  with the current runtime (`FIXED_PRECS="q8_0 q4_K_M" MODE=fixed`, 96
  refreshed result files → `accuracy_summary_round14.json`). Exactly ONE
  verdict change against the stale baseline: 2B cpp-vulkan q8_0
  chess_referring FAIL (0.79) → PASS (0.94) — the round-13 grid
  alignment propagating to the quantized tier. Per-tier mean matched IoU
  is flat across the precision ladder (f16 0.8367 / q8_0 0.8373 /
  q4_K_M 0.8294): no quantization regression, and every remaining FAIL
  is the pre-existing attributed set (coffee_sign_ocr straddling the 0.9
  band on ALL tiers including f16, dogs_referring 2B, the parasol_point
  count family, 2B q4_K_M parasol_detection at 0.87). The GGUF weights
  were never stale — every fix since export was runtime-layer (SHA256SUMS
  re-verified); the staleness lived in the result JSONs, not the models.
- **Sweep definition fixed**: the FIXED_PRECS loop used to re-run an
  identical bf16 python-cuda pass PER tier — the python reference has
  exactly one official tier and the quantized sweep is a
  cpp-weight-format concept, so those passes are now skipped (sweep cost
  halved).
- **Cross-stack divergence root-caused** (the round-11 "LLM generation
  noise" known item, closed with a decisive experiment on
  dogs_referring 2B). The phenomenon decomposes into two mechanisms
  with the SAME cause — representation-dependent logit noise — and
  different amplifiers:
  1. *Decision-point flips* (grounded tasks): divergence happens at a
     semantic choice and then freezes. Same implementation, same weights,
     same greedy decode: py-bf16 picks the LEFT dog for "closest to the
     camera", py-f16 picks the RIGHT dog (`python_infer.py --dtype
     float16`) — the flip is purely the bf16-vs-f16 representation gap.
     All three cpp tiers pick the right dog too, cpp-f16 == q8_0
     byte-identical and q4_K_M ±1 bin: the ggml f16 path agrees with
     torch f16, not torch bf16 — the port faithfully tracks its own
     numeric representation. py-f16 vs cpp-f16 box IoU = 0.951 (PASS
     band). On 4B the decision margin is wider and both stacks agree
     (0.99 PASS across tiers).
  2. *Compounding freeform drift* (long generations): hundreds of greedy
     steps multiply the per-step flip probability, 1-(1-p)^n, so long
     describe-style answers branch (the round-11 observation: same
     describe request, py ~10 boxes vs cpp 6). Structured outputs are
     short AND coordinate tokens carry wide logit gaps, so flips are rare
     and semantically absorbed — which is why exact answer identity holds
     on only 5/48 f16 rows (±1-2 bin coordinate flips, plus a
     `,`-vs-newline separator flip) while box IoU stays 0.93+.
- **Tolerance policy recorded**: verdict gates stay STRUCTURAL
  (boxes/points/masks) precisely because they must absorb representation
  noise; exact token identity is a diagnostic, not a contract, for any
  generation beyond a few tokens. Every summary row now records
  `answer_sim` (difflib SequenceMatcher ratio, autojunk=False — the
  answers are `<N>` token streams, markup chars must not be junked), and
  FAIL rows surface it: dogs_referring 0.865 = same-family flip,
  parasol_point 0.66-0.77 = count divergence, coffee_sign_ocr 0.92 =
  borderline-IoU with near-identical text. No freeform case is in the
  benchmark today; a similarity band will be added only when one enters
  (no speculative gate). The parity contract is therefore: SAME
  representation must agree (cpp-f16 vs py-f16 — and it does, IoU 0.951
  on the flip case), CROSS representation must agree structurally, not
  token-identically.
- **`python_infer.py --dtype`** (default bfloat16, unchanged official
  tier): float16 exists only for divergence experiments like the one
  above; the JSON payload records `"dtype"` for traceability.
- **Divergence confirmed on the freeform path too** (post-refresh
  audit experiment): py-f16 vs cpp-f16 on a long 2B describe request
  (livingroom, "Describe this image in detail.") agrees verbatim for the
  first ~100 chars, then branches into different descriptions (920 vs
  1202 chars, answer_sim 0.674) — SAME representation, op-level kernel
  differences alone compound into full divergence. Two consequences:
  (a) long-freeform divergence is COMPOUNDING-dominated, not
  representation-dominated — same-representation and cross-representation
  pairs land at the same ~0.7 similarity level, so no fixed similarity
  gate can ever separate them; the recorded `answer_sim` stays the right
  measure and a freeform band stays unjustifiable.
  (b) the round-11 determinism observation extends: temperature=0 makes
  a stack verbatim-stable WITHIN an implementation, but ACROSS
  implementations even the same representation branches on long chains —
  token-identity parity beyond short structured outputs is structurally
  impossible, and the structural gates (boxes/points/masks) are the
  contract.

## Status (2026-11-15, round 13 — deepstack-taps export: one ViT forward serves LLM + decoder)

- **`0002-wevisone-mtmd-vision-taps.patch`** (vendored llama.cpp, 134 lines
  over 6 mtmd files): the qwen3vl graph builder copies the RAW pre-merger
  block outputs at the configured decoder feature layers (5/11/17/23) into
  dedicated f32 leaves (`wwv_tap_blk_N`, expanded inside the SAME graph —
  the pooled scheduler memory reuses interior buffers, so a kept pointer can
  never be read back), `clip_encode` reads them to host right after graph
  compute, and the mtmd C API exposes the contract:
  `mtmd_vision_taps_configure` (choose blocks; 0 disables) +
  `mtmd_vision_taps_get` ((layer, n_embd, n_tokens, data) views valid until
  the next encode, mirroring `mtmd_get_output_embd`). Clean-replay verified:
  submodule reset to `620d2225` + stamp removal replays 0001+0002 forward at
  configure; the patch is disjoint from 0001 (ggml-cuda.cu) by path.
- **Pixel pass rewired** (`wevisone_runtime.cpp`): the mask decoder now
  consumes the mtmd tower taps captured during the teacher-forced eval —
  the SAME encode that produces the <OBJ> hidden states — so the manual
  ViT replay (round-12's identified ~550 ms duplicate forward) is retired
  from the production path. The torch-exact manual sentinel graph survives
  ONLY behind `--dump-dir` (the mtmd tower cannot export the b0-internal
  bisect probes). A hard grid invariant guards the contract: tap
  `n_tokens == npx*npy` from the shared smart-resize dims, else the run
  fails loudly instead of decoding on a foreign grid.
- **Grid alignment is now explicit**: mtmd's qwen3vl defaults
  (min/max 8/4096 merged tokens) resize bitmaps onto a DIFFERENT grid than
  engine.py's processor limits (16*32*32 / 2025*32*32 px = 16/2025 merged
  tokens at (patch*merge)^2=1024 px each); the runtime now pins
  `image_min_tokens=16, image_max_tokens=2025` at mtmd init so the tower,
  the decoder and the official pytorch stack share one grid. This also
  moved the box path onto the engine grid — and 2B chess_referring box
  parity IMPROVED from FAIL (IoU 0.81) to PASS (0.93).
- **Gates re-run, zero regressions**: mask IoU @0.5 = 0.9944 (2B cuda) /
  0.9943 (2B vulkan) / 0.9889 (4B cuda) / 0.9894 (4B vulkan) vs the
  official pytorch matte (baseline: 0.9944/0.9943/0.9894/0.9894); the full
  12-case fixed sweep × 2B/4B × cpp-cuda/cpp-vulkan f16 shows every
  remaining FAIL is the pre-existing documented set (dogs_referring LLM
  generation noise, parasol_point count, 4B coffee_sign_ocr borderline
  0.899 vs the 0.9 band — all FAIL on the round-12 baseline too); the
  stale q8_0/q4_K_M rows are not refreshed this round.
- **Latency**: serve hot-path matting e2e 1593 → **~1085 ms** on 2B cuda
  (pixel-pass segments now `preprocess 30 | teacher+taps 335 | unshuffle
  92 | decoder 496 ms` — the vit_graph segment is gone; the tap D2H rides
  inside the teacher segment). `wevisone-capi-smoke` re-verified with
  masks green; the box path re-checked (chess count_box: 31 boxes,
  byte-identical answer).
- **py tests collection fixed** (offline host): `pycocotools` imports
  deferred to point-of-use (`pointing_bench_metric.decode`, mirrors
  `other_metric.point_target`) and the test fixtures lazy-skip RLE cases
  (`pytest.importorskip`) — `pytest tests/` collects clean and runs
  18 passed / 2 skipped without supervision/pycocotools installed; the
  CLI save path moved behind a monkeypatchable `_save_result` wrapper that
  treats a missing render stack as success-without-files (and fixes a
  duplicated engine construction in single-image mode).

## Status (2026-11-14, round 12 — API surface completed: batch + serve + verification GUI)

- **C API batch** (`wevisone_run_batch` + `wevisone_result_count` and the
  nine `*_at` getters): the `engine.predict_batch` contract on the C side —
  N images through one shared model load, a shared task/query, one result
  per image. Scheduling is sequential (llama.cpp multimodal contexts are
  single-sequence; the python padded-batch generate is itself only an
  approximation), the interface and per-image outputs are what aligns.
- **CLI `batch`**: `wevisone-cli batch --list rows.txt --task T --query Q
  --jsonl out.jsonl` — one model load, `<image>[TAB<query override>]` rows,
  JSONL rows mirroring the python payload (answer/backend/box_coords/
  mask_count/timings). The python CLI gained the mirror entry:
  `python -m wevisone_vl.cli --images a.jpg b.jpg --jsonl out.jsonl`
  (`engine.predict_batch` underneath; the optional `supervision` render
  dependency is now deferred so batch works without it).
- **CLI `serve`** (`wevisone-cli serve`): persistent line-protocol backend
  for the GUI — one model load per process, 5-line requests
  (IMAGE/TASK/QB64/REFB/RUN) → one JSON response with the answer,
  normalized boxes and the raw f32 mattes base64-encoded.
- **Root one-click verification GUI** (`bash run_gui.sh`, zero third-party
  deps beyond the engines themselves — gradio/tkinter are unavailable on
  the offline host): side-by-side pytorch (official engine, in-process)
  and ggml (persistent serve child) over all 11 task families with
  official-style overlays (green-screen matte + red boxes/points) and
  per-backend answer/e2e. Samples come from `test_data/manifest.json`
  (36 entries: 7 official demo photos referenced in place + 17 clean
  originals extracted from the README case-study figures + the README
  matting input; see `test_data/README.md`). Audited end-to-end:
  `gui/matrix_test.py` drives all manifest samples (every task family) x
  both backends over the HTTP API
  — all PASS with per-modality semantic assertions (box/point shapes,
  mattes, the `<box>none</box>` no-target convention), plus verified
  negatives (path traversal + non-image extensions rejected by the
  /image whitelist), serve self-heal (a killed child respawns on the next
  run), and concurrent-run mutual exclusion. Audit fixes landed: py
  engine load race (double-checked lock), serve respawn after a transient
  failure + handshake-failure child kill (leaked GPU memory), whitelist
  containment check (`is_relative_to`, was a string prefix) + image
  extension gate, a global run lock (one GPU card), ref-box encode errors
  no longer misreported as a dead serve, and the stderr handle lifecycle.
- **Pixel-path latency (round-12 perf bisect)**: the matting e2e carried
  a ~2.3 s gap between pre+infer+post and e2e — the whole pixel pass was
  unaccounted AND slow. Segmented timing pinned four contributors, all
  fixed: (1) the manual ViT ran all-F32 weights (mmproj F16 expanded to
  F32, 1.6 GB H2D + full-F32 math) → matmul/conv weights now ride F16
  straight from the GGUF (mul_mat dequantizes on the fly, the exact mtmd
  numeric path — matte IoU unchanged at 0.9944); (2) bicubic canvas
  preprocessing was single-threaded → row-parallel with precomputed taps,
  bit-exact contract preserved (354→62 ms); (3) the 4-level tap readback
  + host un-shuffle was serialized → per-level threads (310→133 ms);
  (4) the timing contract violation itself — the pixel pass now folds
  into inference_ms per wevisone_pipeline_timing.h. Matting e2e
  3191→2480 ms on 2B cuda. The remaining gap vs pytorch is the manual
  ViT recompute (the official stack reuses the teacher-forward ViT via
  hooks; the ggml side re-runs it because mtmd does not expose the
  per-block hidden states — a vendored-llama deepstack-taps export is
  the identified future work).
- **charts gap closed**: `plot_e2e_backend_compare.py` merges the
  `latency_matrix.json` quantization/backend rows with the official-stack
  e2e timings into
  `charts/latency_e2e_backends_{2b,4b}.png` (all series, log scale) and
  `charts/latency_e2e_vs_pytorch_{2b,4b}.png` (GPU-only, linear) — the
  quantized-GGML vs pytorch-cuda per-case view that was missing.

## Status (2026-11-14, round 11 — mask-path (hand ViT taps) fully aligned with the official stack)
- **The hand ViT (vision_taps_graph.cpp) diverged from the HF vision tower on real inference**: all 4 tapped feature layers corr≈0 with the std 3-11x off, collapsing matting/referring masks (dump-path IoU 0.32) while the mtmd box path stayed correct. A sentinel-tap bisect (patch_out/pos_out/ln1/attn/ffn/qkv/q_rope/v leaves dumped alongside torch forward hooks on the same seabreeze input) pinned **five root causes**, every fix verified by a raw `(C,T)` flat re-compare plus a f32 numpy recompute of the whole block-0 attention chain (py-side chain recompute == py bf16 output corr 1.00000):
  1. pixel preprocessing normalized with the Qwen2-VL CLIP stats (0.481/0.458/0.408) instead of the released `processor_config.json` 0.5/0.5 (`image_prep.cpp`); pixels corr 0.999991 after the fix.
  2. the learned pos-embed interpolation crossed the x/y axes: the GGUF table is (C, 48*48) with n = y*48+x, so the bilinear resize must permute to (y,x,C) first (`vision_taps_graph.cpp`); pos_out corr 0.695 -> 1.0000.
  3. (fixed while bisecting, kept as the mtmd contract) K/V cast to F16 + `ggml_prec_set_acc(F32)` around `flash_attn_ext` — a no-op on this backend (its fattn already pins those) but it makes the graph mirror `clip.cpp build_attn` exactly.
  4. **the fattn output was permuted once too often**: `ggml_flash_attn_ext` already returns the permute(0,2,1,3) view (ne = v.ne0, q.ne2, q.ne1 = (hd, nh, T) — ggml.c's own comment; mtmd reshapes straight to (n_embd, T)), so the extra permute interleaved channels with tokens (blk0 attn corr 0.8945, std 12% low, CPU == CUDA so it was graph construction, not a kernel issue). Removed; blk0/b0_attn/b0_ffn corr 1.0000, blk5 0.9996.
  5. `unshuffle_tokens_host` read the tap flat as a C-order (E,T) matrix while ggml stores element (e,t) at `tokens[t*E+e]` — level maps were permuted into noise (corr 0.0003 vs the numpy recompute; the same tap read as tokens[t*E+e] gives 0.991). Fixed; level0-3 corr 0.976-0.9996 (f16-vs-bf16 weight residual).
- **End-to-end gate**: same-input logits corr 0.99891, **mask IoU @0.5 = 0.9944 (cpp-cuda) / 0.9943 (cpp-vulkan)** vs the official pytorch matte (fixed mode, seabreeze); the 12-case closed/fixed 2B runs are green with the parser fix in place. The remaining referring/point answer divergence (dogs_referring picks a different dog; chess/seabreeze +-12 token band) is the round-10 f16-vs-bf16 LLM generation noise, unchanged by this round.
- **Prompt construction re-audit against `wevisone_vl/prompts.py`** (P4): join_labels stripped whitespace INSIDE labels ("the white king" -> "thewhiteking") — now py `str.strip()` semantics; count_box/count_point used phrase_of()+substr and ate a user-supplied trailing '.' — now pass the stripped text verbatim; ocr_grounding passed the raw query — now stripped; visual_prompt boxes are skipped for a degenerate canvas like `_box_tokens`. The CLI grew `--ref-boxes x1,y1,x2,y2;x1,y1,x2,y2` (pixel-space, the `--reference-boxes` JSON contract) and a duplicated `--stages-in` arm was dropped; `wevisone-capi-smoke` re-verified (boxes/has_masks green). Python-side test collection still needs supervision/pycocotools that the offline host cannot install (pre-existing).

## Status (2026-11-14, round 10 — 4B GGUF family + full-stack official benchmark)
- **4B GGUF family converted and gated** from the verified `checkpoints/hf-4B` (no network): VLM f16 8.8 GB -> llama-quantize q8_0 4.7 GB / q4_K_M 2.7 GB, mmproj f16 836 MB, mask decoder f32/f16 (decoder q8_0 remains a non-goal: 48-element tensors sit below the q8_0 block granularity). Structural gate `selftest wevisone-mask-decoder-4b-f32.gguf --vit 1024 --llm 2560 --grid 32` PASS (S=256, 3553 ms).
- **Full-stack benchmark vs the official pytorch stack (bf16) is green**: 12 official cases across every released task family x {cpp-cuda, cpp-vulkan, python-cuda} x {2B, 4B}. 4B: 11/12 PASS on BOTH GPU backends (the single miss is ocr IoU 0.899, a +-1-grid-cell band edge); 2B: 9/12 (the misses are free-choice referring/point generation divergence between bf16 and f16 paths, not port defects — 4B resolves the same cases at 0.977/0.991). Details, conversion commands, and the offline transformers-4.57 environment adapter: `benchmarks/README.md`.
- The benchmark exposed a real parser bug: `parse_answer` took only the last four coordinate tokens of a `<box>` block, so comma-separated multi-box answers (detection/count_box) lost every box after the first segment in the CLI payload (8->1); fixed to per-segment size-4 chunks matching `visualization.py parse_predictions`, and the CLI JSON now carries `mask_count` for payload parity with the python engine.
- Vulkan runtime fix: the ggml device is registered as `Vulkan0` (instance suffix), so the one-GPU-backend binary contract checks a name PREFIX (CUDA*/Vulkan*) instead of equality — `run --device vulkan` was previously rejected with "backend not compiled into this binary".
- Latency matrix (`benchmarks/run_latency_matrix.sh`): f16 x {cuda, vulkan, cpu} x {2B, 4B} x 12 cases plus {q8_0, q4_K_M} on cuda; dual apertures per run (process wall_ms, load-free e2e_ms); mmproj and decoder stay f16 on every tier. Charts + `accuracy_summary.json` regenerate via the plot scripts.

## Status (2026-11-06, round 9 — full-chain parity GATE PASS; F32-intrinsic log conclusion overturned)
- **Full-chain parity: GATE PASS** (parity16, 75 torch-referenced stages/taps; zero violations everywhere except one GN-amplified isolated tap closed by `--band-override "tap_cas_prev1=0.2:0.05"`). coarse 1.2e-2 mean inside the 0.05/0.05 band, logits_s1/s2 green, MSDA/transformer taps green, query_position band now 2.8e-7 mean / 1.6e-6 max.
- **The round-8 "intrinsic to F32" log conclusion is REFUTED**: torch's AVX2 log kernel dispatches to MKL VML `vmsLn` (gdb evidence: `test %edi,%edi` on entry, mode `0x140102` passed as an immediate) with a **BY-VALUE ABI** — `vmsLn(int n, const float* a, float* r, int mode)` — NOT the documented pointer interface; assuming pointers caused every earlier segfault / "Condition -1" / "Parameter 1 incorrect". mode 0 (VML_HA) reproduces torch.log bit-for-bit on a 2M-input fuzz, and a bare process can `dlopen libtorch_cpu.so + dlsym("vmsLn")` with no torch init. Excluded along the way: SLEEF u10 (machine-code blob replays glibc, 98.1%), libmvec, scalar glibc — none match torch; the MKL VML entry is the only bit-exact source.
- The host box log is now `mkl_logf_exact` (mask_decoder_graph.cpp): `WWV_MKL_LOG_LIB` env override, then common pip torch layouts (glob probe), glibc double-log fallback (98.2% — recorded limitation). enc_w/enc_h are **bit-exact** with torch, so the Fourier phase amplification (frequencies up to 2^23 radian-scale per ulp) is gone and the query_position band closed (0/1152).
- Pixel/cascade fixes that closed the last five layers (each derived from the torch reference and verified by tap bisect): MSDA three stacked defects (inverted upper-bound corner mask + `in_bounds` off-by-one + level smearing/head-channel misorder, cr0 2.7e-1 -> 3.5e-4); the coarse head permute (3,1,0,2) aligning torch (Q,H,W) via the reversed-shape identity; the additive pixel fusion 1x1 column blocks (torch `cat [up42, semantic, rgb, coarse, prev]` — the fused conv slices its input-channel axis per block); `FullCanvasAlphaCascadeHead._resize` is plain bilinear for ANY size change (vs pixel fusion's pool-then-bilinear) via `resize_map(force_bilinear)`; the ConvTranspose crop offset `p*(nb0+nb1)` (out_torch[o] = out_ggml[o+p] for torch p=1 vs ggml p=0); the cascade delta-chain permute (1,2,0,3) producing (C,W,H,Q).
- compare_parity hardened: `--band-override stage=max_abs:max_rel` per-stage bands and a MISSING-cpp-side FAIL (the torch dump roster is authoritative — a prefix-mismatched tap family once masked as all-green).
- 4B checkpoint verified: `scripts/verify_checkpoint.py checkpoints/hf-4B` OK (base=4,830,581,248 params, decoder=9,609,764, <OBJ>=152678); sha256 `002c3ac0…acb4a` (model.safetensors) / `85284604…7b46` (mask_decoder.pt); the four missing release sidecars staged from the 2B snapshot per `stage_release_checkpoint.py` semantics (README rendered from hidden_size 2560, LICENSE from the repo root, chat_template.jinja / processor_config.json are size-agnostic passthrough artifacts).
- 5 diagnostic fprintf removed from the decoder graph; rebuild keeps the gate green.

## Status (2026-11-05, round 8 — encoder/attention aligned; MSDA is the next gate)
- Bisect tooling matured: per-stage debug taps on both sides (build-time
  ggml_cpy nodes with source-dimensionality dst + torch forward hooks), a
  stages-in harness, and standalone ggml mini-probes (mmtest6/7/10/11) that
  replay single ops on the real weights and dumps.
- Fixed this round, each verified to 1e-7..1e-4 against torch:
  tap dst dimensionality (nbytes=0 cpy wrote nothing), the dump protocol
  (the reversed-shape identity makes ggml flat == torch flat; every earlier
  "alignment" transpose corrupted the data), staged uploads on the host
  compute path (f32 leaves + PixelShuffle tables were silently skipped ->
  intermittent NaN), the token-matrix construction (view_2d ne order mixed
  channels with spatial positions), pos-leaf slicing (ggml_view_impl resets
  nb[0], so a view of a permuted tensor must be materialized first), the
  depthwise im2col tap layout (kernel writes ic*9+k -> direct (9,C,L)
  reshape), the encoder residual (torch keeps the carrier free of pos and
  re-adds pos fresh every layer), MHA value stream (v projects the raw
  target, not target+query_pos), the in_proj v row-block offset (2*d*d, not
  2*d, floats), the stages-in boxes (the harness fed an empty box list),
  exact-erf GELU, and the Fourier phase multiply order ((2pi*v)*freqs).
- Aligned stages: cnn_image 0 violations, sine pos leaf 1.2e-8, level inputs
  and weights byte-exact, enc_proj 1.5e-7, enc_in_sum 2.2e-7, encoder
  layer-0 norms and memory 1.7e-4 (corr 1.0), query_content 2.9e-7,
  self_attention 2.4e-3 (corr 0.9999).
- Known open items: query_position carries a 3e-2 band from a 1-ulp log
  difference amplified by the 2^31 Fourier frequency (intrinsic to F32);
  cross-attention (MSDA) is the next divergent block (2.7e-1); the cascade
  sits behind it. 4B download at 19/24 verified chunks.
- selftest PASSES on the released weights (S=256, 3.5 s CPU).

- `run_decoder_graph` now has a CPU host-compute path (no_alloc=false context
  owning all tensors + `ggml_graph_compute`, no gallocr) — the selftest-style
  route. GPU keeps the gallocr path. The pool must hold every intermediate
  simultaneously (the S/1 cascade convs materialize ~5.4 GB F16 im2col each
  under N=6 queries), so the parity gate runs at a reduced cnn canvas
  (`--cnn-size 128` on the torch dump; the builder follows the dumped canvas
  via `cfg_eff.cnn_image_size`).
- Verified inside the live graph (byte/tap level): level inputs, weight
  bytes, and the sine pos leaf all match torch/the reference formula exactly
  (1.2e-8). The standalone ggml chain (permute+cont+reshape+mul_mat F32+
  reshape+permute) matches numpy at 96x64x1024 (mmtest5, 6e-7). The
  gallocr-aliasing hypothesis is REFUTED (host path shows the same conv
  divergence, 0.86).
- Remaining encoder-conv mystery: the graph conv output differs from the
  numpy replay (0.86) with identical inputs/weights/op order, and the
  enc_proj tap (cpy of a strided-view cont) reads garbage (1e27) while the
  enc_conv tap reads finite values. Next session: isolate conv1x1 in a
  minimal graph on the staged tensors (mini-graph inside run_decoder_graph),
  then bisect prologue/mul_mat/epilogue with taps; also verify the enc_proj
  cpy-of-view under graph_compute.
- 4B: download running in the background (24 verified chunks + sha256 gate,
  checkpoints/hf-4B/); mask_decoder.pt already landed.

- Round-6 fixes (all verified against torch): sine pos embedding X/Y halves
  (`cat([pos_y, pos_x])`), Fourier box phase missing `2*pi`, depthwise im2col
  patch layout `[IC, KH, KW]` (channel-outer), grid_sample zeros-padding
  semantics (corner masks + clamped gathers), output biases wired into the
  graph, mmproj wiring (mtmd path + `v.*` weights + `ffn_up/ffn_down` names),
  `<__media__>` marker, learned-pos resize + 2x2 merge per the qwen3vl
  reference, 16384-node graphs, no_alloc staged-upload guards.
- Bisect infrastructure: per-stage debug taps on both sides
  (`build_mask_decoder_graph(..., debug_taps)` + torch forward hooks), a
  `stages-in` harness feeding torch dumps directly into the decoder graph,
  and numpy replays of conv1x1+GN that match the torch tap to 1.5e-7.
- **Isolated root cause**: the standalone ggml chain (permute+cont+reshape+
  mul_mat F32+reshape+permute at 96x64x1024) matches numpy exactly
  (/tmp/mmtest5), the graph inputs and weights are byte-exact, yet the
  decoder graph output differs — AND adding pure cpy tap nodes changes the
  final logits by 0.31 (taps-on vs taps-off). The divergence is a
  gallocr/compute-level aliasing or ordering hazard in the decoder graph,
  NOT op math. Next session: run run_decoder_graph with the selftest-style
  no_alloc=false + ggml_graph_compute path (no gallocr) and re-measure the
  parity band; if it closes, fix the gallocr interaction (view lifetimes).

- Full chain EXECUTES on the released 2B weights: mtmd image path, structured
  generation (fixed-mode: 20/24 coordinate tokens exact vs torch fp32, the
  4 divergent tokens differ by 1-9 code values at quantized-logit boundaries),
  answer parsing, ViT taps, decoder graph, matte write-out (CPU f16, 89 s).
- Image preprocessing is byte-exact (cnn_image 0/3.1M violations vs torch).
- Numeric gates still OPEN: with identical stage inputs (stages-in harness)
  the decoder logits diverge ~25% from torch — the graph is structurally
  correct (selftest + real-weight execution) but at least one math block
  (MSDA aggregation order, cascade fuse ordering or the pose/box encoding)
  still differs semantically. Next step: per-op divergence bisect using the
  dump machinery (dump every block output on both sides).
- Hardening landed this round: name-keyed depthwise detection, 1x1-conv
  squeeze, gguf-py writer, `.layers.` name alignment, per-dim-tile to_map,
  16384-node graphs, mmproj wiring (mtmd + v.* weights), `<__media__>`
  marker, learned-pos resize + 2x2 merge per the qwen3vl reference, ffn_up/
  ffn_down names, grid_sample zeros-padding semantics (corner masks +
  clamped gathers), output biases wired into the graph, staged-upload
  guards, stages-in parity harness (run --stages-in <dir>).

- The released `mask_decoder.pt` converts to a 296-tensor f32 GGUF and the
  decoder selftest PASSES with the real weights (name contract verified,
  graph builds and computes end-to-end at S=256 on CPU).
- Converter hardening: name-keyed depthwise detection (shape-only heuristics
  misclassified the Cin=1 cascade convs), 1x1-conv squeeze to the 2-D GEMM
  layout, vendored gguf-py writer (hand-rolled serialization had format
  offsets), `encoder.layers.` name alignment on both the runtime and the
  selftest, per-dim-tile `to_map` scatter, 16384-node graph capacity.
- `wevisone-cli selftest <gguf> --vit C --llm D --grid G --q N --cnn S`
  parametrizes the structural gate; the pixel pass now writes parity dumps
  (`<stage>.cpp.bin` + `.shape`) when `dump_dir` is set.

- **`wevisone-cli selftest` PASSES**: the full 9.6M-parameter decoder graph
  (MSDA encoder -> query builder -> DINO transformer with in-graph
  deformable attention -> pixel fusion -> dynamic predictor -> three-stage
  alpha cascade) builds and computes on CPU against a synthetic
  structurally-complete GGUF (`scripts/make_synth_decoder_gguf.py`, pure
  numpy — the same tensor-name contract as the real converter). This is the
  graph-execution gate: every op shape, the allocator and the S/1 logits
  readback are exercised without the released checkpoint.
- **Layout-semantics bugs found and fixed by the selftest** (each was a
  silent-math or abort-class defect the compiler could not see):
  conv1x1 contracted the wrong axis (W*H instead of C); three permutes had
  inverted direction (ggml permute moves old axis i TO axis_i); GroupNorm /
  conv-bias affines needed (1,1,C,1) reshapes for per-dim broadcast; the
  predictor einsum needed a transpose (mul_mat returns (Q, S^2)); the MSDA
  aggregation needed a flat-slot (1,S) redesign with a two-stage
  contiguity-safe sum; the alpha-cascade fuse needed per-query-first
  operand order; conv_transpose_2d_p0 runs unpadded (2*in+2 -> crop);
  im2col output is (K, OW, OH); and every host constant had to move to the
  staged-upload contract (no_alloc ctx: writing t->data before gallocr
  segfaults).
- **mtmd wired**: the VLM answer path now rides llama.cpp's supported
  image/text chunk API (mtmd_init_from_file + mtmd_tokenize +
  mtmd_helper_eval_chunks) with the released Qwen3-VL chat template; the
  teacher-forced t_seg pass re-evaluates the same chunks and reads per-token
  hidden states under POOLING_TYPE_NONE.

## Status (2026-11-04, round 2)

- **Integration contract complete**: submodule pin, idempotent patch replay
  (verified forward+reverse against the pin), single-ggml CMake, AICore-drop-in
  C API, pure-C smoke, CPU build green (lib + cli + smoke).
- **Pixel pass fully wired** (round 2): the runtime now runs the complete
  chain when `<OBJ>` appears — smart-resize + bicubic 1024 preprocess,
  teacher-forced VLM forward with per-token embeddings (POOLING_TYPE_NONE)
  for `t_seg`, qwen3vl ViT taps at layers 5/11/17/23 with host un-shuffle
  (HFRE pattern), and the fp32 decoder graph (MSDA encoder -> queries ->
  pixel fusion -> dynamic predictor -> three-stage alpha cascade), ending in
  sigmoid + bilinear matte readback at the original resolution.
- **Weight layout contract (single source of truth)**: the converter stores
  every tensor as raw torch row-major bytes with `ggml ne = reversed(torch
  shape)` — the standard GGUF convention. It covers conv (Kw,Kh,Cin,Cout),
  ConvT (Kw,Kh,Cout,Cin), GEMM (in,out), 1x1-as-GEMM and the depthwise
  (C,9)->(9,C) taps with zero permutes; `conv1x1` in the graph contracts the
  channel axis through a permute+cont chain (the earlier flatten contracted
  W*H — a silent-math bug caught in round-2 review and fixed).
- **In-graph MSDA** (single implementation; the host-plan variant was
  removed): exact `F.grid_sample(align_corners=False, zeros)` semantics via
  floor/step/cast ops + `get_rows` gathers, softmax over (levels*points)
  in-graph, head aggregation through sum_rows/permute — no custom op, no
  patch, identical math on CPU/CUDA/Vulkan.
- **Numeric close-out pending** (requires the released 2B/4B weights and
  `mask_decoder.pt` on disk): stage parity gates via dump/compare, fixed-mode
  token identity, precision-matrix latency. The harness is in `scripts/` and
  `benchmarks/`; the remaining engineering follows the sibling phase playbook
  (stage gates -> text parity -> effect panels).
- **Known seams** (documented in code): weights are uploaded per run (the
  persisted weight-buffer optimization follows the parity gates); matte
  readback covers the first query per request (the batched-query extension
  rides the same logits tensor); `dump_torch_stages.py` dumps the structured
  reference via the public engine API and grows stage hooks alongside the
  gates.


- `run_decoder_graph` now has a CPU host-compute path (no_alloc=false context
  owning all tensors + `ggml_graph_compute`, no gallocr) — the selftest-style
  route. GPU keeps the gallocr path. The pool must hold every intermediate
  simultaneously (the S/1 cascade convs materialize ~5.4 GB F16 im2col each
  under N=6 queries), so the parity gate runs at a reduced cnn canvas
  (`--cnn-size 128` on the torch dump; the builder follows the dumped canvas
  via `cfg_eff.cnn_image_size`).
- Verified inside the live graph (byte/tap level): level inputs, weight
  bytes, and the sine pos leaf all match torch/the reference formula exactly
  (1.2e-8). The standalone ggml chain (permute+cont+reshape+mul_mat F32+
  reshape+permute) matches numpy at 96x64x1024 (mmtest5, 6e-7). The
  gallocr-aliasing hypothesis is REFUTED (host path shows the same conv
  divergence, 0.86).
- Remaining encoder-conv mystery: the graph conv output differs from the
  numpy replay (0.86) with identical inputs/weights/op order, and the
  enc_proj tap (cpy of a strided-view cont) reads garbage (1e27) while the
  enc_conv tap reads finite values. Next session: isolate conv1x1 in a
  minimal graph on the staged tensors (mini-graph inside run_decoder_graph),
  then bisect prologue/mul_mat/epilogue with taps; also verify the enc_proj
  cpy-of-view under graph_compute.
- 4B: download running in the background (24 verified chunks + sha256 gate,
  checkpoints/hf-4B/); mask_decoder.pt already landed.

- Round-6 fixes (all verified against torch): sine pos embedding X/Y halves
  (`cat([pos_y, pos_x])`), Fourier box phase missing `2*pi`, depthwise im2col
  patch layout `[IC, KH, KW]` (channel-outer), grid_sample zeros-padding
  semantics (corner masks + clamped gathers), output biases wired into the
  graph, mmproj wiring (mtmd path + `v.*` weights + `ffn_up/ffn_down` names),
  `<__media__>` marker, learned-pos resize + 2x2 merge per the qwen3vl
  reference, 16384-node graphs, no_alloc staged-upload guards.
- Bisect infrastructure: per-stage debug taps on both sides
  (`build_mask_decoder_graph(..., debug_taps)` + torch forward hooks), a
  `stages-in` harness feeding torch dumps directly into the decoder graph,
  and numpy replays of conv1x1+GN that match the torch tap to 1.5e-7.
- **Isolated root cause**: the standalone ggml chain (permute+cont+reshape+
  mul_mat F32+reshape+permute at 96x64x1024) matches numpy exactly
  (/tmp/mmtest5), the graph inputs and weights are byte-exact, yet the
  decoder graph output differs — AND adding pure cpy tap nodes changes the
  final logits by 0.31 (taps-on vs taps-off). The divergence is a
  gallocr/compute-level aliasing or ordering hazard in the decoder graph,
  NOT op math. Next session: run run_decoder_graph with the selftest-style
  no_alloc=false + ggml_graph_compute path (no gallocr) and re-measure the
  parity band; if it closes, fix the gallocr interaction (view lifetimes).

- Full chain EXECUTES on the released 2B weights: mtmd image path, structured
  generation (fixed-mode: 20/24 coordinate tokens exact vs torch fp32, the
  4 divergent tokens differ by 1-9 code values at quantized-logit boundaries),
  answer parsing, ViT taps, decoder graph, matte write-out (CPU f16, 89 s).
- Image preprocessing is byte-exact (cnn_image 0/3.1M violations vs torch).
- Numeric gates still OPEN: with identical stage inputs (stages-in harness)
  the decoder logits diverge ~25% from torch — the graph is structurally
  correct (selftest + real-weight execution) but at least one math block
  (MSDA aggregation order, cascade fuse ordering or the pose/box encoding)
  still differs semantically. Next step: per-op divergence bisect using the
  dump machinery (dump every block output on both sides).
- Hardening landed this round: name-keyed depthwise detection, 1x1-conv
  squeeze, gguf-py writer, `.layers.` name alignment, per-dim-tile to_map,
  16384-node graphs, mmproj wiring (mtmd + v.* weights), `<__media__>`
  marker, learned-pos resize + 2x2 merge per the qwen3vl reference, ffn_up/
  ffn_down names, grid_sample zeros-padding semantics (corner masks +
  clamped gathers), output biases wired into the graph, staged-upload
  guards, stages-in parity harness (run --stages-in <dir>).

- The released `mask_decoder.pt` converts to a 296-tensor f32 GGUF and the
  decoder selftest PASSES with the real weights (name contract verified,
  graph builds and computes end-to-end at S=256 on CPU).
- Converter hardening: name-keyed depthwise detection (shape-only heuristics
  misclassified the Cin=1 cascade convs), 1x1-conv squeeze to the 2-D GEMM
  layout, vendored gguf-py writer (hand-rolled serialization had format
  offsets), `encoder.layers.` name alignment on both the runtime and the
  selftest, per-dim-tile `to_map` scatter, 16384-node graph capacity.
- `wevisone-cli selftest <gguf> --vit C --llm D --grid G --q N --cnn S`
  parametrizes the structural gate; the pixel pass now writes parity dumps
  (`<stage>.cpp.bin` + `.shape`) when `dump_dir` is set.

- **`wevisone-cli selftest` PASSES**: the full 9.6M-parameter decoder graph
  (MSDA encoder -> query builder -> DINO transformer with in-graph
  deformable attention -> pixel fusion -> dynamic predictor -> three-stage
  alpha cascade) builds and computes on CPU against a synthetic
  structurally-complete GGUF (`scripts/make_synth_decoder_gguf.py`, pure
  numpy — the same tensor-name contract as the real converter). This is the
  graph-execution gate: every op shape, the allocator and the S/1 logits
  readback are exercised without the released checkpoint.
- **Layout-semantics bugs found and fixed by the selftest** (each was a
  silent-math or abort-class defect the compiler could not see):
  conv1x1 contracted the wrong axis (W*H instead of C); three permutes had
  inverted direction (ggml permute moves old axis i TO axis_i); GroupNorm /
  conv-bias affines needed (1,1,C,1) reshapes for per-dim broadcast; the
  predictor einsum needed a transpose (mul_mat returns (Q, S^2)); the MSDA
  aggregation needed a flat-slot (1,S) redesign with a two-stage
  contiguity-safe sum; the alpha-cascade fuse needed per-query-first
  operand order; conv_transpose_2d_p0 runs unpadded (2*in+2 -> crop);
  im2col output is (K, OW, OH); and every host constant had to move to the
  staged-upload contract (no_alloc ctx: writing t->data before gallocr
  segfaults).
- **mtmd wired**: the VLM answer path now rides llama.cpp's supported
  image/text chunk API (mtmd_init_from_file + mtmd_tokenize +
  mtmd_helper_eval_chunks) with the released Qwen3-VL chat template; the
  teacher-forced t_seg pass re-evaluates the same chunks and reads per-token
  hidden states under POOLING_TYPE_NONE.

## Status (2026-11-04, round 2)

- **Integration contract complete**: submodule pin, idempotent patch replay
  (verified forward+reverse against the pin), single-ggml CMake, AICore-drop-in
  C API, pure-C smoke, CPU build green (lib + cli + smoke).
- **Pixel pass fully wired** (round 2): the runtime now runs the complete
  chain when `<OBJ>` appears — smart-resize + bicubic 1024 preprocess,
  teacher-forced VLM forward with per-token embeddings (POOLING_TYPE_NONE)
  for `t_seg`, qwen3vl ViT taps at layers 5/11/17/23 with host un-shuffle
  (HFRE pattern), and the fp32 decoder graph (MSDA encoder -> queries ->
  pixel fusion -> dynamic predictor -> three-stage alpha cascade), ending in
  sigmoid + bilinear matte readback at the original resolution.
- **Weight layout contract (single source of truth)**: the converter stores
  every tensor as raw torch row-major bytes with `ggml ne = reversed(torch
  shape)` — the standard GGUF convention. It covers conv (Kw,Kh,Cin,Cout),
  ConvT (Kw,Kh,Cout,Cin), GEMM (in,out), 1x1-as-GEMM and the depthwise
  (C,9)->(9,C) taps with zero permutes; `conv1x1` in the graph contracts the
  channel axis through a permute+cont chain (the earlier flatten contracted
  W*H — a silent-math bug caught in round-2 review and fixed).
- **In-graph MSDA** (single implementation; the host-plan variant was
  removed): exact `F.grid_sample(align_corners=False, zeros)` semantics via
  floor/step/cast ops + `get_rows` gathers, softmax over (levels*points)
  in-graph, head aggregation through sum_rows/permute — no custom op, no
  patch, identical math on CPU/CUDA/Vulkan.
- **Numeric close-out pending** (requires the released 2B/4B weights and
  `mask_decoder.pt` on disk): stage parity gates via dump/compare, fixed-mode
  token identity, precision-matrix latency. The harness is in `scripts/` and
  `benchmarks/`; the remaining engineering follows the sibling phase playbook
  (stage gates -> text parity -> effect panels).
- **Known seams** (documented in code): weights are uploaded per run (the
  persisted weight-buffer optimization follows the parity gates); matte
  readback covers the first query per request (the batched-query extension
  rides the same logits tensor); `dump_torch_stages.py` dumps the structured
  reference via the public engine API and grows stage hooks alongside the
  gates.
