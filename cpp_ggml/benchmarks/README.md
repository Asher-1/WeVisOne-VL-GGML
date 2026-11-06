# WeVisOne-VL cpp_ggml benchmarks

Two modes, inherited from the sibling integrations:

- `fixed` — the SAME reference boxes are fed to both backends. This is the
  parity criterion: vision + LLM must be token-identical and mattes must
  stay inside the f32 band; latency differences here measure backend speed.
- `closed` — each backend runs its own full path. This shows the real user
  latency; answers may differ when the proposal path is device-dependent.

## Run

```bash
cmake -S cpp_ggml -B cpp_ggml/build-cuda -DWWV_GGML_CUDA=ON
cmake --build cpp_ggml/build-cuda --parallel 8
bash cpp_ggml/benchmarks/run_benchmark.sh                 # both modes
MODE=fixed bash cpp_ggml/benchmarks/run_benchmark.sh      # parity only
bash cpp_ggml/benchmarks/run_latency_matrix.sh            # precision x backend
python3 cpp_ggml/benchmarks/visualize_cases.py \
    --results cpp_ggml/benchmarks/results --images cpp_ggml/benchmarks/images \
    --out cpp_ggml/benchmarks/charts
```

## Outputs

- `results/<mode>/<backend>/<case>.json` — raw answers, boxes, timings
- `charts/latency_ttft_bar.png` — per-case latency, cpp vs python
- `charts/latency_matrix.png` — {f32, f16, q8_0, q4_K_M} x {cuda, vulkan, cpu}
- `charts/case_*_cpp_py.png` — input | python | cpp effect panels

## Acceptance (aligned with the sibling d10 criteria)

- token identity is the hard constraint for every tier in fixed mode;
- f32/f16 pass the `ratio <= 1.0` noise band, q8_0 `<= 2.5`;
- the mask decoder keeps its f32 gate (0.05/0.05) regardless of the VLM tier.
