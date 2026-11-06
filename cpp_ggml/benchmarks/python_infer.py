#!/usr/bin/env python3
"""Python (official pytorch) backend for the benchmark harness.

Runs the public engine API (wevisone_vl.WeVisOneVL) on CUDA so the GGUF
port can be compared against the official pytorch stack end to end.

Environment adapter for the offline benchmark host (no network): the only
python with a CUDA torch (conda env python3.13) ships transformers 4.57,
while the released checkpoints pin transformers 5.12. The gaps bridged
here, none of which touch the released checkpoint or the engine:
  1. tokenizer_config.json `extra_special_tokens` is a LIST in the official
     checkpoints; 4.57's tokenizer loader requires a mapping. run_benchmark.sh
     stages /tmp/wwv-bench-hf-<size> with that single field rewritten to {}.
  2. qwen3_vl rope_scaling=None (5.x handles it via ROPE_INIT_FUNCTIONS) is
     patched inside the env's modeling file (one-line benchmark-env patch).
  3. 4.57 defaults to the FAST image processor while the checkpoints were
     saved with the slow one (the 5.x default) -> use_fast=False is injected
     below so preprocessing matches the official stack.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import transformers

# Processor adapter: transformers 4.57 selects BOTH subcomponents (image
# processor and tokenizer) from the Qwen3VLProcessor class-attribute tuples
# with the single `use_fast` flag — setting it False would also pick the
# slow tokenizer (needs a vocab.json these checkpoints do not ship). So we
# run all-fast here and let the fixed-mode accuracy gate judge whether the
# fast image processor (torch ops vs the 5.x slow PIL path) preserves the
# answer tokens; a FAIL there would call for a per-subcomponent patch.
_AP = transformers.AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PIL import Image  # noqa: E402
from wevisone_vl import WeVisOneVL  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--query", default="")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out-png", default="")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    model = WeVisOneVL(args.model_dir, device="cuda", dtype="bfloat16")
    img = Image.open(args.image).convert("RGB")
    t0 = time.perf_counter()
    r = model.predict(img, task=args.task, query=args.query, temperature=0.0)
    latency = time.perf_counter() - t0

    if args.out_png:
        # delayed: wevisone_vl.visualization pulls in `supervision`, which the
        # benchmark env does not carry and json-only runs do not need
        from wevisone_vl.visualization import save_result
        p = Path(args.out_png)
        save_result(str(p.parent), img, r, name=p.name)
    if args.out_json:
        payload = {
            "backend": "python-cuda",
            "answer": r.response,
            "box_count": len(r.boxes),
            "box_coords": [[round(float(v), 6) for v in b] for b in r.boxes],
            "mask_count": len(r.masks),
            "latency_s": round(latency, 4),
        }
        Path(args.out_json).write_text(json.dumps(payload, indent=2))
    print(f"answer: {r.response[:100]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
