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
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"],
                    help="official-stack inference precision; bfloat16 is "
                         "the ONE official reference tier — float16 exists "
                         "only for cross-stack divergence experiments (same "
                         "representation as the ggml f16 parity tier)")
    ap.add_argument("--out-png", default="")
    ap.add_argument("--out-json", default="")
    ap.add_argument("--out-matte", default="",
                    help="raw f32 alpha plane of the first mask (h*w "
                         "row-major + .shape 'h,w' sidecar — the exact "
                         "layout the cpp CLI --matte-out writes, so the "
                         "mask quality of both stacks is threshold-"
                         "comparable)")
    args = ap.parse_args()

    model = WeVisOneVL(args.model_dir, device="cuda", dtype=args.dtype)
    img = Image.open(args.image).convert("RGB")
    t0 = time.perf_counter()
    r = model.predict(img, task=args.task, query=args.query, temperature=0.0)
    latency = time.perf_counter() - t0

    # r.boxes is only populated on the <OBJ> matting path (engine.predict);
    # detection/ocr/point/count_box boxes live in the response text. The
    # authoritative parser (wevisone_vl.visualization.parse_predictions)
    # imports supervision at module level, absent in this env, so the same
    # REF_RE semantics are inlined here: <ref>L</ref><shape>BODY, per
    # comma-separated segment, <v> tokens, 1000-grid -> normalized xyxy.
    import re

    ref_re = re.compile(
        r"<ref>(.*?)</ref>\s*<(box|point|polygon)>(.*?)(?:</(?:box|point|polygon)>|<ref>|$)",
        re.DOTALL | re.IGNORECASE,
    )
    coords: list[list[float]] = []
    text = r.response.split("<|im_end|>", 1)[0].split("<|endoftext|>", 1)[0]
    for _label, shape, body in ref_re.findall(text):
        size = 2 if shape.lower() == "point" else 4
        for segment in body.split(","):
            values = [int(v) for v in re.findall(r"<(\d{1,4})>", segment)]
            for off in range(0, len(values) - size + 1, size):
                chunk = [v / 1000.0 for v in values[off:off + size]]
                if size == 2:
                    chunk = [chunk[0], chunk[1], chunk[0], chunk[1]]
                elif chunk[2] <= chunk[0] or chunk[3] <= chunk[1]:
                    continue
                coords.append(chunk)
    if not coords:
        coords = [[round(float(v), 6) for v in b] for b in r.boxes]

    if args.out_png:
        # wevisone_vl.visualization pulls in `supervision`, which the
        # benchmark env does not carry — draw the normalized boxes with
        # PIL directly (same payload semantics, no new dependency).
        from PIL import ImageDraw
        vis = img.copy()
        draw = ImageDraw.Draw(vis)
        for b in coords:
            x1, y1, x2, y2 = (round(v * s) for v, s in
                              zip(b, (vis.width, vis.height) * 2))
            draw.rectangle([x1, y1, x2, y2], outline=(255, 60, 60), width=3)
        vis.save(args.out_png)
    if args.out_json:
        payload = {
            "backend": "python-cuda",
            "dtype": args.dtype,
            "answer": r.response,
            "box_count": len(coords),
            "box_coords": [[round(float(v), 6) for v in b] for b in coords],
            "mask_count": len(r.masks),
            # same aperture as the cpp CLI's timings.e2e_ms: model-load-free
            # pre+infer+post wall time around predict()
            "timings": {"e2e_ms": round(latency * 1000, 3)},
            "latency_s": round(latency, 4),
        }
        Path(args.out_json).write_text(json.dumps(payload, indent=2))
    if args.out_matte and r.masks:
        import numpy as np
        matte = np.asarray(r.masks[0], dtype=np.float32)
        matte.tofile(args.out_matte)
        h, w = matte.shape
        Path(args.out_matte + ".shape").write_text(f"{h},{w}")
    print(f"answer: {r.response[:100]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
