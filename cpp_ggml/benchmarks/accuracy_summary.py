#!/usr/bin/env python3
"""Accuracy summary: GGUF port vs the official pytorch stack (bf16).

Compares the fixed-mode results of run_benchmark.sh per size:
  - answer identity (the token-identity hard constraint of the d10 contract)
  - box agreement: greedy IoU matching against the python-cuda reference
    boxes + mean matched IoU
  - mask presence (matting/refseg families)
The port backend is `cpp-cuda` f16 (the parity tier); `cpp-vulkan` is
compared as well. Verdicts follow the sibling d10 bands: token identity is
the hard constraint; box IoU >= 0.9 for the f16 tier on detection/referring.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def box_match(ref, got):
    """Greedy IoU matching; returns (matched_mean_iou, count_ref, count_got)."""
    pairs = []
    used = set()
    for i, r in enumerate(ref):
        best, best_j = 0.0, -1
        for j, g in enumerate(got):
            if j in used:
                continue
            v = iou(r, g)
            if v > best:
                best, best_j = v, j
        if best_j >= 0 and best > 0:
            pairs.append(best)
            used.add(best_j)
    return (sum(pairs) / len(pairs) if pairs else 0.0,
            len(ref), len(got))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True,
                    help="output json path (accuracy summary)")
    ap.add_argument("--ref-backend", default="python-cuda")
    args = ap.parse_args()
    results = Path(args.results)
    sizes = sorted({p.name for p in results.glob("fixed/*")
                    if p.is_dir()})
    summary = {"sizes": {}, "ref_backend": args.ref_backend}
    ok_all = True
    for size in sizes:
        rows = []
        ref_dir = results / "fixed" / size / args.ref_backend
        cases = sorted(p.stem for p in ref_dir.glob("*.json"))
        for backend in ("cpp-cuda", "cpp-vulkan"):
            bdir = results / "fixed" / size / backend
            for case in cases:
                ref = json.loads((ref_dir / f"{case}.json").read_text())
                got_path = bdir / f"{case}.json"
                if not got_path.exists():
                    rows.append({"case": case, "backend": backend,
                                 "verdict": "MISSING"})
                    ok_all = False
                    continue
                got = json.loads(got_path.read_text())
                answer_eq = ref.get("answer") == got.get("answer")
                miou, n_ref, n_got = box_match(
                    ref.get("box_coords", []), got.get("box_coords", []))
                mask_eq = ref.get("mask_count") == got.get("mask_count")
                verdict = "PASS" if answer_eq and mask_eq and (
                    n_ref == 0 or miou >= 0.9) else "FAIL"
                if verdict == "FAIL":
                    ok_all = False
                rows.append({"case": case, "backend": backend,
                             "answer_eq": answer_eq,
                             "mask_eq": mask_eq,
                             "matched_iou": round(miou, 4),
                             "boxes_ref": n_ref, "boxes_got": n_got,
                             "verdict": verdict})
        summary["sizes"][size] = rows
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    # console table
    for size in sizes:
        print(f"== {size}")
        for r in summary["sizes"][size]:
            extra = (f"iou={r['matched_iou']:.3f} "
                     f"boxes {r['boxes_ref']}->{r['boxes_got']}"
                     if "matched_iou" in r else "")
            print(f"  [{r['verdict']}] {r['backend']:<12} {r['case']:<28} {extra}")
    print("ACCURACY:", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
