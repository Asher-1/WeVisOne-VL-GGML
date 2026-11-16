#!/usr/bin/env python3
"""Accuracy summary: GGUF port vs the official pytorch stack (bf16).

Compares the fixed-mode results of run_benchmark.sh per size:
  - box agreement: greedy IoU matching against the python-cuda reference
    boxes + mean matched IoU (verdict gate: IoU >= --iou-band)
  - point agreement: mean normalized distance <= --point-band
  - mask presence (matting/refseg families)
The port backend is `cpp-cuda` f16 (the parity tier); `cpp-vulkan` is
compared as well; every cpp precision tier (f16/q8_0/q4_K_M, the
FIXED_PRECS sweep) is judged against the same python-cuda bf16 reference.

Answer-text fields are RECORDED, not gated:
  - answer_eq: exact token identity. Coordinate-token ±1-2 flips make it
    noisy even on the f16 parity tier; a diagnostic, not the verdict.
  - answer_sim: difflib SequenceMatcher ratio of the raw answers. The
    verdict gates stay structural (boxes/points/masks) precisely so they
    absorb representation noise; long freeform generations are the one
    family where greedy decode AMPLIFIES the bf16-vs-f16 representation
    noise over token count (README round-14) — answer_sim is the recorded
    divergence measure there, gated only if freeform cases ever enter the
    benchmark (no freeform case today; no speculative gate).
"""
from __future__ import annotations

import argparse
import difflib
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
    ap.add_argument("--iou-band", type=float, default=0.9,
                    help="matched-IoU hard gate for box tasks")
    ap.add_argument("--point-band", type=float, default=0.02,
                    help="max mean point distance (normalized) for point tasks")
    args = ap.parse_args()
    results = Path(args.results)
    sizes = sorted({p.name for p in results.glob("fixed/*")
                    if p.is_dir()})
    summary = {"sizes": {}, "ref_backend": args.ref_backend,
               "bands": {"iou": args.iou_band, "point": args.point_band}}
    ok_all = True
    for size in sizes:
        rows = []
        ref_dir = results / "fixed" / size / args.ref_backend
        cases = sorted(p.stem for p in ref_dir.glob("*.json")
                       if not p.stem.endswith((".q8_0", ".q4_K_M")))
        # The python-cuda bf16 reference is the ONE official tier; every
        # cpp precision tier (f16 parity tier + the quantized accuracy
        # sweep written as <case>.<tier>.json by run_benchmark.sh
        # FIXED_PRECS) is judged against it.
        tiers = ("f16", "q8_0", "q4_K_M")
        for backend in ("cpp-cuda", "cpp-vulkan"):
            bdir = results / "fixed" / size / backend
            for case in cases:
                for tier in tiers:
                    got_path = bdir / (f"{case}.json" if tier == "f16"
                                       else f"{case}.{tier}.json")
                    if not got_path.exists():
                        if tier != "f16":
                            continue  # tier not swept for this case
                        rows.append({"case": case, "backend": backend,
                                     "precision": tier, "verdict": "MISSING"})
                        ok_all = False
                        continue
                    got = json.loads(got_path.read_text())
                    ref = json.loads((ref_dir / f"{case}.json").read_text())
                    ref_answer = ref.get("answer") or ""
                    got_answer = got.get("answer") or ""
                    answer_eq = ref_answer == got_answer
                    mask_eq = ref.get("mask_count") == got.get("mask_count")
                    row = {"case": case, "backend": backend,
                           "precision": tier,
                           "answer_eq": answer_eq,
                           # autojunk=False: the answers are token-streams
                           # (<14><274>...); high-frequency markup chars
                           # must not be dropped as junk on long answers
                           "answer_sim": round(difflib.SequenceMatcher(
                               None, ref_answer, got_answer,
                               autojunk=False).ratio(), 4),
                           "mask_eq": mask_eq}
                    if case.endswith("_point"):
                        # Both stacks now emit points as degenerate boxes
                        # [x, y, x, y] in box_coords (the cpp result_parser
                        # <point> branch mirrors python_infer.py size-2
                        # chunks); the answer-regex parser stays as the
                        # fallback for result trees written before that fix.
                        pr = [b[:2] for b in ref.get("box_coords", [])]
                        pg = [b[:2] for b in got.get("box_coords", [])]
                        if not pr or not pg:
                            import re
                            pt = re.compile(
                                r"<point>(.*?)(?:</point>|<ref>|$)", re.DOTALL)
                            num = re.compile(r"<(\d{1,4})>")

                            def points_of(s: str) -> list[list[float]]:
                                out_pts = []
                                for seg in pt.findall(s or ""):
                                    vs = [int(v) for v in num.findall(seg)]
                                    for o in range(0, len(vs) - 1, 2):
                                        out_pts.append([vs[o] / 1000.0,
                                                        vs[o + 1] / 1000.0])
                                return out_pts
                            pr = points_of(ref.get("answer"))
                            pg = points_of(got.get("answer"))
                        dist = 0.0
                        if len(pr) == len(pg) and pr:
                            dist = sum(
                                ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
                                for a, b in zip(pr, pg)) / len(pr)
                            verdict = ("PASS" if dist <= args.point_band
                                       else "FAIL")
                        else:
                            verdict = "FAIL"
                        if verdict == "FAIL":
                            ok_all = False
                        row.update({"points_ref": len(pr), "points_got": len(pg),
                                    "mean_dist": round(dist, 4),
                                    "verdict": verdict})
                    else:
                        miou, n_ref, n_got = box_match(
                            ref.get("box_coords", []),
                            got.get("box_coords", []))
                        verdict = ("PASS" if n_ref == n_got
                                   and (n_ref == 0 or miou >= args.iou_band)
                                   else "FAIL")
                        if verdict == "FAIL":
                            ok_all = False
                        row.update({"matched_iou": round(miou, 4),
                                    "boxes_ref": n_ref, "boxes_got": n_got,
                                    "verdict": verdict})
                    rows.append(row)
        summary["sizes"][size] = rows
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    # console table
    for size in sizes:
        print(f"== {size}")
        for r in summary["sizes"][size]:
            extra = ""
            if "matched_iou" in r:
                extra = (f"iou={r['matched_iou']:.3f} "
                         f"boxes {r['boxes_ref']}->{r['boxes_got']}")
            # FAIL rows surface answer_sim: a high sim on a FAIL marks a
            # representation-noise divergence (boxes/points flipped late),
            # a low sim marks a full generation divergence — the two
            # failure classes the round-14 note distinguishes.
            if r["verdict"] == "FAIL" and "answer_sim" in r:
                extra += f" sim={r['answer_sim']:.3f}"
            print(f"  [{r['verdict']}] {r['backend']:<12} {r['case']:<28} {extra}")
    print("ACCURACY:", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
