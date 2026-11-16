#!/usr/bin/env python3
"""Regression matrix for the verification GUI: every task family x backend.

Sends one /api/run request per manifest sample with backend=both (the GUI
runs py first, then cpp, under its run lock). Asserts per sample:
  - no "error" field on either backend
  - answer non-empty
  - expected modality present: shapes>0 for box/point tasks, mask_count>0
    for matte tasks — unless the model legitimately answered "none" inside
    the box/point body (the released no-target convention)

Usage: python3 gui/matrix_test.py [base_url]   (default http://localhost:8017)
The GUI must be running (bash run_gui.sh). Exits non-zero on any FAIL.
"""
import json
import re
import sys
import time
import urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8017").rstrip("/")
EXPECT_SHAPES = {"detection", "referring", "count_box", "count_point",
                 "point", "ocr_grounding", "visual_prompt"}
EXPECT_MASKS = {"segmentation", "matting"}
# the model may legitimately answer "no target found"; the released
# convention is a literal "none" inside the box/point body
NONE_RE = re.compile(r"<(box|point)>\s*none\s*<", re.IGNORECASE)

manifest = json.loads(urllib.request.urlopen(BASE + "/api/samples").read())
rows = []
failures = 0
for s in manifest["samples"]:
    body = json.dumps({
        "backend": "both",
        "image": s["image"],
        "task": s["task"],
        "query": s.get("query", ""),
        "ref_boxes": s.get("ref_boxes"),
    }).encode()
    req = urllib.request.Request(BASE + "/api/run", body,
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=900).read())
    except Exception as exc:
        rows.append((s["id"], s["task"], "REQUEST-FAIL", f"{exc!r}", 0.0))
        failures += 1
        continue
    wall = time.time() - t0
    problems = []
    for be in ("py", "cpp"):
        r = resp.get(be)
        if r is None:
            problems.append(f"{be}:missing")
            continue
        if r.get("error"):
            problems.append(f"{be}:error={r['error'][:90]}")
            continue
        ans = (r.get("answer") or "").strip()
        if not ans:
            problems.append(f"{be}:empty-answer")
        n_shapes = r.get("shapes", 0)
        n_masks = r.get("mask_count", 0)
        none_answer = bool(NONE_RE.search(ans))
        if s["task"] in EXPECT_SHAPES and n_shapes < 1 and not none_answer:
            problems.append(f"{be}:no-shapes({n_shapes})")
        if s["task"] in EXPECT_MASKS and n_masks < 1:
            problems.append(f"{be}:no-mattes({n_masks})")
    if problems:
        failures += 1
        rows.append((s["id"], s["task"], "FAIL", "; ".join(problems), wall))
    else:
        rows.append((s["id"], s["task"], "PASS", "", wall))

print(f"{'sample':32} {'task':15} {'res':7} {'detail':60} wall_s")
for rid, task, res, detail, wall in rows:
    print(f"{rid:32} {task:15} {res:7} {detail:60} {wall:5.1f}")
print(f"\n{len(rows) - failures}/{len(rows)} PASS, {failures} FAIL")
sys.exit(1 if failures else 0)
