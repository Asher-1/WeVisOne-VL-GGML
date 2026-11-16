#!/usr/bin/env python3
"""One-click WeVisOne-VL verification GUI (zero third-party deps).

Runs the official pytorch engine in-process and the ggml port as a
persistent `wevisone-cli serve` child, side by side, over every released
task family. Only the python standard library plus what the official
engine already imports (torch/transformers/PIL) is used — gradio is not
available on the offline benchmark host and tkinter is not installed,
so the UI is a plain stdlib http.server page.

Layout (root):
  gui/wevisone_gui.py    this file
  run_gui.sh             one-click launcher (stages the offline HF dir
                         when needed, then serves on :8017)
  test_data/             sample manifest the GUI loads (see test_data/README)

Endpoints:
  GET  /              UI
  GET  /api/status    backend states (py engine, cpp serve child)
  GET  /api/samples   test_data manifest
  GET  /image?src=    whitelisted image files (test_data/ + uploads/)
  POST /api/upload    base64 image -> uploads/, returns the serving name
  POST /api/run       {backend: py|cpp|both, image, task, query, ref_boxes?
                       , image_b64?} -> per-backend result with a rendered
                       overlay data-URL
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import threading
import time
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    # the official engine imports from the repo root; this file lives in
    # gui/ so python's default sys.path[0] is gui/, not ROOT
    sys.path.insert(0, str(ROOT))
TEST_DATA = ROOT / "test_data"
UPLOADS = TEST_DATA / "uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)
SERVE_WAIT_S = 600  # first model load can take minutes on a cold page cache

# Released task families (wevisone_vl/prompts.py TASKS + the C-API aliases
# the official engine exposes as freeform). visual_prompt rides the same
# family with reference boxes.
TASKS = ["detection", "referring", "point", "count_box", "count_point",
         "ocr", "ocr_grounding", "segmentation", "matting", "visual_prompt",
         "freeform"]

# ---------------------------------------------------------------- py engine


class PyBackend:
    """Official torch engine, loaded once in this process."""

    def __init__(self, model_path: str, device: str):
        self.model_path = model_path
        self.device = device
        self.engine = None
        self.error: str | None = None
        self.lock = threading.Lock()

    def ensure_loaded(self) -> bool:
        if self.engine is not None or self.error is not None:
            return self.engine is not None
        try:
            from wevisone_vl import WeVisOneVL

            with self.lock:
                # concurrent requests can race past the pre-lock check;
                # re-check under the lock so the multi-GB load runs once
                if self.engine is None:
                    self.engine = WeVisOneVL(self.model_path,
                                             device=self.device,
                                             dtype="bfloat16")
        except Exception as exc:  # surface the load failure in the UI
            self.error = f"{exc.__class__.__name__}: {exc}"
        return self.engine is not None

    @property
    def ready(self) -> bool:
        return self.engine is not None

    def run(self, image_path: str, task: str, query: str,
            ref_boxes: list | None) -> dict:
        if not self.ensure_loaded():
            return {"error": self.error or "engine failed to load"}
        from PIL import Image

        with Image.open(image_path) as source:
            pil = source.convert("RGB")
        started = time.perf_counter()
        try:
            with self.lock:  # the engine is single-threaded
                result = self.engine.predict(
                    pil, task=task, query=query,
                    reference_boxes=ref_boxes, temperature=0.0)
        except Exception as exc:
            return {"error": f"{exc.__class__.__name__}: {exc}"}
        return {
            "answer": result.response,
            "boxes01": result.boxes,
            "mask_count": len(result.masks),
            "e2e_ms": (time.perf_counter() - started) * 1000.0,
            # raw masks ride along for the overlay renderer
            "_pil": pil,
            "_masks": result.masks,
        }


# --------------------------------------------------------------- cpp serve


class CppBackend:
    """Persistent `wevisone-cli serve` child with the 5-line protocol."""

    def __init__(self, cli: str, models_args: list[str], device: str,
                 precision: str, decoder_precision: str):
        self.cli = cli
        self.models_args = models_args
        self.device = device
        self.precision = precision
        self.decoder_precision = decoder_precision
        self.proc: subprocess.Popen | None = None
        self.stderr_path: str | None = None
        self.stderr_file = None
        self.error: str | None = None
        self.lock = threading.Lock()
        self.device_resolved = None

    def ready(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self) -> bool:
        if self.ready():
            return True
        # a previous failure does not disqualify a retry: the common
        # failure (GPU memory held by another process) is transient, so
        # each run attempt re-spawns and refreshes self.error
        cmd = [self.cli, "serve", "--device", self.device,
               "--precision", self.precision,
               "--decoder-precision", self.decoder_precision,
               *self.models_args]
        import os

        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = str(Path(self.cli).parent) + ":" + \
            env.get("LD_LIBRARY_PATH", "")
        # keep the child's stderr on disk: a failed spawn dies before the
        # handshake and the reason is otherwise invisible
        self.stderr_path = f"/tmp/wwv_serve_{os.getpid()}.log"
        try:
            self.stderr_file = open(self.stderr_path, "wb")
            self.proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self.stderr_file, cwd=str(ROOT), env=env,
                text=True, bufsize=1)
        except OSError as exc:
            self.error = f"cannot spawn {self.cli}: {exc}"
            return False

        def read_ready() -> bool:
            assert self.proc is not None and self.proc.stdout is not None
            line = self.proc.stdout.readline()
            try:
                info = json.loads(line)
            except json.JSONDecodeError:
                tail = ""
                try:
                    with open(self.stderr_path, "rb") as fh:
                        tail = fh.read()[-800:].decode("utf-8", "replace")
                except OSError:
                    pass
                self.error = (f"serve handshake failed: {line[:120]!r} "
                              f"— child stderr tail:\n{tail}")
                return False
            if info.get("event") != "ready":
                self.error = f"serve did not become ready: {info}"
                return False
            self.device_resolved = info.get("device")
            return True

        watcher = threading.Thread(target=read_ready, daemon=True)
        watcher.start()
        watcher.join(SERVE_WAIT_S)
        ok = self.ready() and self.device_resolved is not None
        if not ok:
            if self.error is None:
                self.error = "serve timeout"
            # kill a half-started child: a serve process that booted past
            # a rejected handshake keeps holding GPU memory otherwise
            self.stop()
        return ok

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
        self.proc = None
        self.device_resolved = None
        if self.stderr_file is not None:
            self.stderr_file.close()
            self.stderr_file = None

    def run(self, image_path: str, task: str, query: str,
            ref_boxes: list | None) -> dict:
        if not self.start():
            return {"error": self.error or "cpp serve not running"}
        import numpy as np  # PIL is required by the py backend anyway

        assert self.proc is not None and self.proc.stdin is not None \
            and self.proc.stdout is not None
        b64 = base64.b64encode(query.encode("utf-8")).decode("ascii")
        try:
            refb = refb_encode(ref_boxes)
        except (TypeError, ValueError) as exc:
            # argument encoding failure, NOT a dead child: report it as
            # such instead of falling into the serve-died branch
            return {"error": f"bad ref_boxes: {exc}"}
        with self.lock:
            try:
                self.proc.stdin.write(
                    f"{image_path}\n{task}\n{b64 or 'NONE'}\n{refb}\nRUN\n")
                self.proc.stdin.flush()
                line = self.proc.stdout.readline()
            except (BrokenPipeError, ValueError) as exc:
                self.stop()
                return {"error": f"serve died: {exc}"}
        if not line:
            self.stop()
            return {"error": "serve closed the pipe"}
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            return {"error": f"serve returned garbage: {line[:200]}"}
        if payload.get("error"):
            return {"error": payload["error"]}
        result = {
            "answer": payload.get("answer", ""),
            "boxes01": payload.get("box_coords", []),
            "mask_count": payload.get("mask_count", 0),
            "e2e_ms": (payload.get("timings") or {}).get("e2e_ms", 0.0),
        }
        mattes = payload.get("mattes") or []
        if mattes:
            m = mattes[0]
            flat = np.frombuffer(base64.b64decode(m["b64"]),
                                 dtype=np.float32)
            result["_matte"] = (flat, int(m["w"]), int(m["h"]))
        result["_pil"] = None
        return result


def refb_encode(ref_boxes: list | None) -> str:
    if not ref_boxes:
        return "NONE"
    parts = []
    for box in ref_boxes:
        parts.append(",".join(str(float(v)) for v in box[:4]))
    return ";".join(parts)


# ----------------------------------------------------------------- render

# The official demo UI parses the raw answer for drawing (engine
# InferenceResult.boxes only fills on the <OBJ> mask path), so both
# backends go through the same regex family here: <ref>label</ref><box>
# <x1><y1><x2><y2> [,...] with 0-1000 coordinate tokens.
_REF_RE = __import__("re").compile(
    r"<ref>(.*?)</ref>\s*<(box|point)>(.*?)(?:</(?:box|point)>|<ref>|$)",
    __import__("re").DOTALL | __import__("re").IGNORECASE)
_COORD_RE = __import__("re").compile(r"<(\d{1,4})>")


def parse_answer_shapes(text: str, width: int, height: int) -> list:
    """answer -> [("box", (x1, y1, x2, y2)) | ("point", (x, y))] pixels."""
    import re

    shapes: list = []
    clean = (text or "").split("<|im_end|>", 1)[0]
    for _label, kind, body in _REF_RE.findall(clean):
        step = 2 if kind.lower() == "point" else 4
        values = [int(v) for v in _COORD_RE.findall(body or "")]
        for off in range(0, len(values) - step + 1, step):
            chunk = values[off:off + step]
            if step == 2:
                shapes.append(("point", (
                    chunk[0] / 1000.0 * width,
                    chunk[1] / 1000.0 * height)))
            else:
                x1 = chunk[0] / 1000.0 * width
                y1 = chunk[1] / 1000.0 * height
                x2 = chunk[2] / 1000.0 * width
                y2 = chunk[3] / 1000.0 * height
                if x2 > x1 and y2 > y1:
                    shapes.append(("box", (x1, y1, x2, y2)))
    return shapes


def render_overlay(pil, shapes: list, matte=None) -> str:
    """Official-style visualization -> JPEG data URL (PIL, no numpy needed
    on the box path; the continuous alpha uses the shipped numpy)."""
    import io as _io

    import numpy as np
    from PIL import Image, ImageDraw

    base = (pil or Image.new("RGB", (640, 400), "#20242c")).convert("RGB")
    if matte is not None:
        if isinstance(matte, tuple):  # cpp serve: (flat f32, w, h)
            flat, w, h = matte
            alpha = np.clip(flat.reshape(h, w), 0.0, 1.0)
        else:  # py engine: ndarray (h, w), float or uint8
            alpha = np.asarray(matte, dtype=np.float32)
            if alpha.max() > 1.0:
                alpha = alpha / 255.0
            alpha = np.clip(alpha, 0.0, 1.0)
        canvas = np.asarray(base.resize(
            (alpha.shape[1], alpha.shape[0]), Image.BILINEAR),
            dtype=np.float32)
        green = np.array([0.0, 214.0, 90.0], dtype=np.float32)
        w3 = (alpha * 0.55)[..., None]
        comp = canvas * (1.0 - w3) + green * w3
        base = Image.fromarray(np.clip(comp, 0, 255).astype("uint8")) \
            .resize(base.size, Image.BILINEAR)
    draw = ImageDraw.Draw(base)
    scale = max(0.75, min(max(base.size) / 800.0, 5.0))
    for kind, coords in shapes or []:
        if kind == "box":
            draw.rectangle(coords, outline=(255, 60, 90),
                           width=max(2, int(round(3 * scale))))
        else:  # point: official filled circle with a white ring
            x, y = coords
            r = max(6, int(round(9 * scale)))
            draw.ellipse([x - r - 3, y - r - 3, x + r + 3, y + r + 3],
                         fill=(255, 255, 255))
            draw.ellipse([x - r, y - r, x + r, y + r],
                         fill=(255, 60, 90))
    buf = _io.BytesIO()
    base.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + \
        base64.b64encode(buf.getvalue()).decode("ascii")


# ------------------------------------------------------------------ server


class State:
    py: PyBackend | None = None
    cpp: CppBackend | None = None
    start_s = time.time()
    run_lock = threading.Lock()  # one GPU card: runs are mutually exclusive


STATE = State()


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def resolve_image(src: str) -> Path | None:
    """Whitelist: test_data/**, repo-relative manifest paths (the official
    demo photos live in demo/examples/images and are referenced in place,
    no copies) and uploads/. Containment (not a string prefix, which
    "/other_repo" style names can sneak past) plus an image-extension
    gate so non-image repo files are never served."""
    if not src:
        return None
    for base in (TEST_DATA, ROOT):
        path = (base / src).resolve()
        if path.is_file() and path.is_relative_to(base) \
                and path.suffix.lower() in IMAGE_EXTS:
            return path
    path = (UPLOADS / src).resolve()
    if path.is_file() and path.is_relative_to(UPLOADS):
        return path
    return None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet; errors still print
        if " 200 " not in fmt % args:
            print("[gui]", fmt % args)

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode(),
                   "application/json")

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif parsed.path == "/api/status":
            self._json(200, {
                "py": {
                    "ready": bool(STATE.py and STATE.py.ready),
                    "error": STATE.py.error if STATE.py else None,
                    "model": STATE.py.model_path if STATE.py else None,
                },
                "cpp": {
                    "ready": bool(STATE.cpp and STATE.cpp.ready()),
                    "error": STATE.cpp.error if STATE.cpp else None,
                    "device": STATE.cpp.device_resolved if STATE.cpp else None,
                    "precision": STATE.cpp.precision if STATE.cpp else None,
                },
                "tasks": TASKS,
                "uptime_s": time.time() - STATE.start_s,
            })
        elif parsed.path == "/api/samples":
            manifest = json.loads((TEST_DATA / "manifest.json").read_text())
            self._json(200, manifest)
        elif parsed.path == "/image":
            query = urllib.parse.parse_qs(parsed.query)
            path = resolve_image((query.get("src") or [""])[0])
            if path is None:
                self._json(404, {"error": "image not in whitelist"})
                return
            self._send(200, path.read_bytes(), "image/jpeg")
        else:
            self._json(404, {"error": "unknown endpoint"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "bad json body"})
            return
        if self.path == "/api/upload":
            try:
                raw = base64.b64decode(req.get("image_b64", ""))
                import io as _io

                from PIL import Image

                image = Image.open(_io.BytesIO(raw)).convert("RGB")
            except Exception as exc:
                self._json(400, {"error": f"bad image: {exc}"})
                return
            name = f"upload_{uuid.uuid4().hex[:8]}.jpg"
            image.save(UPLOADS / name, format="JPEG", quality=95)
            self._json(200, {"name": name, "width": image.width,
                             "height": image.height})
            return
        if self.path != "/api/run":
            self._json(404, {"error": "unknown endpoint"})
            return

        # persist an uploaded payload when the request carries one
        if req.get("image_b64"):
            try:
                raw = base64.b64decode(req["image_b64"])
                import io as _io

                from PIL import Image

                image = Image.open(_io.BytesIO(raw)).convert("RGB")
                name = f"upload_{uuid.uuid4().hex[:8]}.jpg"
                image.save(UPLOADS / name, format="JPEG", quality=95)
                req["image"] = name
            except Exception as exc:
                self._json(400, {"error": f"bad image: {exc}"})
                return

        image_path = resolve_image(str(req.get("image", "")))
        if image_path is None:
            self._json(400, {"error": "image not found / not whitelisted"})
            return
        task = str(req.get("task", "freeform"))
        query = str(req.get("query", ""))
        ref_boxes = req.get("ref_boxes") or None
        backend = str(req.get("backend", "both"))
        results: dict = {"backend": backend, "task": task,
                         "image": str(image_path)}

        def run_py() -> None:
            assert STATE.py is not None
            r = STATE.py.run(str(image_path), task, query, ref_boxes)
            pil, masks = r.pop("_pil", None), r.pop("_masks", [])
            matte = masks[0] if masks else None
            if not r.get("error") and pil is not None:
                shapes = parse_answer_shapes(r.get("answer", ""),
                                             pil.width, pil.height)
                r["shapes"] = len(shapes)
                r["overlay"] = render_overlay(pil, shapes, matte)
            else:
                r["overlay"] = None
            results["py"] = r

        def run_cpp() -> None:
            assert STATE.cpp is not None
            r = STATE.cpp.run(str(image_path), task, query, ref_boxes)
            matte = r.pop("_matte", None)
            r.pop("_pil", None)
            if not r.get("error"):
                from PIL import Image

                with Image.open(str(image_path)) as source:
                    pil = source.convert("RGB")
                shapes = parse_answer_shapes(r.get("answer", ""),
                                             pil.width, pil.height)
                r["shapes"] = len(shapes)
                r["overlay"] = render_overlay(pil, shapes, matte)
            else:
                r["overlay"] = None
            results["cpp"] = r

        # sequential, and mutually exclusive across requests: the two GPU
        # stacks share one card, so py-then-cpp must never interleave with
        # another in-flight run
        with STATE.run_lock:
            if backend in ("py", "both") and STATE.py is not None:
                run_py()
            if backend in ("cpp", "both") and STATE.cpp is not None:
                run_cpp()
        self._json(200, results)


def main() -> int:
    ap = argparse.ArgumentParser(description="WeVisOne-VL verification GUI")
    ap.add_argument("--port", type=int, default=8017)
    ap.add_argument("--py-model", default=None,
                    help="official torch checkpoint dir (default: the "
                         "offline benchmark staging, else checkpoints/hf-2B)")
    ap.add_argument("--py-device", default="cuda")
    ap.add_argument("--cpp-cli", default=str(ROOT / "cpp_ggml/build-cuda/"
                                             "wevisone-cli"))
    ap.add_argument("--cpp-device", default="cuda")
    ap.add_argument("--cpp-precision", default="f16")
    ap.add_argument("--cpp-decoder-precision", default="f16")
    ap.add_argument("--cpp-size", default="2B", choices=("2B", "4B"))
    ap.add_argument("--cpp-models-dir", default=str(ROOT / "cpp_ggml/"
                                                    "models/gguf"))
    ap.add_argument("--backend", default="both",
                    choices=("py", "cpp", "both"))
    args = ap.parse_args()

    py_model = args.py_model
    if py_model is None:
        staging = Path("/tmp/wwv-bench-hf-2B")
        py_model = str(staging if staging.is_dir()
                       else ROOT / "checkpoints/hf-2B")

    lower = args.cpp_size.lower()
    models_args = [
        "--vlm-gguf",
        f"{args.cpp_models_dir}/wevisone-vlm-{lower}-{args.cpp_precision}.gguf",
        "--mmproj-gguf",
        f"{args.cpp_models_dir}/mmproj-wevisone-vlm-{lower}-{args.cpp_precision}.gguf",
    ]
    decoder = Path(args.cpp_models_dir) / \
        f"wevisone-mask-decoder-{lower}-{args.cpp_decoder_precision}.gguf"
    if decoder.is_file():
        models_args += ["--decoder-gguf", str(decoder)]

    if args.backend in ("py", "both"):
        STATE.py = PyBackend(py_model, args.py_device)
    if args.backend in ("cpp", "both"):
        STATE.cpp = CppBackend(args.cpp_cli, models_args, args.cpp_device,
                               args.cpp_precision, args.cpp_decoder_precision)

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"GUI  -> http://localhost:{args.port}")
    print(f"py   -> {py_model} (loads on first run)")
    print(f"cpp  -> {args.cpp_cli} {args.cpp_device}/{args.cpp_precision}")
    server.serve_forever()
    return 0


# --------------------------------------------------------------------- UI
# Single-file dark UI; no build step, no CDN (offline host).

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>WeVisOne-VL — dual-backend verification</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; font:14px/1.5 -apple-system,"Segoe UI",Roboto,sans-serif;
         background:#101318; color:#e8eaed; }
  header { padding:18px 28px 10px; border-bottom:1px solid #262c36; }
  header h1 { margin:0; font-size:20px; font-weight:700; }
  header .sub { color:#8b93a1; font-size:12px; margin-top:4px; }
  .dot { display:inline-block; width:9px; height:9px; border-radius:50%;
         margin-right:5px; background:#5f6673; }
  .dot.ok { background:#3ddc84; } .dot.err { background:#ff5c7c; }
  main { display:grid; grid-template-columns: 380px 1fr; gap:18px;
         padding:18px 28px; }
  .card { background:#171b22; border:1px solid #262c36; border-radius:14px;
          padding:14px; }
  label { display:block; font-size:11px; text-transform:uppercase;
          letter-spacing:.08em; color:#8b93a1; margin:10px 0 4px; }
  select, input, textarea { width:100%; box-sizing:border-box; background:#0d1015;
          border:1px solid #2c3442; color:#e8eaed; border-radius:8px;
          padding:7px 9px; font:inherit; }
  textarea { resize:vertical; min-height:44px; }
  .chips { display:flex; flex-wrap:wrap; gap:6px; }
  .chip { padding:5px 12px; border-radius:999px; border:1px solid #2c3442;
          background:transparent; color:#c8cdd6; font:inherit; cursor:pointer; }
  .chip.on { background:#4f6ef7; border-color:#4f6ef7; color:#fff; }
  button.run { margin-top:12px; width:100%; padding:10px; font-size:15px;
          font-weight:700; border:0; border-radius:10px; cursor:pointer;
          background:linear-gradient(90deg,#4f6ef7,#9a5cf5,#00b8d4); color:#fff; }
  #preview { width:100%; border-radius:10px; margin-top:10px; display:block; }
  .results { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
  .res h3 { margin:0 0 8px; font-size:14px; }
  .res img { width:100%; border-radius:10px; background:#0d1015; min-height:200px; }
  .res pre { white-space:pre-wrap; word-break:break-word; font:12px/1.45
          ui-monospace,monospace; background:#0d1015; padding:9px;
          border-radius:8px; max-height:180px; overflow:auto; margin:8px 0 0; }
  .stat { font-size:12px; color:#8b93a1; margin-top:6px; }
  .err { color:#ff5c7c; font-size:12px; }
  .hint { color:#5f6673; font-size:11px; margin-top:4px; }
</style>
</head>
<body>
<header>
  <h1>WeVisOne-VL — dual-backend verification</h1>
  <div class="sub">
    pytorch <span class="dot" id="dot_py"></span><span id="st_py">loading…</span>
    &nbsp;·&nbsp; ggml <span class="dot" id="dot_cpp"></span><span id="st_cpp">loading…</span>
  </div>
</header>
<main>
  <div class="card">
    <label>Sample (test_data manifest)</label>
    <select id="sample"></select>
    <img id="preview" alt="">
    <label>Task</label>
    <div class="chips" id="tasks"></div>
    <label>Query</label>
    <textarea id="query"></textarea>
    <label>Ref boxes (visual_prompt, pixels "x1,y1,x2,y2")</label>
    <input id="refboxes" placeholder="880,120,1500,1070">
    <label>Upload image (overrides sample)</label>
    <input type="file" id="file" accept="image/*">
    <label>Backend</label>
    <div class="chips" id="backends">
      <button class="chip on" data-v="both">both</button>
      <button class="chip" data-v="py">pytorch</button>
      <button class="chip" data-v="cpp">ggml</button>
    </div>
    <button class="run" id="run">Run</button>
    <div class="hint">first pytorch run also loads the model (~1 min);
    first ggml run spawns the persistent serve child</div>
  </div>
  <div class="card">
    <div class="results">
      <div class="res" id="res_py">
        <h3>PyTorch (official)</h3>
        <img id="img_py" alt=""><div class="stat" id="meta_py"></div>
        <pre id="ans_py" hidden></pre><div class="err" id="err_py"></div>
      </div>
      <div class="res" id="res_cpp">
        <h3>GGML (cpp_ggml)</h3>
        <img id="img_cpp" alt=""><div class="stat" id="meta_cpp"></div>
        <pre id="ans_cpp" hidden></pre><div class="err" id="err_cpp"></div>
      </div>
    </div>
  </div>
</main>
<script>
const $ = id => document.getElementById(id);
let TASK = "detection", BACKEND = "both", SAMPLES = [];

function chipRow(el, values, current, cb) {
  el.innerHTML = "";
  values.forEach(v => {
    const b = document.createElement("button");
    b.className = "chip" + (v === current ? " on" : "");
    b.textContent = v; b.onclick = () => cb(v);
    el.appendChild(b);
  });
}

async function boot() {
  const manifest = await (await fetch("/api/samples")).json();
  SAMPLES = manifest.samples;
  const sel = $("sample");
  SAMPLES.forEach((s, i) => {
    const o = document.createElement("option");
    o.value = i; o.textContent = s.id + " · " + s.task;
    sel.appendChild(o);
  });
  sel.onchange = () => loadSample(SAMPLES[sel.value]);
  if (SAMPLES.length) { sel.value = 0; loadSample(SAMPLES[0]); }
  const status = await (await fetch("/api/status")).json();
  status.tasks.forEach(t => {
    const b = document.createElement("button");
    b.className = "chip" + (t === TASK ? " on" : "");
    b.textContent = t;
    b.onclick = () => { TASK = t;
      [...$("tasks").children].forEach(x =>
        x.classList.toggle("on", x === b)); };
    $("tasks").appendChild(b);
  });
  [...$("backends").children].forEach(c =>
    c.onclick = () => { BACKEND = c.dataset.v;
      [...$("backends").children].forEach(x =>
        x.classList.toggle("on", x === c)); });
  pollStatus();
  setInterval(pollStatus, 4000);
  $("file").onchange = upload;
  $("run").onclick = run;
}

function loadSample(s) {
  $("preview").src = "/image?src=" + encodeURIComponent(s.image);
  $("query").value = s.query || "";
  $("refboxes").value = s.ref_boxes ? s.ref_boxes.map(b => b.join(",")).join(";") : "";
  if (s.task) { TASK = s.task;
    [...$("tasks").children].forEach(x =>
      x.classList.toggle("on", x.textContent === s.task)); }
}

async function upload(ev) {
  const f = ev.target.files[0];
  if (!f) return;
  const b64 = await new Promise(res => {
    const r = new FileReader();
    r.onload = () => res(r.result.split(",")[1]); r.readAsDataURL(f);
  });
  const up = await (await fetch("/api/upload", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({image_b64: b64})})).json();
  if (!up.error) $("preview").src = "/image?src=" + encodeURIComponent(up.name);
}

function pollStatus() {
  fetch("/api/status").then(r => r.json()).then(s => {
    [["py", s.py], ["cpp", s.cpp]].forEach(([k, b]) => {
      const dot = $("dot_" + k), txt = $("st_" + k);
      dot.className = "dot" + (b.ready ? " ok" : (b.error ? " err" : ""));
      txt.textContent = b.ready
        ? (k === "cpp" ? `ready (${b.device || "?"}/${b.precision || "?"})`
                       : "ready")
        : (b.error ? "error" : "loading…");
    });
  });
}

function fillPanel(prefix, r) {
  if (!r) return;
  $( "err_" + prefix).textContent = r.error || "";
  if (r.error) { $("img_" + prefix).removeAttribute("src"); 
    $("meta_" + prefix).textContent = ""; return; }
  $("img_" + prefix).src = r.overlay || "";
  $("meta_" + prefix).textContent =
    `e2e ${(r.e2e_ms ?? 0).toFixed(0)}ms · shapes ${r.shapes ?? 0}` +
    (r.mask_count ? ` · mattes ${r.mask_count}` : "");
  const pre = $("ans_" + prefix);
  pre.hidden = false; pre.textContent = r.answer || "";
}

async function run() {
  $("run").disabled = true; $("run").textContent = "Running…";
  const body = {
    backend: BACKEND, task: TASK, query: $("query").value,
    image: ($("preview").src.match(/src=([^&]+)/) || [])[1]
      ? decodeURIComponent(($("preview").src.match(/src=([^&]+)/) || [])[1])
      : "",
  };
  const rb = $("refboxes").value.trim();
  if (rb) body.ref_boxes = rb.split(";").map(s => s.split(",").map(Number));
  try {
    const r = await (await fetch("/api/run", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify(body)})).json();
    fillPanel("py", r.py); fillPanel("cpp", r.cpp);
  } catch (e) { alert("request failed: " + e); }
  $("run").disabled = false; $("run").textContent = "Run";
}

boot();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    raise SystemExit(main())
