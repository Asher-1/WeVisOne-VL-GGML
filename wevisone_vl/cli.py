"""Command-line inference entry point."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from PIL import Image

from .engine import DEFAULT_MODEL_ID, WeVisOneVL
from .prompts import TASKS

logger = logging.getLogger(__name__)

EXIT_FAILURE = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WeVisOne-VL image inference")
    parser.add_argument("--model", default=os.environ.get("WEVISONE_MODEL", DEFAULT_MODEL_ID))
    parser.add_argument("--image", default=None,
                        help="single-image mode (default)")
    parser.add_argument(
        "--images",
        nargs="+",
        default=None,
        help="batch mode: 2+ images -> engine.predict_batch (one shared "
             "model load, one result per image) with a JSONL per image",
    )
    parser.add_argument("--jsonl", default="",
                        help="batch mode output path (JSONL, one row per "
                             "image; mirrors the cpp CLI batch payload)")
    parser.add_argument("--task", choices=TASKS, default="freeform")
    parser.add_argument("--query", default="")
    parser.add_argument("--prompt", help="Use a raw prompt instead of the task template")
    parser.add_argument(
        "--reference-boxes",
        type=json.loads,
        default=None,
        help='Pixel-space JSON boxes, e.g. "[[10,20,100,200]]"',
    )
    parser.add_argument("--output-dir", default="outputs/prediction")
    parser.add_argument("--backend", default="transformers", choices=("transformers", "vllm"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--decoder-batch-size", type=int, default=16)
    parser.add_argument("--revision")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> list[Path]:
    """Check user-supplied values before any heavy work starts.

    Returns the resolved image path list (one entry in single mode, the
    batch in --images mode). Raises ``ValueError`` with a message that is
    safe to show to the user.
    """
    if args.images and args.image:
        raise ValueError("--images and --image are mutually exclusive")
    if args.images:
        paths = [Path(p).expanduser() for p in args.images]
    elif args.image:
        paths = [Path(args.image).expanduser()]
    else:
        raise ValueError("one of --image or --images is required")
    for image_path in paths:
        if not image_path.is_file():
            raise ValueError(f"--image is not an existing file: {image_path}")
    if args.max_new_tokens <= 0:
        raise ValueError(f"--max-new-tokens must be positive, got {args.max_new_tokens}")
    if args.decoder_batch_size <= 0:
        raise ValueError(f"--decoder-batch-size must be positive, got {args.decoder_batch_size}")
    if args.temperature < 0:
        raise ValueError(f"--temperature must be non-negative, got {args.temperature}")
    output_dir = Path(args.output_dir).expanduser()
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"--output-dir exists and is not a directory: {output_dir}")
    return paths


def _load_image(path: Path) -> Image.Image:
    # ``convert`` returns a fully loaded copy, so the file handle can be closed.
    with Image.open(path) as source:
        return source.convert("RGB")


def _save_result(output_dir: str, image: Image.Image, result) -> dict | None:
    # Optional rendering stack (supervision): deferred so the batch entry
    # point works on hosts without it (offline benchmark hosts). A missing
    # stack is not an error — the prediction already succeeded; a failing
    # write is (handled by the caller).
    try:
        from .visualization import save_result
    except ImportError as exc:
        logger.warning("Skipping result files, render stack unavailable: %s", exc)
        return None
    return save_result(output_dir, image, result)


def _run_batch(engine: WeVisOneVL, args: argparse.Namespace,
               paths: list[Path]) -> int:
    """Batch mode: engine.predict_batch over all images with one shared
    model load, JSONL rows mirroring the cpp CLI batch payload."""
    images = [_load_image(p) for p in paths]
    results = engine.predict_batch(
        images,
        task=args.task,
        query=args.query,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    rows: list[dict] = []
    for index, (image_path, result) in enumerate(zip(paths, results)):
        rows.append({
            "image": str(image_path),
            "index": index,
            "answer": result.response,
            "backend": args.backend,
            "task": args.task,
            "query": args.query,
            "box_count": len(result.boxes),
            # normalized 0-1 xyxy, the same payload the cpp CLI writes
            "box_coords": [list(map(float, b)) for b in result.boxes],
            "mask_count": len(result.masks),
            "timings": {"e2e_ms": result.latency_s * 1000.0},
        })
        print(f"[{index + 1}/{len(paths)}] {image_path}")
        print(f"answer: {result.response}")
        print(
            f"timings: e2e {result.latency_s * 1000.0:.1f}ms"
        )
    if args.jsonl:
        out_path = Path(args.jsonl).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nWrote {len(rows)} rows -> {out_path}")
    return 0


def main() -> int:
    """Run inference and return a process exit code (0 on success)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    try:
        paths = validate_args(args)
    except ValueError as exc:
        logger.error("Invalid arguments: %s", exc)
        return EXIT_USAGE

    try:
        engine = WeVisOneVL(
            args.model,
            backend=args.backend,
            device=args.device,
            dtype=args.dtype,
            revision=args.revision,
            decoder_batch_size=args.decoder_batch_size,
        )
    except KeyboardInterrupt:
        logger.error("Interrupted while loading the model")
        return EXIT_INTERRUPTED
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        logger.exception("Failed to initialize model %r: %s", args.model, exc)
        return EXIT_FAILURE

    if len(paths) > 1 or args.jsonl:
        try:
            return _run_batch(engine, args, paths)
        except KeyboardInterrupt:
            logger.error("Interrupted during batch inference")
            return EXIT_INTERRUPTED
        except (OSError, RuntimeError, ValueError) as exc:
            logger.exception("Batch inference failed: %s", exc)
            return EXIT_FAILURE

    image_path = paths[0]
    try:
        image = _load_image(image_path)
    except (OSError, ValueError) as exc:  # includes UnidentifiedImageError
        logger.error("Cannot read image %s: %s", image_path, exc)
        logger.debug("Image load failure", exc_info=True)
        return EXIT_FAILURE

    try:
        result = engine.predict(
            image,
            task=args.task,
            query=args.query,
            prompt=args.prompt,
            reference_boxes=args.reference_boxes,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    except KeyboardInterrupt:
        logger.error("Interrupted during inference")
        return EXIT_INTERRUPTED
    except (OSError, RuntimeError, ValueError) as exc:  # includes CUDA out-of-memory
        logger.exception("Inference failed: %s", exc)
        return EXIT_FAILURE

    try:
        saved = _save_result(args.output_dir, image, result)
    except OSError as exc:
        logger.exception("Cannot write results to %s: %s", args.output_dir, exc)
        # The prediction itself succeeded, so still show it.
        print(result.response)
        return EXIT_FAILURE

    print(result.response)
    if saved is None:
        return 0
    print(f"\nSaved results to {saved['metadata'].parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
