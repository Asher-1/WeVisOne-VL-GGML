#!/usr/bin/env python3
"""Upload the GGUF variant matrix to the Hugging Face hub repo.

Creates/updates `tencent/WeVisOne-VL-GGML-<SIZE>` (or REPO override) with:
  wevisone-vlm-{f32,f16,q8_0,q4_K_M}.gguf
  wevisone-mask-decoder-{f32,f16,q8_0}.gguf
  MODEL_CARD.md, SHA256SUMS
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf-dir", required=True)
    ap.add_argument("--repo", default=None,
                    help="default: tencent/WeVisOne-VL-GGML-<SIZE>")
    ap.add_argument("--size", choices=["2B", "4B"], required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    repo = args.repo or f"tencent/WeVisOne-VL-GGML-{args.size}"
    gguf = Path(args.gguf_dir)
    files = sorted(gguf.glob("*.gguf")) + [
        gguf / "MODEL_CARD.md", gguf / "SHA256SUMS"]
    files = [f for f in files if f.exists()]
    print(f"repo={repo} files={len(files)} dry_run={args.dry_run}")
    for f in files:
        print(f"  upload {f.name} ({f.stat().st_size / 2**30:.2f} GiB)")
    if args.dry_run:
        return 0
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
    for f in files:
        api.upload_file(path_or_fileobj=str(f), path_in_repo=f.name,
                        repo_id=repo, repo_type="model")
    print("upload complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
