#!/usr/bin/env python3
"""Single entry point: publish WeVisOne-VL GGUF models + docs + test data
to the Hugging Face Hub repository Asher-1/WeVisOne-VL-GGML.

Usage:
    python3 cpp_ggml/models/upload_to_hf.py              # full upload
    python3 cpp_ggml/models/upload_to_hf.py --docs-only  # skip GGUF files
    python3 cpp_ggml/models/upload_to_hf.py --zip-only   # rebuild + upload zip

Requires: huggingface_hub (pip install huggingface_hub), authenticated
via `huggingface-cli login` or HF_TOKEN env var.
"""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from huggingface_hub import HfApi, upload_file

HERE = Path(__file__).resolve().parent
GGUF_DIR = HERE / "gguf"
MODEL_CARD_PATH = HERE / "MODEL_CARD.md"
REPO_ROOT = HERE.parent.parent  # WeVisOne-VL-GGML/
TEST_DATA_DIR = REPO_ROOT / "test_data"
REPO_ID = "Asher-1/WeVisOne-VL-GGML"
ZIP_NAME = "wevisone_vl_test_data.zip"

MODEL_FILES = {
    "2B": [
        "wevisone-vlm-2b-f16.gguf",
        "wevisone-vlm-2b-q8_0.gguf",
        "wevisone-vlm-2b-q4_K_M.gguf",
        "mmproj-wevisone-vlm-2b-f16.gguf",
        "wevisone-mask-decoder-2b-f32.gguf",
        "wevisone-mask-decoder-2b-f16.gguf",
    ],
    "4B": [
        "wevisone-vlm-4b-f16.gguf",
        "wevisone-vlm-4b-q8_0.gguf",
        "wevisone-vlm-4b-q4_K_M.gguf",
        "mmproj-wevisone-vlm-4b-f16.gguf",
        "wevisone-mask-decoder-4b-f32.gguf",
        "wevisone-mask-decoder-4b-f16.gguf",
    ],
}

ALL_FILES = [f for files in MODEL_FILES.values() for f in files]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_sha256sums() -> Path:
    """Generate SHA256SUMS for all 12 model files."""
    out = GGUF_DIR / "SHA256SUMS"
    lines = []
    for name in ALL_FILES:
        p = GGUF_DIR / name
        if not p.exists():
            print(f"  WARNING: {name} not found, skipping")
            continue
        print(f"  hashing {name} ({p.stat().st_size / 2**30:.2f} GiB)...")
        lines.append(f"{sha256_file(p)}  {name}")
    out.write_text("\n".join(lines) + "\n")
    print(f"  wrote {out} ({len(lines)} entries)")
    return out


def build_test_data_zip() -> Path:
    """Build a self-contained test-data zip with manifest + README.

    Structure:
        wevisone_vl_test_data/
            manifest.json   (image paths relative to zip root)
            README.md
            *.jpg           (25 images, flat)
    """
    manifest = json.loads((TEST_DATA_DIR / "manifest.json").read_text())
    samples = manifest["samples"]

    # Map repo-relative image paths to flat zip filenames.
    # The assets/demo/seabreeze.jpg (1400x933) gets a distinct name.
    image_map: dict[str, str] = {}
    for s in samples:
        src = s["image"]
        if src in image_map:
            continue
        if src == "assets/demo/seabreeze.jpg":
            image_map[src] = "seabreeze_hires.jpg"
        else:
            image_map[src] = Path(src).name

    # Build zip-relative manifest
    zip_samples = []
    for s in samples:
        entry = dict(s)
        entry["image"] = image_map[s["image"]]
        zip_samples.append(entry)
    zip_manifest = {
        "note": (
            "Self-contained test data for WeVisOne-VL-GGML. "
            "36 entries covering all 11 task families. "
            "Image paths are relative to this directory."
        ),
        "samples": zip_samples,
    }

    zip_readme = """\
# WeVisOne-VL Test Data

Self-contained test images for verifying the WeVisOne-VL GGML inference
pipeline. 36 manifest entries covering all 11 task families.

## Contents

- 25 JPEG images (7 official demo photos + 17 extracted case-study
  originals + 1 alternate-resolution matting input)
- `manifest.json`: task/query/image mapping (paths relative to this dir)
- This README

## Quick start

```bash
# Run matting on the chicken image (2B f16 triplet)
wevisone-cli run --image chicken.jpg --task matting --query chicken \\
    --vlm-gguf /path/to/wevisone-vlm-2b-f16.gguf \\
    --mmproj-gguf /path/to/mmproj-wevisone-vlm-2b-f16.gguf \\
    --decoder-gguf /path/to/wevisone-mask-decoder-2b-f32.gguf \\
    --device cuda --out chicken_matte.png
```

## Task coverage

| Task family | Samples |
|-------------|---------|
| detection | livingroom, seabreeze, parasol |
| referring | dogs, livingroom, chess |
| count_box | boys, dogs, chess, cups, shoe_rack, fake_apples |
| count_point | chess, colored_balls |
| point | parasol |
| ocr / ocr_grounding | coffee_sign, thomas_sign, biscuit_box, receipt, frozen_pipes |
| segmentation | parasol, gray_hair_lady, dog_car, vase_crate, remotes, sheep (none) |
| matting | seabreeze (x2 resolutions), chicken, iceberg, athletes_black, daikin_balloon |
| visual_prompt | seabreeze (pixel-space ref box: [880, 120, 1500, 1070] on 1620x1080) |
| freeform | livingroom |

## Notes

- `sheep_refseg_none` is a deliberate negative case: the query describes
  a deer not present in the photo. A `none`/empty answer is correct.
- `ref_boxes` in the manifest are pixel-space [x1, y1, x2, y2] on the
  original image resolution (seabreeze.jpg = 1620x1080).
- `seabreeze_hires.jpg` is the 1400x933 README matting input (distinct
  from the 1620x1080 official demo version).

## Source

Extracted from the WeVisOne-VL-GGML repository:
https://github.com/Asher-1/WeVisOne-VL-GGML/tree/main/test_data
"""

    out_path = GGUF_DIR / ZIP_NAME
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        prefix = "wevisone_vl_test_data/"
        zf.writestr(prefix + "manifest.json",
                    json.dumps(zip_manifest, indent=2, ensure_ascii=False))
        zf.writestr(prefix + "README.md", zip_readme)
        for src_rel, zip_name in sorted(image_map.items(),
                                        key=lambda kv: kv[1]):
            src_path = REPO_ROOT / src_rel
            if not src_path.exists():
                print(f"  WARNING: {src_rel} not found, skipping")
                continue
            zf.write(src_path, prefix + zip_name)

    print(f"  built {out_path} ({out_path.stat().st_size / 2**20:.2f} MiB, "
          f"{len(image_map)} images + manifest + README)")
    return out_path


def generate_hf_readme() -> str:
    """Generate the HF model-repo README.md."""
    readme = """\
---
library_name: ggml
tags:
  - multimodal
  - vision-language
  - gguf
  - wevisone
  - image-segmentation
  - object-detection
  - alpha-matting
license: apache-2.0
pipeline_tag: image-to-text
---

# WeVisOne-VL-GGML

GGUF exports of [WeVisOne-VL](https://github.com/Tencent/WeVisOne-VL)
(2B and 4B) for efficient local inference with the C++ ggml runtime in
[Asher-1/WeVisOne-VL-GGML](https://github.com/Asher-1/WeVisOne-VL-GGML).

## Model organization

Each inference pipeline needs a **triplet**: VLM (text stack) + mmproj
(vision tower) + mask decoder (pixel tasks). Always match sizes (2B with
2B, 4B with 4B).

### 2B models

| File | Type | Precision | Size |
|------|------|-----------|------|
"""
    for f in MODEL_FILES["2B"]:
        p = GGUF_DIR / f
        size_gb = p.stat().st_size / (1024**3) if p.exists() else 0
        if "mmproj" in f:
            ftype = "Vision Tower + Projector"
        elif "mask-decoder" in f:
            ftype = "Mask Decoder"
        else:
            ftype = "VLM (text stack)"
        prec = _precision_label(f)
        readme += f"| `{f}` | {ftype} | {prec} | {size_gb:.2f} GB |\n"

    readme += """
### 4B models

| File | Type | Precision | Size |
|------|------|-----------|------|
"""
    for f in MODEL_FILES["4B"]:
        p = GGUF_DIR / f
        size_gb = p.stat().st_size / (1024**3) if p.exists() else 0
        if "mmproj" in f:
            ftype = "Vision Tower + Projector"
        elif "mask-decoder" in f:
            ftype = "Mask Decoder"
        else:
            ftype = "VLM (text stack)"
        prec = _precision_label(f)
        readme += f"| `{f}` | {ftype} | {prec} | {size_gb:.2f} GB |\n"

    readme += """
## Precision tiers

| Family | Tier | Guidance |
|--------|------|----------|
| VLM | FP16 | **Default** — token-identical to PyTorch reference |
| VLM | Q8_0 | ~2x smaller; structured grammar preserved |
| VLM | Q4_K_M | Edge/batch; verify on parity suite first |
| mmproj | FP16 | Always f16 (not quantized) |
| Decoder | FP32 | **Parity-gated** — matches the released Python decoder |
| Decoder | FP16 | Memory-tight; MAD may drift, gate before use |

No quantized decoder is offered: 48-element tensors sit below q8_0 block
granularity; GroupNorm/MSDA statistics degrade beyond MAD thresholds.

## Usage

```bash
# Full pipeline (matting) with the 2B f16 triplet
wevisone-cli run --image photo.jpg --task matting --query person \\
    --vlm-gguf wevisone-vlm-2b-f16.gguf \\
    --mmproj-gguf mmproj-wevisone-vlm-2b-f16.gguf \\
    --decoder-gguf wevisone-mask-decoder-2b-f32.gguf \\
    --device cuda --out result.png

# Upstream llama.cpp compatibility (freeform chat, no pixel tasks)
llama-cli -m wevisone-vlm-2b-f16.gguf \\
    --mmproj mmproj-wevisone-vlm-2b-f16.gguf \\
    --image photo.jpg -p "Describe this image."
```

Build instructions and C API docs:
<https://github.com/Asher-1/WeVisOne-VL-GGML/tree/main/cpp_ggml>

## Test data

Self-contained test archive (25 images + manifest covering all 11 tasks):

**[wevisone_vl_test_data.zip](https://huggingface.co/Asher-1/WeVisOne-VL-GGML/blob/main/wevisone_vl_test_data.zip)**

## Integrity

```bash
sha256sum -c SHA256SUMS
```

## Detailed model card

See [MODEL_CARD.md](https://huggingface.co/Asher-1/WeVisOne-VL-GGML/blob/main/MODEL_CARD.md)
for pairing guide, parity criteria, and ACloudViewer integration notes.

## Original models

- [tencent/WeVisOne-VL-2B](https://huggingface.co/tencent/WeVisOne-VL-2B)
- [tencent/WeVisOne-VL-4B](https://huggingface.co/tencent/WeVisOne-VL-4B)

## License

Apache 2.0 (Tencent-authored code). See upstream model cards for weight
licensing terms.
"""
    return readme


def _precision_label(filename: str) -> str:
    if "f16" in filename:
        return "FP16"
    if "f32" in filename:
        return "FP32"
    if "q4_K_M" in filename:
        return "Q4_K_M (4-bit)"
    if "q8_0" in filename:
        return "Q8_0 (8-bit)"
    return "Unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--docs-only", action="store_true",
                    help="skip GGUF model upload (docs + zip + SHA256SUMS)")
    ap.add_argument("--zip-only", action="store_true",
                    help="only rebuild and upload the test-data zip")
    args = ap.parse_args()

    api = HfApi()
    print(f"Target: {REPO_ID}")

    # --- test data zip ---
    print("\n[1/4] Building test-data zip...")
    zip_path = build_test_data_zip()
    if args.zip_only:
        upload_file(path_or_fileobj=str(zip_path), path_in_repo=ZIP_NAME,
                    repo_id=REPO_ID, repo_type="model")
        print(f"  uploaded {ZIP_NAME}")
        return

    # --- SHA256SUMS ---
    print("\n[2/4] Generating SHA256SUMS...")
    sums_path = generate_sha256sums()

    # --- README + MODEL_CARD ---
    print("\n[3/4] Uploading docs...")
    readme_content = generate_hf_readme()
    temp_readme = GGUF_DIR / "README_HF_TEMP.md"
    temp_readme.write_text(readme_content)
    try:
        upload_file(path_or_fileobj=str(temp_readme),
                    path_in_repo="README.md",
                    repo_id=REPO_ID, repo_type="model")
        print("  uploaded README.md")
    finally:
        temp_readme.unlink(missing_ok=True)

    upload_file(path_or_fileobj=str(MODEL_CARD_PATH),
                path_in_repo="MODEL_CARD.md",
                repo_id=REPO_ID, repo_type="model")
    print("  uploaded MODEL_CARD.md")

    upload_file(path_or_fileobj=str(sums_path), path_in_repo="SHA256SUMS",
                repo_id=REPO_ID, repo_type="model")
    print("  uploaded SHA256SUMS")

    upload_file(path_or_fileobj=str(zip_path), path_in_repo=ZIP_NAME,
                repo_id=REPO_ID, repo_type="model")
    print(f"  uploaded {ZIP_NAME}")

    # --- GGUF models ---
    if args.docs_only:
        print("\n[4/4] Skipped (--docs-only).")
    else:
        print("\n[4/4] Uploading GGUF models...")
        total = len(ALL_FILES)
        for i, name in enumerate(ALL_FILES, 1):
            p = GGUF_DIR / name
            if not p.exists():
                print(f"  [{i}/{total}] MISSING: {name}")
                continue
            size_gb = p.stat().st_size / (1024**3)
            print(f"  [{i}/{total}] {name} ({size_gb:.2f} GB)...")
            upload_file(path_or_fileobj=str(p), path_in_repo=name,
                        repo_id=REPO_ID, repo_type="model")
            print(f"  [{i}/{total}] done")

    print(f"\nComplete: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()