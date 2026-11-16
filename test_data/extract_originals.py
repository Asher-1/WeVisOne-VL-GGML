#!/usr/bin/env python3
"""Extract the clean input photos behind the README case-study figures.

assets/case_study/*.jpg are 4-column comparison grids (GT / Rex-Omni /
LocateAnything / WeVisionOne-VL, or Image / ... ). Only the leftmost
column carries the input photo; the other columns are annotated model
outputs. Four figures ship that left column un-annotated (counting,
matting_text, ocr, refseg) - detection / pointing / referring /
matting_box draw GT boxes / contours / prompt boxes on every panel, so
no clean original exists for them in the repo.

Every left panel still carries a small corner badge ("Image", or
"GT: N" which would leak the counting answer), so a fixed top strip is
trimmed off each crop. The column grid is uniform (1600px figures,
10px outer margin, 395px columns); row bands are detected from the
white gutters / caption strips between panel rows.

Run from anywhere:  python3 test_data/extract_originals.py
Outputs land in test_data/images/ and are referenced by manifest.json.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
FIGURES = HERE.parent / "assets" / "case_study"
OUT = HERE / "images"

COL_EDGES = (10, 405, 800, 1195, 1590)  # uniform 4-column grid, 1600px wide
INSET = 3      # stay clear of the gutter anti-aliasing
BADGE_TRIM = 50  # top strip holding the "Image" / "GT: N" corner badge

# figure -> per-row output names (top to bottom), left column only
ROWS = {
    "counting": ["cups", "shoe_rack", "colored_balls", "fake_apples"],
    "matting_text": ["chicken", "iceberg", "athletes_black", "daikin_balloon"],
    "ocr": ["thomas_sign", "biscuit_box", "receipt", "frozen_pipes"],
    "refseg": ["gray_hair_lady", "dog_car", "vase_crate", "remotes", "sheep"],
}


def row_bands(gray: np.ndarray) -> list[tuple[int, int]]:
    """Content row bands = runs of non-white rows; caption text strips
    between panels are thin non-white runs and are dropped by min_size."""
    profile = (gray > 245).mean(axis=1)
    mask = profile <= 0.98
    bands, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if not m and start is not None:
            if i - start >= 40:
                bands.append((start, i))
            start = None
    if start is not None and len(mask) - start >= 40:
        bands.append((start, len(mask)))
    return bands


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for figure, names in ROWS.items():
        with Image.open(FIGURES / f"{figure}.jpg") as src:
            rgb = src.convert("RGB")
        bands = row_bands(np.asarray(rgb.convert("L"), dtype=np.float32))
        if len(bands) != len(names):
            raise SystemExit(f"{figure}: detected {len(bands)} rows, "
                             f"expected {len(names)} - grid changed?")
        x0, x1 = COL_EDGES[0] + INSET, COL_EDGES[1] - INSET
        for (y0, y1), name in zip(bands, names):
            crop = rgb.crop((x0, y0 + BADGE_TRIM, x1, y1 - INSET))
            crop.save(OUT / f"{name}.jpg", quality=95)
            print(f"{figure}.jpg row -> images/{name}.jpg {crop.size}")


if __name__ == "__main__":
    main()