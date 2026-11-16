# test_data — GUI / batch verification samples

Per-task best test images for the one-click GUI (`run_gui.sh`) and the
batch CLIs: **all 11 task families covered**, 36 manifest entries.

Two image sources:

* the official released demo photos (`demo/examples/images`, 7 photos)
  referenced **in place**, plus `assets/demo/seabreeze.jpg` (the README
  matting input) referenced in place;
* 17 clean input photos **extracted** from the README case-study
  comparison figures into `test_data/images/` (see below).

```
test_data/
  manifest.json         # {id, image, task, query, ref_boxes?} × 36 entries
  extract_originals.py  # re-runs the case-study panel extraction
  images/               # extracted clean originals (17 jpgs)
  README.md
```

Task → sample mapping (see `manifest.json`):

| Task family | Sample(s) |
|-------------|-----------|
| detection | livingroom, seabreeze, parasol |
| referring | dogs, livingroom, chess |
| count_box | boys, dogs, chess, cups, shoe_rack, fake_apples |
| count_point | chess, colored_balls |
| point | parasol |
| ocr / ocr_grounding | coffee_sign, thomas_sign, biscuit_box, receipt, frozen_pipes |
| segmentation | parasol, gray_hair_lady, dog_car, vase_crate, remotes, sheep (none case) |
| matting | seabreeze (×2 resolutions), chicken, iceberg, athletes_black, daikin_balloon |
| visual_prompt | seabreeze (+ pixel-space ref box) |
| freeform | livingroom |

## Case-study originals extracted from `assets/`

`assets/case_study/*.jpg` are 4-column comparison grids; only the
leftmost column carries the input photo. Four figures ship that column
un-annotated, so `extract_originals.py` crops it out (uniform 395px
column grid, row bands detected from the white gutters) into
`test_data/images/`:

| Figure | Extracted rows |
|--------|----------------|
| matting_text.jpg | chicken, iceberg, athletes_black, daikin_balloon |
| refseg.jpg | gray_hair_lady, dog_car, vase_crate, remotes, sheep |
| ocr.jpg | thomas_sign, biscuit_box, receipt, frozen_pipes |
| counting.jpg | cups, shoe_rack, colored_balls, fake_apples |

Notes on the extraction:

* Every left panel carries a corner badge (`Image`, or `GT: N` which
  would leak the counting answer), so a fixed 50px top strip is trimmed
  off each crop. This slightly clips the top of `frozen_pipes` (red
  header) and `daikin_balloon` (Daikin lettering); subjects stay intact.
* `detection.jpg`, `pointing.jpg`, `referring.jpg`, `matting_box.jpg`
  draw GT boxes / contours / prompt boxes on **every** panel — no clean
  original exists for them in the repo, so they contribute no samples
  (those task families are already covered by the demo photos).
* `assets/method/*`, `assets/teaser.png`, `assets/wechat-logo.png` are
  diagrams / annotated collages / logos, not test photos.
* Manifest queries for extracted samples reuse the figure captions
  verbatim. `sheep_refseg_none` is a deliberate **negative** case (the
  caption describes a deer that is not in the photo); a `none` / empty
  answer is the correct outcome there, not a failure.

## Full evaluation sets

The full evaluation families behind `evaluation/prepare/` (RefCOCO/+ /g,
ReasonSeg, Ref-L4, PixMo-Count, PointBench, P3M-500-NP, AM-2K, ...) are
deliberately NOT vendored here — they are GB-scale, several ship as
human-download releases, and the GUI verifies functionality, not scores.
Their canonical sources and the conversion flow are documented per
benchmark in `evaluation/prepare/<folder>/README.md`:

```bash
# 1. download each release into $RAW (see the per-folder README)
# 2. convert to the annotation format
python -m evaluation.prepare.<folder>.prepare_<folder> --raw_root $RAW --eval_root $EVAL_ROOT
```

## Manifest schema

`{"id", "image" (repo-relative), "task", "query", "source", "ref_boxes"?}`
— `ref_boxes` are pixel-space `[x1, y1, x2, y2]` quadruples used only by
the `visual_prompt` task. `uploads/` (created on demand) holds images
uploaded through the GUI.
