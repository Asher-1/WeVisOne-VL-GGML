from __future__ import annotations

import numpy as np
import pytest

from evaluation.metrics import counting_metric, other_metric, pointing_bench_metric, refseg_metric
from evaluation.metrics.matting_metric import compute_matting_metrics
from evaluation.metrics.matting_metric import evaluate as matting_evaluate
from evaluation.utils.parser import parse_prediction
from evaluation.utils.prompt_builder import TMPL_MATTE, build_prompt, pointing_bench_prompt


def _rle(mask: np.ndarray) -> dict:
    # Skip (not error) when pycocotools is absent: the offline benchmark host
    # cannot install it. Only the RLE-backed tests depend on it; the parser,
    # box/counting/refseg/matting tests below run without it.
    pytest.importorskip("pycocotools")
    from pycocotools import mask as coco_mask

    rle = coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": rle["size"], "counts": rle["counts"].decode("ascii")}


def test_parser_scales_bins_and_drops_degenerate_boxes() -> None:
    text = "<ref>cat</ref><box><100><200><300><400>,<500><500><400><600></box><|im_end|>"
    assert parse_prediction(text, 1000, 500) == {"cat": [[100.0, 100.0, 300.0, 200.0]]}


def test_parser_reads_points_and_truncated_answers() -> None:
    assert parse_prediction("<ref>a</ref><point><500><250>,<10><20>", 200, 100) == {
        "a": [[100.0, 25.0], [2.0, 2.0]]
    }
    assert parse_prediction("<point><500><500></point>", 10, 10) == {}
    assert parse_prediction("<point><500><500></point>", 10, 10, pointing=True) == {"": [[5.0, 5.0]]}


def test_prompts_follow_the_benchmark_protocol() -> None:
    assert build_prompt("Dense200", ["a", "b"]) == (
        "Locate all the instances that match the following description: a. b."
    )
    assert build_prompt("RefCOCOg_val", ["the dog"]) == (
        "Locate a single instance that best matches the following description: the dog."
    )
    assert build_prompt("PR-Bench", ["red cups?"]).startswith("Locate the instances")
    assert build_prompt("SROIE", ["text"]) == "detect all the text in the box format"
    assert build_prompt("cloc-val", ["wheat heads"]) == (
        "How many wheat heads in the image? Please count them with bounding boxes."
    )
    assert build_prompt("COCO", ["cup"], pointing=True) == "point to: cup"
    assert pointing_bench_prompt("Point to the free space.") == "point to: the free space."
    assert TMPL_MATTE["single"].startswith("\nLocate a single instance")


def test_box_f1_averages_samples_and_keeps_wrong_categories_as_false_positives() -> None:
    rows = [
        {
            "task_name": "common_object_detection",
            "gt": {"cat": [[0, 0, 10, 10]]},
            "extracted_predictions": {"cat": [[0, 0, 10, 10]], "dog": [[20, 20, 30, 30]]},
        },
        {"task_name": "common_object_detection", "gt": {"cat": [[0, 0, 10, 10]]}, "extracted_predictions": {}},
    ]
    result = other_metric.evaluate(rows)
    assert result["precision@0.50"] == pytest.approx(0.25)
    assert result["recall@0.50"] == pytest.approx(0.5)
    assert result["avg_f1"] == pytest.approx(2 * 0.25 * 0.5 / 0.75)


def test_referring_ignores_labels_and_reports_rejection() -> None:
    rows = [
        {"gt": {"the dog": [[0, 0, 10, 10]]}, "extracted_predictions": {"dog": [[0, 0, 10, 10]]}},
        {"gt": {"a unicorn": []}, "extracted_predictions": {}},
        {"gt": {"a unicorn": []}, "extracted_predictions": {"x": [[0, 0, 5, 5]]}},
    ]
    result = other_metric.evaluate(rows)
    assert result["task_name"] == "referring_object_detection"
    assert result["num_absent_target"] == 2
    assert result["rejection_rate"] == pytest.approx(0.5)
    assert result["f1@0.50"] == pytest.approx(2 / 3)


def test_pointing_hits_masks_one_point_per_instance() -> None:
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:10, 5:10] = 1
    row = {"gt": {"cup": [_rle(mask)]}, "extracted_predictions": {"cup": [[6, 6], [7, 7]]}}
    result = other_metric.evaluate([row], pointing=True)
    assert result["precision@0.50"] == pytest.approx(0.5)
    assert result["recall@0.50"] == pytest.approx(1.0)


def test_counting_uses_box_cardinality_with_raw_fallback() -> None:
    rows = [
        {"gt_count": 2, "extracted_predictions": {"a": [[0, 0, 1, 1], [1, 1, 2, 2]]}},
        {"gt_count": 2, "extracted_predictions": {}, "raw_response": "<box><0><0><5><5>,<9><9><1><1>"},
        {"gt": {"count": 0}, "extracted_predictions": {}, "raw_response": ""},
    ]
    result = counting_metric.evaluate(rows)
    assert result["EM"] == pytest.approx(200 / 3)
    assert result["MAE"] == pytest.approx(1 / 3)


def test_pointing_bench_maps_points_to_mask_pixels() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[0:5, 0:5] = 1
    annotation = {
        "sample_id": "s0",
        "gt_mask": {"target": [_rle(mask)]},
        "gt_count": 1,
        "image_width": 100,
        "image_height": 100,
        "benchmark_slice": "single",
    }
    prediction = {"sample_id": "s0", "extracted_predictions": {"t": [[20.0, 20.0], [90.0, 90.0]]}}
    result = pointing_bench_metric.evaluate([prediction], [annotation])
    assert (result["precision"], result["recall"]) == (0.5, 1.0)


def test_refseg_ciou_and_refzom_accuracy() -> None:
    rows = [
        {"split": "test", "source_type": "one", "intersection": 5, "union": 10, "iou": 0.5, "pred_area": 6},
        {"split": "test", "source_type": "zero", "intersection": 0, "union": 0, "iou": 1.0, "pred_area": 0},
        {"split": "test", "source_type": "zero", "intersection": 0, "union": 3, "iou": 0.0, "pred_area": 3},
    ]
    assert refseg_metric.evaluate(rows[:1], "RefCOCO_seg")["cIoU"] == pytest.approx(50.0)
    result = refseg_metric.evaluate(rows, "Ref-ZOM_seg_test")
    assert result["cIoU"] == pytest.approx(50.0)
    assert result["Acc"] == pytest.approx(50.0)


def test_matting_metrics_and_aggregation() -> None:
    alpha = np.zeros((32, 32), dtype=np.uint8)
    alpha[8:24, 8:24] = 255
    perfect = compute_matting_metrics(alpha / 255.0, alpha)
    assert all(value == pytest.approx(0.0) for value in perfect.values())
    empty = compute_matting_metrics(np.zeros((32, 32)), alpha)
    assert empty["mad"] == pytest.approx(0.25)
    rows = [
        {"dataset": "A", "metrics": perfect},
        {"dataset": "B", "metrics": empty},
        {"dataset": "B", "metrics": empty},
    ]
    result = matting_evaluate(rows, {"AB": ["A", "B"]})
    assert result["datasets"]["B"]["MAD"] == pytest.approx(250.0)
    assert result["overall"]["MAD"] == pytest.approx(500.0 / 3)
    assert result["AB"]["count"] == 3
