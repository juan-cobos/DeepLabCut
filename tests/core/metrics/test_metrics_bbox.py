#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
"""Tests that bounding box mAP/mAR computation is correct."""

from __future__ import annotations

import numpy as np
import pytest

from deeplabcut.core.metrics.bbox import (
    _get_metric,
    compute_bbox_metrics,
    with_pycocotools,
)

pytestmark = pytest.mark.skipif(
    not with_pycocotools, reason="pycocotools is not installed"
)

METRIC_NAMES = [
    "mAP@50:95",
    "mAP@50",
    "mAP@75",
    "mAR@50:95",
    "mAR@50",
    "mAR@75",
]


def _ground_truth(bboxes: dict[str, list[list[float]]]) -> dict[str, dict]:
    return {
        image: {
            "width": 640,
            "height": 480,
            "bboxes": np.array(image_bboxes, dtype=float).reshape(-1, 4),
        }
        for image, image_bboxes in bboxes.items()
    }


def _detections(
    bboxes: dict[str, list[list[float]]],
    scores: dict[str, list[float]] | None = None,
) -> dict[str, dict]:
    return {
        image: {
            "bboxes": np.array(image_bboxes, dtype=float).reshape(-1, 4),
            "scores": np.array(
                [1.0] * len(image_bboxes) if scores is None else scores[image],
                dtype=float,
            ),
        }
        for image, image_bboxes in bboxes.items()
    }


def test_perfect_detections_score_100():
    bboxes = {
        "img0": [[10.0, 10.0, 50.0, 50.0], [200.0, 100.0, 80.0, 120.0]],
        "img1": [[0.0, 0.0, 100.0, 100.0]],
    }
    metrics = compute_bbox_metrics(_ground_truth(bboxes), _detections(bboxes))

    assert sorted(metrics) == sorted(METRIC_NAMES)
    for name in METRIC_NAMES:
        assert metrics[name] == pytest.approx(100.0), name


def test_no_detections_scores_zero():
    ground_truth = _ground_truth({"img0": [[10.0, 10.0, 50.0, 50.0]]})
    detections = _detections({"img0": []})
    metrics = compute_bbox_metrics(ground_truth, detections)

    assert metrics == {name: 0.0 for name in METRIC_NAMES}


def test_shifted_detections_score_below_perfect():
    gt_bboxes = {"img0": [[100.0, 100.0, 60.0, 60.0]]}
    # An IoU of ~0.68 with the ground truth: matched at 50% IoU, not at 75%.
    pred_bboxes = {"img0": [[110.0, 110.0, 60.0, 60.0]]}
    metrics = compute_bbox_metrics(
        _ground_truth(gt_bboxes), _detections(pred_bboxes)
    )

    assert metrics["mAP@50"] == pytest.approx(100.0)
    assert metrics["mAP@75"] == pytest.approx(0.0)
    assert metrics["mAR@50"] == pytest.approx(100.0)
    assert metrics["mAR@75"] == pytest.approx(0.0)
    assert 0.0 < metrics["mAP@50:95"] < 100.0
    assert 0.0 < metrics["mAR@50:95"] < 100.0


def test_false_positives_lower_precision_but_not_recall():
    gt_bboxes = {"img0": [[100.0, 100.0, 60.0, 60.0]]}
    pred_bboxes = {"img0": [[100.0, 100.0, 60.0, 60.0], [300.0, 300.0, 40.0, 40.0]]}
    detections = _detections(pred_bboxes, scores={"img0": [0.9, 0.8]})
    metrics = compute_bbox_metrics(_ground_truth(gt_bboxes), detections)

    assert metrics["mAR@50"] == pytest.approx(100.0)
    assert metrics["mAP@50"] < 100.0


def test_extra_columns_in_ground_truth_bboxes_are_ignored():
    gt = {
        "img0": {
            "width": 640,
            "height": 480,
            # e.g. bboxes stored as xywh + score
            "bboxes": np.array([[10.0, 10.0, 50.0, 50.0, 1.0]]),
        }
    }
    detections = _detections({"img0": [[10.0, 10.0, 50.0, 50.0]]})
    metrics = compute_bbox_metrics(gt, detections)

    assert metrics["mAP@50"] == pytest.approx(100.0)


def test_mismatched_number_of_images_raises():
    ground_truth = _ground_truth(
        {"img0": [[10.0, 10.0, 50.0, 50.0]], "img1": [[0.0, 0.0, 20.0, 20.0]]}
    )
    detections = _detections({"img0": [[10.0, 10.0, 50.0, 50.0]]})
    with pytest.raises(ValueError):
        compute_bbox_metrics(ground_truth, detections)


def _coco_eval():
    """Returns an accumulated COCOeval for a small, imperfect set of detections."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    rng = np.random.default_rng(0)
    coco = COCO()
    coco.dataset["categories"] = [{"id": 1, "name": "animals", "supercategory": "obj"}]
    coco.dataset["images"] = []
    coco.dataset["annotations"] = []
    coco.dataset["info"] = {}

    predictions = []
    for img_id in range(1, 5):
        coco.dataset["images"].append(
            {"id": img_id, "file_name": f"img{img_id}", "width": 640, "height": 480}
        )
        bbox = np.array([50.0 * img_id, 40.0 * img_id, 90.0, 110.0])
        coco.dataset["annotations"].append(
            {
                "id": img_id,
                "image_id": img_id,
                "category_id": 1,
                "area": float(bbox[2] * bbox[3]),
                "bbox": bbox,
                "iscrowd": 0,
            }
        )
        noisy = bbox + rng.normal(0, 5, size=4)
        predictions.append(np.array([img_id, *noisy, 0.9, 1]))

    coco.createIndex()
    coco_eval = COCOeval(coco, coco.loadRes(np.stack(predictions)), iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    return coco_eval


@pytest.mark.parametrize(
    "recall, iou_threshold, expected_name",
    [
        (False, None, "mAP@50:95"),
        (False, 0.5, "mAP@50"),
        (False, 0.75, "mAP@75"),
        (True, None, "mAR@50:95"),
        (True, 0.5, "mAR@50"),
        (True, 0.75, "mAR@75"),
    ],
)
def test_get_metric_name(recall: bool, iou_threshold: float | None, expected_name: str):
    name, value = _get_metric(
        _coco_eval(), recall=recall, iou_threshold=iou_threshold
    )
    assert name == expected_name
    assert 0.0 <= value <= 100.0


def test_get_metric_single_threshold_averages_over_all_thresholds():
    """The 50:95 metric must lie between the best and worst single-IoU scores."""
    coco_eval = _coco_eval()
    _, averaged = _get_metric(coco_eval)
    _, at_50 = _get_metric(coco_eval, iou_threshold=0.5)
    _, at_95 = _get_metric(coco_eval, iou_threshold=0.95)

    assert at_95 <= averaged <= at_50


def test_get_metric_returns_minus_one_when_no_data():
    """Area ranges without any ground truth are reported as -1, not averaged in."""
    # All ground truth boxes are "medium" (90 * 110 = 9900 px), so "small" is empty.
    name, value = _get_metric(_coco_eval(), area_rng="small")
    assert name == "mAP@50:95"
    assert value == -1
