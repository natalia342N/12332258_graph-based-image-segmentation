import numpy as np
import pytest
from graph.eval.metrics import iou_fg, dice_fg

def test_iou_and_dice_perfect_match():
    gt = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    pred = np.array([[1, 0], [0, 1]], dtype=np.uint8)

    assert iou_fg(pred, gt) == pytest.approx(1.0, abs=1e-6)
    assert dice_fg(pred, gt) == pytest.approx(1.0, abs=1e-6)

def test_iou_and_dice_no_overlap():
    gt = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    pred = np.array([[0, 0], [0, 1]], dtype=np.uint8)
    assert iou_fg(pred, gt) == pytest.approx(0.0, abs=1e-6)
    assert dice_fg(pred, gt) == pytest.approx(0.0, abs=1e-6)
