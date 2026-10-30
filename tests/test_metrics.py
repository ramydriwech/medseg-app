import math

import numpy as np
import pytest

from medseg.evaluation import compute_all, dice, hausdorff, iou, precision, recall
from medseg.evaluation.similarity import compare_images, correlation, mse, normalized_mutual_information, psnr


def squares():
    a = np.zeros((20, 20), bool)
    b = np.zeros((20, 20), bool)
    a[2:12, 2:12] = True   # 100 px
    b[7:17, 2:12] = True   # 100 px, 50 px overlap
    return a, b


def test_dice_iou_known_values():
    a, b = squares()
    assert dice(a, b) == pytest.approx(0.5)
    assert iou(a, b) == pytest.approx(50 / 150)
    assert precision(a, b) == pytest.approx(0.5)
    assert recall(a, b) == pytest.approx(0.5)


def test_identical_and_disjoint():
    a, _ = squares()
    assert dice(a, a) == 1.0 and iou(a, a) == 1.0
    c = np.zeros_like(a)
    c[15:20, 15:20] = True
    assert dice(a, c) == 0.0 and iou(a, c) == 0.0


def test_empty_conventions():
    e = np.zeros((5, 5), bool)
    full = np.ones((5, 5), bool)
    assert dice(e, e) == 1.0 and iou(e, e) == 1.0
    assert dice(e, full) == 0.0
    assert precision(e, full) == 0.0 and recall(full, e) == 1.0
    assert math.isnan(hausdorff(e, full))


def test_hausdorff_shift():
    a, b = squares()
    assert hausdorff(a, b) == pytest.approx(5.0)
    assert hausdorff(a, a) == 0.0
    assert hausdorff(a, b, 95) <= hausdorff(a, b)


def test_shape_mismatch():
    with pytest.raises(ValueError):
        dice(np.zeros((3, 3)), np.zeros((4, 4)))


def test_compute_all_keys():
    a, b = squares()
    assert set(compute_all(a, b)) == {"Dice", "IoU", "Précision", "Rappel", "HD95 (px)", "Hausdorff (px)"}


def test_similarity_tp5():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (40, 40), dtype=np.uint8)
    assert correlation(img, img) == pytest.approx(1.0)
    assert mse(img, img) == 0.0 and math.isinf(psnr(img, img))
    assert normalized_mutual_information(img, img) == pytest.approx(2.0)
    other = 255 - img
    assert correlation(img, other) == pytest.approx(-1.0)
    assert set(compare_images(img, other)) == {"Corrélation", "EQM", "PSNR (dB)", "IMN"}
