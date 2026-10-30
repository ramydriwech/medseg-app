import numpy as np
import pytest

from medseg.evaluation import dice
from medseg.io import load_image, load_mask
from medseg.pipeline import metrics_table, run_classical, run_preprocessing
from medseg.registry import CLASSICAL, PREPROCESSING, methods_table
from medseg.segmentation.postprocessing import (
    field_of_view,
    filter_components,
    labels_to_mask,
    relabel_by_intensity,
    select_region,
)


@pytest.mark.parametrize("key", list(CLASSICAL))
def test_every_classical_method_finds_the_dark_object(key, dark_disk):
    rgb, gt = dark_disk
    params = {"threshold": 100} if key == "manual" else {}
    res = run_classical(rgb, key, params)
    assert res.mask.shape == gt.shape and res.mask.dtype == bool
    assert dice(res.mask, gt) > 0.85, key
    assert res.seconds >= 0


def test_field_of_view_contains_inner_dark_object(dark_disk):
    rgb, gt = dark_disk
    fov = field_of_view(rgb[..., 0], threshold=28, margin=3)
    assert fov[gt].all()           # holes filled: the dark object belongs to the field
    assert not fov[0, 0]           # vignette excluded
    assert field_of_view(rgb[..., 0], threshold=0).all()


def test_filter_components_border_area_largest():
    m = np.zeros((30, 30), bool)
    m[0:5, 0:5] = True       # touches border
    m[10:20, 10:20] = True   # 100 px
    m[24:26, 24:26] = True   # 4 px
    m[22:24, 5:9] = True     # 8 px
    assert filter_components(m, min_area=5).sum() == 108
    assert filter_components(m, min_area=5, keep="largest").sum() == 100
    assert filter_components(m, exclude_border=False, min_area=0).sum() == 25 + 100 + 4 + 8


def test_select_region_and_relabel():
    gray = np.array([[10, 10, 200], [10, 100, 200]], np.uint8)
    labels = np.array([[2, 2, 0], [2, 1, 0]])
    assert select_region(labels, gray, "darkest").sum() == 3
    assert select_region(labels, gray, "brightest").sum() == 2
    assert np.array_equal(relabel_by_intensity(labels, gray), np.array([[0, 0, 2], [0, 1, 2]]))


def test_labels_to_mask_tp3_mode_equals_image_border_rule(dark_disk):
    rgb, gt = dark_disk
    labels = (rgb[..., 0] > 100).astype(int)
    m = labels_to_mask(labels, rgb[..., 0], fov_threshold=0, opening_size=1, keep="all")
    assert dice(m, gt) > 0.95


def test_pipeline_preprocessing_order_and_timing(dark_disk):
    rgb, _ = dark_disk
    out, steps = run_preprocessing(rgb, [("median", {"ksize": 3}), ("clahe", {})])
    assert [s.key for s in steps] == ["median", "clahe"]
    assert out.shape == rgb.shape and all(s.seconds >= 0 for s in steps)


def test_metrics_table(dark_disk):
    rgb, gt = dark_disk
    rows = metrics_table([run_classical(rgb, "otsu")], gt)
    assert rows[0]["Dice"] > 0.85 and "Temps (s)" in rows[0]
    assert "Dice" not in metrics_table([run_classical(rgb, "otsu")], None)[0]


def test_registry_is_traceable():
    for spec in list(PREPROCESSING.values()) + list(CLASSICAL.values()):
        assert spec.origin and spec.library
        assert spec.from_tp == ("Ajout" not in spec.origin)
        spec.func  # callable exists
    assert len(methods_table()) == len(PREPROCESSING) + len(CLASSICAL) + 1


def test_real_frame442_kmeans_regression(samples_dir):
    """frame442: K-means isolates the glottis once the field restriction is used."""
    img = load_image(samples_dir / "endoscopie_hd" / "frame442.png").image
    gt = load_mask(samples_dir / "endoscopie_hd" / "frame442_glottis.png")
    res = run_classical(img, "kmeans", shared={"exclude_border": False})
    assert dice(res.mask, gt) > 0.6
