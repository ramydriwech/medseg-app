import io
import json
import zipfile

import numpy as np
import pytest

from medseg.export import build_results_zip, rows_to_csv
from medseg.pipeline import metrics_table, run_classical, run_deep
from medseg.samples import load_catalog
from medseg.segmentation.deep import ModelCard, discover_model_cards, modality_warnings, prepare_input
from medseg.segmentation.deep.dataset import split_indices

ROOT_MODELS = __import__("pathlib").Path(__file__).resolve().parents[1] / "models"


def test_model_cards_are_valid():
    cards = discover_model_cards(ROOT_MODELS)
    ids = {c.id for c in cards}
    assert {"unet_tp4_50ep", "unet_tp4_10ep"} <= ids
    for c in cards:
        assert c.modality and c.training_data and c.input_size == (256, 256)


def test_prepare_input_matches_tp4(dark_disk):
    rgb, _ = dark_disk
    card = ModelCard(id="x", name="x", origin="TP4", framework="k", weights="w", modality="m", target="t", training_data="d")
    x = prepare_input(rgb, card)
    assert x.shape == (1, 256, 256, 3) and x.dtype == np.float32 and 0 <= x.min() and x.max() <= 1


def test_modality_warnings():
    card = ModelCard(id="x", name="x", origin="TP4", framework="k", weights="w", modality="m", target="t", training_data="d")
    assert modality_warnings(np.zeros((256, 256, 3), np.uint8), card) == []
    assert len(modality_warnings(np.zeros((640, 640), np.uint8), card)) == 2


class FakeSegmenter:
    """Stands in for a Keras model: thresholds the red channel."""

    def __init__(self):
        self.card = ModelCard(id="fake", name="Fake", origin="test", framework="none", weights="none",
                              modality="m", target="t", training_data="d")

    def segment(self, image, threshold=None):
        prob = 1.0 - image[..., 0].astype(np.float32) / 255.0
        return prob >= (threshold or 0.5), prob


def test_run_deep_and_export(dark_disk):
    rgb, gt = dark_disk
    results = [run_classical(rgb, "kmeans"), run_deep(rgb, FakeSegmenter())]
    assert results[1].kind == "Deep learning" and results[1].probability is not None
    rows = metrics_table(results, gt)
    data = build_results_zip("img.png", results, rows, {"preprocessing": []})
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = zf.namelist()
        assert "metrics.csv" in names and "config.json" in names
        assert sum(n.startswith("masks/") for n in names) == 2
        assert json.loads(zf.read("config.json"))["image"] == "img.png"
    assert rows_to_csv(rows).count("\n") == 3


def test_split_indices():
    tr, va, te = split_indices(100, 0.10, 0.15, "temporal")
    assert (len(tr), len(va), len(te)) == (75, 15, 10) and te.min() > va.max() > tr.max()
    tr, va, te = split_indices(100, 0.10, 0.15, "random")
    assert (len(tr), len(va), len(te)) == (75, 15, 10)
    assert len(set(tr) | set(va) | set(te)) == 100


def test_catalog_files_exist(samples_dir):
    cat = load_catalog(samples_dir)
    assert len(cat) >= 10
    for s in cat:
        assert s.image.is_file()
        assert s.mask is None or s.mask.is_file()


def test_demo_model_inference_if_available(samples_dir):
    pytest.importorskip("tensorflow")
    from medseg.io import load_image, load_mask
    from medseg.segmentation.deep import KerasSegmenter

    cards = [c for c in discover_model_cards(ROOT_MODELS) if c.has_weights]
    if not cards:
        pytest.skip("no model weights available")
    img = load_image(samples_dir / "endoscopie_tp4" / "tp4_test_06.png").image
    gt = load_mask(samples_dir / "endoscopie_tp4" / "tp4_test_06_mask.png")
    res = run_deep(img, KerasSegmenter(cards[0]))
    assert res.mask.shape == gt.shape and res.probability.min() >= 0 and res.probability.max() <= 1
