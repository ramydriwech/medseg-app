import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture
def dark_disk():
    """Synthetic 'glottis': a dark ellipse inside a bright textured field, with a black
    vignette touching the image border (same structure as an endoscopic frame)."""
    rng = np.random.default_rng(0)
    h, w = 160, 200
    yy, xx = np.mgrid[:h, :w]
    img = np.full((h, w), 170, np.float32) + rng.normal(0, 6, (h, w))
    vignette = ((yy - h / 2) / (h * 0.55)) ** 2 + ((xx - w / 2) / (w * 0.55)) ** 2 > 1
    img[vignette] = 8
    gt = ((yy - 80) / 22.0) ** 2 + ((xx - 100) / 12.0) ** 2 <= 1
    img[gt] = 45 + rng.normal(0, 4, gt.sum())
    gray = np.clip(img, 0, 255).astype(np.uint8)
    rgb = np.stack([gray, (gray * 0.8).astype(np.uint8), (gray * 0.7).astype(np.uint8)], -1)
    return rgb, gt


@pytest.fixture
def samples_dir():
    return ROOT / "data" / "samples"
