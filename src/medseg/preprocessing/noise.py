"""Synthetic noise used to evaluate the denoising filters (TP2 ex.1).

Correction w.r.t. TP2: the TP drew the noise with ``np.random.normal`` without a seed,
so results changed at every run. A seeded ``numpy.random.Generator`` makes the
experiment reproducible.
"""

from __future__ import annotations

import numpy as np


def add_gaussian_noise(image: np.ndarray, std: float = 25.0, mean: float = 0.0, seed: int = 0) -> np.ndarray:
    """Add i.i.d. Gaussian noise and clip to [0, 255] (TP2: mean 0, std 25).

    Args:
        image: uint8 gray or RGB image.
        std: standard deviation of the noise (grey levels).
        mean: mean of the noise.
        seed: random seed (reproducibility).
    """
    rng = np.random.default_rng(int(seed))
    noise = rng.normal(float(mean), float(std), image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)
