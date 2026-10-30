"""Image similarity measures (TP5 ex.2), used here to quantify the effect of a
preprocessing step (filtered image vs original) instead of judging it by eye.

Faithful to TP5: correlation coefficient (``np.corrcoef``), mean squared error and
normalised mutual information ``(H1 + H2) / H12`` with a 64-bin joint histogram.
PSNR is added; it is a direct transformation of TP5's MSE: ``10 log10(255² / MSE)``.
Colour images are compared on their grayscale version, as in TP5.
"""

from __future__ import annotations

import math

import numpy as np

from ..io import to_gray


def _pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a, b = to_gray(a), to_gray(b)
    if a.shape != b.shape:
        raise ValueError("Both images must have the same dimensions.")
    return a, b


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of the pixel values (TP5 ``correlation_images``, off-diagonal term)."""
    a, b = _pair(a, b)
    x, y = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    if x.std() == 0 or y.std() == 0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Mean squared error (TP5 ``erreur_quadratique_moyenne``)."""
    a, b = _pair(a, b)
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Peak signal-to-noise ratio in dB (inf for identical images)."""
    e = mse(a, b)
    return math.inf if e == 0 else float(10.0 * math.log10(255.0**2 / e))


def normalized_mutual_information(a: np.ndarray, b: np.ndarray, bins: int = 64) -> float:
    """NMI = (H(A) + H(B)) / H(A, B), in [1, 2] (TP5 ``information_mutuelle_normalisee``)."""
    a, b = _pair(a, b)
    hist, _, _ = np.histogram2d(a.ravel(), b.ravel(), bins=bins, range=[[0, 256], [0, 256]])
    p_joint = hist / hist.sum()

    def entropy(p: np.ndarray) -> float:
        v = p[p > 0]
        return float(-np.sum(v * np.log2(v)))

    h_joint = entropy(p_joint)
    if h_joint == 0:
        return math.nan
    return (entropy(p_joint.sum(axis=1)) + entropy(p_joint.sum(axis=0))) / h_joint


def compare_images(reference: np.ndarray, other: np.ndarray) -> dict[str, float]:
    """All similarity measures between a reference image and a processed one."""
    return {
        "Corrélation": correlation(reference, other),
        "EQM": mse(reference, other),
        "PSNR (dB)": psnr(reference, other),
        "IMN": normalized_mutual_information(reference, other),
    }
