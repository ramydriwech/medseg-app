"""Segmentation metrics against a ground-truth mask.

Origin: Dice and IoU - TP3 ex.3 (``np.logical_and`` / ``np.logical_or``) and TP4
(``segmentation_scores``). Precision, recall and Hausdorff distances are added because
the project statement asks for them.

Corrections w.r.t. the TPs:
- TP3's ``dice``/``iou`` divide by zero when both masks are empty; TP4 used an epsilon.
  Here the convention is explicit: both empty -> 1.0 (perfect agreement), one empty -> 0.0.
- Accuracy is deliberately NOT reported: TP4 showed 99.5 % accuracy for a model that
  predicted nothing (Dice = 0), because the glottis covers < 1 % of the pixels.
- Hausdorff distances are undefined (NaN) when one of the masks is empty.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage as ndi


def _as_bool(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    return a, b


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice = 2|A∩B| / (|A| + |B|)."""
    p, g = _as_bool(pred, gt)
    total = p.sum() + g.sum()
    if total == 0:
        return 1.0
    return float(2.0 * np.logical_and(p, g).sum() / total)


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """IoU (Jaccard) = |A∩B| / |A∪B|."""
    p, g = _as_bool(pred, gt)
    union = np.logical_or(p, g).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(p, g).sum() / union)


def precision(pred: np.ndarray, gt: np.ndarray) -> float:
    """TP / (TP + FP); 1.0 if nothing is predicted and nothing is expected, else 0.0."""
    p, g = _as_bool(pred, gt)
    if p.sum() == 0:
        return 1.0 if g.sum() == 0 else 0.0
    return float(np.logical_and(p, g).sum() / p.sum())


def recall(pred: np.ndarray, gt: np.ndarray) -> float:
    """TP / (TP + FN) (sensitivity); 1.0 if the ground truth is empty."""
    p, g = _as_bool(pred, gt)
    if g.sum() == 0:
        return 1.0
    return float(np.logical_and(p, g).sum() / g.sum())


def _surface(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndi.binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)


def _surface_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distances (px) from every surface pixel of ``a`` to the surface of ``b``."""
    dist_to_b = ndi.distance_transform_edt(~_surface(b))
    return dist_to_b[_surface(a)]


def hausdorff(pred: np.ndarray, gt: np.ndarray, percentile: float = 100.0) -> float:
    """Symmetric Hausdorff distance between contours, in pixels.

    ``percentile=95`` gives HD95, far less sensitive to a single outlier pixel than the
    maximum. NaN if either mask is empty.
    """
    p, g = _as_bool(pred, gt)
    if not p.any() or not g.any():
        return math.nan
    d = np.concatenate([_surface_distances(p, g), _surface_distances(g, p)])
    return float(np.percentile(d, percentile))


def compute_all(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """All metrics displayed in the comparison table."""
    return {
        "Dice": dice(pred, gt),
        "IoU": iou(pred, gt),
        "Précision": precision(pred, gt),
        "Rappel": recall(pred, gt),
        "HD95 (px)": hausdorff(pred, gt, 95.0),
        "Hausdorff (px)": hausdorff(pred, gt, 100.0),
    }
