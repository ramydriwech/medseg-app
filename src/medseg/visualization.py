"""Rendering helpers (pure NumPy/OpenCV, no UI dependency).

Origin: mask overlay - TP1 ex.3; contour overlay - TP2 ex.3 (q.4); region recolouring by
mean colour and label-change contours - TP3 ex.1/ex.2.
"""

from __future__ import annotations

import cv2
import numpy as np

from .io import ensure_rgb
from .segmentation.postprocessing import mask_contour

PALETTE = {
    "mask": (255, 64, 64),
    "gt": (40, 220, 90),
    "tp": (40, 200, 90),
    "fp": (235, 70, 60),
    "fn": (60, 120, 255),
}


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = PALETTE["mask"],
    alpha: float = 0.4,
    contour: bool = True,
    contour_color: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Blend a coloured mask on the image with transparency ``alpha`` (TP1 ex.3)."""
    rgb = ensure_rgb(image).astype(np.float32)
    m = mask.astype(bool)
    out = rgb.copy()
    out[m] = (1.0 - alpha) * rgb[m] + alpha * np.asarray(color, dtype=np.float32)
    out = out.astype(np.uint8)
    if contour and m.any():
        thickness = max(1, round(max(image.shape[:2]) / 320))
        edge = mask_contour(m, 2 * thickness + 1)
        out[edge] = contour_color or color
    return out


def overlay_contours(image: np.ndarray, masks: dict[str, tuple[np.ndarray, tuple[int, int, int]]]) -> np.ndarray:
    """Draw the contours of several masks in different colours (TP2 ex.5 q.3)."""
    out = ensure_rgb(image)
    thickness = max(1, round(max(image.shape[:2]) / 320))
    for mask, color in masks.values():
        out[mask_contour(mask.astype(bool), 2 * thickness + 1)] = color
    return out


def error_map(image: np.ndarray, pred: np.ndarray, gt: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """True positives (green), false positives (red), false negatives (blue) over the image."""
    rgb = ensure_rgb(image).astype(np.float32) * 0.6
    p, g = pred.astype(bool), gt.astype(bool)
    for sel, color in ((p & g, PALETTE["tp"]), (p & ~g, PALETTE["fp"]), (~p & g, PALETTE["fn"])):
        rgb[sel] = (1 - alpha) * rgb[sel] + alpha * np.asarray(color, np.float32)
    return rgb.astype(np.uint8)


def recolor_labels(labels: np.ndarray, image: np.ndarray) -> np.ndarray:
    """Replace every region by its mean colour in the original image (TP3 'recoloriage')."""
    rgb = ensure_rgb(image)
    out = np.zeros_like(rgb)
    for k in np.unique(labels):
        sel = labels == k
        out[sel] = rgb[sel].mean(axis=0)
    return out


def label_edges(labels: np.ndarray) -> np.ndarray:
    """Pixels whose label differs from the right or bottom neighbour (TP3 contours)."""
    edges = np.zeros(labels.shape, dtype=bool)
    edges[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    edges[:-1, :] |= labels[:-1, :] != labels[1:, :]
    return edges


def probability_heatmap(prob: np.ndarray) -> np.ndarray:
    """Colour-mapped probability map (RGB)."""
    u8 = np.clip(prob * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)


def mask_to_png_array(mask: np.ndarray) -> np.ndarray:
    """Binary mask -> uint8 0/255 image (export format, same convention as the TP masks)."""
    return mask.astype(np.uint8) * 255
