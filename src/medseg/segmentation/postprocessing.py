"""Post-processing shared by every conventional method.

All conventional methods produce a *label map*; the same two steps then turn it into a
binary mask of the region of interest, so that the comparison between methods is fair
(TP3 discussion point: in TP3 only Mean-Shift was median-filtered, which biased the
comparison - here preprocessing and post-processing are identical for all methods).

1. ``labels_to_mask`` - TP3 ex.3: select the darkest (or brightest) region, split it into
   connected components (``cv2.connectedComponentsWithStats``), drop the components that
   touch the image border (the black background / vignette) and those with area <= 20 px.
   Added option ``keep="largest"``: keep only the largest remaining component, useful when
   the target is a single object such as the glottis.
2. ``morphological_opening`` - TP2 ex.3: opening with a 5x5 square kernel removes small
   outlier regions while preserving the size of the main region.
"""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np
from scipy import ndimage as ndi

Target = Literal["darkest", "brightest"]
Keep = Literal["all", "largest"]


def relabel_by_intensity(labels: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """Renumber labels 0..K-1 by increasing mean intensity (0 = darkest region).

    Pure cosmetic/deterministic step: clustering algorithms return arbitrary label ids.
    """
    ids = np.unique(labels)
    means = np.array([gray[labels == k].mean() for k in ids])
    order = ids[np.argsort(means, kind="stable")]
    lut = {int(old): new for new, old in enumerate(order)}
    out = np.zeros(labels.shape, dtype=np.int32)
    for old, new in lut.items():
        out[labels == old] = new
    return out


def select_region(
    labels: np.ndarray, gray: np.ndarray, target: Target = "darkest", within: np.ndarray | None = None
) -> np.ndarray:
    """Boolean mask of the label whose mean gray level is the lowest/highest (TP3 ex.3 step 1).

    ``within`` restricts the statistics to the useful field: otherwise, when the black
    vignette forms its own cluster, it is always "the darkest region" and the target is lost.
    """
    if within is None or not within.any():
        within = np.ones(labels.shape, dtype=bool)
    ids = np.unique(labels[within])
    means = np.array([gray[(labels == k) & within].mean() for k in ids])
    k = ids[np.argmin(means)] if target == "darkest" else ids[np.argmax(means)]
    return labels == k


def filter_components(
    mask: np.ndarray,
    exclude_border: bool = True,
    min_area: int = 20,
    keep: Keep = "all",
    field: np.ndarray | None = None,
) -> np.ndarray:
    """Connected-component filtering (TP3 ex.3 steps 2-3).

    Args:
        mask: boolean mask.
        exclude_border: drop components touching the border of ``field`` (TP3 used the
            image border, which is the border of the default field = whole image).
        min_area: drop components whose area is <= ``min_area`` pixels (TP3: 20).
        keep: ``"all"`` remaining components (TP3) or only the ``"largest"`` one.
        field: useful field (boolean); ``None`` = whole image.
    """
    h, w = mask.shape
    if field is None:
        field = np.ones((h, w), dtype=bool)
    edge = field & ~cv2.erode(field.astype(np.uint8), np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    n, cc, stats, _ = cv2.connectedComponentsWithStats((mask & field).astype(np.uint8), connectivity=8)
    touching = set(np.unique(cc[edge]).tolist()) if exclude_border else set()
    kept: list[tuple[int, int]] = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if i in touching or area <= min_area:
            continue
        kept.append((i, area))
    if keep == "largest" and kept:
        kept = [max(kept, key=lambda t: t[1])]
    out = np.zeros((h, w), dtype=bool)
    for i, _ in kept:
        out |= cc == i
    return out


def field_of_view(gray: np.ndarray, threshold: int = 28, margin: int = 5) -> np.ndarray:
    """Useful field of an image with a dark surround (endoscopic vignette, MRI background).

    Re-uses TP2 ex.2's manual threshold: the report reads s ~ 28 on the histogram of
    frame442 as the value "separating the black background from the rest of the image".
    The field is the largest component of ``gray > threshold`` with its holes filled
    (dark structures *inside* the field, such as the glottis or the ventricles, belong to
    it), eroded by ``margin`` pixels to drop the dark transition ring at its edge.
    ``threshold <= 0`` disables the restriction (whole image).
    """
    h, w = gray.shape
    if threshold <= 0:
        return np.ones((h, w), dtype=bool)
    bright = (gray > threshold).astype(np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, cc, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    if n <= 1:
        return np.ones((h, w), dtype=bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    fov = ndi.binary_fill_holes(cc == largest)
    if margin > 0:
        fov = cv2.erode(fov.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=int(margin)).astype(bool)
    return fov


def morphological_opening(mask: np.ndarray, kernel_size: int = 5, iterations: int = 1) -> np.ndarray:
    """Binary opening with a square kernel (TP2 ex.3: ``np.ones((5, 5))``, 1 iteration)."""
    if kernel_size <= 1 or iterations <= 0:
        return mask.astype(bool).copy()
    kernel = np.ones((int(kernel_size), int(kernel_size)), np.uint8)
    opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=int(iterations))
    return opened.astype(bool)


def labels_to_mask(
    labels: np.ndarray,
    gray: np.ndarray,
    target: Target = "darkest",
    exclude_border: bool = True,
    min_area: int = 20,
    keep: Keep = "largest",
    opening_size: int = 5,
    fov_threshold: int = 28,
    fov_margin: int = 5,
) -> np.ndarray:
    """Full shared post-processing.

    region selection (TP3) -> restriction to the useful field (TP2 threshold) ->
    opening (TP2) -> connected-component filtering (TP3).

    Why the field: on frame442 the glottis is connected to the black vignette by a chain
    of dark pixels, so TP3's "drop what touches the image border" rule removed the glottis
    together with the background and every method scored Dice = 0. Restricting the region
    to the useful field and excluding what touches the *field* border generalises TP3's
    rule (with ``fov_threshold = 0`` the field is the whole image: exactly TP3).
    """
    field = field_of_view(gray, fov_threshold, fov_margin)
    region = select_region(labels, gray, target, within=field)
    region = morphological_opening(region & field, opening_size)
    return filter_components(region, exclude_border=exclude_border, min_area=min_area, keep=keep, field=field)


def mask_contour(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Morphological contour = dilation - erosion of the mask (TP2 ex.3 question 3)."""
    kernel = np.ones((int(kernel_size), int(kernel_size)), np.uint8)
    m = mask.astype(np.uint8)
    return cv2.subtract(cv2.dilate(m, kernel), cv2.erode(m, kernel)).astype(bool)
