"""Spatial denoising filters.

Origin:
- mean filter: TP2 ex.1 - ``skimage.filters.rank.mean(img, disk(3))``
- median filter: TP2 ex.1 (``rank.median``, ``disk(3)``) and TP3 ex.1 (``cv2.medianBlur(img, 5)``).
  The TPs used two implementations; a single one is kept here (``cv2.medianBlur``,
  default ``ksize=5`` as in TP3) because it handles RGB natively and is the one
  feeding Mean-Shift in TP3.
- gaussian filter: TP2 ex.1 - ``img_as_ubyte(skimage.filters.gaussian(img, sigma=1))``

Correction w.r.t. TP2: the ``rank`` filters only accept 2-D images, so the TP applied
them to a grayscale copy. Here they are applied channel by channel so that colour
endoscopic images can be filtered without losing their colour information.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.filters import gaussian, rank
from skimage.morphology import disk
from skimage.util import img_as_ubyte


def _per_channel(image: np.ndarray, func) -> np.ndarray:
    if image.ndim == 2:
        return func(image)
    return np.stack([func(image[..., c]) for c in range(image.shape[2])], axis=-1)


def mean_filter(image: np.ndarray, radius: int = 3) -> np.ndarray:
    """Mean (box) filter on a disk-shaped neighbourhood (TP2).

    Args:
        image: uint8 gray or RGB image.
        radius: radius of the ``disk`` structuring element (TP2: 3).
    """
    footprint = disk(int(radius))
    return _per_channel(image, lambda ch: rank.mean(ch, footprint))


def median_filter(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Median filter (TP3: ``cv2.medianBlur(img, 5)``; TP2 used ``rank.median(disk(3))``).

    Args:
        image: uint8 gray or RGB image.
        ksize: odd aperture size (TP3: 5).
    """
    ksize = int(ksize)
    if ksize < 3 or ksize % 2 == 0:
        raise ValueError("ksize must be an odd integer >= 3")
    return cv2.medianBlur(np.ascontiguousarray(image), ksize)


def gaussian_filter(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Gaussian filter (TP2: ``skimage.filters.gaussian(img, sigma=1)``).

    Args:
        image: uint8 gray or RGB image.
        sigma: standard deviation of the Gaussian kernel, in pixels (TP2: 1).
    """
    channel_axis = -1 if image.ndim == 3 else None
    smoothed = gaussian(image, sigma=float(sigma), channel_axis=channel_axis, preserve_range=False)
    return img_as_ubyte(np.clip(smoothed, 0.0, 1.0))
