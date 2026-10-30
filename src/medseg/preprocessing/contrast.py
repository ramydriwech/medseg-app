"""Contrast enhancement.

NOT FROM A TP. The project statement requires "amélioration du contraste" but no TP
implements one (TP1/TP2 only display histograms). CLAHE is added explicitly as the
standard choice for endoscopy: it equalises contrast locally, which compensates the
strong illumination fall-off towards the borders of the endoscopic field, while the
clip limit prevents the noise amplification of a global histogram equalisation.
For RGB images it is applied to the L channel of CIELAB so that hues are preserved.
"""

from __future__ import annotations

import cv2
import numpy as np


def clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int = 8) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalisation (OpenCV).

    Args:
        image: uint8 gray or RGB image.
        clip_limit: contrast limit per tile (OpenCV default 40 is too strong; 2.0 is usual).
        tile_grid_size: number of tiles per side.
    """
    op = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid_size),) * 2)
    if image.ndim == 2:
        return op.apply(image)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lab[..., 0] = op.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
