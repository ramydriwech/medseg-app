"""Region-based active contour (level set).

NOT IMPLEMENTED IN A TP. The TP3 statement asks, as a research exercise, for a
segmentation by active contours (level set) to be compared with Mean-Shift / K-means;
the report notes that this part was not done. The project statement also lists
"contours actifs". It is added explicitly with scikit-image's morphological Chan-Vese
(a level-set formulation without edges, robust to the blurred glottis contours), which
brings the spatial regularity that pixel clustering lacks.
"""

from __future__ import annotations

import numpy as np
from skimage.segmentation import morphological_chan_vese

from ...io import to_gray


def chan_vese_segmentation(
    image: np.ndarray, num_iter: int = 60, smoothing: int = 2, lambda1: float = 1.0, lambda2: float = 1.0
) -> tuple[np.ndarray, dict[str, float]]:
    """Morphological Chan-Vese on the grayscale image; returns a two-label map (0/1)."""
    gray = to_gray(image).astype(np.float64) / 255.0
    level_set = morphological_chan_vese(
        gray,
        num_iter=int(num_iter),
        init_level_set="checkerboard",
        smoothing=int(smoothing),
        lambda1=float(lambda1),
        lambda2=float(lambda2),
    )
    return level_set.astype(np.int32), {"itérations": int(num_iter)}
