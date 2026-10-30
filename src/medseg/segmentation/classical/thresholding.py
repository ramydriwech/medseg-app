"""Global thresholding (TP2 ex.2).

Both functions return a two-label map: 0 = pixels <= threshold (dark), 1 = above (bright),
exactly as ``cv2.threshold(..., THRESH_BINARY)`` in TP2. The choice of which class is the
target (dark glottis vs bright structure) is made by the shared post-processing.

Correction w.r.t. TP2: the report states that Otsu's threshold is "close to" the manual
threshold of 28 read on the histogram. The notebook actually prints 116 for frame442:
s = 28 separates the black vignette from the endoscopic field, whereas Otsu separates the
dark half of the field from the bright mucosa. The two thresholds do not answer the same
question; neither isolates the glottis on its own - the post-processing (border and size
filtering, TP3) is what turns the dark class into a glottis mask.
"""

from __future__ import annotations

import cv2
import numpy as np

from ...io import to_gray


def otsu_threshold(image: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Otsu's optimal global threshold (``cv2.THRESH_BINARY + cv2.THRESH_OTSU``)."""
    gray = to_gray(image)
    value, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (binary > 0).astype(np.int32), {"seuil": float(value)}


def manual_threshold(image: np.ndarray, threshold: int = 28) -> tuple[np.ndarray, dict[str, float]]:
    """Manual global threshold (TP2: s = 28, read on the histogram of frame442)."""
    gray = to_gray(image)
    _, binary = cv2.threshold(gray, int(threshold), 255, cv2.THRESH_BINARY)
    return (binary > 0).astype(np.int32), {"seuil": float(threshold)}
