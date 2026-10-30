"""Unsupervised clustering of pixel intensities / colours (TP3 ex.1 and ex.2).

Faithful to TP3:
- K-means: ``KMeans(n_clusters=3, random_state=0, n_init=10)`` on the N x C pixel matrix.
- Mean-Shift: ``estimate_bandwidth(X, quantile=0.2, n_samples=500, random_state=0)`` and
  ``MeanShift(bandwidth, bin_seeding=True)``.

Corrections / adaptations:
- TP3 applied ``cv2.medianBlur(img, 5)`` inside Mean-Shift only, which biased the
  comparison with K-means (acknowledged in the report). The median filter is now a
  regular preprocessing step applied identically to every method.
- Grayscale images are clustered on 1 feature instead of 3 redundant RGB channels.
- Mean-Shift cost grows with the number of pixels; the modes are estimated on a random
  subsample of at most ``max_fit_pixels`` pixels and every pixel is then assigned to its
  nearest mode (``MeanShift.predict``). With the default (5 000) a 640 x 480 frame is
  segmented in about a second; set 0 to fit on all pixels as in the TP.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth


def _features(image: np.ndarray) -> np.ndarray:
    channels = 1 if image.ndim == 2 else image.shape[2]
    return image.reshape(-1, channels).astype(np.float64)


def kmeans_segmentation(
    image: np.ndarray, n_clusters: int = 3, n_init: int = 10, random_state: int = 0
) -> tuple[np.ndarray, dict[str, float]]:
    """K-means clustering of pixel values (TP3 ex.2: K = 3, n_init = 10, random_state = 0)."""
    h, w = image.shape[:2]
    model = KMeans(n_clusters=int(n_clusters), n_init=int(n_init), random_state=int(random_state))
    labels = model.fit_predict(_features(image)).reshape(h, w)
    return labels.astype(np.int32), {"régions": int(n_clusters), "inertie": float(model.inertia_)}


def meanshift_segmentation(
    image: np.ndarray,
    quantile: float = 0.2,
    n_samples: int = 500,
    max_fit_pixels: int = 5_000,
    random_state: int = 0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Mean-Shift clustering of pixel values (TP3 ex.1: quantile 0.2, 500 samples, bin seeding)."""
    h, w = image.shape[:2]
    X = _features(image)
    bandwidth = estimate_bandwidth(X, quantile=float(quantile), n_samples=int(n_samples), random_state=int(random_state))
    if bandwidth <= 0:  # constant image
        return np.zeros((h, w), dtype=np.int32), {"régions": 1, "bandwidth": 0.0}
    fit_X = X
    if max_fit_pixels and len(X) > max_fit_pixels:
        rng = np.random.default_rng(int(random_state))
        fit_X = X[rng.choice(len(X), size=int(max_fit_pixels), replace=False)]
    model = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    model.fit(fit_X)
    labels = model.predict(X).reshape(h, w)
    return labels.astype(np.int32), {"régions": int(len(model.cluster_centers_)), "bandwidth": float(bandwidth)}
