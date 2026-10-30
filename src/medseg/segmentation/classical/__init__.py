"""Conventional segmentation methods. Each returns ``(label_map, info_dict)``."""

from .active_contours import chan_vese_segmentation
from .clustering import kmeans_segmentation, meanshift_segmentation
from .thresholding import manual_threshold, otsu_threshold
from .watershed import watershed_segmentation

__all__ = [
    "chan_vese_segmentation",
    "kmeans_segmentation",
    "manual_threshold",
    "meanshift_segmentation",
    "otsu_threshold",
    "watershed_segmentation",
]
