"""Evaluation: segmentation metrics (TP3/TP4), image similarity (TP5), timing (TP3)."""

from .metrics import compute_all, dice, hausdorff, iou, precision, recall
from .similarity import compare_images, correlation, mse, normalized_mutual_information, psnr
from .timing import timed

__all__ = [
    "compare_images",
    "compute_all",
    "correlation",
    "dice",
    "hausdorff",
    "iou",
    "mse",
    "normalized_mutual_information",
    "precision",
    "psnr",
    "recall",
    "timed",
]
