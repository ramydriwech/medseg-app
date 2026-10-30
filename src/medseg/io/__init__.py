"""Image / mask loading and normalisation (TP1)."""

from .loaders import (
    SUPPORTED_EXTENSIONS,
    LoadedImage,
    align_mask,
    ensure_rgb,
    load_image,
    load_mask,
    to_gray,
)
from .metadata import describe_image

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "LoadedImage",
    "align_mask",
    "describe_image",
    "ensure_rgb",
    "load_image",
    "load_mask",
    "to_gray",
]
