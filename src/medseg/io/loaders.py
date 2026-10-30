"""Loading and normalisation of images and masks.

Origin: TP1 / Complement TP1 (``cv2.imread`` with ``IMREAD_COLOR`` / ``IMREAD_GRAYSCALE``,
BGR -> RGB conversion before display) and TP2 (mask resized with ``INTER_NEAREST``).

Only raster formats are supported (PNG, JPG, BMP, TIFF): none of the TPs uses DICOM or
NIfTI, so - following the project rule - those formats are deliberately not implemented.
The loader is the single place where the canonical format is enforced:

* ``uint8``; 16-bit / float inputs are min-max rescaled to [0, 255];
* RGB order (OpenCV decodes in BGR, which is converted once here);
* RGB images whose three channels are identical are reduced to grayscale. TP3 ran
  K-means / Mean-Shift on three redundant channels of a grayscale MRI; reducing them
  removes that redundancy without changing any result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import cv2
import numpy as np

SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

Source = Union[str, Path, bytes, bytearray]


@dataclass(frozen=True)
class LoadedImage:
    """A decoded image in canonical format plus the raw information needed for metadata."""

    image: np.ndarray
    name: str
    file_size: int | None = None
    original_dtype: str = "uint8"
    original_channels: int = 1
    notes: tuple[str, ...] = field(default_factory=tuple)


def _read_bytes(source: Source) -> tuple[bytes, str]:
    if isinstance(source, (bytes, bytearray)):
        return bytes(source), "upload"
    path = Path(source)
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported format '{path.suffix}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)} "
            "(DICOM/NIfTI are not used in the TPs and are therefore not implemented)."
        )
    return path.read_bytes(), path.name


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float64)
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.round((arr - lo) / (hi - lo) * 255.0).astype(np.uint8)


def load_image(source: Source, name: str | None = None) -> LoadedImage:
    """Decode an image file (path or raw bytes) into the canonical format.

    Args:
        source: file path or the raw bytes of an uploaded file.
        name: display name (defaults to the file name).

    Returns:
        LoadedImage with ``image`` as uint8 gray ``(H, W)`` or RGB ``(H, W, 3)``.

    Raises:
        ValueError: unsupported extension or undecodable content.
    """
    raw, default_name = _read_bytes(source)
    buffer = np.frombuffer(raw, dtype=np.uint8)
    decoded = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise ValueError("The file could not be decoded as an image.")

    notes: list[str] = []
    original_dtype = str(decoded.dtype)
    original_channels = 1 if decoded.ndim == 2 else decoded.shape[2]

    if decoded.ndim == 3 and decoded.shape[2] == 4:
        decoded = cv2.cvtColor(decoded, cv2.COLOR_BGRA2BGR)
        notes.append("Alpha channel dropped.")
    if decoded.dtype != np.uint8:
        decoded = _to_uint8(decoded)
        notes.append(f"Rescaled from {original_dtype} to uint8 (min-max).")

    if decoded.ndim == 3:
        b, g, r = cv2.split(decoded)
        if np.array_equal(b, g) and np.array_equal(g, r):
            image = b
            notes.append("Identical R, G and B channels: converted to grayscale.")
        else:
            image = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    else:
        image = decoded

    return LoadedImage(
        image=np.ascontiguousarray(image),
        name=name or default_name,
        file_size=len(raw),
        original_dtype=original_dtype,
        original_channels=original_channels,
        notes=tuple(notes),
    )


def load_mask(source: Source, threshold: int = 127) -> np.ndarray:
    """Load a ground-truth mask as a boolean array (TP3/TP4 convention: ``> 127``)."""
    loaded = load_image(source)
    gray = to_gray(loaded.image)
    return gray > threshold


def align_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a binary mask to ``shape`` (H, W) with nearest-neighbour interpolation (TP2)."""
    mask = np.asarray(mask).astype(bool)
    if mask.shape[:2] == tuple(shape[:2]):
        return mask.copy()
    resized = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


def to_gray(image: np.ndarray) -> np.ndarray:
    """Return a uint8 grayscale copy (ITU-R 601 luma, as ``cv2.COLOR_RGB2GRAY``)."""
    if image.ndim == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    """Return an RGB copy (gray images are replicated on three channels)."""
    if image.ndim == 3:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
