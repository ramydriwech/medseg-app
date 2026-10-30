"""Dataset utilities for (re)training the TP4 U-Net.

``load_image_mask_pairs`` is the loader of TP4 (cell 16): files are paired by identifier
(``Copie de 00001_rgb.png`` <-> ``Copie de 00001_mask.png``), images are read in RGB and
resized with ``INTER_AREA``, masks in grayscale with ``INTER_NEAREST``.

``split_indices`` adds a *temporal* split. TP4 used ``train_test_split(shuffle=True)`` on
100 consecutive frames of a single video: the test frames are near-duplicates of training
frames (visible in the notebook figures), so the reported test Dice (0.806) measures
memorisation of the sequence more than generalisation. Splitting the sorted frame indices
into contiguous blocks keeps the test frames away from the training ones.
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def image_key(filename: str) -> str:
    """'Copie de 00001_rgb.png' and 'Copie de 00001_mask.png' -> 'copie de 00001' (TP4)."""
    stem = Path(filename).stem.strip().casefold()
    return re.sub(r"_(rgb|mask)$", "", stem)


def index_folder(folder: Path) -> dict[str, Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")
    indexed: dict[str, Path] = {}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
            key = image_key(path.name)
            if key in indexed:
                raise ValueError(f"Duplicate identifier {key!r} in {folder}")
            indexed[key] = path
    return indexed


def load_image_mask_pairs(images_folder, masks_folder, target_size=(256, 256)):
    """Return ``(keys, images uint8 (N,H,W,3) RGB, masks uint8 (N,H,W))`` (TP4 loader)."""
    image_files = index_folder(Path(images_folder))
    mask_files = index_folder(Path(masks_folder))
    keys = sorted(image_files.keys() & mask_files.keys())
    missing = sorted(image_files.keys() ^ mask_files.keys())
    if not keys:
        raise ValueError("No image/mask pair found (expected XXX_rgb.png in Orig and XXX_mask.png in Mask).")
    if missing:
        raise ValueError(f"Incomplete pairs: {missing[:5]}")
    images, masks = [], []
    for key in keys:
        bgr = cv2.imread(str(image_files[key]), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_files[key]), cv2.IMREAD_GRAYSCALE)
        if bgr is None or mask is None:
            raise ValueError(f"Unreadable pair: {key}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        images.append(cv2.resize(rgb, target_size, interpolation=cv2.INTER_AREA))
        masks.append(cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST))
    return keys, np.asarray(images, dtype=np.uint8), np.asarray(masks, dtype=np.uint8)


def split_indices(n: int, test_size: float, val_size: float, mode: str = "temporal", seed: int = 42):
    """Train / validation / test indices.

    ``mode="random"`` reproduces TP4 (two ``train_test_split`` calls, random_state=42).
    ``mode="temporal"`` keeps contiguous blocks: train = first frames, val = next, test = last.
    """
    n_test = int(round(n * test_size))
    n_val = int(round(n * val_size))
    idx = np.arange(n)
    if mode == "random":
        from sklearn.model_selection import train_test_split

        if n_test:
            idx_tmp, idx_test = train_test_split(idx, test_size=test_size, random_state=seed, shuffle=True)
        else:
            idx_tmp, idx_test = idx, idx[:0]
        if n_val:
            idx_train, idx_val = train_test_split(
                idx_tmp, test_size=val_size / (1 - test_size), random_state=seed, shuffle=True
            )
        else:
            idx_train, idx_val = idx_tmp, idx[:0]
        return idx_train, idx_val, idx_test
    if mode != "temporal":
        raise ValueError("mode must be 'random' or 'temporal'")
    n_train = n - n_val - n_test
    return idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]
