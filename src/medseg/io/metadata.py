"""Image characteristics (TP1 exercise 1: pixel count, height, width, shape, max intensity,
file size; histogram analysis of the Complement TP1)."""

from __future__ import annotations

import numpy as np

from .loaders import LoadedImage


def describe_image(loaded: LoadedImage) -> dict[str, object]:
    """Return the metadata displayed by the application (pure, JSON-serialisable values)."""
    img = loaded.image
    h, w = img.shape[:2]
    channels = 1 if img.ndim == 2 else img.shape[2]
    info: dict[str, object] = {
        "Nom": loaded.name,
        "Dimensions (L x H)": f"{w} x {h}",
        "Nombre de pixels": int(h * w),
        "Canaux": channels,
        "Type": "niveaux de gris" if channels == 1 else "couleur (RGB)",
        "dtype d'origine": loaded.original_dtype,
        "Intensité min / max": f"{int(img.min())} / {int(img.max())}",
        "Intensité moyenne": round(float(img.mean()), 2),
    }
    if loaded.file_size is not None:
        info["Taille du fichier"] = f"{loaded.file_size / 1024:.1f} Ko"
    if loaded.notes:
        info["Remarques"] = " ".join(loaded.notes)
    return info


def histogram(image: np.ndarray) -> dict[str, np.ndarray]:
    """256-bin intensity histograms, one per channel (TP1 / Complement TP1)."""
    if image.ndim == 2:
        return {"gris": np.bincount(image.ravel(), minlength=256)}
    return {
        name: np.bincount(image[..., i].ravel(), minlength=256)
        for i, name in enumerate(("R", "G", "B"))
    }
