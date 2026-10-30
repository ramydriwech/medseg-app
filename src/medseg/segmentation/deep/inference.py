"""Inference with a Keras segmentation model described by a :class:`ModelCard`.

Input preparation reproduces TP4 exactly (``load_image_mask_pairs`` + normalisation):
RGB order, ``cv2.resize(..., INTER_AREA)`` to the training size, ``float32 / 255``.
The probability map is resized back to the original size (bilinear) *before*
thresholding, which gives smoother contours than resizing a binary mask.
"""

from __future__ import annotations

import importlib.util
import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from ...io import ensure_rgb
from .model_card import ModelCard

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def tensorflow_available() -> bool:
    """True if TensorFlow can be imported (the deep-learning part is optional)."""
    return importlib.util.find_spec("tensorflow") is not None


@lru_cache(maxsize=4)
def _load_keras_model(weights_path: str, height: int, width: int, channels: int):
    import keras

    try:
        return keras.models.load_model(weights_path, compile=False)
    except Exception as first_error:  # e.g. weights-only file, or Keras version mismatch
        from .unet_tp4 import simple_unet_model

        model = simple_unet_model(height, width, channels)
        try:
            model.load_weights(weights_path)
        except Exception as second_error:
            raise RuntimeError(
                f"Unable to load '{Path(weights_path).name}': {first_error} / {second_error}"
            ) from second_error
        return model


def prepare_input(image: np.ndarray, card: ModelCard) -> np.ndarray:
    """Image -> ``(1, H, W, C)`` float32 tensor, as in TP4."""
    h, w = card.input_size
    rgb = ensure_rgb(image) if card.channels == 3 else image
    resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    x = resized.astype(np.float32) / 255.0
    if x.ndim == 2:
        x = x[..., None]
    return x[None]


class KerasSegmenter:
    """Callable wrapper around a Keras binary-segmentation model."""

    def __init__(self, card: ModelCard):
        if not card.has_weights:
            raise FileNotFoundError(f"Weights not found: {card.weights_path}")
        if not tensorflow_available():
            raise RuntimeError("TensorFlow is not installed (pip install tensorflow).")
        self.card = card
        h, w = card.input_size
        self.model = _load_keras_model(str(card.weights_path.resolve()), h, w, card.channels)

    def predict_proba(self, image: np.ndarray) -> np.ndarray:
        """Foreground probability map at the original image size, float32 in [0, 1]."""
        prob = self.model.predict(prepare_input(image, self.card), verbose=0)[0, ..., 0]
        h, w = image.shape[:2]
        return np.clip(cv2.resize(prob.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR), 0.0, 1.0)

    def segment(self, image: np.ndarray, threshold: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(binary_mask, probability_map)``; threshold defaults to the card's (0.5, TP4)."""
        prob = self.predict_proba(image)
        thr = self.card.threshold if threshold is None else float(threshold)
        return prob >= thr, prob


def modality_warnings(image: np.ndarray, card: ModelCard) -> list[str]:
    """Warnings displayed when the input is unlike the training data of the model."""
    warnings = []
    if card.channels == 3 and image.ndim == 2:
        warnings.append(
            "Image en niveaux de gris alors que le modèle a été entraîné sur des images couleur : "
            "elle est répliquée sur 3 canaux, le résultat n'a pas de valeur."
        )
    h, w = image.shape[:2]
    th, tw = card.input_size
    if max(h / th, w / tw) > 2.0:
        warnings.append(
            f"Image {w}x{h} réduite à {tw}x{th} pour l'inférence : les petites structures sont "
            "perdues et la résolution/le cadrage diffèrent probablement de ceux de l'entraînement."
        )
    return warnings
