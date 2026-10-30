"""Deep-learning segmentation (TP4 U-Net, TensorFlow/Keras - optional dependency)."""

from .inference import KerasSegmenter, modality_warnings, prepare_input, tensorflow_available
from .model_card import ModelCard, discover_model_cards

__all__ = [
    "KerasSegmenter",
    "ModelCard",
    "discover_model_cards",
    "modality_warnings",
    "prepare_input",
    "tensorflow_available",
]
