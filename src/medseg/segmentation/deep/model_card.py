"""Model cards: every deep model is shipped with a ``model_card.json`` describing the
modality and data it was trained on, so that the interface can state clearly on which
kind of images its output is meaningful (project requirement)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelCard:
    id: str
    name: str
    origin: str
    framework: str
    weights: str
    modality: str
    target: str
    training_data: str
    input_size: tuple[int, int] = (256, 256)
    channels: int = 3
    normalization: str = "x / 255"
    threshold: float = 0.5
    training: dict[str, Any] = field(default_factory=dict)
    reported_metrics: dict[str, Any] = field(default_factory=dict)
    limitations: tuple[str, ...] = ()
    is_demo: bool = False
    selected_by_default: bool = True
    folder: Path = Path(".")

    @property
    def weights_path(self) -> Path:
        return self.folder / self.weights

    @property
    def has_weights(self) -> bool:
        return self.weights_path.is_file()

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelCard":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        known = {f for f in cls.__dataclass_fields__ if f != "folder"}
        kwargs = {k: v for k, v in data.items() if k in known}
        kwargs["input_size"] = tuple(kwargs.get("input_size", (256, 256)))
        kwargs["limitations"] = tuple(kwargs.get("limitations", ()))
        return cls(folder=path.parent, **kwargs)

    def to_json(self) -> str:
        data = {k: getattr(self, k) for k in self.__dataclass_fields__ if k != "folder"}
        data["input_size"] = list(self.input_size)
        data["limitations"] = list(self.limitations)
        return json.dumps(data, indent=2, ensure_ascii=False)


def discover_model_cards(models_dir: str | Path) -> list[ModelCard]:
    """All ``models/<id>/model_card.json`` files, real models first, demo models last."""
    root = Path(models_dir)
    if not root.is_dir():
        return []
    cards = [ModelCard.from_json(p) for p in sorted(root.glob("*/model_card.json"))]
    return sorted(cards, key=lambda c: (c.is_demo, not c.has_weights, c.id))
