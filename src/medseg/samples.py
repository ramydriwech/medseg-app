"""Catalogue of the demo images shipped in ``data/samples`` (all extracted from the TP
reports and notebooks, see ``scripts/extract_samples.py``)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Sample:
    id: str
    label: str
    image: Path
    mask: Path | None
    modality: str
    source: str
    notes: str
    preset: dict[str, Any] = field(default_factory=dict)
    in_demo_training: bool = False
    tp4_prediction: Path | None = None


def load_catalog(samples_dir: str | Path) -> list[Sample]:
    root = Path(samples_dir)
    path = root / "catalog.json"
    if not path.is_file():
        return []
    entries = json.loads(path.read_text(encoding="utf-8"))["samples"]
    out = []
    for e in entries:
        out.append(Sample(
            id=e["id"], label=e["label"], image=root / e["image"],
            mask=root / e["mask"] if e.get("mask") else None,
            modality=e["modality"], source=e["source"], notes=e["notes"],
            preset=e.get("preset", {}), in_demo_training=e.get("in_demo_training", False),
            tp4_prediction=root / e["tp4_prediction"] if e.get("tp4_prediction") else None,
        ))
    return out
