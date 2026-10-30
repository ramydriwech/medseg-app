"""Orchestration of the processing chain (no UI code).

image --(preprocessing steps, in order)--> preprocessed image
      --(each conventional method + shared post-processing)--> masks
      --(each deep model, on the RAW image by default)--> masks
      --(ground truth, optional)--> metrics table
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .evaluation import compute_all, timed
from .io import to_gray
from .registry import CLASSICAL, PREPROCESSING, shared_defaults
from .segmentation.postprocessing import labels_to_mask


@dataclass
class StepResult:
    key: str
    label: str
    image: np.ndarray
    seconds: float
    params: dict[str, Any]


@dataclass
class SegmentationResult:
    key: str
    label: str
    kind: str  # "Classique" | "Deep learning"
    origin: str
    mask: np.ndarray
    seconds: float
    params: dict[str, Any]
    labels: np.ndarray | None = None
    probability: np.ndarray | None = None
    info: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def run_preprocessing(image: np.ndarray, steps: Sequence[tuple[str, dict[str, Any]]]) -> tuple[np.ndarray, list[StepResult]]:
    """Apply the preprocessing steps in order; returns the final image and each intermediate."""
    current = image
    results: list[StepResult] = []
    for key, params in steps:
        spec = PREPROCESSING[key]
        merged = {**spec.defaults(), **(params or {})}
        current, seconds = timed(spec.func, current, **merged)
        results.append(StepResult(key, spec.label, current, seconds, merged))
    return current, results


def run_classical(
    image: np.ndarray, key: str, params: dict[str, Any] | None = None, shared: dict[str, Any] | None = None
) -> SegmentationResult:
    """Run one conventional method followed by the shared post-processing.

    The reported time covers the method and its post-processing.
    """
    spec = CLASSICAL[key]
    merged = {**spec.defaults(), **(params or {})}
    post = {**shared_defaults(), **(shared or {})}
    call_kwargs = dict(merged)
    for name in spec.uses_shared_params:
        call_kwargs[name] = post[name]
    (labels, info), t_method = timed(spec.func, image, **call_kwargs)
    mask, t_post = timed(labels_to_mask, labels, to_gray(image), **post)
    return SegmentationResult(
        key=key, label=spec.label, kind="Classique", origin=spec.origin, mask=mask,
        seconds=t_method + t_post, params={**merged, "post": post}, labels=labels,
        info={**info, "temps_méthode_s": round(t_method, 4), "temps_post_s": round(t_post, 4)},
    )


def run_deep(image: np.ndarray, segmenter, threshold: float | None = None) -> SegmentationResult:
    """Run a deep model (any object exposing ``card`` and ``segment(image, threshold)``)."""
    from .segmentation.deep import modality_warnings

    (mask, prob), seconds = timed(segmenter.segment, image, threshold)
    card = segmenter.card
    return SegmentationResult(
        key=f"dl:{card.id}", label=card.name, kind="Deep learning", origin=card.origin, mask=mask,
        seconds=seconds, params={"seuil": card.threshold if threshold is None else threshold},
        probability=prob, warnings=modality_warnings(image, card),
    )


def metrics_table(results: Sequence[SegmentationResult], ground_truth: np.ndarray | None) -> list[dict[str, Any]]:
    """One row per method: metrics (if a ground truth is given), mask area and time."""
    rows = []
    for r in results:
        row: dict[str, Any] = {"Méthode": r.label, "Type": r.kind, "Origine": r.origin}
        if ground_truth is not None:
            row.update(compute_all(r.mask, ground_truth))
        row["Aire (px)"] = int(r.mask.sum())
        row["Temps (s)"] = r.seconds
        rows.append(row)
    return rows
