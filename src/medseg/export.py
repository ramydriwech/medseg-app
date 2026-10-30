"""Export of the results: masks (PNG 0/255), metrics (CSV) and parameters (JSON) in a ZIP."""

from __future__ import annotations

import csv
import io
import json
import math
import re
import zipfile
from typing import Any, Sequence

import cv2
import numpy as np

from .visualization import mask_to_png_array


def slug(text: str) -> str:
    text = text.lower()
    for a, b in (("é", "e"), ("è", "e"), ("ê", "e"), ("à", "a"), ("ç", "c"), ("ô", "o"), ("î", "i"), ("—", "-")):
        text = text.replace(a, b)
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_") or "x"


def png_bytes(image: np.ndarray) -> bytes:
    """Encode a gray or RGB uint8 image as PNG."""
    arr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.ndim == 3 else image
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()


def rows_to_csv(rows: Sequence[dict[str, Any]]) -> str:
    if not rows:
        return ""
    out = io.StringIO()
    fields = list(rows[0].keys())
    writer = csv.DictWriter(out, fieldnames=fields, delimiter=";")
    writer.writeheader()
    for r in rows:
        writer.writerow({k: ("" if isinstance(v, float) and math.isnan(v) else v) for k, v in r.items()})
    return out.getvalue()


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def build_results_zip(
    image_name: str,
    results: Sequence[Any],
    metrics_rows: Sequence[dict[str, Any]],
    pipeline_config: dict[str, Any],
    preprocessed: np.ndarray | None = None,
) -> bytes:
    """ZIP with ``masks/<method>.png``, ``metrics.csv``, ``config.json`` (+ preprocessed image)."""
    buf = io.BytesIO()
    base = slug(image_name.rsplit(".", 1)[0])
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for r in results:
            zf.writestr(f"masks/{base}__{slug(r.label)}.png", png_bytes(mask_to_png_array(r.mask)))
        if metrics_rows:
            zf.writestr("metrics.csv", rows_to_csv(metrics_rows))
        if preprocessed is not None:
            zf.writestr(f"{base}__pretraitee.png", png_bytes(preprocessed))
        config = {
            "image": image_name,
            "pipeline": pipeline_config,
            "methods": {r.label: {"origin": r.origin, "params": r.params, "seconds": r.seconds} for r in results},
        }
        zf.writestr("config.json", json.dumps(_jsonable(config), indent=2, ensure_ascii=False))
    return buf.getvalue()
