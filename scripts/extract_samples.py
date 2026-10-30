"""Rebuild the demo images shipped in data/samples/ from the TP reports and notebooks.

Provenance (no image was invented):
- endoscopie_hd/: native 640x480 images embedded in the Atelier 1 and Atelier 5 PDF reports
  (frame856, frame442 + its glottis mask, frame266, frame439).
- endoscopie_tp4/: frames of the TP4 dataset (endoscopy_dataset.rar) recovered from the
  matplotlib figures stored in TP4_IM.ipynb (cells 16 and 37). The figures were rendered at
  ~312-348 px and are resampled back to 256x256: they are close to, not identical to, the
  original files. Column 3 of cell 37 holds the predictions of the TP4 U-Net (50 epochs).
- irm/, radio/: panels of TP3_IM.ipynb (Te-no_106) and TP1_IM_Complement.ipynb (Tumor, pneumonia).

Usage: python scripts/extract_samples.py <dir with PDFs/ipynb> <tmp dir> <out dir>
Requires poppler-utils (pdfimages).
"""
from __future__ import annotations

import base64
import io
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def find_file(src: Path, needle: str) -> Path:
    hits = sorted(p for p in src.iterdir() if needle in p.name)
    if not hits:
        raise FileNotFoundError(needle)
    return hits[0]


def notebook_png(nb_path: Path, cell: int) -> np.ndarray:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    for out in nb["cells"][cell].get("outputs", []):
        if "image/png" in out.get("data", {}):
            raw = base64.b64decode(out["data"]["image/png"])
            return np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
    raise ValueError(f"no PNG output in cell {cell} of {nb_path.name}")


def panel_boxes(fig: np.ndarray, min_side: int = 80) -> list[tuple[int, int, int, int]]:
    """Bounding boxes (y0, y1, x0, x1) of the image panels of a matplotlib figure."""
    non_white = (fig.min(axis=2) < 250).astype(np.uint8)
    non_white = cv2.morphologyEx(non_white, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    labels, _ = ndi.label(non_white)
    boxes = []
    for sl in ndi.find_objects(labels):
        y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        if y1 - y0 >= min_side and x1 - x0 >= min_side:
            boxes.append((y0, y1, x0, x1))
    boxes.sort(key=lambda b: (round(b[0] / 50), b[2]))
    return boxes


def crop(fig, box):
    y0, y1, x0, x1 = box
    return fig[y0:y1, x0:x1]


def to_size(img, size, is_mask=False):
    out = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    if is_mask:
        gray = out if out.ndim == 2 else cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
        return np.where(gray > 127, 255, 0).astype(np.uint8)
    return out


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if rgb.ndim == 3 else rgb)


def main(src: Path, tmp: Path, out: Path) -> None:
    tmp.mkdir(parents=True, exist_ok=True)

    # --- native images from the PDFs -------------------------------------------------
    subprocess.run(["pdfimages", "-png", str(find_file(src, "Atelier_1_1.pdf")), str(tmp / "a1")], check=True)
    subprocess.run(["pdfimages", "-j", str(find_file(src, "Atelier_5")), str(tmp / "a5")], check=True)
    rd = lambda p: cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)  # noqa: E731
    hd = out / "endoscopie_hd"
    save_rgb(hd / "frame856.png", rd(tmp / "a1-000.png"))
    save_rgb(hd / "frame442.png", rd(tmp / "a1-017.png"))
    save_rgb(hd / "frame442_glottis.png", to_size(rd(tmp / "a1-018.png"), (640, 480), is_mask=True))
    save_rgb(hd / "frame266.png", rd(tmp / "a5-002.jpg"))
    save_rgb(hd / "frame439.png", rd(tmp / "a5-003.jpg"))

    # --- TP4 frames recovered from notebook figures -----------------------------------
    tp4 = find_file(src, "TP4_IM.ipynb")
    fig = notebook_png(tp4, 37)
    boxes = panel_boxes(fig)
    assert len(boxes) == 30, len(boxes)
    d = out / "endoscopie_tp4"
    for i in range(10):
        img, ref, pred = (crop(fig, boxes[3 * i + k]) for k in range(3))
        save_rgb(d / f"tp4_test_{i:02d}.png", to_size(img, (256, 256)))
        save_rgb(d / f"tp4_test_{i:02d}_mask.png", to_size(ref, (256, 256), is_mask=True))
        save_rgb(d / "tp4_unet50_predictions" / f"tp4_test_{i:02d}_pred.png", to_size(pred, (256, 256), is_mask=True))
    fig = notebook_png(tp4, 16)
    b = panel_boxes(fig)
    save_rgb(d / "tp4_00001.png", to_size(crop(fig, b[0]), (256, 256)))
    save_rgb(d / "tp4_00001_mask.png", to_size(crop(fig, b[1]), (256, 256), is_mask=True))

    # --- MRI / X-ray panels -----------------------------------------------------------
    fig = notebook_png(find_file(src, "TP3_IM.ipynb"), 1)
    b = panel_boxes(fig)
    y0, y1, x0, x1 = b[1]                      # panel 2 has the reliable geometry
    step = b[2][2] - b[1][2]
    mri = fig[y0:y1, x0 - step:x1 - step]      # panel 1 = original Te-no_106
    save_rgb(out / "irm" / "Te-no_106.png", cv2.cvtColor(mri, cv2.COLOR_RGB2GRAY))
    fig = notebook_png(find_file(src, "TP1_IM_Compl"), 6)
    b = sorted(panel_boxes(fig), key=lambda bb: bb[2])
    save_rgb(out / "radio" / "pneumonia.png",
             cv2.cvtColor(to_size(crop(fig, b[1]), (856, 480)), cv2.COLOR_RGB2GRAY))
    save_rgb(out / "irm" / "Tumor.png",
             cv2.cvtColor(to_size(crop(fig, b[2]), (428, 417)), cv2.COLOR_RGB2GRAY))
    print("samples written to", out)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
