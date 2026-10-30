"""Batch evaluation of every method on all images that have a ground truth.

A comparison on a single image is anecdotal; this script runs the same pipeline as the
application on every annotated sample (or on a folder of pairs) and reports, per method,
the mean ± standard deviation of Dice / IoU / HD95 and of the execution time.

    python scripts/evaluate_dataset.py                         # samples of data/samples
    python scripts/evaluate_dataset.py --pairs path/Orig path/Mask --out results/tp4
    python scripts/evaluate_dataset.py --preprocess median gaussian --methods otsu kmeans

Outputs (in --out, default results/): per_image.csv and summary.csv (';' separated).
Per-sample presets of the shared post-processing (catalog.json) are applied, exactly
as in the application.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medseg.evaluation import compute_all  # noqa: E402
from medseg.io import align_mask, load_image, load_mask  # noqa: E402
from medseg.pipeline import run_classical, run_deep, run_preprocessing  # noqa: E402
from medseg.registry import CLASSICAL  # noqa: E402
from medseg.samples import load_catalog  # noqa: E402
from medseg.segmentation.deep import KerasSegmenter, discover_model_cards, tensorflow_available  # noqa: E402
from medseg.segmentation.deep.dataset import index_folder  # noqa: E402


def iter_cases(args):
    if args.pairs:
        imgs, masks = index_folder(Path(args.pairs[0])), index_folder(Path(args.pairs[1]))
        for key in sorted(imgs.keys() & masks.keys()):
            yield key, args.domain or "dossier", imgs[key], masks[key], {}, False, None
    else:
        for s in load_catalog(ROOT / "data" / "samples"):
            if s.mask is not None:
                domain = "Endoscopie TP4 (256x256)" if "tp4" in s.id else "Endoscopie HD (640x480)"
                yield s.id, domain, s.image, s.mask, s.preset, s.in_demo_training, s.tp4_prediction


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", nargs=2, metavar=("IMAGES_DIR", "MASKS_DIR"))
    ap.add_argument("--domain", default=None)
    ap.add_argument("--preprocess", nargs="*", default=["median"], help="preprocessing keys, in order")
    ap.add_argument("--methods", nargs="*", default=list(CLASSICAL), help="conventional method keys")
    ap.add_argument("--no-deep", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args(argv)

    segmenters = []
    if not args.no_deep and tensorflow_available():
        segmenters = [KerasSegmenter(c) for c in discover_model_cards(ROOT / "models") if c.has_weights]

    rows = []
    for case_id, domain, img_path, mask_path, preset, in_demo, tp4_pred in iter_cases(args):
        image = load_image(img_path).image
        gt = align_mask(load_mask(mask_path), image.shape[:2])
        pre_img, _ = run_preprocessing(image, [(k, {}) for k in args.preprocess])
        results = [run_classical(pre_img, k, shared=preset) for k in args.methods]
        results += [run_deep(image, s) for s in segmenters]
        for r in results:
            note = "image d'entraînement" if (in_demo and r.key.startswith("dl:") and "demo" in r.key) else ""
            rows.append({"image": case_id, "domaine": domain, "méthode": r.label, "type": r.kind,
                         **compute_all(r.mask, gt), "temps_s": r.seconds, "remarque": note})
        if tp4_pred is not None and Path(tp4_pred).is_file():
            pred = align_mask(load_mask(tp4_pred), image.shape[:2])
            rows.append({"image": case_id, "domaine": domain, "méthode": "U-Net TP4 50 ép. (prédiction du notebook)",
                         "type": "Référence TP4", **compute_all(pred, gt), "temps_s": math.nan, "remarque": ""})
        print(f"{case_id}: {len(results)} méthodes")

    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "per_image.csv", sep=";", index=False)
    summary = (
        df.groupby(["domaine", "méthode", "type"])
        .agg(n=("Dice", "size"), Dice_moy=("Dice", "mean"), Dice_std=("Dice", "std"),
             IoU_moy=("IoU", "mean"), HD95_med=("HD95 (px)", "median"), temps_moy_s=("temps_s", "mean"))
        .reset_index()
        .sort_values(["domaine", "Dice_moy"], ascending=[True, False])
    )
    summary.to_csv(args.out / "summary.csv", sep=";", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.precision", 3):
        print(summary.to_string(index=False))
    print("Écrit :", args.out / "per_image.csv", "et", args.out / "summary.csv")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
