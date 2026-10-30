"""(Re)train the TP4 U-Net and export it for the application.

Default options reproduce TP4 exactly (TP4_IM.ipynb): 256x256x3 inputs scaled to [0, 1],
masks binarised (> 127), random split 75/15/10 with random_state=42,
``tf.keras.utils.set_random_seed(42)``, Adam, binary cross-entropy, batch 16.

    # exact TP4 reproduction (50 epochs), e.g. on Google Colab with a GPU:
    python scripts/train_unet_tp4.py --data endoscopy_dataset --epochs 50 \
        --out models/unet_tp4_50ep

    # corrected protocol (recommended): temporal split + BCE+Dice + augmentation
    python scripts/train_unet_tp4.py --data endoscopy_dataset --epochs 50 \
        --split temporal --loss bce_dice --augment --out models/unet_tp4_corrige

``--data`` must contain the ``Orig/`` and ``Mask/`` folders of endoscopy_dataset.rar.
The output folder receives the model (``model.keras``), ``model_card.json`` (read by the
application), ``split.json`` (file names of each subset) and ``history.json``.

Corrections available w.r.t. TP4 (all off by default, to keep the TP reproducible):
- ``--loss bce_dice``: BCE alone on a target covering < 1 % of the pixels let the
  10-epoch model collapse to "all background" (accuracy 99.5 %, Dice 0). Adding a soft
  Dice term makes the loss sensitive to the glottis pixels from the first epochs.
- ``--split temporal``: the 100 images are consecutive frames of one video; a random
  split puts near-duplicate frames in train and test (see medseg.segmentation.deep.dataset).
- ``--augment``: horizontal flips, small rotations / zooms, brightness-contrast jitter.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medseg.evaluation import dice, iou  # noqa: E402
from medseg.segmentation.deep.dataset import load_image_mask_pairs, split_indices  # noqa: E402
from medseg.segmentation.deep.unet_tp4 import simple_unet_model  # noqa: E402


def bce_dice_loss(y_true, y_pred):
    import keras
    from keras import ops

    bce = ops.mean(keras.losses.binary_crossentropy(y_true, y_pred))
    inter = ops.sum(y_true * y_pred)
    soft_dice = (2.0 * inter + 1.0) / (ops.sum(y_true) + ops.sum(y_pred) + 1.0)
    return bce + (1.0 - soft_dice)


def augment_pair(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator):
    """Light geometric + photometric augmentation (image float [0,1] HxWx3, mask HxWx1)."""
    h, w = img.shape[:2]
    if rng.random() < 0.5:
        img, mask = img[:, ::-1], mask[:, ::-1]
    angle = rng.uniform(-10, 10)
    scale = rng.uniform(0.9, 1.1)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    m[:, 2] += rng.uniform(-0.05, 0.05, size=2) * (w, h)
    img = cv2.warpAffine(np.ascontiguousarray(img), m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    mask = cv2.warpAffine(np.ascontiguousarray(mask[..., 0]), m, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)[..., None]
    img = np.clip((img - 0.5) * rng.uniform(0.85, 1.15) + 0.5 + rng.uniform(-0.08, 0.08), 0.0, 1.0)
    return img.astype(np.float32), mask.astype(np.float32)


def make_sequence(x, y, batch_size, seed):
    import keras

    class AugmentedBatches(keras.utils.PyDataset):
        def __init__(self):
            super().__init__()
            self.rng = np.random.default_rng(seed)

        def __len__(self):
            return int(np.ceil(len(x) / batch_size))

        def __getitem__(self, i):
            idx = self.rng.permutation(len(x))[:batch_size] if len(x) > batch_size else np.arange(len(x))
            pairs = [augment_pair(x[j], y[j], self.rng) for j in idx]
            return np.stack([p[0] for p in pairs]), np.stack([p[1] for p in pairs])

    return AugmentedBatches()


def evaluate(model, x, y, batch_size):
    if len(x) == 0:
        return {}
    prob = model.predict(x, batch_size=batch_size, verbose=0)
    d = [dice(p[..., 0] >= 0.5, t[..., 0] > 0.5) for p, t in zip(prob, y)]
    j = [iou(p[..., 0] >= 0.5, t[..., 0] > 0.5) for p, t in zip(prob, y)]
    return {"Dice": round(float(np.mean(d)), 4), "IoU": round(float(np.mean(j)), 4), "n_images": int(len(x))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path, help="folder containing Orig/ and Mask/")
    ap.add_argument("--out", required=True, type=Path, help="output model folder (models/<id>)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--loss", choices=["bce", "bce_dice"], default="bce")
    ap.add_argument("--split", choices=["random", "temporal"], default="random")
    ap.add_argument("--test-size", type=float, default=0.10)
    ap.add_argument("--val-size", type=float, default=0.15)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default=None, help="display name in the application")
    ap.add_argument("--modality", default="Endoscopie laryngée (vidéo des cordes vocales), images RGB")
    ap.add_argument("--target", default="Glotte (espace glottique)")
    ap.add_argument("--training-data", default="endoscopy_dataset.rar du TP4 : 100 frames RGB 256x256 + masques binaires")
    ap.add_argument("--demo", action="store_true", help="flag the model as a demonstration model")
    ap.add_argument("--limitation", action="append", default=[], help="extra limitation (repeatable)")
    args = ap.parse_args(argv)

    import tensorflow as tf

    keys, images, masks = load_image_mask_pairs(args.data / "Orig", args.data / "Mask")
    x = images.astype(np.float32) / 255.0
    y = np.expand_dims((masks > 127).astype(np.float32), -1)
    i_tr, i_va, i_te = split_indices(len(x), args.test_size, args.val_size, args.split, args.seed)
    print(f"{len(keys)} pairs - train {len(i_tr)} / val {len(i_va)} / test {len(i_te)} ({args.split} split)")

    tf.keras.utils.set_random_seed(args.seed)
    model = simple_unet_model(256, 256, 3)
    loss = bce_dice_loss if args.loss == "bce_dice" else "binary_crossentropy"
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    val_data = (x[i_va], y[i_va]) if len(i_va) else None
    if args.augment:
        history = model.fit(make_sequence(x[i_tr], y[i_tr], args.batch_size, args.seed),
                            validation_data=val_data, epochs=args.epochs, verbose=2)
    else:
        history = model.fit(x[i_tr], y[i_tr], validation_data=val_data, batch_size=args.batch_size,
                            epochs=args.epochs, verbose=2)

    args.out.mkdir(parents=True, exist_ok=True)
    model.save(args.out / "model.keras")
    metrics = {"validation": evaluate(model, x[i_va], y[i_va], args.batch_size),
               "test": evaluate(model, x[i_te], y[i_te], args.batch_size)}
    print("metrics:", metrics)

    limitations = list(args.limitation)
    if args.split == "random" and len(i_te):
        limitations.append("Split aléatoire sur des frames consécutives d'une même vidéo (protocole TP4) : "
                           "les scores de test sont optimistes.")
    card = {
        "id": args.out.name,
        "name": args.name or f"U-Net TP4 ({args.epochs} époques, {args.loss}, split {args.split})",
        "origin": "TP4 - simple_unet_model (TP4_IM.ipynb)",
        "framework": f"TensorFlow {tf.__version__} / Keras",
        "weights": "model.keras",
        "modality": args.modality,
        "target": args.target,
        "training_data": args.training_data,
        "input_size": [256, 256],
        "channels": 3,
        "normalization": "RGB, resize INTER_AREA 256x256, x / 255",
        "threshold": 0.5,
        "training": {"epochs": args.epochs, "batch_size": args.batch_size, "optimizer": "adam",
                     "loss": args.loss, "split": args.split, "augmentation": args.augment,
                     "seed": args.seed, "date": date.today().isoformat()},
        "reported_metrics": metrics,
        "limitations": limitations,
        "is_demo": bool(args.demo),
    }
    (args.out / "model_card.json").write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
    split = {name: [keys[i] for i in idx] for name, idx in (("train", i_tr), ("val", i_va), ("test", i_te))}
    (args.out / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    (args.out / "history.json").write_text(json.dumps(hist), encoding="utf-8")
    print("saved to", args.out)


if __name__ == "__main__":
    main()
