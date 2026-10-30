# Modèles de segmentation par Deep Learning

Chaque sous-dossier contient un `model_card.json` (fiche du modèle : modalité et données
d'entraînement, prétraitement attendu, scores rapportés, limites) et le fichier de poids
qu'elle référence. L'application liste automatiquement tous les sous-dossiers.

| Dossier | Contenu |
|---|---|
| `unet_tp4_50ep/` | U-Net du TP4, 50 époques (`unet_endoscopie_batch16_epochs50.hdf5`, poids originaux) |
| `unet_tp4_10ep/` | U-Net du TP4, 10 époques (`unet_endoscopie_batch16_epochs10.hdf5`, modèle effondré, Dice 0) |

## Ajouter un modèle

Un dossier `models/<id>/` contenant les poids (`.keras`, `.h5` ou `.hdf5`) et un `model_card.json`
(voir les exemples ; les champs `modality` et `training_data` sont obligatoires).
`scripts/train_unet_tp4.py` produit ce dossier directement :

```bash
# reproduction exacte du TP4
python scripts/train_unet_tp4.py --data endoscopy_dataset --epochs 50 --out models/unet_tp4_repro
# protocole corrigé : split temporel + BCE + Dice + augmentation
python scripts/train_unet_tp4.py --data endoscopy_dataset --epochs 50 \
    --split temporal --loss bce_dice --augment --out models/unet_tp4_corrige
```
