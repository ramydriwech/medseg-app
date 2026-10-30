# MedSeg — analyse et segmentation d'images médicales

Application web (Streamlit) du mini-projet de l'atelier *Applications en imagerie médicale*
(SUP'COM, 2026-2027). Elle couvre le prétraitement, la segmentation conventionnelle, la
segmentation par deep learning (U-Net du TP4, poids originaux) et la comparaison des résultats.
**Toutes les méthodes proviennent des TPs.** Les deux ajouts (CLAHE, Chan-Vese) et chaque
correction d'un TP sont signalés explicitement.

## Installation et lancement

Python **3.11 ou 3.12** recommandé (versions supportées par TensorFlow).

**Méthode simple** : le script crée un environnement `.venv`, installe les dépendances au premier
lancement puis démarre l'application.

| Système | Commande |
|---|---|
| Windows | double-cliquer sur `run.bat`, ou `.\run.bat` dans un terminal |
| Linux / macOS | `./run.sh` |

**Méthode manuelle** :

```bash
python -m venv .venv
.venv\Scripts\activate                 # Linux/macOS : source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-dl.txt     # TensorFlow (facultatif)
python -m streamlit run app/streamlit_app.py
```

L'application s'ouvre sur http://localhost:8501 avec l'image `frame442` et sa vérité terrain.

| Autre commande | Rôle |
|---|---|
| `python -m pytest -q` | tests unitaires (47) |
| `python scripts/evaluate_dataset.py` | toutes les méthodes sur toutes les images annotées |

### Dépannage

| Symptôme | Cause et solution |
|---|---|
| `ModuleNotFoundError: No module named 'cv2'` (ou `skimage`, `sklearn`…) | La commande `streamlit` lancée appartient à un autre Python que celui où les paquets sont installés. Lancer avec **`python -m streamlit run app/streamlit_app.py`** depuis le `.venv` (ce que fait `run.bat`). Dans VS Code : *Python: Select Interpreter* → `.venv`. L'application affiche aussi la commande d'installation exacte pour l'interpréteur utilisé. |
| « TensorFlow n'est pas installé » dans l'application | `requirements-dl.txt` a échoué (souvent une version de Python non supportée) : utiliser Python 3.11 ou 3.12. Le reste de l'application fonctionne sans TensorFlow. |
| Le port 8501 est occupé | `python -m streamlit run app/streamlit_app.py --server.port 8502` |

## Utilisation

Le panneau de gauche décrit la chaîne de traitement, dans l'ordre.

1. **Image** : une image d'exemple (extraite des comptes rendus et notebooks des TPs) ou une
   image importée (PNG, JPG, BMP, TIFF), avec un masque de vérité terrain facultatif.
2. **Prétraitement** : les étapes s'appliquent dans l'ordre de sélection, et chaque paramètre a son curseur.
3. **Segmentation conventionnelle** : choix des méthodes et de leurs paramètres. Le
   *post-traitement commun* s'applique à l'identique à toutes les méthodes.
4. **Deep learning** : choix des modèles et du seuil de décision. Par défaut, le modèle reçoit
   l'image brute, comme à l'entraînement.

| Onglet | Contenu |
|---|---|
| Image | image, vérité terrain superposée, métadonnées, histogramme (TP1) |
| Prétraitement | image après chaque étape, plus l'effet mesuré : corrélation, EQM, PSNR et IMN (TP5) |
| Segmentations | résultats côte à côte. Quatre affichages : superposition (transparence réglable), carte d'erreurs VP/FP/FN, masque binaire, régions avant post-traitement (TP3) ou carte de probabilité (DL) |
| Comparaison | tableau Dice / IoU / précision / rappel / HD95 / Hausdorff / temps, graphiques, contours superposés (TP2), accord entre méthodes (TP3) |
| Export | ZIP regroupant les masques PNG, `metrics.csv` et `config.json` (tous les paramètres) ; CSV ; masques individuels |
| Modèles DL | fiche de chaque modèle (modalité et données d'entraînement, scores, limites), import de poids |
| Méthodes & TPs | traçabilité des méthodes et liste des corrections |

## Méthodes et TP d'origine

| Catégorie | Méthode | Origine | Bibliothèque | Paramètres par défaut |
|---|---|---|---|---|
| Prétraitement | Filtre médian | TP2 ex.1 / TP3 ex.1 | OpenCV `medianBlur` | ksize=5 |
| Prétraitement | Filtre gaussien | TP2 ex.1 | scikit-image `filters.gaussian` | sigma=1 |
| Prétraitement | Filtre moyenneur | TP2 ex.1 | scikit-image `filters.rank.mean` | disk(3) |
| Prétraitement | CLAHE | **Ajout** (contraste exigé par l'énoncé, absent des TPs) | OpenCV `createCLAHE` | clip=2, 8×8 tuiles |
| Prétraitement | Bruit gaussien (test) | TP2 ex.1 | NumPy | σ=25, graine 0 |
| Segmentation | Seuillage d'Otsu | TP2 ex.2 | OpenCV `THRESH_OTSU` | — |
| Segmentation | Seuillage manuel | TP2 ex.2 | OpenCV `threshold` | s=28 |
| Segmentation | K-means | TP3 ex.2 | scikit-learn `KMeans` | K=3, n_init=10, random_state=0 |
| Segmentation | Mean-Shift | TP3 ex.1 | scikit-learn `MeanShift` | quantile=0,2, n_samples=500, bin_seeding |
| Segmentation | Ligne de partage des eaux | TP2 ex.5 (corrigé) | OpenCV `watershed` | érosions 3, dilatations 10, noyau 5×5 |
| Segmentation | Contour actif Chan-Vese | **Ajout** (exercice de recherche du TP3, non réalisé) | scikit-image `morphological_chan_vese` | 60 itérations, lissage 2 |
| Post-traitement commun | Région la plus sombre → champ utile → ouverture → composantes connexes | TP3 ex.3, TP2 ex.2-3 | OpenCV | seuil du champ 28, ouverture 5×5, aire min. 20 |
| Segmentation DL | U-Net `simple_unet_model`, 50 et 10 époques | TP4 (poids originaux) | TensorFlow/Keras | entrée 256×256×3, x/255, seuil 0,5 |
| Évaluation | Dice, IoU | TP3 ex.3 / TP4 | NumPy | — |
| Évaluation | Précision, rappel, HD95, Hausdorff | **Ajout** (demandé par l'énoncé) | NumPy / SciPy | — |
| Évaluation | Corrélation, EQM, IMN, (+ PSNR) | TP5 ex.2 | NumPy | IMN 64 classes |
| Évaluation | Temps d'exécution | TP3 ex.3 | `time.perf_counter` | — |

## Corrections apportées aux TPs

| TP | Problème constaté dans le notebook / le rapport | Traitement dans l'application |
|---|---|---|
| TP2 ex.5 | Les deux variantes du watershed utilisent la vérité terrain pour construire les marqueurs. La variante « gradient » restreint le gradient au masque GT dilaté ; elle repose sur `image_glottis`, une variable jamais définie dans le notebook. | Marqueurs automatiques : seuil d'Otsu restreint au champ utile, érosion ×3 pour l'objet sûr, dilatation ×10 pour le fond sûr. Aucune méthode ne lit la vérité terrain. |
| TP2 ex.2 | Le rapport juge le seuil d'Otsu « proche » du seuil manuel 28, alors que le notebook affiche **116**. Les deux seuils ne séparent pas les mêmes classes. | Le seuil 28 sert à délimiter le champ utile (vignettage) ; Otsu reste une méthode de segmentation. |
| TP2 ex.1 | Bruit tiré sans graine, donc non reproductible. Filtres `rank` appliqués au gris uniquement. | Graine fixée ; filtres appliqués canal par canal. |
| TP3 | Filtre médian appliqué au seul Mean-Shift, ce qui biaise la comparaison (reconnu dans le rapport). Trois canaux RGB redondants sur une IRM. `time.time()`. | Prétraitement commun à toutes les méthodes ; image à canaux identiques réduite au gris ; `perf_counter`. |
| TP3 ex.3 | Dice et IoU divisent par zéro quand les deux masques sont vides. La règle « composante touchant le bord = fond » supprime la glotte de frame442, reliée au vignettage par des pixels sombres. | Conventions explicites pour les masques vides. La région la plus sombre est sélectionnée **dans le champ utile** ; l'exclusion du bord porte sur le bord de ce champ et devient un paramètre. |
| TP4 | Split aléatoire de 100 frames consécutives d'une même vidéo : les images de test sont quasi identiques à des images d'entraînement, donc le Dice de test de 0,806 est optimiste. La perte BCE seule fait s'effondrer le modèle à 10 époques (Dice 0, accuracy 99,5 %). Pas d'augmentation de données. | `scripts/train_unet_tp4.py` reproduit le TP4 par défaut et propose `--split temporal`, `--loss bce_dice` et `--augment`. L'accuracy n'est jamais affichée. |

## Modèles de deep learning (`models/`)

| Dossier | Modèle | Sélection par défaut |
|---|---|---|
| `unet_tp4_50ep/` | U-Net du TP4, 50 époques : `unet_endoscopie_batch16_epochs50.hdf5` (poids originaux) | oui |
| `unet_tp4_10ep/` | U-Net du TP4, 10 époques : `unet_endoscopie_batch16_epochs10.hdf5`. Modèle effondré, Dice 0 | non (intérêt pédagogique) |

Chaque dossier contient une fiche `model_card.json`. Elle précise la modalité (endoscopie
laryngée RGB), les données (une seule vidéo, 100 frames en 256×256), le protocole, les scores
rapportés par le TP et les limites. L'interface affiche ces informations, et avertit lorsque
l'image analysée diffère du domaine d'entraînement (niveaux de gris, autre résolution).

**Validation du portage.** Sur les 10 frames de test du TP4, j'ai comparé le modèle 50 époques
chargé dans l'application aux prédictions enregistrées dans le notebook. Il retrouve les mêmes
glottes : Dice 0,87 sur les 6 frames où la glotte dépasse 300 px, contre 0,92 pour le notebook. Le
modèle 10 époques reproduit l'effondrement (Dice 0). L'écart résiduel vient des images : elles sont
reconstruites depuis les figures du notebook, donc un peu plus floues que les originales. Sur les
glottes de 8 à 22 px, la probabilité maximale du réseau vaut environ 0,5 et bascule sous le seuil ;
le notebook échoue lui aussi sur la frame de test 4.

## Données d'exemple (`data/samples/`)

Aucune image n'a été inventée : toutes proviennent des sources du projet
(`scripts/extract_samples.py` reconstruit le dossier).

| Dossier | Images | Provenance | Vérité terrain |
|---|---|---|---|
| `endoscopie_hd/` | frame442, frame439, frame266, frame856 (640×480) | images natives intégrées aux PDF des Ateliers 1 et 5 | frame442 uniquement (`frame442_glottis.png`) |
| `endoscopie_tp4/` | les 10 frames du **jeu de test** du TP4 (256×256) | figures du notebook TP4 (cellule 37), rééchantillonnées | oui, plus les prédictions du U-Net 50 époques enregistrées dans le notebook |
| `irm/`, `radio/` | Te-no_106 (TP3), Tumor, pneumonia (Complément TP1) | panneaux des figures des notebooks | non |

`frame442` provient d'une autre séquence (autre patient, autre date, autre résolution) que le
dataset du TP4. C'est le seul vrai test de généralisation disponible pour le U-Net.

Si le dataset original du TP4 est disponible, on peut évaluer les méthodes sur les images d'origine :

```bash
python scripts/evaluate_dataset.py --pairs endoscopy_dataset/Orig endoscopy_dataset/Mask --domain "TP4 original"
```

## Résultats de référence

`python scripts/evaluate_dataset.py` (prétraitement : médian 5×5 pour les méthodes classiques ;
image brute pour le U-Net) :

| Domaine | Méthode | Type | n | Dice moyen ± σ | IoU moyen | HD95 médiane (px) | Temps moyen |
|---|---|---|---|---|---|---|---|
| Endoscopie HD (640x480) | Mean-Shift | Classique | 1 | 0.626 | 0.456 | 72.2 | 656 ms |
| Endoscopie HD (640x480) | K-means | Classique | 1 | 0.550 | 0.379 | 210.5 | 975 ms |
| Endoscopie HD (640x480) | Seuillage manuel | Classique | 1 | 0.347 | 0.210 | 114.1 | 20 ms |
| Endoscopie HD (640x480) | Contour actif (Chan-Vese) | Classique | 1 | 0.182 | 0.100 | 271.2 | 5.24 s |
| Endoscopie HD (640x480) | Seuillage d'Otsu | Classique | 1 | 0.181 | 0.099 | 270.8 | 29 ms |
| Endoscopie HD (640x480) | Ligne de partage des eaux | Classique | 1 | 0.176 | 0.097 | 271.2 | 57 ms |
| Endoscopie HD (640x480) | U-Net TP4 — 50 époques | Deep learning | 1 | 0.018 | 0.009 | 68.7 | 399 ms |
| Endoscopie HD (640x480) | U-Net TP4 — 10 époques (effondré) | Deep learning | 1 | 0.000 | 0.000 | — | 1.00 s |
| Endoscopie TP4 (256x256) | U-Net TP4 50 ép. (prédiction du notebook) | Référence TP4 | 10 | 0.788 ± 0.30 | 0.712 | 1.0 | — |
| Endoscopie TP4 (256x256) | Contour actif (Chan-Vese) | Classique | 10 | 0.622 ± 0.31 | 0.516 | 2.2 | 1.05 s |
| Endoscopie TP4 (256x256) | Ligne de partage des eaux | Classique | 10 | 0.614 ± 0.33 | 0.511 | 2.1 | 9 ms |
| Endoscopie TP4 (256x256) | Seuillage d'Otsu | Classique | 10 | 0.610 ± 0.33 | 0.510 | 2.3 | 5 ms |
| Endoscopie TP4 (256x256) | U-Net TP4 — 50 époques | Deep learning | 10 | 0.592 ± 0.37 | 0.504 | 1.0 | 176 ms |
| Endoscopie TP4 (256x256) | K-means | Classique | 10 | 0.513 ± 0.44 | 0.450 | 5.5 | 278 ms |
| Endoscopie TP4 (256x256) | Mean-Shift | Classique | 10 | 0.438 ± 0.38 | 0.350 | 11.3 | 1.38 s |
| Endoscopie TP4 (256x256) | Seuillage manuel | Classique | 10 | 0.231 ± 0.28 | 0.160 | 27.5 | 5 ms |
| Endoscopie TP4 (256x256) | U-Net TP4 — 10 époques (effondré) | Deep learning | 10 | 0.000 ± 0.00 | 0.000 | — | 158 ms |

*Temps mesurés sur un CPU à 2 cœurs.*

Dice moyen sur les frames du TP4, stratifié par taille de glotte :

| Méthode | Glotte > 300 px (6 images) | Glotte de 8 à 22 px (4 images) |
|---|---|---|
| U-Net TP4 50 ép., prédiction du notebook (images originales) | 0.922 | 0.586 |
| U-Net TP4 50 ép., dans l'application (images reconstruites) | 0.869 | 0.178 |
| Seuillage d'Otsu | 0.866 | 0.226 |
| Ligne de partage des eaux | 0.864 | 0.238 |
| Contour actif (Chan-Vese) | 0.862 | 0.263 |
| K-means | 0.855 | 0.000 |
| Mean-Shift | 0.731 | 0.000 |
| Seuillage manuel | 0.385 | 0.000 |

À retenir :

- **Dans le domaine d'entraînement (TP4)**, quand la glotte est visible, le U-Net (0,87 à 0,92) et
  les méthodes classiques les plus simples (Otsu, watershed, Chan-Vese : environ 0,86) font jeu
  égal. La glotte y est la structure la plus sombre d'une scène peu contrastée, et un seuil global
  suffit. Le U-Net garde l'avantage sur les glottes de quelques pixels.
- **Hors domaine (frame442)**, le U-Net échoue complètement (Dice 0,02) : entraîné sur une seule
  vidéo en 256×256, il ne généralise pas à un autre patient ni à une autre résolution. Les méthodes
  à deux classes échouent aussi (environ 0,18), car toute la moitié haute de la scène est sombre.
  Seuls K-means et Mean-Shift, avec 3 classes ou plus, isolent la glotte (0,55 à 0,63), sans aucun
  apprentissage.
- Le modèle 10 époques illustre pourquoi l'accuracy est trompeuse : 99,5 % d'accuracy pour un
  masque vide.
- Ces résultats portent sur 11 images issues de 2 séquences. Ils illustrent les compromis entre
  méthodes mais ne constituent pas une évaluation statistique.

## Architecture

```
medseg-app/
├── run.bat / run.sh            installation (1re fois) + lancement
├── requirements.txt            dépendances (cœur, interface, tests)
├── requirements-dl.txt         TensorFlow (facultatif)
├── CLAUDE.md                   instructions pour Claude Code
├── app/                        interface Streamlit, sans logique de traitement
│   ├── streamlit_app.py        mise en page, onglets, vérification des dépendances
│   └── components.py           widgets générés depuis le registre, cache, graphiques
├── src/medseg/                 logique de traitement : fonctions pures, testables sans l'interface
│   ├── io/                     chargement, normalisation (uint8, RGB), métadonnées (TP1)
│   ├── preprocessing/          smoothing.py (TP2/TP3), contrast.py (CLAHE), noise.py (TP2)
│   ├── segmentation/
│   │   ├── classical/          thresholding, clustering, watershed, active_contours
│   │   ├── postprocessing.py   post-traitement commun (TP3 ex.3 + TP2)
│   │   └── deep/               unet_tp4.py, model_card.py, inference.py, dataset.py
│   ├── evaluation/             metrics.py, similarity.py (TP5), timing.py
│   ├── registry.py             méthodes, TP d'origine, schéma des paramètres
│   ├── pipeline.py             enchaînement prétraitement → segmentation → métriques
│   ├── visualization.py        superpositions, cartes d'erreurs, recoloriage (TP1/TP2/TP3)
│   ├── export.py               ZIP / CSV / PNG
│   └── samples.py              catalogue des images d'exemple
├── models/                     un dossier par modèle : model_card.json + poids
├── data/samples/               images d'exemple + catalog.json
├── scripts/                    extract_samples.py, train_unet_tp4.py, evaluate_dataset.py
├── results/                    sorties de evaluate_dataset.py
└── tests/                      tests unitaires (pytest)
```

Les conventions suivantes valent dans tout le code. Les images sont en `uint8`, gris `(H, W)` ou
RGB `(H, W, 3)`, et la conversion depuis le BGR d'OpenCV n'a lieu qu'une fois, dans `io`. Les
masques sont des booléens `(H, W)`. Chaque méthode est déclarée une seule fois dans
`registry.py` : l'interface en génère les curseurs et le tableau ci-dessus en reprend les
informations.

## Limites connues

- Les formats DICOM et NIfTI ne sont pas pris en charge : aucun TP ne les utilise, conformément à
  la règle du projet. Le module `io` est prévu pour les ajouter.
- Les images du TP4 sont reconstruites depuis les figures du notebook : utiliser le dataset original
  pour des chiffres définitifs.
- Mean-Shift ajuste ses modes sur un sous-échantillon de pixels (5 000 par défaut, 0 = tous comme
  au TP3) pour rester sous la seconde.
- Les temps affichés sont ceux du premier calcul, car les résultats sont mis en cache.
