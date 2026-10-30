"""Single source of truth for the available methods.

Each method is declared once with its TP of origin, library and parameter schema.
The Streamlit interface builds its widgets from these schemas (no processing logic in
the UI) and the README method table is generated from the same data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from . import preprocessing as pre
from .segmentation import classical as cls

Kind = Literal["int", "float", "choice", "bool"]


@dataclass(frozen=True)
class ParamSpec:
    name: str
    label: str
    kind: Kind
    default: Any
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: tuple[Any, ...] = ()
    help: str = ""


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    category: Literal["preprocessing", "classical"]
    origin: str
    library: str
    func: Callable[..., Any]
    params: tuple[ParamSpec, ...] = ()
    description: str = ""
    from_tp: bool = True
    uses_shared_params: tuple[str, ...] = field(default_factory=tuple)

    def defaults(self) -> dict[str, Any]:
        return {p.name: p.default for p in self.params}


PREPROCESSING: dict[str, MethodSpec] = {
    m.key: m
    for m in (
        MethodSpec(
            "median", "Filtre médian", "preprocessing", "TP2 ex.1 / TP3 ex.1", "OpenCV (cv2.medianBlur)",
            pre.median_filter,
            (ParamSpec("ksize", "Taille du noyau", "int", 5, 3, 15, 2, help="Impair. TP3 : 5"),),
            "Débruitage non linéaire préservant les contours ; prétraitement de Mean-Shift au TP3.",
        ),
        MethodSpec(
            "gaussian", "Filtre gaussien", "preprocessing", "TP2 ex.1", "scikit-image (filters.gaussian)",
            pre.gaussian_filter,
            (ParamSpec("sigma", "Sigma", "float", 1.0, 0.3, 5.0, 0.1, help="TP2 : 1"),),
            "Lissage pondéré : compromis entre réduction du bruit et conservation des détails.",
        ),
        MethodSpec(
            "mean", "Filtre moyenneur", "preprocessing", "TP2 ex.1", "scikit-image (filters.rank.mean)",
            pre.mean_filter,
            (ParamSpec("radius", "Rayon du disque", "int", 3, 1, 10, 1, help="TP2 : disk(3)"),),
            "Moyenne locale : réduit le bruit mais floute les contours (référence).",
        ),
        MethodSpec(
            "clahe", "CLAHE (contraste)", "preprocessing", "Ajout (aucun TP)", "OpenCV (createCLAHE)",
            pre.clahe,
            (
                ParamSpec("clip_limit", "Limite de contraste", "float", 2.0, 0.5, 8.0, 0.5),
                ParamSpec("tile_grid_size", "Tuiles par côté", "int", 8, 2, 16, 1),
            ),
            "Égalisation adaptative limitée : exigée par l'énoncé (amélioration du contraste), absente des TPs.",
            from_tp=False,
        ),
        MethodSpec(
            "noise", "Bruit gaussien (test)", "preprocessing", "TP2 ex.1", "NumPy",
            pre.add_gaussian_noise,
            (
                ParamSpec("std", "Écart-type", "float", 25.0, 1.0, 80.0, 1.0, help="TP2 : 25"),
                ParamSpec("seed", "Graine", "int", 0, 0, 9999, 1),
            ),
            "Dégrade volontairement l'image pour évaluer les filtres de débruitage (TP2).",
        ),
    )
}

SHARED_POSTPROCESSING: tuple[ParamSpec, ...] = (
    ParamSpec("target", "Région cible", "choice", "darkest", choices=("darkest", "brightest"),
              help="TP3 : région la plus sombre (glotte, ventricules)."),
    ParamSpec("fov_threshold", "Seuil du champ utile", "int", 28, 0, 120, 1,
              help="TP2 : s = 28 sépare le vignettage noir du champ. 0 = image entière."),
    ParamSpec("fov_margin", "Marge du champ (px)", "int", 5, 0, 30, 1),
    ParamSpec("exclude_border", "Exclure ce qui touche le bord du champ", "bool", True,
              help="TP3 : supprime le fond. À désactiver si l'objet touche le bord (ex. frame442)."),
    ParamSpec("min_area", "Aire minimale (px)", "int", 20, 0, 2000, 5, help="TP3 : 20"),
    ParamSpec("keep", "Composantes conservées", "choice", "largest", choices=("largest", "all"),
              help="TP3 : toutes ; 'largest' pour un objet unique (glotte)."),
    ParamSpec("opening_size", "Ouverture (noyau)", "int", 5, 1, 21, 2, help="TP2 : 5x5. 1 = désactivée."),
)

_WS_SHARED = ("target", "fov_threshold", "fov_margin", "exclude_border", "min_area")

CLASSICAL: dict[str, MethodSpec] = {
    m.key: m
    for m in (
        MethodSpec(
            "otsu", "Seuillage d'Otsu", "classical", "TP2 ex.2", "OpenCV (THRESH_OTSU)",
            cls.otsu_threshold, (), "Seuil global optimal (variance inter-classes maximale).",
        ),
        MethodSpec(
            "manual", "Seuillage manuel", "classical", "TP2 ex.2", "OpenCV (threshold)",
            cls.manual_threshold,
            (ParamSpec("threshold", "Seuil", "int", 28, 0, 255, 1, help="TP2 : 28 (lu sur l'histogramme)"),),
            "Seuil global fixé par l'utilisateur.",
        ),
        MethodSpec(
            "kmeans", "K-means", "classical", "TP3 ex.2", "scikit-learn (KMeans)",
            cls.kmeans_segmentation,
            (
                ParamSpec("n_clusters", "Nombre de régions K", "int", 3, 2, 8, 1, help="TP3 : 3"),
                ParamSpec("n_init", "Initialisations", "int", 10, 1, 20, 1, help="TP3 : 10"),
            ),
            "Partition des pixels en K groupes minimisant l'inertie intra-classe.",
        ),
        MethodSpec(
            "meanshift", "Mean-Shift", "classical", "TP3 ex.1", "scikit-learn (MeanShift)",
            cls.meanshift_segmentation,
            (
                ParamSpec("quantile", "Quantile (bandwidth)", "float", 0.2, 0.05, 0.5, 0.05, help="TP3 : 0,2"),
                ParamSpec("n_samples", "Échantillons (bandwidth)", "int", 500, 100, 2000, 100, help="TP3 : 500"),
                ParamSpec("max_fit_pixels", "Pixels pour l'ajustement", "int", 5000, 0, 50000, 1000,
                          help="Adaptation : sous-échantillon pour la vitesse. 0 = tous (TP3)."),
            ),
            "Recherche de modes de densité ; nombre de régions automatique.",
        ),
        MethodSpec(
            "watershed", "Ligne de partage des eaux", "classical", "TP2 ex.5 (corrigé)", "OpenCV (watershed)",
            cls.watershed_segmentation,
            (
                ParamSpec("fg_erosions", "Érosions (objet sûr)", "int", 3, 0, 10, 1, help="TP2 : 3"),
                ParamSpec("bg_dilations", "Dilatations (fond sûr)", "int", 10, 1, 30, 1, help="TP2 : 10"),
                ParamSpec("kernel_size", "Noyau", "int", 5, 3, 11, 2, help="TP2 : 5x5"),
            ),
            "Marqueurs automatiques (Otsu) au lieu de la vérité terrain utilisée au TP2.",
            uses_shared_params=_WS_SHARED,
        ),
        MethodSpec(
            "chanvese", "Contour actif (Chan-Vese)", "classical", "Ajout (exercice de recherche TP3)",
            "scikit-image (morphological_chan_vese)",
            cls.chan_vese_segmentation,
            (
                ParamSpec("num_iter", "Itérations", "int", 60, 10, 300, 10),
                ParamSpec("smoothing", "Lissage", "int", 2, 0, 4, 1),
            ),
            "Level set sans bords : régularité spatiale absente du clustering.",
            from_tp=False,
        ),
    )
}


def shared_defaults() -> dict[str, Any]:
    return {p.name: p.default for p in SHARED_POSTPROCESSING}


def get_method(key: str) -> MethodSpec:
    if key in PREPROCESSING:
        return PREPROCESSING[key]
    if key in CLASSICAL:
        return CLASSICAL[key]
    raise KeyError(key)


def methods_table() -> list[dict[str, str]]:
    """Rows describing every method (used by the app and to write the README)."""
    rows = []
    for group, specs in (("Prétraitement", PREPROCESSING), ("Segmentation conventionnelle", CLASSICAL)):
        for m in specs.values():
            params = ", ".join(f"{p.name}={p.default}" for p in m.params) or "-"
            rows.append({
                "Catégorie": group, "Méthode": m.label, "Origine": m.origin,
                "Bibliothèque": m.library, "Paramètres par défaut": params, "Description": m.description,
            })
    rows.append({
        "Catégorie": "Segmentation DL", "Méthode": "U-Net", "Origine": "TP4", "Bibliothèque": "TensorFlow/Keras",
        "Paramètres par défaut": "seuil=0.5, entrée 256x256x3",
        "Description": "Encodeur-décodeur 16-256 filtres ; modèles décrits par leur model_card.json.",
    })
    return rows
