"""MedSeg - web interface (Streamlit).

Run from the project root:  streamlit run app/streamlit_app.py
The interface only orchestrates and displays; all processing lives in src/medseg.
"""

from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _check_dependencies() -> None:
    """Fail with an actionable message instead of a bare ModuleNotFoundError.

    The usual cause is launching the global ``streamlit`` command while the packages were
    installed in another interpreter (e.g. a virtual environment): always launch with
    ``python -m streamlit`` from the environment that holds requirements.txt (run.bat / run.sh).
    """
    import importlib.util

    required = {"cv2": "opencv-python-headless", "skimage": "scikit-image", "sklearn": "scikit-learn",
                "scipy": "scipy", "pandas": "pandas", "altair": "altair", "matplotlib": "matplotlib"}
    missing = [pkg for mod, pkg in required.items() if importlib.util.find_spec(mod) is None]
    if missing:
        import streamlit as st

        st.set_page_config(page_title="MedSeg · dépendances manquantes", page_icon=":material/build:")
        st.error(f"Paquets manquants dans l'interpréteur Python utilisé : {', '.join(missing)}")
        st.markdown(
            f"Interpréteur actuel : `{sys.executable}`\n\n"
            "Installez les dépendances **dans cet interpréteur** puis relancez :\n"
            f"```\n\"{sys.executable}\" -m pip install -r requirements.txt -r requirements-dl.txt\n```\n"
            "Ou utilisez `run.bat` (Windows) / `run.sh` (Linux, macOS), qui créent un environnement "
            "`.venv` et lancent l'application avec `python -m streamlit` (même interpréteur garanti)."
        )
        st.stop()


_check_dependencies()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import components as ui  # noqa: E402
from medseg import visualization as viz  # noqa: E402
from medseg.evaluation import compare_images, dice  # noqa: E402
from medseg.export import build_results_zip, png_bytes, rows_to_csv  # noqa: E402
from medseg.io import align_mask, describe_image, load_image, load_mask  # noqa: E402
from medseg.io.metadata import histogram  # noqa: E402
from medseg.pipeline import SegmentationResult, metrics_table  # noqa: E402
from medseg.registry import CLASSICAL, PREPROCESSING, SHARED_POSTPROCESSING, methods_table  # noqa: E402
from medseg.samples import load_catalog  # noqa: E402
from medseg.segmentation.deep import discover_model_cards, tensorflow_available  # noqa: E402

SAMPLES_DIR = ROOT / "data" / "samples"
MODELS_DIR = ROOT / "models"
CONTOUR_COLORS = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
                  (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212)]

st.set_page_config(page_title="MedSeg · imagerie médicale", page_icon=":material/biotech:", layout="wide")


@st.cache_data(show_spinner=False)
def read_file(path: str) -> bytes:
    return Path(path).read_bytes()


# ======================================================================================
# Sidebar: pipeline configuration
# ======================================================================================
catalog = load_catalog(SAMPLES_DIR)
sample = None
loaded = None
gt_source = None

with st.sidebar:
    st.title("MedSeg")
    st.caption("Prétraitement et segmentation d'images médicales : méthodes des TPs, classiques vs deep learning.")

    st.subheader("1 · Image", divider="gray")
    mode = st.radio("Source", ["Exemple des TPs", "Importer une image"], horizontal=True, label_visibility="collapsed")
    if mode == "Exemple des TPs" and catalog:
        labels = {s.id: s.label for s in catalog}
        default_idx = next((i for i, s in enumerate(catalog) if s.id == "frame442"), 0)
        sid = st.selectbox("Image d'exemple", list(labels), index=default_idx, format_func=labels.get)
        sample = next(s for s in catalog if s.id == sid)
        loaded = load_image(read_file(str(sample.image)), name=sample.image.name)
        gt_source = read_file(str(sample.mask)) if sample.mask else None
        image_id = f"sample:{sample.id}"
    else:
        up = st.file_uploader("Image (PNG, JPG, BMP, TIFF)", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
        gt_up = st.file_uploader("Vérité terrain (masque binaire, optionnel)", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
        if up is not None:
            try:
                loaded = load_image(up.getvalue(), name=up.name)
            except ValueError as exc:
                st.error(str(exc))
        gt_source = gt_up.getvalue() if gt_up is not None else None
        image_id = f"upload:{up.name}:{up.size}" if up is not None else "none"

    use_gt = False
    if gt_source is not None:
        use_gt = st.toggle("Évaluer avec la vérité terrain", value=True)

    # Apply the per-image preset of the shared post-processing when the image changes.
    if st.session_state.get("_image_id") != image_id:
        st.session_state["_image_id"] = image_id
        ui.reset_params(SHARED_POSTPROCESSING, "shared", sample.preset if sample else {})

    st.subheader("2 · Prétraitement", divider="gray")
    pre_keys = st.multiselect(
        "Étapes, appliquées dans l'ordre de sélection", list(PREPROCESSING), default=["median"],
        format_func=lambda k: PREPROCESSING[k].label,
    )
    pre_params = {}
    for k in pre_keys:
        spec = PREPROCESSING[k]
        with st.expander(spec.label):
            st.markdown(f"{ui.origin_badge(spec.origin, spec.from_tp)} {spec.description}")
            pre_params[k] = ui.render_params(spec, "pre")

    st.subheader("3 · Segmentation conventionnelle", divider="gray")
    cl_keys = st.multiselect(
        "Méthodes", list(CLASSICAL), default=["otsu", "kmeans", "meanshift", "watershed", "chanvese"],
        format_func=lambda k: CLASSICAL[k].label,
    )
    cl_params = {}
    for k in cl_keys:
        spec = CLASSICAL[k]
        with st.expander(spec.label):
            st.markdown(f"{ui.origin_badge(spec.origin, spec.from_tp)} {spec.description}")
            cl_params[k] = ui.render_params(spec, "cls")
    with st.expander("Post-traitement commun", icon=":material/tune:"):
        st.caption("Appliqué à l'identique à toutes les méthodes classiques (TP3 ex.3 + ouverture TP2) : "
                   "sélection de la région cible → champ utile → ouverture → composantes connexes.")
        shared = {p.name: ui.render_param(p, "shared") for p in SHARED_POSTPROCESSING}

    st.subheader("4 · Deep learning", divider="gray")
    cards = discover_model_cards(MODELS_DIR)
    available = [c for c in cards if c.has_weights]
    missing = [c for c in cards if not c.has_weights]
    tf_ok = tensorflow_available()
    dl_ids: list[str] = []
    if not tf_ok:
        st.warning("TensorFlow n'est pas installé : segmentation DL désactivée.", icon=":material/warning:")
    elif not available:
        st.info("Aucun modèle avec poids. Voir l'onglet « Modèles DL ».")
    else:
        dl_ids = st.multiselect("Modèles", [c.id for c in available],
                                default=[c.id for c in available if c.selected_by_default],
                                format_func=lambda i: next(c.name for c in cards if c.id == i))
    if missing:
        st.caption("Poids absents : " + ", ".join(c.name for c in missing) + " → onglet « Modèles DL ».")
    dl_threshold = st.slider("Seuil de décision", 0.05, 0.95, 0.5, 0.05, help="TP4 : 0,5")
    dl_on_pre = st.toggle(
        "Appliquer le prétraitement à l'entrée du modèle", value=False,
        help="Désactivé par défaut : le U-Net a été entraîné sur des images brutes ; "
             "un prétraitement crée un décalage de distribution.",
    )

# ======================================================================================
# Main page
# ======================================================================================
st.title("Analyse et segmentation d'images médicales")
st.caption("Approches conventionnelles (TP2, TP3) et apprentissage profond (TP4) · Atelier imagerie médicale, SUP'COM")

if loaded is None:
    st.info("Choisissez une image d'exemple ou importez une image dans le panneau de gauche.", icon=":material/upload:")
    st.stop()

image = loaded.image
gt = None
if gt_source is not None and use_gt:
    try:
        gt = align_mask(load_mask(gt_source), image.shape[:2])
    except ValueError as exc:
        st.error(f"Vérité terrain illisible : {exc}")

# ---- computations (cached per image + parameters) ------------------------------------
errors: list[str] = []
with st.spinner("Traitement en cours…"):
    pre_img, steps = ui.cached_preprocessing(image, tuple((k, ui.freeze(pre_params[k])) for k in pre_keys))
    results: list[SegmentationResult] = []
    for k in cl_keys:
        try:
            results.append(ui.cached_classical(pre_img, k, ui.freeze(cl_params[k]), ui.freeze(shared)))
        except Exception as exc:  # noqa: BLE001 - shown to the user, others keep running
            errors.append(f"{CLASSICAL[k].label} : {exc}")
    for cid in dl_ids:
        card = next(c for c in cards if c.id == cid)
        try:
            inp = pre_img if dl_on_pre else image
            res = ui.cached_deep(inp, cid, card.weights_path.stat().st_mtime, dl_threshold, card)
            if card.is_demo and sample is not None and sample.in_demo_training:
                # never mutate a cached object: work on a copy
                res = replace(res, label=f"⚠ {res.label} (vu à l'entraînement)", info={**res.info, "in_sample": True}, warnings=[
                    *res.warnings, "Cette image a servi à entraîner le modèle de démonstration : "
                    "son score est un score d'entraînement, pas de généralisation."])
            results.append(res)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{card.name} : {exc}")
    if sample is not None and sample.tp4_prediction is not None and sample.tp4_prediction.is_file():
        results.append(SegmentationResult(
            key="tp4_saved", label="U-Net TP4 50 ép. (prédiction du notebook)", kind="Référence TP4",
            origin="TP4_IM.ipynb, cellule 37", mask=align_mask(load_mask(sample.tp4_prediction), image.shape[:2]),
            seconds=math.nan, params={},
            warnings=["Prédiction enregistrée du vrai modèle du TP4 (non recalculée) ; temps non mesuré."],
        ))
for e in errors:
    st.error(e, icon=":material/error:")

rows = metrics_table(results, gt)
mdf = ui.metrics_dataframe(rows)

# ---- headline -------------------------------------------------------------------------
h, w = image.shape[:2]
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Image ({'couleur' if image.ndim == 3 else 'niveaux de gris'})", f"{w} × {h}")
c2.metric("Méthodes comparées", len(results))
fair = [r.label for r in results if not r.info.get("in_sample")]
if gt is not None and fair:
    best = mdf.loc[fair, "Dice"].idxmax()
    c3.metric(f"Meilleur Dice · {best}", f"{mdf.loc[best, 'Dice']:.3f}",
              help="Les résultats marqués ⚠ (image vue à l'entraînement) sont exclus.")
    c4.metric(f"Vérité terrain · {100 * gt.mean():.2f} % de l'image", f"{int(gt.sum())} px")
else:
    c3.metric("Meilleur Dice", "-", help="Pas de vérité terrain")
    c4.metric("Prétraitement", " → ".join(PREPROCESSING[k].label for k in pre_keys) or "aucun")

tabs = st.tabs(["Image", "Prétraitement", "Segmentations", "Comparaison", "Export", "Modèles DL", "Méthodes & TPs"])

# ---- Image --------------------------------------------------------------------------
with tabs[0]:
    left, right = st.columns([3, 2], gap="large")
    with left:
        show_gt = gt is not None and st.toggle("Superposer la vérité terrain", value=True)
        disp = viz.overlay_mask(image, gt, viz.PALETTE["gt"], 0.35) if show_gt else image
        st.image(disp, width="stretch", caption=loaded.name)
    with right:
        if sample is not None:
            st.markdown(f"**Modalité** · {sample.modality}")
            st.markdown(f"**Provenance** · {sample.source}")
            st.info(sample.notes, icon=":material/info:")
        meta = describe_image(loaded)
        st.dataframe(pd.DataFrame({"Valeur": [str(v) for v in meta.values()]}, index=list(meta.keys())),
                     width="stretch")
        st.caption("Histogramme des intensités (TP1)")
        st.altair_chart(ui.histogram_chart(histogram(image)), width="stretch")

# ---- Preprocessing ---------------------------------------------------------------------
with tabs[1]:
    if not steps:
        st.info("Aucun prétraitement sélectionné : les méthodes travaillent sur l'image brute.")
    else:
        cols = st.columns(min(len(steps) + 1, 4))
        cols[0].image(image, caption="Originale", width="stretch")
        for i, s in enumerate(steps, start=1):
            cols[i % len(cols)].image(s.image, caption=f"{i}. {s.label} · {ui.fmt_seconds(s.seconds)}", width="stretch")
        st.markdown("##### Effet mesuré par rapport à l'image originale (mesures de similarité du TP5)")
        sim_rows = []
        for i, s in enumerate(steps, start=1):
            sim = compare_images(image, s.image)
            sim_rows.append({"Étape": f"{i}. {s.label}", **sim, "Temps (s)": s.seconds,
                             "Paramètres": ", ".join(f"{k}={v}" for k, v in s.params.items())})
        st.dataframe(pd.DataFrame(sim_rows).set_index("Étape").style.format(
            {"Corrélation": "{:.4f}", "EQM": "{:.1f}", "PSNR (dB)": "{:.2f}", "IMN": "{:.4f}", "Temps (s)": "{:.4f}"}),
            width="stretch")
        st.caption("Corrélation / IMN élevées et EQM faible = image peu modifiée. Pour évaluer un débruitage, "
                   "ajoutez d'abord l'étape « Bruit gaussien (test) » puis un filtre, et comparez les lignes.")

# ---- Segmentations ---------------------------------------------------------------------
with tabs[2]:
    if not results:
        st.info("Sélectionnez au moins une méthode dans le panneau de gauche.")
    else:
        views = ["Superposition", "Masque binaire", "Régions avant post-traitement"]
        if gt is not None:
            views.insert(1, "Erreurs (VP / FP / FN)")
        cc1, cc2 = st.columns([2, 1])
        view = cc1.segmented_control("Affichage", views, default="Superposition", key="seg_view") or "Superposition"
        alpha = cc2.slider("Transparence du masque", 0.0, 1.0, 0.45, 0.05)
        if view.startswith("Erreurs"):
            st.caption(":green[■] vrais positifs · :red[■] faux positifs · :blue[■] faux négatifs")
        grid = st.columns(3)
        panels: list[tuple[str, np.ndarray, str]] = []
        if gt is not None:
            panels.append(("Vérité terrain", viz.overlay_mask(image, gt, viz.PALETTE["gt"], alpha), "référence"))
        for r in results:
            if view == "Masque binaire":
                pic = viz.mask_to_png_array(r.mask)
            elif view.startswith("Erreurs"):
                pic = viz.error_map(image, r.mask, gt, alpha)
            elif view.startswith("Régions"):
                if r.labels is not None:
                    pic = viz.recolor_labels(r.labels, pre_img)
                    pic[viz.label_edges(r.labels)] = (255, 0, 0)
                elif r.probability is not None:
                    pic = viz.probability_heatmap(r.probability)
                else:
                    pic = viz.mask_to_png_array(r.mask)
            else:
                pic = viz.overlay_mask(image, r.mask, viz.PALETTE["mask"], alpha)
            cap = r.origin if math.isnan(r.seconds) else f"{r.origin} · {ui.fmt_seconds(r.seconds)}"
            if gt is not None:
                cap += f" · Dice {dice(r.mask, gt):.3f}"
            panels.append((r.label, pic, cap))
        for i, (title, pic, cap) in enumerate(panels):
            with grid[i % 3]:
                st.markdown(f"**{title}**")
                st.image(pic, caption=cap, width="stretch")
        for r in results:
            for n in r.warnings:
                st.warning(f"**{r.label}** — {n}", icon=":material/warning:")
        if view.startswith("Régions"):
            st.caption("Méthodes classiques : régions recolorées par leur couleur moyenne et contours (TP3) ; "
                       "deep learning : carte de probabilité.")

# ---- Comparison ------------------------------------------------------------------------
with tabs[3]:
    if not results:
        st.info("Aucun résultat à comparer.")
    else:
        if gt is None:
            st.info("Pas de vérité terrain : seules l'aire et le temps sont comparés, ainsi que l'accord "
                    "entre méthodes (comme au TP3, EX3). Importez un masque pour obtenir Dice, IoU, précision, "
                    "rappel et distances de Hausdorff.", icon=":material/info:")
        st.dataframe(ui.style_metrics(mdf), width="stretch")
        if gt is not None:
            a, b = st.columns(2, gap="large")
            a.altair_chart(ui.bar_chart(mdf, "Dice", "Dice (plus haut = meilleur)"), width="stretch")
            if mdf["HD95 (px)"].notna().any():
                b.altair_chart(ui.bar_chart(mdf, "HD95 (px)", "HD95 en pixels (plus bas = meilleur)", ".1f",
                                            ascending=True), width="stretch")
        timed_df = mdf[mdf["Temps (s)"].notna()] if "Temps (s)" in mdf else mdf
        if len(timed_df):
            st.altair_chart(ui.bar_chart(timed_df, "Temps (s)", "Temps d'exécution en secondes (plus bas = plus rapide)",
                                         ".2f", ascending=True), width="stretch")
        st.caption("Temps mesurés avec time.perf_counter ; méthodes classiques : méthode + post-traitement ; "
                   "DL : redimensionnement + inférence CPU. Résultats mis en cache : le temps affiché est celui "
                   "du premier calcul.")
        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("##### Contours superposés (TP2, ex.5)")
            sel = {r.label: (r.mask, CONTOUR_COLORS[i % len(CONTOUR_COLORS)]) for i, r in enumerate(results)}
            if gt is not None:
                sel = {"Vérité terrain": (gt, (255, 255, 255)), **sel}
            st.image(viz.overlay_contours(image, sel), width="stretch")
            st.markdown(" ".join(
                f"<span style='color:rgb{c};font-size:1.2em'>■</span> {name}&nbsp;&nbsp;" for name, (_, c) in sel.items()
            ), unsafe_allow_html=True)
        with right:
            st.markdown("##### Accord entre méthodes (Dice deux à deux)")
            names = [r.label for r in results]
            agree = pd.DataFrame([[dice(a.mask, b.mask) for b in results] for a in results], index=names, columns=names)
            st.dataframe(agree.style.format("{:.2f}").background_gradient(cmap="Greens", vmin=0, vmax=1), width="stretch")
            st.caption("Un accord élevé n'implique pas l'exactitude (TP3, discussion) : deux méthodes peuvent "
                       "se tromper de la même façon.")

# ---- Export ------------------------------------------------------------------------------
with tabs[4]:
    if not results:
        st.info("Aucun résultat à exporter.")
    else:
        config = {
            "preprocessing": [{"step": s.key, "params": s.params} for s in steps],
            "shared_postprocessing": shared,
            "deep_learning": {"threshold": dl_threshold, "on_preprocessed_input": dl_on_pre},
            "ground_truth": gt is not None,
        }
        zip_bytes = build_results_zip(loaded.name, results, rows, config, pre_img if steps else None)
        st.download_button("Télécharger tous les résultats (.zip)", zip_bytes, file_name=f"medseg_{Path(loaded.name).stem}.zip",
                           mime="application/zip", type="primary", icon=":material/download:")
        st.caption("Contenu : un masque PNG (0/255) par méthode, metrics.csv, config.json (tous les paramètres), "
                   "image prétraitée.")
        st.download_button("Tableau des métriques (.csv)", rows_to_csv(rows), file_name="metrics.csv", mime="text/csv",
                           icon=":material/table:")
        st.markdown("##### Masques individuels")
        cols = st.columns(4)
        for i, r in enumerate(results):
            cols[i % 4].download_button(r.label, png_bytes(viz.mask_to_png_array(r.mask)),
                                        file_name=f"{Path(loaded.name).stem}__{r.key.replace(':', '_')}.png",
                                        mime="image/png", key=f"dl_mask_{r.key}", width="stretch")

# ---- Deep models -----------------------------------------------------------------------
with tabs[5]:
    st.markdown("Chaque modèle est décrit par une fiche (`model_card.json`) : **la modalité et les données "
                "d'entraînement déterminent les images sur lesquelles son résultat a un sens.**")
    if not tf_ok:
        st.warning("TensorFlow n'est pas installé (`pip install tensorflow`).")
    for card in cards:
        status = "poids disponibles" if card.has_weights else "poids absents"
        with st.expander(f"{card.name} — {status}", expanded=not card.has_weights or card.is_demo,
                         icon=":material/neurology:"):
            if card.is_demo:
                st.warning("Modèle de démonstration.", icon=":material/science:")
            a, b = st.columns(2)
            a.markdown(f"**Modalité d'entraînement** · {card.modality}")
            a.markdown(f"**Structure segmentée** · {card.target}")
            a.markdown(f"**Données** · {card.training_data}")
            a.markdown(f"**Origine** · {card.origin} · {card.framework}")
            a.markdown(f"**Entrée** · {card.input_size[1]}×{card.input_size[0]}×{card.channels}, {card.normalization}")
            b.markdown("**Entraînement**")
            b.json(card.training, expanded=False)
            b.markdown("**Scores rapportés**")
            b.json(card.reported_metrics, expanded=False)
            if card.limitations:
                st.markdown("**Limites**\n" + "\n".join(f"- {x}" for x in card.limitations))
            if not card.has_weights:
                st.info(f"Fichier attendu : `models/{card.folder.name}/{card.weights}`.")
                up = st.file_uploader(f"Importer les poids ({card.weights})", type=["hdf5", "h5", "keras"],
                                      key=f"weights_{card.id}")
                if up is not None and st.button("Enregistrer", key=f"save_{card.id}", type="primary"):
                    card.weights_path.write_bytes(up.getvalue())
                    st.cache_resource.clear()
                    st.rerun()

# ---- Methods & TPs ---------------------------------------------------------------------
with tabs[6]:
    st.markdown("##### Méthodes intégrées et TP d'origine")
    st.dataframe(pd.DataFrame(methods_table()), width="stretch", hide_index=True)
    st.markdown("""
##### Corrections apportées aux TPs
- **TP2 · watershed** : les marqueurs étaient dérivés de la vérité terrain (y compris la variante « gradient »,
  restreinte par le masque GT dilaté, variable `image_glottis` jamais définie). Ici : marqueurs automatiques (Otsu).
- **TP2 · Otsu** : le rapport juge le seuil d'Otsu « proche » du seuil manuel (28) ; le notebook affiche **116**.
  Les deux seuils ne séparent pas les mêmes classes ; le seuil 28 sert ici à délimiter le champ utile.
- **TP2 · morphologie** : les filtres `rank` étaient appliqués au gris seulement ; ici canal par canal.
- **TP2 · bruit** : tirage sans graine (non reproductible) ; ici graine fixée.
- **TP3** : filtre médian appliqué au seul Mean-Shift (comparaison biaisée) ; ici même prétraitement pour toutes
  les méthodes. Canaux RGB redondants sur une IRM : réduits au gris. `time.time` remplacé par `perf_counter`.
- **TP3 · masque** : la règle « ne touche pas le bord » supprimait la glotte de frame442 (reliée au vignettage) ;
  restriction au champ utile + sélection de la région la plus sombre *dans* ce champ.
- **TP4** : split aléatoire de 100 frames consécutives d'une même vidéo → images de test quasi identiques aux images
  d'entraînement (Dice de test optimiste) ; perte BCE seule → effondrement à 10 époques. Le script
  `scripts/train_unet_tp4.py` propose un split temporel, une perte BCE + Dice et de l'augmentation.
- **Métriques** : divisions par zéro (TP3) gérées ; l'accuracy n'est pas affichée (99,5 % pour un modèle qui ne
  prédit rien, TP4).
""")
