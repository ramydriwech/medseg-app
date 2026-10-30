"""Streamlit building blocks: widgets generated from the method registry, cached wrappers
around the (pure) processing functions, and small display helpers. No image processing
is implemented here - everything is delegated to the ``medseg`` package."""

from __future__ import annotations

import math
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from medseg.pipeline import run_classical, run_deep, run_preprocessing
from medseg.registry import MethodSpec, ParamSpec
from medseg.segmentation.deep import KerasSegmenter, ModelCard

CHOICE_LABELS = {
    "darkest": "Région la plus sombre",
    "brightest": "Région la plus claire",
    "largest": "La plus grande",
    "all": "Toutes",
}


# --------------------------------------------------------------------------------------
# Widgets generated from the registry
# --------------------------------------------------------------------------------------
def widget_key(prefix: str, name: str) -> str:
    return f"{prefix}::{name}"


def render_param(p: ParamSpec, prefix: str) -> Any:
    key = widget_key(prefix, p.name)
    if key not in st.session_state:
        st.session_state[key] = p.default
    if p.kind == "bool":
        return st.checkbox(p.label, key=key, help=p.help or None)
    if p.kind == "choice":
        return st.selectbox(p.label, p.choices, key=key, help=p.help or None,
                            format_func=lambda v: CHOICE_LABELS.get(v, str(v)))
    if p.kind == "int":
        return st.slider(p.label, int(p.min), int(p.max), step=int(p.step or 1), key=key, help=p.help or None)
    return st.slider(p.label, float(p.min), float(p.max), step=float(p.step or 0.1), key=key, help=p.help or None)


def render_params(spec: MethodSpec, prefix: str) -> dict[str, Any]:
    if not spec.params:
        st.caption("Aucun paramètre.")
        return {}
    return {p.name: render_param(p, f"{prefix}:{spec.key}") for p in spec.params}


def reset_params(params: tuple[ParamSpec, ...], prefix: str, overrides: dict[str, Any] | None = None) -> None:
    overrides = overrides or {}
    for p in params:
        st.session_state[widget_key(prefix, p.name)] = overrides.get(p.name, p.default)


def origin_badge(spec_origin: str, from_tp: bool) -> str:
    return f":green-badge[{spec_origin}]" if from_tp else f":orange-badge[{spec_origin}]"


# --------------------------------------------------------------------------------------
# Cached computations (the cache key includes the image and every parameter)
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=64)
def cached_preprocessing(image: np.ndarray, steps: tuple[tuple[str, tuple], ...]):
    return run_preprocessing(image, [(k, dict(p)) for k, p in steps])


@st.cache_data(show_spinner=False, max_entries=256)
def cached_classical(image: np.ndarray, key: str, params: tuple, shared: tuple):
    return run_classical(image, key, dict(params), dict(shared))


@st.cache_resource(show_spinner=False)
def load_segmenter(card_id: str, weights_mtime: float, _card: ModelCard) -> KerasSegmenter:
    return KerasSegmenter(_card)


@st.cache_data(show_spinner=False, max_entries=64)
def cached_deep(image: np.ndarray, card_id: str, weights_mtime: float, threshold: float, _card: ModelCard):
    return run_deep(image, load_segmenter(card_id, weights_mtime, _card), threshold)


def freeze(d: dict[str, Any]) -> tuple:
    return tuple(sorted(d.items()))


# --------------------------------------------------------------------------------------
# Display helpers
# --------------------------------------------------------------------------------------
def fmt_seconds(s: float) -> str:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return "-"
    return f"{s * 1000:.0f} ms" if s < 1 else f"{s:.2f} s"


def metrics_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.set_index("Méthode")


def style_metrics(df: pd.DataFrame):
    higher = [c for c in ("Dice", "IoU", "Précision", "Rappel") if c in df]
    lower = [c for c in ("HD95 (px)", "Hausdorff (px)") if c in df]
    fmt = {c: "{:.3f}" for c in higher}
    fmt.update({c: "{:.1f}" for c in lower})
    fmt["Temps (s)"] = "{:.3f}"
    styler = df.style.format(fmt, na_rep="-")
    if higher:
        styler = styler.highlight_max(subset=higher, props="font-weight:600; background-color:rgba(46,160,110,.18)")
    if lower:
        styler = styler.highlight_min(subset=lower, props="font-weight:600; background-color:rgba(46,160,110,.18)")
    return styler


TYPE_COLORS = alt.Scale(
    domain=["Classique", "Deep learning", "Référence TP4"],
    range=["#4C78A8", "#E45756", "#B279A2"],
)


def bar_chart(df: pd.DataFrame, column: str, title: str, fmt: str = ".3f", ascending: bool = False) -> alt.Chart:
    """Horizontal bar chart, one bar per method, sorted best first, value labels at the end."""
    data = df.reset_index()[["Méthode", "Type", column]].dropna()
    sort = "x" if ascending else "-x"
    base = alt.Chart(data).encode(
        y=alt.Y("Méthode:N", sort=sort, title=None, axis=alt.Axis(labelLimit=220)),
        x=alt.X(f"{column}:Q", title=title),
    )
    bars = base.mark_bar(cornerRadiusEnd=3).encode(
        color=alt.Color("Type:N", scale=TYPE_COLORS, legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["Méthode", "Type", alt.Tooltip(f"{column}:Q", format=fmt)],
    )
    labels = base.mark_text(align="left", dx=4, fontSize=11).encode(text=alt.Text(f"{column}:Q", format=fmt))
    return (bars + labels).properties(height=alt.Step(30))


def histogram_chart(hist: dict[str, np.ndarray]) -> alt.Chart:
    colors = {"gris": "#555555", "R": "#E45756", "G": "#54A24B", "B": "#4C78A8"}
    frames = [pd.DataFrame({"Intensité": np.arange(256), "Pixels": h, "Canal": name}) for name, h in hist.items()]
    data = pd.concat(frames)
    return (
        alt.Chart(data)
        .mark_line(strokeWidth=1.4)
        .encode(
            x=alt.X("Intensité:Q", scale=alt.Scale(domain=[0, 255])),
            y=alt.Y("Pixels:Q"),
            color=alt.Color("Canal:N", scale=alt.Scale(domain=list(hist), range=[colors[k] for k in hist]),
                            legend=alt.Legend(orient="top", title=None)),
        )
        .properties(height=220)
    )
