"""Streamlit helpers shared by every page."""
from __future__ import annotations

import streamlit as st

from . import config as C
from .field import generate_field

CSS = f"""
<style>
h1, h2, h3 {{ font-family: Cambria, Georgia, serif; color: {C.NAVY}; }}
@media (prefers-color-scheme: dark) {{ h1, h2, h3 {{ color: #E6F1F2; }} }}
.kpi {{ border-left: 4px solid {C.TEAL}; padding: 6px 12px; background: rgba(20,128,140,0.06);
        border-radius: 4px; margin-bottom: 6px; }}
.kpi .v {{ font-size: 1.45rem; font-weight: 700; }}
.kpi .l {{ font-size: 0.78rem; opacity: 0.75; text-transform: uppercase; letter-spacing: .04em; }}
.agent {{ border: 1px solid rgba(10,46,54,0.15); border-radius: 6px; padding: 8px 10px; margin: 4px 0;
          border-left: 4px solid {C.AMBER}; }}
.agent b {{ color: {C.TEAL}; }}
.note {{ font-size: 0.8rem; opacity: 0.75; }}
</style>
"""


def header(title: str, subtitle: str | None = None):
    st.markdown(CSS, unsafe_allow_html=True)
    st.title(title)
    if subtitle:
        st.caption(subtitle)


@st.cache_data(show_spinner="Building the synthetic Permian field")
def load_field(seed: int = 7):
    wells, prod, truth, surf = generate_field(seed=seed)
    return wells, prod, truth, surf


@st.cache_resource(show_spinner="Training the Transformer agent: five pad-grouped folds, first run only")
def get_bundle(_wells, _prod, seed: int = 7):
    import os
    from .agents.transformer import TORCH_OK, train_folds
    if not TORCH_OK:
        return None
    cache = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".model_cache")
    return train_folds(_wells, _prod, seed=seed, cache_dir=cache)


def kpi(col, label: str, value: str):
    col.markdown(f"<div class='kpi'><div class='l'>{label}</div><div class='v'>{value}</div></div>",
                 unsafe_allow_html=True)


def require_run():
    ctx = st.session_state.get("ctx")
    if ctx is None or "ensemble" not in ctx.results:
        st.info("Run an analysis from the **Run analysis** page first.")
        st.stop()
    return ctx


def qc_badges(ctx, agent: str | None = None):
    for q in ctx.qc:
        if agent and q["agent"] != agent:
            continue
        fn = {"warn": st.warning, "stop": st.error}.get(q["level"], st.info)
        fn(f"**{q['agent']}** · {q['message']}")
