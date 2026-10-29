import streamlit as st

from prodintel import config as C
from prodintel.agents.analogs import FEATURES
from prodintel.plots import fig_analog_curves, fig_contrib
from prodintel.ui import header, kpi, qc_badges, require_run

header("Agent 2 · Analog selection",
       "Weighted Gower-style dissimilarity over location, completion, spacing, vintage and depth")
ctx = require_run()
qc_badges(ctx, "Analog")
an = ctx.results["analogs"]
c = st.columns(4)
kpi(c[0], "Analogs", len(an))
kpi(c[1], "Median similarity", f"{an.similarity.median():.2f}")
kpi(c[2], "Median distance", f"{an.dist_mi.median():.1f} mi")
kpi(c[3], "Median 12-mo oil", f"{an.cum12_bbl_per_ft.median():.1f} bbl/ft")

stream = st.radio("Stream", C.STREAMS, horizontal=True, format_func=str.title)
left, right = st.columns([3, 2])
with left:
    st.markdown("**Normalised type curves**")
    st.plotly_chart(fig_analog_curves(ctx, stream), width="stretch")
with right:
    st.markdown("**Why each analog ranked where it did**")
    st.plotly_chart(fig_contrib(ctx), width="stretch")
    st.caption("Feature weights: " + " · ".join(f"{k} {v[2]:.2f}" for k, v in FEATURES.items()))

st.dataframe(an[["api", "well_name", "zone", "dist_mi", "lateral_ft", "proppant_lbft", "fluid_bblft", "spacing_ft",
                 "first_prod", "months_on", "similarity", "cum12_bbl_per_ft", "peak_bopd_per_kft"]].round(3),
             hide_index=True, width="stretch")
