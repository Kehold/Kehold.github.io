import pandas as pd
import streamlit as st

from prodintel import config as C
from prodintel.plots import fig_ensemble, fig_eur, fig_hindcast
from prodintel.ui import header, kpi, qc_badges, require_run

header("Agent 5 · Ensemble and uncertainty",
       "Hindcast-weighted blend of DCA, analog type curve and Transformer · Monte Carlo EUR")
ctx = require_run()
qc_badges(ctx, "Ensemble")
e = ctx.results["ensemble"]
st.caption(f"Weighting basis: {e['basis']}")

c = st.columns(len(e["weights"]) + 1)
for col, (m, w) in zip(c, e["weights"].items()):
    kpi(col, f"{m} weight", f"{w:.0%}")
kpi(c[-1], "Oil P10 / P90 ratio", f"{e['eur']['oil']['P10'] / max(e['eur']['oil']['P90'], 1):.2f}")

stream = st.radio("Stream", C.STREAMS, horizontal=True, format_func=str.title)
yrs = st.slider("Years shown after today", 2, 30, 10)
st.plotly_chart(fig_ensemble(ctx, stream, months=yrs * 12), width="stretch")

left, right = st.columns(2)
with left:
    st.markdown("**EUR distribution**")
    st.plotly_chart(fig_eur(ctx, stream), width="stretch")
    rows = []
    for s in C.STREAMS:
        v = e["eur"][s]
        rows.append(dict(stream=s.title(), unit=C.STREAM_VOL_UNITS[s], cum_to_date=v["cum_to_date"] / 1e3,
                         P90=v["P90"] / 1e3, P50=v["P50"] / 1e3, P10=v["P10"] / 1e3))
    st.dataframe(pd.DataFrame(rows).round(0), hide_index=True, width="stretch")
    st.caption("P90 is the low case and P10 the high case (petroleum convention).")
with right:
    if e["hindcast"] is not None:
        st.markdown("**Blind 12-month hindcast**")
        st.plotly_chart(fig_hindcast(ctx), width="stretch")
        st.dataframe(e["hindcast"].assign(weight=lambda d: d.method.map(e["weights"])).round(3),
                     hide_index=True, width="stretch")
    else:
        st.markdown("**Member oil EUR (P50)**")
        st.dataframe(pd.DataFrame([dict(member=k, eur_mbbl=v / 1e3, weight=e["weights"][k])
                                   for k, v in e["member_eur_oil"].items()]).round(2),
                     hide_index=True, width="stretch")
        st.info("No history to hindcast, so weights are priors. Treat the band as indicative.")
