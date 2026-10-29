import pandas as pd
import streamlit as st

from prodintel import config as C
from prodintel.agents.decline import describe
from prodintel.plots import fig_dca
from prodintel.ui import header, qc_badges, require_run

header("Agent 3 · Decline curve analysis",
       "Modified hyperbolic Arps, robust log-space fit, 8 %/yr terminal decline, bootstrap uncertainty")
ctx = require_run()
qc_badges(ctx, "Decline")
basis = ctx.results["tool_outputs"].get("Decline", ctx.results["tool_outputs"].get("fit_decline_curves", {}))
st.caption(f"Basis: {basis.get('basis', '')}")

rows = []
for s in C.STREAMS:
    rows.append(dict(stream=s.title(), unit=C.STREAM_UNITS[s], **describe(ctx.results["dca"][s]["params"])))
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

stream = st.radio("Stream", C.STREAMS, horizontal=True, format_func=str.title)
st.plotly_chart(fig_dca(ctx, stream), width="stretch")

bp = ctx.results["dca"][stream]["boot_params"]
with st.expander("Bootstrap parameter spread"):
    df = pd.DataFrame(bp, columns=["qi", "Di (nominal / month)", "b"])
    st.dataframe(df.describe(percentiles=[0.1, 0.5, 0.9]).T.round(4), width="stretch")
    st.caption("Residual bootstrap captures fit uncertainty only. Reservoir and operational uncertainty "
               "enter through the analog and Transformer members of the ensemble.")
