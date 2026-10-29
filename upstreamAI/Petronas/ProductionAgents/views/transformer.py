import streamlit as st

from prodintel import config as C
from prodintel.plots import fig_loss, fig_transformer
from prodintel.ui import header, kpi, qc_badges, require_run

header("Agent 4 · Transformer forecasting",
       "Encoder-only quantile Transformer · 24-month context · 120-month horizon · "
       "five pad-grouped folds, so the target pad is never in training")
ctx = require_run()
qc_badges(ctx, "Transformer")
tf = ctx.results.get("transformer")
if not tf:
    st.warning("The Transformer agent did not run (PyTorch unavailable). The ensemble used the other members.")
    st.stop()
b = tf["bundle"]
c = st.columns(4)
kpi(c[0], "Training samples", f"{b['n_train_samples']:,}")
kpi(c[1], "Model", "fold average" if b.get("kind") == "folds" else "out-of-fold")
kpi(c[2], "Held-out median APE", f"{b['val_median_ape']:.0%}")
kpi(c[3], "Interval rescale", f"×{b['interval_scale']:.2f}")

stream = st.radio("Stream", C.STREAMS, horizontal=True, format_func=str.title)
st.plotly_chart(fig_transformer(ctx, stream), width="stretch")

left, right = st.columns([3, 2])
with left:
    st.markdown("**Training curve**")
    st.plotly_chart(fig_loss(b), width="stretch")
with right:
    st.markdown("**How it works**")
    st.markdown(
        "- Each training sample cuts a well's history at month *k* (including *k* = 0) and asks for the "
        "next 120 months of oil, gas and water.\n"
        "- Tokens carry normalised log rate per 1,000 ft; a static token carries zone, design, vintage, TVD "
        "and location.\n"
        "- Three quantile heads (P90, P50, P10) are trained with pinball loss and kept ordered.\n"
        f"- Raw held-out coverage of the P90 to P10 band was {b['raw_coverage']:.0%}; widths are rescaled to "
        "80 % on the held-out pads.\n"
        "- Beyond 120 months the forecast is extended at the 8 %/yr terminal decline.")
