import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from prodintel import config as C
from prodintel.field import arps_modified
from prodintel.ui import header, load_field

header("Field data and method", "Seeded synthetic Midland Basin field used by every agent")
wells, prod, truth, surf = load_field()

c = st.columns(4)
c[0].metric("Wells", len(wells))
c[1].metric("Pads", wells.pad_id.nunique())
c[2].metric("Monthly records", f"{len(prod):,}")
c[3].metric("Data through", C.DATA_END)

st.dataframe(wells[["api", "well_name", "operator", "pad_id", "zone", "tvd", "lateral_ft", "proppant_lbft",
                    "fluid_bblft", "spacing_ft", "first_prod", "months_on", "cum_oil_mbbl", "surf_lat", "surf_lon"]]
             .round(4), hide_index=True, width="stretch", height=320)

left, right = st.columns(2)
with left:
    fig = px.box(wells, x="zone", y="cum_oil_mbbl", color="zone", points="all",
                 color_discrete_sequence=[C.TEAL, C.AMBER, C.NAVY], title="Cumulative oil by zone (Mbbl)")
    st.plotly_chart(fig, width="stretch")
with right:
    fig = px.scatter(wells, x="proppant_lbft", y=wells.cum_oil_mbbl * 1e3 / wells.lateral_ft / wells.months_on.clip(upper=60),
                     color="zone", color_discrete_sequence=[C.TEAL, C.AMBER, C.NAVY],
                     labels={"y": "cum oil per ft per month on (capped at 60)"},
                     title="Proppant intensity against normalised productivity")
    st.plotly_chart(fig, width="stretch")

ctx = st.session_state.get("ctx")
if ctx is not None and ctx.target.mode == "existing" and "ensemble" in ctx.results:
    st.subheader("Truth check for the current target")
    tr = truth.set_index("api").loc[ctx.results["target"]["api"]]
    m = np.arange(1, C.EUR_YEARS * 12 + 1)
    q = arps_modified(np.maximum(m - 2, 0), tr.qi_oil, tr.di_oil, tr.b_oil)
    q[0] = 0.58 * tr.qi_oil
    life = np.cumprod(q >= C.ECON_LIMIT_BOPD).astype(bool)
    true_eur = float((q * life * 30.4375).sum()) / 1e3
    e = ctx.results["ensemble"]["eur"]["oil"]
    st.write(f"Because the field is synthetic, the noise-free generating curve is known. Its oil EUR is "
             f"**{true_eur:,.0f} Mbbl** (before frac hits and downtime). The ensemble gives "
             f"{e['P90'] / 1e3:,.0f} / {e['P50'] / 1e3:,.0f} / {e['P10'] / 1e3:,.0f} Mbbl. "
             f"Generating parameters: qi {tr.qi_oil:,.0f} bopd, b {tr.b_oil:.2f}, "
             f"child factor {tr.child_factor:.2f}, frac hits received by offsets {int(tr.frac_hits)}.")

st.subheader("Method")
st.markdown("""
**Generator.** Pads of three to six laterals in the Lower Spraberry Shale, Wolfcamp A and Wolfcamp B. Peak rate
scales with lateral length, a smooth spatial rock-quality field per zone, proppant intensity, spacing and vintage.
Child wells drilled at least 18 months after a close, overlapping same-zone parent take a 20 % penalty, and the
parent sees a frac-hit dip and a small permanent step down. Decline is modified hyperbolic. GOR rises and water-oil
ratio grows with time. Downtime and correlated measurement noise are added.

**Honesty rules.** API numbers use non-existent county codes 901 and 903. Trajectories are proxies, not surveys.
The Transformer never sees the target well, and is validated on whole held-out pads. Ensemble weights come from a
blind hindcast when history allows and are declared as priors otherwise. Every agent writes QC flags that travel
into the report.
""")
