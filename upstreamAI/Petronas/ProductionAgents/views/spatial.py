import pandas as pd
import streamlit as st

from prodintel import config as C
from prodintel.plots import PROXY_NOTE, fig_3d, fig_history, fig_map
from prodintel.ui import header, kpi, qc_badges, require_run

header("Agent 1 · Spatial, formation and geometry",
       "Resolves the input, assigns the landing zone, estimates TVD and gathers offsets")
ctx = require_run()
t = ctx.results["target"]
qc_badges(ctx, "Spatial")

c = st.columns(5)
kpi(c[0], "Landing zone", C.ZONE_NAMES[t["zone"]])
kpi(c[1], "Proxy TVD", f"{t['tvd']:,.0f} ft")
kpi(c[2], "Lateral", f"{t['lateral_ft']:,.0f} ft")
kpi(c[3], "Offsets in 3 mi", f"{len(ctx.results['offsets'])}")
kpi(c[4], "Same-zone offsets", f"{(ctx.results['offsets'].zone == t['zone']).sum()}")

tab3d, tabmap, tabhist = st.tabs(["3D well geometry", "Map", "Production history"])
with tab3d:
    r = st.slider("Show wells within (mi)", 1.0, 6.0, 3.0, 0.5)
    st.plotly_chart(fig_3d(ctx, radius_mi=r), width="stretch")
    st.caption(PROXY_NOTE)
with tabmap:
    st.plotly_chart(fig_map(ctx), width="stretch")
with tabhist:
    hist = ctx.results.get("history")
    if hist is None:
        st.info("Proposed location: no production history. The forecast is driven by analogs and design.")
        if t.get("zone_table"):
            st.markdown(f"**Zone selection** · {t['zone_reason']}")
            st.dataframe(pd.DataFrame(t["zone_table"]).rename(columns={
                "cum12_bbl_per_ft": "median 12-mo oil (bbl/ft)"}), hide_index=True)
    else:
        h = t["history"]
        c = st.columns(6)
        kpi(c[0], "Months on", h["months_on"])
        kpi(c[1], "Cum oil", f"{h['cum_oil_mbbl']:,.0f} Mbbl")
        kpi(c[2], "Cum gas", f"{h['cum_gas_mmcf']:,.0f} MMcf")
        kpi(c[3], "Peak oil", f"{h['peak_oil_bopd']:,} bopd")
        kpi(c[4], "GOR now", f"{h['gor_mcf_per_bbl']:.2f}")
        kpi(c[5], "Water cut now", f"{h['water_cut']:.0%}")
        st.plotly_chart(fig_history(hist, t["first_prod"]), width="stretch")
        st.caption(f"{h['suspected_downtime_months']} months flagged as suspected downtime or frac-hit interference.")

st.markdown("**Offset wells**")
st.dataframe(ctx.results["offsets"][["api", "well_name", "operator", "zone", "dist_mi", "tvd", "lateral_ft",
                                     "proppant_lbft", "spacing_ft", "first_prod", "months_on", "cum_oil_mbbl"]]
             .round(2), hide_index=True, width="stretch")
