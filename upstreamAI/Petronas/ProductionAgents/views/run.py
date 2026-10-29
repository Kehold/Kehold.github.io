import os

import streamlit as st

from prodintel import config as C
from prodintel.agents.spatial import normalise_api
from prodintel.agents.transformer import TORCH_OK
from prodintel.context import ProposedDesign, RunContext, Target
from prodintel.orchestrator import run_deterministic, run_llm
from prodintel.ui import get_bundle, header, kpi, load_field

header(C.APP_NAME, C.APP_TAGLINE + " · synthetic demonstration field")
wells, prod, truth, surf = load_field()


def _secret_key():
    try:
        return st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        return None


with st.sidebar:
    st.subheader("Input")
    mode = st.radio("Target", ["Producing well (API)", "Proposed location (lat/lon)"], horizontal=False)
    if mode.startswith("Producing"):
        labels = {f"{r.api} · {r.zone} · {r.months_on} mo": r.api for r in wells.sort_values("api").itertuples()}
        pick = st.selectbox("API number", list(labels), index=20)
        typed = st.text_input("or type an API", placeholder="42-903-40143")
        target = Target(api=normalise_api(typed) if typed.strip() else labels[pick])
    else:
        lat = st.number_input("Latitude", value=31.86000, format="%.5f", step=0.002)
        lon = st.number_input("Longitude", value=-102.13000, format="%.5f", step=0.002)
        zone = st.selectbox("Landing zone", ["auto"] + C.ZONES,
                            format_func=lambda z: "Let the Spatial agent choose" if z == "auto" else C.ZONE_NAMES[z])
        with st.expander("Proposed design", expanded=True):
            ll = st.slider("Lateral length (ft)", 5_000, 15_000, 10_000, 250)
            prop = st.slider("Proppant (lb/ft)", 1_200, 3_200, 2_400, 50)
            fluid = st.slider("Fluid (bbl/ft)", 25, 70, 48)
            spacing = st.select_slider("Spacing (ft)", [660, 880, 1056, 1320], 880)
            az = st.radio("Azimuth", [0.0, 180.0], format_func=lambda a: "North" if a == 0 else "South",
                          horizontal=True)
        target = Target(lat=lat, lon=lon, zone=zone,
                        design=ProposedDesign(ll, prop, fluid, spacing, az))

    st.subheader("Orchestrator")
    key = _secret_key() or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        key = st.text_input("Anthropic API key (optional)", type="password",
                            help="Without a key the deterministic orchestrator runs the same agents.")
    orch = st.radio("Run with", ["Claude tool-calling agent", "Deterministic rules"],
                    index=0 if key else 1, disabled=not key)
    model = st.selectbox("Model", C.MODEL_CHOICES, disabled=not key or orch.startswith("Deter"))
    go = st.button("Run five agents", type="primary", width="stretch")

AGENTS = [("Spatial", "Spatial / formation / geometry"), ("Analog", "Analog selection"),
          ("Decline", "Decline curve"), ("Transformer", "Transformer forecasting"),
          ("Ensemble", "Ensemble and uncertainty")]

if go:
    ctx = RunContext(wells, prod, surf, target)
    status = st.status("Agents working", expanded=True)
    bar = status.progress(0.0)

    def progress(i, name):
        bar.progress(min(i / 5, 1.0), text=f"{name}")

    try:
        bundle = get_bundle(wells, prod) if TORCH_OK else None
        if orch.startswith("Claude") and key:
            try:
                run_llm(ctx, api_key=key, model=model, bundle=bundle, progress=progress)
            except Exception as exc:
                status.write(f"Claude orchestration unavailable ({exc}); falling back to deterministic rules.")
                ctx = RunContext(wells, prod, surf, target)
                ctx.flag("Orchestrator", "info", f"Claude call failed, deterministic fallback used: {exc}")
                run_deterministic(ctx, bundle=bundle, progress=progress)
        else:
            run_deterministic(ctx, bundle=bundle, progress=progress)
        st.session_state["ctx"] = ctx
        st.session_state.pop("zip_bytes", None)
        status.update(label="All five agents finished", state="complete", expanded=False)
    except ValueError as exc:
        status.update(label="Input problem", state="error")
        st.error(str(exc))
        st.stop()

ctx = st.session_state.get("ctx")
if ctx is None or "ensemble" not in ctx.results:
    st.markdown(
        "Choose a producing well or a proposed surface location in the sidebar, then run the agents. "
        "Each agent writes into a shared run context; the orchestrator (Claude or a rule set) decides the "
        "order and parameters, reacts to QC flags, and writes the summary.")
    cols = st.columns(5)
    for c, (k, name) in zip(cols, AGENTS):
        c.markdown(f"<div class='agent'><b>{k}</b><br>{name}</div>", unsafe_allow_html=True)
    st.stop()

t = ctx.results["target"]
e = ctx.results["ensemble"]
where = t["api"] or "{:.5f}, {:.5f}".format(t["lat"], t["lon"])
st.subheader(f"{t['well_name']} · {where} · {C.ZONE_NAMES[t['zone']]}")
st.caption(f"Orchestrator: {ctx.orchestrator}")
c = st.columns(5)
kpi(c[0], "Oil EUR P50", f"{e['eur']['oil']['P50'] / 1e3:,.0f} Mbbl")
kpi(c[1], "Oil EUR P90 to P10", f"{e['eur']['oil']['P90'] / 1e3:,.0f} to {e['eur']['oil']['P10'] / 1e3:,.0f}")
kpi(c[2], "Gas EUR P50", f"{e['eur']['gas']['P50'] / 1e3:,.0f} MMcf")
kpi(c[3], "Analogs", f"{len(ctx.results['analogs'])} · sim {ctx.results['analogs'].similarity.median():.2f}")
kpi(c[4], "QC flags", f"{sum(q['level'] != 'info' for q in ctx.qc)} warn · {sum(q['level'] == 'info' for q in ctx.qc)} info")

left, right = st.columns([3, 2])
with left:
    from prodintel.plots import fig_ensemble
    st.plotly_chart(fig_ensemble(ctx, "oil", h=420), width="stretch")
    st.markdown("**Summary**")
    st.write(ctx.narrative)
with right:
    st.markdown("**Agent activity**")
    for r in ctx.log:
        colour = C.AMBER if r["agent"] == "Orchestrator" else C.TEAL
        st.markdown(f"<div class='agent' style='border-left-color:{colour}'><span class='note'>{r['t']}</span> "
                    f"<b>{r['agent']}</b> · {r['message']}</div>", unsafe_allow_html=True)
