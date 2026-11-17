"""
Deepwater sandstone and carbonate reservoir characterisation with integrated
Rock-Eval and XRD data. Case study app, Day 1 Session 1.

digitaloilEd.ai · Advanced AI for Conventional & Unconventional Reservoir Petrophysics

Run:  streamlit run app.py
All data are synthetic and generated on start-up.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dwcs import models as M
from dwcs.data import MINERALS, build_field

NAVY, TEAL, AMBER, SLATE, MIST, PAPER = "#0A2E36", "#14808C", "#E8A317", "#5A6B70", "#A9C3C6", "#F7F5F0"
SEQ = [TEAL, AMBER, NAVY, "#7FB8BE", "#B5651D", SLATE, "#C9D8DA", "#6B4E16"]
INTERVAL_COLOUR = {"Sandstone": AMBER, "Carbonate": TEAL}
FLUID_COLOUR = {"Light oil": TEAL, "Bitumen / tar": NAVY, "Water": MIST}

st.set_page_config(page_title="Deepwater case study · digitaloilEd.ai", page_icon="🛢️", layout="wide")

st.markdown(
    f"""
<style>
html, body, [class*="css"], .stMarkdown, p, li, label {{ font-family: Calibri, Carlito, "Segoe UI", sans-serif; }}
h1, h2, h3, h4 {{ font-family: Cambria, Caladea, Georgia, serif !important; color: {NAVY}; }}
h1 {{ font-weight: 700; letter-spacing: -0.5px; }}
section[data-testid="stSidebar"] {{ background: {NAVY}; }}
section[data-testid="stSidebar"] * {{ color: #F4F1EA; }}
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] * {{ color: {NAVY}; }}
section[data-testid="stSidebar"] input {{ color: {NAVY} !important; }}
.dw-note {{ border-left: 6px solid {AMBER}; background: #FFFFFF; padding: 0.8rem 1rem; border-radius: 6px; margin: 0.4rem 0 1rem 0; }}
.dw-key {{ border-left: 6px solid {TEAL}; background: #FFFFFF; padding: 0.8rem 1rem; border-radius: 6px; margin: 0.4rem 0 1rem 0; }}
.dw-brand {{ font-family: Cambria, Caladea, Georgia, serif; font-size: 1.35rem; font-weight: 700; color: {AMBER} !important; }}
div[data-testid="stMetricValue"] {{ color: {NAVY}; font-family: Cambria, Caladea, Georgia, serif; }}
</style>
""",
    unsafe_allow_html=True,
)


def style(fig: go.Figure, height: int = 460) -> go.Figure:
    fig.update_layout(
        height=height, paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font=dict(family="Calibri, Carlito, sans-serif", size=14, color=NAVY),
        margin=dict(l=60, r=20, t=50, b=50), colorway=SEQ,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="#E6ECEC", zeroline=False)
    fig.update_yaxes(gridcolor="#E6ECEC", zeroline=False)
    return fig


def note(text: str, kind: str = "note"):
    st.markdown(f'<div class="dw-{kind}">{text}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# cached computation
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def get_field(seed: int):
    return build_field(seed)


@st.cache_data(show_spinner=False)
def get_plugs(seed, shift, ob, rubble, s2_cut):
    return M.plug_table(get_field(seed), apply_shift=shift, apply_ob=ob, reject_rubble=rubble, s2_cut=s2_cut)


@st.cache_data(show_spinner=False)
def get_scores(seed, shift, ob, rubble, s2_cut, kind, pooled, n_trees, bit):
    return M.scenario_scores(get_plugs(seed, shift, ob, rubble, s2_cut), kind, pooled, n_trees, bit)


@st.cache_data(show_spinner=False)
def get_lowo(seed, shift, ob, rubble, s2_cut, scenario, kind, pooled, n_trees, bit):
    return M.lowo_permeability(get_plugs(seed, shift, ob, rubble, s2_cut), scenario, kind, pooled,
                               n_trees, bitumen_flag=bit)


@st.cache_data(show_spinner=False)
def get_importance(seed, shift, ob, rubble, s2_cut, scenario, interval, n_trees, bit):
    return M.lowo_importance(get_plugs(seed, shift, ob, rubble, s2_cut), scenario, interval,
                             "Random Forest", n_trees, bit)


@st.cache_data(show_spinner=False)
def get_rock_types(seed, shift, ob, rubble, s2_cut, n_types, n_trees):
    rt = M.rock_types(get_plugs(seed, shift, ob, rubble, s2_cut), n_types)
    pred, summary = M.lowo_rock_type(rt, n_trees)
    rt["RT_PRED"] = pred
    return rt, summary


@st.cache_data(show_spinner=False)
def get_kh(seed, shift, ob, rubble, s2_cut, scenario, kind, n_trees, linear_mean):
    f = get_field(seed)
    return M.kh_check(f, get_plugs(seed, shift, ob, rubble, s2_cut), scenario, kind, n_trees,
                      linear_mean=linear_mean)


@st.cache_data(show_spinner=False)
def get_vclay(seed, shift, ob, rubble, s2_cut):
    f = get_field(seed)
    d = get_plugs(seed, shift, ob, rubble, s2_cut)
    g = d[d.GEO].copy()
    g["IS_CB"] = (g.INTERVAL == "Carbonate").astype(int)
    g = M.vclay_methods(g, f.logs)
    g["VCL_ML"] = M.lowo_ml_vclay(g)
    return g


# ---------------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------------

FINDINGS_SLIDE = 25   # "What the case study dashboard proves, in four numbers" in the final Session 1 deck

# Plain names for the driver row, matching the wording on the findings slide
FEATURE_LABEL = {
    "CLR_QTZ": "Quartz (clr)", "CLR_KFS": "K-feldspar (clr)", "CLR_CAL": "Calcite (clr)",
    "CLR_DOL": "Dolomite (clr)", "CLR_ILL": "Illite (clr)", "CLR_KAO": "Kaolinite (clr)",
    "CLR_CHL": "Chlorite (clr)", "NPHI": "Neutron porosity", "RHOB": "Bulk density",
    "GR": "Gamma ray", "DT": "Sonic slowness", "PEF": "Photoelectric factor",
    "LOG_RT": "Deep resistivity (log10)", "ND_SEP": "Neutron-density separation",
    "LOG_S1": "Rock-Eval S1 (log10)", "LOG_S2": "Rock-Eval S2 (log10)", "LOG_TOC": "TOC (log10)",
    "BIT_FLAG": "Bitumen flag",
}

PAGES = ["Case overview", "1 · QC and alignment", "2 · XRD mineralogy", "3 · Rock-Eval and fluids",
         "Rock typing", "4 · Permeability models", "5 · Well-test check", f"Findings for slide {FINDINGS_SLIDE}"]

with st.sidebar:
    st.markdown('<div class="dw-brand">digitaloilEd.ai</div>', unsafe_allow_html=True)
    st.caption("Advanced AI for Reservoir Petrophysics · Day 1 case study")
    page = st.radio("Section", PAGES, label_visibility="collapsed")
    st.divider()
    st.markdown("**Workflow decisions**")
    shift = st.toggle("1 · Depth-match core to logs", value=True)
    ob = st.toggle("1 · Correct core to overburden", value=True)
    rubble = st.toggle("1 · Reject rubble and fractured plugs", value=True)
    bit = st.toggle("3 · Flag bitumen class from Rock-Eval S2", value=True)
    s2_cut = st.slider("S2 cut-off for bitumen (mg HC/g)", 2.0, 10.0, 4.0, 0.5)
    pooled = st.radio("4 · Model structure", ["Separate by interval", "Pooled"], horizontal=False) == "Pooled"
    st.divider()
    kind = st.selectbox("Learner", ["Random Forest", "XGBoost"])
    n_trees = st.slider("Trees", 100, 400, 150, 50)
    seed = int(st.number_input("Field realisation (seed)", 1, 999, 11))
    if kind == "XGBoost" and not M.HAS_XGB:
        st.caption("xgboost not installed: using HistGradientBoosting.")

field = get_field(seed)
plugs = get_plugs(seed, shift, ob, rubble, s2_cut)
KEY = (seed, shift, ob, rubble, s2_cut)


# ---------------------------------------------------------------------------
# pages
# ---------------------------------------------------------------------------

def page_overview():
    st.title("Deepwater sandstone and carbonate with integrated Rock-Eval and XRD")
    st.markdown(
        "#### Can mineralogy and geochemistry explain the permeability and fluid-quality "
        "variation that logs alone miss?"
    )
    note("Field A is synthetic: eight wells through a turbidite sandstone and an underlying carbonate "
         "with a tar mat above its oil-water contact. Numbers in this app illustrate behaviour and "
         "must not be quoted as field results.")

    c1, c2 = st.columns([1.05, 1])
    with c1:
        st.subheader("Data inventory")
        p = field.plugs
        geo = p.XRD_QTZ.notna()
        rows = []
        for iv in ["Sandstone", "Carbonate"]:
            s = p.INTERVAL == iv
            cored = p[s].WELL.nunique()
            length = cored * (120 if iv == "Sandstone" else 120)
            rows.append({
                "Interval": iv,
                "Wells with full log suite": field.wells.shape[0],
                "Cored wells · core length": f"{cored} · {length * 0.3048:.0f} m",
                "Core plugs with φ and k": int(s.sum()),
                "XRD samples": int((s & geo).sum()),
                "Rock-Eval samples": int((s & p.RE_S2.notna()).sum()),
                "Fluid samples (API)": int((s & p.API.notna()).sum()),
                "Well tests (kh)": int((field.tests.INTERVAL == iv).sum()),
            })
        inv = pd.DataFrame(rows).set_index("Interval").T.astype(str)
        st.dataframe(inv, width="stretch")
        st.caption("Synthetic Field A: logs are dense inputs; XRD and Rock-Eval exist only at core points.")
        wells = field.wells[["WELL", "CORED", "MUD", "TAR_MAT"]].rename(
            columns={"CORED": "Cored", "MUD": "Mud", "TAR_MAT": "Carbonate tar mat"})
        st.dataframe(wells, hide_index=True, width="stretch")

    with c2:
        st.subheader("Well viewer")
        w = st.selectbox("Well", field.wells.WELL.tolist(), index=0)
        lg = field.logs[field.logs.WELL == w]
        wp = plugs[plugs.WELL == w]
        fig = make_subplots(rows=1, cols=6, shared_yaxes=True, horizontal_spacing=0.015,
                            subplot_titles=["GR", "RHOB", "NPHI", "RT", "Core k", "S2"])
        fig.add_trace(go.Scatter(x=lg.GR, y=lg.DEPTH, line=dict(color=AMBER, width=1)), 1, 1)
        fig.add_trace(go.Scatter(x=lg.RHOB, y=lg.DEPTH, line=dict(color=NAVY, width=1)), 1, 2)
        fig.add_trace(go.Scatter(x=lg.NPHI, y=lg.DEPTH, line=dict(color=TEAL, width=1)), 1, 3)
        fig.add_trace(go.Scatter(x=lg.RT, y=lg.DEPTH, line=dict(color=NAVY, width=1)), 1, 4)
        if len(wp):
            fig.add_trace(go.Scatter(x=wp.K, y=wp.LOG_DEPTH, mode="markers",
                                     marker=dict(size=4, color=TEAL)), 1, 5)
            g = wp[wp.GEO]
            fig.add_trace(go.Scatter(x=g.RE_S2, y=g.LOG_DEPTH, mode="markers",
                                     marker=dict(size=5, color=np.where(g.BIT_FLAG == 1, NAVY, AMBER))), 1, 6)
        wm = field.wells.set_index("WELL").loc[w]
        # flagged-bitumen band just above the carbonate OWC (from the S2 flag, not from hidden truth)
        if len(wp):
            near = wp[(wp.BIT_FLAG == 1) & (wp.LOG_DEPTH <= wm.CB_OWC) & (wp.LOG_DEPTH >= wm.CB_OWC - 40)]
            if len(near):
                fig.add_hrect(y0=float(near.LOG_DEPTH.min()), y1=float(wm.CB_OWC),
                              fillcolor=NAVY, opacity=0.08, line_width=0)
        for y, lab in [(wm.SS_OWC, "Sandstone OWC"), (wm.CB_TOP, "Carbonate top"), (wm.CB_OWC, "Carbonate OWC")]:
            fig.add_hline(y=y, line=dict(color=SLATE, dash="dot", width=1))
            fig.add_annotation(x=0.02, xref="x domain", y=y, yref="y", text=lab, showarrow=False,
                               xanchor="left", yanchor="bottom", font=dict(size=11, color=SLATE))
        fig.update_xaxes(type="log", row=1, col=4)
        fig.update_xaxes(type="log", row=1, col=5)
        fig.update_yaxes(autorange="reversed", title_text="Depth (ft)", row=1, col=1)
        fig.update_layout(showlegend=False)
        st.plotly_chart(style(fig, 720), width="stretch")
        if not wm.CORED:
            st.caption("Uncored well: logs and a well test only.")
        else:
            st.caption("Dotted lines, top to bottom: sandstone oil-water contact, carbonate top, carbonate "
                       "oil-water contact. Dark S2 points are flagged bitumen; the shaded band above the "
                       "carbonate OWC is the tar mat they define. Gaps in the core tracks are uncored rock.")


def page_qc():
    st.title("QC and alignment")
    note("Decision 1: depth-match core by core gamma, correct to overburden, and reject rubble. "
         "Toggle each in the sidebar and watch the permeability scores on page 4 respond.", "key")

    est = M.estimated_shifts(field)
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.subheader("Core gamma against log GR")
        fig = go.Figure()
        for i, w in enumerate(est.WELL):
            lags, cc, best = M.shift_curve(field, w)
            fig.add_trace(go.Scatter(x=lags, y=cc, name=w, line=dict(color=SEQ[i % len(SEQ)])))
        fig.update_layout(xaxis_title="Shift applied: core depth minus log depth (ft)",
                          yaxis_title="Correlation coefficient")
        st.plotly_chart(style(fig, 420), width="stretch")
    with c2:
        st.subheader("Estimated shifts")
        show = est.rename(columns={"SHIFT_FT": "Shift (ft)", "CC_AT_BEST": "r at best",
                                   "CC_AT_ZERO": "r unshifted"})
        st.dataframe(show.round(3), hide_index=True, width="stretch")
        with st.expander("Instructor answer key"):
            st.write({k: v for k, v in field.truth["shifts"].items() if k in set(est.WELL)})

    o1, o2 = st.columns([1, 1])
    w = o1.selectbox("Overlay for well", est.WELL.tolist())
    view = o2.radio("View", ["30 ft zoom on a distinctive bed", "Whole cored sections"], horizontal=True)
    cg = field.core_gamma[field.core_gamma.WELL == w].sort_values("CORE_DEPTH")
    lg = field.logs[field.logs.WELL == w]
    s = float(est.set_index("WELL").SHIFT_FT[w])

    def broken(df, depth_col):
        """Depth and GR arrays with a break between core runs, so Plotly does not join them."""
        xs, ys = [], []
        for _, run in df.groupby("INTERVAL", sort=False):
            xs += list(run.CORE_GR) + [None]
            ys += list(run[depth_col]) + [None]
        return xs, ys

    cg = cg.assign(SHIFTED=cg.CORE_DEPTH - s)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lg.GR, y=lg.DEPTH, name="Log GR", line=dict(color=SLATE, width=1)))
    x, y = broken(cg, "CORE_DEPTH")
    fig.add_trace(go.Scatter(x=x, y=y, name="Core gamma, driller's depth", connectgaps=False,
                             line=dict(color=AMBER, width=1.5)))
    x, y = broken(cg, "SHIFTED")
    fig.add_trace(go.Scatter(x=x, y=y, name="Core gamma, shifted", connectgaps=False,
                             line=dict(color=TEAL, width=2)))
    if view.startswith("30 ft"):
        # centre on the highest core-gamma spike in the sandstone run: the easiest bed to match by eye
        ss = cg[cg.INTERVAL == "Sandstone"]
        centre = float(ss.loc[ss.CORE_GR.idxmax(), "SHIFTED"]) if len(ss) else float(cg.SHIFTED.median())
        y_range = [centre + 15, centre - 15]
    else:
        y_range = [cg.CORE_DEPTH.max() + 5, cg.CORE_DEPTH.min() - 5]
    fig.update_yaxes(autorange=False, title="Depth (ft)", range=y_range)
    fig.update_xaxes(title="GR (API)")
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(style(fig, 520), width="stretch")
        st.caption(f"{w}: estimated shift {s:+.1f} ft. Amber is core gamma at driller's depth; teal is the "
                   "same curve after the shift, sitting on the log. Gaps are uncored intervals, not bad hole.")
    with c4:
        st.subheader("Rubble and fractured plugs")
        raw = get_plugs(seed, shift, ob, False, s2_cut)
        fig2 = px.scatter(raw, x="PHI", y="K", color="PLUG_NOTE", log_y=True, facet_col="INTERVAL",
                          color_discrete_map={"intact": MIST, "rubble / induced fracture": AMBER},
                          labels={"PHI": "Porosity (frac)", "K": "k (mD)", "PLUG_NOTE": ""})
        st.plotly_chart(style(fig2, 460), width="stretch")
        note("Rubble plugs sit far above the trend: induced fractures raise measured k. They describe "
             "neither the matrix nor the flowing system.")

    st.subheader("Effect on blind-well skill (logs only, Random Forest)")
    with st.spinner("Scoring with and without depth matching"):
        on = get_lowo(seed, True, ob, rubble, s2_cut, "A · Logs only", "Random Forest", False, n_trees, False)
        off = get_lowo(seed, False, ob, rubble, s2_cut, "A · Logs only", "Random Forest", False, n_trees, False)
    cols = st.columns(4)
    for i, iv in enumerate(["Sandstone", "Carbonate"]):
        a, b = on[on.INTERVAL == iv], off[off.INTERVAL == iv]
        cols[2 * i].metric(f"{iv} · depth-matched R²", f"{M.metrics(a.LOG_K, a.PRED)['R2']:.2f}")
        cols[2 * i + 1].metric(f"{iv} · unshifted R²", f"{M.metrics(b.LOG_K, b.PRED)['R2']:.2f}")
    st.caption("Both scores are always computed here, so this comparison does not depend on the sidebar "
               "toggle. The depth-matching toggle changes the scores on page 4 and on the Findings page.")


def page_xrd():
    st.title("XRD mineralogy")
    note("Decision 2: convert XRD weight % to volume fractions with grain densities, then clr-transform "
         "before modelling.", "key")
    g = get_vclay(*KEY)

    t1, t2, t3 = st.tabs(["Clay type and permeability", "Closure", "Vclay methods"])
    with t1:
        d = g.copy()
        d["Illite share of illite + kaolinite"] = d.V_ILL / (d.V_ILL + d.V_KAO)
        ss = d[d.INTERVAL == "Sandstone"]
        fig = px.scatter(ss, x="PHI", y="K", color="Illite share of illite + kaolinite", log_y=True,
                         color_continuous_scale=[[0, TEAL], [0.5, "#E6ECEC"], [1, AMBER]],
                         hover_data=["WELL", "FACIES"], labels={"PHI": "Porosity (frac)", "K": "k (mD)"})
        c1, c2 = st.columns([1.4, 1])
        c1.plotly_chart(style(fig, 480), width="stretch")
        with c2:
            fig2 = px.scatter(ss, x="CLAY_SOLID", y="GR", color="Illite share of illite + kaolinite",
                              color_continuous_scale=[[0, TEAL], [0.5, "#E6ECEC"], [1, AMBER]],
                              labels={"CLAY_SOLID": "Total clay (volume fraction of solids)"})
            fig2.update_layout(coloraxis_showscale=False)
            st.plotly_chart(style(fig2, 480), width="stretch")
        note("At equal porosity, illite-rich sands (amber) sit one to two decades below kaolinitic "
             "sands (teal). GR responds to total clay, not to clay species, so logs cannot see this "
             "diagenetic overprint.")
    with t2:
        raw, cl = M.closure_demo(g)
        c1, c2 = st.columns(2)
        for col, mat, title in [(c1, raw, "Raw volume fractions"), (c2, cl, "Centred log-ratio")]:
            fig = px.imshow(mat.round(2), text_auto=True, zmin=-1, zmax=1,
                            color_continuous_scale=[[0, TEAL], [0.5, "#FFFFFF"], [1, AMBER]], title=title)
            col.plotly_chart(style(fig, 460), width="stretch")
        note("Raw fractions must sum to one, so quartz and calcite correlate strongly negatively across "
             "the field for arithmetic reasons. clr removes the constant-sum constraint, although clr "
             "coefficients still sum to zero; use ilr where orthogonal coordinates are required.")
    with t3:
        methods = {"VCL_LINEAR": "Linear GR", "VCL_STEIBER": "Steiber", "VCL_CLAVIER": "Clavier",
                   "VCL_ML": "ML from logs (leave-one-well-out)"}
        rows = []
        for c, name in methods.items():
            for iv in ["Sandstone", "Carbonate"]:
                x = g[g.INTERVAL == iv]
                err = x[c] - x.VCLAY_XRD
                rows.append({"Method": name, "Interval": iv, "RMSE": np.sqrt((err ** 2).mean()),
                             "Bias": err.mean()})
        st.dataframe(pd.DataFrame(rows).pivot(index="Method", columns="Interval").round(3),
                     width="stretch")
        m = st.selectbox("Method to plot", list(methods), format_func=methods.get)
        d = g.copy()
        d["K-feldspar (vol frac)"] = d.V_KFS
        fig = px.scatter(d, x="VCLAY_XRD", y=m, color="K-feldspar (vol frac)", facet_col="INTERVAL",
                         color_continuous_scale=[[0, MIST], [1, AMBER]],
                         labels={"VCLAY_XRD": "Vclay from XRD (bulk volume)", m: methods[m]})
        fig.add_shape(type="line", x0=0, y0=0, x1=0.5, y1=0.5, line=dict(color=NAVY, dash="dot"),
                      row="all", col="all")
        st.plotly_chart(style(fig, 460), width="stretch")
        note("GR-based methods overstate clay in arkosic lobe sands (amber points): K-feldspar is "
             "radioactive. In the clean carbonate the GR end-points exaggerate the small clay content.")


def page_rockeval():
    st.title("Rock-Eval and fluid quality")
    note("Decision 3: carbonate samples with anomalous S2 form a bitumen-affected class. Set the "
         "cut-off in the sidebar.", "key")
    g = plugs[plugs.GEO].copy()
    cb = g[g.INTERVAL == "Carbonate"]
    owc = field.wells.set_index("WELL").CB_OWC
    cb = cb.assign(HAOWC=owc.reindex(cb.WELL).values - cb.LOG_DEPTH)
    bit = cb[cb.FLUID_CLASS == "Bitumen / tar"]
    oil = cb[cb.FLUID_CLASS == "Light oil"]
    wat = cb[cb.FLUID_CLASS == "Water"]
    TAR_FT = 25          # height above the OWC treated as the tar-mat band
    PAY_RT = 2.0         # a typical resistivity pay cut-off (ohm·m), for illustration
    lab = {"RE_S2": "S2 (mg HC/g rock)", "HAOWC": "Height above OWC (ft)", "PHI": "Porosity (frac)",
           "RT": "Deep resistivity (ohm·m)", "K": "k (mD)", "FLUID_CLASS": "Fluid class (core)"}

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("S2 against height above the carbonate OWC")
        fig = px.scatter(cb, x="RE_S2", y="HAOWC", color="FLUID_CLASS", hover_data=["WELL"],
                         color_discrete_map=FLUID_COLOUR, labels=lab)
        fig.add_hrect(y0=0, y1=TAR_FT, fillcolor=NAVY, opacity=0.07, line_width=0)
        fig.add_hline(y=0, line=dict(color=SLATE, dash="dot"),
                      annotation_text="Carbonate OWC", annotation_position="bottom right")
        fig.add_vline(x=s2_cut, line=dict(color=AMBER, dash="dash"),
                      annotation_text=f"S2 cut-off {s2_cut:g} mg/g", annotation_position="top right")
        xmax = float(cb.RE_S2.max())
        fig.add_annotation(x=xmax, y=TAR_FT, text="Tar mat: dense bitumen within ~20 ft of the OWC",
                           showarrow=False, xanchor="right", yanchor="bottom", font=dict(size=12, color=NAVY))
        high = bit[bit.HAOWC > TAR_FT]
        if len(high):
            fig.add_annotation(x=xmax, y=float(high.HAOWC.max()), text="Scattered bitumen beds",
                               showarrow=False, xanchor="right", yanchor="bottom", font=dict(size=12, color=NAVY))
        st.plotly_chart(style(fig, 480), width="stretch")
        st.caption(f"{(bit.HAOWC <= TAR_FT).sum()} of {len(bit)} bitumen samples sit within {TAR_FT} ft of the "
                   "contact. Oil and water both sit near zero S2, so S2 separates bitumen cleanly where logs cannot.")
    with c2:
        st.subheader("Resistivity cannot isolate tar")
        fig = px.scatter(cb, x="PHI", y="RT", color="FLUID_CLASS", log_y=True, hover_data=["WELL"],
                         color_discrete_map=FLUID_COLOUR, labels=lab)
        fig.add_hline(y=PAY_RT, line=dict(color=AMBER, dash="dash"),
                      annotation_text=f"Typical pay cut-off {PAY_RT:g} ohm·m", annotation_position="top left")
        st.plotly_chart(style(fig, 480), width="stretch")
        share = float((bit.RT > PAY_RT).mean()) if len(bit) else 0.0
        st.caption(f"{share:.0%} of bitumen samples pass a {PAY_RT:g} ohm·m pay cut-off, yet the oil in them won't flow. "
                   f"Median Rt: light oil {oil.RT.median():.1f}, bitumen {bit.RT.median():.1f}, water "
                   f"{wat.RT.median():.1f} ohm·m. Tar sits between water and oil, so no resistivity cut-off "
                   "isolates it: a model trained on logs alone cannot learn this class.")

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Bitumen cuts permeability")
        fig = px.scatter(cb, x="PHI", y="K", color="FLUID_CLASS", log_y=True, hover_data=["WELL"],
                         color_discrete_map=FLUID_COLOUR, labels=lab)
        if len(oil) > 5 and len(bit):
            coef = np.polyfit(oil.PHI, np.log10(oil.K), 1)
            xs = np.linspace(cb.PHI.min(), cb.PHI.max(), 20)
            fig.add_trace(go.Scatter(x=xs, y=10 ** np.polyval(coef, xs), mode="lines", name="Light-oil trend",
                                     line=dict(color=TEAL, dash="dot", width=1.5)))
            factor = 10 ** (-np.median(np.log10(bit.K) - np.polyval(coef, bit.PHI)))
            st.caption(f"At the same porosity, bitumen samples sit about {factor:.0f}× below the light-oil "
                       "trend (median). Porosity logs see the pore volume, not what is blocking it.")
        st.plotly_chart(style(fig, 440), width="stretch")
    with c4:
        st.subheader("Oil-based mud inflates S1")
        oilleg = g[g.FLUID_CLASS == "Light oil"].merge(field.wells[["WELL", "MUD"]], on="WELL")
        fig = px.box(oilleg, x="WELL", y="RE_S1", color="MUD", color_discrete_map={"WBM": TEAL, "OBM": AMBER},
                     labels={"RE_S1": "S1 (mg HC/g rock), oil leg", "MUD": "Mud system", "WELL": ""})
        for w in oilleg[oilleg.MUD == "OBM"].WELL.unique():
            fig.add_annotation(x=w, y=float(oilleg[oilleg.WELL == w].RE_S1.max()), text="OBM: not an oil show",
                               showarrow=True, arrowhead=2, ay=-30, font=dict(size=12, color=NAVY))
        st.plotly_chart(style(fig, 440), width="stretch")
        st.caption("Mud filtrate adds free hydrocarbon to the sample. Record the mud system with every Rock-Eval "
                   "result, or a model will learn the mud, not the rock.")

    tp = ((cb.BIT_FLAG == 1) & (cb.FLUID_CLASS == "Bitumen / tar")).sum()
    fp = ((cb.BIT_FLAG == 1) & (cb.FLUID_CLASS != "Bitumen / tar")).sum()
    fn = ((cb.BIT_FLAG == 0) & (cb.FLUID_CLASS == "Bitumen / tar")).sum()
    m1, m2, m3 = st.columns(3)
    m1.metric("Bitumen samples caught", int(tp))
    m2.metric("False alarms", int(fp))
    m3.metric("Missed", int(fn))
    st.caption("A miss means perforating a tar mat; a false alarm means one more sample or a cautious "
               "completion. The relative cost of the two, not the statistics, should set the S2 cut-off. "
               "This synthetic field separates cleanly at 4 mg/g; real data will overlap.")
    note("High resistivity with low S1 and high S2 is the tar-mat signature. The logs call it pay; "
         "the pyrolysis says the oil is immobile. A-03 was drilled with oil-based mud, so its S1 "
         "values carry contamination and should not be read as oil shows.")


def page_rocktype():
    st.title("Rock typing: core flow units to logs")
    n_types = st.slider("Rock types per interval", 3, 6, 4)
    with st.spinner("Clustering on FZI and training leave-one-well-out classifiers"):
        rt, summ = get_rock_types(*KEY, n_types, n_trees)
    m1, m2 = st.columns(2)
    m1.metric("Blind-well rock-type accuracy", f"{summ['accuracy']:.0%}")
    m2.metric("Majority-class baseline", f"{summ['majority_baseline']:.0%}")
    tabs = st.tabs(["Sandstone", "Carbonate"])
    for tab, iv in zip(tabs, ["Sandstone", "Carbonate"]):
        with tab:
            d = rt[rt.INTERVAL == iv]
            c1, c2 = st.columns([1.3, 1])
            fig = px.scatter(d, x="PHIZ", y="RQI", color="ROCK_TYPE", log_x=True, log_y=True,
                             color_discrete_sequence=[NAVY, TEAL, "#7FB8BE", AMBER, "#B5651D", SLATE],
                             category_orders={"ROCK_TYPE": sorted(d.ROCK_TYPE.unique())},
                             labels={"PHIZ": "φz = φ / (1 − φ)", "RQI": "RQI (µm)",
                                     "ROCK_TYPE": "Rock type (core FZI)"})
            c1.plotly_chart(style(fig, 460), width="stretch")
            labels, cm = M.rt_confusion(rt, rt.RT_PRED.values, iv)
            fig2 = px.imshow(cm, x=labels, y=labels, text_auto=True,
                             color_continuous_scale=[[0, "#FFFFFF"], [1, TEAL]],
                             labels=dict(x="Predicted from logs", y="Core rock type", color="Count"))
            c2.plotly_chart(style(fig2, 460), width="stretch")
            off = cm.sum() - np.trace(cm)
            near = sum(cm[i, j] for i in range(len(cm)) for j in range(len(cm)) if abs(i - j) == 1)
            acc_iv = np.trace(cm) / cm.sum() if cm.sum() else 0
            c2.caption(f"Diagonal = correct ({acc_iv:.0%}). {near / off:.0%} of the errors fall one class away, "
                       "so a wrong prediction is usually a neighbouring flow unit, not a different rock."
                       if off else "Every sample classified correctly.")
            c1.caption("Each colour is a flow unit: samples on one unit-slope line share pore geometry (FZI). "
                       "The labels come from core k, so they train the classifier but never enter the k model.")
    note("RQI, FZI and R35 contain measured k. They define rock-type labels in cored wells and never "
         "enter the permeability model as inputs.")


def page_perm():
    st.title("Permeability models")
    note("Decision 4: separate or pooled models, judged only by leave-one-well-out scores. Decision 5: "
         "report k as P10–P90 with drivers.", "key")
    for sc, desc in M.SCENARIOS.items():
        st.markdown(f"**{sc}** · {desc}")
    with st.spinner("Scoring three feature scenarios, leave-one-well-out"):
        sc = get_scores(*KEY, kind, pooled, n_trees, bit)
    fig = px.bar(sc, x="INTERVAL", y="R2", color="Scenario", barmode="group", text_auto=".2f",
                 color_discrete_sequence=[SLATE, AMBER, TEAL],
                 labels={"R2": "Blind-well R² for log10 k", "INTERVAL": ""})
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(style(fig, 420), width="stretch")
    A_, B_, C_ = list(M.SCENARIOS)
    r2s = sc.set_index(["Scenario", "INTERVAL"]).R2
    g1, g2, g3, g4 = st.columns(4)
    for col, iv in [(g1, "Sandstone"), (g3, "Carbonate")]:
        col.metric(f"{iv} · measured geochemistry (B − A)", f"{r2s[B_, iv] - r2s[A_, iv]:+.2f}")
    for col, iv in [(g2, "Sandstone"), (g4, "Carbonate")]:
        col.metric(f"{iv} · predicted from logs (C − A)", f"{r2s[C_, iv] - r2s[A_, iv]:+.2f}")
    st.caption("Read the bars as two gaps. B − A is what measuring XRD and Rock-Eval is worth. C − A is what "
               "predicting the same geochemistry from logs recovers: almost nothing, because a model cannot "
               "add information its inputs never contained. B is an upper bound: it needs core at every depth.")

    scenario = st.selectbox("Inspect scenario", list(M.SCENARIOS))
    res = get_lowo(*KEY, scenario, kind, pooled, n_trees, bit)
    c1, c2 = st.columns([1.3, 1])
    with c1:
        fig = go.Figure()
        for iv in ["Sandstone", "Carbonate"]:
            r = res[res.INTERVAL == iv]
            fig.add_trace(go.Scatter(
                x=r.LOG_K, y=r.PRED, mode="markers", name=iv,
                marker=dict(color=INTERVAL_COLOUR[iv], size=6, opacity=0.75),
                error_y=dict(type="data", symmetric=False, array=r.CONF_HI - r.PRED,
                             arrayminus=r.PRED - r.CONF_LO, color="rgba(90,107,112,0.18)", thickness=1),
                customdata=np.stack([r.WELL, r.FACIES], axis=1),
                hovertemplate="%{customdata[0]} · %{customdata[1]}<br>core %{x:.2f} · model %{y:.2f}"))
        fig.add_shape(type="line", x0=-3, y0=-3, x1=4, y1=4, line=dict(color=NAVY, dash="dot"))
        fig.add_annotation(x=3.6, y=3.6, text="1:1", showarrow=False, xanchor="left", font=dict(color=NAVY))
        fig.update_layout(xaxis_title="Core log10 k (mD)", yaxis_title="Predicted log10 k (mD)")
        st.plotly_chart(style(fig, 520), width="stretch")
        st.caption("Each point is a plug predicted by a model that never saw its well. Grey bars are the "
                   "cross-conformal P10–P90. Points far from the 1:1 line with narrow bars would be the "
                   "dangerous ones: confident and wrong.")
    with c2:
        st.subheader("Interval coverage (target 80%)")
        cov = M.coverage_table(res)
        st.dataframe(cov.set_index("INTERVAL").style.format("{:.2f}"), width="stretch")
        wid = cov.set_index("INTERVAL")["Median band width (decades)"]
        st.caption("Band width in plain terms: P90 is about "
                   + ", ".join(f"{10 ** wid[iv]:.0f}× P10 in the {iv.lower()}" for iv in wid.index)
                   + ". That spread belongs in the reserves range, not in a footnote.")
        note("Spread across the trees of one forest falls short of 80%: it measures disagreement "
             "among trees trained on the same wells, not the difference between wells. Cross-conformal "
             "bands, sized from other wells' blind residuals, land close to target by construction.")
        per_well = res.groupby(["INTERVAL", "WELL"]).apply(
            lambda g: pd.Series(M.metrics(g.LOG_K, g.PRED))).reset_index()
        st.dataframe(per_well.round(2), hide_index=True, width="stretch", height=250)

    st.subheader("Drivers: permutation importance on held-out wells")
    if scenario.startswith("C") and n_trees > 200:
        st.caption("Scenario C refits the mineralogy models inside every fold; this can take a minute.")
    c3, c4 = st.columns(2)
    for col, iv in zip([c3, c4], ["Sandstone", "Carbonate"]):
        with st.spinner(f"Importance, {iv.lower()}"):
            imp = get_importance(*KEY, scenario, iv, min(n_trees, 200), bit).head(10)
        fig = px.bar(imp[::-1], x="importance", y="feature", orientation="h", title=iv,
                     color_discrete_sequence=[INTERVAL_COLOUR[iv]],
                     labels={"importance": "Drop in blind-well R² when shuffled", "feature": ""})
        col.plotly_chart(style(fig, 420), width="stretch")
        if len(imp):
            top = FEATURE_LABEL.get(imp.iloc[0].feature, imp.iloc[0].feature)
            col.caption(f"Top driver: {top}. Importance is measured on held-out wells, so it shows what "
                        "transfers between wells, not what the model memorised.")


def page_kh():
    st.title("Well-test check")
    note("Well tests measure kh over each oil leg at flow scale. They are the only independent check "
         "in the three uncored wells.", "key")
    c0, c1 = st.columns([1, 1])
    scenario = c0.selectbox("Deployable scenario", [s for s in M.SCENARIOS if not s.startswith("B")])
    linear = c1.toggle("Average trees in linear k (Random Forest)", value=True,
                       help="Averaging log k and then exponentiating gives the geometric mean, which "
                            "underestimates arithmetic kh when high-k streaks carry the flow.")
    kh = get_kh(*KEY, scenario, kind, n_trees, linear)
    fig = px.scatter(kh, x="KH_TEST", y="KH_MODEL", color="INTERVAL", symbol="CORED", log_x=True, log_y=True,
                     text="WELL", color_discrete_map=INTERVAL_COLOUR,
                     symbol_map={True: "circle", False: "diamond-open"},
                     labels={"KH_TEST": "Well-test kh (mD·ft)", "KH_MODEL": "Model kh (mD·ft)",
                             "CORED": "Cored well"})
    lo, hi = kh[["KH_TEST", "KH_MODEL"]].min().min() * 0.7, kh[["KH_TEST", "KH_MODEL"]].max().max() * 1.4
    for f_, dash in [(1, "dot"), (2, "dash"), (0.5, "dash")]:
        fig.add_shape(type="line", x0=lo, y0=lo * f_, x1=hi, y1=hi * f_, line=dict(color=SLATE, dash=dash, width=1))
    fig.update_traces(textposition="top center", marker=dict(size=11))
    c2, c3 = st.columns([1.4, 1])
    c2.plotly_chart(style(fig, 540), width="stretch")
    with c3:
        summ = kh.groupby(["INTERVAL", "CORED"]).RATIO.median().unstack().rename(
            columns={True: "Cored wells", False: "Uncored wells"})
        st.markdown("**Median model / test kh**")
        st.dataframe(summ.round(2), width="stretch")
        st.dataframe(kh.round(2), hide_index=True, width="stretch", height=330)
    note("Cored wells trained the model, so they sit close to the line. The uncored wells are the "
         "honest test. Core and test also sample different volumes: plugs miss fractures and vugs "
         "that a test sees.")


def page_findings():
    st.title(f"Findings for slide {FINDINGS_SLIDE}")
    note("Populated from the current settings. Change a workflow decision in the sidebar and the table "
         "updates. Synthetic results: label them as such on the slide.")
    with st.spinner("Assembling results"):
        sc = get_scores(*KEY, kind, pooled, n_trees, bit).set_index(["Scenario", "INTERVAL"]).R2
        rt, summ = get_rock_types(*KEY, 4, n_trees)
        res_a = get_lowo(*KEY, "A · Logs only", kind, pooled, n_trees, bit)
        cov = M.coverage_table(res_a).set_index("INTERVAL")["Cross-conformal P10–P90"]
        imp = {iv: get_importance(*KEY, "B · Logs + measured XRD and Rock-Eval", iv, min(n_trees, 200), bit)
               .iloc[0].feature for iv in ["Sandstone", "Carbonate"]}
        imp = {iv: FEATURE_LABEL.get(f, f) for iv, f in imp.items()}
        kh = get_kh(*KEY, "A · Logs only", kind, n_trees, True)
        dm_on = get_lowo(seed, True, ob, rubble, s2_cut, "A · Logs only", "Random Forest", False, n_trees, False)
        dm_off = get_lowo(seed, False, ob, rubble, s2_cut, "A · Logs only", "Random Forest", False, n_trees, False)
    rt_acc = {iv: float((g.ROCK_TYPE == g.RT_PRED).mean()) for iv, g in rt.groupby("INTERVAL")}
    khr = kh[~kh.CORED].groupby("INTERVAL").RATIO.median()
    A, B, C = list(M.SCENARIOS)

    # ---- the four numbers on the closing slide, computed live ----------------
    st.subheader(f"The four numbers on slide {FINDINGS_SLIDE}")
    r2 = lambda d, iv: M.metrics(d[d.INTERVAL == iv].LOG_K, d[d.INTERVAL == iv].PRED)["R2"]
    gain = {iv: sc[B, iv] - sc[A, iv] for iv in ["Sandstone", "Carbonate"]}
    tree_cov = M.coverage_table(res_a).set_index("INTERVAL")["Tree-spread P10–P90"]
    unc = kh[~kh.CORED].RATIO
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Blind-well R² from depth matching (sandstone)",
              f"{r2(dm_off, 'Sandstone'):.2f} → {r2(dm_on, 'Sandstone'):.2f}")
    f1.caption(f"Carbonate {r2(dm_off, 'Carbonate'):.2f} → {r2(dm_on, 'Carbonate'):.2f}. "
               "One afternoon of core QC outweighs any choice of algorithm.")
    f2.metric("R² gained from measured geochemistry (B − A)",
              f"+{min(gain.values()):.2f} to {max(gain.values()):.2f}")
    f2.caption(f"Predicting it from logs (C − A) adds "
               f"{min(sc[C, iv] - sc[A, iv] for iv in gain):+.2f} to "
               f"{max(sc[C, iv] - sc[A, iv] for iv in gain):+.2f}: it must be measured.")
    f3.metric("Calibrated P10–P90 coverage", f"{cov.mean():.0%}")
    f3.caption(f"Cross-conformal bands. Tree spread covers {tree_cov.min():.0%} to {tree_cov.max():.0%}.")
    f4.metric("Model / test kh, uncored wells", f"{unc.min():.2f} to {unc.max():.2f}")
    f4.caption("The slide rounds this to × 2 to 3: the honest range for any forecast built on these "
               "predictions.")
    note(f"Return to the show of hands from slide 24: measured geochemistry added "
         f"{gain['Sandstone']:+.2f} in the sandstone and {gain['Carbonate']:+.2f} in the carbonate.", "key")

    st.subheader("Detail behind the numbers")
    rows = [
        ("R² log k · logs only", sc[A, "Sandstone"], sc[A, "Carbonate"], "{:.2f}"),
        ("R² log k · logs + XRD + Rock-Eval", sc[B, "Sandstone"], sc[B, "Carbonate"], "{:.2f}"),
        ("R² log k · logs + log-predicted geochemistry", sc[C, "Sandstone"], sc[C, "Carbonate"], "{:.2f}"),
        ("Rock-type accuracy", rt_acc["Sandstone"] * 100, rt_acc["Carbonate"] * 100, "{:.0f} %"),
        ("P10–P90 coverage of core k", cov["Sandstone"] * 100, cov["Carbonate"] * 100, "{:.0f} %"),
        ("Top driver, permutation (scenario B)", imp["Sandstone"], imp["Carbonate"], "{}"),
        ("Model kh / test kh, uncored wells (median)", khr["Sandstone"], khr["Carbonate"], "{:.2f}"),
    ]
    table = pd.DataFrame([{"Blind-well result": r[0], "Sandstone": r[3].format(r[1]),
                           "Carbonate": r[3].format(r[2])} for r in rows])
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption(f"Learner: {kind} · structure: {'pooled' if pooled else 'separate'} · trees: {n_trees} · "
               f"majority-class rock-type baseline {summ['majority_baseline']:.0%}. The driver row uses "
               "permutation importance on held-out wells, a model-agnostic stand-in for SHAP.")

    md = "| Blind-well result | Sandstone | Carbonate |\n|---|---|---|\n" + "\n".join(
        f"| {r['Blind-well result']} | {r['Sandstone']} | {r['Carbonate']} |" for _, r in table.iterrows())
    c1, c2 = st.columns(2)
    c1.download_button("Download as CSV", table.to_csv(index=False).encode(), f"slide{FINDINGS_SLIDE}_findings.csv", "text/csv")
    c2.download_button("Download as Markdown", md.encode(), f"slide{FINDINGS_SLIDE}_findings.md", "text/markdown")

    st.subheader("Discuss in pairs")
    st.markdown("Where would you refuse to use this model?  \n"
                "Which extra measurement would move the carbonate result most?")
    buf = io.StringIO()
    # drop generator truth columns (prefixed "_", e.g. true k and true depth): they would leak the answer
    plugs.drop(columns=[c for c in plugs.columns if c.startswith("_")]).to_csv(buf, index=False)
    st.download_button("Download the QC'd plug table for Exercise 1", buf.getvalue().encode(),
                       "field_a_plugs_qc.csv", "text/csv")


{
    PAGES[0]: page_overview, PAGES[1]: page_qc, PAGES[2]: page_xrd, PAGES[3]: page_rockeval,
    PAGES[4]: page_rocktype, PAGES[5]: page_perm, PAGES[6]: page_kh, PAGES[7]: page_findings,
}[page]()

st.caption("digitaloilEd.ai · Keith R. Holdaway FGS · synthetic Field A, generated in the app")
