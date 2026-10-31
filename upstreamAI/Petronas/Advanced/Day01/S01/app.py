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

PAGES = ["Case overview", "1 · QC and alignment", "2 · XRD mineralogy", "3 · Rock-Eval and fluids",
         "Rock typing", "4 · Permeability models", "5 · Well-test check", "Findings for slide 22"]

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
        st.caption("These counts fill the placeholders on slide 20.")
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
        for y, lab in [(wm.SS_OWC, "SS OWC"), (wm.CB_OWC, "CB OWC"), (wm.CB_TOP, "Carbonate top")]:
            fig.add_hline(y=y, line=dict(color=SLATE, dash="dot", width=1))
        fig.update_xaxes(type="log", row=1, col=4)
        fig.update_xaxes(type="log", row=1, col=5)
        fig.update_yaxes(autorange="reversed", title_text="Depth (ft)", row=1, col=1)
        fig.update_layout(showlegend=False)
        st.plotly_chart(style(fig, 720), width="stretch")
        if not wm.CORED:
            st.caption("Uncored well: logs and a well test only.")
        else:
            st.caption("Dotted lines: sandstone OWC, carbonate top, carbonate OWC. Dark S2 points are "
                       "flagged bitumen.")


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

    w = st.selectbox("Overlay for well", est.WELL.tolist())
    cg = field.core_gamma[field.core_gamma.WELL == w]
    lg = field.logs[field.logs.WELL == w]
    s = float(est.set_index("WELL").SHIFT_FT[w])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lg.GR, y=lg.DEPTH, name="Log GR", line=dict(color=SLATE, width=1)))
    fig.add_trace(go.Scatter(x=cg.CORE_GR, y=cg.CORE_DEPTH, name="Core gamma, driller's depth",
                             line=dict(color=AMBER, width=1)))
    fig.add_trace(go.Scatter(x=cg.CORE_GR, y=cg.CORE_DEPTH - s, name="Core gamma, shifted",
                             line=dict(color=TEAL, width=1.5)))
    fig.update_yaxes(autorange="reversed", title="Depth (ft)",
                     range=[cg.CORE_DEPTH.max() + 5, cg.CORE_DEPTH.min() - 5])
    fig.update_xaxes(title="GR (API)")
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(style(fig, 520), width="stretch")
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
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("S2 against height above the carbonate OWC")
        owc = field.wells.set_index("WELL").CB_OWC
        cb = cb.assign(HAOWC=owc.reindex(cb.WELL).values - cb.LOG_DEPTH)
        fig = px.scatter(cb, x="RE_S2", y="HAOWC", color="FLUID_CLASS", symbol="WELL",
                         color_discrete_map=FLUID_COLOUR,
                         labels={"RE_S2": "S2 (mg HC/g rock)", "HAOWC": "Height above OWC (ft)",
                                 "FLUID_CLASS": "Fluid class (core)"})
        fig.add_vline(x=s2_cut, line=dict(color=AMBER, dash="dash"))
        fig.add_hline(y=0, line=dict(color=SLATE, dash="dot"))
        st.plotly_chart(style(fig, 480), width="stretch")
    with c2:
        st.subheader("Tar looks like pay on resistivity")
        fig = px.scatter(cb, x="PHI", y="RT", color="FLUID_CLASS", log_y=True,
                         color_discrete_map=FLUID_COLOUR,
                         labels={"PHI": "Porosity (frac)", "RT": "Deep resistivity (ohm·m)",
                                 "FLUID_CLASS": "Fluid class (core)"})
        st.plotly_chart(style(fig, 480), width="stretch")
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Bitumen cuts permeability")
        fig = px.scatter(cb, x="PHI", y="K", color="FLUID_CLASS", log_y=True,
                         color_discrete_map=FLUID_COLOUR, labels={"PHI": "Porosity (frac)", "K": "k (mD)"})
        st.plotly_chart(style(fig, 440), width="stretch")
    with c4:
        st.subheader("Oil-based mud inflates S1")
        oil = g[g.FLUID_CLASS == "Light oil"].merge(field.wells[["WELL", "MUD"]], on="WELL")
        fig = px.box(oil, x="WELL", y="RE_S1", color="MUD", color_discrete_map={"WBM": TEAL, "OBM": AMBER},
                     labels={"RE_S1": "S1 (mg HC/g rock), oil leg"})
        st.plotly_chart(style(fig, 440), width="stretch")
    tp = ((cb.BIT_FLAG == 1) & (cb.FLUID_CLASS == "Bitumen / tar")).sum()
    fp = ((cb.BIT_FLAG == 1) & (cb.FLUID_CLASS != "Bitumen / tar")).sum()
    fn = ((cb.BIT_FLAG == 0) & (cb.FLUID_CLASS == "Bitumen / tar")).sum()
    m1, m2, m3 = st.columns(3)
    m1.metric("Bitumen samples caught", int(tp))
    m2.metric("False alarms", int(fp))
    m3.metric("Missed", int(fn))
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
                             labels={"PHIZ": "φz = φ / (1 − φ)", "RQI": "RQI (µm)"})
            c1.plotly_chart(style(fig, 460), width="stretch")
            labels, cm = M.rt_confusion(rt, rt.RT_PRED.values, iv)
            fig2 = px.imshow(cm, x=labels, y=labels, text_auto=True,
                             color_continuous_scale=[[0, "#FFFFFF"], [1, TEAL]],
                             labels=dict(x="Predicted from logs", y="Core rock type", color="Count"))
            c2.plotly_chart(style(fig2, 460), width="stretch")
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
        fig.update_layout(xaxis_title="Core log10 k (mD)", yaxis_title="Predicted log10 k (mD)")
        st.plotly_chart(style(fig, 520), width="stretch")
    with c2:
        st.subheader("Interval coverage (target 80%)")
        cov = M.coverage_table(res)
        st.dataframe(cov.set_index("INTERVAL").style.format("{:.2f}"), width="stretch")
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
    st.title("Findings for slide 22")
    note("Populated from the current settings. Change a workflow decision in the sidebar and the table "
         "updates. Synthetic results: label them as such on the slide.")
    with st.spinner("Assembling results"):
        sc = get_scores(*KEY, kind, pooled, n_trees, bit).set_index(["Scenario", "INTERVAL"]).R2
        rt, summ = get_rock_types(*KEY, 4, n_trees)
        res_a = get_lowo(*KEY, "A · Logs only", kind, pooled, n_trees, bit)
        cov = M.coverage_table(res_a).set_index("INTERVAL")["Cross-conformal P10–P90"]
        imp = {iv: get_importance(*KEY, "B · Logs + measured XRD and Rock-Eval", iv, min(n_trees, 200), bit)
               .iloc[0].feature for iv in ["Sandstone", "Carbonate"]}
        kh = get_kh(*KEY, "A · Logs only", kind, n_trees, True)
    rt_acc = {iv: float((g.ROCK_TYPE == g.RT_PRED).mean()) for iv, g in rt.groupby("INTERVAL")}
    khr = kh[~kh.CORED].groupby("INTERVAL").RATIO.median()
    A, B, C = list(M.SCENARIOS)
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
    c1.download_button("Download as CSV", table.to_csv(index=False).encode(), "slide22_findings.csv", "text/csv")
    c2.download_button("Download as Markdown", md.encode(), "slide22_findings.md", "text/markdown")

    st.subheader("Discuss in pairs")
    st.markdown("Where would you refuse to use this model?  \n"
                "Which extra measurement would move the carbonate result most?")
    buf = io.StringIO()
    plugs.to_csv(buf, index=False)
    st.download_button("Download the QC'd plug table for Exercise 1", buf.getvalue().encode(),
                       "field_a_plugs_qc.csv", "text/csv")


{
    PAGES[0]: page_overview, PAGES[1]: page_qc, PAGES[2]: page_xrd, PAGES[3]: page_rockeval,
    PAGES[4]: page_rocktype, PAGES[5]: page_perm, PAGES[6]: page_kh, PAGES[7]: page_findings,
}[page]()

st.caption("digitaloilEd.ai · Keith R. Holdaway FGS · synthetic Field A, generated in the app")
