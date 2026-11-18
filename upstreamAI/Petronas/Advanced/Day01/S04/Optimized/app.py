"""
Model Validation and Business Integration · Permian geomechanics (Midland Basin, synthetic)
Day 1 · Session 4 · Advanced AI for Conventional & Unconventional Reservoir Petrophysics · digitaloilEd.ai

Run:  conda activate petro-core, then python -m streamlit run app.py
"""
from __future__ import annotations

import os
import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from pgm import models as M
from pgm import precomputed as PC
from pgm.data import FACIES, FAULT, LATERAL, build, fracture_height, pilot_profile, stage_table

NAVY, TEAL, AMBER, SLATE, MIST = "#0A2E36", "#14808C", "#E8A317", "#5A6B70", "#A9C3C6"
FAC_COL = {"Siliceous mudstone": TEAL, "Calcareous mudstone": "#7FB8BE", "Carbonate debris flow": AMBER,
           "Argillaceous mudstone": SLATE}
st.set_page_config(page_title="Model validation · Permian geomechanics · digitaloilEd.ai", page_icon="🪨", layout="wide")
st.markdown(f"""<style>
html, body, p, li, label {{ font-family: Calibri, Carlito, "Segoe UI", sans-serif; }}
h1, h2, h3, h4 {{ font-family: Cambria, Caladea, Georgia, serif !important; color: {NAVY}; }}
section[data-testid="stSidebar"] {{ background: {NAVY}; }}
section[data-testid="stSidebar"] * {{ color: #F4F1EA; }}
section[data-testid="stSidebar"] div[data-baseweb="select"] * {{ color: {NAVY}; }}
.note {{ border-left: 6px solid {AMBER}; background: #FFFFFF; padding: .8rem 1rem; border-radius: 6px; margin: .4rem 0 1rem 0; }}
.key {{ border-left: 6px solid {TEAL}; background: #FFFFFF; padding: .8rem 1rem; border-radius: 6px; margin: .4rem 0 1rem 0; }}
.brand {{ font-family: Cambria, Caladea, Georgia, serif; font-size: 1.35rem; font-weight: 700; color: {AMBER} !important; }}
div[data-testid="stMetricValue"] {{ color: {NAVY}; font-family: Cambria, Caladea, Georgia, serif; }}
</style>""", unsafe_allow_html=True)


def style(fig, h=440):
    fig.update_layout(height=h, paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", margin=dict(l=60, r=20, t=50, b=50),
                      font=dict(family="Calibri, Carlito, sans-serif", size=14, color=NAVY),
                      colorway=[TEAL, AMBER, NAVY, "#7FB8BE", "#B5651D", SLATE], legend=dict(bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(gridcolor="#E6ECEC", zeroline=False); fig.update_yaxes(gridcolor="#E6ECEC", zeroline=False)
    return fig


def box(text, kind="note"):
    st.markdown(f'<div class="{kind}">{text}</div>', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def data():
    return build()


# Precomputed leave-one-well-out predictions are the default in Cloud.
# Set PGM_LIVE_TRAINING=1 to demonstrate training on demand (or rebuild results offline).
LIVE_TRAINING = os.getenv("PGM_LIVE_TRAINING", "0") == "1"


@st.cache_data(show_spinner="Computing held-out predictions (first run only)")
def oof(fs, model):
    _, stg, _, _ = data()
    if not LIVE_TRAINING:
        cached = PC.load(stg, fs, model, 'oof')
        if cached is not None:
            return cached
    start = time.perf_counter()
    result = M.lowo_proba(stg, M.FEATURE_SETS[fs], model)
    print(f"LOWO predictions {model} / {fs}: {time.perf_counter()-start:.1f}s", flush=True)
    return result


@st.cache_data(show_spinner="Computing permutation importance on held-out wells")
def imp(fs, model):
    _, stg, _, _ = data()
    if not LIVE_TRAINING:
        cached = PC.load(stg, fs, model, 'imp')
        if cached is not None:
            return cached
    start = time.perf_counter()
    result = M.importance(stg, M.FEATURE_SETS[fs], model)
    print(f"LOWO permutation importance {model} / {fs}: {time.perf_counter()-start:.1f}s", flush=True)
    return result


@st.cache_data(show_spinner=False)
def psi_ref(feat):
    _, stg, _, _ = data()
    return M.psi_reference(stg, feat)


@st.cache_resource(show_spinner=False)
def fitted(fs, model):
    _, stg, _, _ = data()
    return M.make_model(model).fit(stg[M.FEATURE_SETS[fs]], stg.EFFECTIVE)


logs, stages, wells, design = data()
PAGES = ["Case overview", "1 · Geomechanics along the lateral", "2 · Stress profile and fracture height",
         "3 · What controls fracture effectiveness", "4 · Model validation", "5 · Business integration",
         "6 · Real-time drilling", "Findings and Day 1 wrap-up"]
with st.sidebar:
    st.markdown('<div class="brand">digitaloilEd.ai</div>', unsafe_allow_html=True)
    st.caption("Day 1 · Session 4 · Permian geomechanics case study · v1.1")
    page = st.radio("Section", PAGES, label_visibility="collapsed")
    st.divider()
    fs = st.selectbox("Feature set", list(M.FEATURE_SETS))
    model_name = st.selectbox("Model", M.MODELS, index=1)
    st.caption("All data are synthetic. Validation is leave-one-well-out throughout.")
    st.caption("Validation mode: " + ("Live model training" if LIVE_TRAINING else "Precomputed held-out predictions"))


# ------------------------------------------------------------------ pages
def overview():
    st.title("Model validation and business integration · Permian geomechanics")
    st.markdown("#### Can the rock we characterise every day tell us which stages will frac well, before we pump them?")
    box("One-size-fits-all completions treat every 200 ft stage the same. In heterogeneous tight rock, brittleness, "
        "closure stress, stress contrast and natural fractures vary within a stage, and so does the result. "
        "This case uses tracer-derived stage outcomes from 12 Midland Basin laterals to build and validate an AI workflow.")
    c = st.columns(4)
    c[0].metric("Horizontal wells", len(wells)); c[1].metric("Frac stages", len(stages))
    c[2].metric("Effective stages", f"{stages.EFFECTIVE.mean():.0%}"); c[3].metric("Lateral samples", f"{len(logs):,}")
    l, r = st.columns([1.1, 1])
    with l:
        st.subheader("Well and completion inventory")
        t = wells.merge(stages.groupby("WELL").EFFECTIVE.mean().rename("EFFECTIVE_SHARE"), on="WELL")
        st.dataframe(t.style.format({"EFFECTIVE_SHARE": "{:.0%}", "PROPPANT_LB_FT": "{:.0f}"}), hide_index=True, width="stretch")
        box("<b>Effective stage:</b> the stage's chemical-tracer share of well production is at least half the well's "
            "mean stage share. It is the label every model on page 4 predicts, and it carries the tracer's measurement noise.", "key")
    with r:
        st.subheader("Map view: pads, laterals and the fault")
        fig = go.Figure()
        s = stages.assign(Y=stages.WELL.map(wells.set_index("WELL").Y_FT), X=(stages.MD_TOE + stages.MD_HEEL) / 2)
        fig.add_trace(go.Scatter(x=s.X, y=s.Y, mode="markers", marker=dict(symbol="square", size=7,
                      color=np.where(s.EFFECTIVE == 1, TEAL, AMBER)), text=s.WELL + " stage " + s.STAGE.astype(str),
                      hoverinfo="text", name="Stage (teal effective, amber not)"))
        yy = np.array([10300, 15700]); fig.add_trace(go.Scatter(x=FAULT["x0"] + FAULT["slope"] * yy, y=yy, mode="lines",
                      line=dict(color=NAVY, dash="dash", width=2), name="Mapped fault"))
        for _, w in wells.iterrows():
            fig.add_annotation(x=-250, y=w.Y_FT, text=w.WELL, showarrow=False, font=dict(size=11))
        fig.update_layout(xaxis_title="Distance along lateral, toe to heel (ft)", yaxis_title="North (ft)", legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(style(fig, 560), width="stretch")


def lateral():
    st.title("Geomechanics along the lateral")
    w = st.selectbox("Well", wells.WELL.tolist(), index=8)
    g = logs[logs.WELL == w]; s = stages[stages.WELL == w]
    fig = make_subplots(rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        subplot_titles=["Facies", "Young's modulus (GPa) and Poisson's ratio", "Brittleness index (%)",
                                        "Closure stress Shmin (psi)", "Natural fracture intensity (per ft)", "Stage outcome"],
                        row_heights=[0.08, 0.2, 0.16, 0.2, 0.16, 0.1])
    for f, col in FAC_COL.items():
        z = g[g.FACIES == f]
        fig.add_trace(go.Scatter(x=z.MD, y=np.ones(len(z)), mode="markers", marker=dict(color=col, size=5, symbol="square"), name=f), 1, 1)
    fig.add_trace(go.Scatter(x=g.MD, y=g.YM_GPA, line=dict(color=NAVY, width=1), name="E (GPa)", showlegend=False), 2, 1)
    fig.add_trace(go.Scatter(x=g.MD, y=g.PR * 100, line=dict(color=AMBER, width=1), name="PR × 100", showlegend=False), 2, 1)
    fig.add_trace(go.Scatter(x=g.MD, y=g.BI, line=dict(color=TEAL, width=1), showlegend=False), 3, 1)
    fig.add_trace(go.Scatter(x=g.MD, y=g.SHMIN_PSI, line=dict(color=NAVY, width=1), showlegend=False), 4, 1)
    fig.add_trace(go.Scatter(x=g.MD, y=g.NFI, line=dict(color=SLATE, width=1), showlegend=False), 5, 1)
    fig.add_trace(go.Bar(x=(s.MD_TOE + s.MD_HEEL) / 2, y=s.TRACER_SHARE, width=190,
                         marker_color=np.where(s.EFFECTIVE == 1, TEAL, AMBER), showlegend=False), 6, 1)
    fig.update_yaxes(showticklabels=False, row=1, col=1); fig.update_xaxes(title_text="Measured depth along lateral (ft)", row=6, col=1)
    fig.update_layout(legend=dict(orientation="h", y=1.08))
    st.plotly_chart(style(fig, 900), width="stretch")
    c = st.columns(4)
    c[0].metric("Effective stages", f"{s.EFFECTIVE.mean():.0%}")
    c[1].metric("Median stress range in a stage", f"{s.STRESS_RANGE_PSI.median():.0f} psi")
    c[2].metric("Closest approach to fault", f"{s.FAULT_DIST_MIN.min():,.0f} ft")
    c[3].metric("Out-of-zone footage", f"{g.OOZ.mean():.0%}")
    box("Read the stress track against the outcome track. Stages that straddle a sharp stress contrast put some clusters in "
        "low-stress rock and some in high-stress rock: the low-stress clusters take the fluid and the others barely open.")


def stress():
    st.title("Stress profile and fracture height")
    pp = pilot_profile()
    c1, c2 = st.columns([1, 1.2])
    with c1:
        target = st.radio("Landing zone", ["Wolfcamp A target", "Wolfcamp B target"], horizontal=True)
        npres = st.slider("Net treating pressure (psi)", 100, 1200, 350, 50)
        top, base, reached = fracture_height(pp, target, npres)
        tr = pp[pp.LAYER == target].iloc[0]
        a, b = st.columns(2)
        a.metric("Fracture height", f"{base - top:,.0f} ft"); b.metric("Target thickness", f"{tr.BASE - tr.TOP:,.0f} ft")
        st.metric("Share of height in target", f"{(tr.BASE - tr.TOP) / (base - top):.0%}")
        st.write("Layers reached: " + ", ".join(reached))
        box("Screening model: the fracture crosses a layer while that layer's closure stress is below the target stress plus "
            "net pressure. It ignores toughness, modulus contrast and interface slip, so treat it as a first look, not a design.", "key")
        st.dataframe(pp[["LAYER", "TOP", "BASE", "YM_GPA", "PR", "SHMIN_PSI", "SHMIN_GRAD"]].style.format(
            {"SHMIN_PSI": "{:,.0f}", "SHMIN_GRAD": "{:.3f}"}), hide_index=True, width="stretch")
    with c2:
        fig = go.Figure()
        for _, r in pp.iterrows():
            fig.add_trace(go.Scatter(x=[r.SHMIN_PSI, r.SHMIN_PSI], y=[r.TOP, r.BASE], mode="lines", line=dict(color=NAVY, width=4), showlegend=False))
            fig.add_annotation(x=r.SHMIN_PSI + 60, y=(r.TOP + r.BASE) / 2, text=r.LAYER, showarrow=False, xanchor="left", font=dict(size=11))
        fig.add_vline(x=tr.SHMIN_PSI + npres, line=dict(color=AMBER, dash="dash"), annotation_text="target stress + net pressure")
        fig.add_shape(type="rect", x0=pp.SHMIN_PSI.min() - 300, x1=pp.SHMIN_PSI.max() + 900, y0=top, y1=base,
                      fillcolor="rgba(20,128,140,0.15)", line_width=0)
        fig.update_yaxes(autorange="reversed", title="TVD (ft)"); fig.update_xaxes(title="Minimum horizontal stress (psi)")
        st.plotly_chart(style(fig, 640), width="stretch")
    box("Raise net pressure past about 450 psi from the Wolfcamp A target and the fracture breaks through the upper barrier, then runs "
        "into the lower-stress Dean and Lower Spraberry. That is out-of-zone proppant, and the classic cause of parent-child "
        "communication on multi-bench pads.")


def controls():
    st.title("What controls fracture effectiveness")
    feats = ["BI", "STRESS_RANGE_PSI", "NFI", "CLAY", "TOC", "SHMIN_PSI", "YM_GPA", "PR", "FAULT_DIST_MIN", "OOZ_FRAC", "GR", "MSE_KSI"]
    f = st.selectbox("Stage property", feats)
    c1, c2 = st.columns(2)
    with c1:
        bins = pd.qcut(stages[f].rank(method="first"), 5, labels=["Lowest 20 %", "2", "3", "4", "Highest 20 %"])
        rate = stages.groupby(bins, observed=True).EFFECTIVE.mean().reset_index()
        fig = px.bar(rate, x=f, y="EFFECTIVE", text_auto=".0%", labels={f: f"{f} quintile", "EFFECTIVE": "Effective stages"},
                     color_discrete_sequence=[TEAL])
        fig.update_yaxes(range=[0, 1], tickformat=".0%")
        st.plotly_chart(style(fig, 420), width="stretch")
    with c2:
        fig = px.box(stages, x="EFFECTIVE", y=f, color="EFFECTIVE", color_discrete_map={1: TEAL, 0: AMBER}, points="outliers",
                     labels={"EFFECTIVE": "Stage effective (1) or not (0)"})
        st.plotly_chart(style(fig, 420), width="stretch")
    st.subheader("Correlations with the outcome")
    corr = stages[feats + ["EFFECTIVE"]].corr(method="spearman")["EFFECTIVE"].drop("EFFECTIVE").sort_values()
    fig = px.bar(corr, orientation="h", color=corr.values, color_continuous_scale=[[0, AMBER], [0.5, "#F4F1EA"], [1, TEAL]],
                 labels={"value": "Spearman correlation with effective stage", "index": ""})
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(style(fig, 460), width="stretch")
    box("Correlation is a first screen, not causation. Brittleness, closure stress and Young's modulus move together, so a "
        "single-variable view overstates each one. Page 4 lets the models sort out their joint effect.")


def validation():
    st.title("Model validation")
    box(f"<b>{model_name}</b> on the <b>{fs}</b> feature set, scored leave-one-well-out: every stage is predicted by a model that never saw its well.", "key")
    y = stages.EFFECTIVE.to_numpy(); p = oof(fs, model_name); rule = M.rule_baseline(stages)
    cm_, cr_ = M.curves(y, p), M.curves(y, rule)
    thr = st.slider("Decision threshold: complete a stage fully when P(effective) is at least", 0.05, 0.95, 0.5, 0.05)
    r = M.at_threshold(y, p, thr)
    c = st.columns(5)
    c[0].metric("ROC AUC", f"{cm_['auc']:.3f}", f"{cm_['auc'] - cr_['auc']:+.3f} vs BI rule")
    c[1].metric("Average precision", f"{cm_['ap']:.3f}", f"{cm_['ap'] - cr_['ap']:+.3f} vs BI rule")
    c[2].metric("Precision", f"{r['precision']:.0%}"); c[3].metric("Recall", f"{r['recall']:.0%}"); c[4].metric("Accuracy", f"{r['accuracy']:.0%}")
    c1, c2, c3 = st.columns(3)
    with c1:
        z = [[r["tn"], r["fp"]], [r["fn"], r["tp"]]]
        fig = px.imshow(z, x=["Predicted not effective", "Predicted effective"], y=["Not effective", "Effective"], text_auto=True,
                        color_continuous_scale=[[0, "#FFFFFF"], [1, TEAL]], title="Confusion matrix")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style(fig, 400), width="stretch")
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cm_["fpr"], y=cm_["tpr"], name=f"{model_name} ({cm_['auc']:.2f})", line=dict(color=TEAL, width=3)))
        fig.add_trace(go.Scatter(x=cr_["fpr"], y=cr_["tpr"], name=f"BI rule ({cr_['auc']:.2f})", line=dict(color=AMBER, dash="dash")))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(color=SLATE, dash="dot"), showlegend=False))
        fig.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate", legend=dict(x=0.35, y=0.08))
        st.plotly_chart(style(fig, 400), width="stretch")
    with c3:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cm_["rec"], y=cm_["prec"], name=f"{model_name} ({cm_['ap']:.2f})", line=dict(color=TEAL, width=3)))
        fig.add_trace(go.Scatter(x=cr_["rec"], y=cr_["prec"], name=f"BI rule ({cr_['ap']:.2f})", line=dict(color=AMBER, dash="dash")))
        fig.add_hline(y=y.mean(), line=dict(color=SLATE, dash="dot"), annotation_text="no-skill")
        fig.update_layout(title="Precision-recall curve", xaxis_title="Recall", yaxis_title="Precision", legend=dict(x=0.02, y=0.08))
        st.plotly_chart(style(fig, 400), width="stretch")
    st.subheader("All models, both feature sets")
    rows = []
    for f_ in M.FEATURE_SETS:
        for m_ in M.MODELS:
            cc = M.curves(y, oof(f_, m_)); rows.append({"Feature set": f_, "Model": m_, "ROC AUC": cc["auc"], "Average precision": cc["ap"]})
    rows.append({"Feature set": "Brittleness only", "Model": "BI rule of thumb", "ROC AUC": cr_["auc"], "Average precision": cr_["ap"]})
    st.dataframe(pd.DataFrame(rows).style.format({"ROC AUC": "{:.3f}", "Average precision": "{:.3f}"}), hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Drivers (permutation importance, held-out wells)")
        im = imp(fs, model_name)
        fig = px.bar(im[::-1], x="importance", y="feature", orientation="h", color_discrete_sequence=[TEAL],
                     labels={"importance": "Drop in ROC AUC when shuffled", "feature": ""})
        st.plotly_chart(style(fig, 440), width="stretch")
    with c2:
        st.subheader("Skill by well")
        pw = pd.DataFrame({"WELL": stages.WELL, "y": y, "p": p}).groupby("WELL").apply(
            lambda g: pd.Series({"ROC AUC": M.curves(g.y, g.p)["auc"] if g.y.nunique() > 1 else np.nan, "Effective share": g.y.mean()}))
        fig = px.bar(pw.reset_index(), x="WELL", y="ROC AUC", color_discrete_sequence=[NAVY]); fig.update_yaxes(range=[0.5, 1])
        st.plotly_chart(style(fig, 440), width="stretch")


def business():
    st.title("Business integration")
    box("Acting on the model: stages predicted effective get the full design; the rest get a reduced, cheaper treatment. "
        "Values below are illustrative defaults: replace them with your own economics.")
    c = st.columns(4)
    vtp = c[0].number_input("Value of a correctly completed effective stage ($k)", 0, 2000, 180, 10)
    cfp = c[1].number_input("Spend wasted on a full completion of a dud stage ($k)", 0, 1000, 60, 10)
    vtn = c[2].number_input("Saving from de-scoping a dud stage ($k)", 0, 1000, 40, 5)
    cfn = c[3].number_input("Value lost by de-scoping an effective stage ($k)", 0, 2000, 120, 10)
    y = stages.EFFECTIVE.to_numpy(); p = oof(fs, model_name)
    vc = M.value_curve(y, p, vtp, cfp, vtn, cfn); rc = M.value_curve(y, M.rule_baseline(stages), vtp, cfp, vtn, cfn)
    base = M.value_curve(y, np.ones(len(y)), vtp, cfp, vtn, cfn).value.iloc[0]
    best = vc.loc[vc.value.idxmax()]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=vc.threshold, y=(vc.value - base) / 1000, name=model_name, line=dict(color=TEAL, width=3)))
    fig.add_trace(go.Scatter(x=rc.threshold, y=(rc.value - base) / 1000, name="BI rule", line=dict(color=AMBER, dash="dash")))
    fig.add_hline(y=0, line=dict(color=SLATE, dash="dot"), annotation_text="one-size-fits-all (complete everything)")
    fig.update_layout(xaxis_title="Decision threshold", yaxis_title="Value vs one-size-fits-all, 600 stages ($MM)")
    l, r = st.columns([1.4, 1])
    l.plotly_chart(style(fig, 440), width="stretch")
    with r:
        st.metric("Best threshold", f"{best.threshold:.2f}")
        st.metric("Uplift over completing everything", f"${(best.value - base) / 1000:,.1f}MM", f"${(best.value - base) / 1000 / 12:,.2f}MM per well")
        st.metric("Uplift using the BI rule at its best", f"${(rc.value.max() - base) / 1000:,.1f}MM")
        box("The optimal threshold is set by the cost ratio, not by 0.5. Change the four values and watch it move.", "key")
        st.caption("The best threshold is chosen on the same held-out predictions it is scored on, so the uplift is slightly "
                   "optimistic. In deployment, fix the threshold on the training wells first.")
    st.subheader("Completion strategy simulator (one well)")
    box("The synthetic field's generating model acts as the simulator here. In a real asset this role is played by a pilot "
        "programme with tracers or fibre-optic monitoring.")
    st.caption("The de-scoping model is fitted on all 12 wells, including the one simulated, so its calls on that well are "
               "in-sample and slightly flattering.")
    a, b, c3 = st.columns(3)
    w = a.selectbox("Well", wells.WELL.tolist(), index=5)
    strat = b.radio("Stage placement", ["Geometric (every 200 ft)", "Engineered (stress-based) staging"])
    skip = c3.slider("De-scope stages with P(effective) below", 0.0, 0.6, 0.0, 0.05)
    mdl = fitted(fs, model_name)
    res = {}
    for sname in ["Geometric (every 200 ft)", "Engineered (stress-based) staging"]:
        res[sname] = M.simulate_completion(logs, wells, design, w, sname, skip if skip > 0 else None, mdl, M.FEATURE_SETS[fs],
                                           cut=float(stages.loc[stages.WELL == w, "LABEL_CUT"].iloc[0]))
    cur = res[strat]
    k = st.columns(4)
    k[0].metric("Stages", len(cur)); k[1].metric("Median stress range in a stage", f"{cur.STRESS_RANGE_PSI.median():.0f} psi")
    k[2].metric("Expected effective stages (fully completed)", f"{(cur.P_TRUE * cur.COMPLETED).sum():.1f}")
    k[3].metric("Stages de-scoped", int((~cur.COMPLETED).sum()))
    comp = pd.DataFrame({s_: {"Stages": len(v), "Median stress range (psi)": v.STRESS_RANGE_PSI.median(),
                               "Expected effective share": v.P_TRUE.mean()} for s_, v in res.items()}).T
    st.dataframe(comp.style.format({"Median stress range (psi)": "{:.0f}", "Expected effective share": "{:.1%}"}), width="stretch")
    fig = px.bar(cur, x=(cur.MD_TOE + cur.MD_HEEL) / 2, y="P_TRUE", color=cur.COMPLETED.map({True: "Full completion", False: "De-scoped"}),
                 color_discrete_map={"Full completion": TEAL, "De-scoped": AMBER},
                 labels={"x": "Stage mid-point MD (ft)", "P_TRUE": "Expected P(effective)", "color": ""})
    st.plotly_chart(style(fig, 360), width="stretch")


def realtime():
    st.title("Real-time drilling application")
    box("Operationalising means predicting while the lateral is still being drilled, from data available at the bit: "
        "gamma ray, drilling mechanical specific energy (MSE) and the landing position. Well W-13 is being drilled now.", "key")
    st.caption("This page always uses logistic regression on the real-time feature set. The sidebar choices do not change it.")
    c1, c2 = st.columns([1, 2])
    drift = c1.slider("Rock difference from training wells (added clay fraction)", 0.0, 0.2, 0.0, 0.02)
    bit = c1.slider("Bit depth along lateral (ft)", 1000, int(LATERAL), 6000, 200)
    thr = c1.slider("Flag stages below P(effective)", 0.2, 0.7, 0.4, 0.05)
    nw = M.new_well(drift=drift)
    drilled = nw[nw.MD < bit]
    fs_rt = "Real-time drilling (GR, MSE, landing, design)"
    mdl = fitted(fs_rt, "Logistic regression")
    stg = stage_table(drilled, design={"W-13": dict(CLUSTERS=5, PROPPANT_LB_FT=2200, FLUID_BBL_FT=40)})
    stg["P_MODEL"] = mdl.predict_proba(stg[M.FEATURE_SETS[fs_rt]])[:, 1]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.3, 0.3, 0.4],
                        subplot_titles=["Gamma ray (gAPI)", "MSE (ksi)", "Predicted P(effective) per 200 ft stage"])
    fig.add_trace(go.Scatter(x=drilled.MD, y=drilled.GR, line=dict(color=AMBER, width=1)), 1, 1)
    fig.add_trace(go.Scatter(x=drilled.MD, y=drilled.MSE_KSI, line=dict(color=NAVY, width=1)), 2, 1)
    fig.add_trace(go.Bar(x=(stg.MD_TOE + stg.MD_HEEL) / 2, y=stg.P_MODEL, width=190,
                         marker_color=np.where(stg.P_MODEL < thr, AMBER, TEAL)), 3, 1)
    fig.add_hline(y=thr, line=dict(color=SLATE, dash="dot"), row=3, col=1)
    fig.update_xaxes(range=[0, LATERAL]); fig.update_layout(showlegend=False)
    c2.plotly_chart(style(fig, 620), width="stretch")
    with c1:
        st.metric("Stages drilled", len(stg)); st.metric("Flagged for review", int((stg.P_MODEL < thr).sum()))
        st.subheader("Drift monitor")
        rows = []
        for f_ in ["GR", "MSE_KSI"]:
            ref = psi_ref(f_)
            rows.append({"Input": f_, "PSI, W-13": M.psi(stages[f_].to_numpy(), stg[f_].to_numpy()),
                         "Largest PSI among training wells": ref.max()})
        d = pd.DataFrame(rows)
        d["Status"] = np.select([d["PSI, W-13"] > 1.5 * d["Largest PSI among training wells"],
                                 d["PSI, W-13"] > d["Largest PSI among training wells"]], ["Review before use", "Watch"], "Within training range")
        st.dataframe(d.style.format({"PSI, W-13": "{:.2f}", "Largest PSI among training wells": "{:.2f}"}), hide_index=True, width="stretch")
        box("Population stability index of the new well against the training stages. Single wells always differ, so the alarm "
            "is calibrated on the training wells themselves: a new well is flagged only when it is more unusual than any of them.")
    st.subheader("Discussion: operationalising for real-time use")
    st.markdown("""
| Question | Points to draw out |
|---|---|
| What decision does the prediction feed, and when? | Stage placement and design are set after TD but before the frac; drilling data gives a head start of days |
| Which inputs arrive in time? | GR and MSE at the bit; image logs and dipole sonic may never be run in the lateral |
| Who acts on a flag? | Completions engineer with the geologist; flags are advice, not automatic changes |
| How do we know it still works? | Drift monitor on inputs; tracer results from every new well added to retraining |
| What if the rig is in new rock? | High PSI: fall back to the physics-based design and label the well for retraining |
""")


def findings():
    st.title("Findings and Day 1 wrap-up")
    y = stages.EFFECTIVE.to_numpy()
    rows = []
    for f_ in M.FEATURE_SETS:
        for m_ in M.MODELS:
            cc = M.curves(y, oof(f_, m_)); rows.append((f_, m_, cc["auc"], cc["ap"]))
    t = pd.DataFrame(rows, columns=["Feature set", "Model", "ROC AUC", "Average precision"])
    cr = M.curves(y, M.rule_baseline(stages))
    best = t.loc[t["ROC AUC"].idxmax()]
    oracle = M.curves(y, stages.P_TRUE.to_numpy())
    c = st.columns(4)
    c[0].metric("Best model ROC AUC", f"{best['ROC AUC']:.3f}", best.Model)
    c[1].metric("Brittleness rule ROC AUC", f"{cr['auc']:.3f}")
    c[2].metric("Ceiling (generating model)", f"{oracle['auc']:.3f}")
    rt = t[t["Feature set"].str.startswith("Real-time")]["ROC AUC"].max()
    c[3].metric("Best real-time drilling model", f"{rt:.3f}")
    st.dataframe(t.style.format({"ROC AUC": "{:.3f}", "Average precision": "{:.3f}"}), hide_index=True, width="stretch")
    box("The ceiling is the ROC AUC of the true probability that each stage is labelled effective, known here because the "
        "field is synthetic. No model can beat it: the label comes from a tracer share with about a factor of three "
        "measurement error, so stages near the cut-off are labelled almost at random. In a real field the ceiling is unknown.", "key")
    st.subheader("Day 1 key takeaways")
    st.markdown("""
- **Robust validation:** every score today was leave-one-well-out; shuffled splits would have flattered every model.
- **Uncertainty quantification:** P10 to P90 bands on permeability, Monte Carlo STOIIP ranges, and here probabilities with a decision threshold set by economics.
- **Business context:** the best threshold comes from the cost of each error, not from 0.5; the best model is the one that pays, which is not always the most complex.
""")
    st.subheader("Preview of Day 2")
    st.markdown("Tomorrow the input becomes an image: borehole images, core photographs, thin sections and SEM. "
                "Natural fracture intensity, a key driver today, becomes something we measure from borehole images rather than infer. "
                "The discipline stays the same: split by well, not by image tile.")


{PAGES[0]: overview, PAGES[1]: lateral, PAGES[2]: stress, PAGES[3]: controls, PAGES[4]: validation,
 PAGES[5]: business, PAGES[6]: realtime, PAGES[7]: findings}[page]()
st.caption("digitaloilEd.ai · Keith R. Holdaway FGS · synthetic Midland Basin data generated in the app")
