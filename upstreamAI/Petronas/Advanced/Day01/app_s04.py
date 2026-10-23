"""Geomechanical Features Controlling Fracture Generation · digitaloilEd.ai case study.

Run:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import geomech_core as gc

NAVY, TEAL, AMBER, INK, MUTED = "#0A2E36", "#14808C", "#E8A317", "#1C2B2F", "#5B6A6E"
GF_COLOURS = [TEAL, AMBER, "#B5533C", "#6A8D3A", "#7A6FA6", "#8C8C8C"]

st.set_page_config(page_title="Geomechanical Facies Case Study", page_icon="🪨", layout="wide")
st.markdown(f"""
<style>
h1, h2, h3 {{ font-family: Cambria, Georgia, serif; color: {NAVY}; }}
html, body, [class*="css"] {{ font-family: Calibri, Carlito, Arial, sans-serif; }}
div[data-testid="stMetricValue"] {{ color: {TEAL}; }}
.brand {{ color:{AMBER}; font-weight:700; letter-spacing:2px; font-size:0.85rem; }}
.callout {{ background:#F4F2EC; border-left:6px solid {TEAL}; padding:0.8rem 1rem; border-radius:6px; margin:0.4rem 0 0.8rem; }}
.trainer {{ background:#FFF6E0; border-left:6px solid {AMBER}; padding:0.8rem 1rem; border-radius:6px; margin:0.4rem 0 0.8rem; }}
</style>""", unsafe_allow_html=True)


def callout(text, kind="callout"):
    st.markdown(f'<div class="{kind}">{text}</div>', unsafe_allow_html=True)


def layout(fig, h=420, **kw):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Calibri, Arial", size=13), **kw)
    return fig


# ----------------------------------------------------------------------------- cached steps
@st.cache_data(show_spinner=False)
def field():
    logs, stages = gc.make_field()
    meas, core, dfit = gc.make_measurements(logs)
    pub = meas[gc.PUBLIC_LOGS + [c for c in meas if c.startswith("_")]]
    return pub, stages, core, dfit


@st.cache_data(show_spinner=False)
def elastic(static_m, ucs_m):
    logs, _, core, _ = field()
    return gc.add_elastic(logs, core, static_m, ucs_m)


@st.cache_data(show_spinner=False)
def indices(rock):
    static_m, ucs_m, eps_t = rock
    L, _ = elastic(static_m, ucs_m)
    return gc.add_fracability(gc.add_indices(L, eps_t))


@st.cache_data(show_spinner="Clustering the laterals...")
def facies(eps_t, features, k, smooth):
    return gc.cluster_facies(indices(eps_t), list(features), k, smooth)   # eps_t carries the full rock-mechanics key


@st.cache_data(show_spinner="Scanning K from 3 to 6...")
def elbow(eps_t, features):
    return gc.elbow_table(indices(eps_t), list(features))


@st.cache_data(show_spinner=False)
def stage_table(eps_t, features, k, smooth):
    stages = field()[1]
    return gc.build_stage_table(indices(eps_t), stages, facies(eps_t, features, k, smooth), k)


@st.cache_data(show_spinner="Training with leave-one-well-out validation...")
def cv(eps_t, features, k, smooth, model_feats, params):
    T = stage_table(eps_t, features, k, smooth)
    return {s: gc.cross_validate(T, list(model_feats), dict(params), s) for s in ("shuffled", "lowo")}


@st.cache_resource(show_spinner=False)
def final_model(eps_t, features, k, smooth, model_feats, params):
    T = stage_table(eps_t, features, k, smooth)
    return gc.make_model(**dict(params)).fit(T[list(model_feats)], T.TARGET_BOPD_PER_100FT)


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown('<p class="brand">DIGITALOILED.AI</p>', unsafe_allow_html=True)
    st.markdown("**Case study · 30 min**  \nGeomechanical features controlling fracture generation")
    st.caption("Synthetic Permian-analog data · six horizontal wells")
    st.divider()
    code = st.text_input("Trainer code", type="password", help="Reveals the synthetic truth and answer notes.")
    try:
        expected = st.secrets.get("TRAINER_CODE", "geomech")
    except Exception:
        expected = "geomech"
    TRAINER = code == expected
    if TRAINER:
        st.success("Trainer view on")
    st.divider()
    st.caption("Rock-mechanics choices (Task 1), facies settings (Task 2) and model settings (Task 3) flow through to Tasks 4 and 5.")
    st.caption("Version 1.1")

logs_raw, stages, core, dfit = field()
W = gc.WELLS.set_index("WELL")

st.markdown('<p class="brand">ADVANCED AI FOR RESERVOIR PETROPHYSICS · UNCONVENTIONAL CONTEXT</p>', unsafe_allow_html=True)
st.title("Geomechanical facies and stage performance")

tabs = st.tabs(["Brief", "1 · Rock mechanics", "2 · Facies", "3 · Quality and model", "4 · SHAP", "5 · Stage design", "Debrief"])

# ============================================================================= BRIEF
with tabs[0]:
    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.subheader("The challenge")
        st.write("Tight sands and shales need massive hydraulic fracturing to be economic, yet one-size-fits-all "
                 "completion designs leave value behind. The rock heterogeneity that petrophysicists characterise "
                 "every day controls where fractures initiate, how fluid divides between clusters and, in the end, "
                 "how much each stage produces.")
        st.subheader("Objective")
        callout("Build an AI workflow that relates geomechanical rock properties to stage-level completion "
                "performance in a Permian Basin tight reservoir, then test whether a facies-based stage design "
                "beats geometric spacing.")
        st.subheader("Dataset")
        st.write("Six horizontal wells, sampled every 5 ft along each lateral, and the measurements that calibrate them:")
        st.markdown("| Dataset | Role |\n|---|---|\n"
                    "| Bulk density (RHOB) and dipole sonic (DTC, DTS) | inputs: dynamic elastic properties |\n"
                    "| XRD-calibrated mineralogy: quartz, carbonate, clay and TOC | inputs |\n"
                    "| Core triaxial tests, 20 plugs in PB-2H and PB-6H: static E, ν, UCS | calibration labels |\n"
                    "| One DFIT closure pressure per well | stress calibration |\n"
                    "| Completion records: stage length and spacing, clusters, cluster spacing, proppant, fluid, pump rate | inputs |\n"
                    "| Production-log stage contributions (%) and initial well rate (bopd) | labels: what the model learns |")
    with c2:
        st.subheader("Plan · 30 minutes")
        st.table(pd.DataFrame({
            "Min": ["0–3", "3–10", "10–14", "14–20", "20–24", "24–30"],
            "Step": ["Brief and data walk-through", "Task 1 · Rock mechanics: moduli, UCS, stress, brittleness",
                     "Task 2 · Geomechanical facies", "Task 3 · Fracability, completion quality and the model",
                     "Task 4 · SHAP drivers", "Task 5 · Stage design, production gain and debrief"],
        }).set_index("Min"))
        st.subheader("Targets")
        st.markdown("UCS · Young's modulus · Poisson's ratio · brittleness · fracability index · geomechanical facies · "
                    "completion quality score · stage production contribution · expected production gain")
        st.caption("Work in your Exercise 1 teams. The Petrophysics reviewer challenges every choice.")

    st.subheader("Wells")
    summary = stages.groupby("WELL").agg(STAGES=("STAGE", "size"), WELL_IP_BOPD=("WELL_IP_BOPD", "first"))
    wt = W.drop(columns=["LANDING"] if not TRAINER else []).join(summary)
    st.dataframe(wt, width="stretch")
    d1, d2, d3, d4 = st.columns(4)
    d1.download_button("Lateral logs (CSV)", logs_raw[[c for c in logs_raw if not c.startswith("_")]].to_csv(index=False),
                       "permian_analog_lateral_logs.csv", "text/csv")
    d2.download_button("Stage records (CSV)", stages[[c for c in stages if not c.startswith("_")]].to_csv(index=False),
                       "permian_analog_stages.csv", "text/csv")
    d3.download_button("Core tests (CSV)", core.to_csv(index=False), "permian_analog_core_tests.csv", "text/csv")
    d4.download_button("DFIT closures (CSV)", dfit.to_csv(index=False), "permian_analog_dfit.csv", "text/csv")
    if TRAINER:
        callout("<b>Trainer note.</b> Four hidden facies generate the logs: siliceous brittle, carbonate-cemented, "
                "clay-rich ductile and organic-rich mudstone. Landing zones differ by well (shown in the table). "
                "The stage simulator uses elastic brittleness, clay, TOC, cluster stress contrast, proppant per cluster, "
                "pump rate and cluster spacing. <b>Fluid volume and UCS play no part in it</b>; watch for fluid in Task 4. "
                "The logged Young's modulus is dynamic: the true static/dynamic ratio falls from about 0.8 in clean rock to 0.55 in clay-rich rock.", "trainer")

# ============================================================================= TASK 1
with tabs[1]:
    st.subheader("Task 1 · Rock mechanics from density and sonic, calibrated to core and DFIT")
    st.markdown(r"""
**Dynamic moduli** from RHOB, DTC and DTS: $\nu_d = \dfrac{V_p^2 - 2V_s^2}{2(V_p^2 - V_s^2)}$, $E_d = \rho V_s^2 \dfrac{3V_p^2 - 4V_s^2}{V_p^2 - V_s^2}$.
Fractures open against **static** properties, which are lower than dynamic ones in soft rock. Choose how to convert, and how to estimate UCS.
""")
    r1, r2 = st.columns(2)
    static_m = r1.radio("Static Young's modulus", gc.STATIC_METHODS, index=2, help="Eissa and Kazi (1988): Es = 0.74 Ed − 0.82 (GPa). Calibrated: linear fit of core static E on log dynamic E.")
    ucs_m = r2.radio("UCS", gc.UCS_METHODS, index=2, help="Horsrud (2001) shale correlations; calibrated: power law fitted to the core UCS.")
    Le, fits = elastic(static_m, ucs_m)
    cc = core.merge(Le[["WELL", "MD_FT", "YM_DYN", "YM_MPSI", "PR", "UCS_PSI"]], on=["WELL", "MD_FT"], suffixes=("", "_LOG"))
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Static E bias against core", f"{(cc.YM_MPSI - cc.STATIC_YM_MPSI).mean():+.2f} Mpsi")
    k2.metric("Poisson's ratio bias", f"{(cc.PR - cc.STATIC_PR).mean():+.3f}")
    k3.metric("UCS error against core (median)", f"{np.median(np.abs(cc.UCS_PSI_LOG / cc.UCS_PSI - 1)):.0%}")
    k4.metric("Core tests", f"{len(core)} in {core.WELL.nunique()} wells")
    cfig = make_subplots(rows=1, cols=2, subplot_titles=["Young's modulus: log against core (Mpsi)", "UCS: log against core (psi)"])
    cfig.add_scatter(x=cc.STATIC_YM_MPSI, y=cc.YM_DYN, mode="markers", marker=dict(color=MUTED, size=7), name="dynamic", row=1, col=1)
    cfig.add_scatter(x=cc.STATIC_YM_MPSI, y=cc.YM_MPSI, mode="markers", marker=dict(color=TEAL, size=8), name="static, your choice", row=1, col=1)
    cfig.add_scatter(x=cc.UCS_PSI, y=cc.UCS_PSI_LOG, mode="markers", marker=dict(color=AMBER, size=8), name="UCS, your choice", row=1, col=2)
    for col, mx in [(1, float(cc.YM_DYN.max()) * 1.05), (2, float(cc.UCS_PSI.max()) * 1.1)]:
        cfig.add_scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(color=MUTED, dash="dot"), showlegend=False, row=1, col=col)
    st.plotly_chart(layout(cfig, 330, legend=dict(orientation="h", y=-0.2)), width="stretch")

    st.markdown(r"""**Closure stress:** $S_{hmin} = \dfrac{\nu}{1-\nu}(S_v - P_p) + P_p + E\,\varepsilon_t$. The tectonic strain $\varepsilon_t$ is unknown; one DFIT per well measures closure.""")
    eps_fit, dfit_tab = gc.fit_eps_t(Le, dfit)
    t1, t2 = st.columns([1, 1.4])
    eps_mode = t1.radio("Tectonic strain", ["Fit to the DFITs", "Set by hand"], index=0)
    if eps_mode.startswith("Fit"):
        eps_t = eps_fit
        t1.metric("Fitted εt", f"{eps_fit * 1e4:.2f} × 10⁻⁴")
    else:
        eps_t = t1.slider("Tectonic strain εt (×10⁻⁴)", 0.0, 2.5, 1.2, 0.05) * 1e-4
    _, dfit_now = gc.fit_eps_t(Le, dfit)
    wcal = gc.WELLS.set_index("WELL")
    dfit_now["PREDICTED_PSI"] = gc.shmin_uniaxial(dfit_now.PR, dfit_now.YM_MPSI, dfit_now.TVD_FT, dfit_now.WELL.map(wcal.PP_GRAD), eps_t)
    dfit_now["MISFIT_PSI"] = dfit_now.PREDICTED_PSI - dfit_now.CLOSURE_PSI
    t1.metric("DFIT misfit (RMS)", f"{np.sqrt((dfit_now.MISFIT_PSI ** 2).mean()):,.0f} psi")
    t2.dataframe(dfit_now[["WELL", "MD_FT", "CLOSURE_PSI", "PREDICTED_PSI", "MISFIT_PSI"]].round(0).set_index("WELL"), width="stretch")
    st.markdown("**Q1.1** Compare the static conversions against the core. What does using dynamic moduli as static do to "
                "the fitted εt and to the DFIT misfit?")
    if TRAINER:
        callout("<b>Answers.</b> Dynamic as static overstates E by about 2.1 Mpsi; Eissa and Kazi by about 0.2; the core "
                "calibration is unbiased. The DFIT fit hides part of a wrong conversion: εt drops to about 0.65 × 10⁻⁴ with "
                "dynamic moduli and 0.9 with Eissa and Kazi, against 1.25 with the core calibration (truth 1.2), and the "
                "RMS misfit roughly doubles (about 135 against 75 psi). Calibrating one parameter to stress data can absorb "
                "an error made elsewhere. For UCS, Horsrud's sonic correlation is close on average; the core fit tightens it. "
                "UCS does not drive stage rate here: it matters for perforating, proppant embedment and wellbore stability.", "trainer")
    rock = (static_m, ucs_m, float(eps_t))
    st.divider()
    st.markdown("#### Brittleness along each lateral")
    st.markdown(r"""
**Mineralogical (Jarvie-type):** $BI_{min} = \dfrac{Qz}{Qz + Carb + Clay}$
**Elastic (Rickman):** $BI_{el} = \tfrac{1}{2}\left(\dfrac{E - E_{min}}{E_{max} - E_{min}} + \dfrac{\nu_{max} - \nu}{\nu_{max} - \nu_{min}}\right)$, with E from 1 to 8 Mpsi and ν from 0.15 to 0.40
**Fracability index:** $FI = \tfrac{1}{2}BI_{el} + \tfrac{1}{2}(1 - S_{hmin,n})$, with closure stress normalised between the P5 and P95 of the lateral
""")
    eps_t = rock                      # the rock-mechanics key carries static method, UCS method and εt downstream
    L = indices(eps_t)
    well = st.selectbox("Well", W.index, key="w1")
    g = L[L.WELL == well]
    st_w = stages[stages.WELL == well]

    fig = make_subplots(rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        row_heights=[1, 1, 1.2, 1, 0.9, 0.9],
                        subplot_titles=["Young's modulus, static (Mpsi)", "Poisson's ratio", "Brittleness and fracability index",
                                        "Shmin (psi)", "UCS (kpsi)", "PLT stage contribution (%)"])
    fig.add_scatter(x=g.MD_FT, y=g.YM_MPSI, line=dict(color=NAVY, width=1), name="E", row=1, col=1)
    fig.add_scatter(x=g.MD_FT, y=g.PR, line=dict(color=NAVY, width=1), name="ν", row=2, col=1)
    fig.add_scatter(x=g.MD_FT, y=g.BI_MIN.rolling(5, center=True).mean(), line=dict(color=AMBER, width=1.4), name="BI mineral", row=3, col=1)
    fig.add_scatter(x=g.MD_FT, y=g.BI_EL.rolling(5, center=True).mean(), line=dict(color=TEAL, width=1.4), name="BI elastic", row=3, col=1)
    fig.add_scatter(x=g.MD_FT, y=g.FI.rolling(5, center=True).mean(), line=dict(color=NAVY, width=1.4, dash="dot"), name="Fracability", row=3, col=1)
    fig.add_scatter(x=g.MD_FT, y=g.SHMIN_PSI, line=dict(color="#B5533C", width=1), name="Shmin", row=4, col=1)
    fig.add_scatter(x=g.MD_FT, y=g.UCS_PSI / 1000, line=dict(color=MUTED, width=1), name="UCS", row=5, col=1)
    dw = dfit[dfit.WELL == well]
    fig.add_scatter(x=dw.MD_FT, y=dw.CLOSURE_PSI, mode="markers", marker=dict(color=NAVY, size=10, symbol="diamond"), name="DFIT closure", row=4, col=1)
    fig.add_bar(x=(st_w.TOP_MD + st_w.BASE_MD) / 2, y=st_w.PLT_CONTRIB_PCT, width=st_w.STAGE_LEN_FT * 0.9,
                marker_color=TEAL, name="PLT %", row=6, col=1)
    for t in st_w.TOP_MD:
        fig.add_vline(x=t, line=dict(color="#DCD8CE", width=0.6), row=6, col=1)
    st.plotly_chart(layout(fig, 940, showlegend=True, legend=dict(orientation="h", y=1.05)), width="stretch")

    r = np.corrcoef(L.BI_MIN, L.BI_EL)[0, 1]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("r (BI mineral, BI elastic), all wells", f"{r:.2f}")
    m2.metric(f"{well} median BI elastic", f"{g.BI_EL.median():.2f}")
    m3.metric(f"{well} median BI mineral", f"{g.BI_MIN.median():.2f}")
    m4.metric(f"{well} Shmin P10–P90 spread", f"{g.SHMIN_PSI.quantile(.9) - g.SHMIN_PSI.quantile(.1):,.0f} psi")

    sc = go.Figure(go.Scattergl(x=L.BI_MIN, y=L.BI_EL, mode="markers",
                                marker=dict(size=3, color=L.CLAY, colorscale="Cividis", colorbar=dict(title="Clay"), opacity=0.5)))
    sc.update_xaxes(title="BI mineral").update_yaxes(title="BI elastic")
    st.plotly_chart(layout(sc, 380, title="Two indices, two measurements"), width="stretch")
    st.markdown("**Q1.2** Where do the two brittleness indices disagree most, and what rock sits there?  \n"
                "**Q1.3** Which index would you trust to predict where a cluster takes fluid, and why?")
    if TRAINER:
        callout("<b>Answers.</b> They disagree in the carbonate-cemented rock: quartz-poor, so mineral BI is low, but "
                "stiff, so elastic BI is high. Its high Poisson's ratio and stiffness also raise Shmin by 800 to 900 psi "
                "over siliceous rock, so clusters placed there take little fluid. Elastic BI and Shmin both come from "
                "the same dipole sonic, which is why stress, not brittleness alone, decides fluid entry. "
                f"The correlation of r = {r:.2f} is low by design: these are different measurements of different things.", "trainer")

# ============================================================================= TASK 2
with tabs[2]:
    st.subheader("Task 2 · Cluster lateral sections into geomechanical facies with k-means")
    all_feats = ["YM_MPSI", "PR", "BI_MIN", "BI_EL", "FI", "UCS_PSI", "CLAY", "TOC", "QUARTZ", "CARBONATE", "SHMIN_PSI"]
    c1, c2, c3 = st.columns([2, 1, 1])
    feats = tuple(c1.multiselect("Clustering inputs (standardised)", all_feats,
                                 default=["YM_MPSI", "PR", "BI_MIN", "CLAY", "TOC"]))
    K = c2.slider("Number of facies K", 3, 6, 4)
    smooth = c3.select_slider("Minimum bed smoothing (ft)", [0, 15, 25, 45], value=25,
                              help="Mode filter along each lateral. Removes single-sample facies flicker.")
    if len(feats) < 2:
        st.warning("Choose at least two inputs.")
        st.stop()
    if "BI_EL" in feats and ("YM_MPSI" in feats or "PR" in feats):
        st.info("BI elastic is built from E and ν. Including both counts the elastic signal twice.")

    el = elbow(eps_t, feats)
    lab = facies(eps_t, feats, K, smooth)
    L2 = L.assign(GF=lab)

    e1, e2 = st.columns([1, 1.6])
    with e1:
        fe = make_subplots(specs=[[{"secondary_y": True}]])
        fe.add_scatter(x=el.K, y=el.Inertia, name="Inertia per sample", line=dict(color=NAVY))
        fe.add_scatter(x=el.K, y=el.Silhouette, name="Silhouette", line=dict(color=AMBER), secondary_y=True)
        st.plotly_chart(layout(fe, 320, title="Elbow and silhouette", legend=dict(orientation="h", y=-0.2)), width="stretch")
        st.dataframe(el.set_index("K").round(3), width="stretch")
    with e2:
        prof = L2.groupby("GF").agg(SHARE=("GF", "size"), E=("YM_MPSI", "mean"), PR=("PR", "mean"), BI_EL=("BI_EL", "mean"),
                                    BI_MIN=("BI_MIN", "mean"), CLAY=("CLAY", "mean"), TOC=("TOC", "mean"), SHMIN=("SHMIN_PSI", "mean"))
        prof["SHARE"] = prof.SHARE / prof.SHARE.sum()
        prof.index = [f"GF{i + 1}" for i in prof.index]
        st.markdown("**Facies profiles** (GF1 is the most elastically brittle)")
        st.dataframe(prof.style.format({"SHARE": "{:.0%}", "E": "{:.2f}", "PR": "{:.3f}", "BI_EL": "{:.2f}",
                                        "BI_MIN": "{:.2f}", "CLAY": "{:.2f}", "TOC": "{:.1f}", "SHMIN": "{:,.0f}"}),
                     width="stretch")
        cx = go.Figure()
        for f in range(K):
            s = L2[L2.GF == f].sample(min(600, (L2.GF == f).sum()), random_state=0)
            cx.add_scattergl(x=s.PR, y=s.YM_MPSI, mode="markers", name=f"GF{f + 1}",
                             marker=dict(size=4, color=GF_COLOURS[f], opacity=0.6))
        cx.update_xaxes(title="Poisson's ratio").update_yaxes(title="Young's modulus (Mpsi)")
        st.plotly_chart(layout(cx, 330, title="Facies in elastic space"), width="stretch")

    well2 = st.selectbox("Well", W.index, key="w2")
    g2 = L2[L2.WELL == well2]
    ft = make_subplots(rows=3 if TRAINER else 2, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                       row_heights=[1, 0.35, 0.35][: 3 if TRAINER else 2],
                       subplot_titles=["BI elastic and Shmin", "Geomechanical facies (yours)"] + (["Hidden truth"] if TRAINER else []))
    ft.add_scatter(x=g2.MD_FT, y=g2.BI_EL.rolling(5, center=True).mean(), line=dict(color=TEAL, width=1.2), name="BI elastic", row=1, col=1)
    ft.add_heatmap(x=g2.MD_FT, z=[g2.GF], colorscale=[[i / max(K - 1, 1), GF_COLOURS[i]] for i in range(K)],
                   showscale=False, hovertemplate="MD %{x}<br>GF%{z}<extra></extra>", row=2, col=1)
    if TRAINER:
        ft.add_heatmap(x=g2.MD_FT, z=[g2._FACIES], colorscale=[[0, "#4B6E74"], [0.33, "#C9A35A"], [0.66, "#B5533C"], [1, "#6A8D3A"]],
                       showscale=False, row=3, col=1)
    for t in stages[stages.WELL == well2].TOP_MD:
        ft.add_vline(x=t, line=dict(color="#BBBBBB", width=0.5), row=1, col=1)
    ft.update_yaxes(showticklabels=False, row=2, col=1)
    st.plotly_chart(layout(ft, 420 if TRAINER else 340, showlegend=False), width="stretch")
    mix = (stage_table(eps_t, feats, K, smooth).FACIES_PURITY < 0.7).mean()
    st.metric("Geometric stages that straddle two or more facies (purity < 70%)", f"{mix:.0%}")
    st.markdown("**Q2.1** Why that number of facies? Give a geomechanical reason, not only a statistical one.  \n"
                "**Q2.2** Thin lines mark the as-completed stage boundaries. What happens to the clusters in a stage that straddles two facies?")
    if TRAINER:
        ari = gc.adjusted_rand_score(L._FACIES, lab)
        callout(f"<b>Trainer view.</b> Agreement with the hidden facies (adjusted Rand index): <b>{ari:.2f}</b>. "
                "Default inputs with K = 4 and 25 ft smoothing recover them well. The silhouette prefers K = 3 because "
                "siliceous and organic-rich rock overlap in elastic space; TOC and Shmin separate them, and they behave "
                "differently in the simulator (organic-rich is better reservoir, siliceous fractures more easily). "
                "That is the geomechanical case for K = 4. Mixed stages put some clusters in high-stress rock, where "
                "they take little fluid because the lowest-stress cluster breaks down first.", "trainer")

# ============================================================================= TASK 3
with tabs[3]:
    st.subheader("Task 3 · Fracability, completion quality and a gradient-boosted model of stage contribution")
    T = stage_table(eps_t, feats, K, smooth)
    st.markdown(r"""**Completion quality score (CQS, 0 to 100)** for each stage: mean fracability at the clusters × fluid-entry uniformity.
Fluid enters the lowest-stress clusters first, each taking a share $w_i \propto e^{-(S_i - S_{min})/150\,psi}$; uniformity is the effective number of clusters $(\sum w)^2/\sum w^2$ over the number placed.
Before any model, test the physics indices against the production log: rank correlation with stage rate, within each well.""")
    yv = T.TARGET_BOPD_PER_100FT
    from scipy.stats import spearmanr
    idx_rows = []
    for c, lab_ in [("CQS", "Completion quality score"), ("CLUSTER_FI", "Fracability at the clusters"), ("FI_MEAN", "Fracability, stage mean"),
                    ("BI_EL_MEAN", "Elastic brittleness"), ("BI_MIN_MEAN", "Mineral brittleness"), ("TOC_MEAN", "TOC"), ("UCS_MEAN", "UCS")]:
        within = T.groupby("WELL").apply(lambda d: spearmanr(d[c], d.TARGET_BOPD_PER_100FT, nan_policy="omit").correlation, include_groups=False)
        idx_rows.append({"Index": lab_, "ρ, all stages": spearmanr(T[c], yv, nan_policy="omit").correlation,
                         "ρ within well (median)": within.median()})
    st.dataframe(pd.DataFrame(idx_rows).set_index("Index").style.format("{:.2f}"), width="stretch")
    st.markdown("The target is stage rate per 100 ft of lateral: PLT contribution × well initial rate ÷ stage length. "
                "This removes the well-level rate, so stages in different wells can be compared.")
    geo_cols = [c for c in T.columns if c.startswith("FRAC_GF")] + ["FACIES_PURITY", "YM_MEAN", "PR_MEAN", "BI_MIN_MEAN",
                "BI_EL_MEAN", "CLAY_MEAN", "TOC_MEAN", "SHMIN_MEAN", "SHMIN_RANGE", "CLUSTER_STRESS_SPREAD", "CLUSTER_BI_EL",
                "FI_MEAN", "CLUSTER_FI", "CQS", "UCS_MEAN"]
    comp_cols = ["CLUSTERS", "STAGE_LEN_FT", "CLUSTER_SPACING_FT", "PROPPANT_LB_FT", "FLUID_BBL_FT", "RATE_BPM"]
    c1, c2 = st.columns(2)
    gsel = c1.multiselect("Geomechanical features", geo_cols, default=geo_cols)
    csel = c2.multiselect("Completion features", comp_cols, default=comp_cols)
    mf = tuple(gsel + csel)
    p1, p2, p3 = st.columns(3)
    params = (("n_estimators", p1.slider("Trees", 100, 600, 300, 50)),
              ("max_depth", p2.slider("Max depth", 2, 6, 3)),
              ("learning_rate", p3.select_slider("Learning rate", [0.02, 0.05, 0.1], 0.05)))
    if len(mf) < 2:
        st.warning("Choose at least two features.")
        st.stop()
    res = cv(eps_t, feats, K, smooth, mf, params)
    (ps, r2s, maes, _), (pl, r2l, mael, pw) = res["shuffled"], res["lowo"]
    rho_ml = T.assign(P=pl).groupby("WELL").apply(lambda d: spearmanr(d.P, d.TARGET_BOPD_PER_100FT).correlation, include_groups=False).median()
    st.caption(f"Leave-one-well-out model predictions: ρ within well (median) {rho_ml:.2f}, for comparison with the indices above.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Shuffled 5-fold R²", f"{r2s:.2f}", help="Stages from the same well land in train and test.")
    m2.metric("Leave-one-well-out R²", f"{r2l:.2f}", delta=f"{r2l - r2s:+.2f} vs shuffled", delta_color="off")
    m3.metric("Leave-one-well-out MAE", f"{mael:.1f} bopd/100 ft")
    m4.metric("Stages", f"{len(T)}")
    a1, a2 = st.columns([1.4, 1])
    with a1:
        cp = go.Figure()
        for i, w in enumerate(W.index):
            s = T.WELL == w
            cp.add_scatter(x=T.TARGET_BOPD_PER_100FT[s], y=pl[s], mode="markers", name=w,
                           marker=dict(size=6, color=GF_COLOURS[i % 6], opacity=0.8))
        mx = float(max(T.TARGET_BOPD_PER_100FT.max(), pl.max()))
        cp.add_scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(color=MUTED, dash="dot"), showlegend=False)
        cp.update_xaxes(title="PLT-derived rate (bopd per 100 ft)").update_yaxes(title="Predicted, well held out")
        st.plotly_chart(layout(cp, 420, title="Leave-one-well-out predictions"), width="stretch")
    with a2:
        st.markdown("**Blind score by well**")
        st.dataframe(pw.round(2), width="stretch")
        st.caption("Six wells means six blind tests. Quote the range, not only the pooled number.")
    st.markdown("**Q3.0** Which index ranks stages best? Does adding fluid-entry uniformity to fracability help, and what does that say about this field?  \n"
                "**Q3.1** Why is the shuffled score higher, and which number do you give the completions engineer?  \n"
                "**Q3.2** Each well was completed with one design. What does that do to the model's ability to "
                "separate a completion effect from a geology effect?")
    if TRAINER:
        callout("<b>Answers.</b> Stages from one well share landing zone, design and pressure, so shuffled folds leak "
                "well identity into training. Quote leave-one-well-out with the per-well range. Completion parameters "
                "vary mostly between wells, and only six wells exist, so proppant, fluid and rate are confounded with "
                "landing zone. The model can learn which wells did well, not why. Expect this to show up in SHAP. "
                "<b>Indices:</b> fracability ranks stages at ρ about 0.8 within wells; CQS at about 0.6, because rock quality at the "
                "clusters matters more here than how evenly fluid divides; elastic brittleness alone about 0.3; UCS about 0. "
                "The model beats every index (about 0.9) because it also sees TOC. Adding FI and CQS to the model changes "
                "its score by about 0.01: a tree builds its own combinations of the same logs. An index is a hypothesis; the PLT tests it.", "trainer")

# ============================================================================= TASK 4
with tabs[4]:
    st.subheader("Task 4 · SHAP: which rock properties control stage performance?")
    model = final_model(eps_t, feats, K, smooth, mf, params)
    X = T[list(mf)]
    sv, base = gc.shap_values(model, X)
    imp = pd.Series(np.abs(sv).mean(0), index=mf).sort_values()
    top = imp.index[::-1][:12]
    b1, b2 = st.columns([1, 1.5])
    with b1:
        bar = go.Figure(go.Bar(x=imp[top[::-1]], y=list(top[::-1]), orientation="h",
                               marker_color=[TEAL if f in geo_cols else AMBER for f in top[::-1]]))
        bar.update_xaxes(title="Mean |SHAP| (bopd per 100 ft)")
        st.plotly_chart(layout(bar, 460, title="Global importance · teal rock, amber completion"), width="stretch")
    with b2:
        bee = go.Figure()
        rng = np.random.default_rng(0)
        for j, f in enumerate(top[::-1]):
            col = list(mf).index(f)
            v = X[f].to_numpy()
            vn = (v - np.nanmin(v)) / (np.nanmax(v) - np.nanmin(v) + 1e-9)
            bee.add_scattergl(x=sv[:, col], y=j + rng.uniform(-0.3, 0.3, len(v)), mode="markers", showlegend=False,
                              marker=dict(size=4, color=vn, colorscale=[[0, TEAL], [0.5, "#DDDDDD"], [1, AMBER]],
                                          showscale=j == 0, colorbar=dict(title="Value", tickvals=[0, 1], ticktext=["low", "high"])),
                              hovertext=[f"{f} = {x:.3g}" for x in v])
        bee.update_yaxes(tickvals=list(range(len(top))), ticktext=list(top[::-1]))
        bee.update_xaxes(title="SHAP value (bopd per 100 ft)")
        bee.add_vline(x=0, line=dict(color=MUTED, width=1))
        st.plotly_chart(layout(bee, 460, title="Per-stage contributions"), width="stretch")
    dsel = st.selectbox("Dependence plot", list(imp.index[::-1]), index=0)
    col = list(mf).index(dsel)
    dp = go.Figure(go.Scattergl(x=X[dsel], y=sv[:, col], mode="markers",
                                marker=dict(size=6, color=T.WELL.map({w: GF_COLOURS[i] for i, w in enumerate(W.index)}), opacity=0.75),
                                hovertext=T.WELL + " · stage " + T.STAGE.astype(str)))
    dp.update_xaxes(title=dsel).update_yaxes(title=f"SHAP for {dsel}")
    st.plotly_chart(layout(dp, 360, title="Coloured by well"), width="stretch")
    st.markdown("**Q4.1** Which rock properties lead, and does the direction of each make physical sense?  \n"
                "**Q4.2** Find a completion parameter with high importance. Is it causing better stages, or riding on "
                "something else? Use the dependence plot coloured by well.")
    if TRAINER:
        rank = list(imp.index[::-1])
        fluid_rank = rank.index("FLUID_BBL_FT") + 1 if "FLUID_BBL_FT" in rank else None
        callout("<b>Answers.</b> Rock quality and brittleness lead (TOC, brittleness, Poisson's ratio), with the "
                "ductile-facies fraction pushing stages down. "
                + (f"FLUID_BBL_FT ranks <b>#{fluid_rank}</b>, yet the simulator never uses fluid volume. " if fluid_rank else "")
                + "Its dependence plot shows six vertical stripes, one per well: it is a well label in disguise, carried by "
                "landing zone. SHAP explains the model, not the reservoir. Ask teams to drop fluid and watch "
                "the leave-one-well-out score barely move.", "trainer")

# ============================================================================= TASK 5
with tabs[5]:
    st.subheader("Task 5 · Facies-based stage design against geometric spacing")
    st.markdown("Keep each well's proppant intensity, pump rate and total cluster count. Place stage breaks at facies "
                "boundaries, merge thin beds into their nearest-stress neighbour, and put clusters in the lowest-stress rock of each stage.")
    c1, c2, c3, c4 = st.columns(4)
    min_len = c1.slider("Minimum stage length (ft)", 80, 200, 120, 10)
    max_len = c2.slider("Maximum stage length (ft)", 200, 400, 300, 20)
    min_sp = c3.slider("Minimum cluster spacing (ft)", 15, 40, 20, 5)
    skip = c4.checkbox("Skip stages that are > 80% the least brittle facies", value=False)
    ductile = K - 1 if skip else None

    rows, designs = [], {}
    for wname, g in L2.groupby("WELL", sort=False):
        g = g.reset_index(drop=True)
        w = W.loc[wname]
        stw = stages[stages.WELL == wname]
        geo = [(r.TOP_MD, r.BASE_MD, np.array([float(x) for x in r.CLUSTER_MD.split(";")])) for r in stw.itertuples()]
        tc = sum(len(c) for *_, c in geo)
        fd = gc.facies_design(g, g.GF.to_numpy(), g.SHMIN_PSI.to_numpy(), tc, min_len, max_len, min_sp, ductile)
        comp = dict(PROPPANT_LB_FT=w.PROPPANT_LB_FT, FLUID_BBL_FT=w.FLUID_BBL_FT, RATE_BPM=w.RATE_BPM)
        Fg = gc.stage_features(g, geo, g.GF.to_numpy(), K, comp)
        Ff = gc.stage_features(g, fd, g.GF.to_numpy(), K, comp)
        Ff_act = Ff[Ff.CLUSTERS > 0]
        pg = (model.predict(Fg[list(mf)]) * Fg.STAGE_LEN_FT / 100).sum()
        pf = (model.predict(Ff_act[list(mf)]) * Ff_act.STAGE_LEN_FT / 100).sum()
        tg = gc.simulate_design(g, geo, w.PROPPANT_LB_FT, w.RATE_BPM, noise=False).sum()
        tf = gc.simulate_design(g, fd, w.PROPPANT_LB_FT * (Fg.STAGE_LEN_FT.sum() / max(Ff_act.STAGE_LEN_FT.sum(), 1)),
                                w.RATE_BPM, noise=False).sum()
        key = [c for c in ("FACIES_PURITY", "CLUSTER_STRESS_SPREAD", "STAGE_LEN_FT", "CLUSTER_SPACING_FT") if c in mf]
        lo, hi = T[key].min(), T[key].max()
        ood = ((Ff_act[key] < lo) | (Ff_act[key] > hi)).any(axis=1).mean() if key else 0.0
        designs[wname] = (g, geo, fd)
        ip = stw.WELL_IP_BOPD.iloc[0]
        rows.append({"WELL": wname, "Geometric stages": len(geo), "Facies stages": int((Ff.CLUSTERS > 0).sum()),
                     "CQS geometric": Fg.CQS.mean(), "CQS facies": Ff_act.CQS.mean(),
                     "Well IP (bopd)": ip, "Expected gain (bopd)": ip * (pf / pg - 1),
                     "Clusters": tc, "Clusters placed": int(Ff.CLUSTERS.sum()),
                     "Purity geometric": Fg.FACIES_PURITY.mean(), "Purity facies": Ff_act.FACIES_PURITY.mean(),
                     "Cluster stress spread geo (psi)": Fg.CLUSTER_STRESS_SPREAD.mean(),
                     "Cluster stress spread facies (psi)": Ff_act.CLUSTER_STRESS_SPREAD.mean(),
                     "Model uplift": pf / pg - 1, "Outside training range": ood, "_true": tf / tg - 1})
    R = pd.DataFrame(rows).set_index("WELL")
    e1, e2 = st.columns(2)
    netback = e1.number_input("Netback ($/bbl)", 20.0, 90.0, 45.0, 5.0)
    yr1 = e2.number_input("First-year oil as days at initial rate", 150, 365, 200, 10,
                          help="A decline assumption: 200 means first-year oil equals 200 days at the initial rate.")
    R["First-year value ($M)"] = R["Expected gain (bopd)"] * yr1 * netback / 1e6
    v1, v2 = st.columns(2)
    v1.metric("Expected production gain, six wells", f"{R['Expected gain (bopd)'].sum():+,.0f} bopd")
    v2.metric("First-year value at your assumptions", f"${R['First-year value ($M)'].sum():+,.1f} M")
    m1, m2, m3 = st.columns(3)
    m1.metric("Mean model-predicted uplift", f"{R['Model uplift'].mean():+.0%}")
    m2.metric("Facies stages outside the training range", f"{R['Outside training range'].mean():.0%}",
              help="Stage purity, cluster stress spread, stage length or cluster spacing beyond anything in the training stages.")
    m3.metric("Mean cluster stress spread", f"{R['Cluster stress spread facies (psi)'].mean():,.0f} psi",
              delta=f"{R['Cluster stress spread facies (psi)'].mean() - R['Cluster stress spread geo (psi)'].mean():,.0f} vs geometric",
              delta_color="inverse")
    show = R.drop(columns=["_true"])
    if TRAINER:
        show["Simulator uplift (truth)"] = R["_true"]
    st.dataframe(show.style.format({c: "{:.0%}" for c in ["Purity geometric", "Purity facies", "Model uplift",
                                                          "Outside training range", "Simulator uplift (truth)"] if c in show}
                                   | {c: "{:,.0f}" for c in show if "psi" in c or "bopd" in c}
                                   | {"CQS geometric": "{:.0f}", "CQS facies": "{:.0f}", "First-year value ($M)": "{:+.2f}"}), width="stretch")

    well5 = st.selectbox("Compare designs along a lateral", W.index, key="w5")
    g, geo, fd = designs[well5]
    fz = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[1, 0.3, 0.6],
                       subplot_titles=["Shmin (psi) with cluster positions", "Facies", "Stage boundaries"])
    fz.add_scatter(x=g.MD_FT, y=g.SHMIN_PSI, line=dict(color=MUTED, width=1), name="Shmin", row=1, col=1)
    gc_md = np.concatenate([c for *_, c in geo])
    fc_md = np.concatenate([c for *_, c in fd if len(c)])
    ymap = lambda m: g.SHMIN_PSI.to_numpy()[gc._idx(g, m)]
    fz.add_scatter(x=gc_md, y=ymap(gc_md), mode="markers", marker=dict(color=AMBER, size=6, symbol="triangle-down"), name="Geometric clusters", row=1, col=1)
    fz.add_scatter(x=fc_md, y=ymap(fc_md), mode="markers", marker=dict(color=TEAL, size=6), name="Facies-based clusters", row=1, col=1)
    fz.add_heatmap(x=g.MD_FT, z=[g.GF], colorscale=[[i / max(K - 1, 1), GF_COLOURS[i]] for i in range(K)], showscale=False, row=2, col=1)
    for t, *_ in geo:
        fz.add_shape(type="line", x0=t, x1=t, y0=0.5, y1=1, line=dict(color=AMBER, width=1), row=3, col=1)
    for t, b, c in fd:
        fz.add_shape(type="line", x0=t, x1=t, y0=0, y1=0.5, line=dict(color=TEAL if len(c) else "#B5533C", width=1), row=3, col=1)
    fz.update_yaxes(showticklabels=False, row=2, col=1)
    fz.update_yaxes(range=[0, 1], tickvals=[0.25, 0.75], ticktext=["Facies", "Geometric"], row=3, col=1)
    st.plotly_chart(layout(fz, 520, legend=dict(orientation="h", y=1.08)), width="stretch")
    st.markdown("**Q5.1** Which wells gain most, and what about their geology explains it?  \n"
                "**Q5.2** The model is predicting stages it has never seen: purer facies and tighter cluster stress. "
                "How far would you trust the uplift, and what would you do before changing the field design?  \n"
                "**Q5.3** The CQS rises with the facies design. Does it rise most in the wells the model expects to gain most?")
    if TRAINER:
        callout(f"<b>Trainer view.</b> Simulator uplift averages <b>{R['_true'].mean():+.0%}</b> "
                f"(range {R['_true'].min():+.0%} to {R['_true'].max():+.0%}). Heterogeneous, clay- or carbonate-rich "
                "laterals (PB-3H, PB-4H) gain most because geometric stages there straddle high-stress rock. PB-5H, landed "
                "in organic-rich rock with tight spacing, gains little or loses: some clusters cannot be placed under the "
                "spacing limit. The model predicts less uplift than the simulator delivers: "
                "facies-based stages have lower cluster stress spread than any geometric stage it trained on, and a "
                "tree model flattens beyond its training range, so it is conservative here. "
                f"<b>The CQS gets the ranking right:</b> its rise from geometric to facies design orders the wells as the simulator does "
                f"(rank correlation {R[['CQS facies']].sub(R['CQS geometric'], axis=0).iloc[:, 0].corr(R['_true'], method='spearman'):.2f}). "
                "The physics index loses to ML at ranking existing stages (Task 3) and wins at judging a design the model has never seen. "
                "The honest recommendation is a pilot: alternate facies-based and geometric stages in one or two wells, "
                "PB-3H and PB-4H first, and read the PLT.", "trainer")

# ============================================================================= DEBRIEF
with tabs[6]:
    st.subheader("Debrief · what to take back to the asset")
    st.markdown("""
1. **Calibrate the rock mechanics before you use it.** Dynamic moduli are not static ones; one core-calibrated conversion and one DFIT per well set the stress profile, and a stress fit can hide a wrong conversion.
2. **Two brittleness indices are two measurements.** Mineralogy says what the rock is made of; the dipole sonic says how it responds and sets closure stress. Fracability adds the stress.
3. **An index is a hypothesis.** Fracability and completion quality are tested against the production log before anyone designs with them.
4. **Facies give stage design a vocabulary.** K is argued on geomechanics as well as on the silhouette.
5. **Validate by well.** Stages within a well are not independent; the leave-one-well-out score is the number to quote.
6. **SHAP explains the model, not the reservoir.** With six wells, completion parameters ride on landing zone.
7. **An engineered design is a hypothesis.** The model ranks it and prices it; a pilot with production logging proves it.
""")
    callout("<b>Petrophysics link.</b> Every input here came from data you already acquire: density and dipole sonic, "
            "XRD-calibrated mineralogy and TOC, a few core triaxial tests and a DFIT. The workflow adds value by calibrating "
            "them to each other and putting them at stage resolution.")
    st.caption("digitaloilEd.ai · synthetic Permian-analog data · for training use only")
