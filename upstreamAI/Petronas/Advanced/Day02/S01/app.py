"""Automated fracture detection in FMI logs for carbonate reservoir characterisation · v1.1
digitaloilEd.ai case study · Day 2 Session 1. TensorFlow / Keras. Run: python -m streamlit run app.py
"""

import fmi_models as fm         # imports TensorFlow: keep it before Streamlit (avoids a pyarrow clash on macOS)
import fmi_core as fc

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy.ndimage import binary_dilation
from sklearn.metrics import average_precision_score, precision_recall_curve, confusion_matrix, balanced_accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier  # no-image screen

NAVY, TEAL, AMBER, INK, MUTED, RUST = "#0A2E36", "#14808C", "#E8A317", "#1C2B2F", "#5B6A6E", "#B5533C"
FMI_CS = [[0, "#2E1C0C"], [0.25, "#5E3814"], [0.5, "#B97A2E"], [0.75, "#E3B45A"], [1, "#FFF2CC"]]
DYN_CS = [[0, "#1B1B1B"], [0.5, "#9C7A4B"], [1, "#FFF6DC"]]
HEAT_CS = [[0, "rgba(0,0,0,0)"], [0.3, "rgba(255,210,63,0.35)"], [1, "rgba(255,59,31,0.85)"]]
_CC = ["#F4F2EC", "#14808C", "#E8A317", "#B5533C", "#8C8C8C", "#0A2E36"]
CLASS_CS = sum([[[k / 6, c], [(k + 1) / 6, c]] for k, c in enumerate(_CC)], [])
FAC_COL = ["#D9D4C7", "#14808C", "#E8A317", "#B5533C"]
FAC_CS = sum([[[k / 4, c], [(k + 1) / 4, c]] for k, c in enumerate(FAC_COL)], [])
TEXTURE = ["tex_anisotropy", "tex_entropy", "tex_contrast_az", "tex_contrast_depth", "tex_homogeneity", "conductive_fraction", "resistive_fraction"]

st.set_page_config(page_title="FMI Fracture Detection Case Study", page_icon="🪨", layout="wide")
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


AZ_DEG = (np.arange(fc.W) + 0.5) * 360 / fc.W


# ----------------------------------------------------------------------------- cached data and models
@st.cache_resource(show_spinner="Generating the synthetic carbonate field...")
def field():
    return fc.make_field()


@st.cache_resource(show_spinner="Preparing and tiling the images...")
def dataset(normalise, mask, dyn, split="well"):
    return fc.build_dataset(field(), normalise, mask, split, dynamic=dyn)


@st.cache_resource(show_spinner=False)
def prepared(well, normalise, mask, dyn):
    return fc.prepare(field()[well], normalise, mask, dyn)


@st.cache_resource(show_spinner="Training the CNN screen on the training wells (about 25 s, once)...")
def cnn_model(normalise, mask, dyn, circular, aug, split="well"):
    (X, y, _, _), (Xt, yt, _, _) = dataset(normalise, mask, dyn, split)
    m, hist = fm.train_classifier(fm.TinyCNN(circular), X, y, Xt, yt, epochs=6, roll_aug=aug)
    return m, pd.DataFrame(hist)


@st.cache_resource(show_spinner="Training the Vision Transformer (about 30 s, once)...")
def vit_model(normalise, mask, dyn, aug):
    (X, y, _, _), (Xt, yt, _, _) = dataset(normalise, mask, dyn, "well")
    m, hist = fm.train_classifier(fm.TinyViT(), X, y, Xt, yt, epochs=15, roll_aug=aug)
    return m, pd.DataFrame(hist)


@st.cache_resource(show_spinner="Training the U-Net segmenter (about 50 s, once)...")
def unet_model(normalise, mask, dyn, circular, aug):
    (X, _, L, _), _ = dataset(normalise, mask, dyn, "well")
    T = np.stack([fc.seg_target(l) for l in L[::2]])
    m, hist = fm.train_unet(fm.MiniUNet(circular), X[::2], T, epochs=10, roll_aug=aug)
    return m, pd.DataFrame(hist)


@st.cache_resource(show_spinner="Segmenting the well and fitting sinusoids...")
def picks_for(well, normalise, mask, dyn, circular, aug, thr):
    m, _ = unet_model(normalise, mask, dyn, circular, aug)
    pr = fm.segment_well(m, prepared(well, normalise, mask, dyn))
    pk = fc.pick_types(fc.extract_sinusoids(pr[1] + pr[2], thr=thr), pr)
    return pr, pk


@st.cache_resource(show_spinner=False)
def beds_for(well):
    return fc.bedding_picks(field()[well])


@st.cache_resource(show_spinner=False)
def texture(well):
    return fc.texture_log(field()[well])


@st.cache_resource(show_spinner="Training the no-image screen on texture and caliper...")
def texture_screen(normalise, mask, dyn):
    (X, y, _, m), (Xt, yt, _, mt) = dataset(normalise, mask, dyn, "well")

    def feats(meta):
        rows = []
        for w_, top in zip(meta.well, meta.top):
            ft = min(int((top + fc.TILE // 2) // fc.PX_PER_FT), field()[w_]["spec"].length_ft - 1)
            t = texture(w_).iloc[ft]; c = field()[w_]["cal"].iloc[ft]
            rows.append([t[k] for k in TEXTURE] + [abs(c.C13_in - c.C24_in), (c.C13_in + c.C24_in) / 2])
        return np.array(rows)
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, random_state=0).fit(feats(m), y)
    return clf.predict_proba(feats(mt))[:, 1]


@st.cache_resource(show_spinner="Building interval logs and calibrating the structural facies rules on CB-1 to CB-4...")
def interval_logs(normalise, mask, dyn, circular, aug, thr=0.45):
    out = {}
    for n, w in field().items():
        if w["spec"].role == "extra":
            continue
        pr, pk = picks_for(n, normalise, mask, dyn, circular, aug, thr)
        Lft = w["spec"].length_ft
        vug = np.array([pr[3, f * fc.PX_PER_FT:(f + 1) * fc.PX_PER_FT][:, fc.VALID_COLS].mean() for f in range(Lft)])
        logs = dict(p10=fc.p10_log(pk.d0, Lft), p32=fc.p32_log(pk.d0, pk.dip, Lft), cprob=fc.conductive_log(pr, Lft), vug=vug)
        logs["X"] = fc.facies_features(w, logs["p10"], logs["p32"], logs["cprob"], vug, texture(n))
        out[n] = logs
    tr = [n for n, w in field().items() if w["spec"].role == "train"]
    Xtr = pd.concat([out[n]["X"] for n in tr]); ytr = np.concatenate([field()[n]["cal"].facies.to_numpy() for n in tr])
    thr, score = fc.calibrate_facies(Xtr, ytr)
    for n in out:
        out[n]["facies"] = fc.facies_rules(out[n]["X"], thr)
    out["_rules"] = dict(thr=thr, train_score=score)
    return out


@st.cache_resource(show_spinner="Training ten CNNs: five random starts, with and without counter-examples...")
def shortcut_experiment(normalise, mask, dyn, circular, aug, seeds=(0, 1, 2, 3, 4)):
    (Xp, yp, _, _), (Xt, yt, Lt, _) = dataset(normalise, mask, dyn, "well")
    (Xe, ye, _, _), _ = fc.build_dataset(field(), normalise, mask, "well", extra=True, dynamic=dyn)
    dif = np.array([(l == 3).sum() > 20 and ((l == 1) | (l == 2)).sum() < 40 for l in Lt])
    rows = []
    for sd in seeds:
        r = {"Random start": sd}
        for tag, X, y in (("CB-1 to CB-4", Xp, yp), ("plus CB-7", Xe, ye)):
            m, _ = fm.train_classifier(fm.TinyCNN(circular), X, y, epochs=6, roll_aug=aug, seed=sd)
            p = fm.predict(m, Xt)
            r[f"Blind AP · {tag}"] = average_precision_score(yt, p)
            r[f"DIF-only called fractured · {tag}"] = (p[dif] >= .5).mean()
        rows.append(r)
    return pd.DataFrame(rows).set_index("Random start")


def first_corridor(well):
    """Default window: the shallowest fracture corridor, so features are visible straight away."""
    return int(min(c[0] for c in field()[well]["corridors"]))


def tile_preds(model, X):
    return fm.predict(model, X)


# ----------------------------------------------------------------------------- plotting helpers
def fmi_fig(img, top_row, well, title="", picks=None, truth=None, beds=None, h=520, zrange=None, cs=FMI_CS, show_scale=False):
    spec = field()[well]["spec"]
    depth = spec.top_ft + (top_row + np.arange(img.shape[0])) / fc.PX_PER_FT
    if zrange is None:
        zrange = np.nanpercentile(img, [2, 98])
    fig = go.Figure(go.Heatmap(z=img, x=AZ_DEG, y=depth, colorscale=cs, zmin=zrange[0], zmax=zrange[1],
                               showscale=show_scale, hovertemplate="Az %{x:.0f}°<br>MD %{y:.2f} ft<extra></extra>"))
    for df, dash, col, name in [(truth, "dot", "#0A2E36", "Interpreter picks"), (picks, "solid", "#2BB3BF", "Model fracture picks"),
                                (beds, "dash", "#7A3FA0", "Bedding picks")]:
        if df is None or not len(df):
            continue
        xs, ys = [], []
        for p in df.itertuples():
            r = fc.sinus(p.d0 - top_row, fc.amp_px(p.dip) if not hasattr(p, "amp_px") else p.amp_px, p.az)
            if r.max() < 0 or r.min() > img.shape[0]:
                continue
            xs += list(AZ_DEG) + [None]
            ys += list(spec.top_ft + (top_row + r) / fc.PX_PER_FT) + [None]
        if xs:
            fig.add_scatter(x=xs, y=ys, mode="lines", line=dict(color=col, width=1.5 if dash == "dot" else 2.5, dash=dash), name=name)
    fig.update_yaxes(range=[depth[-1], depth[0]], title="MD (ft)")
    fig.update_xaxes(title="Azimuth (°)", range=[0, 360], tickvals=[0, 90, 180, 270, 360])
    return layout(fig, h, title=title, legend=dict(orientation="h", y=-0.12))


def tile_fig(x, overlay=None, title="", h=260, cs=FMI_CS):
    fig = go.Figure(go.Heatmap(z=x, colorscale=cs, showscale=False, zmin=np.percentile(x, 2), zmax=np.percentile(x, 98)))
    if overlay is not None:
        fig.add_heatmap(z=overlay, colorscale=HEAT_CS, showscale=False, zmin=0, zmax=1)
    fig.update_yaxes(autorange="reversed", showticklabels=False)
    fig.update_xaxes(showticklabels=False)
    return layout(fig, h, title=title)


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown('<p class="brand">DIGITALOILED.AI</p>', unsafe_allow_html=True)
    st.markdown("**Case study** · Automated fracture detection in carbonate FMI · v1.1")
    st.caption("Synthetic six-well field · Day 2 Session 1 · TensorFlow / Keras")
    st.divider()
    st.markdown("**Pipeline settings**")
    NORM = st.toggle("Normalise each well", True, help="Robust scaling of the static image with each well's own median and spread.")
    MASK = st.toggle("Mask pad gaps", True, help="Treat gaps as missing data, with a valid-data channel. Off: gaps read as zero resistivity.")
    DYN = st.toggle("Dynamic image channel", True, help="Add the dynamic image (rank within a sliding 2 ft window) as a second image channel.")
    CIRC = st.toggle("Circular azimuth padding", True, help="0° and 360° are neighbours in the rock.")
    AUG = st.toggle("Azimuth-shift augmentation", True, help="Random rotation of the tool in the hole during training.")
    st.caption("Each combination trains once and is cached for everyone.")
    st.divider()
    code = st.text_input("Trainer code", type="password")
    try:
        expected = st.secrets.get("TRAINER_CODE", "fracture")
    except Exception:
        expected = "fracture"
    TRAINER = code == expected
    if TRAINER:
        st.success("Trainer view on")
    st.caption(f"TensorFlow {fm.tf.__version__} · Keras {fm.keras.__version__}")

F = field()
TRAIN_W = [n for n, w in F.items() if w["spec"].role == "train"]
BLIND_W = [n for n, w in F.items() if w["spec"].role == "blind"]
CFG = (NORM, MASK, DYN)

st.markdown('<p class="brand">ADVANCED AI FOR RESERVOIR PETROPHYSICS · DAY 2 · CARBONATE CASE STUDY</p>', unsafe_allow_html=True)
st.title("Automated fracture detection in FMI logs")
tabs = st.tabs(["Brief", "1 · Prepare", "2 · Screen", "3 · Segment and extract", "4 · CNN against ViT", "5 · Explain", "6 · Business value", "Debrief"])

# ============================================================================= BRIEF
with tabs[0]:
    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.subheader("The challenge")
        st.write("In many carbonate reservoirs matrix permeability is low and flow follows fractures. Image logs are the main direct "
                 "evidence of fractures in the wellbore, but manual picking is slow and varies between interpreters. Stylolites, "
                 "vugs, bedding and drilling-induced fractures look similar enough to confuse both people and models.")
        st.subheader("Objective")
        callout("On wells the models have never seen: detect natural fractures and tell open (conductive) from closed (resistive), "
                "pick bedding, recover dip and dip azimuth, and deliver fracture density (P10) and intensity (P32), a conductive "
                "feature probability log and structural facies intervals that a petrophysicist can sign off.")
        st.subheader("Six steps")
        st.markdown("1. **Prepare** · static and dynamic images, per-well normalisation, pad-gap masking, caliper and texture QC\n"
                    "2. **Screen** · a CNN flags tiles with natural fractures; a texture-and-caliper screen is the no-image bar\n"
                    "3. **Segment and extract** · a U-Net traces open and closed fractures, vugs and stylolites; sinusoid fitting gives dip, azimuth and bedding\n"
                    "4. **Compare** · CNN against a Vision Transformer, including the 0°/360° seam test\n"
                    "5. **Explain** · Grad-CAM and attention maps, and an audit of what the network looks at\n"
                    "6. **Business value** · P10 and P32, conductive feature probability, structural facies, candidate intervals and limits")
    with c2:
        st.subheader("Synthetic field")
        rows = []
        for n, w in F.items():
            s = w["spec"]
            if s.role == "extra":
                continue
            rows.append({"Well": n, "Role": "Training" if s.role == "train" else "Blind test", "Top MD (ft)": int(s.top_ft),
                         "Imaged (ft)": s.length_ft, "Core": "Yes" if s.core else "",
                         "Picked fractures": str(len(w["fracs"])) if (s.role == "train" or TRAINER) else "held out"})
        st.dataframe(pd.DataFrame(rows).set_index("Well"), width="stretch")
        st.dataframe(pd.DataFrame([
            ("Static and dynamic images", "Input", "Tabs 1 to 5"),
            ("Azimuth and dip", "Input / target", "Sinusoid picks, tadpoles, rose, P32"),
            ("Caliper", "Input", "Washout QC, breakouts, structural facies"),
            ("Image texture metrics", "Input", "No-image screen, structural facies"),
            ("Fracture type, open or closed, bedding, vug", "Target", "Tabs 3 and 5"),
            ("Density, intensity, conductive probability, structural facies", "Target", "Tab 6"),
        ], columns=["Customer dataset", "Role", "Where"]), width="stretch", hide_index=True)
        st.caption("Image grid: 64 azimuth columns (5.6° each) by 24 rows per foot (0.5 in). Pads cover 7 of every 8 columns. "
                   "Pixel labels: open (conductive) fracture, closed (resistive) fracture, drilling-induced fracture, stylolite, vug, background.")
        callout("The models are deliberately small so they train on a laptop CPU in seconds. Production work would fine-tune "
                "pre-trained ResNet, EfficientNet or ViT backbones; every validation and interpretation step here carries over.")
    if TRAINER:
        callout("<b>Trainer view.</b> The field is generated with fracture corridors in every well, two natural fracture sets "
                "(dip azimuth about 130° and 310°, dips 55–78°), stylolite and vuggy intervals, bedding with a 4–11° structural dip, and "
                "drilling-induced fractures at 40° and 220° (SHmax) with breakouts toward 130°. <b>In the four training wells the "
                "drilling-induced fractures sit inside fracture corridors; in the blind wells they do not.</b> That confound is the trap Tab 5 "
                "exposes. Each well has two washouts (both caliper arms 1–2.5 in over bit, soft and noisy image). Tool gain and offset differ "
                "by well, which is what per-well normalisation fixes. CB-5 carries core fracture counts with about 85% recovery and three "
                "rubble intervals. CB-7 is a counter-example interval used only in the Tab 5 experiment.", "trainer")

# ============================================================================= 1 PREPARE
with tabs[1]:
    st.subheader("Step 1 · Prepare the images")
    c1, c2, c0, c3 = st.columns([1, 1.4, 0.8, 1.6])
    well = c1.selectbox("Well", TRAIN_W + BLIND_W, key="w1")
    w = F[well]
    span = c0.selectbox("Window (ft)", [5, 10, 20], index=1, key="s1")
    top_ft = c2.slider("Window top (ft below top of image)", 0, w["spec"].length_ft - span, first_corridor(well), 1, key=f"d1_{well}")
    show_picks = c3.toggle("Show interpreter picks", value=w["spec"].role == "train",
                           disabled=(w["spec"].role == "blind" and not TRAINER),
                           help="Blind-well picks are held out for scoring; trainer view can show them.")
    r0, r1 = top_ft * fc.PX_PER_FT, (top_ft + span) * fc.PX_PER_FT
    prep = prepared(well, NORM, MASK, DYN)
    truth = w["fracs"] if show_picks else None
    a, b, c = st.columns(3)
    a.plotly_chart(fmi_fig(w["raw"][r0:r1], r0, well, "Static image, raw (pad gaps blank)", truth=truth, h=640), width="stretch")
    b.plotly_chart(fmi_fig(prep[0, r0:r1], r0, well, "Static, prepared", truth=truth, zrange=(-3, 3), h=640), width="stretch")
    c.plotly_chart(fmi_fig(w["dyn"][r0:r1], r0, well, "Dynamic image (2 ft window)", zrange=(0, 1), cs=DYN_CS, h=640), width="stretch")
    st.markdown("**Q1.1** Find an open (conductive) fracture, a closed (resistive) fracture, a stylolite, vugs and drilling-induced fractures in the window.  \n"
                "**Q1.2** Switch off *Mask pad gaps* in the sidebar. What do the gaps look like to a network now?  \n"
                "**Q1.3** What does the dynamic image show more clearly than the static, and what has it lost?")

    st.markdown("#### Why normalise each well")
    box = go.Figure()
    for n, ww in F.items():
        if ww["spec"].role == "extra":
            continue
        v = ww["raw"][::7, fc.VALID_COLS].ravel()
        box.add_box(y=v[~np.isnan(v)][::5], name=n, marker_color=TEAL if ww["spec"].role == "train" else AMBER, boxpoints=False)
    box.update_yaxes(title="Raw image value (log resistivity units)")
    nb = go.Figure()
    for n in TRAIN_W + BLIND_W:
        v = prepared(n, NORM, MASK, DYN)[0, ::7, fc.VALID_COLS].ravel()
        nb.add_box(y=v[::5], name=n, marker_color=TEAL if F[n]["spec"].role == "train" else AMBER, boxpoints=False)
    nb.update_yaxes(title="Prepared value", range=[-6, 6])
    a, b = st.columns(2)
    a.plotly_chart(layout(box, 330, title="Raw: tool and mud shift every well · teal training, amber blind", showlegend=False), width="stretch")
    b.plotly_chart(layout(nb, 330, title="Prepared with current settings", showlegend=False), width="stretch")

    st.markdown("#### Caliper and texture logs")
    cal, tex = w["cal"], texture(well)
    f = make_subplots(1, 4, shared_yaxes=True, horizontal_spacing=0.03, column_widths=[0.34, 0.22, 0.22, 0.22],
                      subplot_titles=("Caliper C13, C24 (in)", "Texture anisotropy", "Conductive fraction", "Structural facies"))
    f.add_scatter(x=cal.C13_in, y=cal.depth_ft, name="C13", line=dict(color=NAVY, width=1.2), row=1, col=1)
    f.add_scatter(x=cal.C24_in, y=cal.depth_ft, name="C24", line=dict(color=AMBER, width=1.2), row=1, col=1)
    f.add_vline(x=fc.BIT_IN, line=dict(color=MUTED, dash="dot"), row=1, col=1)
    f.add_scatter(x=tex.tex_anisotropy, y=tex.depth_ft, line=dict(color=TEAL, width=1.2), showlegend=False, row=1, col=2)
    f.add_scatter(x=tex.conductive_fraction, y=tex.depth_ft, line=dict(color=RUST, width=1.2), showlegend=False, row=1, col=3)
    f.add_heatmap(z=cal.facies.map({q: k for k, q in enumerate(fc.FACIES)}).to_numpy(float)[:, None], y=cal.depth_ft, x=[0], colorscale=FAC_CS,
                  zmin=-0.5, zmax=3.5, showscale=False, row=1, col=4, hoverinfo="skip")
    for d in cal.depth_ft[cal.washout]:
        f.add_hrect(y0=d - 0.5, y1=d + 0.5, fillcolor="rgba(0,0,0,0.22)", line_width=0, row=1, col=1)
    f.add_hrect(y0=w["spec"].top_ft + top_ft, y1=w["spec"].top_ft + top_ft + span, line=dict(color=RUST, width=2), fillcolor="rgba(0,0,0,0)", row=1, col=1)
    f.update_yaxes(autorange="reversed", title="MD (ft)", row=1, col=1).update_xaxes(type="log", row=1, col=2).update_xaxes(showticklabels=False, row=1, col=4)
    f.update_xaxes(range=[8, 12], row=1, col=1)
    st.plotly_chart(layout(f, 560, legend=dict(orientation="h", y=-0.06)), width="stretch")
    st.caption("Grey bands: washouts (both arms well over bit). Red box: the image window above. Facies: " + " · ".join(fc.FACIES) +
               " (interpreter, shown on every well for context).")
    (Xtr, ytr, _, _), (Xte, yte, _, _) = dataset(NORM, MASK, DYN, "well")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Training tiles", f"{len(ytr):,}", help="64 × 64 pixels, 2.7 ft tall, stride 0.7 ft")
    m2.metric("Blind tiles", f"{len(yte):,}")
    m3.metric("Fractured share, training", f"{ytr.mean():.0%}")
    m4.metric("Network input channels", f"{Xtr.shape[1]}", help="Static, dynamic (if on) and the valid-data mask")
    if TRAINER:
        callout("<b>Answers.</b> Gaps filled naively become dark vertical stripes, the same signature as a drilling-induced fracture. "
                "Without normalisation the raw boxes sit at different levels because gain and offset differ by well (CB-2 and CB-6 "
                "read high, CB-3 and CB-5 low). The dynamic image lifts faint traces and stylolites for the eye but discards the absolute "
                "level. Caliper: a breakout stretches one arm (stressed intervals); a washout opens both, and the image there goes soft. "
                "Texture anisotropy jumps where drilling-induced streaks run vertically; conductive fraction rises in vuggy intervals.", "trainer")

# ============================================================================= 2 SCREEN
with tabs[2]:
    st.subheader("Step 2 · Screen tiles with a CNN")
    st.write("A small CNN answers one question per tile: does it contain a natural fracture (open or closed)? "
             "It is trained on CB-1 to CB-4 and scored on the blind wells CB-5 and CB-6.")
    cnn, hist = cnn_model(*CFG, CIRC, AUG)
    (Xtr, ytr, Ltr, mtr), (Xte, yte, Lte, mte) = dataset(*CFG, "well")
    p = tile_preds(cnn, Xte)
    ap = average_precision_score(yte, p)
    p_tex = texture_screen(*CFG)
    thr = st.slider("Decision threshold", 0.05, 0.95, 0.5, 0.05, key="t2")
    tn, fp, fn, tp = confusion_matrix(yte, (p >= thr).astype(int)).ravel()
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Average precision, blind wells", f"{ap:.2f}")
    m2.metric("No-image screen (texture + caliper)", f"{average_precision_score(yte, p_tex):.2f}")
    m3.metric("No-skill baseline", f"{yte.mean():.2f}", help="Share of blind tiles that are fractured")
    m4.metric("Precision at threshold", f"{tp / max(tp + fp, 1):.2f}")
    m5.metric("Recall at threshold", f"{tp / max(tp + fn, 1):.2f}")
    a, b = st.columns([1.3, 1])
    with a:
        f = go.Figure()
        for pp, col, name in [(p, TEAL, "CNN"), (p_tex, AMBER, "Texture + caliper (no image)")]:
            pr, rc, _ = precision_recall_curve(yte, pp)
            f.add_scatter(x=rc, y=pr, mode="lines", line=dict(color=col, width=3), name=name)
        f.add_hline(y=yte.mean(), line=dict(color=MUTED, dash="dot"), annotation_text="no skill")
        f.update_xaxes(title="Recall", range=[0, 1]).update_yaxes(title="Precision", range=[0, 1.02])
        st.plotly_chart(layout(f, 380, title="Precision-recall on held-out wells", legend=dict(orientation="h", y=-0.2)), width="stretch")
    with b:
        cm = go.Figure(go.Heatmap(z=[[tp, fn], [fp, tn]], x=["Pred. fractured", "Pred. clean"], y=["Fractured", "Clean"],
                                  text=[[tp, fn], [fp, tn]], texttemplate="%{text}", colorscale=[[0, "#F4F2EC"], [1, TEAL]], showscale=False))
        cm.update_yaxes(autorange="reversed")
        st.plotly_chart(layout(cm, 380, title=f"Confusion matrix at {thr:.2f}"), width="stretch")
    per = pd.DataFrame({"well": mte.well, "y": yte, "p": p}).groupby("well").apply(
        lambda d: pd.Series({"tiles": len(d), "fractured share": d.y.mean(), "AP": average_precision_score(d.y, d.p)}), include_groups=False)
    st.dataframe(per.style.format({"fractured share": "{:.0%}", "AP": "{:.2f}", "tiles": "{:.0f}"}), width="stretch")

    st.markdown("#### Leakage check: random tiles against held-out wells")
    if st.button("Train the same CNN on a random tile split", key="rand"):
        st.session_state["rand_on"] = True
    if st.session_state.get("rand_on"):
        rm, _ = cnn_model(*CFG, CIRC, AUG, "random")
        _, (Xr, yr, _, _) = dataset(*CFG, "random")
        apr = average_precision_score(yr, tile_preds(rm, Xr))
        c1, c2 = st.columns(2)
        c1.metric("AP · random overlapping tiles", f"{apr:.2f}")
        c2.metric("AP · held-out wells", f"{ap:.2f}", delta=f"{ap - apr:+.2f}", delta_color="off")

    st.markdown("#### What the screen gets wrong")
    lab_name = lambda L: max(((L == c).sum(), fc.CLASSES[c]) for c in (3, 4, 5))[1]
    fp_idx = np.flatnonzero((p >= thr) & (yte == 0))[:6]
    fn_idx = np.flatnonzero((p < thr) & (yte == 1))[:6]
    for name, idx in [("False alarms", fp_idx), ("Missed fractures", fn_idx)]:
        if len(idx):
            cols = st.columns(6)
            for c, i in zip(cols, idx):
                c.plotly_chart(tile_fig(Xte[i, 0], title=f"{mte.well[i]} · p={p[i]:.2f}<br>{lab_name(Lte[i]) if name == 'False alarms' else 'faint trace'}", h=220),
                               width="stretch", key=f"{name}{i}")
            st.caption(name)
    st.markdown("**Q2.1** How far above the no-skill line is the CNN, and how much of that could texture and caliper alone deliver?  \n"
                "**Q2.2** Why does the random tile split score higher?")
    if TRAINER:
        callout("<b>Answers.</b> With default settings the CNN scores close to 0.95 against a no-skill baseline of about 0.3. Texture and caliper "
                "alone reach only about 0.5 to 0.6: they know a tile is busy, not whether a sinusoid crosses it. For a screen, a low threshold suits: "
                "the interpreter reviews flagged tiles, so missed fractures cost more than false alarms. The random split puts overlapping "
                "neighbours of every test tile in training; the gap is small here because the synthetic wells are similar, and grows with "
                "real well-to-well variability. The direction is what matters.", "trainer")

# ============================================================================= 3 SEGMENT AND EXTRACT
with tabs[3]:
    st.subheader("Step 3 · Segment fractures and fit sinusoids")
    st.write("A small U-Net labels every pixel as open fracture, closed fracture, vug, stylolite or background. A Hough-style search then "
             "fits a sinusoid through each fracture trace: amplitude gives dip, the deepest point gives dip azimuth. Bedding planes are "
             "picked the classical way: low-dip sinusoids along resistivity steps of the static image.")
    c1, c2, c0, c3, c4 = st.columns([1, 1, 0.7, 1.3, 1])
    well3 = c1.selectbox("Blind well", BLIND_W, key="w3")
    hthr = c2.slider("Pick threshold", 0.25, 0.65, 0.45, 0.05, help="Mean fracture probability along the sinusoid")
    span3 = c0.selectbox("Window (ft)", [5, 10, 20], index=1, key="s3")
    top3 = c3.slider("Window top (ft)", 0, 240 - span3, first_corridor(well3), 1, key=f"d3_{well3}")
    reveal = c4.toggle("Show held-out picks", value=TRAINER)
    probs, picks = picks_for(well3, *CFG, CIRC, AUG, hthr)
    beds = beds_for(well3)
    w3 = F[well3]
    truth3 = w3["fracs"]
    m, npk, nt = fc.match_picks(picks, truth3)
    mb, nbk, ntb = fc.match_picks(beds, w3["beds"].assign(type=0), tol_px=12)
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Fracture picks", npk, help=f"{nt} interpreter picks in this well")
    k2.metric("Precision / recall", f"{len(m) / max(npk, 1):.2f} / {len(m) / max(nt, 1):.2f}")
    k3.metric("Dip error (mean abs.)", f"{m.dip_err.abs().mean():.1f}°" if len(m) else "n/a")
    k4.metric("Azimuth error (median)", f"{m.az_err.median():.0f}°" if len(m) else "n/a")
    k5.metric("Open against closed", f"{(m.type_pred == m.type_true).mean():.0%}" if len(m) else "n/a", help="On matched picks")
    k6.metric("Bedding · dip / azimuth", f"{beds.dip.median():.0f}° / {beds.az.median():.0f}°" if len(beds) else "n/a",
              help=f"{nbk} bedding picks; structural dip {w3['spec'].bed_dip:.0f}° toward {w3['spec'].bed_az:.0f}° (trainer view)")
    r0, r1 = top3 * fc.PX_PER_FT, (top3 + span3) * fc.PX_PER_FT
    a, b = st.columns(2)
    a.plotly_chart(fmi_fig(w3["raw"][r0:r1], r0, well3, "Image with model picks", picks=picks, truth=truth3 if reveal else None, beds=beds, h=640), width="stretch")
    pm = (probs[1] + probs[2])[r0:r1].copy()
    pm[:, fc.GAP_COLS] = np.nan
    b.plotly_chart(fmi_fig(pm, r0, well3, "Fracture probability from the U-Net (open + closed)", picks=picks, zrange=(0, 1),
                           cs=[[0, "#F4F2EC"], [0.5, "#7CC3C9"], [1, NAVY]], show_scale=True, h=640), width="stretch")

    a, b = st.columns([1.2, 1])
    with a:
        tp_ = go.Figure()
        L = 6.0
        series = [(truth3 if reveal else None, "#0A2E36", "Interpreter fractures", "circle-open"), (picks, TEAL, "Model fractures", "circle"),
                  (beds, "#7A3FA0", "Bedding picks", "diamond")]
        for df, col, name, sym in series:
            if df is None or not len(df):
                continue
            depth = w3["spec"].top_ft + df.d0 / fc.PX_PER_FT
            tp_.add_scatter(x=df.dip, y=depth, mode="markers", marker=dict(color=col, size=8, symbol=sym), name=name,
                            hovertext=[f"dip {d:.0f}° az {z:.0f}°" for d, z in zip(df.dip, df.az)])
            xs, ys = [], []
            for d, z, y in zip(df.dip, df.az, depth):
                xs += [d, d + L * np.sin(np.radians(z)) / 2, None]
                ys += [y, y - L * np.cos(np.radians(z)) / 2, None]
            tp_.add_scatter(x=xs, y=ys, mode="lines", line=dict(color=col, width=1.2), showlegend=False, hoverinfo="skip")
        tp_.update_yaxes(autorange="reversed", title="MD (ft)").update_xaxes(title="Dip (°)", range=[0, 90])
        st.plotly_chart(layout(tp_, 520, title="Tadpole plot · tail points to dip azimuth"), width="stretch")
    with b:
        rose = go.Figure()
        bins = np.arange(0, 360, 20)
        for df, col, name in [(truth3 if reveal else None, "#0A2E36", "Interpreter fractures"), (picks, TEAL, "Model fractures"), (beds, "#7A3FA0", "Bedding")]:
            if df is None or not len(df):
                continue
            cnt = np.histogram(df.az % 360, bins=np.append(bins, 360))[0]
            rose.add_barpolar(r=cnt, theta=bins + 10, width=18, name=name, marker_color=col, opacity=0.6)
        rose.update_polars(angularaxis=dict(direction="clockwise", rotation=90))
        st.plotly_chart(layout(rose, 520, title="Rose diagram of dip azimuth"), width="stretch")
    st.caption("Matching tolerance for fractures: centre depth within 4 in. Bedding picks are matched to resistivity steps of at least 0.2 log units within 6 in.")
    st.markdown("**Q3.1** Move the pick threshold. Which do you trade: missed fractures or extra picks, and who pays for each?  \n"
                "**Q3.2** The two natural sets dip toward roughly opposite azimuths. Does the rose diagram recover both, and where does bedding sit?")
    if TRAINER:
        callout(f"<b>Answers.</b> With defaults expect fracture precision near 0.75 and recall near 0.8 to 0.85, dip error under 1°, median azimuth "
                f"error of a few degrees and open against closed agreement above 95% on matched picks. The stylolite class in the U-Net matters: "
                f"without it, stylolites (conductive, bedding-parallel) become false open-fracture picks. Bedding picks recover the structural dip "
                f"to within about 3° of dip and 20° of azimuth ({w3['spec'].bed_dip:.0f}° toward {w3['spec'].bed_az:.0f}° in {well3}); the classical picker finds about half the "
                "bed boundaries, with about one false pick in three. The rose shows fracture sets near 130° and 310° and bedding toward 190–220°. "
                "Remind the room that these are apparent intensities along the well; P32 in Tab 6 corrects for dip.", "trainer")

# ============================================================================= 4 CNN AGAINST VIT
with tabs[4]:
    st.subheader("Step 4 · CNN against Vision Transformer")
    vit, vh = vit_model(NORM, MASK, DYN, AUG)
    (Xtr, ytr, _, _), (Xte, yte, Lte, mte) = dataset(*CFG, "well")
    pc, pv = tile_preds(cnn, Xte), tile_preds(vit, Xte)
    f = go.Figure()
    for pp, col, name in [(pc, TEAL, "CNN"), (pv, AMBER, "ViT")]:
        pr, rc, _ = precision_recall_curve(yte, pp)
        f.add_scatter(x=rc, y=pr, mode="lines", line=dict(color=col, width=3), name=f"{name} · AP {average_precision_score(yte, pp):.2f}")
    f.add_hline(y=yte.mean(), line=dict(color=MUTED, dash="dot"))
    f.update_xaxes(title="Recall", range=[0, 1]).update_yaxes(title="Precision", range=[0, 1.02])
    a, b = st.columns([1.3, 1])
    a.plotly_chart(layout(f, 400, title="Blind wells, same tiles, same labels"), width="stretch")
    with b:
        hh = go.Figure()
        hh.add_scatter(x=hist.epoch, y=hist.blind_AP, name="CNN", line=dict(color=TEAL, width=3))
        hh.add_scatter(x=vh.epoch, y=vh.blind_AP, name="ViT", line=dict(color=AMBER, width=3))
        hh.update_xaxes(title="Epoch").update_yaxes(title="Blind AP", range=[0, 1])
        st.plotly_chart(layout(hh, 400, title="Learning speed"), width="stretch")
    st.caption("Both models are trained from scratch on about 1,400 tiles. The ViT has no built-in notion of locality, which is "
               "why it learns more slowly with small data; in practice ViTs are pre-trained on large image collections first.")

    st.markdown("#### Seam stress test · rotate every blind tile by 180°")
    st.write("Rolling the image by half its width moves the azimuth origin. The rock has not changed, so the prediction should not either.")
    if st.button("Run the seam test (trains one extra CNN with zero padding and no augmentation)", key="seam"):
        st.session_state["seam_on"] = True
    if st.session_state.get("seam_on"):
        zcnn, _ = cnn_model(*CFG, False, False)
        Xroll = np.roll(Xte, fc.W // 2, axis=-1).copy()
        rows = []
        for name, mdl in [("CNN · current settings", cnn), ("CNN · zero padding, no augmentation", zcnn), ("ViT · current settings", vit)]:
            p0, p1 = tile_preds(mdl, Xte), tile_preds(mdl, Xroll)
            rows.append({"Model": name, "Mean change in probability": np.abs(p1 - p0).mean(),
                         "Decisions flipped at 0.5": ((p0 >= .5) != (p1 >= .5)).mean(), "Blind AP": average_precision_score(yte, p0)})
        st.dataframe(pd.DataFrame(rows).set_index("Model").style.format({"Mean change in probability": "{:.3f}",
                     "Decisions flipped at 0.5": "{:.1%}", "Blind AP": "{:.2f}"}), width="stretch")
    st.markdown("**Q4.1** Which model would you deploy for a field-wide screen, and what would make you change your mind?  \n"
                "**Q4.2** A model whose answer depends on where 0° sits: defect or quirk?")
    if TRAINER:
        callout("<b>Answers.</b> With these data the CNN wins clearly and learns faster; the ViT reaches about 0.75 to 0.85 blind AP in 15 epochs and would "
                "need pre-training to compete. With circular padding and azimuth augmentation the CNN's prediction barely changes under the 180° "
                "roll (no decision flips); the zero-padded CNN without augmentation and the ViT flip one to two per cent of decisions. A model that depends on the azimuth origin is a defect: "
                "the tool's reference direction is arbitrary.", "trainer")

# ============================================================================= 5 EXPLAIN
with tabs[5]:
    st.subheader("Step 5 · Explain: where is the network looking?")
    (Xtr, ytr, _, _), (Xte, yte, Lte, mte) = dataset(*CFG, "well")
    pc, pv = tile_preds(cnn, Xte), tile_preds(vit, Xte)
    nat_px = np.array([((l == 1) | (l == 2)).sum() for l in Lte])
    dif_px = np.array([(l == 3).sum() for l in Lte])
    sty_px = np.array([(l == 4).sum() for l in Lte])
    groups = {
        "Fractured tiles the CNN catches": np.flatnonzero((yte == 1) & (pc >= .5)),
        "Drilling-induced only (no natural fracture)": np.flatnonzero((dif_px > 20) & (nat_px < 40)),
        "Stylolite-rich, no natural fracture": np.flatnonzero((sty_px > 60) & (nat_px < 40)),
        "CNN false alarms": np.flatnonzero((yte == 0) & (pc >= .5)),
    }
    c1, c2 = st.columns([2, 1])
    g = c1.selectbox("Tile group", [k for k, v in groups.items() if len(v)])
    idx = groups[g]
    j = c2.slider("Tile", 0, max(len(idx) - 1, 0), 0)
    i = int(idx[j])
    cam, pcam = fm.grad_cam(cnn, Xte[i])
    att, patt = fm.vit_attention(vit, Xte[i])
    cols = st.columns(4)
    lab = Lte[i].astype(float)
    cols[0].plotly_chart(tile_fig(Xte[i, 0], title=f"{mte.well[i]} · MD {F[mte.well[i]]['spec'].top_ft + mte.top[i] / fc.PX_PER_FT:.1f} ft", h=300), width="stretch")
    cols[1].plotly_chart(tile_fig(Xte[i, 0], cam, title=f"CNN Grad-CAM · p = {pcam:.2f}", h=300), width="stretch")
    cols[2].plotly_chart(tile_fig(Xte[i, 0], att, title=f"ViT attention · p = {patt:.2f}", h=300), width="stretch")
    lf = go.Figure(go.Heatmap(z=lab, colorscale=CLASS_CS, zmin=-0.5, zmax=5.5, showscale=False))
    lf.update_yaxes(autorange="reversed", showticklabels=False).update_xaxes(showticklabels=False)
    cols[3].plotly_chart(layout(lf, 300, title="Interpreter picks"), width="stretch")
    st.caption("Picks panel: teal open fracture · amber closed fracture · rust drilling-induced · grey stylolite · navy vug.")

    st.markdown("#### Audit · share of Grad-CAM heat on each feature, blind tiles the CNN calls fractured")

    def audit(model, X, L, P, n=120):
        idx_ = np.flatnonzero(P >= .5)[:n]
        sh, area = np.zeros(3), np.zeros(3)
        gapm = np.zeros((fc.TILE, fc.W), bool); gapm[:, fc.GAP_COLS] = True
        for k in idx_:
            c_, _ = fm.grad_cam(model, X[k]); l_ = L[k]
            nat = binary_dilation((l_ == 1) | (l_ == 2), iterations=2)
            dif = binary_dilation(l_ == 3, iterations=2) & ~nat
            gap = gapm & ~(nat | dif)
            tot = c_.sum() + 1e-9
            for q, msk in enumerate((nat, dif, gap)):
                sh[q] += c_[msk].sum() / tot; area[q] += msk.mean()
        return sh / max(len(idx_), 1), area / max(len(idx_), 1)

    sh, area = audit(cnn, Xte, Lte, pc)
    aud = pd.DataFrame({"Feature": ["Natural fracture", "Drilling-induced", "Pad gap"], "Heat share": sh, "Area share": area})
    aud["Enrichment"] = aud["Heat share"] / aud["Area share"].clip(lower=1e-6)
    a, b = st.columns([1, 1.2])
    a.dataframe(aud.set_index("Feature").style.format({"Heat share": "{:.0%}", "Area share": "{:.0%}", "Enrichment": "{:.1f}×"}), width="stretch")
    fp_dif = (pc[groups["Drilling-induced only (no natural fracture)"]] >= .5).mean() if len(groups["Drilling-induced only (no natural fracture)"]) else np.nan
    b.metric("Drilling-induced-only blind tiles called fractured", f"{fp_dif:.0%}")

    st.markdown("#### Random-start experiment · is the model right by luck?")
    st.write("Train the same CNN, on the same data, from five different random starts. Then add CB-7, an interpreter-labelled "
             "interval with drilling-induced fractures and no corridors, and repeat. Score every model on the blind wells, and on the "
             "drilling-induced-only tiles separately.")
    if st.button("Run the experiment (ten short trainings, about three minutes)", key="seeds"):
        st.session_state["seeds_on"] = True
    if st.session_state.get("seeds_on"):
        ex = shortcut_experiment(*CFG, CIRC, AUG)
        k1, k2 = st.columns(2)
        a_ = ex["DIF-only called fractured · CB-1 to CB-4"]; b_ = ex["DIF-only called fractured · plus CB-7"]
        k1.metric("DIF-only tiles called fractured · range over starts", f"{a_.min():.0%} to {a_.max():.0%}",
                  help=f"Blind AP over the same starts: {ex['Blind AP · CB-1 to CB-4'].min():.2f} to {ex['Blind AP · CB-1 to CB-4'].max():.2f}")
        k2.metric("With CB-7 counter-examples", f"{b_.min():.0%} to {b_.max():.0%}",
                  help=f"Starts at 10% or more: {(a_ >= 0.1).sum()} of {len(ex)} without, {(b_ >= 0.1).sum()} of {len(ex)} with CB-7")
        st.dataframe(ex.style.format({c: ("{:.2f}" if "AP" in c else "{:.0%}") for c in ex.columns}), width="stretch")
        st.caption("DIF-only tiles: blind tiles with drilling-induced streaks and no natural fracture. Read the spread across starts, not one number.")
    st.markdown("**Q5.1** Find a tile where the network is right for the wrong reason.  \n"
                "**Q5.2** Two models with almost the same blind AP behave very differently on drilling-induced fractures. "
                "Which number would have warned you?")
    if TRAINER:
        callout("<b>Answers.</b> In CB-1 to CB-4 the drilling-induced fractures sit inside fracture corridors, so vertical dark "
                "streaks and natural fractures appear together. With these data the share of blind drilling-induced-only tiles the CNN calls "
                "fractured swings between random starts, from 0% to about 25%, while blind AP stays at about 0.94 to 0.97: the tiles are too few "
                "to move AP. Adding one counter-example interval (CB-7) usually narrows the swing (to 0–2% in the reference run), but not in every "
                "run: TensorFlow on CPU is not bit-for-bit repeatable, and in one of our trial runs two starts got worse. The robust fix is a class "
                "of its own, as the stylolite class did for the U-Net in Tab 3. Run the experiment before the session and quote the ranges shown. Lessons: AP hides targeted failures; score the confusers explicitly, across several "
                "starts; teach the model what the confuser is. The image counterpart of Day 1's fluid-volume proxy.", "trainer")

# ============================================================================= 6 BUSINESS VALUE
with tabs[6]:
    st.subheader("Step 6 · From picks to a decision")
    c1, c2, c3 = st.columns(3)
    well6 = c1.selectbox("Blind well", BLIND_W, key="w6")
    meas = c2.radio("Intensity measure for candidate intervals", ["P10 · fractures per ft", "P32 · fracture area per volume"], horizontal=True)
    cut = c3.slider("Cut-off for a candidate interval", 0.2, 1.5, 0.6 if meas.startswith("P10") else 1.2, 0.1)
    _, pk6 = picks_for(well6, *CFG, CIRC, AUG, 0.45)
    logs = interval_logs(*CFG, CIRC, AUG)[well6]
    w6 = F[well6]
    Lft = w6["spec"].length_ft
    depth_ft = w6["spec"].top_ft + np.arange(Lft) + 0.5
    key = "p10" if meas.startswith("P10") else "p32"
    tlog = fc.p10_log(w6["fracs"].d0, Lft) if key == "p10" else fc.p32_log(w6["fracs"].d0, w6["fracs"].dip, Lft)
    ctruth = fc.conductive_truth(w6["lab"], Lft)
    fig = make_subplots(1, 4, shared_yaxes=True, horizontal_spacing=0.03, column_widths=[0.4, 0.2, 0.2, 0.2],
                        subplot_titles=("P10 (per ft)" if key == "p10" else "P32 (1/ft)", "Conductive prob.", "Caliper (in)", "Facies"))
    fig.add_scatter(x=logs[key], y=depth_ft, name=f"Model {key.upper()}", line=dict(color=TEAL, width=3), row=1, col=1)
    fig.add_scatter(x=tlog, y=depth_ft, name=f"Interpreter {key.upper()} (held out)", line=dict(color=NAVY, width=2, dash="dot"), row=1, col=1)
    if w6["core"] is not None and key == "p10":
        cr = w6["core"]
        fig.add_bar(x=cr.core_fractures_per_ft, y=cr.depth_ft, orientation="h", name="Core count per ft", marker_color="rgba(232,163,23,0.55)", row=1, col=1)
    fig.add_vline(x=cut, line=dict(color=RUST, dash="dash"), row=1, col=1)
    fig.add_scatter(x=logs["cprob"], y=depth_ft, line=dict(color=TEAL, width=1.5), showlegend=False, row=1, col=2)
    fig.add_scatter(x=w6["cal"].C13_in, y=depth_ft, line=dict(color=NAVY, width=1), showlegend=False, row=1, col=3)
    fig.add_scatter(x=w6["cal"].C24_in, y=depth_ft, line=dict(color=AMBER, width=1), showlegend=False, row=1, col=3)
    fz = np.column_stack([w6["cal"].facies.map({q: k for k, q in enumerate(fc.FACIES)}), pd.Series(logs["facies"]).map({q: k for k, q in enumerate(fc.FACIES)})]).astype(float)
    fig.add_heatmap(z=fz if TRAINER else fz[:, 1:], y=depth_ft, x=["interp.", "model"] if TRAINER else ["model"], colorscale=FAC_CS, zmin=-0.5, zmax=3.5,
                    showscale=False, row=1, col=4, hoverinfo="skip")

    def intervals(series):
        on = series >= cut
        out, s = [], None
        for k, v in enumerate(on):
            if v and s is None:
                s = k
            if (not v or k == len(on) - 1) and s is not None:
                e = k if not v else k + 1
                if e - s >= 3:
                    out.append((s, e))
                s = None
        return out

    wash = w6["cal"].washout.to_numpy()
    im, it = intervals(logs[key]), intervals(tlog)
    for s, e in im:
        fig.add_hrect(y0=w6["spec"].top_ft + s, y1=w6["spec"].top_ft + e, fillcolor="rgba(20,128,140,0.12)", line_width=0, row=1, col=1)
    for d in depth_ft[wash]:
        fig.add_hrect(y0=d - 0.5, y1=d + 0.5, fillcolor="rgba(0,0,0,0.25)", line_width=0, row=1, col=3)
    fig.update_yaxes(autorange="reversed", title="MD (ft)", row=1, col=1)
    fig.update_xaxes(range=[0, 1], row=1, col=2).update_xaxes(range=[8, 12], row=1, col=3)
    a, b = st.columns([1.5, 1])
    a.plotly_chart(layout(fig, 720, title="Shaded: model candidate intervals (3 ft window)", legend=dict(orientation="h", y=-0.06)), width="stretch")
    with b:
        mset = np.zeros(Lft, bool); tset = np.zeros(Lft, bool)
        for s, e in im: mset[s:e] = True
        for s, e in it: tset[s:e] = True
        jac = (mset & tset).sum() / max((mset | tset).sum(), 1)
        k1, k2 = st.columns(2)
        k1.metric("Intervals · model / interp.", f"{len(im)} / {len(it)}")
        k2.metric("Footage agreement (IoU)", f"{jac:.0%}")
        k1.metric(f"{key.upper()} correlation", f"{np.corrcoef(logs[key], tlog)[0, 1]:.2f}")
        k2.metric("Mean P32 / P10 (model)", f"{logs['p32'].mean() / max(logs['p10'].mean(), 1e-9):.2f}", help="1 / cos(dip): about 2.6 for 68° fractures")
        k1.metric("Conductive probability · AP", f"{average_precision_score(ctruth, logs['cprob']):.2f}", help=f"No skill {ctruth.mean():.2f}")
        k2.metric("Structural facies · balanced accuracy", f"{balanced_accuracy_score(w6['cal'].facies, logs['facies']):.2f}")
        if w6["core"] is not None:
            cc = w6["core"].core_fractures_per_ft.to_numpy()
            ok = ~np.isnan(cc)
            st.metric("P10 correlation with core counts (3 ft)", f"{np.corrcoef(np.convolve(np.nan_to_num(cc), np.ones(3) / 3, 'same')[ok], logs['p10'][ok])[0, 1]:.2f}")
        rl = interval_logs(*CFG, CIRC, AUG)["_rules"]["thr"]
        st.caption(f"Structural facies rules, calibrated on CB-1 to CB-4 (9 ft windows): Stressed if the caliper elongation reaches {rl[1]:.2f} in · "
                   f"Fracture corridor if P10 averages {rl[0]:.2f} per ft or more · Vuggy if the U-Net vug fraction averages {rl[2]:.3f} · else Layered. "
                   f"Washout footage in {well6}: {wash.sum()} ft (grey on the caliper track).")
        st.markdown("##### Interpretation effort")
        st.caption("Illustrative defaults: replace them with your asset's figures.")
        ft_yr = st.number_input("Imaged footage per year (ft)", 0, 500000, 20000, 1000)
        man = st.number_input("Manual picking, hours per 100 ft", 0.0, 20.0, 2.0, 0.5)
        rev = st.number_input("Reviewing model picks, hours per 100 ft", 0.0, 20.0, 0.5, 0.25)
        st.metric("Interpreter hours released per year", f"{ft_yr / 100 * (man - rev):,.0f}")
    callout("<b>Where not to use the output.</b> Washouts and intervals with poor pad contact or many pad gaps, heavy drilling-induced fracturing, "
            "and rock types absent from the training wells. State these limits on the model card next to the blind scores.")
    st.markdown("**Q6.1** Would you select acid stimulation intervals from the model log alone? What else would you check?  \n"
                "**Q6.2** P10 or P32: which would you hand to the reservoir engineer building a fracture model, and why?  \n"
                "**Q6.3** Which number would convince your asset manager: average precision, interval agreement or interpreter hours?")
    if TRAINER:
        callout("<b>Answers.</b> The model logs follow the corridors (P10 correlation about 0.9); candidate-interval footage agrees on about half the footage, and the core in CB-5 "
                "correlates with P10 (about 0.85) while missing rubble intervals. P32 is about three times P10 here because the fractures dip "
                "55–78° in a vertical well; a reservoir engineer needs P32 (or a Terzaghi-corrected P10) for a discrete fracture network. The "
                "conductive probability ranks well (AP about 0.85 to 0.9) but is under-confident: it averages about 0.55 where open fractures or vugs are "
                "present and about 0.2 where they are not, so treat it as a ranking or calibrate it. Structural facies come from four transparent "
                "rules calibrated on the training wells (balanced accuracy about 0.75 to 0.9 on the blind wells): stressed intervals from caliper "
                "breakouts, corridors from P10; vuggy against layered is the main confusion. A learned facies model trained on CB-1 to CB-4 would "
                "inherit the corridor confound and call blind corridors 'stressed'. Use the model as a consistent first pass reviewed by an "
                "interpreter, then check against core, losses and production logs before committing stimulation spend.", "trainer")

# ============================================================================= DEBRIEF
with tabs[7]:
    st.subheader("Debrief")
    st.markdown("""
1. **Preparation is domain knowledge.** Per-well normalisation, pad-gap masking, circular azimuth padding and caliper QC decide what the network can learn and where it can be trusted.
2. **Static for amplitude, dynamic for the eye.** The dynamic image helps interpreters see faint features; the network leans on the calibrated static amplitude for open against closed.
3. **Validate by well.** Random overlapping tiles leak; blind wells are the honest test.
4. **Give confusers their own class.** Stylolites and drilling-induced fractures become false fractures unless the model is taught what they are.
5. **Architecture follows data.** With small labelled sets a CNN beats a ViT trained from scratch; pre-training changes that.
6. **Audit what the network looks at.** A good score can hide a shortcut: drilling-induced fractures and pad gaps are the usual suspects.
7. **Deliver a decision, with limits.** P10 and P32, a conductive probability, structural facies, candidate intervals and a model card that says where not to use them.
""")
    callout("All data are synthetic, generated for the course. The workflow, the checks and the traps are the same on real image logs.")
    st.caption("digitaloilEd.ai · Advanced AI for Conventional & Unconventional Reservoir Petrophysics · Day 2")
