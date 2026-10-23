"""Core logic for the Geomechanical Facies and Stage Performance case study.

Synthetic Permian-analog field: six horizontal wells, dipole-sonic elastic logs,
XRD-calibrated mineralogy, stage-level completion records and production-log
contributions. Everything is generated deterministically from SEED.

The hidden "truth" columns (prefixed with an underscore) drive the stage
productivity simulator and are only exposed in the trainer view.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, generic_filter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, r2_score, mean_absolute_error
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

SEED = 2026
DX = 5.0                      # sample spacing along the lateral (ft)
SV_GRAD = 1.10                # overburden gradient (psi/ft)
EPS_T_TRUE = 1.2e-4           # tectonic strain used by the simulator
YM_MIN, YM_MAX = 1.0, 8.0     # Rickman normalisation limits (Mpsi)
PR_MIN, PR_MAX = 0.15, 0.40

# Hidden geomechanical facies (truth). Students never see these labels.
FACIES = [
    dict(name="Siliceous brittle",     qz=0.50, carb=0.15, clay=0.24, toc=3.4, ym=5.1, pr=0.21),
    dict(name="Carbonate-cemented",    qz=0.20, carb=0.60, clay=0.12, toc=1.0, ym=7.2, pr=0.28),
    dict(name="Clay-rich ductile",     qz=0.28, carb=0.10, clay=0.48, toc=2.2, ym=2.9, pr=0.31),
    dict(name="Organic-rich mudstone", qz=0.42, carb=0.09, clay=0.32, toc=5.2, ym=3.9, pr=0.24),
]

WELLS = pd.DataFrame({
    "WELL":          ["PB-1H", "PB-2H", "PB-3H", "PB-4H", "PB-5H", "PB-6H"],
    "LATERAL_FT":    [9000, 8500, 10000, 9500, 8000, 9200],
    "TVD_FT":        [9800, 10100, 9600, 10300, 9900, 10000],
    "PP_GRAD":       [0.58, 0.60, 0.55, 0.63, 0.59, 0.61],
    "STAGE_LEN_FT":  [200, 180, 240, 200, 150, 220],
    "CLUSTERS":      [5, 6, 5, 4, 7, 5],
    "PROPPANT_LB_FT": [2000, 2400, 1600, 1800, 2600, 2200],
    "FLUID_BBL_FT":  [40, 48, 32, 36, 50, 44],
    "RATE_BPM":      [90, 95, 80, 85, 100, 90],
    "LANDING":       ["Upper target", "Mid target", "Low (clay-prone)", "Upper (cemented)", "Organic-rich", "Mid target"],
})
PROPS = {
    "PB-1H": [0.40, 0.15, 0.20, 0.25],
    "PB-2H": [0.25, 0.25, 0.25, 0.25],
    "PB-3H": [0.20, 0.15, 0.40, 0.25],
    "PB-4H": [0.30, 0.35, 0.15, 0.20],
    "PB-5H": [0.30, 0.10, 0.25, 0.35],
    "PB-6H": [0.30, 0.20, 0.25, 0.25],
}

LOG_CURVES = ["YM_MPSI", "PR", "QUARTZ", "CARBONATE", "CLAY", "TOC"]


# ----------------------------------------------------------------------------- physics
def shmin_uniaxial(pr, ym_mpsi, tvd, pp_grad, eps_t=EPS_T_TRUE):
    """Minimum horizontal stress: uniaxial strain plus a tectonic strain term (psi)."""
    sv = SV_GRAD * tvd
    pp = pp_grad * tvd
    return pr / (1 - pr) * (sv - pp) + pp + ym_mpsi * 1e6 * eps_t


def bi_mineral(qz, carb, clay):
    """Jarvie-type mineralogical brittleness: quartz over quartz + carbonate + clay."""
    return qz / (qz + carb + clay)


def bi_elastic(ym_mpsi, pr):
    """Rickman elastic brittleness from normalised Young's modulus and Poisson's ratio."""
    ymn = np.clip((ym_mpsi - YM_MIN) / (YM_MAX - YM_MIN), 0, 1)
    prn = np.clip((PR_MAX - pr) / (PR_MAX - PR_MIN), 0, 1)
    return 0.5 * (ymn + prn)


# ----------------------------------------------------------------------------- field generator
def _facies_track(n, props, rng):
    lab = np.empty(n, dtype=int)
    i, prev = 0, -1
    p = np.asarray(props, float)
    while i < n:
        q = p.copy()
        if prev >= 0:
            q[prev] *= 0.25
        q /= q.sum()
        f = rng.choice(4, p=q)
        seg = int(max(3, round((15 + rng.exponential(105)) / DX)))
        lab[i:i + seg] = f
        prev, i = f, i + seg
    return lab


def make_field(seed: int = SEED):
    """Return (logs, stages). logs holds one row per 5 ft sample; stages one row per stage."""
    rng = np.random.default_rng(seed)
    log_rows, stage_rows = [], []
    for w in WELLS.itertuples():
        n = int(w.LATERAL_FT / DX)
        md = 10500 + np.arange(n) * DX            # heel at the shallowest MD
        lab = _facies_track(n, PROPS[w.WELL], rng)
        base = {k: np.array([FACIES[f][k] for f in lab]) for k in ["qz", "carb", "clay", "toc", "ym", "pr"]}
        z = {k: gaussian_filter1d(rng.standard_normal(n), 6) * 2.4 for k in ["a", "b", "c", "d", "e"]}

        clay = np.clip(base["clay"] + 0.085 * z["a"], 0.03, 0.75)
        qz = np.clip(base["qz"] + 0.085 * z["b"], 0.05, 0.80)
        carb = np.clip(base["carb"] + 0.080 * z["c"], 0.01, 0.80)
        tot = qz + carb + clay
        scale = np.where(tot > 0.95, 0.95 / tot, 1.0)
        qz, carb, clay = qz * scale, carb * scale, clay * scale
        toc = np.clip(base["toc"] * (1 + 0.30 * z["d"]), 0.3, 9.0)
        ym = np.clip(base["ym"] * (1 + 0.16 * z["e"]) - 3.0 * (clay - base["clay"]), 1.2, 8.5)
        pr = np.clip(base["pr"] + 0.022 * z["a"] + 0.016 * gaussian_filter1d(rng.standard_normal(n), 4) * 2, 0.16, 0.39)

        # tool resolution: logs average across bed boundaries
        sm = lambda a: gaussian_filter1d(a, 1.2)
        qz, carb, clay, toc, ym, pr = map(sm, (qz, carb, clay, toc, ym, pr))
        tvd = w.TVD_FT + 15 * np.sin(np.arange(n) / 140.0) + gaussian_filter1d(rng.standard_normal(n), 30) * 20
        shmin = shmin_uniaxial(pr, ym, tvd, w.PP_GRAD)

        df = pd.DataFrame({
            "WELL": w.WELL, "MD_FT": md, "TVD_FT": tvd.round(1),
            "YM_MPSI": (ym + rng.normal(0, 0.25, n)).round(3),
            "PR": (pr + rng.normal(0, 0.012, n)).round(4),
            "QUARTZ": np.clip(qz + rng.normal(0, 0.035, n), 0, 1).round(3),
            "CARBONATE": np.clip(carb + rng.normal(0, 0.035, n), 0, 1).round(3),
            "CLAY": np.clip(clay + rng.normal(0, 0.035, n), 0, 1).round(3),
            "TOC": np.clip(toc + rng.normal(0, 0.25, n), 0.1, None).round(2),
            "_FACIES": lab, "_YM": ym, "_PR": pr, "_CLAY": clay, "_TOC": toc, "_SHMIN": shmin,
        })
        log_rows.append(df)

        # as-completed geometric design
        design = geometric_design(df, w.STAGE_LEN_FT, w.CLUSTERS)
        sr = simulate_design(df, design, w.PROPPANT_LB_FT, w.RATE_BPM, rng=rng)
        well_ip = sr.sum()
        contrib = sr / well_ip * 100 * rng.lognormal(0, 0.08, len(sr))   # PLT measurement error
        contrib = contrib / contrib.sum() * 100
        for k, (top, bot, cl) in enumerate(design):
            stage_rows.append({
                "WELL": w.WELL, "STAGE": k + 1, "TOP_MD": top, "BASE_MD": bot,
                "STAGE_LEN_FT": bot - top, "CLUSTERS": len(cl),
                "CLUSTER_MD": ";".join(f"{c:.0f}" for c in cl),
                "PROPPANT_LB_FT": round(w.PROPPANT_LB_FT * rng.normal(1, 0.06)),
                "FLUID_BBL_FT": round(w.FLUID_BBL_FT * rng.normal(1, 0.05), 1),
                "RATE_BPM": round(w.RATE_BPM + rng.normal(0, 3), 1),
                "PLT_CONTRIB_PCT": round(contrib[k], 3),
                "WELL_IP_BOPD": round(well_ip, 0),
                "_RATE_TRUE": sr[k],
            })
    return pd.concat(log_rows, ignore_index=True), pd.DataFrame(stage_rows)


# ----------------------------------------------------------------------------- designs and simulator
def geometric_design(df_well, stage_len, n_clusters):
    md = df_well.MD_FT.to_numpy()
    top, end = md[0], md[-1] + DX
    out = []
    while top + 0.5 * stage_len <= end:
        bot = min(top + stage_len, end)
        L = bot - top
        cl = top + (np.arange(n_clusters) + 0.5) * L / n_clusters
        out.append((float(top), float(bot), cl))
        top = bot
    return out


def _idx(df_well, mds):
    md0 = df_well.MD_FT.iloc[0]
    return np.clip(np.round((np.asarray(mds) - md0) / DX).astype(int), 0, len(df_well) - 1)


def simulate_design(df_well, design, proppant_lb_ft, rate_bpm, rng=None, noise=True):
    """Synthetic field 'truth': stage oil rate (bopd) from hidden rock properties.

    Fluid divides between clusters by their stress relative to the lowest-stress
    cluster (limited-entry imbalance). Each cluster's productivity depends on
    elastic brittleness, clay content, organic richness and proppant placed.
    """
    rates = []
    for top, bot, cl in design:
        if len(cl) == 0:
            rates.append(0.0)
            continue
        i = _idx(df_well, cl)
        s = df_well._SHMIN.to_numpy()[i]
        lam = 140 * (rate_bpm / 90)                 # higher rate spreads fluid more evenly
        eff = np.exp(-(s - s.min()) / lam)
        share = eff / eff.sum()
        bi = bi_elastic(df_well._YM.to_numpy()[i], df_well._PR.to_numpy()[i])
        clay = df_well._CLAY.to_numpy()[i]
        toc = df_well._TOC.to_numpy()[i]
        frac = 1 / (1 + np.exp(-(bi - 0.42) / 0.07)) * (1 - 0.5 * clay)
        q = 0.35 + toc / 5
        P = proppant_lb_ft * (bot - top) * share
        prod = q * frac * (1 - np.exp(-P / 45000))
        spacing = (bot - top) / len(cl)
        shadow = 1 - 0.35 * np.exp(-max(spacing - 15, 0) / 15)
        r = 120 * prod.sum() * shadow
        if noise and rng is not None:
            r *= rng.lognormal(0, 0.15)
            if rng.random() < 0.04:
                r *= 0.3                              # screen-out
        rates.append(r)
    return np.array(rates)


def facies_design(df_well, labels, shmin_est, total_clusters, min_len=120, max_len=300,
                  min_spacing=20, skip_ductile=None):
    """Engineered design: stage breaks at facies changes, clusters in the lowest-stress rock."""
    md = df_well.MD_FT.to_numpy()
    n = len(md)
    cuts = [0] + list(np.flatnonzero(np.diff(labels)) + 1) + [n]
    segs = [[cuts[k], cuts[k + 1]] for k in range(len(cuts) - 1)]
    minn = int(min_len / DX)
    while len(segs) > 1:
        lens = [b - a for a, b in segs]
        j = int(np.argmin(lens))
        if lens[j] >= minn:
            break
        sm = lambda s: shmin_est[s[0]:s[1]].mean()
        left = abs(sm(segs[j - 1]) - sm(segs[j])) if j > 0 else np.inf
        right = abs(sm(segs[j + 1]) - sm(segs[j])) if j < len(segs) - 1 else np.inf
        if left <= right:
            segs[j - 1][1] = segs[j][1]
        else:
            segs[j + 1][0] = segs[j][0]
        segs.pop(j)
    maxn = int(max_len / DX)
    out_segs = []
    for a, b in segs:
        k = int(np.ceil((b - a) / maxn))
        edges = np.linspace(a, b, k + 1).round().astype(int)
        out_segs += [(edges[i], edges[i + 1]) for i in range(k)]

    dom = [np.bincount(labels[a:b], minlength=labels.max() + 1).argmax() for a, b in out_segs]
    active = [not (skip_ductile is not None and d == skip_ductile and
                   (labels[a:b] == d).mean() > 0.8) for (a, b), d in zip(out_segs, dom)]
    act_len = sum((b - a) for (a, b), ok in zip(out_segs, active) if ok)
    raw = np.array([total_clusters * (b - a) / act_len if ok else 0.0 for (a, b), ok in zip(out_segs, active)])
    alloc = np.where(np.array(active), np.maximum(2, np.floor(raw)), 0).astype(int)
    for j in np.argsort(-(raw - np.floor(raw))):
        if alloc.sum() >= total_clusters:
            break
        if active[j]:
            alloc[j] += 1
    design = []
    for (a, b), ok, nc in zip(out_segs, active, alloc):
        top, bot = md[a], (md[b - 1] + DX)
        if not ok:
            design.append((float(top), float(bot), np.array([])))
            continue
        order = np.argsort(shmin_est[a:b])
        chosen = []
        for o in order:
            if all(abs(o - c) * DX >= min_spacing for c in chosen):
                chosen.append(o)
            if len(chosen) == nc:
                break
        design.append((float(top), float(bot), np.sort(md[a + np.array(chosen)])))
    return design


# ----------------------------------------------------------------------------- task 1
def add_indices(logs, eps_t=EPS_T_TRUE):
    out = logs.copy()
    out["BI_MIN"] = bi_mineral(out.QUARTZ, out.CARBONATE, out.CLAY)
    out["BI_EL"] = bi_elastic(out.YM_MPSI, out.PR)
    w = WELLS.set_index("WELL")
    out["SHMIN_PSI"] = shmin_uniaxial(out.PR, out.YM_MPSI, out.TVD_FT,
                                      out.WELL.map(w.PP_GRAD), eps_t)
    return out


# ----------------------------------------------------------------------------- task 2
def mode_filter(labels, window):
    if window <= 1:
        return labels
    k = labels.max() + 1
    return generic_filter(labels, lambda v: np.bincount(v.astype(int), minlength=k).argmax(),
                          size=window, mode="nearest").astype(int)


def cluster_facies(logs, features, k, smooth_ft=0, seed=0):
    X = StandardScaler().fit_transform(logs[features].to_numpy())
    km = KMeans(k, n_init=10, random_state=seed).fit(X)
    raw = km.labels_
    # order clusters from most to least brittle (elastic) so GF1 is always the best rock
    order = logs.groupby(raw).BI_EL.mean().sort_values(ascending=False).index.to_numpy()
    remap = np.empty(k, int)
    remap[order] = np.arange(k)
    lab = remap[raw]
    win = int(smooth_ft / DX) | 1 if smooth_ft else 1
    if win > 1:
        lab = np.concatenate([mode_filter(lab[g.index.to_numpy()], win)
                              for _, g in logs.groupby("WELL", sort=False)])
    return lab


def elbow_table(logs, features, ks=range(3, 7), seed=0):
    X = StandardScaler().fit_transform(logs[features].to_numpy())
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(X), min(3000, len(X)), replace=False)
    rows = []
    for k in ks:
        km = KMeans(k, n_init=10, random_state=seed).fit(X)
        rows.append({"K": k, "Inertia": km.inertia_ / len(X), "Silhouette": silhouette_score(X[sub], km.labels_[sub])})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- task 3
def stage_features(df_well, design, labels, k, completion):
    """Aggregate logs over each stage. completion: dict or list of dicts per stage."""
    md = df_well.MD_FT.to_numpy()
    rows = []
    for j, (top, bot, cl) in enumerate(design):
        sel = (md >= top) & (md < bot)
        lab = labels[sel]
        g = df_well.loc[sel]
        c = completion[j] if isinstance(completion, list) else completion
        row = {f"FRAC_GF{f + 1}": (lab == f).mean() for f in range(k)}
        row["FACIES_PURITY"] = np.bincount(lab, minlength=k).max() / len(lab)
        row["YM_MEAN"] = g.YM_MPSI.mean()
        row["PR_MEAN"] = g.PR.mean()
        row["BI_MIN_MEAN"] = g.BI_MIN.mean()
        row["BI_EL_MEAN"] = g.BI_EL.mean()
        row["CLAY_MEAN"] = g.CLAY.mean()
        row["TOC_MEAN"] = g.TOC.mean()
        row["SHMIN_MEAN"] = g.SHMIN_PSI.mean()
        row["SHMIN_RANGE"] = g.SHMIN_PSI.quantile(0.9) - g.SHMIN_PSI.quantile(0.1)
        if len(cl):
            s = df_well.SHMIN_PSI.to_numpy()[_idx(df_well, cl)]
            row["CLUSTER_STRESS_SPREAD"] = s.max() - s.min()
            row["CLUSTER_BI_EL"] = df_well.BI_EL.to_numpy()[_idx(df_well, cl)].mean()
        else:
            row["CLUSTER_STRESS_SPREAD"] = np.nan
            row["CLUSTER_BI_EL"] = np.nan
        if "FI" in df_well:                                  # v1.1 indices
            row["FI_MEAN"] = g.FI.mean()
            row["UCS_MEAN"] = g.UCS_PSI.mean() if "UCS_PSI" in g else np.nan
            if len(cl):
                ii = _idx(df_well, cl)
                row["CLUSTER_FI"] = df_well.FI.to_numpy()[ii].mean()
                row["CQS"] = completion_quality(df_well.FI.to_numpy()[ii], df_well.SHMIN_PSI.to_numpy()[ii])
            else:
                row["CLUSTER_FI"] = row["CQS"] = np.nan
        row["CLUSTERS"] = len(cl)
        row["STAGE_LEN_FT"] = bot - top
        row["CLUSTER_SPACING_FT"] = (bot - top) / max(len(cl), 1)
        row["PROPPANT_LB_FT"] = c["PROPPANT_LB_FT"]
        row["FLUID_BBL_FT"] = c["FLUID_BBL_FT"]
        row["RATE_BPM"] = c["RATE_BPM"]
        rows.append(row)
    return pd.DataFrame(rows)


def build_stage_table(logs_i, stages, labels, k):
    out = []
    for wname, g in logs_i.groupby("WELL", sort=False):
        st = stages[stages.WELL == wname].reset_index(drop=True)
        design = [(r.TOP_MD, r.BASE_MD, np.array([float(x) for x in r.CLUSTER_MD.split(";")])) for r in st.itertuples()]
        comp = st[["PROPPANT_LB_FT", "FLUID_BBL_FT", "RATE_BPM"]].to_dict("records")
        f = stage_features(g.reset_index(drop=True), design, labels[g.index.to_numpy()], k, comp)
        f.insert(0, "WELL", wname)
        f.insert(1, "STAGE", st.STAGE.to_numpy())
        rate = st.PLT_CONTRIB_PCT / 100 * st.WELL_IP_BOPD
        f["TARGET_BOPD_PER_100FT"] = (rate / st.STAGE_LEN_FT * 100).to_numpy()
        out.append(f)
    return pd.concat(out, ignore_index=True)


def make_model(n_estimators=300, max_depth=3, learning_rate=0.05, monotone=None):
    return xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                            monotone_constraints=monotone, random_state=0, n_jobs=2)


def cross_validate(table, features, params, scheme="lowo"):
    X, y, g = table[features], table.TARGET_BOPD_PER_100FT, table.WELL
    pred = np.zeros(len(y))
    splits = GroupKFold(6).split(X, y, g) if scheme == "lowo" else KFold(5, shuffle=True, random_state=0).split(X)
    for tr, te in splits:
        m = make_model(**params).fit(X.iloc[tr], y.iloc[tr])
        pred[te] = m.predict(X.iloc[te])
    per_well = pd.DataFrame({"WELL": g, "y": y, "p": pred}).groupby("WELL").apply(
        lambda d: pd.Series({"R2": r2_score(d.y, d.p), "MAE": mean_absolute_error(d.y, d.p)}), include_groups=False)
    return pred, r2_score(y, pred), mean_absolute_error(y, pred), per_well


def shap_values(model, X):
    contrib = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    return contrib[:, :-1], contrib[:, -1]


# ============================================================================= v1.1 additions
# Raw measurements behind the elastic logs (RHOB, DTC, DTS), core triaxial tests with UCS, and one DFIT
# closure per well. They come from their own random stream, so the v1.0 field and stage records are unchanged.
MPSI_TO_PA = 6.894757e9
RHO_GRAIN = dict(qz=2.65, carb=2.71, clay=2.70, ker=1.25)
CORE_WELLS = ["PB-2H", "PB-6H"]


def _static_ratio(clay):
    """Truth: static/dynamic Young's modulus ratio falls in softer, clay-rich rock."""
    return 0.85 - 0.6 * clay


def _ucs_truth_psi(ym_static_mpsi, carb, toc, rng):
    e_gpa = ym_static_mpsi * MPSI_TO_PA / 1e9
    mpa = 7.22 * e_gpa ** 0.712 * (1 + 0.5 * carb - 0.03 * toc) * rng.lognormal(0, 0.08, len(e_gpa))
    return mpa * 145.038


def make_measurements(logs, seed=SEED + 11):
    """Public sonic and density logs, core tests and DFITs from the hidden truth."""
    rng = np.random.default_rng(seed)
    out = logs.copy()
    clay, toc = out._CLAY.to_numpy(), out._TOC.to_numpy()
    qz, carb = out.QUARTZ.clip(0.01).to_numpy(), out.CARBONATE.clip(0.01).to_numpy()
    phi = np.clip(0.065 + 0.04 * gaussian_filter1d(rng.standard_normal(len(out)), 5) * 2 + 0.03 * (toc - 3) / 3, 0.02, 0.14)
    ker = np.clip(toc / 100 * 2.6 / 1.25, 0, 0.25)                 # wt% to volume of solids, roughly
    minerals = np.clip(1 - ker, 0.6, 1)
    tot = qz + carb + clay
    rho_ma = minerals * (qz * RHO_GRAIN["qz"] + carb * RHO_GRAIN["carb"] + clay * RHO_GRAIN["clay"]) / tot + ker * RHO_GRAIN["ker"]
    rhob = (1 - phi) * rho_ma + phi * 1.0
    e_dyn = out._YM.to_numpy() / _static_ratio(clay)
    nu_dyn = np.clip(out._PR.to_numpy() + 0.02, 0.1, 0.45)
    E = e_dyn * MPSI_TO_PA
    G = E / (2 * (1 + nu_dyn))
    K = E / (3 * (1 - 2 * nu_dyn))
    rho = rhob * 1000
    vp, vs = np.sqrt((K + 4 * G / 3) / rho), np.sqrt(G / rho)
    out["RHOB"] = (rhob + rng.normal(0, 0.012, len(out))).round(3)
    out["DTC"] = (304800 / vp * (1 + rng.normal(0, 0.008, len(out)))).round(2)
    out["DTS"] = (304800 / vs * (1 + rng.normal(0, 0.012, len(out)))).round(2)
    out["_UCS"] = _ucs_truth_psi(out._YM.to_numpy(), out._CLAY.to_numpy() * 0 + carb, toc, rng)

    rows = []
    for w in CORE_WELLS:
        g = out[out.WELL == w]
        for i in np.sort(rng.choice(len(g), 10, replace=False)):
            r = g.iloc[i]
            rows.append(dict(WELL=w, MD_FT=r.MD_FT, STATIC_YM_MPSI=round(r._YM * rng.lognormal(0, 0.06), 3),
                             STATIC_PR=round(r._PR + rng.normal(0, 0.01), 3), UCS_PSI=round(r._UCS * rng.lognormal(0, 0.07), -1)))
    core = pd.DataFrame(rows)

    drows = []
    for w, g in out.groupby("WELL", sort=False):
        i = int(rng.integers(int(0.3 * len(g)), int(0.7 * len(g))))
        r = g.iloc[i]
        drows.append(dict(WELL=w, MD_FT=r.MD_FT, TVD_FT=r.TVD_FT, CLOSURE_PSI=round(r._SHMIN + rng.normal(0, 60), -1)))
    return out, core, pd.DataFrame(drows)


PUBLIC_LOGS = ["WELL", "MD_FT", "TVD_FT", "RHOB", "DTC", "DTS", "QUARTZ", "CARBONATE", "CLAY", "TOC"]


def dynamic_moduli(logs):
    """Dynamic Young's modulus (Mpsi) and Poisson's ratio from RHOB (g/cc), DTC and DTS (us/ft)."""
    vp, vs = 304800 / logs.DTC.to_numpy(), 304800 / logs.DTS.to_numpy()
    rho = logs.RHOB.to_numpy() * 1000
    nu = (vp ** 2 - 2 * vs ** 2) / (2 * (vp ** 2 - vs ** 2))
    e = rho * vs ** 2 * (3 * vp ** 2 - 4 * vs ** 2) / (vp ** 2 - vs ** 2)
    return e / MPSI_TO_PA, nu


STATIC_METHODS = ["None (dynamic = static)", "Eissa and Kazi (1988)", "Calibrated to core tests"]
UCS_METHODS = ["Horsrud (2001), from DTC", "Horsrud (2001), from static E", "Calibrated to core tests"]


def _core_rows(logs, core):
    return core.merge(logs, on=["WELL", "MD_FT"], how="left")


def add_elastic(logs, core, static_method, ucs_method):
    """YM_MPSI and PR (static), YM_DYN and PR_DYN, and UCS_PSI, by the chosen methods. Returns (logs, fits)."""
    out = logs.copy()
    e_d, nu_d = dynamic_moduli(out)
    out["YM_DYN"], out["PR_DYN"] = e_d, nu_d
    fits = {}
    if static_method.startswith("None"):
        out["YM_MPSI"] = e_d
    elif static_method.startswith("Eissa"):
        e_gpa = e_d * MPSI_TO_PA / 1e9
        out["YM_MPSI"] = np.clip(0.74 * e_gpa - 0.82, 0.3, None) * 1e9 / MPSI_TO_PA
    else:
        c = _core_rows(out, core)
        b, a = np.polyfit(c.YM_DYN, c.STATIC_YM_MPSI, 1)
        fits["static"] = (a, b)
        out["YM_MPSI"] = np.clip(a + b * e_d, 0.3, None)
    out["PR"] = nu_d - (0.0 if static_method.startswith("None") else
                        float(np.mean(_core_rows(out, core).PR_DYN - core.STATIC_PR)) if static_method.startswith("Calibrated") else 0.0)
    e_gpa = out.YM_MPSI.to_numpy() * MPSI_TO_PA / 1e9
    if ucs_method.endswith("DTC"):
        out["UCS_PSI"] = 0.77 * (304.8 / out.DTC) ** 2.93 * 145.038
    elif "static E" in ucs_method:
        out["UCS_PSI"] = 7.22 * e_gpa ** 0.712 * 145.038
    else:
        c = _core_rows(out.assign(_EG=e_gpa), core)
        b, a = np.polyfit(np.log(c._EG), np.log(c.UCS_PSI), 1)
        fits["ucs"] = (np.exp(a), b)
        out["UCS_PSI"] = np.exp(a) * e_gpa ** b
    return out, fits


def core_check(logs, core):
    """Each method against the core tests (static E and UCS)."""
    rows = []
    for sm in STATIC_METHODS:
        for um in UCS_METHODS:
            L, _ = add_elastic(logs, core, sm, um)
            c = _core_rows(L, core)
            rows.append(dict(static=sm, ucs=um,
                             E_bias=float(np.mean(c.YM_MPSI - c.STATIC_YM_MPSI)), E_RMSE=float(np.sqrt(np.mean((c.YM_MPSI - c.STATIC_YM_MPSI) ** 2))),
                             UCS_bias=float(np.mean(c.UCS_PSI - c.UCS_PSI_y)) if "UCS_PSI_y" in c else np.nan))
    return pd.DataFrame(rows)


def fit_eps_t(logs, dfit):
    """Tectonic strain that best matches the DFIT closures (least squares), with the misfit per well."""
    w = WELLS.set_index("WELL")
    d = dfit.merge(logs[["WELL", "MD_FT", "PR", "YM_MPSI"]], on=["WELL", "MD_FT"])
    base = shmin_uniaxial(d.PR, d.YM_MPSI, d.TVD_FT, d.WELL.map(w.PP_GRAD), 0.0)
    slope = d.YM_MPSI * 1e6
    eps = float(np.sum(slope * (d.CLOSURE_PSI - base)) / np.sum(slope ** 2))
    d["PREDICTED_PSI"] = base + slope * eps
    d["MISFIT_PSI"] = d.PREDICTED_PSI - d.CLOSURE_PSI
    return eps, d


def add_fracability(logs):
    """Fracability index (0 to 1): elastic brittleness and low closure stress relative to the rest of the lateral."""
    out = logs.copy()
    lo = out.groupby("WELL").SHMIN_PSI.transform(lambda s: s.quantile(0.05))
    hi = out.groupby("WELL").SHMIN_PSI.transform(lambda s: s.quantile(0.95))
    s_n = ((out.SHMIN_PSI - lo) / (hi - lo)).clip(0, 1)
    out["FI"] = 0.5 * out.BI_EL + 0.5 * (1 - s_n)
    return out


def completion_quality(cluster_fi, cluster_shmin, entry_psi=150.0):
    """Completion quality score (0 to 100) for one stage: cluster fracability times fluid-entry uniformity.

    Fluid enters the lowest-stress clusters first: each cluster's expected share falls off as
    exp(-(S - S_min) / entry_psi), with entry_psi a typical limited-entry perforation-friction scale.
    Uniformity is the effective number of clusters taking fluid, (sum w)^2 / sum w^2, over the number placed.
    """
    s = np.asarray(cluster_shmin, float)
    w = np.exp(-(s - s.min()) / entry_psi)
    uniformity = (w.sum() ** 2 / np.sum(w ** 2)) / len(w)
    return float(100 * np.mean(np.asarray(cluster_fi, float)) * uniformity)
