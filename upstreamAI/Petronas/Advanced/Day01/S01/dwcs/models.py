"""
Analytics for the case study. Every model is scored leave-one-well-out, and every
preprocessing step that learns from data is fitted on training wells only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score

from .data import CLAYS, MINERALS, STEP_FT, Field, weight_to_vol

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:  # pragma: no cover
    HAS_XGB = False

LOG_FEATURES = ["GR", "RHOB", "NPHI", "DT", "PEF", "LOG_RT", "ND_SEP"]
CLR_FEATURES = [f"CLR_{m}" for m in MINERALS]
RE_FEATURES = ["LOG_S1", "LOG_S2", "LOG_TOC"]
SCENARIOS = {
    "A · Logs only": "Deployable in any well.",
    "B · Logs + measured XRD and Rock-Eval": "Upper bound: needs geochemistry at every depth.",
    "C · Logs + log-predicted mineralogy and S2": "Deployable: geochemistry learned from logs in cored wells.",
}


# ---------------------------------------------------------------------------
# generic
# ---------------------------------------------------------------------------

def clr(frac: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Centred log-ratio (Aitchison) for closed compositions."""
    lx = np.log(np.clip(frac, eps, None))
    return lx - lx.mean(axis=1, keepdims=True)


def add_log_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["LOG_RT"] = np.log10(out["RT"])
    out["ND_SEP"] = out["NPHI"] - (2.71 - out["RHOB"]) / 1.71
    return out


def make_regressor(kind: str, n_trees: int = 250, seed: int = 0):
    if kind == "XGBoost" and HAS_XGB:
        return XGBRegressor(n_estimators=n_trees * 2, max_depth=4, learning_rate=0.04,
                            subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                            min_child_weight=3, random_state=seed, n_jobs=-1)
    if kind == "XGBoost":
        return HistGradientBoostingRegressor(max_iter=n_trees * 2, learning_rate=0.04,
                                             max_depth=4, random_state=seed)
    return RandomForestRegressor(n_estimators=n_trees, min_samples_leaf=3,
                                 max_features=0.5, random_state=seed, n_jobs=-1)


def metrics(y, p) -> dict:
    return {"R2": float(r2_score(y, p)), "RMSE": float(mean_squared_error(y, p) ** 0.5), "N": int(len(y))}


# ---------------------------------------------------------------------------
# 1. QC and alignment
# ---------------------------------------------------------------------------

def shift_curve(field: Field, well: str, lags=np.arange(-8, 8.5, 0.5)):
    """Correlation of core gamma with log GR for each candidate shift (core - log, ft)."""
    cg = field.core_gamma[field.core_gamma.WELL == well]
    lg = field.logs[field.logs.WELL == well].sort_values("DEPTH")
    out = []
    for lag in lags:
        gr = np.interp(cg.CORE_DEPTH - lag, lg.DEPTH, lg.GR)
        out.append(np.corrcoef(cg.CORE_GR, gr)[0, 1])
    out = np.array(out)
    return lags, out, float(lags[int(np.nanargmax(out))])


def estimated_shifts(field: Field) -> pd.DataFrame:
    rows = []
    for w in sorted(field.core_gamma.WELL.unique()):
        lags, cc, best = shift_curve(field, w)
        rows.append({"WELL": w, "SHIFT_FT": best, "CC_AT_BEST": float(cc.max()),
                     "CC_AT_ZERO": float(cc[np.argmin(np.abs(lags))])})
    return pd.DataFrame(rows)


def plug_table(field: Field, apply_shift=True, apply_ob=True, reject_rubble=True,
               ob_k=(0.72, 0.85), ob_phi=0.965, s2_cut=4.0) -> pd.DataFrame:
    """Core plugs matched to logs after the chosen QC steps."""
    p = field.plugs.copy()
    shifts = estimated_shifts(field).set_index("WELL").SHIFT_FT if apply_shift else None
    p["LOG_DEPTH"] = p.CORE_DEPTH - (p.WELL.map(shifts) if apply_shift else 0.0)
    p["LOG_DEPTH"] = np.round(p.LOG_DEPTH / STEP_FT) * STEP_FT

    logs = add_log_features(field.logs)
    keep = ["WELL", "DEPTH", "FACIES", "INTERVAL"] + ["GR", "RHOB", "NPHI", "DT", "PEF", "RT"] + \
           ["LOG_RT", "ND_SEP"]
    m = p.merge(logs[keep].rename(columns={"INTERVAL": "LOG_INTERVAL"}),
                left_on=["WELL", "LOG_DEPTH"], right_on=["WELL", "DEPTH"], how="left")
    m = m[m.GR.notna()].copy()

    fk = np.where(m.INTERVAL == "Sandstone", ob_k[0], ob_k[1]) if apply_ob else 1.0
    m["PHI"] = m.PHI_AMB * (ob_phi if apply_ob else 1.0)
    m["K"] = m.K_AMB * fk
    m["LOG_K"] = np.log10(m.K)
    if reject_rubble:
        m = m[m.PLUG_NOTE == "intact"].copy()

    # geochemistry: weight % -> volume fraction of solids, clr, Rock-Eval logs
    geo = m[f"XRD_{MINERALS[0]}"].notna()
    wt = m.loc[geo, [f"XRD_{x}" for x in MINERALS]].to_numpy() / 100
    vol = weight_to_vol(wt)
    for j, x in enumerate(MINERALS):
        m.loc[geo, f"V_{x}"] = vol[:, j]
    c = clr(vol)
    for j, x in enumerate(MINERALS):
        m.loc[geo, f"CLR_{x}"] = c[:, j]
    m["CLAY_SOLID"] = m[[f"V_{x}" for x in CLAYS]].sum(axis=1, min_count=1)
    m["VCLAY_XRD"] = m.CLAY_SOLID * (1 - m.PHI)
    m["LOG_S1"] = np.log10(m.RE_S1.clip(lower=0.01))
    m["LOG_S2"] = np.log10(m.RE_S2.clip(lower=0.01))
    m["LOG_TOC"] = np.log10(m.RE_TOC.clip(lower=0.01))
    m["HI"] = 100 * m.RE_S2 / m.RE_TOC
    m["GEO"] = geo.values
    # decision 3: carbonate samples with anomalous S2 form a bitumen-affected class
    m["BIT_FLAG"] = ((m.INTERVAL == "Carbonate") & (m.RE_S2 > s2_cut)).astype(float)
    m.loc[~m.GEO, "BIT_FLAG"] = np.nan
    return m.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Mineralogy and Vclay
# ---------------------------------------------------------------------------

def vclay_methods(df: pd.DataFrame, logs: pd.DataFrame) -> pd.DataFrame:
    """Linear, Steiber and Clavier Vclay; GR end-points from each well's interval (P5/P95)."""
    out = df.copy()
    ends = logs.groupby(["WELL", "INTERVAL"]).GR.quantile([0.05, 0.95]).unstack()
    ends.columns = ["GR_MIN", "GR_MAX"]
    out = out.join(ends, on=["WELL", "INTERVAL"])
    igr = ((out.GR - out.GR_MIN) / (out.GR_MAX - out.GR_MIN)).clip(0, 1)
    out["IGR"] = igr
    out["VCL_LINEAR"] = igr
    out["VCL_STEIBER"] = igr / (3 - 2 * igr)
    out["VCL_CLAVIER"] = (1.7 - np.sqrt(np.clip(3.38 - (igr + 0.7) ** 2, 0, None))).clip(0, 1)
    return out


def lowo_ml_vclay(df: pd.DataFrame, n_trees=200) -> np.ndarray:
    pred = np.full(len(df), np.nan)
    X = df[LOG_FEATURES + ["IS_CB"]].to_numpy()
    for w in df.WELL.unique():
        tr, te = df.WELL != w, df.WELL == w
        mdl = RandomForestRegressor(n_estimators=n_trees, min_samples_leaf=3, random_state=0, n_jobs=-1)
        mdl.fit(X[tr], df.VCLAY_XRD[tr])
        pred[te.values] = mdl.predict(X[te])
    return pred


def closure_demo(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = df[[f"V_{m}" for m in MINERALS]].dropna()
    raw.columns = MINERALS
    c = pd.DataFrame(clr(raw.to_numpy()), columns=MINERALS)
    return raw.corr(), c.corr()


# ---------------------------------------------------------------------------
# 3. Rock typing
# ---------------------------------------------------------------------------

def rqi_fzi(k, phi):
    rqi = 0.0314 * np.sqrt(k / phi)
    phiz = phi / (1 - phi)
    return rqi, phiz, rqi / phiz


def rock_types(df: pd.DataFrame, n_types=4, seed=0) -> pd.DataFrame:
    out = df.copy()
    out["RQI"], out["PHIZ"], out["FZI"] = rqi_fzi(out.K, out.PHI)
    out["RT_LABEL"] = -1
    for iv in out.INTERVAL.unique():
        s = out.INTERVAL == iv
        x = np.log10(out.loc[s, ["FZI"]].to_numpy())
        km = KMeans(n_clusters=n_types, n_init=10, random_state=seed).fit(x)
        order = np.argsort(km.cluster_centers_.ravel())          # RT1 = poorest
        remap = {c: i + 1 for i, c in enumerate(order)}
        out.loc[s, "RT_LABEL"] = [remap[c] for c in km.labels_]
    out["ROCK_TYPE"] = out.INTERVAL.str[:2].str.upper() + "-RT" + out.RT_LABEL.astype(str)
    return out


def lowo_rock_type(df: pd.DataFrame, n_trees=250) -> tuple[np.ndarray, dict]:
    pred = np.empty(len(df), dtype=object)
    for iv in df.INTERVAL.unique():
        d = df[df.INTERVAL == iv]
        X = d[LOG_FEATURES].to_numpy()
        for w in d.WELL.unique():
            tr, te = (d.WELL != w).values, (d.WELL == w).values
            clf = RandomForestClassifier(n_estimators=n_trees, min_samples_leaf=2,
                                         class_weight="balanced", random_state=0, n_jobs=-1)
            clf.fit(X[tr], d.ROCK_TYPE.values[tr])
            pred[d.index[te]] = clf.predict(X[te])
    acc = accuracy_score(df.ROCK_TYPE, pred)
    base = df.groupby("INTERVAL").ROCK_TYPE.agg(lambda s: s.value_counts(normalize=True).iloc[0])
    wts = df.INTERVAL.value_counts(normalize=True)
    return pred, {"accuracy": float(acc), "majority_baseline": float((base * wts).sum())}


def rt_confusion(df, pred, interval):
    s = (df.INTERVAL == interval).values
    labels = sorted(df.ROCK_TYPE[s].unique())
    return labels, confusion_matrix(df.ROCK_TYPE[s], pred[s], labels=labels)


# ---------------------------------------------------------------------------
# 4. Permeability scenarios
# ---------------------------------------------------------------------------

AUX_TARGETS = CLR_FEATURES + ["LOG_S2"]


def _aux_fit_predict(train: pd.DataFrame, apply: pd.DataFrame, n_trees=150) -> pd.DataFrame:
    """Learn clr mineralogy and log S2 from logs; returns predicted columns P_*."""
    X = train[LOG_FEATURES + ["IS_CB"]].to_numpy()
    mdl = RandomForestRegressor(n_estimators=n_trees, min_samples_leaf=3, random_state=0, n_jobs=-1)
    mdl.fit(X, train[AUX_TARGETS].to_numpy())
    p = mdl.predict(apply[LOG_FEATURES + ["IS_CB"]].to_numpy())
    return pd.DataFrame(p, columns=[f"P_{c}" for c in AUX_TARGETS], index=apply.index)


def _cross_fitted_aux(train: pd.DataFrame, n_trees=150) -> pd.DataFrame:
    """Out-of-fold (by well) auxiliary predictions for the training wells."""
    parts = []
    for w in train.WELL.unique():
        parts.append(_aux_fit_predict(train[train.WELL != w], train[train.WELL == w], n_trees))
    return pd.concat(parts).loc[train.index]


def scenario_features(scenario: str, pooled: bool, bitumen_flag: bool = False) -> list[str]:
    f = list(LOG_FEATURES)
    if scenario.startswith("B"):
        f += CLR_FEATURES + RE_FEATURES
        if bitumen_flag:
            f += ["BIT_FLAG"]
    elif scenario.startswith("C"):
        f += [f"P_{c}" for c in AUX_TARGETS]
    if pooled:
        f += ["IS_CB"]
    return f


def lowo_permeability(df: pd.DataFrame, scenario: str, kind: str, pooled: bool,
                      n_trees=250, coverage=0.8, bitumen_flag=False) -> pd.DataFrame:
    """Leave-one-well-out log10 k predictions with P10/P90 from trees and from conformal."""
    d = df[df.GEO].copy()
    d["IS_CB"] = (d.INTERVAL == "Carbonate").astype(int)
    feats = scenario_features(scenario, pooled, bitumen_flag)
    groups = [("All", d)] if pooled else [(iv, d[d.INTERVAL == iv]) for iv in d.INTERVAL.unique()]
    rows = []
    for _, g in groups:
        for w in g.WELL.unique():
            tr, te = g[g.WELL != w].copy(), g[g.WELL == w].copy()
            if scenario.startswith("C"):
                tr = tr.join(_cross_fitted_aux(tr))
                te = te.join(_aux_fit_predict(tr, te))
            mdl = make_regressor(kind, n_trees)
            mdl.fit(tr[feats].to_numpy(), tr.LOG_K.to_numpy())
            p = mdl.predict(te[feats].to_numpy())
            if isinstance(mdl, RandomForestRegressor):
                trees = np.stack([t.predict(te[feats].to_numpy()) for t in mdl.estimators_])
                lo, hi = np.percentile(trees, [(1 - coverage) / 2 * 100, (1 + coverage) / 2 * 100], axis=0)
            else:
                lo = hi = np.full(len(te), np.nan)
            rows.append(pd.DataFrame({"idx": te.index, "PRED": p, "TREE_LO": lo, "TREE_HI": hi}))
    r = pd.concat(rows).set_index("idx")
    out = d.join(r)
    out["RESID"] = out.LOG_K - out.PRED
    # cross-conformal: half-width from the other wells' residuals (same interval)
    hw = np.empty(len(out))
    for i, (w, iv) in enumerate(zip(out.WELL, out.INTERVAL)):
        other = out[(out.WELL != w) & (out.INTERVAL == iv)].RESID.abs()
        hw[i] = np.quantile(other, coverage)
    out["CONF_LO"], out["CONF_HI"] = out.PRED - hw, out.PRED + hw
    return out


def coverage_table(res: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for iv, g in res.groupby("INTERVAL"):
        tree = ((g.LOG_K >= g.TREE_LO) & (g.LOG_K <= g.TREE_HI)).mean() if g.TREE_LO.notna().any() else np.nan
        conf = ((g.LOG_K >= g.CONF_LO) & (g.LOG_K <= g.CONF_HI)).mean()
        rows.append({"INTERVAL": iv, "Tree-spread P10–P90": tree, "Cross-conformal P10–P90": conf,
                     "Median band width (decades)": float((g.CONF_HI - g.CONF_LO).median())})
    return pd.DataFrame(rows)


def scenario_scores(df: pd.DataFrame, kind: str, pooled: bool, n_trees=250,
                    bitumen_flag=False) -> pd.DataFrame:
    rows = []
    for sc in SCENARIOS:
        r = lowo_permeability(df, sc, kind, pooled, n_trees, bitumen_flag=bitumen_flag)
        for iv, g in r.groupby("INTERVAL"):
            rows.append({"Scenario": sc, "INTERVAL": iv, **metrics(g.LOG_K, g.PRED)})
    return pd.DataFrame(rows)


def lowo_importance(df: pd.DataFrame, scenario: str, interval: str, kind="Random Forest",
                    n_trees=200, bitumen_flag=False) -> pd.DataFrame:
    """Permutation importance on each held-out well, averaged (R2 drop)."""
    d = df[df.GEO & (df.INTERVAL == interval)].copy()
    d["IS_CB"] = int(interval == "Carbonate")
    feats = scenario_features(scenario, pooled=False, bitumen_flag=bitumen_flag)
    acc = []
    for w in d.WELL.unique():
        tr, te = d[d.WELL != w].copy(), d[d.WELL == w].copy()
        if scenario.startswith("C"):
            tr = tr.join(_cross_fitted_aux(tr))
            te = te.join(_aux_fit_predict(tr, te))
        mdl = make_regressor(kind, n_trees).fit(tr[feats].to_numpy(), tr.LOG_K.to_numpy())
        pi = permutation_importance(mdl, te[feats].to_numpy(), te.LOG_K.to_numpy(),
                                    n_repeats=4, random_state=0, scoring="r2")
        acc.append(pi.importances_mean)
    imp = np.mean(acc, axis=0)
    return pd.DataFrame({"feature": feats, "importance": imp}).sort_values("importance", ascending=False)


# ---------------------------------------------------------------------------
# 5. Well-test check
# ---------------------------------------------------------------------------

def smearing_factor(res: pd.DataFrame) -> dict:
    """Duan smearing: mean of 10**residual corrects the log-to-linear back-transform bias."""
    return res.groupby("INTERVAL").RESID.apply(lambda r: float(np.mean(10 ** r))).to_dict()


def kh_check(field: Field, df: pd.DataFrame, scenario: str, kind: str, n_trees=250,
             smear: dict | None = None, linear_mean: bool = False) -> pd.DataFrame:
    """Train a deployable model on all cored geochem plugs, apply to every well's oil leg."""
    if scenario.startswith("B"):
        raise ValueError("Scenario B needs measured geochemistry at every depth; it cannot be deployed.")
    d = df[df.GEO].copy()
    d["IS_CB"] = (d.INTERVAL == "Carbonate").astype(int)
    logs = add_log_features(field.logs)
    logs["IS_CB"] = (logs.INTERVAL == "Carbonate").astype(int)
    rows = []
    for iv in ["Sandstone", "Carbonate"]:
        tr = d[d.INTERVAL == iv].copy()
        ap = logs[logs.INTERVAL == iv].copy()
        feats = scenario_features(scenario, pooled=False)
        if scenario.startswith("C"):
            tr = tr.join(_cross_fitted_aux(tr))
            ap = ap.join(_aux_fit_predict(d[d.INTERVAL == iv], ap))
        mdl = make_regressor(kind, n_trees).fit(tr[feats].to_numpy(), tr.LOG_K.to_numpy())
        Xa = ap[feats].to_numpy()
        if linear_mean and isinstance(mdl, RandomForestRegressor):
            # average the trees in linear k space instead of log space
            ap["K_PRED"] = np.mean([10 ** t.predict(Xa) for t in mdl.estimators_], axis=0)
        else:
            ap["K_PRED"] = 10 ** mdl.predict(Xa) * (smear or {}).get(iv, 1.0)
        for _, t in field.tests[field.tests.INTERVAL == iv].iterrows():
            seg = ap[(ap.WELL == t.WELL) & (ap.DEPTH >= t.TEST_TOP) & (ap.DEPTH <= t.TEST_BASE)]
            rows.append({"WELL": t.WELL, "INTERVAL": iv,
                         "CORED": bool(field.wells.set_index("WELL").CORED[t.WELL]),
                         "KH_TEST": t.KH_TEST_MDFT, "KH_MODEL": float((seg.K_PRED * STEP_FT).sum())})
    out = pd.DataFrame(rows)
    out["RATIO"] = out.KH_MODEL / out.KH_TEST
    return out
