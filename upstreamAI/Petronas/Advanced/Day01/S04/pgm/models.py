"""Classifiers, validation curves, business thresholds, completion simulator and drift checks."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import build, engineered_boundaries, lateral_logs, p_effective, stage_table, truth_logit

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:  # pragma: no cover
    HAS_XGB = False

FEATURE_SETS = {
    "Full geomechanical (logs + design)": ["YM_GPA", "PR", "BI", "SHMIN_PSI", "STRESS_RANGE_PSI", "NFI", "CLAY",
                                           "TOC", "GR", "OOZ_FRAC", "FAULT_DIST_MIN", "CLUSTERS", "PROPPANT_LB_FT"],
    "Real-time drilling (GR, MSE, landing, design)": ["GR", "MSE_KSI", "MSE_CV", "TVD_DEV_ABS", "OOZ_FRAC",
                                                      "CLUSTERS", "PROPPANT_LB_FT"],
}
MODELS = ["Logistic regression", "Random Forest", "XGBoost"]


def make_model(name, seed=0):
    if name == "Logistic regression":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
    if name == "XGBoost" and HAS_XGB:
        return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, random_state=seed, n_jobs=-1, eval_metric="logloss")
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=seed, n_jobs=-1)


def lowo_proba(st: pd.DataFrame, feats, model_name, group="WELL"):
    p = np.empty(len(st))
    for g in st[group].unique():
        tr, te = (st[group] != g).values, (st[group] == g).values
        m = make_model(model_name).fit(st.loc[tr, feats], st.EFFECTIVE[tr])
        p[te] = m.predict_proba(st.loc[te, feats])[:, 1]
    return p


def rule_baseline(st):
    """Industry rule of thumb: brittleness index scaled to 0-1."""
    return (st.BI / 100).clip(0, 1).to_numpy()


def curves(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    return dict(fpr=fpr, tpr=tpr, prec=prec, rec=rec, auc=roc_auc_score(y, p), ap=average_precision_score(y, p))


def at_threshold(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), precision=tp / max(tp + fp, 1),
                recall=tp / max(tp + fn, 1), accuracy=(tp + tn) / len(y))


def value_curve(y, p, value_tp, cost_fp, value_tn, cost_fn, thresholds=np.linspace(0.05, 0.95, 91)):
    """Business value of acting on the model at each threshold.

    Acting = full completion on stages predicted effective, reduced (saved) completion elsewhere.
    value_tp: gain from a correctly completed effective stage; cost_fp: spend wasted on a dud;
    value_tn: saving on a correctly de-scoped dud; cost_fn: value lost by de-scoping a good stage.
    """
    out = []
    for t in thresholds:
        r = at_threshold(y, p, t)
        out.append(dict(threshold=t, value=r["tp"] * value_tp - r["fp"] * cost_fp + r["tn"] * value_tn - r["fn"] * cost_fn))
    return pd.DataFrame(out)


def importance(st, feats, model_name="Random Forest"):
    rows = []
    for g in st.WELL.unique():
        tr, te = (st.WELL != g).values, (st.WELL == g).values
        m = make_model(model_name).fit(st.loc[tr, feats], st.EFFECTIVE[tr])
        pi = permutation_importance(m, st.loc[te, feats], st.EFFECTIVE[te], scoring="roc_auc", n_repeats=3, random_state=0)
        rows.append(pi.importances_mean)
    return pd.DataFrame({"feature": feats, "importance": np.mean(rows, axis=0)}).sort_values("importance", ascending=False)


def simulate_completion(logs, wells, design, well, strategy, skip_thr=None, model=None, feats=None, cut=None):
    """Expected effective stages and cost for a completion strategy on one well (truth simulator)."""
    g = logs[logs.WELL == well]
    if strategy == "Engineered (stress-based) staging":
        edges = {well: engineered_boundaries(g)}
        st = stage_table(g, boundaries=edges, design={well: design[well]})
    else:
        st = stage_table(g, design={well: design[well]})
    st["P_TRUE"] = p_effective(truth_logit(st), cut if cut is not None else 0.0)
    st["COMPLETED"] = True
    if skip_thr is not None and model is not None:
        st["P_MODEL"] = model.predict_proba(st[feats])[:, 1]
        st["COMPLETED"] = st.P_MODEL >= skip_thr
    return st


def psi(expected, actual, bins=10):
    """Population stability index of a feature between training and a new well."""
    q = np.quantile(expected, np.linspace(0, 1, bins + 1))
    q[0], q[-1] = -np.inf, np.inf
    e = np.histogram(expected, q)[0] / len(expected) + 1e-4
    a = np.histogram(actual, q)[0] / len(actual) + 1e-4
    return float(np.sum((a - e) * np.log(a / e)))


def new_well(seed=99, bench="Wolfcamp B", y=13500.0, drift=0.0):
    """A well being drilled now, optionally in rock that differs from the training wells."""
    rng = np.random.default_rng(seed)
    g = lateral_logs(rng, "W-13", bench, y, 5, 2200)
    if drift:
        g["CLAY"] = (g.CLAY + drift).clip(0, 0.9)
        g["GR"] = g.GR + 170 * drift
        g["MSE_KSI"] = g.MSE_KSI * (1 - 0.8 * drift)
    return g


def psi_reference(st: pd.DataFrame, feat: str) -> np.ndarray:
    """PSI of each training well against all the others: how different an ordinary well already is."""
    return np.array([psi(st.loc[st.WELL != w, feat].to_numpy(), st.loc[st.WELL == w, feat].to_numpy())
                     for w in st.WELL.unique()])
