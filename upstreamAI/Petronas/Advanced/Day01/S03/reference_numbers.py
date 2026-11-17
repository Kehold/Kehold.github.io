"""Every number quoted in the v1.1 guides and the Day 1 Session 4 deck, from the app's default settings."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.model_selection import GroupKFold, KFold
import xgboost as xgb
import geomech_core as gc

logs, stages = gc.make_field()
meas, core, dfit = gc.make_measurements(logs)
pub = meas[gc.PUBLIC_LOGS + [c for c in meas if c.startswith("_")]]
L, fits = gc.add_elastic(pub, core, gc.STATIC_METHODS[2], gc.UCS_METHODS[2])
eps, dtab = gc.fit_eps_t(L, dfit)
L = gc.add_fracability(gc.add_indices(L, eps))
FEATS, K, SM = ["YM_MPSI", "PR", "BI_MIN", "CLAY", "TOC"], 4, 25
lab = gc.cluster_facies(L, FEATS, K, SM)
T = gc.build_stage_table(L, stages, lab, K)
geo = [c for c in T.columns if c.startswith("FRAC_GF")] + ["FACIES_PURITY", "YM_MEAN", "PR_MEAN", "BI_MIN_MEAN", "BI_EL_MEAN", "CLAY_MEAN",
       "TOC_MEAN", "SHMIN_MEAN", "SHMIN_RANGE", "CLUSTER_STRESS_SPREAD", "CLUSTER_BI_EL", "FI_MEAN", "CLUSTER_FI", "CQS", "UCS_MEAN"]
comp = ["CLUSTERS", "STAGE_LEN_FT", "CLUSTER_SPACING_FT", "PROPPANT_LB_FT", "FLUID_BBL_FT", "RATE_BPM"]
MF = geo + comp
P = dict(n_estimators=300, max_depth=3, learning_rate=0.05)
out = {}
g1 = L[L.WELL == "PB-1H"]
out["task1"] = dict(eps=eps, dfit_rms=float(np.sqrt((dtab.MISFIT_PSI ** 2).mean())), r_bi=float(np.corrcoef(L.BI_MIN, L.BI_EL)[0, 1]),
                    pb1_bi_el=float(g1.BI_EL.median()), pb1_bi_min=float(g1.BI_MIN.median()), pb1_spread=float(g1.SHMIN_PSI.quantile(.9) - g1.SHMIN_PSI.quantile(.1)),
                    fits={k: list(map(float, v)) for k, v in fits.items()})
el = gc.elbow_table(L, FEATS); out["elbow"] = el.round(3).to_dict("records")
L2 = L.assign(GF=lab)
prof = L2.groupby("GF").agg(SHARE=("GF", "size"), E=("YM_MPSI", "mean"), PR=("PR", "mean"), BI_EL=("BI_EL", "mean"), BI_MIN=("BI_MIN", "mean"),
                            FI=("FI", "mean"), CLAY=("CLAY", "mean"), TOC=("TOC", "mean"), SHMIN=("SHMIN_PSI", "mean"), UCS=("UCS_PSI", "mean"))
prof["SHARE"] = prof.SHARE / prof.SHARE.sum(); out["profile"] = prof.round(3).to_dict("index")
out["ari"] = float(gc.adjusted_rand_score(L._FACIES, lab)); out["mixed"] = float((T.FACIES_PURITY < 0.7).mean())
pl, r2l, mael, pw = gc.cross_validate(T, MF, P, "lowo"); ps, r2s, *_ = gc.cross_validate(T, MF, P, "shuffled")
out["model"] = dict(lowo=r2l, shuffled=r2s, mae=mael, per_well=pw.round(2).to_dict("index"), stages=len(T))
y = T.TARGET_BOPD_PER_100FT
out["indices"] = {c: dict(all=float(spearmanr(T[c], y, nan_policy="omit").correlation),
                          within=float(T.groupby("WELL").apply(lambda d: spearmanr(d[c], d.TARGET_BOPD_PER_100FT).correlation, include_groups=False).median()))
                  for c in ["CQS", "CLUSTER_FI", "FI_MEAN", "BI_EL_MEAN", "BI_MIN_MEAN", "TOC_MEAN", "UCS_MEAN"]}
out["ml_rho_within"] = float(T.assign(P=pl).groupby("WELL").apply(lambda d: spearmanr(d.P, d.TARGET_BOPD_PER_100FT).correlation, include_groups=False).median())
m = gc.make_model(**P).fit(T[MF], y); sv, _ = gc.shap_values(m, T[MF])
imp = pd.Series(np.abs(sv).mean(0), index=MF).sort_values(ascending=False); out["shap_top"] = imp.head(10).round(2).to_dict()
pl_nf, r2_nf, *_ = gc.cross_validate(T, [f for f in MF if f != "FLUID_BBL_FT"], P, "lowo"); out["lowo_without_fluid"] = r2_nf
mf10 = [f for f in MF if f not in ("FI_MEAN", "CLUSTER_FI", "CQS", "UCS_MEAN")]
out["lowo_without_new_indices"] = gc.cross_validate(T, mf10, P, "lowo")[1]

# ---- Session 4 classifier: stage in its well's bottom 25 %
T["UNDER"] = (T.groupby("WELL").TARGET_BOPD_PER_100FT.transform(lambda s: s < s.quantile(0.25))).astype(int)
def clf(): return xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=3, random_state=0, n_jobs=2)
def oof(split):
    p = np.zeros(len(T))
    for tr, te in split:
        p[te] = clf().fit(T[MF].iloc[tr], T.UNDER.iloc[tr]).predict_proba(T[MF].iloc[te])[:, 1]
    return p
p_lowo = oof(GroupKFold(6).split(T, T.UNDER, T.WELL)); p_sh = oof(KFold(5, shuffle=True, random_state=0).split(T))
yb = T.UNDER.to_numpy()
def cm(th):
    f = p_lowo >= th
    return dict(tp=int((f & (yb == 1)).sum()), fn=int((~f & (yb == 1)).sum()), fp=int((f & (yb == 0)).sum()), tn=int((~f & (yb == 0)).sum()))
c5 = cm(0.5)
out["s4"] = dict(n=len(T), pos=int(yb.sum()), cm=c5, recall=c5["tp"] / (c5["tp"] + c5["fn"]), precision=c5["tp"] / (c5["tp"] + c5["fp"]),
                 specificity=c5["tn"] / (c5["tn"] + c5["fp"]), accuracy=(c5["tp"] + c5["tn"]) / len(T),
                 auc_lowo=roc_auc_score(yb, p_lowo), auc_sh=roc_auc_score(yb, p_sh), ap_lowo=average_precision_score(yb, p_lowo), ap_sh=average_precision_score(yb, p_sh),
                 auc_per_well=[float(roc_auc_score(yb[T.WELL == w], p_lowo[T.WELL == w])) for w in T.WELL.unique()],
                 thresholds={str(th): dict(**cm(th), flags=int((p_lowo >= th).sum())) for th in (0.3, 0.5, 0.6)},
                 shuffled_r2=r2s, lowo_r2=r2l)
for th, d in out["s4"]["thresholds"].items():
    d["precision"] = d["tp"] / max(d["flags"], 1); d["cost1"] = d["fn"] * 1 + d["fp"]; d["cost5"] = d["fn"] * 5 + d["fp"]
np.save(Path(__file__).with_name("s4_scores.npy"), np.vstack([yb, p_lowo, p_sh]))   # beside this script, on any OS
json.dump(out, open(Path(__file__).with_name("reference_numbers.json"), "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float)[:6000])
