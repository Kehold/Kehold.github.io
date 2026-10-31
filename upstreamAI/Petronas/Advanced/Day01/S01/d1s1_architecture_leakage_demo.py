"""
Day 1 · Session 1 companion demo
Advanced AI for Conventional & Unconventional Reservoir Petrophysics (Level 3)
digitaloilEd.ai

What it shows (about 5 minutes live, or as a take-home notebook):
  1. A synthetic multi-well dataset (5 sandstone wells, 3 carbonate wells) with
     depth-correlated logs, XRD mineralogy, and core porosity/permeability
     generated from hydraulic flow units (FZI).
  2. Petrophysical feature engineering: centred log-ratio (clr) of XRD fractions,
     Jarvie-style mineralogical brittleness, RQI and FZI.
  3. Random Forest vs XGBoost vs MLP for log10(k) prediction.
  4. Validation leakage: shuffled K-fold vs leave-one-well-out (GroupKFold).

All data are synthetic. Numbers printed here illustrate behaviour and must not
be quoted as field results.

Run:  python d1s1_architecture_leakage_demo.py
Needs: numpy, pandas, scikit-learn, matplotlib; xgboost optional (falls back to
       scikit-learn HistGradientBoostingRegressor).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import KFold, GroupKFold, cross_val_predict
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:  # pragma: no cover
    HAS_XGB = False

RNG = np.random.default_rng(42)

NAVY, TEAL, AMBER = "#0A2E36", "#14808C", "#E8A317"

# ---------------------------------------------------------------------------
# 1. Synthetic multi-well dataset
# ---------------------------------------------------------------------------

# Rock types: mean porosity, FZI (micron), XRD composition [qtz, calc, dol, clay]
ROCK_TYPES = {
    "SS_clean":   dict(phi=0.24, fzi=4.0,  xrd=[0.85, 0.03, 0.02, 0.10]),
    "SS_shaly":   dict(phi=0.17, fzi=1.2,  xrd=[0.62, 0.04, 0.02, 0.32]),
    "SS_cemented":dict(phi=0.10, fzi=0.9,  xrd=[0.70, 0.22, 0.02, 0.06]),
    "Siltstone":  dict(phi=0.08, fzi=0.3,  xrd=[0.45, 0.05, 0.02, 0.48]),
    "LS_grain":   dict(phi=0.20, fzi=2.5,  xrd=[0.03, 0.90, 0.04, 0.03]),
    "LS_mud":     dict(phi=0.12, fzi=0.4,  xrd=[0.04, 0.84, 0.04, 0.08]),
    "DOL_vuggy":  dict(phi=0.16, fzi=6.0,  xrd=[0.03, 0.12, 0.82, 0.03]),
}
SS_TYPES = ["SS_clean", "SS_shaly", "SS_cemented", "Siltstone"]
CARB_TYPES = ["LS_grain", "LS_mud", "DOL_vuggy"]
MATRIX_RHO = np.array([2.65, 2.71, 2.87, 2.62])   # qtz, calc, dol, clay (g/cc)
MATRIX_DT = np.array([55.5, 47.5, 43.5, 90.0])    # us/ft


def ar1(n: int, rho: float, sigma: float) -> np.ndarray:
    """Depth-correlated noise: AR(1) with lag-1 correlation rho."""
    x = np.zeros(n)
    x[0] = RNG.normal(0, sigma)
    innov = sigma * np.sqrt(1 - rho**2)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + RNG.normal(0, innov)
    return x


def facies_sequence(n: int, types: list[str], mean_bed: int = 30) -> list[str]:
    """Beds of random thickness drawn along depth (a crude Markov stacking)."""
    seq: list[str] = []
    while len(seq) < n:
        bed = max(4, int(RNG.exponential(mean_bed)))
        seq.extend([RNG.choice(types)] * bed)
    return seq[:n]


def make_well(name: str, setting: str, n: int = 400, top_ft: float = 8000.0) -> pd.DataFrame:
    types = SS_TYPES if setting == "sandstone" else CARB_TYPES
    rt = facies_sequence(n, types)
    depth = top_ft + 0.5 * np.arange(n)

    phi = np.array([ROCK_TYPES[r]["phi"] for r in rt]) + ar1(n, 0.95, 0.025)
    phi = np.clip(phi, 0.02, 0.35)

    comp = np.array([ROCK_TYPES[r]["xrd"] for r in rt], dtype=float)
    comp = np.clip(comp + np.column_stack([ar1(n, 0.9, 0.03) for _ in range(4)]), 0.005, None)
    comp /= comp.sum(axis=1, keepdims=True)               # closure: fractions sum to 1
    qtz, calc, dol, clay = comp.T

    # Per-well diagenetic overprint and depth-correlated pore-throat variation
    # that the logs only partly see: this is what makes blind wells hard.
    well_fzi = np.exp(RNG.normal(0, 0.35))
    fzi = np.array([ROCK_TYPES[r]["fzi"] for r in rt]) * well_fzi * np.exp(ar1(n, 0.97, 0.45))
    k = 1014.24 * fzi**2 * phi**3 / (1 - phi) ** 2         # mD, from RQI/FZI definitions
    k = np.clip(k, 0.001, 20000)

    sw = np.clip(0.25 + 0.9 * clay + ar1(n, 0.9, 0.05), 0.1, 1.0)
    rw = 0.05 * np.exp(RNG.normal(0, 0.3))                  # formation water varies by well
    a, m, nsat = 1.0, (2.0 if setting == "sandstone" else 2.3), 2.0
    rt_ohm = a * rw / (phi**m * sw**nsat)

    rho_ma = comp @ MATRIX_RHO
    dt_ma = comp @ MATRIX_DT

    gr_shift, rhob_shift = RNG.normal(0, 8), RNG.normal(0, 0.02)   # tool / calibration offsets
    df = pd.DataFrame({
        "WELL": name, "SETTING": setting, "DEPTH": depth, "ROCK_TYPE": rt,
        "GR":   20 + gr_shift + 140 * clay + ar1(n, 0.8, 6),
        "RHOB": rhob_shift + phi * 1.0 + (1 - phi) * rho_ma + ar1(n, 0.7, 0.025),
        "NPHI": phi + 0.30 * clay + ar1(n, 0.7, 0.012),
        "DT":   phi * 189 + (1 - phi) * dt_ma + ar1(n, 0.7, 1.5),
        "RT":   rt_ohm * np.exp(ar1(n, 0.7, 0.08)),
        "XRD_QTZ": qtz, "XRD_CALC": calc, "XRD_DOL": dol, "XRD_CLAY": clay,
        "CORE_PHI": phi, "CORE_K": k,
    })
    return df


def build_dataset() -> pd.DataFrame:
    wells = [make_well(f"SS-{i}", "sandstone") for i in range(1, 6)]
    wells += [make_well(f"CB-{i}", "carbonate") for i in range(1, 4)]
    return pd.concat(wells, ignore_index=True)


# ---------------------------------------------------------------------------
# 2. Petrophysical feature engineering
# ---------------------------------------------------------------------------

def clr(frac: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Centred log-ratio transform for closed compositional data (Aitchison)."""
    logx = np.log(np.clip(frac, eps, None))
    return logx - logx.mean(axis=1, keepdims=True)


def brittleness_jarvie(qtz, carb, clay) -> np.ndarray:
    """Mineralogical brittleness index: quartz / (quartz + carbonate + clay)."""
    return qtz / (qtz + carb + clay)


def rqi_fzi(k_md, phi_e):
    """Reservoir Quality Index (micron) and Flow Zone Indicator (micron)."""
    rqi = 0.0314 * np.sqrt(k_md / phi_e)
    phi_z = phi_e / (1 - phi_e)
    return rqi, rqi / phi_z


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["LOG_RT"] = np.log10(out["RT"])
    xrd = out[["XRD_QTZ", "XRD_CALC", "XRD_DOL", "XRD_CLAY"]].to_numpy()
    for name, col in zip(["QTZ", "CALC", "DOL", "CLAY"], clr(xrd).T):
        out[f"CLR_{name}"] = col
    out["BI_JARVIE"] = brittleness_jarvie(out["XRD_QTZ"], out["XRD_CALC"] + out["XRD_DOL"], out["XRD_CLAY"])
    # RQI/FZI use core k: they are rock-typing tools (Exercise 1), NOT predictors of k.
    out["RQI"], out["FZI"] = rqi_fzi(out["CORE_K"], out["CORE_PHI"])
    out["LOG_K"] = np.log10(out["CORE_K"])
    return out


LOG_FEATURES = ["GR", "RHOB", "NPHI", "DT", "LOG_RT"]
XRD_FEATURES = ["CLR_QTZ", "CLR_CALC", "CLR_DOL", "CLR_CLAY", "BI_JARVIE"]
LEAKY = ["RQI", "FZI"]   # derived from the target: never use as features for k


# ---------------------------------------------------------------------------
# 3. Models
# ---------------------------------------------------------------------------

def models() -> dict:
    boost = (XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                          random_state=0, n_jobs=-1)
             if HAS_XGB else
             HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0))
    return {
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                               random_state=0, n_jobs=-1),
        "XGBoost" if HAS_XGB else "HistGB": boost,
        "MLP": make_pipeline(StandardScaler(),
                             MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3,
                                          max_iter=1500, early_stopping=True,
                                          random_state=0)),
    }


def evaluate(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    X, y, groups = df[features].to_numpy(), df["LOG_K"].to_numpy(), df["WELL"].to_numpy()
    schemes = {
        "Shuffled 5-fold": (KFold(5, shuffle=True, random_state=0), None),
        "Leave-one-well-out": (GroupKFold(n_splits=df["WELL"].nunique()), groups),
    }
    rows = []
    for mname, model in models().items():
        for sname, (cv, g) in schemes.items():
            pred = cross_val_predict(model, X, y, cv=cv, groups=g)
            rows.append({"model": mname, "cv": sname,
                         "R2": r2_score(y, pred),
                         "RMSE_logk": mean_squared_error(y, pred) ** 0.5})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Run
# ---------------------------------------------------------------------------

def main() -> None:
    df = engineer(build_dataset())
    print(f"Samples: {len(df)} · wells: {df['WELL'].nunique()} · "
          f"booster: {'XGBoost' if HAS_XGB else 'HistGradientBoosting'}\n")

    lag1 = df.groupby("WELL")["GR"].apply(lambda s: s.autocorr(1)).mean()
    print(f"Mean lag-1 autocorrelation of GR along depth: {lag1:.2f} "
          "(neighbouring samples are near-copies)\n")

    res_logs = evaluate(df, LOG_FEATURES).assign(features="Logs only")
    res_all = evaluate(df, LOG_FEATURES + XRD_FEATURES).assign(features="Logs + XRD")
    res = pd.concat([res_logs, res_all], ignore_index=True)
    pd.set_option("display.width", 120)
    print(res.pivot_table(index=["features", "model"], columns="cv",
                          values="R2").round(3).to_string())

    print("\nNote: XRD exists only where core was cut. 'Logs + XRD' shows the value of "
          "mineralogy as information; in uncored wells it must first be predicted from logs.")

    # Target-leakage illustration: add FZI (computed from core k) as a feature
    leaky = evaluate(df, LOG_FEATURES + LEAKY)
    best = leaky[leaky["cv"] == "Leave-one-well-out"]["R2"].max()
    print(f"\nWith FZI/RQI (built from core k) as inputs, blind-well R2 rises to {best:.3f}."
          " That is circular: the target is inside the features.")

    # Figure for the leakage slide
    sub = res[res["features"] == "Logs only"]
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
    names = sub["model"].unique()
    x = np.arange(len(names))
    for i, (scheme, colour) in enumerate([("Shuffled 5-fold", AMBER), ("Leave-one-well-out", TEAL)]):
        vals = [sub[(sub.model == n) & (sub.cv == scheme)]["R2"].iloc[0] for n in names]
        ax.bar(x + (i - 0.5) * 0.36, vals, 0.36, label=scheme, color=colour)
        for xi, v in zip(x + (i - 0.5) * 0.36, vals):
            ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=9, color=NAVY)
    ax.set_xticks(x, names)
    ax.set_ylabel("R² for log10(k)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Same models, same data: the split decides the score (synthetic)", color=NAVY)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig("d1s1_cv_leakage.png")
    df.to_csv("d1s1_synthetic_multiwell.csv", index=False)
    print("\nSaved d1s1_cv_leakage.png and d1s1_synthetic_multiwell.csv")


if __name__ == "__main__":
    main()
