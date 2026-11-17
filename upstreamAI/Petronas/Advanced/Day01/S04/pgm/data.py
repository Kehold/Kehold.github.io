"""
Synthetic Midland Basin case: 12 horizontal wells in Wolfcamp A and B, three pads, 50 stages each.

Along every lateral (2 ft step): mineralogy, TOC, dynamic Young's modulus and Poisson's ratio,
Rickman brittleness, pore pressure, minimum horizontal stress (uniaxial strain plus tectonic strain),
natural fracture intensity (higher near a fault that crosses pad 3), GR and drilling MSE.

Stage outcome: each stage's true contribution is a known function of the geomechanics; a chemical tracer
observes it with multiplicative noise. The label "effective" is a tracer share of at least half the
well's mean stage share. Because the generating function is known, the app can act as a completion
simulator. All numbers are synthetic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STEP = 2.0
LATERAL = 10000.0
STAGE_FT = 200.0
TRACER_SIGMA = 1.2
FACIES = {  # quartz, carbonate, clay, TOC wt%
    "Siliceous mudstone":    dict(p=0.34, q=0.46, c=0.14, cl=0.30, toc=4.2),
    "Calcareous mudstone":   dict(p=0.26, q=0.30, c=0.36, cl=0.26, toc=2.6),
    "Carbonate debris flow": dict(p=0.16, q=0.14, c=0.70, cl=0.12, toc=1.2),
    "Argillaceous mudstone": dict(p=0.24, q=0.24, c=0.10, cl=0.52, toc=3.0),
}
BENCH = {"Wolfcamp A": dict(tvd=8600, pp_grad=0.62, clay_shift=0.0),
         "Wolfcamp B": dict(tvd=9150, pp_grad=0.66, clay_shift=0.05)}
FAULT = dict(x0=-10600.0, slope=1.2)         # fault line x = x0 + slope * y (ft): crosses Pad 3 (W-09 to W-12) only


def _ar1(rng, n, rho, s):
    x = np.empty(n); x[0] = rng.normal(0, s); e = s * np.sqrt(1 - rho ** 2)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + rng.normal(0, e)
    return x


def _beds(rng, n, mean_ft):
    names = list(FACIES); p = np.array([FACIES[f]["p"] for f in names]); out = []
    while len(out) < n:
        out += [rng.choice(names, p=p)] * max(3, int(rng.exponential(mean_ft) / STEP))
    return np.array(out[:n])


def stress_params():
    return dict(sv_grad=1.05, strain=5e-5, e_lim=(15.0, 50.0), nu_lim=(0.18, 0.32))


def shmin(nu, e_gpa, tvd, pp_grad, sv_grad=1.05, strain=5e-5):
    sv, pp = sv_grad * tvd, pp_grad * tvd
    return nu / (1 - nu) * (sv - pp) + pp + e_gpa * 145038.0 * strain


def brittleness(e, nu, e_lim=(15.0, 50.0), nu_lim=(0.18, 0.32)):
    be = (e - e_lim[0]) / (e_lim[1] - e_lim[0]); bn = (nu - nu_lim[1]) / (nu_lim[0] - nu_lim[1])
    return np.clip(50 * (be + bn), 0, 100)


def lateral_logs(rng, well, bench, y, clusters, proppant):
    n = int(LATERAL / STEP); md = np.arange(n) * STEP
    b = BENCH[bench]
    beds = _beds(rng, n, 140.0)
    g = lambda k: np.array([FACIES[f][k] for f in beds])
    dev = _ar1(rng, n, 0.998, 9.0)                                    # landing: TVD offset from target (ft)
    ooz = np.abs(dev) > 18
    q = np.clip(g("q") * np.exp(_ar1(rng, n, 0.95, 0.12)), 0.02, 0.9)
    c = np.clip(g("c") * np.exp(_ar1(rng, n, 0.95, 0.15)), 0.02, 0.9)
    cl = np.clip((g("cl") + b["clay_shift"] + 0.12 * ooz) * np.exp(_ar1(rng, n, 0.95, 0.12)), 0.03, 0.8)
    s = q + c + cl; q, c, cl = q / s, c / s, cl / s
    toc = np.clip(g("toc") * np.exp(_ar1(rng, n, 0.95, 0.15)) - 1.0 * ooz, 0.3, 8)
    e = np.clip(10 + 40 * q + 46 * c + 8 * cl - 2.2 * toc + _ar1(rng, n, 0.9, 1.5), 8, 60)
    nu = np.clip(0.17 + 0.22 * cl - 0.04 * q + 0.008 * toc + _ar1(rng, n, 0.9, 0.006), 0.12, 0.40)
    tvd = b["tvd"] + dev
    sh = shmin(nu, e, tvd, b["pp_grad"])
    bi = brittleness(e, nu)
    x = md
    dist = np.abs(x - (FAULT["x0"] + FAULT["slope"] * y)) / np.sqrt(1 + FAULT["slope"] ** 2)
    nfi = np.clip(0.08 + 0.55 * (bi / 100) ** 2 + 1.1 * np.exp(-dist / 350) + _ar1(rng, n, 0.9, 0.05), 0, 3)
    gr = 35 + 170 * cl + 6 * toc + _ar1(rng, n, 0.8, 5)
    mse = np.exp(np.log(6 + 0.95 * e) + _ar1(rng, n, 0.85, 0.20) + rng.normal(0, 0.12, n))   # ksi, drilling mechanical specific energy
    return pd.DataFrame(dict(WELL=well, BENCH=bench, MD=md, X=x, Y=y, FACIES=beds, TVD_DEV=dev, OOZ=ooz,
                             QUARTZ=q, CARB=c, CLAY=cl, TOC=toc, YM_GPA=e, PR=nu, BI=bi, SHMIN_PSI=sh,
                             PP_PSI=b["pp_grad"] * tvd, NFI=nfi, GR=gr, MSE_KSI=mse, FAULT_DIST=dist))


def truth_logit(st: pd.DataFrame) -> np.ndarray:
    """The generating model for stage effectiveness (the 'simulator'). Hidden from the ML models."""
    return (0.55 + 0.045 * (st.BI - 50) - 0.0055 * (st.STRESS_RANGE_PSI - 250)
            + 1.6 * (st.NFI - 0.4) - 2.2 * np.exp(-st.FAULT_DIST_MIN / 220)
            - 3.5 * (st.CLAY - 0.30) + 0.22 * (st.TOC - 3.0) - 1.6 * st.OOZ_FRAC
            + 0.0007 * (st.PROPPANT_LB_FT - 2200) - 0.25 * (st.CLUSTERS - 5) * (st.STRESS_RANGE_PSI / 300)
            - 0.0012 * (st.SHMIN_PSI - st.SHMIN_WELL_MEAN))


def p_effective(z, cut, sigma=None):
    """Probability that a stage with true log-contribution z is labelled effective, given tracer noise."""
    from scipy.stats import norm
    return norm.cdf((np.asarray(z) - np.asarray(cut)) / (sigma or TRACER_SIGMA))


def stage_table(logs: pd.DataFrame, boundaries=None, design=None) -> pd.DataFrame:
    """Aggregate lateral samples into stages. boundaries: sorted MD edges (default: every 200 ft)."""
    rows = []
    for w, g in logs.groupby("WELL", sort=False):
        edges = boundaries[w] if boundaries else np.arange(0, LATERAL + 1, STAGE_FT)
        d = design[w] if design else dict(CLUSTERS=5, PROPPANT_LB_FT=2200, FLUID_BBL_FT=40)
        mean_sh = g.SHMIN_PSI.mean()
        for i in range(len(edges) - 1):
            z = g[(g.MD >= edges[i]) & (g.MD < edges[i + 1])]
            if z.empty:
                continue
            rows.append(dict(WELL=w, BENCH=z.BENCH.iloc[0], STAGE=i + 1, MD_TOE=edges[i], MD_HEEL=edges[i + 1],
                             LENGTH_FT=edges[i + 1] - edges[i],
                             YM_GPA=z.YM_GPA.mean(), PR=z.PR.mean(), BI=z.BI.mean(), SHMIN_PSI=z.SHMIN_PSI.mean(),
                             STRESS_RANGE_PSI=z.SHMIN_PSI.quantile(0.9) - z.SHMIN_PSI.quantile(0.1),
                             NFI=z.NFI.mean(), CLAY=z.CLAY.mean(), QUARTZ=z.QUARTZ.mean(), CARB=z.CARB.mean(),
                             TOC=z.TOC.mean(), GR=z.GR.mean(), MSE_KSI=z.MSE_KSI.mean(),
                             MSE_CV=z.MSE_KSI.std() / z.MSE_KSI.mean(), TVD_DEV_ABS=z.TVD_DEV.abs().mean(),
                             OOZ_FRAC=z.OOZ.mean(), FAULT_DIST_MIN=z.FAULT_DIST.min(),
                             SHMIN_WELL_MEAN=mean_sh, **d))
    return pd.DataFrame(rows)


def build(seed=7):
    rng = np.random.default_rng(seed)
    wells, logs, design = [], [], {}
    pads = {"Pad 1": ["Wolfcamp A"] * 4, "Pad 2": ["Wolfcamp B"] * 4, "Pad 3": ["Wolfcamp A", "Wolfcamp A", "Wolfcamp B", "Wolfcamp B"]}
    k = 0
    for p, (pad, benches) in enumerate(pads.items()):
        for j, bench in enumerate(benches):
            k += 1
            w = f"W-{k:02d}"
            y = p * 6000 + j * 660
            clusters = int(rng.choice([4, 5, 6]))
            prop = float(rng.choice([1800, 2200, 2600]))
            design[w] = dict(CLUSTERS=clusters, PROPPANT_LB_FT=prop, FLUID_BBL_FT=round(prop / 55))
            logs.append(lateral_logs(rng, w, bench, y, clusters, prop))
            wells.append(dict(WELL=w, PAD=pad, BENCH=bench, Y_FT=y, CLUSTERS=clusters, PROPPANT_LB_FT=prop,
                              FLUID_BBL_FT=round(prop / 55)))
    logs = pd.concat(logs, ignore_index=True)
    wells = pd.DataFrame(wells)
    stages = stage_table(logs, design=design)
    stages = stages.merge(wells[["WELL", "PAD"]], on="WELL")
    # Tracer measurement: each stage's true contribution (from the geomechanics) is observed through a
    # chemical tracer with multiplicative error (TRACER_SIGMA in natural-log units, about a factor of 3).
    z = truth_logit(stages)
    raw = z + rng.normal(0, TRACER_SIGMA, len(stages))
    well_mean = np.exp(raw).groupby(stages.WELL).transform("mean")
    stages["TRACER_SHARE"] = np.exp(raw) / well_mean                     # 1.0 = the well's mean stage share
    stages["EFFECTIVE"] = (stages.TRACER_SHARE >= 0.5).astype(int)       # the label: at least half the mean
    cut = np.log(0.5) + np.log(well_mean)
    stages["LABEL_CUT"] = cut
    stages["P_TRUE"] = p_effective(z, cut)                               # probability the tracer calls it effective
    return logs, stages, wells, design


# ------------------------------------------------------------------ pilot well: layered stress profile
LAYERS = [  # name, top TVD, base TVD, E GPa, PR, pp grad
    ("Lower Spraberry", 7900, 8250, 30, 0.24, 0.52),
    ("Dean", 8250, 8420, 40, 0.21, 0.55),
    ("Wolfcamp A upper", 8420, 8560, 20, 0.32, 0.62),
    ("Wolfcamp A target", 8560, 8680, 34, 0.23, 0.62),
    ("Wolfcamp A lower", 8680, 8900, 24, 0.29, 0.63),
    ("Wolfcamp B upper", 8900, 9080, 20, 0.32, 0.66),
    ("Wolfcamp B target", 9080, 9220, 30, 0.25, 0.66),
    ("Wolfcamp B lower", 9220, 9450, 21, 0.31, 0.67),
    ("Wolfcamp C", 9450, 9700, 28, 0.27, 0.68),
]


def pilot_profile():
    rows = []
    for name, top, base, e, nu, ppg in LAYERS:
        mid = (top + base) / 2
        rows.append(dict(LAYER=name, TOP=top, BASE=base, YM_GPA=e, PR=nu, PP_GRAD=ppg,
                         SHMIN_PSI=shmin(nu, e, mid, ppg), SHMIN_GRAD=shmin(nu, e, mid, ppg) / mid))
    return pd.DataFrame(rows)


def fracture_height(profile: pd.DataFrame, target: str, net_pressure: float):
    """Equilibrium-height screening: the fracture crosses a layer while its closure stress stays below
    the target-layer stress plus net pressure. Returns (top TVD, base TVD, layers reached)."""
    idx = profile.index[profile.LAYER == target][0]
    s0 = profile.loc[idx, "SHMIN_PSI"] + net_pressure
    up, down = idx, idx
    while up - 1 >= 0 and profile.loc[up - 1, "SHMIN_PSI"] < s0:
        up -= 1
    while down + 1 < len(profile) and profile.loc[down + 1, "SHMIN_PSI"] < s0:
        down += 1
    return profile.loc[up, "TOP"], profile.loc[down, "BASE"], profile.loc[up:down, "LAYER"].tolist()


def engineered_boundaries(g: pd.DataFrame, min_ft=150, max_ft=250, win=20):
    """Place stage boundaries at the largest closure-stress contrast inside each allowed window."""
    sh = g.SHMIN_PSI.to_numpy(); md = g.MD.to_numpy()
    k = int(win / STEP)
    edges, start = [0.0], 0.0
    while start + max_ft < LATERAL:
        lo, hi = int((start + min_ft) / STEP), int((start + max_ft) / STEP)
        best, bi = -1, hi
        for i in range(lo, hi):
            if i - k < 0 or i + k >= len(sh):
                continue
            jump = abs(sh[i:i + k].mean() - sh[i - k:i].mean())
            if jump > best:
                best, bi = jump, i
        start = float(md[bi]); edges.append(start)
    edges.append(LATERAL)
    return np.array(edges)
