"""
Synthetic deepwater field: an upper turbidite sandstone above a lower carbonate.

Everything here is generated. The field is built so that the teaching points of
the Day 1 case study are present in the data, with known "truth" kept aside:

* Arkosic lobe sands carry K-feldspar, so gamma ray overstates clay.
* Illite damages permeability far more than kaolinite; chlorite coats preserve it.
* The carbonate has a tar mat above its oil-water contact and scattered bitumen
  beds. Bitumen cuts permeability, raises Rock-Eval S2, and reads as pay on
  resistivity.
* Core depths are driller's depths, offset from log depth by a per-well shift.
* Core plugs are measured at ambient stress; a few are rubble or fractured.
* Two wells were drilled with oil-based mud, which contaminates Rock-Eval S1.
* Well tests give kh over each oil leg, an independent flow-scale check.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

STEP_FT = 0.5

MINERALS = ["QTZ", "KFS", "CALC", "DOL", "ILL", "KAO", "CHL"]
CLAYS = ["ILL", "KAO", "CHL"]
GRAIN_RHO = {"QTZ": 2.65, "KFS": 2.56, "CALC": 2.71, "DOL": 2.87,
             "ILL": 2.75, "KAO": 2.61, "CHL": 2.95}
GRAIN_DT = {"QTZ": 55.5, "KFS": 60.0, "CALC": 47.5, "DOL": 43.5,
            "ILL": 85.0, "KAO": 80.0, "CHL": 75.0}
GRAIN_PEF = {"QTZ": 1.81, "KFS": 2.86, "CALC": 5.08, "DOL": 3.14,
             "ILL": 3.45, "KAO": 1.83, "CHL": 6.30}

# facies: probability, porosity, FZI (micron), vug fraction, volume composition of solids
SS_FACIES = {
    "Channel sand":  dict(p=0.30, phi=0.26, fzi=5.5, vug=0.0,
                          comp=[0.78, 0.10, 0.02, 0.01, 0.02, 0.04, 0.03]),
    "Lobe sand":     dict(p=0.30, phi=0.22, fzi=3.0, vug=0.0,
                          comp=[0.60, 0.24, 0.03, 0.01, 0.04, 0.05, 0.03]),
    "Heterolithic":  dict(p=0.25, phi=0.15, fzi=1.0, vug=0.0,
                          comp=[0.50, 0.12, 0.03, 0.01, 0.18, 0.10, 0.06]),
    "Mudstone":      dict(p=0.15, phi=0.08, fzi=0.25, vug=0.0,
                          comp=[0.30, 0.08, 0.04, 0.01, 0.35, 0.15, 0.07]),
}
CB_FACIES = {
    "Grainstone":        dict(p=0.30, phi=0.20, fzi=3.5, vug=0.05,
                              comp=[0.02, 0.00, 0.92, 0.03, 0.02, 0.005, 0.005]),
    "Packstone":         dict(p=0.30, phi=0.14, fzi=1.2, vug=0.02,
                              comp=[0.03, 0.00, 0.86, 0.04, 0.05, 0.01, 0.01]),
    "Wackestone":        dict(p=0.20, phi=0.10, fzi=0.35, vug=0.0,
                              comp=[0.04, 0.00, 0.80, 0.04, 0.09, 0.02, 0.01]),
    "Vuggy dolostone":   dict(p=0.20, phi=0.17, fzi=6.0, vug=0.35,
                              comp=[0.02, 0.00, 0.15, 0.80, 0.02, 0.005, 0.005]),
}
INTERVALS = {"Sandstone": SS_FACIES, "Carbonate": CB_FACIES}


@dataclass
class Field:
    logs: pd.DataFrame
    plugs: pd.DataFrame
    core_gamma: pd.DataFrame
    tests: pd.DataFrame
    wells: pd.DataFrame
    truth: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _ar1(rng, n, rho, sigma):
    x = np.empty(n)
    x[0] = rng.normal(0, sigma)
    innov = sigma * np.sqrt(1 - rho ** 2)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + rng.normal(0, innov)
    return x


def _beds(rng, n, facies: dict, mean_ft: float):
    names = list(facies)
    probs = np.array([facies[f]["p"] for f in names])
    out: list[str] = []
    while len(out) < n:
        thick = max(2, int(rng.exponential(mean_ft) / STEP_FT))
        out.extend([rng.choice(names, p=probs)] * thick)
    return np.array(out[:n])


def _smooth(x, sigma_samples):
    """Gaussian vertical response of a logging tool."""
    if sigma_samples <= 0:
        return x
    half = int(4 * sigma_samples)
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma_samples) ** 2)
    k /= k.sum()
    pad = np.pad(x, half, mode="edge")
    return np.convolve(pad, k, mode="valid")


def vol_to_weight(vol: np.ndarray) -> np.ndarray:
    rho = np.array([GRAIN_RHO[m] for m in MINERALS])
    w = vol * rho
    return w / w.sum(axis=1, keepdims=True)


def weight_to_vol(wt: np.ndarray) -> np.ndarray:
    rho = np.array([GRAIN_RHO[m] for m in MINERALS])
    v = wt / rho
    return v / v.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# one interval in one well
# ---------------------------------------------------------------------------

def _interval(rng, well, interval, top, thick_ft, owc, wp):
    fac = INTERVALS[interval]
    n = int(thick_ft / STEP_FT)
    depth = top + STEP_FT * np.arange(n)
    beds = _beds(rng, n, fac, 7.0 if interval == "Sandstone" else 6.0)

    base_comp = np.array([fac[b]["comp"] for b in beds], float)
    noise = np.column_stack([_ar1(rng, n, 0.9, 0.25) for _ in MINERALS])
    comp = base_comp * np.exp(noise) + 1e-4
    comp /= comp.sum(axis=1, keepdims=True)
    # diagenetic illitisation: kaolinite converts to fibrous illite. Total clay (and
    # so GR) barely changes, but pore throats are bridged and permeability falls.
    illite_frac = np.clip(wp["illitisation"] + _ar1(rng, n, 0.96, 0.30), 0, 0.95)
    if interval == "Carbonate":
        illite_frac *= 0.3
    ki, kk = MINERALS.index("ILL"), MINERALS.index("KAO")
    moved = illite_frac * comp[:, kk]
    comp[:, ki] += moved
    comp[:, kk] -= moved
    d = dict(zip(MINERALS, comp.T))
    b = dict(zip(MINERALS, base_comp.T))

    phi = np.array([fac[x]["phi"] for x in beds]) + _ar1(rng, n, 0.93, 0.02)
    phi += 0.35 * (d["CHL"] - b["CHL"])                       # chlorite coats preserve porosity
    phi = np.clip(phi, 0.02, 0.33)
    vug = np.array([fac[x]["vug"] for x in beds])

    fzi = np.array([fac[x]["fzi"] for x in beds]) * wp["diag"]
    fzi *= np.exp(-2.6 * illite_frac * np.clip(d["ILL"] + d["KAO"], 0, 0.5) / 0.1
                  + 4.0 * (d["CHL"] - b["CHL"]) + 3.0 * (d["DOL"] - b["DOL"]))
    fzi *= np.exp(_ar1(rng, n, 0.95, 0.35))

    # bitumen: tar mat above the carbonate OWC plus scattered beds
    bitumen = np.zeros(n, bool)
    if interval == "Carbonate":
        if wp["tar_mat"]:
            bitumen |= (depth > owc - wp["tar_thick"]) & (depth <= owc)
        patches = _beds(rng, n, {"b": dict(p=0.1), "n": dict(p=0.9)}, 4.0) == "b"
        bitumen |= patches & (depth < owc)
    fzi = np.where(bitumen, fzi * rng.uniform(0.06, 0.15, n), fzi)
    phi = np.where(bitumen, phi * 0.93, phi)

    k = np.clip(1014.24 * fzi ** 2 * phi ** 3 / (1 - phi) ** 2, 0.002, 15000)

    # saturation: capillary transition above OWC
    h = np.clip(owc - depth, 0, None)
    swirr = np.clip(0.95 - 0.18 * np.log10(k * 10), 0.08, 0.95)
    tz = np.clip(8 + 40 / np.sqrt(np.maximum(k, 0.01)), 8, 60)
    sw = np.where(depth > owc, 1.0, swirr + (1 - swirr) * np.exp(-h / tz))
    sw = np.clip(sw + _ar1(rng, n, 0.8, 0.02), 0.05, 1.0)
    so = 1 - sw

    clay_solid = d["ILL"] + d["KAO"] + d["CHL"]
    vclay_bulk = clay_solid * (1 - phi)

    # Rock-Eval (pyrolysis on crushed rock, mg HC / g rock, TOC wt %)
    toc = 0.08 + 1.4 * clay_solid + rng.gamma(2, 0.03, n)
    toc = np.where(bitumen, toc + rng.uniform(1.0, 3.0, n), toc)
    s2 = 1.5 * toc * np.exp(rng.normal(0, 0.15, n))
    s2 = np.where(bitumen, rng.uniform(6, 16, n), s2)
    s1 = 40 * phi * so * wp["retention"] * np.exp(rng.normal(0, 0.15, n))
    s1 = np.where(bitumen, s1 * 0.4, s1)
    s1 = s1 + (wp["obm_s1"] if wp["mud"] == "OBM" else 0.0)
    s3 = 0.15 + 0.25 * toc * np.exp(rng.normal(0, 0.2, n))
    tmax = np.where(bitumen, rng.normal(424, 3, n), rng.normal(441, 3, n))

    # true (unsmoothed) log responses
    rho_ma = comp @ np.array([GRAIN_RHO[m] for m in MINERALS])
    dt_ma = comp @ np.array([GRAIN_DT[m] for m in MINERALS])
    pef_ma = comp @ np.array([GRAIN_PEF[m] for m in MINERALS])
    rho_fl = 0.98 - 0.12 * so * 0.3                       # mostly filtrate in the flushed zone
    m_exp = np.where(interval == "Sandstone", 2.0, 2.0 + 0.9 * vug)
    rw = wp["rw"]
    rt = rw / (phi ** m_exp * np.maximum(sw, 0.05) ** 2.0)
    rt = rt / (1 + 2.5 * vclay_bulk * rt / 8.0)           # clay conduction (crude)

    gr_true = 12 + 150 * clay_solid + 115 * d["KFS"]
    rhob_true = phi * rho_fl + (1 - phi) * rho_ma
    nphi_true = phi + 0.35 * vclay_bulk - 0.035 * d["QTZ"] + 0.02 * d["DOL"]
    dt_true = phi * (1 - vug) * 189 + (1 - phi * (1 - vug)) * dt_ma
    pef_true = pef_ma * (1 - phi) + 0.1

    logs = pd.DataFrame({
        "WELL": well, "INTERVAL": interval, "DEPTH": depth, "FACIES": beds,
        "GR": _smooth(gr_true, 1.5) + wp["gr_shift"] + _ar1(rng, n, 0.7, 3.0),
        "RHOB": _smooth(rhob_true, 1.5) + wp["rhob_shift"] + _ar1(rng, n, 0.6, 0.015),
        "NPHI": _smooth(nphi_true, 1.5) + _ar1(rng, n, 0.6, 0.01),
        "DT": _smooth(dt_true, 2.0) + _ar1(rng, n, 0.6, 1.2),
        "PEF": _smooth(pef_true, 1.0) + _ar1(rng, n, 0.6, 0.08),
        "RT": np.exp(_smooth(np.log(rt), 3.0) + _ar1(rng, n, 0.7, 0.06)),
        # truth columns (hidden from the models)
        "T_PHI": phi, "T_K": k, "T_SW": sw, "T_VCLAY": vclay_bulk, "T_BITUMEN": bitumen,
        "T_GR_CORE": gr_true,
    })
    for m in MINERALS:
        logs[f"T_V_{m}"] = d[m]
    for name, arr in [("TOC", toc), ("S1", s1), ("S2", s2), ("S3", s3), ("TMAX", tmax)]:
        logs[f"T_{name}"] = arr
    logs["T_OWC"] = owc
    return logs


# ---------------------------------------------------------------------------
# field
# ---------------------------------------------------------------------------

def build_field(seed: int = 7) -> Field:
    rng = np.random.default_rng(seed)
    names = [f"A-{i:02d}" for i in range(1, 9)]
    cored = {"A-01", "A-02", "A-03", "A-05", "A-07"}
    obm = {"A-03", "A-06"}

    all_logs, wells_meta = [], []
    for i, w in enumerate(names):
        wp = dict(
            diag=float(np.exp(rng.normal(0, 0.15))),
            illitisation=float(rng.uniform(0.25, 0.40)),
            rw=float(0.045 * np.exp(rng.normal(0, 0.2))),
            gr_shift=float(rng.normal(0, 6)),
            rhob_shift=float(rng.normal(0, 0.015)),
            retention=float(rng.uniform(0.15, 0.35)),
            mud="OBM" if w in obm else "WBM",
            obm_s1=float(rng.uniform(1.5, 3.0)),
            tar_mat=bool(rng.random() < 0.75),
            tar_thick=float(rng.uniform(10, 22)),
            shift=float(np.round(rng.uniform(1.0, 5.0) * 2) / 2 * rng.choice([-1, 1])),
        )
        ss_top = 9800 + rng.uniform(-60, 60)
        ss_top = np.round(ss_top * 2) / 2
        cb_top = ss_top + 240
        ss_owc = ss_top + 180 + rng.uniform(-20, 20)
        cb_owc = cb_top + 140 + rng.uniform(-20, 20)
        all_logs.append(_interval(rng, w, "Sandstone", ss_top, 240, ss_owc, wp))
        all_logs.append(_interval(rng, w, "Carbonate", cb_top, 200, cb_owc, wp))
        wells_meta.append(dict(WELL=w, CORED=w in cored, MUD=wp["mud"],
                               SS_TOP=ss_top, CB_TOP=cb_top, SS_OWC=ss_owc, CB_OWC=cb_owc,
                               TAR_MAT=wp["tar_mat"], _SHIFT=wp["shift"]))
    logs = pd.concat(all_logs, ignore_index=True)
    wells = pd.DataFrame(wells_meta)

    # ---- core: plugs every 1 ft, core gamma every 0.5 ft, driller's depth -------
    plugs, cgam = [], []
    for _, wm in wells[wells.CORED].iterrows():
        wl = logs[logs.WELL == wm.WELL]
        for interval, top, (a, b_) in [("Sandstone", wm.SS_TOP, (40, 160)),
                                       ("Carbonate", wm.CB_TOP, (30, 150))]:
            seg = wl[(wl.INTERVAL == interval) & (wl.DEPTH >= top + a) & (wl.DEPTH < top + b_)]
            cgam.append(pd.DataFrame({
                "WELL": wm.WELL, "INTERVAL": interval,
                "CORE_DEPTH": seg.DEPTH + wm._SHIFT,
                "CORE_GR": seg.T_GR_CORE + rng.normal(0, 4, len(seg)),
            }))
            p = seg.iloc[::2].copy()
            npl = len(p)
            ob_k = 0.72 if interval == "Sandstone" else 0.85
            rubble = rng.random(npl) < 0.04
            k_amb = p.T_K / ob_k * np.exp(rng.normal(0, 0.12, npl))
            k_amb = np.where(rubble, k_amb * rng.uniform(5, 30, npl), k_amb)
            sw_ds = p.T_SW + np.where(wm.MUD == "WBM", 0.10, 0.02) * (1 - p.T_SW)
            pl = pd.DataFrame({
                "WELL": wm.WELL, "INTERVAL": interval,
                "CORE_DEPTH": p.DEPTH.values + wm._SHIFT,
                "PHI_AMB": p.T_PHI.values * 1.035 + rng.normal(0, 0.005, npl),
                "K_AMB": k_amb.values if hasattr(k_amb, "values") else k_amb,
                "SW_DS": np.clip(sw_ds.values + rng.normal(0, 0.03, npl), 0, 1),
                "PLUG_NOTE": np.where(rubble, "rubble / induced fracture", "intact"),
                "_TRUE_DEPTH": p.DEPTH.values,
            })
            # geochemistry on every second plug (plug end trims): XRD weight % and Rock-Eval
            geo = np.zeros(npl, bool)
            geo[::2] = True
            vol = p[[f"T_V_{m}" for m in MINERALS]].to_numpy()
            wt = vol_to_weight(vol) * np.exp(rng.normal(0, 0.06, vol.shape))
            wt = wt / wt.sum(axis=1, keepdims=True) * 100
            for j, m in enumerate(MINERALS):
                pl[f"XRD_{m}"] = np.where(geo, wt[:, j], np.nan)
            for name in ["TOC", "S1", "S2", "S3", "TMAX"]:
                pl[f"RE_{name}"] = np.where(geo, p[f"T_{name}"].values, np.nan)
            api = np.where((p.T_SW.values < 0.95) & ~p.T_BITUMEN.values,
                           rng.normal(32, 2, npl), np.where(p.T_BITUMEN.values,
                                                            rng.normal(12, 2, npl), np.nan))
            pl["API"] = np.where(geo, api, np.nan)
            pl["FLUID_CLASS"] = np.select(
                [p.T_BITUMEN.values, p.T_SW.values >= 0.95], ["Bitumen / tar", "Water"], "Light oil")
            pl["_T_K"] = p.T_K.values
            plugs.append(pl)
    plugs = pd.concat(plugs, ignore_index=True)
    core_gamma = pd.concat(cgam, ignore_index=True)

    # ---- well tests: kh over each oil leg (independent of core) ------------------
    tests = []
    for _, wm in wells.iterrows():
        for interval, owc in [("Sandstone", wm.SS_OWC), ("Carbonate", wm.CB_OWC)]:
            seg = logs[(logs.WELL == wm.WELL) & (logs.INTERVAL == interval) & (logs.DEPTH < owc)]
            kh = float((seg.T_K * STEP_FT).sum() * np.exp(rng.normal(0, 0.2)))
            tests.append(dict(WELL=wm.WELL, INTERVAL=interval, TEST_TOP=seg.DEPTH.min(),
                              TEST_BASE=seg.DEPTH.max(), KH_TEST_MDFT=kh))
    tests = pd.DataFrame(tests)

    truth = {"shifts": wells.set_index("WELL")._SHIFT.to_dict()}
    return Field(logs=logs, plugs=plugs, core_gamma=core_gamma, tests=tests,
                 wells=wells, truth=truth)


LOG_CURVES = ["GR", "RHOB", "NPHI", "DT", "PEF", "RT"]
