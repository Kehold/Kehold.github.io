"""Agent 3 · Decline curve analysis.

Modified hyperbolic Arps (hyperbolic with a switch to terminal exponential at
8 % per year effective) fitted per stream on log rate with a robust loss.
Uncertainty comes from a residual bootstrap for producing wells, or from
resampling the analog set for a proposed location.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .. import config as C
from ..context import RunContext
from ..field import arps_modified, dmin_nominal_monthly
from .analogs import type_curve

AGENT = "Decline"
HORIZON = C.EUR_YEARS * 12


def _clean_mask(q: np.ndarray) -> np.ndarray:
    s = pd.Series(q)
    roll = s.rolling(5, center=True, min_periods=2).median()
    return ((s >= 0.6 * roll) & (s > 0)).values


def fit_arps(t, q, weights=None, x0=None):
    """Fit log q = log arps(t). Returns dict of parameters and diagnostics."""
    t, q = np.asarray(t, float), np.asarray(q, float)
    qmax = float(np.nanmax(q))
    w = np.ones_like(q) if weights is None else np.asarray(weights, float)

    def resid(p):
        qi, di, b = np.exp(p[0]), np.exp(p[1]), p[2]
        return w * (np.log(arps_modified(t, qi, di, b)) - np.log(q))

    lo = [np.log(0.3 * qmax), np.log(0.004), 0.3]
    hi = [np.log(3.0 * qmax), np.log(1.2), 1.8]
    starts = ([[np.log(qmax), np.log(di0), b0] for b0 in (0.8, 1.2) for di0 in (0.08, 0.2)]
              if x0 is None else [np.clip(x0, np.add(lo, 1e-6), np.subtract(hi, 1e-6))])
    best = None
    for s0 in starts:
        r = least_squares(resid, s0, bounds=(lo, hi), loss="soft_l1", f_scale=0.1)
        if best is None or r.cost < best.cost:
            best = r
    qi, di, b = float(np.exp(best.x[0])), float(np.exp(best.x[1])), float(best.x[2])
    pred = np.log(arps_modified(t, qi, di, b))
    ss_res = np.sum((np.log(q) - pred) ** 2)
    ss_tot = np.sum((np.log(q) - np.log(q).mean()) ** 2) or 1.0
    return dict(qi=qi, di=di, b=b, r2_log=float(1 - ss_res / ss_tot), x=best.x,
                resid=np.log(q) - pred, at_b_bound=bool(b > 1.78 or b < 0.32))


def describe(p: dict) -> dict:
    di_yr = p["di"] * 12
    b = p["b"]
    de = 1 - (1 + b * di_yr) ** (-1 / b)
    dlim = dmin_nominal_monthly()
    t_sw = (p["di"] / dlim - 1) / (b * p["di"]) if p["di"] > dlim else 0.0
    return dict(qi=round(p["qi"], 1), b=round(b, 3), Di_nominal_annual=round(di_yr, 3),
                Di_effective_annual=round(de, 3), switch_to_terminal_years=round(t_sw / 12, 1),
                r2_log=round(p.get("r2_log", np.nan), 3))


def dca_from_history(h: pd.DataFrame, nboot: int = 150, seed: int = 0) -> dict:
    """Fit each stream from its own history. Returns stream -> result dict."""
    rng = np.random.default_rng(seed)
    n = len(h)
    out = {}
    for s in C.STREAMS:
        q = h[s].values.astype(float)
        k_peak = int(np.argmax(q[: min(4, n)]))
        qq, mo = q[k_peak:], h.month_on.values[k_peak:]
        mask = _clean_mask(qq)
        t = (mo - mo[0]).astype(float)
        p = fit_arps(t[mask], qq[mask])
        fut_mo = np.arange(n + 1, HORIZON + 1)
        tf = fut_mo - mo[0]
        base_fit = arps_modified(t[mask], p["qi"], p["di"], p["b"])
        sims = np.empty((nboot, len(fut_mo)))
        params = []
        for i in range(nboot):
            rb = rng.choice(p["resid"], size=mask.sum(), replace=True)
            pb = fit_arps(t[mask], base_fit * np.exp(rb), x0=p["x"])
            params.append((pb["qi"], pb["di"], pb["b"]))
            sims[i] = arps_modified(tf, pb["qi"], pb["di"], pb["b"])
        # Anchor to the last observed rate level (removes small level offset)
        out[s] = dict(params=p, boot_params=np.array(params), month_on=fut_mo,
                      quant=_quant(sims, fut_mo),
                      fit=pd.Series(arps_modified(t, p["qi"], p["di"], p["b"]), index=mo),
                      used_mask=pd.Series(mask, index=mo))
    return out


def dca_from_type_curve(ctx: RunContext, lateral_ft: float, nboot: int = 150, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    fut_mo = np.arange(1, HORIZON + 1)
    for s in C.STREAMS:
        mat = ctx.results["analog_curves"][s]
        tc = type_curve(ctx, s, lateral_ft)
        q = tc.P50.values
        mo = tc.index.values
        k_peak = int(np.argmax(q[: min(4, len(q))]))
        t = (mo[k_peak:] - mo[k_peak]).astype(float)
        p = fit_arps(t, q[k_peak:])
        peak_mo = int(mo[k_peak])
        ramp = np.ones(len(fut_mo))
        for j in range(peak_mo - 1):          # months before peak keep the analog ramp shape
            ramp[j] = q[j] / q[k_peak]
        sims = np.empty((nboot, len(fut_mo)))
        params = []
        cols = mat.columns.values
        for i in range(nboot):
            pick = rng.choice(cols, size=len(cols), replace=True)
            m = mat[pick]
            ok = m.notna().sum(axis=1) >= 3
            curve = m[ok].median(axis=1) * lateral_ft / 1_000
            cq, cmo = curve.values, curve.index.values
            kp = int(np.argmax(cq[: min(4, len(cq))]))
            pb = fit_arps((cmo[kp:] - cmo[kp]).astype(float), cq[kp:], x0=p["x"])
            params.append((pb["qi"], pb["di"], pb["b"]))
            sims[i] = ramp * arps_modified(np.maximum(fut_mo - peak_mo, 0), pb["qi"], pb["di"], pb["b"])
        out[s] = dict(params=p, boot_params=np.array(params), month_on=fut_mo,
                      quant=_quant(sims, fut_mo),
                      fit=pd.Series(arps_modified(t, p["qi"], p["di"], p["b"]), index=mo[k_peak:]),
                      used_mask=None)
    return out


def _quant(sims, mo):
    return pd.DataFrame({"P90": np.percentile(sims, 10, axis=0),
                         "P50": np.percentile(sims, 50, axis=0),
                         "P10": np.percentile(sims, 90, axis=0)}, index=pd.Index(mo, name="month_on"))


def fit_decline_curves(ctx: RunContext, nboot: int = 150) -> dict:
    tgt = ctx.results["target"]
    if ctx.target.mode == "existing":
        res = dca_from_history(ctx.results["history"], nboot=nboot, seed=ctx.seed)
        basis = "target well history"
    else:
        if "analog_curves" not in ctx.results:
            raise RuntimeError("Run analog selection before decline analysis for a proposed well.")
        res = dca_from_type_curve(ctx, tgt["lateral_ft"], nboot=nboot, seed=ctx.seed)
        basis = "analog P50 type curve scaled to the proposed lateral"
    ctx.results["dca"] = res
    summary = {"basis": basis}
    for s in C.STREAMS:
        d = describe(res[s]["params"])
        summary[s] = d
        if res[s]["params"]["at_b_bound"]:
            ctx.flag(AGENT, "warn", f"{s.title()} b-factor is at a fitting bound ({d['b']}).")
        elif s == "oil" and d["b"] > 1.5:
            ctx.flag(AGENT, "info", f"{s.title()} b-factor {d['b']} implies extended transient flow; "
                                    "terminal decline carries much of the EUR.")
    ctx.emit(AGENT, f"Arps fits on {basis} · oil b={summary['oil']['b']}, "
                    f"Di={summary['oil']['Di_effective_annual']:.0%}/yr effective, "
                    f"R²(log)={summary['oil']['r2_log']}")
    return summary
