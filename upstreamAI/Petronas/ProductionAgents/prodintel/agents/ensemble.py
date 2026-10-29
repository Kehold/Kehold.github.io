"""Agent 5 · Ensemble and uncertainty.

Blends the decline curve, Transformer and analog type-curve forecasts. For a
producing well with at least 24 months of history, weights come from a blind
12-month hindcast; for a proposed location, fixed prior weights are used and
the report says so. Each member is treated as a split log-normal per month and
the blend is sampled by Monte Carlo, with one shared draw per trajectory so
that cumulative volumes keep their full spread.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C
from ..context import RunContext
from .analogs import type_curve
from .decline import dca_from_history
from .transformer import extend_tail, predict, TORCH_OK

AGENT = "Ensemble"
DAYS = 30.4375
Z90 = 1.2816
PRIOR = {"DCA": 0.35, "Analog": 0.35, "Transformer": 0.30}


def _analog_forecast(ctx, hist: pd.DataFrame | None, lateral_ft: float, start: int, end: int):
    """Analog type curve for months start..end, rate-matched to history if present."""
    out = {}
    for s in C.STREAMS:
        tc = type_curve(ctx, s, lateral_ft)
        if tc.empty:
            return None
        if hist is not None and len(hist):
            n = len(hist)
            common = [m for m in range(max(1, n - 5), n + 1) if m in tc.index]
            if len(common) < 3:
                return None
            obs = hist.set_index("month_on").loc[common, s]
            factor = float(np.median(obs.values / tc.loc[common, "P50"].values))
            tc = tc[["P90", "P50", "P10"]] * factor
        tc = tc[["P90", "P50", "P10"]]
        tc = extend_tail(tc, end)
        out[s] = tc.loc[start:end]
    return out


def _dca_member(dca: dict, start: int, end: int):
    return {s: dca[s]["quant"].loc[start:end] for s in C.STREAMS}


def _hindcast(ctx: RunContext, members: list, holdout: int = 12) -> pd.DataFrame:
    h = ctx.results["history"]
    tgt = ctx.results["target"]
    n = len(h)
    cut = n - holdout
    hc, actual = h.iloc[:cut], h.iloc[cut:]
    act_cum = float(actual.oil_bbl.sum())
    rows = []
    preds = {}
    if "DCA" in members:
        d = dca_from_history(hc, nboot=20, seed=ctx.seed)
        preds["DCA"] = d["oil"]["quant"].loc[cut + 1:n, "P50"]
    if "Analog" in members:
        a = _analog_forecast(ctx, hc, tgt["lateral_ft"], cut + 1, n)
        if a is not None:
            preds["Analog"] = a["oil"]["P50"]
    if "Transformer" in members and ctx.results.get("transformer"):
        tf = predict(ctx.results["transformer"]["bundle"], tgt, h, k=cut)
        preds["Transformer"] = tf["oil"].loc[cut + 1:n, "P50"]
    for m, p in preds.items():
        days = actual.days.values
        cum = float((p.values * days).sum())
        mape = float(np.mean(np.abs(p.values - actual.oil.values) / np.maximum(actual.oil.values, 1)))
        rows.append(dict(method=m, cum_oil_error=round(cum / act_cum - 1, 4), monthly_mape=round(mape, 4)))
    ctx.results["hindcast_paths"] = dict(cut=cut, actual=actual[["month_on", "oil"]], preds=preds)
    return pd.DataFrame(rows)


def _split_lognormal_draw(q: pd.DataFrame, z: np.ndarray) -> np.ndarray:
    """q: P90/P50/P10 rows for months; z: (N,) shared draws. Returns (N, months)."""
    p50 = np.log(np.maximum(q.P50.values, 1e-3))
    s_lo = np.maximum(p50 - np.log(np.maximum(q.P90.values, 1e-3)), 1e-4) / Z90
    s_hi = np.maximum(np.log(np.maximum(q.P10.values, 1e-3)) - p50, 1e-4) / Z90
    zz = z[:, None]
    return np.exp(p50 + np.where(zz > 0, zz * s_hi, zz * s_lo))


def build_ensemble(ctx: RunContext, n_samples: int = 2_000, weights: dict | None = None) -> dict:
    tgt = ctx.results["target"]
    hist = ctx.results.get("history")
    n = 0 if hist is None else len(hist)
    start, end = n + 1, C.EUR_YEARS * 12

    members = {}
    if ctx.results.get("dca"):
        members["DCA"] = _dca_member(ctx.results["dca"], start, end)
    if ctx.results.get("analog_curves") is not None:
        a = _analog_forecast(ctx, hist, tgt["lateral_ft"], start, end)
        if a is not None:
            members["Analog"] = a
        else:
            ctx.flag(AGENT, "info", "Analog type curve does not reach the current month; analog member dropped.")
    if TORCH_OK and ctx.results.get("transformer"):
        members["Transformer"] = {s: v.loc[start:end] for s, v in ctx.results["transformer"]["forecast"].items()}
    if not members:
        raise RuntimeError("No forecast members available to blend.")

    hind = None
    if weights:
        w = {m: weights.get(m, 0.0) for m in members}
        basis = "user-specified weights"
    elif ctx.target.mode == "existing" and n >= 24:
        hind = _hindcast(ctx, list(members))
        err = {r.method: max(abs(r.cum_oil_error), 0.03) for r in hind.itertuples()}
        inv = {m: 1 / err.get(m, 0.5) for m in members}
        tot = sum(inv.values())
        w = {m: 0.7 * inv[m] / tot + 0.3 / len(members) for m in members}
        basis = "blind 12-month hindcast (70 % inverse cumulative error, 30 % equal)"
    else:
        w = {m: PRIOR[m] for m in members}
        basis = "fixed prior weights (no history to hindcast)"
        ctx.flag(AGENT, "info", "Weights are priors, not data-derived: no production history to hindcast.")
    tot = sum(w.values()) or 1.0
    w = {m: v / tot for m, v in w.items()}

    rng = np.random.default_rng(ctx.seed)
    names = list(members)
    pick = rng.choice(len(names), size=n_samples, p=[w[m] for m in names])
    z = rng.standard_normal(n_samples)
    months = np.arange(start, end + 1)
    samples = {}
    for s in C.STREAMS:
        arr = np.empty((n_samples, len(months)))
        for i, m in enumerate(names):
            idx = pick == i
            if idx.any():
                q = members[m][s].reindex(months).interpolate().bfill().ffill()
                arr[idx] = _split_lognormal_draw(q, z[idx])
        samples[s] = arr
    # Economic limit on oil sets the producing life of each trajectory
    alive = np.cumprod(samples["oil"] >= C.ECON_LIMIT_BOPD, axis=1).astype(bool)
    bands, eur = {}, {}
    cum_hist = {"oil": 0.0, "gas": 0.0, "water": 0.0}
    if hist is not None:
        cum_hist = {"oil": hist.oil_bbl.sum(), "gas": hist.gas_mcf.sum(), "water": hist.water_bbl.sum()}
    for s in C.STREAMS:
        arr = samples[s]
        b = pd.DataFrame({"P90": np.percentile(arr, 10, 0), "P50": np.percentile(arr, 50, 0),
                          "P10": np.percentile(arr, 90, 0)}, index=pd.Index(months, name="month_on"))
        for m in names:
            b[f"{m}_P50"] = members[m][s].reindex(months).P50.values
        bands[s] = b
        vol = (arr * alive * DAYS).sum(axis=1) + cum_hist[s]
        eur[s] = dict(P90=float(np.percentile(vol, 10)), P50=float(np.percentile(vol, 50)),
                      P10=float(np.percentile(vol, 90)), mean=float(vol.mean()),
                      samples=vol, cum_to_date=float(cum_hist[s]))
    member_eur = {}
    for m in names:
        q = members[m]["oil"].reindex(months).P50.values
        life = np.cumprod(q >= C.ECON_LIMIT_BOPD).astype(bool)
        member_eur[m] = float((q * life * DAYS).sum() + cum_hist["oil"])
    ctx.results["ensemble"] = dict(weights=w, basis=basis, hindcast=hind, bands=bands, eur=eur,
                                   member_eur_oil=member_eur, members=names)

    vals = list(member_eur.values())
    if len(vals) > 1 and max(vals) / max(min(vals), 1) > 1.5:
        ctx.flag(AGENT, "warn", "Members disagree by more than 50 % on oil EUR; review before use.")
    ratio = eur["oil"]["P10"] / max(eur["oil"]["P90"], 1)
    ctx.emit(AGENT, f"Blended {len(names)} members ({basis}) · oil EUR P90/P50/P10 "
                    f"{eur['oil']['P90'] / 1e3:,.0f} / {eur['oil']['P50'] / 1e3:,.0f} / "
                    f"{eur['oil']['P10'] / 1e3:,.0f} Mbbl")
    summary = dict(
        weights={k: round(v, 3) for k, v in w.items()}, weighting_basis=basis,
        hindcast=None if hind is None else hind.to_dict("records"),
        eur_oil_mbbl={k: round(eur["oil"][k] / 1e3, 1) for k in ("P90", "P50", "P10")},
        eur_gas_mmcf={k: round(eur["gas"][k] / 1e3, 1) for k in ("P90", "P50", "P10")},
        eur_water_mbbl={k: round(eur["water"][k] / 1e3, 1) for k in ("P90", "P50", "P10")},
        member_oil_eur_mbbl={k: round(v / 1e3, 1) for k, v in member_eur.items()},
        p10_p90_ratio_oil=round(ratio, 2),
        cum_oil_to_date_mbbl=round(cum_hist["oil"] / 1e3, 1),
    )
    return summary
