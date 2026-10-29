"""Agent 2 · Analog selection.

Scores candidate wells with a weighted Gower-style dissimilarity over location,
completion design, spacing, vintage and depth. Each feature's share of the
score is kept, so the ranking is explainable feature by feature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C
from ..context import RunContext

AGENT = "Analog"
FT_PER_MILE = 5_280.0

FEATURES = {
    # name: (target key, well column, weight)
    "distance": (None, "dist_mi", 0.25),
    "lateral length": ("lateral_ft", "lateral_ft", 0.15),
    "proppant intensity": ("proppant_lbft", "proppant_lbft", 0.20),
    "fluid intensity": ("fluid_bblft", "fluid_bblft", 0.10),
    "spacing": ("spacing_ft", "spacing_ft", 0.15),
    "vintage": ("first_prod_year", "first_prod_year", 0.10),
    "TVD": ("tvd", "tvd", 0.05),
}


def select_analogs(ctx: RunContext, radius_mi: float = 5.0, k: int = 8,
                   same_zone_only: bool = True, min_months: int = 12) -> dict:
    tgt = ctx.results["target"]
    w = ctx.wells.copy()
    w["dist_mi"] = np.hypot(w.mid_x - tgt["mid_x"], w.mid_y - tgt["mid_y"]) / FT_PER_MILE
    cand = w[(w.api != tgt["api"]) & (w.dist_mi <= radius_mi) & (w.months_on >= min_months)]
    if same_zone_only:
        cand = cand[cand.zone == tgt["zone"]]
    if cand.empty:
        ctx.flag(AGENT, "stop", "No analog candidates met the criteria.")
        ctx.results["analogs"] = cand
        return dict(n_analogs=0, radius_mi=radius_mi, same_zone_only=same_zone_only)

    contrib = {}
    for name, (tkey, col, wt) in FEATURES.items():
        if name == "distance":
            diff = cand[col] / max(radius_mi, 1e-6)
        else:
            rng = w[col].quantile(0.95) - w[col].quantile(0.05)
            diff = (cand[col] - tgt[tkey]).abs() / max(rng, 1e-6)
        contrib[name] = wt * diff.clip(0, 1)
    contrib = pd.DataFrame(contrib, index=cand.index)
    cand = cand.assign(dissimilarity=contrib.sum(axis=1))
    cand["similarity"] = 1 - cand.dissimilarity / sum(v[2] for v in FEATURES.values())
    top = cand.sort_values("dissimilarity").head(k)
    top_contrib = contrib.loc[top.index]
    metrics = ctx.results["metrics"].set_index("api")
    top = top.join(metrics[["cum12_bbl_per_ft", "peak_bopd_per_kft"]], on="api")
    ctx.results["analogs"] = top.reset_index(drop=True)
    ctx.results["analog_contrib"] = top_contrib.reset_index(drop=True).assign(api=top.api.values)

    # Type curves per 1,000 ft of lateral
    curves = {}
    hist = ctx.prod[ctx.prod.api.isin(top.api)]
    ll = top.set_index("api").lateral_ft
    for s in C.STREAMS:
        mat = hist.pivot(index="month_on", columns="api", values=s)
        curves[s] = mat.div(ll.reindex(mat.columns) / 1_000, axis=1)
    ctx.results["analog_curves"] = curves

    n = len(top)
    msim = float(top.similarity.median())
    ctx.emit(AGENT, f"Selected {n} analogs within {radius_mi:g} mi "
                    f"({'same zone' if same_zone_only else 'any zone'}) · median similarity {msim:.2f}")
    if n < 5:
        ctx.flag(AGENT, "warn", f"Only {n} analogs: type-curve percentiles are unstable.")
    if msim < 0.45:
        ctx.flag(AGENT, "warn", f"Median analog similarity is low ({msim:.2f}).")
    mean_contrib = top_contrib.mean().sort_values(ascending=False)
    return dict(
        n_analogs=n, radius_mi=radius_mi, same_zone_only=same_zone_only,
        median_similarity=round(msim, 3),
        main_dissimilarity_drivers=[f"{k} ({v:.3f})" for k, v in mean_contrib.head(3).items()],
        analogs=top[["api", "zone", "dist_mi", "lateral_ft", "proppant_lbft", "spacing_ft",
                     "first_prod", "months_on", "similarity", "cum12_bbl_per_ft"]]
        .round(3).to_dict("records"),
    )


def type_curve(ctx: RunContext, stream: str, lateral_ft: float, min_wells: int = 3) -> pd.DataFrame:
    """P90/P50/P10 type curve (calendar-day rate) scaled to a lateral length."""
    mat = ctx.results["analog_curves"][stream]
    counts = mat.notna().sum(axis=1)
    mat = mat[counts >= min_wells]
    scale = lateral_ft / 1_000
    q = pd.DataFrame({
        "P90": mat.quantile(0.10, axis=1) * scale,
        "P50": mat.quantile(0.50, axis=1) * scale,
        "P10": mat.quantile(0.90, axis=1) * scale,
        "n": counts[counts >= min_wells],
    })
    q.index.name = "month_on"
    # Light smoothing past the ramp so single downtime months in a few analogs do not
    # print into the forecast; the first three months keep their shape.
    for c in ("P90", "P50", "P10"):
        sm = np.exp(np.log(q[c].clip(lower=1e-3)).rolling(5, center=True, min_periods=1).median())
        q.loc[q.index > 3, c] = sm[q.index > 3]
    return q
