"""Agent 1 · Spatial, formation and geometry.

Resolves the input (API number or latitude/longitude) to a lateral, assigns the
landing zone, estimates landing TVD from offset control, gathers offset wells
and summarises historical oil, gas and water performance.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .. import config as C
from ..context import RunContext
from ..field import latlon_to_xy, xy_to_latlon

AGENT = "Spatial"
FT_PER_MILE = 5_280.0


def normalise_api(api: str) -> str:
    digits = re.sub(r"\D", "", str(api))
    if len(digits) >= 10:
        digits = digits[:10]
        return f"{digits[:2]}-{digits[2:5]}-{digits[5:10]}"
    return str(api).strip()


def well_metrics(wells: pd.DataFrame, prod: pd.DataFrame) -> pd.DataFrame:
    """Per-well productivity metrics normalised by lateral length."""
    g = prod.sort_values("month_on").groupby("api")
    cum6 = g.apply(lambda d: d.oil_bbl.iloc[:6].sum() if len(d) >= 6 else np.nan, include_groups=False)
    cum12 = g.apply(lambda d: d.oil_bbl.iloc[:12].sum() if len(d) >= 12 else np.nan, include_groups=False)
    peak = g.oil.max()
    m = wells.set_index("api")[["lateral_ft", "zone"]].copy()
    m["cum6_bbl_per_ft"] = cum6 / m.lateral_ft
    m["cum12_bbl_per_ft"] = cum12 / m.lateral_ft
    m["peak_bopd_per_kft"] = peak / m.lateral_ft * 1_000
    return m.reset_index()


def _history_summary(h: pd.DataFrame) -> dict:
    roll = h.oil.rolling(5, center=True, min_periods=2).median()
    downtime = int((h.oil < 0.6 * roll).sum())
    last3 = h.tail(3)
    gor = last3.gas.sum() / max(last3.oil.sum(), 1e-9)
    wc = last3.water.sum() / max(last3.water.sum() + last3.oil.sum(), 1e-9)
    return dict(
        months_on=int(len(h)), first_month=h.month.iloc[0], last_month=h.month.iloc[-1],
        cum_oil_mbbl=round(h.oil_bbl.sum() / 1e3, 1), cum_gas_mmcf=round(h.gas_mcf.sum() / 1e3, 1),
        cum_water_mbbl=round(h.water_bbl.sum() / 1e3, 1),
        peak_oil_bopd=round(float(h.oil.max())), peak_month=int(h.oil.idxmax()) + 1,
        last3_oil_bopd=round(float(last3.oil.mean()), 1),
        last3_gas_mcfd=round(float(last3.gas.mean()), 1),
        last3_water_bwpd=round(float(last3.water.mean()), 1),
        gor_mcf_per_bbl=round(float(gor), 2), water_cut=round(float(wc), 3),
        suspected_downtime_months=downtime,
    )


def _idw(x, y, xs, ys, vals, power=2.0, k=8):
    d = np.hypot(xs - x, ys - y)
    idx = np.argsort(d)[:k]
    w = 1 / np.maximum(d[idx], 200.0) ** power
    return float(np.sum(w * vals[idx]) / np.sum(w))


def resolve_target(ctx: RunContext, offset_radius_mi: float = 3.0) -> dict:
    wells, prod, t = ctx.wells, ctx.prod, ctx.target
    metrics = well_metrics(wells, prod)
    ctx.results["metrics"] = metrics

    if t.mode == "existing":
        api = normalise_api(t.api)
        hit = wells[wells.api == api]
        if hit.empty:
            close = wells.api[wells.api.str.endswith(api[-3:])].head(5).tolist()
            raise ValueError(f"API {api} is not in the field database. Similar: {close}")
        w = hit.iloc[0]
        tgt = dict(api=api, well_name=w.well_name, operator=w.operator, zone=w.zone,
                   tvd=float(w.tvd), lateral_ft=float(w.lateral_ft),
                   proppant_lbft=float(w.proppant_lbft), fluid_bblft=float(w.fluid_bblft),
                   spacing_ft=float(w.spacing_ft), first_prod=w.first_prod,
                   first_prod_year=float(w.first_prod_year),
                   surf_x=w.surf_x, surf_y=w.surf_y, heel_x=w.heel_x, heel_y=w.heel_y,
                   toe_x=w.toe_x, toe_y=w.toe_y, mid_x=w.mid_x, mid_y=w.mid_y,
                   lat=float(w.surf_lat), lon=float(w.surf_lon), azimuth=float(w.azimuth),
                   tvd_source="reported")
        hist = prod[prod.api == api].sort_values("month_on").reset_index(drop=True)
        ctx.results["history"] = hist
        tgt["history"] = _history_summary(hist)
        if len(hist) < 12:
            ctx.flag(AGENT, "warn", f"Short history ({len(hist)} months): decline parameters are weakly constrained.")
        ctx.emit(AGENT, f"Resolved API {api} · {w.well_name} · {C.ZONE_NAMES[w.zone]} · "
                        f"{len(hist)} months on production")
    else:
        if t.lat is None or t.lon is None:
            raise ValueError("Provide either an API number or a latitude and longitude.")
        d = t.design
        sx, sy = latlon_to_xy(t.lat, t.lon)
        sx, sy = float(sx), float(sy)
        sign = 1 if d.azimuth == 0 else -1
        heel_x, heel_y = sx, sy + sign * 450
        toe_x, toe_y = heel_x, heel_y + sign * d.lateral_ft
        mid_x, mid_y = heel_x, (heel_y + toe_y) / 2
        dist_all = np.hypot(wells.mid_x - mid_x, wells.mid_y - mid_y) / FT_PER_MILE
        if dist_all.min() > 2.0:
            ctx.flag(AGENT, "warn", f"Nearest well control is {dist_all.min():.1f} mi away; "
                                    "the location sits outside the well-controlled area.")
        # Landing zone selection from nearby normalised productivity
        near = metrics.assign(dist=dist_all.values)
        near = near[near.dist <= max(offset_radius_mi, 3.0)]
        zone_table = (near.groupby("zone").agg(wells=("api", "size"),
                                               cum12_bbl_per_ft=("cum12_bbl_per_ft", "median"))
                      .reindex(C.ZONES).reset_index())
        if t.zone == "auto":
            ok = zone_table[zone_table.wells >= 3].dropna()
            zone = (ok.sort_values("cum12_bbl_per_ft", ascending=False).zone.iloc[0]
                    if len(ok) else "WCA")
            reason = "highest median 12-month oil per foot among nearby offsets"
        else:
            zone, reason = t.zone, "specified by user"
        zw = wells[wells.zone == zone]
        tvd = _idw(mid_x, mid_y, zw.mid_x.values, zw.mid_y.values, zw.tvd.values.astype(float))
        tgt = dict(api=None, well_name="Proposed well", operator="Proposed", zone=zone,
                   tvd=round(tvd), lateral_ft=d.lateral_ft, proppant_lbft=d.proppant_lbft,
                   fluid_bblft=d.fluid_bblft, spacing_ft=d.spacing_ft,
                   first_prod=str(pd.Period(C.DATA_END, "M") + 1),
                   first_prod_year=float(pd.Period(C.DATA_END, "M").year + 1.0),
                   surf_x=sx, surf_y=sy, heel_x=heel_x, heel_y=heel_y, toe_x=toe_x, toe_y=toe_y,
                   mid_x=mid_x, mid_y=mid_y, lat=float(t.lat), lon=float(t.lon),
                   azimuth=d.azimuth, tvd_source="IDW of 8 nearest same-zone wells",
                   zone_reason=reason,
                   zone_table=zone_table.round(2).to_dict("records"))
        ctx.results["history"] = None
        ctx.emit(AGENT, f"Proposed location {t.lat:.5f}, {t.lon:.5f} · landing zone "
                        f"{C.ZONE_NAMES[zone]} ({reason}) · proxy TVD {tvd:,.0f} ft")

    # Offsets within radius of the lateral midpoint
    dist = np.hypot(wells.mid_x - tgt["mid_x"], wells.mid_y - tgt["mid_y"]) / FT_PER_MILE
    off = wells.assign(dist_mi=dist.values)
    off = off[(off.dist_mi <= offset_radius_mi) & (off.api != tgt["api"])].sort_values("dist_mi")
    ctx.results["offsets"] = off
    ctx.results["target"] = tgt
    mid_lat, mid_lon = xy_to_latlon(tgt["mid_x"], tgt["mid_y"])
    tgt["mid_lat"], tgt["mid_lon"] = float(mid_lat), float(mid_lon)
    ctx.emit(AGENT, f"{len(off)} offset wells within {offset_radius_mi:g} mi "
                    f"({(off.zone == tgt['zone']).sum()} in the same zone)")
    if (off.zone == tgt["zone"]).sum() < 4:
        ctx.flag(AGENT, "warn", "Fewer than four same-zone offsets in the search radius.")

    summary = {k: v for k, v in tgt.items() if k not in
               ("surf_x", "surf_y", "heel_x", "heel_y", "toe_x", "toe_y", "mid_x", "mid_y")}
    summary["mode"] = t.mode
    summary["offsets_in_radius"] = int(len(off))
    summary["same_zone_offsets"] = int((off.zone == tgt["zone"]).sum())
    return summary
