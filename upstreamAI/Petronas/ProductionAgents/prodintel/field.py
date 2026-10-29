"""Synthetic Midland Basin field generator.

Builds a seeded, physics-based teaching dataset: pads of horizontal wells in
three landing zones, with completion design, spatial rock quality, parent-child
interference, frac hits, downtime and monthly oil, gas and water volumes.

All API numbers use county codes 901 and 903, which do not exist in Texas, so
nothing here can be mistaken for a real well.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C

FT_PER_DEG_LON = C.FT_PER_DEG_LAT * np.cos(np.radians(C.FIELD_LAT))

ZONE_BASE_TVD = {"LSS": 7_800.0, "WCA": 9_050.0, "WCB": 9_550.0}
ZONE_QI_PER_KFT = {"LSS": 78.0, "WCA": 98.0, "WCB": 82.0}   # peak bopd per 1,000 ft
ZONE_GOR0 = {"LSS": 1.15, "WCA": 1.75, "WCB": 2.40}          # mcf/bbl
ZONE_WOR0 = {"LSS": 1.40, "WCA": 1.90, "WCB": 2.50}          # bbl/bbl
ZONE_B = {"LSS": 1.05, "WCA": 1.15, "WCB": 1.10}

OPERATORS = ["Caprock Energy", "Ector Ridge Operating", "Llano Estacado Resources",
             "Mustang Draw Oil"]


# ---------------------------------------------------------------- coordinates
def xy_to_latlon(x_ft, y_ft):
    lat = C.FIELD_LAT + np.asarray(y_ft) / C.FT_PER_DEG_LAT
    lon = C.FIELD_LON + np.asarray(x_ft) / FT_PER_DEG_LON
    return lat, lon


def latlon_to_xy(lat, lon):
    y = (np.asarray(lat) - C.FIELD_LAT) * C.FT_PER_DEG_LAT
    x = (np.asarray(lon) - C.FIELD_LON) * FT_PER_DEG_LON
    return x, y


# ------------------------------------------------------------ spatial surfaces
@dataclass
class Surfaces:
    bumps: dict  # zone -> list of (x0, y0, width, amplitude)
    struct: list  # structural bumps shared by all zones

    def quality(self, zone, x, y):
        q = np.ones_like(np.asarray(x, dtype=float))
        for x0, y0, w, a in self.bumps[zone]:
            q = q + a * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * w ** 2))
        return np.clip(q, 0.55, 1.50)

    def tvd(self, zone, x, y):
        base = ZONE_BASE_TVD[zone] - 0.0055 * np.asarray(x) + 0.0015 * np.asarray(y)
        for x0, y0, w, a in self.struct:
            base = base + a * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * w ** 2))
        return base


def _make_surfaces(rng) -> Surfaces:
    bumps = {}
    for z in C.ZONES:
        bumps[z] = [(rng.uniform(-45e3, 45e3), rng.uniform(-50e3, 50e3),
                     rng.uniform(12e3, 28e3), rng.choice([-1, 1]) * rng.uniform(0.15, 0.38))
                    for _ in range(6)]
    struct = [(rng.uniform(-40e3, 40e3), rng.uniform(-45e3, 45e3),
               rng.uniform(15e3, 30e3), rng.uniform(-120, 120)) for _ in range(4)]
    return Surfaces(bumps, struct)


# --------------------------------------------------------------- Arps helpers
def dmin_nominal_monthly(d_eff_annual: float = C.DMIN_EFF_ANNUAL) -> float:
    return -np.log(1 - d_eff_annual) / 12.0


def arps_modified(t, qi, di, b, dlim=None):
    """Modified hyperbolic rate. t in months from peak, di nominal per month."""
    dlim = dmin_nominal_monthly() if dlim is None else dlim
    t = np.asarray(t, dtype=float)
    b = max(float(b), 1e-6)
    if di <= dlim:
        return qi * np.exp(-di * t)
    t_sw = (di / dlim - 1.0) / (b * di)
    q_sw = qi / (1 + b * di * t_sw) ** (1 / b)
    hyp = qi / (1 + b * di * np.minimum(t, t_sw)) ** (1 / b)
    return np.where(t <= t_sw, hyp, q_sw * np.exp(-dlim * (t - t_sw)))


# ------------------------------------------------------------------ generator
def _months_between(start: pd.Period, end: pd.Period) -> int:
    return (end.year - start.year) * 12 + (end.month - start.month) + 1


def generate_field(seed: int = 7, n_wells: int = 140):
    """Return (wells, production, truth) DataFrames."""
    rng = np.random.default_rng(seed)
    surf = _make_surfaces(rng)
    end = pd.Period(C.DATA_END, "M")

    rows = []
    pad_id = 0
    while len(rows) < n_wells:
        pad_id += 1
        px, py = rng.uniform(-40e3, 40e3), rng.uniform(-45e3, 45e3)
        az = float(rng.choice([0.0, 180.0]))
        n_pad = int(rng.integers(3, 7))
        start = pd.Period("2015-03", "M") + int(rng.integers(0, 120))
        spacing = float(rng.choice([660, 880, 880, 1056, 1320]))
        prop = rng.uniform(1_500, 3_000)
        fluid = prop / rng.uniform(42, 60)
        ll_pad = float(rng.choice([7_500, 10_000, 10_000, 12_500, 15_000]))
        operator = OPERATORS[int(rng.integers(0, len(OPERATORS)))]
        zones_pad = rng.choice(C.ZONES, size=n_pad, p=[0.32, 0.40, 0.28])
        for i, zone in enumerate(zones_pad):
            if len(rows) >= n_wells:
                break
            offset = (i - (n_pad - 1) / 2) * spacing
            ll = ll_pad + rng.normal(0, 250)
            heel_x = px + offset
            heel_y = py + (450 if az == 0 else -450)
            sign = 1 if az == 0 else -1
            toe_y = heel_y + sign * ll
            mid_x, mid_y = heel_x, (heel_y + toe_y) / 2
            tvd = float(surf.tvd(zone, mid_x, mid_y)) + rng.normal(0, 25)
            county = "901" if px < 0 else "903"
            api = f"42-{county}-{40000 + len(rows) * 7 + int(rng.integers(0, 6)):05d}"
            fp = start + int(rng.integers(0, 2))
            rows.append(dict(
                api=api, well_name=f"{operator.split()[0].upper()} {pad_id:02d}-{i + 1} {i + 1}H",
                operator=operator, pad_id=pad_id, zone=zone,
                surf_x=px + offset * 0.15, surf_y=py, heel_x=heel_x, heel_y=heel_y,
                toe_x=heel_x + rng.normal(0, 60), toe_y=toe_y, mid_x=mid_x, mid_y=mid_y,
                azimuth=az, tvd=round(tvd), lateral_ft=round(ll),
                proppant_lbft=round(prop * rng.uniform(0.95, 1.05)),
                fluid_bblft=round(fluid * rng.uniform(0.95, 1.05), 1),
                spacing_ft=spacing, first_prod=str(fp),
                quality=float(surf.quality(zone, mid_x, mid_y)),
            ))
    wells = pd.DataFrame(rows)
    wells = wells[pd.PeriodIndex(wells.first_prod, freq="M") <= end - 3].reset_index(drop=True)
    wells["surf_lat"], wells["surf_lon"] = xy_to_latlon(wells.surf_x, wells.surf_y)
    wells["mid_lat"], wells["mid_lon"] = xy_to_latlon(wells.mid_x, wells.mid_y)
    fp_idx = pd.PeriodIndex(wells.first_prod, freq="M")
    wells["first_prod_year"] = fp_idx.year + (fp_idx.month - 1) / 12

    # Parent-child relationships: same zone, laterally close, overlapping, earlier
    child_factor = np.ones(len(wells))
    frac_hits = {i: [] for i in range(len(wells))}
    for i, wi in wells.iterrows():
        for j, wj in wells.iterrows():
            if i == j or wi.zone != wj.zone:
                continue
            dx = abs(wi.mid_x - wj.mid_x)
            dy = abs(wi.mid_y - wj.mid_y)
            if dx < 1_500 and dy < (wi.lateral_ft + wj.lateral_ft) / 2:
                lag = (wi.first_prod_year - wj.first_prod_year) * 12
                if lag >= 18:              # wi is a child of wj
                    child_factor[i] = min(child_factor[i], 0.80)
                    frac_hits[j].append(str(pd.Period(wi.first_prod, "M")))
    wells["child_factor"] = child_factor

    prod_rows, truth_rows = [], []
    dlim = dmin_nominal_monthly()
    for i, w in wells.iterrows():
        prop_f = (w.proppant_lbft / 2_000) ** 0.35
        sp_f = np.clip((w.spacing_ft / 880) ** 0.18, 0.85, 1.08)
        vintage_f = 1.0 - 0.012 * max(0.0, w.first_prod_year - 2019)
        qi = (ZONE_QI_PER_KFT[w.zone] * w.lateral_ft / 1_000 * w.quality * prop_f * sp_f
              * w.child_factor * vintage_f * rng.lognormal(0, 0.10))
        b = float(np.clip(ZONE_B[w.zone] + rng.normal(0, 0.12), 0.7, 1.5))
        di_eff = float(np.clip(rng.normal(0.72, 0.05), 0.58, 0.84))   # first-year effective
        di = ((1 - di_eff) ** (-b) - 1) / (12 * b)                  # nominal per month
        fp = pd.Period(w.first_prod, "M")
        n = _months_between(fp, end)
        m = np.arange(1, n + 1)
        base = arps_modified(np.maximum(m - 2, 0), qi, di, b, dlim)
        base[0] = 0.58 * qi
        noise = np.zeros(n)
        e = 0.0
        for k in range(n):
            e = 0.5 * e + rng.normal(0, 0.055)
            noise[k] = e
        rate = base * np.exp(noise)
        downtime = rng.random(n) < 0.035
        rate[downtime] *= rng.uniform(0.30, 0.75, downtime.sum())
        periods = pd.period_range(fp, end, freq="M")
        for hit in frac_hits[i]:
            k = periods.get_loc(pd.Period(hit, "M")) if pd.Period(hit, "M") in periods else None
            if k is not None:
                rate[k] *= 0.35
                if k + 1 < n:
                    rate[k + 1] *= 0.80
                rate[k + 2:] *= 0.93
        gor = ZONE_GOR0[w.zone] * (1 + 0.65 * (1 - np.exp(-m / 36)))
        wor = ZONE_WOR0[w.zone] * (1 + 0.30 * m / 60)
        wor[:3] *= np.array([1.8, 1.35, 1.1])[: min(3, n)]
        gas = rate * gor * np.exp(rng.normal(0, 0.04, n))
        water = rate * wor * np.exp(rng.normal(0, 0.05, n))
        days = periods.days_in_month.values
        for k in range(n):
            prod_rows.append((w.api, str(periods[k]), k + 1, int(days[k]),
                              rate[k] * days[k], gas[k] * days[k], water[k] * days[k]))
        truth_rows.append(dict(api=w.api, qi_oil=qi, b_oil=b, di_oil=di,
                               quality=w.quality, child_factor=w.child_factor,
                               frac_hits=len(frac_hits[i])))

    prod = pd.DataFrame(prod_rows, columns=["api", "month", "month_on", "days",
                                            "oil_bbl", "gas_mcf", "water_bbl"])
    prod["oil"] = prod.oil_bbl / prod.days
    prod["gas"] = prod.gas_mcf / prod.days
    prod["water"] = prod.water_bbl / prod.days
    wells["months_on"] = prod.groupby("api").size().reindex(wells.api).values
    wells["cum_oil_mbbl"] = (prod.groupby("api").oil_bbl.sum() / 1e3).reindex(wells.api).values
    truth = pd.DataFrame(truth_rows)
    wells = wells.drop(columns=["quality", "child_factor"])
    return wells, prod, truth, surf


def well_history(prod: pd.DataFrame, api: str) -> pd.DataFrame:
    return prod[prod.api == api].sort_values("month_on").reset_index(drop=True)
