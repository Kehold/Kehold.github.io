"""Synthetic carbonate FMI field, preparation, tiling, sinusoid extraction and interval logs · v1.1

Six wells (CB-1 to CB-6) of unwrapped microresistivity images, 240 ft each, plus CB-7, a counter-example interval.
Image grid: 64 azimuth columns (5.6 degrees each) by 24 rows per foot (0.5 in). Pads cover 7 of every 8 columns;
every 8th column is a pad gap (NaN).

Inputs per well (the customer's FMI list): static image (calibrated, well gain and offset), dynamic image (rank
within a sliding 2 ft window), two-arm caliper C13 and C24 with pad 1 azimuth, interpreter dip and azimuth picks,
and texture metrics computed on 64-row tiles.

Pixel classes (interpreter picks; in this synthetic field they equal truth):
0 background, 1 open fracture (conductive), 2 closed fracture (resistive), 3 drilling-induced fracture,
4 stylolite, 5 vug. Bedding planes are picked as planes (dip, azimuth), not pixels.
Structural facies per foot: Layered, Fracture corridor, Vuggy, Stressed (drilling-induced fractures, breakouts; takes priority).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, gaussian_filter1d, maximum_filter1d

import fmi_measure as fmm

SEED = 2027
W = 64                      # azimuth columns
PX_PER_FT = 24              # 0.5 in per row
PX_IN = 0.5
R_IN = 4.25                 # borehole radius, 8.5 in bit
BIT_IN = 8.5
GAP_COLS = np.arange(7, W, 8)
VALID_COLS = np.setdiff1d(np.arange(W), GAP_COLS)
THETA = 2 * np.pi * (np.arange(W) + 0.5) / W
AZ_SH = 40.0                # SHmax azimuth, controls drilling-induced fractures; breakouts at 130 degrees
CLASSES = ["Background", "Open fracture (conductive)", "Closed fracture (resistive)",
           "Drilling-induced", "Stylolite", "Vug"]
FACIES = ["Layered", "Fracture corridor", "Vuggy", "Stressed"]
TILE = 64


def amp_px(dip_deg):
    return R_IN * np.tan(np.radians(dip_deg)) / PX_IN


def dip_from_amp(a_px):
    return np.degrees(np.arctan(np.asarray(a_px) * PX_IN / R_IN))


def sinus(d0, a, az_deg):
    """Row of a planar feature in every column; deepest point at the dip azimuth."""
    return d0 + a * np.cos(THETA - np.radians(az_deg))


@dataclass
class WellSpec:
    name: str
    top_ft: float
    role: str          # "train" or "blind"
    gain: float        # tool / mud response
    offset: float
    bed_dip: float
    bed_az: float
    bg_int: float      # background natural fracture intensity, per ft
    confound: bool     # drilling-induced fractures sit inside fracture corridors
    core: bool
    length_ft: int = 240
    n_dif: int = 0              # 0: two or three zones at random


WELLS = [
    WellSpec("CB-1", 9120, "train", 1.00, 0.00, 6, 200, 0.04, True, False),
    WellSpec("CB-2", 9340, "train", 0.80, 0.35, 9, 210, 0.06, True, False),
    WellSpec("CB-3", 9010, "train", 1.25, -0.30, 4, 190, 0.03, True, False),
    WellSpec("CB-4", 9460, "train", 0.90, 0.20, 11, 220, 0.05, True, False),
    WellSpec("CB-5", 9230, "blind", 1.15, -0.25, 7, 205, 0.04, False, True),
    WellSpec("CB-6", 9390, "blind", 0.75, 0.40, 8, 215, 0.05, False, False),
    WellSpec("CB-7", 9150, "extra", 1.05, 0.10, 7, 200, 0.05, False, False, 240, 7),
]


# ----------------------------------------------------------------------------- generator
def _draw_curve(img, lab, rows, cols, contrast, cls, thick, overwrite=(0, 4, 5)):
    H = img.shape[0]
    for r, c in zip(rows, cols):
        for t in range(thick):
            rr = int(round(r)) + t
            if 0 <= rr < H:
                img[rr, c] += contrast
                if lab[rr, c] in overwrite:
                    lab[rr, c] = cls


def make_well(spec: WellSpec, rng):
    H = spec.length_ft * PX_PER_FT
    # bedding: layered resistivity with structural dip
    prof = np.zeros(H + 600)
    pos = 0
    bounds = []
    while pos < len(prof):
        t = int(rng.uniform(1.0, 6.0) * PX_PER_FT)
        prof[pos:pos + t] = rng.normal(2.0, 0.3)
        bounds.append(pos)
        pos += t
    step = np.array([abs(prof[b] - prof[b - 1]) if b > 0 else 0.0 for b in bounds])
    prof = gaussian_filter1d(prof, 3) + gaussian_filter1d(rng.normal(0, 0.10, len(prof)), 2)
    bshift = amp_px(spec.bed_dip) * np.cos(THETA - np.radians(spec.bed_az))
    rows = np.arange(H)[:, None] - bshift[None, :] + 300
    img = np.interp(rows, np.arange(len(prof)), prof)
    img += gaussian_filter(rng.normal(0, 0.28, (H, W)), (1.2, 0.8), mode=("nearest", "wrap"))
    lab = np.zeros((H, W), np.uint8)
    beds = pd.DataFrame({"d0": np.array(bounds, float) - 300, "step": step})
    beds = beds[(beds.d0 > 12) & (beds.d0 < H - 12) & (beds.step > 0.2)].reset_index(drop=True)
    beds["dip"] = float(spec.bed_dip); beds["az"] = float(spec.bed_az)

    # fracture corridors and natural fractures
    corridors = []
    for _ in range(int(rng.integers(3, 6))):
        L = rng.uniform(6, 18)
        s = rng.uniform(5, spec.length_ft - L - 5)
        corridors.append((s, s + L, rng.uniform(0.5, 1.4)))
    lam = np.full(spec.length_ft, spec.bg_int)
    for s, e, i in corridors:
        lam[int(s):int(e)] = i
    fracs = []
    for ft in range(spec.length_ft):
        for _ in range(rng.poisson(lam[ft])):
            st = "A" if rng.random() < 0.6 else "B"
            az = (rng.normal(130, 12) if st == "A" else rng.normal(310, 12)) % 360
            dip = float(np.clip(rng.normal(68, 5), 55, 78))
            in_cor = lam[ft] > spec.bg_int
            typ = 1 if rng.random() < (0.7 if in_cor else 0.4) else 2
            fracs.append(dict(d0=(ft + rng.uniform()) * PX_PER_FT, dip=dip, az=az, type=typ,
                              set=st, in_corridor=bool(in_cor),
                              contrast=float(rng.uniform(0.35, 1.2))))

    # stylolites: jagged, bedding-parallel, conductive
    sty_zones = [(rng.uniform(0, spec.length_ft - 30), None) for _ in range(2)]
    sty_lam = np.full(spec.length_ft, 0.05)
    for s, _ in sty_zones:
        sty_lam[int(s):int(s) + 25] = 0.6
    for ft in range(spec.length_ft):
        for _ in range(rng.poisson(sty_lam[ft])):
            d0 = (ft + rng.uniform()) * PX_PER_FT
            jag = np.clip(np.cumsum(rng.choice([-1, 0, 1], W)), -3, 3)
            y = d0 + bshift + jag
            _draw_curve(img, lab, y, np.arange(W), -0.7, 4, 1, overwrite=(0, 5))

    # vugs: small conductive blobs in vuggy intervals
    vug_lam = np.full(spec.length_ft, 0.3)
    vug_zones = []
    for _ in range(2):
        s = int(rng.uniform(0, spec.length_ft - 20))
        vug_lam[s:s + 20] = 4.0
        vug_zones.append((s, s + 20))
    for ft in range(spec.length_ft):
        for _ in range(rng.poisson(vug_lam[ft])):
            r0, c0 = (ft + rng.uniform()) * PX_PER_FT, rng.uniform(0, W)
            rv, rh = rng.uniform(1.0, 2.5), rng.uniform(0.8, 1.8)
            for r in range(int(r0 - rv), int(r0 + rv) + 1):
                for c in range(int(c0 - rh), int(c0 + rh) + 1):
                    if 0 <= r < H and ((r - r0) / rv) ** 2 + ((c - c0) / rh) ** 2 <= 1:
                        img[r, c % W] -= 1.0
                        if lab[r, c % W] == 0:
                            lab[r, c % W] = 5

    # natural fractures drawn last of the geology so they win the label
    for f in fracs:
        a = amp_px(f["dip"])
        y = sinus(f["d0"], a, f["az"])
        keep = np.ones(W, bool)
        if rng.random() < 0.5:                       # partial trace
            n_miss = int(rng.integers(4, 16))
            st = int(rng.integers(0, W))
            keep[(st + np.arange(n_miss)) % W] = False
        cols = np.arange(W)[keep]
        c = -f["contrast"] if f["type"] == 1 else 1.1 * f["contrast"]
        _draw_curve(img, lab, y[keep], cols, c, f["type"], int(rng.integers(1, 3)), overwrite=(0, 3, 4, 5))

    # drilling-induced fractures: vertical streaks 180 degrees apart
    dif_zones = []
    for _ in range(spec.n_dif or int(rng.integers(2, 4))):
        L = rng.uniform(8, 20)
        if spec.confound:                            # training wells: the drilling-induced zone lies inside a corridor
            s0, e0, _ = corridors[int(rng.integers(len(corridors)))]
            L = min(L, e0 - s0)
            s = float(s0 + rng.uniform(0, 1) * (e0 - s0 - L))
        else:
            for _try in range(50):
                s = rng.uniform(2, spec.length_ft - L - 2)
                if lam[int(s):int(s + L)].max() <= spec.bg_int:
                    break
        dif_zones.append((s, s + L))
        r = s * PX_PER_FT
        while r < (s + L) * PX_PER_FT:
            seg = int(rng.integers(4, 14))
            for az in (AZ_SH, AZ_SH + 180):
                col = int(round((az + rng.normal(0, 4)) % 360 / 360 * W - 0.5)) % W
                rows_ = np.arange(int(r), int(r) + seg)
                _draw_curve(img, lab, rows_, np.full(seg, col), -1.0, 3, 1, overwrite=(0, 4, 5))
            r += int(rng.integers(6, 20))

    # structural facies per foot (stressed over corridor over vuggy over layered: a stressed interval is flagged even inside a corridor)
    fac = np.zeros(spec.length_ft, int)
    for s, e in vug_zones:
        fac[s:e] = 2
    for s, e, _ in corridors:
        fac[int(s):int(np.ceil(e))] = 1
    for s, e in dif_zones:
        fac[int(s):int(np.ceil(e))] = 3

    # washouts: both caliper arms enlarged, pads lose contact, the image goes soft and noisy
    wash = np.zeros(spec.length_ft, bool)
    for _ in range(2):
        L = int(rng.integers(4, 9)); s = int(rng.uniform(5, spec.length_ft - L - 5))
        wash[s:s + L] = True
    wrows = np.repeat(wash, PX_PER_FT)
    m0 = img[wrows].mean()
    img[wrows] = m0 + 0.45 * (img[wrows] - m0) + gaussian_filter(rng.normal(0, 0.45, (wrows.sum(), W)), (0.8, 0.6), mode=("nearest", "wrap"))

    img = gaussian_filter(img, (0.6, 0.4), mode=("nearest", "wrap"))
    raw = img * spec.gain + spec.offset
    raw[:, GAP_COLS] = np.nan

    # caliper per foot: breakouts elongate the hole toward SHmin in stressed intervals; the tool rotates slowly
    n = spec.length_ft
    p1az = (rng.uniform(0, 360) + np.cumsum(rng.normal(0, 3, n))) % 360
    elong = np.abs(rng.normal(0, 0.04, n))
    bo = (fac == 3) & (rng.random(n) < 0.6)
    elong[bo] = rng.uniform(0.4, 1.4, bo.sum())
    phi = np.radians(p1az - (AZ_SH + 90))
    c13 = BIT_IN + 0.05 + elong * np.cos(phi) ** 2 + rng.normal(0, 0.03, n)
    c24 = BIT_IN + 0.05 + elong * np.sin(phi) ** 2 + rng.normal(0, 0.03, n)
    wo = rng.uniform(1.0, 2.5, wash.sum())
    c13[wash] += wo; c24[wash] += wo * rng.uniform(0.7, 1.0, wash.sum())
    cal = pd.DataFrame({"depth_ft": spec.top_ft + np.arange(n) + 0.5, "C13_in": c13.round(2), "C24_in": c24.round(2),
                        "P1AZ_deg": p1az.round(1), "facies": [FACIES[k] for k in fac], "washout": wash, "breakout": bo})

    fr = pd.DataFrame(fracs)
    if len(fr):
        fr["depth_ft"] = spec.top_ft + fr.d0 / PX_PER_FT
        fr["amp_px"] = amp_px(fr.dip)
        fr["well"] = spec.name
    core = None
    if spec.core:
        counts = np.zeros(spec.length_ft, int)
        for f in fracs:
            counts[min(int(f["d0"] // PX_PER_FT), spec.length_ft - 1)] += 1
        core_cnt = rng.binomial(counts, 0.85).astype(float)
        for _ in range(3):
            s = int(rng.uniform(0, spec.length_ft - 5))
            core_cnt[s:s + int(rng.integers(2, 5))] = np.nan
        core = pd.DataFrame({"depth_ft": spec.top_ft + np.arange(spec.length_ft) + 0.5, "core_fractures_per_ft": core_cnt})
    return dict(spec=spec, raw=raw, dyn=dynamic_image(raw), lab=lab, fracs=fr, beds=beds, corridors=corridors, dif=dif_zones,
                vug_zones=vug_zones, core=core, cal=cal, depth=spec.top_ft + np.arange(H) / PX_PER_FT)


def dynamic_image(raw, window_ft=2.0):
    """Dynamic normalisation: rank of each valid pixel among the pixels of a sliding window (centred, window_ft tall),
    computed in half-foot blocks. Output in [0, 1], gaps NaN."""
    H = raw.shape[0]
    half = int(window_ft * PX_PER_FT / 2)
    blk = PX_PER_FT // 2
    out = np.full(raw.shape, np.nan, np.float32)
    for r0 in range(0, H, blk):
        r1 = min(r0 + blk, H)
        win = raw[max(0, r0 - half + blk // 2):min(H, r1 + half - blk // 2)]
        ref = np.sort(win[~np.isnan(win)])
        v = raw[r0:r1]
        ok = ~np.isnan(v)
        out[r0:r1][ok] = np.searchsorted(ref, v[ok]) / max(len(ref) - 1, 1)
    return np.clip(out, 0, 1)


def make_field(seed=SEED):
    rng = np.random.default_rng(seed)
    return {s.name: make_well(s, rng) for s in WELLS}


# ----------------------------------------------------------------------------- preparation
def prepare(w, normalise=True, mask_gaps=True, dynamic=True):
    """Network input (channels, rows, columns): normalised static image, optional dynamic image (2 x rank - 1) and a
    valid-data mask. w: a well dict from make_field."""
    raw = w["raw"]
    valid = ~np.isnan(raw)
    if normalise:
        med = np.nanmedian(raw)
        q1, q3 = np.nanpercentile(raw, [25, 75])
        scale = (q3 - q1) / 1.349
    else:
        med, scale = 2.0, 0.35          # one fixed field-wide scaling
    x = (raw - med) / scale
    if mask_gaps:
        x[~valid] = 0.0
        m = valid.astype(np.float32)
    else:
        x[~valid] = (0.0 - med) / scale  # naive fill: gaps read as zero resistivity
        m = np.ones_like(x, np.float32)
    ch = [x.astype(np.float32)]
    if dynamic:
        d = 2 * np.nan_to_num(w["dyn"], nan=0.5) - 1
        if not mask_gaps:
            d[~valid] = -1.0
        ch.append(d.astype(np.float32))
    ch.append(m)
    return np.stack(ch)


SEG_CLASSES = ["Background", "Open fracture", "Closed fracture", "Vug", "Stylolite"]


def seg_target(lab):
    """Segmentation target: 0 background, 1 open (conductive) fracture, 2 closed (resistive) fracture, 3 vug,
    4 stylolite (conductive but not open: a separate class keeps it out of the fracture picks)."""
    t = np.zeros(lab.shape, np.int64)
    t[lab == 1] = 1
    t[lab == 2] = 2
    t[lab == 5] = 3
    t[lab == 4] = 4
    return t


def make_tiles(prep, lab, stride=32, min_px=40):
    H = prep.shape[1]
    tops = np.arange(0, H - TILE + 1, stride)
    X = np.stack([prep[:, t:t + TILE] for t in tops])
    L = np.stack([lab[t:t + TILE] for t in tops])
    nat = ((L == 1) | (L == 2)).reshape(len(tops), -1).sum(1)
    y = (nat >= min_px).astype(np.float32)
    return X, y, L, tops


def interval_labels(w, tops):
    """Coarse labels: tile centre inside an interpreter's fractured-interval list (the corridors)."""
    mid_ft = (tops + TILE / 2) / PX_PER_FT
    y = np.zeros(len(tops), np.float32)
    for s, e, _ in w["corridors"]:
        y[(mid_ft >= s) & (mid_ft < e)] = 1
    return y


def build_dataset(field, normalise=True, mask_gaps=True, split="well", seed=0, labels="pixel", extra=False, dynamic=True):
    """Tiles for train and blind sets. split='well' holds out CB-5 and CB-6;
    split='random' shuffles overlapping tiles from all wells."""
    out = {"train": [], "test": []}
    allt = []
    for name, w in field.items():
        if w["spec"].role == "extra":
            if not extra or split != "well":
                continue
        p = prepare(w, normalise, mask_gaps, dynamic)
        stride = 32 if w["spec"].role == "blind" and split == "well" else 16
        X, y, L, tops = make_tiles(p, w["lab"], stride)
        if labels == "interval" and w["spec"].role == "train":
            y = interval_labels(w, tops)
        meta = pd.DataFrame({"well": name, "top": tops})
        if split == "well":
            out["test" if w["spec"].role == "blind" else "train"].append((X, y, L, meta))
        else:
            allt.append((X, y, L, meta))
    if split == "random":
        X = np.concatenate([a[0] for a in allt]); y = np.concatenate([a[1] for a in allt])
        L = np.concatenate([a[2] for a in allt]); meta = pd.concat([a[3] for a in allt], ignore_index=True)
        idx = np.random.default_rng(seed).permutation(len(y))
        n = int(0.7 * len(y))
        tr, te = idx[:n], idx[n:]
        out["train"] = [(X[tr], y[tr], L[tr], meta.iloc[tr].reset_index(drop=True))]
        out["test"] = [(X[te], y[te], L[te], meta.iloc[te].reset_index(drop=True))]
    cat = lambda parts: (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
                         np.concatenate([p[2] for p in parts]), pd.concat([p[3] for p in parts], ignore_index=True))
    return cat(out["train"]), cat(out["test"])


# ----------------------------------------------------------------------------- texture metrics per foot
def texture_log(w):
    """GLCM texture of the dynamic image and conductive / resistive fractions of the static image, per foot
    (64-row windows centred on each foot)."""
    H = w["raw"].shape[0]
    n = w["spec"].length_ft
    Z = (w["raw"] - np.nanmedian(w["raw"])) / ((np.subtract(*np.nanpercentile(w["raw"], [75, 25]))) / 1.349)
    c = (np.arange(n) * PX_PER_FT + PX_PER_FT // 2)
    t0 = np.clip(c - TILE // 2, 0, H - TILE)
    D = np.stack([w["dyn"][t:t + TILE] for t in t0]); Zs = np.stack([Z[t:t + TILE] for t in t0])
    tex = fmm.texture_metrics(D, Zs)
    out = pd.DataFrame(tex); out.insert(0, "depth_ft", w["spec"].top_ft + np.arange(n) + 0.5)
    return out


# ----------------------------------------------------------------------------- sinusoid extraction
_AMPS = np.arange(0, 45)
_AZS = np.arange(0, 360, 360 / 32)


def extract_sinusoids(prob, thr=0.45, nms_px=10, amps=_AMPS, azs=_AZS):
    """Hough-style search: mean probability (or feature strength) along every candidate sinusoid.
    prob: (H, W). Returns a DataFrame of picks."""
    H = prob.shape[0]
    P = maximum_filter1d(prob, 3, axis=0)
    pad = int(max(amps)) + 5
    Pp = np.pad(P, ((pad, pad), (0, 0)))
    best = np.full(H, -1.0); barg = np.zeros((H, 2), int)
    for ia, a in enumerate(amps):
        for iz, az in enumerate(azs):
            off = np.round(a * np.cos(THETA - np.radians(az))).astype(int)
            acc = np.zeros(H)
            for c in VALID_COLS:
                acc += Pp[pad + off[c]: pad + off[c] + H, c]
            acc /= len(VALID_COLS)
            better = acc > best
            best[better] = acc[better]
            barg[better] = (ia, iz)
    cand = np.flatnonzero((best >= thr) & (best == maximum_filter1d(best, 9)))
    cand = cand[np.argsort(-best[cand])]
    picks = []
    for d in cand:
        if all(abs(d - p["d0"]) >= nms_px for p in picks):
            a = amps[barg[d, 0]]; az = azs[barg[d, 1]]
            picks.append(dict(d0=float(d), amp_px=float(a), az=float(az), dip=float(dip_from_amp(a)), score=float(best[d])))
    return pd.DataFrame(picks, columns=["d0", "amp_px", "az", "dip", "score"]).sort_values("d0").reset_index(drop=True)


def bedding_picks(w, thr=4.0, max_dip=15):
    """Bedding planes: low-dip sinusoids along depth-gradient edges of the static image (classical, no network).
    thr is in units of the well's median edge strength."""
    raw = w["raw"]
    Z = np.nan_to_num((raw - np.nanmedian(raw)) / np.nanstd(raw), nan=0.0)
    Zs = gaussian_filter(Z, (2.0, 1.5), mode=("nearest", "wrap"))
    g = np.abs(np.gradient(Zs, axis=0))
    g = g / (np.median(g[:, VALID_COLS]) + 1e-9)
    amps = np.arange(0, int(np.ceil(amp_px(max_dip))) + 1)
    pk = extract_sinusoids(g, thr=thr, nms_px=PX_PER_FT // 2, amps=amps, azs=np.arange(0, 360, 15.0))
    return pk[pk.dip > 0.5].reset_index(drop=True) if len(pk) else pk


def pick_types(picks, cls_prob):
    """Open (conductive) or closed (resistive) from the mean class probability along each pick."""
    if not len(picks):
        return picks.assign(type=[])
    H = cls_prob.shape[1]
    types = []
    for p in picks.itertuples():
        rows = np.clip(np.round(sinus(p.d0, p.amp_px, p.az)).astype(int), 0, H - 1)
        r = rows[VALID_COLS]
        c1 = cls_prob[1, r, VALID_COLS].mean()
        c2 = cls_prob[2, r, VALID_COLS].mean()
        types.append(1 if c1 >= c2 else 2)
    return picks.assign(type=types)


def match_picks(picks, truth, tol_px=8):
    """Greedy match by centre depth. Returns matched pairs plus counts."""
    t = truth.sort_values("d0").reset_index(drop=True)
    used = np.zeros(len(t), bool)
    rows = []
    for p in picks.itertuples():
        if not len(t):
            break
        d = np.abs(t.d0.to_numpy() - p.d0)
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= tol_px:
            used[j] = True
            daz = abs((p.az - t.az[j] + 180) % 360 - 180)
            rows.append(dict(d0=p.d0, dip_pred=p.dip, dip_true=t.dip[j], az_pred=p.az, az_true=t.az[j],
                             az_err=daz, dip_err=p.dip - t.dip[j],
                             type_pred=getattr(p, "type", np.nan), type_true=t.type[j] if "type" in t else np.nan))
    m = pd.DataFrame(rows)
    return m, len(picks), len(t)


def p10_log(d0_px, length_ft, window_ft=3):
    counts = np.bincount(np.clip((np.asarray(d0_px) // PX_PER_FT).astype(int), 0, length_ft - 1), minlength=length_ft).astype(float)
    k = np.ones(window_ft) / window_ft
    return np.convolve(counts, k, mode="same")


def p32_log(d0_px, dip_deg, length_ft, window_ft=3, max_dip=80.0):
    """Fracture intensity P32 (area per volume, 1/ft) in a vertical well: each fracture counts 1 / cos(dip)."""
    w_ = 1 / np.cos(np.radians(np.clip(np.asarray(dip_deg, float), 0, max_dip)))
    counts = np.bincount(np.clip((np.asarray(d0_px) // PX_PER_FT).astype(int), 0, length_ft - 1), weights=w_, minlength=length_ft)
    return np.convolve(counts, np.ones(window_ft) / window_ft, mode="same")


def conductive_log(prob, length_ft):
    """Conductive feature probability per foot: the largest smoothed pixel probability of an open fracture or a vug
    within the foot (valid pad columns only)."""
    pc = gaussian_filter(prob[1] + prob[3], (1.0, 1.0), mode=("nearest", "wrap"))[:, VALID_COLS]
    H = pc.shape[0]
    return np.array([pc[f * PX_PER_FT:min((f + 1) * PX_PER_FT, H)].max() for f in range(length_ft)])


def conductive_truth(lab, length_ft, min_px=10):
    """A foot holds a conductive feature if it has at least min_px open-fracture or vug pixels."""
    H = lab.shape[0]
    return np.array([(np.isin(lab[f * PX_PER_FT:min((f + 1) * PX_PER_FT, H)], (1, 5))).sum() >= min_px for f in range(length_ft)])


def facies_features(w, p10, p32, cprob, vug_frac, tex, half=4):
    """Per-foot interval features for structural facies, as centred running means over 2 x half + 1 ft."""
    cal = w["cal"]
    f = pd.DataFrame({"p10": p10, "p32": p32, "cprob": cprob, "vug_frac": vug_frac,
                      "cal_elong": (cal.C13_in - cal.C24_in).abs().to_numpy(), "cal_mean": ((cal.C13_in + cal.C24_in) / 2).to_numpy(),
                      "tex_anisotropy": tex.tex_anisotropy.to_numpy(), "conductive_fraction": tex.conductive_fraction.to_numpy(),
                      "tex_entropy": tex.tex_entropy.to_numpy()})
    r = f.rolling(2 * half + 1, center=True, min_periods=1)
    out = r.mean().add_suffix("_mean")
    out["cal_elong_max"] = r.cal_elong.max().to_numpy()
    return out


FACIES_RULE_GRID = (np.arange(0.15, 1.0, 0.05), np.arange(0.15, 1.0, 0.05), np.arange(0.005, 0.04, 0.0025))


def facies_rules(X, thr):
    """Interpretation rules, in priority order: Stressed (caliper breakout in the window), Fracture corridor (window P10),
    Vuggy (U-Net vug fraction), else Layered. thr = (p10, caliper elongation in, vug fraction)."""
    out = np.array(["Layered"] * len(X), dtype=object)
    out[X.vug_frac_mean.to_numpy() >= thr[2]] = "Vuggy"
    out[X.p10_mean.to_numpy() >= thr[0]] = "Fracture corridor"
    out[X.cal_elong_max.to_numpy() >= thr[1]] = "Stressed"
    return out


def calibrate_facies(X, y):
    """Thresholds that maximise balanced accuracy on the training wells (grid search)."""
    from itertools import product
    from sklearn.metrics import balanced_accuracy_score
    best = (-1.0, None)
    for t in product(*FACIES_RULE_GRID):
        s = balanced_accuracy_score(y, facies_rules(X, t))
        if s > best[0]:
            best = (s, tuple(float(v) for v in t))
    return best[1], best[0]
