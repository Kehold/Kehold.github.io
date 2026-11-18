"""Classical image-log measurements (numpy only): automatic sinusoid picks and image texture metrics.

digitaloilEd.ai · Day 2 Session 2. Used by the data generator to produce the supplied inputs, and by
fmi_ex3 so teams can recompute them. Patches are (N, 64 rows of depth, 64 azimuth columns).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

W = 64
ROWS_PER_FT = 32
R_IN, PX_IN = 4.25, 12 / ROWS_PER_FT          # 8.5 in hole; 0.375 in per row
GAP_COLS = np.arange(7, W, 8)
VALID_COLS = np.setdiff1d(np.arange(W), GAP_COLS)
THETA = 2 * np.pi * (np.arange(W) + 0.5) / W
AMPS = np.r_[np.arange(0, 6), np.arange(6, 42, 2)]   # sinusoid amplitude (rows): fine steps for bedding, up to about 75 degrees
AZS = np.arange(0, 360, 360 / 24)              # dip azimuth search, 15 degree steps


def dip_from_amp(a_px):
    return np.degrees(np.arctan(np.asarray(a_px, float) * PX_IN / R_IN))


def pick_sinusoids(Z, batch=128, return_row=False):
    """Automatic sinusoid pick on each patch: the planar trace with the strongest mean thin-feature contrast.

    Z: (N, 64, 64) robust-scaled static image, NaN in pad gaps. Returns dip (deg), dip azimuth (deg) and a
    score (mean |contrast| along the pick divided by the patch's mean |contrast|; about 1 for no feature).
    return_row=True also returns the centre row of each pick."""
    N, H, _ = Z.shape
    S = np.abs(high_pass(Z))[:, :, VALID_COLS].astype(np.float32)
    base = S.mean(axis=(1, 2)) + 1e-6
    best = np.full(N, -1.0, np.float32); arg = np.zeros((N, 3))
    d0 = np.arange(H)[:, None]
    for a in AMPS:
        for az in AZS:
            off = np.round(a * np.cos(THETA[VALID_COLS] - np.radians(az))).astype(int)
            rows = d0 + off[None, :]                                   # (H, n_valid)
            inside = (rows >= 0) & (rows < H)
            rr = np.clip(rows, 0, H - 1)
            cc = np.broadcast_to(np.arange(len(VALID_COLS))[None, :], rr.shape)
            cover = inside.mean(1)                                     # share of the trace inside the patch
            for s in range(0, N, batch):
                acc = (S[s:s + batch][:, rr, cc] * inside).sum(2) / np.maximum(inside.sum(1), 1)
                acc = np.where(cover[None, :] >= 0.6, acc, 0)          # most of the trace must be visible
                m = acc.max(1); r0 = acc.argmax(1)
                upd = m > best[s:s + batch]
                best[s:s + batch][upd] = m[upd]
                arg[s:s + batch][upd] = np.column_stack([np.full(upd.sum(), a), np.full(upd.sum(), az), r0[upd]])
    out = (dip_from_amp(arg[:, 0]), arg[:, 1].copy(), best / base)
    return out + (arg[:, 2].astype(int),) if return_row else out


def high_pass(Z, sigma=2.0):
    """Thin-feature image: the static image minus its gap-aware Gaussian smooth (sigma in pixels); gaps become 0."""
    Zf = np.nan_to_num(Z, nan=0.0).astype(np.float32); w = (~np.isnan(Z)).astype(np.float32)
    mode = ("nearest", "nearest", "wrap")
    sm = gaussian_filter(Zf, (0, sigma, sigma), mode=mode) / np.maximum(gaussian_filter(w, (0, sigma, sigma), mode=mode), 1e-3)
    return np.where(w > 0, Zf - sm, 0.0)


def texture_metrics(D, Z=None, levels=16):
    """Grey-level co-occurrence (GLCM) texture of the dynamic image, plus conductive and resistive pixel
    fractions of the static image.

    D: (N, 64, 64) dynamic image in [0, 1], NaN in gaps. Z: robust-scaled static image (optional).
    Contrast is reported along azimuth (high for vertical features) and along depth (high for bedding)."""
    N = len(D)
    q = np.clip(np.nan_to_num(D, nan=-1) * levels, -1, levels - 1).astype(int)       # -1 marks gaps
    out = {}
    feats = []
    for name, a, b in (("az", q[:, :, :-1], q[:, :, 1:]), ("depth", q[:, :-1, :], q[:, 1:, :])):
        ok = (a >= 0) & (b >= 0)
        idx = (np.arange(N)[:, None, None] * levels * levels + a * levels + b)[ok]
        P = np.bincount(idx, minlength=N * levels * levels).reshape(N, levels, levels).astype(float)
        P = P + P.transpose(0, 2, 1)
        P /= P.sum(axis=(1, 2), keepdims=True)
        feats.append(P)
        i, j = np.meshgrid(np.arange(levels), np.arange(levels), indexing="ij")
        out[f"tex_contrast_{name}"] = (P * (i - j) ** 2).sum(axis=(1, 2))
    P = (feats[0] + feats[1]) / 2
    i, j = np.meshgrid(np.arange(levels), np.arange(levels), indexing="ij")
    out["tex_homogeneity"] = (P / (1 + np.abs(i - j))).sum(axis=(1, 2))
    out["tex_energy"] = np.sqrt((P ** 2).sum(axis=(1, 2)))
    out["tex_entropy"] = -(P * np.log2(P + 1e-12)).sum(axis=(1, 2))
    mu_i = (P * i).sum(axis=(1, 2)); mu_j = (P * j).sum(axis=(1, 2))
    sd_i = np.sqrt((P * (i - mu_i[:, None, None]) ** 2).sum(axis=(1, 2)))
    sd_j = np.sqrt((P * (j - mu_j[:, None, None]) ** 2).sum(axis=(1, 2)))
    out["tex_correlation"] = (P * (i - mu_i[:, None, None]) * (j - mu_j[:, None, None])).sum(axis=(1, 2)) / (sd_i * sd_j + 1e-9)
    out["tex_anisotropy"] = out["tex_contrast_az"] / (out["tex_contrast_depth"] + 1e-6)
    if Z is not None:
        out["conductive_fraction"] = np.nanmean(Z < -2.0, axis=(1, 2))
        out["resistive_fraction"] = np.nanmean(Z > 2.0, axis=(1, 2))
    return {k: np.asarray(v, float) for k, v in out.items()}


def robust_scale(X, groups):
    """Median and IQR/1.349 per group (well); NaN gaps stay NaN."""
    Z = np.empty_like(X, dtype=np.float32)
    for idx in groups.values():
        v = X[idx]
        med = np.nanmedian(v); q1, q3 = np.nanpercentile(v, [25, 75])
        Z[idx] = (v - med) / ((q3 - q1) / 1.349)
    return Z


def dynamic_normalise(X):
    """Dynamic normalisation: rank of every valid pixel within its 2 ft window (the patch), in [0, 1]."""
    out = np.full(X.shape, np.nan, np.float32)
    for n in range(len(X)):
        v = X[n]; ok = ~np.isnan(v)
        r = np.argsort(np.argsort(v[ok]))
        out[n][ok] = r / max(ok.sum() - 1, 1)
    return out
