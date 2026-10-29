"""Agent 4 · Transformer forecasting.

A small encoder-only Transformer that maps (static well descriptors, the last
24 observed months) to quantile forecasts of oil, gas and water for the next
120 months. The same network handles a proposed well: with zero observed
months the context is empty and the forecast is conditioned on design and
location alone.

Trust spine:
* the target well is never in the training set;
* validation holds out whole pads (grouped split), so neighbouring laterals on
  one pad cannot leak into the score;
* the P90-P10 interval is recalibrated on the held-out pads to 80 % coverage.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import config as C
from ..context import RunContext
from ..field import dmin_nominal_monthly

import os

AGENT = "Transformer"
DEFAULT_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                             ".model_cache")
L_CTX = 24
H = 120
QUANTILES = (0.1, 0.5, 0.9)

try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
except Exception:   # pragma: no cover
    TORCH_OK = False


def _static(w: dict, k: int) -> np.ndarray:
    z = [1.0 if w["zone"] == zz else 0.0 for zz in C.ZONES]
    return np.array(z + [w["lateral_ft"] / 10_000, w["proppant_lbft"] / 2_500,
                         w["fluid_bblft"] / 50, w["spacing_ft"] / 1_000,
                         (w["first_prod_year"] - 2020) / 5, w["tvd"] / 10_000,
                         w["mid_x"] / 50_000, w["mid_y"] / 50_000,
                         math.log1p(k) / 5], dtype=np.float32)


N_STATIC = 12


@dataclass
class Norm:
    mu: np.ndarray
    sd: np.ndarray


def _series(h: pd.DataFrame, ll: float) -> np.ndarray:
    return np.log1p(h[C.STREAMS].values / (ll / 1_000)).astype(np.float32)


def _sample(w: dict, s: np.ndarray, k: int, norm: Norm):
    """Build one (static, ctx, ctx_mask, target, target_mask) sample cut at month k."""
    n = len(s)
    zs = (s - norm.mu) / norm.sd
    ctx = np.zeros((L_CTX, 5), np.float32)
    cmask = np.ones(L_CTX, bool)          # True = padded
    lo = max(0, k - L_CTX)
    seg = zs[lo:k]
    if len(seg):
        ctx[-len(seg):, :3] = seg
        ctx[-len(seg):, 3] = 1.0
        ctx[-len(seg):, 4] = np.arange(lo + 1, k + 1) / 120.0
        cmask[-len(seg):] = False
    tgt = np.zeros((H, 3), np.float32)
    tmask = np.zeros(H, bool)
    fut = zs[k:k + H]
    tgt[:len(fut)] = fut
    tmask[:len(fut)] = True
    return _static(w, k), ctx, cmask, tgt, tmask


if TORCH_OK:
    class ProdTransformer(nn.Module):
        def __init__(self, d=64, nhead=4, layers=2, dropout=0.1):
            super().__init__()
            self.static = nn.Sequential(nn.Linear(N_STATIC, d), nn.GELU(), nn.Linear(d, d))
            self.tok = nn.Linear(5, d)
            self.pos = nn.Parameter(torch.randn(1, L_CTX + 1, d) * 0.02)
            enc = nn.TransformerEncoderLayer(d, nhead, 2 * d, dropout, batch_first=True,
                                             activation="gelu")
            self.encoder = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
            self.head = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, H * 3 * 3))

        def forward(self, st, ctx, cmask):
            x = torch.cat([self.static(st)[:, None], self.tok(ctx)], 1) + self.pos
            pad = torch.cat([torch.zeros(len(st), 1, dtype=torch.bool), cmask], 1)
            hcls = self.encoder(x, src_key_padding_mask=pad)[:, 0]
            o = self.head(hcls).view(-1, H, 3, 3)
            med = o[..., 1]
            lo = med - nn.functional.softplus(o[..., 0])
            hi = med + nn.functional.softplus(o[..., 2])
            return torch.stack([lo, med, hi], -1)          # (B, H, 3 streams, 3 quantiles)

    def _pinball(pred, tgt, tmask):
        q = torch.tensor(QUANTILES).view(1, 1, 1, 3)
        e = tgt[..., None] - pred
        loss = torch.maximum(q * e, (q - 1) * e)
        m = tmask[:, :, None, None].float()
        return (loss * m).sum() / (m.sum() * 9 + 1e-9)


def _cuts(n: int) -> list:
    base = [0, 1, 2, 3, 4, 6, 9, 12, 15, 18, 24, 30, 36]
    base += list(range(42, n - 2, 6))
    return [k for k in base if k <= n - 3]


def train_model(wells: pd.DataFrame, prod: pd.DataFrame, exclude_api=None, seed: int = 7,
                epochs: int = 40, val_frac: float = 0.15, val_pads=None):
    """Train on every well except exclude_api and the held-out pads. Returns a bundle dict."""
    if not TORCH_OK:
        raise RuntimeError("PyTorch is not installed")
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    rng = np.random.default_rng(seed)
    w_all = wells[wells.api != exclude_api].reset_index(drop=True)
    pads = w_all.pad_id.unique()
    if val_pads is None:
        val_pads = set(rng.choice(pads, size=max(2, int(len(pads) * val_frac)), replace=False))
    val_pads = set(int(v) for v in val_pads)
    series = {r.api: _series(prod[prod.api == r.api].sort_values("month_on"), r.lateral_ft)
              for r in w_all.itertuples()}
    train_stack = np.concatenate([series[a] for a, p in zip(w_all.api, w_all.pad_id)
                                  if p not in val_pads])
    norm = Norm(train_stack.mean(0), train_stack.std(0) + 1e-6)

    def build(split_val: bool):
        out = []
        for r in w_all.to_dict("records"):
            if (r["pad_id"] in val_pads) != split_val:
                continue
            s = series[r["api"]]
            ks = _cuts(len(s)) if not split_val else [k for k in (0, 6, 12, 24) if k <= len(s) - 6]
            out += [_sample(r, s, k, norm) for k in ks]
        return [torch.tensor(np.stack(x)) for x in zip(*out)]

    tr, va = build(False), build(True)
    model = ProdTransformer()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = len(tr[0])
    hist = []
    best, best_state = 1e9, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, 256):
            b = perm[i:i + 256]
            loss = _pinball(model(tr[0][b], tr[1][b], tr[2][b]), tr[3][b], tr[4][b])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(b)
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = _pinball(model(va[0], va[1], va[2]), va[3], va[4]).item()
        hist.append((ep + 1, tot / n, vl))
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()

    # Recalibrate the interval width on the held-out pads (oil, log space)
    with torch.no_grad():
        pv = model(va[0], va[1], va[2]).numpy()
    tgt, m = va[3].numpy(), va[4].numpy()
    med = pv[..., 0, 1][m]
    lo_w = (pv[..., 0, 1] - pv[..., 0, 0])[m]
    hi_w = (pv[..., 0, 2] - pv[..., 0, 1])[m]
    y = tgt[..., 0][m]
    scale = 1.0
    for s_ in np.linspace(0.6, 3.0, 49):
        cov = np.mean((y >= med - s_ * lo_w) & (y <= med + s_ * hi_w))
        if cov >= 0.80:
            scale = float(s_)
            break
    raw_cov = float(np.mean((y >= med - lo_w) & (y <= med + hi_w)))
    # Back-transform for an interpretable P50 error on held-out pads
    sd, mu = norm.sd[0], norm.mu[0]
    yr = np.expm1(y * sd + mu)
    pr = np.expm1(med * sd + mu)
    mape = float(np.median(np.abs(pr - yr) / np.maximum(yr, 1.0)))
    return dict(model=model, norm=norm, interval_scale=scale, raw_coverage=raw_cov,
                val_median_ape=mape, history=pd.DataFrame(hist, columns=["epoch", "train", "val"]),
                n_train_samples=n, n_val_samples=len(va[0]), val_pads=sorted(int(p) for p in val_pads),
                excluded=exclude_api)


def predict(bundle, w: dict, hist: pd.DataFrame | None, k: int | None = None) -> dict:
    """Forecast from cut month k (default: all of hist). Returns stream -> DataFrame.

    bundle is either one fold model or a fold collection; a collection is averaged
    (geometric mean per quantile) with the between-fold spread added in quadrature.
    """
    if bundle.get("kind") == "folds":
        outs = [_predict_one(b, w, hist, k) for b in bundle["folds"]]
        res = {}
        for sname in C.STREAMS:
            logs = np.stack([np.log(np.maximum(o[sname][["P90", "P50", "P10"]].values, 1e-3)) for o in outs])
            mean = logs.mean(0)
            sd_between = logs[:, :, 1].std(0)
            lo = mean[:, 1] - np.sqrt((mean[:, 1] - mean[:, 0]) ** 2 + (Z * sd_between) ** 2)
            hi = mean[:, 1] + np.sqrt((mean[:, 2] - mean[:, 1]) ** 2 + (Z * sd_between) ** 2)
            res[sname] = pd.DataFrame({"P90": np.exp(lo), "P50": np.exp(mean[:, 1]), "P10": np.exp(hi)},
                                      index=outs[0][sname].index)
        return res
    return _predict_one(bundle, w, hist, k)


Z = 1.2816


def _predict_one(bundle, w, hist, k=None):
    model, norm = bundle["model"], bundle["norm"]
    if hist is None or len(hist) == 0:
        s, k = np.zeros((0, 3), np.float32), 0
    else:
        s = _series(hist, w["lateral_ft"])
        k = len(s) if k is None else k
        s = s[:k]
    st, ctx, cmask, _, _ = _sample(w, np.vstack([s, np.zeros((H, 3), np.float32)]), k, norm)
    with torch.no_grad():
        out = model(torch.tensor(st[None]), torch.tensor(ctx[None]), torch.tensor(cmask[None]))[0].numpy()
    sc = bundle["interval_scale"]
    mo = np.arange(k + 1, k + H + 1)
    res = {}
    for j, sname in enumerate(C.STREAMS):
        med = out[:, j, 1]
        lo = med - sc * (out[:, j, 1] - out[:, j, 0])
        hi = med + sc * (out[:, j, 2] - out[:, j, 1])
        conv = lambda z: np.maximum(np.expm1(z * norm.sd[j] + norm.mu[j]), 0) * w["lateral_ft"] / 1_000
        res[sname] = pd.DataFrame({"P90": conv(lo), "P50": conv(med), "P10": conv(hi)},
                                  index=pd.Index(mo, name="month_on"))
    return res


def train_folds(wells: pd.DataFrame, prod: pd.DataFrame, seed: int = 7, k: int = 5, epochs: int = 40,
                cache_dir: str | None = None, progress=None) -> dict:
    """Pad-grouped k-fold training. Every pad is held out by exactly one fold model.

    A producing target is forecast by the fold that never saw its pad, which is
    stricter than excluding the well alone. A proposed location is forecast by
    all folds together.
    """
    import hashlib
    import os
    key = hashlib.md5(f"{seed}-{k}-{epochs}-{len(wells)}-{len(prod)}-{prod.oil_bbl.sum():.0f}".encode()).hexdigest()[:10]
    path = os.path.join(cache_dir, f"folds_{key}.pt") if cache_dir else None
    rng = np.random.default_rng(seed)
    pads = np.array(sorted(wells.pad_id.unique()))
    rng.shuffle(pads)
    groups = [set(int(p) for p in g) for g in np.array_split(pads, k)]
    folds = None
    if path and os.path.exists(path):
        try:
            saved = torch.load(path, weights_only=False)
            folds = []
            for item in saved:
                m = ProdTransformer()
                m.load_state_dict(item.pop("state"))
                m.eval()
                folds.append(dict(item, model=m))
        except Exception:
            folds = None
    if folds is None:
        folds = []
        for i, g in enumerate(groups):
            if progress:
                progress(i, k)
            folds.append(train_model(wells, prod, seed=seed + i, epochs=epochs, val_pads=g))
        if path:
            os.makedirs(cache_dir, exist_ok=True)
            torch.save([dict({kk: v for kk, v in f.items() if kk != "model"},
                             state=f["model"].state_dict()) for f in folds], path)
    pad_fold = {p: i for i, f in enumerate(folds) for p in f["val_pads"]}
    hist = sum(f["history"][["train", "val"]] for f in folds) / len(folds)
    hist.insert(0, "epoch", folds[0]["history"].epoch)
    return dict(kind="folds", folds=folds, pad_fold=pad_fold, k=k, excluded=None,
                val_median_ape=float(np.mean([f["val_median_ape"] for f in folds])),
                raw_coverage=float(np.mean([f["raw_coverage"] for f in folds])),
                interval_scale=float(np.mean([f["interval_scale"] for f in folds])),
                n_train_samples=int(np.mean([f["n_train_samples"] for f in folds])),
                val_pads=sorted(pad_fold), history=hist)


def select_model(collection: dict, wells: pd.DataFrame, api: str | None) -> dict:
    """The fold whose held-out pads include the target pad, or the whole collection."""
    if collection.get("kind") != "folds" or api is None:
        return collection
    pad = int(wells.loc[wells.api == api, "pad_id"].iloc[0])
    f = dict(collection["folds"][collection["pad_fold"][pad]])
    f["cv_median_ape"] = collection["val_median_ape"]
    f["excluded"] = api
    return f


def extend_tail(q: pd.DataFrame, end_month: int) -> pd.DataFrame:
    """Extend a quantile forecast to end_month with terminal exponential decline."""
    last = q.index.max()
    if last >= end_month:
        return q.loc[:end_month]
    dlim = dmin_nominal_monthly()
    mo = np.arange(last + 1, end_month + 1)
    f = np.exp(-dlim * (mo - last))
    ext = pd.DataFrame({c: q[c].iloc[-1] * f for c in ("P90", "P50", "P10")},
                       index=pd.Index(mo, name="month_on"))
    return pd.concat([q, ext])


def run_transformer_forecast(ctx: RunContext, bundle=None, epochs: int = 40, cache_dir=None) -> dict:
    if not TORCH_OK:
        ctx.flag(AGENT, "warn", "PyTorch is not installed; the Transformer agent was skipped.")
        ctx.results["transformer"] = None
        ctx.emit(AGENT, "Skipped: PyTorch unavailable")
        return dict(status="skipped", reason="PyTorch not installed")
    tgt = ctx.results["target"]
    if bundle is None:
        bundle = train_folds(ctx.wells, ctx.prod, seed=ctx.seed, epochs=epochs,
                             cache_dir=cache_dir or DEFAULT_CACHE)
    bundle = select_model(bundle, ctx.wells, tgt["api"])
    hist = ctx.results.get("history")
    fc = predict(bundle, tgt, hist)
    end = C.EUR_YEARS * 12
    fc = {s: extend_tail(v, end) for s, v in fc.items()}
    ctx.results["transformer"] = dict(bundle=bundle, forecast=fc)
    how = ("fold model that never saw the target pad" if bundle.get("kind") != "folds"
           else f"mean of {bundle['k']} pad-grouped fold models")
    ctx.emit(AGENT, f"Transformer forecast from the {how} · held-out median APE "
                    f"{bundle['val_median_ape']:.0%} · interval scale {bundle['interval_scale']:.2f}")
    if bundle["val_median_ape"] > 0.35:
        ctx.flag(AGENT, "warn", f"Held-out median APE is {bundle['val_median_ape']:.0%}; treat the "
                                "Transformer as a weak member of the ensemble.")
    first12 = fc["oil"].iloc[:12]
    return dict(status="ok", model_used=how, context_months=0 if hist is None else int(len(hist)),
                held_out_pads=bundle["val_pads"], held_out_median_ape=round(bundle["val_median_ape"], 3),
                raw_80pct_coverage=round(bundle["raw_coverage"], 3),
                interval_scale=round(bundle["interval_scale"], 2),
                next12_oil_mbbl_P50=round(float((first12.P50 * 30.4).sum() / 1e3), 1))
