"""Plotly figures for the Streamlit app."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import config as C
from .geometry import well_trajectory, proxy_trajectory

PROXY_NOTE = "Depth and trajectory are geometric proxies, not measured directional surveys."
MEMBER_COLOURS = {"DCA": C.TEAL, "Analog": C.AMBER, "Transformer": "#7B4FA0"}


def _layout(fig, title=None, h=430, **kw):
    fig.update_layout(title=title, height=h, margin=dict(l=10, r=10, t=50 if title else 20, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Calibri, Segoe UI, sans-serif", size=13),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), **kw)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(120,140,150,0.18)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(120,140,150,0.18)")
    return fig


def month_dates(first_prod: str, months) -> pd.DatetimeIndex:
    p0 = pd.Period(first_prod, "M")
    return pd.PeriodIndex([p0 + int(m) - 1 for m in months], freq="M").to_timestamp()


# ------------------------------------------------------------------ 3D and map
def fig_3d(ctx, radius_mi: float = 3.0, h: int = 620):
    tgt = ctx.results["target"]
    w = ctx.wells
    an = set(ctx.results.get("analogs", pd.DataFrame()).get("api", []))
    ox, oy = tgt["surf_x"], tgt["surf_y"]
    dist = np.hypot(w.mid_x - tgt["mid_x"], w.mid_y - tgt["mid_y"]) / 5280
    show = w[(dist <= radius_mi) | w.api.isin(an)]
    fig = go.Figure()
    legend_done = set()
    for r in show.itertuples():
        x, y, z = well_trajectory(r)
        role = "Selected analog" if r.api in an else "Nearby well"
        col = C.AMBER if role == "Selected analog" else "#3C78B4"
        fig.add_trace(go.Scatter3d(
            x=x - ox, y=y - oy, z=z, mode="lines", line=dict(color=col, width=6 if role != "Nearby well" else 3),
            name=role, legendgroup=role, showlegend=role not in legend_done,
            hovertext=f"{r.api} · {r.zone} · TVD {r.tvd:,} ft · {r.lateral_ft:,} ft", hoverinfo="text"))
        legend_done.add(role)
    x, y, z = proxy_trajectory(tgt["surf_x"], tgt["surf_y"], tgt["heel_x"], tgt["heel_y"],
                               tgt["toe_x"], tgt["toe_y"], tgt["tvd"])
    fig.add_trace(go.Scatter3d(x=x - ox, y=y - oy, z=z, mode="lines", line=dict(color="#2FD3B8", width=11),
                               name="Target well" if ctx.target.mode == "existing" else "Proposed well"))
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers",
                               marker=dict(symbol="diamond", size=7, color="#FFD54A"),
                               name="Requested location"))
    toes = show
    fig.add_trace(go.Scatter3d(x=toes.toe_x - ox, y=toes.toe_y - oy, z=toes.tvd, mode="markers",
                               marker=dict(size=3, color=toes.tvd, colorscale="Turbo", reversescale=True,
                                           colorbar=dict(title="Proxy TVD (ft)", len=0.6)),
                               name="Toe (coloured by TVD)", hoverinfo="skip"))
    fig.update_layout(scene=dict(
        xaxis_title="East offset (ft)", yaxis_title="North offset (ft)", zaxis_title="TVD (ft)",
        zaxis=dict(autorange="reversed"), aspectmode="manual", aspectratio=dict(x=1, y=1.3, z=0.55),
        camera=dict(eye=dict(x=1.5, y=-1.6, z=0.9))))
    _layout(fig, h=h)
    fig.add_annotation(text=PROXY_NOTE, xref="paper", yref="paper", x=0, y=0, showarrow=False,
                       font=dict(size=11, color=C.SLATE))
    return fig


def fig_map(ctx, h: int = 460):
    tgt = ctx.results["target"]
    w = ctx.wells
    an = set(ctx.results.get("analogs", pd.DataFrame()).get("api", []))
    from .field import xy_to_latlon
    fig = go.Figure()
    for role, sub, col, width in [("Field well", w[~w.api.isin(an)], "#7C93A3", 2),
                                  ("Selected analog", w[w.api.isin(an)], C.AMBER, 4)]:
        lats, lons, txt = [], [], []
        for r in sub.itertuples():
            la, lo = xy_to_latlon([r.heel_x, r.toe_x], [r.heel_y, r.toe_y])
            lats += list(la) + [None]
            lons += list(lo) + [None]
        fig.add_trace(go.Scattermap(lat=lats, lon=lons, mode="lines", line=dict(color=col, width=width),
                                    name=role, hoverinfo="skip"))
        fig.add_trace(go.Scattermap(lat=sub.mid_lat, lon=sub.mid_lon, mode="markers",
                                    marker=dict(size=5, color=col), showlegend=False,
                                    text=sub.api + " · " + sub.zone, hoverinfo="text"))
    la, lo = xy_to_latlon([tgt["heel_x"], tgt["toe_x"]], [tgt["heel_y"], tgt["toe_y"]])
    fig.add_trace(go.Scattermap(lat=la, lon=lo, mode="lines", line=dict(color=C.TEAL, width=6),
                                name="Target lateral"))
    fig.add_trace(go.Scattermap(lat=[tgt["lat"]], lon=[tgt["lon"]], mode="markers",
                                marker=dict(size=13, color=C.AMBER), name="Requested location"))
    fig.update_layout(map=dict(style="carto-positron", zoom=10.3,
                               center=dict(lat=tgt["mid_lat"], lon=tgt["mid_lon"])))
    return _layout(fig, h=h)


# ----------------------------------------------------------------- production
def fig_history(hist: pd.DataFrame, first_prod: str, h=380):
    fig = go.Figure()
    d = month_dates(first_prod, hist.month_on)
    for s in C.STREAMS:
        fig.add_trace(go.Scatter(x=d, y=hist[s], name=f"{s.title()} ({C.STREAM_UNITS[s]})",
                                 mode="lines+markers", marker=dict(size=4),
                                 line=dict(color=C.STREAM_COLOURS[s], width=2)))
    fig.update_yaxes(type="log", title="Calendar-day rate")
    return _layout(fig, h=h)


def fig_analog_curves(ctx, stream: str = "oil", h=420):
    from .agents.analogs import type_curve
    mat = ctx.results["analog_curves"][stream]
    tgt = ctx.results["target"]
    fig = go.Figure()
    for col in mat.columns:
        fig.add_trace(go.Scatter(x=mat.index, y=mat[col], mode="lines", line=dict(color="rgba(232,163,23,0.35)", width=1),
                                 name=col, showlegend=False, hovertemplate=f"{col}<br>month %{{x}}: %{{y:.1f}}"))
    tc = type_curve(ctx, stream, 1_000)
    fig.add_trace(go.Scatter(x=tc.index, y=tc.P10, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=tc.index, y=tc.P90, fill="tonexty", fillcolor="rgba(20,128,140,0.18)",
                             line=dict(width=0), name="Analog P90 to P10"))
    fig.add_trace(go.Scatter(x=tc.index, y=tc.P50, line=dict(color=C.TEAL, width=3), name="Analog P50"))
    hist = ctx.results.get("history")
    if hist is not None:
        fig.add_trace(go.Scatter(x=hist.month_on, y=hist[stream] / (tgt["lateral_ft"] / 1_000),
                                 mode="markers", marker=dict(color=C.NAVY, size=5), name="Target history"))
    fig.update_yaxes(type="log", title=f"{stream.title()} rate per 1,000 ft ({C.STREAM_UNITS[stream]})")
    fig.update_xaxes(title="Months on production")
    return _layout(fig, h=h)


def fig_contrib(ctx, h=360):
    c = ctx.results["analog_contrib"].set_index("api")
    fig = go.Figure()
    pal = ["#0A2E36", "#14808C", "#E8A317", "#5B6B73", "#7B4FA0", "#B5413A", "#2E7D32"]
    for i, col in enumerate(c.columns):
        fig.add_trace(go.Bar(y=c.index, x=c[col], name=col, orientation="h", marker_color=pal[i % len(pal)]))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title="Weighted dissimilarity (lower is more similar)")
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, h=h)


def fig_dca(ctx, stream: str = "oil", h=420):
    tgt = ctx.results["target"]
    r = ctx.results["dca"][stream]
    fig = go.Figure()
    hist = ctx.results.get("history")
    if hist is not None:
        used = r["used_mask"]
        hh = hist.set_index("month_on")[stream]
        ok = used.reindex(hh.index).fillna(False).astype(bool)
        fig.add_trace(go.Scatter(x=month_dates(tgt["first_prod"], hh.index[ok]), y=hh[ok], mode="markers",
                                 marker=dict(color=C.STREAM_COLOURS[stream], size=5), name="History (fitted)"))
        fig.add_trace(go.Scatter(x=month_dates(tgt["first_prod"], hh.index[~ok]), y=hh[~ok], mode="markers",
                                 marker=dict(color="#999", size=7, symbol="x"), name="Excluded (downtime or ramp)"))
    fit = r["fit"]
    fig.add_trace(go.Scatter(x=month_dates(tgt["first_prod"], fit.index), y=fit.values,
                             line=dict(color=C.NAVY, width=2), name="Arps fit"))
    q = r["quant"].loc[: (0 if hist is None else len(hist)) + C.FORECAST_MONTHS]
    d = month_dates(tgt["first_prod"], q.index)
    fig.add_trace(go.Scatter(x=d, y=q.P10, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=d, y=q.P90, fill="tonexty", fillcolor="rgba(20,128,140,0.2)", line=dict(width=0),
                             name="Bootstrap P90 to P10"))
    fig.add_trace(go.Scatter(x=d, y=q.P50, line=dict(color=C.TEAL, width=3, dash="dash"), name="Forecast P50"))
    fig.update_yaxes(type="log", title=f"{stream.title()} ({C.STREAM_UNITS[stream]})")
    return _layout(fig, h=h)


def fig_transformer(ctx, stream: str = "oil", h=420):
    tgt = ctx.results["target"]
    fc = ctx.results["transformer"]["forecast"][stream]
    hist = ctx.results.get("history")
    n = 0 if hist is None else len(hist)
    fc = fc.loc[: n + C.FORECAST_MONTHS]
    fig = go.Figure()
    if hist is not None:
        fig.add_trace(go.Scatter(x=month_dates(tgt["first_prod"], hist.month_on), y=hist[stream], mode="markers",
                                 marker=dict(color=C.STREAM_COLOURS[stream], size=5), name="History"))
        ctxm = hist.month_on.iloc[-24:]
        fig.add_vrect(x0=month_dates(tgt["first_prod"], [ctxm.iloc[0]])[0],
                      x1=month_dates(tgt["first_prod"], [ctxm.iloc[-1]])[0],
                      fillcolor="rgba(232,163,23,0.10)", line_width=0, annotation_text="context window",
                      annotation_position="top left")
    d = month_dates(tgt["first_prod"], fc.index)
    fig.add_trace(go.Scatter(x=d, y=fc.P10, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=d, y=fc.P90, fill="tonexty", fillcolor="rgba(123,79,160,0.2)", line=dict(width=0),
                             name="Calibrated P90 to P10"))
    fig.add_trace(go.Scatter(x=d, y=fc.P50, line=dict(color="#7B4FA0", width=3), name="Transformer P50"))
    fig.update_yaxes(type="log", title=f"{stream.title()} ({C.STREAM_UNITS[stream]})")
    return _layout(fig, h=h)


def fig_loss(bundle, h=300):
    hst = bundle["history"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hst.epoch, y=hst.train, name="Train pinball", line=dict(color=C.TEAL)))
    fig.add_trace(go.Scatter(x=hst.epoch, y=hst.val, name="Held-out pads pinball", line=dict(color=C.AMBER)))
    fig.update_xaxes(title="Epoch")
    fig.update_yaxes(title="Loss")
    return _layout(fig, h=h)


def fig_ensemble(ctx, stream: str = "oil", h=460, months: int = C.FORECAST_MONTHS):
    tgt = ctx.results["target"]
    e = ctx.results["ensemble"]
    b = e["bands"][stream]
    hist = ctx.results.get("history")
    n = 0 if hist is None else len(hist)
    b = b.loc[: n + months]
    d = month_dates(tgt["first_prod"], b.index)
    fig = go.Figure()
    if hist is not None:
        fig.add_trace(go.Scatter(x=month_dates(tgt["first_prod"], hist.month_on), y=hist[stream], mode="lines",
                                 line=dict(color=C.STREAM_COLOURS[stream], width=2), name="History"))
    fig.add_trace(go.Scatter(x=d, y=b.P10, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=d, y=b.P90, fill="tonexty", fillcolor="rgba(10,46,54,0.16)", line=dict(width=0),
                             name="Ensemble P90 to P10"))
    for m in e["members"]:
        fig.add_trace(go.Scatter(x=d, y=b[f"{m}_P50"], line=dict(color=MEMBER_COLOURS[m], width=1.6, dash="dot"),
                                 name=f"{m} P50 ({e['weights'][m]:.0%})"))
    fig.add_trace(go.Scatter(x=d, y=b.P50, line=dict(color=C.NAVY, width=3.5), name="Ensemble P50"))
    fig.update_yaxes(type="log", title=f"{stream.title()} ({C.STREAM_UNITS[stream]})")
    return _layout(fig, h=h)


def fig_eur(ctx, stream: str = "oil", h=320):
    e = ctx.results["ensemble"]["eur"][stream]
    v = e["samples"] / 1e3
    lo, hi = np.percentile(v, [0.5, 99.5])
    v = v[(v >= lo) & (v <= hi)]
    fig = go.Figure(go.Histogram(x=v, nbinsx=50, marker_color="rgba(20,128,140,0.65)", name="Trajectories"))
    for k, col in (("P90", C.RED), ("P50", C.NAVY), ("P10", C.AMBER)):
        fig.add_vline(x=e[k] / 1e3, line=dict(color=col, width=2, dash="dash"),
                      annotation_text=f"{k} {e[k] / 1e3:,.0f}", annotation_position="top")
    fig.update_xaxes(title=f"{stream.title()} EUR ({C.STREAM_VOL_UNITS[stream]})")
    fig.update_yaxes(title="Count")
    return _layout(fig, h=h, showlegend=False)


def fig_hindcast(ctx, h=340):
    hp = ctx.results.get("hindcast_paths")
    tgt = ctx.results["target"]
    hist = ctx.results["history"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=month_dates(tgt["first_prod"], hist.month_on), y=hist.oil, mode="lines",
                             line=dict(color="#999", width=1.5), name="Oil history"))
    a = hp["actual"]
    fig.add_trace(go.Scatter(x=month_dates(tgt["first_prod"], a.month_on), y=a.oil, mode="markers",
                             marker=dict(color=C.STREAM_COLOURS["oil"], size=7), name="Held-out actual"))
    for m, p in hp["preds"].items():
        fig.add_trace(go.Scatter(x=month_dates(tgt["first_prod"], p.index), y=p.values, mode="lines",
                                 line=dict(color=MEMBER_COLOURS[m], width=2.5), name=f"{m} blind P50"))
    fig.update_yaxes(type="log", title="Oil (bbl/d)")
    lo = month_dates(tgt["first_prod"], [max(1, hp["cut"] - 24)])[0]
    fig.update_xaxes(range=[lo, month_dates(tgt["first_prod"], [len(hist) + 1])[0]])
    return _layout(fig, h=h)
