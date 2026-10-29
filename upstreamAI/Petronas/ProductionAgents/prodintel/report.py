"""Engineering report package: PDF summary, CSV tables, interactive 3D HTML and a run manifest, zipped."""
from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

from . import __version__
from . import config as C
from .agents.decline import describe
from .plots import fig_3d, month_dates

NAVY = colors.HexColor(C.NAVY)
TEAL = colors.HexColor(C.TEAL)
AMBER = colors.HexColor(C.AMBER)


# --------------------------------------------------------------------- tables
def tables(ctx) -> dict:
    tgt = ctx.results["target"]
    out = {}
    hist = ctx.results.get("history")
    if hist is not None:
        out["history.csv"] = hist[["month", "month_on", "days", "oil_bbl", "gas_mcf", "water_bbl",
                                   "oil", "gas", "water"]]
    e = ctx.results["ensemble"]
    fc = []
    for s in C.STREAMS:
        b = e["bands"][s].copy()
        b.columns = [f"{s}_{c}" for c in b.columns]
        fc.append(b)
    fc = pd.concat(fc, axis=1).reset_index()
    fc.insert(1, "month", [str(pd.Period(tgt["first_prod"], "M") + int(m) - 1) for m in fc.month_on])
    out["forecast_monthly_rates.csv"] = fc.round(2)
    out["analogs.csv"] = ctx.results["analogs"][[
        "api", "well_name", "operator", "zone", "dist_mi", "lateral_ft", "proppant_lbft", "fluid_bblft",
        "spacing_ft", "first_prod", "months_on", "tvd", "similarity", "cum12_bbl_per_ft"]].round(3)
    out["analog_dissimilarity_contributions.csv"] = ctx.results["analog_contrib"].round(4)
    out["offset_wells.csv"] = ctx.results["offsets"][[
        "api", "well_name", "zone", "dist_mi", "tvd", "lateral_ft", "first_prod", "months_on",
        "cum_oil_mbbl"]].round(3)
    rows = []
    for s in C.STREAMS:
        rows.append(dict(stream=s, **describe(ctx.results["dca"][s]["params"])))
    out["decline_parameters.csv"] = pd.DataFrame(rows)
    eur = []
    for s in C.STREAMS:
        v = e["eur"][s]
        eur.append(dict(stream=s, unit=C.STREAM_VOL_UNITS[s], cum_to_date=v["cum_to_date"] / 1e3,
                        P90=v["P90"] / 1e3, P50=v["P50"] / 1e3, P10=v["P10"] / 1e3, mean=v["mean"] / 1e3))
    out["eur_summary.csv"] = pd.DataFrame(eur).round(1)
    if e["hindcast"] is not None:
        out["hindcast.csv"] = e["hindcast"]
    out["qc_flags.csv"] = pd.DataFrame(ctx.qc or [dict(agent="", level="", message="none")])
    out["agent_log.csv"] = pd.DataFrame([{k: v for k, v in r.items() if k != "data"} for r in ctx.log])
    return out


# --------------------------------------------------------------------- charts
def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def charts(ctx) -> dict:
    tgt = ctx.results["target"]
    e = ctx.results["ensemble"]
    hist = ctx.results.get("history")
    n = 0 if hist is None else len(hist)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    out = {}
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.2), sharex=True)
    for ax, s in zip(axes, C.STREAMS):
        b = e["bands"][s].loc[: n + C.FORECAST_MONTHS]
        d = month_dates(tgt["first_prod"], b.index)
        if hist is not None:
            ax.semilogy(month_dates(tgt["first_prod"], hist.month_on), hist[s], color=C.STREAM_COLOURS[s], lw=1.4,
                        label="History")
        ax.fill_between(d, b.P90, b.P10, color=C.NAVY, alpha=0.15, label="Ensemble P90 to P10")
        for m, col in (("DCA", C.TEAL), ("Analog", C.AMBER), ("Transformer", "#7B4FA0")):
            if f"{m}_P50" in b:
                ax.semilogy(d, b[f"{m}_P50"], color=col, lw=1, ls=":", label=f"{m} P50")
        ax.semilogy(d, b.P50, color=C.NAVY, lw=2.2, label="Ensemble P50")
        ax.set_ylabel(f"{s.title()} ({C.STREAM_UNITS[s]})")
        ax.grid(alpha=0.25, which="both")
    axes[0].legend(fontsize=7, ncol=3, loc="upper right")
    axes[0].set_title("Rate history and ensemble forecast", loc="left", color=C.NAVY, fontweight="bold")
    out["fig_forecast.png"] = _png(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for ax, s in zip(axes, ("oil", "gas")):
        v = e["eur"][s]
        x = v["samples"] / 1e3
        lo, hi = np.percentile(x, [0.5, 99.5])
        ax.hist(x[(x >= lo) & (x <= hi)], bins=45, color=C.TEAL, alpha=0.7)
        for (k, col), yf, ha in zip((("P90", C.RED), ("P50", C.NAVY), ("P10", C.AMBER)),
                                    (0.95, 0.80, 0.95), ("right", "left", "left")):
            ax.axvline(v[k] / 1e3, color=col, ls="--", lw=1.5)
            ax.text(v[k] / 1e3, ax.get_ylim()[1] * yf, f" {k} {v[k] / 1e3:,.0f} ", color=col, ha=ha, fontsize=7)
        ax.set_xlabel(f"{s.title()} EUR ({C.STREAM_VOL_UNITS[s]})")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Trajectories")
    out["fig_eur.png"] = _png(fig)

    from .agents.analogs import type_curve
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    mat = ctx.results["analog_curves"]["oil"]
    for c in mat.columns:
        ax.semilogy(mat.index, mat[c], color=C.AMBER, alpha=0.35, lw=0.8)
    tc = type_curve(ctx, "oil", 1_000)
    ax.fill_between(tc.index, tc.P90, tc.P10, color=C.TEAL, alpha=0.2)
    ax.semilogy(tc.index, tc.P50, color=C.TEAL, lw=2.2, label="Analog P50")
    if hist is not None:
        ax.semilogy(hist.month_on, hist.oil / (tgt["lateral_ft"] / 1_000), "o", ms=2.5, color=C.NAVY,
                    label="Target history")
    ax.set_xlabel("Months on production")
    ax.set_ylabel("Oil per 1,000 ft (bbl/d)")
    ax.set_title("Analog oil type curve, normalised by lateral length", loc="left", color=C.NAVY,
                 fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25, which="both")
    out["fig_analogs.png"] = _png(fig)
    return out


# ------------------------------------------------------------------------ PDF
def _tbl(data, widths=None, head=True):
    t = Table(data, colWidths=widths, hAlign="LEFT")
    st = [("FONT", (0, 0), (-1, -1), "Helvetica", 8), ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#C9D3D6")),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 2),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if head:
        st += [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
               ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8)]
    t.setStyle(TableStyle(st))
    return t


def pdf(ctx, figs: dict) -> bytes:
    tgt = ctx.results["target"]
    e = ctx.results["ensemble"]
    ss = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontName="Times-Bold", textColor=NAVY, fontSize=18)
    H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName="Times-Bold", textColor=TEAL, fontSize=12.5,
                        spaceBefore=8, spaceAfter=4)
    B = ParagraphStyle("B", parent=ss["BodyText"], fontName="Helvetica", fontSize=9, leading=12.5)
    S = ParagraphStyle("S", parent=B, fontSize=7.5, textColor=colors.HexColor(C.SLATE))
    buf = io.BytesIO()

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor(C.SLATE))
        canvas.drawString(18 * mm, 10 * mm, f"Digital Oil · Production Intelligence · synthetic demonstration data")
        canvas.drawRightString(192 * mm, 10 * mm, f"Page {doc.page}")
        canvas.setFillColor(AMBER)
        canvas.rect(18 * mm, 285 * mm, 174 * mm, 1.2 * mm, fill=1, stroke=0)
        canvas.restoreState()

    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm, title="Production Intelligence Report")
    el = [Paragraph("Production Intelligence Report", H1)]
    label = tgt["api"] if tgt["api"] else f"proposed location {tgt['lat']:.5f}, {tgt['lon']:.5f}"
    el.append(Paragraph(f"{tgt['well_name']} · {label} · {C.ZONE_NAMES[tgt['zone']]} · "
                        f"generated {datetime.now():%d %b %Y %H:%M} · orchestrator: {ctx.orchestrator}", S))
    el.append(Spacer(1, 6))
    el.append(Paragraph("Summary", H2))
    el.append(Paragraph(ctx.narrative.replace("\n", "<br/>"), B))

    el.append(Paragraph("Estimated ultimate recovery", H2))
    rows = [["Stream", "Unit", "Cum to date", "P90 (low)", "P50", "P10 (high)"]]
    for s in C.STREAMS:
        v = e["eur"][s]
        rows.append([s.title(), C.STREAM_VOL_UNITS[s], f"{v['cum_to_date'] / 1e3:,.0f}",
                     f"{v['P90'] / 1e3:,.0f}", f"{v['P50'] / 1e3:,.0f}", f"{v['P10'] / 1e3:,.0f}"])
    el.append(_tbl(rows))
    el.append(Spacer(1, 4))
    w = " · ".join(f"{k} {v:.0%}" for k, v in e["weights"].items())
    el.append(Paragraph(f"Ensemble weights: {w}. Basis: {e['basis']}. EUR horizon {C.EUR_YEARS} years or the "
                        f"{C.ECON_LIMIT_BOPD:g} bopd oil limit, whichever comes first.", S))
    if e["hindcast"] is not None:
        rows = [["Member", "12-month cum oil error", "Monthly MAPE"]]
        rows += [[r.method, f"{r.cum_oil_error:+.1%}", f"{r.monthly_mape:.1%}"] for r in e["hindcast"].itertuples()]
        el.append(Spacer(1, 4))
        el.append(_tbl(rows))
    el.append(Image(io.BytesIO(figs["fig_forecast.png"]), width=165 * mm, height=142 * mm))
    el.append(Image(io.BytesIO(figs["fig_eur.png"]), width=170 * mm, height=66 * mm))

    el.append(PageBreak())
    el.append(Paragraph("Target", H2))
    rows = [["Landing zone", C.ZONE_NAMES[tgt["zone"]], "TVD (ft)", f"{tgt['tvd']:,.0f} ({tgt['tvd_source']})"],
            ["Lateral (ft)", f"{tgt['lateral_ft']:,.0f}", "Proppant (lb/ft)", f"{tgt['proppant_lbft']:,.0f}"],
            ["Fluid (bbl/ft)", f"{tgt['fluid_bblft']:,.1f}", "Spacing (ft)", f"{tgt['spacing_ft']:,.0f}"],
            ["First production", tgt["first_prod"], "Surface lat, lon", f"{tgt['lat']:.5f}, {tgt['lon']:.5f}"]]
    if tgt.get("history"):
        h = tgt["history"]
        rows += [["Months on", str(h["months_on"]), "Cum oil (Mbbl)", f"{h['cum_oil_mbbl']:,.1f}"],
                 ["Peak oil (bopd)", f"{h['peak_oil_bopd']:,}", "Last 3 mo oil (bopd)", f"{h['last3_oil_bopd']:,.0f}"],
                 ["GOR (mcf/bbl)", f"{h['gor_mcf_per_bbl']:.2f}", "Water cut", f"{h['water_cut']:.0%}"]]
    el.append(_tbl(rows, widths=[32 * mm, 50 * mm, 34 * mm, 58 * mm], head=False))

    el.append(Paragraph("Selected analogs", H2))
    an = ctx.results["analogs"]
    rows = [["API", "Zone", "Dist (mi)", "Lateral", "Prop lb/ft", "Spacing", "First prod", "Similarity"]]
    rows += [[r.api, r.zone, f"{r.dist_mi:.2f}", f"{r.lateral_ft:,}", f"{r.proppant_lbft:,}",
              f"{r.spacing_ft:,.0f}", r.first_prod, f"{r.similarity:.2f}"] for r in an.itertuples()]
    el.append(_tbl(rows))
    el.append(Image(io.BytesIO(figs["fig_analogs.png"]), width=170 * mm, height=71 * mm))

    el.append(Paragraph("Decline curve parameters (modified hyperbolic)", H2))
    rows = [["Stream", "qi", "b", "Di nominal / yr", "Di effective / yr", "Switch (yr)", "R² (log)"]]
    for s in C.STREAMS:
        d = describe(ctx.results["dca"][s]["params"])
        rows.append([s.title(), f"{d['qi']:,.0f}", f"{d['b']:.2f}", f"{d['Di_nominal_annual']:.2f}",
                     f"{d['Di_effective_annual']:.0%}", f"{d['switch_to_terminal_years']:.1f}", f"{d['r2_log']:.3f}"])
    el.append(_tbl(rows))

    tf = ctx.results.get("transformer")
    if tf:
        bd = tf["bundle"]
        el.append(Paragraph("Transformer validation", H2))
        el.append(Paragraph(
            f"Trained on {bd['n_train_samples']:,} cut-samples with the target well excluded. Validation holds "
            f"out {len(bd['val_pads'])} whole pads. Held-out median absolute percentage error on monthly oil is "
            f"{bd['val_median_ape']:.0%}. The raw P90 to P10 interval covered {bd['raw_coverage']:.0%} of held-out "
            f"months and was rescaled by {bd['interval_scale']:.2f} to reach 80 % coverage.", B))

    el.append(Paragraph("Quality control flags", H2))
    if ctx.qc:
        rows = [["Agent", "Level", "Message"]] + [[q["agent"], q["level"], Paragraph(q["message"], B)] for q in ctx.qc]
        el.append(_tbl(rows, widths=[26 * mm, 16 * mm, 132 * mm]))
    else:
        el.append(Paragraph("No flags raised.", B))

    el.append(Paragraph("Method and limitations", H2))
    for txt in [
        "Data are a seeded synthetic Midland Basin field. API numbers use county codes 901 and 903, which do not exist.",
        "Trajectories are geometric proxies built from surface location, landing TVD, heel and toe. They are not "
        "measured directional surveys.",
        "Type curves are normalised linearly by lateral length; completion differences enter through analog ranking, "
        "not through a scaling law.",
        "Decline fits use a robust log-space loss, exclude suspected downtime, and switch to 8 % per year terminal "
        "decline. Bootstrap intervals reflect fit uncertainty, not reservoir uncertainty.",
        "The Transformer forecast beyond 120 months is extended with terminal exponential decline.",
        "Members are partly correlated (the proposed-well DCA member is fitted to the analog P50), so the blended "
        "interval may understate total uncertainty.",
    ]:
        el.append(Paragraph(f"· {txt}", B))
    doc.build(el, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()


# ------------------------------------------------------------------------ ZIP
def build_zip(ctx) -> bytes:
    figs = charts(ctx)
    tabs = tables(ctx)
    tgt = ctx.results["target"]
    stem = (tgt["api"] or f"proposed_{tgt['lat']:.4f}_{tgt['lon']:.4f}").replace("-", "")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{stem}/report.pdf", pdf(ctx, figs))
        for name, png in figs.items():
            z.writestr(f"{stem}/figures/{name}", png)
        for name, df in tabs.items():
            z.writestr(f"{stem}/data/{name}", df.to_csv(index=False))
        z.writestr(f"{stem}/well_geometry_3d.html",
                   fig_3d(ctx).to_html(include_plotlyjs="cdn", full_html=True))
        z.writestr(f"{stem}/summary.md", f"# Production intelligence summary\n\n{ctx.narrative}\n")
        manifest = dict(app=C.APP_NAME, version=__version__, generated=datetime.now().isoformat(timespec="seconds"),
                        orchestrator=ctx.orchestrator, seed=ctx.seed, mode=ctx.target.mode,
                        target={k: v for k, v in tgt.items() if isinstance(v, (str, int, float, type(None)))},
                        ensemble_weights=ctx.results["ensemble"]["weights"],
                        weighting_basis=ctx.results["ensemble"]["basis"],
                        tool_outputs=ctx.results.get("tool_outputs", {}))
        z.writestr(f"{stem}/run_manifest.json", json.dumps(manifest, indent=2, default=_jd))
    return buf.getvalue()


def _jd(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
