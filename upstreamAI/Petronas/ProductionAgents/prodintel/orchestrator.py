"""Orchestration: a deterministic rule-based runner and a Claude tool-calling runner.

Both drive the same five agent functions through the same RunContext, so the
outputs are directly comparable. The deterministic runner is the fallback
whenever no API key is configured or the model call fails.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from . import config as C
from .agents import analogs, decline, ensemble, spatial, transformer
from .context import RunContext

ORDER = ["resolve_target", "select_analogs", "fit_decline_curves",
         "run_transformer_forecast", "build_ensemble"]


# --------------------------------------------------------------- deterministic
def run_deterministic(ctx: RunContext, bundle=None, progress=None) -> RunContext:
    ctx.orchestrator = "deterministic"
    steps = [
        ("Spatial", lambda: spatial.resolve_target(ctx)),
        ("Analog", lambda: _analogs_with_relaxation(ctx)),
        ("Decline", lambda: decline.fit_decline_curves(ctx)),
        ("Transformer", lambda: transformer.run_transformer_forecast(ctx, bundle=bundle)),
        ("Ensemble", lambda: ensemble.build_ensemble(ctx)),
    ]
    for i, (name, fn) in enumerate(steps):
        if progress:
            progress(i, name)
        t0 = time.time()
        out = fn()
        ctx.results.setdefault("tool_outputs", {})[name] = out
        ctx.emit("Orchestrator", f"{name} agent finished in {time.time() - t0:.1f} s")
    ctx.narrative = template_narrative(ctx)
    if progress:
        progress(len(steps), "Done")
    return ctx


def _analogs_with_relaxation(ctx: RunContext):
    out = analogs.select_analogs(ctx, radius_mi=5, k=8, same_zone_only=True)
    if out["n_analogs"] < 5:
        ctx.emit("Orchestrator", "Too few analogs at 5 mi; widening the search to 8 mi")
        out = analogs.select_analogs(ctx, radius_mi=8, k=8, same_zone_only=True)
    if out["n_analogs"] < 5:
        ctx.emit("Orchestrator", "Still too few; relaxing the same-zone constraint")
        out = analogs.select_analogs(ctx, radius_mi=8, k=8, same_zone_only=False)
    return out


def template_narrative(ctx: RunContext) -> str:
    t = ctx.results["target"]
    e = ctx.results["ensemble"]
    eo = e["eur"]["oil"]
    eg = e["eur"]["gas"]
    d = decline.describe(ctx.results["dca"]["oil"]["params"])
    an = ctx.results["analogs"]
    w = ", ".join(f"{k} {v:.0%}" for k, v in e["weights"].items())
    if ctx.target.mode == "existing":
        h = t["history"]
        head = (f"{t['well_name']} (API {t['api']}) is a {t['lateral_ft']:,.0f} ft "
                f"{C.ZONE_NAMES[t['zone']]} lateral with {h['months_on']} months of production and "
                f"{h['cum_oil_mbbl']:,.0f} Mbbl oil to date. The last three months averaged "
                f"{h['last3_oil_bopd']:,.0f} bopd at a GOR of {h['gor_mcf_per_bbl']:.2f} mcf/bbl "
                f"and a water cut of {h['water_cut']:.0%}.")
    else:
        head = (f"The proposed {t['lateral_ft']:,.0f} ft lateral at {t['lat']:.5f}, {t['lon']:.5f} "
                f"lands in the {C.ZONE_NAMES[t['zone']]} ({t.get('zone_reason', '')}) at a proxy TVD of "
                f"{t['tvd']:,.0f} ft, completed at {t['proppant_lbft']:,.0f} lb/ft on "
                f"{t['spacing_ft']:,.0f} ft spacing.")
    body = (f" {len(an)} analogs were selected with a median similarity of "
            f"{an.similarity.median():.2f}. The oil decline fit gives b = {d['b']} and a first-year "
            f"effective decline of {d['Di_effective_annual']:.0%}. Ensemble weights are {w}, based on "
            f"{e['basis']}. Oil EUR is {eo['P90'] / 1e3:,.0f} / {eo['P50'] / 1e3:,.0f} / "
            f"{eo['P10'] / 1e3:,.0f} Mbbl (P90 / P50 / P10) and gas EUR is "
            f"{eg['P50'] / 1e3:,.0f} MMcf at P50.")
    flags = [q for q in ctx.qc if q["level"] != "info"]
    tail = (" Review items: " + " ".join(q["message"] for q in flags)) if flags else \
        " No warning-level QC flags were raised."
    return head + body + tail + (" All inputs are synthetic and trajectories are geometric proxies, "
                                 "not measured surveys.")


# ------------------------------------------------------------------- LLM agent
TOOLS = [
    dict(name="resolve_target",
         description="Agent 1. Resolve the API number or lat/lon to a lateral, landing zone, proxy TVD, "
                     "offset wells and a production history summary. Must run first.",
         input_schema=dict(type="object", properties=dict(
             offset_radius_mi=dict(type="number", description="Offset search radius in miles, 1 to 8",
                                   default=3)))),
    dict(name="select_analogs",
         description="Agent 2. Rank analog wells by weighted dissimilarity of location, completion, spacing, "
                     "vintage and TVD. Returns analog list, similarity and dissimilarity drivers. Re-run with a "
                     "wider radius or same_zone_only=false if fewer than 5 analogs are found.",
         input_schema=dict(type="object", properties=dict(
             radius_mi=dict(type="number", default=5), k=dict(type="integer", default=8),
             same_zone_only=dict(type="boolean", default=True)))),
    dict(name="fit_decline_curves",
         description="Agent 3. Fit modified hyperbolic Arps curves to oil, gas and water with bootstrap "
                     "uncertainty. For a proposed well this needs analogs first.",
         input_schema=dict(type="object", properties={})),
    dict(name="run_transformer_forecast",
         description="Agent 4. Train (target excluded, pad-grouped validation) and run the quantile "
                     "Transformer forecaster.",
         input_schema=dict(type="object", properties={})),
    dict(name="build_ensemble",
         description="Agent 5. Blend available forecasts with hindcast or prior weights and return "
                     "P90/P50/P10 EUR for oil, gas and water.",
         input_schema=dict(type="object", properties={})),
    dict(name="finalize",
         description="Submit the engineer-facing summary once the ensemble is built. 150 to 250 words.",
         input_schema=dict(type="object", properties=dict(narrative=dict(type="string")),
                           required=["narrative"])),
]

SYSTEM = """You are the orchestrating agent of a production intelligence system for Permian Basin \
horizontal wells. You coordinate five specialist tools in a sensible order, decide parameters, \
react to QC flags, and finish by calling finalize with a concise engineering summary.

Rules:
- Use only numbers returned by the tools. Never invent values.
- If select_analogs returns fewer than 5 analogs, widen radius_mi (up to 8), then relax \
same_zone_only, and say so in the summary.
- In the summary cover: target and landing zone, history or design, analog basis, decline \
character, ensemble weighting basis, EUR P90/P50/P10 for oil and gas, and every warning-level QC flag.
- State that data are synthetic and trajectories are proxies, not measured surveys.
- Petroleum convention: P90 is the low case and P10 the high case.
- Write in British English. Do not use em dashes. Plain, direct engineering prose."""


def _json(obj) -> str:
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    return json.dumps(obj, default=default)


def run_llm(ctx: RunContext, api_key: str | None = None, model: str = C.DEFAULT_MODEL,
            bundle=None, progress=None, max_turns: int = 16, client=None) -> RunContext:
    """Run the pipeline with Claude choosing the tool sequence and parameters."""
    if client is None:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
    ctx.orchestrator = f"Claude ({model})"
    done = set()
    t = ctx.target
    brief = (f"Target: API {t.api}" if t.mode == "existing" else
             f"Target: proposed well at lat {t.lat}, lon {t.lon}, landing zone '{t.zone}', design "
             f"{t.design.__dict__}")
    messages = [dict(role="user", content=f"{brief}. Produce the production intelligence assessment.")]
    for turn in range(max_turns):
        resp = client.messages.create(model=model, max_tokens=2_000, system=SYSTEM,
                                      tools=TOOLS, messages=messages)
        messages.append(dict(role="assistant", content=resp.content))
        for block in resp.content:
            if getattr(block, "type", "") == "text" and block.text.strip():
                ctx.emit("Orchestrator", block.text.strip()[:400])
        uses = [b for b in resp.content if getattr(b, "type", "") == "tool_use"]
        if not uses:
            break
        results = []
        for u in uses:
            ctx.emit("Orchestrator", f"Claude called {u.name}({_json(u.input)})")
            if progress and u.name in ORDER:
                progress(ORDER.index(u.name), u.name)
            try:
                out = _dispatch(ctx, u.name, dict(u.input or {}), done, bundle)
                is_err = False
            except Exception as exc:  # report back to the model so it can recover
                out, is_err = dict(error=str(exc)), True
            results.append(dict(type="tool_result", tool_use_id=u.id,
                                content=_json(out)[:12_000], is_error=is_err))
        messages.append(dict(role="user", content=results))
        if "finalize" in done:
            break
    if "build_ensemble" not in done:
        ctx.flag("Orchestrator", "warn", "Claude did not complete the pipeline; finished deterministically.")
        for name in ORDER:
            if name not in done:
                _dispatch(ctx, name, {}, done, bundle)
    if not ctx.narrative:
        ctx.narrative = template_narrative(ctx)
    ctx.narrative = ctx.narrative.replace(" — ", ", ").replace("—", ", ")
    if progress:
        progress(len(ORDER), "Done")
    return ctx


def _dispatch(ctx, name, args, done, bundle):
    need = {"select_analogs": ["resolve_target"],
            "fit_decline_curves": ["resolve_target"] + (["select_analogs"] if ctx.target.mode == "proposed" else []),
            "run_transformer_forecast": ["resolve_target"],
            "build_ensemble": ["fit_decline_curves"],
            "finalize": ["build_ensemble"]}.get(name, [])
    missing = [n for n in need if n not in done]
    if missing:
        raise RuntimeError(f"Prerequisite not met: run {', '.join(missing)} first.")
    t0 = time.time()
    if name == "resolve_target":
        out = spatial.resolve_target(ctx, offset_radius_mi=float(np.clip(args.get("offset_radius_mi", 3), 1, 8)))
    elif name == "select_analogs":
        out = analogs.select_analogs(ctx, radius_mi=float(np.clip(args.get("radius_mi", 5), 1, 10)),
                                     k=int(np.clip(args.get("k", 8), 3, 15)),
                                     same_zone_only=bool(args.get("same_zone_only", True)))
    elif name == "fit_decline_curves":
        out = decline.fit_decline_curves(ctx)
    elif name == "run_transformer_forecast":
        out = transformer.run_transformer_forecast(ctx, bundle=bundle)
    elif name == "build_ensemble":
        if "analog_curves" not in ctx.results:
            analogs.select_analogs(ctx)
            done.add("select_analogs")
        out = ensemble.build_ensemble(ctx)
        out["qc_flags"] = ctx.qc
    elif name == "finalize":
        ctx.narrative = args.get("narrative", "").strip()
        out = dict(status="accepted")
    else:
        raise RuntimeError(f"Unknown tool {name}")
    done.add(name)
    ctx.results.setdefault("tool_outputs", {})[name] = out
    if name != "finalize":
        ctx.emit("Orchestrator", f"{name} finished in {time.time() - t0:.1f} s")
    return out
