"""End-to-end tests. Run with:  python -m pytest -q"""
import io
import zipfile
from types import SimpleNamespace as NS

import numpy as np
import pytest

from prodintel.agents.decline import fit_arps
from prodintel.context import RunContext, Target
from prodintel.field import arps_modified, generate_field
from prodintel.orchestrator import run_deterministic, run_llm
from prodintel.report import build_zip


@pytest.fixture(scope="module")
def field():
    return generate_field(seed=7)


def test_arps_recovers_known_parameters():
    t = np.arange(0, 60, dtype=float)
    q = arps_modified(t, 800, 0.2, 1.1)
    p = fit_arps(t, q)
    assert abs(p["qi"] - 800) / 800 < 0.02
    assert abs(p["b"] - 1.1) < 0.05


def test_existing_well_pipeline(field):
    w, p, truth, s = field
    ctx = run_deterministic(RunContext(w, p, s, Target(api=w.api.iloc[20])))
    e = ctx.results["ensemble"]["eur"]["oil"]
    assert e["P90"] <= e["P50"] <= e["P10"]
    assert e["P90"] >= e["cum_to_date"]
    assert abs(sum(ctx.results["ensemble"]["weights"].values()) - 1) < 1e-9
    assert "—" not in ctx.narrative


def test_proposed_well_pipeline_and_zip(field):
    w, p, truth, s = field
    ctx = run_deterministic(RunContext(w, p, s, Target(lat=31.86, lon=-102.13)))
    assert ctx.results["target"]["zone"] in ("LSS", "WCA", "WCB")
    z = zipfile.ZipFile(io.BytesIO(build_zip(ctx)))
    names = z.namelist()
    assert any(n.endswith("report.pdf") for n in names)
    assert any(n.endswith("forecast_monthly_rates.csv") for n in names)


def test_target_well_never_in_transformer_training(field):
    from prodintel.agents.transformer import TORCH_OK, train_model
    if not TORCH_OK:
        pytest.skip("torch not installed")
    w, p, truth, s = field
    api = w.api.iloc[5]
    b = train_model(w, p, exclude_api=api, epochs=2)
    assert b["excluded"] == api
    from prodintel.agents.transformer import select_model, train_folds
    folds = train_folds(w, p, k=3, epochs=2)
    f = select_model(folds, w, api)
    pad = int(w.loc[w.api == api, "pad_id"].iloc[0])
    assert pad in f["val_pads"]


class FakeClient:
    """Mimics the Anthropic client: scripted tool calls, then finalize."""

    def __init__(self):
        self.script = [("resolve_target", {"offset_radius_mi": 3}),
                       ("build_ensemble", {}),            # wrong order on purpose: must be refused
                       ("select_analogs", {"radius_mi": 2, "k": 8}),
                       ("fit_decline_curves", {}), ("run_transformer_forecast", {}),
                       ("build_ensemble", {}), ("finalize", {"narrative": "Test summary — done."})]
        self.messages = self

    def create(self, **kw):
        if not self.script:
            return NS(content=[NS(type="text", text="Finished.")])
        name, args = self.script.pop(0)
        return NS(content=[NS(type="tool_use", name=name, input=args, id=f"tu_{len(self.script)}")])


def test_llm_orchestrator_with_fake_client(field):
    w, p, truth, s = field
    ctx = run_llm(RunContext(w, p, s, Target(api=w.api.iloc[20])), client=FakeClient())
    assert "ensemble" in ctx.results
    assert ctx.narrative == "Test summary, done."
    refused = [r for r in ctx.log if "build_ensemble" in r["message"]]
    assert refused
