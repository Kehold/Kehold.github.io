"""Run context shared by every agent: inputs, intermediate results, log and QC flags."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd


@dataclass
class ProposedDesign:
    lateral_ft: float = 10_000
    proppant_lbft: float = 2_400
    fluid_bblft: float = 48
    spacing_ft: float = 880
    azimuth: float = 0.0


@dataclass
class Target:
    api: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    zone: str = "auto"
    design: ProposedDesign = field(default_factory=ProposedDesign)

    @property
    def mode(self) -> str:
        return "existing" if self.api else "proposed"


@dataclass
class RunContext:
    wells: pd.DataFrame
    prod: pd.DataFrame
    surfaces: Any
    target: Target
    seed: int = 7
    results: dict = field(default_factory=dict)
    log: list = field(default_factory=list)
    qc: list = field(default_factory=list)
    narrative: str = ""
    orchestrator: str = "deterministic"

    def emit(self, agent: str, message: str, **data):
        self.log.append(dict(t=time.strftime("%H:%M:%S"), agent=agent,
                             message=message, data=data or None))

    def flag(self, agent: str, level: str, message: str):
        """level: info | warn | stop"""
        self.qc.append(dict(agent=agent, level=level, message=message))
