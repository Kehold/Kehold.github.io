"""Shared constants, brand palette and unit conventions."""

APP_NAME = "Production Intelligence Agents"
APP_TAGLINE = "Digital Oil · Permian Basin · five coordinated agents"

# Digital Oil brand palette
NAVY = "#0A2E36"
TEAL = "#14808C"
AMBER = "#E8A317"
SLATE = "#5B6B73"
LIGHT = "#F4F7F8"
RED = "#B5413A"
STREAM_COLOURS = {"oil": "#2E7D32", "gas": "#B5413A", "water": "#1F5FA8"}

# Synthetic field anchor (Midland Basin, Ector/Midland county line)
FIELD_LAT = 31.86
FIELD_LON = -102.13
FT_PER_DEG_LAT = 364_000.0

# Reporting cut-off for the synthetic production history
DATA_END = "2026-09"

ZONES = ["LSS", "WCA", "WCB"]
ZONE_NAMES = {
    "LSS": "Lower Spraberry Shale",
    "WCA": "Wolfcamp A",
    "WCB": "Wolfcamp B",
}

# Decline curve conventions
DMIN_EFF_ANNUAL = 0.08            # terminal effective annual decline
EUR_YEARS = 30                    # EUR horizon from first production
ECON_LIMIT_BOPD = 5.0             # oil economic limit
FORECAST_MONTHS = 120             # displayed forecast horizon

STREAMS = ["oil", "gas", "water"]
STREAM_UNITS = {"oil": "bbl/d", "gas": "mcf/d", "water": "bbl/d"}
STREAM_VOL_UNITS = {"oil": "Mbbl", "gas": "MMcf", "water": "Mbbl"}

# Petroleum convention: P90 is the low case, P10 the high case
PCTS = {"P90": 10, "P50": 50, "P10": 90}

DEFAULT_MODEL = "claude-sonnet-5-5"
MODEL_CHOICES = ["claude-sonnet-5-5", "claude-opus-5-5", "claude-haiku-5-5"]
