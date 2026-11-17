"""Export the public case-study datasets to data/ (same content as the app's Brief-tab downloads, plus the wells table).
Hidden truth columns (leading underscore) and landing zones are never exported. Deterministic: same seed, same files."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geomech_core as gc

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(out, exist_ok=True)
logs, stages = gc.make_field()
meas, core, dfit = gc.make_measurements(logs)
files = {
    "permian_analog_lateral_logs.csv": meas[gc.PUBLIC_LOGS],
    "permian_analog_stages.csv": stages[[c for c in stages if not c.startswith("_")]],
    "permian_analog_core_tests.csv": core,
    "permian_analog_dfit.csv": dfit,
    "permian_analog_wells.csv": gc.WELLS.drop(columns=["LANDING"]),
}
for name, df in files.items():
    assert not any(c.startswith("_") for c in df.columns)
    df.to_csv(os.path.join(out, name), index=False)
    print(f"{name}: {len(df)} rows, {df.shape[1]} columns")
