"""Bundled, reproducible held-out results for fast Streamlit Cloud deployment."""
from pathlib import Path
import hashlib
import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / 'precomputed'


def fingerprint(st):
    """Checks labels and all model features against bundled training data."""
    from .models import FEATURE_SETS
    feats = list(dict.fromkeys(f for group in FEATURE_SETS.values() for f in group))
    arr = np.round(st[feats + ['EFFECTIVE']].to_numpy(dtype=np.float64), 6)
    h = hashlib.sha256()
    h.update(arr.tobytes())
    h.update('\x1f'.join(st.WELL.astype(str)).encode())
    return h.hexdigest()


def load(st, fs, model, kind='oof'):
    from .models import FEATURE_SETS, MODELS
    path = RESULTS / f'{kind}_f{list(FEATURE_SETS).index(fs)}_m{MODELS.index(model)}.npz'
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        if str(z['fingerprint']) != fingerprint(st):
            return None
        if kind == 'oof':
            p = z['probabilities']
            return p if len(p) == len(st) else None
        import pandas as pd
        expected = FEATURE_SETS[fs]
        if list(z['features'].astype(str)) != expected:
            return None
        return pd.DataFrame({'feature': expected, 'importance': z['importance']}).sort_values('importance', ascending=False)
