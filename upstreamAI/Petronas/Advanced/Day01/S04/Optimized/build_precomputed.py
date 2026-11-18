"""Recreate deployment results using: python build_precomputed.py
Run offline, then commit precomputed/*.npz with the Streamlit app.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import time
import numpy as np
from pgm import models as M
from pgm.data import build
from pgm.precomputed import RESULTS, fingerprint

_, st, _, _ = build()
RESULTS.mkdir(parents=True, exist_ok=True)
fpr = fingerprint(st)
print(f'Generated {len(st)} stages, fingerprint {fpr[:16]}', flush=True)
for fi, (fs, feats) in enumerate(M.FEATURE_SETS.items()):
    for mi, model in enumerate(M.MODELS):
        t = time.perf_counter()
        pred_file = RESULTS / f'oof_f{fi}_m{mi}.npz'
        if pred_file.exists():
            print('Already built:', pred_file.name, flush=True)
        else:
            p = M.lowo_proba(st, feats, model)
            np.savez_compressed(pred_file, probabilities=p, fingerprint=fpr)
        print(f'OOF: {model}, {fs}: {time.perf_counter()-t:.1f}s', flush=True)
        t = time.perf_counter()
        imp_file = RESULTS / f'imp_f{fi}_m{mi}.npz'
        if imp_file.exists():
            print('Already built:', imp_file.name, flush=True)
        else:
            imp = M.importance(st, feats, model).set_index('feature').loc[feats, 'importance'].to_numpy()
            np.savez_compressed(imp_file, features=np.array(feats), importance=imp, fingerprint=fpr)
        print(f'Importance: {model}, {fs}: {time.perf_counter()-t:.1f}s', flush=True)
print('Wrote', len(list(RESULTS.glob('*.npz'))), 'result files')
