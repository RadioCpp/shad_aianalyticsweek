import numpy as np, pandas as pd, lightgbm as lgb, time, sys
from sim import *
al = pd.read_pickle('al2.pkl'); base = base_feats(al)
print('f_01 check', float(np.abs(base.f_01 - al.f_01).max()))
def make(cuts, seeds, hmax):
    fr = []
    for c in cuts:
        for s in seeds:
            rng = np.random.default_rng(1000 * c + s)
            ce, cg = pick_cold(al, c * D, 60 * D, rng)
            h = min(60, hmax - c)
            fr.append(scenario(al, base, c * D, h * D, ce, cg).assign(seed=s))
    return pd.concat(fr, ignore_index=True)
t = time.time()
trn = make([20, 30, 40, 50, 60, 75, 90], [0, 1], 120); print('train scen', trn.shape, time.time() - t)
val = make([120], [0, 1, 2, 3], 180); print('val scen', val.shape, val.cold.mean(), val.cold_e.mean(), val.cold_g.mean(), time.time() - t)
trn.to_pickle('trn.pkl'); val.to_pickle('val.pkl')
