import sys, time
from hon import *
al = pd.read_pickle('al2.pkl'); base = pd.read_pickle('base.pkl')
yrec = pd.read_pickle('yrec.pkl').y_rec.values
# ---- A) train-only validation: fit on <=150, validate on (150,180] with simulated cold
T = 150
trnA = train_frame(al, base, T, [30, 60, 90, 105, 120, 135])
vals = []
for s in range(4):
    rng = np.random.default_rng(777 + s); ce, cg = pick(al, T * D, rng, 0.155, 0.035)
    vals.append(build(al, base, T * D, 180 * D, 180 * D, ce, cg, T * D))
# ---- B) real test check with reconstructed labels (analysis only)
trnB = train_frame(al, base, 180, [30, 60, 90, 120, 135, 150, 165])
te = build(al, base, 180 * D, 241 * D, 241 * D)
tre = set(al[al.is_te == 0].entity_id); trg = set(al[al.is_te == 0].group_id)
te['cold'] = ((~te.entity_id.isin(tre)) | (~te.group_id.isin(trg))).astype(int)
te['target'] = yrec[te.index.values] if False else yrec[(al.is_te == 1).values]
print(trnA.shape, trnB.shape, te.shape, (te.target >= 0).mean(), te.cold.mean())
import pickle; pickle.dump((trnA, vals, trnB, te), open('frames.pkl', 'wb'))
def evalA(feats, tau, rounds=400, params=P0, cold_w=1.0, name=''):
    m = fit(trnA, feats, T, tau, params, rounds, cold_w)
    r = pd.DataFrame([metric(v, m.predict(v[feats]), 150, 180) for v in vals]).mean()
    mB = fit(trnB, feats, 180, tau, params, rounds, cold_w); k = te.target >= 0
    rB = metric(te[k], mB.predict(te[k][feats]), 180, 240)
    print(f'{name:22s} tau={tau} | val150 ' + ' '.join(f'{k_}={v:.4f}' for k_, v in r.items()) + ' | TEST ' + ' '.join(f'{k_}={v:.4f}' for k_, v in rB.items()), flush=True)
    return mB
for tau in [None, 60, 30, 20, 12]:
    evalA(FEATS, tau, name='all feats')
