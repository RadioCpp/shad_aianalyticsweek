#!/usr/bin/env python3
"""Reproduces every number quoted in README.md of task C.

Usage:  python3 evidence.py data/train_signals.csv data/test_signals.csv answers.csv probes > evidence_output.txt
"""
import collections, io, sys, warnings, zipfile
import numpy as np, pandas as pd
from solve_c import runs, fit, predict

warnings.filterwarnings('ignore')
pd.set_option('display.width', 200)
def head(t): print('\n' + '=' * 10 + ' ' + t, flush=True)

tr = pd.read_csv(sys.argv[1]); te = pd.read_csv(sys.argv[2]); ans = pd.read_csv(sys.argv[3]); probes_dir = sys.argv[4]
R = [runs(s) for s in tr.string]; Rt = [runs(s) for s in te.string]; y = tr.source.values

head('1. Data overview')
print(f'train {len(tr)} strings, test {len(te)}; lengths {sorted(set(map(len, tr.string)) | set(map(len, te.string)))}; alphabet {"".join(sorted(set("".join(tr.string))))}')
print('train class counts:', dict(collections.Counter(y)), '| duplicate strings in train:', int(tr.string.duplicated().sum()), '| test strings present in train:', int(te.string.isin(tr.string).sum()))
for s in (0, 1, 2): print(f'station {s} examples:', tr[tr.source == s].string.head(3).tolist())

head('2. Raw characters carry nothing: letter frequencies per station, %')
print(pd.DataFrame({s: pd.Series(collections.Counter(''.join(tr[tr.source == s].string))) for s in (0, 1, 2)}).pipe(lambda d: (100 * d / d.sum()).round(1)).T.to_string())

head('3. Run lengths per station (inner runs = all except the first and the last one)')
for s in (0, 1, 2):
    inner = collections.Counter(x for r, t in zip(R, y) if t == s for x in r[1:-1]); edge = collections.Counter(x for r, t in zip(R, y) if t == s for x in (r[0], r[-1]))
    print(f'station {s} inner:', dict(sorted(inner.items()))); print(f'station {s} edge: ', dict(sorted(edge.items())))
c = collections.Counter(x for r, t in zip(R, y) if t == 0 for x in r[1:-1]); tot = sum(c.values()); cum = 0; hz = {}
for k in sorted(c)[:7]: hz[k] = round(c[k] / (tot - cum), 3); cum += c[k]
print('station 0: P(run ends | it reached length k):', hz)
print('station 2: number of runs with length 2..5 per string:', dict(sorted(collections.Counter(sum(2 <= x <= 5 for x in r) for r, t in zip(R, y) if t == 2).items())))
aba = {}
for s in (0, 1, 2):
    a = b = 0
    for z in tr[tr.source == s].string:
        sym = [z[0]] + [z[i] for i in range(1, len(z)) if z[i] != z[i - 1]]; a += sum(sym[i] == sym[i + 2] for i in range(len(sym) - 2)); b += len(sym) - 2
    aba[s] = round(a / b, 3)
print('P(run symbol equals the symbol two runs earlier):', aba, '(uniform choice among the 5 other letters gives 0.2)')

head('4. Strict grammars (complete runs + possibly truncated last run, either orientation)')
fit1 = lambda r: any(all(x in (2, 3) for x in o[:-1]) and o[-1] <= 3 for o in (r, r[::-1]))
fit2 = lambda r: any(all(x == 1 or 6 <= x <= 10 for x in o[:-1]) and o[-1] <= 10 for o in (r, r[::-1]))
g = pd.DataFrame({'station': y, 'fits_grammar_1': [fit1(r) for r in R], 'fits_grammar_2': [fit2(r) for r in R]})
print('train, share of strings fitting each grammar:'); print(g.groupby('station').mean().round(4).to_string())
print('train, counts:'); print(g.groupby('station')[['fits_grammar_1', 'fits_grammar_2']].sum().to_string())

head('5. Maximum-likelihood classifier')
P = fit(tr); pr = predict(tr.string, P)
print('fitted parameters:', {k: (np.round(v, 4).tolist() if hasattr(v, '__len__') else round(float(v), 4)) for k, v in P.items()})
print(f'train accuracy {np.mean(pr == y):.6f}'); print(pd.crosstab(pd.Series(y, name='true'), pd.Series(pr, name='pred')).to_string())
for s_, a, b in zip(tr.string[pr != y], y[pr != y], pr[pr != y]): print('  train error: true', a, 'pred', b, s_, runs(s_))
pt = predict(te.string, P)
print('answers.csv identical to a fresh prediction:', bool((pt == ans.source.values).all()), '| predicted class counts in test:', dict(sorted(collections.Counter(pt).items())))
gt = pd.DataFrame({'pred': pt, 'g1': [fit1(r) for r in Rt], 'g2': [fit2(r) for r in Rt]})
print('test: predicted class vs grammar fit (counts):'); print(gt.groupby('pred')[['g1', 'g2']].sum().to_string())

head('6. How often would a station-0 string imitate another station? (simulation, geometric runs with the fitted hazard)')
rng = np.random.default_rng(0); N = 400000; c1 = c2 = 0
for _ in range(N):
    r = []; tot = 0
    while tot < 28:
        k = min(int(rng.geometric(P['h'])), 28 - tot); r.append(k); tot += k
    c1 += fit1(r); c2 += fit2(r)
n0 = int((pt == 0).sum())
print(f'P(fits grammar 1) = {c1/N:.2e}, P(fits grammar 2) = {c2/N:.2e}; expected among ~{n0} station-0 test strings: {n0*c1/N:.3f} and {n0*c2/N:.3f}')
print(f'observed in train: {int(g[g.station == 0].fits_grammar_1.sum())} and {int(g[g.station == 0].fits_grammar_2.sum())} of {int((y == 0).sum())}')

head('7. Independent cross-check: gradient boosting on invariant features')
try:
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    def feats(s):
        n = len(s); r = runs(s); cnt = sorted([s.count(ch) for ch in 'ABCDEF'], reverse=True)
        lag = [sum(s[i] == s[i + k] for i in range(n - k)) for k in range(1, 28)]
        hist = [sum(x == k for x in r) for k in range(1, 9)] + [sum(x >= 9 for x in r)]
        sr = (sorted(r, reverse=True) + [0] * 6)[:6]
        return cnt + lag + hist + sr + [min(r[0], r[-1]), max(r[0], r[-1]), len(set(s)), len(r), float(np.mean(r)), float(np.std(r))]
    X = np.array([feats(s) for s in tr.string], float); Xt = np.array([feats(s) for s in te.string], float)
    mk = lambda: lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=15, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, verbose=-1, n_jobs=2, random_state=0)
    cv = cross_val_predict(mk(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
    print(f'LightGBM 5-fold CV accuracy on train: {np.mean(cv == y):.4f}; confusion:'); print(pd.crosstab(pd.Series(y, name='true'), pd.Series(cv, name='pred')).to_string())
    for i in np.where(cv != y)[0]: print('  CV error: true', y[i], 'pred', cv[i], tr.string[i], R[i], '| strict grammar 2 fit:', fit2(R[i]))
    gb = mk().fit(X, y).predict(Xt); dis = np.where(gb != pt)[0]
    print('test strings where LightGBM disagrees with answers.csv:', len(dis))
    for i in dis: print('  answers', pt[i], 'gbm', gb[i], te.string[i], Rt[i])
except ImportError:
    print('lightgbm is not installed, section skipped')

head('8. Scoring formula in float64')
score = lambda acc: int(100 * max(5 * (acc - 0.8), 0))
print('accuracy = 1.0 ->', score(1.0), '(raw value', repr(100 * max(5 * (1.0 - 0.8), 0)) + ')')
print('score by number of wrong answers out of 3500:', {e: score((3500 - e) / 3500) for e in list(range(0, 9)) + [13, 14, 15]})

head('9. Probe submissions: which truth is consistent with all 20 observed scores?')
obs = {'round1_probes.zip': [16] * 7 + [25] * 3 + [80], 'round2_probes.zip': [99, 99, 99, 17, 17, 15, 16, 16, 16]}
base = ans.source.values; PR = []; OB = []
for zname, sc in obs.items():
    z = zipfile.ZipFile(f'{probes_dir}/{zname}'); names = sorted(n for n in z.namelist() if n.endswith('answers.csv'))
    for nme, o in zip(names, sc):
        p = pd.read_csv(io.BytesIO(z.read(nme))); assert (p.string.values == ans.string.values).all()
        PR.append(p.source.values); OB.append(o)
PR = np.array(PR); OB = np.array(OB); k = (PR != base).sum(1)
sc_by_e = np.array([score((3500 - e) / 3500) for e in range(3501)])
print('probe sizes (answers changed):', k.tolist()); print('observed scores:            ', OB.tolist())
print('predicted if answers.csv has 0 errors:', sc_by_e[k].tolist(), '-> all match:', bool((sc_by_e[k] == OB).all()))
lo = np.array([np.where(sc_by_e == o)[0].min() for o in OB]); hi = np.array([np.where(sc_by_e == o)[0].max() for o in OB])
# hypotheses "string i is wrong and its true label is L": change of the error count in every probe
H = [];
for L in (0, 1, 2):
    m = base != L
    d = (PR[:, m] != L).astype(np.int8) - (PR[:, m] != base[m]).astype(np.int8)       # probes x strings
    H.append(d.T)
H = np.vstack(H); print('single-error hypotheses:', len(H))
ok1 = ((k + H >= lo) & (k + H <= hi)).all(1); print('  consistent with all 20 scores (and base score 99):', int(ok1.sum()))
U, cnts = np.unique(H, axis=0, return_counts=True); print('distinct probe signatures among them:', len(U))
tot = 0
for i in range(len(U)):
    e = k + U[i] + U[i:]                                                               # pairs (i, j>=i)
    good = ((e >= lo) & (e <= hi)).all(1)
    tot += int((cnts[i] * cnts[i:][good]).sum())
print('  two-error hypotheses (pairs of signatures, with multiplicity) consistent with all 20 scores:', tot)
