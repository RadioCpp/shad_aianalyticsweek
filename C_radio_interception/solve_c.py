#!/usr/bin/env python3
"""C. Radio Interception.  usage: python solve_c.py train_signals.csv test_signals.csv answers.csv
Invariant under letter permutation + reversal: the sequence of run lengths (up to reversal).
  station 0: geometric runs (P(run ends) ~ 0.52 at every step)
  station 1: every run has length 2 or 3
  station 2: every run has length 1 or 6..10
the message is cut at 28 chars, so the last run (first one after reversal) may be truncated.
Classifier = maximum likelihood over the three run-length grammars."""
import sys, numpy as np, pandas as pd

def runs(s):
    r = []; c = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1]: c += 1
        else: r.append(c); c = 1
    r.append(c); return r

def fit(tr):
    R = [runs(s) for s in tr.string]; y = tr.source.values
    inner = lambda k: np.array([x for r, t in zip(R, y) if t == k for x in r[1:-1]])
    a = inner(0); h = 1 / a.mean()                                  # geometric hazard of station 0
    b = inner(1); p2 = (b == 2).mean()                              # station 1: P(run = 2)
    c = inner(2); q = (c == 1).mean()                               # station 2: P(single)
    return dict(h=h, p2=p2, q=q, prior=np.bincount(y, minlength=3) / len(y))

def lik(r, P):
    h, p2, q = P['h'], P['p2'], P['q']
    def grammar(full, tail):                                         # complete runs then a possibly truncated last one
        tot = 0.0
        for o in (r, r[::-1]):
            v = tail(o[-1])
            for x in o[:-1]: v *= full(x)
            tot += 0.5 * v
        return tot
    L0 = grammar(lambda x: h * (1 - h) ** (x - 1), lambda x: (1 - h) ** (x - 1))
    f1 = lambda x: p2 if x == 2 else (1 - p2 if x == 3 else 0.0)
    L1 = grammar(f1, lambda x: 1.0 if x <= 2 else (1 - p2 if x == 3 else 0.0))
    f2 = lambda x: q if x == 1 else ((1 - q) / 5 if 6 <= x <= 10 else 0.0)
    L2 = grammar(f2, lambda x: 1.0 if x == 1 else ((1 - q) * min(1.0, (11 - x) / 5) if x <= 10 else 0.0))
    return np.array([L0, L1, L2]) * P['prior']

def predict(strings, P):
    return np.array([int(np.argmax(lik(runs(s), P))) for s in strings])

if __name__ == '__main__':
    tr = pd.read_csv(sys.argv[1]); te = pd.read_csv(sys.argv[2])
    P = fit(tr); print('params', {k: (np.round(v, 4).tolist() if hasattr(v, '__len__') else round(float(v), 4)) for k, v in P.items()})
    p = predict(tr.string, P); print('train accuracy', (p == tr.source.values).mean(), 'errors', int((p != tr.source.values).sum()))
    for s, a, b in zip(tr.string[p != tr.source.values], tr.source.values[p != tr.source.values], p[p != tr.source.values]): print('  true', a, 'pred', b, s, runs(s))
    pd.DataFrame({'string': te.string, 'source': predict(te.string, P)}).to_csv(sys.argv[3], index=False)
