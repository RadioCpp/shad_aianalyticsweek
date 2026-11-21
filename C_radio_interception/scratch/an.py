import numpy as np, pandas as pd, lightgbm as lgb, collections
from f import X, Xt, y, te, tr, runs
base = pd.read_csv('/home/claude/work/out/answers.csv'); pred = base.source.values
# 1) exact-boundary float behaviour of the scoring formula for the probe sizes
for m in (83, 74, 19, 1):
    for e in (7*m-1, 7*m, 7*m+1):
        acc = float(np.mean(np.arange(3500) >= e)); print('m',m,'errors',e,'score',int(100*max(5*(acc-0.8),0)), repr(100*max(5*(acc-0.8),0)))
# 2) how often does a station-0-like string satisfy grammar 1 / grammar 2 (simulation with fitted hazard)
rng = np.random.default_rng(0); h = 0.5196
def sim0():
    r=[];tot=0
    while tot<28:
        k=rng.geometric(h); k=min(k,28-tot); r.append(k); tot+=k
    return r
fit2=lambda r: any(all(x==1 or 6<=x<=10 for x in o[:-1]) and o[-1]<=10 for o in (r,r[::-1]))
fit1=lambda r: any(all(x in(2,3) for x in o[:-1]) and o[-1]<=3 for o in (r,r[::-1]))
N=400000; c1=c2=0
for _ in range(N):
    r=sim0(); c1+=fit1(r); c2+=fit2(r)
print('P(station0 fits grammar1)',c1/N,'P(fits grammar2)',c2/N,'expected in test (1170 st0):',1170*c1/N,1170*c2/N)
# 3) near misses among answers "0": number of runs violating each grammar (best orientation)
def viol(r, okfull, oktail):
    return min(sum(not okfull(x) for x in o[:-1]) + (not oktail(o[-1])) for o in (r, r[::-1]))
v1=lambda r: viol(r, lambda x: x in(2,3), lambda x: x<=3)
v2=lambda r: viol(r, lambda x: x==1 or 6<=x<=10, lambda x: x<=10)
R=[runs(s) for s in te.string]; Rt=[runs(s) for s in tr.string]
z=np.where(pred==0)[0]
print('test answers 0: violations of grammar1', collections.Counter(v1(R[i]) for i in z).most_common(6))
print('test answers 0: violations of grammar2', sorted(collections.Counter(v2(R[i]) for i in z).items())[:6])
t0=np.where(tr.source.values==0)[0]
print('train station0: violations of grammar1', sorted(collections.Counter(v1(Rt[i]) for i in t0).items())[:4], 'grammar2', sorted(collections.Counter(v2(Rt[i]) for i in t0).items())[:6])
m=lgb.LGBMClassifier(n_estimators=400,learning_rate=0.05,num_leaves=15,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,verbose=-1,n_jobs=2).fit(X,y); P=m.predict_proba(Xt)
print('--- answers 0 with <=1 violation of grammar 2 or 1')
for i in z:
    a,b=v1(R[i]),v2(R[i])
    if a<=1 or b<=1: print(i, te.string[i], R[i], 'v1',a,'v2',b,'gbm',P[i].round(3))
print('--- answers 1/2 where GBM is least sure')
o=np.argsort(P[np.arange(len(pred)),pred])[:12]
for i in o: print(i, pred[i], te.string[i], R[i], P[i].round(3))
