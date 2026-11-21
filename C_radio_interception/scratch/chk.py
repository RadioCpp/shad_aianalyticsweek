import pandas as pd, numpy as np, lightgbm as lgb
from f import X, Xt, y, te, runs
a=pd.read_csv('/home/claude/work/out/answers.csv'); ex=pd.read_csv('/mnt/user-data/uploads/c2f5399a-example.csv')
print(a.shape, list(a.columns), (a.string.values==te.string.values).all(), (a.string.values==ex.string.values).all(), a.source.value_counts().sort_index().to_dict(), a.source.dtype)
m=lgb.LGBMClassifier(n_estimators=400,learning_rate=0.05,num_leaves=15,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,verbose=-1,n_jobs=2).fit(X,y)
g=m.predict(Xt); d=np.where(g!=a.source.values)[0]; print('disagreement with GBM', len(d))
for i in d: print(' rule', a.source[i], 'gbm', g[i], te.string[i], runs(te.string[i]))
# test strings that fit grammar of station 2 but would be plausible station 0, and strings fitting no strict grammar
R=[runs(s) for s in te.string]
fit2=lambda r: any(all(x==1 or 6<=x<=10 for x in o[:-1]) and o[-1]<=10 for o in (r,r[::-1]))
fit1=lambda r: any(all(x in(2,3) for x in o[:-1]) and o[-1]<=3 for o in (r,r[::-1]))
f1=np.array([fit1(r) for r in R]); f2=np.array([fit2(r) for r in R])
print('fit1',f1.sum(),'fit2',f2.sum(),'both',(f1&f2).sum(),'pred counts within', pd.crosstab(a.source, [f1,f2]))
k=[i for i in range(len(R)) if f2[i]]; ml=sorted(k,key=lambda i: max(R[i]))[:8]
for i in ml: print(' weakest station-2 candidates:', te.string[i], R[i], 'pred', a.source[i])
