import pandas as pd, numpy as np, lightgbm as lgb, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold, cross_val_predict
U='/mnt/user-data/uploads/'
tr=pd.read_csv(U+'60ebf667-train_signals.csv'); te=pd.read_csv(U+'fd7c4ad3-test_signals.csv')
def runs(s):
    r=[];c=1
    for i in range(1,len(s)):
        if s[i]==s[i-1]: c+=1
        else: r.append(c); c=1
    r.append(c); return r
def feats(s):
    n=len(s); cnt=sorted([s.count(c) for c in 'ABCDEF'],reverse=True)
    lag=[sum(s[i]==s[i+k] for i in range(n-k)) for k in range(1,28)]
    r=runs(s); inner=r[1:-1]
    hist=[sum(x==k for x in r) for k in range(1,9)]+[sum(x>=9 for x in r)]
    ih=[sum(x==k for x in inner) for k in range(1,5)]
    sr=sorted(r,reverse=True)+[0]*28
    edge=[min(r[0],r[-1]),max(r[0],r[-1])]
    # run-level symbol sequence: distinct symbols, returns (ABA patterns), pair structure
    sym=[s[0]]+[s[i] for i in range(1,n) if s[i]!=s[i-1]]
    aba=sum(sym[i]==sym[i+2] for i in range(len(sym)-2)); abca=sum(sym[i]==sym[i+3] for i in range(len(sym)-3))
    big=set(frozenset(p) for p in zip(sym,sym[1:])); dbig=set(zip(sym,sym[1:]))
    # longest stretch of singletons, number of singles adjacent to long run
    ls=0;c=0
    for x in r:
        c=c+1 if x==1 else 0; ls=max(ls,c)
    return cnt+lag+hist+ih+sr[:6]+edge+[len(set(s)),len(r),aba,abca,len(big),len(dbig),ls,np.mean(r),np.std(r)]
X=np.array([feats(s) for s in tr.string],float); Xt=np.array([feats(s) for s in te.string],float); y=tr.source.values
if __name__=='__main__':
    m=lgb.LGBMClassifier(n_estimators=400,learning_rate=0.05,num_leaves=15,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,verbose=-1,n_jobs=2)
    p=cross_val_predict(m,X,y,cv=StratifiedKFold(5,shuffle=True,random_state=0),method='predict_proba'); pr=p.argmax(1)
    print('CV acc',(pr==y).mean()); print(pd.crosstab(y,pr))
    bad=np.where(pr!=y)[0]
    for i in bad[:40]: print(y[i],pr[i],tr.string[i],runs(tr.string[i]),p[i].round(2))
    # station 1 rule check
    for s in (0,1,2):
        d=tr[tr.source==s].string; ok=d.apply(lambda z: all(x in(2,3) for x in runs(z)[1:-1]) and max(runs(z))<=3)
        print(s,'all inner runs in {2,3}',ok.mean(), 'maxrun dist', d.apply(lambda z:max(runs(z))).value_counts().sort_index().to_dict())
