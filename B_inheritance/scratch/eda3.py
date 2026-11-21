import pandas as pd, numpy as np, itertools
from sklearn.metrics import roc_auc_score
U='/mnt/user-data/uploads/'
tr=pd.read_csv(U+'train.csv').sort_values('event_time').reset_index(drop=True); te=pd.read_csv(U+'9a51432a-test.csv')
pd.set_option('display.width',280); pd.set_option('display.max_columns',50); pd.set_option('display.max_rows',200)
gm=tr.target.mean()
fs=[f'f_{i:02d}' for i in range(24)]
sp=tr[fs+['event_time','target']].rank().corr()
print(sp.round(2).to_string())
# find key of f_18
cands=['entity_id','group_id','cat_00','cat_01','cat_02','cat_03','cat_04']
tr['day']=(tr.event_time//86400)
best=[]
for r in (1,2,3):
    for ks in itertools.combinations(cands+['day'],r):
        g=tr.groupby(list(ks)).target
        est=(g.transform('sum')+1.25*gm)/(g.transform('size')+1.25)
        best.append((np.mean(np.abs(est-tr.f_18)<1e-9), ks))
best.sort(reverse=True); print(best[:8])
# f_22 smoothing
g=tr.groupby('entity_id').target; s=g.transform('sum'); n=g.transform('size')
for a in [1,5,10,20,50]:
    print('f_22 a',a, np.mean(np.abs((s+a*gm)/(n+a)-tr.f_22)<1e-9))
g=tr.groupby('group_id').target; s=g.transform('sum'); n=g.transform('size')
for a in [0,1,5,10,20,50,100]:
    print('f_11 a',a, np.mean(np.abs((s+a*gm)/(n+a)-tr.f_11)<1e-9))
