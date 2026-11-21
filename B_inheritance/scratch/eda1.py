import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
U='/mnt/user-data/uploads/'
tr=pd.read_csv(U+'train.csv'); te=pd.read_csv(U+'9a51432a-test.csv')
pd.set_option('display.width',250); pd.set_option('display.max_columns',50); pd.set_option('display.max_rows',200)
print(tr.shape, te.shape, tr.target.mean())
print('time', tr.event_time.min(), tr.event_time.max(), te.event_time.min(), te.event_time.max())
print('row_id', tr.row_id.min(), tr.row_id.max(), te.row_id.min(), te.row_id.max())
print('rowid-time corr', np.corrcoef(tr.row_id.rank(), tr.event_time.rank())[0,1])
tr=tr.sort_values('event_time').reset_index(drop=True)
print('row_id monotone in time:', (tr.row_id.diff().dropna()>0).mean())
for c in ['entity_id','group_id','cat_00','cat_01','cat_02','cat_03','cat_04']:
    print(c, tr[c].nunique(), te[c].nunique(), 'test unseen frac', (~te[c].isin(tr[c])).mean())
cold=(~te.entity_id.isin(tr.entity_id))|(~te.group_id.isin(tr.group_id))
print('cold frac test', cold.mean())
# target by time decile
tr['tb']=pd.qcut(tr.event_time,10,labels=False)
print(tr.groupby('tb').target.agg(['mean','size']))
feats=[c for c in te.columns if c!='row_id']
rows=[]
for c in feats:
    a=roc_auc_score(tr.target, tr[c].fillna(-999))
    # auc within early and late
    e=tr[tr.tb<5]; l=tr[tr.tb>=5]
    rows.append((c, tr[c].nunique(), tr[c].isna().mean(), te[c].isna().mean(), a, roc_auc_score(e.target,e[c].fillna(-999)), roc_auc_score(l.target,l[c].fillna(-999)), np.corrcoef(tr[c].fillna(0).rank(), tr.event_time.rank())[0,1], tr[c].mean(), te[c].mean(), tr[c].std(), te[c].std()))
print(pd.DataFrame(rows, columns=['c','nuniq','na_tr','na_te','auc','auc_early','auc_late','corr_time','mean_tr','mean_te','std_tr','std_te']).to_string())
