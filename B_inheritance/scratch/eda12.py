import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
pd.set_option('display.width',250); pd.set_option('display.max_columns',60); pd.set_option('display.max_rows',400)
al=pd.read_pickle('al2.pkl'); b=pd.read_pickle('base.pkl'); D=86400
al['la']=np.log(al.f_16); al['r']=al.la-al.groupby('entity_id').la.transform('median')
al['night']=((al.f_23>=23)|(al.f_23<=5)).astype(int)
for c in ['p_cnt','p_dt','e_ngrp','e_cnt','e_nl','e_sl','g_nl','g_sl','g_nent']: al[c]=b[c]
al['day']=al.event_time/D
tr=al[al.is_te==0].copy(); tr['per']=(tr.day//30).astype(int)
cols=['r','f_20','f_16','f_19','f_00','f_14','f_09','night','f_17','f_01','f_07','f_06','f_10','f_08','f_12','f_04','f_03','f_15','f_13','f_02','cat_00','cat_01','cat_02','cat_03','cat_04','p_cnt','p_dt','e_ngrp','e_sl','g_sl','f_22','f_11','f_18']
res={}
for c in cols:
    res[c]=[roc_auc_score(d.target,d[c]) for _,d in tr.groupby('per')]
print(pd.DataFrame(res).T.round(3))
# categorical rate per period
for c in ['cat_00','cat_01','cat_02','cat_03','cat_04']:
    print(c); print((tr.pivot_table(index='per',columns=c,values='target',aggfunc='mean')*100).round(2))
