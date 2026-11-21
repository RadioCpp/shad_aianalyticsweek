import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
pd.set_option('display.width',250); pd.set_option('display.max_columns',60); pd.set_option('display.max_rows',400)
al=pd.read_pickle('al2.pkl'); b=pd.read_pickle('base.pkl'); D=86400
al['la']=np.log(al.f_16); al['r']=al.la-al.groupby('entity_id').la.transform('median')
al['night']=((al.f_23>=23)|(al.f_23<=5)).astype(int); al['day']=al.event_time/D
for c in ['p_cnt','p_dt','e_ngrp']: al[c]=b[c]
tr=al[al.is_te==0].copy(); tr['b10']=(tr.day//10).astype(int)
tr['hi']=(tr.r>0.75).astype(int); tr['c1']=tr.cat_01.isin([4,5]).astype(int); tr['c0']=(tr.cat_00==2).astype(int)
t=tr.pivot_table(index='b10',columns=['c1','c0'],values='target',aggfunc='mean')*100; print(t.round(2))
t=tr.pivot_table(index='b10',columns=['hi','c1'],values='target',aggfunc='mean')*100; print(t.round(2))
print(tr.pivot_table(index='b10',columns=['c1','c0'],values='target',aggfunc='size'))
late=tr[tr.day>=150]
print(late.groupby(['c1','c0','hi']).target.agg(['mean','size']).round(4))
print(late.groupby(['c1','c0','night']).target.agg(['mean','size']).round(4))
print(late.groupby(['c1','c0','f_17']).target.agg(['mean','size']).round(4))
print((late.pivot_table(index='cat_01',columns='cat_00',values='target',aggfunc='mean')*100).round(2))
print(late.pivot_table(index='cat_01',columns='cat_00',values='target',aggfunc='size'))
# test distribution of cat_00/cat_01 over time
te=al[al.is_te==1]
print('share cat_01 in{4,5} by 20d', al.groupby((al.day//20).astype(int)).cat_01.apply(lambda s:s.isin([4,5]).mean()).round(3).values)
print('share cat_00==2 by 20d', al.groupby((al.day//20).astype(int)).cat_00.apply(lambda s:(s==2).mean()).round(3).values)
