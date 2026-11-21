import pandas as pd, numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
pd.set_option('display.width',280); pd.set_option('display.max_columns',60); pd.set_option('display.max_rows',300)
al=pd.read_pickle('al.pkl')
tr=al[al.is_te==0]; te=al[al.is_te==1]
late=tr[tr.event_time>tr.event_time.quantile(0.67)]
fs=[f'f_{i:02d}' for i in range(24)]
print('KS late-train vs test, and late-train(target0) vs test')
for f in fs:
    print(f, round(ks_2samp(late[f],te[f]).statistic,4), round(ks_2samp(late[late.target==0][f],te[f]).statistic,4), round(ks_2samp(tr[f],late[f]).statistic,4))
# f_20 vs f_16
al['r']=al.f_16/al.f_20
print(al[['f_16','f_20','r','f_19','f_14','f_00','f_09']].head(20))
print('r nunique', al.r.round(6).nunique(), 'per entity nunique of r', al.groupby('entity_id').r.apply(lambda s:s.round(5).nunique()).describe())
print('per group', al.groupby('group_id').r.apply(lambda s:s.round(5).nunique()).describe())
ent=al[al.entity_id==al.entity_id.value_counts().index[50]]
print(ent[['event_time','is_te','group_id','target','f_16','f_20','r','f_19','f_14','f_00','f_09','f_11','f_22','f_18']].head(30).to_string())
# decode others
al['g_first']=al.groupby('group_id').event_time.transform('min')
print('f_05 = group age?', np.abs(al.f_05-(al.event_time-al.g_first)/86400).max())
print('f_07 = g_dt?', np.mean(np.abs(al.f_07-al.g_dt.fillna(0))<1e-6), 'eg_dt', np.mean(np.abs(al.f_07-al.eg_dt.fillna(0))<1e-6))
print('f_01=e_dt', np.mean(np.abs(al.f_01-al.e_dt.fillna(0))<1e-6), 'f_21=e_age', np.abs(al.f_21-al.e_age).max(), 'f_17', np.mean(al.f_17==(al.eg_cnt==0)), 'f_23 hour', np.mean(al.f_23==(al.event_time%86400)//3600))
