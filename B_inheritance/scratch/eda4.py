import pandas as pd, numpy as np, itertools
U='/mnt/user-data/uploads/'
tr=pd.read_csv(U+'train.csv'); te=pd.read_csv(U+'9a51432a-test.csv')
pd.set_option('display.width',280); pd.set_option('display.max_columns',60); pd.set_option('display.max_rows',300)
tr['is_te']=0; te['is_te']=1; te['target']=np.nan
al=pd.concat([tr,te]).sort_values(['event_time','row_id']).reset_index(drop=True)
gm=tr.target.mean()
# functional dependencies
for a in ['entity_id','group_id']:
    for c in ['cat_00','cat_01','cat_02','cat_03','cat_04','group_id','entity_id']:
        if a!=c: print(a,'->',c, al.groupby(a)[c].nunique().describe()[['mean','max']].values)
al['e_cnt']=al.groupby('entity_id').cumcount()
al['g_cnt']=al.groupby('group_id').cumcount()
al['eg_cnt']=al.groupby(['entity_id','group_id']).cumcount()
al['t0']=al.target.fillna(0)
al['e_sum']=al.groupby('entity_id').t0.cumsum()-al.t0
al['g_sum']=al.groupby('group_id').t0.cumsum()-al.t0
al['e_first']=al.groupby('entity_id').event_time.transform('min')
al['e_prev']=al.groupby('entity_id').event_time.shift(1)
al['g_prev']=al.groupby('group_id').event_time.shift(1)
al['eg_prev']=al.groupby(['entity_id','group_id']).event_time.shift(1)
al['e_dt']=(al.event_time-al.e_prev); al['g_dt']=al.event_time-al.g_prev; al['eg_dt']=al.event_time-al.eg_prev
al['e_age']=(al.event_time-al.e_first)/86400
al['e_ngroups']=al.groupby('entity_id').group_id.transform(lambda s: (~s.duplicated()).cumsum())
ent=al[al.entity_id==al.entity_id.value_counts().index[50]]
cols=['event_time','is_te','group_id','target','e_cnt','eg_cnt','e_sum','e_dt','eg_dt','e_age','e_ngroups','f_00','f_01','f_03','f_04','f_05','f_06','f_07','f_08','f_10','f_12','f_13','f_15','f_17','f_21','f_23','f_02']
print(ent[cols].head(40).to_string())
print(ent[cols].tail(25).to_string())
al.to_pickle('al.pkl')
