import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
U='/mnt/user-data/uploads/'
tr=pd.read_csv(U+'train.csv').sort_values('event_time').reset_index(drop=True); te=pd.read_csv(U+'9a51432a-test.csv')
pd.set_option('display.width',250); pd.set_option('display.max_columns',50); pd.set_option('display.max_rows',200)
print((tr.f_05-tr.event_time/86400).describe())
# f_18 analysis
print(tr.f_18.describe()); print(te.f_18.describe())
print(tr.groupby('target').f_18.describe())
print('train f_18 value counts top'); print(tr.f_18.value_counts().head(15))
print('test f_18 value counts top'); print(te.f_18.value_counts().head(15))
gm=tr.target.mean()
# candidate keys for target encoding
keys={'entity':['entity_id'],'group':['group_id'],'cat_00':['cat_00'],'cat_01':['cat_01'],'cat_02':['cat_02'],'cat_03':['cat_03'],'cat_04':['cat_04']}
for f in ['f_11','f_13','f_15','f_18','f_22']:
    for k,v in keys.items():
        g=tr.groupby(v)
        m=g.target.transform('mean'); n=g.target.transform('size')
        nun=g[f].transform('nunique')
        print(f,k,'corr with group mean',round(np.corrcoef(m,tr[f])[0,1],4),'nunique f within key (median)',nun.median(), 'test nunique within key median', te.groupby(v)[f].nunique().median())
