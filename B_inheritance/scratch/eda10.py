import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
pd.set_option('display.width',250); pd.set_option('display.max_columns',60); pd.set_option('display.max_rows',400)
al=pd.read_pickle('al2.pkl'); b=pd.read_pickle('base.pkl'); D=86400
al['la']=np.log(al.f_16)
tr=al[al.is_te==0].copy()
auc=lambda x: round(roc_auc_score(tr.target, x.loc[tr.index]),4)
print('f_20',auc(al.f_20),'la',auc(al.la))
ge=al.groupby('entity_id').la
al['e_med']=ge.transform('median'); al['e_mean']=ge.transform('mean'); al['e_sd']=ge.transform('std')
print('la-e_med',auc(al.la-al.e_med),'la-e_mean',auc(al.la-al.e_mean),'z',auc((al.la-al.e_med)/al.e_sd.fillna(1)))
print('e_sd describe', al.groupby('entity_id').la.std().describe())
print('entity mean la describe', al.groupby('entity_id').la.mean().describe())
# does group / cats shift amount?
r=al.la-al.e_med
for c in ['group_id','cat_00','cat_01','cat_02','cat_03','cat_04','f_23','f_17']:
    m=r.groupby(al[c]).mean(); print(c,'between-level sd of mean residual', round(m.std(),4), 'n levels', len(m), 'expected noise sd', round(r.std()/np.sqrt(len(al)/len(m)),4))
al['r']=r
al['g_eff']=al.groupby('group_id').r.transform('mean')
print('r - g_eff', auc(al.r-al.g_eff))
# residual distribution by target
tr=al[al.is_te==0]
print(tr.groupby('target').r.describe())
print(tr.groupby(pd.cut(tr.r,[-9,-2,-1.5,-1,-0.5,-0.25,0,0.25,0.5,0.75,1,1.25,1.5,2,2.5,3,9])).target.agg(['mean','size']))
# entity persistence: split-half
h=tr.event_time<90*D
a=tr[h].groupby('entity_id').target.agg(['sum','count']); c=tr[~h].groupby('entity_id').target.agg(['sum','count'])
j=a.join(c,lsuffix='1',rsuffix='2',how='inner'); j=j[(j.count1>=20)&(j.count2>=20)]
print('entity split-half corr of rates', np.corrcoef(j.sum1/j.count1, j.sum2/j.count2)[0,1], len(j))
a=tr[h].groupby('group_id').target.agg(['sum','count']); c=tr[~h].groupby('group_id').target.agg(['sum','count'])
j=a.join(c,lsuffix='1',rsuffix='2',how='inner'); print('group split-half corr', np.corrcoef(j.sum1/j.count1, j.sum2/j.count2)[0,1], len(j))
# overdispersion of entity positive counts vs binomial
e=tr.groupby('entity_id').target.agg(['sum','count']); p=tr.target.mean()
print('entity var of sum', e['sum'].var(), 'binomial expected', (e['count']*p*(1-p)).mean()+ (p**2)*e['count'].var())
g=tr.groupby('group_id').target.agg(['sum','count']); print('group var of sum', g['sum'].var(), 'expected', (g['count']*p*(1-p)).mean()+(p**2)*g['count'].var())
# random odd/even split (non-temporal persistence)
tr=tr.assign(odd=(np.arange(len(tr))%2))
a=tr[tr.odd==0].groupby('entity_id').target.mean(); c=tr[tr.odd==1].groupby('entity_id').target.mean(); print('entity odd-even corr', a.corr(c))
a=tr[tr.odd==0].groupby('group_id').target.mean(); c=tr[tr.odd==1].groupby('group_id').target.mean(); print('group odd-even corr', a.corr(c))
a=tr[tr.odd==0].groupby('pair').target.agg(['sum','count']); c=tr[tr.odd==1].groupby('pair').target.agg(['sum','count']); j=a.join(c,lsuffix='1',rsuffix='2',how='inner'); j=j[(j.count1>=5)&(j.count2>=5)]; print('pair odd-even corr', np.corrcoef(j.sum1/j.count1, j.sum2/j.count2)[0,1], len(j))
