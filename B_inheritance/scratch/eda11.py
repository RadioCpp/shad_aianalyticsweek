import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
pd.set_option('display.width',250); pd.set_option('display.max_columns',60); pd.set_option('display.max_rows',400)
al=pd.read_pickle('al2.pkl'); b=pd.read_pickle('base.pkl'); D=86400
al['la']=np.log(al.f_16); al['r']=al.la-al.groupby('entity_id').la.transform('median')
al['night']=((al.f_23>=23)|(al.f_23<=5)).astype(int)
al['p_cnt']=b.p_cnt; al['day']=al.event_time/D
tr=al[al.is_te==0].copy()
tr['blk']=(tr.day//15).astype(int)
g=tr.groupby('blk')
print(pd.DataFrame({'rate':g.target.mean()*100,'auc_r':g.apply(lambda d: roc_auc_score(d.target,d.r)),'auc_f20':g.apply(lambda d: roc_auc_score(d.target,d.f_20)),'f_02':g.f_02.mean(),'night_lift':g.apply(lambda d: d[d.night==1].target.mean()/d[d.night==0].target.mean()),'new_lift':g.apply(lambda d: d[d.f_17==1].target.mean()/d[d.f_17==0].target.mean()),'hi_r_rate':g.apply(lambda d: d[d.r>0.75].target.mean()*100),'lo_r_rate':g.apply(lambda d: d[d.r<=0.75].target.mean()*100)}).round(3))
# f_02 by day (shape)
d=al.groupby((al.day//5).astype(int)).f_02.mean(); print(d.round(2).values)
# 2-way tables
tr['hi']=(tr.r>0.75).astype(int)
print(tr.groupby(['hi','night','f_17']).target.agg(['mean','size']).round(4))
# pos fraction by entity sd bucket and r
tr['e_sd']=al.groupby('entity_id').la.transform('std').loc[tr.index]
tr['sdb']=pd.qcut(tr.e_sd,4,labels=False)
print(tr.pivot_table(index=pd.cut(tr.r,[-9,0,0.5,0.75,1,1.5,9]),columns='sdb',values='target',aggfunc='mean').round(3))
# autocorrelation: prev entity event positive, by gap
tr=tr.sort_values(['entity_id','event_time'])
tr['prev_y']=tr.groupby('entity_id').target.shift(1); tr['gap']=(tr.event_time-tr.groupby('entity_id').event_time.shift(1))/D
print(tr.groupby([pd.cut(tr.gap,[0,0.25,1,2,5,10,60]),'prev_y']).target.agg(['mean','size']).round(4))
tr=tr.sort_values(['group_id','event_time'])
tr['prev_yg']=tr.groupby('group_id').target.shift(1); tr['gapg']=(tr.event_time-tr.groupby('group_id').event_time.shift(1))/D
print(tr.groupby([pd.cut(tr.gapg,[0,0.1,0.5,1,2,5,60]),'prev_yg']).target.agg(['mean','size']).round(4))
