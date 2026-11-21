import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
pd.set_option('display.width',250); pd.set_option('display.max_columns',60); pd.set_option('display.max_rows',400)
al=pd.read_pickle('al2.pkl'); b=pd.read_pickle('base.pkl'); D=86400
y=pd.read_pickle('yrec.pkl').y_rec.values
al['y']=np.where(al.is_te==1, np.where(y>=0,y,np.nan), al.target)
al['la']=np.log(al.f_16); al['r']=al.la-al.groupby('entity_id').la.transform('median')
al['night']=((al.f_23>=23)|(al.f_23<=5)).astype(int); al['day']=al.event_time/D
al['hi']=(al.r>0.75).astype(int); al['c1']=al.cat_01.isin([4,5]).astype(int); al['c0']=(al.cat_00==2).astype(int); al['cc']=al.c1*al.c0
d=al[al.y.notna()].copy(); d['b']=(d.day//15).astype(int)
tr_e=set(al[al.is_te==0].entity_id); tr_g=set(al[al.is_te==0].group_id)
d['cold']=((~d.entity_id.isin(tr_e))|(~d.group_id.isin(tr_g))).astype(int)
g=d.groupby('b')
print(pd.DataFrame({'n':g.size(),'rate':g.y.mean()*100,'auc_r':g.apply(lambda x: roc_auc_score(x.y,x.r)),'auc_f20':g.apply(lambda x: roc_auc_score(x.y,x.f_20)),
 'hi_rate':g.apply(lambda x:x[x.hi==1].y.mean()*100),'lo_rate':g.apply(lambda x:x[x.hi==0].y.mean()*100),
 'cc_rate':g.apply(lambda x:x[x.cc==1].y.mean()*100),'c1only':g.apply(lambda x:x[(x.c1==1)&(x.c0==0)].y.mean()*100),'c0only':g.apply(lambda x:x[(x.c1==0)&(x.c0==1)].y.mean()*100),'none':g.apply(lambda x:x[(x.c1==0)&(x.c0==0)].y.mean()*100),
 'night_lift':g.apply(lambda x:x[x.night==1].y.mean()/x[x.night==0].y.mean()),'new_lift':g.apply(lambda x:x[x.f_17==1].y.mean()/x[x.f_17==0].y.mean())}).round(3))
te=d[d.is_te==1]
print('test cold rate', te.groupby('cold').y.agg(['mean','size']))
print((te.pivot_table(index='cat_01',columns='cat_00',values='y',aggfunc='mean')*100).round(2))
for c in ['cat_02','cat_03','cat_04','f_23']:
    print(c, (te.groupby(c).y.mean()*100).round(2).values)
print(te.groupby(['cc','hi']).y.agg(['mean','size']).round(4))
print(te.groupby(['cold','hi']).y.agg(['mean','size']).round(4)); print(te.groupby(['cold','cc']).y.agg(['mean','size']).round(4)); print(te.groupby(['cold','f_17']).y.agg(['mean','size']).round(4)); print(te.groupby(['cold','night']).y.agg(['mean','size']).round(4))
