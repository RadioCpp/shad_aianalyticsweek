import pandas as pd, numpy as np
pd.set_option('display.width',250); pd.set_option('display.max_columns',60); pd.set_option('display.max_rows',400)
al=pd.read_pickle('al2.pkl'); b=pd.read_pickle('base.pkl'); D=86400
tr=al[al.is_te==0].copy(); bt=b.loc[tr.index]
for c in ['p_cnt','p_dt','e_ngrp','e_cnt','e_nl','e_sl','g_nl','g_sl','g_nent']: tr[c]=bt[c]
def show(c, q=10):
    x=tr[c]
    if x.nunique()>q+2: bins=pd.qcut(x,q,duplicates='drop')
    else: bins=x
    g=tr.groupby(bins,observed=True).target.agg(['mean','size']); print(c); print((g['mean']*100).round(2).values, g['size'].values);
    if x.nunique()<=q+2: print(list(g.index))
for c in ['f_20','f_16','f_19','f_00','f_14','f_09','f_23','f_17','f_01','f_07','f_06','f_10','f_08','f_12','f_04','f_03','f_15','f_13','f_02','f_21','f_05','cat_00','cat_01','cat_02','cat_03','cat_04','p_cnt','p_dt','e_ngrp','e_cnt','e_sl','g_sl']:
    show(c, 12 if c not in('f_23',) else 30)
# f_20 fine tail
tr['lr']=np.log(tr.f_20)
print(tr.groupby(pd.cut(tr.lr,[-10,-3,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2,2.5,3,4,10])).target.agg(['mean','size']))
