import pandas as pd, numpy as np
pd.set_option('display.width',280); pd.set_option('display.max_columns',60); pd.set_option('display.max_rows',300)
al=pd.read_pickle('al.pkl')
D=86400
tr=al[al.is_te==0]
def roll(df,key,W,delay,valcol=None):
    out_n=np.zeros(len(df)); out_s=np.zeros(len(df))
    for k,idx in df.groupby(key).indices.items():
        t=df.event_time.values[idx]
        hi=np.searchsorted(t, t-delay, side='right'); hi=np.minimum(hi, np.arange(len(t)))
        lo=np.searchsorted(t, t-W, side='left') if W else np.zeros(len(t),int); lo=np.minimum(lo,hi)
        out_n[idx]=hi-lo
        if valcol is not None:
            c=np.concatenate([[0],np.cumsum(df[valcol].values[idx])]); out_s[idx]=c[hi]-c[lo]
    return out_n,out_s
trs=tr.reset_index(drop=True)
gm=trs.target.mean()
# implied n assuming s integer: f_15=(s+1)/(n+40)
res=[]
for W in [30,45,60,90,120,0]:
    for dl in [1,1.5,2,2.5,3]:
        n,s=roll(trs,'entity_id',W*D,dl*D,'target')
        for a,b in [(1,40),(gm*40,40)]:
            res.append((np.mean(np.abs((s+a)/(n+b)-trs.f_15)<1e-9),W,dl,a,b))
res.sort(reverse=True); print('f_15',res[:5])
res=[]
for W in [30,45,60,90,120,0]:
    for dl in [1,1.5,2,2.5,3]:
        n,s=roll(trs,'group_id',W*D,dl*D,'target')
        for a,b in [(2,81),(2,80),(gm*80,80)]:
            res.append((np.mean(np.abs((s+a)/(n+b)-trs.f_13)<1e-9),W,dl,a,b))
res.sort(reverse=True); print('f_13',res[:5])
