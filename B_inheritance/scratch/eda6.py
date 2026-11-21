import pandas as pd, numpy as np
pd.set_option('display.width',280); pd.set_option('display.max_columns',60); pd.set_option('display.max_rows',300)
al=pd.read_pickle('al.pkl')
D=86400
# arrivals of new entities / groups / pairs per 20-day block
al['blk']=(al.event_time//(20*D))
first=al.groupby('entity_id').event_time.min()//(20*D); print('new entities per 20d', first.value_counts().sort_index().values)
first=al.groupby('group_id').event_time.min()//(20*D); print('new groups per 20d', first.value_counts().sort_index().values)
print('rows per 20d', al.blk.value_counts().sort_index().values)
print('frac rows new pair per blk', al.groupby('blk').f_17.mean().round(3).values)
print('frac rows e_age<... first-20d-of-entity', al.assign(n=(al.f_21<20)).groupby('blk').n.mean().round(3).values)
# decode windows by brute force on a sample of entities, vectorised with searchsorted
def roll(al,key,W,delay,valcol=None):
    out_n=np.zeros(len(al)); out_s=np.zeros(len(al))
    for k,idx in al.groupby(key).indices.items():
        t=al.event_time.values[idx]
        hi=np.searchsorted(t, t-delay, side='right')  # events with time <= t-delay
        lo=np.searchsorted(t, t-W, side='left') if W else np.zeros(len(t),int)
        hi=np.minimum(hi, np.arange(len(t)))
        lo=np.minimum(lo,hi)
        out_n[idx]=hi-lo
        if valcol is not None:
            c=np.concatenate([[0],np.cumsum(al[valcol].values[idx])])
            out_s[idx]=c[hi]-c[lo]
    return out_n,out_s
tr=al[al.is_te==0].reset_index(drop=True)   # shuffled? al is time sorted
for W in [7,14,21,28,30,45,60,90]:
    n,_=roll(al,'entity_id',W*D,0)
    print('f_08 W',W, np.mean(n==al.f_08), 'f_10', np.mean(n==al.f_10), 'f_06', np.mean(n==al.f_06))
for W in [0.5,1,2,3,7]:
    n,_=roll(al,'entity_id',W*D,0); print('W',W,'f_06', np.mean(n==al.f_06),'f_10', np.mean(n==al.f_10))
for W in [0.5,1,2,3,7,14,28,30,60]:
    n,_=roll(al,'group_id',W*D,0); print('grp W',W,'f_03', np.mean(n==al.f_03),'f_04', np.mean(n==al.f_04),'f_12', np.mean(n==al.f_12))
