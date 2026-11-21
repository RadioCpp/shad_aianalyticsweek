import pandas as pd, numpy as np
pd.set_option('display.width',280); pd.set_option('display.max_columns',60); pd.set_option('display.max_rows',300)
al=pd.read_pickle('al.pkl').reset_index(drop=True)
D=86400
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
n,_=roll(al,'entity_id',30*D,2*D)
al['e_n_lag']=n; al['e_s_rec']=al.f_15*(n+40)-1
te=al[al.is_te==1]
print('test entity s integrality', np.abs(te.e_s_rec-te.e_s_rec.round()).max(), te.e_s_rec.round().value_counts().sort_index().to_dict())
n,_=roll(al,'group_id',30*D,2*D)
al['g_n_lag']=n; al['g_s_rec']=al.f_13*(n+80)-2
te=al[al.is_te==1]
print('test group s integrality', np.abs(te.g_s_rec-te.g_s_rec.round()).max())
# amount features
al['amt']=al.f_16
g=al.groupby('entity_id').amt
al['exp_mean']=(g.cumsum()-al.amt)/al.e_cnt.replace(0,np.nan)
print('f_19 expanding mean', np.nanmean(np.abs(al.exp_mean-al.f_19)<1e-6), 'frac f_20==amt/(f_19+10)', np.mean(np.abs(al.f_20-al.amt/(al.f_19+10))<1e-9))
for W in [7,30]:
    n,s=roll(al,'entity_id',W*D,0,'amt'); m=np.where(n>0,s/np.maximum(n,1),0)
    print('rolling mean W',W,'f_00',np.mean(np.abs(m-al.f_00)<1e-6),'f_19',np.mean(np.abs(m-al.f_19)<1e-6))
for k in [3,5,10,20]:
    m=g.transform(lambda s: s.shift(1).rolling(k,min_periods=1).mean()).fillna(0)
    sd=g.transform(lambda s: s.shift(1).rolling(k,min_periods=2).std()).fillna(0)
    print('last-k mean',k,'f_00',np.mean(np.abs(m-al.f_00)<1e-6),'f_19',np.mean(np.abs(m-al.f_19)<1e-6),'std f_14',np.mean(np.abs(sd-al.f_14)<1e-6))
sd=g.transform(lambda s: s.shift(1).expanding(2).std()).fillna(0); print('exp std f_14',np.mean(np.abs(sd-al.f_14)<1e-6))
for W in [7,30]:
    n,s=roll(al,'entity_id',W*D,0,'amt'); al['a2']=al.amt**2; n,s2=roll(al,'entity_id',W*D,0,'a2')
    m=s/np.maximum(n,1); sd=np.sqrt(np.maximum(s2/np.maximum(n,1)-m**2,0)); sd1=np.where(n>1, sd*np.sqrt(n/np.maximum(n-1,1)),0)
    print('roll std W',W, np.mean(np.abs(sd-al.f_14)<1e-6), np.mean(np.abs(sd1-al.f_14)<1e-6))
# f_09: group-level?
gg=al.groupby('group_id').amt
al['g_exp_mean']=(gg.cumsum()-al.amt)/al.g_cnt.replace(0,np.nan)
print(al[['f_09','g_exp_mean','f_16','f_19','f_00','f_14']].corr(method='spearman').round(3))
for W in [1,7,30]:
    n,s=roll(al,'group_id',W*D,0,'amt'); m=np.where(n>0,s/np.maximum(n,1),0); print('grp roll mean',W, np.mean(np.abs(m-al.f_09)<1e-6), np.corrcoef(m,al.f_09)[0,1])
grp=al[al.group_id==al.group_id.value_counts().index[100]]
print(grp[['event_time','entity_id','f_16','f_09','g_exp_mean','f_03','f_04','f_12']].head(25).to_string())
al.to_pickle('al.pkl')
