"""Leak-free feature builder + scenario simulator (cut time + simulated cold entities/groups)."""
import numpy as np, pandas as pd
D = 86400
U = '/mnt/user-data/uploads/'

def load():
    tr = pd.read_csv(U + 'train.csv'); te = pd.read_csv(U + '9a51432a-test.csv')
    tr['is_te'] = 0; te['is_te'] = 1; te['target'] = np.nan
    al = pd.concat([tr, te]).sort_values(['event_time', 'row_id']).reset_index(drop=True)
    al['pair'] = al.entity_id.astype(np.int64) * 100000 + al.group_id
    return al

def _roll(t, W, delay, vals=()):
    """for sorted times t: stats over events j<i with t_i-W <= t_j <= t_i-delay"""
    ar = np.arange(len(t))
    hi = np.minimum(np.searchsorted(t, t - delay, side='right'), ar)
    lo = np.minimum(np.searchsorted(t, t - W, side='left'), hi)
    out = [hi - lo]
    for v in vals:
        c = np.concatenate([[0.0], np.cumsum(v)]); out.append(c[hi] - c[lo])
    return out

def ent_feats(df):
    """entity-history features (names = original f_xx) for events in df (time-sorted)."""
    n = len(df); o = {k: np.zeros(n) for k in ['f_01','f_21','f_06','f_10','f_08','f_00','f_19','f_14','e_cnt','e_nl','e_sl']}
    T = df.event_time.values.astype(float); A = df.f_16.values; Y = np.nan_to_num(df.target.values)
    for _, idx in df.groupby('entity_id').indices.items():
        t = T[idx]; a = A[idx]
        o['f_01'][idx] = np.minimum(np.concatenate([[0], np.diff(t)]), 60 * D); o['f_21'][idx] = (t - t[0]) / D; o['e_cnt'][idx] = np.arange(len(t))
        o['f_06'][idx] = _roll(t, D, 0)[0]
        n7, s7 = _roll(t, 7 * D, 0, [a]); o['f_10'][idx] = n7; o['f_00'][idx] = np.where(n7 > 0, s7 / np.maximum(n7, 1), 0)
        n30, s30, q30 = _roll(t, 30 * D, 0, [a, a * a]); m = s30 / np.maximum(n30, 1)
        o['f_08'][idx] = n30; o['f_19'][idx] = np.where(n30 > 0, m, 0)
        o['f_14'][idx] = np.where(n30 > 0, np.sqrt(np.maximum(q30 / np.maximum(n30, 1) - m * m, 0)), 0)
        nl, sl = _roll(t, 30 * D, 2 * D, [Y[idx]]); o['e_nl'][idx] = nl; o['e_sl'][idx] = sl
    o = pd.DataFrame(o, index=df.index)
    o['f_20'] = df.f_16 / (o.f_19 + 10); o['f_15'] = (o.e_sl + 1) / (o.e_nl + 40)
    return o

def grp_feats(df):
    n = len(df); o = {k: np.zeros(n) for k in ['f_07','f_05','f_12','f_04','f_03','f_09','g_nl','g_sl']}
    T = df.event_time.values.astype(float); A = df.f_16.values; Y = np.nan_to_num(df.target.values)
    for _, idx in df.groupby('group_id').indices.items():
        t = T[idx]; a = A[idx]
        o['f_07'][idx] = np.concatenate([[0], np.diff(t)]); o['f_05'][idx] = (t - t[0]) / D
        o['f_12'][idx] = _roll(t, D, 0)[0]; o['f_04'][idx] = _roll(t, 7 * D, 0)[0]
        n30, s30 = _roll(t, 30 * D, 0, [a]); o['f_03'][idx] = n30; o['f_09'][idx] = np.where(n30 > 0, s30 / np.maximum(n30, 1), 0)
        nl, sl = _roll(t, 30 * D, 2 * D, [Y[idx]]); o['g_nl'][idx] = nl; o['g_sl'][idx] = sl
    o = pd.DataFrame(o, index=df.index); o['f_13'] = (o.g_sl + 2) / (o.g_nl + 80)
    return o

def pair_feats(df):
    g = df.groupby('pair'); o = pd.DataFrame(index=df.index)
    o['p_cnt'] = g.cumcount().values
    o['p_dt'] = (df.event_time - g.event_time.shift(1)).fillna(-1).values
    o['f_17'] = (o.p_cnt == 0).astype(float)
    o['e_ngrp'] = o.f_17.groupby(df.entity_id).cumsum().values       # distinct groups seen so far (incl. current)
    o['g_nent'] = o.f_17.groupby(df.group_id).cumsum().values
    return o

ENT = ['f_01','f_21','f_06','f_10','f_08','f_00','f_19','f_14','f_20','f_15','e_cnt','e_nl','e_sl']
GRP = ['f_07','f_05','f_12','f_04','f_03','f_09','f_13','g_nl','g_sl']
PAIR = ['p_cnt','p_dt','f_17','e_ngrp','g_nent']

TE_KEYS = {'ent': (['entity_id'], 5), 'grp': (['group_id'], 10), 'pair': (['pair'], 1.25)}

def add_te(rows, ref, keys=TE_KEYS, cold_e=None, cold_g=None):
    gm = ref.target.mean(); out = pd.DataFrame(index=rows.index); out['gm'] = gm
    for name, (ks, a) in keys.items():
        st = ref.groupby(ks).target.agg(['sum', 'count']).reset_index()
        m = rows[ks].merge(st, on=ks, how='left')
        s = m['sum'].fillna(0).values.astype(float).copy(); n = m['count'].fillna(0).values.astype(float).copy()
        if cold_e is not None and ('entity_id' in ks or 'pair' in ks):
            z = rows.entity_id.isin(cold_e).values; s[z] = 0; n[z] = 0
        if cold_g is not None and ('group_id' in ks or 'pair' in ks):
            z = rows.group_id.isin(cold_g).values; s[z] = 0; n[z] = 0
        out[f'te_{name}'] = (s + a * gm) / (n + a); out[f'n_{name}'] = n; out[f's_{name}'] = s
    return out

def scenario(al, base, cut, horizon, cold_e=(), cold_g=(), keys=TE_KEYS):
    """pseudo-test = events in (cut, cut+horizon]; cold keys have their pre-cut history erased.
    base = features computed on the full universe (index aligned with al)."""
    cold_e = set(cold_e); cold_g = set(cold_g)
    m = (al.event_time > cut) & (al.event_time <= cut + horizon)
    rows = al[m]; F = base.loc[rows.index].copy()
    post = al[al.event_time > cut]
    if cold_e:
        sub = post[post.entity_id.isin(cold_e)]; f = ent_feats(sub); ix = rows.index.intersection(f.index); F.loc[ix, ENT] = f.loc[ix, ENT].values
    if cold_g:
        sub = post[post.group_id.isin(cold_g)]; f = grp_feats(sub); ix = rows.index.intersection(f.index); F.loc[ix, GRP] = f.loc[ix, GRP].values
    if cold_e or cold_g:
        ce = post.entity_id.isin(cold_e); cg = post.group_id.isin(cold_g)
        univ = pd.concat([al[(al.event_time <= cut) & ~al.entity_id.isin(cold_e) & ~al.group_id.isin(cold_g)], post])
        # pair-features recomputed on the truncated universe (cheap enough)
        f = pair_feats(univ); F.loc[rows.index, PAIR] = f.loc[rows.index, PAIR].values
    ref = al[(al.event_time <= cut) & (al.is_te == 0)]
    te = add_te(rows, ref, keys, cold_e, cold_g)
    F = pd.concat([F, te], axis=1)
    F['cold'] = (rows.entity_id.isin(cold_e) | rows.group_id.isin(cold_g)).values.astype(int)
    F['cold_e'] = rows.entity_id.isin(cold_e).values.astype(int); F['cold_g'] = rows.group_id.isin(cold_g).values.astype(int)
    F['rel_t'] = (rows.event_time.values - cut) / D
    F['cut'] = cut / D
    for c in ['row_id','event_time','entity_id','group_id','pair','cat_00','cat_01','cat_02','cat_03','cat_04','f_16','f_23','f_02','target','is_te']:
        F[c] = rows[c].values
    return F

def base_feats(al):
    return pd.concat([ent_feats(al), grp_feats(al), pair_feats(al)], axis=1)

def pick_cold(al, cut, horizon, rng, fe=0.155, fg=0.035):
    """choose cold keys among those known before the cut (so that they become 'new' at the cut)"""
    pre = al[al.event_time <= cut]
    e = pre.entity_id.unique(); g = pre.group_id.unique()
    return rng.choice(e, int(round(fe * len(e))), replace=False), rng.choice(g, int(round(fg * len(g))), replace=False)

def robust(df, pred):
    from sklearn.metrics import roc_auc_score
    b = pd.qcut(df.rel_t, 3, labels=False) if False else np.minimum((df.rel_t // (df.rel_t.max() / 3 + 1e-9)).astype(int), 2)
    A = [roc_auc_score(df.target[b == i], pred[b == i]) for i in range(3)]
    c = df.cold.values == 1; Ac = roc_auc_score(df.target[c], pred[c]) if c.sum() and df.target[c].nunique() > 1 else np.nan
    R = 0.5 * np.mean(A) + 0.25 * Ac + 0.25 * min(A + [Ac])
    return dict(A1=A[0], A2=A[1], A3=A[2], Acold=Ac, warm=roc_auc_score(df.target[~c], pred[~c]), R=R)

if __name__ == '__main__':
    al = load(); base = base_feats(al)
    # sanity: recomputed == given
    for c in ENT + GRP + ['f_17']:
        if c in al.columns:
            m = al.is_te == 0 if c in ('f_15', 'f_13') else slice(None)
            print(c, float(np.abs(base[c][m] - al[c][m]).max()))
    # sanity: TE == given in test
    te = al[al.is_te == 1]; t = add_te(te, al[al.is_te == 0])
    print('f_22', np.abs(t.te_ent.values - te.f_22.values).max(), 'f_11', np.abs(t.te_grp.values - te.f_11.values).max(), 'f_18', np.abs(t.te_pair.values - te.f_18.values).max())
    base.to_pickle('base.pkl'); al.to_pickle('al2.pkl')
