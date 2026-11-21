"""Honest pipeline: label-free features (+ the legit lagged rates f_13/f_15), simulated cold start, recency weights."""
import numpy as np, pandas as pd, lightgbm as lgb, warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score
from sim import ent_feats, grp_feats, pair_feats, ENT, GRP, PAIR, D

def build(al, base, lo, hi, t_end, cold_e=(), cold_g=(), cut=None):
    """rows with lo < t <= hi. cold keys: history before `cut` erased. entity amount profile uses events <= t_end."""
    cold_e = set(cold_e); cold_g = set(cold_g)
    rows = al[(al.event_time > lo) & (al.event_time <= hi)]
    F = base.loc[rows.index].copy()
    univ = al[al.event_time <= t_end]
    if cold_e or cold_g:
        post = univ[univ.event_time > cut]
        if cold_e:
            f = ent_feats(post[post.entity_id.isin(cold_e)]); ix = rows.index.intersection(f.index); F.loc[ix, ENT] = f.loc[ix, ENT].values
        if cold_g:
            f = grp_feats(post[post.group_id.isin(cold_g)]); ix = rows.index.intersection(f.index); F.loc[ix, GRP] = f.loc[ix, GRP].values
        univ = pd.concat([univ[(univ.event_time <= cut) & ~univ.entity_id.isin(cold_e) & ~univ.group_id.isin(cold_g)], post])
        f = pair_feats(univ); F.loc[rows.index, PAIR] = f.loc[rows.index, PAIR].values
    for c in ['row_id','event_time','entity_id','group_id','cat_00','cat_01','cat_02','cat_03','cat_04','f_16','f_23','f_02','target','is_te']:
        F[c] = rows[c].values
    la = np.log(univ.f_16); st = la.groupby(univ.entity_id).agg(['median', 'std', 'count'])
    m = st.reindex(rows.entity_id.values)
    F['la'] = np.log(F.f_16); F['e_med'] = m['median'].values; F['e_sd'] = m['std'].fillna(-1).values; F['e_nall'] = m['count'].values
    F['r_med'] = F.la - F.e_med
    F['r30'] = np.where(F.f_19 > 0, F.la - np.log(np.maximum(F.f_19, 1e-9)), np.nan)
    F['cv30'] = F.f_14 / (F.f_19 + 1)
    F['night'] = ((F.f_23 >= 23) | (F.f_23 <= 5)).astype(int)
    F['age'] = np.minimum(F.f_21, 60); F['g_age'] = np.minimum(F.f_05, 60)
    F['p_cnt_c'] = np.minimum(F.p_cnt, 30); F['e_ngrp_c'] = np.minimum(F.e_ngrp, 30)
    F['c01'] = F.cat_00 * 6 + F.cat_01
    F['cold'] = (rows.entity_id.isin(cold_e) | rows.group_id.isin(cold_g)).values.astype(int)
    F['day'] = F.event_time / D
    return F

FEATS = ['la','r_med','e_sd','e_nall','f_20','r30','cv30','f_19','f_00','f_14','f_09','f_23','f_17','p_cnt_c','p_dt','e_ngrp_c','age','g_age',
         'cat_00','cat_01','c01','cat_02','cat_03','cat_04','f_15','e_sl','f_13','g_sl','f_01','f_07','f_06','f_10','f_08','f_12','f_04','f_03']

def pick(al, cut, rng, fe, fg):
    pre = al[al.event_time <= cut]; e = pre.entity_id.unique(); g = pre.group_id.unique()
    return rng.choice(e, int(round(fe * len(e))), replace=False), rng.choice(g, int(round(fg * len(g))), replace=False)

def train_frame(al, base, T, cuts, fe=0.3, fg=0.06, seed=0):
    """warm copy of all rows <= T plus simulated-cold copies (only the cold rows) for several cuts."""
    fr = [build(al, base, -1, T * D, 240 * D if T >= 180 else T * D).assign(sim=0)]
    for i, c in enumerate(cuts):
        rng = np.random.default_rng(seed * 100 + i)
        ce, cg = pick(al, c * D, rng, fe, fg)
        h = min(c + 60, T)
        f = build(al, base, c * D, h * D, h * D, ce, cg, c * D)
        fr.append(f[f.cold == 1].assign(sim=1))
    return pd.concat(fr, ignore_index=True)

def metric(df, pred, t0, t1):
    b = np.minimum(((df.day.values - t0) // ((t1 - t0) / 3)).astype(int), 2); y = df.target.values
    A = [roc_auc_score(y[b == i], pred[b == i]) for i in range(3)]; c = df.cold.values == 1
    Ac = roc_auc_score(y[c], pred[c]); R = 0.5 * np.mean(A) + 0.25 * Ac + 0.25 * min(A + [Ac])
    return dict(A1=A[0], A2=A[1], A3=A[2], Acold=Ac, R=R, score=100 * np.clip((R - 0.55) / 0.2, 0, 1))

P0 = dict(objective='binary', learning_rate=0.03, num_leaves=8, min_data_in_leaf=300, feature_fraction=0.6, bagging_fraction=0.7, bagging_freq=1, lambda_l2=20, verbose=-1, num_threads=2, seed=0, max_cat_to_onehot=64)

def fit(trn, feats, T, tau, params=P0, rounds=400, cold_w=1.0):
    w = np.exp(-(T - trn.day.values) / tau) if tau else np.ones(len(trn))
    w = w * np.where(trn.sim.values == 1, cold_w, 1.0)
    return lgb.train(params, lgb.Dataset(trn[feats], trn.target, weight=w), rounds)
