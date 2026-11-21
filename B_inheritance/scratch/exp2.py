import numpy as np, pandas as pd, lightgbm as lgb, time, sys, warnings
warnings.filterwarnings('ignore')
from sim import robust
trn = pd.read_pickle('trn.pkl'); val = pd.read_pickle('val.pkl')
print(trn.shape, val.shape, val.cold.mean(), trn.target.mean(), val.target.mean())
RAW = ['f_16','f_20','f_19','f_00','f_14','f_09','f_23','f_17','f_01','f_07','f_06','f_10','f_08','f_12','f_04','f_03','f_15','f_13','cat_00','cat_01','cat_02','cat_03','cat_04']
TE = ['te_ent','n_ent','te_grp','n_grp','te_pair','n_pair']
EXTRA = ['e_nl','e_sl','g_nl','g_sl','p_cnt','p_dt','e_ngrp']
def run(feats, name, params=None, rounds=600, ret=False):
    p = dict(objective='binary', learning_rate=0.03, num_leaves=15, min_data_in_leaf=200, feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10, verbose=-1, num_threads=2, seed=0)
    if params: p.update(params)
    m = lgb.train(p, lgb.Dataset(trn[feats], trn.target), rounds)
    out = []
    for r in (200, 400, rounds):
        pr = m.predict(val[feats], num_iteration=r)
        res = pd.DataFrame([robust(val[val.seed == s], pr[(val.seed == s).values]) for s in sorted(val.seed.unique())]).mean()
        out.append(res); print(f'{name:28s} it={r:4d} ' + ' '.join(f'{k}={v:.4f}' for k, v in res.items()), flush=True)
    if ret: return m
if __name__ == '__main__':
    # single-feature reference points
    for f in ['f_20','te_ent','te_pair','f_15']:
        print(f, pd.DataFrame([robust(val[val.seed == s], val[f][val.seed == s].values) for s in sorted(val.seed.unique())]).mean().round(4).to_dict())
    run(RAW, 'raw(no TE)')
    run(RAW + TE, 'raw+TE')
    m = run(RAW + TE + EXTRA, 'raw+TE+extra', ret=True)
    print(pd.Series(m.feature_importance('gain'), RAW + TE + EXTRA).sort_values(ascending=False).round(0).to_string())
