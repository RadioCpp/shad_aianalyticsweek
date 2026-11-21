import pickle
from hon import *
trnA, vals, trnB, te = pickle.load(open('frames.pkl', 'rb'))
k = te.target >= 0; tk = te[k]
def ev(feats, tau=60, rounds=400, params=P0, cold_w=1.0, name='', seeds=(0,)):
    pv = [np.zeros(len(v)) for v in vals]; pt = np.zeros(len(tk))
    for s in seeds:
        p = dict(params, seed=s)
        m = fit(trnA, feats, 150, tau, p, rounds, cold_w)
        for i, v in enumerate(vals): pv[i] += m.predict(v[feats])
        pt += fit(trnB, feats, 180, tau, p, rounds, cold_w).predict(tk[feats])
    r = pd.DataFrame([metric(v, pv[i], 150, 180) for i, v in enumerate(vals)]).mean(); rB = metric(tk, pt, 180, 240)
    print(f'{name:26s} | val150 ' + ' '.join(f'{k_}={v:.4f}' for k_, v in r.items()) + ' | TEST ' + ' '.join(f'{k_}={v:.4f}' for k_, v in rB.items()), flush=True)
COMPACT = ['la','r_med','e_sd','e_nall','f_20','r30','cv30','f_23','f_17','p_cnt_c','p_dt','e_ngrp_c','age','cat_00','cat_01','c01','f_15','f_13']
MINI = ['r_med','e_sd','e_nall','r30','f_23','f_17','p_cnt_c','age','cat_00','cat_01','c01']
ev(FEATS, name='all')
ev(COMPACT, name='compact')
ev(MINI, name='mini')
ev(COMPACT, name='compact r200', rounds=200)
ev(COMPACT, name='compact r800 lr.015', rounds=800, params=dict(P0, learning_rate=0.015))
ev(COMPACT, name='compact leaves4', params=dict(P0, num_leaves=4), rounds=600)
ev(COMPACT, name='compact cold_w=3', cold_w=3)
ev(COMPACT, name='compact tau=90', tau=90)
ev(COMPACT, name='compact tau=40', tau=40)
ev(COMPACT, name='compact 5 seeds', seeds=range(5))
ev(FEATS, name='all 5 seeds', seeds=range(5))
