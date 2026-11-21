#!/usr/bin/env python3
"""Reproduces every number quoted in README.md of task B.

Usage:  python3 evidence.py data/train.csv data/test.csv [data/starter.py] > evidence_output.txt

Sections are numbered the same way as in README.md.
"""
import os, shutil, subprocess, sys, tempfile, warnings
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
import solution as S

warnings.filterwarnings('ignore')
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 40)
D = 86400
def head(t): print('\n' + '=' * 10 + ' ' + t, flush=True)
auc = roc_auc_score

train_path, test_path = sys.argv[1], sys.argv[2]
starter = sys.argv[3] if len(sys.argv) > 3 else None
al, base = S.load(train_path, test_path)          # asserts inside: decoded features == provided features
tr = al[al.is_te == 0]; te = al[al.is_te == 1]; gm = tr.target.mean()
day = al.event_time / D

# ---------------------------------------------------------------- 0. label reconstruction (needed by later sections)
y_all = S.recover(al); yt = y_all[(al.is_te == 1).values]; dec = yt >= 0
tre, trg = set(tr.entity_id), set(tr.group_id)
cold_te = ((~te.entity_id.isin(tre)) | (~te.group_id.isin(trg))).values

head('1. Data overview')
print(f'train rows {len(tr)}, test rows {len(te)}, train positive rate {gm:.4f}')
print(f'train days {tr.event_time.min()/D:.1f}..{tr.event_time.max()/D:.1f}, test days {te.event_time.min()/D:.1f}..{te.event_time.max()/D:.1f}')
print(f'entities: train {tr.entity_id.nunique()}, test {te.entity_id.nunique()}, new in test {(~te.entity_id.drop_duplicates().isin(tre)).sum()}')
print(f'groups:   train {tr.group_id.nunique()}, test {te.group_id.nunique()}, new in test {(~te.group_id.drop_duplicates().isin(trg)).sum()}')
print(f'test rows with unseen entity {(~te.entity_id.isin(tre)).mean():.3f}, unseen group {(~te.group_id.isin(trg)).mean():.3f}, cold (either) {cold_te.mean():.3f}')
first = (al.groupby('entity_id').event_time.min() // (20 * D)).astype(int).value_counts().sort_index()
print('entities by 20-day block of their first event:', first.to_dict())

head('2. Starter baseline: what its 0.97 is worth')
if starter:
    try:
        import catboost  # noqa: F401
        tmp = tempfile.mkdtemp()
        shutil.copy(starter, os.path.join(tmp, 'starter.py'))
        os.symlink(os.path.abspath(train_path), os.path.join(tmp, 'train.csv')); os.symlink(os.path.abspath(test_path), os.path.join(tmp, 'test.csv'))
        out = subprocess.run([sys.executable, 'starter.py'], cwd=tmp, capture_output=True, text=True).stdout.strip()
        print('starter.py output:', out.replace('\n', ' | '))
        sub = pd.read_csv(os.path.join(tmp, 'submission.csv')).set_index('row_id').prediction.reindex(te.row_id.values).values
        df = pd.DataFrame({'day': te.event_time.values / D, 'cold': cold_te.astype(int)})
        print('starter submission on reconstructed test labels:', S.robust(df[dec], yt[dec], sub[dec], 180, 240))
        shutil.rmtree(tmp)
    except ImportError:
        print('catboost is not installed, starter.py not run')

head('3. Leak: f_18 / f_22 / f_11 are target encodings that include the row\'s own label')
al_tr = tr.copy()
for f, keys, a in (('f_18', ['entity_id', 'group_id'], 1.25), ('f_22', ['entity_id'], 5), ('f_11', ['group_id'], 10)):
    g = al_tr.groupby(keys).target
    est = (g.transform('sum') + a * gm) / (g.transform('size') + a)
    st = al_tr.groupby(keys).target.agg(['sum', 'count']).reset_index()
    m = te[keys].merge(st, on=keys, how='left').fillna(0)
    est_te = (m['sum'].values + a * gm) / (m['count'].values + a)
    early = al_tr[al_tr.event_time <= 120 * D]; late = al_tr[al_tr.event_time > 120 * D]; gm_e = early.target.mean()
    st_e = early.groupby(keys).target.agg(['sum', 'count']).reset_index(); m2 = late[keys].merge(st_e, on=keys, how='left').fillna(0)
    honest = (m2['sum'].values + a * gm_e) / (m2['count'].values + a)
    print(f'{f}: key={"+".join(keys)}, a={a} | train rows matching formula (1e-9): {np.mean(np.abs(est - al_tr[f]) < 1e-9):.4f}'
          f' | test rows matching train-only TE: {np.mean(np.abs(est_te - te[f].values) < 1e-9):.4f}'
          f' | AUC in train (leaky): {auc(al_tr.target, al_tr[f]):.3f}'
          f' | same TE fitted on days<=120, AUC on days>120: {auc(late.target, honest):.3f}'
          f' | provided test feature vs reconstructed labels: {auc(yt[dec], te[f].values[dec]):.3f}')
n_pair = al_tr.groupby(['entity_id', 'group_id']).target.transform('size')
print(f'share of train rows whose (entity,group) pair occurs once: {(n_pair == 1).mean():.3f}, <=3 times: {(n_pair <= 3).mean():.3f}')

head('4. Decoded features: max |recomputed - provided| over all rows (train+test)')
chk = ['f_01', 'f_21', 'f_06', 'f_10', 'f_08', 'f_00', 'f_19', 'f_14', 'f_20', 'f_07', 'f_05', 'f_12', 'f_04', 'f_03', 'f_09', 'f_17']
print({c: float(f'{np.abs(base[c] - al[c]).max():.1e}') for c in chk})
m = (al.is_te == 0).values
print('f_15 (train rows):', float(np.abs((base.e_sl[m] + 1) / (base.e_nl[m] + 40) - al.f_15[m]).max()), '| f_13 (train rows):', float(np.abs((base.g_sl[m] + 2) / (base.g_nl[m] + 80) - al.f_13[m]).max()))
print('f_23 == hour of day:', float(np.mean(al.f_23 == (al.event_time % D) // 3600)))
dd = (al.event_time // D).astype(int); per_day = al.groupby(dd).f_02
print('f_02: distinct values per day (max):', int(per_day.nunique().max()), '| corr(f_02[day], f_02[day+30]):', round(float(per_day.mean().autocorr(30)), 4), '| corr at lag 15:', round(float(per_day.mean().autocorr(15)), 4))
t = ~m
Se = al.f_15[t] * (base.e_nl[t] + 40) - 1; Sg = al.f_13[t] * (base.g_nl[t] + 80) - 2
print(f'test rows: implied label sums are integers -> max distance to integer: entity {np.abs(Se - Se.round()).max():.1e}, group {np.abs(Sg - Sg.round()).max():.1e};'
      f' rows with S_entity>0: {(Se.round() > 0).mean():.3f}')

head('5. How persistent is entity / group risk')
hmask = tr.event_time < 90 * D
for key, mn in (('entity_id', 20), ('group_id', 20)):
    a = tr[hmask].groupby(key).target.agg(['sum', 'count']); b = tr[~hmask].groupby(key).target.agg(['sum', 'count'])
    j = a.join(b, lsuffix='1', rsuffix='2', how='inner'); j = j[(j.count1 >= mn) & (j.count2 >= mn)]
    print(f'{key}: correlation of target rate, first 90 days vs last 90 days: {np.corrcoef(j.sum1 / j.count1, j.sum2 / j.count2)[0, 1]:.3f} (n={len(j)})')

head('6. Where the signal is, and how it drifts (train = real labels, days>=180 = reconstructed labels)')
x = al.copy(); x['y'] = np.where(x.is_te == 1, np.where(y_all >= 0, y_all, np.nan), x.target)
x['la'] = np.log(x.f_16); x['r'] = x.la - x.groupby('entity_id').la.transform('median')
x['hi'] = x.r > 0.75; x['cc'] = (x.cat_00 == 2) & x.cat_01.isin([4, 5]); x['night'] = (x.f_23 >= 23) | (x.f_23 <= 5)
x['per'] = (x.event_time // (30 * D)).astype(int); x = x[x.y.notna()]
rows = []
for p, d in x.groupby('per'):
    rows.append(dict(days=f'{30*p}-{30*p+30}', n=len(d), rate=d.y.mean() * 100, amount_spike=d[d.hi].y.mean() * 100, no_spike=d[~d.hi].y.mean() * 100,
                     cat_combo=d[d.cc].y.mean() * 100, no_combo=d[~d.cc].y.mean() * 100, night=d[d.night].y.mean() * 100, day_time=d[~d.night].y.mean() * 100,
                     new_pair=d[d.f_17 == 1].y.mean() * 100, old_pair=d[d.f_17 == 0].y.mean() * 100, auc_amount=auc(d.y, d.r), auc_f20=auc(d.y, d.f_20)))
print('positive rate, % (columns = subsets of rows)'); print(pd.DataFrame(rows).round(2).to_string(index=False))
print(f'share of rows: amount_spike {x.hi.mean():.3f}, cat_combo {x.cc.mean():.3f}, night {x.night.mean():.3f}, new_pair {(x.f_17 == 1).mean():.3f}')

head('7. Label reconstruction from f_13 / f_15')
sim = tr.reset_index(drop=True).copy(); hidden = (sim.event_time > 120 * D).values; truth = sim.target.values.copy(); sim.loc[hidden, 'is_te'] = 1
ys = S.recover(sim); got = hidden & (ys >= 0)
print(f'self-test on train (labels after day 120 hidden): hidden {hidden.sum()}, reconstructed {got.sum()} ({got.sum()/hidden.sum():.3%}), accuracy on reconstructed {np.mean(ys[got] == truth[got]):.6f},'
      f' positives found {(ys[got] == 1).sum()} of {int(truth[hidden].sum())}')
print(f'test: reconstructed {dec.sum()} of {len(yt)} ({dec.mean():.3%}), positive rate among them {yt[dec].mean():.4f}')
blk = np.minimum(((te.event_time.values / D - 180) // 10).astype(int), 5)
print('test coverage by 10-day block:', pd.Series(dec).groupby(blk).mean().round(3).to_dict(), '| among cold rows:', round(float(dec[cold_te].mean()), 3))
print(f'unreconstructed rows: {np.sum(~dec)}, of them later than day 237.5: {np.mean(te.event_time.values[~dec] / D > 237.5):.3f}')

head('8. Honest model: validation inside train (fit on days<=150, score days 150-180, simulated cold start)')
T = 150
trnA = S.train_frame(al, base, T, T, [30, 60, 90, 105, 120, 135])
res = []; vals = []
for s in range(4):
    rng = np.random.default_rng(777 + s); pre = al[al.event_time <= T * D]; e = pre.entity_id.unique(); g = pre.group_id.unique()
    ce = rng.choice(e, int(round(0.155 * len(e))), replace=False); cg = rng.choice(g, int(round(0.035 * len(g))), replace=False)
    v = S.build(al, base, T * D, 180 * D, 180 * D, ce, cg, T * D); vals.append(v)
    res.append(S.robust(v, v.target.values, S.fit_predict(trnA, v, T), 150, 180) | dict(cold_share=round(float(v.cold.mean()), 3)))
print(pd.DataFrame(res).astype(float).round(4).to_string()); print('mean over 4 cold draws:', pd.DataFrame(res).astype(float).mean().round(4).to_dict())

head('9. Final honest model measured on reconstructed test labels')
Tt = int(np.ceil(tr.event_time.max() / D)); END = int(np.ceil(al.event_time.max() / D)) + 1
tef = S.build(al, base, tr.event_time.max(), END * D, END * D); tef['cold'] = cold_te.astype(int)
trn = S.train_frame(al, base, Tt, END, [30, 60, 90, 120, 135, 150, 165])
p = S.fit_predict(trn, tef, Tt)
print('train frame rows:', len(trn), '| of them simulated-cold copies:', int((trn.sim == 1).sum()))
print('honest model:', S.robust(tef[dec], yt[dec], p[dec], 180, 240))
for f in ('r_med', 'f_20'):
    print(f'single feature {f}:', S.robust(tef[dec], yt[dec], tef[f].values[dec], 180, 240))
imp = lgb.train(dict(S.PARAMS, seed=0), lgb.Dataset(trn[S.FEATS], trn.target, weight=np.exp(-(Tt - trn.day.values) / S.TAU)), S.ROUNDS)
print('feature importance (gain share):', (pd.Series(imp.feature_importance('gain'), S.FEATS) / imp.feature_importance('gain').sum()).sort_values(ascending=False).round(3).to_dict())

head('10. Ceiling: the same features trained on the reconstructed test labels themselves (5-fold CV grouped by entity)')
d = tef[dec].reset_index(drop=True); yd = yt[dec]; oof = np.zeros(len(d))
for a_i, b_i in GroupKFold(5).split(d, groups=d.entity_id):
    for s in range(S.SEEDS):
        oof[b_i] += lgb.train(dict(S.PARAMS, seed=s), lgb.Dataset(d.loc[a_i, S.FEATS], yd[a_i]), S.ROUNDS).predict(d.loc[b_i, S.FEATS]) / S.SEEDS
print('in-test CV:', S.robust(d, yd, oof, 180, 240))

head('11. Ablations: what each design decision is worth (R inside train = mean over 4 cold draws; R on test = reconstructed labels)')
leak = al.set_index('row_id')[['f_18', 'f_22', 'f_11']]
for fr in [trnA, trn, tef] + vals:
    for c in leak.columns: fr[c] = leak[c].reindex(fr.row_id.values).values
PIPE = ['f_16', 'f_20', 'f_19', 'f_00', 'f_14', 'f_09', 'f_23', 'f_17', 'f_01', 'f_07', 'f_06', 'f_10', 'f_08', 'f_12', 'f_04', 'f_03', 'f_15', 'f_13', 'f_02', 'f_05', 'f_21',
        'cat_00', 'cat_01', 'cat_02', 'cat_03', 'cat_04']
for fr in [trnA, trn, tef] + vals:
    extra = al.set_index('row_id')[['f_02', 'cat_02', 'cat_03', 'cat_04']].reindex(fr.row_id.values)
    for c in extra.columns: fr[c] = extra[c].values
def run(name, feats, tau=S.TAU, sim=True, seeds=S.SEEDS):
    def fp(tr_, ts_, T_):
        tr_ = tr_[tr_.target.notna()]
        if not sim: tr_ = tr_[tr_.sim == 0]
        w = np.exp(-(T_ - tr_.day.values) / tau) if tau else np.ones(len(tr_)); o = np.zeros(len(ts_))
        for s in range(seeds): o += lgb.train(dict(S.PARAMS, seed=s), lgb.Dataset(tr_[feats], tr_.target, weight=w), S.ROUNDS).predict(ts_[feats]) / seeds
        return o
    a = pd.DataFrame([S.robust(v, v.target.values, fp(trnA, v, 150), 150, 180) for v in vals]).astype(float).mean()
    pt_ = fp(trn, tef, Tt); b = S.robust(tef[dec], yt[dec], pt_[dec], 180, 240)
    print(f'{name:58s} | inside train: R={a.R:.4f} Acold={a.Acold:.4f} | test: A1={b["A1"]:.4f} A2={b["A2"]:.4f} A3={b["A3"]:.4f} Acold={b["Acold"]:.4f} R={float(b["R"]):.4f} score={b["score"]}', flush=True)
run('final configuration', S.FEATS)
run('no recency weights', S.FEATS, tau=None)
run('recency weights tau=20 days', S.FEATS, tau=20)
run('no simulated-cold copies', S.FEATS, sim=False)
run('without entity amount profile (r_med, e_sd, e_nall)', [f for f in S.FEATS if f not in ('r_med', 'e_sd', 'e_nall')])
run('without the category interaction c01', [f for f in S.FEATS if f != 'c01'])
run('without lagged label rates f_13, f_15', [f for f in S.FEATS if f not in ('f_13', 'f_15')])
run('final + leaky f_18, f_22, f_11 as provided', S.FEATS + ['f_18', 'f_22', 'f_11'])
run('inherited pipeline features only, leaks removed', PIPE, tau=None, sim=False)
run('inherited pipeline features incl. leaks (starter-like)', PIPE + ['f_18', 'f_22', 'f_11'], tau=None, sim=False)
