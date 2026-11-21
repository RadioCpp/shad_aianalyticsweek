import pickle
from hon import *
from sklearn.model_selection import GroupKFold
trnA, vals, trnB, te = pickle.load(open('frames.pkl', 'rb'))
k = te.target >= 0; d = te[k].reset_index(drop=True)
# oracle ceiling: CV inside the test period on reconstructed labels (analysis only)
for nl, mdl, rounds in [(8, 300, 300), (8, 300, 600), (16, 200, 400)]:
    p = dict(P0, num_leaves=nl, min_data_in_leaf=mdl); oof = np.zeros(len(d))
    for tr_i, va_i in GroupKFold(5).split(d, groups=d.entity_id):
        m = lgb.train(p, lgb.Dataset(d.loc[tr_i, FEATS], d.target[tr_i]), rounds); oof[va_i] = m.predict(d.loc[va_i, FEATS])
    print('ORACLE in-test CV', nl, mdl, rounds, {k_: round(v, 4) for k_, v in metric(d, oof, 180, 240).items()})
# same CV protocol inside late train (150-180) for comparison of regime difficulty
lt = trnB[(trnB.sim == 0) & (trnB.day > 120)].reset_index(drop=True); lt['cold'] = (np.arange(len(lt)) % 6 == 0).astype(int)
oof = np.zeros(len(lt))
for tr_i, va_i in GroupKFold(5).split(lt, groups=lt.entity_id):
    m = lgb.train(P0, lgb.Dataset(lt.loc[tr_i, FEATS], lt.target[tr_i]), 300); oof[va_i] = m.predict(lt.loc[va_i, FEATS])
print('late-train CV AUC', roc_auc_score(lt.target, oof))
m = fit(trnB, FEATS, 180, 60)
print(pd.Series(m.feature_importance('gain'), FEATS).sort_values(ascending=False).round(0).to_string())
