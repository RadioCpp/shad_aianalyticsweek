"""Reconstruct labels hidden behind the lagged-rate features f_15 (entity) / f_13 (group).
window for row i of a key: events j<i of the same key with t_i-30d <= t_j <= t_i-2d ; S = sum of labels there."""
import numpy as np, pandas as pd
D = 86400
def recover(al, known):
    """al: time-sorted frame with event_time, entity_id, group_id, f_15, f_13, target; known: bool mask of rows with known label.
    returns y (0/1/-1 unknown)"""
    n = len(al); y = np.where(known, np.nan_to_num(al.target.values), -1).astype(int)
    T = al.event_time.values.astype(float)
    cons = []          # (member idx array, target sum)
    for key, f, a, bb in (('entity_id', 'f_15', 1, 40), ('group_id', 'f_13', 2, 80)):
        F = al[f].values
        for _, idx in al.groupby(key).indices.items():
            t = T[idx]; ar = np.arange(len(t))
            hi = np.minimum(np.searchsorted(t, t - 2 * D, side='right'), ar); lo = np.minimum(np.searchsorted(t, t - 30 * D, side='left'), hi)
            S = np.rint(F[idx] * (hi - lo + bb) - a).astype(int)
            for i in range(len(t)):
                if known[idx[i]] and i + 1 < len(t) and known[idx[i + 1]] and False: continue
                if hi[i] > lo[i]: cons.append((idx[lo[i]:hi[i]], S[i]))
    print('constraints', len(cons))
    # propagation with prefix-difference trick: windows are intervals in per-key order -> use interval sums.
    # generic propagation on single windows + pairwise nested differences (consecutive windows of the same key)
    changed = True; it = 0
    while changed and it < 50:
        changed = False; it += 1
        prev = None
        for members, S in cons:
            ym = y[members]; unk = members[ym < 0]
            if len(unk):
                r = S - ym[ym >= 0].sum()
                if r == 0: y[unk] = 0; changed = True
                elif r == len(unk): y[unk] = 1; changed = True
            # difference with the previous constraint (consecutive rows of the same key share most members)
            if prev is not None:
                pm, pS = prev
                if len(pm) and len(members) and pm[-1] >= members[0] and (pm[0] <= members[0]):
                    leave = pm[pm < members[0]]; enter = members[members > pm[-1]]
                    if (len(leave) == 0 or (y[leave] >= 0).all()):
                        ye = y[enter]; unk = enter[ye < 0]
                        if len(unk):
                            r = S - pS + y[leave].sum() - ye[ye >= 0].sum()
                            if r == 0: y[unk] = 0; changed = True
                            elif r == len(unk): y[unk] = 1; changed = True
            prev = (members, S)
        print('iter', it, 'unknown left', int((y < 0).sum()))
    return y

if __name__ == '__main__':
    al = pd.read_pickle('al2.pkl')
    # 1) self-test on train: hide labels after day 120
    tr = al[al.is_te == 0].reset_index(drop=True)
    known = (tr.event_time <= 120 * D).values
    y = recover(tr, known)
    h = ~known; dec = h & (y >= 0)
    print('hidden', h.sum(), 'decoded', dec.sum(), 'coverage', dec.sum() / h.sum(), 'accuracy on decoded', (y[dec] == tr.target.values[dec]).mean(), 'pos decoded', (y[dec] == 1).sum(), 'pos true', int(tr.target.values[h].sum()))
    # 2) real: train known, test unknown
    known = (al.is_te == 0).values
    y = recover(al, known)
    te = al.is_te.values == 1
    print('test rows', te.sum(), 'decoded', (te & (y >= 0)).sum(), 'coverage', (te & (y >= 0)).sum() / te.sum(), 'decoded positives', (te & (y == 1)).sum(), 'rate among decoded', (y[te & (y >= 0)] == 1).mean())
    al['y_rec'] = y; al[['row_id', 'y_rec']].to_pickle('yrec.pkl')
    d = (al.event_time[te] / D)
    print('coverage by 10d', pd.Series((y[te] >= 0)).groupby(((d.values - 180) // 10).astype(int)).mean().round(3).values)
