import pandas as pd, numpy as np, collections
from f import tr, runs
for s in (0,2):
    d=tr[tr.source==s].string
    inner=collections.Counter(); edge=collections.Counter();
    for z in d:
        r=runs(z)
        for x in r[1:-1]: inner[x]+=1
        edge[r[0]]+=1; edge[r[-1]]+=1
    print(s,'inner',sorted(inner.items())); print(s,'edge',sorted(edge.items()))
d=tr[tr.source==2].string
print('num long runs(>=6) per string', d.apply(lambda z: sum(x>=6 for x in runs(z))).value_counts().sort_index().to_dict())
print('num runs len 2-5 per string', d.apply(lambda z: sum(2<=x<=5 for x in runs(z))).value_counts().sort_index().to_dict())
for z in d.head(25): print(z, runs(z))
d0=tr[tr.source==0].string
c=collections.Counter()
for z in d0:
    for x in runs(z)[1:-1]: c[x]+=1
tot=sum(c.values()); cum=0
for k in sorted(c):
    print(k, c[k], 'hazard', round(c[k]/(tot-cum),3)); cum+=c[k]
for s in (0,1,2):
    a=b=0
    for z in tr[tr.source==s].string:
        sym=[z[0]]+[z[i] for i in range(1,28) if z[i]!=z[i-1]]
        a+=sum(sym[i]==sym[i+2] for i in range(len(sym)-2)); b+=len(sym)-2
    print(s,'ABA rate',a/b)
