import json, sys
def L(p): return {r["id"]: r for r in map(json.loads, open(p))}
a, b, tag = L(sys.argv[1]), L(sys.argv[2]), sys.argv[3]
ids = list(b)
ex = [i for i in ids if a[i]["reply"] == b[i]["reply"]]
lab = [i for i in ids if a[i]["your_label"] == b[i]["your_label"]]
tb = [i for i in ids if i.startswith("tb")]
print(f"== {tag}: new={sys.argv[1].split('/')[-1]} ref={sys.argv[2].split('/')[-1]}")
print(f" exact reply match {len(ex)}/{len(ids)} (tb {sum(i in ex for i in tb)}/24); label agree {len(lab)}/{len(ids)} (tb {sum(i in lab for i in tb)}/24)")
print(" held new", sum(a[i]['your_label']=='hold' for i in tb), "ref", sum(b[i]['your_label']=='hold' for i in tb))
print(" label disagreements:", [(i, a[i]["your_label"], b[i]["your_label"]) for i in ids if i not in lab])
import os
pre = []
for i in ids:
    x, y = a[i]["reply"], b[i]["reply"]; k = next((j for j in range(min(len(x), len(y))) if x[j] != y[j]), min(len(x), len(y)))
    if x != y: pre.append((i, k))
print(" first-diverge char idx for non-matching:", pre)
print(" caps new", sum(a[i].get("hit_token_cap",0) for i in ids), "ref", sum(b[i].get("hit_token_cap",0) for i in ids))
