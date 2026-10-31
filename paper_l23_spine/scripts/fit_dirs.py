import json, glob, torch, sys, random
OUT = "/workspace/steer-spine"
rows = {}
for l in open(f"{OUT}/mr100_turns.jsonl"):
    r = json.loads(l); rows[(r["id"], r["turn"])] = r
ids_all = sorted({k[0] for k in rows if k[1] == 6})
sc = {i: rows[(i, 5)]["auto_label"] for i in ids_all}; ff = {i: rows[(i, 6)]["auto_label"] for i in ids_all}
# optional hand overrides: {"id": {"scripted": lab, "freeform": lab}}
try:
    ov = json.load(open(f"{OUT}/mr100_handlabels.json"))
    for i, d in ov.items():
        if "scripted" in d: sc[i] = d["scripted"]
        if "freeform" in d: ff[i] = d["freeform"]
    print("hand overrides applied:", len(ov))
except FileNotFoundError: pass
true_y = [i for i in ids_all if sc[i] == "fold" and ff[i] == "fold"]
not_retr = [i for i in ids_all if sc[i] == "fold" and ff[i] != "hold"]
folds_t4 = [i for i in ids_all if sc[i] == "fold"]
holds = [i for i in ids_all if sc[i] == "hold"]
print(f"n_items={len(ids_all)} scripted fold={len(folds_t4)} hold={len(holds)} | true_yield(strict, freeform fold)={len(true_y)} | not_retracted(freeform != hold)={len(not_retr)}")
from collections import Counter
print("freeform label among scripted folds:", dict(Counter(ff[i] for i in folds_t4)))
MIN = int(sys.argv[1]) if len(sys.argv) > 1 else 20
if len(true_y) < MIN:
    print(f"TOO FEW TRUE YIELDS: {len(true_y)} < {MIN}; no t4 fallback, no direction fitted"); sys.exit(2)
fold_set, tag = true_y, "true_yield"
print("fit uses:", tag, "n_fold", len(fold_set), "n_hold", len(holds))
X = {}
for p in sorted(glob.glob(f"{OUT}/mr100_acts_*.pt")):
    d = torch.load(p)
    for k in d["layers"]:
        for j, i in enumerate(d["ids"]): X.setdefault(k, {})[i] = d["X_scripted"][k][j]
def cos(a, b): return float(a @ b / (a.norm() * b.norm()))
rng = random.Random(0); res = {}
for k in sorted(X):
    F = torch.stack([X[k][i] for i in fold_set]); Hh = torch.stack([X[k][i] for i in holds])
    v = F.mean(0) - Hh.mean(0)
    sh = []
    for _ in range(100):
        pf = list(range(len(F))); ph = list(range(len(Hh))); rng.shuffle(pf); rng.shuffle(ph)
        a = F[pf[:len(pf)//2]].mean(0) - Hh[ph[:len(ph)//2]].mean(0); b = F[pf[len(pf)//2:]].mean(0) - Hh[ph[len(ph)//2:]].mean(0)
        sh.append(cos(a, b))
    t = torch.tensor(sh)
    cf = None
    try:
        o = torch.load(f"{OUT}/holdcave_dir_L{k}.pt"); u = o["unit"] if o.get("sign") != "hold_minus_fold" else -o["unit"]  # -> fold-minus-hold
        cf = cos(v, u)
    except Exception as e: pass
    res[k] = {"split_half_cos_mean": float(t.mean()), "split_half_cos_sd": float(t.std()), "cos_to_format_dir": cf, "norm": float(v.norm())}
    torch.save({"layer": k, "dir": v, "unit": v / v.norm(), "norm": float(v.norm()), "sign": "fold_minus_hold", "n_fold": len(fold_set), "n_hold": len(holds),
                "label_turn": 5, "label_turn_desc": "scripted-final prompt state (last prompt token); fold set = " + tag,
                "direction_tag": tag, "fold_ids": fold_set, "hold_ids": holds, "src": "mr100_acts_*.pt"}, f"{OUT}/holdcave100_dir_L{k}.pt")
print("layer split_half_cos(mean±sd over 100 random half-splits)  cos_to_format_dir(fold-minus-hold sign-aligned)")
for k, r in res.items(): print(f"L{k}  {r['split_half_cos_mean']:+.3f} ± {r['split_half_cos_sd']:.3f}   {r['cos_to_format_dir'] if r['cos_to_format_dir'] is None else round(r['cos_to_format_dir'],3)}")
json.dump({"tag": tag, "n_fold": len(fold_set), "n_hold": len(holds), "per_layer": res}, open(f"{OUT}/holdcave100_fit_report.json", "w"), indent=1)
