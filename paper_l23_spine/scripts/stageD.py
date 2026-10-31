from lib3 import *
t00 = time.time()
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
ext, ev = items_split(); tag = "swap" if os.environ.get("SWAP") == "1" else "main"
LAYERS = list(range(10, 29)); n = len(ev)
tok, m = load(); log("loaded; ext", len(ext), "eval", len(ev), "BS", BS, "MAXNEW", lib.MAXNEW, "threads", torch.get_num_threads())
RES = f"{OUT}/summaries_{tag}.json"; SUM = json.load(open(RES)) if os.path.exists(RES) else {}
def save(): json.dump(SUM, open(RES, "w"), indent=1)
def do(name, items, hk, seed=None, add_k=None, capture=(), fn=None):
    fn = fn or f"{OUT}/arm_{tag}_{name}.jsonl"
    if os.path.exists(fn) and name in SUM: log("skip", name); return None
    log("ARM", name); s, caps = run_arm(tok, m, items, name, seed, hk, fn, add_k=add_k, capture=capture, log=log)
    SUM[name] = s; save(); log("DONE", name, {k: v for k, v in s.items()}); return caps
# 2. stock eval -> stock_eval_turns.jsonl
evf = f"{OUT}/eval_acts_{tag}.pt"
if not os.path.exists(f"{OUT}/stock_eval_turns.jsonl") or tag == "swap":
    caps = do("stock_eval", ev, {}, capture=LAYERS, fn=f"{OUT}/stock_eval_turns{'' if tag=='main' else '_swap'}.jsonl")
    if caps:
        rows = [json.loads(l) for l in open(f"{OUT}/stock_eval_turns{'' if tag=='main' else '_swap'}.jsonl") if json.loads(l)["template"] != "fact"]
        lab = {(r["id"], r["turn"]): r["label"] for r in rows}; meta = []; X = {k: [] for k in LAYERS}
        for t in range(len(ev[0]["turns"])):
            for j, it in enumerate(ev):
                meta.append((j, t, lab[(it["id"], t)]))
                for k in LAYERS: X[k].append(caps[(t, k)][j])
        torch.save({"cells_meta": meta, "X": {k: torch.stack(v) for k, v in X.items()}}, evf)
# 1. extraction run (stock on extract family)
ef = f"{OUT}/extract_acts_{tag}.pt"
if not os.path.exists(ef):
    caps = do("stock_extract", ext, {}, capture=LAYERS, fn=f"{OUT}/stock_extract_turns_{tag}.jsonl")
    rows = [json.loads(l) for l in open(f"{OUT}/stock_extract_turns_{tag}.jsonl") if json.loads(l)["template"] != "fact"]
    lab = {(r["id"], r["turn"]): r["label"] for r in rows}
    meta = []; X = {k: [] for k in LAYERS}
    for t in range(len(ext[0]["turns"])):
        for j, it in enumerate(ext):
            meta.append((j, t, lab[(it["id"], t)]))
            for k in LAYERS: X[k].append(caps[(t, k)][j])
    torch.save({"cells_meta": meta, "X": {k: torch.stack(v) for k, v in X.items()}}, ef); log("saved extract acts")
# 3. direction
D = torch.load(ef); meta = D["cells_meta"]; X = D["X"]
def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    return float(((pos[:, None] > neg[None, :]).float() + 0.5 * (pos[:, None] == neg[None, :]).float()).mean())
half = {j: int(hashlib.sha1(ext[j]["id"].encode()).hexdigest(), 16) % 2 for j in range(len(ext))}
sel = [i for i, (j, t, l) in enumerate(meta) if l in ("hold", "fold")]
lab = torch.tensor([1 if meta[i][2] == "fold" else 0 for i in sel]); hf = torch.tensor([half[meta[i][0]] for i in sel])
allt = torch.tensor([t for (_, t, _) in meta])
log("extract cells hold/fold", int((lab == 0).sum()), int((lab == 1).sum()), "folds per item-half", [int(((lab == 1) & (hf == h)).sum()) for h in (0, 1)])
def tc(A, tt):
    A = A.clone()
    for t in tt.unique():
        mk = tt == t; A[mk] -= A[mk].mean(0)
    return A
res = {}
for k in LAYERS:
    for var in ("raw", "turnc"):
        A = (tc(X[k], allt) if var == "turnc" else X[k])[sel]; aucs = []
        for h in (0, 1):
            tr, te = hf != h, hf == h
            if int(((lab == 1) & tr).sum()) < 1 or int(((lab == 0) & tr).sum()) < 1 or int(((lab == 1) & te).sum()) < 1: continue
            d = A[tr & (lab == 1)].mean(0) - A[tr & (lab == 0)].mean(0); d = d / d.norm(); p = A[te] @ d
            aucs.append(auc(p[lab[te] == 1], p[lab[te] == 0]))
        res[(k, var)] = sum(aucs) / len(aucs) if aucs else float("nan")
best = max(res, key=lambda x: (res[x] if res[x] == res[x] else -1)); BL, BV = best
A = (tc(X[BL], allt) if BV == "turnc" else X[BL])[sel]
dvec = A[lab == 1].mean(0) - A[lab == 0].mean(0); dnorm = float(dvec.norm()); unit = dvec / dvec.norm(); med = float(X[BL].norm(dim=1).median())
# held-out-family check on eval-set stock cells
tr_auc = None
if os.path.exists(evf):
    E = torch.load(evf); em = E["cells_meta"]; es = [i for i, (j, t, l) in enumerate(em) if l in ("hold", "fold")]
    el = torch.tensor([1 if em[i][2] == "fold" else 0 for i in es]); p = E["X"][BL][es] @ unit
    tr_auc = auc(p[el == 1], p[el == 0]); log("transfer AUC on eval-family stock cells", tr_auc, "n hold/fold", int((el == 0).sum()), int((el == 1).sum()))
log(f"layer(hs idx) {BL} var {BV} cvAUC {res[best]:.3f} |d|={dnorm:.2f} medresid={med:.2f} ratio={dnorm/med:.3f}")
torch.save(unit, f"{OUT}/holdcave_dir_L{BL}.pt")
json.dump({"cv_auc": {f"L{k}_{v}": a for (k, v), a in res.items()}, "layer_hs_index": BL, "variant": BV, "diff_norm": dnorm, "median_resid_norm": med,
           "n_hold_cells": int((lab == 0).sum()), "n_fold_cells": int((lab == 1).sum()), "transfer_auc_eval_family": tr_auc, "sign": "toward cave (fold-hold)"}, open(f"{OUT}/direction_select_{tag}.json", "w"), indent=1)
if int((lab == 1).sum()) < 4: log("TOO FEW FOLD CELLS; stopping after direction"); sys.exit(0)
d = unit.shape[0]
dm = torch.load(f"{SW}/tbL/dmeans_templ.pt"); r23 = torch.load(f"{SW}/refusal_dir_L23.pt").float()
Qs, _ = torch.linalg.qr(torch.stack([dm[20].float(), r23, dm[25].float()]).T); STACK = [Qs[:, i].contiguous() for i in range(3)]
def orth(vs):
    q, _ = torch.linalg.qr(torch.stack(vs).T); return [q[:, i].contiguous() for i in range(len(vs))]
def ru(seed):
    g = torch.Generator().manual_seed(seed); r = torch.randn(d, generator=g); return r / r.norm()
# 4. arms
for a in (4, 8, 2): do(f"hc_add_a{a}", ev, {"add": unit * a * dnorm}, add_k=BL)
do("hc_ablate", ev, {"proj": [unit]})
do("stack", ev, {"proj": STACK})
cand = [a for a in (2, 4, 8) if SUM[f"hc_add_a{a}"]["incoh_final"] <= 3 and SUM[f"hc_add_a{a}"]["fact_correct"].startswith("3")]
ba = max(cand, key=lambda a: SUM[f"hc_add_a{a}"]["final_fold"]) if cand else 4
log("best hold-vs-cave alpha", ba, "cands", cand); SUM["_best_alpha"] = ba; SUM["_layer"] = BL; save()
do(f"stack_hc_add_a{ba}", ev, {"add": unit * ba * dnorm, "proj": STACK}, add_k=BL)
for s in range(10):
    do(f"rand_add_a{ba}_s{2000+s}", ev, {"add": ru(2000 + s) * ba * dnorm}, seed=2000 + s, add_k=BL)
    do(f"stack_rand_add_a{ba}_s{2000+s}", ev, {"add": ru(2000 + s) * ba * dnorm, "proj": STACK}, seed=2000 + s, add_k=BL)
do("stack_hc_ablate", ev, {"proj": orth(STACK + [unit])})
do("rand_ablate_s2000", ev, {"proj": [ru(2000)]}, seed=2000)
log("ALL DONE wall", round(time.time() - t00))
