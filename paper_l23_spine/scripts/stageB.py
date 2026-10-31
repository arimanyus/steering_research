from lib import *
t00 = time.time()
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
ext, ev = items_split(); tag = "swap" if os.environ.get("SWAP") == "1" else "main"
D = torch.load(f"{OUT}/extract_acts{'_swap' if tag=='swap' else ''}.pt") if False else torch.load(f"{OUT}/extract_acts.pt")
meta = D["cells_meta"]; X = D["X"]
LAYERS = sorted(X)
def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    return float(((pos[:, None] > neg[None, :]).float() + 0.5 * (pos[:, None] == neg[None, :]).float()).mean())
# fold-halves of extract items (by position parity of hashed id)
half = {j: int(hashlib.sha1(ext[j]["id"].encode()).hexdigest(), 16) % 2 for j in range(len(ext))}
sel = [i for i, (j, t, l) in enumerate(meta) if l in ("hold", "fold")]
lab = torch.tensor([1 if meta[i][2] == "fold" else 0 for i in sel]); hf = torch.tensor([half[meta[i][0]] for i in sel])
tt = torch.tensor([meta[i][1] for i in sel])
log("cells hold/fold", int((lab == 0).sum()), int((lab == 1).sum()), "per half fold", [int(((lab == 1) & (hf == h)).sum()) for h in (0, 1)])
def turn_center(A, tt_all):
    A = A.clone()
    for t in tt_all.unique():
        mk = tt_all == t; A[mk] -= A[mk].mean(0)
    return A
allt = torch.tensor([t for (_, t, _) in meta])
res = {}
for k in LAYERS:
    for var in ("raw", "turnc"):
        A = X[k]; A = turn_center(A, allt) if var == "turnc" else A
        A = A[sel]; aucs = []
        for h in (0, 1):
            tr, te = hf != h, hf == h
            if int(((lab == 1) & tr).sum()) < 1 or int(((lab == 0) & tr).sum()) < 1: continue
            d = A[tr & (lab == 1)].mean(0) - A[tr & (lab == 0)].mean(0); d = d / d.norm()
            p = A[te] @ d; aucs.append(auc(p[lab[te] == 1], p[lab[te] == 0]))
        res[(k, var)] = sum(aucs) / max(1, len(aucs))
best = max(res, key=lambda x: (res[x] if res[x] == res[x] else -1))
log("heldout AUC (item-half CV) top:", sorted(res.items(), key=lambda x: -(x[1] if x[1]==x[1] else -1))[:6])
BL, BV = best
A = X[BL]; A = turn_center(A, allt) if BV == "turnc" else A
A = A[sel]
dvec = A[lab == 1].mean(0) - A[lab == 0].mean(0); dnorm = float(dvec.norm()); unit = dvec / dvec.norm()
med = float(X[BL].norm(dim=1).median())
log(f"chosen layer(hs idx) {BL} variant {BV} cvAUC {res[best]:.3f} |d|={dnorm:.2f} median last-tok resid norm={med:.2f} ratio={dnorm/med:.3f}")
torch.save(unit, f"{OUT}/holdcave_dir_L{BL}.pt")
torch.save({"unit": unit, "diff_norm": dnorm, "median_resid_norm": med, "layer_hs_index": BL, "variant": BV, "sign": "toward cave (fold - hold)"}, f"{OUT}/holdcave_dir_L{BL}_meta.pt")
json.dump({"cv_auc": {f"L{k}_{v}": a for (k, v), a in res.items()}, "layer": BL, "variant": BV, "diff_norm": dnorm, "median_norm": med,
           "n_hold_cells": int((lab == 0).sum()), "n_fold_cells": int((lab == 1).sum())}, open(f"{OUT}/direction_select.json", "w"), indent=1)

tok, m = load(); log("model loaded")
g = torch.Generator().manual_seed(1000); rnd = torch.randn(unit.shape[0], generator=g); rnd = rnd / rnd.norm()
# refusal stack: saved L23 dir, + dmeans (templ) L20, L25 ; QR
dm = torch.load(f"{SW}/tbL/dmeans_templ.pt"); r23 = torch.load(f"{SW}/refusal_dir_L23.pt").float()
Qs, _ = torch.linalg.qr(torch.stack([dm[20].float(), r23, dm[25].float()]).T); Qstack = Qs.T.contiguous()
Q1 = (unit / unit.norm())[None]; Qr = rnd[None]
def cond_run(name, hk, capture=()):
    fn = f"{OUT}/eval_{tag}_{name}.json"
    if os.path.exists(fn): log("skip", name); return json.load(open(fn))
    t0 = time.time(); log("RUN", name)
    recs, caps = run_dialogues(tok, m, ev, hookkw=hk, log=log, capture=capture)
    fact = run_fact(tok, m, hk)
    s = summarize(recs); s["fact_correct"] = f'{fact["correct"]}/{fact["n"]}'; s["fact_incoh"] = fact["incoh"]; s["wall_s"] = round(time.time() - t0)
    json.dump({"summary": s, "recs": recs, "fact": fact}, open(fn, "w"))
    if capture: torch.save({"cells_meta": None, "caps": {k: (a, v) for k, (a, v) in caps.items()}}, f"{OUT}/eval_{tag}_{name}_caps.pt")
    log("DONE", name, {k: v for k, v in s.items()}); return {"summary": s}
S = {}
def put(k, name, hk, capture=()): S[k] = cond_run(name, hk, capture)
def dump(): json.dump({k: v["summary"] for k, v in S.items()}, open(f"{OUT}/results_{tag}.json", "w"), indent=1)
put("stock", "stock", {}, capture=[BL]); dump()
for a in (4, 8): put(f"add_a{a}", f"add_a{a}", {"add": [(BL, unit, a * dnorm)]}); dump()
put("rand_a8", "rand_a8", {"add": [(BL, rnd, 8 * dnorm)]}); dump()
put("ablate", "ablate_dir", {"Q": Q1}); dump()
put("rand_ablate", "rand_ablate", {"Q": Qr}); dump()
put("stack", "stack_L20_23_25", {"Q": Qstack}); dump()
cand = [a for a in (4, 8) if S[f"add_a{a}"]["summary"]["incoh_final"] <= 2 and S[f"add_a{a}"]["summary"]["fact_correct"].startswith("3")]
ba = max(cand, key=lambda a: S[f"add_a{a}"]["summary"]["final_fold"]) if cand else 4
log("best alpha for combined:", ba)
put("stack_add", f"stack_add_a{ba}", {"Q": Qstack, "add": [(BL, unit, ba * dnorm)]}); dump()
put("stack_ablate", "stack_ablate_dir", {"Q": torch.linalg.qr(torch.cat([Qstack, Q1]).T)[0].T.contiguous()}); dump()
put("add_a2", "add_a2", {"add": [(BL, unit, 2 * dnorm)]}); dump()
put("rand_a4", "rand_a4", {"add": [(BL, rnd, 4 * dnorm)]}); dump()
put("stack_rand_add", f"stack_rand_a{ba}", {"Q": Qstack, "add": [(BL, rnd, ba * dnorm)]}); dump()
log("ALL DONE wall", round(time.time() - t00))
