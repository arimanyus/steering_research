from lib2 import *
t00 = time.time()
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
ext, ev = items_split(); tag = "swap" if os.environ.get("SWAP") == "1" else "main"
D = torch.load(f"{OUT}/extract_acts.pt"); meta = D["cells_meta"]; X = D["X"]; LAYERS = sorted(X)
def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    return float(((pos[:, None] > neg[None, :]).float() + 0.5 * (pos[:, None] == neg[None, :]).float()).mean())
half = {j: int(hashlib.sha1(ext[j]["id"].encode()).hexdigest(), 16) % 2 for j in range(len(ext))}
sel = [i for i, (j, t, l) in enumerate(meta) if l in ("hold", "fold")]
lab = torch.tensor([1 if meta[i][2] == "fold" else 0 for i in sel]); hf = torch.tensor([half[meta[i][0]] for i in sel])
allt = torch.tensor([t for (_, t, _) in meta])
log("cells hold/fold", int((lab == 0).sum()), int((lab == 1).sum()))
def turn_center(A):
    A = A.clone()
    for t in allt.unique():
        mk = allt == t; A[mk] -= A[mk].mean(0)
    return A
res = {}
for k in LAYERS:
    for var in ("raw", "turnc"):
        A = turn_center(X[k]) if var == "turnc" else X[k]; A = A[sel]; aucs = []
        for h in (0, 1):
            tr, te = hf != h, hf == h
            if int(((lab == 1) & tr).sum()) < 1 or int(((lab == 0) & tr).sum()) < 1: continue
            d = A[tr & (lab == 1)].mean(0) - A[tr & (lab == 0)].mean(0); d = d / d.norm()
            p = A[te] @ d; aucs.append(auc(p[lab[te] == 1], p[lab[te] == 0]))
        res[(k, var)] = sum(aucs) / max(1, len(aucs)) if aucs else float("nan")
best = max(res, key=lambda x: (res[x] if res[x] == res[x] else -1)); BL, BV = best
log("top CV AUC:", sorted(res.items(), key=lambda x: -(x[1] if x[1]==x[1] else -1))[:6])
A = (turn_center(X[BL]) if BV == "turnc" else X[BL])[sel]
dvec = A[lab == 1].mean(0) - A[lab == 0].mean(0); dnorm = float(dvec.norm()); unit = dvec / dvec.norm(); med = float(X[BL].norm(dim=1).median())
log(f"layer(hs idx) {BL} variant {BV} cvAUC {res[best]:.3f} |d|={dnorm:.2f} medresid={med:.2f} ratio={dnorm/med:.3f}")
torch.save(unit, f"{OUT}/holdcave_dir_L{BL}.pt")
torch.save({"unit": unit, "diff_norm": dnorm, "median_resid_norm": med, "layer_hs_index": BL, "variant": BV, "sign": "toward cave (fold-hold)"}, f"{OUT}/holdcave_dir_L{BL}_meta.pt")
json.dump({"cv_auc": {f"L{k}_{v}": a for (k, v), a in res.items()}, "layer": BL, "variant": BV, "diff_norm": dnorm, "median_norm": med,
           "n_hold_cells": int((lab == 0).sum()), "n_fold_cells": int((lab == 1).sum())}, open(f"{OUT}/direction_select.json", "w"), indent=1)
tok, m = load(); log("model loaded"); d = unit.shape[0]
dm = torch.load(f"{SW}/tbL/dmeans_templ.pt"); r23 = torch.load(f"{SW}/refusal_dir_L23.pt").float()
Qs, _ = torch.linalg.qr(torch.stack([dm[20].float(), r23, dm[25].float()]).T); STACK = [Qs[:, i].contiguous() for i in range(3)]
def orth(vs):
    q, _ = torch.linalg.qr(torch.stack(vs).T); return [q[:, i].contiguous() for i in range(len(vs))]
def rand_unit(seed):
    g = torch.Generator().manual_seed(seed); r = torch.randn(d, generator=g); return r / r.norm()
def mk(name, add=None, proj=None, seed=None): return {"name": name, "seed": seed, "add": add, "proj": proj, "d": d}
def finish(recs, fr, arms, n, fn):
    out = {}
    for a, arm in enumerate(arms):
        rr = recs[a*n:(a+1)*n]; s = summarize(rr); f = fr[a]
        s["fact_correct"] = f"{f['correct']}/{f['n']}"; s["fact_incoh"] = f["incoh"]; s["seed"] = arm["seed"]
        out[arm["name"]] = s
    json.dump({"summary": out, "recs": recs, "fact": {arms[a]["name"]: fr[a] for a in fr}}, open(fn, "w")); return out
n = len(ev); ALL = {}
def run_group(gname, arms, capture=()):
    fn = f"{OUT}/eval_{tag}_{gname}.json"
    if os.path.exists(fn): log("skip", gname); return json.load(open(fn))["summary"]
    log("GROUP", gname, "arms", len(arms), "rows", len(arms) * n); t0 = time.time()
    recs, fr, caps = run_arms(tok, m, ev, arms, BL, capture=capture, log=log)
    if capture: torch.save(caps, f"{OUT}/eval_{tag}_{gname}_caps.pt")
    o = finish(recs, fr, arms, n, fn); log("GROUP DONE", gname, round(time.time() - t0), "s")
    for k, v in o.items(): log("  ", k, {kk: vv for kk, vv in v.items()})
    return o
# phase 1: stock, holdcave add sweep, ablate, random ablate, stack, stack+add sweep
arms1 = [mk("stock")] + [mk(f"hc_add_a{a}", add=unit * a * dnorm) for a in (2, 4, 8)] + \
        [mk("hc_ablate", proj=[unit]), mk("rand_ablate_s2000", proj=[rand_unit(2000)], seed=2000), mk("stack", proj=STACK)] + \
        [mk(f"stack_hc_add_a{a}", add=unit * a * dnorm, proj=STACK) for a in (2, 4, 8)] + \
        [mk("stack_hc_ablate", proj=orth(STACK + [unit]))]
P1 = run_group("phase1", arms1, capture=[BL]); ALL.update(P1)
cand = [a for a in (2, 4, 8) if P1[f"hc_add_a{a}"]["incoh_final"] <= 3 and P1[f"hc_add_a{a}"]["fact_correct"].startswith("3")]
ba = max(cand, key=lambda a: P1[f"hc_add_a{a}"]["final_fold"]) if cand else 4
log("best hold-vs-cave alpha:", ba, "cands", cand)
# phase 2: 10 random seeds at best alpha, alone and with stack (norm-matched: unit rand * ba * dnorm)
for part, (lo, hi) in enumerate([(0, 5), (5, 10)]):
    arms2 = [mk(f"rand_add_a{ba}_s{2000+s}", add=rand_unit(2000 + s) * ba * dnorm, seed=2000 + s) for s in range(lo, hi)] + \
            [mk(f"stack_rand_add_a{ba}_s{2000+s}", add=rand_unit(2000 + s) * ba * dnorm, proj=STACK, seed=2000 + s) for s in range(lo, hi)]
    ALL.update(run_group(f"phase2_{part}", arms2))
json.dump({"layer": BL, "best_alpha": ba, "summary": ALL}, open(f"{OUT}/results_{tag}.json", "w"), indent=1)
log("ALL DONE wall", round(time.time() - t00))
