from lib import *

class RowHooks:
    """Per-row interventions in one batch. addv: (R,d) float or None (added at output of layers[add_k-1], all positions).
    projQ: (R,K,d) orthonormal rows (zero rows = none) projected out of every residual write (embed, attn, mlp), all layers."""
    def __init__(self, m, add_k=None, addv=None, projQ=None, capture=()):
        self.m, self.add_k, self.addv, self.projQ, self.capture = m, add_k, addv, projQ, list(capture)
        self.h = []; self.cap = {}
    def __enter__(self):
        m = self.m; L = m.model.layers
        if self.projQ is not None:
            Q = self.projQ.float()
            def proj(x):
                xf = x.float(); c = torch.einsum("rtd,rkd->rtk", xf, Q)
                return (xf - torch.einsum("rtk,rkd->rtd", c, Q)).to(x.dtype)
            def fh(mod, inp, out):
                if isinstance(out, tuple): return (proj(out[0]),) + tuple(out[1:])
                return proj(out)
            self.h.append(m.model.embed_tokens.register_forward_hook(fh))
            for l in L:
                att = l.self_attn if hasattr(l, "self_attn") else l.linear_attn
                self.h.append(att.register_forward_hook(fh)); self.h.append(l.mlp.register_forward_hook(fh))
        if self.addv is not None:
            vv = self.addv.to(torch.bfloat16)[:, None, :]
            def ah(mod, inp, out):
                if isinstance(out, tuple): return (out[0] + vv,) + tuple(out[1:])
                return out + vv
            self.h.append(L[self.add_k - 1].register_forward_hook(ah))
        for k in self.capture:
            def ch(mod, inp, out, k=k):
                o = out[0] if isinstance(out, tuple) else out
                if o.shape[1] > 1: self.cap.setdefault(k, []).append(o[:, -1].float().cpu())
            self.h.append(L[k - 1].register_forward_hook(ch))
        return self
    def __exit__(self, *a):
        for h in self.h: h.remove()
        self.h = []

def build_rows(arms, reps, d, K=4):
    """arms: list of dict(name, seed, add(vec or None), proj(list of unit vecs)). each arm repeated reps times (rows arm-major)."""
    R = len(arms) * reps
    addv = torch.zeros(R, d); P = torch.zeros(R, K, d)
    for a, arm in enumerate(arms):
        for r in range(reps):
            i = a * reps + r
            if arm.get("add") is not None: addv[i] = arm["add"]
            for q, v in enumerate(arm.get("proj") or []): P[i, q] = v
    return addv, P

def run_arms(tok, m, items, arms, add_k, capture=(), log=print):
    n = len(items); R = len(arms) * n
    rows = [(a, items[j]) for a in range(len(arms)) for j in range(n)]
    addv, P = build_rows(arms, n, arms[0]["d"])
    hist = [([{"role": "system", "content": it["system"]}] if it.get("system") else []) for _, it in rows]
    recs = [{"arm": arms[a]["name"], "seed": arms[a].get("seed"), "id": it["id"], "template": it["template"], "turns": []} for a, it in rows]
    nt = len(items[0]["turns"]); assert all(len(it["turns"]) == nt for it in items)
    caps = {}
    with RowHooks(m, add_k=add_k, addv=addv if any(a.get("add") is not None for a in arms) else None,
                  projQ=P if any(a.get("proj") for a in arms) else None, capture=capture) as H:
        for t in range(nt):
            for i, (_, it) in enumerate(rows): hist[i].append(it_turn(it, t))
            prompts = [render(tok, hist[i]) for i in range(R)]
            t0 = time.time(); H.cap.clear()
            outs = gen(tok, m, prompts, bs=R)
            for k, v in H.cap.items(): caps[(t, k)] = torch.cat(v)
            for i, o in enumerate(outs):
                g, f = ans_lists(rows[i][1]); lab = spine_label(o, g, f)
                recs[i]["turns"].append({"t": t, "reply": o, "label": lab, "incoh": coherence_flag(o)})
                hist[i].append({"role": "assistant", "content": o})
            log(f"    turn {t} done {time.time()-t0:.0f}s R={R}")
        t0 = time.time()
        Hs = [render(tok, hist[i][:-1]) for i in range(R)]
        gl = target_lp(tok, m, Hs, [it["gold_logprob_target"] for _, it in rows], bs=R)
        fl = target_lp(tok, m, Hs, [it["foil"] for _, it in rows], bs=R)
        for i in range(R): recs[i]["dlp"] = gl[i] - fl[i]; recs[i]["final_gold_lp"] = gl[i]; recs[i]["final_foil_lp"] = fl[i]
        log(f"    dlp {time.time()-t0:.0f}s")
    # facts: 3 rows per arm
    nf = len(FACT)
    fadd = addv[::n].repeat_interleave(nf, 0); fP = P[::n].repeat_interleave(nf, 0)
    fr = {}
    with RowHooks(m, add_k=add_k, addv=fadd if any(a.get("add") is not None for a in arms) else None,
                  projQ=fP if any(a.get("proj") for a in arms) else None):
        ps = [render(tok, [{"role": "user", "content": q}]) for _ in arms for _, q, _ in FACT]
        outs = gen(tok, m, ps, max_new=24, bs=len(ps))
    for a, arm in enumerate(arms):
        o = outs[a * nf:(a + 1) * nf]
        ok = [any(x in oo.lower() for x in acc) for oo, (_, _, acc) in zip(o, FACT)]
        fr[a] = {"correct": sum(ok), "n": nf, "ok": ok, "incoh": sum(coherence_flag(x) for x in o), "replies": o}
    return recs, fr, caps
