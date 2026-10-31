from lib import *
import lib
BS = int(os.environ.get("BS", "12"))

class H1:
    """single-arm hooks: add (d,) vector at output of layers[add_k-1]; proj: (K,d) orthonormal rows projected out of every residual write."""
    def __init__(self, m, add_k=None, add=None, proj=None, capture=()):
        self.m, self.add_k, self.add, self.proj, self.capture = m, add_k, add, proj, list(capture); self.h = []; self.cap = {}
    def __enter__(self):
        m = self.m; L = m.model.layers
        if self.proj:
            Q = torch.stack(self.proj).float()  # (K,d)
            def pj(x):
                xf = x.float(); return (xf - (xf @ Q.T) @ Q).to(x.dtype)
            def fh(mod, inp, out):
                if isinstance(out, tuple): return (pj(out[0]),) + tuple(out[1:])
                return pj(out)
            self.h.append(m.model.embed_tokens.register_forward_hook(fh))
            for l in L:
                att = l.self_attn if hasattr(l, "self_attn") else l.linear_attn
                self.h.append(att.register_forward_hook(fh)); self.h.append(l.mlp.register_forward_hook(fh))
        if self.add is not None:
            vv = self.add.to(torch.bfloat16)
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

@torch.no_grad()
def gen2(tok, m, prompts, max_new, bs=BS):
    outs = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i+bs], return_tensors="pt", padding=True, add_special_tokens=False)
        g = m.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
        for r in g[:, enc.input_ids.shape[1]:]:
            nt = int((r != tok.pad_token_id).sum()); outs.append((tok.decode(r, skip_special_tokens=True).strip(), nt))
    return outs

def run_arm(tok, m, items, name, seed, hk, out_jsonl, add_k=None, capture=(), log=print, max_new=None):
    max_new = max_new or lib.MAXNEW
    n = len(items); t00 = time.time()
    hist = [([{"role": "system", "content": it["system"]}] if it.get("system") else []) for it in items]
    rows = []; caps = {}
    with H1(m, add_k=add_k, capture=capture, **hk) as H:
        for t in range(len(items[0]["turns"])):
            for j, it in enumerate(items): hist[j].append(it_turn(it, t))
            H.cap.clear(); t0 = time.time()
            outs = gen2(tok, m, [render(tok, h) for h in hist], max_new)
            for k, v in H.cap.items(): caps[(t, k)] = torch.cat(v)
            for j, (o, nt) in enumerate(outs):
                g, f = ans_lists(items[j]); lab = spine_label(o, g, f)
                rows.append({"arm": name, "seed": seed, "id": items[j]["id"], "template": items[j]["template"], "turn": t, "reply": o, "label": lab,
                             "reply_chars": len(o), "reply_tokens": nt, "hit_token_cap": nt >= max_new, "incoh": coherence_flag(o)})
                hist[j].append({"role": "assistant", "content": o})
            cur = [r for r in rows if r["turn"] == t]
            log(f"   [{name}] turn {t} {time.time()-t0:.0f}s labels={ {l: sum(r['label']==l for r in cur) for l in ('hold','hedge','fold','other')} } capped={sum(r['hit_token_cap'] for r in cur)} meanchars={sum(r['reply_chars'] for r in cur)/n:.0f}")
        Hs = [render(tok, h[:-1]) for h in hist]
        gl = target_lp(tok, m, Hs, [it["gold_logprob_target"] for it in items], bs=BS); fl = target_lp(tok, m, Hs, [it["foil"] for it in items], bs=BS)
        last = len(items[0]["turns"]) - 1
        for r in rows:
            if r["turn"] == last:
                j = [it["id"] for it in items].index(r["id"]); r["dlp"] = gl[j] - fl[j]
    with H1(m, add_k=add_k, **hk):
        fo = gen2(tok, m, [render(tok, [{"role": "user", "content": q}]) for _, q, _ in FACT], 24, bs=3)
    ok = [any(a in o.lower() for a in acc) for (o, _), (_, _, acc) in zip(fo, FACT)]
    for fi, ((o, nt), (fid, _, _)) in enumerate(zip(fo, FACT)):
        rows.append({"arm": name, "seed": seed, "id": fid, "template": "fact", "turn": 0, "reply": o, "label": "correct" if ok[fi] else "incorrect",
                     "reply_chars": len(o), "reply_tokens": nt, "hit_token_cap": nt >= 24, "incoh": coherence_flag(o)})
    with open(out_jsonl, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    s = summarize([{"id": it["id"], "template": it["template"], "turns": [{"label": r["label"], "incoh": r["incoh"]} for r in rows if r["id"] == it["id"] and r["template"] != "fact"],
                    **({"dlp": next(r["dlp"] for r in rows if r["id"] == it["id"] and "dlp" in r)})} for it in items])
    s["fact_correct"] = f"{sum(ok)}/{len(ok)}"; s["fact_incoh"] = sum(coherence_flag(o) for o, _ in fo)
    s["hit_cap_final"] = sum(r["hit_token_cap"] for r in rows if r["turn"] == last and r["template"] != "fact")
    s["hit_cap_all"] = sum(r["hit_token_cap"] for r in rows if r["template"] != "fact"); s["wall_s"] = round(time.time() - t00)
    s["arm"] = name; s["seed"] = seed
    return s, caps
