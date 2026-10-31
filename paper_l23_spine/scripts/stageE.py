import os
os.environ["MAXNEW"] = "512"
from lib3 import *
import lib
assert lib.MAXNEW == 512, lib.MAXNEW
MAXNEW_ = 512
torch.set_num_threads(int(os.environ.get("NT", "4")))
t00 = time.time()
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
ext, ev = items_split()
tok, m = load()
assert lib.MAXNEW == 512
EOS = set(x for x in ([m.generation_config.eos_token_id] if isinstance(m.generation_config.eos_token_id, int) else (m.generation_config.eos_token_id or [])))
EOS |= {tok.eos_token_id, tok.pad_token_id}; EOS.discard(None)
log("loaded; ext", len(ext), "eval", len(ev), "BS", BS, "MAXNEW", lib.MAXNEW, "eos ids", sorted(EOS), "threads", torch.get_num_threads())
LAYERS = list(range(10, 29))

@torch.no_grad()
def gen3(prompts, bs=BS):
    outs = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i+bs], return_tensors="pt", padding=True, add_special_tokens=False)
        g = m.generate(**enc, max_new_tokens=MAXNEW_, do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
        for r in g[:, enc.input_ids.shape[1]:]:
            ids = r.tolist(); n = len(ids); hit_eos = False
            for p, x in enumerate(ids):
                if x in EOS: n = p; hit_eos = True; break
            outs.append((tok.decode(ids[:n], skip_special_tokens=True).strip(), n, not hit_eos and n >= MAXNEW_))
    return outs

def run_set(items, out_jsonl, tag):
    if os.path.exists(out_jsonl): os.rename(out_jsonl, out_jsonl + f".old{int(time.time())}")
    hist = [([{"role": "system", "content": it["system"]}] if it.get("system") else []) for it in items]
    nt = len(items[0]["turns"]); caps = {}; labs = {}
    with H1(m, capture=LAYERS) as H:
        for t in range(nt):
            for j, it in enumerate(items): hist[j].append(it_turn(it, t))
            H.cap.clear(); t0 = time.time()
            outs = gen3([render(tok, h) for h in hist])
            for k, v in H.cap.items(): caps[(t, k)] = torch.cat(v)
            with open(out_jsonl, "a") as f:
                for j, (o, n, cap) in enumerate(outs):
                    g, fo = ans_lists(items[j]); lab = spine_label(o, g, fo)
                    labs[(items[j]["id"], t)] = lab
                    f.write(json.dumps({"id": items[j]["id"], "template": items[j]["template"], "turn": t, "reply": o, "auto_label": lab,
                                        "new_tokens": n, "hit_cap": bool(cap), "maxnew": MAXNEW_, "arm": "stock", "seed": None}) + "\n")
                    hist[j].append({"role": "assistant", "content": o})
                f.flush(); os.fsync(f.fileno())
            cur = [labs[(it["id"], t)] for it in items]
            log(f"[{tag}] turn {t} done {time.time()-t0:.0f}s rows written -> {out_jsonl} labels={ {l: cur.count(l) for l in ('hold','hedge','fold','other')} } capped={sum(c for _,_,c in outs)} meantok={sum(n for _,n,_ in outs)/len(outs):.0f}")
    # AFTER all rows are on disk: activations then logprob pass
    last = nt - 1
    meta = []; X = {k: [] for k in LAYERS}
    for t in range(nt):
        for j, it in enumerate(items):
            meta.append((j, t, labs[(it["id"], t)]))
            for k in LAYERS: X[k].append(caps[(t, k)][j])
    torch.save({"cells_meta": meta, "X": {k: torch.stack(v) for k, v in X.items()}}, f"{OUT}/{tag}_acts_512.pt")
    log(f"[{tag}] acts saved")
    try:
        Hs = [render(tok, h[:-1]) for h in hist]
        gl = target_lp(tok, m, Hs, [it["gold_logprob_target"] for it in items], bs=BS); fl = target_lp(tok, m, Hs, [it["foil"] for it in items], bs=BS)
        json.dump({it["id"]: {"gold_lp": gl[j], "foil_lp": fl[j], "dlp": gl[j]-fl[j]} for j, it in enumerate(items)}, open(f"{OUT}/{tag}_dlp_512.json", "w"))
        log(f"[{tag}] dlp saved")
    except Exception as e: log(f"[{tag}] dlp failed {e!r}")
    return labs

run_set(ext, f"{OUT}/stock_extract_turns.jsonl", "extract")
run_set(ev, f"{OUT}/stock_eval_turns.jsonl", "eval")
log("ALL DONE wall", round(time.time()-t00))
