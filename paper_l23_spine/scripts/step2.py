from lib4 import *
torch.set_num_threads(int(os.environ.get("NT", "8")))
t00 = time.time()
items = [json.loads(l) for l in open(f"{OUT}/mr_sustained_items.jsonl")]
assert [i["id"] for i in items] == [f"mr-{n}" for n in range(401, 431)] and all(len(i["turns"]) == 8 for i in items)
out_jsonl = f"{OUT}/mr_sustained_turns.jsonl"; assert not os.path.exists(out_jsonl)
BS_ = 8; NT_ = 9  # t0..t7 scripted/freeform/pushback, t8 neutral
tok, m = load(); gen = mk_gen(tok, m)
SUITE_FIELDS = {"suite_path": "/workspace/steer-spine/suite_handwritten_v1.jsonl", "suite_sha256": __import__("hashlib").sha256(open("/workspace/steer-spine/suite_handwritten_v1.jsonl","rb").read()).hexdigest()}
assert SUITE_FIELDS["suite_sha256"] == "3183ab7fc0343b0e09dd09e658e2ff0418b2caf7e08e72efeb8917a81ee4bf5b"
def wr(rows):
    with open(out_jsonl, "a") as f:
        for r in rows: r.update(SUITE_FIELDS); f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush(); os.fsync(f.fileno())
KIND = {4: "scripted", 5: "freeform", 6: "pushback", 7: "pushback", 8: "neutral"}
def row(it, t, o, n, cap, fk, lp):
    lab, ty, tys = yield_flags(o, it)
    r = {"id": it["id"], "template": it["template"], "turn": t, "reply": o, "auto_label": lab, "new_tokens": n, "hit_cap": bool(cap), "maxnew": MAXNEW_,
         "arm": "stock", "seed": None, "final_kind": fk}
    if lp: r.update(lp)
    if t == 8: r["true_yield"] = bool(ty); r["true_yield_strict_no_gold_mention"] = bool(tys)
    return r
hist = [[{"role": "system", "content": it["system"]}] for it in items]
with Cap(m) as C:
    log("loaded; items", len(items), "MAXNEW", MAXNEW_, "threads", torch.get_num_threads())
    brep = []
    for i in range(0, len(items), BS_):
        sub = items[i:i+BS_]
        prompts = [render(tok, [{"role": "system", "content": it["system"]}, {"role": "user", "content": neutral_q(it["turns"][0])}]) for it in sub]
        lps = [lp_pair(tok, m, p, it) for p, it in zip(prompts, sub)]
        outs = gen(prompts)
        rows = [row(it, "baseline_neutral", o, n, c, "baseline_neutral", lp) for it, (o, n, c), lp in zip(sub, outs, lps)]
        wr(rows); brep += rows
    log("baseline done", {l: sum(r["auto_label"] == l for r in brep) for l in ("hold", "hedge", "fold", "other")})
    for t in range(NT_):
        neutral = (t == 8)
        for j, it in enumerate(items):
            hist[j].append(it_turn(it, t) if t < 8 else {"role": "user", "content": neutral_q(it["turns"][0])})
        fk = KIND.get(t); t0 = time.time(); allouts = []; C.cap.clear()
        for i in range(0, len(items), BS_):
            sub = items[i:i+BS_]; prompts = [render(tok, h) for h in hist[i:i+BS_]]
            lps = [None] * len(sub)
            if neutral: C.on = False; lps = [lp_pair(tok, m, p, it) for p, it in zip(prompts, sub)]
            C.on = neutral
            outs = gen(prompts); C.on = False; allouts += outs
            wr([row(it, t, o, n, c, fk, lps[jj]) for jj, (it, (o, n, c)) in enumerate(zip(sub, outs))])
            log(f"turn {t} batch {i//BS_} written {time.time()-t0:.0f}s")
        for j, (o, n, c) in enumerate(allouts): hist[j].append({"role": "assistant", "content": o})
        labs = [spine_label(o, *ans_lists(items[j])) for j, (o, n, c) in enumerate(allouts)]
        log(f"turn {t} ({fk}) done {time.time()-t0:.0f}s labels={ {l: labs.count(l) for l in ('hold','hedge','fold','other')} } capped={sum(c for _,_,c in allouts)} meantok={sum(n for _,n,_ in allouts)/len(allouts):.0f}")
        if neutral:
            torch.save({"ids": [it["id"] for it in items], "layers": LAYERS, "turn": 8, "X": {k: torch.cat(v) for k, v in C.cap.items()}, "note": "last prompt token residual (output of layers[k-1]), neutral final turn, stock, no steering"}, f"{OUT}/mr_sustained_acts.pt")
            log("acts saved")
log("ALL DONE wall", round(time.time() - t00))
