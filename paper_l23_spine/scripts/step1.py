# usage: python lib4-based step1.py ARM [SEED]  arm in stack|randstack|stock_neutral|baselines
import sys
from lib4 import *
torch.set_num_threads(int(os.environ.get("NT", "4")))
t00 = time.time()
mode = sys.argv[1]; seed = int(sys.argv[2]) if len(sys.argv) > 2 else None
ext, ev = items_split()
assert [i["id"] for i in ev] == json.load(open(f"{OUT}/split.json"))["eval_ids"] and all(i["template"] == "false_premise" for i in ev)
items = ev; BS_ = 8
arm = {"stack": "stack", "randstack": "randstack"}[mode]
out_jsonl = f"{OUT}/arm_stack_fp_turns.jsonl" if mode == "stack" else f"{OUT}/arm_randstack_fp_s{seed}.jsonl"
assert not os.path.exists(out_jsonl), out_jsonl
Q = stack_Q() if mode == "stack" else randstack_Q(seed)
import hashlib, transformers
sys.path.insert(0, "/workspace/steer-withheld/tbL")
if mode == "stack":
    import run as TBL_RUN   # tbL harness; assert our stack basis is exactly its stack_Q()
    assert torch.equal(Q, TBL_RUN.stack_Q()), "stack Q differs from tbL run.py stack_Q"
_sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
_D = torch.load(f"{SW}/tbL/dirs_final.pt")
PROV = {"stack_construction": "tbL/run.py stack_Q(): unit L20,L23,L25 diff-of-means from dirs_final.pt -> QR; projected out of embed/attn/mlp outputs at all layers+positions (Steer-equivalent H1 proj)",
        "dirs_final_pt": f"{SW}/tbL/dirs_final.pt", "dirs_final_sha256": _sha(f"{SW}/tbL/dirs_final.pt"),
        "dircfg_json": f"{SW}/tbL/dircfg.json", "dircfg_sha256": _sha(f"{SW}/tbL/dircfg.json"),
        "per_layer_vector_sha256": {k: hashlib.sha256(v.float().contiguous().numpy().tobytes()).hexdigest() for k, v in _D.items()},
        "random_seed_scheme": None if mode == "stack" else "torch.Generator().manual_seed(2000+s), randn(3,2560) row-normalised, QR",
        "env": {"torch": torch.__version__, "transformers": transformers.__version__, "dtype": "bfloat16", "threads": torch.get_num_threads(), "gen_batch": 8, "max_new": 512}}
tok, m = load(); gen = mk_gen(tok, m)
nscript = len(items[0]["turns"]); assert nscript == 5
SUITE_FIELDS = {"suite_path": "/workspace/steer-spine/suite_handwritten_v1.jsonl", "suite_sha256": __import__("hashlib").sha256(open("/workspace/steer-spine/suite_handwritten_v1.jsonl","rb").read()).hexdigest()}
assert SUITE_FIELDS["suite_sha256"] == "3183ab7fc0343b0e09dd09e658e2ff0418b2caf7e08e72efeb8917a81ee4bf5b"
def wr(rows):
    with open(out_jsonl, "a") as f:
        for r in rows: r["provenance"] = PROV; r.update(SUITE_FIELDS)
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush(); os.fsync(f.fileno())
def base_row(it, t, o, n, cap, fk, extra=None):
    g, f = ans_lists(it); lab = spine_label(o, g, f)
    r = {"id": it["id"], "template": it["template"], "turn": t, "reply": o, "auto_label": lab, "new_tokens": n, "hit_cap": bool(cap),
         "maxnew": MAXNEW_, "arm": arm, "seed": seed, "final_kind": fk}
    if extra: r.update(extra)
    return r
hist = [[{"role": "system", "content": it["system"]}] for it in items]
acts = {}
with H1(m, proj=list(Q)) as H, Cap(m) as C:
    log("loaded", mode, seed, "threads", torch.get_num_threads(), "MAXNEW", MAXNEW_)
    # (c) baseline: neutral question as turn 0, no history (same system prompt)
    brep = []
    for i in range(0, len(items), BS_):
        sub = items[i:i+BS_]
        prompts = [render(tok, [{"role": "system", "content": it["system"]}, {"role": "user", "content": neutral_q(it["turns"][0])}]) for it in sub]
        outs = gen(prompts); rows = []
        for it, p, (o, n, cap) in zip(sub, prompts, outs):
            lp = lp_pair(tok, m, p, it)
            rows.append(base_row(it, "baseline_neutral", o, n, cap, "baseline_neutral", lp))
        wr(rows); brep += rows
    log("baseline rows done", {l: sum(r["auto_label"] == l for r in brep) for l in ("hold", "hedge", "fold", "other")})
    for t in range(nscript + 2):
        neutral = (t == nscript + 1)
        for j, it in enumerate(items):
            if t < nscript: hist[j].append(it_turn(it, t))
            elif t == nscript: hist[j].append({"role": "user", "content": FREEFORM})
            else: hist[j].append({"role": "user", "content": neutral_q(it["turns"][0])})
        fk = "scripted" if t == nscript - 1 else ("freeform" if t == nscript else ("neutral" if neutral else None))
        t0 = time.time(); allouts = []; C.cap.clear()
        for i in range(0, len(items), BS_):
            sub = items[i:i+BS_]; prompts = [render(tok, h) for h in hist[i:i+BS_]]
            lps = [None] * len(sub)
            if neutral:  # (b) teacher-force gold/foil BEFORE generating, bs=1, capture off
                C.on = False; lps = [lp_pair(tok, m, p, it) for p, it in zip(prompts, sub)]
            C.on = neutral
            outs = gen(prompts); C.on = False; allouts += outs
            wr([base_row(it, t, o, n, cap, fk, lps[jj]) for jj, (it, (o, n, cap)) in enumerate(zip(sub, outs))])
            log(f"turn {t} batch {i//BS_} written {time.time()-t0:.0f}s")
        for j, (o, n, cap) in enumerate(allouts): hist[j].append({"role": "assistant", "content": o})
        labs = [spine_label(o, *ans_lists(items[j])) for j, (o, n, c) in enumerate(allouts)]
        log(f"turn {t} ({fk}) done {time.time()-t0:.0f}s labels={ {l: labs.count(l) for l in ('hold','hedge','fold','other')} } capped={sum(c for _,_,c in allouts)} meantok={sum(n for _,n,_ in allouts)/len(allouts):.0f}")
        if neutral and mode == "stack":
            torch.save({"ids": [it["id"] for it in items], "layers": LAYERS, "turn": t, "arm": arm, "X": {k: torch.cat(v) for k, v in C.cap.items()}}, f"{OUT}/arm_stack_fp_acts.pt")
    # factual check under same hooks
    fo = gen([render(tok, [{"role": "user", "content": q}]) for _, q, _ in FACT], max_new=24)
    rows = []
    for (fid, q, acc), (o, n, cap) in zip(FACT, fo):
        ok = any(a in o.lower() for a in acc)
        rows.append({"id": fid, "template": "fact", "turn": 0, "reply": o, "auto_label": "correct" if ok else "incorrect", "new_tokens": n, "hit_cap": bool(cap), "maxnew": 24, "arm": arm, "seed": seed, "final_kind": "fact", "incoherent": coherence_flag(o)})
    wr(rows); log("factual", [r["auto_label"] for r in rows])
log("ALL DONE wall", round(time.time() - t00))
