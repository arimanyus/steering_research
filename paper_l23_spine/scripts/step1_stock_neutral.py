# stock arm: replay stored stock t0-t5 replies, add neutral t6 (+lp, acts not saved); also stock baseline
import sys
from lib4 import *
torch.set_num_threads(int(os.environ.get("NT", "4")))
ext, ev = items_split(); items = ev; BS_ = 8
out_jsonl = f"{OUT}/arm_stock_fp_neutral.jsonl"; assert not os.path.exists(out_jsonl)
rep = {}
for l in open(f"{OUT}/stock_eval_turns.jsonl"):
    if l.startswith('{"id"'):
        r = json.loads(l); rep[(r["id"], r["turn"])] = r["reply"]
tok, m = load(); gen = mk_gen(tok, m)
def wr(rows):
    with open(out_jsonl, "a") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush(); os.fsync(f.fileno())
def row(it, t, o, n, cap, fk, lp):
    g, f = ans_lists(it); r = {"id": it["id"], "template": it["template"], "turn": t, "reply": o, "auto_label": spine_label(o, g, f), "new_tokens": n, "hit_cap": bool(cap), "maxnew": MAXNEW_, "arm": "stock", "seed": None, "final_kind": fk}
    r.update(lp); return r
for kind in ("baseline_neutral", "neutral"):
    for i in range(0, len(items), BS_):
        sub = items[i:i+BS_]; prompts = []
        for it in sub:
            if kind == "baseline_neutral": h = [{"role": "system", "content": it["system"]}]
            else:
                h = [{"role": "system", "content": it["system"]}]
                for t in range(5): h += [it_turn(it, t), {"role": "assistant", "content": rep[(it["id"], t)]}]
                h += [{"role": "user", "content": FREEFORM}, {"role": "assistant", "content": rep[(it["id"], 5)]}]
            h.append({"role": "user", "content": neutral_q(it["turns"][0])}); prompts.append(render(tok, h))
        lps = [lp_pair(tok, m, p, it) for p, it in zip(prompts, sub)]
        outs = gen(prompts)
        wr([row(it, kind if kind == "baseline_neutral" else 6, o, n, c, kind, lp) for it, (o, n, c), lp in zip(sub, outs, lps)])
        log(kind, i // BS_, "written")
log("ALL DONE")
