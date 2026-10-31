import json,sys,collections
for fn in ["stock_extract_turns.jsonl","stock_eval_turns.jsonl"]:
    try: R=[json.loads(l) for l in open("/workspace/steer-spine/"+fn)]
    except FileNotFoundError: print(fn,"missing"); continue
    print(fn,"rows",len(R))
    for t in sorted({r["turn"] for r in R}):
        c=[r for r in R if r["turn"]==t]; L=collections.Counter(r["auto_label"] for r in c)
        print(f"  turn {t}: n={len(c)} {dict(L)} hit_cap={sum(r['hit_cap'] for r in c)} max_new_tokens={max(r['new_tokens'] for r in c)}")
