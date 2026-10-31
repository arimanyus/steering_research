#!/usr/bin/env python3
"""Freeze benchmark subsets. Run with ./dprep_venv/bin/python (needs pyarrow). No model, no torch.
Inputs : data/raw/*.parquet  (HF: cais/mmlu all/{test,dev}, allenai/ai2_arc ARC-Challenge/test, Salesforce/wikitext wikitext-2-raw-v1/test)
Outputs: data/mmlu500.jsonl, data/arc300.jsonl, data/wiki200.jsonl, data/calib200.jsonl, data/ids_*.json, data/MANIFEST.json (sha256 of everything)
All sampling uses seed 0. Every file is ORDERED so any prefix is a (near-)uniform sub-sample -> `--items` staged screens reuse the same items.
"""
import json, hashlib, random, os, sys, math
import pandas as pd
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"); R = f"{D}/raw"
SEED = 0
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def dump(name, rows):
    p = f"{D}/{name}"
    with open(p, "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return p
def prefix_stratified_order(items, key, seed):
    """order so each prefix is ~proportionally stratified by `key`: within each stratum shuffle, then sort by (rank+u)/n."""
    rng = random.Random(seed); by = {}
    for it in items: by.setdefault(it[key], []).append(it)
    keyed = []
    for k in sorted(by):
        g = by[k][:]; rng.shuffle(g)
        for r, it in enumerate(g): keyed.append(((r + rng.random()) / len(g), k, it["item_id"], it))
    keyed.sort(key=lambda t: t[:3]); return [t[3] for t in keyed]
def largest_remainder(sizes, total):
    tot = sum(sizes.values()); raw = {k: v * total / tot for k, v in sizes.items()}
    base = {k: int(math.floor(x)) for k, x in raw.items()}; rem = total - sum(base.values())
    for k in sorted(raw, key=lambda k: (-(raw[k] - base[k]), k))[:rem]: base[k] += 1
    return base
L = "ABCDE"
# ---- MMLU: 500 stratified by subject (proportional, largest remainder), seed 0
m = pd.read_parquet(f"{R}/all/test-00000-of-00001.parquet")
rows = [{"item_id": f"mmlu-test-{i}", "subject": r.subject, "question": r.question, "choices": [str(c) for c in r.choices], "gold": L[int(r.answer)]} for i, r in enumerate(m.itertuples())]
assert all(len(r["choices"]) == 4 for r in rows)
alloc = largest_remainder(m.subject.value_counts().to_dict(), 500)
rng = random.Random(SEED); sel = []
for s in sorted(alloc):
    pool = [r for r in rows if r["subject"] == s]; sel += rng.sample(pool, alloc[s])
assert len(sel) == 500
mm = prefix_stratified_order(sel, "subject", SEED + 1)
for i, r in enumerate(mm): r["order"] = i
dump("mmlu500.jsonl", mm)
# ---- ARC-Challenge test: keep 4-choice items only (1165/1172; 7 with 3 or 5 choices dropped, uniform A-D scoring), 300 random, seed 0
a = pd.read_parquet(f"{R}/ARC-Challenge/test-00000-of-00001.parquet")
arows = []
for r in a.itertuples():
    labs, txt = list(r.choices["label"]), [str(t) for t in r.choices["text"]]
    if len(txt) != 4 or r.answerKey not in labs: continue
    arows.append({"item_id": f"arc-c-test-{r.id}", "question": r.question, "choices": txt, "gold": L[labs.index(r.answerKey)]})
rng = random.Random(SEED); asel = rng.sample(arows, 300)
for i, r in enumerate(asel): r["order"] = i      # random sample => every prefix is a uniform random sub-sample
dump("arc300.jsonl", asel)
# ---- wikitext-2-raw test: 200 paragraphs (>=500 chars, not headings), seed 0; truncated to 256 tokens at run time
w = pd.read_parquet(f"{R}/wikitext-2-raw-v1/test-00000-of-00001.parquet")
paras = [{"item_id": f"wt2-test-{i}", "text": t.strip()} for i, t in enumerate(w.text) if len(t) >= 500 and not t.strip().startswith("=")]
rng = random.Random(SEED); wsel = rng.sample(paras, 200)
for i, r in enumerate(wsel): r["order"] = i
dump("wiki200.jsonl", wsel)
# ---- calibration prompts: 200 from MMLU *dev* split (disjoint from the 500 test items), seed 0, same zero-shot template as eval
d = pd.read_parquet(f"{R}/all/dev-00000-of-00001.parquet")
drows = [{"item_id": f"mmlu-dev-{i}", "subject": r.subject, "question": r.question, "choices": [str(c) for c in r.choices], "gold": L[int(r.answer)]} for i, r in enumerate(d.itertuples())]
rng = random.Random(SEED); csel = rng.sample(drows, 200)
for i, r in enumerate(csel): r["order"] = i
dump("calib200.jsonl", csel)
# ---- screen set: 200 MMLU test items, stratified by subject (largest remainder over the REMAINING pool), seed 1, disjoint from mmlu500 and calib200
sel_ids = {r["item_id"] for r in mm}; pool = [r for r in rows if r["item_id"] not in sel_ids]
salloc = largest_remainder({s: sum(1 for r in pool if r["subject"] == s) for s in sorted({r["subject"] for r in pool})}, 200)
rng = random.Random(SEED + 1); ssel = []
for s in sorted(salloc):
    ssel += rng.sample([r for r in pool if r["subject"] == s], salloc[s])
assert len(ssel) == 200
scr = prefix_stratified_order(ssel, "subject", SEED + 2)
for i, r in enumerate(scr): r["order"] = i
dump("mmlu_screen200.jsonl", scr)
for name, rr in [("mmlu500", mm), ("mmlu_screen200", scr), ("arc300", asel), ("wiki200", wsel), ("calib200", csel)]:
    json.dump([r["item_id"] for r in rr], open(f"{D}/ids_{name}.json", "w"))
man = {"seed": SEED, "screen_seed": SEED + 1, "mmlu_alloc_by_subject": alloc, "arc_dropped_non4": len(a) - len(arows), "files": {}}
for f in sorted(os.listdir(D)):
    p = f"{D}/{f}"
    if os.path.isfile(p) and f != "MANIFEST.json": man["files"][f] = sha(p)
for root, _, fs in os.walk(R):
    for f in fs: man["files"][os.path.relpath(f"{root}/{f}", D)] = sha(f"{root}/{f}")
man["sources"] = {"mmlu": "cais/mmlu all/test (14042 rows)", "arc": "allenai/ai2_arc ARC-Challenge/test (1172 rows)", "wikitext": "Salesforce/wikitext wikitext-2-raw-v1/test", "calib": "cais/mmlu all/dev"}
Q = {n: set(r["question"] for r in rr) for n, rr in [("calib", csel), ("mmlu500", mm), ("screen", scr)]}
assert not (Q["calib"] & Q["mmlu500"]) and not (Q["screen"] & Q["mmlu500"]) and not (Q["screen"] & Q["calib"]), "question-text overlap"
I = {n: set(r["item_id"] for r in rr) for n, rr in [("calib", csel), ("mmlu500", mm), ("screen", scr)]}
assert not (I["calib"] & I["mmlu500"]) and not (I["screen"] & I["mmlu500"]) and not (I["screen"] & I["calib"]), "id overlap"
man["screen_alloc_by_subject"] = salloc
json.dump(man, open(f"{D}/MANIFEST.json", "w"), indent=1)
print("mmlu", len(mm), "arc", len(asel), "arc pool", len(arows), "wiki", len(wsel), "wiki pool", len(paras), "calib", len(csel))
