from lib import *
import numpy as np
t00 = time.time()
def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)
ext, ev = items_split()
log("extract", len(ext), "eval", len(ev), "templates ext", {t: sum(i['template']==t for i in ext) for t in {i['template'] for i in ext}})
json.dump({"extract_ids": [i["id"] for i in ext], "eval_ids": [i["id"] for i in ev], "split": "by template: extract=memory_rewrite, eval=false_premise"}, open(f"{OUT}/split.json", "w"))
tok, m = load(); log("loaded")
LAYERS = list(range(10, 29))
recs, caps = run_dialogues(tok, m, ext, capture=LAYERS, log=log)
json.dump(recs, open(f"{OUT}/stock_extract.json", "w"))
# collect cells
cells = []  # (item_idx, t, label)
X = {k: [] for k in LAYERS}
for (t, k), (act, A) in caps.items():
    for r, j in enumerate(act):
        if k == LAYERS[0]: cells.append((j, t, recs[j]["turns"][t]["label"]))
        X[k].append((j, t, A[r]))
torch.save({"cells_meta": [(j, t, recs[j]["turns"][t]["label"]) for (j, t, _) in X[LAYERS[0]]],
            "X": {k: torch.stack([a for (_, _, a) in v]) for k, v in X.items()}}, f"{OUT}/extract_acts.pt")
log("saved acts", summarize(recs), "total wall", round(time.time()-t00))
