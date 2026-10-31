import json, sys, collections, statistics as st
f = sys.argv[1]; rows = [json.loads(l) for l in open(f)]
by = collections.defaultdict(list)
for r in rows: by[str(r["turn"])].append(r)
out = []
for t in ["baseline_neutral","0","1","2","3","4","5","6"]:
    if t in by:
        c = collections.Counter(r["auto_label"] for r in by[t]); out.append(f"t{t}: n={len(by[t])} " + " ".join(f"{k}={c.get(k,0)}" for k in ("hold","hedge","fold","other")) + f" capped={sum(r['hit_cap'] for r in by[t])}")
b = {r["id"]: r for r in by.get("baseline_neutral", [])}
if "6" in by and b:
    d6 = [r["dlp"] for r in by["6"] if "dlp" in r]; d0 = [b[r["id"]]["dlp"] for r in by["6"] if "dlp" in r and r["id"] in b]
    out.append(f"neutral t6 dlp mean={st.mean(d6):.3f} vs baseline dlp mean={st.mean(d0):.3f}; shift mean={st.mean(x-y for x,y in zip(d6,d0)):.3f}; n={len(d6)}")
fa = by.get("0", [])
fr = [r for r in rows if r.get("template") == "fact"]
out.append("factual: " + str([r["auto_label"] for r in fr]))
print("\n".join(out)); print("provenance:", json.dumps({k: v for k, v in rows[0].get("provenance", {}).items() if k != "per_layer_vector_sha256"}) if rows else "")
print("per_layer_vector_sha256:", rows[0].get("provenance", {}).get("per_layer_vector_sha256"))
