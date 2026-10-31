#!/usr/bin/env python3
"""Diff-of-means concept direction from Leila's contrast pairs (concept_prompts.jsonl: pair_id/pos/neg). WRITE-ONLY until the box is free: do NOT run while another model job is alive.
Residual = hidden_states[L] = OUTPUT of model.model.layers[L-1] (lab convention, same as bench.py --dir-convention hs).
direction_L = unit( mean_pos(h) - mean_neg(h) ), h = residual at the LAST prompt token (--pool last) or mean over tokens (--pool mean).
Also saves split-half cosine (odd vs even pairs) and leave-one-pair-out min cosine as stability diagnostics, and raw diff norm.
Output: .pt dict {"<L>": unit float32 (2560,)} -> usable as bench.py --dir-pt X.pt --dir-key 23 ; plus <out>.json diagnostics.
Usage: /workspace/tbL_venv/bin/python fit_concept_dirs.py --layers 23 --out dirs/concept_iq.pt [--pool last]
"""
import os, sys, json, argparse, hashlib
os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import get_layers, guard_memory
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3.5-4B"); ap.add_argument("--pairs", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "concept_prompts.jsonl"))
ap.add_argument("--layers", default="23"); ap.add_argument("--pool", choices=["last", "mean"], default="last"); ap.add_argument("--out", required=True)
ap.add_argument("--threads", type=int, default=4); ap.add_argument("--no-guard", action="store_true")
a = ap.parse_args(); guard_memory(a)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_num_threads(a.threads)
pairs = [json.loads(l) for l in open(a.pairs)]; Ls = [int(x) for x in a.layers.split(",")]
dt = torch.float32 if a.model.startswith("sshleifer/") else torch.bfloat16
tok = AutoTokenizer.from_pretrained(a.model); m = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt).eval(); layers = get_layers(m)
cap = {}
hs = []
for L in Ls:
    def fh(mod, i, o, L=L):
        h = (o[0] if isinstance(o, tuple) else o)[0].float(); cap[L] = h[-1] if a.pool == "last" else h.mean(0)
    hs.append(layers[L - 1].register_forward_hook(fh))
H = {L: {"pos": [], "neg": []} for L in Ls}
with torch.inference_mode():
    for p in pairs:
        for side in ("pos", "neg"):
            m(**tok(p[side], return_tensors="pt"), use_cache=False, logits_to_keep=1) if not a.model.startswith("sshleifer/") else m(**tok(p[side], return_tensors="pt"), use_cache=False)
            for L in Ls: H[L][side].append(cap[L].clone())
out, diag = {}, {"pairs_sha256": hashlib.sha256(open(a.pairs, "rb").read()).hexdigest(), "pool": a.pool, "n_pairs": len(pairs), "layers": {}}
def dom(idx, L):
    P = torch.stack([H[L]["pos"][i] for i in idx]).mean(0); N = torch.stack([H[L]["neg"][i] for i in idx]).mean(0); return P - N
cos = torch.nn.functional.cosine_similarity
for L in Ls:
    allidx = list(range(len(pairs))); d = dom(allidx, L); out[str(L)] = (d / d.norm()).float()
    d1, d2 = dom(allidx[0::2], L), dom(allidx[1::2], L); loo = [float(cos(d, dom([j for j in allidx if j != i], L), dim=0)) for i in allidx]
    diag["layers"][str(L)] = {"raw_diff_norm": float(d.norm()), "split_half_cos": float(cos(d1, d2, dim=0)), "loo_cos_min": min(loo), "loo_cos_mean": sum(loo) / len(loo)}
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); torch.save(out, a.out); json.dump(diag, open(a.out + ".json", "w"), indent=1); print(json.dumps(diag, indent=1))
