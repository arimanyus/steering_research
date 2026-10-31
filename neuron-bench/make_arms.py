# Generates arms.jsonl. No model loaded. Indices checked against config.json intermediate_size.
import json, torch, glob
cfg = json.load(open(glob.glob("/home/box/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/*/config.json")[0]))["text_config"]
I, NL = cfg["intermediate_size"], cfg["num_hidden_layers"]
DOSES = [-5, 0, 2, 5, 10]
TARGET = (8, 1955); LYRA = [(14, 8949, +1), (13, 7956, -1)]   # sign convention: /workspace/steer-withheld/tbL/run.py line 54-55
g = torch.Generator().manual_seed(4242)
perm = torch.randperm(I, generator=g).tolist()
rand = [n for n in perm if n != TARGET[1]][:20]          # without replacement, excluding 1955
OPNOTE = "layer = 0-indexed model.model.layers[layer]; neuron = index into MLP intermediate act (input to down_proj); op: dose_mult>0 or <0 => ADD dose_mult * mean|act| (this neuron, calibration set) to the activation at all positions; dose_mult==0 => ABLATE (set activation to 0), not add-0"
rows = [{"arm_id": "stock", "layer": None, "neuron": None, "dose_mult": None, "sign_note": "no edit"}]
def op(d): return "ablate" if d == 0 else "add"
for d in DOSES:
    rows.append({"arm_id": f"nisten_L8N1955_d{d}", "layer": 8, "neuron": 1955, "dose_mult": d, "op": op(d), "sign_note": f"{op(d)} {d} x mean|act| (nisten claim; layer 0-indexed, ambiguity noted in /workspace/neuronbench/target_check.md)"})
for d in DOSES:
    comps = [{"layer": l, "neuron": n, "dose_mult": d * s, "sign": s} for l, n, s in LYRA]
    rows.append({"arm_id": f"lyra_pair_d{d}", "layer": [l for l, _, _ in LYRA], "neuron": [n for _, n, _ in LYRA], "dose_mult": [d * s for _, _, s in LYRA], "op": op(d), "components": comps,
                 "sign_note": "sign convention as /workspace/steer-withheld/tbL/run.py:54-55 (L14 n8949 +1, L13 n7956 -1); dose_mult list = dose*sign per component, each in multiples of its OWN mean|act|. Note run.py scaled doses by resid-norm/||down_proj col||, not mean|act|; here Maya's mean|act| unit is used"})
for k, n in enumerate(rand):
    for d in DOSES:
        rows.append({"arm_id": f"rand_L8_k{k:02d}_n{n}_d{d}", "layer": 8, "neuron": n, "dose_mult": d, "op": op(d), "random_rank_k": k, "sign_note": f"random control neuron {k}; {op(d)} {d} x its own mean|act|"})
for r in rows:
    for l, n in zip(*(x if isinstance(x, list) else [x] for x in (r["layer"], r["neuron"]))):
        if l is not None: assert 0 <= l < NL and 0 <= n < I, r
assert len(set(rand)) == 20 and TARGET[1] not in rand
with open("arms.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
open("README.md", "w").write(f"""# arms.jsonl
{len(rows)} rows: stock(1) + nisten L8N1955 x{len(DOSES)} doses + Lyra pair x{len(DOSES)} + 20 random L8 neurons x{len(DOSES)} doses.
Doses (multiples of the neuron's own mean|activation| on the calibration set; Maya computes scale): {DOSES}.
{OPNOTE}.
Validation: all layer < {NL}, all neuron < intermediate_size {I} (config.json text_config).
Random neurons: `g = torch.Generator().manual_seed(4242); perm = torch.randperm({I}, generator=g).tolist(); rand = [n for n in perm if n != 1955][:20]` -> {rand}
Lyra sign convention copied from /workspace/steer-withheld/tbL/run.py lines 54-55: (L14, n8949, +1), (L13, n7956, -1); applied as one combined arm, per-component dose_mult = dose*sign.
Regenerate: python3 make_arms.py (torch from /workspace/tbL_venv or system).
""")
print(len(rows), rand)
