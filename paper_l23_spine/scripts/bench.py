#!/usr/bin/env python3
"""CPU bench harness: single-MLP-neuron pushes + residual-direction pushes on Qwen3.5-4B (MMLU/ARC by answer-letter logprob, wikitext ppl).
NO generation. One forward pass per item, batch size 1 (fixed; logged). Run with /workspace/tbL_venv/bin/python (torch 2.14.1 / transformers 5.21.0).

Conventions (match Kenji's arms.jsonl README):
  * layer = 0-indexed model.model.layers[layer]; neuron = index into MLP intermediate activation = INPUT of mlp.down_proj.
  * neuron arm: dose_mult == 0 -> ABLATE (activation := 0); else ADD dose_mult * mean|act| (that neuron, stock model, calib200 prompts, all token positions) at all positions.
  * multi-neuron rows (Lyra): row["components"] = [{layer, neuron, dose_mult}], else parallel lists layer/neuron/dose_mult. dose_mult already carries the sign.
  * direction arm: ADD alpha * scale * u at ALL positions to the residual stream, u = unit vector, scale = mean |<h,u>| on calib200 (stock) at the same site
    (--dir-unit signed uses signed mean instead). Site: --dir-convention hs (default; lab convention, "L23" = hidden_states[23] = OUTPUT of layers[22], as lib3.py) or layer (output of layers[L]).
  * random controls: --dir-random N random unit directions (torch.Generator seed --dir-seed) each with its own scale, same alpha.
Stages: --stage screen (mmlu_screen200 only) | confirm (mmlu500+arc300+wiki200); results in <out>/<stage>/, no cross-stage item reuse (asserted at startup together with calib/screen/confirm disjointness). --items N or mmlu=..,arc=..,wiki=.. restricts to a prefix inside the stage;
  a later run with larger --items only computes the missing items; a config mismatch with existing rows aborts (use --overwrite).
Outputs: <out>/<arm_id>.jsonl (per item rows), <out>/<arm_id>.summary.json, <out>/calib/*.pt|json.
"""
import os, sys, json, time, math, glob, fnmatch, hashlib, argparse, resource
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = f"{HERE}/data"
os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
LET = "ABCD"

def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
def rss_mb(): return int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
def peak_rss_mb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

# ---------------- data / prompts ----------------
def load_jsonl(p): return [json.loads(l) for l in open(p)]
def mmlu_prompt(it):
    s = it["subject"].replace("_", " ")
    return f"The following is a multiple choice question about {s}.\n\n{it['question'].strip()}\n" + "".join(f"{LET[i]}. {c}\n" for i, c in enumerate(it["choices"])) + "Answer:"
def arc_prompt(it):
    return f"Question: {it['question'].strip()}\n" + "".join(f"{LET[i]}. {c}\n" for i, c in enumerate(it["choices"])) + "Answer:"
STAGES = {"screen": {"mmlu_screen": ("mmlu_screen200.jsonl", 200)},
          "confirm": {"mmlu": ("mmlu500.jsonl", 500), "arc": ("arc300.jsonl", 300), "wiki": ("wiki200.jsonl", 200)}}
MMLU_LIKE = ("mmlu", "mmlu_screen")
def verify_manifest():
    man = json.load(open(f"{DATA}/MANIFEST.json"))
    for f, h in man["files"].items():
        if f.startswith("raw/"): continue
        assert sha_file(f"{DATA}/{f}") == h, f"data file changed since freeze: {f}"
    return man
def parse_items(spec, stage):
    """--items: None = full stage; N = first N of the stage's MMLU-type bench only; 'mmlu=200,arc=0,wiki=50' (confirm) or 'mmlu_screen=100' / 'screen=100' (screen)."""
    full = {b: n for b, (_, n) in STAGES[stage].items()}
    if spec is None: return dict(full)
    first = next(iter(full))
    if spec.isdigit(): return {b: (min(int(spec), full[b]) if b == first else 0) for b in full}
    out = {b: 0 for b in full}
    for kv in spec.split(","):
        k, v = kv.split("="); k = "mmlu_screen" if k == "screen" else k; assert k in out, f"bench {k} not in stage {stage}: {list(out)}"; out[k] = min(int(v), full[k])
    return out
def check_disjoint():
    """startup assertion: calibration (MMLU dev), screen200 and confirm mmlu500 share no item id and no question text; arc/wiki ids disjoint from all."""
    sets = {}
    for name, f in [("mmlu500", "mmlu500.jsonl"), ("mmlu_screen200", "mmlu_screen200.jsonl"), ("calib200", "calib200.jsonl"), ("arc300", "arc300.jsonl"), ("wiki200", "wiki200.jsonl")]:
        rows = load_jsonl(f"{DATA}/{f}"); ids = [r["item_id"] for r in rows]
        assert len(ids) == len(set(ids)), f"duplicate ids in {name}"
        assert ids == json.load(open(f"{DATA}/ids_{name}.json")), f"ids file != jsonl for {name}"
        sets[name] = (set(ids), {r["question"] for r in rows if "question" in r})
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not (sets[a][0] & sets[b][0]), f"id overlap {a} & {b}"
            assert not (sets[a][1] & sets[b][1]), f"question-text overlap {a} & {b}"
    return {n: len(v[0]) for n, v in sets.items()}

# ---------------- model adapter ----------------
def get_layers(m):
    for path in ("model.layers", "model.language_model.layers", "transformer.h"):
        o = m
        try:
            for a in path.split("."): o = getattr(o, a)
            return o
        except AttributeError: pass
    raise RuntimeError("cannot find decoder layers")
def get_down(layer):
    mlp = layer.mlp
    return getattr(mlp, "down_proj", None) or mlp.c_proj     # c_proj = GPT-2 style (stub model only)
def text_cfg(m):
    c = m.config; return getattr(c, "text_config", c)
def dims(m):
    c = text_cfg(m); nl = getattr(c, "num_hidden_layers", None) or c.n_layer
    inter = getattr(c, "intermediate_size", None) or getattr(c, "n_inner", None) or 4 * (getattr(c, "hidden_size", None) or c.n_embd)
    return nl, inter, (getattr(c, "hidden_size", None) or c.n_embd)

# ---------------- arms ----------------
def listify(x): return x if isinstance(x, list) else [x]
def parse_neuron_arm(r):
    if r.get("layer") is None: return []
    if "components" in r: return [(c["layer"], c["neuron"], c["dose_mult"]) for c in r["components"]]
    L, N, D = listify(r["layer"]), listify(r["neuron"]), listify(r["dose_mult"])
    assert len(L) == len(N) == len(D), r["arm_id"]
    return list(zip(L, N, D))
_PT = {}
def load_dirs(pt, key=None, index=None):
    """-> (unit (d,), file_scale or None, file_site_layer_index or None). Formats:
      * plain tensor (d,)                         -> scale None (calibrated), site None (--dir-layer + --dir-convention)
      * dict {key: tensor (d,)} (e.g. dirs_final.pt) -> pick `key`
      * Leila concept_dir_L{z}.pt: {"dir","scale","layer_zero_idx",...}   (use key None or "dir")
      * Leila concept_rand_dirs_L{z}.pt: {"dirs":(K,d),"scales":(K,),"layer_zero_idx":z,...}; pick row `index`
    file site = layer_zero_idx = OUTPUT of model.model.layers[z] (lib.Hooks: hs index z+1). Scales come from the file and are NOT recomputed."""
    import torch
    if pt not in _PT: _PT[pt] = torch.load(pt, map_location="cpu")
    o = _PT[pt]; scale = None; site = None
    if isinstance(o, dict) and "dirs" in o:
        assert index is not None, f"{pt}: random-direction file, need row index"
        u = o["dirs"][index].float(); scale = float(o["scales"][index]); site = o.get("layer_zero_idx")
    elif isinstance(o, dict) and "dir" in o and key in (None, "dir"):
        u = o["dir"].float(); scale = float(o["scale"]) if "scale" in o else None; site = o.get("layer_zero_idx")
    elif isinstance(o, dict):
        k = key if key is not None else (list(o)[0] if len(o) == 1 else None)
        assert k is not None and k in o, f"{pt}: need pt_key from {list(o)}"; u = o[k].float()
    else: u = o.float()
    u = u.flatten()
    if scale is not None: assert abs(float(u.norm()) - 1) < 1e-3, f"{pt}: file-provided scale but direction is not unit (norm {float(u.norm())})"
    return u / u.norm(), scale, (int(site) if site is not None else None)
def load_unit(pt, key): return load_dirs(pt, key)[0]
def expand_arms(args, d_model):
    """arms.jsonl rows (+ CLI direction arms) -> list of dict. Direction vectors are loaded lazily by the runner."""
    arms = [json.loads(l) for l in open(args.arms)] if args.arms else []
    for a in arms:
        a.setdefault("type", "direction" if a.get("pt") else "neuron")
    alphas = [float(x) for x in args.alphas.split(",")]
    if args.dir_pt:
        stem = os.path.splitext(os.path.basename(args.dir_pt))[0]
        _, _, fsite = load_dirs(args.dir_pt, args.dir_key)
        tag = f"L{fsite}f" if fsite is not None else f"L{args.dir_layer}"
        for al in alphas:
            arms.append({"arm_id": f"dir_{stem}_{tag}_a{al:g}", "type": "direction", "pt": args.dir_pt, "pt_key": args.dir_key, "layer": args.dir_layer, "alpha": al, "role": "target"})
    if args.dir_rand_pt:
        import torch
        o = torch.load(args.dir_rand_pt, map_location="cpu"); _PT[args.dir_rand_pt] = o; assert "dirs" in o and "scales" in o, "need keys dirs, scales"
        assert o["dirs"].shape[0] == o["scales"].shape[0]; z = o.get("layer_zero_idx", args.dir_layer)
        for k in range(o["dirs"].shape[0]):
            for al in alphas:
                arms.append({"arm_id": f"dirrand_file_k{k:02d}_L{z}f_a{al:g}", "type": "direction", "pt": args.dir_rand_pt, "pt_index": k, "layer": args.dir_layer, "alpha": al, "role": "random_control_file"})
    if args.dir_random:
        assert args.dir_layer is not None, "--dir-random needs --dir-layer"
        for k in range(args.dir_random):
            for al in alphas:
                arms.append({"arm_id": f"dirrand_k{k:02d}_L{args.dir_layer}_a{al:g}", "type": "direction", "pt": None, "rand_k": k, "layer": args.dir_layer, "alpha": al, "role": "random_control"})
    ids = [a["arm_id"] for a in arms]; assert len(ids) == len(set(ids)), "duplicate arm_id"
    if args.arm_ids:
        pats = args.arm_ids.split(","); arms = [a for a in arms if any(fnmatch.fnmatch(a["arm_id"], p) for p in pats)]
    arms.sort(key=lambda a: a["arm_id"] != "stock")      # stock first
    return arms

# ---------------- hooks ----------------
class Edits:
    """context manager installing neuron edits (pre-hook on down_proj) and direction edits (forward hook on a layer output)."""
    def __init__(self, m, neuron_edits=(), dir_edits=()):
        self.m, self.ne, self.de, self.h = m, list(neuron_edits), list(dir_edits), []
    def __enter__(self):
        import torch
        layers = get_layers(self.m); by = {}
        for (l, n, op, delta) in self.ne: by.setdefault(l, []).append((n, op, delta))
        for l, lst in by.items():
            def pre(mod, a, lst=lst):
                x = a[0].clone()
                for n, op, delta in lst:
                    if op == "ablate": x[..., n] = 0
                    else: x[..., n] = x[..., n] + delta
                return (x,) + tuple(a[1:])
            self.h.append(get_down(layers[l]).register_forward_pre_hook(pre))
        dby = {}
        for (site, vec) in self.de: dby.setdefault(site, []).append(vec)
        for site, vs in dby.items():
            tot = sum(vs)
            def fh(mod, a, out, tot=tot):
                h = out[0] if isinstance(out, tuple) else out
                h2 = h + tot.to(h.dtype)
                return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2
            self.h.append(layers[site].register_forward_hook(fh))
        return self
    def __exit__(self, *e):
        for h in self.h: h.remove()

# ---------------- calibration ----------------
def calibrate(m, tok, layers_needed, dir_specs, args, out_dir):
    """one stock pass over calib200: mean|act| per neuron for each needed layer, and mean |<h,u>| / signed mean for each direction spec.
    dir_specs: {key: (site_layer_index, unit_vec)}. Cached by (model, calib sha, dtype)."""
    import torch
    csha = sha_file(f"{DATA}/calib200.jsonl")
    cp = f"{out_dir}/calib/calib_{args.model.replace('/', '_')}_{csha[:12]}_{args.dtype}.pt"; os.makedirs(os.path.dirname(cp), exist_ok=True)
    cache = torch.load(cp) if os.path.exists(cp) else {"neuron_mean_abs": {}, "dirs": {}, "n_tokens": None, "calib_sha256": csha}
    need_l = [l for l in layers_needed if l not in cache["neuron_mean_abs"] or l not in cache.get("neuron_mean_signed", {})]
    need_d = {k: v for k, v in dir_specs.items() if k not in cache["dirs"]}
    if need_l or need_d:
        layers = get_layers(m); sums = {l: None for l in need_l}; ssums = {l: None for l in need_l}; dsum = {k: [0.0, 0.0] for k in need_d}; ntok = [0]; hs = []
        for l in need_l:
            def pre(mod, a, l=l):
                x = a[0].float().abs().sum(dim=tuple(range(a[0].dim() - 1)))
                sums[l] = x if sums[l] is None else sums[l] + x
                y = a[0].float().sum(dim=tuple(range(a[0].dim() - 1))); ssums[l] = y if ssums[l] is None else ssums[l] + y
            hs.append(get_down(layers[l]).register_forward_pre_hook(pre))
        for k, (site, u) in need_d.items():
            def fh(mod, a, out, k=k, u=u):
                h = (out[0] if isinstance(out, tuple) else out).float(); p = h @ u
                dsum[k][0] += float(p.abs().sum()); dsum[k][1] += float(p.sum())
            hs.append(layers[site].register_forward_hook(fh))
        t0 = time.time()
        with torch.inference_mode():
            for it in load_jsonl(f"{DATA}/calib200.jsonl"):
                ids = tok(mmlu_prompt(it), return_tensors="pt")
                ntok[0] += ids["input_ids"].shape[1]; forward_last(m, ids)
        for h in hs: h.remove()
        cache.setdefault("neuron_mean_signed", {})
        for l in need_l: cache["neuron_mean_abs"][l] = (sums[l] / ntok[0]).cpu(); cache["neuron_mean_signed"][l] = (ssums[l] / ntok[0]).cpu()
        for k in need_d: cache["dirs"][k] = {"mean_abs": dsum[k][0] / ntok[0], "mean_signed": dsum[k][1] / ntok[0]}
        cache["n_tokens"] = ntok[0]; torch.save(cache, cp)
        print(f"[calib] {len(need_l)} layers, {len(need_d)} dirs, {ntok[0]} tokens, {time.time()-t0:.0f}s -> {cp}", flush=True)
    return cache

def forward_last(m, ids):
    import torch
    try: return m(**ids, use_cache=False, logits_to_keep=1).logits[0, -1].float()
    except TypeError: return m(**ids, use_cache=False).logits[0, -1].float()

# ---------------- running ----------------
def guard_memory(args):
    if args.no_guard or args.model.startswith("sshleifer/"): return
    me = os.getpid()
    for p in glob.glob("/proc/[0-9]*/status"):
        try:
            s = open(p).read(); pid = int(p.split("/")[2])
            if pid == me: continue
            name = s.split("Name:")[1].split()[0]; rss = int(s.split("VmRSS:")[1].split()[0]) / 1e6   # GB
            if rss > 2.0 and "python" in name: sys.exit(f"GUARD: another model-sized python process alive (pid {pid}, {rss:.1f} GB). Shared box fits ONE 9GB job. Aborting (use --no-guard to override).")
        except Exception: pass
    avail = int([l for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0].split()[1]) / 1e6
    if avail < 10.5: sys.exit(f"GUARD: MemAvailable {avail:.1f} GB < 10.5 GB. Aborting.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B"); ap.add_argument("--dtype", default=None, help="bfloat16 (default for Qwen) | float32")
    ap.add_argument("--arms", default=f"{HERE}/arms.jsonl"); ap.add_argument("--arm-ids", default=None, help="comma list of fnmatch patterns")
    ap.add_argument("--stage", required=True, choices=["screen", "confirm"], help="screen = mmlu_screen200 only; confirm = mmlu500+arc300+wiki200. Results go to <out>/<stage>/; no cross-stage item reuse.")
    ap.add_argument("--items", default=None); ap.add_argument("--out", default=f"{HERE}/results"); ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--dir-pt", default=None); ap.add_argument("--dir-key", default=None); ap.add_argument("--dir-layer", type=int, default=None, help="needed for plain tensors / dict-keyed .pt (hs convention: 23 = output of layers[22]); ignored when the .pt has layer_zero_idx")
    ap.add_argument("--dir-rand-pt", default=None, help="Leila format: {dirs (K,d), scales (K,), layer_zero_idx}; provided scales are used as-is (no recompute)")
    ap.add_argument("--dir-scale", choices=["file", "calib"], default="file", help="for .pt files that carry a scale: use it (file, default) or recompute on calib200")
    ap.add_argument("--alphas", default="-2,-1,1,2"); ap.add_argument("--dir-random", type=int, default=0); ap.add_argument("--dir-seed", type=int, default=4242)
    ap.add_argument("--dir-unit", choices=["abs", "signed"], default="abs"); ap.add_argument("--dir-convention", choices=["hs", "layer"], default="hs")
    ap.add_argument("--no-arms-file", action="store_true"); ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--plan", action="store_true", help="print plan (arms, #forward passes) and exit; loads no model"); ap.add_argument("--no-guard", action="store_true")
    ap.add_argument("--calibrate-only", action="store_true")
    args = ap.parse_args()
    if args.no_arms_file: args.arms = None
    if args.dtype is None: args.dtype = "float32" if args.model.startswith("sshleifer/") else "bfloat16"
    items_n = parse_items(args.items, args.stage); man = verify_manifest(); disj = check_disjoint()
    print(f"[disjoint] ok {disj}; stage={args.stage}", flush=True)
    sout = f"{args.out}/{args.stage}"
    cfgp = None
    if not args.model.startswith("sshleifer/"):
        from huggingface_hub import hf_hub_download
        cfgp = hf_hub_download(args.model, "config.json", local_files_only=True)
    d_model = json.load(open(cfgp))["text_config"]["hidden_size"] if cfgp else 2
    arms = expand_arms(args, d_model)
    nfwd = sum(items_n.values())
    print(f"[plan] {len(arms)} arms x {nfwd} forward passes ({items_n}) bs=1 + calibration 200; out={args.out}", flush=True)
    if args.plan:
        for a in arms: print(" ", a["arm_id"], a.get("type"))
        return
    guard_memory(args)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(args.threads); os.makedirs(sout, exist_ok=True)
    dt = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model)
    m = AutoModelForCausalLM.from_pretrained(args.model, dtype=dt).eval()
    NL, INTER, D = dims(m)
    letter_ids = []
    for c in LET:
        t = tok.encode(" " + c, add_special_tokens=False); assert len(t) == 1, f"' {c}' is not a single token: {t}"; letter_ids.append(t[0])
    model_rev = os.path.dirname(os.path.realpath(cfgp)) if cfgp else args.model
    env = {"stage": args.stage, "torch": torch.__version__, "dtype": args.dtype, "threads": args.threads, "bs": 1, "model": args.model, "model_snapshot": model_rev, "model_hash_note": "snapshot dir name = HF revision"}
    # ---- resolve arms: validate indices, direction vectors
    neuron_layers = set(); dir_specs = {}; resolved = []
    rand_dirs = None
    for a in arms:
        if a["type"] == "direction":
            fscale = None; fsite = None
            if a.get("pt"): u, fscale, fsite = load_dirs(a["pt"], a.get("pt_key"), a.get("pt_index"))
            else:
                if rand_dirs is None:
                    g = torch.Generator().manual_seed(args.dir_seed); rd = torch.randn(max(args.dir_random, a["rand_k"] + 1), D, generator=g); rand_dirs = rd / rd.norm(dim=1, keepdim=True)
                u = rand_dirs[a["rand_k"]]
            if fsite is not None: site = fsite; conv = "file layer_zero_idx = output of layers[z]"
            else:
                assert a.get("layer") is not None, f"{a['arm_id']}: no layer (give --dir-layer)"
                site = a["layer"] - 1 if args.dir_convention == "hs" else a["layer"]; conv = args.dir_convention
            assert 0 <= site < NL
            assert u.numel() == D, f"direction dim {u.numel()} != hidden {D}"
            use_file = fscale is not None and args.dir_scale == "file"
            if use_file and args.dir_unit == "signed": sys.exit("file scales are mean|proj|; --dir-unit signed is incompatible with --dir-scale file")
            key = f"{hashlib.sha256(u.numpy().tobytes()).hexdigest()[:16]}@{site}"
            if not use_file: dir_specs[key] = (site, u)
            resolved.append({"arm": a, "kind": "dir", "site": site, "u": u, "dkey": key, "fscale": fscale if use_file else None, "conv": conv, "fscale_ignored": fscale if (fscale is not None and not use_file) else None})
        else:
            comps = parse_neuron_arm(a)
            for (l, n, dm) in comps: assert 0 <= l < NL and 0 <= n < INTER, f"{a['arm_id']}: L{l} N{n} out of range (NL={NL}, I={INTER})"; neuron_layers.add(l)
            resolved.append({"arm": a, "kind": "neuron", "comps": comps})
    cal = calibrate(m, tok, sorted(neuron_layers), dir_specs, args, args.out)
    if args.calibrate_only: return
    # ---- per-bench frozen data
    data = {}; dsha = {}
    for b, (f, _) in STAGES[args.stage].items():
        dsha[b] = sha_file(f"{DATA}/{f}")
        if items_n[b] > 0: data[b] = load_jsonl(f"{DATA}/{f}")[: items_n[b]]
    stage_ids = {it["item_id"] for b in data for it in data[b]}
    for other, spec in STAGES.items():
        if other != args.stage:
            for b, (f, _) in spec.items(): assert not (stage_ids & {r["item_id"] for r in load_jsonl(f"{DATA}/{f}")}), f"cross-stage item reuse: {args.stage} vs {other}/{b}"
    scales_used = {}
    for R in resolved:
        a = R["arm"]; aid = a["arm_id"]
        if R["kind"] == "neuron":
            nedits = []; desc = []
            for (l, n, dm) in R["comps"]:
                sc = float(cal["neuron_mean_abs"][l][n]); ms = float(cal["neuron_mean_signed"][l][n])
                # PRIMARY dose = additive (dose_mult * mean|act|). mult_equiv_* are descriptive only:
                #  abs   : neuron magnitude at its typical level is scaled by 1 + dose_mult (ablate -> 0.0)
                #  signed: factor m with ms*(m-1) = delta, i.e. 1 + delta/mean_signed_act (None if |mean_signed| < 1e-3*mean_abs; ablate -> 0.0)
                if dm == 0: nedits.append((l, n, "ablate", 0.0)); desc.append({"layer": l, "neuron": n, "dose_mult": 0, "op": "ablate", "scale_mean_abs": sc, "mean_signed_act": ms, "delta": None, "mult_equiv_abs": 0.0, "mult_equiv_signed": 0.0})
                else: nedits.append((l, n, "add", dm * sc)); desc.append({"layer": l, "neuron": n, "dose_mult": dm, "op": "add", "scale_mean_abs": sc, "mean_signed_act": ms, "delta": dm * sc, "mult_equiv_abs": 1 + dm, "mult_equiv_signed": (1 + dm * sc / ms) if abs(ms) >= 1e-3 * sc else None})
            edits = Edits(m, nedits, []); info = {"kind": "neuron", "edits": desc}
        else:
            if R["fscale"] is not None: sc = R["fscale"]; other = None; ssrc = "file"
            else:
                sc = cal["dirs"][R["dkey"]]["mean_abs" if args.dir_unit == "abs" else "mean_signed"]; other = cal["dirs"][R["dkey"]]["mean_signed" if args.dir_unit == "abs" else "mean_abs"]; ssrc = "calib200"
                if args.dir_unit == "signed" and sc == 0: sys.exit("zero signed scale")
            vec = (a["alpha"] * sc) * R["u"]
            edits = Edits(m, [], [(R["site"], vec)])
            info = {"kind": "direction", "site_layer_index": R["site"], "convention": R["conv"], "alpha": a["alpha"], "unit": "abs(file)" if ssrc == "file" else args.dir_unit, "scale": sc, "scale_source": ssrc, "scale_other": other, "dir_sha16": R["dkey"].split("@")[0], "role": a.get("role", "arm_file"), "pt": a.get("pt"), "pt_index": a.get("pt_index")}
        scales_used[aid] = info
        cfg_hash = hashlib.sha256(json.dumps([info, env["dtype"], env["bs"], env["model_snapshot"], dsha, args.dir_unit, args.stage], sort_keys=True, default=str).encode()).hexdigest()[:16]
        fp = f"{sout}/{aid}.jsonl"; done = set(); 
        if os.path.exists(fp):
            old = load_jsonl(fp)
            if old and old[0]["config_hash"] != cfg_hash:
                if not args.overwrite: sys.exit(f"{aid}: existing rows have different config_hash ({old[0]['config_hash']} vs {cfg_hash}); use --overwrite")
                os.remove(fp); old = []
            assert all(r["bench"] in STAGES[args.stage] for r in old), f"{fp}: rows from another stage"
            done = {(r["bench"], r["item_id"]) for r in old}
        t_arm = time.perf_counter(); nnew = 0
        with open(fp, "a") as fo, torch.inference_mode(), edits:
            for b in STAGES[args.stage]:
                for it in data.get(b, []):
                    if (b, it["item_id"]) in done: continue
                    t0 = time.perf_counter()
                    base = {"arm_id": aid, "bench": b, "item_id": it["item_id"], "order": it["order"], "config_hash": cfg_hash}
                    if b == "wiki":
                        ids = tok(it["text"], return_tensors="pt", add_special_tokens=False)["input_ids"][:, :256]
                        lg = m(input_ids=ids, use_cache=False).logits[0].float()
                        nll = torch.nn.functional.cross_entropy(lg[:-1], ids[0, 1:], reduction="sum").item(); n = ids.shape[1] - 1
                        row = {**base, "nll_sum": nll, "n_pred_tokens": n, "ppl_item": math.exp(nll / n)}
                    else:
                        ids = tok(mmlu_prompt(it) if b in MMLU_LIKE else arc_prompt(it), return_tensors="pt")
                        lp = torch.log_softmax(forward_last(m, ids), -1)[letter_ids]
                        pred = LET[int(lp.argmax())]
                        row = {**base, "correct": pred == it["gold"], "pred": pred, "gold": it["gold"], "logprobs": {c: float(lp[i]) for i, c in enumerate(LET)}, "n_tokens": int(ids["input_ids"].shape[1])}
                    row.update({"edit": info, "dtype": args.dtype, "threads": args.threads, "bs": 1, "torch": env["torch"], "dataset_sha256": dsha[b],
                                "peak_rss_mb_process": round(peak_rss_mb(), 1), "rss_mb_now": round(rss_mb(), 1), "wall_s": round(time.perf_counter() - t0, 4)})
                    fo.write(json.dumps(row) + "\n"); fo.flush(); os.fsync(fo.fileno()); nnew += 1
        rows = load_jsonl(fp); summ = {"arm_id": aid, "config_hash": cfg_hash, "edit": info, "env": env, "new_rows_this_run": nnew, "arm_wall_s_this_run": round(time.perf_counter() - t_arm, 1), "peak_rss_mb_process": round(peak_rss_mb(), 1)}
        for b in ("mmlu", "mmlu_screen", "arc"):
            rr = [r for r in rows if r["bench"] == b]
            if rr: summ[b] = {"n": len(rr), "acc": sum(r["correct"] for r in rr) / len(rr)}
        ww = [r for r in rows if r["bench"] == "wiki"]
        if ww: summ["wiki"] = {"n": len(ww), "ppl": math.exp(sum(r["nll_sum"] for r in ww) / sum(r["n_pred_tokens"] for r in ww))}
        json.dump(summ, open(f"{sout}/{aid}.summary.json", "w"), indent=1)
        print(f"[arm] {aid}: new={nnew} " + " ".join(f"{b}={summ[b]}" for b in ("mmlu", "mmlu_screen", "arc", "wiki") if b in summ), flush=True)
    sp = f"{sout}/scales_used.json"; old = json.load(open(sp)) if os.path.exists(sp) else {}
    old.update(scales_used); json.dump(old, open(sp, "w"), indent=1)

if __name__ == "__main__": main()
