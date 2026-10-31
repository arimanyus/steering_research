#!/usr/bin/env python3
"""Try to reproduce refusal_dir_L23.pt (sha256 1fa2cd8e...) by refitting diff-of-means directions.

NOT RUN when written (RAM is held by another job). Needs the 4B model in bf16, ~9.5 GB peak. One job at a time.

Method (reuses tbL/common.py load()/render() and the activation code of tbL/dirs.py exactly)
  * contrast128_full.jsonl, 128 rows. Chat template on, enable_thinking=False (common.render).
  * Last prompt token, hidden_states[L], L in 22..25, left padding, batch 16, bf16, 4 threads.
  * Activations for all 128 refuse + 128 comply prompts are computed ONCE and cached to
    refit_L23_acts.pt. Later runs reuse the cache and need no model.
  * Each candidate fit set gives d = mean(refuse[S]) - mean(comply[S]) and u = d/|d|. Report cos(u, saved).

Recovered provenance (read before changing candidates)
  The "missing" fit script exists: arimanyus/steering_research (public), PR #4 branch
  cursor/dom-direction-break-c20b, commit 0857706, steer/fit_refusal_dir.py. Copies are in
  recovered_fit/. What it does:
    - pairs = contrast128 (128, file order); fit = pairs at 0-based index i with i%4 != 0 (96); held = i%4 == 0 (32)
    - fit_used = fit pairs whose REFUSE prompt the unsteered model refuses in a 16-token greedy reply (marker check)
    - r = mean(refuse) - mean(comply) over fit_used at hidden_states[23], last prompt token, chat template,
      enable_thinking=False; unit-normalised; saved as the L23 vector (fp32).
  The 75 ids it used survive in steer/results/stack_ablate.json (commit f1aa22e) -> recovered_fit/fit_used_ids.json.
  The refuse text used then is a template re-derivation (contrast128_build.py); the box file matches it exactly.
  The original code ran batch 8 with explicit position_ids and a different transformers build, so tiny bf16
  differences are expected. --orig-protocol repeats the activations that way as a sensitivity check.

Registered primary test (fixed before running): candidate FIT75_RERUN, hs index 23, raw diff-of-means, 'dirs' protocol.
  cos >= 0.99 -> "reproduced". Everything else is context; any other cell >= 0.99 is exploratory, not a registered pass.

Usage (after the RAM job is done):
  python refit_L23.py                      # acts via dirs.py protocol, evaluate all candidates
  python refit_L23.py --orig-protocol      # also recompute acts the way fit_refusal_dir.py did
  python refit_L23.py --eval-only          # no model: reuse caches
"""
import argparse, hashlib, json, os, platform, resource, sys, time

SW = "/workspace/steer-withheld"
os.environ.setdefault("TBL_THREADS", "4")          # common.py reads this; dirs.py default is 4
sys.path.insert(0, f"{SW}/tbL")

SAVED = f"{SW}/refusal_dir_L23.pt"
SAVED_SHA = "1fa2cd8e93032a555c5e1032f7463e00be3bd6da3b4f15ae021697673c0a4b97"
DATA = f"{SW}/contrast128_full.jsonl"
DATA_SHA = "a1e1902303ff8719e6b5ecef87684d0def57ef6712b0380e17d254ad68476734"
IDS = f"{SW}/recovered_fit/fit_used_ids.json"
IDS_SHA = "b91657b3964fc133b5bcb3cd9962e6303296a56efc1b266a25ec7cb6a283fa56"
CACHE = f"{SW}/refit_L23_acts.pt"
OUT = f"{SW}/refit_L23_results.json"
HS_INDICES = (22, 23, 24, 25)
BATCH = 16            # dirs.py
BATCH_ORIG = 8        # direction_break.FWD_BATCH used by the original fit
THRESH = 0.99
PRIMARY = ("FIT75_RERUN", 23, "raw", "dirs")
MIN_AVAIL_GB = 11.0


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def mem_avail_gb():
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable"):
            return int(l.split()[1]) / 1024 / 1024
    return float("nan")


T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def candidates(rows, ids):
    """Preregistered candidate fit sets -> (index list, note). Order is the table order."""
    pos = {r["id"]: i for i, r in enumerate(rows)}
    n = len(rows)
    mod4_fit = [i for i in range(n) if i % 4]
    mod4_held = [i for i in range(n) if i % 4 == 0]
    c = {
        "ALL128": (list(range(n)), "reference: what tbL/dirs.py computed (logged cos 0.9469 at hs 23)"),
        "FIRST96": (list(range(96)), "naive reading of '96 fit pairs' = first 96 in file order; contradicted by the recovered code (i%4 split)"),
        "FIT96_MOD4": (mod4_fit, "recovered code's fit split before the refusal filter: 0-based i%4 != 0"),
        "FIT75_RERUN": ([pos[x] for x in ids["rerun75"]], "PRIMARY: the 75 ids fit_refusal_dir.py kept (stack_ablate.json fit_used_ids)"),
        "FIT69_ROUND2": ([pos[x] for x in ids["round2_69"]], "round-2 69 ids (original wording); documented alternative, not the saved vector's run"),
        "HELD32": (mod4_held, "negative control: the 32 held-out pairs (i%4 == 0), never in any fit"),
    }
    sizes = {"ALL128": 128, "FIRST96": 96, "FIT96_MOD4": 96, "FIT75_RERUN": 75, "FIT69_ROUND2": 69, "HELD32": 32}
    for k, (ix, _) in c.items():
        assert len(ix) == sizes[k] and len(set(ix)) == len(ix), (k, len(ix))
    assert set(c["FIT75_RERUN"][0]) <= set(mod4_fit), "75 ids must lie inside the i%4 fit split"
    return {k: (sorted(ix), note) for k, (ix, note) in c.items()}


def compute_acts(rows, orig):
    """Return {'meta':..., 'refuse':{L:[128,H]}, 'comply':{L:[128,H]}}. orig=False is the dirs.py code path."""
    import torch
    from common import load, render, MODEL
    tok, m = load()
    log(f"model loaded; rss {rss_gb():.1f} GB; threads {torch.get_num_threads()}")
    rendered = {k: [render(tok, [{"role": "user", "content": r[k]}]) for r in rows] for k in ("refuse", "comply")}
    rhash = hashlib.sha256("\x00".join(rendered["refuse"] + rendered["comply"]).encode()).hexdigest()
    same_ids = all(tok(p, add_special_tokens=False).input_ids == tok(p).input_ids for k in rendered for p in rendered[k])
    bs = BATCH_ORIG if orig else BATCH
    out = {}
    t_acts = time.time()
    for key in ("refuse", "comply"):
        acc = {L: [] for L in HS_INDICES}
        for i in range(0, len(rows), bs):
            ps = rendered[key][i:i + bs]
            if orig:   # direction_break.Runner._batch: manual left pad, mask, position_ids
                seqs = [tok(p).input_ids for p in ps]
                n = max(len(s) for s in seqs)
                ids = torch.tensor([[tok.pad_token_id] * (n - len(s)) + s for s in seqs])
                mask = torch.tensor([[0] * (n - len(s)) + [1] * len(s) for s in seqs])
                pos = (mask.cumsum(-1) - 1).clamp(min=0)
                kw = dict(input_ids=ids, attention_mask=mask, position_ids=pos)
            else:      # tbL/dirs.py
                kw = dict(tok(ps, return_tensors="pt", padding=True, add_special_tokens=False))
            with torch.no_grad():
                hs = m(**kw, output_hidden_states=True).hidden_states
            for L in HS_INDICES:
                acc[L].append(hs[L][:, -1].float())
            log(f"  {'orig' if orig else 'dirs'} {key} {min(i + bs, len(rows))}/{len(rows)} rss {rss_gb():.1f} GB")
        out[key] = {L: torch.cat(v) for L, v in acc.items()}
    meta = {"protocol": "orig" if orig else "dirs", "batch": bs, "n_hidden_states": len(hs), "dtype": "bfloat16",
            "model": MODEL, "model_name_or_path": getattr(m.config, "_name_or_path", None),
            "rendered_prompts_sha256": rhash, "tok_with_and_without_special_tokens_identical": same_ids,
            "acts_seconds": time.time() - t_acts, "peak_rss_gb": rss_gb(),
            "torch": torch.__version__, "transformers": __import__("transformers").__version__}
    try:
        from huggingface_hub import snapshot_download
        meta["snapshot_dir"] = snapshot_download(MODEL, local_files_only=True)
    except Exception as e:                                   # informational only
        meta["snapshot_dir"] = f"unavailable: {type(e).__name__}"
    return {"meta": meta, "refuse": out["refuse"], "comply": out["comply"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig-protocol", action="store_true", help="also recompute acts like fit_refusal_dir.py (batch 8, position_ids)")
    ap.add_argument("--eval-only", action="store_true", help="never load the model; use caches only")
    ap.add_argument("--force-mem", action="store_true", help=f"skip the MemAvailable >= {MIN_AVAIL_GB} GB guard")
    ap.add_argument("--overwrite", action="store_true", help="allow replacing results json / cache")
    args = ap.parse_args()
    import torch
    torch.set_num_threads(int(os.environ["TBL_THREADS"]))

    # ---- inputs, pinned ----
    inputs = {p: sha256(p) for p in (SAVED, DATA, IDS, f"{SW}/tbL/dirs.py", f"{SW}/tbL/common.py", os.path.abspath(__file__),
                                     f"{SW}/recovered_fit/fit_refusal_dir.py", f"{SW}/recovered_fit/contrast128_build.py")}
    assert inputs[SAVED] == SAVED_SHA, "saved vector changed"
    assert inputs[DATA] == DATA_SHA, "contrast set changed"
    assert inputs[IDS] == IDS_SHA, "recovered id list changed"
    rows = [json.loads(l) for l in open(DATA) if l.strip()]
    assert len(rows) == 128 and len({r["id"] for r in rows}) == 128
    ids = json.load(open(IDS))
    saved = torch.load(SAVED).float()
    assert saved.shape == (2560,) and abs(float(saved.norm()) - 1) < 1e-5
    cands = candidates(rows, ids)
    log(f"inputs pinned; saved norm {float(saved.norm()):.7f}; candidates: " + ", ".join(f"{k}={len(v[0])}" for k, v in cands.items()))

    # ---- activations (ONCE), cached ----
    caches = {}
    want = [("dirs", CACHE)] + ([("orig", CACHE.replace(".pt", "_orig.pt"))] if args.orig_protocol else [])
    for proto, path in want:
        if os.path.exists(path):
            c = torch.load(path)
            assert c["meta"]["protocol"] == proto
            log(f"reusing cache {path}")
        elif args.eval_only:
            log(f"--eval-only and no cache {path}; skipping protocol {proto}")
            continue
        else:
            av = mem_avail_gb()
            if av < MIN_AVAIL_GB and not args.force_mem:
                sys.exit(f"MemAvailable {av:.1f} GB < {MIN_AVAIL_GB} GB: another job holds RAM. Re-run when it finishes (or --force-mem).")
            c = compute_acts(rows, orig=(proto == "orig"))
            c["meta"]["rows_sha256"] = inputs[DATA]
            torch.save(c, path + ".tmp"); os.replace(path + ".tmp", path)
            log(f"cached {path} ({c['meta']['acts_seconds']:.0f}s acts, peak rss {c['meta']['peak_rss_gb']:.1f} GB)")
        caches[proto] = c
    if not caches:
        sys.exit("no activations available")

    # ---- evaluate ----
    cs = torch.nn.functional.cosine_similarity
    results, units = [], {}
    for proto, c in caches.items():
        for variant in ("raw", "unitacts"):   # unitacts: normalise every activation before averaging (sensitivity only)
            for name, (ix, _) in cands.items():
                for L in HS_INDICES:
                    R, C = c["refuse"][L][ix].double(), c["comply"][L][ix].double()
                    if variant == "unitacts":
                        R, C = R / R.norm(dim=1, keepdim=True), C / C.norm(dim=1, keepdim=True)
                    d = (R.mean(0) - C.mean(0)).float()
                    u = d / d.norm()
                    cos = float(cs(u, saved, dim=0))
                    results.append({"protocol": proto, "variant": variant, "cand": name, "n": len(ix), "hs": L, "cos": cos,
                                    "raw_norm": float(d.norm()), "max_abs_diff_unit": float((u - saved).abs().max()),
                                    "registered_primary": (name, L, variant, proto) == PRIMARY, "ge_0.99": cos >= THRESH})
                    if (variant, L) == ("raw", 23):
                        units[(proto, name)] = u
    # pairwise cosines at hs 23 (raw, dirs): shows how far each fit set is from the others
    pw = None
    if "dirs" in caches:
        names = list(cands)
        pw = {a: {b: float(cs(units[("dirs", a)], units[("dirs", b)], dim=0)) for b in names} for a in names}

    # ---- print ----
    print()
    hdr = f"{'protocol':8} {'variant':8} {'candidate':13} {'n':>4} " + " ".join(f"hs{L:>2}  " for L in HS_INDICES)
    print(hdr); print("-" * len(hdr))
    for proto in caches:
        for variant in ("raw", "unitacts"):
            for name in cands:
                cells = []
                for L in HS_INDICES:
                    r = next(x for x in results if (x["protocol"], x["variant"], x["cand"], x["hs"]) == (proto, variant, name, L))
                    cells.append(f"{r['cos']:.4f}{'*' if r['ge_0.99'] else ' '}{'P' if r['registered_primary'] else ' '}")
                print(f"{proto:8} {variant:8} {name:13} {len(cands[name][0]):4d} " + " ".join(cells))
        print()
    print("* = cos >= 0.99   P = registered primary cell (FIT75_RERUN, hs 23, raw, dirs)")
    prim = next((x for x in results if x["registered_primary"]), None)
    best = max((x for x in results if x["cand"] not in ("HELD32",)), key=lambda x: x["cos"])
    if prim:
        verdict = "REPRODUCED" if prim["cos"] >= THRESH else "NOT REPRODUCED at the registered cell"
        print(f"\nPRIMARY cos = {prim['cos']:.6f} -> {verdict}")
    print(f"best non-control cell: {best['protocol']}/{best['variant']}/{best['cand']}/hs{best['hs']} cos {best['cos']:.6f}")
    chk = next((x for x in results if (x["protocol"], x["variant"], x["cand"], x["hs"]) == ("dirs", "raw", "ALL128", 23)), None)
    if chk:
        print(f"consistency check vs logged tbL/dirs.log: ALL128 hs23 cos {chk['cos']:.4f} (logged 0.9469)")

    # ---- write ----
    if os.path.exists(OUT) and not args.overwrite:
        sys.exit(f"{OUT} exists; use --overwrite")
    json.dump({
        "task": "reproduce refusal_dir_L23.pt by diff-of-means refit", "threshold": THRESH,
        "registered_primary": dict(zip(("cand", "hs", "variant", "protocol"), PRIMARY)),
        "primary_cos": prim["cos"] if prim else None,
        "primary_verdict": ("reproduced" if prim and prim["cos"] >= THRESH else "not reproduced") if prim else None,
        "logged_all128_hs23_cos_in_dirs_log": 0.9469,
        "candidates": {k: {"n": len(v[0]), "note": v[1], "indices": v[0], "ids": [rows[i]["id"] for i in v[0]]} for k, v in cands.items()},
        "results": results, "pairwise_cos_hs23_raw_dirs": pw,
        "input_sha256": inputs, "acts_meta": {p: c["meta"] for p, c in caches.items()},
        "env": {"python": platform.python_version(), "torch": torch.__version__, "threads": torch.get_num_threads()},
        "wall_seconds": time.time() - T0,
    }, open(OUT, "w"), indent=1)
    log(f"wrote {OUT}")


if __name__ == "__main__":
    main()
