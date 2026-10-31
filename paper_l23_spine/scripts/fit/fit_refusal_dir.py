#!/usr/bin/env python3
"""Recompute round 2's refusal direction on Qwen3.5-4B (CPU) and export the L23 unit vector.

Same recipe as refusal_direction.py's `last` variant: r_L = unit(mean(refuse) - mean(comply))
over the residual stream at the last prompt token, fitted on the 96 fit pairs of contrast128
(every index not divisible by 4) that survive the Arditi-style filter -- only pairs whose refuse
side the unsteered model actually refuses on a 16-token greedy check.

Writes every layer's direction plus the median residual norms to the gitignored
steer/local/dirs.pt, and the single L23 unit vector (fp32) to the artifacts path for steer/demo.
Reads no tool_bait prompt text: the refusal markers come from the committed id manifest.
"""
import argparse
import json
import os
import time

import torch

import direction_break as db
from direction_break import LOCAL, PROBES, Runner, load_jsonl, maxrss_gb

LAYER = 23
SELECT_NEW = 16
ARTIFACT_DIR = "/opt/cursor/artifacts"


def refused(reply, markers):
    return any(m.lower() in reply.lower() for m in markers)


@torch.no_grad()
def last_token_feats(run, prompts):
    """[n, n_layers+1, hidden] float32 residual stream at the last prompt token, plus median norms."""
    last, norms = [], []
    for i in range(0, len(prompts), db.FWD_BATCH):
        seqs = [run.tok(p).input_ids for p in prompts[i:i + db.FWD_BATCH]]
        ids, mask, pos = run._batch(seqs)
        hs = run.model(input_ids=ids, attention_mask=mask, position_ids=pos,
                       output_hidden_states=True).hidden_states
        h = torch.stack([x.float() for x in hs], 1)  # [B, L+1, T, H]
        last.append(h[:, :, -1].clone())
        norms.append(h.norm(dim=-1).permute(0, 2, 1)[mask.bool()].clone())
        del hs, h
    return torch.cat(last), torch.cat(norms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.time()

    def log(m):
        print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)

    pairs = load_jsonl(os.path.join(LOCAL, "contrast128.jsonl"))
    fit = [p for i, p in enumerate(pairs) if i % 4]
    held = [p for i, p in enumerate(pairs) if i % 4 == 0]
    manifest = load_jsonl(os.path.join(PROBES, "refusal_probes.ids.jsonl"))
    markers = manifest[0]["refusal_markers"] + db.EXTRA_REFUSAL_MARKERS
    log(f"pairs={len(pairs)} fit={len(fit)} held={len(held)} markers={len(markers)}")

    run = Runner(db.MODEL)
    user = lambda t: run.chat([{"role": "user", "content": t}])
    log(f"model loaded: {run.n_layers} layers, hidden={run.model.config.hidden_size}")

    reps = run.generate([user(p["refuse"]) for p in fit], SELECT_NEW)
    fit_used = [p for p, x in zip(fit, reps) if refused(x, markers)]
    fams = sorted({p["topic"].split("/")[0] for p in fit})
    log(f"fit pairs refused by the unsteered model: {len(fit_used)}/{len(fit)} | "
        + " ".join(f"{f}={sum(p['topic'].startswith(f + '/') for p in fit_used)}"
                   f"/{sum(p['topic'].startswith(f + '/') for p in fit)}" for f in fams))

    rl, rn = last_token_feats(run, [user(p["refuse"]) for p in fit_used])
    cl, cn = last_token_feats(run, [user(p["comply"]) for p in fit_used])
    r = rl.mean(0) - cl.mean(0)
    dirs = r / r.norm(dim=-1, keepdim=True)  # [L+1, H] fp32 unit
    median_norm = torch.cat([rn, cn]).median(0).values
    log(f"directions fitted; |r| at L{LAYER} before norm = {r[LAYER].norm():.3f}, "
        f"median residual norm at L{LAYER} = {median_norm[LAYER]:.2f}")

    # reproduction check: held-out refusal-marker rate, unsteered vs L23 ablation
    held_p = [user(p["refuse"]) for p in held]
    base = sum(refused(x, markers) for x in run.generate(held_p, SELECT_NEW)) / len(held)
    run.steer.set(ablate=dirs[LAYER])
    abl = sum(refused(x, markers) for x in run.generate(held_p, SELECT_NEW)) / len(held)
    run.steer.set()
    log(f"held-out ({len(held)} refuse prompts, {SELECT_NEW} tokens): unsteered refusal={base:.3f}, "
        f"L{LAYER} ablated={abl:.3f}, drop={base - abl:+.3f}")

    os.makedirs(LOCAL, exist_ok=True)
    torch.save({"dirs": dirs, "median_norm": median_norm, "layer": LAYER,
                "fit_used_ids": [p["id"] for p in fit_used], "n_fit": len(fit),
                "heldout_base": base, "heldout_ablate": abl},
               os.path.join(LOCAL, "dirs.pt"))
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    torch.save(dirs[LAYER].contiguous().float(), os.path.join(ARTIFACT_DIR, "refusal_dir_L23.pt"))
    meta = {"model": db.MODEL, "layer": LAYER, "hidden": dirs.shape[-1], "dtype": "float32",
            "method": "unit(mean(refuse) - mean(comply)) at the last prompt token",
            "fit_pairs_used": len(fit_used), "fit_pairs_total": len(fit),
            "heldout_refusal_unsteered": base, "heldout_refusal_L23_ablated": abl,
            "seconds": time.time() - t0, "maxrss_gb": maxrss_gb()}
    json.dump(meta, open(os.path.join(LOCAL, "dirs_meta.json"), "w"), indent=1)
    log(f"saved {ARTIFACT_DIR}/refusal_dir_L23.pt  shape={tuple(dirs[LAYER].shape)} "
        f"norm={dirs[LAYER].norm():.6f}  maxrss={maxrss_gb():.1f}GB")


if __name__ == "__main__":
    main()
