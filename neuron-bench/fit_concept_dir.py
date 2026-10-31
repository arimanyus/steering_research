#!/usr/bin/env python3
"""Fit a 'genius/intelligent' concept direction by diff-of-means on contrastive prompts,
in our Qwen3.5-4B checkpoint. Also emits 20 random unit-direction controls at the push layer.
CPU-only. Loads the model, so run it from the queue, NOT alongside another 9GB job.

Outputs (in OUT dir):
  concept_dir_L{z}.pt         dict: dir(unit, d), layer_zero_idx=z, scale(mean|proj| on calib),
                              n_pos, n_neg, source, prompts_sha256
  concept_rand_dirs_L{z}.pt   dict: dirs(20,d) unit, seed, layer_zero_idx
Convention matches steer-spine/lib.py Hooks: an 'add' at 0-indexed layer z uses hs index k=z+1,
and the harness should push by coef = dose * scale along the unit dir (same dose convention as arms.jsonl).
"""
import os, sys, json, hashlib, argparse
sys.path.insert(0, "/workspace/steer-spine")
import torch
import lib  # load(), Hooks, render

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 16), b""):
            h.update(c)
    return h.hexdigest()

@torch.no_grad()
def capture_last_tok(tok, m, texts, zlayers, bs=8):
    """Mean-free last-prompt-token residuals at each 0-indexed layer z (lib hs index k=z+1).
    Returns {z: tensor(N, d)}."""
    ks = [z + 1 for z in zlayers]
    acc = {z: [] for z in zlayers}
    for i in range(0, len(texts), bs):
        batch = texts[i:i+bs]
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False)
        with lib.Hooks(m, capture=ks) as H:
            m(**enc)
        for z, k in zip(zlayers, ks):
            acc[z].append(torch.cat(H.cap[k]))   # H.cap[k] is a list of (b, d); last prompt token
    return {z: torch.cat(v) for z, v in acc.items()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="/workspace/neuron-bench/concept_prompts.jsonl")
    ap.add_argument("--layers", default="8,14,20", help="0-indexed layers (model.model.layers[z])")
    ap.add_argument("--out", default="/workspace/neuron-bench")
    ap.add_argument("--n_rand", type=int, default=20)
    ap.add_argument("--rand_layer", type=int, default=14, help="0-indexed layer for random-direction control; must match the primary arm layer")
    ap.add_argument("--seed", type=int, default=4242)
    args = ap.parse_args()
    zlayers = [int(x) for x in args.layers.split(",")]

    pairs = [json.loads(l) for l in open(args.prompts) if l.strip()]
    pos = [p["pos"] for p in pairs]
    neg = [p["neg"] for p in pairs]
    psha = sha256(args.prompts)

    tok, m = lib.load()
    # Render as bare prefixes continued by a short neutral stub so the last token is comparable.
    def render_plain(s):
        return tok.apply_chat_template(
            [{"role": "user", "content": "Continue appropriately."},
             {"role": "assistant", "content": s}],
            add_generation_prompt=False, tokenize=False, enable_thinking=False)
    P = [render_plain(s) for s in pos]
    N = [render_plain(s) for s in neg]
    allrows = [json.loads(l) for l in open(args.prompts) if l.strip()]

    actP = capture_last_tok(tok, m, P, zlayers)
    actN = capture_last_tok(tok, m, N, zlayers)

    os.makedirs(args.out, exist_ok=True)
    for z in zlayers:
        mu_p = actP[z].mean(0); mu_n = actN[z].mean(0)
        diff = (mu_p - mu_n).float()
        unit = diff / (diff.norm() + 1e-8)
        # dose scale: mean |residual . unit| across all calibration prompts (pos+neg)
        allact = torch.cat([actP[z], actN[z]]).float()
        scale = float((allact @ unit).abs().mean())
        torch.save({"dir": unit, "layer_zero_idx": z, "scale": scale,
                    "raw_norm": float(diff.norm()), "n_pos": len(pos), "n_neg": len(neg),
                    "source": "diff_of_means last-tok residual on concept_prompts",
                    "prompts_sha256": psha},
                   os.path.join(args.out, f"concept_dir_L{z}.pt"))
        print(f"L{z}: |diff|={diff.norm():.3f} scale(mean|proj|)={scale:.3f} d={unit.numel()}")

    # random unit-direction controls, matched to the primary arm's layer and
    # scaled by THAT layer's own mean |residual . dir| (per Asha: a control at a
    # different layer, or scaled off a different layer, is not matched).
    rz = args.rand_layer
    if rz not in actP:
        raise SystemExit(f"rand_layer {rz} not among captured layers {zlayers}; add it to --layers")
    d = actP[rz].shape[1]
    g = torch.Generator().manual_seed(args.seed)
    R = torch.randn(args.n_rand, d, generator=g)
    R = R / R.norm(dim=1, keepdim=True)
    allact_rz = torch.cat([actP[rz], actN[rz]]).float()  # (N, d) at the primary layer
    rand_scales = (allact_rz @ R.t().float()).abs().mean(0)  # (n_rand,) per-dir mean|proj|
    torch.save({"dirs": R, "scales": rand_scales, "seed": args.seed,
                "layer_zero_idx": rz,
                "source": "random unit dirs at primary layer, each scaled by its own mean|residual.dir|"},
               os.path.join(args.out, f"concept_rand_dirs_L{rz}.pt"))
    print(f"wrote {args.n_rand} random unit dirs at L{rz}, per-dir scales "
          f"min={float(rand_scales.min()):.3f} max={float(rand_scales.max()):.3f}")

if __name__ == "__main__":
    main()
