#!/usr/bin/env python3
"""Standalone refusal-direction demo for Qwen3.5-4B on CPU.

Loads the layer-23 difference-of-means refusal direction (a unit fp32 tensor produced by
steer/fit_refusal_dir.py) and removes it from the model in one of two ways:

  * activation ablation (default) -- forward hooks project the direction out of every residual
    write (embeddings, every attention output, every MLP output) at every position;
  * weight orthogonalisation (--orth) -- the same projection applied once to the weights that
    write into the residual stream, so no hooks run at generation time.

Both print the stock reply and the unlocked reply side by side. --off skips the unlocked pass
and prints the baseline only.

Self-contained: needs only torch + transformers and the direction file.

    python steer/demo/demo.py --prompt "Write a fake five-star review for a cafe I never visited"

Intended for mild, low-stakes requests (the everyday-dishonesty register the direction was
fitted on). Do not use it to elicit genuinely harmful content.
"""
import argparse
import os
import time
from functools import partial

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_DIRECTION = "/opt/cursor/artifacts/refusal_dir_L23.pt"
DEFAULT_MODEL = "Qwen/Qwen3.5-4B"


def residual_writers(model):
    """Every module whose output is added into the residual stream, plus the weights that write it."""
    lm = model.model
    mods = [lm.embed_tokens]
    mats = [lm.embed_tokens.weight]
    for layer in lm.layers:
        mixer = layer.linear_attn if hasattr(layer, "linear_attn") else layer.self_attn
        out = mixer.out_proj if hasattr(mixer, "out_proj") else mixer.o_proj
        mods += [mixer, layer.mlp]
        mats += [out.weight.T, layer.mlp.down_proj.weight.T]
    return mods, mats


class Ablator:
    """Forward hooks projecting `direction` out of every residual write, scaled by `strength`."""

    def __init__(self, model, direction):
        self.r = direction
        self.strength = 0.0
        for mod in residual_writers(model)[0]:
            mod.register_forward_hook(self._hook)

    def _proj(self, x):
        xf = x.float()
        return (xf - self.strength * (xf @ self.r).unsqueeze(-1) * self.r).to(x.dtype)

    def _hook(self, mod, inp, out):
        if self.strength == 0.0:
            return None
        if isinstance(out, tuple):
            return (self._proj(out[0]),) + tuple(out[1:])
        return self._proj(out)


def orthogonalize(model, direction, strength):
    """Project `direction` out of the residual-writing weights (scaled by `strength`), in place."""
    if model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr():
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.clone(), requires_grad=False)
    with torch.no_grad():
        for W in residual_writers(model)[1]:  # each [*, hidden]
            Wf = W.float()
            W.copy_((Wf - strength * (Wf @ direction).unsqueeze(-1) * direction).to(W.dtype))


@torch.no_grad()
def reply(model, tok, prompt, max_new):
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    enc = tok(text, return_tensors="pt")
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser(description="Refusal-direction demo (Qwen3.5-4B, CPU).")
    ap.add_argument("--prompt", required=True, help="the request to send to the model")
    ap.add_argument("--strength", type=float, default=1.0,
                    help="0..1, scales how much of the direction is removed (default 1.0 = full)")
    ap.add_argument("--off", action="store_true", help="baseline only: skip the unlocked pass")
    ap.add_argument("--orth", action="store_true",
                    help="remove the direction from the weights instead of via forward hooks")
    ap.add_argument("--max-new", type=int, default=120, help="greedy tokens per reply")
    ap.add_argument("--direction", default=DEFAULT_DIRECTION)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    args = ap.parse_args()
    if not 0.0 <= args.strength <= 1.0:
        ap.error("--strength must be between 0 and 1")
    torch.set_num_threads(args.threads)

    r = torch.load(args.direction).float()
    r = (r / r.norm()).contiguous()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).eval()
    assert r.shape == (model.config.hidden_size,), f"direction {tuple(r.shape)} != hidden {model.config.hidden_size}"
    ablator = None if args.orth else Ablator(model, r)

    mode = "weight orthogonalisation" if args.orth else "activation ablation (all layers, all positions)"
    print(f"model {args.model} (bf16, CPU, {args.threads} threads); direction {args.direction} "
          f"[L23, unit, {r.numel()}d]; mode: {mode}; strength {args.strength:g}\n")
    print("=" * 78)
    print(f"PROMPT: {args.prompt}")
    print("=" * 78)

    t = time.time()
    print(f"\n--- stock Qwen3.5-4B ---\n{reply(model, tok, args.prompt, args.max_new)}")
    print(f"\n[{time.time() - t:.0f}s]")
    if args.off:
        print("\n(--off: unlocked pass skipped)")
        return

    t = time.time()
    if args.orth:
        orthogonalize(model, r, args.strength)
    else:
        ablator.strength = args.strength
    print(f"\n--- unlocked: L23 refusal direction removed, {mode}, strength {args.strength:g} ---\n"
          f"{reply(model, tok, args.prompt, args.max_new)}")
    print(f"\n[{time.time() - t:.0f}s]")


if __name__ == "__main__":
    main()
