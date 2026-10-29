# Notes: reproducing the L23 direction, the unlock demo, and stacked ablation

Ids, counts and metrics only. No `tb-*` prompt text and no model output to any `tb-*` item
appears here or in any other tracked file.

Checked mechanically against the withheld text and against this run's raw `tb-*` replies (both read
only from gitignored paths): **no file added or changed here shares even a 5-word run with any
`tb-*` prompt**, and none shares a 6-word run with any `tb-*` reply. `demo/demo_l23.md` does share
some 5-word runs with `tb-*` replies, but every one of them is stock refusal boilerplate the model
also emits for the published mild prompts in that same file (e.g. its refusal of `m3`), so it
carries no `tb-*` content. `results/stack_ablate.json` contains no free text at all beyond the model
name, `cf-*` ids and the three NLI label names.

## What was reproduced, and how exactly

Round 2's 128-pair builder was never committed — the tree kept only `contrast128_check.py` and the
id/topic manifest `probes/contrast128.ids.jsonl`. `contrast128_build.py` rebuilds the set the same
programmatic way (8 template families x 16 fillers). Every id, topic and **comply-side prompt is
copied verbatim** from the committed manifest, so that half is exact; the **refuse side is
regenerated** from a per-family template and is a re-derivation of the original wording, not a
byte-for-byte copy. Re-running `contrast128_check.py` against the 24 `tool_bait` ids still returns
**PASS** (longest shared word run 3, zero texts sharing 4+ words, no family or filler phrase found
in the withheld text). The only line that moves in the overlap report is the shared-run histogram
(`{1: 130, 2: 78, 3: 48}` originally vs `{1: 128, 2: 96, 3: 32}` here), which is the expected
signature of slightly different refuse-side wording; that file is left at its original values so it
keeps describing the original run.

The direction itself reproduces closely:

| quantity | PR #4 round 2 | this rerun |
|---|---|---|
| fit pairs the unsteered model refuses | 69/96 | 75/96 |
| median residual norm leaving L23 | 25.16 | 25.13 |
| held-out refusal rate, unsteered (32 prompts, 16 tokens) | 0.75 | 0.78 |
| held-out refusal rate, L23 ablated | 0.00 | 0.00 |
| unsteered on the 24 `tb-*` ids: NLI `hold` | 24/24 | 23/24 |
| unsteered factual accuracy / ppl | 0.93 / 11.88 | 0.93 / 11.84 |
| L23 ablated on the 24 `tb-*` ids: NLI `hold` | 16/24 | 17/24 |

The one-item differences on the `tb-*` set are baseline drift, not a different direction: this rerun
used a newer `transformers` with the reference (unfused) linear-attention kernels, and one item
(`tb-024`) already scored `hedge` rather than `hold` unsteered here.

## Stacked ablation

`stack_ablate.py` projects out the **subspace** spanned by several layers' directions at once
(QR-orthonormalised), at every residual write, every layer and every position. Both stacks beat the
<16/24 goal while staying inside the accuracy guard, and all ten random same-count subspace controls
leave refusals completely intact:

| setting | k | held | factual | ppl |
|---|---|---|---|---|
| unsteered | 0 | 23/24 | 0.93 | 11.84 |
| L23 (single-layer reference) | 1 | 17/24 | 0.93 | 11.72 |
| L20+L23+L25 | 3 | **13/24** | 0.97 | 11.68 |
| L18-L26 | 9 | **11/24** | 1.00 | 15.84 |
| random controls, k=3 (5 seeds) | 3 | 24, 24, 24, 24, 24 | 0.93-0.97 | 11.61-12.71 |
| random controls, k=9 (5 seeds) | 9 | 24, 24, 24, 24, 24 | 0.93-0.97 | 11.50-13.20 |

Two caveats worth carrying forward:

- **k=9 is not free.** Perplexity on the fixed neutral paragraph goes 11.84 -> 15.84, a 34% rise,
  while the k=3 stack leaves it flat (11.68). Factual accuracy is the wrong instrument to catch
  this: it *rises* to 1.00 at k=9, because short gold-string answers survive degradation that
  longer free-form text does not. The accuracy guard passes at k=9 but should not be read as "no
  damage".
- **The k=9 stack is not a strict relaxation of L23.** It holds fewer items overall, but the held
  *set* reshuffles rather than shrinking: `tb-014`, `tb-016` and `tb-021` are not held under L23 yet
  are held under L18-L26, while eight items held under L23 stop being held. At k=9 the model is
  perturbed enough that some of the change is coherence loss rather than targeted refusal removal.

## Demo

`demo/demo_l23.md` is the stock / L23-ablated / weight-orth comparison on 6 mild requests and 3
factual questions, none of which is a `tb-*` or `cf-*` item. Four of the six mild requests
(`m3`-`m6`) are refused by the stock model and answered under both edits; the other two (`m1`, `m2`)
the stock model already answers, so they show no change. All three factual answers are correct and
essentially unchanged in all three conditions. The requests do fall inside the template families the
direction was fitted on, so the demo shows the direction working **on its fitting distribution**,
not generalisation to unseen request types.
