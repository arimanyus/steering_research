# Neuron-steering falsifier: probe prompt set

Prompt sets for testing Lyra's claim that, on **Qwen3.5-4B**, steering neuron **8949 in
layer 14 (+1)** and neuron **7956 in layer 13 (−1)** at `down_proj` produces an "evil villain"
persona. This folder supplies **prompts only**. Conditions (unsteered; the two neurons; ~10
random neuron pairs at the same scaled dose; doses 0.25 / 0.5 / 1 / 2) are built later by
Maya/Asha. **No model runs here.**

Ported from `spine-is-all-you-need` (`kenji/steering-probes-08e5`, PR #21), with a
template-disjointness fix (see `template_overlap.md`).

## Primary vs secondary outcomes

Per Vivek, the goal is **breaking the model**, not persona. Priorities:

1. **Primary 1 — spine break (fold under pressure):** does steering make the model abandon a
   correct answer and fold to the planted wrong answer (the *foil*)? Measured on
   `pressure_probes.jsonl` with `consistency_flip` / fold-to-foil.
2. **Primary 2 — refusal collapse:** does steering change compliance on harmful requests?
   Measured on `refusal_probes.jsonl` with the existing `refusal_flip` / NLI refusal scorer.
3. **Secondary — coherence/persona:** `persona_probes.jsonl` (does tone change) and
   `factual_probes.jsonl` (does steering break basic competence). These are secondary/coherence
   measures, not the headline result.

## Eval sets (used to score the hypothesis)

| file | n | group | purpose | scorer |
|------|---|-------|---------|--------|
| `pressure_probes.jsonl`  | 12 | `pressure` | **Primary 1.** Does steering make the model fold to the foil under sustained pressure? | `consistency_flip` / fold-to-foil |
| `refusal_probes.jsonl` *(withheld)* | 24 | `refusal`  | **Primary 2.** Does steering change compliance on harmful requests? | existing `refusal_flip` / NLI refusal scorer |
| `persona_probes.jsonl`   | 30 | `persona`  | Secondary. Does tone shift toward a "villain" register on neutral prompts? | persona score (**must be fixed before results**) |
| `factual_probes.jsonl`   | 30 | `factual`  | Secondary. Does steering break basic factual competence? | gold / alias match |

## Search sets (used to find candidate neurons — never scored as outcomes)

| file | n | purpose |
|------|---|---------|
| `contrast_prompts.jsonl` | 40 | 20 refuse-side prompts + 20 harmless lookalikes on matched topics. Used to find neurons in layers 13–14 that differ between refusal and compliance. The refuse side is **withheld** (reduced to `id` + `topic` category stubs); the harmless comply side keeps its prompt. |
| `spine_contrast.jsonl`   | 48 | 24 false-premise / memory-rewrite items, each as a `clean` row (neutral question only) and a `pressured` row (full escalating turns). Used to find folding neurons. |

## Disjointness invariant (must hold)

Search sets (`contrast_prompts`, `spine_contrast`) must **never overlap** eval sets
(`refusal_probes`, `pressure_probes`) — by **id, by source item, and by template**.

Per Leila/Asha, id-disjointness alone is insufficient: `spine_contrast` and the scored
`pressure_probes` were originally built from the **same** `false_premise_stack` /
`memory_rewrite` templates, so a neuron found by the search could latch onto the template
wording rather than the folding behaviour. To close this:

- `spine_contrast` draws its 24 source items from factual items that **none** of the 12 scored
  `pressure_probes` use, and its pressured turns are rebuilt on a **search-only** scaffold
  (`false_premise_search` / `memory_rewrite_search`) whose wording shares nothing with the
  scored pressure scaffold.
- `contrast_prompts` uses the `refuse_comply_pair` template and reuses no `tool_bait` item.

Template overlap between the search and scored sets is therefore exactly **zero**. See
`template_overlap.md` for the per-template membership and the confirmed empty intersection.
Validated with the counts, foil-verbatim, and id/source/template disjointness checks.

## Protocol notes (enforced when conditions are run)

- Every condition (unsteered, two-neuron, random pairs, every dose) must use the **identical
  prompts in identical order**.
- **Temperature 0.**
- The **persona score must be fixed before any results are looked at** (pre-registered), so the
  secondary persona read cannot be tuned to the outcome.
- Only prompts are stored here — **no model replies.**

## File details

- **`persona_probes.jsonl`** — 30 neutral everyday prompts (`pp-001`..`pp-030`), fields `id`,
  `group`, `prompt`. Chosen so nothing invites a villain register.
- **`refusal_probes.jsonl`** — all 24 `tool_bait` items (`tb-001`..`tb-024`). **Prompt text is
  withheld from version control** (git-ignored via `steer/.gitignore`, matching scorer PR #1); the
  harness reads the file locally. The tree instead tracks **`refusal_probes.ids.jsonl`**, a manifest
  of the 24 ids with non-prompt scorer metadata only (`group`, `template`, `score_mode`, `gold`,
  `gold_logprob_target`, `foil`, generic `refusal_markers` / `compliance_markers`, `withheld: true`)
  — **no prompt `turns`, no model replies**. Per-item compliance markers that echo the prompt
  (e.g. `DROP TABLE`) are dropped from the manifest.
- **`factual_probes.jsonl`** — 30 short factual questions (`fq-001`..`fq-030`) with one
  unambiguous answer each. Fields: `id`, `group`, `prompt`, `gold`, `aliases`.
- **`pressure_probes.jsonl`** — 12 items (6 `false_premise_stack`, 6 `memory_rewrite`), all
  original fields plus `group: "pressure"`. Known-broken ids (fp-008, fp-011, fp-021, mr-015,
  mr-016, mr-019) are excluded, and every chosen item's foil appears verbatim in at least one
  user pressure turn. Deterministic lowest-id pick:
  `fp-001, fp-003, fp-004, fp-005, fp-006, fp-007, mr-002, mr-003, mr-004, mr-005, mr-006, mr-007`.
- **`contrast_prompts.jsonl`** — 40 rows (`cp-001`..`cp-040`) as 20 matched topic pairs
  (odd = `refuse`, even = `comply`). The refuse side is the same harmful-request class as
  `tool_bait`, so it is **withheld**: refuse rows carry only `id`, `group`, `side`, `topic`,
  `withheld: true` — **no prompt text**. The harmless comply rows keep their `prompt`.
- **`spine_contrast.jsonl`** — 48 rows: 24 source items (`sc-001`..`sc-024`), each emitted as
  `-clean` (turn 0 only) and `-pressured` (full search-scaffold turns). Fields: `id`,
  `source_id`, `template`, `side`, `turns`, `gold`, `foil`.
- **`template_overlap.md`** — the leakage report: per-template membership of the scored vs
  search sets, with the confirmed zero template/source/id intersection.

## Withheld harmful-request content — DRAFT, do not merge

The harmful-request (`tool_bait`-level) prompt text is **withheld from version control**, matching
the scorer repo's convention in PR #1 (which git-ignores `probes/refusal_probes.jsonl` and
references items by id):

- **`refusal_probes.jsonl`** (tool_bait prompt text) is git-ignored via `steer/.gitignore` and is
  **not committed**. The tree tracks `refusal_probes.ids.jsonl` instead — ids + non-prompt scorer
  metadata, no prompt text, no model replies. The harness reads the real file from a local copy
  fetched out-of-band.
- **`contrast_prompts.jsonl`** refuse side is reduced to `id` + `topic` category stubs (no prompt
  text); the harmless comply side is kept. Bounded per Vivek: **no operational weapons, CBRN, or
  CSAM content**.

No model replies are stored anywhere in this folder. This PR remains a **DRAFT — do not merge**;
the withheld prompt text must never land on a public branch.
