# Template-overlap / leakage report

Leila/Asha's point: id-disjointness between the neuron-search sets and the scored
sets is not sufficient. If the search set is built from the **same pressure
templates** as the scored set, candidate neurons can latch onto the template
wording rather than the folding behaviour, and the evaluation leaks. This report
confirms the search sets share **no template** (and no source item) with the
scored sets.

## Scored sets

### `pressure_probes.jsonl` (Primary 1, fold-to-foil)

| template | ids |
|---|---|
| `false_premise_stack` | fp-001, fp-003, fp-004, fp-005, fp-006, fp-007 |
| `memory_rewrite` | mr-002, mr-003, mr-004, mr-005, mr-006, mr-007 |

Template set: {`false_premise_stack`, `memory_rewrite`}

### `refusal_probes.jsonl` (Primary 2, refusal collapse)

Prompt text is **withheld** (git-ignored, matching scorer PR #1); referenced in-tree by id via
`refusal_probes.ids.jsonl`. Templates below are read from the id manifest, not from prompt text.

| template | count |
|---|---|

Template set: {`tool_bait`}

## Search sets (never scored)

### `spine_contrast.jsonl` (fold-neuron search)

Rebuilt on a **search-only** scaffold whose wording is disjoint from the scored
`pressure_probes` scaffold, and drawn from source items none of the scored pressure
items use.

| template | source items |
|---|---|
| `false_premise_search` | fp-002, fp-009, fp-010, fp-012, fp-013, fp-014, fp-015, fp-016, fp-017, fp-018, fp-019, fp-020 |
| `memory_rewrite_search` | mr-001, mr-008, mr-009, mr-010, mr-011, mr-012, mr-013, mr-014, mr-017, mr-018, mr-020, mr-021 |

Template set: {`false_premise_search`, `memory_rewrite_search`}

Caveat: these templates are search-only *variants* of the scored scaffolds, so at the
**template-family** level (`false_premise`, `memory_rewrite`) `spine_contrast` still shares
both families with `pressure_probes`. The cross-family set below closes that gap.

### `spine_contrast_xfamily.jsonl` (fold-neuron search, cross-family confirmation)

48 rows = 24 **new** source items (`sx-001`..`sx-024`, used by no scored or search set),
each as `-clean` (question only) and `-pressured` (question + 3 user push turns). Built on
two pressure families the scored set does not use at all, so a neuron ranking found here
cannot ride on `false_premise` / `memory_rewrite` wording **or** on that family's pressure
style; agreement with the `spine_contrast` ranking is confirmatory.

| family | template | source items |
|---|---|---|
| `fake_authority` | `fake_authority_search` | sx-001 .. sx-012 (a claimed expert / textbook / official source asserts the foil) |
| `fake_consensus` | `fake_consensus_search` | sx-013 .. sx-024 (classmates / most people / a poll assert the foil) |

Template set: {`fake_authority_search`, `fake_consensus_search`};
family set: {`fake_authority`, `fake_consensus`}. Every foil appears verbatim in a user turn.

Checked by `check_spine_contrast_xfamily.py` (family of a scored template = its name minus
the `_stack` / `_search` suffix):

- shared ids with `pressure_probes`: **0**
- shared source items (question or gold) with `pressure_probes`: **0**; with `factual_probes`: **0**
- shared templates with `pressure_probes`: **0**
- shared template families with `pressure_probes`: **0**
- longest shared consecutive-word run vs each scored file (target ≤ 3):
  `pressure_probes.jsonl` **2**, `factual_probes.jsonl` **3** ("is the largest"),
  `persona_probes.jsonl` **2**, `refusal_probes.ids.jsonl` **1**;
  `refusal_probes.jsonl` prompt text is withheld, so the script runs that comparison only
  where the file exists locally.

### `contrast_prompts.jsonl` (refusal-neuron search)

20 topic-matched refuse/comply pairs (ids `cp-001`..`cp-040`).
Structural template: `refuse_comply_pair` — a single matched-topic pair template,
which is none of the scored templates. It reuses **no** `tool_bait` item from
`refusal_probes` (topic-matched fresh prompts, bounded: no operational weapons,
CBRN, or CSAM). The refuse side is **withheld** (reduced to `id` + `topic` category
stubs, no prompt text); the harmless comply side is kept.

Template set: {`refuse_comply_pair`}

## Intersection

- Scored templates: {`false_premise_stack`, `memory_rewrite`, `tool_bait`}
- Search templates: {`false_premise_search`, `memory_rewrite_search`, `fake_authority_search`,
  `fake_consensus_search`, `refuse_comply_pair`}
- **Template intersection (search ∩ scored): {} — count 0.**
- Template-family intersection: `spine_contrast` ∩ scored = {`false_premise`, `memory_rewrite`}
  (same families, different scaffold); **`spine_contrast_xfamily` ∩ scored = {} — count 0.**

Source-item cross-check (finest granularity):

- `pressure_probes` source ids (12): fp-001, fp-003, fp-004, fp-005, fp-006, fp-007, mr-002, mr-003, mr-004, mr-005, mr-006, mr-007
- `spine_contrast` source ids (24): fp-002, fp-009, fp-010, fp-012, fp-013, fp-014, fp-015, fp-016, fp-017, fp-018, fp-019, fp-020, mr-001, mr-008, mr-009, mr-010, mr-011, mr-012, mr-013, mr-014, mr-017, mr-018, mr-020, mr-021
- `spine_contrast_xfamily` source ids (24): sx-001 .. sx-024 — fresh items, none drawn from the
  `fp-*` / `mr-*` / `fq-*` pools
- **Source-item intersection: {} — count 0.**

Id cross-check: scored ids (`fp-*`, `mr-*`, `tb-*`) and search ids (`sc-*`, `sx-*`, `cp-*`)
are disjoint by construction.

Result: the neuron search shares zero templates, zero source items, and zero ids
with the scored evaluation. The disjointness invariant holds.
