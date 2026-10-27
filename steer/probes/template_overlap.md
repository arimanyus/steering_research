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

| template | count |
|---|---|
| `tool_bait` | 24 |

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

### `contrast_prompts.jsonl` (refusal-neuron search)

20 topic-matched refuse/comply pairs (ids `cp-001`..`cp-040`).
Structural template: `refuse_comply_pair` — a single matched-topic pair template,
which is none of the scored templates. It reuses **no** `tool_bait` item from
`refusal_probes` (topic-matched fresh prompts, bounded: no operational weapons,
CBRN, or CSAM).

Template set: {`refuse_comply_pair`}

## Intersection

- Scored templates: {`false_premise_stack`, `memory_rewrite`, `tool_bait`}
- Search templates: {`false_premise_search`, `memory_rewrite_search`, `refuse_comply_pair`}
- **Template intersection (search ∩ scored): {} — count 0.**

Source-item cross-check (finest granularity):

- `pressure_probes` source ids (12): fp-001, fp-003, fp-004, fp-005, fp-006, fp-007, mr-002, mr-003, mr-004, mr-005, mr-006, mr-007
- `spine_contrast` source ids (24): fp-002, fp-009, fp-010, fp-012, fp-013, fp-014, fp-015, fp-016, fp-017, fp-018, fp-019, fp-020, mr-001, mr-008, mr-009, mr-010, mr-011, mr-012, mr-013, mr-014, mr-017, mr-018, mr-020, mr-021
- **Source-item intersection: {} — count 0.**

Id cross-check: scored ids (`fp-*`, `mr-*`, `tb-*`) and search ids (`sc-*`, `cp-*`)
are disjoint by construction.

Result: the neuron search shares zero templates, zero source items, and zero ids
with the scored evaluation. The disjointness invariant holds.
