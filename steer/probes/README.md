# Probe sets (not tracked on this branch)

Kenji's rebuilt probe files live on [PR #2](https://github.com/arimanyus/steering_research/pull/2)
(branch `cursor/steering-probes-5853`, `steer/probes/`, with its own `README.md` and
`template_overlap.md`). `steer/scorer.py` and `steer/search.py` read them from this directory
(`--probes steer/probes`); this branch copies none of their text. Ids used by the protocol:

| file | n | ids | template(s) | role | tracked here? |
|---|---|---|---|---|---|
| `pressure_probes.jsonl` | 12 | `fp-001, fp-003..fp-007, mr-002..mr-007` | `false_premise_stack`, `memory_rewrite` | scored, B2 | no |
| `refusal_probes.jsonl` | 24 | `tb-001`..`tb-024` | `tool_bait` | scored, B1 | **withheld**, git-ignored |
| `persona_probes.jsonl` | 30 | `pp-001`..`pp-030` | | scored, B3 ppl + persona | no |
| `factual_probes.jsonl` | 30 | `fq-001`..`fq-030` | | scored, B3 accuracy | no |
| `contrast_prompts.jsonl` | 40 | `cp-001`..`cp-040` (odd = `refuse`, even = `comply`) | `single_turn_<side>` (inferred) | search, safety | **withheld** (refuse side), git-ignored |
| `spine_contrast.jsonl` | 48 | `sc-001`..`sc-024` x `-clean` / `-pressured`; sources `fp-002, fp-009, fp-010, fp-012..fp-020, mr-001, mr-008..mr-014, mr-017, mr-018, mr-020, mr-021` | `false_premise_search`, `memory_rewrite_search` | search, spine | no |

The two withheld files hold harmful-request (`tool_bait`-level) prompt text and must never land on a
public branch; `steer/.gitignore` ignores them. To run against the real files, fetch them into this
directory from the PR #2 branch:

```bash
git fetch origin cursor/steering-probes-5853
for f in pressure_probes refusal_probes persona_probes factual_probes contrast_prompts spine_contrast; do
  git show origin/cursor/steering-probes-5853:steer/probes/$f.jsonl > steer/probes/$f.jsonl
done
python -m steer.search --probes steer/probes   # disjointness + lexical audit (PROTOCOL.md 8.2)
```

`tests/test_steer_search.py::test_kenji_probes_disjointness_invariant` runs only when
`spine_contrast.jsonl` is present and is skipped otherwise.
