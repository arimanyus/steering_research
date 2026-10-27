# Probe sets (not tracked here)

Kenji's probe files live on spine PR #21
([branch `kenji/steering-probes-08e5`](https://github.com/arimanyus/spine-is-all-you-need/pull/21),
`steer/probes/`). `steer/scorer.py` and `steer/search.py` read them from this directory
(`--probes steer/probes`); nothing in this repository copies their text. Ids used by the protocol:

| file | n | ids | role | tracked here? |
|---|---|---|---|---|
| `pressure_probes.jsonl` | 12 | `fp-001, fp-003..fp-007, mr-002..mr-007` | scored, B2 | no |
| `refusal_probes.jsonl` | 24 | `tb-001`..`tb-024` | scored, B1 | **withheld**, git-ignored |
| `persona_probes.jsonl` | 30 | `pp-001`..`pp-030` | scored, B3 ppl + persona | no |
| `factual_probes.jsonl` | 30 | `fq-001`..`fq-030` | scored, B3 accuracy | no |
| `contrast_prompts.jsonl` | 40 | `cp-001`..`cp-040` (odd = `refuse`, even = `comply`) | search, safety | **withheld** (refuse side), git-ignored |
| `spine_contrast.jsonl` | 48 | `sc-001`..`sc-024` x `-clean` / `-pressured` | search, spine | no |

The two withheld files hold harmful-request (`tool_bait`-level) prompt text and must never land on a
public branch; `steer/.gitignore` ignores them. To run against the real files, fetch them into this
directory from the PR #21 branch:

```bash
git -C /path/to/spine-is-all-you-need fetch origin kenji/steering-probes-08e5
for f in pressure_probes refusal_probes persona_probes factual_probes contrast_prompts spine_contrast; do
  git -C /path/to/spine-is-all-you-need show origin/kenji/steering-probes-08e5:steer/probes/$f.jsonl > steer/probes/$f.jsonl
done
```

`tests/test_steer_search.py::test_kenji_probes_disjointness_invariant` runs only when
`spine_contrast.jsonl` is present and is skipped otherwise.
