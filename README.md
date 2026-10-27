# steering_research

Home for the neuron-steering work.

## `steer/`: pre-registered eval of the two-neuron "break" claim

Eval side only, ported from [spine PR #22](https://github.com/arimanyus/spine-is-all-you-need/pull/22).
Nothing here downloads or runs a model: a steering harness emits `outputs.jsonl` (schema in
[`steer/PROTOCOL.md`](steer/PROTOCOL.md) section 7) and `steer/` scores it against Kenji's probe
sets ([spine PR #21](https://github.com/arimanyus/spine-is-all-you-need/pull/21), loaded from
`steer/probes/`, not tracked here; see [`steer/probes/README.md`](steer/probes/README.md)).

| module | what |
|---|---|
| `steer/PROTOCOL.md` | frozen protocol: B1 safety, B2 spine, B3 validity, thresholds, verdicts, specificity, neuron search, template-overlap audit |
| `steer/controls.py` | seeded random neuron pairs, single-neuron ablations, normalized dose math (`dose_scale`, `steering_vector`) |
| `steer/scorer.py` | `outputs.jsonl` -> `summary.json` + `summary.md`: per-condition metrics with Wilson / bootstrap CIs, effects vs unsteered, verdicts, specificity vs random controls, dose-response, min-k |
| `steer/search.py` | Cohen's-d neuron ranking on activation dumps, top-k / random-k sets, held-out-template rankings, search-vs-scored disjointness audit |
| `steer/spine.py` | vendored text scorers: hold/hedge/fold/other labels with alias matching, marker-based refusal, first-sentence NLI refusal scorer |
| `steer/rubric.py` | 0-3 villain-persona rubric, judge prompt, parser, lexical backup (secondary endpoint) |
| `steer/human_spotcheck.py` | blind 40-reply sample and Cohen's kappa for the persona judge |
| `steer/redact.py` | stubs replies to withheld probes before anything is committed |

```bash
pip install -r requirements.txt            # numpy + pytest; CPU only, no model downloads
python -m pytest                           # tests/test_steer_*.py
python -m steer.controls --intermediate-size 9728 --out steer/random_pairs.json
python -m steer.scorer outputs.jsonl --probes steer/probes --out artifacts/steer/run1 [--nli] [--kappa K]
python -m steer.search --probes steer/probes                       # disjointness audit
python -m steer.redact artifacts/steer/run1/scored.jsonl -o artifacts/steer/run1/scored.redacted.jsonl
```

`--nli` needs the optional `torch` + `transformers` extras (`pip install -e .[nli]`) and downloads
the DeBERTa checkpoint; without it the scorer uses the harness's precomputed `refusal` field, else
the lexical marker fallback, and records which.

Withheld content: the refusal probes (`tb-*`) and the refuse side of `contrast_prompts` hold
harmful-request prompt text. They are referenced by id only, git-ignored under `steer/probes/`, and
raw `outputs.jsonl` / `scored.jsonl` are git-ignored too; commit only `steer.redact` output.
