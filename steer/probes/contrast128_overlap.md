# contrast128 vs tool_bait overlap check

128 refuse/comply pairs (`cf-001`..`cf-128`, 8 template families x 16 fillers) used only to fit the
refusal direction. Refuse-side text lives in gitignored `steer/local/contrast128.jsonl`; this tree keeps
ids, topics and the comply side (`contrast128.ids.jsonl`). Checked against the 24 `tb-*` items read at
runtime from commit 5bf8f29 (words lower-cased, punctuation stripped).

- ids shared with tb-*: 0
- texts checked: 256 (128 refuse + 128 comply) x 24 tb items
- longest shared word run over all text pairs: 3 words
- texts sharing a run of 4+ words with any tb item: 0
- histogram of per-text longest shared run: {0: 0, 1: 130, 2: 78, 3: 48}
- template-family phrases found in tb text (16 checked): none
- filler topic phrases found in tb text (128 checked): none
- families: fake_comments, fake_excuse, fake_reviews, guilt_trip, homework_cheat, insult, lie_to, sneak_in

Result: PASS.
