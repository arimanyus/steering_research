# neuron-bench harness (Maya). Built 2026-10-30; NO real-model run has happened.
Python: /workspace/tbL_venv/bin/python (torch 2.14.1, transformers 5.21.0). Data prep: ./dprep_venv/bin/python (pyarrow only there).
- prep_data.py  -> data/{mmlu500,arc300,wiki200,calib200}.jsonl, ids_*.json, MANIFEST.json (sha256, verified at every bench.py start). Seed 0.
  Screen set: data/mmlu_screen200.jsonl (MMLU test, 200 stratified by subject over the pool excluding mmlu500; seed 1; all 57 subjects). `bench.py --stage screen` uses it only; `--stage confirm` uses mmlu500+arc300+wiki200. Results in results/<stage>/. Disjointness of calib/screen/confirm is asserted at every start.
  MMLU: cais/mmlu all/test, 500 stratified by subject (largest remainder), ordered so any prefix is ~stratified (first 200 cover all 57 subjects).
  ARC-C: allenai/ai2_arc test, 4-choice items only (1165/1172), 300 random. Wikitext-2-raw test: 200 paragraphs >=500 chars, truncated to 256 tokens. Calibration: 200 prompts from MMLU *dev* (disjoint from eval).
- bench.py: arms from arms.jsonl (Kenji schema) + `--dir-pt/--alphas/--dir-random` direction arms. Zero-shot prompt, score = log-softmax of " A".." D" at last position, bs=1, no generation.
  Resumable per (arm,bench,item); config_hash mismatch aborts (use --overwrite). Staged screen: `--items 200` (MMLU only) or `--items mmlu=200,arc=0,wiki=50`; a later bigger --items only adds missing items.
  `--plan` prints arms + pass count, loads nothing. `--calibrate-only` runs the 200-prompt stock pass (caches mean|act| for every neuron of each needed layer + direction scales).
  Guard: refuses to load if another python proc has >2GB RSS or MemAvailable<10.5GB.
- fit_concept_dirs.py: diff-of-means from concept_prompts.jsonl (WRITE-ONLY, not run). (Leila also wrote fit_concept_dir.py; either .pt works: use --dir-key 23 for mine, --dir-key dir for hers.)
- tests/run_dry.sh, tests/test_hooks.py: tiny-gpt2 dry run + hook unit tests (passed).
Direction convention: plain tensors need --dir-layer (hs: 23 = output of layers[22]); Leila-format files (concept_dir_L{z}.pt, concept_rand_dirs_L{z}.pt) carry layer_zero_idx = output of layers[z] and their own scale(s), used as-is (no recompute, no calibration pass): `--dir-pt concept_dir_L14.pt --dir-rand-pt concept_rand_dirs_L14.pt --alphas 1,2,5`.
Suggested first real steps once the box is free: `bench.py --calibrate-only`, then `--arm-ids stock --items 200` to measure tok/s before queueing anything.
