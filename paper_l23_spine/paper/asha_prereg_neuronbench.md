# Single-neuron push vs established benches: pre-registered rule (Asha, 2026-10-30 09:37 IST)
Fixed before any run. Not changed after seeing results.

Model: our Qwen3.5-4B checkpoint. Step 0: verify L8-N1955 indexing (MLP intermediate neuron, layer 8, 0-based) matches nisten's model; if checkpoint differs, label the arm "index-transplant", not a replication.
Benches (fixed, seeded subsets, ids saved before run): MMLU 500 stratified, ARC-Challenge 300, GSM8K 100 (greedy, exact-match, optional).
Scoring: answer-letter log-prob argmax, same prompt template for all arms; batch size and torch build logged and held fixed.
Arms: stock; target clamped at 0, +2x, +5x, +10x, -5x of its mean |activation| on a fixed 200-prompt calibration set.
Control: 20 random L8 neurons, same dose rule (scale matched to each neuron's own activation).

A push counts as "improves capability" only if ALL hold:
1. Paired McNemar vs stock p<0.05 on the bench, Holm-corrected across (5 doses x benches).
2. Gain exceeds all 20 random neurons at that dose (rank 1/21, p<=0.048).
3. No other bench drops beyond its 95% paired CI (no trade-off).
4. Perplexity on a fixed 200-passage text set rises <5% vs stock.
Report effect in accuracy points with 95% paired bootstrap CI for every arm, nulls included.
Expected prior: flat or worse. A controlled null is a reportable result.

## Amendment A1 — 2026-10-30 09:50 IST (frozen before any Qwen bench run)
- Dose = additive: dose × target's own mean|act| (neurons) or mean|residual·dir| (directions); dose 0 = ablation (secondary analysis). mult_equiv_abs logged, descriptive only.
- Targets in confirmatory family: L8-N1955 (transplant, not replication), Lyra pair (L14 n8949 +1 / L13 n7956 −1), concept dir at L14 (primary; L8/L20 descriptive unless own 20 controls).
- Matched controls: 20 random L8 neurons (seed 4242) for N1955; 20 random L14/L13 pairs, same signs (seed 4343) for Lyra; 20 random unit dirs at L14 with own L14 scale for concept dir. Pass requires beating all 20 own controls.
- Screen: data/mmlu_screen200.jsonl (sha256 5d33cf30…f7a4), disjoint from confirmatory 500/calib/ARC/wiki by id and text. Advance rule: arm accuracy outside min–max of its matched controls at that dose. Matched controls of advancing arms also run on confirm set.
- Holm: over advancing (target × non-zero dose × bench) cells; count logged. Screen results never pooled with confirm.

## Amendment A2 — 2026-10-30 09:51 IST (ordering; frozen before any Qwen bench run)
- Screen targets first (N1955, Lyra pair, L14 concept dir) at all doses.
- "Moved off stock" = screen McNemar vs stock, unadjusted p < 0.20 (deliberately lenient gate). Matched controls run only at target×dose cells that move.
- No cell moves -> report null with stock + target screen numbers; controls skipped, stated as such. A screen null is not evidence of no effect at confirm scale.
