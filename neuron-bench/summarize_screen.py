#!/usr/bin/env python3
"""results/screen/*.jsonl -> RESULTS_screen.md: per arm, MMLU-screen acc and mean gold-logprob delta vs stock (paired on item_id)."""
import os, json, glob, statistics
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = f"{HERE}/results/screen"

def load(aid): return {r["item_id"]: r for r in map(json.loads, open(f"{OUT}/{aid}.jsonl"))}
stock = load("stock")
arm_order = [json.loads(l)["arm_id"] for l in open(f"{HERE}/arms.jsonl")]
done = {os.path.basename(p)[:-6] for p in glob.glob(f"{OUT}/*.jsonl")}

rows = []
for aid in arm_order:
    if aid not in done: continue
    rs = load(aid); common = sorted(set(rs) & set(stock))
    n = len(rs); acc = sum(r["correct"] for r in rs.values()) / n
    sacc = sum(stock[i]["correct"] for i in common) / len(common)
    d = [rs[i]["logprobs"][rs[i]["gold"]] - stock[i]["logprobs"][stock[i]["gold"]] for i in common]
    md = statistics.fmean(d); se = statistics.stdev(d) / len(d) ** 0.5 if len(d) > 1 else float("nan")
    peak = max(r["peak_rss_mb_process"] for r in rs.values())
    rows.append((aid, n, acc, acc - sacc, md, se, peak))

missing = [a for a in arm_order if a not in done]
env = json.load(open(f"{OUT}/stock.summary.json"))["env"]
L = ["# MMLU damage screen: Qwen3.5-4B, CPU", "",
     f"Stage `screen` = `data/mmlu_screen200.jsonl` (200 MMLU test items, all 57 subjects, disjoint from calib200 and mmlu500). "
     f"Zero-shot, score = log-softmax over ' A'..' D' at the last position, bs=1, no generation. dtype {env['dtype']}, torch {env['torch']}, threads {env['threads']}.",
     "", f"Arms done here: {len(rows)}/{len(arm_order)}." + (f" Not in this table: {', '.join(missing)}" if missing else ""),
     "This table covers arms.jsonl lines 1-80 (this run); lines 81-211 are run by shard agents and land under results/screen/shard{1..4}/.", "",
     "`acc` = fraction correct on the n items. `d_acc` = acc minus stock acc on the same items. `d_gold_lp` = mean over items of (gold-letter logprob under the arm minus under stock), paired per item, with standard error. Gold logprob is within the 4-letter log-softmax.",
     "", "## Caveats", "",
     "- Doses are in units of each neuron's own mean|act| over calib200 (stock model, all token positions), NOT Lyra run.py's resid-norm/||down_proj col|| units. The lyra_pair_* arms therefore do not reproduce Lyra's dose scale and the comparison to Lyra is inexact.",
     "- The screen measures MMLU damage only (accuracy and gold logprob on 200 items); it says nothing about whether an arm produces the claimed behavioural effect.",
     "- n=200 items per arm: 1 item = 0.5 pp of accuracy; the binomial SE of an acc near 0.76 is about 3 pp.",
     "", "## Per-arm results", "",
     "| arm | n | acc | d_acc | d_gold_lp (mean +/- se) | peak RSS MB |", "|---|---|---|---|---|---|"]
for aid, n, acc, dacc, md, se, peak in rows:
    L.append(f"| {aid} | {n} | {acc:.3f} | {dacc:+.3f} | {md:+.4f} +/- {se:.4f} | {peak:.0f} |")
open(f"{HERE}/RESULTS_screen.md", "w").write("\n".join(L) + "\n")
print(f"{len(rows)} arms -> RESULTS_screen.md; missing {len(missing)}")
