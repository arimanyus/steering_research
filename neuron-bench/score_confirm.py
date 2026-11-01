#!/usr/bin/env python3
"""Confirm-stage scoring per pre-registration amendment A3 (paper_l23_spine/paper/asha_prereg_neuronbench.md). stdlib + numpy only.
Inputs: results/confirm/<arm>.jsonl (stage 1 + stage-2 controls, all paired with the one stage-1 stock run),
        results/confirm_rerun/confirm/stock.jsonl (second stock run, descriptive only), results/confirm/scales_used.json.
Outputs: confirm_scores.json, RESULTS_confirm.md. Screen results are never pooled (the screen is read only for the calibration-scale comparison)."""
import os, json, math, glob, fnmatch
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = f"{HERE}/results/confirm"; RERUN = f"{HERE}/results/confirm_rerun/confirm/stock.jsonl"; SCREEN = f"{HERE}/results/screen"
FULL = {"mmlu": 500, "arc": 300, "wiki": 200}
DOSES = [-5, 0, 2, 5, 10]; PRIMARY = [-5, 2, 5, 10]
TARGETS = {"L8-N1955": ("nisten_L8N1955_d{}", "rand_L8_k*_d{}"), "Lyra pair": ("lyra_pair_d{}", "randpair_*_d{}")}
NBOOT, SEED, GATE_P, ALPHA, PPL_MAX = 10000, 0, 0.20, 0.05, 1.05
ARM_IDS = [json.loads(l)["arm_id"] for l in open(f"{HERE}/arms.jsonl")]
NOTES = ["Environment: results/confirm/ENV.txt (pip freeze identical to results/screen/ENV_orig.txt; torch 2.15.0+cpu, transformers 5.22.1; model snapshot 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a). "
         "4 vCPU Intel Xeon (AMX/AVX512-bf16), 15.6 GB RAM visible to the guest plus a VM memory balloon.",
         "bench.py's memory guard aborted the first stage-1 launch on MemAvailable (7.1 GB < 10.5 GB; the balloon hides free memory) with no other python process above 2 GB RSS, so stage 1 ran with --no-guard. The stock rerun and the stage-2 runs passed the guard (the balloon had deflated) and ran without --no-guard.",
         "Stage-1 process 1 (plain bench.py) ran stock, nisten_L8N1955_d-5 and the first 845 rows of nisten_L8N1955_d0. It was stopped (SIGTERM, 17:30 IST) because the balloon re-inflated and evicted the mmapped weights (~5 s/item instead of ~0.5 s). "
         "Process 2 resumed (bench.py's per-item resume; no duplicate rows, one config_hash per arm) through `run_anon.py`, which runs bench.py unchanged after copying the weights bit-exactly into anonymous memory. "
         "All later stage-1 arms and all stage-2 controls ran through run_anon.py. The stage-1 stock run and the stock rerun both ran as plain bench.py.",
         "Because process 2 re-listed stock and nisten_L8N1955_d-5 with 0 new rows, their summary.json files were restored to the process-1 versions. nisten_L8N1955_d0's summary covers only the 155 resumed rows.",
         "The VM was suspended once for about 15 minutes during nisten_L8N1955_d2 (kernel 'virtual machine fork' events); the guest clock did not advance, so its wall s in the table above (657 s) excludes the pause (wall-clock duration about 26 min). Results are unaffected.",
         "Calibration (calib200, layers 8, 13, 14) was computed once in stage-1 process 1 and cached in results/calib/; process 2 and the stage-2 controls reuse that cache."]

def load(fp):
    """-> {bench: {item_id: row}} or None if the file is missing"""
    if not os.path.exists(fp): return None
    out = {}
    for l in open(fp):
        r = json.loads(l); out.setdefault(r["bench"], {})[r["item_id"]] = r
    return out
def complete(arm, b): return arm is not None and len(arm.get(b, {})) == FULL[b]

_IDX = {}
def boot_idx(n):
    if n not in _IDX: _IDX[n] = np.random.default_rng(SEED).integers(0, n, size=(NBOOT, n))
    return _IDX[n]
def ci(x): return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

def mcnemar_p(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)
def wilcoxon_p(d):
    """two-sided signed-rank test, zeros dropped, average ranks for ties, normal approximation with tie correction, no continuity correction"""
    d = np.asarray(d, float); d = d[d != 0]; n = len(d)
    if n == 0: return 1.0
    a = np.abs(d); order = np.argsort(a, kind="stable"); ranks = np.empty(n); sa = a[order]; i = 0
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]: j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    _, t = np.unique(a, return_counts=True)
    var = n * (n + 1) * (2 * n + 1) / 24 - float(np.sum(t ** 3 - t)) / 48
    z = (float(ranks[d > 0].sum()) - n * (n + 1) / 4) / math.sqrt(var)
    return math.erfc(abs(z) / math.sqrt(2))
def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [0.0] * len(ps); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i])); adj[i] = run
    return adj

def acc_cell(stock, arm, b):
    ids = sorted(stock[b], key=lambda i: stock[b][i]["order"])
    s = np.array([stock[b][i]["correct"] for i in ids], float); a = np.array([arm[b][i]["correct"] for i in ids], float)
    glp = lambda R, i: R[b][i]["logprobs"][R[b][i]["gold"]]
    dlp = np.array([glp(arm, i) - glp(stock, i) for i in ids])
    nb = int(((s == 1) & (a == 0)).sum()); nc = int(((s == 0) & (a == 1)).sum()); idx = boot_idx(len(ids))
    return {"n": len(ids), "stock_correct": int(s.sum()), "arm_correct": int(a.sum()), "stock_acc": float(s.mean()), "arm_acc": float(a.mean()),
            "b": nb, "c": nc, "net": nc - nb, "p": mcnemar_p(nb, nc), "test": "McNemar exact",
            "d_acc_pts": 100 * float(a.mean() - s.mean()), "d_acc_ci": ci(100 * (a[idx].mean(1) - s[idx].mean(1))),
            "pred_differs": sum(stock[b][i]["pred"] != arm[b][i]["pred"] for i in ids),
            "d_goldlp": float(dlp.mean()), "d_goldlp_ci": ci(dlp[idx].mean(1)), "mean_abs_d_goldlp": float(np.abs(dlp).mean())}
def wiki_cell(stock, arm):
    ids = sorted(stock["wiki"], key=lambda i: stock["wiki"][i]["order"])
    assert all(stock["wiki"][i]["n_pred_tokens"] == arm["wiki"][i]["n_pred_tokens"] for i in ids)
    sn = np.array([stock["wiki"][i]["nll_sum"] for i in ids]); an = np.array([arm["wiki"][i]["nll_sum"] for i in ids])
    nt = np.array([stock["wiki"][i]["n_pred_tokens"] for i in ids], float); idx = boot_idx(len(ids))
    sp, ap = math.exp(sn.sum() / nt.sum()), math.exp(an.sum() / nt.sum())
    d = an / nt - sn / nt
    return {"n": len(ids), "stock_ppl": sp, "arm_ppl": ap, "ratio": ap / sp, "ratio_ci": ci(np.exp((an[idx].sum(1) - sn[idx].sum(1)) / nt[idx].sum(1))),
            "p": wilcoxon_p(d), "test": "Wilcoxon signed-rank (per-passage mean NLL)", "mean_d_nll_per_tok": float(d.mean()), "mean_abs_d_nll_per_tok": float(np.abs(d).mean())}
def moved(c, b): return c["p"] < GATE_P if b != "wiki" else not (c["ratio_ci"][0] <= 1 <= c["ratio_ci"][1])
def effect(c, b): return c["d_acc_pts"] if b != "wiki" else c["ratio"]

def calib_comparison():
    su = json.load(open(f"{RES}/scales_used.json")) if os.path.exists(f"{RES}/scales_used.json") else {}
    def scale(edits, l, n): return next((e["scale_mean_abs"] for e in edits if e["layer"] == l and e["neuron"] == n), None)
    out = []
    for (l, n, a) in [(8, 1955, "nisten_L8N1955_d5"), (14, 8949, "lyra_pair_d5"), (13, 7956, "lyra_pair_d5")]:
        conf = scale(su[a]["edits"], l, n) if a in su else None; scr = set()
        for fp in glob.glob(f"{SCREEN}/**/*.jsonl", recursive=True):
            with open(fp) as f: r = json.loads(f.readline() or "null")
            if r and r.get("edit", {}).get("kind") == "neuron":
                s = scale(r["edit"]["edits"], l, n)
                if s is not None: scr.add(s)
        out.append({"layer": l, "neuron": n, "confirm": conf, "screen_values": sorted(scr),
                    "rel_diff": [(conf - s) / s for s in sorted(scr)] if conf is not None else None})
    return out

def main():
    stock = load(f"{RES}/stock.jsonl"); assert stock and all(complete(stock, b) for b in FULL), "stage-1 stock run incomplete"
    cells = []
    for tname, (tpat, cpat) in TARGETS.items():
        for d in DOSES:
            arm = load(f"{RES}/{tpat.format(d)}.jsonl")
            for b in FULL:
                c = {"target": tname, "arm": tpat.format(d), "dose": d, "bench": b, "primary": d in PRIMARY}
                if not complete(arm, b): cells.append({**c, "missing": True}); continue
                c.update(acc_cell(stock, arm, b) if b != "wiki" else wiki_cell(stock, arm)); c["moved"] = moved(c, b); cells.append(c)
    prim = [c for c in cells if c["primary"] and not c.get("missing")]
    n_primary_expected = len(TARGETS) * len(PRIMARY) * len(FULL)
    for c, h in zip(prim, holm([c["p"] for c in prim])): c["holm_p"] = h
    # ---- stage 2: matched controls at the moved cell's dose, same bench, same stock
    for c in cells:
        if not c.get("moved"): continue
        b = c["bench"]; pat = TARGETS[c["target"]][1].format(c["dose"]); ctrl_ids = [a for a in ARM_IDS if fnmatch.fnmatch(a, pat)]
        ctrl = {}
        for a in ctrl_ids:
            r = load(f"{RES}/{a}.jsonl")
            if complete(r, b): ctrl[a] = acc_cell(stock, r, b) if b != "wiki" else wiki_cell(stock, r)
        s2 = {"control_pattern": pat, "n_controls_expected": len(ctrl_ids), "n_controls": len(ctrl), "controls": ctrl}
        if len(ctrl) == len(ctrl_ids) == 20:
            t = effect(c, b); ce = [effect(v, b) for v in ctrl.values()]
            if b != "wiki":
                s2["rank_p_gain"] = (1 + sum(x >= t for x in ce)) / 21; s2["rank_p_drop"] = (1 + sum(x <= t for x in ce)) / 21
            else:
                s2["rank_p_increase"] = (1 + sum(x >= t for x in ce)) / 21
                s2["rank_p_abs_log"] = (1 + sum(abs(math.log(x)) >= abs(math.log(t)) for x in ce)) / 21
            s2["control_effects"] = {"min": min(ce), "median": float(np.median(ce)), "max": max(ce)}
        c["stage2"] = s2
    # ---- verdicts (primary cells only; Holm p is defined only there)
    get = {(c["target"], c["dose"], c["bench"]): c for c in cells}
    verdicts = {}
    for tname in TARGETS:
        imp, dmg, pending = [], [], []
        for d in PRIMARY:
            for b in FULL:
                c = get[(tname, d, b)]
                if c.get("missing"): pending.append(f"d{d}/{b}: stage-1 arm missing"); continue
                s2 = c.get("stage2", {}); ranked = "rank_p_gain" in s2 or "rank_p_increase" in s2
                sig = c["holm_p"] < ALPHA
                if c.get("moved") and sig and not ranked: pending.append(f"d{d}/{b}: stage-2 controls incomplete ({s2.get('n_controls', 0)}/20)")
                if b != "wiki":
                    o = get[(tname, d, "arc" if b == "mmlu" else "mmlu")]; w = get[(tname, d, "wiki")]
                    if (sig and c["d_acc_pts"] > 0 and ranked and s2["rank_p_gain"] == 1 / 21 and not o.get("missing") and o["d_acc_ci"][1] >= 0
                            and not w.get("missing") and w["ratio"] < PPL_MAX): imp.append(f"d{d}/{b}")
                    if sig and c["d_acc_pts"] < 0 and ranked and s2["rank_p_drop"] == 1 / 21: dmg.append(f"d{d}/{b}")
                elif sig and c["ratio"] > 1 and ranked and s2["rank_p_increase"] == 1 / 21: dmg.append(f"d{d}/{b}")
        v = " + ".join(x for x, y in (("improves capability", imp), ("damages capability", dmg)) if y) or "no detectable effect"
        verdicts[tname] = {"verdict": v, "improves_cells": imp, "damages_cells": dmg, "pending": pending}
    # ---- stock rerun (descriptive)
    rerun = load(RERUN); agree = None
    if rerun and all(complete(rerun, b) for b in FULL):
        agree = {b: (acc_cell(stock, rerun, b) if b != "wiki" else wiki_cell(stock, rerun)) for b in FULL}
        for b in FULL: agree[b]["moved_if_it_were_an_arm"] = moved(agree[b], b)
    out = {"prereg": "A3", "n_boot": NBOOT, "seed": SEED, "gate_p": GATE_P, "holm_family_size": len(prim), "holm_family_expected": n_primary_expected,
           "stock": {b: ({"acc": sum(r["correct"] for r in stock[b].values()) / FULL[b]} if b != "wiki" else
                         {"ppl": math.exp(sum(r["nll_sum"] for r in stock[b].values()) / sum(r["n_pred_tokens"] for r in stock[b].values()))}) for b in FULL},
           "cells": cells, "verdicts": verdicts, "stock_rerun": agree, "calibration": calib_comparison()}
    json.dump(out, open(f"{HERE}/confirm_scores.json", "w"), indent=1)
    write_md(out)
    print(f"{len(cells)} cells ({sum(1 for c in cells if c.get('missing'))} missing), moved: {[(c['arm'], c['bench']) for c in cells if c.get('moved')]}; verdicts: { {k: v['verdict'] for k, v in verdicts.items()} }")

def f3(x): return "-" if x is None else f"{x:.3g}" if x < 0.001 else f"{x:.3f}"
def write_md(o):
    S = o["stock"]; cells = o["cells"]
    L = ["# Qwen3.5-4B single-neuron pushes: confirm stage (amendment A3)", "",
         "Generated by `score_confirm.py` from `results/confirm/` (machine-readable copy: `confirm_scores.json`). Pre-registration: amendment A3 in "
         "`paper_l23_spine/paper/asha_prereg_neuronbench.md`, committed before any confirm item was scored. This run goes beyond the A2 gate (no screen cell moved) and is a pre-registered extension, not part of the original plan. "
         "Screen and confirm results are not pooled.", "",
         f"Benches: MMLU-500, ARC-Challenge-300 (answer-letter log-prob argmax, zero-shot), Wikitext-200 perplexity (256-token passages). bf16 CPU, batch size 1, 4 threads. "
         f"Every arm is paired with the one stage-1 stock run (MMLU {100 * S['mmlu']['acc']:.1f}%, ARC {100 * S['arc']['acc']:.1f}%, Wiki ppl {S['wiki']['ppl']:.3f}). "
         f"Paired bootstrap: {o['n_boot']} resamples over items/passages, seed {o['seed']}, percentile 95% CI. Holm over the {o['holm_family_size']} primary cells (expected {o['holm_family_expected']}: 2 targets x doses -5,2,5,10 x 3 benches); dose 0 (ablation) is secondary and outside the Holm family.", "",
         "b = stock right & arm wrong, c = stock wrong & arm right; net = c - b. McNemar = exact two-sided binomial on b vs c. Wiki p = two-sided Wilcoxon signed-rank on per-passage mean NLL differences (normal approximation with tie correction). "
         f"Moved (stage-2 gate) = MMLU/ARC unadjusted McNemar p < {o['gate_p']}; Wiki perplexity-ratio 95% CI excludes 1.", ""]
    mv = [c for c in cells if c.get("moved")]
    moved_txt = ", ".join(c["arm"] + " / " + c["bench"] for c in mv) or "none"
    s2_done = [c for c in mv if c["stage2"]["n_controls"] == 20]
    L += ["## Summary", "", f"- Cells that moved (stage-2 gate): {moved_txt}.",
          f"- Stage 2: {'not triggered' if not mv else f'{len(s2_done)}/{len(mv)} moved cells have all 20 controls'}.",
          "- Verdicts: " + "; ".join(f"{t}: {v['verdict']}" for t, v in o["verdicts"].items()) + ".",
          f"- Smallest Holm-adjusted p over the primary cells: {f3(min(c['holm_p'] for c in cells if 'holm_p' in c))}"
          + ("; no primary cell reaches Holm p < 0.05, so neither verdict can depend on the stage-2 ranks." if all(c["holm_p"] >= ALPHA for c in cells if "holm_p" in c) else "."), "",
          "## Stage 1: accuracy cells", "",
         "| target | dose | bench | stock acc | arm acc | b | c | net | McNemar p | Holm p | d acc pts [95% CI] | d gold logprob [95% CI] | moved |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        if c["bench"] == "wiki": continue
        if c.get("missing"): L.append(f"| {c['target']} | {c['dose']} | {c['bench']} | not run | | | | | | | | | |"); continue
        L.append(f"| {c['target']} | {c['dose']} | {c['bench']} | {100 * c['stock_acc']:.1f} | {100 * c['arm_acc']:.1f} | {c['b']} | {c['c']} | {c['net']:+d} | {f3(c['p'])} | "
                 f"{f3(c['holm_p']) if 'holm_p' in c else '(secondary)'} | {c['d_acc_pts']:+.1f} [{c['d_acc_ci'][0]:+.1f}, {c['d_acc_ci'][1]:+.1f}] | "
                 f"{c['d_goldlp']:+.4f} [{c['d_goldlp_ci'][0]:+.4f}, {c['d_goldlp_ci'][1]:+.4f}] | {'**yes**' if c['moved'] else 'no'} |")
    L += ["", "## Stage 1: Wikitext-200 perplexity cells", "",
          "| target | dose | stock ppl | arm ppl | ratio arm/stock [95% CI] | Wilcoxon p | Holm p | moved |", "|---|---|---|---|---|---|---|---|"]
    for c in cells:
        if c["bench"] != "wiki": continue
        if c.get("missing"): L.append(f"| {c['target']} | {c['dose']} | not run | | | | | |"); continue
        L.append(f"| {c['target']} | {c['dose']} | {c['stock_ppl']:.4f} | {c['arm_ppl']:.4f} | {c['ratio']:.5f} [{c['ratio_ci'][0]:.5f}, {c['ratio_ci'][1]:.5f}] | {f3(c['p'])} | "
                 f"{f3(c['holm_p']) if 'holm_p' in c else '(secondary)'} | {'**yes**' if c['moved'] else 'no'} |")
    L += ["", "## Stage 2: matched controls (moved cells only)", ""]
    if not mv: L.append("No cell moved; stage 2 is not triggered.")
    else:
        L += ["Each control is paired with the same stage-1 stock run. Add-one rank p = (1 + controls at least as extreme as the target) / 21, ties counted against the target. "
              "Accuracy effect = d acc pts; perplexity effect = ratio arm/stock.", "",
              "| target | dose | bench | target effect | controls done | control min / median / max | rank p (gain) | rank p (drop) | rank p (ppl increase) | rank p (abs log ratio) |", "|---|---|---|---|---|---|---|---|---|---|"]
        for c in mv:
            s2 = c["stage2"]; ce = s2.get("control_effects"); fmt = (lambda x: f"{x:+.1f}") if c["bench"] != "wiki" else (lambda x: f"{x:.5f}")
            L.append(f"| {c['target']} | {c['dose']} | {c['bench']} | {fmt(effect(c, c['bench']))} | {s2['n_controls']}/{s2['n_controls_expected']} | "
                     f"{' / '.join(fmt(ce[k]) for k in ('min', 'median', 'max')) if ce else 'pending'} | "
                     + " | ".join(f"{s2[k]:.3f}" if k in s2 else "-" for k in ("rank_p_gain", "rank_p_drop", "rank_p_increase", "rank_p_abs_log")) + " |")
        L += ["", "Per-control effects (moved cells):", ""]
        for c in mv:
            ctrl = c["stage2"]["controls"]
            if not ctrl: continue
            L.append(f"- {c['arm']} / {c['bench']}: " + ", ".join(f"{a} {effect(v, c['bench']):+.1f} (p {f3(v['p'])})" if c["bench"] != "wiki" else f"{a} {v['ratio']:.5f}" for a, v in ctrl.items()))
    L += ["", "## Verdicts (A3; primary doses only)", "",
          "Improves capability: gain with Holm p < 0.05, gain rank 1/21 among the 20 controls, other accuracy bench's 95% CI not entirely below 0, perplexity ratio < 1.05 at that dose. "
          "Damages capability: drop with Holm p < 0.05 and drop rank 1/21 (Wiki: perplexity increase with Holm p < 0.05 and increase rank 1/21). Otherwise no detectable effect.", "",
          "| target | verdict | cells | pending |", "|---|---|---|---|"]
    for t, v in o["verdicts"].items():
        L.append(f"| {t} | **{v['verdict']}** | {', '.join(v['improves_cells'] + v['damages_cells']) or '-'} | {'; '.join(v['pending']) or '-'} |")
    L += ["", "## Stock rerun (descriptive; run-to-run noise)", ""]
    a = o["stock_rerun"]
    if not a: L.append("Second stock run not available.")
    else:
        L += ["Second stock run (`results/confirm_rerun/confirm/stock.jsonl`, separate process, same config) scored against the stage-1 stock run exactly like an arm.", "",
              "| bench | items whose correctness differs (b+c) | preds differing | d acc pts [95% CI] | McNemar p | mean abs d gold logprob | d gold logprob [95% CI] |", "|---|---|---|---|---|---|---|"]
        for b in ("mmlu", "arc"):
            x = a[b]; L.append(f"| {b} | {x['b'] + x['c']} (b={x['b']}, c={x['c']}) | {x['pred_differs']} | {x['d_acc_pts']:+.1f} [{x['d_acc_ci'][0]:+.1f}, {x['d_acc_ci'][1]:+.1f}] | {f3(x['p'])} | {x['mean_abs_d_goldlp']:.4f} | {x['d_goldlp']:+.4f} [{x['d_goldlp_ci'][0]:+.4f}, {x['d_goldlp_ci'][1]:+.4f}] |")
        w = a["wiki"]
        L += ["", f"Wiki: stock ppl {w['stock_ppl']:.4f}, rerun ppl {w['arm_ppl']:.4f} (difference {w['arm_ppl'] - w['stock_ppl']:+.4f}); ratio {w['ratio']:.5f} [{w['ratio_ci'][0]:.5f}, {w['ratio_ci'][1]:.5f}]; "
              f"Wilcoxon p {f3(w['p'])}; mean abs per-passage d NLL/token {w['mean_abs_d_nll_per_tok']:.5f}.",
              f"Would the rerun pass the stage-2 gate if it were an arm? MMLU {'yes' if a['mmlu']['moved_if_it_were_an_arm'] else 'no'}, ARC {'yes' if a['arc']['moved_if_it_were_an_arm'] else 'no'}, Wiki {'yes' if w['moved_if_it_were_an_arm'] else 'no'}."]
    L += ["", "## Per-arm wall time (last invocation of each arm)", "", "| arm | rows | new rows | wall s | peak RSS MB |", "|---|---|---|---|---|"]
    for fp in sorted(glob.glob(f"{RES}/*.summary.json")) + sorted(glob.glob(f"{os.path.dirname(RERUN)}/*.summary.json")):
        s = json.load(open(fp)); rows = sum(s[b]["n"] for b in FULL if b in s)
        L.append(f"| {os.path.relpath(fp, os.path.dirname(RES))[:-len('.summary.json')]} | {rows} | {s['new_rows_this_run']} | {s['arm_wall_s_this_run']:.0f} | {s['peak_rss_mb_process']:.0f} |")
    L += ["", "## Run notes", ""] + [f"- {n}" for n in NOTES]
    L += ["", "## Calibration scales (mean|act| on calib200): confirm recomputation vs screen rows", "",
          "| neuron | confirm | screen value(s) | relative difference |", "|---|---|---|---|"]
    for r in o["calibration"]:
        conf = "-" if r["confirm"] is None else f"{r['confirm']:.6g}"
        L.append(f"| L{r['layer']} n{r['neuron']} | {conf} | {', '.join(f'{s:.6g}' for s in r['screen_values'])} | "
                 f"{', '.join(f'{x:+.2e}' for x in r['rel_diff']) if r['rel_diff'] else '-'} |")
    open(f"{HERE}/RESULTS_confirm.md", "w").write("\n".join(L) + "\n")

if __name__ == "__main__": main()
