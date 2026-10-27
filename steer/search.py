"""Neuron search (layers 13/14) and the search-vs-scored disjointness audit.

Ranking is pure numpy over activation matrices the harness dumps: for each search prompt, the
mean (over positions) MLP activation of every neuron in a layer, shape (n_prompts, intermediate_size).
Two contrasts, each giving a ranked, signed candidate list:

- safety:  `refuse` vs `comply` prompts of `contrast_prompts.jsonl`
- spine:   `clean` vs `pressured` rows of `spine_contrast.jsonl` (hold vs fold frame)

The steering sign for a neuron pushes toward the *break* class (comply / pressured):
sign = sign(mean_break - mean_keep). Neurons are ranked by |Cohen's d| across both layers jointly;
top-k sets for k in {1, 2, 4, 8} are steered against random-k controls at matched dose
(`controls.dose_scale` with per-neuron dose = dose / k, so the summed perturbation budget is
the same for every k).

Disjointness: search prompts must never be scored, and the search set must share no *template*
with the scored set for the search result to count as a claim (otherwise: exploratory).
"""

import json
import random
import re

import numpy as np

from steer import load_jsonl, load_probe_set
from steer.controls import POSTED, _as_array

LAYERS = (13, 14)
KS = (1, 2, 4, 8)
SEARCH_FILES = ("contrast_prompts", "spine_contrast")
SCORED_FILES = ("pressure_probes", "refusal_probes", "persona_probes", "factual_probes")
_PREFIX_TEMPLATE = {"fp": "false_premise_stack", "mr": "memory_rewrite", "tb": "tool_bait"}


# ---- ranking -------------------------------------------------------------------------------

def cohens_d(a, b):
    """Per-column standardized mean difference (a - b) / pooled sd; 0 where the pooled sd is 0."""
    a, b = _as_array(a), _as_array(b)
    var = ((len(a) - 1) * a.var(0, ddof=1) + (len(b) - 1) * b.var(0, ddof=1)) / max(len(a) + len(b) - 2, 1)
    sd = np.sqrt(var)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(sd > 0, (a.mean(0) - b.mean(0)) / sd, 0.0)
    return d


def rank_neurons(acts_break, acts_keep, exclude=POSTED):
    """`acts_*` map layer -> (n_prompts, intermediate_size). Returns [{layer, neuron, sign, d}] over all
    layers, |d| descending, with the posted neurons (`exclude`: layer -> neuron) dropped so the
    search cannot simply rediscover the claim."""
    rows = []
    for layer in sorted(acts_break):
        d = cohens_d(acts_break[layer], acts_keep[layer])
        for n in np.argsort(-np.abs(d), kind="stable"):
            if exclude.get(layer) == int(n) or d[n] == 0:
                continue
            rows.append({"layer": layer, "neuron": int(n), "sign": int(np.sign(d[n])), "d": float(d[n])})
    return sorted(rows, key=lambda r: -abs(r["d"]))


def top_k_sets(ranking, target, ks=KS):
    """[{pair_id: topk_<target>_k<k>, k, neurons}] with the top-k of `ranking`."""
    return [{"pair_id": f"topk_{target}_k{k}", "k": k,
             "neurons": [{k2: r[k2] for k2 in ("layer", "neuron", "sign")} for r in ranking[:k]]}
            for k in ks if k <= len(ranking)]


def random_k_sets(intermediate_size, ranking, ks=KS, n=10, seed=0, layers=LAYERS, exclude=POSTED, avoid_top=64):
    """`n` random-k controls per k: k neurons drawn uniformly from the union of `layers`, random sign,
    never a posted neuron and never one of the `avoid_top` best-ranked (so controls are not
    accidentally the signal)."""
    rng = random.Random(seed)
    banned = {(r["layer"], r["neuron"]) for r in ranking[:avoid_top]} | set(exclude.items())
    pool = [(l, i) for l in layers for i in range(intermediate_size) if (l, i) not in banned]
    out = []
    for k in ks:
        for i in range(n):
            picks = rng.sample(pool, k)
            out.append({"pair_id": f"randk_k{k}_{i:02d}", "k": k,
                        "neurons": [{"layer": l, "neuron": j, "sign": rng.choice((-1, 1))} for l, j in picks]})
    return out


def held_out_rankings(acts_break, acts_keep, templates, exclude=POSTED):
    """Robustness: rank with every template except T, for each template T in `templates` (one label per
    prompt row, aligned with the activation rows). Returns {T: ranking}. Steering the held-out top-k
    on scored probes of template T (`pair_id` = heldout_<T>_k<k>) checks that the search did not
    just learn a template."""
    templates = np.asarray(templates)
    out = {}
    for t in sorted(set(templates.tolist())):
        keep = templates != t
        if keep.sum() < 2:
            continue
        out[t] = rank_neurons({l: _as_array(a)[keep] for l, a in acts_break.items()},
                              {l: _as_array(a)[keep] for l, a in acts_keep.items()}, exclude)
    return out


def top_k_overlap(ranking_a, ranking_b, k):
    """Jaccard overlap of the top-k neuron sets (layer, neuron) of two rankings."""
    a = {(r["layer"], r["neuron"]) for r in ranking_a[:k]}
    b = {(r["layer"], r["neuron"]) for r in ranking_b[:k]}
    return len(a & b) / len(a | b) if a | b else 0.0


# ---- disjointness --------------------------------------------------------------------------

def item_template(item):
    """The pressure/refusal template of a probe: its `template`, else inferred from `source_id` / `id`
    prefix (fp -> false_premise_stack, mr -> memory_rewrite, tb -> tool_bait), else for single-turn
    contrast prompts `single_turn_<side>`."""
    if item.get("template"):
        return item["template"]
    for key in ("source_id", "id"):
        m = re.match(r"([a-z]+)-", str(item.get(key, "")))
        if m and m.group(1) in _PREFIX_TEMPLATE:
            return _PREFIX_TEMPLATE[m.group(1)]
    if item.get("side") in ("refuse", "comply"):
        return f"single_turn_{item['side']}"
    return None


def _norm(text):
    return re.sub(r"\W+", " ", text.lower()).strip()


def first_turn(item):
    if "turns" in item:
        return item["turns"][0]["content"]
    return item.get("prompt", "")


def user_text(item):
    """All user turns of a probe (or its single prompt), joined."""
    if "turns" in item:
        return " ".join(t["content"] for t in item["turns"] if t["role"] == "user")
    return item.get("prompt", "")


def longest_shared_ngram(a, b):
    """Length in words of the longest common contiguous word sequence of token lists `a` and `b`."""
    best, prev = 0, [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            if x == y:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def lexical_overlap(search_items, scored_items):
    """Worst-case wording overlap over all (search, scored) pairs, on the normalized user text: max
    unigram Jaccard and the longest shared word n-gram (with the pair that attains it). Templates
    can be renamed; shared wording cannot, so this is reported next to the template audit.
    Withheld items (id stubs / manifest rows, no text) cannot be checked; `items_with_text` says
    how many on each side were."""
    search = [(s["id"], _norm(user_text(s)).split()) for s in search_items]
    scored = [(p["id"], _norm(user_text(p)).split()) for p in scored_items]
    search, scored = [x for x in search if x[1]], [x for x in scored if x[1]]
    out = {"items_with_text": [len(search), len(scored)], "max_unigram_jaccard": 0.0,
           "max_shared_ngram_words": 0, "max_shared_ngram_pair": None}
    for sid, st in search:
        for pid, pt in scored:
            out["max_unigram_jaccard"] = max(out["max_unigram_jaccard"], len(set(st) & set(pt)) / len(set(st) | set(pt)))
            n = longest_shared_ngram(st, pt)
            if n > out["max_shared_ngram_words"]:
                out["max_shared_ngram_words"], out["max_shared_ngram_pair"] = n, [sid, pid]
    return out


def overlap_report(search_items, scored_items):
    """Overlap between a search set and a scored set at four levels: ids (and source ids), templates,
    normalized first-turn text (plus shared gold/foil where present), and wording (`lexical`).
    `claim_status` is "confirmatory" only when ids, templates and first turns are all disjoint;
    otherwise neuron-search results on this pair are exploratory."""
    def ids(items):
        return {i["id"] for i in items} | {i["source_id"] for i in items if "source_id" in i}
    t_search = {item_template(i) for i in search_items} - {None}
    t_scored = {item_template(i) for i in scored_items} - {None}
    shared_ids = sorted(ids(search_items) & ids(scored_items))
    shared_templates = sorted(t_search & t_scored)
    shared_text = {_norm(first_turn(i)) for i in search_items} & {_norm(first_turn(i)) for i in scored_items} - {""}
    gold = {i.get("gold") for i in search_items} & {i.get("gold") for i in scored_items} - {None}
    foil = {i.get("foil") for i in search_items} & {i.get("foil") for i in scored_items} - {None}
    return {"n_search": len(search_items), "n_scored": len(scored_items),
            "shared_ids": shared_ids,
            "search_templates": sorted(t_search), "scored_templates": sorted(t_scored),
            "shared_templates": shared_templates,
            "template_overlap": len(shared_templates) / len(t_scored) if t_scored else 0.0,
            "shared_first_turns": len(shared_text), "shared_gold": sorted(gold), "shared_foil": sorted(foil),
            "lexical": lexical_overlap(search_items, scored_items),
            "claim_status": "confirmatory" if not (shared_ids or shared_templates or shared_text) else "exploratory"}


def disjointness_audit(probes_dir):
    """The two search/scored pairings the protocol names, plus a global id check. A withheld set may
    be present only as its `.ids.jsonl` manifest; ids and templates are still audited, wording is not."""
    p = {name: load_probe_set(probes_dir, name) for name in SEARCH_FILES + SCORED_FILES}
    search_ids = {i["id"] for n in SEARCH_FILES for i in p[n]}
    scored_ids = {i["id"] for n in SCORED_FILES for i in p[n]}
    return {"spine_contrast_vs_pressure_probes": overlap_report(p["spine_contrast"], p["pressure_probes"]),
            "contrast_prompts_vs_refusal_probes": overlap_report(p["contrast_prompts"], p["refusal_probes"]),
            "any_search_id_scored": sorted(search_ids & scored_ids)}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="search-vs-scored disjointness audit")
    ap.add_argument("--probes", default="steer/probes")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    report = disjointness_audit(a.probes)
    text = json.dumps(report, indent=1)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text)
    print(text)


if __name__ == "__main__":
    main()
