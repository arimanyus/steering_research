#!/usr/bin/env python3
"""Disjointness / integrity check for the search set spine_contrast_xfamily.jsonl.

Run from anywhere:  python3 steer/probes/check_spine_contrast_xfamily.py

Checks (exit 1 on any failure):
  - 48 rows = 24 source items x {clean, pressured}; 12 fake_authority + 12 fake_consensus
  - required fields present, every turn is a user turn (no model replies stored)
  - clean row = question only; pressured row starts with the same question
  - every foil appears verbatim in a user turn of its pressured row
  - vs pressure_probes.jsonl: zero shared ids, source items, templates, template families
  - vs factual_probes.jsonl: zero shared source items
  - longest shared consecutive-word run vs every scored file is <= 3
  - no tool_bait template and no withheld rows
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEW = HERE / "spine_contrast_xfamily.jsonl"
PRESSURE = HERE / "pressure_probes.jsonl"
FACTUAL = HERE / "factual_probes.jsonl"
SCORED = ["pressure_probes.jsonl", "factual_probes.jsonl", "persona_probes.jsonl",
          "refusal_probes.ids.jsonl", "refusal_probes.jsonl"]  # last one is local-only
MAX_RUN = 3
REQUIRED = ["id", "group", "family", "template", "turns", "gold", "gold_logprob_target", "foil"]

failures = []


def check(cond, msg):
    if not cond:
        failures.append(msg)


def load(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def family_of(row):
    """Template family: explicit `family` field, else the template name minus its variant suffix."""
    if "family" in row:
        return row["family"]
    return re.sub(r"_(stack|search)$", "", row["template"])


def norm(s):
    return " ".join(re.findall(r"[a-z0-9]+", s.lower()))


def question(row):
    return norm(row["turns"][0]["content"] if "turns" in row else row["prompt"])


def strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from strings(v)


def ngrams(rows, n):
    out = set()
    for s in strings(rows):
        toks = norm(s).split()
        out.update(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))
    return out


def longest_shared_run(a, b):
    n, example = 0, ()
    while True:
        shared = ngrams(a, n + 1) & ngrams(b, n + 1)
        if not shared:
            return n, example
        n, example = n + 1, min(shared)


new = load(NEW)
pressure = load(PRESSURE)
factual = load(FACTUAL)

# --- structure -----------------------------------------------------------------
check(len(new) == 48, f"expected 48 rows, got {len(new)}")
check(len({r["id"] for r in new}) == len(new), "duplicate ids")
for r in new:
    for k in REQUIRED:
        check(k in r and r[k] not in ("", None, []), f"{r.get('id')}: missing field {k}")
    check(all(t["role"] == "user" for t in r["turns"]), f"{r['id']}: non-user turn (no model replies allowed)")
    check(r["template"] != "tool_bait" and not r.get("withheld"), f"{r['id']}: tool_bait/withheld row")

by_source = {}
for r in new:
    by_source.setdefault(r["source_id"], {})[r["side"]] = r
check(len(by_source) == 24, f"expected 24 source items, got {len(by_source)}")
for sid, sides in by_source.items():
    check(set(sides) == {"clean", "pressured"}, f"{sid}: needs exactly one clean and one pressured row")
    if set(sides) != {"clean", "pressured"}:
        continue
    clean, pressured = sides["clean"], sides["pressured"]
    check(len(clean["turns"]) == 1, f"{sid}: clean row must be the question only")
    check(len(pressured["turns"]) >= 2, f"{sid}: pressured row needs push turns after the question")
    check(clean["turns"][0] == pressured["turns"][0], f"{sid}: clean/pressured question mismatch")
    check(any(pressured["foil"] in t["content"] for t in pressured["turns"][1:]),
          f"{sid}: foil {pressured['foil']!r} not verbatim in a user push turn")
    check(clean["family"] == pressured["family"] and clean["gold"] == pressured["gold"], f"{sid}: clean/pressured metadata mismatch")

fam_counts = {}
for sid, sides in by_source.items():
    fam = next(iter(sides.values()))["family"]
    fam_counts[fam] = fam_counts.get(fam, 0) + 1
check(fam_counts == {"fake_authority": 12, "fake_consensus": 12}, f"family counts {fam_counts}")

# --- disjointness vs pressure_probes -----------------------------------------
new_ids = {r["id"] for r in new} | set(by_source)
pressure_ids = {r["id"] for r in pressure}
shared_ids = new_ids & pressure_ids

new_items = {(question(r), norm(r["gold"])) for r in new}
new_qs = {q for q, _ in new_items}
new_golds = {g for _, g in new_items}
pressure_qs = {question(r) for r in pressure}
pressure_golds = {norm(r["gold"]) for r in pressure}
shared_items_pressure = (new_qs & pressure_qs) | (new_golds & pressure_golds)

new_templates = {r["template"] for r in new}
pressure_templates = {r["template"] for r in pressure}
shared_templates = new_templates & pressure_templates

new_families = {family_of(r) for r in new}
pressure_families = {family_of(r) for r in pressure}
shared_families = new_families & pressure_families

factual_qs = {question(r) for r in factual}
factual_golds = {norm(r["gold"]) for r in factual}
shared_items_factual = (new_qs & factual_qs) | (new_golds & factual_golds)

check(not shared_ids, f"shared ids with pressure_probes: {sorted(shared_ids)}")
check(not shared_items_pressure, f"shared source items with pressure_probes: {sorted(shared_items_pressure)}")
check(not shared_templates, f"shared templates with pressure_probes: {sorted(shared_templates)}")
check(not shared_families, f"shared template families with pressure_probes: {sorted(shared_families)}")
check(not shared_items_factual, f"shared source items with factual_probes: {sorted(shared_items_factual)}")

# --- lexical leakage vs every scored file -----------------------------------
runs = {}
for name in SCORED:
    path = HERE / name
    if not path.exists():
        runs[name] = None
        continue
    runs[name] = longest_shared_run(new, load(path))
    check(runs[name][0] <= MAX_RUN, f"{name}: shared run of {runs[name][0]} words > {MAX_RUN}: {' '.join(runs[name][1])}")

# --- report ------------------------------------------------------------------
print(f"rows: {len(new)}  source items: {len(by_source)}  families: {fam_counts}")
print(f"new templates: {sorted(new_templates)}  families: {sorted(new_families)}")
print(f"pressure_probes templates: {sorted(pressure_templates)}  families: {sorted(pressure_families)}")
print(f"shared with pressure_probes -> ids: {len(shared_ids)}  source items: {len(shared_items_pressure)}  "
      f"templates: {len(shared_templates)}  template families: {len(shared_families)}")
print(f"shared source items with factual_probes: {len(shared_items_factual)}")
print(f"longest shared consecutive-word run (target <= {MAX_RUN}):")
for name, run in runs.items():
    if run is None:
        print(f"  {name}: not present locally (withheld), skipped")
    else:
        print(f"  {name}: {run[0]}  e.g. {' '.join(run[1])!r}")

if failures:
    print("\nFAIL")
    for f in failures:
        print(" -", f)
    sys.exit(1)
print("\nOK")
