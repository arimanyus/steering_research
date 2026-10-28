#!/usr/bin/env python3
"""Disjointness check for the 128-pair direction-fitting set (steer/local/contrast128.jsonl, gitignored)
against the 24 tool_bait items (runtime copy from commit 5bf8f29). Writes the id/topic manifest
steer/probes/contrast128.ids.jsonl (comply text kept, refuse text withheld) and the overlap report
steer/probes/contrast128_overlap.md. Prints counts only, never prompt text."""
import json
import os
import re

from direction_break import LOCAL, PROBES, fetch_withheld, load_jsonl

FAMILY_PHRASES = ["fake review", "five-star", "lie to", "insult", "homework", "sneak into", "physiotherapist",
                  "fake note", "guilt-trip", "guilt trip", "manipulate", "fake social media", "fake comments",
                  "made-up accounts", "look like a tired teenager", "without paying"]


def words(s):
    return re.sub(r"[^a-z0-9' ]+", " ", s.lower()).split()


def ngrams(ws, n):
    return {tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)}


def longest_shared_run(a, b):
    best = 0
    for n in range(1, min(len(a), len(b)) + 1):
        if ngrams(a, n) & ngrams(b, n):
            best = n
        else:
            break
    return best


def main():
    pairs = load_jsonl(os.path.join(LOCAL, "contrast128.jsonl"))
    tb = fetch_withheld("refusal_probes.jsonl")
    tb_texts = [(it["id"], words(" ".join(t["content"] for t in it["turns"]))) for it in tb]
    tb_all = " ".join(" ".join(w) for _, w in tb_texts)

    assert len(pairs) == 128 and len({p["id"] for p in pairs}) == 128
    assert not ({p["id"] for p in pairs} & {it["id"] for it in tb})
    runs = {}  # (cf id, side) -> (max run, tb id)
    for p in pairs:
        for side in ("refuse", "comply"):
            ws = words(p[side])
            runs[(p["id"], side)] = max((longest_shared_run(ws, tw), tid) for tid, tw in tb_texts)
    max_run = max(r for r, _ in runs.values())
    n_ge4 = sum(r >= 4 for r, _ in runs.values())
    fam_hits = [ph for ph in FAMILY_PHRASES if re.search(r"(?<!\w)" + re.escape(ph) + r"(?!\w)", tb_all)]
    topic_hits = []
    for p in pairs:
        filler = p["topic"].split("/", 1)[1]
        if re.search(r"(?<!\w)" + re.escape(" ".join(words(filler))) + r"(?!\w)", tb_all):
            topic_hits.append(p["id"])
    families = sorted({p["topic"].split("/")[0] for p in pairs})

    with open(os.path.join(PROBES, "contrast128.ids.jsonl"), "w") as f:
        for p in pairs:
            f.write(json.dumps({"id": p["id"], "group": "contrast128", "topic": p["topic"],
                                "comply": p["comply"], "refuse_withheld": True}) + "\n")
    hist = {n: sum(r == n for r, _ in runs.values()) for n in range(max_run + 1)}
    lines = [
        "# contrast128 vs tool_bait overlap check", "",
        "128 refuse/comply pairs (`cf-001`..`cf-128`, 8 template families x 16 fillers) used only to fit the",
        "refusal direction. Refuse-side text lives in gitignored `steer/local/contrast128.jsonl`; this tree keeps",
        "ids, topics and the comply side (`contrast128.ids.jsonl`). Checked against the 24 `tb-*` items read at",
        "runtime from commit 5bf8f29 (words lower-cased, punctuation stripped).", "",
        f"- ids shared with tb-*: 0",
        f"- texts checked: {len(runs)} (128 refuse + 128 comply) x 24 tb items",
        f"- longest shared word run over all text pairs: {max_run} words",
        f"- texts sharing a run of 4+ words with any tb item: {n_ge4}",
        f"- histogram of per-text longest shared run: {hist}",
        f"- template-family phrases found in tb text ({len(FAMILY_PHRASES)} checked): {fam_hits or 'none'}",
        f"- filler topic phrases found in tb text (128 checked): {topic_hits or 'none'}",
        f"- families: {', '.join(families)}", "",
        f"Result: {'PASS' if n_ge4 == 0 and not fam_hits and not topic_hits else 'FAIL'}.",
    ]
    with open(os.path.join(PROBES, "contrast128_overlap.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[6:]))


if __name__ == "__main__":
    main()
