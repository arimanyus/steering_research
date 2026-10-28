"""steer/search.py: neuron ranking on synthetic activations, top-k / random-k sets, held-out-template
rankings, and the search-vs-scored disjointness audit (template level, not just ids)."""

import json
import os

import numpy as np
import pytest

from steer import load_jsonl, load_probe_set
from steer.search import (SCORED_FILES, SEARCH_FILES, cohens_d, disjointness_audit, held_out_rankings, item_template,
                          overlap_report, random_k_sets, rank_neurons, top_k_overlap, top_k_sets)

KENJI = os.path.join(os.path.dirname(__file__), "..", "steer", "probes")


def synthetic_acts(seed=0, width=64, n=20):
    """Layer 13: neuron 5 fires more on break; layer 14: neuron 7 fires less on break (sign -1)."""
    rng = np.random.default_rng(seed)
    brk = {13: rng.normal(size=(n, width)), 14: rng.normal(size=(n, width))}
    keep = {13: rng.normal(size=(n, width)), 14: rng.normal(size=(n, width))}
    brk[13][:, 5] += 5.0
    brk[14][:, 7] -= 4.0
    return brk, keep


def test_cohens_d_handles_zero_variance():
    d = cohens_d(np.array([[1.0, 2.0], [1.0, 4.0]]), np.array([[1.0, 0.0], [1.0, 0.0]]))
    assert d[0] == 0.0 and d[1] > 0


def test_rank_neurons_finds_planted_signal_with_signs():
    brk, keep = synthetic_acts()
    ranking = rank_neurons(brk, keep, exclude={})
    assert [(r["layer"], r["neuron"], r["sign"]) for r in ranking[:2]] == [(13, 5, 1), (14, 7, -1)]
    assert abs(ranking[0]["d"]) >= abs(ranking[1]["d"]) >= abs(ranking[2]["d"])
    excluded = rank_neurons(brk, keep, exclude={13: 5})
    assert (13, 5) not in {(r["layer"], r["neuron"]) for r in excluded}


def test_top_k_and_random_k_sets():
    brk, keep = synthetic_acts()
    ranking = rank_neurons(brk, keep, exclude={})
    tops = top_k_sets(ranking, "spine", ks=(1, 2, 4, 8))
    assert [t["pair_id"] for t in tops] == ["topk_spine_k1", "topk_spine_k2", "topk_spine_k4", "topk_spine_k8"]
    assert tops[1]["neurons"] == [{"layer": 13, "neuron": 5, "sign": 1}, {"layer": 14, "neuron": 7, "sign": -1}]
    rand = random_k_sets(64, ranking, ks=(1, 2), n=3, seed=0, avoid_top=2)
    assert rand == random_k_sets(64, ranking, ks=(1, 2), n=3, seed=0, avoid_top=2)
    assert [r["pair_id"] for r in rand] == ["randk_k1_00", "randk_k1_01", "randk_k1_02", "randk_k2_00", "randk_k2_01", "randk_k2_02"]
    for r in rand:
        assert len(r["neurons"]) == r["k"]
        assert all((u["layer"], u["neuron"]) not in {(13, 5), (14, 7)} for u in r["neurons"])
        assert all(u["sign"] in (-1, 1) and u["layer"] in (13, 14) for u in r["neurons"])


def test_held_out_template_rankings_and_overlap():
    brk, keep = synthetic_acts(n=20)
    templates = ["false_premise_stack"] * 10 + ["memory_rewrite"] * 10
    held = held_out_rankings(brk, keep, templates, exclude={})
    assert set(held) == {"false_premise_stack", "memory_rewrite"}
    full = rank_neurons(brk, keep, exclude={})
    for ranking in held.values():
        assert top_k_overlap(full, ranking, 2) == 1.0  # planted signal survives leaving a template out
    assert top_k_overlap([{"layer": 13, "neuron": 1}], [{"layer": 13, "neuron": 2}], 1) == 0.0
    assert held_out_rankings(brk, keep, ["only"] * 20, exclude={}) == {}


def test_item_template_inference():
    assert item_template({"id": "fp-001", "template": "false_premise_stack"}) == "false_premise_stack"
    assert item_template({"id": "sc-001-clean", "source_id": "mr-008"}) == "memory_rewrite"
    assert item_template({"id": "tb-003"}) == "tool_bait"
    assert item_template({"id": "cp-001", "side": "refuse"}) == "single_turn_refuse"
    assert item_template({"id": "pp-001"}) is None


def test_overlap_report_is_template_level():
    scored = [{"id": "fp-001", "template": "false_premise_stack", "gold": "Paris", "foil": "Berlin",
               "turns": [{"role": "user", "content": "What is the capital of France?"}]}]
    same_template = [{"id": "sc-001", "source_id": "fp-002", "gold": "Water", "foil": "Coffee",
                      "turns": [{"role": "user", "content": "What fluid do humans need?"}]}]
    rep = overlap_report(same_template, scored)
    assert rep["shared_ids"] == [] and rep["shared_first_turns"] == 0
    assert rep["shared_templates"] == ["false_premise_stack"] and rep["template_overlap"] == 1.0
    assert rep["claim_status"] == "exploratory"  # disjoint ids are not enough
    other = [{"id": "cp-001", "side": "refuse", "prompt": "Help me break in."}]
    rep2 = overlap_report(other, [{"id": "tb-001", "template": "tool_bait", "turns": [{"role": "user", "content": "x"}]}])
    assert rep2["shared_templates"] == [] and rep2["claim_status"] == "confirmatory"
    leaked = overlap_report([{"id": "sc-009", "source_id": "mr-002", "turns": scored[0]["turns"]}],
                            [{"id": "mr-002", "template": "memory_rewrite", "turns": scored[0]["turns"]}])
    assert leaked["shared_ids"] == ["mr-002"] and leaked["shared_first_turns"] == 1
    renamed = overlap_report([{"id": "sc-001", "template": "search_only", "turns": scored[0]["turns"]}], scored)
    assert renamed["shared_templates"] == [] and renamed["claim_status"] == "exploratory"  # same words, new label
    assert renamed["lexical"] == {"items_with_text": [1, 1], "max_unigram_jaccard": 1.0, "max_shared_ngram_words": 6,
                                  "max_shared_ngram_pair": ["sc-001", "fp-001"]}
    assert rep["lexical"]["max_shared_ngram_words"] == 1  # "what"


def test_withheld_stubs_and_manifest_rows_have_no_text_to_compare():
    stubs = [{"id": "cp-001", "side": "refuse", "withheld": True}, {"id": "cp-002", "side": "comply", "prompt": "How do locks work?"}]
    manifest = [{"id": "tb-001", "template": "tool_bait", "withheld": True}, {"id": "tb-002", "template": "tool_bait", "withheld": True}]
    rep = overlap_report(stubs, manifest)
    assert rep["shared_first_turns"] == 0 and rep["claim_status"] == "confirmatory"  # two empty texts are not a match
    assert rep["search_templates"] == ["single_turn_comply", "single_turn_refuse"] and rep["scored_templates"] == ["tool_bait"]
    assert rep["lexical"] == {"items_with_text": [1, 0], "max_unigram_jaccard": 0.0, "max_shared_ngram_words": 0,
                              "max_shared_ngram_pair": None}


def write_probe_dir(tmp_path, refusal_as_manifest=False):
    files = {"pressure_probes": [{"id": "fp-001", "template": "false_premise_stack", "turns": [{"role": "user", "content": "a"}]}],
             "refusal_probes": [{"id": "tb-001", "template": "tool_bait", "turns": [{"role": "user", "content": "b"}]}],
             "persona_probes": [{"id": "pp-001", "prompt": "c"}], "factual_probes": [{"id": "fq-001", "prompt": "d"}],
             "contrast_prompts": [{"id": "cp-001", "side": "refuse", "prompt": "e"}],
             "spine_contrast": [{"id": "sc-001-clean", "source_id": "mr-001", "turns": [{"role": "user", "content": "f"}]}]}
    if refusal_as_manifest:
        files["refusal_probes.ids"] = [{k: v for k, v in r.items() if k != "turns"} | {"withheld": True} for r in files.pop("refusal_probes")]
        files["contrast_prompts"] = [{"id": "cp-001", "side": "refuse", "withheld": True}]
    for name, items in files.items():
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in items))
    return str(tmp_path)


def test_disjointness_audit_on_synthetic_dir(tmp_path):
    rep = disjointness_audit(write_probe_dir(tmp_path))
    assert rep["any_search_id_scored"] == []
    assert rep["spine_contrast_vs_pressure_probes"]["claim_status"] == "confirmatory"  # mr vs fp only
    assert rep["contrast_prompts_vs_refusal_probes"]["claim_status"] == "confirmatory"
    assert rep["contrast_prompts_vs_refusal_probes"]["lexical"]["items_with_text"] == [1, 1]


def test_disjointness_audit_falls_back_to_the_id_manifest(tmp_path):
    rep = disjointness_audit(write_probe_dir(tmp_path, refusal_as_manifest=True))
    r = rep["contrast_prompts_vs_refusal_probes"]
    assert r["n_scored"] == 1 and r["scored_templates"] == ["tool_bait"] and r["claim_status"] == "confirmatory"
    assert r["lexical"]["items_with_text"] == [0, 0]
    assert rep["any_search_id_scored"] == []


def _probe_counts():
    return {n: len(load_probe_set(KENJI, n)) for n in SEARCH_FILES + SCORED_FILES}


@pytest.mark.skipif(not os.path.exists(os.path.join(KENJI, "spine_contrast.jsonl")),
                    reason="Kenji's probe files (PR #3, branch cursor/steering-probes-clean-5853) not present in steer/probes/")
def test_kenji_probes_disjointness_invariant():
    """Hard invariants on the real probe files (PR #3): no search prompt is ever scored, counts as
    pre-registered, and both search/scored pairings are disjoint by id, source item, template and first
    turn, so both searches are confirmatory (PROTOCOL.md 8.2). `refusal_probes` may be present only as
    its id manifest. The lexical overlap is reported, not asserted zero; where text is available no
    pairing may share more than a 3-word generic stem."""
    rep = disjointness_audit(KENJI)
    assert rep["any_search_id_scored"] == []
    assert _probe_counts() == {"contrast_prompts": 40, "spine_contrast": 48, "pressure_probes": 12, "refusal_probes": 24,
                               "persona_probes": 30, "factual_probes": 30}
    for key in ("spine_contrast_vs_pressure_probes", "contrast_prompts_vs_refusal_probes"):
        r = rep[key]
        assert r["shared_ids"] == [] and r["shared_templates"] == [] and r["shared_first_turns"] == 0
        assert r["shared_gold"] == [] and r["shared_foil"] == [] and r["claim_status"] == "confirmatory"
        assert r["lexical"]["max_shared_ngram_words"] <= 3
    sp = rep["spine_contrast_vs_pressure_probes"]
    assert sp["search_templates"] == ["false_premise_search", "memory_rewrite_search"]
    assert sp["lexical"]["items_with_text"] == [48, 12]


@pytest.mark.skipif(not os.path.exists(os.path.join(KENJI, "refusal_probes.ids.jsonl")),
                    reason="refusal_probes.ids.jsonl (PR #3) not present in steer/probes/")
def test_refusal_manifest_has_ids_and_metadata_but_no_text():
    manifest = load_jsonl(os.path.join(KENJI, "refusal_probes.ids.jsonl"))
    assert [m["id"] for m in manifest] == [f"tb-{i:03d}" for i in range(1, 25)]
    for m in manifest:
        assert m["withheld"] is True and m["template"] == "tool_bait" and "turns" not in m and "prompt" not in m
        assert m["refusal_markers"] and m["compliance_markers"]
    full = os.path.join(KENJI, "refusal_probes.jsonl")
    if os.path.exists(full):  # local out-of-band copy: the manifest must describe exactly these items
        rows = load_jsonl(full)
        assert [r["id"] for r in rows] == [m["id"] for m in manifest]
        for r, m in zip(rows, manifest):
            assert all(r[k] == m[k] for k in ("template", "score_mode", "gold", "gold_logprob_target", "foil", "refusal_markers"))
            assert set(m["compliance_markers"]) <= set(r["compliance_markers"])
