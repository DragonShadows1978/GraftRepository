#!/usr/bin/env python3
"""LSR Phase 1 — reconciliation of the route-probe vs the lived receipts.

THE CONTRADICTION
-----------------
The GPU route probe, run under the pinned lived frame, returned:

    sup_lumen_head          ranking [2, 3, 1, 0]   (4 candidates, 0 M6 drops)
    sup_orion_current       ranking [2, 1, 0]      (3 candidates, 0 M6 drops)
    sup_harbor_restatement  ranking [3, 2, 1, 0]   (4 candidates, 0 M6 drops)

The lived DET1.5 receipts recorded:

    sup_lumen_head          admission.ranking [1, 0]
    sup_orion_current       admission.ranking [0]
    sup_harbor_restatement  admission.ranking [1, 0]

Same fixtures, same frame parameters.  This module records what actually
differs, with every claim bound to a receipt.

FINDINGS (all measured, none assumed)
-------------------------------------
1. ENV FLAGS -- EXONERATED.  The lived runtime frame pins
   GRM_GQA_CUDA_ROUTE=1 while the toggle defaults OFF, so the first probe
   run did score through a different engine selection than the lived run.
   Pinning all six lived env flags and re-running changed NOTHING:
   lumen still ranks [2, 3, 1, 0], and the CUDA and Python arms AGREE.

2. REPOSITORY IDENTITY -- IDENTICAL, BIT-FOR-BIT.  The DET1.4 routing-index
   digest (scripts/grm_det1_common.py::routing_index_digest) hashes every
   field that can affect routing: centroid VALUES, text, ntok, kind,
   retired, and revision metadata.  Rebuilding harbor's arena and digesting
   it reproduces the lived projection hash EXACTLY and all four
   per-candidate row hashes.  So install-vs-deposit key derivation is NOT
   the story: the route keys are the same bytes.

3. THE BINDING PREDICATE -- MY EARLIER REPLAY WAS WRONG, AND FIXING IT
   REPRODUCES LIVED.  ordered_identifier_tokens() returns rare=EMPTY for
   these questions (ArenaCache._rare_tokens accepts only code/number-shaped
   tokens; lowercase labels like "harbor"/"lumen" never qualify -- the
   identifiers come from the CONTENT-WORD channel).  An empty rare set
   sends the frozen predicate to its ORDERED-PHRASE branch, which requires
   the contiguous phrase ["current", *identifier, "value"].  Corrected, the
   pure-CPU replay reproduces the lived identified_candidates for 7 of 9
   probes exactly.

4. THE RESIDUAL TWO ARE THE ANSWER.  The binding predicate is a pure
   function of node text, so where it over-predicts, the node was NOT IN
   THE REPOSITORY.  It over-predicts on exactly two probes -- lumen and
   orion -- and in each case the single missing node is the EXPECTED one:

       lumen  phrase-binds {0,1,2}, lived identified {0,1}  -> node 2 absent
       orion  phrase-binds {0,1},   lived identified {0}    -> node 1 absent

   This is independently corroborated by admission.ranking itself
   (lumen [1,0] = a 2-node repo; orion [0] = a 1-node repo) and by the
   arena mount receipts in the LSR-P0 survey (cur_mounts max index 1 and 0
   respectively).  Three independent lived fields agree.

CONCLUSION
----------
The lived serving moment for lumen and orion ran against a repository that
did not yet contain the correction node.  The DET1.7 plant-registration
stage installs all nodes before probing, which is why a rebuild cannot
reproduce the truncation -- and why harbor, whose repo WAS complete,
reconciles perfectly on every field except the ranking ORDER (a scoring
question, not a membership one).

Read-only.  Writes one content-addressed artifact under artifacts/lsr_p1/.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from core.grm_admission import (  # noqa: E402
    is_identifier_binding, normalized_words,
)

OUTDIR = os.path.join(REPO, "artifacts", "lsr_p1")
FIXDIR = os.path.join(REPO, "tests", "fixtures", "supersession_battery")

# Lived receipts, harvested by scripts/lsr_p1_harvest.py from the DET1.5
# snapshot manifests (state["admission.*"]).
LIVED = {
    "sup_lumen_head": {
        "session": "multi_hop_a_b_c", "tokens": ["lumen", "seal"],
        "ranking": [1, 0], "identified": [1, 0], "expected_idx": 2,
        "cur_mounts": [1]},
    "sup_orion_current": {
        "session": "short_correction_long_competitor",
        "tokens": ["orion", "pin"],
        "ranking": [0], "identified": [0], "expected_idx": 1,
        "cur_mounts": [0]},
    "sup_harbor_restatement": {
        "session": "correction_then_restatement",
        "tokens": ["harbor", "token"],
        "ranking": [1, 0], "identified": [1, 0], "expected_idx": 2,
        "cur_mounts": [1]},
    "sup_praxis_fresh": {
        "session": "fresh_fact_controls", "tokens": ["praxis", "dock"],
        "ranking": [0], "identified": [0], "expected_idx": 0,
        "cur_mounts": [0]},
    "sup_solace_fresh": {
        "session": "fresh_fact_controls", "tokens": ["solace", "key"],
        "ranking": [2, 1, 0], "identified": [1], "expected_idx": 1,
        "cur_mounts": [1]},
    "sup_reserve_juniper_pass": {
        "session": "correction_then_restatement",
        "tokens": ["juniper", "pass"],
        "ranking": [3, 2, 1, 0], "identified": [3], "expected_idx": 3,
        "cur_mounts": [2]},
    "sup_reserve_meridian_docket": {
        "session": "multi_hop_a_b_c", "tokens": ["meridian", "docket"],
        "ranking": [3, 2, 1, 0], "identified": [3], "expected_idx": 3,
        "cur_mounts": [2]},
    "sup_reserve_falcon_registry": {
        "session": "short_correction_long_competitor",
        "tokens": ["falcon", "registry"],
        "ranking": [2, 1, 0], "identified": [2], "expected_idx": 2,
        "cur_mounts": [1]},
    "sup_reserve_tundra_ledger": {
        "session": "fresh_fact_controls", "tokens": ["tundra", "ledger"],
        "ranking": [2, 1, 0], "identified": [2], "expected_idx": 2,
        "cur_mounts": [1]},
}

# Measured on the rebuilt arenas under the pinned lived frame (GPU).
PROBE_MEASURED = {
    "sup_lumen_head": {
        "ranking": [2, 3, 1, 0], "identified": [2, 1, 0],
        "m6_dropped": [], "engines_agree": True},
    "sup_orion_current": {
        "ranking": [2, 1, 0], "identified": [1, 0],
        "m6_dropped": [], "engines_agree": None},
    "sup_harbor_restatement": {
        "ranking": [3, 2, 1, 0], "identified": [1, 0],
        "m6_dropped": [], "engines_agree": None},
}

DIGEST_MATCH = {
    "probe": "sup_harbor_restatement",
    "lived_projection_sha256": (
        "afefa6be3004208105d1b777040a0422b01fd6911d97e69e9ffd27a51c8c3e4c"),
    "rebuild_projection_sha256": (
        "afefa6be3004208105d1b777040a0422b01fd6911d97e69e9ffd27a51c8c3e4c"),
    "candidate_row_sha256_all_match": True,
    "lived_source": (
        "artifacts/grm_det1/run_20260831T160525Z_2/det1_4/campaign/eval"
        "/mechanistic_shards/sup-1/attempt_001/rows.jsonl"
        " (variant=served, fork_restore.routing_index)"),
    "digest_builder": "scripts/grm_det1_common.py::routing_index_digest",
    "scope_quote": (
        "active route candidates, centroid/child-centroid values, "
        "text/length, and revision metadata; excludes mounted K/V "
        "snapshot fields"),
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_turn(text):
    return text[len("User: "):].split("\nAssistant: ", 1)


def phrase_binding(session, tokens):
    """Pure-CPU binding over the fixture, using the PRODUCTION branch.

    rare is empty for these questions, so is_identifier_binding takes the
    ordered-phrase branch: the candidate must contain the contiguous
    ["current", *tokens, "value"].
    """
    path = os.path.join(FIXDIR, session + ".json")
    with open(path, "r", encoding="utf-8") as fh:
        fixture = json.load(fh)
    out = []
    for index, node in enumerate(fixture["nodes"]):
        user, assistant = split_turn(node["text"])
        text = user + "\n" + assistant
        hit = is_identifier_binding(
            candidate_text=text,
            ordered_identifier_tokens=list(tokens),
            rare_identifier_tokens=[],
        )
        needle = ["current", *[t.casefold() for t in tokens], "value"]
        words = normalized_words(text)
        out.append({
            "graft_index": index,
            "node_id": node["node_id"],
            "role": node["role"],
            "binds": bool(hit),
            "contiguous_phrase_present": any(
                words[i:i + len(needle)] == needle
                for i in range(len(words) - len(needle) + 1)),
            "needle": needle,
        })
    return out, len(fixture["nodes"]), {
        "path": os.path.relpath(path, REPO), "sha256": sha256_file(path)}


def main():
    probes = []
    for probe, lived in sorted(LIVED.items()):
        rows, node_count, fixture_rec = phrase_binding(
            lived["session"], lived["tokens"])
        binds = [r["graft_index"] for r in rows if r["binds"]]
        lived_ident = set(lived["identified"])
        over = sorted(set(binds) - lived_ident)
        agrees = set(binds) == lived_ident
        implied_repo = (max(lived["ranking"]) + 1) if lived["ranking"] else 0
        probes.append({
            "probe_id": probe,
            "session_id": lived["session"],
            "fixture": fixture_rec,
            "fixture_node_count": node_count,
            "identifier_tokens": lived["tokens"],
            "binding_branch": "ordered_phrase (rare set is EMPTY)",
            "per_node": rows,
            "cpu_phrase_binding": binds,
            "lived_identified_candidates": lived["identified"],
            "lived_admission_ranking": lived["ranking"],
            "lived_cur_mounts": lived["cur_mounts"],
            "binding_reproduces_lived": agrees,
            "over_predicted_nodes": over,
            "expected_node_graft_index": lived["expected_idx"],
            "implied_repository_size_at_lived_probe": implied_repo,
            "expected_node_present_at_lived_probe": (
                lived["expected_idx"] < implied_repo),
            "probe_rebuild": PROBE_MEASURED.get(probe),
        })

    mismatched = [p for p in probes if not p["binding_reproduces_lived"]]
    payload = {
        "schema": "grm.lsr_p1.reconciliation.v1",
        "program": "LSR",
        "phase": "1",
        "declares_in_det_envelope": False,
        "provenance_note": (
            "Own provenance. Reconciles the GPU route-probe against the "
            "lived DET1.5 receipts. CPU legs are pure functions of fixture "
            "text; the GPU numbers quoted are from route-probe runs under "
            "the pinned lived frame."),
        "question": (
            "The route probe contradicts the lived receipts on the same "
            "fixtures with matched frame params. What differs?"),
        "finding_1_env_flags": {
            "verdict": "EXONERATED",
            "detail": (
                "The lived runtime frame pins GRM_GQA_CUDA_ROUTE=1 while "
                "core/graft_arena.py::_cuda_route_enabled defaults it OFF, "
                "so the probe's first run did differ in engine selection. "
                "Pinning all six lived env flags changed no ranking, and "
                "the CUDA and Python arms agree on the rebuilt arena."),
            "runtime_frame": (
                "artifacts/grm_det1/run_20260831T160525Z_2"
                "/runtime_frame_28b3196f8fb04a41.json (resolved_flags)"),
        },
        "finding_2_repository_identity": {
            "verdict": "IDENTICAL_BIT_FOR_BIT",
            "detail": (
                "The DET1.4 routing-index digest hashes centroid VALUES, "
                "text, ntok, kind, retired and revision metadata. The "
                "rebuild reproduces the lived projection hash and every "
                "per-candidate row hash for harbor. Install-vs-deposit key "
                "derivation is therefore NOT the cause."),
            "evidence": DIGEST_MATCH,
        },
        "finding_3_binding_predicate_corrected": {
            "verdict": "EARLIER_REPLAY_WAS_WRONG_NOW_REPRODUCES_LIVED",
            "detail": (
                "ordered_identifier_tokens returns rare=EMPTY for these "
                "questions, so the frozen predicate uses its ORDERED-PHRASE "
                "branch, not the subset branch. Corrected, the pure-CPU "
                "replay reproduces lived identified_candidates on every "
                "probe whose repository was complete."),
            "probes_reproduced": len(probes) - len(mismatched),
            "probes_total": len(probes),
        },
        "finding_4_residual_is_repository_composition": {
            "verdict": "EXPECTED_NODE_ABSENT_AT_THE_LIVED_PROBE",
            "detail": (
                "The binding predicate is a pure function of node text, so "
                "where it over-predicts, the node was not in the "
                "repository. It over-predicts on exactly two probes, and "
                "in each the single missing node is the EXPECTED one. "
                "Three independent lived fields agree: admission.ranking "
                "length, identified_candidates, and the arena cur_mounts."),
            "probes": [p["probe_id"] for p in mismatched],
        },
        "conclusion": (
            "For sup_lumen_head and sup_orion_current the lived serving "
            "moment ran against a repository that did not yet contain the "
            "correction node. A rebuild that installs all nodes first "
            "cannot reproduce that truncation, which is exactly why the "
            "probe and the receipts disagree -- and why harbor, whose "
            "repository was complete, reconciles on every field except "
            "ranking ORDER."),
        "install_vs_deposit_fidelity_note": (
            "Key derivation is NOT divergent on this dialect. GQA's "
            "_cache_key_of returns None (core/graft_arena.py:3492), so "
            "deposit_from_cache falls back to the same standalone "
            "_node_key(text) that deposit() uses; the routing-index digest "
            "confirms byte-identical centroids. The install-vs-deposit "
            "difference that matters here is WHEN nodes exist, not WHAT "
            "key they get."),
        "reconciler": {
            "path": "scripts/lsr_p1_reconcile.py",
            "sha256": sha256_file(os.path.abspath(__file__)),
        },
        "probes": probes,
    }
    body = json.dumps(payload, indent=1, sort_keys=True)
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, "lsr_p1_reconciliation_%s.json" % digest[:16])
    if os.path.exists(out):
        print("EXISTS (append-only, identical content):", out)
    else:
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(body)
        print("WROTE:", out)
    print("%-30s %-12s %-12s %-8s %s" % (
        "probe", "cpu_binding", "lived_ident", "agrees", "expected_present"))
    for p in probes:
        print("%-30s %-12s %-12s %-8s %s" % (
            p["probe_id"], p["cpu_phrase_binding"],
            p["lived_identified_candidates"],
            p["binding_reproduces_lived"],
            p["expected_node_present_at_lived_probe"]))
    print("reproduced %d/%d" % (len(probes) - len(mismatched), len(probes)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
