#!/usr/bin/env python3
"""LSR Phase 1 — CPU replay of the identifier / A-DEC / L2 stages.

Replays the model-free stages of the production routing decision on the
fixture texts exactly as the lived campaign built them:

  * turn text   : scripts.grm_det1_3_gpu._split_fixture_turn +
                  scripts.grm_e2e_session.harmony_turn (the lived
                  _install_lived_nodes path)
  * identifier  : core.grm_admission.is_identifier_binding (frozen rule
                  c304609f...), driven by the identifier_tokens the lived
                  receipts actually recorded
  * A-DEC       : core.grm_admission.policy_plan (frozen branch precedence)
  * L2          : the _revision_mount_heads contract, replayed as a pure
                  set filter over the fixtures' declared supersedes edges

This settles which stage could and could not have discarded the expected
node, WITHOUT a model: the route SCORES need the model, but the branch
arithmetic and the M5 filter do not.

Read-only. Writes one content-addressed artifact under artifacts/lsr_p1/.
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
    is_identifier_binding, normalized_words, policy_plan,
)

FIXDIR = os.path.join(REPO, "tests", "fixtures", "supersession_battery")
OUTDIR = os.path.join(REPO, "artifacts", "lsr_p1")

# The four DET1.9-invented reserve probes, verbatim from
# scripts/grm_det1_5_workers.py::SUP_RESERVE_PROBES.
RESERVE_PROBES = {
    "correction_then_restatement": {
        "fixture_id": "sup_reserve_juniper_pass",
        "question": "What is the current Juniper pass value?",
        "expected_values": ["Opal-7-Green"],
    },
    "fresh_fact_controls": {
        "fixture_id": "sup_reserve_tundra_ledger",
        "question": "What is the current Tundra ledger value?",
        "expected_values": ["Sable-0-Copper"],
    },
    "multi_hop_a_b_c": {
        "fixture_id": "sup_reserve_meridian_docket",
        "question": "What is the current Meridian docket value?",
        "expected_values": ["Delta-4-Drift"],
    },
    "short_correction_long_competitor": {
        "fixture_id": "sup_reserve_falcon_registry",
        "question": "What is the current Falcon registry value?",
        "expected_values": ["Vortex-3-Sierra"],
    },
}

# Identifier token lists exactly as the lived receipts recorded them
# (state["admission.identifier_tokens"]).  Using the receipt's own tokens
# keeps the replay bound to measured production behavior instead of a
# re-derivation that would need the arena's rare-token model state.
LIVED_IDENTIFIER_TOKENS = {
    "sup_lumen_head": ["lumen", "seal"],
    "sup_orion_current": ["orion", "pin"],
    "sup_harbor_restatement": ["harbor", "token"],
    "sup_praxis_fresh": ["praxis", "dock"],
    "sup_solace_fresh": ["solace", "key"],
    "sup_reserve_juniper_pass": ["juniper", "pass"],
    "sup_reserve_tundra_ledger": ["tundra", "ledger"],
    "sup_reserve_meridian_docket": ["meridian", "docket"],
    "sup_reserve_falcon_registry": ["falcon", "registry"],
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_fixture_turn(text):
    prefix = "User: "
    marker = "\nAssistant: "
    user, assistant = text[len(prefix):].split(marker, 1)
    return user, assistant


def revision_mount_heads(picks, supersedes_by_index, node_count):
    """Replay of ArenaCache._revision_mount_heads (resolution ON arm)."""
    unique = []
    for raw in picks or ():
        idx = int(raw)
        if 0 <= idx < node_count and idx not in unique:
            unique.append(idx)
    if len(unique) < 2:
        return list(unique)
    present = set(unique)
    memo = {}

    def ancestors(idx, visiting):
        if idx in visiting:
            raise ValueError("cycle")
        if idx in memo:
            return memo[idx]
        visiting.add(idx)
        direct = tuple(supersedes_by_index.get(idx, ()))
        found = set(direct)
        for old in direct:
            found.update(ancestors(old, visiting))
        visiting.remove(idx)
        memo[idx] = found
        return found

    superseded_present = set()
    try:
        for idx in unique:
            superseded_present.update(ancestors(idx, set()) & present)
    except ValueError:
        return unique
    return [idx for idx in unique if idx not in superseded_present]


def analyze_probe(fixture, probe_question, expected_values, fixture_id,
                  target_node_id):
    nodes = fixture["nodes"]
    node_to_idx = {n["node_id"]: i for i, n in enumerate(nodes)}
    supersedes_by_index = {
        i: [node_to_idx[v] for v in n.get("supersedes", [])]
        for i, n in enumerate(nodes)
    }
    tokens = LIVED_IDENTIFIER_TOKENS[fixture_id]
    # CORRECTED 2026-09-01 (LSR-P1 reconciliation). An earlier revision of
    # this replay passed rare_identifier_tokens=tokens, which selects
    # is_identifier_binding's SUBSET branch (rare <= words) and made all
    # four nodes bind. That was WRONG.
    #
    # Measured on the rebuilt lived arena (GPU, pinned frame):
    #     ordered_identifier_tokens(arena, question)
    #       -> ordered=['harbor','token'], rare=set()   <-- rare is EMPTY
    # because ArenaCache._rare_tokens only accepts code/number-shaped
    # tokens (a digit, or ALL-CAPS length>=3); lowercase labels like
    # "harbor"/"token" never qualify. The identifier list comes from the
    # CONTENT-WORD channel instead, and an empty rare set sends the frozen
    # predicate to its ORDERED-PHRASE branch: the candidate must contain
    # the contiguous phrase ["current", *ordered, "value"].
    #
    # That reproduces the lived identified_candidates EXACTLY (harbor
    # [1,0]; see the reconciliation artifact), so the replay now models
    # production instead of contradicting it.
    rare_for_predicate = []
    binding = []
    per_node = []
    for i, n in enumerate(nodes):
        user, assistant = split_fixture_turn(n["text"])
        # The deposited graft text is the harmony turn built from the pair;
        # identifier binding reads word membership, which the harmony
        # wrapper does not change for these fixtures.
        candidate_text = user + "\n" + assistant
        hit = is_identifier_binding(
            candidate_text=candidate_text,
            ordered_identifier_tokens=tokens,
            rare_identifier_tokens=rare_for_predicate,
        )
        if hit:
            binding.append(i)
        per_node.append({
            "graft_index": i,
            "node_id": n["node_id"],
            "role": n["role"],
            "value": n["value"],
            "text_chars": len(n["text"]),
            "identifier_binding": bool(hit),
            "supersedes": supersedes_by_index[i],
            "word_count": len(normalized_words(candidate_text)),
        })
    expected_idx = (node_to_idx.get(target_node_id)
                    if target_node_id else None)
    if expected_idx is None:
        low = {str(v).casefold() for v in expected_values}
        for i, n in enumerate(nodes):
            if str(n["value"]).casefold() in low:
                expected_idx = i
                break
    return {
        "fixture_id": fixture_id,
        "session_id": fixture["session_id"],
        "question": probe_question,
        "identifier_tokens_from_lived_receipt": tokens,
        "rare_identifier_tokens_used": list(rare_for_predicate),
        "binding_branch": (
            "ordered_phrase [current, *identifier, value] (rare set is "
            "EMPTY in production: _rare_tokens accepts only code/number "
            "shaped tokens, never lowercase labels)"),
        "expected_values": list(expected_values),
        "expected_node_graft_index": expected_idx,
        "identifier_binding_candidates_full_universe": binding,
        "expected_node_binds_identifier": (
            expected_idx in binding if expected_idx is not None else None),
        "nodes": per_node,
        "supersedes_edges": supersedes_by_index,
        "l2_full_universe_heads": revision_mount_heads(
            list(range(len(nodes))), supersedes_by_index, len(nodes)),
    }


def main():
    fixtures = {}
    for name in sorted(os.listdir(FIXDIR)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(FIXDIR, name)
        with open(path, "r", encoding="utf-8") as fh:
            fixtures[name] = (json.load(fh), path)

    results = []
    for name, (fixture, path) in fixtures.items():
        session = fixture["session_id"]
        for probe in fixture["probes"]:
            fid = "sup_" + probe["probe_id"]
            if fid not in LIVED_IDENTIFIER_TOKENS:
                continue
            row = analyze_probe(
                fixture, probe["question"], probe["expected_values"], fid,
                probe.get("target_node"))
            row["probe_origin"] = "fixture_declared_probe"
            row["fixture_file"] = os.path.relpath(path, REPO)
            row["fixture_sha256"] = sha256_file(path)
            results.append(row)
        reserve = RESERVE_PROBES.get(session)
        if reserve and reserve["fixture_id"] in LIVED_IDENTIFIER_TOKENS:
            row = analyze_probe(
                fixture, reserve["question"], reserve["expected_values"],
                reserve["fixture_id"], None)
            row["probe_origin"] = (
                "det1_9_invented_reserve_probe "
                "(scripts/grm_det1_5_workers.py::SUP_RESERVE_PROBES) — "
                "NOT declared by the fixture file")
            row["fixture_file"] = os.path.relpath(path, REPO)
            row["fixture_sha256"] = sha256_file(path)
            results.append(row)

    payload = {
        "schema": "grm.lsr_p1.stage_replay.v1",
        "program": "LSR",
        "phase": "1",
        "declares_in_det_envelope": False,
        "provenance_note": (
            "Own provenance. Model-free replay of the frozen identifier / "
            "A-DEC / L2 stages against the fixture texts. Route SCORES are "
            "model-dependent and are NOT reconstructed here; they are read "
            "from the lived receipts by scripts/lsr_p1_harvest.py."),
        "frozen_rule_sha256": (
            "c304609f81475bd2ae3399ad180d3bb00cc5d2b44b1a49db570387c810defb91"),
        "replay_source": {
            "path": "scripts/lsr_p1_stage_replay.py",
            "sha256": sha256_file(os.path.abspath(__file__)),
        },
        "policy_plan_selfcheck": {
            "sole_hit_at_rank1": policy_plan(
                ranking=[3, 2, 1, 0], identified_candidates=[3],
                route_margin_1_2=0.0),
            "sole_hit_off_rank1_low_margin": policy_plan(
                ranking=[2, 3, 1, 0], identified_candidates=[3],
                route_margin_1_2=0.0),
            "two_hits": policy_plan(
                ranking=[1, 0], identified_candidates=[1, 0],
                route_margin_1_2=0.0),
        },
        "probes": results,
    }
    body = json.dumps(payload, indent=1, sort_keys=True)
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, "lsr_p1_stage_replay_%s.json" % digest[:16])
    if os.path.exists(out):
        print("EXISTS (append-only, identical content):", out)
    else:
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(body)
        print("WROTE:", out)
    print("probes:", len(results), "sha256:", digest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
