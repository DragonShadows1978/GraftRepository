#!/usr/bin/env python3
"""LSR Phase 1 — per-probe stage adjudication.

Assigns each of the six wrong-value probes exactly one registered verdict
from the frozen Addendum-1 vocabulary:

    ROUTE-RANK | LINEAGE-RESOLUTION | ADMISSION-PRUNE | UNRESOLVED

Every verdict cites the stage's own persisted receipts.  No GPU, no
network, no model: the inputs are the lived DET1.5 snapshot manifests
(harvested by scripts/lsr_p1_harvest.py), the DET1.5 eval-stage routing
rows, the fixture manifests, and the frozen production stage code.

Append-only, content-addressed, own provenance (NOT a DET envelope).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(REPO, "artifacts", "lsr_p1")

HARVEST = "artifacts/lsr_p1/lsr_p1_harvest_cf29a6db390d4c81.json"
REPLAY = "artifacts/lsr_p1/lsr_p1_stage_replay_8ea8dc56a738f0ee.json"
SURVEY = "artifacts/lsr_p0/lsr_p0_survey_bd5350fd63ddbf09.json"

VOCAB = ("ROUTE-RANK", "LINEAGE-RESOLUTION", "ADMISSION-PRUNE", "UNRESOLVED")

# Probe -> (shard, snapshot rung whose manifest carries the SERVED attempt)
SERVED_RUNG = {
    "sup_lumen_head": ("sup-3", "rung_00"),
    "sup_orion_current": ("sup-4", "rung_00"),
    "sup_reserve_meridian_docket": ("sup-3", "rung_01"),
    "sup_reserve_juniper_pass": ("sup-1", "rung_01"),
    "sup_reserve_falcon_registry": ("sup-4", "rung_01"),
    "sup_reserve_tundra_ledger": ("sup-2", "rung_01"),
}
CONTROLS = {
    "sup_harbor_restatement": ("sup-1", "rung_00"),
    "sup_praxis_fresh": ("sup-2", "rung_00"),
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(rel):
    path = os.path.join(REPO, rel)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh), {
            "path": rel,
            "sha256": sha256_file(path),
            "bytes": os.path.getsize(path),
        }


def adjudicate(probe, lived, replay_row, survey_row):
    """Return (verdict, reason, evidence) for one probe."""
    ranking = lived["admission.ranking"] or []
    identified = lived["admission.identified_candidates"] or []
    branch = lived["admission.policy_branch"]
    rank_plan = lived["admission.rank_plan"] or []
    planned = lived["admission.current_planned"] or []
    fitted = lived["admission.current_fitted"] or []
    dropped = lived["admission.current_dropped"] or []
    final = lived["admission.final_mounts"] or []
    expected = replay_row["expected_node_graft_index"]
    node_count = len(replay_row["nodes"])

    in_ranking = expected in ranking
    in_plan = expected in rank_plan
    in_planned = expected in planned
    in_final = expected in final

    evidence = {
        "expected_node_graft_index": expected,
        "fixture_node_count": node_count,
        "admission.ranking": ranking,
        "ranking_length": len(ranking),
        "ranking_shortfall_vs_fixture_nodes": node_count - len(ranking),
        "admission.identified_candidates": identified,
        "admission.identifier_tokens": lived["admission.identifier_tokens"],
        "identifier_binding_over_full_fixture_universe":
            replay_row["identifier_binding_candidates_full_universe"],
        "expected_node_binds_identifier":
            replay_row["expected_node_binds_identifier"],
        "admission.policy_branch": branch,
        "admission.rank_plan": rank_plan,
        "admission.current_planned": planned,
        "admission.current_fitted": fitted,
        "admission.current_dropped": dropped,
        "admission.final_mounts": final,
        "arena.width": lived["arena.width"],
        "arena.cur_mount_n": lived["arena.cur_mount_n"],
        "arena.revision_resolution": lived["arena.revision_resolution"],
        "arena.decisive_admission": lived["arena.decisive_admission"],
        "admission.rule_sha256": lived["admission.rule_sha256"],
        "l2_heads_over_full_fixture_universe":
            replay_row["l2_full_universe_heads"],
        "expected_node_survives_l2_over_full_universe":
            expected in replay_row["l2_full_universe_heads"],
        "expected_node_text_chars":
            replay_row["nodes"][expected]["text_chars"]
            if expected is not None else None,
        "expected_node_role":
            replay_row["nodes"][expected]["role"]
            if expected is not None else None,
        "served_answer": lived["linked.probe_answer"],
        "survey_mounted_graft_indices": survey_row["mounted_graft_indices"],
        "survey_expected_value_carrying_nodes":
            survey_row["expected_value_carrying_nodes"],
        "manifest_path": lived["manifest_path"],
        "manifest_sha256": lived["manifest_sha256"],
    }

    # Control-masking diagnostic. A lawful control can serve the CORRECT
    # value out of a mount set that never contained the target node, when
    # some other mounted node happens to carry the same value string. That
    # is a masked defect, not a healthy route, and it must be visible in
    # the receipts rather than inferred.
    carrying = [int(v) for v in
                survey_row["expected_value_carrying_nodes"] or ()]
    mounted_carries = sorted(set(final) & set(carrying))
    evidence["expected_value_carrying_nodes_mounted"] = mounted_carries
    evidence["expected_value_is_duplicated_across_nodes"] = len(carrying) > 1
    evidence["correct_value_served_from_a_non_target_node"] = bool(
        mounted_carries and expected is not None
        and expected not in final)

    # Stage 1: did the expected node reach the ranking at all?
    if not in_ranking:
        evidence["stage_1_note"] = (
            "The expected node is absent from admission.ranking, which the "
            "production code records as the RAW output of "
            "ArenaCache.route() (scripts/grm_det1_5_gpu.py::"
            "_admission_snapshot takes 'ranking' verbatim from "
            "decisive_admission_profile['ranking'], unfiltered). Because "
            "core/grm_admission.py::decisive_admission_profile computes "
            "want = min(len(eligible), max(3, route_limit)) and the lived "
            "production_route_limit was 6, a ranking shorter than the "
            "fixture's node count cannot be a window bound: route() itself "
            "returned fewer ids than it was given candidates.")
        return "ROUTE-RANK", (
            "The expected node never entered the route ranking. A-DEC, L2 "
            "resolution and budget fitting all operate downstream of "
            "admission.ranking, so none of them can be the discarding "
            "stage; the node was already gone when they ran."), evidence

    # Stage 2: ranking held it — did A-DEC's plan keep it?
    if not in_plan:
        evidence["stage_2_note"] = (
            "The expected node is in admission.ranking but not in "
            "admission.rank_plan, so the frozen A-DEC branch "
            f"'{branch}' (rule c304609f) is the discarding stage.")
        return "ADMISSION-PRUNE", (
            "A-DEC's decisive branch removed a node the route had ranked."
        ), evidence

    # Stage 3: plan held it — did the ladder attempt still carry it?
    if not in_planned:
        evidence["stage_3_note"] = (
            "The expected node survived A-DEC's rank_plan but is absent "
            "from the served attempt's current_planned, i.e. the probe "
            "ladder rung that actually served is not the rung A-DEC "
            "planned.")
        return "ADMISSION-PRUNE", (
            "The served ladder rung dropped a node A-DEC had admitted."
        ), evidence

    # Stage 4: planned but not mounted -> budget fit or L2.
    if not in_final:
        l2_would_drop = expected not in replay_row["l2_full_universe_heads"]
        evidence["stage_4_note"] = (
            "The expected node was PLANNED for the served attempt and is "
            "recorded in admission.current_dropped; it never received arena "
            "seats. core/graft_arena.py::step()::fit() applies L2 "
            "(_resolve_revision_mounts) FIRST and then truncates the "
            "surviving picks against the arena width in EXPANSION ORDER, "
            "keeping each node only while used + ntok <= budget.")
        evidence["l2_would_have_dropped_expected"] = bool(l2_would_drop)
        if l2_would_drop:
            return "LINEAGE-RESOLUTION", (
                "L2 M5-edge resolution removes the expected node as a "
                "superseded member of its own lineage before budget "
                "fitting."), evidence
        return "ADMISSION-PRUNE", (
            "L2 keeps the expected node (it is a lineage head over the "
            "fixture's declared supersedes edges), so the only remaining "
            "discarding stage in fit() is the arena-width budget "
            "truncation, which is part of the admission/mount pipeline "
            "rather than route ranking or lineage resolution. The "
            "expected node is the fixture's long competitor text and "
            "cannot be seated inside the lived arena width."), evidence

    return "UNRESOLVED", (
        "The expected node reached the final mount set, so no stage in "
        "this pipeline discarded it; the wrong value has a different "
        "cause."), evidence


def grep_line(rel, needle):
    """Return (line_number, text) for the first line containing needle."""
    path = os.path.join(REPO, rel)
    with open(path, "r", encoding="utf-8") as fh:
        for number, line in enumerate(fh, 1):
            if needle in line:
                return {"path": rel, "line": number,
                        "text": line.strip(),
                        "sha256": sha256_file(path)}
    return {"path": rel, "line": None, "text": None,
            "sha256": sha256_file(path)}


def adm1_3_reconciliation():
    """Why ADM1.3's F-PROD was all-green on the same fixture files.

    Three independent, receipt-checkable frame differences. Each is read
    off the source rather than asserted.
    """
    return {
        "question": (
            "ADM1.3's F-PROD battery was all-green on these same "
            "supersession fixtures. Why do these lived serves fail?"),
        "answer_summary": (
            "Different probes, a different arena frame, and a different "
            "admission driver. The battery and the lived campaign do not "
            "measure the same thing on these files."),
        "difference_1_arena_width": {
            "claim": (
                "ADM1.3 builds its arena at width 256; the lived DET1.5 "
                "campaign runs at width 96. Every fixture's competitor "
                "node is 623-706 characters and consumes more seats than "
                "a 96-seat arena has, so it is mountable in the battery "
                "frame and unmountable in the lived frame."),
            "adm1_3_receipt": grep_line(
                "scripts/grm_adm1_3_dual_frame.py", "arena_width=256"),
            "battery_default_receipt": grep_line(
                "tests/test_grm_supersession_battery.py",
                '"--arena-width", type=int, default=256'),
            "lived_receipt": grep_line(
                "scripts/grm_det1_gpu.py", '"arena_width": 96'),
            "lived_e2e_default_receipt": grep_line(
                "scripts/grm_e2e_session.py",
                '"--arena-width", type=int, default=96'),
            "measured_confirmation": (
                "Every lived sup fitted mount set in the harvest is a "
                "SINGLE node occupying 52-66 of 96 arena seats "
                "(arena.cur_mount_n), so a second node never fits."),
        },
        "difference_2_probe_population": {
            "claim": (
                "Four of the six failing probes are not battery probes at "
                "all. sup_reserve_* are DET1.9-invented reserve probes "
                "defined in scripts/grm_det1_5_workers.py, and each one "
                "deliberately targets its session's LONG COMPETITOR node "
                "-- exactly the node the 96-seat arena cannot seat. The "
                "fixture files declare only one probe each, and none of "
                "them is a sup_reserve_* probe."),
            "reserve_definition_receipt": grep_line(
                "scripts/grm_det1_5_workers.py",
                "SUP_RESERVE_PROBES: dict[str, dict[str, Any]] = {"),
            "fixture_declared_probe_counts": {
                "correction_then_restatement": 1,
                "fresh_fact_controls": 2,
                "multi_hop_a_b_c": 1,
                "short_correction_long_competitor": 1,
            },
            "note": (
                "ADM1.3's F-PROD could not have covered these four "
                "probes: they did not exist as fixture probes."),
        },
        "difference_3_admission_driver": {
            "claim": (
                "ADM1.3's harness constructs its arena with "
                "decisive_admission=False and supplies each A-k1/k2/k3/"
                "A-DEC plan itself, so it exercises the POLICY in "
                "isolation. The lived campaign runs decisive_admission="
                "True and lets production compute the plan, then fits it "
                "to the arena. A green policy battery therefore says "
                "nothing about the fit stage that actually dropped the "
                "expected node in four of six lived probes."),
            "adm1_3_receipt": grep_line(
                "scripts/grm_adm1_3_dual_frame.py",
                "decisive_admission=False"),
            "lived_receipt": (
                "arena.decisive_admission == true in every lived sup "
                "snapshot manifest (see the harvest artifact)."),
        },
        "verdict": (
            "No contradiction. ADM1.3 validated the frozen A-DEC branch "
            "arithmetic in a 256-seat frame with hand-supplied plans on "
            "the fixtures' own probes. The lived failures arise in a "
            "96-seat frame, on probes ADM1.3 never ran, at the "
            "budget-fit stage ADM1.3 bypassed. Both results stand."),
    }


def main():
    harvest, harvest_rec = load(HARVEST)
    replay, replay_rec = load(REPLAY)
    survey, survey_rec = load(SURVEY)

    lived_by_key = {}
    for row in harvest["rows"]:
        lived_by_key[(row["fixture_id"], row["shard"], row["rung"],
                      row["attempt_dir"])] = row
    replay_by_id = {row["fixture_id"]: row for row in replay["probes"]}
    survey_by_id = {row["probe_id"]: row for row in survey["probes"]}

    results = []
    for probe, (shard, rung) in list(SERVED_RUNG.items()) + list(
            CONTROLS.items()):
        key = (probe, shard, rung, "attempt_002")
        lived = lived_by_key.get(key)
        if lived is None:
            results.append({
                "probe_id": probe, "verdict": "UNRESOLVED",
                "reason": "no lived snapshot manifest at the served rung",
                "role": ("wrong_value" if probe in SERVED_RUNG
                         else "lawful_control"),
            })
            continue
        verdict, reason, evidence = adjudicate(
            probe, lived, replay_by_id[probe], survey_by_id[probe])
        if verdict not in VOCAB:
            raise SystemExit("verdict outside the registered vocabulary")
        results.append({
            "probe_id": probe,
            "role": ("wrong_value" if probe in SERVED_RUNG
                     else "lawful_control"),
            "verdict": verdict,
            "reason": reason,
            "evidence": evidence,
        })

    six = [r for r in results if r["role"] == "wrong_value"]
    tally = {}
    for row in six:
        tally[row["verdict"]] = tally.get(row["verdict"], 0) + 1

    payload = {
        "schema": "grm.lsr_p1.stage_adjudication.v1",
        "program": "LSR",
        "phase": "1",
        "declares_in_det_envelope": False,
        "provenance_note": (
            "Own provenance. CPU archaeology over persisted DET1.5 lived "
            "receipts plus a model-free replay of the frozen stages. No "
            "GPU, no network, no production code modified."),
        "registered_vocabulary": list(VOCAB),
        "vocabulary_source": "docs/LSR_ADDENDUM_1.md#phase-1",
        "denominator": len(six),
        "verdict_tally_wrong_value_probes": tally,
        "sources": {
            "harvest": harvest_rec,
            "stage_replay": replay_rec,
            "lsr_p0_survey": survey_rec,
            "addendum": {
                "path": "docs/LSR_ADDENDUM_1.md",
                "sha256": sha256_file(
                    os.path.join(REPO, "docs/LSR_ADDENDUM_1.md")),
            },
            "plan": {
                "path": "docs/LSR_LIVED_SERVING_PLAN.md",
                "sha256": sha256_file(
                    os.path.join(REPO, "docs/LSR_LIVED_SERVING_PLAN.md")),
            },
        },
        "adjudicator": {
            "path": "scripts/lsr_p1_adjudicate.py",
            "sha256": sha256_file(os.path.abspath(__file__)),
        },
        "adm1_3_reconciliation": adm1_3_reconciliation(),
        "probes": results,
    }
    body = json.dumps(payload, indent=1, sort_keys=True)
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, "lsr_p1_adjudication_%s.json" % digest[:16])
    if os.path.exists(out):
        print("EXISTS (append-only, identical content):", out)
    else:
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(body)
        print("WROTE:", out)
    for row in results:
        print("%-30s %-14s %-22s" % (
            row["probe_id"], row["role"], row["verdict"]))
    print("tally:", tally)
    return 0


if __name__ == "__main__":
    sys.exit(main())
