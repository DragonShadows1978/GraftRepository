#!/usr/bin/env python3
"""LSR-P2B G2 — replay proof: persisted receipts reproduce Phase 1's evidence.

CPU only, model-free, no re-serve.  For each of the 8 probes the Phase 1
adjudication ruled on, this script:

  1. reads the LIVED DET1.5 receipt the probe served from (the snapshot
     manifest that ``scripts/lsr_p1_harvest.py`` harvests),
  2. builds the ``grm.route_receipt.v1`` record a P2B-instrumented serve
     would have written at that turn, from that lived state ALONE,
  3. renders it through ``scripts/lsr_p2b_route_receipt_dump.py``, and
  4. compares the rendered evidence block, field for field, with the Phase 1
     adjudication artifact.

Fields the new receipt WOULD carry but the lived receipts never recorded are
listed explicitly as "closed by P2B, not retro-provable" -- they are not
scored as matches.

The DET1.5 shard tree and the Phase 1 artifacts are read-only inputs and may
live outside this worktree; point at them with --det-root / --p1-adjudication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from core.grm_three_pass import build_route_receipt  # noqa: E402
from scripts.lsr_p2b_route_receipt_dump import (  # noqa: E402
    EVIDENCE_FIELDS,
    turn_block,
)

# Fields the P2B receipt carries that no lived DET1.5 receipt recorded.  A
# future serve fills them; this replay cannot, and says so rather than
# fabricating a match.
NOT_RETRO_PROVABLE = (
    ("route.candidate_count",
     "candidate base size at route time; lived manifests record the returned "
     "ranking only"),
    ("route.route_limit",
     "the limit ARGUMENT passed to route(); lived manifests record the "
     "resulting ranking, not the bound"),
    ("route.ranking_scores",
     "per-candidate route scores; never persisted at serve time"),
    ("route.excluded_live_ids",
     "the live/recency exclusion set handed to route()"),
    ("route.excluded_recency_ids",
     "the recency exclusion set handed to route()"),
    ("route.fallbacks",
     "route fallback reason codes; the arena route receipt is profile-gated "
     "and was off in the lived campaign"),
    ("provenance.arena_state_sha256_before_serve",
     "canonical arena-state digest before the serve"),
    ("provenance.request_sha256",
     "sha256 of the request text"),
    ("provenance.output_sha256",
     "sha256 of the served output"),
    ("fit.trips[].mount_set / fit.trips[].grounded",
     "per-trip seated set and grounded flag; lived ladder_attempts record "
     "planned sets only"),
    ("fit.info_pass_through",
     "P2A's fit_*/abstain*/served_without_plan_head fields; they do not exist "
     "yet in the lived code"),
)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


class LivedArenaView:
    """Read-only view of the lived arena geometry, straight from a manifest."""

    grafts = ()

    def __init__(self, state):
        self.cur_mounts = list(state.get("arena.cur_mounts") or ())
        self.cur_mount_n = state.get("arena.cur_mount_n")
        self.width = state.get("arena.width")
        self.last_route_backend = state.get("admission.route_backend")
        self.last_route_receipt = None


def lived_receipt_record(manifest, *, probe_id, session_id):
    """The route receipt a P2B serve WOULD have written at that lived turn.

    Built from the manifest's own persisted state -- no re-serve, no model.
    """
    state = manifest.get("state") or {}
    ranking = list(state.get("admission.ranking") or ())
    trips = []
    for row in state.get("admission.ladder_attempts") or ():
        trips.append({
            "ordinal": row.get("ordinal"),
            "clean_room": row.get("clean_room"),
            "planned": row.get("planned"),
            # Not recorded in the lived receipts -- left empty/None on purpose.
            "mount_set": [],
            "grounded": None,
        })
    info = {
        "mounts": [int(i) + 1 for i in (
            state.get("admission.final_mounts") or ())],
        "mount_plan": list(state.get("admission.current_planned") or ()),
        "mount_fitted": list(state.get("admission.current_fitted") or ()),
        "mount_dropped_for_width": list(
            state.get("admission.current_dropped") or ()),
        "ranking_ids": ranking,
        "admission_policy": "A-DEC" if state.get(
            "arena.decisive_admission") else None,
        "admission_policy_branch": state.get("admission.policy_branch"),
        "admission_rank_plan": list(state.get("admission.rank_plan") or ()),
        "admission_identified_candidates": list(
            state.get("admission.identified_candidates") or ()),
        "admission_route_margin_1_2": state.get("admission.route_margin_1_2"),
        "admission_margin_threshold": state.get("admission.margin_threshold"),
        "admission_rule_sha256": state.get("admission.rule_sha256"),
    }
    profile = {
        "identifier_tokens": state.get("admission.identifier_tokens"),
        "rare_identifier_tokens": None,
        "ranking": ranking,
        "rank_plan": list(state.get("admission.rank_plan") or ()),
        "identified_candidates": list(
            state.get("admission.identified_candidates") or ()),
    }
    # Phase 1's stage_0_note: the lived repository size is implied by
    # max(ranking)+1 and corroborated by identified_candidates.
    repository_size = (max(ranking) + 1) if ranking else 0
    return build_route_receipt(
        session_id=session_id,
        turn_id=probe_id,
        info=info,
        arena=LivedArenaView(state),
        repository_size=repository_size,
        serving_path="lsr_p2b_g2_replay.lived_det1_5_receipt",
        trips=trips,
        admission_profile=profile,
        turn_kind="probe",
    )


def find_manifest(det_root, rel_path):
    candidate = os.path.join(det_root, rel_path)
    if os.path.isfile(candidate):
        return candidate
    trimmed = rel_path.split("artifacts/", 1)[-1]
    candidate = os.path.join(det_root, "artifacts", trimmed)
    if os.path.isfile(candidate):
        return candidate
    return None


def replay_probe(probe, det_root):
    evidence = probe.get("evidence") or {}
    rel = evidence.get("manifest_path")
    manifest_path = find_manifest(det_root, rel) if rel else None
    row = {
        "probe_id": probe.get("probe_id"),
        "phase1_verdict": probe.get("verdict"),
        "manifest_path": rel,
        "manifest_found": manifest_path is not None,
    }
    if manifest_path is None:
        row["fields"] = None
        row["all_match"] = False
        row["error"] = "lived manifest not found under --det-root"
        return row
    manifest = load_json(manifest_path)
    row["manifest_sha256"] = sha256_file(manifest_path)
    row["manifest_sha256_matches_phase1"] = (
        row["manifest_sha256"] == evidence.get("manifest_sha256"))
    record = lived_receipt_record(
        manifest, probe_id=str(probe.get("probe_id")),
        session_id="det1_5_lived")
    block = turn_block(record)
    replayed = block["evidence"]
    fields = {}
    for field in EVIDENCE_FIELDS:
        present = field in evidence
        fields[field] = {
            "phase1": evidence.get(field),
            "replayed": replayed.get(field),
            "in_phase1_block": present,
            "match": (evidence[field] == replayed.get(field)
                      if present else None),
        }
    row["fields"] = fields
    row["compared_field_count"] = sum(
        1 for f in fields.values() if f["in_phase1_block"])
    row["match_count"] = sum(1 for f in fields.values() if f["match"] is True)
    row["mismatched_fields"] = sorted(
        name for name, f in fields.items() if f["match"] is False)
    row["all_match"] = not row["mismatched_fields"]
    row["receipt_sha256"] = record["receipt_sha256"]
    row["receipt_sha256_ok"] = block["receipt_sha256_ok"]
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--det-root", default=REPO,
        help="repository root holding artifacts/grm_det1/... (read-only)")
    parser.add_argument(
        "--p1-adjudication", required=True,
        help="path to lsr_p1_adjudication_31e5c89453f4aa42.json (read-only)")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    adjudication = load_json(args.p1_adjudication)
    probes = [replay_probe(probe, args.det_root)
              for probe in adjudication.get("probes") or ()]

    payload = {
        "schema": "grm.lsr_p2b.g2_replay.v1",
        "program": "LSR",
        "phase": "2B",
        "declares_in_det_envelope": False,
        "provenance_note": (
            "Own provenance. Reads DET1.5 lived receipts and the Phase 1 "
            "adjudication read-only; no re-serve, no model, no GPU."),
        "p1_adjudication": {
            "path": args.p1_adjudication,
            "sha256": sha256_file(args.p1_adjudication),
        },
        "det_root": args.det_root,
        "probe_count": len(probes),
        "probes_all_match": sum(1 for p in probes if p["all_match"]),
        "verdict": (
            "GREEN" if probes and all(p["all_match"] for p in probes)
            else "RED"),
        "closed_by_p2b_not_retro_provable": [
            {"field": name, "reason": reason}
            for name, reason in NOT_RETRO_PROVABLE
        ],
        "probes": probes,
    }
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)),
                    exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)

    for probe in probes:
        status = "MATCH" if probe["all_match"] else "MISMATCH"
        detail = ("" if probe["all_match"]
                  else json.dumps(probe.get("mismatched_fields")
                                  or probe.get("error")))
        print("%-30s %-9s %s" % (probe["probe_id"], status, detail))
    print("verdict: %s  (%d/%d probes match field-for-field)" % (
        payload["verdict"], payload["probes_all_match"], len(probes)))
    return 0 if payload["verdict"] == "GREEN" else 1


if __name__ == "__main__":
    raise SystemExit(main())
