#!/usr/bin/env python3
"""LSR Phase 1 — stage archaeology harvester (CPU only, read-only).

Reconstructs, for every lived DET1.5 supersession probe serve on disk, the
full routing decision as the production code recorded it:

  candidate base (admission.ranking universe)
    -> identifier-hit computation (admission.identifier_tokens /
       identified_candidates)
    -> A-DEC branch (admission.policy_branch, frozen rule c304609f...)
    -> rank_plan / ladder attempts
    -> L2 M5-edge resolution (arena.revision_resolution + the
       _revision_mount_heads contract)
    -> final mounts (admission.final_mounts, linked_answer.probe_mounts)

Reads only persisted receipts.  Writes nothing outside artifacts/lsr_p1/.
Emits a content-addressed harvest artifact with its own provenance (NOT a
DET envelope).
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARDS = os.path.join(
    REPO, "artifacts", "grm_det1", "run_20260831T160525Z_2", "det1_4",
    "campaign", "det1_7", "plant_registration", "shards")
OUTDIR = os.path.join(REPO, "artifacts", "lsr_p1")

ADM_KEYS = (
    "admission.ranking", "admission.identified_candidates",
    "admission.identifier_tokens", "admission.policy_branch",
    "admission.rank_plan", "admission.final_mounts",
    "admission.authoritative_mounts", "admission.current_planned",
    "admission.current_fitted", "admission.current_dropped",
    "admission.ladder_attempts", "admission.route_margin_1_2",
    "admission.margin_threshold", "admission.rule_sha256",
    "admission.route_backend", "admission.selection_state",
    "admission.current_clean_room", "admission.current_attempt_ordinal",
)
ARENA_KEYS = (
    "arena.cur_mounts", "arena.topk", "arena.revision_resolution",
    "arena.decisive_admission", "arena.route_backend", "arena.route_layer",
    "arena.width", "arena.cur_mount_n", "arena.n_sink", "arena.live_shift",
    "arena.live_turns",
)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def harvest_one(path):
    man = load_manifest(path)
    state = man.get("state") or {}
    prov = man.get("provenance") or {}
    linked = man.get("linked_answer") or {}
    rel = os.path.relpath(path, REPO)
    parts = rel.split(os.sep)
    row = {
        "manifest_path": rel,
        "manifest_sha256": sha256_file(path),
        "shard": parts[-6],
        "attempt_dir": parts[-5],
        "fixture_id": parts[-3],
        "rung": parts[-2],
        "phase": man.get("phase"),
        "label": man.get("label"),
        "arm": prov.get("arm"),
        "probe_id": prov.get("probe_id"),
        "provenance_fixture_id": prov.get("fixture_id"),
        "fixture_sha256": prov.get("fixture_sha256"),
        "attempt_ordinal": prov.get("attempt_ordinal"),
        "max_trips": prov.get("max_trips"),
        "topk_declared": prov.get("topk"),
        "probe_ladder": prov.get("probe_ladder"),
        "probe_driver": prov.get("probe_driver"),
        "protocol": prov.get("protocol"),
        "defer_memory": prov.get("defer_memory"),
        "process_instance_sha256": prov.get("process_instance_sha256"),
        "capture_boundary": prov.get("capture_boundary"),
        "run_id": prov.get("run_id"),
    }
    for key in ADM_KEYS + ARENA_KEYS:
        row[key] = state.get(key)
    for key in ("probe_answer", "probe_answer_correct", "probe_mounts",
                "attempt_answer", "attempt_answer_correct", "attempt_mounts",
                "probe_refusal", "probe_selected_attempt"):
        row["linked." + key] = linked.get(key)
    return row


def main():
    pattern = os.path.join(SHARDS, "*", "attempt_*", "snapshots", "*",
                           "rung_*", "manifest.json")
    files = sorted(glob.glob(pattern))
    rows = [harvest_one(p) for p in files]
    payload = {
        "schema": "grm.lsr_p1.stage_harvest.v1",
        "program": "LSR",
        "phase": "1",
        "declares_in_det_envelope": False,
        "provenance_note": (
            "Own provenance. Reads DET1.5 lived receipts read-only; "
            "does not declare inside any DET envelope."),
        "source_root": os.path.relpath(SHARDS, REPO),
        "manifest_count": len(rows),
        "harvester": {
            "path": "scripts/lsr_p1_harvest.py",
            "sha256": sha256_file(os.path.abspath(__file__)),
        },
        "rows": rows,
    }
    body = json.dumps(payload, indent=1, sort_keys=True)
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, "lsr_p1_harvest_%s.json" % digest[:16])
    if os.path.exists(out):
        print("EXISTS (append-only, identical content):", out)
    else:
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(body)
        print("WROTE:", out)
    print("manifests:", len(rows), "sha256:", digest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
