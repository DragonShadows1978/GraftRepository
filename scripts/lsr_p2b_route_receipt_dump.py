#!/usr/bin/env python3
"""LSR-P2B — print the Phase-1 evidence block from persisted route receipts.

LSR Phase 1 (docs/LSR_ADDENDUM_1.md, principle 2) had to RECONSTRUCT every
routing decision from DET1.5 snapshot manifests because route receipts were
not persisted at serving time.  P2B persists them.  This script is the proof:
given a session directory or a single ledger/receipt file, it prints, per
turn, the same evidence fields Phase 1 reconstructed -- using nothing but the
persisted ``grm.route_receipt.v1`` records.

Inputs accepted:
  * a session directory (``memory_ledger/turn_*.json`` and/or
    ``instrumentation.jsonl``),
  * a memory-ledger turn file (``grm.memory_ledger.turn.v1`` with the
    ``route_receipt`` sidecar),
  * a route-receipt file (``grm.route_receipt.v1``),
  * an ``instrumentation.jsonl`` file,
  * a route-receipt ledger directory written by
    ``GraftRepository.write_route_receipt``.

CPU only.  Reads; never writes outside ``--json-out`` if given.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from core.grm_three_pass import (  # noqa: E402
    ROUTE_RECEIPT_SCHEMA,
    route_receipt_sha256,
)

# The Phase 1 adjudication evidence block, field for field
# (artifacts/lsr_p1/lsr_p1_adjudication_31e5c89453f4aa42.json).
EVIDENCE_FIELDS = (
    "admission.ranking",
    "admission.identified_candidates",
    "admission.identifier_tokens",
    "admission.policy_branch",
    "admission.rank_plan",
    "admission.current_planned",
    "admission.current_fitted",
    "admission.current_dropped",
    "admission.final_mounts",
    "admission.rule_sha256",
    "arena.cur_mount_n",
    "arena.width",
    "lived_repository_size_at_probe",
    "ranking_length",
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def receipts_from_object(obj):
    """Pull every route receipt out of one loaded JSON object."""
    if not isinstance(obj, dict):
        return []
    if obj.get("schema") == ROUTE_RECEIPT_SCHEMA:
        return [obj]
    out = []
    for key in ("route_receipt", "route_receipt_record"):
        nested = obj.get(key)
        if (isinstance(nested, dict)
                and nested.get("schema") == ROUTE_RECEIPT_SCHEMA):
            out.append(nested)
    info = obj.get("info")
    if isinstance(info, dict):
        out.extend(receipts_from_object(info))
    return out


def collect_receipts(target):
    """Gather route receipts from a file or directory, ordered by turn."""
    receipts = []
    seen = set()

    def take(rows):
        for row in rows:
            key = str(row.get("receipt_sha256") or id(row))
            if key in seen:
                continue
            seen.add(key)
            receipts.append(row)

    def eat_file(path):
        if path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        take(receipts_from_object(json.loads(line)))
                    except json.JSONDecodeError:
                        continue
            return
        if not path.endswith(".json"):
            return
        try:
            take(receipts_from_object(load_json(path)))
        except (json.JSONDecodeError, OSError):
            return

    if os.path.isfile(target):
        eat_file(target)
    elif os.path.isdir(target):
        for base, _dirs, files in os.walk(target):
            for name in sorted(files):
                eat_file(os.path.join(base, name))
    else:
        raise SystemExit("no such session dir or file: %s" % target)

    def order(row):
        try:
            return (0, int(row.get("turn_id", 0)))
        except (TypeError, ValueError):
            return (1, str(row.get("turn_id", "")))

    receipts.sort(key=order)
    return receipts


def evidence_block(receipt):
    """Phase-1-style evidence, derived from the persisted receipt ALONE."""
    route = receipt.get("route") or {}
    admission = receipt.get("admission") or {}
    fit = receipt.get("fit") or {}
    provenance = receipt.get("provenance") or {}
    ranking = list(route.get("ranking_ids") or ())
    return {
        "admission.ranking": ranking,
        "admission.identified_candidates": list(
            admission.get("identified_candidates") or ()),
        "admission.identifier_tokens": admission.get("identifier_tokens"),
        "admission.policy_branch": admission.get("policy_branch"),
        "admission.rank_plan": list(admission.get("rank_plan") or ()),
        "admission.current_planned": list(fit.get("planned") or ()),
        "admission.current_fitted": list(fit.get("seated") or ()),
        "admission.current_dropped": list(fit.get("dropped") or ()),
        "admission.final_mounts": list(fit.get("final_mounts") or ()),
        "admission.rule_sha256": admission.get("rule_sha256"),
        "arena.cur_mount_n": fit.get("cur_mount_n"),
        "arena.width": fit.get("width"),
        "lived_repository_size_at_probe": provenance.get(
            "repository_size_at_probe"),
        "ranking_length": route.get("ranking_length", len(ranking)),
    }


def turn_block(receipt):
    route = receipt.get("route") or {}
    fit = receipt.get("fit") or {}
    provenance = receipt.get("provenance") or {}
    recomputed = route_receipt_sha256(receipt)
    return {
        "session_id": receipt.get("session_id"),
        "turn_id": receipt.get("turn_id"),
        "turn_kind": receipt.get("turn_kind"),
        "serving_path": receipt.get("serving_path"),
        "receipt_sha256": receipt.get("receipt_sha256"),
        "receipt_sha256_recomputed": recomputed,
        "receipt_sha256_ok": receipt.get("receipt_sha256") == recomputed,
        "evidence": evidence_block(receipt),
        "route": {
            "backend": route.get("backend"),
            "route_limit": route.get("route_limit"),
            "candidate_count": route.get("candidate_count"),
            "ranking_scores": route.get("ranking_scores"),
            "excluded_live_ids": route.get("excluded_live_ids"),
            "excluded_recency_ids": route.get("excluded_recency_ids"),
            "fallbacks": route.get("fallbacks"),
        },
        "fit": {
            "trips": fit.get("trips"),
            "trip_count": fit.get("trip_count"),
            "no_mount_fit": fit.get("no_mount_fit"),
            "info_pass_through": fit.get("info_pass_through"),
        },
        "provenance": provenance,
    }


def render(blocks):
    lines = []
    for block in blocks:
        lines.append("=" * 72)
        lines.append("turn %s  kind=%s  session=%s" % (
            block["turn_id"], block["turn_kind"], block["session_id"]))
        lines.append("  serving_path: %s" % block["serving_path"])
        lines.append("  receipt_sha256: %s (recomputes: %s)" % (
            block["receipt_sha256"], block["receipt_sha256_ok"]))
        lines.append("  -- Phase 1 evidence block, from this receipt alone --")
        for field in EVIDENCE_FIELDS:
            lines.append("    %s = %s" % (
                field, json.dumps(block["evidence"][field])))
        for section in ("route", "fit", "provenance"):
            lines.append("  -- %s --" % section)
            for key, value in block[section].items():
                lines.append("    %s = %s" % (key, json.dumps(value)))
    if not blocks:
        lines.append("no grm.route_receipt.v1 records found")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        help="session directory, memory-ledger turn file, route-receipt file, "
             "instrumentation.jsonl, or route-receipt ledger directory")
    parser.add_argument("--json", action="store_true",
                        help="emit machine-readable JSON instead of text")
    parser.add_argument("--json-out", default=None,
                        help="also write the JSON payload to this path")
    args = parser.parse_args(argv)

    receipts = collect_receipts(args.target)
    blocks = [turn_block(receipt) for receipt in receipts]
    payload = {
        "schema": "grm.lsr_p2b.route_receipt_dump.v1",
        "target": args.target,
        "receipt_schema": ROUTE_RECEIPT_SCHEMA,
        "turn_count": len(blocks),
        "all_sha256_verify": all(b["receipt_sha256_ok"] for b in blocks),
        "turns": blocks,
    }
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)),
                    exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(render(blocks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
