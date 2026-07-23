#!/usr/bin/env python3
"""Assemble the registered P0/P1 gate table from bounded-run artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _without_wall_clock(value):
    if isinstance(value, dict):
        return {
            key: _without_wall_clock(item)
            for key, item in value.items()
            if key != "created_at"
        }
    if isinstance(value, list):
        return [_without_wall_clock(item) for item in value]
    return value


def _canonical_bytes(value) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")


def _repository_projection(session: Path) -> dict:
    repository = session / "repository"
    manifest = _without_wall_clock(_read_json(repository / "manifest.json"))
    arrays = []
    paths = [repository / "index.npz"] + sorted(
        (repository / "nodes").glob("*.npz"))
    for path in paths:
        with np.load(path) as packed:
            fields = []
            for key in sorted(packed.files):
                array = packed[key]
                fields.append({
                    "key": key,
                    "dtype": str(array.dtype),
                    "shape": [int(x) for x in array.shape],
                    "sha256": _sha(array.tobytes(order="C")),
                })
        arrays.append({"path": str(path.relative_to(repository)),
                       "fields": fields})
    projection = {"manifest": manifest, "arrays": arrays}
    projection["canonical_sha256"] = _sha(_canonical_bytes(projection))
    native = repository / "native" / "grm_store.bin"
    projection["native_checkpoint_sha256"] = _sha(native.read_bytes())
    return projection


def _transcript(session: Path) -> tuple[str, bytes]:
    payload = (session / "transcript.jsonl").read_bytes()
    return _sha(payload), payload


def _ledger_audit(session: Path, expected_turns: int) -> dict:
    receipts = sorted((session / "memory_ledger").glob("turn_*.json"))
    rows = _read_jsonl(session / "instrumentation.jsonl")
    valid = []
    kinds = {}
    for path in receipts:
        receipt = _read_json(path)
        contiguous = all(
            mutation.get("sequence") == idx
            for idx, mutation in enumerate(receipt.get("mutations", ())))
        hashes_valid = all(
            len(mutation.get("before_sha256", "")) == 64
            and len(mutation.get("after_sha256", "")) == 64
            and len(mutation.get("provenance", {}).get("source_sha256", "")) == 64
            and len(mutation.get("provenance", {}).get("decision_sha256", "")) == 64
            for mutation in receipt.get("mutations", ()))
        valid.append(
            receipt.get("schema") == "grm.memory_ledger.turn.v1"
            and receipt.get("turn_pipeline") == "three_pass"
            and receipt.get("pass") == 3
            and receipt.get("mutation_count") == len(receipt.get("mutations", ()))
            and contiguous and hashes_valid)
        for mutation in receipt.get("mutations", ()):
            kind = mutation.get("kind")
            kinds[kind] = kinds.get(kind, 0) + 1
    row_audits = [row.get("memory_ledger_audit", {}) for row in rows]
    supersession_rows = [
        row for row in rows
        if row.get("kind") == "supersede"
        and (row.get("correction_result") or {}).get("action") == "correct"
    ]
    supersession_receipted = any(
        mutation.get("kind") == "supersession"
        for path in receipts
        for mutation in _read_json(path).get("mutations", ()))
    return {
        "expected_turns": int(expected_turns),
        "receipt_files": len(receipts),
        "one_receipt_per_turn": len(receipts) == int(expected_turns),
        "all_receipts_schema_valid": all(valid) and bool(valid),
        "all_turn_audits_complete": (
            len(row_audits) == int(expected_turns)
            and all(audit.get("complete") is True for audit in row_audits)),
        "missing_target_count": sum(
            len(audit.get("missing_targets", ())) for audit in row_audits),
        "mutation_kind_counts": kinds,
        "supersession_command_inside_pass3": (
            bool(supersession_rows) and supersession_receipted),
    }


def _gate_line(gate: dict) -> str:
    details = " ".join(
        f"{key}={value}" for key, value in gate["headline"].items())
    return f"{gate['gate']} {gate['status']} {details}"


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-session", type=Path, required=True)
    parser.add_argument("--single-session", type=Path, required=True)
    parser.add_argument("--three-pass-session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv) -> int:
    args = parse_args(argv)
    head_sha, head_transcript = _transcript(args.head_session)
    single_sha, single_transcript = _transcript(args.single_session)
    three_sha, three_transcript = _transcript(args.three_pass_session)
    head_repo = _repository_projection(args.head_session)
    single_repo = _repository_projection(args.single_session)
    three_repo = _repository_projection(args.three_pass_session)

    single_timing = _read_json(args.single_session / "stage_timing.json")
    three_timing = _read_json(args.three_pass_session / "stage_timing.json")
    baseline_ms = float(single_timing["turn_wall_ms_total"])
    three_ms = float(three_timing["turn_wall_ms_total"])
    ratio = three_ms / baseline_ms
    overhead = float(three_timing["pass2_visible_memory_overhead_ms_max"])
    pass2_read_only = bool(three_timing["pass2_all_arena_read_only"])

    score_single = _read_json(args.single_session / "probe_scorecard.json")
    score_three = _read_json(args.three_pass_session / "probe_scorecard.json")
    ledger = _ledger_audit(
        args.three_pass_session, three_timing["frame"]["turns"])
    supersession_probes = [
        row for row in score_three.get("probes", ()) if row.get("old_value")]
    decisive_supersession_pass = (
        bool(supersession_probes)
        and all(row.get("pass") is True for row in supersession_probes))

    equiv_checks = {
        "single_transcript_byte_equal_to_head": single_transcript == head_transcript,
        "three_pass_transcript_byte_equal_to_single": (
            three_transcript == single_transcript),
        "single_canonical_arena_equal_to_head": (
            single_repo["canonical_sha256"] == head_repo["canonical_sha256"]),
        "three_pass_canonical_arena_equal_to_single": (
            three_repo["canonical_sha256"] == single_repo["canonical_sha256"]),
    }
    gates = [
        {
            "gate": "G-EQUIV",
            "status": "PASS" if all(equiv_checks.values()) else "RED",
            "headline": {
                "head_single_transcript_sha256": head_sha,
                "single_transcript_sha256": single_sha,
                "three_pass_transcript_sha256": three_sha,
                "canonical_arena_byte_equal": equiv_checks[
                    "three_pass_canonical_arena_equal_to_single"],
            },
            "checks": equiv_checks,
            "evidence": {
                "head_repository": head_repo,
                "single_repository": single_repo,
                "three_pass_repository": three_repo,
                "raw_native_checkpoint_equal_single_vs_three": (
                    single_repo["native_checkpoint_sha256"]
                    == three_repo["native_checkpoint_sha256"]),
                "canonicalization": (
                    "manifest and exact NPZ arrays; existing wall-clock "
                    "provenance created_at excluded by frozen receipt law"),
            },
        },
        {
            "gate": "G-SERVE",
            "status": "PASS" if (
                overhead <= 5.0 and ratio <= 1.25 and pass2_read_only) else "RED",
            "headline": {
                "pass2_visible_memory_overhead_ms_max": overhead,
                "overhead_limit_ms": 5.0,
                "total_latency_ratio": ratio,
                "ratio_limit": 1.25,
            },
            "checks": {
                "pass2_all_arena_read_only": pass2_read_only,
                "overhead_within_limit": overhead <= 5.0,
                "total_within_ratio": ratio <= 1.25,
            },
            "evidence": {
                "single_turn_wall_ms_total": baseline_ms,
                "three_pass_turn_wall_ms_total": three_ms,
            },
        },
        {
            "gate": "G-COMPACT",
            "status": "PASS" if (
                ledger["one_receipt_per_turn"]
                and ledger["all_receipts_schema_valid"]
                and ledger["all_turn_audits_complete"]
                and ledger["supersession_command_inside_pass3"]
                and decisive_supersession_pass
                and int(score_three["passed"]) >= int(score_single["passed"])
            ) else "RED",
            "headline": {
                "ledger_complete": ledger["all_turn_audits_complete"],
                "receipts": f"{ledger['receipt_files']}/{ledger['expected_turns']}",
                "decisive_supersession_probe_pass": decisive_supersession_pass,
                "recall_three_pass": f"{score_three['passed']}/{score_three['total']}",
                "recall_single_inline": f"{score_single['passed']}/{score_single['total']}",
            },
            "checks": {
                **ledger,
                "recall_non_regression_vs_inline_management": (
                    int(score_three["passed"]) >= int(score_single["passed"])),
                "decisive_supersession_probe_pass": decisive_supersession_pass,
            },
        },
    ]
    artifact = {
        "schema": "grm.three_pass.p0_p1.gates.v1",
        "sessions": {
            "head": str(args.head_session),
            "single": str(args.single_session),
            "three_pass": str(args.three_pass_session),
        },
        "gates": gates,
        "gate_lines": [_gate_line(gate) for gate in gates],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    for line in artifact["gate_lines"]:
        print(line)
    print(f"artifact={args.output}")
    return 0 if all(gate["status"] == "PASS" for gate in gates) else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
