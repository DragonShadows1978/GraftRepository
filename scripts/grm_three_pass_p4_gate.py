#!/usr/bin/env python3
"""Assemble the registered GRM3P-P4 composed-E2E gate tables from artifacts.

Two gate families, one per registered frame:

- F-FULL  -> G-E2E-EQ, G-E2E-SERVE          (single vs three_pass equivalence)
- F-COLD  -> G-E2E-RECALL, G-E2E-IO, G-E2E-LEDGER
                                            (cold page-in leg, both pipelines)

The assembler is pure instrumentation: it reads committed session artifacts
and computes comparative pass/fail. No thresholds are invented here; every
bound is the one registered in orders/GRM3P_P4.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.grm_three_pass_gate import (  # noqa: E402
    _ledger_audit,
    _read_json,
    _repository_projection,
    _transcript,
)


def _gate_line(gate: dict) -> str:
    details = " ".join(
        f"{key}={value}" for key, value in gate["headline"].items())
    return f"{gate['gate']} {gate['status']} {details}"


def _working_sets(session: Path) -> list[dict]:
    return [
        _read_json(path)
        for path in sorted(
            (session / "working_set").glob("turn_*/working_set.json"))
    ]


def _scorecard(session: Path) -> dict:
    return _read_json(session / "probe_scorecard.json")


def _step_io_totals(working_sets: list[dict]) -> dict:
    steps = ("1_prep", "2_inference", "3_cleanup")
    out = {step: {"page_ins": 0, "uploads": 0} for step in steps}
    for row in working_sets:
        step_io = (row.get("step_io") or {}).get("steps", {})
        for step in steps:
            leg = step_io.get(step, {})
            out[step]["page_ins"] += int(leg.get("page_in_count", 0))
            out[step]["uploads"] += int(leg.get("upload_count", 0))
    return out


def _per_probe_rows(scorecard: dict) -> list[dict]:
    rows = []
    for probe in scorecard.get("probes", ()):
        rows.append({
            "turn": probe.get("turn"),
            "fact_id": probe.get("fact_id"),
            "expected": probe.get("expected"),
            "old_value": probe.get("old_value"),
            "pass": bool(probe.get("pass")),
            "contains_expected": bool(probe.get("contains_expected")),
            "contains_stale": bool(probe.get("contains_stale")),
            "source_rank": (probe.get("route_ranking") or {}).get(
                "source_rank"),
            "route_backend": probe.get("route_backend"),
        })
    return rows


# --------------------------------------------------------------- F-FULL
def _f_full_gates(single: Path, three: Path) -> list[dict]:
    single_sha, single_bytes = _transcript(single)
    three_sha, three_bytes = _transcript(three)
    single_repo = _repository_projection(single)
    three_repo = _repository_projection(three)

    transcripts_equal = single_bytes == three_bytes
    arena_equal = (
        single_repo["canonical_sha256"] == three_repo["canonical_sha256"])

    single_timing = _read_json(single / "stage_timing.json")
    three_timing = _read_json(three / "stage_timing.json")
    single_ms = float(single_timing["turn_wall_ms_total"])
    three_ms = float(three_timing["turn_wall_ms_total"])
    ratio = three_ms / single_ms if single_ms else float("inf")
    pass2_read_only = bool(three_timing.get("pass2_all_arena_read_only"))

    eq = {
        "gate": "G-E2E-EQ",
        "status": "PASS" if (transcripts_equal and arena_equal) else "RED",
        "headline": {
            "single_transcript_sha256": single_sha,
            "three_pass_transcript_sha256": three_sha,
            "transcripts_byte_identical": transcripts_equal,
            "canonical_arena_byte_equal": arena_equal,
        },
        "checks": {
            "transcripts_byte_identical": transcripts_equal,
            "canonical_arena_byte_equal": arena_equal,
        },
        "evidence": {
            "single_repository": single_repo,
            "three_pass_repository": three_repo,
        },
    }
    serve = {
        "gate": "G-E2E-SERVE",
        "status": "PASS" if (ratio <= 1.25 and pass2_read_only) else "RED",
        "headline": {
            "single_turn_wall_ms_total": single_ms,
            "three_pass_turn_wall_ms_total": three_ms,
            "total_latency_ratio": ratio,
            "ratio_limit": 1.25,
        },
        "checks": {
            "total_within_ratio": ratio <= 1.25,
            "pass2_all_arena_read_only": pass2_read_only,
        },
    }
    return [eq, serve]


# --------------------------------------------------------------- F-COLD
def _f_cold_gates(single: Path, three: Path, expected_turns: int) -> list[dict]:
    score_single = _scorecard(single)
    score_three = _scorecard(three)
    single_pass = int(score_single["passed"])
    three_pass = int(score_three["passed"])
    total = int(score_three["total"])

    three_ws = _working_sets(three)
    io = _step_io_totals(three_ws)

    per_turn_present = all(
        row.get("step_io") is not None for row in three_ws)
    step2_clean = (io["2_inference"]["page_ins"] == 0
                   and io["2_inference"]["uploads"] == 0)
    positive_page_ins_steps13 = (
        io["1_prep"]["page_ins"] + io["3_cleanup"]["page_ins"])

    ledger = _ledger_audit(three, int(expected_turns))

    recall = {
        "gate": "G-E2E-RECALL",
        "status": "PASS" if (three_pass >= single_pass) else "RED",
        "headline": {
            "recall_three_pass": f"{three_pass}/{total}",
            "recall_single": f"{single_pass}/{int(score_single['total'])}",
            "three_pass_ge_single": three_pass >= single_pass,
        },
        "checks": {"three_pass_recall_ge_single": three_pass >= single_pass},
        "evidence": {
            "three_pass_per_probe": _per_probe_rows(score_three),
            "single_per_probe": _per_probe_rows(score_single),
        },
    }
    io_gate = {
        "gate": "G-E2E-IO",
        "status": "PASS" if (step2_clean and per_turn_present) else "RED",
        "headline": {
            "step1_page_ins": io["1_prep"]["page_ins"],
            "step2_page_ins": io["2_inference"]["page_ins"],
            "step3_page_ins": io["3_cleanup"]["page_ins"],
            "step2_uploads": io["2_inference"]["uploads"],
            "positive_page_ins_in_steps_1_or_3": positive_page_ins_steps13,
            "working_set_receipts": f"{len(three_ws)}/{expected_turns}",
        },
        "checks": {
            "step2_page_ins_zero": io["2_inference"]["page_ins"] == 0,
            "step2_uploads_zero": io["2_inference"]["uploads"] == 0,
            "step_io_present_every_turn": per_turn_present,
            "working_set_receipt_per_turn": len(three_ws) == int(expected_turns),
        },
        "evidence": {"step_io_totals": io},
    }
    ledger_gate = {
        "gate": "G-E2E-LEDGER",
        "status": "PASS" if (
            ledger["one_receipt_per_turn"]
            and ledger["all_receipts_schema_valid"]
            and ledger["all_turn_audits_complete"]
        ) else "RED",
        "headline": {
            "ledger_complete": ledger["all_turn_audits_complete"],
            "receipts": f"{ledger['receipt_files']}/{ledger['expected_turns']}",
            "all_receipts_schema_valid": ledger["all_receipts_schema_valid"],
        },
        "checks": ledger,
    }
    return [recall, io_gate, ledger_gate]


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frame", choices=("full", "cold"), required=True)
    p.add_argument("--single-session", type=Path, required=True)
    p.add_argument("--three-pass-session", type=Path, required=True)
    p.add_argument("--expected-turns", type=int, default=None)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args(argv)


def main(argv) -> int:
    args = parse_args(argv)
    if args.frame == "full":
        gates = _f_full_gates(args.single_session, args.three_pass_session)
        frame = "F-FULL"
    else:
        expected = args.expected_turns
        if expected is None:
            expected = len(
                _read_json(
                    args.three_pass_session / "run_config.json")["script"])
        gates = _f_cold_gates(
            args.single_session, args.three_pass_session, int(expected))
        frame = "F-COLD"
    artifact = {
        "schema": "grm.three_pass.p4.gates.v1",
        "frame": frame,
        "sessions": {
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
