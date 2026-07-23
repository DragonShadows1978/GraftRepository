#!/usr/bin/env python3
"""Assemble the registered GRM3P-P2 F1 gate table from session artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.grm_three_pass_gate import (
    _ledger_audit,
    _read_json,
    _repository_projection,
    _transcript,
)


COMMITTED_TRANSCRIPT_SHA256 = (
    "68da84b8824eb79a14f65a692a602215bc53d84232cd041002410f51657bba4f"
)


def _working_sets(session: Path) -> list[dict]:
    return [
        _read_json(path)
        for path in sorted(
            (session / "working_set").glob("turn_*/working_set.json"))
    ]


def _gate_line(gate: dict) -> str:
    details = " ".join(
        f"{key}={value}" for key, value in gate["headline"].items())
    return f"{gate['gate']} {gate['status']} {details}"


def _frame_checks(session: Path, pipeline: str) -> dict:
    config = _read_json(session / "run_config.json")
    return {
        "mode_smoke": config.get("mode") == "smoke",
        "restart_after_5": config.get("restart_after") == 5,
        "resume_exercised": (session / "restart.json").exists(),
        "live_turns_default_1": config.get("live_turns") == 1,
        "arena_width_default_96": config.get("arena_width") == 96,
        "topk_default_3": config.get("topk") == 3,
        "ngen_default_24": config.get("ngen") == 24,
        "max_trips_default_1": config.get("max_trips") == 1,
        "turn_pipeline": config.get("turn_pipeline") == pipeline,
        "cuda_env_1": (
            config.get("env", {}).get("GRM_GQA_CUDA_ROUTE") == "1"),
        "storage_env_8": (
            config.get("env", {}).get("GRM_GRAFT_STORAGE_BITS") == "8"),
        "gpu_idle_check_skipped": (
            config.get("gpu_idle_check", {}).get("idle") == "skipped"),
    }


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-session", type=Path, required=True)
    parser.add_argument("--single-session", type=Path, required=True)
    parser.add_argument("--three-pass-session", type=Path, required=True)
    parser.add_argument("--baseline-timing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv) -> int:
    args = parse_args(argv)
    reference_sha, reference_transcript = _transcript(args.reference_session)
    single_sha, single_transcript = _transcript(args.single_session)
    three_sha, three_transcript = _transcript(args.three_pass_session)
    reference_repo = _repository_projection(args.reference_session)
    single_repo = _repository_projection(args.single_session)
    three_repo = _repository_projection(args.three_pass_session)
    single_frame = _frame_checks(args.single_session, "single")
    three_frame = _frame_checks(args.three_pass_session, "three_pass")

    timing = _read_json(args.three_pass_session / "stage_timing.json")
    baseline = _read_json(args.baseline_timing)
    baseline_ms = float(baseline["turn_wall_ms_total"])
    three_ms = float(timing["turn_wall_ms_total"])
    ratio = three_ms / baseline_ms
    overhead = float(timing["pass2_visible_memory_overhead_ms_max"])
    pass2_read_only = bool(timing["pass2_all_arena_read_only"])

    working_sets = _working_sets(args.three_pass_session)
    routed = [row for row in working_sets if row.get("routed")]
    probe_sets = [row for row in working_sets if row.get("event_kind") == "probe"]
    route_all_cuda = (
        bool(routed)
        and all(row.get("route_backend") == "cuda" for row in routed)
    )
    route_all_python_equal = (
        bool(routed)
        and all(
            row.get("python_backend") == "python"
            and row.get("ranking_ids_byte_equal") is True
            and row.get("ranking_ids") == row.get("python_ranking_ids")
            for row in routed
        )
    )
    direct_recall = sum(
        row.get("direct_route_recall_at_3") is True for row in probe_sets)
    staged_recall = sum(
        row.get("source_present_in_staged_set") is True for row in probe_sets)
    source_present_all = (
        bool(probe_sets)
        and all(
            row.get("source_present_in_staged_set") is True
            for row in probe_sets)
    )
    l2_misses = sum(
        int(row.get("l1", {}).get("l2_miss_count", 0))
        for row in working_sets)
    step2_page_ins = sum(
        int(row.get("step_io", {}).get("steps", {})
            .get("2_inference", {}).get("page_in_count", 0))
        for row in working_sets)
    step2_uploads = sum(
        int(row.get("step_io", {}).get("steps", {})
            .get("2_inference", {}).get("upload_count", 0))
        for row in working_sets)

    score_single = _read_json(args.single_session / "probe_scorecard.json")
    score_three = _read_json(args.three_pass_session / "probe_scorecard.json")
    ledger = _ledger_audit(
        args.three_pass_session, timing["frame"]["turns"])
    supersession_probes = [
        row for row in score_three.get("probes", ()) if row.get("old_value")]
    decisive_supersession_pass = (
        bool(supersession_probes)
        and all(row.get("pass") is True for row in supersession_probes)
    )

    equiv_checks = {
        "reference_sha_is_committed": (
            reference_sha == COMMITTED_TRANSCRIPT_SHA256),
        "single_transcript_sha_is_committed": (
            single_sha == COMMITTED_TRANSCRIPT_SHA256),
        "single_transcript_byte_equal_to_reference": (
            single_transcript == reference_transcript),
        "three_pass_transcript_byte_equal_to_single": (
            three_transcript == single_transcript),
        "single_canonical_arena_equal_to_reference": (
            single_repo["canonical_sha256"]
            == reference_repo["canonical_sha256"]),
        "three_pass_canonical_arena_equal_to_single": (
            three_repo["canonical_sha256"] == single_repo["canonical_sha256"]),
        "single_frame_exact": all(single_frame.values()),
        "three_pass_frame_exact": all(three_frame.values()),
    }
    gates = [
        {
            "gate": "G-EQUIV-P2",
            "status": "PASS" if all(equiv_checks.values()) else "RED",
            "headline": {
                "committed_transcript_sha256": COMMITTED_TRANSCRIPT_SHA256,
                "single_transcript_sha256": single_sha,
                "three_pass_transcript_sha256": three_sha,
                "canonical_arena_byte_equal": equiv_checks[
                    "three_pass_canonical_arena_equal_to_single"],
            },
            "checks": equiv_checks,
            "evidence": {
                "single_frame": single_frame,
                "three_pass_frame": three_frame,
                "reference_repository": reference_repo,
                "single_repository": single_repo,
                "three_pass_repository": three_repo,
            },
        },
        {
            "gate": "G-ROUTE",
            "status": "PASS" if (
                len(working_sets) == 10
                and route_all_cuda
                and route_all_python_equal
            ) else "RED",
            "headline": {
                "cuda_routed_turns": (
                    f"{sum(row.get('route_backend') == 'cuda' for row in routed)}"
                    f"/{len(routed)}"),
                "python_ranking_byte_equal_turns": (
                    f"{sum(row.get('ranking_ids_byte_equal') is True for row in routed)}"
                    f"/{len(routed)}"),
                "working_set_receipts": f"{len(working_sets)}/10",
            },
            "checks": {
                "non_vacuous_routed_turns": bool(routed),
                "route_backend_cuda_every_routed_turn": route_all_cuda,
                "ranking_ids_equal_python_every_routed_turn": (
                    route_all_python_equal),
                "one_working_set_receipt_per_turn": len(working_sets) == 10,
            },
        },
        {
            "gate": "G-PREP",
            "status": "PASS" if (
                len(probe_sets) == 2
                and direct_recall == 2
                and staged_recall == direct_recall
                and source_present_all
                and l2_misses == 0
                and step2_page_ins == 0
                and step2_uploads == 0
            ) else "RED",
            "headline": {
                "staged_set_recall": f"{staged_recall}/{len(probe_sets)}",
                "direct_route_recall": f"{direct_recall}/{len(probe_sets)}",
                "fact_source_present": (
                    f"{sum(row.get('source_present_in_staged_set') is True for row in probe_sets)}"
                    f"/{len(probe_sets)}"),
                "l1_misses": l2_misses,
                "step2_page_ins": step2_page_ins,
                "step2_uploads": step2_uploads,
            },
            "checks": {
                "two_probe_fixture": len(probe_sets) == 2,
                "staged_recall_equals_direct": staged_recall == direct_recall,
                "direct_recall_2_of_2": direct_recall == 2,
                "source_present_every_probe": source_present_all,
                "l1_misses_zero": l2_misses == 0,
                "step2_page_ins_zero": step2_page_ins == 0,
                "step2_uploads_zero": step2_uploads == 0,
            },
        },
        {
            "gate": "G-SERVE-P2",
            "status": "PASS" if (
                overhead <= 5.0 and ratio <= 1.25 and pass2_read_only
            ) else "RED",
            "headline": {
                "pass2_visible_memory_overhead_ms_max": overhead,
                "overhead_limit_ms": 5.0,
                "total_latency_ratio_vs_p0": ratio,
                "ratio_limit": 1.25,
                "route_wall_ms_mean_per_turn": (
                    timing["stages"]["route"]["wall_ms_mean_per_turn"]),
            },
            "checks": {
                "pass2_all_arena_read_only": pass2_read_only,
                "overhead_within_limit": overhead <= 5.0,
                "total_within_ratio": ratio <= 1.25,
            },
            "evidence": {
                "p0_baseline_turn_wall_ms_total": baseline_ms,
                "three_pass_turn_wall_ms_total": three_ms,
                "route_stage": timing["stages"]["route"],
                "prep_stage": timing["stages"]["prep"],
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
                and int(score_three["passed"]) == 2
                and int(score_single["passed"]) == 2
            ) else "RED",
            "headline": {
                "ledger_complete": ledger["all_turn_audits_complete"],
                "receipts": (
                    f"{ledger['receipt_files']}/{ledger['expected_turns']}"),
                "decisive_supersession_probe_pass": (
                    decisive_supersession_pass),
                "recall_three_pass": (
                    f"{score_three['passed']}/{score_three['total']}"),
                "recall_single_inline": (
                    f"{score_single['passed']}/{score_single['total']}"),
            },
            "checks": ledger,
        },
    ]
    artifact = {
        "schema": "grm.three_pass.p2.gates.v1",
        "frame": "F1",
        "sessions": {
            "reference": str(args.reference_session),
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
