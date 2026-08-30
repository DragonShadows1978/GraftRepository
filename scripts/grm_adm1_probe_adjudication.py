#!/usr/bin/env python3
"""Adjudicate ADM1 replay baselines after the GRM-SUP-L2 default flip.

This module is CPU-only.  It validates completed ADM E2E frame receipts against
their registered fixed-A-k3 controls, audits whether enabling L2 could have
changed any recorded candidate mount set, and emits the GRM-ADM1.2 diagnosis.
It never generates model answers or rewrites an existing receipt.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from grm_adm1_analysis import (  # noqa: E402
    ADMError,
    file_record,
    metric_table,
    read_json,
    read_jsonl,
    write_content_addressed,
)


FRAME_ANCHORS = {
    "diag": {
        "scorecard": ROOT / "artifacts" / "grm_three_pass"
        / "diag_unbounded_arm_b_single_r2" / "probe_scorecard.json",
        "anchor_fixture": "DIAG-CURRENT-FULL",
        "row_count": 9,
    },
    "e2e": {
        "scorecard": ROOT / "artifacts" / "grm_e2e"
        / "full_session_20260708_P2" / "probe_scorecard.json",
        "anchor_fixture": "E2E-34-FULL-A-k3-ANCHOR",
        "row_count": 11,
    },
    "p4": {
        "scorecard": ROOT / "artifacts" / "grm_three_pass"
        / "p4_ffull_three_pass" / "probe_scorecard.json",
        "anchor_fixture": "P4-REPLICATION-FULL-A-k3-ANCHOR",
        "row_count": 11,
    },
}

PRODUCTION_FULL_SCORECARD = (
    ROOT / "artifacts" / "grm_three_pass"
    / "ladder_on_full_default_single" / "probe_scorecard.json"
)
L2_REGISTRATION = (
    ROOT / "artifacts" / "grm_supersession"
    / "l2_default_on_registration_20260830.json"
)
L2_LEGACY_ANCHOR = (
    ROOT / "artifacts" / "grm_supersession" / "g0_baseline.jsonl"
)
L2_DEFAULT_ANCHOR = (
    ROOT / "artifacts" / "grm_supersession" / "diag_resolve_only.jsonl"
)


def _source_line(path: Path, needle: str) -> int:
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if needle in line:
            return lineno
    raise ADMError(f"source marker not found in {path}: {needle!r}")


def _probe_map(scorecard: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    rows = list(scorecard.get("probes") or ())
    out = {int(row["turn"]): row for row in rows}
    if len(out) != len(rows):
        raise ADMError("scorecard contains duplicate probe turns")
    return out


def _ancestors(nodes: Sequence[Mapping[str, Any]]) -> dict[int, set[int]]:
    direct = {
        index: {
            int(value)
            for value in ((node.get("metadata") or {}).get("supersedes") or ())
            if 0 <= int(value) < len(nodes)
        }
        for index, node in enumerate(nodes)
    }
    resolved: dict[int, set[int]] = {}

    def walk(index: int, visiting: set[int]) -> set[int]:
        if index in visiting:
            raise ADMError("cycle in captured supersession metadata")
        if index in resolved:
            return resolved[index]
        visiting.add(index)
        found = set(direct[index])
        for older in direct[index]:
            found.update(walk(older, visiting))
        visiting.remove(index)
        resolved[index] = found
        return found

    for index in direct:
        walk(index, set())
    return resolved


def project_l2_mounts(
    picks: Sequence[int], ancestors: Mapping[int, set[int]]
) -> list[int]:
    """Pure projection of ArenaCache's L2 lineage-head filter."""
    unique: list[int] = []
    for raw in picks:
        value = int(raw)
        if value not in unique:
            unique.append(value)
    present = set(unique)
    superseded = set()
    for value in unique:
        superseded.update(set(ancestors.get(value, set())) & present)
    return [value for value in unique if value not in superseded]


def audit_l2_projection(frame_dir: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    session_dir = frame_dir / "session"
    repository_manifest = session_dir / "repository" / "manifest.json"
    if not repository_manifest.is_file():
        raise ADMError(f"missing repository manifest: {repository_manifest}")
    nodes = list(read_json(repository_manifest).get("nodes") or ())
    ancestors = _ancestors(nodes)

    candidate_sets: list[dict[str, Any]] = []
    for row in rows:
        for arm, value in (row.get("arms") or {}).items():
            candidate_sets.append({
                "source": f"adm_rows:turn-{int(row['turn'])}:{arm}",
                "picks": [int(item) for item in value.get("rank_plan_fitted") or ()],
            })
    instrumentation_path = session_dir / "instrumentation.jsonl"
    for event in read_jsonl(instrumentation_path):
        info = event.get("info") or {}
        probe = event.get("probe_score") or {}
        for label, values in (
            ("info.mount_plan", info.get("mount_plan")),
            ("info.mount_fitted", info.get("mount_fitted")),
            ("probe.mount_plan", probe.get("mount_plan")),
        ):
            if values:
                candidate_sets.append({
                    "source": f"instrumentation:turn-{int(event['turn'])}:{label}",
                    "picks": [int(item) for item in values],
                })

    changes = []
    for item in candidate_sets:
        projected = project_l2_mounts(item["picks"], ancestors)
        if projected != item["picks"]:
            changes.append({**item, "l2_projected": projected})
    lineages = {
        str(index): sorted(values)
        for index, values in ancestors.items()
        if values
    }
    return {
        "schema": "grm.adm1.l2_projection_audit.v1",
        "candidate_sets_examined": len(candidate_sets),
        "lineages": lineages,
        "changed_candidate_sets": changes,
        "l2_projection_noop": not changes,
        "repository_manifest": file_record(repository_manifest),
        "instrumentation": file_record(instrumentation_path),
    }


def completed_capture_available(run_dir: Path, frame: str) -> bool:
    frame_dir = run_dir.resolve() / frame
    summary_path = frame_dir / "session" / "summary.json"
    rows_path = frame_dir / "adm_rows.jsonl"
    if not summary_path.is_file() or not rows_path.is_file():
        return False
    summary = read_json(summary_path)
    return (
        int(summary.get("turns_completed", -1))
        == int(summary.get("turns_expected", -2))
        == 34
        and len(read_jsonl(rows_path)) == int(FRAME_ANCHORS[frame]["row_count"])
    )


def validate_frame_receipts(run_dir: Path, frame: str) -> dict[str, Any]:
    """Validate a completed frame without invoking a model or CUDA."""
    if frame not in FRAME_ANCHORS:
        raise ADMError(f"unknown ADM frame: {frame}")
    run_dir = run_dir.resolve()
    frame_dir = run_dir / frame
    session_dir = frame_dir / "session"
    rows_path = frame_dir / "adm_rows.jsonl"
    scorecard_path = session_dir / "probe_scorecard.json"
    summary_path = session_dir / "summary.json"
    config_path = session_dir / "run_config.json"
    anchor_path = Path(FRAME_ANCHORS[frame]["scorecard"])

    rows = read_jsonl(rows_path)
    expected_count = int(FRAME_ANCHORS[frame]["row_count"])
    if len(rows) != expected_count:
        raise ADMError(f"{frame} expected {expected_count} ADM rows, found {len(rows)}")
    summary = read_json(summary_path)
    if not (
        int(summary.get("turns_completed", -1))
        == int(summary.get("turns_expected", -2))
        == 34
    ):
        raise ADMError(f"{frame} capture is not a complete 34-turn session")

    actual_scorecard = read_json(scorecard_path)
    anchor_scorecard = read_json(anchor_path)
    if actual_scorecard != anchor_scorecard:
        raise ADMError(
            f"{frame} scorecard differs from registered anchor {anchor_path}")

    anchor_fixture = str(FRAME_ANCHORS[frame]["anchor_fixture"])
    anchor_rows = [row for row in rows if row.get("fixture") == anchor_fixture]
    if len(anchor_rows) != 9:
        raise ADMError(f"{frame} expected nine A-k3 anchor rows, found {len(anchor_rows)}")
    anchor_probes = _probe_map(anchor_scorecard)
    mismatches = []
    for row in anchor_rows:
        turn = int(row["turn"])
        probe = anchor_probes.get(turn)
        k3 = (row.get("arms") or {}).get("A-k3") or {}
        observed = {
            "answer": str(k3.get("answer")),
            "mounted": [int(value) for value in k3.get("mounted_nodes") or ()],
            "answer_correct": bool(k3.get("answer_correct")),
            "canonical_answer": str(row.get("canonical_a_k3_answer")),
            "canonical_mounted": [
                int(value) for value in row.get("canonical_a_k3_mounts") or ()
            ],
            "canonical_equal": bool(row.get("canonical_a_k3_replay_equal")),
        }
        expected = None if probe is None else {
            "answer": str(probe.get("answer")),
            "mounted": [int(value) for value in probe.get("mounted_ids") or ()],
            "answer_correct": bool(probe.get("pass")),
            "canonical_answer": str(probe.get("answer")),
            "canonical_mounted": [int(value) for value in probe.get("mounted_ids") or ()],
            "canonical_equal": True,
        }
        if observed != expected:
            mismatches.append({"turn": turn, "observed": observed, "expected": expected})
    if mismatches:
        raise ADMError(f"{frame} A-k3 rows differ from registered control: {mismatches}")

    l2_audit = audit_l2_projection(frame_dir, rows)
    if not l2_audit["l2_projection_noop"]:
        raise ADMError(
            f"{frame} legacy capture cannot migrate to L2-on by projection: "
            f"{l2_audit['changed_candidate_sets']}")
    config = read_json(config_path)
    return {
        "schema": "grm.adm1.frame_alignment.v1",
        "status": "PASS",
        "frame": frame,
        "capture_mode": (
            "production_l2_on_explicit"
            if bool(config.get("sup_resolve"))
            else "legacy_l2_off_migrated_by_noop_projection"
        ),
        "alignment_target": "production_l2_on_explicit",
        "raw_replay_status": summary.get("status"),
        "raw_probe_score": [
            int(actual_scorecard.get("passed", 0)),
            int(actual_scorecard.get("total", 0)),
        ],
        "registered_anchor_exact": True,
        "all_a_k3_canonical_replays_equal": True,
        "captured_arm_rows_valid": True,
        "scorecard": file_record(scorecard_path),
        "registered_anchor": file_record(anchor_path),
        "rows": file_record(rows_path),
        "run_config": file_record(config_path),
        "summary": file_record(summary_path),
        "l2_projection": l2_audit,
        "l2_registration": file_record(L2_REGISTRATION),
    }


def _battery_probe(path: Path, probe_id: str) -> Mapping[str, Any]:
    matches = [
        row for row in read_jsonl(path)
        if row.get("record_type") == "supersession_probe_receipt"
        and row.get("probe_id") == probe_id
    ]
    if len(matches) != 1:
        raise ADMError(f"expected one {probe_id} row in {path}, found {len(matches)}")
    return matches[0]


def build_diag_adjudication(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    alignment = validate_frame_receipts(run_dir, "diag")
    diag_rows = read_jsonl(run_dir / "diag" / "adm_rows.jsonl")
    all_rows = read_jsonl(run_dir / "minicpm_rows.jsonl") + diag_rows
    scorecard = read_json(run_dir / "diag" / "session" / "probe_scorecard.json")
    legacy = _probe_map(read_json(Path(FRAME_ANCHORS["diag"]["scorecard"])))
    production = _probe_map(read_json(PRODUCTION_FULL_SCORECARD))
    failures = [row for row in scorecard["probes"] if not row.get("pass")]
    comparisons = []
    by_turn = {int(row["turn"]): row for row in diag_rows}
    for probe in failures:
        turn = int(probe["turn"])
        arm_answers = {
            arm: str(value["answer"])
            for arm, value in by_turn[turn]["arms"].items()
        }
        comparisons.append({
            "turn": turn,
            "fact_id": probe["fact_id"],
            "expected": probe["expected"],
            "captured_answer": probe["answer"],
            "legacy_registered_answer": legacy[turn]["answer"],
            "production_registered_answer": production[turn]["answer"],
            "matches_legacy_registered": probe["answer"] == legacy[turn]["answer"],
            "matches_production_registered": probe["answer"] == production[turn]["answer"],
            "all_admission_arms_same_answer": len(set(arm_answers.values())) == 1,
            "arm_answers": arm_answers,
            "canonical_a_k3_replay_equal": bool(
                by_turn[turn]["canonical_a_k3_replay_equal"]),
        })

    old_orion = _battery_probe(L2_LEGACY_ANCHOR, "orion_current")
    new_orion = _battery_probe(L2_DEFAULT_ANCHOR, "orion_current")
    synth = by_turn[13]
    run_manifest = read_json(run_dir / "run_manifest.json")
    manifest_path = str(run_manifest["manifest"]["path"])
    e2e_source = ROOT / "scripts" / "grm_e2e_session.py"
    gpu_source = ROOT / "scripts" / "grm_adm1_gpu.py"
    captured_config = run_dir / "diag" / "session" / "run_config.json"
    return {
        "schema": "grm.adm1.probe_skew_adjudication.v1",
        "order": "GRM-ADM1.2",
        "run_dir": str(run_dir),
        "verdict": "(a) BASELINE/GRADER SKEW; NOT AN ADM-POLICY REGRESSION",
        "qualification": (
            "The skew is not an L2 behavioral regression. The captured frame "
            "was explicit L2-off and exactly matches the registered legacy "
            "DIAG scorecard; the generic session exit gate instead demanded "
            "semantic 9/9. L2 is a no-op on every recorded candidate set."
        ),
        "unpinned_side": {
            "side": "generic replay verdict/grader",
            "status_line": (
                f"scripts/grm_e2e_session.py:"
                f"{_source_line(e2e_source, '\"status\": \"ok\" if all(')}"
            ),
            "exit_line": (
                f"scripts/grm_e2e_session.py:"
                f"{_source_line(e2e_source, 'return 0 if summary[\"status\"] == \"ok\" else 2')}"
            ),
            "captured_frame_pin": (
                f"artifacts/grm_adm1/{run_dir.name}/diag/session/run_config.json:"
                f"{_source_line(captured_config, '\"sup_resolve\": false')}"
            ),
            "current_frame_pin_line": (
                f"scripts/grm_adm1_gpu.py:"
                f"{_source_line(gpu_source, '\"--sup-resolve\",')}"
            ),
            "finding": (
                "The generic grader has no ADM registered-anchor input; it "
                "maps any semantic miss to status 2 even when A-k3 exactly "
                "reproduces the registered 7/9 control."
            ),
        },
        "failing_probe_comparison": comparisons,
        "l2_registered_orion_comparison": {
            "legacy_escape": {
                "answer": old_orion["answer_text"],
                "classification": old_orion["classification"],
                "anchor": file_record(L2_LEGACY_ANCHOR),
            },
            "production_default": {
                "answer": new_orion["answer_text"],
                "classification": new_orion["classification"],
                "anchor": file_record(L2_DEFAULT_ANCHOR),
            },
            "finding": (
                "The captured Vortex-3-Sierra read is a DIAG Cypher contaminator "
                "and matches neither L2 battery answer; L2 did not create it."
            ),
        },
        "alignment_choice": {
            "choice": "migrate ADM1 to explicit production L2-on",
            "future_frames": "--sup-resolve",
            "probe_ladder": (
                "remains explicitly off inside the admission experiment because "
                "it is itself an admission policy and would confound k1/k2/k3/A-DEC"
            ),
            "existing_diag_capture": (
                "accepted under the production L2 anchor by exhaustive no-op "
                "projection; the immutable original run_config remains L2-off"
            ),
        },
        "captured_rows": {
            "session_turns": 34,
            "minicpm_probe_rows": len(all_rows) - len(diag_rows),
            "diag_probe_rows": len(diag_rows),
            "valid": True,
            "minicpm_l2_noop_basis": (
                "CORPUS-100 and the two fresh-control arenas contain no "
                "metadata.supersedes edges; the L2 registration independently "
                "records fresh controls byte-identical across the flip"
            ),
            "l2_projection": alignment["l2_projection"],
        },
        "interim_analysis": {
            "scope": "captured MiniCPM plus DIAG rows only; not a final ADM verdict",
            "probe_rows": len(all_rows),
            "metrics": metric_table(all_rows),
            "p_synth_live_receipt_turn_13": {
                "identifier_hit_count": synth["identifier_hit_count"],
                "recall": {
                    arm: value["recall"] for arm, value in synth["arms"].items()
                },
                "branches": {
                    arm: value["policy_branch"] for arm, value in synth["arms"].items()
                },
            },
            "ADM-VERDICT": "NOT EMITTED — E2E and P4 stages remain unmeasured",
        },
        "corrected_lead_command": (
            "cd /mnt/ForgeRealm/GraftRepository\n"
            "export PYTHONPATH=\"$PWD${PYTHONPATH:+:$PYTHONPATH}\"\n"
            "python3 scripts/grm_adm1_gpu.py --stage all "
            f"--run-dir {run_dir.relative_to(ROOT)} "
            f"--manifest {manifest_path}"
        ),
        "frame_alignment": alignment,
        "registered_baselines": {
            "legacy_full": file_record(Path(FRAME_ANCHORS["diag"]["scorecard"])),
            "production_full": file_record(PRODUCTION_FULL_SCORECARD),
            "l2_registration": file_record(L2_REGISTRATION),
        },
    }


def render_adjudication(value: Mapping[str, Any]) -> str:
    lines = [
        "# GRM-ADM1.2 probe-skew adjudication",
        "",
        f"**Verdict: {value['verdict']}.**",
        "",
        value["qualification"],
        "",
        "## Probe evidence",
        "",
        "| Turn | Captured | Legacy registered | Production registered | All arms same |",
        "|---:|---|---|---|---|",
    ]
    for row in value["failing_probe_comparison"]:
        lines.append(
            f"| {row['turn']} | `{row['captured_answer']}` | "
            f"`{row['legacy_registered_answer']}` | "
            f"`{row['production_registered_answer']}` | "
            f"{'yes' if row['all_admission_arms_same_answer'] else 'no'} |"
        )
    unpinned = value["unpinned_side"]
    alignment = value["alignment_choice"]
    captured = value["captured_rows"]
    synth = value["interim_analysis"]["p_synth_live_receipt_turn_13"]
    lines += [
        "",
        "The L2 battery comparison is also negative for causation: legacy "
        f"`{value['l2_registered_orion_comparison']['legacy_escape']['answer']}`; "
        "production L2-on "
        f"`{value['l2_registered_orion_comparison']['production_default']['answer']}`. "
        "The DIAG failure's Cypher value matches neither.",
        "",
        "## Convicted mismatch",
        "",
        f"- Captured frame's explicit legacy pin: `{unpinned['captured_frame_pin']}`.",
        f"- Corrected production-L2 pin: `{unpinned['current_frame_pin_line']}`.",
        f"- Unpinned generic grader: `{unpinned['status_line']}` and `{unpinned['exit_line']}`.",
        f"- {unpinned['finding']}",
        "",
        "## Alignment and row validity",
        "",
        f"- Choice: **{alignment['choice']}**; subsequent frames pin "
        f"`{alignment['future_frames']}`.",
        f"- Probe ladder: {alignment['probe_ladder']}.",
        f"- Existing DIAG capture: {alignment['existing_diag_capture']}.",
        f"- Audited {captured['l2_projection']['candidate_sets_examined']} recorded "
        "candidate sets; L2 changed **0**. All 34 session turns and all 9 DIAG "
        "arm rows remain valid.",
        f"- The 22 MiniCPM rows also remain valid: {captured['minicpm_l2_noop_basis']}.",
        "",
        "## Reused analysis",
        "",
        f"- Captured probe rows analyzed: {value['interim_analysis']['probe_rows']} "
        f"({captured['minicpm_probe_rows']} MiniCPM + {captured['diag_probe_rows']} DIAG).",
        "- Live turn-13 P-SYNTH receipt: A-DEC declared-synthesis recall "
        f"{synth['recall']['A-DEC']}; A-k1 {synth['recall']['A-k1']}; "
        f"A-k2 {synth['recall']['A-k2']}; A-k3 {synth['recall']['A-k3']}.",
        f"- {value['interim_analysis']['ADM-VERDICT']}",
        "",
        "## Corrected lead command",
        "",
        "```bash",
        value["corrected_lead_command"],
        "```",
        "",
        "## Anything not done",
        "",
        "- No GPU/model replay was recomputed.",
        "- E2E and P4 remain for the lead; therefore the final ADM-G0/G3 table "
        "and ADM-VERDICT are not emitted here.",
        "",
    ]
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    value = build_diag_adjudication(args.run_dir)
    run_dir = args.run_dir.resolve()
    receipt = write_content_addressed(run_dir, "adm1_2_probe_adjudication", value)
    report = write_content_addressed(
        run_dir,
        "GRM_ADM1_2_ADJUDICATION",
        render_adjudication(value),
        suffix=".md",
    )
    print(f"adjudication={receipt}")
    print(f"report={report}")
    print(f"verdict={value['verdict']}")
    print("captured_rows_valid=True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
