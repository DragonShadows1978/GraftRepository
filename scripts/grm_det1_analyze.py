#!/usr/bin/env python3
"""Fit frozen DET1 thresholds and produce the registered four-arm race report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.grm_det1_common import (  # noqa: E402
    CALIBRATION_IDS,
    DETECTORS,
    DETError,
    file_record,
    fit_thresholds,
    race_metrics,
    read_json,
    read_jsonl,
    utc_now,
    write_content_addressed,
    write_json_exclusive,
)
from scripts.grm_det1_baseline_registry import (  # noqa: E402
    BaselineRegistryError,
    load_live_registry,
)


def _one(directory: Path, pattern: str) -> Path:
    paths = sorted(Path(directory).glob(pattern))
    if len(paths) != 1:
        raise DETError(f"expected one {pattern} under {directory}, found {len(paths)}")
    return paths[0]


def _record_path(record: Mapping[str, Any]) -> Path:
    path = Path(str(record["path"]))
    return path if path.is_absolute() else ROOT / path


def _valid_file_record(record: Mapping[str, Any]) -> bool:
    try:
        path = _record_path(record)
        return path.is_file() and file_record(path) == dict(record)
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _current_reanchor(
    run_dir: Path,
    baseline_anchor: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    candidates = []
    for path in sorted(Path(run_dir).glob("det1_1_reanchor_*.json")):
        value = read_json(path)
        if value.get("baseline_anchor") != dict(baseline_anchor):
            continue
        if not (
            value.get("schema") == "grm.det1.reanchor.v1"
            and value.get("status") == "READY_TO_RESUME"
            and _valid_file_record(value.get("order") or {})
        ):
            raise DETError(f"invalid current-anchor reanchor receipt: {path}")
        amendment = value.get("source_amendment")
        if amendment is not None and not _valid_file_record(amendment):
            raise DETError(f"reanchor source amendment no longer validates: {path}")
        frozen = value.get("frozen_artifacts_unchanged") or []
        if not frozen or not all(_valid_file_record(record) for record in frozen):
            raise DETError(f"reanchor frozen-artifact inventory drifted: {path}")
        audit = value.get("historical_row_audit") or {}
        if int(audit.get("comparison_count", -1)) != len(
            audit.get("comparisons") or []
        ):
            raise DETError(f"reanchor historical audit is incomplete: {path}")
        candidates.append((len(frozen), path, value))
    if not candidates:
        raise DETError("no validated DET1.1 reanchor receipt matches live registry")
    _count, path, value = max(candidates, key=lambda item: (item[0], str(item[1])))
    return path, value


def _current_harbor_adjudication(
    run_dir: Path,
    baseline_anchor: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    fixture_marker_path = (
        Path(run_dir) / "eval" / "sup_fixtures"
        / "correction_then_restatement" / "complete.json")
    fixture_marker = read_json(fixture_marker_path)
    record = fixture_marker.get("harbor_adjudication") or {}
    if not _valid_file_record(record):
        raise DETError("Harbor fixture marker lacks a valid adjudication binding")
    path = _record_path(record)
    value = read_json(path)
    if not (
        value.get("baseline_anchor") == dict(baseline_anchor)
        and value.get("status")
            == "PASS_STALE_CANONICAL_NOT_INSTRUMENTATION_LEAK"
        and value.get("instrumented_matches_live_registration") is True
        and value.get("instrumented_matches_live_harbor_raw_projection") is True
        and value.get("instrumented_matches_authoritative_private_rerun") is True
        and value.get("retired_guard_matches_instrumented_served") is False
        and _valid_file_record(value.get("served_rows") or {})
    ):
        raise DETError(f"Harbor adjudication did not validate: {path}")
    return path, value


def fit(run_dir: Path) -> Path:
    run_dir = Path(run_dir).resolve()
    registration_path = _one(run_dir, "registration_*.json")
    runtime_frame_path = _one(run_dir, "runtime_frame_*.json")
    registration = read_json(registration_path)
    runtime_frame = read_json(runtime_frame_path)
    rows_path = run_dir / "calibration" / "e2e_rows.jsonl"
    rows = read_jsonl(rows_path)
    ids = {str(row["fixture_id"]) for row in rows}
    if ids != set(CALIBRATION_IDS) or len(rows) != 2:
        raise DETError(
            f"calibration must contain exactly the two frozen served rows; "
            f"ids={sorted(ids)} rows={len(rows)}")
    if any(row.get("variant") != "served" for row in rows):
        raise DETError("calibration contains a non-served variant")
    bad_rows = []
    for row in rows:
        aliases = {int(value) for value in row.get("logical_alias_ids") or ()}
        mounted = {int(value) for value in row.get("mounted_ids") or ()}
        if not (
            row.get("split") == "calibration"
            and row.get("stage") == "calibration"
            and Path(str(row.get("registration_path", ""))).resolve()
                == registration_path
            and row.get("runtime_frame") == runtime_frame
            and row.get("answer_correct") is True
            and row.get("target_contains_expected") is True
            and row.get("router_rank1_admitted") is True
            and bool(aliases & mounted)
            and row.get("mounted_contains_expected") is True
            and row.get("full_index_contains_all_aliases") is True
        ):
            bad_rows.append(str(row.get("row_id")))
    if bad_rows:
        raise DETError(f"invalid served calibration controls: {bad_rows}")
    calibration_started = [
        int(row["evaluation_started_unix_ns"]) for row in rows]
    if int(registration["registered_unix_ns"]) >= min(calibration_started):
        raise DETError("registration does not predate calibration data")
    expected_registration = {
        str(row["fixture_id"]): row
        for row in registration["fixtures"] if row["split"] == "calibration"
    }
    if set(expected_registration) != ids:
        raise DETError("calibration IDs differ from the frozen registration")
    thresholds = fit_thresholds(rows)
    thresholds.update({
        "registration": file_record(registration_path),
        "runtime_frame": file_record(runtime_frame_path),
        "resolved_flags": dict(runtime_frame["resolved_flags"]),
        "calibration_rows": file_record(rows_path),
        "frozen_before_eval": True,
    })
    path = write_content_addressed(run_dir, "thresholds", thresholds)
    print(str(path))
    return path


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _f(value: float) -> str:
    return f"{float(value):.4f}"


def _latency(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.1f}"


def race_table_markdown(table: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| Detector | Recall | False-positive rate | Precision | F1 | Viable | Median latency (token, 0-based) |",
        "|---|---:|---:|---:|---:|:---:|---:|",
    ]
    by_name = {str(row["detector"]): row for row in table}
    for detector in DETECTORS:
        row = by_name[detector]
        lines.append(
            f"| {detector} | {_pct(row['recall'])} ({row['tp']}/{row['positive_n']}) "
            f"| {_pct(row['false_positive_rate'])} ({row['fp']}/{row['negative_n']}) "
            f"| {_f(row['precision'])} | {_f(row['f1'])} "
            f"| {'YES' if row['viable'] else 'NO'} "
            f"| {_latency(row['median_trigger_latency_token'])} |"
        )
    return "\n".join(lines)


def _eval_fixture_ids(registration: Mapping[str, Any]) -> list[str]:
    return [
        str(row["fixture_id"])
        for row in registration["fixtures"]
        if row["split"] == "eval"
    ]


def _gate_g0(
    rows: Sequence[Mapping[str, Any]],
    registration: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    expected_ids = _eval_fixture_ids(registration)
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["fixture_id"]), []).append(row)
    pair_invalid = []
    for fixture_id in expected_ids:
        variants = sorted(str(row.get("variant")) for row in grouped.get(fixture_id, ()))
        if variants != ["planted_miss", "served"]:
            pair_invalid.append(fixture_id)
    served = [row for row in rows if row.get("variant") == "served"]
    planted = [row for row in rows if row.get("variant") == "planted_miss"]
    served_bad = []
    for row in served:
        aliases = {int(value) for value in row.get("logical_alias_ids") or ()}
        mounted = {int(value) for value in row.get("mounted_ids") or ()}
        if not (
            row.get("answer_correct") is True
            and row.get("target_contains_expected") is True
            and row.get("router_rank1_admitted") is True
            and int(row.get("raw_router_rank1", -1)) in aliases
            and int(row.get("logical_router_rank1", -1)) in aliases
            and bool(aliases & mounted)
            and row.get("mounted_contains_expected") is True
            and row.get("full_index_contains_all_aliases") is True
        ):
            served_bad.append(str(row["fixture_id"]))
    planted_bad = []
    for row in planted:
        checks = row.get("plant_checks") or {}
        if not (
            row.get("target_contains_expected") is True
            and checks.get("withheld_aliases_absent") is True
            and checks.get("logical_target_absent") is True
            and checks.get("raw_router_rank1_absent") is True
            and checks.get("expected_value_absent_from_mounted_text") is True
            and checks.get("withheld_aliases_remain_in_full_detector_index") is True
            and checks.get("admission_ranking_unchanged") is True
            and checks.get("admission_branch_unchanged") is True
            and checks.get("only_aliases_removed_from_admission_plan") is True
            and checks.get("withheld_aliases_absent_from_every_ladder_attempt") is True
        ):
            planted_bad.append(str(row["fixture_id"]))
    observed_ids = sorted(grouped)
    passed = bool(
        len(served) == 12
        and len(planted) == 12
        and observed_ids == sorted(expected_ids)
        and not pair_invalid
        and not served_bad
        and not planted_bad
    )
    return ("PASS" if passed else "RED"), {
        "served_count": len(served),
        "planted_count": len(planted),
        "served_incorrect": served_bad,
        "planted_miss_invalid": planted_bad,
        "pair_invalid": pair_invalid,
        "expected_fixture_ids": expected_ids,
        "observed_fixture_ids": observed_ids,
    }


def _gate_g2(
    registration: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    registration_path: Path,
    runtime_frame_path: Path,
) -> tuple[str, dict[str, Any]]:
    runtime_frame = read_json(runtime_frame_path)
    calibration_path = _record_path(thresholds.get("calibration_rows") or {})
    calibration_rows = (
        read_jsonl(calibration_path) if calibration_path.is_file() else [])
    calibration_ids = {str(row.get("fixture_id")) for row in calibration_rows}
    expected_eval_ids = sorted(_eval_fixture_ids(registration))
    observed_eval_ids = sorted({str(row["fixture_id"]) for row in rows})
    registered_calibration = [
        row for row in registration["fixtures"] if row["split"] == "calibration"]
    registered_evaluation = [
        row for row in registration["fixtures"] if row["split"] == "eval"]
    query_fact_disjoint = bool(
        not (
            {str(row["question"]).casefold() for row in registered_calibration}
            & {str(row["question"]).casefold() for row in registered_evaluation}
        )
        and not (
            {str(row["fact_id"]).casefold() for row in registered_calibration}
            & {
                str(row["fact_id"]).casefold() for row in registered_evaluation
                if "fact_id" in row
            }
        )
    )
    eval_started = [int(row["evaluation_started_unix_ns"]) for row in rows]
    calibration_started = [
        int(row["evaluation_started_unix_ns"]) for row in calibration_rows
        if row.get("evaluation_started_unix_ns") is not None
    ]
    threshold_bindings = bool(
        thresholds.get("registration") == file_record(registration_path)
        and thresholds.get("runtime_frame") == file_record(runtime_frame_path)
        and calibration_path.is_file()
        and thresholds.get("calibration_rows") == file_record(calibration_path)
        and thresholds.get("resolved_flags") == runtime_frame.get("resolved_flags")
    )
    calibration_rows_valid = bool(
        len(calibration_rows) == 2
        and calibration_ids == set(CALIBRATION_IDS)
        and set(thresholds.get("fit_row_ids") or ())
            == {f"{fixture_id}:served" for fixture_id in CALIBRATION_IDS}
        and all(
            row.get("split") == "calibration"
            and row.get("stage") == "calibration"
            and row.get("variant") == "served"
            and Path(str(row.get("registration_path", ""))).resolve()
                == registration_path
            and row.get("runtime_frame") == runtime_frame
            and row.get("answer_correct") is True
            and row.get("target_contains_expected") is True
            and row.get("router_rank1_admitted") is True
            and bool(
                {int(value) for value in row.get("logical_alias_ids") or ()}
                & {int(value) for value in row.get("mounted_ids") or ()}
            )
            and row.get("mounted_contains_expected") is True
            and row.get("full_index_contains_all_aliases") is True
            for row in calibration_rows
        )
    )
    eval_rows_bound = bool(all(
        row.get("split") == "eval"
        and row.get("stage") == "eval"
        and Path(str(row.get("registration_path", ""))).resolve()
            == registration_path
        and row.get("runtime_frame") == runtime_frame
        for row in rows
    ))
    frozen = bool(
        thresholds.get("frozen_before_eval") is True
        and set(thresholds.get("fit_fixture_ids") or ()) == set(CALIBRATION_IDS)
        and threshold_bindings
        and calibration_rows_valid
        and eval_rows_bound
        and query_fact_disjoint
        and calibration_started
        and eval_started
        and int(registration["registered_unix_ns"]) < min(calibration_started)
        and max(calibration_started) < int(thresholds["created_unix_ns"])
        and int(thresholds["created_unix_ns"]) < min(eval_started)
        and observed_eval_ids == expected_eval_ids
    )
    return ("PASS" if frozen else "RED"), {
        "registration_unix_ns": int(registration["registered_unix_ns"]),
        "thresholds_unix_ns": int(thresholds["created_unix_ns"]),
        "first_calibration_unix_ns": (
            min(calibration_started) if calibration_started else None),
        "last_calibration_unix_ns": (
            max(calibration_started) if calibration_started else None),
        "first_eval_unix_ns": min(eval_started) if eval_started else None,
        "fit_fixture_ids": list(thresholds.get("fit_fixture_ids") or ()),
        "eval_fixture_ids": observed_eval_ids,
        "threshold_bindings_valid": threshold_bindings,
        "calibration_rows_valid": calibration_rows_valid,
        "eval_rows_bound": eval_rows_bound,
        "query_fact_disjoint": query_fact_disjoint,
        "split_disjoint": not bool(
            set(thresholds.get("fit_fixture_ids") or ())
            & {str(row["fixture_id"]) for row in rows}
        ),
        "eval_matches_registration": (
            observed_eval_ids == expected_eval_ids
        ),
    }


def fixture_table_markdown(
    rows: Sequence[Mapping[str, Any]],
    registration: Mapping[str, Any],
) -> str:
    by_key = {
        (str(row["fixture_id"]), str(row["variant"])): row
        for row in rows
    }
    registered = {
        str(row["fixture_id"]): row
        for row in registration["fixtures"] if row["split"] == "eval"
    }
    lines = [
        "| Fixture | Source probe | Expected | Raw rank-1 | Logical target | Alias IDs | Served mounts | Planted mounts | Plant valid |",
        "|---|---|---|---:|---:|---|---|---|:---:|",
    ]
    for fixture_id in _eval_fixture_ids(registration):
        fixture = registered[fixture_id]
        served = by_key[(fixture_id, "served")]
        planted = by_key[(fixture_id, "planted_miss")]
        source = (
            f"certified turn {fixture['turn']}"
            if fixture["source_family"] == "certified_34_turn"
            else f"sup {fixture['probe_id']}"
        )
        checks = planted.get("plant_checks") or {}
        valid = all(checks.get(name) is True for name in (
            "withheld_aliases_absent",
            "logical_target_absent",
            "raw_router_rank1_absent",
            "expected_value_absent_from_mounted_text",
            "withheld_aliases_remain_in_full_detector_index",
            "admission_ranking_unchanged",
            "admission_branch_unchanged",
            "only_aliases_removed_from_admission_plan",
            "withheld_aliases_absent_from_every_ladder_attempt",
        ))
        lines.append(
            f"| {fixture_id} | {source} | {fixture['expected_values']} "
            f"| {served['raw_router_rank1']} "
            f"| {served['logical_router_rank1']} "
            f"| {served['logical_alias_ids']} | {served['mounted_ids']} "
            f"| {planted['mounted_ids']} | {'YES' if valid else 'NO'} |"
        )
    return "\n".join(lines)


def report(run_dir: Path) -> Path:
    run_dir = Path(run_dir).resolve()
    registration_path = _one(run_dir, "registration_*.json")
    runtime_frame_path = _one(run_dir, "runtime_frame_*.json")
    thresholds_path = _one(run_dir, "thresholds_*.json")
    g1_path = _one(run_dir / "g1", "g1_receipt_*.json")
    registration = read_json(registration_path)
    runtime_frame = read_json(runtime_frame_path)
    thresholds = read_json(thresholds_path)
    g1 = read_json(g1_path)
    e2e_rows_path = run_dir / "eval" / "e2e_rows.jsonl"
    sup_rows_path = run_dir / "eval" / "sup_rows.jsonl"
    rows = read_jsonl(e2e_rows_path) + read_jsonl(sup_rows_path)
    if len(rows) != 24 or len({str(row["row_id"]) for row in rows}) != 24:
        raise DETError(f"expected 24 unique eval rows, observed {len(rows)}")
    table, winners, verdict = race_metrics(rows, thresholds)
    g0_status, g0_detail = _gate_g0(rows, registration)
    try:
        _registry, baseline_anchor = load_live_registry()
    except BaselineRegistryError as exc:
        raise DETError(f"final live baseline registry is invalid: {exc}") from exc
    reanchor_path, reanchor = _current_reanchor(run_dir, baseline_anchor)
    harbor_path, harbor_adjudication = _current_harbor_adjudication(
        run_dir, baseline_anchor)
    sup_rows = read_jsonl(sup_rows_path)
    sup_anchor_bad = [
        str(row["row_id"]) for row in sup_rows
        if row.get("baseline_anchor") != baseline_anchor
        or (
            row.get("variant") == "served"
            and (
                (row.get("served_baseline_comparison") or {}).get(
                    "live_match") is not True
                or (row.get("served_baseline_comparison") or {}).get(
                    "instrumented_matches_authoritative_rerun") is not True
            )
        )
        or (
            row.get("variant") == "planted_miss"
            and row.get("baseline_comparison_role") != "served_path_reference"
        )
    ]
    if sup_anchor_bad:
        raise DETError(
            f"post-DET1.1 supersession rows lack a live registry anchor: {sup_anchor_bad}")
    baseline_audit = {
        "baseline_anchor": baseline_anchor,
        "reanchor_receipt": file_record(reanchor_path),
        "immutable_pre_fix_rows": reanchor["historical_row_audit"],
        "harbor_adjudication": file_record(harbor_path),
        "post_fix_sup_row_count": len(sup_rows),
        "post_fix_sup_anchor_invalid": sup_anchor_bad,
    }
    g0_detail["registered_baseline_audit"] = baseline_audit
    g1_valid = bool(
        g1.get("status") == "PASS"
        and g1.get("byte_identical") is True
        and g1.get("runtime_frame") == file_record(runtime_frame_path)
        and _valid_file_record(g1.get("baseline") or {})
        and _valid_file_record(g1.get("instrumentation_off") or {})
    )
    g1_status = "PASS" if g1_valid else "RED"
    g2_status, g2_detail = _gate_g2(
        registration, thresholds, rows, registration_path, runtime_frame_path)
    if not g2_detail["split_disjoint"] or not g2_detail["eval_matches_registration"]:
        g2_status = "RED"
    gates = {
        "DET-G0": {"status": g0_status, **g0_detail},
        "DET-G1": {"status": g1_status, "receipt": file_record(g1_path)},
        "DET-G2": {"status": g2_status, **g2_detail},
        "DET-G3": {
            "status": (
                "PASS" if verdict in ("SUPPORTED", "PARTIAL", "REFUTED")
                else "RED"
            ),
            "four_rows": len(table),
            "verdict_present": True,
        },
    }
    prediction_sentence = (
        f"PREDICTION VERDICT: {verdict} — "
        + (
            "D-LQR won the viable race and every viable mechanistic detector outscores D-VERB."
            if verdict == "SUPPORTED" else
            "mechanistic detection beat D-VERB, but D-LQR did not win the viable race."
            if verdict == "PARTIAL" else
            "D-VERB won the viable race or tied the best mechanistic F1."
            if verdict == "REFUTED" else
            "The registered iff clauses do not cover this measured ordering."
        )
    )
    race = {
        "schema": "grm.det1.race.v1",
        "created_utc": utc_now(),
        "gates": gates,
        "table": table,
        "race_winners": winners,
        "prediction_verdict": verdict,
        "prediction_verdict_sentence": prediction_sentence,
        "thresholds": file_record(thresholds_path),
        "registration": file_record(registration_path),
        "runtime_frame": file_record(runtime_frame_path),
        "resolved_flags": dict(runtime_frame["resolved_flags"]),
        "eval_rows": [file_record(e2e_rows_path), file_record(sup_rows_path)],
        "baseline_anchor": baseline_anchor,
        "reanchor_receipt": file_record(reanchor_path),
        "harbor_adjudication": file_record(harbor_path),
        "registered_baseline_audit": baseline_audit,
    }
    race_path = write_content_addressed(run_dir, "race", race)
    table_text = race_table_markdown(table)
    fixture_text = fixture_table_markdown(rows, registration)
    incomplete = [name for name, value in gates.items() if value["status"] != "PASS"]
    incomplete_text = (
        "None." if not incomplete else
        "; ".join(f"{name}={gates[name]['status']}" for name in incomplete) + "."
    )
    frame_flags = runtime_frame["resolved_flags"]
    lines = [
        "# GRM-DET1 demand-detector race",
        "",
        f"Run: `{run_dir.name}`. Registered before GPU data; all 14 source probes "
        "were rerun on GPT-OSS-20B, with 2 calibration probes and 12 paired eval probes.",
        "",
        "## Production frame",
        "",
        f"- Probe ladder: `{frame_flags['probe_ladder']}`.",
        f"- L2 supersession resolution: `{frame_flags['sup_resolve']}`.",
        f"- Admission: `{frame_flags['admission_mode']}`; adm_decisive "
        f"`{frame_flags['adm_decisive']}`; top-k `{frame_flags['topk']}`.",
        f"- Turn pipeline: `{frame_flags['turn_pipeline']}`; L1 length debias: `{frame_flags['length_debias']}`.",
        f"- Exact verbal wording: `{registration['verbal_question_exact']}`.",
        f"- Live baseline registration: `{baseline_anchor['registration_id']}` "
        f"(`{baseline_anchor['registration_hash']}`).",
        "",
        "## Gates",
        "",
        "| Gate | Status | Receipt detail |",
        "|---|:---:|---|",
        f"| DET-G0 | {gates['DET-G0']['status']} | served incorrect: "
        f"{g0_detail['served_incorrect'] or 'none'}; invalid planted misses: "
        f"{g0_detail['planted_miss_invalid'] or 'none'} |",
        f"| DET-G1 | {gates['DET-G1']['status']} | flags-off raw transcript and scorecard bytes compared |",
        f"| DET-G2 | {gates['DET-G2']['status']} | calibration IDs {g2_detail['fit_fixture_ids']}; row disjoint={g2_detail['split_disjoint']}; query/fact disjoint={g2_detail['query_fact_disjoint']} |",
        f"| DET-G3 | {gates['DET-G3']['status']} | four detector rows and registered verdict below |",
        "",
        "## Four-row race table (verbatim)",
        "",
        table_text,
        "",
        prediction_sentence,
        "",
        f"RACE WINNER(S): {', '.join(winners) if winners else 'NONE (no viable detector)' }.",
        "",
        "Latency medians are zero-based generated-token indices over triggered planted misses only.",
        "",
        "## Enumerated evaluation pairs",
        "",
        fixture_text,
        "",
        "## Frozen thresholds",
        "",
        f"- D-NGH mounted-mass threshold: `{thresholds['D-NGH']['threshold']:.9g}` (strictly below).",
        f"- D-ENT entropy threshold: `{thresholds['D-ENT']['entropy_threshold']:.9g}` (strictly above).",
        f"- D-ENT margin threshold: `{thresholds['D-ENT']['margin_threshold']:.9g}` (strictly below).",
        "",
        "## DET1.1 live-baseline anchor",
        "",
        f"- Registry SHA-256: `{baseline_anchor['registry']['sha256']}`.",
        f"- Active registration SHA-256: `{baseline_anchor['registration_hash']}`.",
        f"- Harbor adjudication: `{harbor_adjudication['status']}`; Harbor-only "
        f"nonportable raw corroboration "
        f"live match `{harbor_adjudication['instrumented_matches_live_harbor_raw_projection']}`; "
        f"same-run authoritative match "
        f"`{harbor_adjudication['instrumented_matches_authoritative_private_rerun']}`.",
        f"- Immutable pre-fix served-row mismatches: "
        f"`{[value['row_id'] for value in reanchor['historical_row_audit']['mismatches']]}`.",
        f"- Post-fix supersession rows with invalid anchors: `{sup_anchor_bad}`.",
        "",
        "## Files",
        "",
        f"- Registration: `{registration_path.relative_to(ROOT)}`",
        f"- Runtime frame: `{runtime_frame_path.relative_to(ROOT)}`",
        f"- Thresholds: `{thresholds_path.relative_to(ROOT)}`",
        f"- DET-G1 receipt: `{g1_path.relative_to(ROOT)}`",
        f"- DET1.1 reanchor receipt: `{reanchor_path.relative_to(ROOT)}`",
        f"- Harbor adjudication: `{harbor_path.relative_to(ROOT)}`",
        f"- Eval rows: `{e2e_rows_path.relative_to(ROOT)}`, `{sup_rows_path.relative_to(ROOT)}`",
        f"- Race JSON: `{race_path.relative_to(ROOT)}`",
        "",
        "## Anything not done",
        "",
        incomplete_text,
        "",
    ]
    report_path = run_dir / "GRM_DET1_REPORT.md"
    gate_status_receipt = {
        "schema": "grm.det1.gate_statuses.v2",
        "gates": gates,
        "baseline_anchor": baseline_anchor,
        "reanchor_receipt": file_record(reanchor_path),
        "harbor_adjudication": file_record(harbor_path),
        "race": file_record(race_path),
    }
    write_json_exclusive(run_dir / "gate_statuses.json", gate_status_receipt)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(str(report_path))
    print(table_text)
    print(prediction_sentence)
    return report_path


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fit", "report", "selftest"))
    parser.add_argument("--run-dir", type=Path)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.command == "selftest":
        sample = [{
            "detector": name, "recall": 1.0, "tp": 12, "positive_n": 12,
            "false_positive_rate": 0.0, "fp": 0, "negative_n": 12,
            "precision": 1.0, "f1": 1.0, "viable": True,
            "median_trigger_latency_token": 0.0,
        } for name in DETECTORS]
        text = race_table_markdown(sample)
        if text.count("\n") != 5:
            raise DETError("race table selftest did not emit exactly four rows")
        print(json.dumps({"schema": "grm.det1.analyze_selftest.v1", "status": "PASS"}))
        return 0
    if args.run_dir is None:
        raise DETError("--run-dir is required")
    if args.command == "fit":
        fit(args.run_dir)
    else:
        report(args.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
