#!/usr/bin/env python3
"""Fail-closed analyzer for the fresh DET1.5 fork-substrate race.

This file never consumes the historical DET1 rows.  It accepts only the
ordered, content-addressed receipts below ``det1_4/campaign`` and emits a race
artifact only after every registered gate and all 12+12 observations validate.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.grm_det1_common import (  # noqa: E402
    CALIBRATION_IDS,
    DETECTORS,
    MECHANISTIC,
    VERBAL_QUESTION,
    DETError,
    canonical_json_bytes,
    file_record,
    fit_thresholds,
    normalize_value_text,
    race_metrics,
    read_json,
    read_jsonl,
    utc_now,
    write_content_addressed,
)
from scripts.grm_det1_5_gpu import (  # noqa: E402
    DELTA_AMENDMENT,
    DET1_7_ORDER,
    DET1_9_ORDER,
    DET1_7_SOURCE_AUTH,
    DET1_7_TERMINAL_AMENDMENT,
    aggregate_process_instances,
    det1_9_finding_receipt_path,
    effective_registration_projection,
    is_lineage_source_authorization,
    merge_detector_rows,
    plant_registry_path,
    validate_det1_7_source_authorization,
    validate_det1_7_terminal_amendment,
    validate_det1_9_finding_receipt,
    validate_g0_rows,
    validate_g1_pair,
    validate_served_control,
    validate_split,
    validate_verbal_chronology,
)
from scripts.grm_det1_7_registry import (  # noqa: E402
    ACHIEVED_EVAL_PAIR_FLOOR,
    REGISTERED_EVAL_PAIR_COUNT,
    REGISTERED_FIXTURE_COUNT,
    validate_plant_registry,
)


DET1_5_ORDER = ROOT / "orders" / "GRM_DET1_5_RACE_CAMPAIGN.md"
DET1_ORDER = ROOT / "orders" / "GRM_DET1_DEMAND_DETECTOR_RACE.md"
AMENDMENT_NAME = "det1_5_race_authorization_amendment.json"
CAMPAIGN_RELATIVE = Path("det1_4") / "campaign"
FROZEN_RUN = ROOT / "artifacts" / "grm_det1" / "run_20260831T160525Z_2"
VERDICTS = ("SUPPORTED", "PARTIAL", "REFUTED")

STAGES: tuple[tuple[str, Path, str], ...] = (
    (
        "cross_process_zero",
        Path("cross_process_zero") / "stage_complete.json",
        "grm.det1_5.cross_process_zero.v1",
    ),
    (
        "plant_registration",
        Path("det1_7") / "plant_registration" / "stage_complete.json",
        "grm.det1_7.plant_registration.v1",
    ),
    ("g0", Path("g0") / "stage_complete.json", "grm.det1_5.g0.v1"),
    ("g1", Path("g1") / "stage_complete.json", "grm.det1_5.g1.v1"),
    (
        "calibration",
        Path("calibration") / "stage_complete.json",
        "grm.det1_5.calibration.v1",
    ),
    (
        "eval_mechanistic",
        Path("eval") / "mechanistic_stage_complete.json",
        "grm.det1_5.mechanistic_eval.v1",
    ),
    (
        "eval_verbal",
        Path("eval") / "verbal_stage_complete.json",
        "grm.det1_5.chronological_verbal.v1",
    ),
)

SHARD_LAYOUT: dict[str, tuple[Path, tuple[str, ...]]] = {
    "plant_registration": (
        Path("det1_7") / "plant_registration" / "shards",
        (
            "e2e-cal",
            "e2e-1",
            "e2e-2",
            "e2e-3",
            "e2e-4",
            "sup-1",
            "sup-2",
            "sup-3",
            "sup-4",
        ),
    ),
    "g0": (
        Path("g0") / "shards",
        (
            "e2e-1",
            "e2e-2",
            "e2e-3",
            "e2e-4",
            "sup-1",
            "sup-2",
            "sup-3",
            "sup-4",
        ),
    ),
    "g1": (Path("g1") / "shards", ("control", "all-hooks")),
    "calibration": (Path("calibration") / "shards", ("e2e-cal",)),
    "eval_mechanistic": (
        Path("eval") / "mechanistic_shards",
        (
            "e2e-1",
            "e2e-2",
            "e2e-3",
            "e2e-4",
            "sup-1",
            "sup-2",
            "sup-3",
            "sup-4",
        ),
    ),
    "eval_verbal": (
        Path("eval") / "verbal_shards",
        (
            "e2e-1",
            "e2e-2",
            "e2e-3",
            "e2e-4",
            "sup-1",
            "sup-2",
            "sup-3",
            "sup-4",
        ),
    ),
}

# The predecessor DET1.5 authorization is append-only and therefore retains
# the stage order it authorized.  DET1.7's terminal amendment, validated
# separately below, inserts plant_registration between ZERO and G0.
DET1_5_AUTHORIZED_STAGE_ORDER = [
    "cross_process_zero",
    "g0",
    "g1",
    "calibration",
    "eval_mechanistic",
    "eval_verbal",
    "analyze",
]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DETError(message)


def _one(directory: Path, pattern: str) -> Path:
    paths = sorted(Path(directory).glob(pattern))
    _require(
        len(paths) == 1,
        f"expected exactly one {pattern} under {directory}, found {len(paths)}",
    )
    return paths[0]


def _record_path(record: Mapping[str, Any]) -> Path:
    try:
        path = Path(str(record["path"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise DETError(f"malformed file record: {record!r}") from exc
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _validated_record(
    record: Mapping[str, Any],
    where: str,
    *,
    beneath: Path | None = None,
) -> Path:
    _require(isinstance(record, Mapping), f"{where} is not a file record")
    _require(
        set(record) == {"path", "bytes", "sha256"},
        f"{where} must contain exactly path/bytes/sha256",
    )
    path = _record_path(record)
    _require(path.is_file(), f"{where} target is absent: {path}")
    if beneath is not None:
        _require(
            path.is_relative_to(Path(beneath).resolve()),
            f"{where} escapes the fresh namespace: {path}",
        )
    _require(file_record(path) == dict(record), f"{where} content record drifted")
    return path


def _same_record(actual: Any, expected: Mapping[str, Any], where: str) -> None:
    _require(isinstance(actual, Mapping), f"{where} is not a file record")
    _require(dict(actual) == dict(expected), f"{where} does not bind expected file")


def _require_projection(
    receipt: Mapping[str, Any],
    projection: Mapping[str, Any],
    where: str,
) -> None:
    for key, expected in projection.items():
        _require(
            receipt.get(key) == expected,
            f"{where}.{key} differs from independently recomputed validation",
        )


def _finite_probability(value: Any, where: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{where} is not numeric",
    )
    result = float(value)
    _require(math.isfinite(result) and 0.0 <= result <= 1.0, f"bad {where}: {value!r}")
    return result


def _finite_nonnegative(value: Any, where: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{where} is not numeric",
    )
    result = float(value)
    _require(math.isfinite(result) and result >= 0.0, f"bad {where}: {value!r}")
    return result


def prediction_verdict(table: Sequence[Mapping[str, Any]]) -> str:
    """Apply the registered DET1 adjudication, rejecting its uncovered corner.

    The function is intentionally pure so the frozen verdict precedence can be
    tested without manufacturing campaign artifacts.  No fourth verdict word
    is permitted by DET1.5.
    """

    _require(
        len(table) == len(DETECTORS),
        "verdict requires exactly four detector rows",
    )
    names = [str(row.get("detector")) for row in table]
    _require(names == list(DETECTORS), f"detector rows must be ordered {DETECTORS}")
    by_name: dict[str, Mapping[str, Any]] = {}
    for row in table:
        name = str(row["detector"])
        _finite_probability(row.get("f1"), f"{name}.f1")
        _require(type(row.get("viable")) is bool, f"{name}.viable is not boolean")
        by_name[name] = row

    viable = [row for row in table if row["viable"]]
    winners: list[str] = []
    if viable:
        best_viable = max(float(row["f1"]) for row in viable)
        winners = [
            str(row["detector"])
            for row in viable
            if float(row["f1"]) == best_viable
        ]
    best_mechanistic = max(float(by_name[name]["f1"]) for name in MECHANISTIC)
    verbal = float(by_name["D-VERB"]["f1"])
    viable_mechanistic = [name for name in MECHANISTIC if by_name[name]["viable"]]

    if "D-VERB" in winners or verbal == best_mechanistic:
        return "REFUTED"
    if "D-LQR" in winners and all(
        float(by_name[name]["f1"]) > verbal for name in viable_mechanistic
    ):
        return "SUPPORTED"
    if best_mechanistic > verbal and "D-LQR" not in winners:
        return "PARTIAL"
    raise DETError(
        "complete measurements fall into the registered DET1 adjudication gap; "
        "DET1.5 forbids inventing or emitting a fourth verdict"
    )


def _achieved_clause(achieved_pairs: int) -> str:
    """The DET1.11 population clause every verdict/table carries."""
    achieved = int(achieved_pairs)
    _require(
        ACHIEVED_EVAL_PAIR_FLOOR <= achieved <= REGISTERED_EVAL_PAIR_COUNT,
        f"achieved pair count {achieved} is outside the registered floor "
        f"{ACHIEVED_EVAL_PAIR_FLOOR}..{REGISTERED_EVAL_PAIR_COUNT}",
    )
    return f"at {achieved} of {REGISTERED_EVAL_PAIR_COUNT} registered pairs"


def _prediction_sentence(verdict: str, achieved_pairs: int) -> str:
    detail = {
        "SUPPORTED": (
            "D-LQR is a highest-F1 viable winner and every viable "
            "mechanistic detector outscores D-VERB."
        ),
        "PARTIAL": (
            "the best mechanistic raw F1 exceeds D-VERB, but D-LQR is not "
            "a highest-F1 viable winner."
        ),
        "REFUTED": (
            "D-VERB is a highest-F1 viable winner or ties the best "
            "mechanistic raw F1."
        ),
    }
    _require(verdict in detail, f"forbidden prediction verdict: {verdict}")
    # DET1.11: the achieved population is named IN the verdict sentence, so
    # the verdict can never be read as if it rested on the full registered
    # twelve.  The verdict vocabulary itself is unchanged.
    return (
        f"PREDICTION VERDICT: {verdict} "
        f"({_achieved_clause(achieved_pairs)}) — {detail[verdict]}"
    )


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _latency(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.1f}"


def race_table_markdown(
    table: Sequence[Mapping[str, Any]],
    *,
    achieved_pairs: int | None = None,
    excluded_fixture_ids: Sequence[str] = (),
) -> str:
    _require(
        [str(row.get("detector")) for row in table] == list(DETECTORS),
        "race table requires the four registered detector rows in order",
    )
    lines: list[str] = []
    if achieved_pairs is not None:
        # DET1.11: the achieved population heads the table, so N is read
        # before any metric is.
        lines.append(
            f"**Population: {_achieved_clause(achieved_pairs)}** "
            f"({achieved_pairs} planted-miss / {achieved_pairs} served)."
        )
        dropped = [str(value) for value in excluded_fixture_ids]
        if dropped:
            lines.append("")
            lines.append(
                f"Excluded ({len(dropped)}): {', '.join(dropped)} — lived "
                f"served control unlawful, reserve pool exhausted."
            )
        lines.append("")
    lines.extend([
        "| Detector | Recall | False-positive rate | Precision | F1 | Viable | Median latency (token, 0-based) |",
        "|---|---:|---:|---:|---:|:---:|---:|",
    ])
    for row in table:
        lines.append(
            f"| {row['detector']} | {_pct(row['recall'])} "
            f"({row['tp']}/{row['positive_n']}) | "
            f"{_pct(row['false_positive_rate'])} "
            f"({row['fp']}/{row['negative_n']}) | "
            f"{float(row['precision']):.4f} | {float(row['f1']):.4f} | "
            f"{'YES' if row['viable'] else 'NO'} | "
            f"{_latency(row['median_trigger_latency_token'])} |"
        )
    return "\n".join(lines)


def _registration(run_dir: Path) -> tuple[Path, dict[str, Any]]:
    path = _one(run_dir, "registration_*.json")
    value = read_json(path)
    _require(value.get("schema") == "grm.det1.registration.v1", "bad registration schema")
    fixtures = list(value.get("fixtures") or ())
    calibration = [row for row in fixtures if row.get("split") == "calibration"]
    evaluation = [row for row in fixtures if row.get("split") == "eval"]
    _require(
        [str(row.get("fixture_id")) for row in calibration] == list(CALIBRATION_IDS),
        "registration calibration split drifted",
    )
    # The BASE registration always carries the full registered twelve; only
    # the achieved population downstream of it varies (DET1.11).
    _require(
        len(evaluation) == REGISTERED_EVAL_PAIR_COUNT,
        f"registration no longer has {REGISTERED_EVAL_PAIR_COUNT} eval "
        f"fixtures",
    )
    _require(
        value.get("verbal_question_exact") == VERBAL_QUESTION,
        "registered D-VERB wording drifted",
    )
    _require(
        set((value.get("detectors") or {}).keys()) == set(DETECTORS),
        "registered detector set drifted",
    )
    adjudication = value.get("prediction_adjudication") or {}
    _require(
        all(name in adjudication for name in (*VERDICTS, "UNADJUDICATED")),
        "registered adjudication contract is incomplete",
    )
    return path, value


def _runtime_frame(run_dir: Path) -> tuple[Path, dict[str, Any]]:
    path = _one(run_dir, "runtime_frame_*.json")
    value = read_json(path)
    _require(value.get("schema") == "grm.det1.runtime_frame.v1", "bad runtime frame schema")
    flags = value.get("resolved_flags") or {}
    _require(flags.get("probe_ladder") is True, "runtime frame does not pin ladder ON")
    _require(flags.get("sup_resolve") is True, "runtime frame does not pin L2 ON")
    return path, value


def _zero_gate(run_dir: Path) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    marker_path = run_dir / "det1_4" / "zero_gate" / "stage_complete.json"
    _require(marker_path.is_file(), "DET1.4 ZERO marker is missing")
    marker = read_json(marker_path)
    _require(marker.get("schema") == "grm.det1_4.stage_marker.v1", "bad ZERO marker schema")
    _require(
        marker.get("stage") == "zero_gate" and marker.get("status") == "COMPLETE",
        "DET1.4 ZERO stage is not COMPLETE",
    )
    receipt_path = _validated_record(marker.get("receipt") or {}, "ZERO marker receipt")
    receipt = read_json(receipt_path)
    substrate = receipt.get("substrate_comparison") or {}
    claims = receipt.get("claims") or {}
    _require(receipt.get("schema") == "grm.det1_4.zero_fork_gate.v1", "bad ZERO receipt schema")
    _require(receipt.get("status") == "PASS", "DET1.4 ZERO receipt is not PASS")
    _require(receipt.get("gate_pass") is True, "DET1.4 ZERO gate_pass is not true")
    _require(
        claims.get("canonical_value_bytes_equal") is True,
        "DET1.4 ZERO lacks canonical-value-byte equality",
    )
    _require(
        substrate.get("gate_pass") is True
        and substrate.get("status") == "PASS_EXACT_CANONICAL_VALUE_BYTES",
        "DET1.4 ZERO substrate did not pass exact canonical value bytes",
    )
    return marker_path, marker, receipt_path, receipt


def _source_records(amendment: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw = amendment.get("source_inventory")
    records: dict[str, dict[str, Any]] = {}
    if isinstance(raw, Mapping):
        for label, value in raw.items():
            _require(isinstance(value, Mapping), f"source_inventory[{label}] is malformed")
            record = dict(value)
            if "path" not in record:
                record["path"] = str(label)
            records[str(record["path"])] = record
    elif isinstance(raw, list):
        for value in raw:
            _require(isinstance(value, Mapping), "source_inventory list member is malformed")
            record = dict(value)
            records[str(record.get("path"))] = record
    else:
        raise DETError("amendment source_inventory is absent")
    return records


def _validated_historical_source_record(
    record: Mapping[str, Any],
    source: str,
    where: str,
    *,
    source_rebindings: Mapping[str, Mapping[str, Any]],
) -> None:
    """Accept a DET1.5 source record only if current or exactly rebound."""

    _require(isinstance(record, Mapping), f"{where} is not a file record")
    _require(
        set(record) == {"path", "bytes", "sha256"},
        f"{where} must contain exactly path/bytes/sha256",
    )
    source_path = (ROOT / source).resolve()
    _require(
        _record_path(record) == source_path,
        f"{where} binds the wrong source path",
    )
    current = file_record(source_path)
    if dict(record) == current:
        _validated_record(record, where)
        return

    rebinding = source_rebindings.get(source)
    _require(
        isinstance(rebinding, Mapping),
        f"{where} is stale without a DET1.6 source rebinding",
    )
    _require(
        set(rebinding) == {"before", "after"},
        f"{where} DET1.6 source rebinding is malformed",
    )
    _same_record(rebinding.get("before"), record, f"{where}.rebinding.before")
    _same_record(rebinding.get("after"), current, f"{where}.rebinding.after")
    _require(
        rebinding.get("before") != rebinding.get("after"),
        f"{where} DET1.6 source rebinding is empty",
    )
    _validated_record(rebinding["after"], f"{where}.rebinding.after")


def _amendment(
    run_dir: Path,
    zero_marker_path: Path,
    zero_receipt_path: Path,
    registration_path: Path,
    runtime_frame_path: Path,
    *,
    source_rebindings: Mapping[str, Mapping[str, Any]],
) -> tuple[Path, dict[str, Any]]:
    path = run_dir / AMENDMENT_NAME
    _require(
        path.is_file(),
        "DET1.5 race authorization is GPU-pending/incomplete",
    )
    value = read_json(path)
    _require(
        value.get("schema") == "grm.det1_5.race_authorization_amendment.v1",
        "bad DET1.5 race-authorization schema",
    )
    _require(
        value.get("status") == "AUTHORIZED_RACE_ON_FORKED_SUBSTRATE",
        "DET1.5 amendment does not authorize the forked race",
    )
    _require(
        (value.get("invariants") or {}).get("race_resume_authorized") is True,
        "DET1.5 amendment race_resume_authorized is not true",
    )
    _same_record(value.get("order"), file_record(DET1_5_ORDER), "amendment.order")
    _same_record(value.get("det1_order"), file_record(DET1_ORDER), "amendment.det1_order")
    _same_record(
        value.get("registration"),
        file_record(registration_path),
        "amendment.registration",
    )
    _same_record(
        value.get("runtime_frame"),
        file_record(runtime_frame_path),
        "amendment.runtime_frame",
    )
    prior = run_dir / "det1_4_source_amendment.json"
    _validated_record(value.get("prior_amendment") or {}, "amendment.prior_amendment")
    _same_record(
        value.get("prior_amendment"),
        file_record(prior),
        "amendment.prior_amendment",
    )
    prior_value = read_json(prior)
    _require(
        prior_value.get("schema") == "grm.det1_4.source_amendment.v1"
        and prior_value.get("status")
        == "AUTHORIZED_FORK_FROM_SNAPSHOT_AMENDMENT"
        and (prior_value.get("invariants") or {}).get(
            "reconstruction_as_counterfactual"
        )
        == "FORBIDDEN"
        and (prior_value.get("invariants") or {}).get("race_resume_authorized")
        is False,
        "DET1.4 prior amendment chain is invalid",
    )
    _same_record(
        value.get("zero_gate_marker"),
        file_record(zero_marker_path),
        "amendment.zero_gate_marker",
    )
    _same_record(
        value.get("zero_gate_receipt"),
        file_record(zero_receipt_path),
        "amendment.zero_gate_receipt",
    )

    inventory = _source_records(value)
    required = (
        "scripts/grm_det1_5_gpu.py",
        "scripts/grm_det1_5_analyze.py",
        "scripts/grm_det1_5_workers.py",
    )
    for source in required:
        _require(source in inventory, f"amendment source inventory omits {source}")
        _validated_historical_source_record(
            inventory[source],
            source,
            f"amendment.source_inventory[{source}]",
            source_rebindings=source_rebindings,
        )
    lead = value.get("lead_script") or {}
    lead_source = "scripts/grm_det1_5_lead.sh"
    _validated_historical_source_record(
        lead,
        lead_source,
        "amendment.lead_script",
        source_rebindings=source_rebindings,
    )
    lead_path = (ROOT / lead_source).resolve()
    _require(
        lead_path == (ROOT / "scripts" / "grm_det1_5_lead.sh").resolve(),
        "amendment binds the wrong DET1.5 lead script",
    )
    lead_key = lead_source
    _require(lead_key in inventory, "lead script is absent from source inventory")
    _same_record(inventory[lead_key], lead, "source inventory lead script")
    invariants = value.get("invariants") or {}
    _require(invariants.get("fork_substrate_only") is True, "amendment permits reconstruction")
    _require(
        invariants.get("historical_rows_reusable") is False,
        "amendment permits historical row reuse",
    )
    _require(
        invariants.get("d_verb_execution")
        == "SEPARATE_CHRONOLOGICAL_LIVED_PROCESS",
        "amendment D-VERB process law drifted",
    )
    _require(
        invariants.get("reconstruction_as_counterfactual") == "FORBIDDEN",
        "amendment permits reconstruction-as-counterfactual",
    )
    _require(
        invariants.get("fresh_namespace") == str(CAMPAIGN_RELATIVE),
        "amendment fresh campaign namespace drifted",
    )
    _require(
        invariants.get("stage_order") == DET1_5_AUTHORIZED_STAGE_ORDER,
        "predecessor DET1.5 amendment stage order drifted",
    )
    _require(
        invariants.get("verdict_vocabulary") == list(VERDICTS),
        "amendment verdict vocabulary drifted",
    )
    return path, value


def _load_stage(
    campaign_dir: Path,
    stage: str,
    marker_relative: Path,
    receipt_schema: str,
) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    marker_path = campaign_dir / marker_relative
    _require(marker_path.is_file(), f"{stage} is GPU-pending/incomplete: missing {marker_path}")
    marker = read_json(marker_path)
    _require(
        marker.get("schema") == "grm.det1_5.stage_marker.v1",
        f"{stage} marker schema is not fresh DET1.5",
    )
    _require(marker.get("stage") == stage, f"{stage} marker names a different stage")
    _require(marker.get("status") == "COMPLETE", f"{stage} marker is not COMPLETE")
    receipt_path = _validated_record(
        marker.get("receipt") or {},
        f"{stage} marker receipt",
        beneath=campaign_dir,
    )
    receipt = read_json(receipt_path)
    _require(receipt.get("schema") == receipt_schema, f"{stage} receipt schema drifted")
    _require(receipt.get("status") == "PASS", f"{stage} receipt is not PASS")
    if "gate_pass" in receipt:
        _require(receipt.get("gate_pass") is True, f"{stage} gate_pass is not true")
    digest_prefix = str(file_record(receipt_path)["sha256"])[:16]
    _require(
        receipt_path.stem.endswith(digest_prefix),
        f"{stage} marker target is not content-addressed",
    )
    same_schema = []
    for candidate in sorted(receipt_path.parent.glob("*.json")):
        if candidate == marker_path:
            continue
        try:
            candidate_value = read_json(candidate)
        except (DETError, json.JSONDecodeError, OSError):
            continue
        if candidate_value.get("schema") == receipt_schema:
            same_schema.append(candidate.resolve())
    _require(
        same_schema == [receipt_path.resolve()],
        f"{stage} has duplicate or unbound same-schema receipts: {same_schema}",
    )
    return marker_path, marker, receipt_path, receipt


def _binding(
    receipt: Mapping[str, Any],
    key: str,
    expected_path: Path,
    where: str,
) -> None:
    _require(expected_path.is_file(), f"{where}.{key} target is absent")
    _same_record(receipt.get(key), file_record(expected_path), f"{where}.{key}")


def _validate_common_stage_bindings(
    receipts: Mapping[str, Mapping[str, Any]],
    *,
    amendment_path: Path,
    source_amendment_path: Path,
    precollection_authorization_path: Path,
    plant_registry_path_value: Path,
    terminal_amendment_path: Path,
    registration_path: Path,
    runtime_frame_path: Path,
) -> None:
    for stage, receipt in receipts.items():
        _binding(receipt, "registration", registration_path, stage)
        _binding(receipt, "runtime_frame", runtime_frame_path, stage)
        # DET1.11: receipts frozen in earlier rounds name the envelope that
        # governed them; lineage membership rather than current-envelope
        # equality.
        if stage == "plant_registration":
            _require_lineage_authorization(
                receipt.get("precollection_authorization"), stage,
            )
        elif stage != "cross_process_zero":
            _binding(receipt, "race_authorization_amendment", amendment_path, stage)
            _binding(receipt, "source_amendment", source_amendment_path, stage)
            _require_lineage_authorization(
                receipt.get("precollection_authorization"), stage,
            )
            _binding(receipt, "plant_registry", plant_registry_path_value, stage)
            _binding(receipt, "terminal_amendment", terminal_amendment_path, stage)


def _validate_marker_chain(
    markers: Mapping[str, Mapping[str, Any]],
    marker_paths: Mapping[str, Path],
    *,
    zero_marker_path: Path,
    zero_receipt: Mapping[str, Any],
    amendment_path: Path,
    source_amendment_path: Path,
    precollection_authorization_path: Path,
    plant_registry_path_value: Path,
    terminal_amendment_path: Path,
    registration_path: Path,
    runtime_frame_path: Path,
) -> None:
    predecessor = {
        "cross_process_zero": zero_marker_path,
        "plant_registration": marker_paths["cross_process_zero"],
        "g0": marker_paths["plant_registration"],
        "g1": marker_paths["g0"],
        "calibration": marker_paths["g1"],
        "eval_mechanistic": marker_paths["calibration"],
        "eval_verbal": marker_paths["eval_mechanistic"],
    }
    for stage, marker in markers.items():
        _same_record(
            marker.get("registration"),
            file_record(registration_path),
            f"{stage} marker.registration",
        )
        _same_record(
            marker.get("runtime_frame"),
            file_record(runtime_frame_path),
            f"{stage} marker.runtime_frame",
        )
        expected = [file_record(predecessor[stage])]
        authorization_slots: list[int] = []
        if stage == "cross_process_zero":
            expected.extend([
                dict(zero_receipt.get("lived_snapshot") or {}),
                dict(zero_receipt.get("fork_snapshot") or {}),
            ])
        elif stage == "plant_registration":
            # DET1.11: the envelope occupies a positional slot in the
            # prerequisite chain.  Mark it so the comparison below accepts
            # any lineage member in that one position while every other
            # prerequisite stays pinned exactly.
            authorization_slots.append(len(expected))
            expected.append(None)
            provenance = marker.get("det1_7_provenance")
            _require(
                isinstance(provenance, Mapping)
                and set(provenance) == {"precollection_authorization"},
                "plant_registration marker DET1.7 provenance drifted",
            )
            _require_lineage_authorization(
                provenance.get("precollection_authorization"),
                "plant_registration marker",
            )
        else:
            _binding(marker, "source_amendment", source_amendment_path, f"{stage} marker")
            expected.extend(
                [
                    file_record(terminal_amendment_path),
                    file_record(plant_registry_path_value),
                ]
            )
            provenance = marker.get("det1_7_provenance")
            _require(
                isinstance(provenance, Mapping)
                and set(provenance) == {
                    "precollection_authorization",
                    "plant_registry_record",
                    "terminal_amendment",
                }
                and provenance.get("plant_registry_record") == file_record(
                    plant_registry_path_value)
                and provenance.get("terminal_amendment") == file_record(
                    terminal_amendment_path),
                f"{stage} marker DET1.7 provenance drifted",
            )
            _require_lineage_authorization(
                provenance.get("precollection_authorization"),
                f"{stage} marker",
            )
        prerequisites = list(marker.get("prerequisites") or ())
        _require(
            len(prerequisites) == len(expected),
            f"{stage} marker prerequisite chain drifted",
        )
        for index, want in enumerate(expected):
            if index in authorization_slots:
                _require_lineage_authorization(
                    prerequisites[index], f"{stage} marker prerequisite",
                )
                continue
            _require(
                prerequisites[index] == want,
                f"{stage} marker prerequisite chain drifted",
            )


def _require_lineage_authorization(record: Any, where: str) -> None:
    """Accept any validated member of the DET1.9 envelope lineage.

    DET1.11: markers, receipts, and shards frozen in earlier campaign rounds
    name the envelope that governed THAT round.  Pinning equality against the
    current envelope makes every advance invalidate all prior artifacts —
    the r10 fail-close.  Membership of the validated lineage is the correct
    check; a record naming a file outside the series is still rejected.
    """
    _require(
        is_lineage_source_authorization(record),
        f"{where} names an authorization outside the validated DET1.9 "
        f"envelope lineage",
    )


def _validate_cross_process_zero(
    receipt: Mapping[str, Any],
    *,
    zero_receipt_path: Path,
    zero_receipt: Mapping[str, Any],
) -> None:
    expected_zero = file_record(zero_receipt_path)
    _same_record(receipt.get("zero_gate"), expected_zero, "cross.zero_gate")
    _same_record(
        receipt.get("parent_zero_gate"),
        expected_zero,
        "cross.parent_zero_gate",
    )
    lived_record = zero_receipt.get("lived_snapshot") or {}
    fork_record = zero_receipt.get("fork_snapshot") or {}
    _same_record(receipt.get("lived_snapshot"), lived_record, "cross.lived_snapshot")
    _same_record(receipt.get("fork_snapshot"), fork_record, "cross.fork_snapshot")
    lived_path = _validated_record(lived_record, "ZERO lived snapshot")
    fork_path = _validated_record(fork_record, "ZERO fork snapshot")
    lived_process = str(
        (read_json(lived_path).get("provenance") or {}).get(
            "process_instance_sha256", ""
        )
    )
    fork_process = str(
        (read_json(fork_path).get("provenance") or {}).get(
            "process_instance_sha256", ""
        )
    )
    _require(
        len(lived_process) == 64
        and len(fork_process) == 64
        and lived_process != fork_process,
        "existing ZERO does not bind distinct lived/fork processes",
    )
    _require(
        receipt.get("lived_process_instance_sha256") == lived_process
        and receipt.get("fork_process_instance_sha256") == fork_process
        and receipt.get("processes_distinct") is True,
        "cross-process ZERO process projection drifted",
    )
    substrate = zero_receipt.get("substrate_comparison") or {}
    _require(
        substrate.get("status") == "PASS_EXACT_CANONICAL_VALUE_BYTES"
        and substrate.get("gate_pass") is True
        and substrate.get("counts") == {"EQUAL": 345},
        "existing ZERO substrate is not exactly 345/345 equal",
    )
    _require(
        receipt.get("substrate_status") == substrate["status"]
        and receipt.get("substrate_counts") == {"EQUAL": 345},
        "cross-process ZERO substrate projection drifted",
    )
    answer = zero_receipt.get("answer") or {}
    normalized = normalize_value_text(str(answer.get("normalized", "")))
    lived_normalized = normalize_value_text(str(answer.get("lived_normalized", "")))
    _require(
        answer.get("normalized_match") is True
        and bool(normalized)
        and normalized == lived_normalized,
        "existing ZERO normalized answer does not match lived",
    )
    _require(
        normalize_value_text(str(receipt.get("normalized_answer", ""))) == normalized,
        "cross-process ZERO normalized-answer projection drifted",
    )


def _worker_shard(
    record: Mapping[str, Any],
    where: str,
    *,
    campaign_dir: Path,
    stage: str,
    amendment_path: Path,
    source_amendment_path: Path,
    precollection_authorization_path: Path,
    plant_registry_path_value: Path,
    terminal_amendment_path: Path,
    registration_path: Path,
    runtime_frame_path: Path,
) -> dict[str, Any]:
    path = _validated_record(record, where, beneath=campaign_dir)
    _require(
        path.stem.endswith(file_record(path)["sha256"][:16]),
        f"{where} is not content-addressed",
    )
    value = read_json(path)
    _require(value.get("schema") == "grm.det1_7.worker_shard.v1", f"{where} schema drifted")
    _require(value.get("status") == "COMPLETE", f"{where} is not COMPLETE")
    _require(value.get("stage") == stage, f"{where} names a different stage")
    _binding(value, "registration", registration_path, where)
    _binding(value, "runtime_frame", runtime_frame_path, where)
    _require(value.get("historical_rows_reused") is False, f"{where} reused history")
    expected_phase = (
        "PRECOLLECTION_SOURCE_AUTHORIZATION"
        if stage == "plant_registration"
        else "POST_REGISTRATION_EXACT_BINDING"
    )
    # DET1.11: the authorization is checked for lineage membership; every
    # other provenance record is still pinned exactly.
    expected_keys = {"precollection_authorization"}
    expected_provenance: dict[str, Any] = {}
    if stage != "plant_registration":
        expected_keys |= {"plant_registry_record", "terminal_amendment"}
        expected_provenance.update(
            {
                "plant_registry_record": file_record(plant_registry_path_value),
                "terminal_amendment": file_record(terminal_amendment_path),
            }
        )
    _require(
        value.get("det1_7_provenance_phase") == expected_phase,
        f"{where} DET1.7 provenance phase drifted",
    )
    provenance = value.get("det1_7_provenance")
    _require(
        isinstance(provenance, Mapping)
        and set(provenance) == expected_keys
        and all(
            provenance.get(key) == expected
            for key, expected in expected_provenance.items()
        ),
        f"{where} DET1.7 provenance records drifted",
    )
    _require_lineage_authorization(
        provenance.get("precollection_authorization"), where,
    )
    return value


def _process_instances(receipt: Mapping[str, Any], where: str) -> list[str]:
    values = [str(value) for value in receipt.get("process_instance_sha256s") or ()]
    aggregate = aggregate_process_instances(values)
    _require(
        receipt.get("process_instance_sha256") == aggregate,
        f"{where} process_instance_sha256 is not the ordered-list aggregate",
    )
    _require(
        receipt.get("process_identity_semantics")
        == "SHA256_OF_ORDERED_OS_PROCESS_INSTANCE_LIST",
        f"{where} does not label its aggregate process semantics",
    )
    return values


def _require_row_processes(
    rows: Sequence[Mapping[str, Any]],
    process_instances: Sequence[str],
    where: str,
) -> None:
    allowed = set(process_instances)
    for index, row in enumerate(rows):
        _require(
            str(row.get("process_instance_sha256", "")) in allowed,
            f"{where}[{index}] is not bound to a registered worker process",
        )


def _validate_stage_shards(
    receipt: Mapping[str, Any],
    where: str,
    *,
    campaign_dir: Path,
    amendment_path: Path,
    source_amendment_path: Path,
    precollection_authorization_path: Path,
    plant_registry_path_value: Path,
    terminal_amendment_path: Path,
    registration_path: Path,
    runtime_frame_path: Path,
    combined_rows: Sequence[Mapping[str, Any]] | None = None,
    combined_record_key: str = "rows",
    combined_count_key: str = "row_count",
    historical_outputs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    _require(where in SHARD_LAYOUT, f"no registered shard layout for {where}")
    shard_relative, expected_specs = SHARD_LAYOUT[where]
    records = list(receipt.get("shard_receipts") or ())
    process_instances = _process_instances(receipt, where)
    _require(
        len(records) == len(process_instances) == len(expected_specs),
        f"{where} shard/process/spec cardinality drifted",
    )
    shards: list[dict[str, Any]] = []
    for index, (record, expected_spec) in enumerate(
        zip(records, expected_specs, strict=True)
    ):
        receipt_path = _record_path(record)
        expected_spec_root = (campaign_dir / shard_relative / expected_spec).resolve()
        _require(
            receipt_path.is_relative_to(expected_spec_root),
            f"{where}.shard_receipts[{index}] escapes registered spec {expected_spec}",
        )
        shard = _worker_shard(
            record,
            f"{where}.shard_receipts[{index}]",
            campaign_dir=campaign_dir,
            stage=where,
            amendment_path=amendment_path,
            source_amendment_path=source_amendment_path,
            precollection_authorization_path=precollection_authorization_path,
            plant_registry_path_value=plant_registry_path_value,
            terminal_amendment_path=terminal_amendment_path,
            registration_path=registration_path,
            runtime_frame_path=runtime_frame_path,
        )
        _require(
            shard.get("spec") == expected_spec,
            f"{where}.shard_receipts[{index}] names a different worker spec",
        )
        # DET1.11: bind THIS round's attempt by reference from the marker
        # chain, not by globbing the shard directory.  Eleven append-only
        # campaign rounds have left completed receipts from earlier attempts
        # in place; they are historical evidence, not competitors for this
        # round's binding.  The receipt named by the marker identifies its
        # own attempt directory, so the attempt is resolved rather than
        # searched for, and stray completed receipts elsewhere in the shard
        # directory are inert by construction.
        bound_attempt_dir = receipt_path.parent.resolve()
        _require(
            bound_attempt_dir.parent == expected_spec_root
            and bound_attempt_dir.name.startswith("attempt_"),
            f"{where}.shard_receipts[{index}] is not inside a registered "
            f"attempt directory of {expected_spec}",
        )
        completed = []
        for output_path in sorted(expected_spec_root.glob("attempt_*/worker_output.json")):
            output = read_json(output_path)
            if output.get("status") not in ("PASS", "COMPLETE"):
                continue
            if output_path.parent.resolve() != bound_attempt_dir:
                # A completed attempt from an earlier round.  Record it as
                # historical evidence when the caller collects that, and
                # move on: only the marker-referenced attempt binds.
                if (
                    historical_outputs is not None
                    and output.get("schema") == "grm.det1_7.worker_shard.v1"
                ):
                    historical_outputs.append(file_record(output_path))
                continue
            finished_path = output_path.parent / "attempt_finished.json"
            if not finished_path.is_file():
                continue
            finished = read_json(finished_path)
            parent_process = finished.get("parent_process_instance") or {}
            parent_process_sha = str(
                parent_process.get("process_instance_sha256", "")
            )
            _require(
                finished.get("schema")
                == "grm.det1_5.worker_attempt_finished.v1"
                and finished.get("status") == "RETURNED"
                and finished.get("returncode") == 0
                and finished.get("worker") == where.replace("_", "-")
                and Path(str(finished.get("attempt_dir", ""))).resolve()
                == output_path.parent.resolve()
                and isinstance(finished.get("finished_unix_ns"), int)
                and not isinstance(finished.get("finished_unix_ns"), bool)
                and int(finished["finished_unix_ns"]) > 0
                and len(parent_process_sha) == 64
                and all(character in "0123456789abcdef" for character in parent_process_sha),
                f"{where}/{expected_spec} parent finish observation did not pass",
            )
            output_schema = output.get("schema")
            if output_schema != "grm.det1_7.worker_shard.v1":
                _require(
                    output_schema in (None, "grm.det1_5.worker_shard.v1"),
                    f"{where}/{expected_spec} has an unrecognized historical "
                    "shard schema",
                )
                if historical_outputs is not None:
                    historical_outputs.append(file_record(output_path))
                continue
            bound_path = _validated_record(
                output.get("receipt_file") or {},
                f"{where}/{expected_spec} completed worker output",
                beneath=campaign_dir,
            )
            bound = read_json(bound_path)
            _require(
                bound.get("schema") == "grm.det1_7.worker_shard.v1"
                and bound.get("status") == "COMPLETE"
                and bound.get("stage") == where
                and bound.get("spec") == expected_spec,
                f"{where}/{expected_spec} completed worker envelope drifted",
            )
            _require(
                set(output) == set(bound) | {"receipt_file"}
                and all(output.get(key) == value for key, value in bound.items()),
                f"{where}/{expected_spec} worker output differs from its receipt",
            )
            completed.append(bound_path.resolve())
        # Exactly one completed output may live in the BOUND attempt
        # directory, and it must be the receipt the marker chain named.
        # Historical attempts were skipped above, so this no longer fails
        # merely because earlier rounds also succeeded.
        _require(
            completed == [receipt_path],
            f"{where}/{expected_spec} bound attempt "
            f"{bound_attempt_dir.name} does not carry exactly its "
            f"marker-referenced completed shard receipt",
        )
        shards.append(shard)
    _require(
        [str(shard.get("process_instance_sha256", "")) for shard in shards]
        == process_instances,
        f"{where} shard process order differs from final receipt",
    )
    if combined_rows is not None:
        shard_rows: list[dict[str, Any]] = []
        for index, shard in enumerate(shards):
            rows_path = _validated_record(
                shard.get(combined_record_key) or {},
                f"{where}.shard[{index}].{combined_record_key}",
                beneath=campaign_dir,
            )
            rows = read_jsonl(rows_path)
            _require(
                int(shard.get(combined_count_key, -1)) == len(rows),
                f"{where}.shard[{index}] {combined_count_key} drifted",
            )
            _require(
                all(
                    row.get("process_instance_sha256")
                    == shard.get("process_instance_sha256")
                    for row in rows
                ),
                f"{where}.shard[{index}] rows bind a different worker process",
            )
            shard_rows.extend(rows)
        _require(
            shard_rows == list(combined_rows),
            f"{where} consolidated rows differ from ordered shard rows",
        )
    return shards


def _validate_verbal_sessions(
    records: Sequence[Mapping[str, Any]],
    shards: Sequence[Mapping[str, Any]],
    processes: Sequence[str],
    *,
    campaign_dir: Path,
) -> None:
    expected_specs = SHARD_LAYOUT["eval_verbal"][1]
    _require(
        len(records) == len(shards) == len(processes) == len(expected_specs),
        "D-VERB chronological session cardinality drifted",
    )
    _require(
        list(records)
        == [shard.get("chronological_session") for shard in shards],
        "D-VERB final session list differs from ordered shard evidence",
    )

    evidence_by_spec: dict[str, dict[str, Any]] = {}
    evidence_record_by_spec: dict[str, Mapping[str, Any]] = {}
    shard_record_by_spec: dict[str, Mapping[str, Any]] = {}
    session_dir_by_spec: dict[str, Path] = {}
    for index, (record, shard, process, spec) in enumerate(
        zip(records, shards, processes, expected_specs, strict=True)
    ):
        evidence_path = _validated_record(
            record,
            f"eval_verbal.chronological_session[{index}]",
            beneath=campaign_dir,
        )
        _require(
            evidence_path.stem.endswith(file_record(evidence_path)["sha256"][:16]),
            f"D-VERB {spec} session evidence is not content-addressed",
        )
        # Immutable shard files omit their self-record; recover the one exact
        # content match beside the session evidence.
        candidates = sorted(evidence_path.parent.glob("worker_shard_*.json"))
        candidates = [path for path in candidates if read_json(path) == dict(shard)]
        _require(
            len(candidates) == 1,
            f"D-VERB {spec} cannot recover its immutable shard record",
        )
        shard_path = candidates[0]
        _require(
            evidence_path.parent == shard_path.parent,
            f"D-VERB {spec} session evidence is outside its worker attempt",
        )
        evidence = read_json(evidence_path)
        _require(
            evidence.get("schema") == "grm.det1_5.chronological_session.v1"
            and evidence.get("stage") == "eval_verbal"
            and evidence.get("spec") == spec
            and evidence.get("process_instance_sha256") == process,
            f"D-VERB {spec} chronological session envelope drifted",
        )
        evidence_by_spec[spec] = evidence
        evidence_record_by_spec[spec] = record
        shard_record_by_spec[spec] = file_record(shard_path)

        if spec.startswith("e2e-"):
            _require(
                evidence.get("status") == "COMPLETE_BOUNDARY",
                f"D-VERB {spec} did not reach its chronological boundary",
            )
            session_dir = Path(str(evidence.get("session_dir", ""))).resolve()
            _require(
                session_dir == evidence_path.parent / "session"
                and session_dir.is_dir(),
                f"D-VERB {spec} active session directory is not its attempt session",
            )
            session_dir_by_spec[spec] = session_dir
            inventory = evidence.get("files") or {}
            _require(
                isinstance(inventory, Mapping),
                f"D-VERB {spec} session file inventory is malformed",
            )
            required = {
                "run_config.json",
                "transcript.jsonl",
                "instrumentation.jsonl",
                "probe_scorecard.json",
            }
            _require(required <= set(inventory), f"D-VERB {spec} evidence is incomplete")
            observed: set[str] = set()
            for relative, file_value in inventory.items():
                file_path = _validated_record(
                    file_value,
                    f"D-VERB {spec} session file {relative}",
                    beneath=session_dir,
                )
                _require(
                    file_path == (session_dir / str(relative)).resolve(),
                    f"D-VERB {spec} session inventory path drifted: {relative}",
                )
                observed.add(str(relative))
            actual = {
                str(path.relative_to(session_dir))
                for path in session_dir.rglob("*")
                if path.is_file()
            }
            _require(actual == observed, f"D-VERB {spec} session inventory drifted")
            run_config = read_json(session_dir / "run_config.json")
            declared = str(run_config.get("session_dir", ""))
            _require(
                evidence.get("run_config_declared_session_dir") == declared
                and evidence.get("active_session_dir") == str(session_dir),
                f"D-VERB {spec} run-config/active-session projection drifted",
            )
        else:
            _require(
                evidence.get("status") == "COMPLETE_LIVED_FIXTURE"
                and evidence.get("protocol")
                == "CHRONOLOGICAL_ARENA_FEED_THEN_INLINE_FORK",
                f"D-VERB {spec} lived supersession session drifted",
            )
            source_path = _validated_record(
                evidence.get("fixture_source") or {},
                f"D-VERB {spec} fixture source",
            )
            fixture_sources = tuple(
                sorted((ROOT / "tests" / "fixtures" / "supersession_battery").glob("*.json"))
            )
            source_index = int(spec.removeprefix("sup-")) - 1
            _require(
                len(fixture_sources) == 4
                and 0 <= source_index < len(fixture_sources)
                and source_path == fixture_sources[source_index].resolve(),
                f"D-VERB {spec} binds the wrong registered fixture source",
            )
            source = read_json(source_path)
            # DET1.9 binds one distinct reserve probe to each certified
            # supersession session, so the fed session carries the fixture's
            # own probes PLUS that reserve.  The worker records the split
            # explicitly; check every component rather than comparing the
            # total against the raw fixture, which never included it.
            source_probes = len(source.get("probes") or ())
            reserve_probes = evidence.get("reserve_probe_count")
            _require(
                evidence.get("node_count") == len(source.get("nodes") or ())
                and evidence.get("source_probe_count") == source_probes
                and reserve_probes == 1
                and evidence.get("probe_count") == source_probes + 1,
                f"D-VERB {spec} fixture cardinality projection drifted",
            )

    e2e_specs = ("e2e-1", "e2e-2", "e2e-3", "e2e-4")
    origin_dir = session_dir_by_spec["e2e-1"]
    for index, spec in enumerate(e2e_specs):
        evidence = evidence_by_spec[spec]
        session_dir = session_dir_by_spec[spec]
        expected_resume = (
            None if index == 0 else shard_record_by_spec[e2e_specs[index - 1]]
        )
        _require(
            evidence.get("resumed_from") == expected_resume,
            f"D-VERB {spec} immediate resumed_from chain drifted",
        )
        expected_names = [
            f"det1_5_resume_{e2e_specs[position - 1]}_to_{e2e_specs[position]}.json"
            for position in range(1, index + 1)
        ]
        inventory = evidence["files"]
        _require(
            all(name in inventory for name in expected_names),
            f"D-VERB {spec} omits registered resume metadata",
        )
        expected_metadata = [inventory[name] for name in expected_names]
        _require(
            evidence.get("resume_metadata") == expected_metadata,
            f"D-VERB {spec} cumulative resume metadata drifted",
        )
        if index == 0:
            _require(
                evidence.get("run_config_role") == "ACTIVE_SESSION_CONFIG"
                and Path(str(evidence["run_config_declared_session_dir"])).resolve()
                == session_dir,
                "D-VERB e2e-1 is not an active-origin session",
            )
        else:
            _require(
                evidence.get("run_config_role")
                == "COPIED_ORIGIN_CONFIG_WITH_EXPLICIT_RESUME_CHAIN"
                and Path(str(evidence["run_config_declared_session_dir"])).resolve()
                == origin_dir,
                f"D-VERB {spec} does not preserve its immutable origin config",
            )
        for position, metadata_record in enumerate(expected_metadata, start=1):
            predecessor = e2e_specs[position - 1]
            target = e2e_specs[position]
            metadata_path = _validated_record(
                metadata_record,
                f"D-VERB {spec} resume metadata {predecessor}->{target}",
                beneath=session_dir,
            )
            metadata = read_json(metadata_path)
            _require(
                metadata.get("schema") == "grm.det1_5.resume_metadata.v1"
                and metadata.get("status")
                == "COPIED_IMMUTABLE_PRODUCTION_BOUNDARY"
                and metadata.get("stage") == "eval_verbal"
                and metadata.get("spec") == target
                and metadata.get("predecessor_spec") == predecessor
                and metadata.get("process_instance_sha256")
                == processes[expected_specs.index(target)],
                f"D-VERB {spec} resume metadata envelope drifted",
            )
            _require(
                Path(str(metadata.get("source_session_dir", ""))).resolve()
                == session_dir_by_spec[predecessor]
                and Path(str(metadata.get("active_session_dir", ""))).resolve()
                == session_dir_by_spec[target],
                f"D-VERB {spec} resume metadata session path drifted",
            )
            _same_record(
                metadata.get("origin_run_config"),
                file_record(session_dir_by_spec[predecessor] / "run_config.json"),
                f"D-VERB {spec} resume origin config",
            )
            _same_record(
                metadata.get("copied_run_config"),
                file_record(session_dir_by_spec[target] / "run_config.json"),
                f"D-VERB {spec} resume copied config",
            )
            _same_record(
                metadata.get("predecessor_shard"),
                shard_record_by_spec[predecessor],
                f"D-VERB {spec} resume predecessor shard",
            )
            _same_record(
                metadata.get("predecessor_chronological_session"),
                evidence_record_by_spec[predecessor],
                f"D-VERB {spec} resume predecessor session",
            )
            _require(
                Path(str(metadata.get("origin_declared_session_dir", ""))).resolve()
                == origin_dir,
                f"D-VERB {spec} resume origin declaration drifted",
            )


def _registry_entries(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    # DET1.11: the registry carries the ACHIEVED population, so the entry
    # count is registered-minus-excluded rather than a fixed 14.  Identity
    # uniqueness and the reconstruction back to the registered fixture count
    # are still enforced exactly as before.
    raw = registry.get("entries")
    excluded = list(registry.get("excluded_slots") or ())
    expected = REGISTERED_FIXTURE_COUNT - len(excluded)
    _require(
        isinstance(raw, list) and len(raw) == expected,
        f"plant registry lacks {expected} entries "
        f"({REGISTERED_FIXTURE_COUNT} registered - {len(excluded)} excluded)",
    )
    entries = [dict(value) for value in raw if isinstance(value, Mapping)]
    _require(len(entries) == expected, "plant registry contains a malformed entry")
    _require(
        len({str(entry.get("fixture_id", "")) for entry in entries}) == expected,
        "plant registry base fixture identities are not unique",
    )
    _require(
        len({str(entry.get("effective_fixture_id", "")) for entry in entries})
        == expected,
        "plant registry effective fixture identities are not unique",
    )
    # An excluded slot must never reappear as an entry.
    excluded_ids = {
        str(row.get("fixture_id")) for row in excluded
        if isinstance(row, Mapping)
    }
    collisions = sorted(
        excluded_ids & {str(entry.get("fixture_id", "")) for entry in entries}
    )
    _require(
        not collisions,
        f"plant registry both excludes and carries entries for: {collisions}",
    )
    return entries


def _entry_by_effective_fixture(
    registry: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    return {
        str(entry["effective_fixture_id"]): entry
        for entry in _registry_entries(registry)
    }


def _split_ids(registry: Mapping[str, Any], split: str) -> list[str]:
    return [
        str(entry["effective_fixture_id"])
        for entry in _registry_entries(registry)
        if entry.get("split") == split
    ]


def _eval_ids(registry: Mapping[str, Any]) -> list[str]:
    """The ACHIEVED evaluation slot order the registry certified.

    DET1.11: the registered population is twelve; the achieved population is
    what actually produced lawful lived controls.  The registered floor is
    re-enforced here because this list is the choke point every downstream
    coverage check derives from.
    """
    values = _split_ids(registry, "eval")
    excluded = [
        str(row.get("fixture_id"))
        for row in registry.get("excluded_slots") or ()
    ]
    _require(
        len(values) + len(excluded) == REGISTERED_EVAL_PAIR_COUNT,
        f"plant registry eval fixtures plus exclusions must equal the "
        f"registered {REGISTERED_EVAL_PAIR_COUNT}, got "
        f"{len(values)}+{len(excluded)}",
    )
    _require(
        len(values) >= ACHIEVED_EVAL_PAIR_FLOOR,
        f"DET1.11 achieved-pair floor not met: {len(values)} evaluation "
        f"pairs is below the registered floor of {ACHIEVED_EVAL_PAIR_FLOOR}",
    )
    return values


def _expected_eval_keys(registry: Mapping[str, Any]) -> set[tuple[str, str]]:
    return {
        (fixture_id, variant)
        for fixture_id in _eval_ids(registry)
        for variant in ("served", "planted_miss")
    }


def _expected_eval_order(registry: Mapping[str, Any]) -> list[tuple[str, str]]:
    return [
        (fixture_id, variant)
        for fixture_id in _eval_ids(registry)
        for variant in ("served", "planted_miss")
    ]


def _row_key(row: Mapping[str, Any], where: str) -> tuple[str, str]:
    fixture_id = str(row.get("fixture_id"))
    variant = str(row.get("variant"))
    _require(variant in ("served", "planted_miss"), f"{where} has bad variant")
    _require(
        row.get("row_id") == f"{fixture_id}:{variant}",
        f"{where} row_id is not fixture_id:variant",
    )
    return fixture_id, variant


def _validate_mechanistic_row(
    row: Mapping[str, Any],
    where: str,
    *,
    split: str,
    registration_path: Path,
    runtime_frame: Mapping[str, Any],
) -> tuple[str, str]:
    _require(row.get("schema") == "grm.det1_5.mechanistic_row.v1", f"{where} is not fresh")
    key = _row_key(row, where)
    _require(row.get("split") == split, f"{where}.split drifted")
    _require(row.get("stage") == split, f"{where}.stage drifted")
    _require(
        Path(str(row.get("registration_path", ""))).resolve()
        == registration_path.resolve(),
        f"{where} registration path drifted",
    )
    _require(row.get("runtime_frame") == runtime_frame, f"{where} runtime frame drifted")
    _require(
        row.get("label") == (1 if key[1] == "planted_miss" else 0),
        f"{where} label differs from its registered variant",
    )
    started = row.get("evaluation_started_unix_ns")
    completed = row.get("mechanistic_completed_unix_ns")
    _require(
        isinstance(started, int)
        and not isinstance(started, bool)
        and isinstance(completed, int)
        and not isinstance(completed, bool)
        and 0 < started <= completed,
        f"{where} has invalid mechanistic chronology",
    )
    signals = row.get("signals") or {}
    _require(isinstance(signals, Mapping), f"{where}.signals is malformed")
    active = list(signals.get("active_detectors") or ())
    _require(
        len(active) == len(set(active)) and set(active) == set(MECHANISTIC),
        f"{where} does not bind all three mechanistic detectors",
    )
    tokens = list(signals.get("tokens") or ())
    _require(tokens, f"{where} has no mechanistic answer-token signals")
    for index, token in enumerate(tokens):
        _require(isinstance(token, Mapping), f"{where}.tokens[{index}] is malformed")
        _require(token.get("token_index") == index, f"{where} token indexing drifted")
        lqr = token.get("lqr")
        ngh = token.get("ngh")
        ent = token.get("ent")
        _require(isinstance(lqr, Mapping), f"{where} lacks D-LQR")
        _require(isinstance(ngh, Mapping), f"{where} lacks D-NGH")
        _require(isinstance(ent, Mapping), f"{where} lacks D-ENT")
        _require(type(lqr.get("trigger")) is bool, f"{where} has non-boolean D-LQR")
        _finite_probability(ngh.get("mounted_mass"), f"{where}.tokens[{index}].ngh")
        _finite_nonnegative(
            ent.get("entropy_nats"), f"{where}.tokens[{index}].entropy"
        )
        _finite_nonnegative(
            ent.get("top1_top2_margin"), f"{where}.tokens[{index}].margin"
        )
    return key


def _validate_verbal_row(
    row: Mapping[str, Any],
    where: str,
    *,
    registration_path: Path,
    runtime_frame: Mapping[str, Any],
) -> tuple[str, str]:
    _require(row.get("schema") == "grm.det1_5.verbal_row.v1", f"{where} is not fresh")
    key = _row_key(row, where)
    _require(row.get("split") == "eval", f"{where}.split drifted")
    _require(row.get("stage") == "eval", f"{where}.stage drifted")
    _require(
        Path(str(row.get("registration_path", ""))).resolve()
        == registration_path.resolve(),
        f"{where} registration path drifted",
    )
    _require(row.get("runtime_frame") == runtime_frame, f"{where} runtime frame drifted")
    _require(
        row.get("verbal_question_exact") == VERBAL_QUESTION,
        f"{where} registered question drifted",
    )
    _require(
        row.get("prompt_suffix_exact") == VERBAL_QUESTION,
        f"{where} wording drifted",
    )
    started = row.get("evaluation_started_unix_ns")
    verbal_started = row.get("verbal_started_unix_ns")
    mechanistic_completed = row.get("mechanistic_completed_unix_ns")
    _require(
        isinstance(started, int)
        and not isinstance(started, bool)
        and isinstance(verbal_started, int)
        and not isinstance(verbal_started, bool)
        and isinstance(mechanistic_completed, int)
        and not isinstance(mechanistic_completed, bool)
        and started == verbal_started > mechanistic_completed > 0,
        f"{where} has invalid chronological D-VERB timestamps",
    )
    verbal = row.get("verbal") or {}
    _require(type(verbal.get("trigger")) is bool, f"{where} lacks a boolean trigger")
    latency = verbal.get("latency_token_index")
    _require(
        latency is None or (isinstance(latency, int) and latency >= 0),
        f"{where} has invalid verbal latency",
    )
    return key


def _validate_fixture_projection(
    row: Mapping[str, Any],
    where: str,
    *,
    registry: Mapping[str, Any],
    mechanistic: bool,
) -> None:
    fixture_id = str(row.get("fixture_id", ""))
    entries = _entry_by_effective_fixture(registry)
    _require(fixture_id in entries, f"{where} does not name one effective fixture")
    entry = entries[fixture_id]
    fixture = entry.get("effective_fixture") or {}
    _require(isinstance(fixture, Mapping), f"{where} effective fixture is malformed")
    _require(
        entry.get("effective_fixture_id") == fixture.get("fixture_id") == fixture_id,
        f"{where} effective fixture identity drifted",
    )
    for key in ("question", "source_family"):
        _require(
            row.get(key) == fixture.get(key),
            f"{where}.{key} differs from registration",
        )
    _require(
        row.get("split") == fixture.get("split"),
        f"{where}.split differs from registered fixture",
    )
    if mechanistic:
        _require(
            row.get("expected_values") == fixture.get("expected_values"),
            f"{where}.expected_values differ from registration",
        )
        _require(
            row.get("stale_values")
            == fixture.get("stale_values", fixture.get("old_values", [])),
            f"{where}.stale_values differ from registration",
        )
        _require(
            row.get("wrong_fact_values") == fixture.get("wrong_fact_values", []),
            f"{where}.wrong_fact_values differ from registration",
        )


def _plant_entry_sha256(entry: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(dict(entry))).hexdigest()


def _validate_row_plant_binding(
    row: Mapping[str, Any],
    where: str,
    *,
    registry: Mapping[str, Any],
    registry_path: Path,
    mechanistic: bool,
) -> dict[str, Any]:
    fixture_id = str(row.get("fixture_id", ""))
    entries = _entry_by_effective_fixture(registry)
    _require(fixture_id in entries, f"{where} lacks a DET1.7 registry entry")
    entry = entries[fixture_id]
    target = int(entry["selected_target_id"])
    aliases = [int(value) for value in entry.get("alias_ids") or ()]
    _require(bool(aliases) and target in aliases, f"{where} registry target is empty")
    _require(
        row.get("plant_target_id") == target,
        f"{where}.plant_target_id differs from lived registry",
    )
    _require(
        row.get("plant_alias_ids") == aliases,
        f"{where}.plant_alias_ids differs from lived registry",
    )
    _require(
        row.get("plant_target_source") == "DET1_7_LIVED_ADMISSION_REGISTRY",
        f"{where}.plant_target_source drifted",
    )
    _same_record(
        row.get("plant_registry"),
        file_record(registry_path),
        f"{where}.plant_registry",
    )
    _require(
        row.get("plant_entry_sha256") == _plant_entry_sha256(entry),
        f"{where}.plant_entry_sha256 differs from the exact entry",
    )
    variant = str(row.get("variant", ""))
    expected_row_id = (
        entry.get("served_row_id")
        if variant == "served"
        else entry.get("planted_row_id")
    )
    _require(
        row.get("row_id") == expected_row_id,
        f"{where}.row_id differs from its registered effective pair",
    )
    if mechanistic:
        _require(
            row.get("logical_router_rank1") == target
            and row.get("production_admitted_rank1") == target,
            f"{where} logical/admitted target differs from DET1.7",
        )
        _require(
            row.get("logical_alias_ids") == aliases,
            f"{where}.logical_alias_ids differs from DET1.7",
        )
    return entry


def _planted_miss_valid(row: Mapping[str, Any]) -> bool:
    checks = row.get("plant_checks") or {}
    names = (
        "withheld_aliases_absent",
        "logical_target_absent",
        "registered_plant_target_absent",
        "registered_plant_aliases_absent",
        "expected_value_absent_from_mounted_text",
        "withheld_aliases_remain_in_full_detector_index",
        "admission_ranking_unchanged",
        "admission_branch_unchanged",
        "only_aliases_removed_from_admission_plan",
        "withheld_aliases_absent_from_every_ladder_attempt",
    )
    return bool(
        row.get("target_contains_expected") is True
        and all(checks.get(name) is True for name in names)
    )


def _utc_unix_ns(value: Any, where: str) -> int:
    _require(isinstance(value, str) and bool(value), f"{where} is not a timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DETError(f"{where} is not ISO-8601 UTC: {value!r}") from exc
    _require(
        parsed.tzinfo is not None and parsed.utcoffset() == timezone.utc.utcoffset(parsed),
        f"{where} is not UTC",
    )
    return int(parsed.timestamp() * 1_000_000_000)


def _plant_registration_table(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "base_fixture_id": entry["fixture_id"],
            "effective_fixture_id": entry["effective_fixture_id"],
            "split": entry["split"],
            "policy_branch": entry["policy_branch"],
            "actual_lived_mounts": entry["actual_authoritative_mounts"],
            "plant_target_id": entry["selected_target_id"],
            "plant_alias_ids": entry["alias_ids"],
            "selection_rule": entry["rule_id"],
            "source_snapshot": entry["source_snapshot"],
            "substitution": entry["substitution"],
        }
        for entry in _registry_entries(registry)
    ]


def _receipt_substitutions(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    from scripts.grm_det1_7_registry import SUBSTITUTION_SELECTION_RULE_ID

    result: list[dict[str, Any]] = []
    for entry in _registry_entries(registry):
        substitution = entry.get("substitution") or {}
        if substitution.get("status") != "SUBSTITUTED":
            continue
        result.append(
            {
                "base_fixture_id": entry["fixture_id"],
                "effective_fixture_id": entry["effective_fixture_id"],
                "source_family": entry["effective_fixture"]["source_family"],
                "session_id": entry["effective_fixture"]["session_id"],
                "selector": entry["effective_fixture"]["selector"],
                "reason": substitution.get("reason"),
                "same_certified_session": (
                    entry["effective_fixture"]["session_id"]
                    == entry["session"]["session_id"]
                ),
                "selection_rule_id": SUBSTITUTION_SELECTION_RULE_ID,
            }
        )
    return result


def _validate_plant_registration(
    receipt: Mapping[str, Any],
    *,
    campaign_dir: Path,
    amendment_path: Path,
    source_amendment_path: Path,
    precollection_authorization_path: Path,
    finding_path: Path,
    finding_validation: Mapping[str, Any],
    finding_text: Sequence[str],
    registry_path: Path,
    terminal_amendment_path: Path,
    registration_path: Path,
    runtime_frame_path: Path,
    registry: Mapping[str, Any],
    historical_outputs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    validation = validate_plant_registry(registry, record_root=ROOT)
    # DET1.11: pairing must be COMPLETE over the ACHIEVED population — every
    # certified eval slot carries both a served and a planted row — and the
    # achieved count must clear the registered floor.  The registry validator
    # already re-enforces the floor; this re-checks its projection so the
    # analyzer never trusts a count it did not verify.
    achieved = validation.get("achieved_eval_pairs")
    _require(
        validation.get("status") == "PASS"
        and validation.get("calibration_count") == 2
        and isinstance(achieved, int)
        and validation.get("eval_count") == achieved
        and validation.get("eval_served_count") == achieved
        and validation.get("eval_planted_count") == achieved,
        "DET1.7 plant registry did not validate complete pairing",
    )
    _require(
        validation.get("registered_eval_pairs") == REGISTERED_EVAL_PAIR_COUNT
        and int(achieved) + int(validation.get("excluded_eval_pairs", -1))
        == REGISTERED_EVAL_PAIR_COUNT,
        "plant registry achieved+excluded does not reconstruct the "
        "registered evaluation population",
    )
    _require(
        int(achieved) >= ACHIEVED_EVAL_PAIR_FLOOR,
        f"DET1.11 achieved-pair floor not met: {achieved} evaluation pairs "
        f"is below the registered floor of {ACHIEVED_EVAL_PAIR_FLOOR}",
    )
    _require(
        validation.get("fixture_count")
        == int(achieved) + int(validation.get("calibration_count", 0)),
        "plant registry fixture count differs from its achieved population",
    )
    _binding(receipt, "plant_registry", registry_path, "plant_registration")
    _require(
        receipt.get("registry_validation") == validation,
        "plant-registration validation projection drifted",
    )
    _require(
        receipt.get("plant_registry_payload_sha256")
        == registry.get("registry_payload_sha256"),
        "plant-registration payload hash differs from registry",
    )
    _require(
        receipt.get("plant_registry_file_sha256")
        == file_record(registry_path)["sha256"],
        "plant-registration file hash differs from registry",
    )
    _require(
        receipt.get("per_turn_table") == _plant_registration_table(registry),
        "plant-registration per-turn table differs from frozen entries",
    )
    _require(
        receipt.get("substitution_policy") == registry.get("substitution_policy"),
        "plant-registration DET1.9 substitution policy drifted",
    )
    _require(
        receipt.get("substitutions") == _receipt_substitutions(registry),
        "plant-registration substitution enumeration drifted",
    )
    _require(
        receipt.get("campaign_r4_lived_control_finding")
        == file_record(finding_path)
        and receipt.get("campaign_r4_lived_control_finding_validation")
        == dict(finding_validation)
        and receipt.get("campaign_r4_lived_control_finding_text")
        == list(finding_text),
        "plant-registration campaign-r4 finding binding drifted",
    )
    _require(
        receipt.get("counts")
        == {
            "served": int(achieved),
            "planted_miss": int(achieved),
            "eval_pairs": int(achieved),
        },
        "plant-registration did not preserve achieved-pair pairing",
    )
    _require(
        receipt.get("achieved_counts") == registry.get("achieved_counts"),
        "plant-registration achieved-count projection differs from registry",
    )
    _require(
        receipt.get("excluded_slots") == list(
            registry.get("excluded_slots") or ()),
        "plant-registration exclusion enumeration differs from registry",
    )
    _require(
        receipt.get("excluded_count")
        == len(registry.get("excluded_slots") or ()),
        "plant-registration excluded_count differs from its enumeration",
    )
    # Every candidate is still collected — DET1.11 shrinks the SELECTED
    # population, never the evidence gathered about it.
    _require(
        receipt.get("candidate_count") == 19
        and int(receipt.get("selected_count", -1))
        + int(receipt.get("excluded_count", -1)) == 14,
        "plant-registration candidate/selection cardinality drifted",
    )
    _require(
        receipt.get("detector_arms_active") == []
        and receipt.get("race_rows_emitted") is False
        and receipt.get("thresholds_fitted") is False,
        "plant-registration crossed its collection-only boundary",
    )
    _require(
        receipt.get("frozen_before_g0") is True
        and receipt.get("frozen_before_eval") is True,
        "plant-registration does not claim a pre-eval freeze",
    )

    candidates_path = _validated_record(
        receipt.get("candidate_observations") or {},
        "plant_registration.candidate_observations",
        beneath=campaign_dir,
    )
    selected_path = _validated_record(
        receipt.get("selected_observations") or {},
        "plant_registration.selected_observations",
        beneath=campaign_dir,
    )
    candidates = read_jsonl(candidates_path)
    selected = read_jsonl(selected_path)
    # Every candidate is still collected; DET1.11 shrinks only the selected
    # population, which must match the registry's achieved entry list.
    _require(len(candidates) == 19, "candidate observation file is not 19 rows")
    _require(
        len(selected) == len(_registry_entries(registry)),
        f"selected observation file is not "
        f"{len(_registry_entries(registry))} rows",
    )
    _require(
        [str(row.get("fixture_id", "")) for row in selected]
        == [str(entry["fixture_id"]) for entry in _registry_entries(registry)],
        "selected observations differ from frozen base-slot order",
    )
    for index, row in enumerate(selected):
        entry = _registry_entries(registry)[index]
        _require(
            row.get("effective_fixture_id") == entry.get("effective_fixture_id")
            and row.get("substitution") == entry.get("substitution")
            and row.get("snapshot_path")
            == str(_record_path(entry["source_snapshot"])),
            f"selected observation {index} differs from its registry entry",
        )
    shards = _validate_stage_shards(
        receipt,
        "plant_registration",
        campaign_dir=campaign_dir,
        amendment_path=amendment_path,
        source_amendment_path=source_amendment_path,
        precollection_authorization_path=precollection_authorization_path,
        plant_registry_path_value=registry_path,
        terminal_amendment_path=terminal_amendment_path,
        registration_path=registration_path,
        runtime_frame_path=runtime_frame_path,
        combined_rows=candidates,
        combined_record_key="observations",
        combined_count_key="observation_count",
        historical_outputs=historical_outputs,
    )
    _require(
        all(
            shard.get("detector_hooks") == []
            and shard.get("variants") == ["served"]
            for shard in shards
        ),
        "plant-registration shard activated a detector or planted arm",
    )
    processes = [str(shard.get("process_instance_sha256", "")) for shard in shards]
    _require(
        len(processes) == len(set(processes)) == 9
        and receipt.get("process_instance_sha256s") == processes
        and receipt.get("process_instance_sha256")
        == aggregate_process_instances(processes),
        "plant-registration process aggregation drifted",
    )
    return shards


def _thresholds_equal(frozen: Mapping[str, Any], fitted: Mapping[str, Any]) -> bool:
    return bool(
        frozen.get("D-NGH") == fitted.get("D-NGH")
        and frozen.get("D-ENT") == fitted.get("D-ENT")
        and frozen.get("fit_fixture_ids") == fitted.get("fit_fixture_ids")
        and frozen.get("fit_row_ids") == fitted.get("fit_row_ids")
        and frozen.get("fit_side") == fitted.get("fit_side")
    )


def _write_text_rendering(path: Path, text: str) -> Path:
    """Write a derived rendering, refreshing it when its source advances.

    Distinct from :func:`_write_text_exclusive_or_verify`, which guards
    write-once EVIDENCE.  A rendering carries no information the receipt
    does not already hold, so refusing to refresh it would strand the
    campaign on a cosmetic difference (DET1.11).
    """
    payload = text.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == payload:
        return path
    path.write_bytes(payload)
    return path


def _write_text_exclusive_or_verify(path: Path, text: str) -> Path:
    payload = text.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require(path.read_bytes() == payload, f"existing report differs: {path}")
        return path
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
    return path


def analyze(run_dir: Path) -> Path:
    """Validate the complete fresh campaign and write its registered report."""

    run_dir = Path(run_dir).resolve()
    required_root = (ROOT / "artifacts" / "grm_det1").resolve()
    _require(run_dir.is_relative_to(required_root), f"run is outside {required_root}")
    _require(
        run_dir == FROZEN_RUN.resolve(),
        f"DET1.5 analyzer is bound to the frozen run {FROZEN_RUN}",
    )
    campaign_dir = run_dir / CAMPAIGN_RELATIVE
    _require(campaign_dir.is_dir(), f"fresh campaign namespace is absent: {campaign_dir}")

    registration_path, registration = _registration(run_dir)
    runtime_frame_path, runtime_frame = _runtime_frame(run_dir)
    zero_marker_path, _zero_marker, zero_receipt_path, zero_receipt = _zero_gate(run_dir)
    precollection = validate_det1_7_source_authorization(run_dir)
    _require(
        precollection.get("collection_authorized") is True
        and precollection.get("race_resume_authorized") is False
        and precollection.get("amendment_is_evidence") is False,
        "DET1.7 precollection authority crossed into evaluation",
    )
    precollection_authorization_path = _validated_record(
        precollection.get("record") or {}, "DET1.7 precollection authorization"
    )
    finding_path = det1_9_finding_receipt_path(run_dir)
    finding_validation = validate_det1_9_finding_receipt(run_dir)
    finding_receipt = read_json(finding_path)
    finding_text = [
        str(row["text"]) for row in finding_receipt["ordered_findings"]
    ]
    source_amendment_path = (run_dir / DELTA_AMENDMENT.name).resolve()
    _require(source_amendment_path.is_file(), "DET1.6 predecessor amendment is absent")
    source_rebindings = precollection.get("det1_5_source_rebindings")
    _require(
        isinstance(source_rebindings, Mapping),
        "DET1.7 direct DET1.5 source rebindings are malformed",
    )
    amendment_path, authorization = _amendment(
        run_dir,
        zero_marker_path,
        zero_receipt_path,
        registration_path,
        runtime_frame_path,
        source_rebindings=source_rebindings,
    )
    terminal = validate_det1_7_terminal_amendment(run_dir)
    terminal_amendment_path = _validated_record(
        terminal.get("record") or {}, "DET1.7 terminal amendment"
    )
    registry_path = plant_registry_path(run_dir)
    _same_record(
        terminal.get("plant_registry"),
        file_record(registry_path),
        "DET1.7 terminal plant registry",
    )
    registry = read_json(registry_path)
    _require(
        terminal.get("plant_registry_validation")
        == validate_plant_registry(registry, record_root=ROOT),
        "DET1.7 terminal registry validation drifted",
    )

    stage_files: dict[str, tuple[Path, Path]] = {}
    stage_markers: dict[str, dict[str, Any]] = {}
    receipts: dict[str, dict[str, Any]] = {}
    for stage, marker_relative, schema in STAGES:
        marker_path, marker, receipt_path, receipt = _load_stage(
            campaign_dir, stage, marker_relative, schema
        )
        stage_files[stage] = (marker_path, receipt_path)
        stage_markers[stage] = marker
        receipts[stage] = receipt
    _validate_marker_chain(
        stage_markers,
        {stage: paths[0] for stage, paths in stage_files.items()},
        zero_marker_path=zero_marker_path,
        zero_receipt=zero_receipt,
        amendment_path=amendment_path,
        source_amendment_path=source_amendment_path,
        precollection_authorization_path=precollection_authorization_path,
        plant_registry_path_value=registry_path,
        terminal_amendment_path=terminal_amendment_path,
        registration_path=registration_path,
        runtime_frame_path=runtime_frame_path,
    )
    _validate_common_stage_bindings(
        receipts,
        amendment_path=amendment_path,
        source_amendment_path=source_amendment_path,
        precollection_authorization_path=precollection_authorization_path,
        plant_registry_path_value=registry_path,
        terminal_amendment_path=terminal_amendment_path,
        registration_path=registration_path,
        runtime_frame_path=runtime_frame_path,
    )

    cross = receipts["cross_process_zero"]
    _validate_cross_process_zero(
        cross,
        zero_receipt_path=zero_receipt_path,
        zero_receipt=zero_receipt,
    )
    cross_marker_path, cross_receipt_path = stage_files["cross_process_zero"]
    _same_record(
        authorization.get("cross_process_zero_marker"),
        file_record(cross_marker_path),
        "amendment.cross_process_zero_marker",
    )
    _same_record(
        authorization.get("cross_process_zero_receipt"),
        file_record(cross_receipt_path),
        "amendment.cross_process_zero_receipt",
    )

    historical_outputs: list[dict[str, Any]] = []
    _validate_plant_registration(
        receipts["plant_registration"],
        campaign_dir=campaign_dir,
        amendment_path=amendment_path,
        source_amendment_path=source_amendment_path,
        precollection_authorization_path=precollection_authorization_path,
        finding_path=finding_path,
        finding_validation=finding_validation,
        finding_text=finding_text,
        registry_path=registry_path,
        terminal_amendment_path=terminal_amendment_path,
        registration_path=registration_path,
        runtime_frame_path=runtime_frame_path,
        registry=registry,
        historical_outputs=historical_outputs,
    )

    g0 = receipts["g0"]
    _require(g0.get("gate_pass") is True, "DET-G0 did not pass")
    g0_rows_path = campaign_dir / "g0" / "mechanistic_rows.jsonl"
    _binding(g0, "rows", g0_rows_path, "g0")
    g0_rows = read_jsonl(g0_rows_path)
    g0_keys: list[tuple[str, str]] = []
    for index, row in enumerate(g0_rows):
        where = f"g0[{index}]"
        key = _validate_mechanistic_row(
            row,
            where,
            split="eval",
            registration_path=registration_path,
            runtime_frame=runtime_frame,
        )
        _validate_fixture_projection(row, where, registry=registry, mechanistic=True)
        _validate_row_plant_binding(
            row, where, registry=registry, registry_path=registry_path,
            mechanistic=True,
        )
        if key[1] == "served":
            validate_served_control(row, f"dedicated G0 served row {key[0]}")
        else:
            _require(_planted_miss_valid(row), f"invalid dedicated G0 planted row: {key}")
        g0_keys.append(key)
    _require(
        g0_keys == _expected_eval_order(registry),
        "dedicated DET-G0 rows differ from registered 12+12 coverage",
    )
    _require_projection(
        g0, validate_g0_rows(g0_rows, expected_pairs=len(_eval_ids(registry))),
        "g0")
    _require_row_processes(g0_rows, _process_instances(g0, "g0"), "g0")
    _validate_stage_shards(
        g0,
        "g0",
        campaign_dir=campaign_dir,
        amendment_path=amendment_path,
        source_amendment_path=source_amendment_path,
        precollection_authorization_path=precollection_authorization_path,
        plant_registry_path_value=registry_path,
        terminal_amendment_path=terminal_amendment_path,
        registration_path=registration_path,
        runtime_frame_path=runtime_frame_path,
        combined_rows=g0_rows,
        historical_outputs=historical_outputs,
    )

    g1 = receipts["g1"]
    control = _worker_shard(
        g1.get("control") or {},
        "g1.control",
        campaign_dir=campaign_dir,
        stage="g1",
        amendment_path=amendment_path,
        source_amendment_path=source_amendment_path,
        precollection_authorization_path=precollection_authorization_path,
        plant_registry_path_value=registry_path,
        terminal_amendment_path=terminal_amendment_path,
        registration_path=registration_path,
        runtime_frame_path=runtime_frame_path,
    )
    hooked = _worker_shard(
        g1.get("hooked") or {},
        "g1.hooked",
        campaign_dir=campaign_dir,
        stage="g1",
        amendment_path=amendment_path,
        source_amendment_path=source_amendment_path,
        precollection_authorization_path=precollection_authorization_path,
        plant_registry_path_value=registry_path,
        terminal_amendment_path=terminal_amendment_path,
        registration_path=registration_path,
        runtime_frame_path=runtime_frame_path,
    )
    for arm_name, arm in (("control", control), ("hooked", hooked)):
        for byte_name, record in (arm.get("byte_records") or {}).items():
            _validated_record(
                record,
                f"g1.{arm_name}.byte_records.{byte_name}",
                beneath=campaign_dir,
            )
    _require_projection(g1, validate_g1_pair(control, hooked), "g1")
    _require(
        g1.get("gate_pass") is True
        and g1.get("byte_identical") is True
        and g1.get("all_hooks_active") is True,
        "DET-G1 all-hooks byte identity did not pass",
    )
    _require(
        control.get("process_instance_sha256")
        != hooked.get("process_instance_sha256"),
        "DET-G1 control and hooked arms are not distinct processes",
    )
    _require(
        _process_instances(g1, "g1")
        == [
            str(control["process_instance_sha256"]),
            str(hooked["process_instance_sha256"]),
        ],
        "DET-G1 ordered process list does not match its two arms",
    )
    _require(
        g1.get("shard_receipts") == [g1.get("control"), g1.get("hooked")],
        "DET-G1 final arm records differ from ordered shard records",
    )
    _validate_stage_shards(
        g1,
        "g1",
        campaign_dir=campaign_dir,
        amendment_path=amendment_path,
        source_amendment_path=source_amendment_path,
        precollection_authorization_path=precollection_authorization_path,
        plant_registry_path_value=registry_path,
        terminal_amendment_path=terminal_amendment_path,
        registration_path=registration_path,
        runtime_frame_path=runtime_frame_path,
        historical_outputs=historical_outputs,
    )

    calibration_path = campaign_dir / "calibration" / "mechanistic_rows.jsonl"
    threshold_path = _one(campaign_dir / "calibration", "thresholds_*.json")
    calibration_receipt = receipts["calibration"]
    _binding(calibration_receipt, "rows", calibration_path, "calibration")
    _binding(calibration_receipt, "thresholds", threshold_path, "calibration")
    calibration_rows = read_jsonl(calibration_path)
    _require(len(calibration_rows) == 2, "fresh calibration is not exactly two rows")
    calibration_ids: list[str] = []
    calibration_served_validations: list[dict[str, Any]] = []
    for index, row in enumerate(calibration_rows):
        where = f"calibration[{index}]"
        key = _validate_mechanistic_row(
            row,
            where,
            split="calibration",
            registration_path=registration_path,
            runtime_frame=runtime_frame,
        )
        _validate_fixture_projection(row, where, registry=registry, mechanistic=True)
        _validate_row_plant_binding(
            row, where, registry=registry, registry_path=registry_path,
            mechanistic=True,
        )
        _require(key[1] == "served", f"calibration[{index}] is not served")
        calibration_served_validations.append(
            validate_served_control(row, f"calibration served row {key[0]}")
        )
        calibration_ids.append(key[0])
    _require(
        calibration_ids == _split_ids(registry, "calibration"),
        "calibration effective row order/split drifted",
    )
    _require(
        calibration_receipt.get("served_control_validations")
        == calibration_served_validations,
        "calibration served-control validation projection drifted",
    )
    _require_row_processes(
        calibration_rows,
        _process_instances(calibration_receipt, "calibration"),
        "calibration",
    )
    _validate_stage_shards(
        calibration_receipt,
        "calibration",
        campaign_dir=campaign_dir,
        amendment_path=amendment_path,
        source_amendment_path=source_amendment_path,
        precollection_authorization_path=precollection_authorization_path,
        plant_registry_path_value=registry_path,
        terminal_amendment_path=terminal_amendment_path,
        registration_path=registration_path,
        runtime_frame_path=runtime_frame_path,
        combined_rows=calibration_rows,
        historical_outputs=historical_outputs,
    )

    thresholds = read_json(threshold_path)
    _require(
        threshold_path.stem.endswith(file_record(threshold_path)["sha256"][:16]),
        "threshold receipt is not content-addressed",
    )
    _require(
        thresholds.get("schema") == "grm.det1_5.thresholds.v1",
        "threshold receipt is not fresh DET1.5",
    )
    _require(thresholds.get("frozen_before_eval") is True, "thresholds were not frozen")
    _binding(thresholds, "registration", registration_path, "thresholds")
    _binding(thresholds, "runtime_frame", runtime_frame_path, "thresholds")
    _binding(thresholds, "race_authorization_amendment", amendment_path, "thresholds")
    _binding(thresholds, "source_amendment", source_amendment_path, "thresholds")
    _require_lineage_authorization(
        thresholds.get("precollection_authorization"), "thresholds",
    )
    _binding(thresholds, "plant_registry", registry_path, "thresholds")
    _binding(thresholds, "terminal_amendment", terminal_amendment_path, "thresholds")
    _binding(thresholds, "calibration_rows", calibration_path, "thresholds")
    fitted = fit_thresholds(calibration_rows)
    _require(_thresholds_equal(thresholds, fitted), "frozen thresholds differ from fresh fit")

    mech_path = campaign_dir / "eval" / "mechanistic_rows.jsonl"
    verbal_path = campaign_dir / "eval" / "verbal_rows.jsonl"
    _binding(receipts["eval_mechanistic"], "rows", mech_path, "eval_mechanistic")
    _binding(receipts["eval_verbal"], "rows", verbal_path, "eval_verbal")
    mechanistic_rows = read_jsonl(mech_path)
    verbal_rows = read_jsonl(verbal_path)
    # DET1.11: both arms must cover the ACHIEVED population exactly — the
    # per-key coverage checks below prove identity, this proves cardinality.
    achieved_pairs = len(_eval_ids(registry))
    _require(
        len(mechanistic_rows) == 2 * achieved_pairs,
        f"mechanistic eval is not exactly {achieved_pairs}+{achieved_pairs}",
    )
    _require(
        len(verbal_rows) == 2 * achieved_pairs,
        f"verbal eval is not exactly {achieved_pairs}+{achieved_pairs}",
    )

    expected_keys = _expected_eval_keys(registry)
    mechanistic_keys: list[tuple[str, str]] = []
    for index, row in enumerate(mechanistic_rows):
        where = f"mechanistic_eval[{index}]"
        key = _validate_mechanistic_row(
            row,
            where,
            split="eval",
            registration_path=registration_path,
            runtime_frame=runtime_frame,
        )
        _validate_fixture_projection(row, where, registry=registry, mechanistic=True)
        _validate_row_plant_binding(
            row, where, registry=registry, registry_path=registry_path,
            mechanistic=True,
        )
        mechanistic_keys.append(key)
        if key[1] == "served":
            validate_served_control(row, f"evaluation served row {key[0]}")
        else:
            _require(_planted_miss_valid(row), f"invalid DET-G0 planted row: {key}")
    _require(
        mechanistic_keys == _expected_eval_order(registry)
        and set(mechanistic_keys) == expected_keys,
        "mechanistic eval coverage differs from the achieved population",
    )
    repeated_g0 = validate_g0_rows(
        mechanistic_rows, expected_pairs=achieved_pairs)
    effective_registration = effective_registration_projection(registration, registry)
    split_projection = validate_split(
        effective_registration, calibration_rows, mechanistic_rows,
        achieved_eval_ids=_eval_ids(registry),
    )
    _require(
        receipts["eval_mechanistic"].get("repeated_g0_validation")
        == repeated_g0,
        "eval_mechanistic repeated G0 projection drifted",
    )
    _require(
        receipts["eval_mechanistic"].get("split_validation")
        == split_projection,
        "eval_mechanistic split projection drifted",
    )
    mechanistic_processes = _process_instances(
        receipts["eval_mechanistic"], "eval_mechanistic"
    )
    _require_row_processes(
        mechanistic_rows, mechanistic_processes, "eval_mechanistic"
    )
    _validate_stage_shards(
        receipts["eval_mechanistic"],
        "eval_mechanistic",
        campaign_dir=campaign_dir,
        amendment_path=amendment_path,
        source_amendment_path=source_amendment_path,
        precollection_authorization_path=precollection_authorization_path,
        plant_registry_path_value=registry_path,
        terminal_amendment_path=terminal_amendment_path,
        registration_path=registration_path,
        runtime_frame_path=runtime_frame_path,
        combined_rows=mechanistic_rows,
        historical_outputs=historical_outputs,
    )

    verbal_keys: list[tuple[str, str]] = []
    for index, (mechanistic, row) in enumerate(zip(mechanistic_rows, verbal_rows)):
        where = f"verbal_eval[{index}]"
        verbal_keys.append(
            _validate_verbal_row(
                row,
                where,
                registration_path=registration_path,
                runtime_frame=runtime_frame,
            )
        )
        _validate_fixture_projection(row, where, registry=registry, mechanistic=False)
        _validate_row_plant_binding(
            row, where, registry=registry, registry_path=registry_path,
            mechanistic=False,
        )
        for binding in (
            "plant_target_id", "plant_alias_ids", "plant_target_source",
            "plant_registry", "plant_entry_sha256",
        ):
            _require(
                row.get(binding) == mechanistic.get(binding),
                f"verbal_eval[{index}] plant binding differs from mechanistic row: "
                f"{binding}",
            )
        mechanistic_completed = int(mechanistic["mechanistic_completed_unix_ns"])
        _require(
            int(row.get("mechanistic_completed_unix_ns", -1))
            == mechanistic_completed,
            f"verbal_eval[{index}] does not bind matching mechanistic completion",
        )
        _require(
            int(row.get("verbal_started_unix_ns", -1)) > mechanistic_completed,
            f"verbal_eval[{index}] did not start after mechanistic completion",
        )
        selection = (mechanistic.get("signals") or {}).get(
            "ladder_attempt_selection"
        ) or {}
        mechanistic_ordinal = selection.get(
            "selected_call_ordinal", selection.get("production_trip")
        )
        _require(
            isinstance(mechanistic_ordinal, int)
            and not isinstance(mechanistic_ordinal, bool),
            f"mechanistic_eval[{index}] lacks a selected ladder ordinal",
        )
        mechanistic_mounts = list(mechanistic.get("mounted_ids") or ())
        _require(
            row.get("selected_attempt_ordinal") == mechanistic_ordinal
            and row.get("mechanistic_selected_attempt_ordinal")
            == mechanistic_ordinal,
            f"verbal_eval[{index}] selected a different mechanistic ladder rung",
        )
        _require(
            row.get("mounted_ids") == mechanistic_mounts
            and row.get("mechanistic_mounted_ids") == mechanistic_mounts,
            f"verbal_eval[{index}] mounted set differs from mechanistic row",
        )
    _require(
        verbal_keys == mechanistic_keys,
        "isolated chronological D-VERB rows do not have identical ordered coverage",
    )

    verbal_receipt = receipts["eval_verbal"]
    verbal_processes = _process_instances(verbal_receipt, "eval_verbal")
    _require(
        set(mechanistic_processes).isdisjoint(verbal_processes),
        "D-VERB worker processes overlap mechanistic worker processes",
    )
    _require(
        verbal_receipt.get("mechanistic_process_instance_sha256s")
        == mechanistic_processes,
        "D-VERB receipt does not bind the ordered mechanistic process list",
    )
    _require_row_processes(verbal_rows, verbal_processes, "eval_verbal")
    verbal_shards = _validate_stage_shards(
        verbal_receipt,
        "eval_verbal",
        campaign_dir=campaign_dir,
        amendment_path=amendment_path,
        source_amendment_path=source_amendment_path,
        precollection_authorization_path=precollection_authorization_path,
        plant_registry_path_value=registry_path,
        terminal_amendment_path=terminal_amendment_path,
        registration_path=registration_path,
        runtime_frame_path=runtime_frame_path,
        combined_rows=verbal_rows,
        historical_outputs=historical_outputs,
    )
    chronology_input = {**verbal_receipt, "rows": verbal_rows}
    chronology_projection = validate_verbal_chronology(chronology_input)
    _require(
        verbal_receipt.get("chronology_validation") == chronology_projection,
        "eval_verbal chronology projection drifted",
    )
    chronological = verbal_receipt.get("chronological_session")
    _require(
        isinstance(chronological, list)
        and len(chronological) == len(verbal_processes),
        "D-VERB chronological session receipts do not match worker processes",
    )
    _validate_verbal_sessions(
        chronological,
        verbal_shards,
        verbal_processes,
        campaign_dir=campaign_dir,
    )

    first_eval = min(int(row["evaluation_started_unix_ns"]) for row in mechanistic_rows)
    first_verbal = min(int(row["evaluation_started_unix_ns"]) for row in verbal_rows)
    first_g0 = min(int(row["evaluation_started_unix_ns"]) for row in g0_rows)
    calibration_times = [int(row["evaluation_started_unix_ns"]) for row in calibration_rows]
    threshold_time = int(thresholds["created_unix_ns"])
    registry_time = _utc_unix_ns(registry.get("created_utc"), "plant registry created_utc")
    plant_receipt_time = _utc_unix_ns(
        receipts["plant_registration"].get("created_utc"),
        "plant-registration receipt created_utc",
    )
    plant_marker_time = _utc_unix_ns(
        stage_markers["plant_registration"].get("created_utc"),
        "plant-registration marker created_utc",
    )
    terminal_time = _utc_unix_ns(
        read_json(terminal_amendment_path).get("created_utc"),
        "terminal amendment created_utc",
    )
    _require(
        registry_time <= plant_receipt_time <= plant_marker_time <= terminal_time
        < min(first_g0, min(calibration_times), first_eval, first_verbal),
        "plant registry/terminal amendment were not frozen before G0 and evaluation",
    )
    _require(
        int(registration["registered_unix_ns"]) < min(calibration_times)
        and max(calibration_times) < threshold_time < min(first_eval, first_verbal),
        "calibration/threshold/evaluation chronology is not frozen pre-eval",
    )

    merged = merge_detector_rows(mechanistic_rows, verbal_rows)
    table, winners, legacy_verdict = race_metrics(merged, thresholds)
    verdict = prediction_verdict(table)
    _require(
        legacy_verdict == verdict,
        f"registered race arithmetic disagrees: {legacy_verdict} != {verdict}",
    )
    _require(verdict in VERDICTS, "forbidden prediction-verdict vocabulary")
    prediction_sentence = _prediction_sentence(verdict, achieved_pairs)

    source_rows = [
        file_record(g0_rows_path),
        file_record(calibration_path),
        file_record(mech_path),
        file_record(verbal_path),
    ]
    analysis_dir = campaign_dir / "analysis"
    existing_analysis = sorted(analysis_dir.glob("analysis_receipt_*.json"))
    _require(
        len(existing_analysis) <= 1,
        "analysis namespace contains duplicate receipts",
    )
    created_utc = (
        str(read_json(existing_analysis[0]).get("created_utc", ""))
        if existing_analysis
        else utc_now()
    )
    _require(bool(created_utc), "existing analysis receipt lacks created_utc")
    analysis = {
        "schema": "grm.det1_5.analysis.v1",
        "created_utc": created_utc,
        "status": "PASS",
        "gates": {
            "ZERO": "PASS_EXACT_CANONICAL_VALUE_BYTES",
            "CROSS_PROCESS_ZERO": "PASS",
            "DET-G0": "PASS",
            "DET-G1": "PASS_ALL_HOOKS_BYTE_IDENTITY",
            "DET-G2": "PASS_FROZEN_PRE_EVAL",
            "DET-G3": "PASS_FOUR_ROWS_AND_VERDICT",
        },
        "gate_details": {
            "DET1.7_plant_registration": {
                "registry_validation": receipts["plant_registration"][
                    "registry_validation"
                ],
                "registry_payload_sha256": registry["registry_payload_sha256"],
                "registry_file_sha256": file_record(registry_path)["sha256"],
                "substitution_policy": registry["substitution_policy"],
                "substitution_table": registry["substitutions"],
                "per_turn_table": _plant_registration_table(registry),
                "substitutions": _receipt_substitutions(registry),
                "campaign_r4_lived_control_finding": file_record(finding_path),
                "campaign_r4_lived_control_finding_validation": (
                    finding_validation
                ),
                "frozen_before_g0_and_eval": True,
            },
            "DET-G0_dedicated": validate_g0_rows(
                g0_rows, expected_pairs=len(_eval_ids(registry))),
            "DET-G0_eval_repeat": repeated_g0,
            "DET-G1": {
                "byte_identity": dict(g1["byte_identity"]),
                "active_hooks": list(g1["active_hooks"]),
            },
            "DET-G2": split_projection,
        },
        "table": table,
        "race_winners": winners,
        "prediction_verdict": verdict,
        "prediction_verdict_sentence": prediction_sentence,
        "registration": file_record(registration_path),
        "runtime_frame": file_record(runtime_frame_path),
        "zero_gate": file_record(zero_receipt_path),
        "race_authorization_amendment": file_record(amendment_path),
        "source_amendment": file_record(source_amendment_path),
        "det1_7_order": file_record(DET1_7_ORDER),
        "det1_9_amendment_order": file_record(DET1_9_ORDER),
        "precollection_authorization": file_record(
            precollection_authorization_path
        ),
        "campaign_r4_lived_control_finding": file_record(finding_path),
        "campaign_r4_lived_control_finding_validation": finding_validation,
        "campaign_r4_lived_control_finding_text": finding_text,
        "plant_registry": file_record(registry_path),
        "terminal_amendment": file_record(terminal_amendment_path),
        "thresholds": file_record(threshold_path),
        "fresh_rows": source_rows,
        "stage_receipts": {
            stage: file_record(paths[1]) for stage, paths in stage_files.items()
        },
        "historical_rows_consumed": False,
        "pre_det1_7_shard_outputs_consumed": False,
        "ignored_pre_det1_7_shard_outputs": historical_outputs,
    }
    if existing_analysis:
        analysis_path = existing_analysis[0]
        stored = read_json(analysis_path)
        # DET1.11: a receipt frozen in an earlier round names the envelope
        # that governed THAT round.  Every other field must still match
        # exactly; the authorization is checked for lineage membership.
        _require_lineage_authorization(
            stored.get("precollection_authorization"), "existing analysis",
        )
        _require(
            {k: v for k, v in stored.items()
             if k != "precollection_authorization"}
            == {k: v for k, v in analysis.items()
                if k != "precollection_authorization"},
            "existing analysis receipt no longer matches validated inputs",
        )
        _require(
            analysis_path.stem.endswith(file_record(analysis_path)["sha256"][:16]),
            "existing analysis receipt is not content-addressed",
        )
    else:
        analysis_path = write_content_addressed(
            analysis_dir, "analysis_receipt", analysis
        )

    excluded_ids = [
        str(row.get("fixture_id"))
        for row in registry.get("excluded_slots") or ()
    ]
    table_text = race_table_markdown(
        table,
        achieved_pairs=achieved_pairs,
        excluded_fixture_ids=excluded_ids,
    )
    sentence = prediction_sentence
    lines = [
        "# GRM-DET1.5 fork-substrate demand-detector race",
        "",
        "All ordered fork-substrate stages are complete. Historical DET1 rows were not consumed.",
        "",
        "## Gates",
        "",
        "| Gate | Status |",
        "|---|:---:|",
        "| DET1.4 ZERO | PASS_EXACT_CANONICAL_VALUE_BYTES |",
        "| Cross-process ZERO | PASS |",
        "| DET-G0 | PASS |",
        "| DET-G1 | PASS_ALL_HOOKS_BYTE_IDENTITY |",
        "| DET-G2 | PASS_FROZEN_PRE_EVAL |",
        "| DET-G3 | PASS_FOUR_ROWS_AND_VERDICT |",
        "",
        f"## Four-row registered race table ({_achieved_clause(achieved_pairs)})",
        "",
        table_text,
        "",
        sentence,
        "",
        f"RACE WINNER(S): {', '.join(winners) if winners else 'NONE (no viable detector)' }.",
        "",
        "Latency medians are zero-based generated-token indices over triggered planted misses.",
        "",
        "## DET1.7 lived-admission plant registry",
        "",
        f"Registry payload SHA-256: `{registry['registry_payload_sha256']}`",
        f"Registry file SHA-256: `{file_record(registry_path)['sha256']}`",
        "",
        "| Base slot | Effective fixture | Split | Branch | Lived mounts | Target | Aliases | Rule | Substitution |",
        "|---|---|---|---|---|---:|---|---|---|",
        *[
            "| {base_fixture_id} | {effective_fixture_id} | {split} | "
            "{policy_branch} | {actual_lived_mounts} | {plant_target_id} | "
            "{plant_alias_ids} | {selection_rule} | {substitution} |".format(
                **row
            )
            for row in _plant_registration_table(registry)
        ],
        "",
        f"Substitutions: `{json.dumps(_receipt_substitutions(registry), sort_keys=True)}`",
        "",
        "## DET1.9 recorded finding",
        "",
        *finding_text,
        "",
        "Finding status: `OPEN_POST_RACE_INVESTIGATION`; causal investigation was not performed here.",
        "",
        "## Evidence",
        "",
        f"- Race authorization: `{file_record(amendment_path)['path']}`",
        f"- DET1.6 terminal source amendment: `{file_record(source_amendment_path)['path']}`",
        f"- DET1.7 precollection authorization: `{file_record(precollection_authorization_path)['path']}`",
        f"- DET1.9 amendment order: `{file_record(DET1_9_ORDER)['path']}`",
        f"- Campaign-r4 lived-control finding: `{file_record(finding_path)['path']}`",
        f"- DET1.7 plant registration: `{file_record(stage_files['plant_registration'][1])['path']}`",
        f"- DET1.7 plant registry: `{file_record(registry_path)['path']}`",
        f"- DET1.7 terminal amendment: `{file_record(terminal_amendment_path)['path']}`",
        f"- Cross-process ZERO: `{file_record(stage_files['cross_process_zero'][1])['path']}`",
        f"- DET-G0: `{file_record(stage_files['g0'][1])['path']}`",
        f"- DET-G1: `{file_record(stage_files['g1'][1])['path']}`",
        f"- Calibration thresholds: `{file_record(threshold_path)['path']}`",
        f"- Mechanistic rows: `{file_record(mech_path)['path']}`",
        f"- Isolated chronological D-VERB rows: `{file_record(verbal_path)['path']}`",
        f"- Analysis receipt: `{file_record(analysis_path)['path']}`",
        "",
        "## Anything not done",
        "",
        "The open t30/t33 post-race causal investigation was intentionally not performed under DET1.9.",
        "",
    ]
    report_path = analysis_dir / "GRM_DET1_5_REPORT.md"
    # DET1.11: the report is a RENDERING of the analysis receipt, not
    # independent evidence.  When the envelope advances, the receipt stays
    # valid under the lineage rule but the rendered evidence list names the
    # newer envelope; regenerate rather than refuse.  The receipt itself is
    # still write-once and content-addressed — that is what is immutable.
    _write_text_rendering(report_path, "\n".join(lines))
    return analysis_path


def selftest() -> dict[str, Any]:
    def row(name: str, f1: float, viable: bool) -> dict[str, Any]:
        return {"detector": name, "f1": f1, "viable": viable}

    supported = [
        row("D-LQR", 0.9, True),
        row("D-NGH", 0.8, True),
        row("D-ENT", 0.7, False),
        row("D-VERB", 0.2, False),
    ]
    partial = [
        row("D-LQR", 0.8, True),
        row("D-NGH", 0.9, True),
        row("D-ENT", 0.7, False),
        row("D-VERB", 0.2, False),
    ]
    refuted = [
        row("D-LQR", 0.9, True),
        row("D-NGH", 0.8, True),
        row("D-ENT", 0.7, False),
        row("D-VERB", 0.9, False),
    ]
    _require(prediction_verdict(supported) == "SUPPORTED", "SUPPORTED selftest failed")
    _require(prediction_verdict(partial) == "PARTIAL", "PARTIAL selftest failed")
    _require(prediction_verdict(refuted) == "REFUTED", "REFUTED selftest failed")
    uncovered = [
        row("D-LQR", 0.9, True),
        row("D-NGH", 0.1, True),
        row("D-ENT", 0.3, False),
        row("D-VERB", 0.2, False),
    ]
    try:
        prediction_verdict(uncovered)
    except DETError:
        pass
    else:
        raise DETError("uncovered adjudication corner did not fail closed")
    return {
        "schema": "grm.det1_5.analyze_selftest.v1",
        "status": "PASS",
        "verdict_vocabulary": list(VERDICTS),
        "uncovered_corner": "REFUSED",
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("report", "selftest"))
    parser.add_argument("--run-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "selftest":
            print(json.dumps(selftest(), sort_keys=True))
            return 0
        if args.run_dir is None:
            raise DETError("--run-dir is required for report")
        path = analyze(args.run_dir)
        print(path)
        return 0
    except DETError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
