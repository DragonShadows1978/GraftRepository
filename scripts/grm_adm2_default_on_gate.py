#!/usr/bin/env python3
"""Registered GRM-ADM2.2 A-DEC default-on acceptance runner.

CPU-only invocation validates the frozen ADM1.3 and CMC receipts and emits
the complete pre-live re-registration enumeration. ``--stage all`` creates a
new run directory and executes each GPU-bearing leg under the repository GPU
flock, with repo-root ``PYTHONPATH``. Existing artifacts are never overwritten.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.grm_admission import (  # noqa: E402
    FROZEN_RULE_SHA256,
    MARGIN_THRESHOLD,
)


ARTIFACT_ROOT = ROOT / "artifacts" / "grm_adm2"
ADM13_RUN = (
    ROOT / "artifacts" / "grm_adm1"
    / "adm1_3_prod_run_20260830T161236Z_3eaccb"
)
ADM13_ADJUDICATION = (
    ADM13_RUN / "dual_frame_adjudication_c83a45c60e9fe50b.json"
)
FROZEN_RULE = (
    ROOT / "artifacts" / "grm_adm1" / "run_20260830T093538Z_3defaa"
    / "decisiveness_rule_c304609f81475bd2.json"
)
CMC_G0 = (
    ROOT / "artifacts" / "grm_cmc1"
    / "live_arms_20260830T092447Z_4056005"
    / "g0_95d6dcb605db1bf0.json"
)
CMC_ADJUDICATION = (
    ROOT / "artifacts" / "grm_cmc2"
    / "live_sweep_20260830T170021Z_4124360"
    / "adjudication_ba6821ccf299ac17.json"
)
L2_LEGACY_DEFAULT = (
    ROOT / "artifacts" / "grm_supersession" / "diag_resolve_only.jsonl"
)
DET1_ORDER = ROOT / "orders" / "GRM_DET1_DEMAND_DETECTOR_RACE.md"

EXPECTED_HASHES = {
    FROZEN_RULE: FROZEN_RULE_SHA256,
    ADM13_ADJUDICATION: (
        "c83a45c60e9fe50b76e63fd63e680f6dab1aaab47628cf030057a8a655600da9"
    ),
    CMC_G0: (
        "95d6dcb605db1bf0c16d2c54f70ba531040b6a7f799e50f0dcac0837c982e39c"
    ),
    CMC_ADJUDICATION: (
        "ba6821ccf299ac170ff27e685e91af5c76c82ba09ebc62e1150e6b40714788e3"
    ),
    L2_LEGACY_DEFAULT: (
        "8053f9c3437bb6ff4c0e81533159b2e5870c32ca20f895e799ca61f8585d2a78"
    ),
}

STAGES = (
    "lookup_default",
    "lookup_escape",
    "supersession_default",
    "supersession_escape",
    "e2e_default",
    "e2e_escape",
    "p4_default",
    "p4_escape",
    "suite_shared",
    "suite_full",
)
MAX_LEASE_SECONDS = 590
ENVELOPE_KEYS = frozenset(("schema", "record_type", "dialect", "mode"))
ADMISSION_INFO_KEYS = frozenset((
    "admission_policy",
    "admission_policy_branch",
    "admission_rank_plan",
    "admission_identifier_hit_count",
    "admission_identified_candidates",
    "admission_route_margin_1_2",
    "admission_route_margin_evaluated",
    "admission_margin_threshold",
    "admission_rule_sha256",
))


class GateError(RuntimeError):
    pass


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": relative(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GateError(f"{path} is not a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise GateError(f"{path}:{line_no} is not a JSON object")
        rows.append(value)
    return rows


def write_exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def canonical_rows_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(
            dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8") + b"\n"
        for row in rows
    )


def _adm_rows() -> list[dict[str, Any]]:
    rows = []
    for stage in ("lookup", "diag", "e2e", "p4"):
        for row in read_jsonl(ADM13_RUN / stage / "adm_rows.jsonl"):
            rows.append({**row, "_stage": stage})
    if len(rows) != 53:
        raise GateError(f"ADM1.3 F-PROD expected 53 rows, observed {len(rows)}")
    return rows


def _physical_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        stage = row["_stage"]
        fixture = row["fixture"]
        if stage == "lookup" or stage == "diag":
            selected.append(dict(row))
        elif stage == "e2e" and fixture == "E2E-34-FULL-A-k3-ANCHOR":
            selected.append(dict(row))
        elif stage == "p4" and fixture == "P4-REPLICATION-FULL-A-k3-ANCHOR":
            selected.append(dict(row))
    if len(selected) != 49:
        raise GateError(
            f"ADM2 physical re-registration expected 49 turns, got {len(selected)}"
        )
    return selected


def changed_keys(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[str]:
    return sorted(
        key for key in set(left) | set(right) if left.get(key) != right.get(key)
    )


def _find_probe_answers(value: Any, probe_id: str) -> list[str]:
    answers = []
    if isinstance(value, dict):
        if value.get("probe_id") == probe_id:
            for key in ("answer", "answer_text"):
                if isinstance(value.get(key), str):
                    answers.append(value[key])
        for child in value.values():
            answers.extend(_find_probe_answers(child, probe_id))
    elif isinstance(value, list):
        for child in value:
            answers.extend(_find_probe_answers(child, probe_id))
    return answers


def registered_acceptance() -> dict[str, Any]:
    observed_hashes = {relative(path): sha256_file(path) for path in EXPECTED_HASHES}
    drift = [
        relative(path) for path, expected in EXPECTED_HASHES.items()
        if observed_hashes[relative(path)] != expected
    ]
    if drift:
        raise GateError(f"registered receipt hash drift: {drift}")

    rule = read_json(FROZEN_RULE)
    if float(rule["margin_threshold"]) != MARGIN_THRESHOLD:
        raise GateError("frozen margin threshold drifted")
    adjudication = read_json(ADM13_ADJUDICATION)
    if adjudication["adoption"]["recommendation"] != "ADOPT-RECOMMENDED":
        raise GateError("ADM1.3 receipt is not ADOPT-RECOMMENDED")

    rows = _adm_rows()
    answer_changes = []
    registrations = []
    for row in _physical_rows(rows):
        legacy = row["arms"]["A-k3"]
        adopted = row["arms"]["A-DEC"]
        if legacy["answer"] != adopted["answer"]:
            answer_changes.append({
                "stage": row["_stage"],
                "probe_id": row["probe_id"],
                "before": legacy["answer"],
                "after": adopted["answer"],
                "before_correct": bool(legacy["answer_correct"]),
                "after_correct": bool(adopted["answer_correct"]),
            })
        registrations.append({
            "stage": row["_stage"],
            "probe_id": row["probe_id"],
            "fixture": row["fixture"],
            "policy_branch": adopted["policy_branch"],
            "legacy_k3_rank_plan": list(legacy["rank_plan"]),
            "adec_rank_plan": list(adopted["rank_plan"]),
            "legacy_final_mounts": list(legacy["mounted_nodes"]),
            "adec_final_mounts": list(adopted["mounted_nodes"]),
            "answer_changed": legacy["answer"] != adopted["answer"],
            "registered_changed_result_fields": changed_keys(legacy, adopted),
            "registered_changed_arena_fields": changed_keys(
                legacy.get("arena_info") or {}, adopted.get("arena_info") or {}),
        })

    expected_change = [{
        "stage": "lookup",
        "probe_id": "supersession:praxis_fresh",
        "before": "The current Praxis dock value is Raven-9-Ivory.",
        "after": "The current Praxis dock value is Quartz-8-Jade.",
        "before_correct": False,
        "after_correct": True,
    }]
    if answer_changes != expected_change:
        raise GateError(f"answer-change enumeration is not exactly praxis: {answer_changes}")
    if any(
        row["legacy_k3_rank_plan"] == row["adec_rank_plan"]
        for row in registrations
    ):
        raise GateError("physical registration includes a non-deviating plan")

    def correct_count(stage: str, fixture: str) -> list[int]:
        selected = [
            row for row in rows
            if row["_stage"] == stage and row["fixture"] == fixture
        ]
        return [
            sum(bool(row["arms"]["A-DEC"]["answer_correct"]) for row in selected),
            len(selected),
        ]

    acceptance_numbers = {
        "CORPUS-100": correct_count("lookup", "CORPUS-100"),
        "SUPERSESSION-FRESH-CONTROLS": correct_count(
            "lookup", "SUPERSESSION-FRESH-CONTROLS"),
        "E2E-34": correct_count("e2e", "E2E-34-FULL-A-k3-ANCHOR"),
        "P4-34": correct_count("p4", "P4-REPLICATION-FULL-A-k3-ANCHOR"),
    }
    expected_numbers = {
        "CORPUS-100": [20, 20],
        "SUPERSESSION-FRESH-CONTROLS": [2, 2],
        "E2E-34": [9, 9],
        "P4-34": [9, 9],
    }
    if acceptance_numbers != expected_numbers:
        raise GateError(f"ADM1.3 A-DEC acceptance drift: {acceptance_numbers}")

    cmc_answers = _find_probe_answers(read_json(CMC_G0), "praxis_fresh")
    if "The current Praxis dock value is Raven-9-Ivory." not in cmc_answers:
        raise GateError("CMC G0 no longer contains the praxis wrong-read")
    det_text = " ".join(DET1_ORDER.read_text(encoding="utf-8").split())
    det_phrases = (
        "admission = whatever production default holds",
        "record exact flags in every receipt",
        "planted misses make admission policy a non-variable here",
    )
    if not all(phrase in det_text for phrase in det_phrases):
        raise GateError("DET1 production-frame recording assumption drifted")

    return {
        "schema": "grm.adm2.registered_acceptance.v1",
        "status": "PASS",
        "flag": {
            "constructor": "decisive_admission",
            "cli": ["--adm-decisive", "--no-adm-decisive"],
            "env": "GRM_ADM_DECISIVE",
            "default": True,
            "legacy_escape": "GRM_ADM_DECISIVE=0",
        },
        "frozen_rule": {
            **file_record(FROZEN_RULE),
            "margin_threshold": MARGIN_THRESHOLD,
        },
        "source_receipts": {
            "adm1_3": file_record(ADM13_ADJUDICATION),
            "cmc_wrong_read": file_record(CMC_G0),
            "cmc_value_blending_adjudication": file_record(CMC_ADJUDICATION),
            "legacy_l2_default": file_record(L2_LEGACY_DEFAULT),
        },
        "acceptance_numbers": acceptance_numbers,
        "answer_changes": answer_changes,
        "answer_change_count": len(answer_changes),
        "re_registered_physical_turn_count": len(registrations),
        "re_registered_turns": registrations,
        "duplicate_metric_rows_retained": 4,
        "det1_coordination": {
            "pin_required": False,
            "reason": (
                "DET1 inherits the production admission default at run time, "
                "must record the exact flag in every receipt, and plants the "
                "withheld winner so admission policy is registered non-variable."
            ),
            "order": file_record(DET1_ORDER),
        },
        "registered_receipt_hashes": observed_hashes,
    }


def _expected_physical(stage: str) -> list[dict[str, Any]]:
    return [row for row in _physical_rows(_adm_rows()) if row["_stage"] == stage]


def _strip_admission_info(info: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in info.items() if key not in ADMISSION_INFO_KEYS}


def run_lookup(run_dir: Path, *, decisive: bool) -> dict[str, Any]:
    from scripts.grm_adm1_analysis import corpus100_fixture_rows
    from scripts.grm_adm1_3_dual_frame import _supersession_arena
    from scripts.grm_adm1_gpu import _corpus_bank, _load_minicpm

    stage = "lookup_default" if decisive else "lookup_escape"
    stage_dir = run_dir / stage
    stage_dir.mkdir(parents=True, exist_ok=False)
    expected = {row["probe_id"]: row for row in _expected_physical("lookup")}
    actual_rows = []
    model, tokenizer, model_info, tc = _load_minicpm()
    try:
        arena, _meta = _corpus_bank(model, tokenizer)
        arena.decisive_admission = bool(decisive)
        for probe in corpus100_fixture_rows():
            expected_row = expected[str(probe["probe_id"])]
            answer, info = arena.step(
                str(probe["question"]), ngen=40, deposit=False, max_trips=2)
            actual_rows.append(_live_lookup_row(
                expected_row, answer, info, arena.cur_mounts,
                decisive=decisive))
        del arena
        if hasattr(tc, "empty_cache"):
            tc.empty_cache()

        arena, fixture, _node_to_idx, _values, registered = _supersession_arena(
            model, tokenizer)
        arena.decisive_admission = bool(decisive)
        for probe in fixture["probes"]:
            probe_id = f"supersession:{probe['probe_id']}"
            if probe_id not in expected:
                probe_id = str(registered[probe["probe_id"]]["probe_id"])
            expected_row = expected[probe_id]
            answer, info = arena.step(
                str(probe["question"]), ngen=32, deposit=False, max_trips=0)
            actual_rows.append(_live_lookup_row(
                expected_row, answer, info, arena.cur_mounts,
                decisive=decisive))
    finally:
        del model
        if hasattr(tc, "empty_cache"):
            tc.empty_cache()

    if len(actual_rows) != 22 or not all(row["match"] for row in actual_rows):
        failures = [row["probe_id"] for row in actual_rows if not row["match"]]
        raise GateError(f"{stage} mismatch: count={len(actual_rows)} failures={failures}")
    receipt = stage_dir / "lookup_receipts.jsonl"
    write_jsonl(receipt, actual_rows)
    result = {
        "schema": "grm.adm2.lookup_stage.v1",
        "stage": stage,
        "status": "PASS",
        "decisive_admission": bool(decisive),
        "model_info": str(model_info),
        "row_count": len(actual_rows),
        "answer_correct": [
            sum(bool(row["answer_correct"]) for row in actual_rows),
            len(actual_rows),
        ],
        "corpus100": [
            sum(row["fixture"] == "CORPUS-100" and row["answer_correct"]
                for row in actual_rows),
            sum(row["fixture"] == "CORPUS-100" for row in actual_rows),
        ],
        "supersession_fresh": [
            sum(row["fixture"] == "SUPERSESSION-FRESH-CONTROLS"
                and row["answer_correct"] for row in actual_rows),
            sum(row["fixture"] == "SUPERSESSION-FRESH-CONTROLS"
                for row in actual_rows),
        ],
        "receipt": file_record(receipt),
        "transcript_sha256": sha256_bytes(canonical_rows_bytes(actual_rows)),
    }
    write_exclusive_json(stage_dir / "stage_complete.json", result)
    return result


def _live_lookup_row(
    expected_row: Mapping[str, Any],
    answer: str,
    info: Mapping[str, Any],
    mounts: Sequence[int],
    *,
    decisive: bool,
) -> dict[str, Any]:
    arm = expected_row["arms"]["A-DEC" if decisive else "A-k3"]
    compact = dict(info or {})
    base_info = _strip_admission_info(compact)
    answer_match = str(answer) == str(arm["answer"])
    mount_match = [int(value) for value in mounts] == [
        int(value) for value in arm["mounted_nodes"]
    ]
    arena_match = base_info == dict(arm.get("arena_info") or {})
    policy_match = (
        compact.get("admission_policy") == "A-DEC"
        and compact.get("admission_policy_branch")
        == arm.get("policy_branch")
        and compact.get("admission_rank_plan") == arm.get("rank_plan")
        and compact.get("admission_rule_sha256") == FROZEN_RULE_SHA256
        if decisive else not (set(compact) & ADMISSION_INFO_KEYS)
    )
    return {
        "probe_id": expected_row["probe_id"],
        "fixture": expected_row["fixture"],
        "answer": str(answer),
        "answer_correct": bool(arm["answer_correct"] and answer_match),
        "mounted_nodes": [int(value) for value in mounts],
        "arena_info": compact,
        "expected_arm": "A-DEC" if decisive else "A-k3",
        "answer_match": bool(answer_match),
        "mount_match": bool(mount_match),
        "arena_match_after_admission_fields_removed": bool(arena_match),
        "policy_receipt_match": bool(policy_match),
        "match": bool(answer_match and mount_match and arena_match and policy_match),
    }


def _probe_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows
            if row.get("record_type") == "supersession_probe_receipt"]


def _supersession_answer_deltas(
    anchor: Mapping[str, Mapping[str, Any]],
    live: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Separate semantic answer changes from byte-only phrasing changes."""
    if set(anchor) != set(live):
        raise GateError(
            "supersession probe set drift: "
            f"anchor={sorted(anchor)} live={sorted(live)}")

    semantic_changes = []
    byte_only_changes = []
    for probe_id in sorted(anchor):
        before = anchor[probe_id]
        after = live[probe_id]
        delta = {
            "probe_id": probe_id,
            "before_answer": str(before["answer_text"]),
            "after_answer": str(after["answer_text"]),
            "before_classification": before.get("classification"),
            "after_classification": after.get("classification"),
            "before_extracted_value": before.get("classification_match"),
            "after_extracted_value": after.get("classification_match"),
        }
        semantic_changed = (
            delta["before_classification"] != delta["after_classification"]
            or delta["before_extracted_value"] != delta["after_extracted_value"]
        )
        answer_bytes_changed = (
            delta["before_answer"] != delta["after_answer"])
        if semantic_changed:
            semantic_changes.append(delta)
        elif answer_bytes_changed:
            byte_only_changes.append(delta)
    return {
        "semantic_changes": semantic_changes,
        "byte_only_changes": byte_only_changes,
    }


def _assert_adm2_2_supersession_enumeration(
    anchor: Mapping[str, Mapping[str, Any]],
    live: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Enforce the two explicitly registered ADM2.2 answer deltas."""
    deltas = _supersession_answer_deltas(anchor, live)
    required_semantic_change = [{
        "probe_id": "praxis_fresh",
        "before_answer": "The current Praxis dock value is Raven-9-Ivory.",
        "after_answer": "The current Praxis dock value is Quartz-8-Jade.",
        "before_classification": "wrong-fact",
        "after_classification": "correct",
        "before_extracted_value": "raven-9-ivory",
        "after_extracted_value": "quartz-8-jade",
    }]
    allowed_byte_only_change = [{
        "probe_id": "orion_current",
        "before_answer": "The current Orion pin value is Kestrel-9-Tango.",
        "after_answer": (
            "The current Orion pin value is Kestrel-9-Tango, replacing "
            "Auric-4-Alpha."
        ),
        "before_classification": "correct",
        "after_classification": "correct",
        "before_extracted_value": "kestrel-9-tango",
        "after_extracted_value": "kestrel-9-tango",
    }]
    if deltas["semantic_changes"] != required_semantic_change:
        raise GateError(
            "supersession semantic-change enumeration drift: "
            f"{deltas['semantic_changes']}")
    if deltas["byte_only_changes"] != allowed_byte_only_change:
        raise GateError(
            "supersession byte-only enumeration drift: "
            f"{deltas['byte_only_changes']}")
    return deltas


def _sup_projection(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return canonical_rows_bytes(
        {key: value for key, value in row.items() if key not in ENVELOPE_KEYS}
        for row in _probe_rows(rows)
    )


def run_supersession(run_dir: Path, *, decisive: bool) -> dict[str, Any]:
    stage = "supersession_default" if decisive else "supersession_escape"
    stage_dir = run_dir / stage
    stage_dir.mkdir(parents=True, exist_ok=False)
    receipt = stage_dir / "battery.jsonl"
    log = stage_dir / "battery.log"
    env = _base_env()
    if decisive:
        env.pop("GRM_ADM_DECISIVE", None)
    else:
        env["GRM_ADM_DECISIVE"] = "0"
    command = [
        sys.executable,
        "tests/test_grm_supersession_battery.py",
        "--run-gpu",
        "--receipt-jsonl", str(receipt),
    ]
    with log.open("xb") as handle:
        completed = subprocess.run(
            command, cwd=ROOT, env=env, stdout=handle,
            stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        raise GateError(f"{stage} failed rc={completed.returncode}; see {log}")
    rows = read_jsonl(receipt)
    probes = _probe_rows(rows)
    if len(probes) != 5:
        raise GateError(f"{stage} expected 5 probes, got {len(probes)}")
    counts = {
        name: sum(row["classification"] == name for row in probes)
        for name in ("correct", "stale", "wrong-fact")
    }
    anchor_rows = read_jsonl(L2_LEGACY_DEFAULT)
    anchor = {row["probe_id"]: row for row in _probe_rows(anchor_rows)}
    live = {row["probe_id"]: row for row in probes}
    answer_byte_changes = [
        probe_id for probe_id in sorted(anchor)
        if anchor[probe_id]["answer_text"] != live[probe_id]["answer_text"]
    ]
    answer_deltas = _supersession_answer_deltas(anchor, live)
    if decisive:
        if counts != {"correct": 5, "stale": 0, "wrong-fact": 0}:
            raise GateError(f"A-DEC supersession battery score drift: {counts}")
        answer_deltas = _assert_adm2_2_supersession_enumeration(anchor, live)
    else:
        if _sup_projection(rows) != _sup_projection(anchor_rows):
            raise GateError("GRM_ADM_DECISIVE=0 supersession transcript drifted")

    changes = {}
    for probe_id in sorted(anchor):
        keys = changed_keys(anchor[probe_id], live[probe_id])
        if keys:
            changes[probe_id] = keys
    result = {
        "schema": "grm.adm2.supersession_stage.v2",
        "stage": stage,
        "status": "PASS",
        "decisive_admission": bool(decisive),
        "classification_counts": counts,
        "answer_byte_changes_vs_legacy_default": answer_byte_changes,
        "semantic_changes_vs_legacy_default": answer_deltas["semantic_changes"],
        "byte_only_reregistrations_vs_legacy_default": (
            answer_deltas["byte_only_changes"]),
        "changed_probe_fields_vs_legacy_default": changes,
        "receipt": file_record(receipt),
        "transcript_sha256": sha256_bytes(_sup_projection(rows)),
        "legacy_transcript_sha256": sha256_bytes(_sup_projection(anchor_rows)),
        "log": file_record(log),
    }
    write_exclusive_json(stage_dir / "stage_complete.json", result)
    return result


def _transcript_projection(path: Path) -> bytes:
    rows = read_jsonl(path)
    return canonical_rows_bytes({
        "turn": int(row["turn"]),
        "kind": row["kind"],
        "fact_id": row.get("fact_id"),
        "user": row["user"],
        "assistant": row["assistant"],
    } for row in rows)


def run_session(
    run_dir: Path,
    *,
    frame: str,
    decisive: bool,
) -> dict[str, Any]:
    stage = f"{frame}_{'default' if decisive else 'escape'}"
    stage_dir = run_dir / stage
    stage_dir.mkdir(parents=True, exist_ok=False)
    session_dir = stage_dir / "session"
    log = stage_dir / "session.log"
    pipeline = "three_pass" if frame == "p4" else "single"
    command = [
        sys.executable,
        "scripts/grm_e2e_session.py",
        "--mode", "full",
        "--session-dir", str(session_dir),
        "--turn-pipeline", pipeline,
        "--topk", "3",
        "--ngen", "32",
        "--max-trips", "1",
        "--live-turns", "2",
        "--restart-after", "17",
        "--probe-ladder",
        "--sup-resolve",
        "--skip-gpu-idle-check",
    ]
    env = _base_env()
    if decisive:
        env.pop("GRM_ADM_DECISIVE", None)
    else:
        env["GRM_ADM_DECISIVE"] = "0"
    with log.open("xb") as handle:
        completed = subprocess.run(
            command, cwd=ROOT, env=env, stdout=handle,
            stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        raise GateError(f"{stage} failed rc={completed.returncode}; see {log}")

    config = read_json(session_dir / "run_config.json")
    if (
        config.get("probe_ladder") is not True
        or config.get("sup_resolve") is not True
        or config.get("adm_decisive") is not bool(decisive)
    ):
        raise GateError(f"{stage} escaped its production-frame pins")
    scorecard = read_json(session_dir / "probe_scorecard.json")
    if scorecard.get("passed") != 9 or scorecard.get("total") != 9:
        raise GateError(f"{stage} score is not 9/9")

    expected_rows = _expected_physical(frame)
    expected_by_turn = {int(row["turn"]): row for row in expected_rows}
    arm_name = "A-DEC" if decisive else "A-k3"
    mismatches = []
    for probe in scorecard["probes"]:
        expected = expected_by_turn[int(probe["turn"])]["arms"][arm_name]
        if str(probe["answer"]) != str(expected["answer"]):
            mismatches.append(int(probe["turn"]))
        if decisive:
            admission = probe.get("admission") or {}
            if (
                admission.get("admission_policy_branch")
                != expected.get("policy_branch")
                or admission.get("admission_rank_plan")
                != expected.get("rank_plan")
                or admission.get("admission_rule_sha256")
                != FROZEN_RULE_SHA256
            ):
                mismatches.append(int(probe["turn"]))
        elif "admission" in probe:
            mismatches.append(int(probe["turn"]))
    if mismatches:
        raise GateError(f"{stage} ADM1.3 comparison mismatch turns={sorted(set(mismatches))}")

    old_session = ADM13_RUN / frame / "session"
    old_transcript = _transcript_projection(old_session / "transcript.jsonl")
    new_transcript = _transcript_projection(session_dir / "transcript.jsonl")
    if new_transcript != old_transcript:
        raise GateError(f"{stage} changed a 34-turn answer transcript")

    old_instrumentation = {
        int(row["turn"]): row for row in read_jsonl(
            old_session / "instrumentation.jsonl")
        if row.get("kind") == "probe"
    }
    new_instrumentation = {
        int(row["turn"]): row for row in read_jsonl(
            session_dir / "instrumentation.jsonl")
        if row.get("kind") == "probe"
    }
    info_changes = {
        str(turn): changed_keys(
            old_instrumentation[turn].get("info") or {},
            new_instrumentation[turn].get("info") or {},
        )
        for turn in sorted(old_instrumentation)
        if (old_instrumentation[turn].get("info") or {})
        != (new_instrumentation[turn].get("info") or {})
    }
    if decisive:
        if set(info_changes) != {str(turn) for turn in expected_by_turn}:
            raise GateError(f"{stage} arena-field enumeration incomplete: {info_changes}")
    else:
        old_score = (old_session / "probe_scorecard.json").read_bytes()
        new_score = (session_dir / "probe_scorecard.json").read_bytes()
        if new_score != old_score or info_changes:
            raise GateError(f"{stage} GRM_ADM_DECISIVE=0 probe bytes drifted")

    result = {
        "schema": "grm.adm2.session_stage.v1",
        "stage": stage,
        "status": "PASS",
        "frame": frame,
        "turn_pipeline": pipeline,
        "decisive_admission": bool(decisive),
        "score": [9, 9],
        "answer_transcript_byte_identical": True,
        "transcript_sha256": sha256_bytes(new_transcript),
        "legacy_transcript_sha256": sha256_bytes(old_transcript),
        "probe_info_changed_fields": info_changes,
        "scorecard": file_record(session_dir / "probe_scorecard.json"),
        "instrumentation": file_record(session_dir / "instrumentation.jsonl"),
        "transcript": file_record(session_dir / "transcript.jsonl"),
        "run_config": file_record(session_dir / "run_config.json"),
        "log": file_record(log),
    }
    write_exclusive_json(stage_dir / "stage_complete.json", result)
    return result


def _base_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    prior = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + prior if prior else "")
    return env


def run_suite(run_dir: Path, *, full: bool) -> dict[str, Any]:
    stage = "suite_full" if full else "suite_shared"
    stage_dir = run_dir / stage
    stage_dir.mkdir(parents=True, exist_ok=False)
    log = stage_dir / "pytest.log"
    if full:
        paths = sorted(relative(path) for path in ROOT.glob("tests/test_grm_*.py"))
        paths.append("tests/test_gqa_ragged_cuda_bank.py")
    else:
        paths = [
            "tests/test_grm_admission.py",
            "tests/test_grm_probe_ladder.py",
            "tests/test_grm_supersession_battery.py",
            "tests/test_grm_adm1_snapshot.py",
            "tests/test_grm_adm1_3_dual_frame.py",
            "tests/test_grm_importance_telemetry.py",
            "tests/test_grm_fold_recovered_guard.py",
        ]
    command = [
        sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *paths,
    ]
    env = _base_env()
    env.pop("GRM_ADM_DECISIVE", None)
    with log.open("xb") as handle:
        completed = subprocess.run(
            command, cwd=ROOT, env=env, stdout=handle,
            stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        raise GateError(f"{stage} failed rc={completed.returncode}; see {log}")
    result = {
        "schema": "grm.adm2.pytest_stage.v1",
        "stage": stage,
        "status": "PASS",
        "path_count": len(paths),
        "log": file_record(log),
    }
    write_exclusive_json(stage_dir / "stage_complete.json", result)
    return result


def _run_gpu_stage(run_dir: Path, stage: str, args: argparse.Namespace) -> dict[str, Any]:
    from scripts.grm_adm1_gpu import gpu_lease

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    with gpu_lease(int(args.lease_seconds), int(args.lock_wait_seconds)):
        if stage == "lookup_default":
            return run_lookup(run_dir, decisive=True)
        if stage == "lookup_escape":
            return run_lookup(run_dir, decisive=False)
        if stage == "supersession_default":
            return run_supersession(run_dir, decisive=True)
        if stage == "supersession_escape":
            return run_supersession(run_dir, decisive=False)
        if stage == "e2e_default":
            return run_session(run_dir, frame="e2e", decisive=True)
        if stage == "e2e_escape":
            return run_session(run_dir, frame="e2e", decisive=False)
        if stage == "p4_default":
            return run_session(run_dir, frame="p4", decisive=True)
        if stage == "p4_escape":
            return run_session(run_dir, frame="p4", decisive=False)
        if stage == "suite_shared":
            return run_suite(run_dir, full=False)
        if stage == "suite_full":
            return run_suite(run_dir, full=True)
    raise GateError(f"unknown stage {stage}")


def _new_run_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = ARTIFACT_ROOT / f"default_on_{stamp}_{os.getpid()}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _write_manifest(run_dir: Path, args: argparse.Namespace) -> None:
    sources = (
        ROOT / "orders" / "GRM_ADM2_ADEC_DEFAULT_ON.md",
        ROOT / "orders" / "GRM_ADM2_1_PROCEED.md",
        ROOT / "orders" / "GRM_ADM2_2_ORION_PHRASING.md",
        ROOT / "core" / "grm_admission.py",
        ROOT / "core" / "graft_arena.py",
        ROOT / "scripts" / "grm_e2e_session.py",
        ROOT / "tests" / "test_grm_admission.py",
        Path(__file__).resolve(),
    )
    write_exclusive_json(run_dir / "run_manifest.json", {
        "schema": "grm.adm2.run_manifest.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": str(args.gpu),
        "lease_seconds": int(args.lease_seconds),
        "lock_wait_seconds": int(args.lock_wait_seconds),
        "stages": list(STAGES),
        "bare_invocation": (
            "PYTHONPATH=/mnt/ForgeRealm/GraftRepository python3 "
            "scripts/grm_adm2_default_on_gate.py --stage all"
        ),
        "sources": {relative(path): file_record(path) for path in sources},
    })
    registration = registered_acceptance()
    write_exclusive_json(run_dir / "registered_acceptance.json", registration)


def summarize(run_dir: Path) -> dict[str, Any]:
    registration = read_json(run_dir / "registered_acceptance.json")
    stages = {
        stage: read_json(run_dir / stage / "stage_complete.json")
        for stage in STAGES
    }
    if not all(row.get("status") == "PASS" for row in stages.values()):
        raise GateError("not every ADM2 stage passed")
    gates = {
        "ADM2-FROZEN-RULE": "PASS",
        "ADM2-SEMANTIC-CHANGE-ENUMERATION": "PASS",
        "ADM2-ORION-BYTE-REREGISTRATION": "PASS",
        "ADM2-LOOKUP-F-PROD": "PASS",
        "ADM2-SUPERSESSION-BATTERY": "PASS",
        "ADM2-ESCAPE-BYTE-IDENTITY": "PASS",
        "ADM2-E2E-34": "PASS",
        "ADM2-P4-34": "PASS",
        "ADM2-SHARED-SUITE": "PASS",
        "ADM2-FULL-GRM-SUITE": "PASS",
        "ADM2-DET1-COORDINATION": "PASS",
    }
    result = {
        "schema": "grm.adm2.default_on_gate.v2",
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": relative(run_dir),
        "gates": gates,
        "flag": registration["flag"],
        "acceptance_numbers": {
            "CORPUS-100": stages["lookup_default"]["corpus100"],
            "SUPERSESSION-FRESH-CONTROLS": (
                stages["lookup_default"]["supersession_fresh"]),
            "SUPERSESSION-FULL": [
                stages["supersession_default"]["classification_counts"]["correct"],
                5,
            ],
            "E2E-34": stages["e2e_default"]["score"],
            "P4-34": stages["p4_default"]["score"],
        },
        "registered_physical_semantic_changes": registration["answer_changes"],
        "live_semantic_changes": (
            stages["supersession_default"]
            ["semantic_changes_vs_legacy_default"]),
        "live_byte_only_reregistrations": (
            stages["supersession_default"]
            ["byte_only_reregistrations_vs_legacy_default"]),
        "re_registered_turns": registration["re_registered_turns"],
        "live_supersession_field_changes": (
            stages["supersession_default"]["changed_probe_fields_vs_legacy_default"]),
        "live_session_field_changes": {
            "e2e": stages["e2e_default"]["probe_info_changed_fields"],
            "p4": stages["p4_default"]["probe_info_changed_fields"],
        },
        "escape_receipts": {
            "lookup": stages["lookup_escape"],
            "supersession": stages["supersession_escape"],
            "e2e": stages["e2e_escape"],
            "p4": stages["p4_escape"],
        },
        "det1_coordination": registration["det1_coordination"],
        "source_receipts": registration["source_receipts"],
        "run_manifest": file_record(run_dir / "run_manifest.json"),
        "stage_receipts": {
            stage: file_record(run_dir / stage / "stage_complete.json")
            for stage in STAGES
        },
        "anything_not_done": [],
    }
    result_path = run_dir / "gate_result.json"
    write_exclusive_json(result_path, result)
    result["gate_result"] = file_record(result_path)
    return result


def orchestrate(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve() if args.run_dir else _new_run_dir()
    if not run_dir.exists():
        run_dir.mkdir(parents=True)
    if not (run_dir / "run_manifest.json").is_file():
        _write_manifest(run_dir, args)
    print(f"run_dir={run_dir}", flush=True)
    for index, stage in enumerate(STAGES):
        marker = run_dir / stage / "stage_complete.json"
        if marker.is_file():
            print(f"stage={stage} status=already_complete", flush=True)
            continue
        if index:
            time.sleep(2)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--stage", stage,
            "--run-dir", str(run_dir),
            "--gpu", str(args.gpu),
            "--lease-seconds", str(args.lease_seconds),
            "--lock-wait-seconds", str(args.lock_wait_seconds),
        ]
        completed = subprocess.run(command, cwd=ROOT, env=_base_env())
        if completed.returncode != 0:
            print(f"stage={stage} status=FAILED rc={completed.returncode}", flush=True)
            return int(completed.returncode)
        print(f"stage={stage} status=PASS", flush=True)
    result = summarize(run_dir)
    print(json.dumps({
        "status": result["status"],
        "run_dir": result["run_dir"],
        "gate_result": result["gate_result"],
    }, sort_keys=True), flush=True)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("cpu", "all", "summarize", *STAGES),
                        default="cpu")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--lease-seconds", type=int, default=580)
    parser.add_argument("--lock-wait-seconds", type=int, default=21600)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not (1 <= int(args.lease_seconds) <= MAX_LEASE_SECONDS):
        raise GateError(f"lease seconds must be in 1..{MAX_LEASE_SECONDS}")
    if args.stage == "cpu":
        print(json.dumps(registered_acceptance(), indent=2, sort_keys=True))
        return 0
    if args.stage == "all":
        return orchestrate(args)
    if args.run_dir is None:
        raise GateError(f"--stage {args.stage} requires --run-dir")
    run_dir = args.run_dir.resolve()
    if args.stage == "summarize":
        print(json.dumps(summarize(run_dir), indent=2, sort_keys=True))
        return 0
    result = _run_gpu_stage(run_dir, args.stage, args)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
