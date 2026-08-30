#!/usr/bin/env python3
"""CPU-only per-fixture reproduce-first audit for ORDER GRM-CMC1.1.

CMC-G0 is evaluated independently for each registered fixture.  Mechanism
arms are authorized for every fixture that reproduces under current defaults;
a healed fixture is recorded and excluded without stopping a reproducing
sibling fixture.  This audit deliberately does not load a model.  It
cross-checks the registered on-disk receipts against the live default-selection
code, records content-addressed provenance, and emits an append-only receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = ROOT / "artifacts" / "grm_cmc1"

ORDER = ROOT / "orders" / "GRM_CMC1_COMOUNT_MECHANISM.md"
AMENDMENT = ROOT / "orders" / "GRM_CMC1_1_PER_FIXTURE.md"
PROBE_LADDER = ROOT / "scripts" / "grm_probe_ladder.py"
E2E_DRIVER = ROOT / "scripts" / "grm_e2e_session.py"
SUP_HARNESS = ROOT / "tests" / "test_grm_supersession_battery.py"
SUP_FIXTURE = (
    ROOT / "tests" / "fixtures" / "supersession_battery"
    / "fresh_fact_controls.json"
)
SUP_BASELINE = ROOT / "artifacts" / "grm_supersession" / "g0_baseline.jsonl"
DIAG_LEGACY = (
    ROOT / "artifacts" / "grm_three_pass"
    / "diag_unbounded_arm_b_single_r2" / "probe_scorecard.json"
)
DIAG_CURRENT_DEFAULT = (
    ROOT / "artifacts" / "grm_three_pass"
    / "ladder_on_full_default_single" / "probe_scorecard.json"
)
DIAG_REPORT = ROOT / "docs" / "GRM3P_DIAG_UNBOUNDED_REPORT.md"
CMC_REPORT = DEFAULT_ARTIFACT_DIR / "GRM_CMC1_REPORT.md"

PROVENANCE_FILES = (
    ORDER,
    AMENDMENT,
    PROBE_LADDER,
    E2E_DRIVER,
    SUP_HARNESS,
    SUP_FIXTURE,
    SUP_BASELINE,
    DIAG_LEGACY,
    DIAG_CURRENT_DEFAULT,
    DIAG_REPORT,
)


class AuditError(RuntimeError):
    """The frozen audit inputs are missing, malformed, or non-finite."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read JSON receipt {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditError(f"JSON receipt must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise AuditError(f"cannot read JSONL receipt {path}: {exc}") from exc
    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AuditError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise AuditError(f"{path}:{lineno}: row must be an object")
        rows.append(row)
    return rows


def _probe(scorecard: dict[str, Any], turn: int) -> dict[str, Any]:
    hits = [
        row for row in scorecard.get("probes", ())
        if isinstance(row, dict) and int(row.get("turn", -1)) == int(turn)
    ]
    if len(hits) != 1:
        raise AuditError(
            f"expected exactly one turn-{turn} probe, found {len(hits)}")
    return hits[0]


def _load_probe_ladder_module():
    spec = importlib.util.spec_from_file_location(
        "grm_cmc1_live_probe_ladder", PROBE_LADDER)
    if spec is None or spec.loader is None:
        raise AuditError(f"cannot import {PROBE_LADDER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def current_probe_ladder_default() -> bool:
    """Resolve the live default with no CLI choice and no env override."""
    module = _load_probe_ladder_module()
    sentinel = object()
    previous: object | str = os.environ.get("GRM_PROBE_LADDER", sentinel)
    try:
        os.environ.pop("GRM_PROBE_LADDER", None)
        return bool(module.probe_ladder_enabled(None))
    finally:
        if previous is sentinel:
            os.environ.pop("GRM_PROBE_LADDER", None)
        else:
            os.environ["GRM_PROBE_LADDER"] = str(previous)


def supersession_evidence(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    fresh = [
        row for row in rows
        if row.get("record_type") == "supersession_probe_receipt"
        and row.get("scenario") == "fresh_fact_controls"
    ]
    by_id = {str(row.get("probe_id")): row for row in fresh}
    if set(by_id) != {"praxis_fresh", "solace_fresh"}:
        raise AuditError(
            "supersession baseline must contain praxis_fresh and solace_fresh")
    praxis = by_id["praxis_fresh"]
    correct = sum(row.get("classification") == "correct" for row in fresh)
    wrong_read = (
        correct == 1
        and praxis.get("classification") == "wrong-fact"
        and "raven-9-ivory" in str(praxis.get("answer_text", "")).casefold()
        and praxis.get("target_rank") == 1
        and "praxis_fact" in (praxis.get("mounted_nodes") or ())
        and "solace_fact" in (praxis.get("mounted_nodes") or ())
    )
    return {
        "receipt_kind": "historical_registered_gpu_receipt",
        "fresh_control_correct": int(correct),
        "fresh_control_total": len(fresh),
        "praxis_answer": praxis.get("answer_text"),
        "praxis_classification": praxis.get("classification"),
        "praxis_target_rank": praxis.get("target_rank"),
        "praxis_mounted_nodes": praxis.get("mounted_nodes"),
        "wrong_read_reproduced_in_receipt": bool(wrong_read),
    }


def diag_evidence(
    legacy: dict[str, Any], current: dict[str, Any]
) -> dict[str, Any]:
    old_t5 = _probe(legacy, 5)
    new_t5 = _probe(current, 5)
    old_wrong = (
        old_t5.get("pass") is False
        and "vortex-3-sierra" in str(old_t5.get("answer", "")).casefold()
    )
    current_wrong = (
        new_t5.get("pass") is False
        and "vortex-3-sierra" in str(new_t5.get("answer", "")).casefold()
    )
    current_healed = (
        current.get("all_passed") is True
        and int(current.get("passed", -1)) == 9
        and int(current.get("total", -1)) == 9
        and new_t5.get("pass") is True
        and "auric-4-alpha" in str(new_t5.get("answer", "")).casefold()
        and new_t5.get("mount_plan") == [0]
        and new_t5.get("mounted_ids") == [0]
    )
    return {
        "legacy": {
            "score": [legacy.get("passed"), legacy.get("total")],
            "turn5_answer": old_t5.get("answer"),
            "turn5_pass": old_t5.get("pass"),
            "wrong_read_reproduced": bool(old_wrong),
        },
        "current_default": {
            "receipt_kind": "registered_lead_gpu_receipt",
            "score": [current.get("passed"), current.get("total")],
            "all_passed": current.get("all_passed"),
            "turn5_answer": new_t5.get("answer"),
            "turn5_pass": new_t5.get("pass"),
            "turn5_mount_plan": new_t5.get("mount_plan"),
            "turn5_mounted_ids": new_t5.get("mounted_ids"),
            "wrong_read_reproduced": bool(current_wrong),
            "healed": bool(current_healed),
        },
    }


def _assert_finite(value: Any, where: str = "$") -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AuditError(f"non-finite value at {where}: {value!r}")
    elif isinstance(value, dict):
        for key, item in value.items():
            _assert_finite(item, f"{where}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_finite(item, f"{where}[{index}]")


def canonical_json_bytes(value: Any) -> bytes:
    _assert_finite(value)
    return (
        json.dumps(
            value, sort_keys=True, indent=2, ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def write_content_addressed(directory: Path, stem: str, value: Any) -> Path:
    payload = canonical_json_bytes(value)
    digest = hashlib.sha256(payload).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stem}_{digest[:16]}.json"
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise AuditError(f"append-only content-address collision: {path}")
    return path


def build_audit() -> dict[str, Any]:
    missing = [str(path) for path in PROVENANCE_FILES if not path.is_file()]
    if missing:
        raise AuditError(f"missing frozen input(s): {missing}")

    ladder_on = current_probe_ladder_default()
    sup = supersession_evidence(_read_jsonl(SUP_BASELINE))
    diag = diag_evidence(_read_json(DIAG_LEGACY), _read_json(DIAG_CURRENT_DEFAULT))

    sup_reproduces = bool(sup["wrong_read_reproduced_in_receipt"])
    diag_reproduces = bool(diag["current_default"]["wrong_read_reproduced"])
    authorized = []
    excluded = []
    fixture_status = {}
    if sup_reproduces:
        authorized.append("supersession_fresh_control")
        fixture_status["supersession_fresh_control"] = "REPRODUCES"
    else:
        excluded.append("supersession_fresh_control")
        fixture_status["supersession_fresh_control"] = "HEALED_OR_NOT_REPRODUCED"
    if diag_reproduces:
        authorized.append("diag_turn5_orion_cypher")
        fixture_status["diag_turn5_orion_cypher"] = "REPRODUCES"
    else:
        excluded.append("diag_turn5_orion_cypher")
        fixture_status["diag_turn5_orion_cypher"] = (
            "HEALED_PROBE_LADDER_CURE"
            if diag["current_default"]["healed"]
            else "NOT_REPRODUCED"
        )
    if authorized:
        g0 = "PASS_PER_FIXTURE"
        stop_reason = None
    else:
        g0 = "STOP_NO_REPRODUCING_FIXTURE"
        stop_reason = "No registered fixture reproduces under current defaults."

    return {
        "schema": "grm.cmc1.preflight.v2",
        "order": "GRM-CMC1.1",
        "execution_scope": "cpu_receipt_audit_no_gpu",
        "live_default": {
            "probe_ladder_enabled_with_cli_unset_and_env_absent": ladder_on,
            "expected_permanent_default": True,
            "matches_expected": ladder_on is True,
        },
        "fixtures": {
            "supersession_fresh_control": sup,
            "diag_turn5_orion_cypher": diag,
        },
        "fixture_status": fixture_status,
        "gates": {
            "CMC-G0": g0,
            "CMC-G1": "PENDING_LIVE_ARM_RUN" if authorized else "NOT_RUN_G0_STOP",
            "CMC-G2": "PENDING_LIVE_WITNESS" if authorized else "NOT_RUN_G0_STOP",
            "CMC-G3": "PENDING_CMC1_1_RESULTS" if authorized else (
                "PASS" if CMC_REPORT.is_file() else "PENDING_REPORT"),
        },
        "arms_authorized": bool(authorized),
        "arms_authorized_fixtures": authorized,
        "excluded_fixtures": excluded,
        "arms_run": [],
        "findings": ([{
            "fixture": "diag_turn5_orion_cypher",
            "finding": "probe_ladder_cure",
            "score": diag["current_default"]["score"],
            "turn5_answer": diag["current_default"]["turn5_answer"],
        }] if diag["current_default"]["healed"] else []),
        "stop_reason": stop_reason,
        "provenance": {
            str(path.relative_to(ROOT)): {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in PROVENANCE_FILES
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR,
        help="append-only destination for the content-addressed receipt")
    parser.add_argument(
        "--require-repro", action="store_true",
        help="return status 3 unless at least one fixture reproduces")
    parser.add_argument("--json", action="store_true", help="print full receipt")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    audit = build_audit()
    path = write_content_addressed(
        args.artifact_dir.expanduser().resolve(), "preflight", audit)
    if args.json:
        print(canonical_json_bytes(audit).decode("utf-8"), end="")
    print(f"receipt={path}")
    print(f"CMC-G0={audit['gates']['CMC-G0']}")
    print(f"arms_authorized={str(audit['arms_authorized']).lower()}")
    if audit.get("stop_reason"):
        print(f"stop_reason={audit['stop_reason']}")
    if args.require_repro and not audit["arms_authorized"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
