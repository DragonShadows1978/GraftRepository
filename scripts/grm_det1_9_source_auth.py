#!/usr/bin/env python3
"""Append-only DET1.9 successor authorization for registry collection.

DET1.8 remains immutable.  This module validates that exact predecessor from
its frozen records, reconstructs the effective source inventory through
DET1.8, and authorizes only a caller-supplied DET1.9 overlay.  The envelope is
strictly collection-only: it is not evaluation evidence and does not
authorize race resume.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from scripts import grm_det1_7_source_auth as det17
from scripts import grm_det1_8_source_auth as det18


ROOT = Path(__file__).resolve().parents[1]
FROZEN_RUN = ROOT / "artifacts/grm_det1/run_20260831T160525Z_2"
ORDER = ROOT / "orders/GRM_DET1_9_SUBSTITUTION_POOL.md"
REGISTRATION = FROZEN_RUN / "registration_62cb6c09cbec211d.json"
RUNTIME_FRAME = FROZEN_RUN / "runtime_frame_28b3196f8fb04a41.json"
DET1_6_AMENDMENT = FROZEN_RUN / "det1_6_fork_hydration_delta_amendment.json"
PREDECESSOR_AUTHORIZATION = (
    FROZEN_RUN / "det1_8_precollection_source_authorization.json"
)
PRECOLLECTION_AUTHORIZATION = (
    FROZEN_RUN / "det1_9_precollection_source_authorization.json"
)

SCHEMA = "grm.det1_9.precollection_source_authorization.v1"
STATUS = "AUTHORIZED_LIVED_PLANT_REGISTRY_COLLECTION_ONLY"
PREDECESSOR_SCHEMA = "grm.det1_8.precollection_source_authorization.v1"
PREDECESSOR_STATUS = "AUTHORIZED_LIVED_PLANT_REGISTRY_COLLECTION_ONLY"

# Immutable production anchor.  Isolated repositories may provide their own
# expected predecessor record explicitly.
PRODUCTION_PREDECESSOR_RECORD = {
    "path": (
        "artifacts/grm_det1/run_20260831T160525Z_2/"
        "det1_8_precollection_source_authorization.json"
    ),
    "bytes": 17727,
    "sha256": "f5f5901a14e759bd14bcdc788e6f36e77eeea7aaec770541b907d2cb7c594edb",
}

SourceAuthorizationError = det17.SourceAuthorizationError


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceAuthorizationError(message)


def _validate_created_utc(created_utc: str) -> None:
    try:
        parsed = datetime.fromisoformat(created_utc.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise SourceAuthorizationError(
            "created_utc is not an ISO-8601 time"
        ) from exc
    _require(
        created_utc.endswith("Z")
        and parsed.tzinfo is not None
        and parsed.utcoffset() == timezone.utc.utcoffset(parsed),
        "created_utc must be UTC with a Z suffix",
    )


def _canonical_purpose(value: Any, label: str) -> str:
    _require(
        isinstance(value, str)
        and value == value.strip()
        and bool(value),
        f"{label} purpose is empty or non-canonical",
    )
    return value


def _merge_direct_rebindings(
    *maps: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for mapping in maps:
        for source, rebinding in mapping.items():
            prior = result.get(source)
            _require(
                prior is None or prior == dict(rebinding),
                f"direct older-validator rebinding conflict: {source}",
            )
            result[source] = dict(rebinding)
    return result


def _authorization_invariants() -> dict[str, Any]:
    return {
        "amendment_is_evidence": False,
        "authorization_boundary": "LIVED_PLANT_REGISTRY_COLLECTION_ONLY",
        "collection_authorized": True,
        "evaluation_authorized": False,
        "historical_rows_reusable": False,
        "plant_lawfulness_rules_unchanged": True,
        "plant_registry_freeze_required_before_evaluation": True,
        "race_resume_authorized": False,
        "registration_change_scope": "PLANTED_MISS_TARGETS_ONLY",
        "served_control_count_required": 12,
        "served_planted_count_required": 12,
        "source_change_scope": (
            "CERTIFIED_LIVED_CAMPAIGN_CROSS_SESSION_SUBSTITUTION_ONLY"
        ),
        "substitution_scope": (
            "ANY_CERTIFIED_LIVED_COLLECTED_SESSION_IN_CAMPAIGN"
        ),
    }


def _expected_predecessor_record(
    *,
    repo_root: Path,
    predecessor_path: Path,
    expected_record: Mapping[str, Any] | None,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    path = Path(predecessor_path).resolve()
    observed = det17.file_record(path, repo_root=root)
    if expected_record is not None:
        expected = dict(expected_record)
    elif root == ROOT.resolve() and path == PREDECESSOR_AUTHORIZATION.resolve():
        expected = dict(PRODUCTION_PREDECESSOR_RECORD)
    else:
        expected = dict(observed)
    _require(
        observed == expected,
        "DET1.8 predecessor authorization file record drift",
    )
    return observed


def _validate_predecessor_authorization(
    *,
    repo_root: Path,
    registration_path: Path,
    runtime_frame_path: Path,
    det1_6_amendment_path: Path,
    predecessor_authorization_path: Path,
    expected_predecessor_record: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate DET1.8 structurally without consulting newer source bytes."""
    root = Path(repo_root).resolve()
    path = det17._inside_repo(  # type: ignore[attr-defined]
        predecessor_authorization_path, root, "DET1.8 predecessor authorization"
    )
    record = _expected_predecessor_record(
        repo_root=root,
        predecessor_path=path,
        expected_record=expected_predecessor_record,
    )
    value = det17._read_json(  # type: ignore[attr-defined]
        path, "DET1.8 predecessor authorization"
    )
    _require(
        path.read_bytes() == det17.canonical_json_bytes(value),
        "DET1.8 predecessor authorization is not canonically encoded",
    )
    _require(
        value.get("schema") == PREDECESSOR_SCHEMA,
        "DET1.8 predecessor authorization schema drift",
    )
    _require(
        value.get("status") == PREDECESSOR_STATUS,
        "DET1.8 predecessor authorization status drift",
    )
    _validate_created_utc(value.get("created_utc"))

    source_lineage = value.get("source_lineage")
    _require(
        isinstance(source_lineage, Mapping),
        "DET1.8 predecessor source_lineage is malformed",
    )
    det1_7_record = source_lineage.get(
        "det1_7_precollection_source_authorization"
    )
    _require(
        isinstance(det1_7_record, Mapping),
        "DET1.8 predecessor lacks DET1.7 authorization lineage",
    )
    _require(
        value.get("predecessor_authorization") == det1_7_record,
        "DET1.8 predecessor DET1.7 binding drift",
    )
    det1_7_path = det17._record_path(  # type: ignore[attr-defined]
        det1_7_record,
        repo_root=root,
        label="DET1.8 embedded DET1.7 authorization",
    )
    det1_7 = det18._validate_predecessor_authorization(  # type: ignore[attr-defined]
        repo_root=root,
        registration_path=registration_path,
        runtime_frame_path=runtime_frame_path,
        det1_6_amendment_path=det1_6_amendment_path,
        predecessor_authorization_path=det1_7_path,
        expected_predecessor_record=det1_7_record,
    )
    lineage = det1_7["lineage"]

    order_record = value.get("order") or {}
    order_path = det17._validate_live_record(  # type: ignore[attr-defined]
        order_record, repo_root=root, label="DET1.8 predecessor order"
    )
    _require(
        order_path.name == "GRM_DET1_8_SYNTH_TURN_SNAPSHOTS.md",
        "DET1.8 predecessor binds the wrong order",
    )
    for key, expected in (
        ("registration", lineage["registration_record"]),
        ("runtime_frame", lineage["runtime_record"]),
    ):
        _require(
            value.get(key) == expected,
            f"DET1.8 predecessor {key} binding drift",
        )

    det1_7_inventory = {
        source: dict(source_record)
        for source, source_record in det1_7["effective_inventory"].items()
    }
    effective = {
        source: dict(source_record)
        for source, source_record in det1_7_inventory.items()
    }
    origins = dict(det1_7["effective_origins"])

    raw_changes = value.get("allowed_source_changes") or {}
    raw_additions = value.get("added_sources") or {}
    _require(
        isinstance(raw_changes, Mapping),
        "DET1.8 predecessor allowed_source_changes is malformed",
    )
    _require(
        isinstance(raw_additions, Mapping),
        "DET1.8 predecessor added_sources is malformed",
    )
    _require(
        not (set(raw_changes) & set(raw_additions)),
        "DET1.8 predecessor change/addition overlap",
    )

    normalized_changes: dict[str, dict[str, Any]] = {}
    for source, raw in raw_changes.items():
        _require(
            isinstance(source, str) and isinstance(raw, Mapping),
            "DET1.8 predecessor source-change entry is malformed",
        )
        normalized_source, _source_path = det17._relative_source(  # type: ignore[attr-defined]
            source, root
        )
        _require(
            normalized_source in det1_7_inventory,
            f"DET1.8 predecessor change lacks DET1.7 source: {source}",
        )
        after = det17._normalized_historical_record(  # type: ignore[attr-defined]
            source, raw.get("after") or {}, f"det1_8.after.{source}"
        )
        expected_entry = {
            "purpose": _canonical_purpose(
                raw.get("purpose"), f"DET1.8 predecessor {source}"
            ),
            "predecessor_origin": origins[source],
            "before": dict(det1_7_inventory[source]),
            "after": after,
        }
        _require(
            dict(raw) == expected_entry,
            f"DET1.8 predecessor source-change content drift: {source}",
        )
        _require(
            expected_entry["before"] != after,
            f"DET1.8 predecessor source change is empty: {source}",
        )
        normalized_changes[source] = expected_entry
        effective[source] = after
        origins[source] = "det1_8.allowed_source_changes.after"

    normalized_additions: dict[str, dict[str, Any]] = {}
    for source, raw in raw_additions.items():
        _require(
            isinstance(source, str) and isinstance(raw, Mapping),
            "DET1.8 predecessor added-source entry is malformed",
        )
        normalized_source, _source_path = det17._relative_source(  # type: ignore[attr-defined]
            source, root
        )
        _require(
            normalized_source not in det1_7_inventory,
            f"DET1.8 predecessor addition already has DET1.7 source: {source}",
        )
        source_record = det17._normalized_historical_record(  # type: ignore[attr-defined]
            source, raw.get("record") or {}, f"det1_8.added.{source}"
        )
        expected_entry = {
            "purpose": _canonical_purpose(
                raw.get("purpose"), f"DET1.8 predecessor {source}"
            ),
            "record": source_record,
        }
        _require(
            dict(raw) == expected_entry,
            f"DET1.8 predecessor added-source content drift: {source}",
        )
        normalized_additions[source] = expected_entry
        effective[source] = source_record
        origins[source] = "det1_8.added_sources.record"

    det1_4_rebindings = det17._direct_rebindings(  # type: ignore[attr-defined]
        lineage["det1_4_inventory"], effective
    )
    det1_5_rebindings = det17._direct_rebindings(  # type: ignore[attr-defined]
        lineage["det1_5_inventory"], effective
    )
    det1_6_rebindings = det17._direct_rebindings(  # type: ignore[attr-defined]
        lineage["immediate_inventory"], effective
    )
    det1_7_rebindings = det17._direct_rebindings(  # type: ignore[attr-defined]
        det1_7_inventory, effective
    )
    older_rebindings = _merge_direct_rebindings(
        det1_4_rebindings, det1_5_rebindings
    )
    maps = {
        "det1_4": det1_4_rebindings,
        "det1_5": det1_5_rebindings,
        "det1_6": det1_6_rebindings,
        "det1_7": det1_7_rebindings,
        "older_validators": older_rebindings,
    }
    expected_document = {
        "schema": PREDECESSOR_SCHEMA,
        "status": PREDECESSOR_STATUS,
        "created_utc": value["created_utc"],
        "order": dict(order_record),
        "registration": lineage["registration_record"],
        "runtime_frame": lineage["runtime_record"],
        "predecessor_authorization": dict(det1_7_record),
        "source_lineage": {
            "det1_4_source_amendment": lineage["det1_4_record"],
            "det1_5_race_authorization_amendment": lineage["det1_5_record"],
            "det1_6_fork_hydration_delta_amendment": lineage["det1_6_record"],
            "det1_7_precollection_source_authorization": dict(det1_7_record),
        },
        "allowed_source_changes": normalized_changes,
        "added_sources": normalized_additions,
        "direct_transitive_rebindings": maps,
        "source_counts": {
            "added": len(normalized_additions),
            "changed": len(normalized_changes),
            "det1_4_direct_rebindings": len(det1_4_rebindings),
            "det1_5_direct_rebindings": len(det1_5_rebindings),
            "det1_6_direct_rebindings": len(det1_6_rebindings),
            "det1_7_direct_rebindings": len(det1_7_rebindings),
            "frozen_predecessors_checked": len(det1_7_inventory),
        },
        "invariants": det18._authorization_invariants(),  # type: ignore[attr-defined]
    }
    _require(
        value == expected_document,
        "DET1.8 predecessor authorization content drift",
    )
    return {
        "record": record,
        "effective_inventory": effective,
        "effective_origins": origins,
        "det1_7_inventory": det1_7_inventory,
        "det1_7_record": dict(det1_7_record),
        "lineage": lineage,
    }


def _build_authorization(
    *,
    repo_root: Path,
    order_path: Path,
    registration_path: Path,
    runtime_frame_path: Path,
    det1_6_amendment_path: Path,
    predecessor_authorization_path: Path,
    changed_sources: Mapping[str, str],
    added_sources: Mapping[str, str],
    created_utc: str,
    expected_predecessor_record: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, Any]]]]:
    root = Path(repo_root).resolve()
    order = det17._inside_repo(  # type: ignore[attr-defined]
        order_path, root, "DET1.9 order"
    )
    _require(
        order.name == "GRM_DET1_9_SUBSTITUTION_POOL.md",
        "wrong DET1.9 order",
    )
    order_record = det17.file_record(order, repo_root=root)
    changes = det17._purpose_map(  # type: ignore[attr-defined]
        changed_sources, "changed_sources", root
    )
    additions = det17._purpose_map(  # type: ignore[attr-defined]
        added_sources, "added_sources", root
    )
    _require(
        not (set(changes) & set(additions)),
        "changed_sources and added_sources overlap",
    )
    _require(changes or additions, "DET1.9 source allowlist is empty")
    _validate_created_utc(created_utc)

    predecessor = _validate_predecessor_authorization(
        repo_root=root,
        registration_path=registration_path,
        runtime_frame_path=runtime_frame_path,
        det1_6_amendment_path=det1_6_amendment_path,
        predecessor_authorization_path=predecessor_authorization_path,
        expected_predecessor_record=expected_predecessor_record,
    )
    immediate = predecessor["effective_inventory"]
    origins = predecessor["effective_origins"]
    lineage = predecessor["lineage"]
    for source in changes:
        _require(
            source in immediate,
            f"changed DET1.9 source has no DET1.8 predecessor: {source}",
        )
    for source in additions:
        _require(
            source not in immediate,
            f"added DET1.9 source already has a DET1.8 predecessor: {source}",
        )

    current: dict[str, dict[str, Any]] = {}
    allowed_changes: dict[str, dict[str, Any]] = {}
    added: dict[str, dict[str, Any]] = {}
    for source, predecessor_record in immediate.items():
        _source, source_path = det17._relative_source(  # type: ignore[attr-defined]
            source, root
        )
        current_record = det17.file_record(source_path, repo_root=root)
        current[source] = current_record
        if source not in changes:
            _require(
                current_record == predecessor_record,
                f"unrelated source drift after DET1.8: {source}",
            )
            continue
        _require(
            current_record != predecessor_record,
            f"allowlisted DET1.9 source change is empty: {source}",
        )
        allowed_changes[source] = {
            "purpose": changes[source],
            "predecessor_origin": origins[source],
            "before": dict(predecessor_record),
            "after": current_record,
        }

    for source, purpose in additions.items():
        _source, source_path = det17._relative_source(  # type: ignore[attr-defined]
            source, root
        )
        current_record = det17.file_record(source_path, repo_root=root)
        current[source] = current_record
        added[source] = {"purpose": purpose, "record": current_record}

    det1_8_rebindings = det17._direct_rebindings(  # type: ignore[attr-defined]
        immediate, current
    )
    _require(
        set(det1_8_rebindings) == set(changes),
        "DET1.9 changes are not exactly predecessor-bound",
    )
    det1_7_rebindings = det17._direct_rebindings(  # type: ignore[attr-defined]
        predecessor["det1_7_inventory"], current
    )
    det1_6_rebindings = det17._direct_rebindings(  # type: ignore[attr-defined]
        lineage["immediate_inventory"], current
    )
    det1_4_rebindings = det17._direct_rebindings(  # type: ignore[attr-defined]
        lineage["det1_4_inventory"], current
    )
    det1_5_rebindings = det17._direct_rebindings(  # type: ignore[attr-defined]
        lineage["det1_5_inventory"], current
    )
    older_rebindings = _merge_direct_rebindings(
        det1_4_rebindings, det1_5_rebindings
    )
    maps = {
        "det1_4": det1_4_rebindings,
        "det1_5": det1_5_rebindings,
        "det1_6": det1_6_rebindings,
        "det1_7": det1_7_rebindings,
        "det1_8": det1_8_rebindings,
        "older_validators": older_rebindings,
    }
    document = {
        "schema": SCHEMA,
        "status": STATUS,
        "created_utc": created_utc,
        "order": order_record,
        "registration": lineage["registration_record"],
        "runtime_frame": lineage["runtime_record"],
        "predecessor_authorization": predecessor["record"],
        "source_lineage": {
            "det1_4_source_amendment": lineage["det1_4_record"],
            "det1_5_race_authorization_amendment": lineage["det1_5_record"],
            "det1_6_fork_hydration_delta_amendment": lineage["det1_6_record"],
            "det1_7_precollection_source_authorization": predecessor[
                "det1_7_record"
            ],
            "det1_8_precollection_source_authorization": predecessor["record"],
        },
        "allowed_source_changes": allowed_changes,
        "added_sources": added,
        "direct_transitive_rebindings": maps,
        "source_counts": {
            "added": len(added),
            "changed": len(allowed_changes),
            "det1_4_direct_rebindings": len(det1_4_rebindings),
            "det1_5_direct_rebindings": len(det1_5_rebindings),
            "det1_6_direct_rebindings": len(det1_6_rebindings),
            "det1_7_direct_rebindings": len(det1_7_rebindings),
            "det1_8_direct_rebindings": len(det1_8_rebindings),
            "frozen_predecessors_checked": len(immediate),
        },
        "invariants": _authorization_invariants(),
    }
    return document, maps


def validate_precollection_source_authorization(
    path: Path = PRECOLLECTION_AUTHORIZATION,
    *,
    repo_root: Path = ROOT,
    order_path: Path = ORDER,
    registration_path: Path = REGISTRATION,
    runtime_frame_path: Path = RUNTIME_FRAME,
    det1_6_amendment_path: Path = DET1_6_AMENDMENT,
    predecessor_authorization_path: Path = PREDECESSOR_AUTHORIZATION,
    changed_sources: Mapping[str, str],
    added_sources: Mapping[str, str],
    expected_predecessor_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the exact DET1.9 overlay, lineage, and current bytes."""
    root = Path(repo_root).resolve()
    authorization_path = det17._inside_repo(  # type: ignore[attr-defined]
        path, root, "DET1.9 source authorization"
    )
    value = det17._read_json(  # type: ignore[attr-defined]
        authorization_path, "DET1.9 source authorization"
    )
    _require(
        authorization_path.read_bytes() == det17.canonical_json_bytes(value),
        "DET1.9 source authorization is not canonically encoded",
    )
    _require(value.get("schema") == SCHEMA, "DET1.9 authorization schema drift")
    _require(value.get("status") == STATUS, "DET1.9 authorization status drift")
    expected, maps = _build_authorization(
        repo_root=root,
        order_path=order_path,
        registration_path=registration_path,
        runtime_frame_path=runtime_frame_path,
        det1_6_amendment_path=det1_6_amendment_path,
        predecessor_authorization_path=predecessor_authorization_path,
        changed_sources=changed_sources,
        added_sources=added_sources,
        created_utc=value.get("created_utc"),
        expected_predecessor_record=expected_predecessor_record,
    )
    _require(value == expected, "DET1.9 source authorization content drift")
    return {
        "schema": SCHEMA,
        "status": STATUS,
        "record": det17.file_record(authorization_path, repo_root=root),
        "predecessor_authorization": expected["predecessor_authorization"],
        "source_count": len(changed_sources) + len(added_sources),
        "det1_4_source_rebindings": maps["det1_4"],
        "det1_5_source_rebindings": maps["det1_5"],
        "det1_6_source_rebindings": maps["det1_6"],
        "det1_7_source_rebindings": maps["det1_7"],
        "det1_8_source_rebindings": maps["det1_8"],
        "transitive_source_rebindings": maps["older_validators"],
        "collection_authorized": True,
        "race_resume_authorized": False,
        "amendment_is_evidence": False,
    }


def author_precollection_source_authorization(
    path: Path = PRECOLLECTION_AUTHORIZATION,
    *,
    repo_root: Path = ROOT,
    order_path: Path = ORDER,
    registration_path: Path = REGISTRATION,
    runtime_frame_path: Path = RUNTIME_FRAME,
    det1_6_amendment_path: Path = DET1_6_AMENDMENT,
    predecessor_authorization_path: Path = PREDECESSOR_AUTHORIZATION,
    changed_sources: Mapping[str, str],
    added_sources: Mapping[str, str],
    created_utc: str | None = None,
    expected_predecessor_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and exclusively write the collection-only DET1.9 successor."""
    timestamp = created_utc or datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    value, _maps = _build_authorization(
        repo_root=repo_root,
        order_path=order_path,
        registration_path=registration_path,
        runtime_frame_path=runtime_frame_path,
        det1_6_amendment_path=det1_6_amendment_path,
        predecessor_authorization_path=predecessor_authorization_path,
        changed_sources=changed_sources,
        added_sources=added_sources,
        created_utc=timestamp,
        expected_predecessor_record=expected_predecessor_record,
    )
    det17.write_json_exclusive(path, value)
    return validate_precollection_source_authorization(
        path,
        repo_root=repo_root,
        order_path=order_path,
        registration_path=registration_path,
        runtime_frame_path=runtime_frame_path,
        det1_6_amendment_path=det1_6_amendment_path,
        predecessor_authorization_path=predecessor_authorization_path,
        changed_sources=changed_sources,
        added_sources=added_sources,
        expected_predecessor_record=expected_predecessor_record,
    )


__all__ = [
    "DET1_6_AMENDMENT",
    "ORDER",
    "PRECOLLECTION_AUTHORIZATION",
    "PREDECESSOR_AUTHORIZATION",
    "PRODUCTION_PREDECESSOR_RECORD",
    "REGISTRATION",
    "ROOT",
    "RUNTIME_FRAME",
    "SCHEMA",
    "STATUS",
    "SourceAuthorizationError",
    "author_precollection_source_authorization",
    "validate_precollection_source_authorization",
]
