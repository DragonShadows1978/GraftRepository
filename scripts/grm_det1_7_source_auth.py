#!/usr/bin/env python3
"""Append-only source authorization for GRM-DET1.7 pre-collection work.

This module authorizes only the source bytes needed to collect and freeze the
DET1.7 lived-admission plant registry.  It deliberately does *not* authorize
resuming the detector race and is not evaluation evidence.

The immediate source predecessor is the DET1.6 amendment.  Sources which were
not changed by DET1.6 inherit their predecessor from the frozen DET1.4 or
DET1.5 inventories.  Direct (not chained) rebinding maps are emitted for the
older validators so they can compare their own frozen record with current
bytes without accepting an intermediate record by implication.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
FROZEN_RUN = ROOT / "artifacts/grm_det1/run_20260831T160525Z_2"
ORDER = ROOT / "orders/GRM_DET1_7_PLANT_REALIGN.md"
REGISTRATION = FROZEN_RUN / "registration_62cb6c09cbec211d.json"
RUNTIME_FRAME = FROZEN_RUN / "runtime_frame_28b3196f8fb04a41.json"
DET1_6_AMENDMENT = FROZEN_RUN / "det1_6_fork_hydration_delta_amendment.json"
PRECOLLECTION_AUTHORIZATION = (
    FROZEN_RUN / "det1_7_precollection_source_authorization_r2.json"
)

SCHEMA = "grm.det1_7.precollection_source_authorization.v1"
STATUS = "AUTHORIZED_LIVED_PLANT_REGISTRY_COLLECTION_ONLY"
DET1_4_SCHEMA = "grm.det1_4.source_amendment.v1"
DET1_4_STATUS = "AUTHORIZED_FORK_FROM_SNAPSHOT_AMENDMENT"
DET1_5_SCHEMA = "grm.det1_5.race_authorization_amendment.v1"
DET1_5_STATUS = "AUTHORIZED_RACE_ON_FORKED_SUBSTRATE"
DET1_6_SCHEMA = "grm.det1_6.fork_hydration_delta_amendment.v1"
DET1_6_STATUS = "AUTHORIZED_FORK_HYDRATION_DELTA_SOURCE_REBINDING"


class SourceAuthorizationError(RuntimeError):
    """A DET1.7 source-lineage or authorization invariant failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceAuthorizationError(message)


def _validate_json(value: Any, where: str = "root") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        _require(math.isfinite(value), f"non-finite JSON number at {where}")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _require(isinstance(key, str), f"non-string JSON key at {where}")
            _validate_json(child, f"{where}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_json(child, f"{where}[{index}]")
        return
    raise SourceAuthorizationError(
        f"non-JSON value at {where}: {type(value).__name__}"
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Return the canonical, human-readable bytes used for this artifact."""
    _validate_json(value)
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside_repo(path: Path, repo_root: Path, label: str) -> Path:
    root = Path(repo_root).resolve()
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise SourceAuthorizationError(
            f"{label} escapes the repository root: {resolved}"
        ) from exc
    return resolved


def file_record(path: Path, *, repo_root: Path = ROOT) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    resolved = _inside_repo(path, root, "file record")
    _require(resolved.is_file(), f"registered file is absent: {resolved}")
    return {
        "path": resolved.relative_to(root).as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _require(Path(path).is_file(), f"{label} is absent: {path}")
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SourceAuthorizationError(f"{label} is not valid JSON: {path}") from exc
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def write_json_exclusive(path: Path, value: Any) -> Path:
    """Canonically write ``value`` once; an existing path is never reused."""
    payload = canonical_json_bytes(value)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        handle.write(payload)
        handle.flush()
    return destination


def _relative_source(source: str, repo_root: Path) -> tuple[str, Path]:
    _require(isinstance(source, str) and bool(source), "source path is empty")
    _require("\\" not in source, f"source path is not POSIX-normalized: {source}")
    pure = PurePosixPath(source)
    _require(not pure.is_absolute(), f"source path must be relative: {source}")
    _require(
        pure.as_posix() == source
        and all(part not in ("", ".", "..") for part in pure.parts),
        f"source path is not canonical: {source}",
    )
    path = _inside_repo(Path(repo_root) / source, repo_root, f"source {source}")
    return source, path


def _purpose_map(value: Mapping[str, str], label: str, repo_root: Path) -> dict[str, str]:
    _require(isinstance(value, Mapping), f"{label} must be a mapping")
    result: dict[str, str] = {}
    for source, purpose in value.items():
        normalized, _path = _relative_source(source, repo_root)
        _require(
            isinstance(purpose, str)
            and purpose == purpose.strip()
            and bool(purpose),
            f"{label} purpose is empty or non-canonical: {source}",
        )
        result[normalized] = purpose
    return result


def _normalized_historical_record(
    source: str,
    record: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    _require(isinstance(record, Mapping), f"{label} is not a file record")
    shown = record.get("path", source)
    _require(shown == source, f"{label} path drift")
    byte_count = record.get("bytes")
    digest = record.get("sha256")
    _require(
        isinstance(byte_count, int)
        and not isinstance(byte_count, bool)
        and byte_count >= 0,
        f"{label} byte count is invalid",
    )
    _require(
        isinstance(digest, str)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest),
        f"{label} digest is invalid",
    )
    return {"path": source, "bytes": byte_count, "sha256": digest}


def _record_path(
    record: Mapping[str, Any],
    *,
    repo_root: Path,
    label: str,
) -> Path:
    _require(isinstance(record, Mapping), f"{label} is not a file record")
    shown = record.get("path")
    _require(isinstance(shown, str) and bool(shown), f"{label} has no path")
    _source, path = _relative_source(shown, repo_root)
    return path


def _validate_live_record(
    record: Mapping[str, Any],
    *,
    repo_root: Path,
    label: str,
) -> Path:
    path = _record_path(record, repo_root=repo_root, label=label)
    _require(
        dict(record) == file_record(path, repo_root=repo_root),
        f"{label} bytes drift",
    )
    return path


def _add_frozen_record(
    inventory: dict[str, dict[str, Any]],
    origins: dict[str, str],
    *,
    source: str,
    record: Mapping[str, Any],
    origin: str,
) -> None:
    normalized = _normalized_historical_record(source, record, f"{origin}.{source}")
    prior = inventory.get(source)
    _require(
        prior is None or prior == normalized,
        f"frozen source lineage conflicts for {source}",
    )
    inventory[source] = normalized
    origins.setdefault(source, origin)


def _frozen_inventories(
    det1_4: Mapping[str, Any],
    det1_5: Mapping[str, Any],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    det1_4_inventory: dict[str, dict[str, Any]] = {}
    det1_5_inventory: dict[str, dict[str, Any]] = {}
    origins: dict[str, str] = {}

    for field in ("allowed_source_changes", "added_sources", "test_sources"):
        section = det1_4.get(field) or {}
        _require(isinstance(section, Mapping), f"DET1.4 {field} is malformed")
        for source, raw in section.items():
            _require(
                isinstance(source, str) and isinstance(raw, Mapping),
                f"DET1.4 {field} entry is malformed",
            )
            record = raw.get("after") if field == "allowed_source_changes" else raw
            _require(isinstance(record, Mapping), f"DET1.4 {field}.{source} is malformed")
            _add_frozen_record(
                det1_4_inventory,
                origins,
                source=source,
                record=record,
                origin=f"det1_4.{field}",
            )

    section = det1_5.get("source_inventory") or {}
    _require(isinstance(section, Mapping), "DET1.5 source_inventory is malformed")
    for source, record in section.items():
        _require(
            isinstance(source, str) and isinstance(record, Mapping),
            "DET1.5 source_inventory entry is malformed",
        )
        _add_frozen_record(
            det1_5_inventory,
            origins,
            source=source,
            record=record,
            origin="det1_5.source_inventory",
        )

    overlap = set(det1_4_inventory) & set(det1_5_inventory)
    for source in overlap:
        _require(
            det1_4_inventory[source] == det1_5_inventory[source],
            f"DET1.4/DET1.5 frozen source conflict: {source}",
        )
    return det1_4_inventory, det1_5_inventory, origins


def _lineage(
    *,
    repo_root: Path,
    registration_path: Path,
    runtime_frame_path: Path,
    det1_6_amendment_path: Path,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    registration_record = file_record(registration_path, repo_root=root)
    runtime_record = file_record(runtime_frame_path, repo_root=root)
    registration = _read_json(registration_path, "base registration")
    runtime = _read_json(runtime_frame_path, "runtime frame")
    _require(
        registration.get("schema") == "grm.det1.registration.v1",
        "base registration schema drift",
    )
    _require(
        runtime.get("schema") == "grm.det1.runtime_frame.v1",
        "runtime-frame schema drift",
    )
    _require(
        runtime.get("registration") == registration_record,
        "runtime frame does not bind the base registration",
    )

    det1_6_path = _inside_repo(
        det1_6_amendment_path, root, "DET1.6 amendment"
    )
    det1_6 = _read_json(det1_6_path, "DET1.6 amendment")
    _require(det1_6.get("schema") == DET1_6_SCHEMA, "DET1.6 schema drift")
    _require(det1_6.get("status") == DET1_6_STATUS, "DET1.6 status drift")
    _require(
        det1_6.get("registration") == registration_record,
        "DET1.6 registration binding drift",
    )
    _require(
        det1_6.get("runtime_frame") == runtime_record,
        "DET1.6 runtime-frame binding drift",
    )
    det1_6_invariants = det1_6.get("invariants") or {}
    _require(
        isinstance(det1_6_invariants, Mapping)
        and det1_6_invariants.get("race_resume_authorized") is True
        and det1_6_invariants.get("amendment_is_evidence") is False,
        "DET1.6 authorization invariants drift",
    )
    _validate_live_record(
        det1_6.get("order") or {}, repo_root=root, label="DET1.6 order"
    )

    det1_4_path = _validate_live_record(
        det1_6.get("det1_4_source_amendment") or {},
        repo_root=root,
        label="DET1.6 DET1.4 predecessor",
    )
    det1_5_path = _validate_live_record(
        det1_6.get("det1_5_race_amendment") or {},
        repo_root=root,
        label="DET1.6 DET1.5 predecessor",
    )
    det1_4 = _read_json(det1_4_path, "DET1.4 amendment")
    det1_5 = _read_json(det1_5_path, "DET1.5 amendment")
    _require(
        det1_4.get("schema") == DET1_4_SCHEMA
        and det1_4.get("status") == DET1_4_STATUS,
        "DET1.4 predecessor identity drift",
    )
    _require(
        det1_5.get("schema") == DET1_5_SCHEMA
        and det1_5.get("status") == DET1_5_STATUS,
        "DET1.5 predecessor identity drift",
    )
    for label, value in (("DET1.4", det1_4), ("DET1.5", det1_5)):
        _require(
            value.get("registration") == registration_record,
            f"{label} registration binding drift",
        )
        _require(
            value.get("runtime_frame") == runtime_record,
            f"{label} runtime-frame binding drift",
        )

    det1_4_inventory, det1_5_inventory, origins = _frozen_inventories(
        det1_4, det1_5
    )
    historical: dict[str, dict[str, Any]] = {}
    for inventory in (det1_4_inventory, det1_5_inventory):
        for source, record in inventory.items():
            prior = historical.get(source)
            _require(
                prior is None or prior == record,
                f"pre-DET1.6 source conflict: {source}",
            )
            historical[source] = record

    immediate = {source: dict(record) for source, record in historical.items()}
    det1_6_changes = det1_6.get("allowed_source_changes") or {}
    _require(
        isinstance(det1_6_changes, Mapping),
        "DET1.6 allowed_source_changes is malformed",
    )
    for source, raw in det1_6_changes.items():
        _require(
            isinstance(source, str) and isinstance(raw, Mapping),
            "DET1.6 source-change entry is malformed",
        )
        _require(
            source in historical,
            f"DET1.6 source lacks a frozen predecessor: {source}",
        )
        before = _normalized_historical_record(
            source, raw.get("before") or {}, f"det1_6.before.{source}"
        )
        after = _normalized_historical_record(
            source, raw.get("after") or {}, f"det1_6.after.{source}"
        )
        _require(
            before == historical[source],
            f"DET1.6 immediate predecessor drift: {source}",
        )
        _require(before != after, f"DET1.6 source rebinding is empty: {source}")
        immediate[source] = after
        origins[source] = "det1_6.allowed_source_changes.after"

    return {
        "registration_record": registration_record,
        "runtime_record": runtime_record,
        "det1_4_record": file_record(det1_4_path, repo_root=root),
        "det1_5_record": file_record(det1_5_path, repo_root=root),
        "det1_6_record": file_record(det1_6_path, repo_root=root),
        "det1_4_inventory": det1_4_inventory,
        "det1_5_inventory": det1_5_inventory,
        "immediate_inventory": immediate,
        "immediate_origins": origins,
    }


def _direct_rebindings(
    frozen: Mapping[str, Mapping[str, Any]],
    current: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        source: {"before": dict(record), "after": dict(current[source])}
        for source, record in frozen.items()
        if current[source] != record
    }


def _authorization_invariants() -> dict[str, Any]:
    return {
        "amendment_is_evidence": False,
        "authorization_boundary": "LIVED_PLANT_REGISTRY_COLLECTION_ONLY",
        "collection_authorized": True,
        "evaluation_authorized": False,
        "historical_rows_reusable": False,
        "plant_registry_freeze_required_before_evaluation": True,
        "race_resume_authorized": False,
        "registration_change_scope": "PLANTED_MISS_TARGETS_ONLY",
    }


def _build_authorization(
    *,
    repo_root: Path,
    order_path: Path,
    registration_path: Path,
    runtime_frame_path: Path,
    det1_6_amendment_path: Path,
    changed_sources: Mapping[str, str],
    added_sources: Mapping[str, str],
    created_utc: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(repo_root).resolve()
    order = _inside_repo(order_path, root, "DET1.7 order")
    _require(order.name == "GRM_DET1_7_PLANT_REALIGN.md", "wrong DET1.7 order")
    order_record = file_record(order, repo_root=root)
    changes = _purpose_map(changed_sources, "changed_sources", root)
    additions = _purpose_map(added_sources, "added_sources", root)
    _require(
        not (set(changes) & set(additions)),
        "changed_sources and added_sources overlap",
    )
    _require(changes or additions, "DET1.7 source allowlist is empty")

    try:
        parsed_time = datetime.fromisoformat(created_utc.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise SourceAuthorizationError("created_utc is not an ISO-8601 time") from exc
    _require(
        created_utc.endswith("Z")
        and parsed_time.tzinfo is not None
        and parsed_time.utcoffset() == timezone.utc.utcoffset(parsed_time),
        "created_utc must be UTC with a Z suffix",
    )

    lineage = _lineage(
        repo_root=root,
        registration_path=registration_path,
        runtime_frame_path=runtime_frame_path,
        det1_6_amendment_path=det1_6_amendment_path,
    )
    immediate = lineage["immediate_inventory"]
    for source in changes:
        _require(
            source in immediate,
            f"changed DET1.7 source has no immediate predecessor: {source}",
        )
    for source in additions:
        _require(
            source not in immediate,
            f"added DET1.7 source already has a frozen predecessor: {source}",
        )

    current: dict[str, dict[str, Any]] = {}
    allowed_changes: dict[str, dict[str, Any]] = {}
    added: dict[str, dict[str, Any]] = {}
    for source, predecessor in immediate.items():
        _name, path = _relative_source(source, root)
        record = file_record(path, repo_root=root)
        current[source] = record
        if source not in changes:
            _require(
                record == predecessor,
                f"unrelated source drift after DET1.6: {source}",
            )
            continue
        _require(
            record != predecessor,
            f"allowlisted DET1.7 source change is empty: {source}",
        )
        allowed_changes[source] = {
            "purpose": changes[source],
            "predecessor_origin": lineage["immediate_origins"][source],
            "before": dict(predecessor),
            "after": record,
        }

    for source, purpose in additions.items():
        _name, path = _relative_source(source, root)
        record = file_record(path, repo_root=root)
        current[source] = record
        added[source] = {"purpose": purpose, "record": record}

    det1_6_rebindings = {
        source: {
            "before": dict(immediate[source]),
            "after": dict(current[source]),
        }
        for source in changes
    }
    det1_4_rebindings = _direct_rebindings(
        lineage["det1_4_inventory"], current
    )
    det1_5_rebindings = _direct_rebindings(
        lineage["det1_5_inventory"], current
    )
    older_rebindings = {
        source: dict(rebinding)
        for source, rebinding in det1_4_rebindings.items()
    }
    for source, rebinding in det1_5_rebindings.items():
        prior = older_rebindings.get(source)
        _require(
            prior is None or prior == rebinding,
            f"direct older-validator rebinding conflict: {source}",
        )
        older_rebindings[source] = dict(rebinding)

    maps = {
        "det1_4": det1_4_rebindings,
        "det1_5": det1_5_rebindings,
        "det1_6": det1_6_rebindings,
        "older_validators": older_rebindings,
    }
    document = {
        "schema": SCHEMA,
        "status": STATUS,
        "created_utc": created_utc,
        "order": order_record,
        "registration": lineage["registration_record"],
        "runtime_frame": lineage["runtime_record"],
        "predecessor_amendment": lineage["det1_6_record"],
        "source_lineage": {
            "det1_4_source_amendment": lineage["det1_4_record"],
            "det1_5_race_authorization_amendment": lineage["det1_5_record"],
            "det1_6_fork_hydration_delta_amendment": lineage["det1_6_record"],
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
    changed_sources: Mapping[str, str],
    added_sources: Mapping[str, str],
) -> dict[str, Any]:
    """Validate the exact allowlist, lineage, current bytes, and flags."""
    root = Path(repo_root).resolve()
    authorization_path = _inside_repo(path, root, "DET1.7 source authorization")
    value = _read_json(authorization_path, "DET1.7 source authorization")
    _require(
        authorization_path.read_bytes() == canonical_json_bytes(value),
        "DET1.7 source authorization is not canonically encoded",
    )
    _require(value.get("schema") == SCHEMA, "DET1.7 authorization schema drift")
    _require(value.get("status") == STATUS, "DET1.7 authorization status drift")
    expected, maps = _build_authorization(
        repo_root=root,
        order_path=order_path,
        registration_path=registration_path,
        runtime_frame_path=runtime_frame_path,
        det1_6_amendment_path=det1_6_amendment_path,
        changed_sources=changed_sources,
        added_sources=added_sources,
        created_utc=value.get("created_utc"),
    )
    _require(value == expected, "DET1.7 source authorization content drift")
    return {
        "schema": SCHEMA,
        "status": STATUS,
        "record": file_record(authorization_path, repo_root=root),
        "source_count": len(changed_sources) + len(added_sources),
        "det1_6_source_rebindings": maps["det1_6"],
        "det1_4_source_rebindings": maps["det1_4"],
        "det1_5_source_rebindings": maps["det1_5"],
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
    changed_sources: Mapping[str, str],
    added_sources: Mapping[str, str],
    created_utc: str | None = None,
) -> dict[str, Any]:
    """Build and exclusively write the collection-only authorization."""
    timestamp = created_utc or datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    value, _maps = _build_authorization(
        repo_root=repo_root,
        order_path=order_path,
        registration_path=registration_path,
        runtime_frame_path=runtime_frame_path,
        det1_6_amendment_path=det1_6_amendment_path,
        changed_sources=changed_sources,
        added_sources=added_sources,
        created_utc=timestamp,
    )
    write_json_exclusive(path, value)
    return validate_precollection_source_authorization(
        path,
        repo_root=repo_root,
        order_path=order_path,
        registration_path=registration_path,
        runtime_frame_path=runtime_frame_path,
        det1_6_amendment_path=det1_6_amendment_path,
        changed_sources=changed_sources,
        added_sources=added_sources,
    )


__all__ = [
    "DET1_6_AMENDMENT",
    "ORDER",
    "PRECOLLECTION_AUTHORIZATION",
    "REGISTRATION",
    "ROOT",
    "RUNTIME_FRAME",
    "SCHEMA",
    "STATUS",
    "SourceAuthorizationError",
    "author_precollection_source_authorization",
    "canonical_json_bytes",
    "file_record",
    "validate_precollection_source_authorization",
    "write_json_exclusive",
]
