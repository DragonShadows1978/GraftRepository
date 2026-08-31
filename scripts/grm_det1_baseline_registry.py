#!/usr/bin/env python3
"""Live registered-baseline lookup for the DET1 purity guard.

The stable registry contains pointers and hashes, not copied canonical rows.
Every lookup revalidates the selected gate chain, then applies the registered
family comparator: same-model exact bytes or cross-model DET fixture semantics,
always with authoritative ordered physical mounts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from scripts.grm_det1_common import normalize_value_text


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "config" / "grm_live_registered_baselines.json"
REGISTRY_ENV = "GRM_REGISTERED_BASELINE_REGISTRY"
EXACT_COMPARATOR = {
    "schema": "grm.registered_baseline_comparator.v1",
    "kind": "same_model_exact_answer_bytes_ordered_mount_ids_v1",
    "answer": "exact_bytes",
    "mounts": "ordered_authoritative_final_physical_ids",
}
CROSS_MODEL_SUP_COMPARATOR = {
    "schema": "grm.registered_baseline_comparator.v1",
    "kind": "cross_model_det_fixture_accept_reject_ordered_physical_identity_v1",
    "answer": (
        "det1_fixture_expected_present_and_rejected_absent_"
        "case_insensitive_whole_value"
    ),
    "mounts": "ordered_fixture_physical_ids_and_nodes",
}


class BaselineRegistryError(RuntimeError):
    """The live registry or a registered projection failed closed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def file_record(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    try:
        shown = str(path.relative_to(ROOT.resolve()))
    except ValueError:
        shown = str(path)
    return {
        "path": shown,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _validate_record(value: Mapping[str, Any], label: str) -> Path:
    try:
        path = _path(str(value["path"])).resolve()
    except (KeyError, TypeError, ValueError) as exc:
        raise BaselineRegistryError(f"invalid {label} path record") from exc
    if not path.is_file() or file_record(path) != dict(value):
        raise BaselineRegistryError(f"{label} hash/size record does not validate")
    return path


def registry_path(explicit: Path | None = None) -> Path:
    """Use the live production registry unless a caller explicitly overrides it.

    ``REGISTRY_ENV`` remains a compatibility symbol, but ambient environment
    state cannot redirect a production DET1 run to a stale/private registry.
    """
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    return DEFAULT_REGISTRY.resolve()


def load_live_registry(
    explicit: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = registry_path(explicit)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BaselineRegistryError(
            f"cannot read live baseline registry {path}") from exc
    if not isinstance(value, dict) or value.get("schema") != (
        "grm.live_registered_baseline_registry.v1"
    ):
        raise BaselineRegistryError("unsupported live baseline registry schema")
    active = value.get("active_registration") or {}
    if active.get("status") != "PASS":
        raise BaselineRegistryError("active baseline registration is not PASS")
    registration_path = _validate_record(
        active.get("registration_receipt") or {}, "active registration receipt")
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    if registration.get("status") != "PASS":
        raise BaselineRegistryError(
            "active registration receipt no longer says PASS")
    families = value.get("families") or {}
    if not isinstance(families, dict) or not families:
        raise BaselineRegistryError("live registry has no baseline families")
    for family, binding in families.items():
        if not isinstance(binding, dict):
            raise BaselineRegistryError(f"invalid baseline family {family}")
        _validate_binding(binding, f"active {family}")
        source_record = binding.get("source") or {}
        stage_record = binding.get("stage_receipt") or {}
        stage_path = _validate_record(
            stage_record, f"{family} stage receipt")
        stage = json.loads(stage_path.read_text(encoding="utf-8"))
        if stage.get("status") != "PASS":
            raise BaselineRegistryError(f"{family} stage receipt is not PASS")
        gate_stage = str(binding.get("gate_stage", ""))
        if (registration.get("stage_receipts") or {}).get(gate_stage) != stage_record:
            raise BaselineRegistryError(
                f"active registration does not bind {family} stage receipt")
        source_field = str(binding.get("stage_source_field", ""))
        if stage.get(source_field) != source_record:
            raise BaselineRegistryError(
                f"{family} stage receipt does not bind its baseline source")
        if stage.get("transcript_sha256") != binding.get(
            "registered_projection_sha256"):
            raise BaselineRegistryError(
                f"{family} registered projection hash is not stage-bound")
    previous_values = value.get("previous_registrations", ())
    if not isinstance(previous_values, list):
        raise BaselineRegistryError("previous registrations must be a list")
    seen_previous = set()
    for previous in previous_values:
        if not isinstance(previous, dict):
            raise BaselineRegistryError("invalid previous registration")
        previous_id = str(previous.get("id", ""))
        if not previous_id or previous_id in seen_previous:
            raise BaselineRegistryError("duplicate/empty previous registration id")
        seen_previous.add(previous_id)
        if previous.get("status") != "RETIRED":
            raise BaselineRegistryError(
                f"previous registration {previous_id} is not RETIRED")
        previous_receipt_path = _validate_record(
            previous.get("registration_receipt") or {},
            f"previous registration {previous_id}",
        )
        previous_receipt = json.loads(
            previous_receipt_path.read_text(encoding="utf-8"))
        if previous_receipt.get("status") != "PASS":
            raise BaselineRegistryError(
                f"previous registration {previous_id} is not a completed PASS gate")
        for family, binding in (previous.get("families") or {}).items():
            if not isinstance(binding, dict):
                raise BaselineRegistryError(
                    f"invalid previous family {previous_id}/{family}")
            _validate_binding(binding, f"previous {previous_id} {family}")
            source_record = binding.get("source") or {}
            gate_key = str(binding.get("gate_source_key", ""))
            gate_source = (previous_receipt.get("live_receipts") or {}).get(gate_key)
            if (
                not isinstance(gate_source, dict)
                or gate_source.get("path") != source_record.get("path")
                or gate_source.get("receipt_sha256") != source_record.get("sha256")
                or gate_source.get("transcript_sha256")
                    != binding.get("registered_projection_sha256")
            ):
                raise BaselineRegistryError(
                    f"previous registration {previous_id} does not bind {family}")
    anchor = {
        "registry": file_record(path),
        "registration_id": str(active.get("id")),
        "registration_hash": str(active["registration_receipt"]["sha256"]),
        "registration_receipt": dict(active["registration_receipt"]),
    }
    return value, anchor


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise BaselineRegistryError(
                f"non-object baseline row {path}:{line_no}")
        rows.append(value)
    return rows


def _comparator(binding: Mapping[str, Any], label: str) -> dict[str, Any]:
    value = binding.get("comparator")
    if value == EXACT_COMPARATOR:
        return dict(EXACT_COMPARATOR)
    if value == CROSS_MODEL_SUP_COMPARATOR:
        return dict(CROSS_MODEL_SUP_COMPARATOR)
    raise BaselineRegistryError(f"{label} has an unsupported comparator contract")


def _fixture_documents(
    binding: Mapping[str, Any], label: str,
) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    records = binding.get("fixture_sources")
    if not isinstance(records, dict) or not records:
        raise BaselineRegistryError(f"{label} lacks fixture-source bindings")
    documents = {}
    for session_id, record in records.items():
        if not isinstance(record, dict):
            raise BaselineRegistryError(f"{label} has an invalid fixture record")
        path = _validate_record(record, f"{label} fixture {session_id}")
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BaselineRegistryError(
                f"cannot read {label} fixture {session_id}") from exc
        if not (
            isinstance(document, dict)
            and document.get("schema") == "grm_supersession_battery_v1"
            and str(document.get("session_id")) == str(session_id)
            and isinstance(document.get("nodes"), list)
            and isinstance(document.get("probes"), list)
        ):
            raise BaselineRegistryError(
                f"{label} fixture {session_id} identity does not validate")
        node_ids = [str(value.get("node_id")) for value in document["nodes"]]
        probe_ids = [str(value.get("probe_id")) for value in document["probes"]]
        if (
            not node_ids or len(node_ids) != len(set(node_ids))
            or not probe_ids or len(probe_ids) != len(set(probe_ids))
        ):
            raise BaselineRegistryError(
                f"{label} fixture {session_id} has duplicate/empty identities")
        documents[str(session_id)] = (dict(record), document)
    return documents


def _sup_fixture_probe(
    fixture: Mapping[str, Any], binding: Mapping[str, Any], label: str,
) -> tuple[dict[str, Any], dict[str, Any], list[str], dict[str, Any]]:
    documents = _fixture_documents(binding, label)
    session_id = str(fixture.get("session_id", ""))
    if session_id not in documents:
        raise BaselineRegistryError(
            f"{label} has no registered fixture session {session_id}")
    source_record, document = documents[session_id]
    if fixture.get("source") != source_record:
        raise BaselineRegistryError(
            f"{label} DET fixture source is not the registered fixture source")
    probe_id = str(fixture.get("probe_id", ""))
    matches = [
        value for value in document["probes"]
        if str(value.get("probe_id")) == probe_id
    ]
    if len(matches) != 1:
        raise BaselineRegistryError(
            f"{label} expected one fixture probe {probe_id}, found {len(matches)}")
    probe = dict(matches[0])
    expected = {
        "session_id": document.get("session_id"),
        "scenario": document.get("scenario"),
        "probe_id": probe.get("probe_id"),
        "question": probe.get("question"),
        "target_node": probe.get("target_node"),
        "expected_values": probe.get("expected_values"),
        "stale_values": probe.get("stale_values"),
        "wrong_fact_values": probe.get("wrong_fact_values"),
    }
    mismatches = {
        key: {"fixture": fixture.get(key), "source": value}
        for key, value in expected.items() if fixture.get(key) != value
    }
    if mismatches:
        raise BaselineRegistryError(
            f"{label} DET fixture fields differ from source: {mismatches}")
    node_ids = [str(value["node_id"]) for value in document["nodes"]]
    return source_record, document, node_ids, probe


def _validate_sup_row(
    fixture: Mapping[str, Any],
    binding: Mapping[str, Any],
    row: Mapping[str, Any],
    label: str,
) -> tuple[list[int], list[str], dict[str, Any]]:
    source_record, document, node_ids, probe = _sup_fixture_probe(
        fixture, binding, label)
    identity = {
        "fixture": Path(str(source_record["path"])).name,
        "session_id": document.get("session_id"),
        "scenario": document.get("scenario"),
        "probe_id": probe.get("probe_id"),
        "question": probe.get("question"),
        "target_node": probe.get("target_node"),
        "correction_node": probe.get("correction_node"),
        "stale_nodes": probe.get("stale_nodes"),
        "competitor_node": probe.get("competitor_node"),
    }
    mismatches = {
        key: {"registered_row": row.get(key), "fixture": value}
        for key, value in identity.items() if row.get(key) != value
    }
    if mismatches:
        raise BaselineRegistryError(
            f"{label} registered row differs from fixture: {mismatches}")
    try:
        mounted = [int(value) for value in row["mounted_indices"]]
        mounted_nodes = [str(value) for value in row["mounted_nodes"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise BaselineRegistryError(
            f"{label} lacks ordered physical mount identity") from exc
    if any(value < 0 or value >= len(node_ids) for value in mounted):
        raise BaselineRegistryError(f"{label} registered mount ID is out of range")
    if mounted_nodes != [node_ids[value] for value in mounted]:
        raise BaselineRegistryError(
            f"{label} registered mount IDs/nodes disagree with fixture order")
    ranking = row.get("ranking_nodes")
    if not (
        isinstance(ranking, list)
        and len(ranking) == len(set(str(value) for value in ranking))
        and all(str(value) in node_ids for value in ranking)
    ):
        raise BaselineRegistryError(
            f"{label} registered ranking nodes are not fixture identities")
    value_sets = {
        "correct": [str(value).casefold() for value in probe["expected_values"]],
        "stale": [str(value).casefold() for value in probe.get("stale_values", ())],
        "wrong-fact": [
            str(value).casefold() for value in probe.get("wrong_fact_values", ())
        ],
    }
    producer_class = str(row.get("classification", ""))
    producer_match = str(row.get("classification_match", "")).casefold()
    if producer_class not in value_sets or producer_match not in value_sets[producer_class]:
        raise BaselineRegistryError(
            f"{label} producer classification/match is not fixture-defined")
    required = str(binding.get("producer_classification_requirement", ""))
    if required == "correct" and producer_class != "correct":
        raise BaselineRegistryError(
            f"{label} producer row is not registered-correct")
    if required not in ("correct", "registered_evidence_only"):
        raise BaselineRegistryError(
            f"{label} has an unsupported producer classification requirement")
    fixture_contract = {
        "fixture_source": source_record,
        "session_id": str(document["session_id"]),
        "scenario": str(document["scenario"]),
        "probe_id": str(probe["probe_id"]),
        "node_order": node_ids,
        "expected_values": [str(value) for value in probe["expected_values"]],
        "stale_values": [str(value) for value in probe.get("stale_values", ())],
        "wrong_fact_values": [
            str(value) for value in probe.get("wrong_fact_values", ())
        ],
        "rejected_values": [
            str(value) for value in (
                list(probe.get("stale_values") or ())
                + list(probe.get("wrong_fact_values") or ())
            )
        ],
    }
    return mounted, mounted_nodes, fixture_contract


def _validate_sup_source(binding: Mapping[str, Any], label: str) -> None:
    source_path = _validate_record(binding.get("source") or {}, f"{label} source")
    rows = [
        value for value in _read_jsonl(source_path)
        if value.get("record_type") == "supersession_probe_receipt"
    ]
    documents = _fixture_documents(binding, label)
    expected = {}
    for session_id, (source_record, document) in documents.items():
        for probe in document["probes"]:
            fixture = {
                "source": source_record,
                "session_id": session_id,
                "scenario": document.get("scenario"),
                "probe_id": probe.get("probe_id"),
                "question": probe.get("question"),
                "target_node": probe.get("target_node"),
                "expected_values": probe.get("expected_values"),
                "stale_values": probe.get("stale_values"),
                "wrong_fact_values": probe.get("wrong_fact_values"),
            }
            expected[(session_id, str(probe.get("probe_id")))] = fixture
    observed = {}
    for row in rows:
        key = (str(row.get("session_id")), str(row.get("probe_id")))
        if key in observed or key not in expected:
            raise BaselineRegistryError(
                f"{label} source has an extra/duplicate probe identity {key}")
        _validate_sup_row(expected[key], binding, row, label)
        observed[key] = row
    if set(observed) != set(expected):
        raise BaselineRegistryError(
            f"{label} source/fixture probe identities are incomplete")


def _validate_binding(binding: Mapping[str, Any], label: str) -> None:
    comparator = _comparator(binding, label)
    source_format = str(binding.get("format", ""))
    _validate_record(binding.get("source") or {}, f"{label} source")
    if comparator == EXACT_COMPARATOR:
        if source_format != "grm_e2e_probe_scorecard_v1":
            raise BaselineRegistryError(f"{label} exact comparator format mismatch")
    elif source_format != "grm_supersession_probe_receipt_v1":
        raise BaselineRegistryError(f"{label} cross-model comparator format mismatch")
    else:
        _validate_sup_source(binding, label)


def _binding_projection(
    fixture: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    comparator = _comparator(binding, "baseline binding")
    source_path = _validate_record(binding.get("source") or {}, "baseline source")
    source_format = str(binding.get("format"))
    if source_format == "grm_e2e_probe_scorecard_v1":
        if comparator != EXACT_COMPARATOR:
            raise BaselineRegistryError("E2E baseline comparator is not exact")
        value = json.loads(source_path.read_text(encoding="utf-8"))
        matches = [
            row for row in value.get("probes", ())
            if int(row.get("turn", -1)) == int(fixture.get("turn", -2))
        ]
        if len(matches) != 1:
            raise BaselineRegistryError(
                f"expected one registered turn {fixture.get('turn')}, "
                f"found {len(matches)}")
        row = dict(matches[0])
        answer = row.get("answer")
        mounted = row.get("mounted_ids")
        selector = {"turn": int(fixture["turn"])}
        if not isinstance(answer, str) or not isinstance(mounted, list):
            raise BaselineRegistryError(
                "registered E2E baseline lacks answer/mounted projection")
        mounted_ids = [int(value) for value in mounted]
        extra: dict[str, Any] = {}
    elif source_format == "grm_supersession_probe_receipt_v1":
        if comparator != CROSS_MODEL_SUP_COMPARATOR:
            raise BaselineRegistryError(
                "supersession baseline comparator is not cross-model safe")
        probe_id = str(fixture.get("probe_id", ""))
        matches = [
            row for row in _read_jsonl(source_path)
            if row.get("record_type") == "supersession_probe_receipt"
            and str(row.get("probe_id")) == probe_id
            and str(row.get("session_id")) == str(fixture.get("session_id"))
        ]
        if len(matches) != 1:
            raise BaselineRegistryError(
                f"expected one registered probe {probe_id}, found {len(matches)}")
        row = dict(matches[0])
        answer = row.get("answer_text")
        mounted_ids, mounted_nodes, fixture_contract = _validate_sup_row(
            fixture, binding, row, "baseline binding")
        selector = {
            "session_id": str(fixture["session_id"]),
            "probe_id": probe_id,
        }
        extra = {
            "mounted_nodes": mounted_nodes,
            "fixture_contract": fixture_contract,
            "producer_classification": {
                "classification": row.get("classification"),
                "classification_match": row.get("classification_match"),
            },
        }
    else:
        raise BaselineRegistryError(
            f"unsupported baseline source format {source_format}")
    if not isinstance(answer, str):
        raise BaselineRegistryError("registered baseline lacks answer evidence")
    return {
        "answer": answer,
        "mounted_ids": mounted_ids,
        "selector": selector,
        "source": dict(binding["source"]),
        "comparator": comparator,
        "registered_source_row": dict(row),
        **extra,
    }


def registered_projection(
    fixture: Mapping[str, Any],
    *,
    registry: Mapping[str, Any],
    previous_registration_id: str | None = None,
) -> dict[str, Any]:
    family = str(fixture.get("source_family", ""))
    if previous_registration_id is None:
        families = registry.get("families") or {}
        registration_id = str((registry.get("active_registration") or {}).get("id"))
    else:
        registrations = [
            value for value in registry.get("previous_registrations", ())
            if str(value.get("id")) == str(previous_registration_id)
        ]
        if len(registrations) != 1:
            raise BaselineRegistryError(
                f"unknown previous registration {previous_registration_id}")
        families = registrations[0].get("families") or {}
        registration_id = str(previous_registration_id)
    if family not in families:
        raise BaselineRegistryError(
            f"registration {registration_id} has no family {family}")
    projection = _binding_projection(fixture, families[family])
    projection.update({
        "family": family,
        "registration_id": registration_id,
    })
    return projection


def _contains_value(text: str, value: str) -> bool:
    normalized_text = normalize_value_text(text)
    normalized_value = normalize_value_text(value)
    return bool(re.search(
        rf"(?<![A-Za-z0-9_-]){re.escape(normalized_value)}(?![A-Za-z0-9_-])",
        normalized_text,
        flags=re.IGNORECASE,
    ))


def _classify_det_answer(
    answer: str, contract: Mapping[str, Any],
) -> dict[str, Any]:
    matches = {
        "correct": [
            value for value in contract.get("expected_values", ())
            if _contains_value(answer, str(value))
        ],
        "stale": [
            value for value in contract.get("stale_values", ())
            if _contains_value(answer, str(value))
        ],
        "wrong-fact": [
            value for value in contract.get("wrong_fact_values", ())
            if _contains_value(answer, str(value))
        ],
    }
    rejected = matches["stale"] + matches["wrong-fact"]
    if matches["correct"] and not rejected:
        classification = "correct"
        classification_match = str(matches["correct"][0])
    elif matches["stale"]:
        classification = "stale"
        classification_match = str(matches["stale"][0])
    elif matches["wrong-fact"]:
        classification = "wrong-fact"
        classification_match = str(matches["wrong-fact"][0])
    else:
        classification = "unmatched"
        classification_match = None
    return {
        "classification": classification,
        "classification_match": classification_match,
        "expected_value_matches": [str(value) for value in matches["correct"]],
        "rejected_value_matches": [str(value) for value in rejected],
    }


def _compare_projection(
    observed: Mapping[str, Any], expected: Mapping[str, Any],
) -> tuple[bool, dict[str, Any], list[str] | None]:
    comparator = expected.get("comparator")
    mounted_ids = [int(value) for value in observed.get("mounted_ids", ())]
    if comparator == EXACT_COMPARATOR:
        checks = {
            "answer_exact_bytes": str(observed.get("answer", ""))
                == str(expected["answer"]),
            "ordered_authoritative_final_physical_ids": mounted_ids
                == [int(value) for value in expected["mounted_ids"]],
        }
        return all(checks.values()), checks, None
    if comparator != CROSS_MODEL_SUP_COMPARATOR:
        raise BaselineRegistryError("projection has an unsupported comparator")
    fixture_contract = expected.get("fixture_contract") or {}
    observed_class = _classify_det_answer(
        str(observed.get("answer", "")), fixture_contract)
    producer_class = expected.get("producer_classification") or {}
    semantic_match = (
        observed_class["classification"] == producer_class.get("classification")
        and str(observed_class["classification_match"] or "").casefold()
            == str(producer_class.get("classification_match") or "").casefold()
    )
    node_order = [str(value) for value in fixture_contract.get("node_order", ())]
    ids_in_range = all(0 <= value < len(node_order) for value in mounted_ids)
    observed_nodes = [node_order[value] for value in mounted_ids] if ids_in_range else None
    reported_correct = observed.get("answer_correct")
    answer_correct_consistent = (
        reported_correct is None
        or bool(reported_correct) == (observed_class["classification"] == "correct")
    )
    checks = {
        "det_fixture_observed_classification": observed_class,
        "registered_producer_classification": dict(producer_class),
        "det_fixture_semantic_class_and_match": semantic_match,
        "served_answer_correct_consistent": answer_correct_consistent,
        "ordered_fixture_physical_ids": mounted_ids
            == [int(value) for value in expected["mounted_ids"]],
        "ordered_fixture_physical_nodes": observed_nodes
            == [str(value) for value in expected["mounted_nodes"]],
    }
    passed = bool(
        checks["det_fixture_semantic_class_and_match"]
        and checks["served_answer_correct_consistent"]
        and checks["ordered_fixture_physical_ids"]
        and checks["ordered_fixture_physical_nodes"]
    )
    return passed, checks, observed_nodes


def compare_served_to_live_registry(
    fixture: Mapping[str, Any],
    served: Mapping[str, Any],
    *,
    registry_path_override: Path | None = None,
) -> dict[str, Any]:
    registry, anchor = load_live_registry(registry_path_override)
    expected = registered_projection(fixture, registry=registry)
    observed = {
        "answer": str(served.get("answer", "")),
        "mounted_ids": [int(value) for value in served.get("mounted_ids", ())],
    }
    comparison_observed = dict(observed)
    if "answer_correct" in served:
        comparison_observed["answer_correct"] = bool(served["answer_correct"])
    live_match, checks, observed_nodes = _compare_projection(
        comparison_observed, expected)
    previous_matches = []
    previous_comparisons = []
    for previous in registry.get("previous_registrations", ()):
        previous_id = str(previous.get("id"))
        try:
            projection = registered_projection(
                fixture,
                registry=registry,
                previous_registration_id=previous_id,
            )
        except BaselineRegistryError as exc:
            if "has no family" in str(exc):
                continue
            raise
        matched, previous_checks, previous_nodes = _compare_projection(
            comparison_observed, projection)
        previous_comparisons.append({
            "registration_id": previous_id,
            "match": bool(matched),
            "checks": previous_checks,
            "observed_mounted_nodes": previous_nodes,
        })
        if matched:
            previous_matches.append(previous_id)
    return {
        "schema": "grm.det1.live_baseline_comparison.v1",
        "anchor": anchor,
        "fixture_id": str(fixture.get("fixture_id")),
        "source_family": str(fixture.get("source_family")),
        "expected": expected,
        "observed": observed,
        "observed_mounted_nodes": observed_nodes,
        "comparator": dict(expected["comparator"]),
        "checks": checks,
        "live_match": bool(live_match),
        "previous_registration_matches": previous_matches,
        "previous_comparisons": previous_comparisons,
    }


def selftest() -> dict[str, Any]:
    registry, anchor = load_live_registry()
    fixture_source = dict(
        registry["families"]["supersession_battery_on_gpt_oss"]
        ["fixture_sources"]["correction_then_restatement"]
    )
    fixture = {
        "fixture_id": "sup_harbor_restatement",
        "source_family": "supersession_battery_on_gpt_oss",
        "source": fixture_source,
        "session_id": "correction_then_restatement",
        "scenario": "correction_then_restatement",
        "probe_id": "harbor_restatement",
        "question": "What is the current Harbor token value?",
        "target_node": "harbor_c",
        "expected_values": ["nacre-6-blue"],
        "stale_values": ["morrow-5-red"],
        "wrong_fact_values": ["opal-7-green"],
    }
    current = registered_projection(fixture, registry=registry)
    previous_id = str(registry["previous_registrations"][0]["id"])
    previous = registered_projection(
        fixture, registry=registry, previous_registration_id=previous_id)
    if not (
        current["answer"] == previous["answer"]
        and current["mounted_ids"] == [1]
        and current["mounted_nodes"] == ["harbor_b"]
        and previous["mounted_ids"] == [2]
        and previous["mounted_nodes"] == ["harbor_c"]
    ):
        raise BaselineRegistryError(
            "Harbor active/retired projection selftest failed")
    return {
        "schema": "grm.det1.baseline_registry_selftest.v1",
        "status": "PASS",
        "anchor": anchor,
        "harbor": {"active": current, "retired": previous},
    }


if __name__ == "__main__":
    print(json.dumps(selftest(), sort_keys=True))
