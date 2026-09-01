#!/usr/bin/env python3
"""Pure-CPU derivation and validation for the GRM-DET1.7 plant registry.

The registry is deliberately a registration artifact, not an evaluator.  It
reads finalized lived snapshot manifests, derives the only lawful mounted
target for each base fixture, and freezes enough evidence to replay that
derivation later.  Importing this module and deriving a registry have no write
side effects; only :func:`write_content_addressed_registry` writes a file.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


REGISTRY_SCHEMA = "grm.det1_7.plant_registry.v2"
BASE_REGISTRATION_SCHEMA = "grm.det1.registration.v1"
SNAPSHOT_SCHEMA = "grm.det1_3.model_visible_snapshot.v2"
ORDER_ID = "GRM-DET1.7"
AMENDMENT_ORDER_ID = "GRM-DET1.9"
BASE_ORDER_ID = "GRM-DET1"
CAPTURE_PHASE = "before_probe_prefill"
DECLARED_SYNTHESIS_BRANCH = "declared_synthesis_identified_set"
DETECTOR_IDS = ("D-LQR", "D-NGH", "D-ENT", "D-VERB")
ADJUDICATION_VOCABULARY = (
    "PARTIAL", "REFUTED", "SUPPORTED", "UNADJUDICATED",
)

ORDINARY_RULE_ID = "LIVED_FIRST_ACTUAL_IN_ADMISSION_ORDER_V1"
SYNTHESIS_RULE_ID = (
    "LIVED_UNIQUE_ESSENTIAL_IDENTIFIED_EXPECTED_CARRIER_V1"
)
SUBSTITUTION_SELECTION_RULE_ID = (
    "DET1_9_PRIMARY_THEN_CANONICAL_UNUSED_LAWFUL_RESERVE_V1"
)


class RegistryError(RuntimeError):
    """A DET1.7 registration invariant was not proved."""


def _validate_json(value: Any, where: str = "root") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RegistryError(f"non-finite JSON number at {where}")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise RegistryError(f"non-string JSON key at {where}: {key!r}")
            _validate_json(child, f"{where}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_json(child, f"{where}[{index}]")
        return
    raise RegistryError(f"non-JSON value at {where}: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Return the single compact UTF-8 representation used for all hashes."""
    _validate_json(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, *, record_root: Path | None = None) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file():
        raise RegistryError(f"registered file is absent: {path}")
    shown = str(path)
    if record_root is not None:
        try:
            shown = str(path.relative_to(Path(record_root).resolve()))
        except ValueError:
            pass
    return {
        "path": shown,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _record_path(record: Mapping[str, Any], record_root: Path | None) -> Path:
    shown = record.get("path")
    if not isinstance(shown, str) or not shown:
        raise RegistryError("file record has no path")
    path = Path(shown)
    if not path.is_absolute():
        if record_root is None:
            raise RegistryError(f"relative file record needs record_root: {shown}")
        path = Path(record_root) / path
    return path.resolve()


def _validate_file_record(
    record: Mapping[str, Any],
    *,
    record_root: Path | None,
    label: str,
) -> Path:
    if not isinstance(record, Mapping):
        raise RegistryError(f"{label} is not a file record")
    path = _record_path(record, record_root)
    if not path.is_file():
        raise RegistryError(f"{label} is absent: {path}")
    observed = {
        "path": record.get("path"),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if dict(record) != observed:
        raise RegistryError(f"{label} record does not match the file: {path}")
    return path


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RegistryError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise RegistryError(f"{label} is not a JSON object: {path}")
    return value


def _parse_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise RegistryError(f"{label} must be an explicit UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RegistryError(f"invalid {label}: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise RegistryError(f"{label} must be UTC: {value!r}")
    return parsed


def _id_list(value: Any, label: str, *, nonempty: bool = False) -> list[int]:
    if not isinstance(value, list):
        raise RegistryError(f"{label} must be a list")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise RegistryError(f"{label} contains an invalid graft id: {item!r}")
        result.append(item)
    if len(result) != len(set(result)):
        raise RegistryError(f"{label} contains duplicate graft ids")
    if nonempty and not result:
        raise RegistryError(f"{label} has no lived mounted seats")
    return result


def _base_fixture_map(registration: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if registration.get("schema") != BASE_REGISTRATION_SCHEMA:
        raise RegistryError("base registration has the wrong schema")
    if registration.get("order") != BASE_ORDER_ID:
        raise RegistryError("base registration does not bind GRM-DET1")
    fixtures = registration.get("fixtures")
    if not isinstance(fixtures, list) or len(fixtures) != 14:
        raise RegistryError("base registration must contain exactly 14 fixtures")
    result: dict[str, dict[str, Any]] = {}
    for index, fixture in enumerate(fixtures):
        if not isinstance(fixture, Mapping):
            raise RegistryError(f"base fixture {index} is not an object")
        fixture_id = fixture.get("fixture_id")
        if not isinstance(fixture_id, str) or not fixture_id:
            raise RegistryError(f"base fixture {index} has no fixture_id")
        if fixture_id in result:
            raise RegistryError(f"duplicate base fixture: {fixture_id}")
        if fixture.get("split") not in ("calibration", "eval"):
            raise RegistryError(f"invalid split for {fixture_id}")
        if not isinstance(fixture.get("source_family"), str):
            raise RegistryError(f"missing source_family for {fixture_id}")
        expected = fixture.get("expected_values")
        if not isinstance(expected, list) or not expected or not all(
            isinstance(item, str) and item for item in expected
        ):
            raise RegistryError(f"invalid expected_values for {fixture_id}")
        source = fixture.get("source")
        if not isinstance(source, Mapping) or not isinstance(
            source.get("sha256"), str
        ):
            raise RegistryError(f"invalid source binding for {fixture_id}")
        result[fixture_id] = dict(fixture)

    calibration = {
        key for key, fixture in result.items()
        if fixture["split"] == "calibration"
    }
    evaluation = set(result) - calibration
    if len(calibration) != 2 or len(evaluation) != 12:
        raise RegistryError("base split must be exactly 2 calibration + 12 eval")
    split_rule = registration.get("split_rule")
    if not isinstance(split_rule, Mapping):
        raise RegistryError("base registration lacks split_rule")
    if split_rule.get("calibration_pair_count") != 2:
        raise RegistryError("base calibration pair count changed")
    if split_rule.get("eval_pair_count") != 12:
        raise RegistryError("base eval pair count changed")
    if set(split_rule.get("calibration") or ()) != calibration:
        raise RegistryError("base calibration fixture projection changed")
    registered_eval = set(split_rule.get("eval_e2e") or ()) | set(
        split_rule.get("eval_supersession") or ()
    )
    if registered_eval != evaluation:
        raise RegistryError("base eval fixture projection changed")

    detectors = registration.get("detectors")
    if not isinstance(detectors, Mapping) or set(detectors) != set(DETECTOR_IDS):
        raise RegistryError("base detector arms changed")
    adjudication = registration.get("prediction_adjudication")
    if not isinstance(adjudication, Mapping) or set(adjudication) != set(
        ADJUDICATION_VOCABULARY
    ):
        raise RegistryError("base adjudication vocabulary changed")
    return result


def _session_projection(fixture: Mapping[str, Any]) -> dict[str, Any]:
    family = str(fixture["source_family"])
    source = fixture["source"]
    source_sha256 = str(source["sha256"])
    if family == "certified_34_turn":
        turn = fixture.get("turn")
        if isinstance(turn, bool) or not isinstance(turn, int) or turn < 0:
            raise RegistryError(f"certified fixture {fixture['fixture_id']} has no turn")
        return {
            "source_family": family,
            "session_id": f"certified_34_turn:{source_sha256}",
            "selector": {"turn": turn},
        }
    session_id = fixture.get("session_id")
    probe_id = fixture.get("probe_id")
    if not isinstance(session_id, str) or not session_id:
        raise RegistryError(f"fixture {fixture['fixture_id']} has no session_id")
    if not isinstance(probe_id, str) or not probe_id:
        raise RegistryError(f"fixture {fixture['fixture_id']} has no probe_id")
    return {
        "source_family": family,
        "session_id": session_id,
        "selector": {"probe_id": probe_id},
    }


def _certified_campaign_sessions(
    fixture_map: Mapping[str, Mapping[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Freeze every source/session pair certified by the base registration."""
    sessions: dict[tuple[str, str], dict[str, Any]] = {}
    for fixture in fixture_map.values():
        projection = _session_projection(fixture)
        key = (
            str(projection["source_family"]),
            str(projection["session_id"]),
        )
        source = dict(fixture["source"])
        prior = sessions.get(key)
        if prior is not None and prior != source:
            raise RegistryError(
                "certified campaign session resolves to multiple source records"
            )
        sessions[key] = source
    return sessions


def _substitution_policy(amendment_record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "amendment_order": AMENDMENT_ORDER_ID,
        "amendment_order_record": dict(amendment_record),
        "pool_scope": (
            "ANY_CERTIFIED_LIVED_COLLECTED_SESSION_IN_THIS_CAMPAIGN_"
            "INCLUDING_SUPERSESSION_BATTERY"
        ),
        "primary_precedence": "KEEP_EACH_LAWFUL_PRIMARY",
        "failed_slot_order": "FROZEN_BASE_REGISTRATION_FIXTURE_ORDER",
        "reserve_eligibility": (
            "LAWFUL_LIVED_TARGET_AND_SAME_FROZEN_SPLIT_AND_UNUSED_"
            "NON_BASE_EFFECTIVE_FIXTURE"
        ),
        "reserve_order": [
            "split",
            "source_family",
            "session_id",
            "canonical_selector_json",
            "effective_fixture_id",
            "row_id",
        ],
        "consume_each_reserve_at_most_once": True,
        "selection_rule_id": SUBSTITUTION_SELECTION_RULE_ID,
    }


def _string_list(value: Any, label: str, *, nonempty: bool = False) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise RegistryError(f"{label} must be a list of nonempty strings")
    if nonempty and not value:
        raise RegistryError(f"{label} must not be empty")
    if len({item.casefold() for item in value}) != len(value):
        raise RegistryError(f"{label} contains duplicate values")
    return list(value)


def _effective_fixture_from_base(
    fixture: Mapping[str, Any], session: Mapping[str, Any]
) -> dict[str, Any]:
    question = fixture.get("question")
    if not isinstance(question, str) or not question:
        raise RegistryError(f"fixture {fixture['fixture_id']} has no question")
    stale = fixture.get("stale_values")
    if stale is None:
        stale = fixture.get("old_values") or []
    return {
        "fixture_id": str(fixture["fixture_id"]),
        "split": str(fixture["split"]),
        "source_family": str(fixture["source_family"]),
        "session_id": str(session["session_id"]),
        "selector": dict(session["selector"]),
        "question": question,
        "expected_values": _string_list(
            fixture.get("expected_values"), "base expected_values", nonempty=True
        ),
        "stale_values": _string_list(stale, "base stale_values"),
        "wrong_fact_values": _string_list(
            fixture.get("wrong_fact_values") or [], "base wrong_fact_values"
        ),
        "source": dict(fixture["source"]),
    }


def _validate_substitution(
    value: Any,
    fixture: Mapping[str, Any],
    session: Mapping[str, Any],
    certified_sessions: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise RegistryError(
            f"fixture {fixture['fixture_id']} needs an explicit substitution object"
        )
    substitution = dict(value)
    status = substitution.get("status")
    if status == "NONE":
        if substitution != {"status": "NONE"}:
            raise RegistryError("NONE substitution must be exactly {'status': 'NONE'}")
        return substitution, _effective_fixture_from_base(fixture, session)
    if status != "SUBSTITUTED":
        raise RegistryError(f"invalid substitution status: {status!r}")
    required = {
        "status", "original_fixture_id", "effective_fixture", "reason",
    }
    if set(substitution) != required:
        raise RegistryError(
            "SUBSTITUTED object must enumerate exactly " + ", ".join(sorted(required))
        )
    original = substitution["original_fixture_id"]
    if original != fixture["fixture_id"]:
        raise RegistryError("substitution original_fixture_id differs from base slot")
    if not isinstance(substitution["reason"], str) or not substitution["reason"].strip():
        raise RegistryError("substitution must enumerate a reason")
    raw_effective = substitution["effective_fixture"]
    if not isinstance(raw_effective, Mapping):
        raise RegistryError("substitution effective_fixture is not an object")
    effective_required = {
        "fixture_id", "split", "source_family", "session_id", "selector",
        "question", "expected_values", "stale_values", "wrong_fact_values",
        "source",
    }
    if set(raw_effective) != effective_required:
        raise RegistryError(
            "effective_fixture must carry exactly "
            + ", ".join(sorted(effective_required))
        )
    effective = dict(raw_effective)
    replacement = effective["fixture_id"]
    if not isinstance(replacement, str) or not replacement or replacement == original:
        raise RegistryError("substitution effective fixture_id is invalid")
    if effective["split"] != fixture["split"]:
        raise RegistryError("substitution crosses the frozen split")
    session_key = (
        str(effective["source_family"]),
        str(effective["session_id"]),
    )
    certified_source = certified_sessions.get(session_key)
    if certified_source is None:
        raise RegistryError(
            "substitution source/session is not certified by this campaign"
        )
    if effective["source"] != certified_source:
        raise RegistryError(
            "substitution source record differs from its certified campaign session"
        )
    selector = effective["selector"]
    if not isinstance(selector, Mapping) or not selector:
        raise RegistryError("substitution must enumerate an effective selector")
    family = str(effective["source_family"])
    if family == "certified_34_turn":
        valid_selector = (
            set(selector) == {"turn"}
            and isinstance(selector.get("turn"), int)
            and not isinstance(selector.get("turn"), bool)
            and int(selector["turn"]) >= 0
        )
    else:
        valid_selector = (
            set(selector) == {"probe_id"}
            and isinstance(selector.get("probe_id"), str)
            and bool(str(selector["probe_id"]).strip())
        )
    if not valid_selector:
        raise RegistryError("substitution selector is invalid for its source family")
    if (
        effective["session_id"] == session["session_id"]
        and dict(selector) == dict(session["selector"])
    ):
        raise RegistryError("substitution selector does not replace the original probe")
    question = effective["question"]
    if not isinstance(question, str) or not question:
        raise RegistryError("substitution effective question is empty")
    effective["selector"] = dict(selector)
    effective["expected_values"] = _string_list(
        effective["expected_values"],
        "substitution effective expected_values",
        nonempty=True,
    )
    effective["stale_values"] = _string_list(
        effective["stale_values"], "substitution effective stale_values"
    )
    effective["wrong_fact_values"] = _string_list(
        effective["wrong_fact_values"],
        "substitution effective wrong_fact_values",
    )
    effective["source"] = dict(effective["source"])
    normalized = {
        "status": "SUBSTITUTED",
        "original_fixture_id": original,
        "effective_fixture": effective,
        "reason": substitution["reason"],
    }
    return normalized, effective


def _snapshot_payload(path: Path) -> dict[str, Any]:
    manifest = _read_object(path, "lived snapshot manifest")
    if manifest.get("schema") != SNAPSHOT_SCHEMA:
        raise RegistryError("source snapshot has the wrong schema")
    digest = manifest.get("manifest_payload_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_payload_sha256", None)
    if not isinstance(digest, str) or digest != canonical_sha256(unsigned):
        raise RegistryError("source snapshot manifest payload digest mismatch")
    return manifest


def _normalize_value_text(value: str) -> str:
    return (
        str(value)
        .replace("\u2010", "-")
        .replace("\u2011", "-")
        .replace("**", "")
        .casefold()
    )


def _contains_value(text: str, value: str) -> bool:
    # DET1.10: separator glyphs (ASCII hyphen, U+2010, U+2011, space) are
    # presentation within the value token sequence; token payload and token
    # count stay load-bearing.  Shares the campaign comparator's pattern
    # builder so registry lawfulness and served-control classification cannot
    # drift apart.
    from scripts.grm_det1_common import value_separator_regex

    haystack = _normalize_value_text(text)
    needle = _normalize_value_text(value)
    return bool(re.search(
        rf"(?<![a-z0-9_-]){value_separator_regex(needle)}(?![a-z0-9_-])",
        haystack,
    ))


def _answer_semantics(answer: str, effective_fixture: Mapping[str, Any]) -> dict[str, Any]:
    expected = [
        item for item in effective_fixture["expected_values"]
        if _contains_value(answer, item)
    ]
    stale = [
        item for item in effective_fixture["stale_values"]
        if _contains_value(answer, item)
    ]
    wrong = [
        item for item in effective_fixture["wrong_fact_values"]
        if _contains_value(answer, item)
    ]
    return {
        "expected_matches": expected,
        "stale_matches": stale,
        "wrong_fact_matches": wrong,
        "correct": (
            len(expected) == len(effective_fixture["expected_values"])
            and not stale
            and not wrong
        ),
    }


def _snapshot_admission(
    manifest: Mapping[str, Any],
    *,
    effective_fixture: Mapping[str, Any],
    registration_sha256: str,
) -> dict[str, Any]:
    if manifest.get("complete") is not True:
        raise RegistryError("source snapshot is incomplete")
    if manifest.get("capture_finalized") is not True:
        raise RegistryError("source snapshot is not finalized")
    if manifest.get("label") != "lived" or manifest.get("phase") != CAPTURE_PHASE:
        raise RegistryError("source snapshot is not lived before_probe_prefill evidence")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping) or not (
        provenance.get("arm") == "lived"
        and provenance.get("capture_boundary") == CAPTURE_PHASE
        and provenance.get("fixture_id") == effective_fixture["fixture_id"]
        and provenance.get("registration_sha256") == registration_sha256
    ):
        raise RegistryError("source snapshot provenance does not bind the lived fixture")

    state = manifest.get("state")
    if not isinstance(state, Mapping):
        raise RegistryError("source snapshot has no state")
    authoritative = _id_list(
        state.get("admission.authoritative_mounts"),
        "admission.authoritative_mounts",
        nonempty=True,
    )
    final_mounts = _id_list(
        state.get("admission.final_mounts"),
        "admission.final_mounts",
        nonempty=True,
    )
    arena_mounts = _id_list(
        state.get("arena.cur_mounts"), "arena.cur_mounts", nonempty=True
    )
    if authoritative != final_mounts or authoritative != arena_mounts:
        raise RegistryError("authoritative/final/arena lived mounts disagree")
    fields = {
        "actual_authoritative_mounts": authoritative,
        "actual_final_mounts": final_mounts,
        "rank_plan": _id_list(state.get("admission.rank_plan"), "rank_plan"),
        "current_planned": _id_list(
            state.get("admission.current_planned"), "current_planned"
        ),
        "ranking": _id_list(state.get("admission.ranking"), "ranking"),
        "identified_candidates": _id_list(
            state.get("admission.identified_candidates"), "identified_candidates"
        ),
    }
    branch = state.get("admission.policy_branch")
    if not isinstance(branch, str) or not branch:
        raise RegistryError("snapshot has no admission policy branch")
    fields["policy_branch"] = branch
    if not fields["rank_plan"] or not fields["current_planned"]:
        raise RegistryError("snapshot has no admission order")
    if not set(fields["rank_plan"]).issubset(fields["ranking"]):
        raise RegistryError("rank_plan is not a projection of ranking")
    if not set(fields["identified_candidates"]).issubset(fields["ranking"]):
        raise RegistryError("identified candidates are not in ranking")

    full_plan = state.get("admission.full_plan")
    if full_plan is not None:
        if not isinstance(full_plan, Mapping):
            raise RegistryError("admission.full_plan is not an object")
        for flat_key, plan_key in (
            ("rank_plan", "rank_plan"),
            ("current_planned", "current_planned"),
            ("ranking", "ranking"),
            ("identified_candidates", "identified_candidates"),
            ("actual_final_mounts", "final_mounts"),
            ("policy_branch", "policy_branch"),
        ):
            if fields[flat_key] != full_plan.get(plan_key):
                raise RegistryError(f"flattened admission differs from full_plan.{plan_key}")

    answer = manifest.get("linked_answer")
    if not isinstance(answer, Mapping):
        raise RegistryError("finalized snapshot has no linked served answer")
    if answer.get("attempt_answer_correct") is not True or answer.get(
        "probe_answer_correct"
    ) is not True:
        raise RegistryError("linked served answer is not correct")
    if answer.get("attempt_refusal") is not False or answer.get("probe_refusal") is not False:
        raise RegistryError("linked served answer is a refusal")
    if not isinstance(answer.get("attempt_answer"), str) or not answer["attempt_answer"]:
        raise RegistryError("linked attempt answer is empty")
    if not isinstance(answer.get("probe_answer"), str) or not answer["probe_answer"]:
        raise RegistryError("linked probe answer is empty")
    for label, answer_text in (
        ("attempt", answer["attempt_answer"]),
        ("probe", answer["probe_answer"]),
    ):
        semantics = _answer_semantics(answer_text, effective_fixture)
        if not semantics["correct"]:
            raise RegistryError(
                f"linked served {label} answer differs from effective fixture "
                f"semantics: {semantics}"
            )
    attempt_mounts = _id_list(answer.get("attempt_mounts"), "linked attempt_mounts")
    probe_mounts = _id_list(answer.get("probe_mounts"), "linked probe_mounts")
    if attempt_mounts != authoritative or probe_mounts != authoritative:
        raise RegistryError("linked served answer mounts differ from lived mounts")
    if answer.get("process_instance_sha256") != provenance.get(
        "process_instance_sha256"
    ):
        raise RegistryError("linked served answer comes from another process")
    if answer.get("captured_attempt_ordinal") != provenance.get("attempt_ordinal"):
        raise RegistryError("linked served answer comes from another ladder attempt")
    if answer.get("probe_selected_attempt") != answer.get("captured_attempt_ordinal"):
        raise RegistryError("linked served answer selected another ladder attempt")
    return fields


def _coverage_projection(
    value: Any,
    *,
    actual_mounts: Sequence[int],
    expected_values: Sequence[str],
) -> tuple[dict[str, list[str]], dict[int, set[str]], set[str]]:
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise RegistryError("expected_value_coverage must be an object")
    expected = {item.casefold() for item in expected_values}
    normalized: dict[str, list[str]] = {}
    coverage: dict[int, set[str]] = {}
    for raw_id, raw_values in value.items():
        try:
            graft_id = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise RegistryError(f"invalid expected-value carrier id: {raw_id!r}") from exc
        if str(graft_id) != str(raw_id) or graft_id not in actual_mounts:
            raise RegistryError("expected-value coverage names a non-mounted graft")
        if not isinstance(raw_values, list) or not all(
            isinstance(item, str) and item for item in raw_values
        ):
            raise RegistryError(f"invalid expected-value coverage for graft {graft_id}")
        values = {item.casefold() for item in raw_values}
        if not values.issubset(expected):
            raise RegistryError("coverage evidence contains an unregistered expected value")
        normalized[str(graft_id)] = list(raw_values)
        coverage[graft_id] = values
    return normalized, coverage, expected


def _select_target(
    admission: Mapping[str, Any],
    *,
    coverage: Mapping[int, set[str]],
    expected: set[str],
) -> tuple[int, str, str]:
    actual = list(admission["actual_authoritative_mounts"])
    branch = admission["policy_branch"]
    if branch == DECLARED_SYNTHESIS_BRANCH:
        total = set().union(*(coverage.get(item, set()) for item in actual))
        if not expected.issubset(total):
            raise RegistryError(
                "declared synthesis lacks mounted expected-value coverage"
            )
        identified_actual = [
            item for item in admission["identified_candidates"] if item in actual
        ]
        essential = []
        for candidate in identified_actual:
            remaining = set().union(*(
                coverage.get(item, set())
                for item in actual if item != candidate
            ))
            if not expected.issubset(remaining):
                essential.append(candidate)
        if len(essential) != 1:
            raise RegistryError(
                "declared synthesis needs exactly one actually mounted identified "
                f"essential expected-value carrier; observed {essential}"
            )
        target = essential[0]
        return (
            target,
            SYNTHESIS_RULE_ID,
            "unique actually mounted identified carrier whose removal eliminates "
            "registered expected-value coverage",
        )

    ordered = list(dict.fromkeys(
        list(admission["rank_plan"]) + list(admission["current_planned"])
    ))
    target = next((item for item in ordered if item in actual), None)
    if target is None:
        raise RegistryError("no actual lived mount occurs in rank_plan/current_planned")
    return (
        target,
        ORDINARY_RULE_ID,
        "first actually mounted graft in frozen rank_plan/current_planned order",
    )


def _validate_behavioral_breaker(
    value: Any,
    *,
    manifest: Mapping[str, Any],
    admission: Mapping[str, Any],
    effective_fixture: Mapping[str, Any],
    target: int,
    aliases: Sequence[int],
) -> dict[str, Any]:
    if admission["policy_branch"] != DECLARED_SYNTHESIS_BRANCH:
        not_applicable = {"status": "NOT_REQUIRED_FOR_NON_DECLARED_BRANCH"}
        if value is not None and value != not_applicable:
            raise RegistryError(
                "behavioral breaker evidence is only registered for declared synthesis"
            )
        return not_applicable
    if not isinstance(value, Mapping):
        raise RegistryError(
            "declared synthesis requires explicit target-only behavioral breaker evidence"
        )
    breaker = dict(value)
    required = {
        "schema", "source_manifest_payload_sha256", "withheld_target_id",
        "withheld_alias_ids", "served", "counterfactual", "delta_receipt",
    }
    allowed = required | {"delta_receipt_sha256"}
    if set(breaker) not in (required, allowed):
        raise RegistryError(
            "behavioral breaker must carry source, target, exact aliases, served, "
            "counterfactual, and delta receipt"
        )
    if breaker["schema"] != "grm.det1_7.target_only_behavioral_breaker.v1":
        raise RegistryError("behavioral breaker has the wrong schema")
    if breaker["source_manifest_payload_sha256"] != manifest.get(
        "manifest_payload_sha256"
    ):
        raise RegistryError("behavioral breaker does not bind the same lived snapshot")
    if breaker["withheld_target_id"] != target:
        raise RegistryError("behavioral breaker withholds a different target")
    withheld_aliases = _id_list(
        breaker["withheld_alias_ids"], "behavioral breaker withheld_alias_ids",
        nonempty=True,
    )
    if withheld_aliases != list(aliases):
        raise RegistryError("behavioral breaker does not withhold the exact target aliases")

    linked = manifest["linked_answer"]
    expected_served = {
        "answer": linked["probe_answer"],
        "answer_correct": True,
        "refusal": False,
        "mounted_ids": list(admission["actual_authoritative_mounts"]),
    }
    if breaker["served"] != expected_served:
        raise RegistryError(
            "behavioral breaker served arm is not the same correct non-refusal snapshot"
        )
    counterfactual = breaker["counterfactual"]
    counterfactual_keys = {
        "answer", "answer_correct", "refusal", "mounted_ids", "classification",
    }
    if not isinstance(counterfactual, Mapping) or set(counterfactual) != counterfactual_keys:
        raise RegistryError("behavioral breaker counterfactual is incomplete")
    if not isinstance(counterfactual["answer"], str):
        raise RegistryError("behavioral breaker counterfactual answer is not text")
    if counterfactual["answer_correct"] is not False:
        raise RegistryError("behavioral breaker counterfactual did not become incorrect")
    if not isinstance(counterfactual["refusal"], bool):
        raise RegistryError("behavioral breaker counterfactual refusal is not boolean")
    remaining = [
        item for item in admission["actual_authoritative_mounts"]
        if item not in aliases
    ]
    if _id_list(
        counterfactual["mounted_ids"], "behavioral breaker counterfactual mounts"
    ) != remaining:
        raise RegistryError("behavioral breaker counterfactual is not target-only")
    semantics = _answer_semantics(counterfactual["answer"], effective_fixture)
    if counterfactual["refusal"]:
        classification = "refusal"
    elif semantics["stale_matches"]:
        classification = "stale"
    elif semantics["wrong_fact_matches"]:
        classification = "wrong_fact"
    elif not semantics["correct"]:
        classification = "incorrect"
    else:
        raise RegistryError(
            "behavioral breaker counterfactual remains semantically correct"
        )
    if counterfactual["classification"] != classification:
        raise RegistryError("behavioral breaker classification is not evidence-derived")

    receipt = breaker["delta_receipt"]
    if not isinstance(receipt, Mapping):
        raise RegistryError("behavioral breaker delta receipt is not an object")
    if not (
        receipt.get("schema") == "grm.det1_6.fork_hydration_delta.v1"
        and receipt.get("status")
        == "PASS_EXACT_REGISTERED_DELTA_CANONICAL_VALUE_BYTES"
        and receipt.get("gate_pass") is True
        and receipt.get("exact_divergence_set") is True
        and receipt.get("non_delta_fields_equal") is True
        and receipt.get("retained_arrays_exact") is True
        and receipt.get("already_absent_aliases") == []
        and receipt.get("unexpected_divergent_fields") == []
        and receipt.get("missing_expected_divergent_fields") == []
        and receipt.get("expected_fork_value_mismatches") == []
        and receipt.get("withheld_logical_aliases") == list(aliases)
        and receipt.get("source_mounted_aliases") == list(aliases)
        and receipt.get("fork_mounts") == remaining
    ):
        raise RegistryError("behavioral breaker delta receipt did not prove exact target-only delta")
    target_absence = receipt.get("target_absence")
    if not isinstance(target_absence, Mapping) or target_absence.get(
        "gate_pass"
    ) is not True:
        raise RegistryError("behavioral breaker delta receipt lacks target absence")
    absence_checks = target_absence.get("checks")
    if not isinstance(absence_checks, Mapping) or not absence_checks or not all(
        check is True for check in absence_checks.values()
    ):
        raise RegistryError("behavioral breaker target-absence checks did not all pass")
    receipt_digest = canonical_sha256(dict(receipt))
    claimed_receipt_digest = breaker.get("delta_receipt_sha256")
    if claimed_receipt_digest is not None and claimed_receipt_digest != receipt_digest:
        raise RegistryError("behavioral breaker delta receipt digest mismatch")
    breaker["served"] = dict(breaker["served"])
    breaker["counterfactual"] = dict(counterfactual)
    breaker["delta_receipt"] = dict(receipt)
    breaker["delta_receipt_sha256"] = receipt_digest
    return breaker


def _policy_bindings(registration: Mapping[str, Any]) -> dict[str, Any]:
    detectors = dict(registration["detectors"])
    threshold_policy = {
        detector: {
            "threshold_fit_present": "threshold_fit" in detectors[detector],
            "threshold_fit": detectors[detector].get("threshold_fit"),
            "trigger": detectors[detector].get("trigger"),
        }
        for detector in DETECTOR_IDS
    }
    adjudication = dict(registration["prediction_adjudication"])
    return {
        "detectors": {
            "projection": detectors,
            "sha256": canonical_sha256(detectors),
        },
        "threshold_policy": {
            "projection": threshold_policy,
            "sha256": canonical_sha256(threshold_policy),
        },
        "adjudication": {
            "projection": adjudication,
            "sha256": canonical_sha256(adjudication),
            "vocabulary": list(ADJUDICATION_VOCABULARY),
        },
    }


def _validate_effective_uniqueness(
    entries: Sequence[Mapping[str, Any]], base_fixture_ids: set[str]
) -> None:
    effective_ids = [str(entry["effective_fixture_id"]) for entry in entries]
    if len(effective_ids) != len(set(effective_ids)):
        raise RegistryError("duplicate effective fixture ids/replacements")
    replacement_ids = [
        str(entry["effective_fixture_id"])
        for entry in entries
        if entry["substitution"]["status"] == "SUBSTITUTED"
    ]
    if len(replacement_ids) != len(set(replacement_ids)):
        raise RegistryError("duplicate substitution replacements")
    collisions = sorted(set(replacement_ids) & base_fixture_ids)
    if collisions:
        raise RegistryError(
            f"substitution replacement reuses another base fixture id: {collisions}"
        )
    selectors = [
        (
            str(entry["effective_fixture"]["session_id"]),
            canonical_sha256(entry["effective_fixture"]["selector"]),
        )
        for entry in entries
    ]
    if len(selectors) != len(set(selectors)):
        raise RegistryError("duplicate effective campaign-session selectors/replacements")


def _observation_map(observations: Any) -> dict[str, dict[str, Any]]:
    if isinstance(observations, Mapping):
        result = {}
        for key, value in observations.items():
            if not isinstance(value, Mapping):
                raise RegistryError(f"observation {key!r} is not an object")
            row = dict(value)
            if "fixture_id" in row and row["fixture_id"] != str(key):
                raise RegistryError(f"observation key/fixture mismatch: {key}")
            row["fixture_id"] = str(key)
            result[str(key)] = row
        return result
    if not isinstance(observations, Sequence) or isinstance(observations, (str, bytes)):
        raise RegistryError("observations must be a mapping or sequence")
    result = {}
    for row in observations:
        if not isinstance(row, Mapping):
            raise RegistryError("observation is not an object")
        fixture_id = row.get("fixture_id")
        if not isinstance(fixture_id, str) or not fixture_id or fixture_id in result:
            raise RegistryError(f"invalid/duplicate observation fixture_id: {fixture_id!r}")
        result[fixture_id] = dict(row)
    return result


def _derive_entry(
    fixture: Mapping[str, Any],
    observation: Mapping[str, Any],
    *,
    registration_sha256: str,
    certified_sessions: Mapping[tuple[str, str], Mapping[str, Any]],
    record_root: Path | None,
) -> dict[str, Any]:
    fixture_id = str(fixture["fixture_id"])
    session = _session_projection(fixture)
    substitution, effective_fixture = _validate_substitution(
        observation.get("substitution"), fixture, session, certified_sessions
    )
    effective_fixture_id = str(effective_fixture["fixture_id"])
    raw_path = observation.get("snapshot_path")
    if not isinstance(raw_path, (str, Path)):
        raise RegistryError(f"observation {fixture_id} has no snapshot_path")
    snapshot_path = Path(raw_path).resolve()
    snapshot_record = file_record(snapshot_path, record_root=record_root)
    manifest = _snapshot_payload(snapshot_path)
    admission = _snapshot_admission(
        manifest,
        effective_fixture=effective_fixture,
        registration_sha256=registration_sha256,
    )
    coverage_projection, coverage, expected = _coverage_projection(
        observation.get("expected_value_coverage"),
        actual_mounts=admission["actual_authoritative_mounts"],
        expected_values=effective_fixture["expected_values"],
    )
    target, rule_id, reason = _select_target(
        admission, coverage=coverage, expected=expected
    )
    raw_aliases = observation.get("alias_ids", [target])
    aliases = _id_list(raw_aliases, "alias_ids", nonempty=True)
    actual = set(admission["actual_authoritative_mounts"])
    if target not in aliases:
        raise RegistryError("selected target is absent from alias_ids")
    if not set(aliases).issubset(actual):
        raise RegistryError("alias_ids include grafts without lived mounted seats")
    breaker = _validate_behavioral_breaker(
        observation.get("behavioral_breaker"),
        manifest=manifest,
        admission=admission,
        effective_fixture=effective_fixture,
        target=target,
        aliases=aliases,
    )
    evidence_utc = observation.get("evidence_utc")
    _parse_utc(evidence_utc, f"{fixture_id} evidence_utc")
    entry = {
        "fixture_id": fixture_id,
        "effective_fixture_id": effective_fixture_id,
        "effective_fixture": effective_fixture,
        "base_fixture_sha256": canonical_sha256(dict(fixture)),
        "split": fixture["split"],
        "session": session,
        "source_snapshot": snapshot_record,
        "manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "evidence_utc": evidence_utc,
        **admission,
        "expected_value_coverage": coverage_projection,
        "selected_target_id": target,
        "alias_ids": aliases,
        "rule_id": rule_id,
        "rule_reason": reason,
        "behavioral_breaker": breaker,
        "served_row_id": f"{effective_fixture_id}:served",
        "planted_row_id": f"{effective_fixture_id}:planted_miss",
        "substitution": substitution,
    }
    if target not in entry["actual_authoritative_mounts"]:
        raise RegistryError("selected target has no lived mounted seats")
    return entry


def derive_plant_registry(
    base_registration_path: Path,
    order_path: Path,
    observations: Any,
    *,
    created_utc: str,
    new_eval_evidence_utc: Sequence[str] = (),
    record_root: Path | None = None,
) -> dict[str, Any]:
    """Derive a frozen DET1.7 registry without writing any artifact."""
    created = _parse_utc(created_utc, "registry created_utc")
    base_registration_path = Path(base_registration_path).resolve()
    order_path = Path(order_path).resolve()
    registration_record = file_record(
        base_registration_path, record_root=record_root
    )
    order_record = file_record(order_path, record_root=record_root)
    registration = _read_object(base_registration_path, "base registration")
    fixture_map = _base_fixture_map(registration)
    certified_sessions = _certified_campaign_sessions(fixture_map)
    if ORDER_ID not in order_path.read_text(encoding="utf-8"):
        raise RegistryError("order record does not identify GRM-DET1.7")
    amendment_order_path = order_path.with_name("GRM_DET1_9_SUBSTITUTION_POOL.md")
    amendment_record = file_record(
        amendment_order_path, record_root=record_root
    )
    if AMENDMENT_ORDER_ID not in amendment_order_path.read_text(encoding="utf-8"):
        raise RegistryError("amendment order record does not identify GRM-DET1.9")
    observed = _observation_map(observations)
    if set(observed) != set(fixture_map):
        missing = sorted(set(fixture_map) - set(observed))
        extra = sorted(set(observed) - set(fixture_map))
        raise RegistryError(
            f"observations must cover exactly all 14 base fixtures; "
            f"missing={missing}, extra={extra}"
        )
    entries = [
        _derive_entry(
            fixture,
            observed[fixture_id],
            registration_sha256=registration_record["sha256"],
            certified_sessions=certified_sessions,
            record_root=record_root,
        )
        for fixture_id, fixture in fixture_map.items()
    ]
    _validate_effective_uniqueness(entries, set(fixture_map))
    for entry in entries:
        if _parse_utc(entry["evidence_utc"], "lived evidence_utc") > created:
            raise RegistryError("registry was created before its lived evidence")
    eval_times = list(new_eval_evidence_utc)
    for value in eval_times:
        if _parse_utc(value, "new eval evidence_utc") <= created:
            raise RegistryError("registry was not frozen before new eval evidence")
    substitutions = [
        {
            "fixture_id": entry["fixture_id"],
            **entry["substitution"],
        }
        for entry in entries if entry["substitution"]["status"] == "SUBSTITUTED"
    ]
    registry: dict[str, Any] = {
        "schema": REGISTRY_SCHEMA,
        "order": ORDER_ID,
        "order_record": order_record,
        "base_registration": registration_record,
        "created_utc": created_utc,
        "chronology": {
            "lived_evidence_utc": {
                entry["fixture_id"]: entry["evidence_utc"] for entry in entries
            },
            "new_eval_evidence_utc": eval_times,
            "freeze_status": (
                "FROZEN_BEFORE_ALL_RECORDED_NEW_EVAL_EVIDENCE"
                if eval_times else "FROZEN_WITH_NO_NEW_EVAL_EVIDENCE"
            ),
        },
        "fixture_count": 14,
        "pair_counts": {
            "all_fixture_pairs": 14,
            "calibration_fixture_pairs": 2,
            "eval_fixture_pairs": 12,
            "eval_served_rows": 12,
            "eval_planted_miss_rows": 12,
        },
        "policy_bindings": _policy_bindings(registration),
        "substitution_policy": _substitution_policy(amendment_record),
        "substitution_count": len(substitutions),
        "substitutions": substitutions,
        "entries": entries,
    }
    registry["registry_payload_sha256"] = canonical_sha256(registry)
    validate_plant_registry(registry, record_root=record_root)
    return registry


def validate_plant_registry(
    registry: Mapping[str, Any],
    *,
    record_root: Path | None = None,
) -> dict[str, Any]:
    """Re-read all bound files and replay every DET1.7 selection proof."""
    if not isinstance(registry, Mapping):
        raise RegistryError("registry is not an object")
    value = dict(registry)
    if value.get("schema") != REGISTRY_SCHEMA or value.get("order") != ORDER_ID:
        raise RegistryError("unsupported DET1.7 registry schema/order")
    claimed_digest = value.pop("registry_payload_sha256", None)
    if not isinstance(claimed_digest, str) or claimed_digest != canonical_sha256(value):
        raise RegistryError("registry payload digest mismatch")
    created = _parse_utc(value.get("created_utc"), "registry created_utc")
    order_path = _validate_file_record(
        value.get("order_record") or {}, record_root=record_root, label="order"
    )
    if ORDER_ID not in order_path.read_text(encoding="utf-8"):
        raise RegistryError("bound order does not identify GRM-DET1.7")
    registration_path = _validate_file_record(
        value.get("base_registration") or {},
        record_root=record_root,
        label="base registration",
    )
    registration = _read_object(registration_path, "base registration")
    fixture_map = _base_fixture_map(registration)
    certified_sessions = _certified_campaign_sessions(fixture_map)
    if value.get("policy_bindings") != _policy_bindings(registration):
        raise RegistryError("detector/threshold/adjudication binding changed")
    policy = value.get("substitution_policy")
    if not isinstance(policy, Mapping):
        raise RegistryError("registry lacks the DET1.9 substitution policy")
    amendment_path = _validate_file_record(
        policy.get("amendment_order_record") or {},
        record_root=record_root,
        label="DET1.9 amendment order",
    )
    if amendment_path != order_path.with_name(
        "GRM_DET1_9_SUBSTITUTION_POOL.md"
    ).resolve():
        raise RegistryError("registry binds the wrong DET1.9 amendment order path")
    if AMENDMENT_ORDER_ID not in amendment_path.read_text(encoding="utf-8"):
        raise RegistryError("bound amendment order does not identify GRM-DET1.9")
    if dict(policy) != _substitution_policy(policy["amendment_order_record"]):
        raise RegistryError("DET1.9 substitution policy drifted")
    expected_counts = {
        "all_fixture_pairs": 14,
        "calibration_fixture_pairs": 2,
        "eval_fixture_pairs": 12,
        "eval_served_rows": 12,
        "eval_planted_miss_rows": 12,
    }
    if value.get("fixture_count") != 14 or value.get("pair_counts") != expected_counts:
        raise RegistryError("registry did not preserve the 2+12 / 12+12 counts")
    entries = value.get("entries")
    if not isinstance(entries, list) or len(entries) != 14:
        raise RegistryError("registry must contain exactly 14 entries")
    if [entry.get("fixture_id") for entry in entries if isinstance(entry, Mapping)] != list(
        fixture_map
    ):
        raise RegistryError("registry entries differ from base fixture order")

    replayed = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise RegistryError("registry entry is not an object")
        fixture_id = str(entry["fixture_id"])
        source_path = _validate_file_record(
            entry.get("source_snapshot") or {},
            record_root=record_root,
            label=f"{fixture_id} source snapshot",
        )
        observation = {
            "fixture_id": fixture_id,
            "snapshot_path": source_path,
            "evidence_utc": entry.get("evidence_utc"),
            "expected_value_coverage": entry.get("expected_value_coverage"),
            "alias_ids": entry.get("alias_ids"),
            "substitution": entry.get("substitution"),
            "behavioral_breaker": entry.get("behavioral_breaker"),
        }
        expected = _derive_entry(
            fixture_map[fixture_id],
            observation,
            registration_sha256=(value["base_registration"])["sha256"],
            certified_sessions=certified_sessions,
            record_root=record_root,
        )
        if dict(entry) != expected:
            raise RegistryError(f"registry entry derivation changed: {fixture_id}")
        if _parse_utc(entry["evidence_utc"], "lived evidence_utc") > created:
            raise RegistryError("registry predates lived evidence")
        replayed.append(expected)

    _validate_effective_uniqueness(replayed, set(fixture_map))

    chronology = value.get("chronology")
    if not isinstance(chronology, Mapping):
        raise RegistryError("registry lacks chronology")
    evidence_projection = {
        entry["fixture_id"]: entry["evidence_utc"] for entry in replayed
    }
    if chronology.get("lived_evidence_utc") != evidence_projection:
        raise RegistryError("chronology differs from per-entry lived evidence")
    eval_times = chronology.get("new_eval_evidence_utc")
    if not isinstance(eval_times, list):
        raise RegistryError("new_eval_evidence_utc must be a list")
    for stamp in eval_times:
        if _parse_utc(stamp, "new eval evidence_utc") <= created:
            raise RegistryError("registry was not frozen before new eval evidence")
    expected_status = (
        "FROZEN_BEFORE_ALL_RECORDED_NEW_EVAL_EVIDENCE"
        if eval_times else "FROZEN_WITH_NO_NEW_EVAL_EVIDENCE"
    )
    if chronology.get("freeze_status") != expected_status:
        raise RegistryError("freeze chronology status is inconsistent")

    substitutions = [
        {"fixture_id": entry["fixture_id"], **entry["substitution"]}
        for entry in replayed
        if entry["substitution"]["status"] == "SUBSTITUTED"
    ]
    if value.get("substitution_count") != len(substitutions) or value.get(
        "substitutions"
    ) != substitutions:
        raise RegistryError("substitution enumeration differs from entries")
    served = [entry["served_row_id"] for entry in replayed]
    planted = [entry["planted_row_id"] for entry in replayed]
    if len(set(served)) != 14 or len(set(planted)) != 14:
        raise RegistryError("served/planted row ids are not one-to-one")
    return {
        "schema": "grm.det1_7.plant_registry_validation.v1",
        "status": "PASS",
        "registry_payload_sha256": claimed_digest,
        "fixture_count": 14,
        "calibration_count": 2,
        "eval_count": 12,
        "eval_served_count": 12,
        "eval_planted_count": 12,
        "substitution_count": len(substitutions),
    }


def write_content_addressed_registry(
    directory: Path,
    registry: Mapping[str, Any],
    *,
    record_root: Path | None = None,
    stem: str = "grm_det1_7_plant_registry",
) -> Path:
    """Validate, then exclusively write deterministic content-addressed JSON."""
    validate_plant_registry(registry, record_root=record_root)
    payload = canonical_json_bytes(dict(registry)) + b"\n"
    digest = hashlib.sha256(payload).hexdigest()
    path = Path(directory) / f"{stem}_{digest[:16]}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RegistryError(f"content-address collision: {path}")
        return path
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
    return path


# Short aliases for callers that use the repository's generic registry naming.
build_registry = derive_plant_registry
validate_registry = validate_plant_registry
write_registry = write_content_addressed_registry


__all__ = [
    "AMENDMENT_ORDER_ID",
    "RegistryError",
    "SUBSTITUTION_SELECTION_RULE_ID",
    "build_registry",
    "canonical_json_bytes",
    "canonical_sha256",
    "derive_plant_registry",
    "file_record",
    "sha256_file",
    "validate_plant_registry",
    "validate_registry",
    "write_content_addressed_registry",
    "write_registry",
]
