"""GRM lived-serving reliability census — classification and receipt contracts.

The census is evidence, not analysis.  These tests pin the two properties
that make it trustworthy as evidence:

1. The classification is a genuine partition — refusal, wrong-value, and
   lawful (with the DET1.10 separator rescue visible as its own subclass).
2. It refuses to absorb anything it does not understand: an unknown
   unplantable reason, a comparator disagreement, or a probe with no
   expected values all fail closed rather than being silently binned.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.grm_det1_common import DETError  # noqa: E402
from scripts.grm_det1_11_census import (  # noqa: E402
    CLASS_LAWFUL,
    CLASS_REFUSAL,
    CLASS_WRONG_VALUE,
    SUBCLASS_EXACT,
    SUBCLASS_SEPARATOR,
    TITLE,
    UNPLANTABLE_REASON,
    build_census,
    classify_observation,
    collect_latest_observations,
    selftest,
    write_census,
)


def _observation(
    probe_id: str,
    *,
    status: str,
    answer: str,
    expected: list[str],
    reserve: bool = False,
    split: str = "eval",
    mounted: list[int] | None = None,
) -> dict:
    row = {
        "fixture_id": probe_id,
        "effective_fixture_id": probe_id,
        "status": status,
        "served_answer": answer,
        "served_answer_correct": status == "LAWFUL_LIVED_TARGET",
        "reserve_candidate": reserve,
        "mounted_ids": [1] if mounted is None else mounted,
        "selected_target_id": 1 if status == "LAWFUL_LIVED_TARGET" else None,
        "effective_fixture": {
            "split": split,
            "source_family": "certified_34_turn",
            "expected_values": expected,
        },
    }
    if status != "LAWFUL_LIVED_TARGET":
        row["unplantable_reason"] = UNPLANTABLE_REASON
    return row


def _write_shard(root: Path, spec: str, attempt: int, rows: list[dict]) -> None:
    directory = root / spec / f"attempt_{attempt:03d}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "observations.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_selftest_passes_and_declares_cause_not_investigated():
    result = selftest()
    assert result["status"] == "PASS"
    assert result["cause_investigated"] is False


def test_refusal_is_classified_apart_from_a_wrong_value():
    refusal = classify_observation(_observation(
        "p", status="UNPLANTABLE",
        answer="I'm sorry, but I don't have that information.",
        expected=["Marble-4-Juliet"],
    ))
    wrong = classify_observation(_observation(
        "p", status="UNPLANTABLE", answer="Birch-2-Beacon.",
        expected=["cobalt-3-comet"],
    ))
    assert refusal["census_class"] == CLASS_REFUSAL
    assert wrong["census_class"] == CLASS_WRONG_VALUE


def test_separator_variant_is_lawful_but_flagged_as_rescued():
    """DET1.10 rescued this class; the census must not hide that."""
    rescued = classify_observation(_observation(
        "p", status="LAWFUL_LIVED_TARGET", answer="Cobalt 1 India",
        expected=["Cobalt-1-India"],
    ))
    assert rescued["census_class"] == CLASS_LAWFUL
    assert rescued["census_subclass"] == SUBCLASS_SEPARATOR
    assert rescued["separator_artifact_rescued"] is True

    exact = classify_observation(_observation(
        "p", status="LAWFUL_LIVED_TARGET", answer="Cobalt-1-India",
        expected=["Cobalt-1-India"],
    ))
    assert exact["census_subclass"] == SUBCLASS_EXACT
    assert exact["separator_artifact_rescued"] is False


def test_comparator_disagreement_fails_closed():
    """An 'unlawful' answer containing an expected value is a defect."""
    with pytest.raises(DETError, match="census/comparator disagreement"):
        classify_observation(_observation(
            "p", status="UNPLANTABLE", answer="The value is Auric-4-Alpha.",
            expected=["Auric-4-Alpha"],
        ))


def test_probe_without_expected_values_fails_closed():
    with pytest.raises(DETError, match="no expected values"):
        classify_observation(_observation(
            "p", status="UNPLANTABLE", answer="anything", expected=[],
        ))


def test_census_uses_the_latest_attempt_and_keeps_the_history(tmp_path):
    root = tmp_path / "shards"
    _write_shard(root, "e2e-1", 1, [_observation(
        "probe_a", status="UNPLANTABLE", answer="Cobalt 1 India",
        expected=["Cobalt-1-India"],
    )])
    _write_shard(root, "e2e-1", 2, [_observation(
        "probe_a", status="LAWFUL_LIVED_TARGET", answer="Cobalt 1 India",
        expected=["Cobalt-1-India"],
    )])

    latest, history = collect_latest_observations(root)
    assert len(latest) == 1
    assert latest[0]["_census_attempt"] == 2
    assert latest[0]["status"] == "LAWFUL_LIVED_TARGET"
    assert history[0]["status_changed_across_attempts"] is True


def test_unknown_unplantable_reason_fails_closed(tmp_path):
    root = tmp_path / "shards"
    row = _observation(
        "probe_a", status="UNPLANTABLE", answer="Birch-2-Beacon.",
        expected=["cobalt-3-comet"],
    )
    row["unplantable_reason"] = "SOMETHING_NEW"
    _write_shard(root, "e2e-1", 1, [row])
    with pytest.raises(DETError, match="unexpected unplantable reason"):
        build_census(root)


def test_census_partitions_and_counts_the_population(tmp_path):
    root = tmp_path / "shards"
    _write_shard(root, "e2e-1", 1, [
        _observation("lawful_a", status="LAWFUL_LIVED_TARGET",
                     answer="Auric-4-Alpha", expected=["Auric-4-Alpha"]),
        _observation("rescued_a", status="LAWFUL_LIVED_TARGET",
                     answer="Cobalt 1 India", expected=["Cobalt-1-India"]),
        _observation("refusal_a", status="UNPLANTABLE",
                     answer="I'm sorry, but I don't know.",
                     expected=["Marble-4-Juliet"], reserve=True),
        _observation("wrong_a", status="UNPLANTABLE",
                     answer="Birch-2-Beacon.", expected=["cobalt-3-comet"]),
    ])
    census = build_census(root)
    counts = census["counts"]
    assert counts["probes_total"] == 4
    assert counts["lawful"] == 2
    assert counts["lawful_exact"] == 1
    assert counts["lawful_separator_artifact_rescued"] == 1
    assert counts["unlawful"] == 2
    assert counts["unlawful_refusal"] == 1
    assert counts["unlawful_wrong_value"] == 1
    assert counts["lawful"] + counts["unlawful"] == counts["probes_total"]
    assert counts["reserves"] == 1
    assert counts["reserves_lawful"] == 0
    # The defect class the successor investigation inherits.
    assert counts["wrong_value_with_correct_single_mount"] == 1


def test_wrong_value_with_multiple_mounts_is_not_counted_as_single_mount(
    tmp_path,
):
    root = tmp_path / "shards"
    _write_shard(root, "e2e-1", 1, [_observation(
        "wrong_a", status="UNPLANTABLE", answer="Birch-2-Beacon.",
        expected=["cobalt-3-comet"], mounted=[1, 2],
    )])
    census = build_census(root)
    assert census["counts"]["unlawful_wrong_value"] == 1
    assert census["counts"]["wrong_value_with_correct_single_mount"] == 0


def test_writing_twice_keeps_exactly_one_receipt(tmp_path):
    """created_utc moves; the receipt must not multiply because of it."""
    root = tmp_path / "shards"
    _write_shard(root, "e2e-1", 1, [_observation(
        "lawful_a", status="LAWFUL_LIVED_TARGET",
        answer="Auric-4-Alpha", expected=["Auric-4-Alpha"],
    )])
    out = tmp_path / "census"
    first, _ = write_census(out, shard_root=root)
    second, _ = write_census(out, shard_root=root)
    assert first == second
    assert len(list(out.glob("lived_serving_census_*.json"))) == 1


def test_changed_evidence_refuses_to_overwrite_a_frozen_receipt(tmp_path):
    root = tmp_path / "shards"
    _write_shard(root, "e2e-1", 1, [_observation(
        "lawful_a", status="LAWFUL_LIVED_TARGET",
        answer="Auric-4-Alpha", expected=["Auric-4-Alpha"],
    )])
    out = tmp_path / "census"
    write_census(out, shard_root=root)

    _write_shard(root, "e2e-2", 1, [_observation(
        "wrong_a", status="UNPLANTABLE", answer="Birch-2-Beacon.",
        expected=["cobalt-3-comet"],
    )])
    with pytest.raises(DETError, match="census evidence changed"):
        write_census(out, shard_root=root)


def test_census_receipt_is_titled_and_scoped_to_receipts_only(tmp_path):
    root = tmp_path / "shards"
    _write_shard(root, "e2e-1", 1, [_observation(
        "lawful_a", status="LAWFUL_LIVED_TARGET",
        answer="Auric-4-Alpha", expected=["Auric-4-Alpha"],
    )])
    path, census = write_census(tmp_path / "census", shard_root=root)
    assert census["title"] == TITLE
    assert census["cause_investigated"] is False
    assert census["status"] == "RECEIPTS_ONLY_CAUSE_NOT_INVESTIGATED"
    assert TITLE in census["markdown"]
    # Content-addressed, and re-derivable from the file.
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["counts"] == census["counts"]
    assert path.stem.startswith("lived_serving_census_")
