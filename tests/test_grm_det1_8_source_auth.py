from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import grm_det1_7_source_auth as det17
from scripts.grm_det1_8_source_auth import (
    SCHEMA,
    STATUS,
    SourceAuthorizationError,
    author_precollection_source_authorization,
    validate_precollection_source_authorization,
)


CREATED_7_UTC = "2026-09-01T01:02:03Z"
CREATED_8_UTC = "2026-09-01T02:03:04Z"
DET1_8_CHANGES = {
    "src/a.py": "complete_mounted_member_snapshot_payload",
    "src/b.py": "collect_complete_declared_synthesis_snapshots",
}
DET1_8_ADDITIONS = {
    "src/eight.py": "append_only_det1_8_source_authorization",
}


def _write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(det17.canonical_json_bytes(value))


def _fixture(tmp_path: Path) -> dict[str, object]:
    root = tmp_path / "repo"
    order6 = root / "orders/GRM_DET1_6_PLANT_ON_FORK.md"
    order7 = root / "orders/GRM_DET1_7_PLANT_REALIGN.md"
    order8 = root / "orders/GRM_DET1_8_SYNTH_TURN_SNAPSHOTS.md"
    registration = root / "artifacts/run/registration.json"
    runtime = root / "artifacts/run/runtime.json"
    det14 = root / "artifacts/run/det1_4.json"
    det15 = root / "artifacts/run/det1_5.json"
    det16 = root / "artifacts/run/det1_6.json"
    det17_authorization = root / "artifacts/run/det1_7_source_r2.json"
    det18_authorization = root / "artifacts/run/det1_8_source.json"
    source_a = root / "src/a.py"
    source_b = root / "src/b.py"
    source_stable = root / "src/stable.py"
    source_seven = root / "src/seven.py"
    source_eight = root / "src/eight.py"

    _write(order6, "ORDER DET1.6\n")
    _write(order7, "ORDER DET1.7\n")
    _write(order8, "ORDER DET1.8\n")
    _write(registration, json.dumps({"schema": "grm.det1.registration.v1"}))
    registration_record = det17.file_record(registration, repo_root=root)
    _write_json(runtime, {
        "schema": "grm.det1.runtime_frame.v1",
        "registration": registration_record,
    })
    runtime_record = det17.file_record(runtime, repo_root=root)

    _write(source_a, "a4\n")
    _write(source_b, "b5\n")
    _write(source_stable, "stable5\n")
    a4 = det17.file_record(source_a, repo_root=root)
    b5 = det17.file_record(source_b, repo_root=root)
    stable5 = det17.file_record(source_stable, repo_root=root)
    _write_json(det14, {
        "schema": "grm.det1_4.source_amendment.v1",
        "status": "AUTHORIZED_FORK_FROM_SNAPSHOT_AMENDMENT",
        "registration": registration_record,
        "runtime_frame": runtime_record,
        "allowed_source_changes": {
            "src/a.py": {
                "before": {"bytes": 1, "sha256": "0" * 64},
                "after": a4,
            },
        },
        "added_sources": {},
        "test_sources": {},
    })
    _write_json(det15, {
        "schema": "grm.det1_5.race_authorization_amendment.v1",
        "status": "AUTHORIZED_RACE_ON_FORKED_SUBSTRATE",
        "registration": registration_record,
        "runtime_frame": runtime_record,
        "source_inventory": {
            "src/b.py": b5,
            "src/stable.py": stable5,
        },
    })

    _write(source_a, "a6\n")
    a6 = det17.file_record(source_a, repo_root=root)
    _write_json(det16, {
        "schema": "grm.det1_6.fork_hydration_delta_amendment.v1",
        "status": "AUTHORIZED_FORK_HYDRATION_DELTA_SOURCE_REBINDING",
        "order": det17.file_record(order6, repo_root=root),
        "registration": registration_record,
        "runtime_frame": runtime_record,
        "det1_4_source_amendment": det17.file_record(det14, repo_root=root),
        "det1_5_race_amendment": det17.file_record(det15, repo_root=root),
        "allowed_source_changes": {
            "src/a.py": {
                "purpose": "det1_6_change",
                "before": a4,
                "after": a6,
            },
        },
        "invariants": {
            "race_resume_authorized": True,
            "amendment_is_evidence": False,
        },
    })

    _write(source_b, "b7\n")
    _write(source_seven, "new seven\n")
    b7 = det17.file_record(source_b, repo_root=root)
    det17.author_precollection_source_authorization(
        det17_authorization,
        repo_root=root,
        order_path=order7,
        registration_path=registration,
        runtime_frame_path=runtime,
        det1_6_amendment_path=det16,
        changed_sources={"src/b.py": "det1_7_change"},
        added_sources={"src/seven.py": "det1_7_addition"},
        created_utc=CREATED_7_UTC,
    )
    det17_record = det17.file_record(det17_authorization, repo_root=root)

    # DET1.8 changes happen after the immutable DET1.7 authorization exists.
    # In particular, a.py was unchanged by DET1.7, whereas b.py was changed.
    _write(source_a, "a8\n")
    _write(source_b, "b8\n")
    _write(source_eight, "new eight\n")
    a8 = det17.file_record(source_a, repo_root=root)
    b8 = det17.file_record(source_b, repo_root=root)

    return {
        "root": root,
        "order8": order8,
        "registration": registration,
        "runtime": runtime,
        "det16": det16,
        "det17_authorization": det17_authorization,
        "det17_record": det17_record,
        "det18_authorization": det18_authorization,
        "source_a": source_a,
        "source_b": source_b,
        "source_stable": source_stable,
        "source_seven": source_seven,
        "a4": a4,
        "a6": a6,
        "a8": a8,
        "b5": b5,
        "b7": b7,
        "b8": b8,
    }


def _kwargs(paths: dict[str, object]) -> dict[str, object]:
    return {
        "repo_root": paths["root"],
        "order_path": paths["order8"],
        "registration_path": paths["registration"],
        "runtime_frame_path": paths["runtime"],
        "det1_6_amendment_path": paths["det16"],
        "predecessor_authorization_path": paths["det17_authorization"],
        "expected_predecessor_record": paths["det17_record"],
        "changed_sources": DET1_8_CHANGES,
        "added_sources": DET1_8_ADDITIONS,
    }


def test_successor_is_exclusive_canonical_collection_only_and_direct(tmp_path: Path):
    paths = _fixture(tmp_path)
    result = author_precollection_source_authorization(
        paths["det18_authorization"],
        **_kwargs(paths),
        created_utc=CREATED_8_UTC,
    )
    authorization_path = Path(paths["det18_authorization"])
    value = json.loads(authorization_path.read_text(encoding="utf-8"))

    assert value["schema"] == SCHEMA
    assert value["status"] == STATUS
    assert authorization_path.read_bytes() == det17.canonical_json_bytes(value)
    assert value["order"] == det17.file_record(
        Path(paths["order8"]), repo_root=Path(paths["root"])
    )
    assert value["predecessor_authorization"] == paths["det17_record"]
    assert value["source_lineage"][
        "det1_7_precollection_source_authorization"
    ] == paths["det17_record"]
    assert value["invariants"]["collection_authorized"] is True
    assert value["invariants"]["evaluation_authorized"] is False
    assert value["invariants"]["race_resume_authorized"] is False
    assert value["invariants"]["amendment_is_evidence"] is False

    assert value["allowed_source_changes"]["src/a.py"]["before"] == paths["a6"]
    assert value["allowed_source_changes"]["src/b.py"]["before"] == paths["b7"]
    assert result["det1_7_source_rebindings"] == {
        "src/a.py": {"before": paths["a6"], "after": paths["a8"]},
        "src/b.py": {"before": paths["b7"], "after": paths["b8"]},
    }
    assert result["det1_6_source_rebindings"] == {
        "src/a.py": {"before": paths["a6"], "after": paths["a8"]},
        "src/b.py": {"before": paths["b5"], "after": paths["b8"]},
    }
    assert result["det1_4_source_rebindings"] == {
        "src/a.py": {"before": paths["a4"], "after": paths["a8"]},
    }
    assert result["det1_5_source_rebindings"] == {
        "src/b.py": {"before": paths["b5"], "after": paths["b8"]},
    }
    assert result["source_count"] == 3
    assert result["collection_authorized"] is True
    assert result["race_resume_authorized"] is False
    assert result["amendment_is_evidence"] is False
    assert value["source_counts"] == {
        "added": 1,
        "changed": 2,
        "det1_4_direct_rebindings": 1,
        "det1_5_direct_rebindings": 1,
        "det1_6_direct_rebindings": 2,
        "det1_7_direct_rebindings": 2,
        "frozen_predecessors_checked": 4,
    }

    with pytest.raises(FileExistsError):
        author_precollection_source_authorization(
            authorization_path,
            **_kwargs(paths),
            created_utc=CREATED_8_UTC,
        )


def test_unallowlisted_effective_predecessor_drift_is_rejected(tmp_path: Path):
    paths = _fixture(tmp_path)
    _write(Path(paths["source_stable"]), "unrelated drift\n")

    with pytest.raises(SourceAuthorizationError, match="unrelated source drift"):
        author_precollection_source_authorization(
            paths["det18_authorization"],
            **_kwargs(paths),
            created_utc=CREATED_8_UTC,
        )
    assert not Path(paths["det18_authorization"]).exists()


def test_det1_7_added_source_cannot_be_redeclared_added(tmp_path: Path):
    paths = _fixture(tmp_path)
    kwargs = _kwargs(paths)
    kwargs["added_sources"] = {
        "src/seven.py": "misclassified_existing_predecessor",
        **DET1_8_ADDITIONS,
    }

    with pytest.raises(
        SourceAuthorizationError,
        match="already has a DET1.7 predecessor",
    ):
        author_precollection_source_authorization(
            paths["det18_authorization"],
            **kwargs,
            created_utc=CREATED_8_UTC,
        )


def test_predecessor_file_record_tamper_is_rejected(tmp_path: Path):
    paths = _fixture(tmp_path)
    predecessor_path = Path(paths["det17_authorization"])
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    predecessor["invariants"]["race_resume_authorized"] = True
    _write_json(predecessor_path, predecessor)

    with pytest.raises(SourceAuthorizationError, match="file record drift"):
        author_precollection_source_authorization(
            paths["det18_authorization"],
            **_kwargs(paths),
            created_utc=CREATED_8_UTC,
        )


def test_predecessor_content_tamper_is_rejected_even_if_reanchored(tmp_path: Path):
    paths = _fixture(tmp_path)
    predecessor_path = Path(paths["det17_authorization"])
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    predecessor["invariants"]["race_resume_authorized"] = True
    _write_json(predecessor_path, predecessor)
    kwargs = _kwargs(paths)
    kwargs["expected_predecessor_record"] = det17.file_record(
        predecessor_path, repo_root=Path(paths["root"])
    )

    with pytest.raises(SourceAuthorizationError, match="content drift"):
        author_precollection_source_authorization(
            paths["det18_authorization"],
            **kwargs,
            created_utc=CREATED_8_UTC,
        )


def test_validation_rejects_successor_authority_tamper(tmp_path: Path):
    paths = _fixture(tmp_path)
    authorization_path = Path(paths["det18_authorization"])
    author_precollection_source_authorization(
        authorization_path,
        **_kwargs(paths),
        created_utc=CREATED_8_UTC,
    )
    value = json.loads(authorization_path.read_text(encoding="utf-8"))
    value["invariants"]["race_resume_authorized"] = True
    _write_json(authorization_path, value)

    with pytest.raises(SourceAuthorizationError, match="content drift"):
        validate_precollection_source_authorization(
            authorization_path,
            **_kwargs(paths),
        )
