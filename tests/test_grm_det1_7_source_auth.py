from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.grm_det1_7_source_auth import (
    SCHEMA,
    STATUS,
    SourceAuthorizationError,
    author_precollection_source_authorization,
    canonical_json_bytes,
    file_record,
    validate_precollection_source_authorization,
)


CREATED_UTC = "2026-09-01T01:02:03Z"


def _write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def _fixture(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "repo"
    order7 = root / "orders/GRM_DET1_7_PLANT_REALIGN.md"
    order6 = root / "orders/GRM_DET1_6_PLANT_ON_FORK.md"
    registration = root / "artifacts/run/registration.json"
    runtime = root / "artifacts/run/runtime.json"
    det14 = root / "artifacts/run/det1_4.json"
    det15 = root / "artifacts/run/det1_5.json"
    det16 = root / "artifacts/run/det1_6.json"
    authorization = root / "artifacts/run/det1_7_source_authorization.json"
    source_a = root / "src/a.py"
    source_b = root / "src/b.py"
    source_c = root / "src/c.py"

    _write(order7, "ORDER DET1.7\n")
    _write(order6, "ORDER DET1.6\n")
    _write(registration, json.dumps({"schema": "grm.det1.registration.v1"}))
    registration_record = file_record(registration, repo_root=root)
    _write_json(runtime, {
        "schema": "grm.det1.runtime_frame.v1",
        "registration": registration_record,
    })
    runtime_record = file_record(runtime, repo_root=root)

    _write(source_a, "a4\n")
    _write(source_b, "b5\n")
    a4 = file_record(source_a, repo_root=root)
    b5 = file_record(source_b, repo_root=root)
    _write_json(det14, {
        "schema": "grm.det1_4.source_amendment.v1",
        "status": "AUTHORIZED_FORK_FROM_SNAPSHOT_AMENDMENT",
        "registration": registration_record,
        "runtime_frame": runtime_record,
        "allowed_source_changes": {
            "src/a.py": {
                "before": {"bytes": 1, "sha256": "0" * 64},
                "after": {"bytes": a4["bytes"], "sha256": a4["sha256"]},
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
        "source_inventory": {"src/b.py": b5},
    })

    _write(source_a, "a6\n")
    a6 = file_record(source_a, repo_root=root)
    _write_json(det16, {
        "schema": "grm.det1_6.fork_hydration_delta_amendment.v1",
        "status": "AUTHORIZED_FORK_HYDRATION_DELTA_SOURCE_REBINDING",
        "order": file_record(order6, repo_root=root),
        "registration": registration_record,
        "runtime_frame": runtime_record,
        "det1_4_source_amendment": file_record(det14, repo_root=root),
        "det1_5_race_amendment": file_record(det15, repo_root=root),
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
    _write(source_c, "new det1.7 source\n")
    return {
        "root": root,
        "order": order7,
        "registration": registration,
        "runtime": runtime,
        "det16": det16,
        "authorization": authorization,
        "source_a": source_a,
        "source_b": source_b,
        "source_c": source_c,
    }


def _kwargs(paths: dict[str, Path]) -> dict:
    return {
        "repo_root": paths["root"],
        "order_path": paths["order"],
        "registration_path": paths["registration"],
        "runtime_frame_path": paths["runtime"],
        "det1_6_amendment_path": paths["det16"],
    }


def test_authorization_is_exclusive_canonical_and_collection_only(tmp_path: Path):
    paths = _fixture(tmp_path)
    a6 = file_record(paths["source_a"], repo_root=paths["root"])
    _write(paths["source_a"], "a7\n")
    a7 = file_record(paths["source_a"], repo_root=paths["root"])
    changed = {"src/a.py": "derive_targets_from_lived_admission"}
    added = {"src/c.py": "pure_precollection_contract"}

    result = author_precollection_source_authorization(
        paths["authorization"],
        **_kwargs(paths),
        changed_sources=changed,
        added_sources=added,
        created_utc=CREATED_UTC,
    )
    value = json.loads(paths["authorization"].read_text(encoding="utf-8"))

    assert value["schema"] == SCHEMA
    assert value["status"] == STATUS
    assert paths["authorization"].read_bytes() == canonical_json_bytes(value)
    assert value["invariants"]["collection_authorized"] is True
    assert value["invariants"]["race_resume_authorized"] is False
    assert value["invariants"]["amendment_is_evidence"] is False
    assert result["det1_6_source_rebindings"]["src/a.py"] == {
        "before": a6,
        "after": a7,
    }
    assert result["det1_4_source_rebindings"]["src/a.py"]["after"] == a7
    assert result["transitive_source_rebindings"]["src/a.py"]["after"] == a7
    assert result["det1_5_source_rebindings"] == {}

    with pytest.raises(FileExistsError):
        author_precollection_source_authorization(
            paths["authorization"],
            **_kwargs(paths),
            changed_sources=changed,
            added_sources=added,
            created_utc=CREATED_UTC,
        )


def test_unchanged_by_det16_source_uses_det15_predecessor(tmp_path: Path):
    paths = _fixture(tmp_path)
    b5 = file_record(paths["source_b"], repo_root=paths["root"])
    _write(paths["source_b"], "b7\n")
    b7 = file_record(paths["source_b"], repo_root=paths["root"])

    result = author_precollection_source_authorization(
        paths["authorization"],
        **_kwargs(paths),
        changed_sources={"src/b.py": "bind_collection_worker"},
        added_sources={"src/c.py": "new_registry_module"},
        created_utc=CREATED_UTC,
    )

    assert result["det1_6_source_rebindings"]["src/b.py"] == {
        "before": b5,
        "after": b7,
    }
    assert result["det1_5_source_rebindings"]["src/b.py"] == {
        "before": b5,
        "after": b7,
    }
    assert result["transitive_source_rebindings"]["src/b.py"]["before"] == b5


def test_unallowlisted_frozen_source_drift_is_rejected(tmp_path: Path):
    paths = _fixture(tmp_path)
    _write(paths["source_b"], "unrelated drift\n")

    with pytest.raises(SourceAuthorizationError, match="unrelated source drift"):
        author_precollection_source_authorization(
            paths["authorization"],
            **_kwargs(paths),
            changed_sources={},
            added_sources={"src/c.py": "new_registry_module"},
            created_utc=CREATED_UTC,
        )
    assert not paths["authorization"].exists()


def test_det16_must_bind_its_immediate_frozen_predecessor(tmp_path: Path):
    paths = _fixture(tmp_path)
    det16 = json.loads(paths["det16"].read_text(encoding="utf-8"))
    det16["allowed_source_changes"]["src/a.py"]["before"]["sha256"] = "f" * 64
    _write_json(paths["det16"], det16)

    with pytest.raises(SourceAuthorizationError, match="immediate predecessor drift"):
        author_precollection_source_authorization(
            paths["authorization"],
            **_kwargs(paths),
            changed_sources={},
            added_sources={"src/c.py": "new_registry_module"},
            created_utc=CREATED_UTC,
        )


def test_known_source_cannot_be_declared_added(tmp_path: Path):
    paths = _fixture(tmp_path)

    with pytest.raises(SourceAuthorizationError, match="already has a frozen predecessor"):
        author_precollection_source_authorization(
            paths["authorization"],
            **_kwargs(paths),
            changed_sources={},
            added_sources={"src/a.py": "misclassified_as_new"},
            created_utc=CREATED_UTC,
        )


def test_validation_rejects_authorization_flag_tamper(tmp_path: Path):
    paths = _fixture(tmp_path)
    changed = {}
    added = {"src/c.py": "new_registry_module"}
    result = author_precollection_source_authorization(
        paths["authorization"],
        **_kwargs(paths),
        changed_sources=changed,
        added_sources=added,
        created_utc=CREATED_UTC,
    )
    # DET1.4's validator gets a direct a4 -> a6 map even though DET1.7 did not
    # touch the source.  It never has to infer a transitive hop through DET1.6.
    assert result["det1_4_source_rebindings"]["src/a.py"]["after"] == file_record(
        paths["source_a"], repo_root=paths["root"]
    )
    value = json.loads(paths["authorization"].read_text(encoding="utf-8"))
    value["invariants"]["race_resume_authorized"] = True
    _write_json(paths["authorization"], value)

    with pytest.raises(SourceAuthorizationError, match="content drift"):
        validate_precollection_source_authorization(
            paths["authorization"],
            **_kwargs(paths),
            changed_sources=changed,
            added_sources=added,
        )
