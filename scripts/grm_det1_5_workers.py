#!/usr/bin/env python3
"""Concrete bounded GPU workers for the DET1.5 fork-substrate campaign.

Each invocation owns one model process and one immutable shard.  Heavy stages
are deliberately split by ``spec`` so the campaign parent can place every
shard under its own standing GPU lease.  The workers only emit measurements
produced by lived chronological execution and same-process snapshot forks;
they never synthesize completed rows or gate receipts.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]

from scripts.grm_det1_common import (  # noqa: E402
    DETError,
    MECHANISTIC,
    canonical_json_bytes,
    file_record,
    read_json,
    read_jsonl,
    utc_now,
    write_content_addressed,
)
from scripts import grm_det1_5_gpu as campaign  # noqa: E402


SHARD_SCHEMA = "grm.det1_5.worker_shard.v1"
E2E_SPECS: dict[str, dict[str, Any]] = {
    "e2e-cal": {
        "predecessor": None,
        "stop_after_turns": 17,
        "expected_rows": 2,
    },
    "e2e-1": {
        "predecessor": None,
        "stop_after_turns": 10,
        "expected_rows": 2,
    },
    "e2e-2": {
        "predecessor": "e2e-1",
        "stop_after_turns": 17,
        "expected_rows": 2,
    },
    "e2e-3": {
        "predecessor": "e2e-2",
        "stop_after_turns": 25,
        "expected_rows": 6,
    },
    "e2e-4": {
        "predecessor": "e2e-3",
        "stop_after_turns": 31,
        "expected_rows": 4,
    },
}
SUP_SOURCES = tuple(sorted(
    (ROOT / "tests/fixtures/supersession_battery").glob("*.json")
))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DETError(message)


def _normalize_worker(value: str) -> str:
    worker = str(value).strip().replace("-", "_")
    _require(worker in {
        "g0", "g1", "calibration", "eval_mechanistic", "eval_verbal",
    }, f"unknown DET1.5 worker: {value}")
    return worker


def _spec(args: Any) -> str:
    value = getattr(args, "spec", None)
    _require(isinstance(value, str) and bool(value.strip()),
             "DET1.5 bounded worker requires --spec")
    return value.strip().lower().replace("_", "-")


def _one(directory: Path, pattern: str) -> Path:
    values = sorted(Path(directory).glob(pattern))
    _require(len(values) == 1,
             f"expected exactly one {pattern} under {directory}, got {len(values)}")
    return values[0]


def _record_path(record: Mapping[str, Any]) -> Path:
    path = Path(str(record.get("path", "")))
    return path if path.is_absolute() else ROOT / path


def _validated_record(record: Mapping[str, Any], label: str) -> Path:
    _require(isinstance(record, Mapping), f"{label} is not a file record")
    path = _record_path(record)
    _require(path.is_file() and file_record(path) == dict(record),
             f"{label} file record drifted: {record}")
    return path


def _context(args: Any, worker: str) -> dict[str, Any]:
    run_dir = Path(args.run_dir).resolve()
    _require(run_dir == campaign.FROZEN_RUN.resolve(),
             "DET1.5 workers are bound to the frozen registered run")
    attempt_dir = Path(args.attempt_dir).resolve()
    root = campaign.campaign_root(run_dir).resolve()
    _require(attempt_dir.is_relative_to(root),
             f"worker attempt escapes fresh campaign namespace: {attempt_dir}")
    _require(attempt_dir.is_dir(), f"worker attempt directory is absent: {attempt_dir}")
    registration_path = _one(run_dir, "registration_*.json")
    runtime_path = _one(run_dir, "runtime_frame_*.json")
    amendment_path = run_dir / campaign.RACE_AMENDMENT.name
    campaign.validate_race_amendment(run_dir)
    registration = read_json(registration_path)
    runtime = read_json(runtime_path)
    _require(runtime.get("registration") == file_record(registration_path),
             "runtime frame registration binding drifted")
    lease_seconds = int(args.lease_seconds)
    _require(lease_seconds == int(runtime["gpu_lease"]["actual_cap_seconds"]),
             "worker lease differs from frozen runtime frame")
    from scripts.grm_det1_gpu import NATIVE_LIB, _runtime_env, _validate_visibility

    _require(runtime.get("native_library") == file_record(NATIVE_LIB),
             "runtime native-library binding drifted")
    os.environ.update(_runtime_env(runtime))
    visible = _validate_visibility()
    _require(str(visible) == str(runtime["cuda_visible_devices"]),
             "worker visible GPU differs from frozen runtime frame")
    process = campaign._process_instance()
    return {
        "worker": worker,
        "run_dir": run_dir,
        "attempt_dir": attempt_dir,
        "registration_path": registration_path,
        "registration": registration,
        "runtime_path": runtime_path,
        "runtime": runtime,
        "amendment_path": amendment_path,
        "lease_seconds": lease_seconds,
        "process": process,
    }


def _mechanistic_bindings(run_dir: Path) -> dict[str, dict[str, Any]]:
    rows_path = campaign.campaign_root(run_dir) / "eval/mechanistic_rows.jsonl"
    _require(rows_path.is_file(),
             "D-VERB opened before fixed mechanistic evaluation rows exist")
    bindings: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(rows_path):
        row_id = str(row.get("row_id", ""))
        selection = (row.get("signals") or {}).get("ladder_attempt_selection") or {}
        ordinal = selection.get("selected_call_ordinal",
                                selection.get("production_trip"))
        _require(bool(row_id) and isinstance(ordinal, int),
                 f"mechanistic row lacks selected ladder ordinal: {row_id}")
        bindings[row_id] = {
            "mechanistic_completed_unix_ns": int(
                row["mechanistic_completed_unix_ns"]),
            "selected_attempt_ordinal": int(ordinal),
            "mounted_ids": [int(value) for value in row["mounted_ids"]],
        }
    _require(len(bindings) == 24,
             f"D-VERB requires 24 mechanistic bindings, got {len(bindings)}")
    return bindings


def _stamp_rows(raw_path: Path, output_path: Path,
                process_sha256: str) -> list[dict[str, Any]]:
    rows = read_jsonl(raw_path)
    stamped = []
    for value in rows:
        row = dict(value)
        existing = row.get("process_instance_sha256")
        _require(existing in (None, process_sha256),
                 f"row process identity drifted: {row.get('row_id')}")
        row["process_instance_sha256"] = process_sha256
        stamped.append(row)
    campaign._write_jsonl_exclusive_or_verify(output_path, stamped)
    return stamped


def _fixture_measure(
    *,
    ctx: Mapping[str, Any],
    repo: Any,
    e2e: Any,
    original: Any,
    fixture: Mapping[str, Any],
    split: str,
    raw_rows: Path,
    active_detectors: Sequence[str],
    variants: Sequence[str],
    verbal: bool,
    turn_idx: int | None,
    canonical_defer_memory: bool,
    snapshot_root: Path,
) -> tuple[str, dict[str, Any]]:
    flags = ctx["runtime"]["resolved_flags"]
    bindings = (
        _mechanistic_bindings(Path(ctx["run_dir"])) if verbal else None
    )
    return campaign._measure_fixture_inline(
        repo=repo,
        e2e=e2e,
        original=original,
        fixture=fixture,
        split=split,
        rows_path=raw_rows,
        runtime=ctx["runtime"],
        registration=ctx["registration"],
        model=repo.arena.m,
        tokenizer=None,
        topk=int(flags["topk"]),
        ngen=int(flags["ngen"]),
        max_trips=int(flags["max_trips"]),
        turn_idx=turn_idx,
        canonical_defer_memory=canonical_defer_memory,
        active_detectors=active_detectors,
        variants=variants,
        verbal=verbal,
        mechanistic_times=bindings,
        snapshot_root=snapshot_root,
    )


def _session_evidence(
    ctx: Mapping[str, Any], spec: str, session_dir: Path,
    *, resumed_from: Mapping[str, Any] | None,
) -> dict[str, Any]:
    names = (
        "run_config.json", "transcript.jsonl", "instrumentation.jsonl",
        "probe_scorecard.json", "restart.json", "arena_state.json",
    )
    files = {
        str(path.relative_to(session_dir)): file_record(path)
        for path in sorted(session_dir.rglob("*")) if path.is_file()
    }
    _require({"run_config.json", "transcript.jsonl", "instrumentation.jsonl",
              "probe_scorecard.json"} <= set(files),
             f"chronological session evidence is incomplete for {spec}")
    run_config = read_json(session_dir / "run_config.json")
    declared_session = str(run_config.get("session_dir", ""))
    resume_metadata = [
        file_record(path)
        for path in sorted(session_dir.glob("det1_5_resume_*.json"))
    ]
    if resumed_from:
        _require(bool(resume_metadata),
                 f"resumed session {spec} lacks explicit resume metadata")
        config_role = "COPIED_ORIGIN_CONFIG_WITH_EXPLICIT_RESUME_CHAIN"
    else:
        _require(Path(declared_session).resolve() == session_dir.resolve(),
                 f"new session {spec} run_config names another directory")
        _require(not resume_metadata,
                 f"new session {spec} unexpectedly contains resume metadata")
        config_role = "ACTIVE_SESSION_CONFIG"
    value = {
        "schema": "grm.det1_5.chronological_session.v1",
        "status": "COMPLETE_BOUNDARY",
        "created_utc": utc_now(),
        "stage": ctx["worker"],
        "spec": spec,
        "process_instance_sha256": ctx["process"]["process_instance_sha256"],
        "session_dir": str(session_dir),
        "files": files,
        "resumed_from": dict(resumed_from) if resumed_from else None,
        "run_config_role": config_role,
        "run_config_declared_session_dir": declared_session,
        "active_session_dir": str(session_dir),
        "resume_metadata": resume_metadata,
    }
    path = write_content_addressed(
        Path(ctx["attempt_dir"]), "chronological_session", value)
    return file_record(path)


def _validate_session_evidence(
    ctx: Mapping[str, Any],
    predecessor: str,
    shard: Mapping[str, Any],
) -> Path:
    evidence_path = _validated_record(
        shard.get("chronological_session") or {},
        f"{predecessor} chronological session")
    evidence = read_json(evidence_path)
    _require(
        evidence.get("schema") == "grm.det1_5.chronological_session.v1"
        and evidence.get("status") == "COMPLETE_BOUNDARY"
        and evidence.get("stage") == ctx["worker"]
        and evidence.get("spec") == predecessor
        and evidence.get("process_instance_sha256")
        == shard.get("process_instance_sha256"),
        f"{predecessor} chronological session envelope drifted",
    )
    session_dir = Path(str(evidence.get("session_dir", ""))).resolve()
    _require(session_dir.is_dir(),
             f"predecessor session directory is absent: {session_dir}")
    records = evidence.get("files") or {}
    _require(isinstance(records, Mapping) and bool(records),
             f"{predecessor} chronological session has no file inventory")
    observed: set[str] = set()
    for relative, record in records.items():
        path = _validated_record(
            record, f"{predecessor} session file {relative}")
        _require(path.resolve() == (session_dir / str(relative)).resolve()
                 and path.resolve().is_relative_to(session_dir),
                 f"{predecessor} session file escapes its directory: {relative}")
        observed.add(str(relative))
    actual = {
        str(path.relative_to(session_dir))
        for path in session_dir.rglob("*") if path.is_file()
    }
    _require(actual == observed,
             f"{predecessor} session inventory gained or lost files")
    run_config = read_json(session_dir / "run_config.json")
    declared = str(run_config.get("session_dir", ""))
    _require(evidence.get("run_config_declared_session_dir") == declared
             and evidence.get("active_session_dir") == str(session_dir),
             f"{predecessor} session config/active-path projection drifted")
    resumed_from = evidence.get("resumed_from")
    resume_records = list(evidence.get("resume_metadata") or ())
    if resumed_from:
        _require(
            evidence.get("run_config_role")
            == "COPIED_ORIGIN_CONFIG_WITH_EXPLICIT_RESUME_CHAIN"
            and bool(resume_records),
            f"{predecessor} resumed session lacks an explicit origin role",
        )
    else:
        _require(evidence.get("run_config_role") == "ACTIVE_SESSION_CONFIG"
                 and Path(declared).resolve() == session_dir,
                 f"{predecessor} initial session config is not active")
    for index, record in enumerate(resume_records):
        metadata_path = _validated_record(
            record, f"{predecessor} resume metadata {index}")
        _require(metadata_path.resolve().is_relative_to(session_dir),
                 f"{predecessor} resume metadata escapes session")
    return session_dir


def _write_resume_metadata(
    *,
    ctx: Mapping[str, Any],
    spec: str,
    predecessor: str,
    predecessor_path: Path,
    predecessor_value: Mapping[str, Any],
    source_session: Path,
    target_session: Path,
) -> Path:
    source_config = source_session / "run_config.json"
    target_config = target_session / "run_config.json"
    _require(source_config.is_file() and target_config.is_file(),
             f"resume {predecessor}->{spec} lacks copied run_config")
    _require(source_config.read_bytes() == target_config.read_bytes(),
             f"resume {predecessor}->{spec} altered the origin run_config")
    value = {
        "schema": "grm.det1_5.resume_metadata.v1",
        "status": "COPIED_IMMUTABLE_PRODUCTION_BOUNDARY",
        "stage": ctx["worker"],
        "spec": spec,
        "predecessor_spec": predecessor,
        "process_instance_sha256": ctx["process"]["process_instance_sha256"],
        "source_session_dir": str(source_session),
        "active_session_dir": str(target_session),
        "origin_run_config": file_record(source_config),
        "copied_run_config": file_record(target_config),
        "origin_declared_session_dir": str(
            read_json(target_config).get("session_dir", "")),
        "predecessor_shard": file_record(predecessor_path),
        "predecessor_chronological_session": dict(
            predecessor_value.get("chronological_session") or {}),
        "law": (
            "run_config is an immutable origin record; active_session_dir "
            "is declared separately for this production resume"
        ),
    }
    return campaign._write_json_exclusive_or_verify(
        target_session / f"det1_5_resume_{predecessor}_to_{spec}.json", value)


def _predecessor_shard(ctx: Mapping[str, Any], predecessor: str) -> tuple[Path, dict[str, Any]]:
    stage_root = campaign.stage_shard_root(
        Path(ctx["run_dir"]), str(ctx["worker"])) / predecessor
    output = campaign._completed_shard_output(
        stage_root, stage=str(ctx["worker"]), spec=predecessor)
    _require(output is not None,
             f"no parent-recognized completed predecessor {predecessor}")
    path = _validated_record(
        output.get("receipt_file") or {}, f"{predecessor} predecessor receipt")
    return path, read_json(path)


def _run_e2e(ctx: Mapping[str, Any], spec: str, *, verbal: bool) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    _require(spec in E2E_SPECS, f"invalid chronological E2E spec: {spec}")
    config = E2E_SPECS[spec]
    worker = str(ctx["worker"])
    _require((worker == "calibration") == (spec == "e2e-cal"),
             f"{worker} cannot run spec {spec}")
    from scripts import grm_e2e_session as e2e
    from scripts import grm_det1_e2e as det_e2e

    attempt_dir = Path(ctx["attempt_dir"])
    session_dir = attempt_dir / "session"
    resumed_from: Mapping[str, Any] | None = None
    predecessor = config["predecessor"]
    resume = predecessor is not None
    if resume:
        predecessor_path, predecessor_value = _predecessor_shard(ctx, str(predecessor))
        prior_session = _validate_session_evidence(
            ctx, str(predecessor), predecessor_value)
        _require(Path(str(predecessor_value.get("session_dir", ""))).resolve()
                 == prior_session,
                 f"predecessor session path binding drifted: {predecessor}")
        shutil.copytree(prior_session, session_dir, symlinks=True)
        _write_resume_metadata(
            ctx=ctx,
            spec=spec,
            predecessor=str(predecessor),
            predecessor_path=predecessor_path,
            predecessor_value=predecessor_value,
            source_session=prior_session,
            target_session=session_dir,
        )
        resumed_from = file_record(predecessor_path)

    raw_rows = attempt_dir / "raw_rows.jsonl"
    selected_split = "calibration" if worker == "calibration" else "eval"
    selected = {
        int(row["turn"]): row
        for row in ctx["registration"]["fixtures"]
        if row.get("source_family") == "certified_34_turn"
        and row.get("split") == selected_split
    }
    active = () if verbal else MECHANISTIC
    variants = ("served",) if worker == "calibration" else (
        "served", "planted_miss")
    original_probe = e2e.probe_multimount_chat
    original_turn = e2e.run_turn

    def wrapped(repo, user_text: str, *, topk: int, ngen: int,
                defer_memory: bool = False, turn_idx: int | None = None,
                probe_ladder: bool = True, max_trips: int = 1):
        if turn_idx is None or int(turn_idx) not in selected:
            return original_probe(
                repo, user_text, topk=topk, ngen=ngen,
                defer_memory=defer_memory, turn_idx=turn_idx,
                probe_ladder=probe_ladder, max_trips=max_trips)
        fixture = selected[int(turn_idx)]
        _require(str(fixture["question"]) == str(user_text),
                 f"turn {turn_idx} question drifted from registration")
        return _fixture_measure(
            ctx=ctx, repo=repo, e2e=e2e, original=original_probe,
            fixture=fixture, split=selected_split, raw_rows=raw_rows,
            active_detectors=active, variants=variants, verbal=verbal,
            turn_idx=int(turn_idx), canonical_defer_memory=bool(defer_memory),
            snapshot_root=attempt_dir / "snapshots" / str(fixture["fixture_id"]),
        )

    e2e.probe_multimount_chat = wrapped
    det_e2e.install_stop_boundary(e2e, int(config["stop_after_turns"]))
    flags = ctx["runtime"]["resolved_flags"]
    argv = [
        "--mode", "full",
        "--session-dir", str(session_dir),
        "--model-dir", str(ctx["runtime"]["model"]["path"]),
        "--native-lib", str(_record_path(ctx["runtime"]["native_library"])),
        "--ngen", str(flags["ngen"]),
        "--max-trips", str(flags["max_trips"]),
        "--live-turns", str(flags["live_turns"]),
        "--arena-width", str(flags["arena_width"]),
        "--max-live", str(flags["max_live"]),
        "--topk", str(flags["topk"]),
        "--turn-pipeline", str(flags["turn_pipeline"]),
        "--restart-after", "999",
        "--skip-gpu-idle-check", "--probe-ladder", "--sup-resolve",
        "--adm-decisive" if flags["adm_decisive"] else "--no-adm-decisive",
    ]
    if resume:
        argv.append("--resume")
    stopped = False
    try:
        e2e.main(argv)
    except det_e2e._DETStageStop:
        stopped = True
    finally:
        e2e.probe_multimount_chat = original_probe
        e2e.run_turn = original_turn
    _require(stopped, f"E2E spec {spec} did not stop at its registered boundary")
    output_rows = attempt_dir / "rows.jsonl"
    rows = _stamp_rows(
        raw_rows, output_rows, str(ctx["process"]["process_instance_sha256"]))
    _require(len(rows) == int(config["expected_rows"]),
             f"{spec} expected {config['expected_rows']} rows, got {len(rows)}")
    evidence = _session_evidence(
        ctx, spec, session_dir, resumed_from=resumed_from)
    return output_rows, rows, {
        "chronological_session": evidence,
        "session_dir": str(session_dir),
        "resume": bool(resume),
        "stop_after_turns": int(config["stop_after_turns"]),
    }


def _close_model(repo: Any, model: Any, tokenizer: Any) -> None:
    try:
        if repo is not None:
            repo.close()
    finally:
        if model is not None:
            try:
                from core import kv_graft
                kv_graft.clear_injection(model)
            except BaseException:
                pass
        del repo, model, tokenizer
        gc.collect()
        try:
            import tensor_cuda as tc
            if hasattr(tc, "empty_cache"):
                tc.empty_cache()
        except BaseException:
            pass


def _run_sup(ctx: Mapping[str, Any], spec: str, *, verbal: bool,
             variants: Sequence[str] = ("served", "planted_miss"),
             active_detectors: Sequence[str] | None = None) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    _require(spec.startswith("sup-") and spec[4:].isdigit(),
             f"invalid supersession spec: {spec}")
    index = int(spec[4:])
    _require(len(SUP_SOURCES) == 4 and 1 <= index <= 4,
             "supersession battery layout is not the registered four files")
    source = SUP_SOURCES[index - 1]
    fixture_source = read_json(source)
    registered = {
        str(row["probe_id"]): row
        for row in ctx["registration"]["fixtures"]
        if row.get("source_family") == "supersession_battery_on_gpt_oss"
    }
    from scripts.grm_det1_2_gpu import _load_model_repo
    from scripts.grm_det1_3_gpu import _install_lived_nodes

    attempt_dir = Path(ctx["attempt_dir"])
    raw_rows = attempt_dir / "raw_rows.jsonl"
    repo = model = tokenizer = None
    try:
        e2e, model, tokenizer, repo, model_info = _load_model_repo(
            attempt_dir, ctx["runtime"])
        _install_lived_nodes(repo, e2e, fixture_source)
        original = e2e.probe_multimount_chat
        for probe in fixture_source["probes"]:
            fixture = registered[str(probe["probe_id"])]
            _fixture_measure(
                ctx=ctx, repo=repo, e2e=e2e, original=original,
                fixture=fixture, split=(
                    "calibration" if ctx["worker"] == "calibration" else "eval"),
                raw_rows=raw_rows,
                active_detectors=(
                    tuple(active_detectors) if active_detectors is not None
                    else (() if verbal else MECHANISTIC)),
                variants=variants, verbal=verbal, turn_idx=None,
                canonical_defer_memory=True,
                snapshot_root=(attempt_dir / "snapshots"
                               / str(fixture["fixture_id"])),
            )
    finally:
        _close_model(repo, model, tokenizer)
    output_rows = attempt_dir / "rows.jsonl"
    rows = _stamp_rows(
        raw_rows, output_rows, str(ctx["process"]["process_instance_sha256"]))
    expected = 2 * len(fixture_source["probes"]) if len(variants) == 2 else len(
        fixture_source["probes"])
    _require(len(rows) == expected,
             f"{spec} expected {expected} rows, got {len(rows)}")
    evidence_value = {
        "schema": "grm.det1_5.chronological_session.v1",
        "status": "COMPLETE_LIVED_FIXTURE",
        "created_utc": utc_now(),
        "stage": ctx["worker"],
        "spec": spec,
        "process_instance_sha256": ctx["process"]["process_instance_sha256"],
        "fixture_source": file_record(source),
        "node_count": len(fixture_source["nodes"]),
        "probe_count": len(fixture_source["probes"]),
        "protocol": "CHRONOLOGICAL_ARENA_FEED_THEN_INLINE_FORK",
    }
    evidence_path = write_content_addressed(
        attempt_dir, "chronological_session", evidence_value)
    return output_rows, rows, {
        "chronological_session": file_record(evidence_path),
        "fixture_source": file_record(source),
        "model_info": model_info,
    }


def _write_g1_bytes(attempt_dir: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    projection = {
        "schema": "grm.det1_5.g1_served_projection.v1",
        "fixture_id": row["fixture_id"],
        "variant": row["variant"],
        "question": row["question"],
        "answer": row["answer"],
        "answer_correct": row["answer_correct"],
        "mounted_ids": row["mounted_ids"],
        "expected_values": row["expected_values"],
    }
    transcript_path = campaign._write_bytes_exclusive_or_verify(
        attempt_dir / "transcript.jsonl", campaign._jsonl_bytes([projection]))
    scorecard_path = campaign._write_json_exclusive_or_verify(
        attempt_dir / "scorecard.json", {
            "schema": "grm.det1_5.g1_scorecard.v1",
            "passed": 1 if row["answer_correct"] else 0,
            "total": 1,
            "all_passed": bool(row["answer_correct"]),
        })
    projection_path = campaign._write_json_exclusive_or_verify(
        attempt_dir / "served_projection.json", projection)
    return {
        "transcript": file_record(transcript_path),
        "scorecard": file_record(scorecard_path),
        "served_projection": file_record(projection_path),
    }


def _run_g1(ctx: Mapping[str, Any], spec: str) -> dict[str, Any]:
    _require(spec in ("control", "all-hooks"), f"invalid G1 arm: {spec}")
    active = () if spec == "control" else MECHANISTIC
    # Harbor is the registered planted-miss incident fixture and supplies a
    # compact lived same-process snapshot test for hook noninterference.
    rows_path, rows, extra = _run_sup(
        ctx, "sup-1", verbal=False, variants=("served",),
        active_detectors=active)
    selected = [row for row in rows
                if row.get("fixture_id") == "sup_harbor_restatement"]
    _require(len(selected) == 1,
             "G1 arm did not produce exactly one Harbor served projection")
    _require(selected[0].get("answer_correct") is True,
             f"G1 {spec} Harbor served control is incorrect")
    campaign.validate_served_control(
        selected[0], f"G1 {spec} Harbor served control")
    byte_records = _write_g1_bytes(Path(ctx["attempt_dir"]), selected[0])
    return {
        "arm": spec,
        "active_hooks": list(active),
        "d_verb_isolated": True,
        "byte_records": byte_records,
        "rows": file_record(rows_path),
        "row_ids": [str(selected[0]["row_id"])],
        "row_count": 1,
        **extra,
    }


def _write_shard(ctx: Mapping[str, Any], spec: str,
                 payload: Mapping[str, Any]) -> dict[str, Any]:
    value = {
        "schema": SHARD_SCHEMA,
        "status": "COMPLETE",
        "created_utc": utc_now(),
        "stage": ctx["worker"],
        "spec": spec,
        "process_instance_sha256": ctx["process"]["process_instance_sha256"],
        "process": dict(ctx["process"]),
        "registration": file_record(ctx["registration_path"]),
        "runtime_frame": file_record(ctx["runtime_path"]),
        "race_authorization_amendment": file_record(ctx["amendment_path"]),
        "lease_seconds": int(ctx["lease_seconds"]),
        "fork_from_lived_snapshot": True,
        "same_process_full_index_required": True,
        "historical_rows_reused": False,
        **dict(payload),
    }
    path = write_content_addressed(Path(ctx["attempt_dir"]), "worker_shard", value)
    output = dict(value)
    output["receipt_file"] = file_record(path)
    return output


def run_worker(worker: str, args: Any) -> dict[str, Any]:
    """Execute one bounded DET1.5 GPU shard and return its receipt envelope."""
    stage = _normalize_worker(worker)
    spec = _spec(args)
    ctx = _context(args, stage)
    started = time.monotonic()
    if stage == "g1":
        payload = _run_g1(ctx, spec)
    else:
        verbal = stage == "eval_verbal"
        if spec in E2E_SPECS:
            rows_path, rows, extra = _run_e2e(ctx, spec, verbal=verbal)
        elif spec.startswith("sup-"):
            _require(stage in ("g0", "eval_mechanistic", "eval_verbal"),
                     f"{stage} cannot run supersession spec {spec}")
            rows_path, rows, extra = _run_sup(ctx, spec, verbal=verbal)
        else:
            raise DETError(f"unsupported {stage} shard spec: {spec}")
        payload = {
            "rows": file_record(rows_path),
            "row_ids": [str(row["row_id"]) for row in rows],
            "row_count": len(rows),
            **extra,
        }
    payload["elapsed_seconds"] = time.monotonic() - started
    return _write_shard(ctx, spec, payload)
