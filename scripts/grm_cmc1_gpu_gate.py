#!/usr/bin/env python3
"""Lead-side live per-fixture CMC-G0 recheck under disciplined GPU leases.

ORDER GRM-CMC1.1 authorizes mechanism arms independently for every fixture
that reproduces.  A healed fixture is recorded and excluded; it does not stop
another reproducing fixture.  Each selected fixture receives its own
flock-protected, single-GPU lease capped below 590 seconds, with a 30-second
gap between fixtures.  This gate runner itself does not launch T1-T4.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "artifacts" / "grm_cmc1"
LOCK_PATH = Path("/tmp/forge-gpu.lock")
MAX_LEASE_SECONDS = 590
DEFAULT_LEASE_SECONDS = 580
GAP_SECONDS = 30


class GateError(RuntimeError):
    pass


def _finite(value: Any, where: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise GateError(f"non-finite receipt value at {where}: {value!r}")
    if isinstance(value, dict):
        for key, item in value.items():
            _finite(item, f"{where}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _finite(item, f"{where}[{index}]")


def _json_bytes(value: Any) -> bytes:
    _finite(value)
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _json_bytes(value)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as exc:
        raise GateError(f"append-only target already exists: {path}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_file_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise GateError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def gpu_lease(seconds: int, wait_seconds: int) -> Iterator[None]:
    if seconds <= 0 or seconds > MAX_LEASE_SECONDS:
        raise GateError(
            f"lease must be in 1..{MAX_LEASE_SECONDS}s, got {seconds}")
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    handle = LOCK_PATH.open("a+")
    started_wait = time.monotonic()
    last_notice = -30
    while True:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            waited = int(time.monotonic() - started_wait)
            if waited >= int(wait_seconds):
                handle.close()
                raise GateError(
                    f"GPU lock unavailable after {wait_seconds}s: {LOCK_PATH}")
            if waited - last_notice >= 30:
                print(f"waiting_for_gpu_lock_s={waited}", flush=True)
                last_notice = waited
            time.sleep(5)

    old_handler = signal.getsignal(signal.SIGALRM)

    def _expired(_signum, _frame):
        raise TimeoutError(f"GPU lease exceeded {seconds}s")

    signal.signal(signal.SIGALRM, _expired)
    signal.alarm(int(seconds))
    lease_started = time.monotonic()
    try:
        print(f"gpu_lease_acquired={LOCK_PATH} cap_s={seconds}", flush=True)
        yield
    finally:
        elapsed = time.monotonic() - lease_started
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
        print(f"gpu_lease_released_s={elapsed:.3f}", flush=True)


def assess_supersession(receipt_path: Path) -> dict[str, Any]:
    rows = [
        json.loads(line) for line in receipt_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    fresh = [
        row for row in rows
        if row.get("record_type") == "supersession_probe_receipt"
        and row.get("scenario") == "fresh_fact_controls"
    ]
    by_id = {row.get("probe_id"): row for row in fresh}
    praxis = by_id.get("praxis_fresh", {})
    correct = sum(row.get("classification") == "correct" for row in fresh)
    wrong = (
        len(fresh) == 2
        and correct == 1
        and praxis.get("classification") == "wrong-fact"
        and "raven-9-ivory" in str(praxis.get("answer_text", "")).casefold()
    )
    return {
        "fixture": "supersession_fresh_control",
        "fresh_control_correct": correct,
        "fresh_control_total": len(fresh),
        "praxis_answer": praxis.get("answer_text"),
        "wrong_read_reproduced": bool(wrong),
        "receipt": str(receipt_path),
        "receipt_sha256": _sha256(receipt_path),
    }


def assess_diag(scorecard_path: Path) -> dict[str, Any]:
    scorecard = json.loads(scorecard_path.read_text(encoding="utf-8"))
    rows = [row for row in scorecard.get("probes", ()) if row.get("turn") == 5]
    if len(rows) != 1:
        raise GateError(f"expected one DIAG turn-5 row, found {len(rows)}")
    row = rows[0]
    wrong = (
        row.get("pass") is False
        and "vortex-3-sierra" in str(row.get("answer", "")).casefold()
    )
    healed = (
        row.get("pass") is True
        and "auric-4-alpha" in str(row.get("answer", "")).casefold()
    )
    return {
        "fixture": "diag_turn5_orion_cypher",
        "turn5_answer": row.get("answer"),
        "turn5_pass": row.get("pass"),
        "turn5_mount_plan": row.get("mount_plan"),
        "turn5_mounted_ids": row.get("mounted_ids"),
        "wrong_read_reproduced": bool(wrong),
        "healed": bool(healed),
        "scorecard": str(scorecard_path),
        "scorecard_sha256": _sha256(scorecard_path),
    }


def run_supersession(run_dir: Path) -> dict[str, Any]:
    module = _load_file_module(
        "grm_cmc1_sup_harness",
        ROOT / "tests" / "test_grm_supersession_battery.py",
    )
    loaded = [
        item for item in module.load_battery_fixtures()
        if item[2]["scenario"] == "fresh_fact_controls"
    ]
    if len(loaded) != 1:
        raise GateError(f"expected one fresh-control fixture, found {len(loaded)}")
    receipt = run_dir / "supersession_fresh_control.jsonl"
    args = module.parse_args([
        "--run-gpu", "--dialect", "mla", "--receipt-jsonl", str(receipt),
    ])
    sink = module._ReceiptSink(args.receipt_jsonl)
    try:
        module.run_gpu_battery(args, loaded, sink)
    finally:
        sink.close()
    return assess_supersession(receipt)


def run_diag_prefix(run_dir: Path) -> dict[str, Any]:
    # Imports tensor_cuda/transformers only after the GPU lock is held.
    module = _load_file_module(
        "grm_cmc1_e2e_driver", ROOT / "scripts" / "grm_e2e_session.py")
    ladder = _load_file_module(
        "grm_cmc1_probe_ladder", ROOT / "scripts" / "grm_probe_ladder.py")

    # Current production default means no escape env and no explicit CLI arm.
    os.environ.pop("GRM_PROBE_LADDER", None)
    os.environ["GRM_GQA_CUDA_ROUTE"] = "1"
    os.environ["GRM_GRAFT_STORAGE_BITS"] = "8"
    args = module.parse_args([
        "--mode", "full",
        "--turn-pipeline", "single",
        "--session-dir", str(run_dir / "diag_current_default"),
        "--skip-gpu-idle-check",
    ])
    args.ngen = 32
    args.live_turns = 2
    args.restart_after = len(module.build_full_script()) // 2
    if ladder.probe_ladder_enabled(args) is not True:
        raise GateError("current production probe ladder did not resolve ON")

    session_dir = args.session_dir.resolve()
    if session_dir.exists():
        raise GateError(f"append-only DIAG session already exists: {session_dir}")
    session_dir.mkdir(parents=True)
    paths = module.stage_paths(session_dir)
    module.write_json(paths["config"], {
        "schema": "grm.cmc1.diag_prefix.config.v1",
        "mode": "full_prefix_through_turn5",
        "turn_pipeline": "single",
        "probe_ladder": True,
        "live_turns": 2,
        "topk": int(args.topk),
        "arena_width": int(args.arena_width),
        "ngen": int(args.ngen),
        "env": {
            "GRM_GQA_CUDA_ROUTE": "1",
            "GRM_GRAFT_STORAGE_BITS": "8",
            "GRM_PROBE_LADDER": "ABSENT_DEFAULT_ON",
        },
        "script": module.build_full_script()[:6],
    })

    transcript: list[dict[str, Any]] = []
    turn_records: dict[int, dict[str, Any]] = {}
    probe_rows: list[dict[str, Any]] = []
    model = tokenizer = repo = None
    try:
        model, tokenizer, repo, _model_info = module.load_model_and_repo(
            args, session_dir)
        for turn_idx, event in enumerate(module.build_full_script()[:6]):
            module.run_turn(
                repo, event, turn_idx,
                paths=paths,
                transcript=transcript,
                turn_records=turn_records,
                probe_rows=probe_rows,
                args=args,
                resumed=False,
            )
            module.write_scorecard(paths, probe_rows)
    finally:
        if repo is not None:
            try:
                repo.flush_now()
                repo.close()
            except Exception as exc:
                print(f"diag_close_warning={exc!r}", flush=True)
        del repo, model, tokenizer
        gc.collect()
        tc_mod = sys.modules.get("tensor_cuda")
        if tc_mod is not None and hasattr(tc_mod, "empty_cache"):
            tc_mod.empty_cache()

    transcript_path = paths["transcript"]
    result = assess_diag(paths["scorecard"])
    result.update({
        "transcript": str(transcript_path),
        "transcript_sha256": _sha256(transcript_path),
        "production_default_probe_ladder": True,
        "turns_completed": len(transcript),
    })
    return result


def _new_run_dir(base: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = base.resolve() / f"live_g0_{stamp}_{os.getpid()}"
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise GateError(f"append-only run directory exists: {path}") from exc
    return path


def self_test() -> dict[str, Any]:
    sup_path = Path("/tmp/grm_cmc1_selftest_sup.jsonl")
    sup_path.write_text(
        "\n".join([
            json.dumps({
                "record_type": "supersession_probe_receipt",
                "scenario": "fresh_fact_controls", "probe_id": "praxis_fresh",
                "classification": "wrong-fact",
                "answer_text": "The current Praxis dock value is Raven-9-Ivory.",
            }),
            json.dumps({
                "record_type": "supersession_probe_receipt",
                "scenario": "fresh_fact_controls", "probe_id": "solace_fresh",
                "classification": "correct", "answer_text": "Raven-9-Ivory",
            }),
        ]) + "\n",
        encoding="utf-8",
    )
    diag_path = Path("/tmp/grm_cmc1_selftest_diag.json")
    diag_path.write_text(json.dumps({
        "probes": [{
            "turn": 5, "pass": True, "answer": "Auric-4-Alpha",
            "mount_plan": [0], "mounted_ids": [0],
        }]
    }), encoding="utf-8")
    try:
        sup = assess_supersession(sup_path)
        diag = assess_diag(diag_path)
        assert sup["wrong_read_reproduced"] is True
        assert diag["wrong_read_reproduced"] is False and diag["healed"] is True
        try:
            with gpu_lease(MAX_LEASE_SECONDS + 1, 0):
                pass
        except GateError:
            lease_guard = True
        else:
            lease_guard = False
        assert lease_guard
        return {
            "schema": "grm.cmc1_1.gpu_gate.selftest.v2",
            "status": "PASS",
            "g0_semantics": "per_fixture",
            "supersession_assessor": "PASS",
            "diag_healed_assessor": "PASS",
            "lease_cap_guard": "PASS",
            "gpu_imported_or_used": False,
        }
    finally:
        sup_path.unlink(missing_ok=True)
        diag_path.unlink(missing_ok=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture", choices=("all", "supersession", "diag"),
        default="supersession",
        help="CMC1.1 defaults to the sole reproducing fixture")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--gpu", default="0", help="single visible GPU index")
    parser.add_argument("--lease-seconds", type=int, default=DEFAULT_LEASE_SECONDS)
    parser.add_argument("--lock-wait-seconds", type=int, default=7200)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        print(_json_bytes(self_test()).decode("utf-8"), end="")
        return 0
    if int(args.lease_seconds) > MAX_LEASE_SECONDS:
        raise GateError(f"lease cap exceeds {MAX_LEASE_SECONDS}s")
    gpu = str(args.gpu).strip()
    if not gpu or "," in gpu:
        raise GateError("--gpu must name exactly one GPU index")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    run_dir = _new_run_dir(args.artifact_dir)
    selected = (
        ["supersession", "diag"] if args.fixture == "all" else [args.fixture]
    )
    results: dict[str, Any] = {}
    for index, fixture in enumerate(selected):
        if index:
            print(f"gpu_gap_s={GAP_SECONDS}", flush=True)
            time.sleep(GAP_SECONDS)
        with gpu_lease(int(args.lease_seconds), int(args.lock_wait_seconds)):
            if fixture == "supersession":
                results[fixture] = run_supersession(run_dir)
            else:
                results[fixture] = run_diag_prefix(run_dir)

    all_selected_reproduced = all(
        result.get("wrong_read_reproduced") is True
        for result in results.values()
    )
    fixture_status = {}
    arms_authorized_fixtures = []
    excluded_fixtures = []
    for fixture, result in results.items():
        if result.get("wrong_read_reproduced") is True:
            fixture_status[fixture] = "REPRODUCES"
            arms_authorized_fixtures.append(fixture)
        else:
            fixture_status[fixture] = (
                "HEALED_EXCLUDED" if result.get("healed") is True
                else "NOT_REPRODUCED_EXCLUDED")
            excluded_fixtures.append(fixture)
    status = (
        "PASS_PER_FIXTURE" if arms_authorized_fixtures
        else "STOP_NO_REPRODUCING_SELECTED_FIXTURE")
    receipt = {
        "schema": "grm.cmc1.live_g0.v2",
        "order": "GRM-CMC1.1",
        "gate": "CMC-G0",
        "status": status,
        "fixture_selection": selected,
        "all_selected_reproduced": all_selected_reproduced,
        "fixture_status": fixture_status,
        "arms_authorized": bool(arms_authorized_fixtures),
        "arms_authorized_fixtures": arms_authorized_fixtures,
        "excluded_fixtures": excluded_fixtures,
        "arms_run": [],
        "gpu_discipline": {
            "CUDA_VISIBLE_DEVICES": gpu,
            "lock": str(LOCK_PATH),
            "lease_cap_seconds": int(args.lease_seconds),
            "inter_fixture_gap_seconds": GAP_SECONDS,
        },
        "results": results,
    }
    _write_new(run_dir / "CMC_G0_RECEIPT.json", receipt)
    print(_json_bytes(receipt).decode("utf-8"), end="")
    print(f"run_dir={run_dir}")
    # Return non-zero only when none of the selected fixtures may enter arms.
    return 0 if arms_authorized_fixtures else 3


if __name__ == "__main__":
    raise SystemExit(main())
