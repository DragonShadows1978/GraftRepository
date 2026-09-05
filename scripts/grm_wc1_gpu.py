"""Foreground, leased WC1 runs using the existing deposit and shard harnesses."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.grm_wc1_sweep import (BASELINE, CENSUS, FLAGS, FRAME, OUT, WIDTHS,
                                  read, record, write)

REGISTRATION_SHA256 = "d27b4d79ef5cfd7eea5d6ee134e6f5a1f2e0c9a72275351d3b815b97e5479b0f"
NATIVE = OUT / "native_build/libgrm_runtime.so"


def check_registration():
    if record(OUT / "registration.json")["sha256"] != REGISTRATION_SHA256:
        raise RuntimeError("immutable registration changed")


def split_counts(arena):
    return {"split_parents": sum(bool((g.get("metadata") or {}).get("width_guard_parent"))
                                 for g in arena.grafts),
            "split_children": sum(bool((g.get("metadata") or {}).get("width_guard_child"))
                                  for g in arena.grafts),
            "repository_nodes": len(arena.grafts)}


@contextmanager
def measurement_bindings(width, battery, turns, counts):
    """Constructor substitution happens before RoPE extension or any deposit.

    Delegation preserves all other loader arguments, including fixture feed
    order. Wrappers only inspect scalar state after the real serve returns.
    """
    from core import graft_repository
    from scripts import grm_det1_2_gpu as loader
    from scripts import grm_e2e_session as e2e
    from scripts import lsr_p2c_replay_gpu as sup
    original_init = graft_repository.GraftRepository.__init__
    original_serve = sup._serve_probe_arm1
    original_turn = e2e.run_turn

    def init(repo, *args, **kwargs):
        kwargs["arena_width"] = width
        original_init(repo, *args, **kwargs)
        if repo.arena.width != width or not repo.arena.ephemeral:
            raise RuntimeError("observed arena does not match the registration")

    def serve(repo, e2e_module, question, flags):
        # Keep all the production info fields before P2C's receipt filter.
        original = e2e_module._probe_ladder_chat
        full_info = {}
        def chat(*args, **kwargs):
            answer, info = original(*args, **kwargs)
            full_info.update(info)
            return answer, info
        start = time.perf_counter()
        with patch.object(e2e_module, "_probe_ladder_chat", chat):
            result = original_serve(repo, e2e_module, question, flags)
        turns.append({"question": question, "wall_ms": (time.perf_counter() - start) * 1000,
                      "resident": full_info.get("resident"),
                      "mounted_seats": repo.arena.cur_mount_n,
                      "arena_width": repo.arena.width,
                      "live_shift": repo.arena.live_shift, "info": full_info})
        counts.update(split_counts(repo.arena))
        return result

    def turn(repo, event, turn_idx, **kwargs):
        result = original_turn(repo, event, turn_idx, **kwargs)
        # Driver persists its own timing. Cache length is read without GPU work.
        turns.append({"turn": turn_idx, "kind": event["kind"],
                      "resident": 0 if repo.arena.caches is None else repo.arena._cache_len(),
                      "mounted_seats": repo.arena.cur_mount_n,
                      "arena_width": repo.arena.width,
                      "frame_ephemeral": repo.arena.ephemeral})
        counts.update(split_counts(repo.arena))
        return result

    with patch.object(graft_repository.GraftRepository, "__init__", init), \
         patch.object(loader, "NATIVE_LIB", NATIVE), \
         patch.object(sup, "_serve_probe_arm1", serve), \
         patch.object(e2e, "run_turn", turn):
        yield


def run(width, battery, spec):
    from scripts import lsr_p2c_replay_gpu as sup
    from scripts import lsr_p2c_e2e_gpu as census
    from scripts import grm_eb1_longhorizon_gpu as lh
    from scripts.grm_det1_gpu import _runtime_env
    frame = copy.deepcopy(read(FRAME))
    frame["resolved_flags"]["arena_width"] = width
    frame["native_library"] = record(NATIVE)
    frame_path = OUT / f"width_{width}/runtime_frame.json"
    if frame_path.exists():
        if read(frame_path) != frame:
            raise RuntimeError("width runtime frame differs from earlier shard")
    else:
        write(frame_path, frame, exclusive=True)
    os.environ.update(_runtime_env(frame))
    os.environ.update(FLAGS)
    # Disable inherited experimental detectors; production flags are pinned above.
    os.environ["GRM_DET1_ENABLED"] = "0"
    sup.CENSUS = CENSUS
    sup.RUNTIME_FRAME = census.RUNTIME_FRAME = lh.RUNTIME_FRAME = frame_path
    census.CENSUS = CENSUS
    result_path = OUT / f"width_{width}/{battery}/{spec}.json"
    if result_path.exists():
        raise RuntimeError(f"receipt already exists: {result_path}")
    turns, counts = [], {}
    with measurement_bindings(width, battery, turns, counts):
        if battery == "sup":
            # tempfile's working directories remain below the authorized root.
            tmp = OUT / f"width_{width}/sup/repositories"
            tmp.mkdir(parents=True, exist_ok=True)
            import tempfile
            with patch.object(tempfile, "tempdir", str(tmp)):
                result = sup.serve_fixture(spec, arm=1, capture_pin="live", seat_near_live=True)
        elif battery == "census":
            result = census.run_shard(spec, arm=1,
                                      run_dir=OUT / f"width_{width}/census/run",
                                      capture_pin="live", seat_near_live=True)
        else:
            result = lh.run_shard(spec, run_dir=OUT / f"width_{width}/longhorizon/run")
    result.update({"wc1_width": width, "wc1_battery": battery,
                   "wc1_registration_sha256": REGISTRATION_SHA256,
                   "wc1_flags": dict(FLAGS), "wc1_counts": counts,
                   "wc1_turns": turns, "wc1_native": record(NATIVE)})
    receipt = write(result_path, result, exclusive=True)
    print(json.dumps({"receipt": receipt, "counts": counts}), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "pytest"))
    parser.add_argument("--width", type=int, choices=WIDTHS, default=BASELINE)
    parser.add_argument("--battery", choices=("sup", "census", "longhorizon"))
    parser.add_argument("--spec")
    parser.add_argument("--test-files", nargs="+")
    args = parser.parse_args()
    check_registration()
    if args.command == "run":
        if not args.battery or not args.spec:
            parser.error("run requires --battery and --spec")
        if args.width != BASELINE:
            gate = OUT / "g2.json"
            if not gate.exists() or read(gate).get("status") != "GREEN":
                raise RuntimeError("G2 is not GREEN; sweep is forbidden")
    from scripts.grm_cmc1_gpu_arms import gpu_lease
    inherited = read(FRAME)["gpu_lease"]
    cap = int(inherited["actual_cap_seconds"])
    gap = int(inherited["inter_stage_gap_seconds"])
    started = time.monotonic()
    print(f"foreground_gap_seconds={gap}", flush=True)
    time.sleep(gap)
    with gpu_lease(cap, cap - gap):
        remaining = int(cap - (time.monotonic() - started))
        if remaining <= gap:
            raise TimeoutError("lock wait consumed this foreground call; no GPU workload started")
        # The wait is included in the foreground bound, not added to it.
        signal.alarm(remaining)
        print(f"foreground_work_budget_seconds={remaining}", flush=True)
        if args.command == "pytest":
            command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                       "-p", "tests.grm_wc1_guard",
                       "--continue-on-collection-errors", *(args.test_files or [])]
            print(json.dumps({"command": command}), flush=True)
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
            result = subprocess.run(command, env=env, timeout=remaining - gap, check=False)
            print(f"SHARD_RC={result.returncode}", flush=True)
            return result.returncode
        try:
            return run(args.width, args.battery, args.spec)
        except Exception as exc:
            write(OUT / f"width_{args.width}/{args.battery}/{args.spec}_error.json",
                  {"status": "BLOCKED_EXECUTION", "width": args.width,
                   "battery": args.battery, "spec": args.spec,
                   "error_type": type(exc).__name__, "error": str(exc),
                   "registration_sha256": REGISTRATION_SHA256}, exclusive=True)
            raise


if __name__ == "__main__":
    raise SystemExit(main())
