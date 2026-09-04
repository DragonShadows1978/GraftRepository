#!/usr/bin/env python3
"""GRM-LSR-P2C G3 — the ten e2e_t* census probes, deposit-order replay.

Program: LSR (Lived-Serving Reliability), Phase 2C.
Order:   orders/GRM_LSR_P2C_UNSEATABLE_SPLIT_DESCENT.md

WHAT THIS IS.  The ten ``e2e_t*`` rows of the DET1.11 lived-serving census
are TURNS INSIDE ONE CERTIFIED 34-TURN SESSION, not standalone fixtures.  A
deposit-order replay therefore means re-running that whole session, which is
what this module does — SHARDED by turn range, with repository state carried
between shards by the driver's own save/restore (``flush_now`` + the restart
receipt + ``--resume``), exactly the mechanism ``grm_det1_5_workers._run_e2e``
uses.  No shard exceeds the GPU bounds.

SHARD PLAN (the frozen DET1.5 ``E2E_SPECS`` turn ranges, reused not
re-invented): turns 1-10, 1-17, 1-25, 1-34.  Each shard copies the previous
shard's session directory and resumes into it, so shard N restores the
repository the previous shard persisted and then runs the new turns.
Measured lived wall times for these shards were 89-153 s each
(``artifacts/grm_det1/.../shards/e2e-*/attempt_*/worker_output.json``),
comfortably inside the 580 s lease.

ARMS, as for G2.
  * ``--arm 0``  fixes OFF (``GRM_LSR_FIXES=0``): the pre-P2A fit stage.
    Must reproduce the lived served values on the ten census probes.
  * ``--arm 1``  fixes ON (default): P2A fit honesty + P2C split/descent.

Turn 33 is the driver's final FILLER in the certified script; DET1.9 replaced
it with the registered Polaris recall probe to create the census's
``e2e_t33_polaris_mark`` row, and this module installs the same replacement
via ``grm_det1_5_workers._install_polaris_probe`` so the census's tenth probe
exists in the replay at all.

GPU DISCIPLINE.  One SHARD per process, one lease per shard, <= 580 s,
operator right of way; the caller inserts the inter-run gap.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.grm_frame import (  # noqa: E402
    ENV_NAME as PERSISTENT_BOAT_ENV,
    env_persistent_boat,
    ephemeral_frame_enabled,
)
from core.grm_frame import (  # noqa: E402
    CAPTURE_PIN_ENV, CAPTURE_PIN_OFF, CAPTURE_PINS, SEAT_NEAR_LIVE_ENV,
    capture_pin_mode, seat_near_live_enabled,
)
from scripts.grm_cmc1_mechanism import canonical_json_bytes, sha256_bytes  # noqa: E402
from scripts.grm_det1_common import contains_value, file_record  # noqa: E402
from scripts.lsr_p2c_replay_gpu import (  # noqa: E402
    CENSUS,
    RUNTIME_FRAME,
    census_rows,
    reproduction_verdict,
)

ARTIFACT_DIR = ROOT / "artifacts" / "lsr_p2c"
SCHEMA_PREFIX = "grm.lsr_p2c"
LEASE_SECONDS = 580
LOCK_WAIT_SECONDS = 7200

#: The frozen DET1.5 E2E turn ranges, in order.  Shard N resumes shard N-1.
SHARDS = ("e2e-1", "e2e-2", "e2e-3", "e2e-4")


class P2CE2EError(RuntimeError):
    """The G3 replay could not be performed as registered."""


def _read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def e2e_census_probes() -> list[dict[str, Any]]:
    """The ten ``e2e_t*`` census rows, keyed to their session turn."""
    rows = census_rows()
    out = []
    for probe_id, row in sorted(rows.items()):
        if not probe_id.startswith("e2e_"):
            continue
        turn = int(str(probe_id).split("_")[1].lstrip("t"))
        out.append({
            "probe_id": probe_id,
            "turn": turn,
            "expected_values": [
                str(v) for v in row.get("expected_values", ())],
            "lived_answer": str(row.get("served_answer", "")),
            "lived_correct": bool(row.get("served_answer_correct", False)),
            "role": str(row.get("role", "")),
        })
    out.sort(key=lambda r: r["turn"])
    return out


def _stop_after(spec: str) -> int:
    from scripts.grm_det1_5_workers import E2E_SPECS

    if spec not in E2E_SPECS:
        raise P2CE2EError(f"unknown E2E shard spec: {spec}")
    stop = E2E_SPECS[spec]["stop_after_turns"]
    # The last shard runs the full 34-turn script so turn 33 (the Polaris
    # replacement probe) is reached; _run_e2e widens e2e-4 the same way.
    return 34 if spec == "e2e-4" else int(stop)


def run_shard(spec: str, *, arm: int, run_dir: Path,
              capture_pin: str | None = None,
              seat_near_live: bool | None = None) -> dict[str, Any]:
    """Run one shard of the certified session, resuming the previous one.

    GRM-RS3 adds ``capture_pin`` and ``seat_near_live``: FLAG PLUMBING ONLY.
    Both default ``None``, which resolves through the (default-OFF, fail-
    closed) env, so a shard that passes neither is the legacy shard byte for
    byte.  Passing them pins the process env for the whole shard, so every
    deposit and every bootstrap seating in those turns runs under the named
    lever.  The resolved values ride on the shard's return.

    EVERY SHARD IN A CHAIN MUST BE RUN WITH THE SAME LEVERS.  A shard resumes
    the previous shard's persisted repository, so mixing levers across a chain
    would serve later turns against grafts captured under a different
    geometry — a confound, not an arm.  The caller owns that discipline; the
    receipt records what each shard actually ran with so a mixed chain is
    visible rather than silent.
    """
    from scripts import grm_det1_e2e as det_e2e
    from scripts import grm_e2e_session as e2e
    from scripts.grm_det1_5_workers import _install_polaris_probe

    if arm not in (0, 1):
        raise P2CE2EError(f"arm must be 0 or 1, got {arm}")
    frame = _read(RUNTIME_FRAME)
    flags = frame["resolved_flags"]
    os.environ["GRM_LSR_FIXES"] = "1" if arm else "0"
    resolved_pin = capture_pin_mode(capture_pin)
    resolved_seat = seat_near_live_enabled(seat_near_live)
    if resolved_pin == CAPTURE_PIN_OFF:
        os.environ.pop(CAPTURE_PIN_ENV, None)
    else:
        os.environ[CAPTURE_PIN_ENV] = resolved_pin
    if resolved_seat:
        os.environ[SEAT_NEAR_LIVE_ENV] = "1"
    else:
        os.environ.pop(SEAT_NEAR_LIVE_ENV, None)

    arm_dir = Path(run_dir) / f"arm{int(arm)}"
    session_dir = arm_dir / spec / "session"
    index = SHARDS.index(spec)
    resume = index > 0
    if session_dir.exists():
        shutil.rmtree(session_dir)
    session_dir.parent.mkdir(parents=True, exist_ok=True)
    if resume:
        prior = arm_dir / SHARDS[index - 1] / "session"
        if not prior.is_dir():
            raise P2CE2EError(
                f"shard {spec} needs {SHARDS[index - 1]} to have run first: "
                f"{prior} is missing")
        # The driver's OWN save/restore: copy the persisted session (the
        # repository directory, the restart receipt, the transcript) and
        # resume into it.  This is exactly what _run_e2e does.
        shutil.copytree(prior, session_dir, symlinks=True)

    stop_after = _stop_after(spec)
    restore_script = None
    if spec == "e2e-4":
        # DET1.9's registered turn-33 recall probe: the census's tenth e2e
        # row exists only because this replacement was installed.
        restore_script = _install_polaris_probe(e2e)
    det_e2e.install_stop_boundary(e2e, stop_after)
    argv = [
        "--mode", "full",
        "--session-dir", str(session_dir),
        "--model-dir", str(frame["model"]["path"]),
        "--native-lib", str(ROOT / frame["native_library"]["path"]),
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
    original_turn = e2e.run_turn
    started = time.monotonic()
    stopped = False
    try:
        e2e.main(argv)
    except det_e2e._DETStageStop:
        stopped = True
    finally:
        e2e.run_turn = original_turn
        if restore_script is not None:
            e2e.build_full_script = restore_script
    elapsed = time.monotonic() - started
    if not stopped:
        raise P2CE2EError(f"shard {spec} did not stop at turn {stop_after}")
    scorecard = session_dir / "probe_scorecard.json"
    if not scorecard.is_file():
        raise P2CE2EError(f"shard {spec} produced no probe scorecard")
    return {
        "spec": spec,
        "stop_after_turns": int(stop_after),
        "resumed": bool(resume),
        "session_dir": str(session_dir),
        "elapsed_seconds": float(elapsed),
        "scorecard": file_record(scorecard),
        "probe_rows": _read(scorecard).get("probes", []),
        # GRM-RS3: the levers THIS shard served under.
        "rs3_levers": {
            "capture_pin": resolved_pin,
            "seat_near_live": bool(resolved_seat),
            "capture_pin_env": os.environ.get(CAPTURE_PIN_ENV),
            "seat_near_live_env": os.environ.get(SEAT_NEAR_LIVE_ENV),
            "both_off_is_the_legacy_run": (
                resolved_pin == CAPTURE_PIN_OFF and not resolved_seat),
        },
    }


def score_run(run_dir: Path, *, arm: int) -> dict[str, Any]:
    """Score the finished shard chain against the ten census rows."""
    arm_dir = Path(run_dir) / f"arm{int(arm)}"
    final = arm_dir / SHARDS[-1] / "session" / "probe_scorecard.json"
    if not final.is_file():
        raise P2CE2EError(
            f"arm {arm} has no final scorecard at {final}; run every shard")
    probe_rows = _read(final).get("probes", [])
    by_turn = {int(row["turn"]): row for row in probe_rows if "turn" in row}
    served = []
    for probe in e2e_census_probes():
        row = by_turn.get(int(probe["turn"]))
        answer = str((row or {}).get("answer", ""))
        hit = [
            v for v in probe["expected_values"] if contains_value(answer, v)]
        served.append({
            **probe,
            "served_answer": answer,
            "driver_pass": bool((row or {}).get("pass")),
            "verdict": {"expected_hits": hit, "correct": bool(hit)},
            "reproduction": reproduction_verdict(
                answer, str(probe["lived_answer"])),
            "turn_row_found": row is not None,
        })
    # GRM-EB1: the FRAME this census ran under, read from the restart receipts
    # the driver wrote (which record the arena's observed ``ephemeral`` and
    # whether the transcript re-feed was skipped), not assumed from the env.
    frame_rows = []
    for spec in SHARDS:
        restart = arm_dir / spec / "session" / "restart.json"
        if restart.is_file():
            payload = _read(restart)
            frame_rows.append({
                "shard": spec,
                "frame_ephemeral": payload.get("frame_ephemeral"),
                "frame_escape_active": payload.get("frame_escape_active"),
                "refeed": payload.get("refeed"),
                "refeed_skipped_reason": payload.get("refeed_skipped_reason"),
            })

    return {
        "schema": f"{SCHEMA_PREFIX}.e2e_replay.v1",
        "program": "LSR",
        "phase": "2C",
        "gate": "G3",
        "frame": {
            "ephemeral_declared": bool(ephemeral_frame_enabled()),
            "escape_env": os.environ.get(PERSISTENT_BOAT_ENV),
            "escape_active": bool(env_persistent_boat()),
            "per_shard_restart_receipts": frame_rows,
            "spec": (
                "GRM-EB1: the chat log is not kept in memory context; any "
                "chat recall on facts is pulled via GRM"),
            "arm0_reproduction_claimed": False,
            "arm0_reproduction_note": (
                "A different frame is a NEW baseline. This run does NOT claim "
                "or attempt Arm-0 reproduction of persistent-frame rows; the "
                "lived column is carried for CHANGE, not for identity."),
        },
        "arm": int(arm),
        "arm_label": (
            "arm0_reproduction_fixes_off" if arm == 0
            else "arm1_p2a_p2c_fixes_on"),
        "shards": list(SHARDS),
        "probes": served,
        "reproduced": all(
            bool(r["reproduction"]["reproduced"]) for r in served),
        "correct_count": sum(1 for r in served if r["verdict"]["correct"]),
        "declares_in_det_envelope": False,
        "runtime_frame": file_record(RUNTIME_FRAME),
        "census": file_record(CENSUS),
        "sources": {
            "lsr_p2c_e2e_gpu": file_record(Path(__file__).resolve()),
            "graft_arena": file_record(ROOT / "core" / "graft_arena.py"),
            "graft_repository": file_record(
                ROOT / "core" / "graft_repository.py"),
            "grm_admission": file_record(ROOT / "core" / "grm_admission.py"),
            "grm_runtime": file_record(ROOT / "core" / "grm_runtime.py"),
            "grm_e2e_session": file_record(
                ROOT / "scripts" / "grm_e2e_session.py"),
        },
    }


def emit(payload: Mapping[str, Any], stem: str) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    body = canonical_json_bytes(payload)
    digest = sha256_bytes(body)
    path = ARTIFACT_DIR / f"{stem}_{digest[:16]}.json"
    path.write_bytes(body)
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "shard", "score"))
    parser.add_argument("--spec", default=None, choices=SHARDS)
    parser.add_argument("--arm", type=int, default=None, choices=(0, 1))
    # GRM-RS3 flag plumbing. Both default to the legacy run. Pass the SAME
    # values to every shard in a chain: a shard resumes the previous shard's
    # persisted repository, so mixing levers mid-chain is a confound.
    parser.add_argument("--capture-pin", default=None, choices=CAPTURE_PINS,
                        help="GRM-RS3 capture geometry pin (default: off)")
    parser.add_argument("--seat-near-live", action="store_true",
                        default=False,
                        help="GRM-RS3 seat the plan head next to the live "
                             "band (default: off)")
    parser.add_argument("--run-dir", type=Path,
                        default=ROOT / "artifacts" / "lsr_p2c" / "g3_run")
    parser.add_argument("--lease-seconds", type=int, default=LEASE_SECONDS)
    parser.add_argument("--lock-wait-seconds", type=int,
                        default=LOCK_WAIT_SECONDS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "plan":
        print(json.dumps({
            "shards": [
                {"spec": s, "stop_after_turns": _stop_after(s),
                 "resumes": (SHARDS[i - 1] if i else None)}
                for i, s in enumerate(SHARDS)
            ],
            "probes": e2e_census_probes(),
        }, indent=1))
        return 0

    if args.arm is None:
        raise P2CE2EError("--arm is required")
    if args.command == "score":
        payload = score_run(args.run_dir, arm=int(args.arm))
        path = emit(payload, f"lsr_p2c_g3_arm{int(args.arm)}")
        print(f"receipt={path}")
        print(f"reproduced_all={payload['reproduced']} "
              f"correct={payload['correct_count']}/10")
        for row in payload["probes"]:
            print(json.dumps({
                "probe_id": row["probe_id"],
                "turn": row["turn"],
                "lived": row["lived_answer"][:70],
                "served": row["served_answer"][:70],
                "reproduced": row["reproduction"]["reproduced"],
                "correct": row["verdict"]["correct"],
            }))
        return 0

    if not args.spec:
        raise P2CE2EError("--spec is required for the shard command")

    from scripts.grm_cmc1_gpu_arms import gpu_lease

    with gpu_lease(int(args.lease_seconds), int(args.lock_wait_seconds)):
        out = run_shard(str(args.spec), arm=int(args.arm),
                        run_dir=args.run_dir,
                        capture_pin=args.capture_pin,
                        seat_near_live=(
                            True if args.seat_near_live else None))
    print(json.dumps({
        "spec": out["spec"],
        "stop_after_turns": out["stop_after_turns"],
        "resumed": out["resumed"],
        "elapsed_seconds": round(out["elapsed_seconds"], 1),
        "probe_rows": len(out["probe_rows"]),
        "rs3_levers": out["rs3_levers"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
