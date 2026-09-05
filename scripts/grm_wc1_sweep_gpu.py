#!/usr/bin/env python3
"""GRM-WC1 — run ONE (width, battery) cell of the arena-width sweep.

Order: ``orders/GRM_WC1_ARENA_WIDTH_CURVE.md``.  Registration:
``artifacts/grm_wc1/registration.json`` (IMMUTABLE).

ONE PROCESS = ONE GPU LEASE = ONE UNIT OF WORK, so no call runs long enough to
need a background wait.  The caller inserts the inter-run gap and holds the
whole sweep together; this module never spawns, never polls, and never signals
anything.

THE WIDTH INJECTION, in full, because it is the only thing WC1 does that the
registered harnesses do not already do for themselves:

  * ``scripts/lsr_p2c_e2e_gpu.run_shard`` and
    ``scripts/grm_eb1_longhorizon_gpu.run_shard`` both build their driver argv
    with ``--arena-width str(flags["arena_width"])``, where ``flags`` is
    ``RUNTIME_FRAME``'s ``resolved_flags``.  Rebinding that module constant to
    a width-overridden COPY of the frozen frame is therefore the whole
    injection for those two batteries — no wrapper, no edit.
  * ``scripts/lsr_p2c_replay_gpu.serve_fixture`` reads the same frame for
    ``topk``/``ngen``/``max_trips``, but its ARENA comes from
    ``grm_det1_2_gpu._load_model_repo``, which passes a hard-coded
    ``arena_width=96``.  So that one function is WRAPPED for the duration of
    the call and the single keyword rewritten.  The read-only module is
    imported unchanged; the wrapper lives on the importing module's namespace
    and is removed in a ``finally``.

Both harness modules are READ-ONLY under this order's file boundary and
neither is edited.  Every receipt carries the frozen frame's sha256, the
derived frame's sha256, the machine-checked single-variable verdict, and the
arena width READ OFF THE ARENA THAT ACTUALLY SERVED — not the width that was
requested.  A width that fails to take is a RED result, not a silent legacy
run.

GPU DISCIPLINE (house rules, enforced here).
  * self-lease on /tmp/forge-gpu.lock via ``grm_cmc1_gpu_arms.gpu_lease``;
    a parallel seat runs the SAME order in another worktree, so the lock is
    WAITED ON and never cleared;
  * one lease per process, under the house cap; the operator has absolute
    right of way; nothing is ever killed, signalled or interrupted.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.grm_cmc1_mechanism import (  # noqa: E402
    canonical_json_bytes, sha256_bytes,
)
from scripts.grm_det1_common import file_record  # noqa: E402
from scripts.grm_wc1_common import (  # noqa: E402
    ARTIFACT_DIR, FRAME_DIR, LEVERS, SCHEMA_PREFIX, WIDTH_GRID, WC1Error,
    battery_result, mean_or_none, probe_row, read_json, split_census,
    unchanged_fields, width_frame_payload,
)

REGISTRATION = ARTIFACT_DIR / "registration.json"
RUN_ROOT = ARTIFACT_DIR / "runs"
LEASE_SECONDS = 580
LOCK_WAIT_SECONDS = 7200

SUP_SESSIONS = (
    "correction_then_restatement",
    "fresh_fact_controls",
    "multi_hop_a_b_c",
    "short_correction_long_competitor",
)
CENSUS_SHARDS = ("e2e-1", "e2e-2", "e2e-3", "e2e-4")
LONGHORIZON_SHARDS = ("lh-1", "lh-2", "lh-3", "lh-4", "lh-5")


# ---------------------------------------------------------------------------
# Registration binding + the derived frame
# ---------------------------------------------------------------------------

def registration_record() -> dict[str, Any]:
    if not REGISTRATION.is_file():
        raise WC1Error(
            f"{REGISTRATION} is missing: register BEFORE any gate run")
    return file_record(REGISTRATION)


def derived_frame_path(width: int) -> Path:
    """Write (once) and return the width-overridden frame for ``width``.

    Content-addressed, so re-deriving the same width is a no-op that yields
    the same path and the same digest.
    """
    from scripts.lsr_p2c_replay_gpu import RUNTIME_FRAME as FROZEN

    base = read_json(FROZEN)
    payload = width_frame_payload(base, width)
    if not unchanged_fields(base, payload):
        raise WC1Error(
            f"derived frame for width {width} differs from the frozen frame "
            "in more than resolved_flags.arena_width")
    body = canonical_json_bytes(payload)
    digest = sha256_bytes(body)
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    path = FRAME_DIR / f"runtime_frame_w{int(width)}_{digest[:16]}.json"
    if not path.exists():
        path.write_bytes(body)
    return path


def frame_binding(width: int, frame_path: Path) -> dict[str, Any]:
    from scripts.lsr_p2c_replay_gpu import RUNTIME_FRAME as FROZEN

    base = read_json(FROZEN)
    derived = read_json(frame_path)
    return {
        "requested_width": int(width),
        "frozen_frame": file_record(FROZEN),
        "frozen_width": int(base["resolved_flags"]["arena_width"]),
        "derived_frame": file_record(frame_path),
        "derived_width": int(derived["resolved_flags"]["arena_width"]),
        "single_variable_check": bool(unchanged_fields(base, derived)),
        "single_variable_check_note": (
            "True means: strip the wc1_override block, reset arena_width to "
            "the frozen value, and the derived frame is byte-equal to the "
            "frozen one. Nothing but the width moved."),
    }


@contextmanager
def frame_override(frame_path: Path) -> Iterator[None]:
    """Rebind ``RUNTIME_FRAME`` on every module that reads it, then restore.

    The three registered harnesses import the constant by VALUE
    (``from scripts.lsr_p2c_replay_gpu import RUNTIME_FRAME``), so each
    module's own binding has to be moved, not just the defining module's.
    """
    from scripts import lsr_p2c_replay_gpu as replay

    modules = [replay]
    for name in ("scripts.lsr_p2c_e2e_gpu", "scripts.grm_eb1_longhorizon_gpu"):
        module = sys.modules.get(name)
        if module is not None and hasattr(module, "RUNTIME_FRAME"):
            modules.append(module)
    saved = [(m, m.RUNTIME_FRAME) for m in modules]
    try:
        for module, _old in saved:
            module.RUNTIME_FRAME = frame_path
        yield
    finally:
        for module, old in saved:
            module.RUNTIME_FRAME = old


@contextmanager
def rs3_lever_env() -> Iterator[None]:
    """Pin the registered lever pair through the ENV, then restore it.

    ``lsr_p2c_replay_gpu`` and ``lsr_p2c_e2e_gpu`` take the levers as keyword
    arguments; ``grm_eb1_longhorizon_gpu`` does NOT — it is read-only under
    this order's file boundary, so its levers are supplied the way RS3 Part 4
    supplied them (its own ``G4_longhorizon.note`` records this): through the
    registered environment switches.  Same mechanism, same resolved values;
    ``core.grm_frame`` is the single resolver either way.
    """
    from core.grm_frame import CAPTURE_PIN_ENV, SEAT_NEAR_LIVE_ENV

    wanted = {
        "GRM_LSR_FIXES": "1",
        CAPTURE_PIN_ENV: str(LEVERS["capture_pin"]),
        SEAT_NEAR_LIVE_ENV: "1" if LEVERS["seat_near_live"] else None,
    }
    saved = {k: os.environ.get(k) for k in wanted}
    try:
        for key, value in wanted.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, old in saved.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


@contextmanager
def width_patched_loader(width: int) -> Iterator[None]:
    """Wrap ``_load_model_repo`` so its hard-coded ``arena_width`` becomes ours.

    ``grm_det1_2_gpu._load_model_repo`` is read-only under this order and
    passes ``arena_width=96`` literally.  The wrapper calls the ORIGINAL with
    everything else untouched and then asserts the arena it got back really
    carries the requested width — a width that silently failed to take would
    otherwise masquerade as a legacy 96 run.
    """
    from scripts import lsr_p2c_replay_gpu as replay
    from core import graft_repository as repo_mod

    original_init = repo_mod.GraftRepository.__init__
    requested = int(width)

    def patched_init(self, *args, **kwargs):
        # The ONLY rewrite: the arena width keyword. Positional callers are
        # refused rather than guessed at, because the seat count is the whole
        # measurement.
        if "arena_width" not in kwargs:
            raise WC1Error(
                "GraftRepository was constructed without an arena_width "
                "keyword; WC1 cannot pin the width it is measuring")
        kwargs["arena_width"] = requested
        return original_init(self, *args, **kwargs)

    repo_mod.GraftRepository.__init__ = patched_init
    try:
        yield
    finally:
        repo_mod.GraftRepository.__init__ = original_init
    del replay  # imported only to force the read-only module to be loaded


# ---------------------------------------------------------------------------
# Per-battery cells
# ---------------------------------------------------------------------------

def run_sup_fixture(width: int, session_id: str, run_dir: Path
                    ) -> dict[str, Any]:
    """One sup fixture at one width: the P2C arm-1 replay, width-overridden."""
    from scripts import lsr_p2c_replay_gpu as replay

    frame_path = derived_frame_path(width)
    started = time.monotonic()
    with frame_override(frame_path), width_patched_loader(width):
        payload = replay.serve_fixture(
            session_id, arm=1, capture_pin=str(LEVERS["capture_pin"]),
            seat_near_live=bool(LEVERS["seat_near_live"]))
    elapsed = time.monotonic() - started

    served_width = int(payload.get("arena_width", -1))
    if served_width != int(width):
        raise WC1Error(
            f"sup fixture {session_id}: requested width {width} but the "
            f"receipt records arena_width={served_width}")
    rows = [
        {
            **probe_row(
                str(p["probe_id"]), bool(p["verdict"]["correct"]),
                served=str(p["served_answer"]),
                expected=[str(v) for v in p.get("expected_values", ())]),
            "lived_answer": str(p.get("lived_answer", "")),
            "reproduced_lived": bool(
                (p.get("reproduction") or {}).get("reproduced", False)),
            "elapsed_ns": int(p.get("elapsed_ns", 0)),
            "fit": dict(p.get("info") or {}),
        }
        for p in payload["probes"]
    ]
    out = {
        "schema": f"{SCHEMA_PREFIX}.sup_fixture.v1",
        "width": int(width),
        "battery": "sup",
        "session_id": session_id,
        "rows": rows,
        "correct": sum(1 for r in rows if r["correct"]),
        "total": len(rows),
        "elapsed_seconds": float(elapsed),
        "frame_binding": frame_binding(width, frame_path),
        "arena_width_on_receipt": served_width,
        "rs3_levers": dict(payload["rs3_levers"]),
        "frame": dict(payload["frame"]),
        "registration": registration_record(),
        "replay_receipt_fields": {
            "reproduced_all": bool(payload["reproduced"]),
            "live_graft_ids_after_install": list(
                payload["live_graft_ids_after_install"]),
        },
    }
    _write_run(run_dir, f"sup_{session_id}", out)
    return out


def run_census_shard(width: int, spec: str, run_dir: Path) -> dict[str, Any]:
    """One census shard at one width: the P2C arm-1 E2E chain."""
    from scripts import lsr_p2c_e2e_gpu as census

    frame_path = derived_frame_path(width)
    session_root = run_dir / "census_session"
    started = time.monotonic()
    with frame_override(frame_path):
        result = census.run_shard(
            spec, arm=1, run_dir=session_root,
            capture_pin=str(LEVERS["capture_pin"]),
            seat_near_live=bool(LEVERS["seat_near_live"]))
    elapsed = time.monotonic() - started
    out = {
        "schema": f"{SCHEMA_PREFIX}.census_shard.v1",
        "width": int(width),
        "battery": "census",
        "spec": spec,
        "elapsed_seconds": float(elapsed),
        "shard": result,
        "frame_binding": frame_binding(width, frame_path),
        "registration": registration_record(),
    }
    _write_run(run_dir, f"census_{spec}", out)
    return out


def score_census(width: int, run_dir: Path) -> dict[str, Any]:
    """Score the finished census chain against the ten lived census rows."""
    from scripts import lsr_p2c_e2e_gpu as census

    frame_path = derived_frame_path(width)
    session_root = run_dir / "census_session"
    with frame_override(frame_path):
        payload = census.score_run(session_root, arm=1)
    # The census scorer records its verdict under ``verdict.correct`` (the
    # DET1 unified semantic comparator), NOT as a top-level ``correct``.
    # Reading the wrong key would score every probe False and fake a collapse.
    rows = [
        {
            **probe_row(
                str(p["probe_id"]),
                bool((p.get("verdict") or {}).get("correct", False)),
                served=str(p.get("served_answer", "")),
                expected=[str(v) for v in p.get("expected_values", ())]),
            "turn": int(p.get("turn", 0)),
            "lived_answer": str(p.get("lived_answer", "")),
            "driver_pass": bool(p.get("driver_pass", False)),
            "turn_row_found": bool(p.get("turn_row_found", False)),
        }
        for p in payload["probes"]
    ]
    out = {
        "schema": f"{SCHEMA_PREFIX}.census_score.v1",
        "width": int(width),
        "battery": "census",
        "rows": rows,
        "correct": sum(1 for r in rows if r["correct"]),
        "total": len(rows),
        "frame_binding": frame_binding(width, frame_path),
        "registration": registration_record(),
        "source_payload_keys": sorted(payload.keys()),
    }
    _write_run(run_dir, "census_score", out)
    return out


def run_longhorizon_shard(width: int, spec: str, run_dir: Path
                          ) -> dict[str, Any]:
    """One long-horizon shard at one width: the EB1 104-turn script."""
    from scripts import grm_eb1_longhorizon_gpu as lh

    frame_path = derived_frame_path(width)
    session_root = run_dir / "lh_session"
    from core.grm_frame import capture_pin_mode, seat_near_live_enabled

    started = time.monotonic()
    with frame_override(frame_path), rs3_lever_env():
        resolved = {
            "capture_pin": capture_pin_mode(None),
            "seat_near_live": bool(seat_near_live_enabled(None)),
        }
        result = lh.run_shard(spec, run_dir=session_root)
    elapsed = time.monotonic() - started
    if (resolved["capture_pin"] != str(LEVERS["capture_pin"])
            or resolved["seat_near_live"] != bool(LEVERS["seat_near_live"])):
        raise WC1Error(
            f"long-horizon shard {spec}: levers did not resolve as registered "
            f"({resolved} != {dict(LEVERS)})")
    out = {
        "schema": f"{SCHEMA_PREFIX}.longhorizon_shard.v1",
        "width": int(width),
        "battery": "longhorizon",
        "spec": spec,
        "elapsed_seconds": float(elapsed),
        "rs3_levers_resolved": resolved,
        "shard": result,
        "frame_binding": frame_binding(width, frame_path),
        "registration": registration_record(),
    }
    _write_run(run_dir, f"lh_{spec}", out)
    return out


#: The four EB1 G5 spot probes the order scores: distances 36-60.  Read from
#: the EB1 receipt rather than retyped, so "the same 4 probes" is a binding
#: and not a claim.
EB1_SPOT_RECEIPT = Path(
    "/mnt/ForgeRealm/GraftRepository/artifacts/grm_rs3/grm_rs3_part4.json")


def eb1_spot_probes() -> list[dict[str, Any]]:
    payload = json.loads(EB1_SPOT_RECEIPT.read_text(encoding="utf-8"))
    return [dict(r) for r in payload["G4_longhorizon"]["spot_check_probes"]]


def score_longhorizon(width: int, run_dir: Path) -> dict[str, Any]:
    """Score the long-horizon chain, and pull out the 4 registered spots."""
    from scripts import grm_eb1_longhorizon_gpu as lh

    frame_path = derived_frame_path(width)
    session_root = run_dir / "lh_session"
    with frame_override(frame_path):
        payload = lh.score_run(session_root)
    by_turn = {int(r["turn"]): r for r in payload["probes"]}
    spot_rows = []
    for spot in eb1_spot_probes():
        turn = int(spot["turn"])
        row = by_turn.get(turn)
        spot_rows.append({
            **probe_row(
                f"lh_t{turn}_{str(spot['fact']).replace(' ', '_')}",
                bool(row.get("correct", False)) if row else False,
                served=str((row or {}).get("served_answer", "")),
                expected=[str(spot["expected"])]),
            "turn": turn,
            "distance": int(spot["distance"]),
            "eb1_expected": str(spot["expected"]),
            "eb1_served": str(spot["served"]),
            "present_in_run": row is not None,
        })
    out = {
        "schema": f"{SCHEMA_PREFIX}.longhorizon_score.v1",
        "width": int(width),
        "battery": "longhorizon",
        "rows": spot_rows,
        "correct": sum(1 for r in spot_rows if r["correct"]),
        "total": len(spot_rows),
        "all_probes": payload["probes"],
        "measured_count": int(payload.get("measured_count", 0)),
        "correct_count_all": int(payload.get("correct_count", 0)),
        "recall_by_distance_bucket": payload.get(
            "recall_by_distance_bucket", {}),
        "frame_binding": frame_binding(width, frame_path),
        "registration": registration_record(),
    }
    _write_run(run_dir, "lh_score", out)
    return out


# ---------------------------------------------------------------------------
# Receipts
# ---------------------------------------------------------------------------

def _write_run(run_dir: Path, stem: str, payload: Mapping[str, Any]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    body = canonical_json_bytes(payload)
    digest = sha256_bytes(body)
    path = run_dir / f"{stem}_{digest[:16]}.json"
    path.write_bytes(body)
    print(f"receipt={path}")
    print(f"sha256={digest}")
    return path


def run_dir_for(width: int) -> Path:
    return RUN_ROOT / f"w{int(width)}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("plan", "sup", "census-shard", "census-score",
                 "lh-shard", "lh-score", "frame"))
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--session-id", default=None, choices=SUP_SESSIONS)
    parser.add_argument("--spec", default=None)
    parser.add_argument("--lease-seconds", type=int, default=LEASE_SECONDS)
    parser.add_argument("--lock-wait-seconds", type=int,
                        default=LOCK_WAIT_SECONDS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.command == "plan":
        print(json.dumps({
            "widths": list(WIDTH_GRID),
            "sup_sessions": list(SUP_SESSIONS),
            "census_shards": list(CENSUS_SHARDS),
            "longhorizon_shards": list(LONGHORIZON_SHARDS),
            "gpu_units_per_width": (
                len(SUP_SESSIONS) + len(CENSUS_SHARDS)
                + len(LONGHORIZON_SHARDS)),
            "registration": registration_record(),
        }, indent=1))
        return 0

    if args.width is None:
        raise WC1Error("--width is required")
    width = int(args.width)
    if width not in WIDTH_GRID:
        raise WC1Error(
            f"width {width} is not on the registered grid {list(WIDTH_GRID)}")
    run_dir = run_dir_for(width)

    if args.command == "frame":
        path = derived_frame_path(width)
        print(json.dumps(frame_binding(width, path), indent=1))
        return 0

    # Scoring commands are CPU-only: no lease, no GPU.
    if args.command == "census-score":
        out = score_census(width, run_dir)
        print(f"census_correct={out['correct']}/{out['total']}")
        return 0
    if args.command == "lh-score":
        out = score_longhorizon(width, run_dir)
        print(f"lh_spot_correct={out['correct']}/{out['total']}")
        print(f"lh_all={out['correct_count_all']}/{out['measured_count']}")
        return 0

    from scripts.grm_cmc1_gpu_arms import gpu_lease

    with gpu_lease(int(args.lease_seconds), int(args.lock_wait_seconds)):
        if args.command == "sup":
            if not args.session_id:
                raise WC1Error("--session-id is required for sup")
            out = run_sup_fixture(width, str(args.session_id), run_dir)
            print(f"sup_correct={out['correct']}/{out['total']}")
            for row in out["rows"]:
                print(json.dumps({
                    "probe_id": row["probe_id"],
                    "correct": row["correct"],
                    "served": row["served_answer"][:80],
                }))
        elif args.command == "census-shard":
            if args.spec not in CENSUS_SHARDS:
                raise WC1Error(f"--spec must be one of {CENSUS_SHARDS}")
            out = run_census_shard(width, str(args.spec), run_dir)
            print(f"shard_elapsed_s={out['elapsed_seconds']:.1f}")
        elif args.command == "lh-shard":
            if args.spec not in LONGHORIZON_SHARDS:
                raise WC1Error(f"--spec must be one of {LONGHORIZON_SHARDS}")
            out = run_longhorizon_shard(width, str(args.spec), run_dir)
            print(f"shard_elapsed_s={out['elapsed_seconds']:.1f}")
        else:  # pragma: no cover - argparse constrains the choices
            raise WC1Error(f"unknown command {args.command!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
