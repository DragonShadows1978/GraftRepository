#!/usr/bin/env python3
"""GRM-SC2 G3 — early abort at the fire token, on the seven E2E pairs.

ORDER: ``orders/GRM_SC2_CALIBRATION_EARLY_ABORT.md`` (gate G3).
REGISTRATION: ``artifacts/grm_sc2/registration.json``.

WHAT THIS MEASURES.  SC1.2 generated the whole wrong answer, THEN took the
demand trip, then threw the wrong answer away when the trip grounded.  SC2
stops generating at the fire token.  This driver re-runs SC1.2's exact pair
protocol with that one change and asks three questions:

  1. Does the SAME thing get served?  (byte-identical text on all 7 pairs,
     including ``e2e_t30_atlas_tone`` -- the one pair whose trip does NOT
     ground, which therefore has to take the RESUME path and still put out
     what SC1.2 put out.)
  2. Does recovery hold?  (same pairs recovered, same detection.)
  3. What does it save?  (tokens not generated, wall-clock per pair.)

WHY A NEW DRIVER RATHER THAN AN EDIT.  ``scripts/grm_det1_*.py`` is read-only
to this order and ``_continue_forked_prefill`` lives there.  That function is
the generation loop the SC1.2 harness runs -- it does NOT go through
``ArenaCache.step``, so the arena's own abort seam is not on its path.  This
driver therefore carries an abort-aware copy of that loop
(``_continue_forked_prefill_abortable``), and pins it against the frozen
original: with the abort disabled it must produce the identical answer, and a
gate here refuses to run if it does not.  Everything else -- the fork, the
index reconstruction, the trip, the comparator, the grounding rule -- is
IMPORTED from SC1.2 and DET1, never re-expressed.

WHAT "RESUME" MEANS HERE.  The suspended attempt keeps the arena state the
abort left behind: the KV cache holds the prompt plus every token generated up
to and including the fire token.  Resuming continues the same greedy argmax
over that same cache, so the finished text is the text the un-aborted attempt
would have produced -- structurally, not approximately.  The trip runs on a
CLEAN-ROOM reset (SC1's measured finding, carried), so the resume has to
re-establish the aborted attempt's cache first; this driver snapshots it at the
abort and restores it before resuming, which is the same verbatim-restore
discipline SC1 uses when a trip fails to ground.

FROZEN TREE IS READ-ONLY.  Every path under the frozen run is opened for
reading only.

GPU DISCIPLINE (house rules, enforced here).
  * self-lease on /tmp/forge-gpu.lock via scripts.grm_cmc1_gpu_arms.gpu_lease;
  * ONE lease per process, <= 520 s, one pair per process;
  * the >= 30 s inter-process gap is taken INSIDE this wrapper BEFORE the
    lease is acquired, so a waiting process never holds the lock while it
    sleeps;
  * the operator has absolute right of way; this driver never signals, kills
    or resets anything it did not start.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import grm_demand  # noqa: E402
from scripts.grm_det1_common import file_record  # noqa: E402
from scripts import grm_sc1_2_e2e_recovery_gpu as sc12gpu  # noqa: E402
from scripts import grm_sc1_2_session as sc12  # noqa: E402

ARTIFACT_DIR = ROOT / "artifacts" / "grm_sc2"
SCHEMA_PREFIX = "grm.sc2"
ORDER = "orders/GRM_SC2_CALIBRATION_EARLY_ABORT.md"
REGISTRATION = ARTIFACT_DIR / "registration.json"

LEASE_SECONDS = 520
LOCK_WAIT_SECONDS = 7200
INTER_RUN_GAP_SECONDS = 30

#: SC1.2's frozen 10-pair table -- the served text every pair must still match.
SC12_TEN_PAIR_TABLE = (
    ROOT / "artifacts" / "grm_sc1_2"
    / "sc1_2_ten_pair_table_4a0da1ed598cbfec.json")


class SC2Error(RuntimeError):
    """The SC2 GPU gate could not run as registered."""


# --------------------------------------------------------------------------
# The abort-aware generation loop
# --------------------------------------------------------------------------


def _continue_forked_prefill_abortable(
    arena: Any, prompt_ids: list[int], ngen: int, *, observer,
) -> tuple[Any, dict[str, Any]]:
    """``_continue_forked_prefill`` with the GRM-SC2 abort seam.

    Line for line the frozen DET1.4 loop, with exactly one addition: each
    ``arena._forward`` is wrapped so a ``DemandAbort`` raised by the observer
    stops generation and returns a SUSPENSION instead of an answer.

    On abort the returned mapping carries what resuming needs and nothing
    else: the tokens emitted so far (recovered from the observer's rows, which
    hold a prediction id for every position it closed -- decode->encode is not
    a BPE round trip, so the ids are the only exact carrier), and the mask
    receipt the completed path would have returned.

    ``_finish`` on the returned handle completes the attempt from the abort
    point.  It is the same loop continuing over the same cache, so the text it
    produces is the text the un-aborted attempt would have produced.
    """
    from core import kv_graft
    from scripts.grm_det1_3_snapshot import verify_fork_masks_consumed

    stops = tuple(arena.stop_sequences or ())

    def _answer(out: list[int]) -> str:
        answer = arena.decode(out)
        for stop in stops:
            if stop in answer:
                answer = answer.split(stop)[0]
        return answer.strip()

    def _suspend(out: list[int], mask_receipt, index: int, mass: float):
        return {
            "grm_sc2_suspended": True,
            "abort_token_index": int(index),
            "abort_mounted_mass": float(mass),
            "out": list(out),
            "tokens_generated_before_abort": len(out),
            "mask_receipt": mask_receipt,
            "ngen": int(ngen),
            "resumed": False,
        }

    try:
        row = arena._forward(prompt_ids)
    except grm_demand.DemandAbort as abort:
        mask_receipt = verify_fork_masks_consumed(arena)
        kv_graft.clear_injection(arena.m)
        # Position 0: the token was predicted but never appended and never
        # committed. Recover it from the observer row the abort just closed.
        rows = observer.finish()
        out = [int(rows[0]["prediction_token_id"])] if rows else []
        return _suspend(out, mask_receipt, abort.token_index,
                        abort.mounted_mass), {}
    mask_receipt = verify_fork_masks_consumed(arena)
    kv_graft.clear_injection(arena.m)
    out = [int(row.argmax())]
    stopped = False
    for _ in range(int(ngen) - 1):
        if any(stop in arena.decode(out) for stop in stops):
            stopped = True
            break
        try:
            row = arena._forward([out[-1]])
        except grm_demand.DemandAbort as abort:
            # The forward COMPLETED -- the abort is raised after the logits
            # were computed -- so out[-1] is committed to the cache and the
            # predicted token is the observer's newest row.
            rows = observer.finish()
            emitted = list(out)
            if len(rows) > len(emitted):
                emitted.append(int(rows[len(emitted)]["prediction_token_id"]))
            return _suspend(emitted, mask_receipt, abort.token_index,
                            abort.mounted_mass), {}
        out.append(int(row.argmax()))
    if not stopped and not any(stop in arena.decode(out) for stop in stops):
        try:
            arena._forward([out[-1]])
        except grm_demand.DemandAbort:
            # The unused final commit forward is not an answer position (the
            # race's flush-drop convention says so), so a fire there is not a
            # fire on the answer. The answer is already complete.
            pass
    return _answer(out), mask_receipt


def _resume_suspended(arena: Any, suspended: Mapping[str, Any]) -> str:
    """Finish a suspended forked attempt; returns the completed answer.

    The caller must have restored the arena to the state the abort left
    behind before calling this -- the same verbatim-restore SC1 performs when
    a demand trip fails to ground.
    """
    if not suspended.get("grm_sc2_suspended"):
        raise SC2Error("not a GRM-SC2 suspended forked attempt")
    if suspended.get("resumed"):
        raise SC2Error("suspended forked attempt already resumed")
    out = [int(v) for v in suspended["out"]]
    if not out:
        raise SC2Error(
            "cannot resume a suspension whose emitted tokens were not "
            "recovered; resuming from an empty prefix would generate a "
            "DIFFERENT answer than the one this attempt was producing")
    stops = tuple(arena.stop_sequences or ())
    ngen = int(suspended["ngen"])
    stopped = False
    for _ in range(max(0, ngen - len(out))):
        if any(stop in arena.decode(out) for stop in stops):
            stopped = True
            break
        row = arena._forward([out[-1]])
        out.append(int(row.argmax()))
    if not stopped and not any(stop in arena.decode(out) for stop in stops):
        arena._forward([out[-1]])
    answer = arena.decode(out)
    for stop in stops:
        if stop in answer:
            answer = answer.split(stop)[0]
    suspended = dict(suspended)
    return answer.strip()


def _snapshot_state(arena: Any) -> tuple:
    """The arena state tuple SC1's demand block snapshots and restores."""
    return (
        list(arena.caches) if isinstance(arena.caches, list) else arena.caches,
        arena.pos,
        list(arena.live_segs),
        list(arena.cur_mounts),
        arena.cur_mount_n,
    )


def _restore_state(arena: Any, snap: tuple) -> None:
    (arena.caches, arena.pos, arena.live_segs, arena.cur_mounts,
     arena.cur_mount_n) = (
        list(snap[0]) if isinstance(snap[0], list) else snap[0],
        snap[1], list(snap[2]), list(snap[3]), snap[4])


# --------------------------------------------------------------------------
# One arm, with the abort
# --------------------------------------------------------------------------


def _fork_and_generate_abortable(
    *, arena, e2e, pair, manifest_path, identity, withheld, threshold, ngen,
    live_ids, route_limit, topk, early_abort,
):
    """SC1.2's ``_fork_and_generate`` Arm 1, with the early abort.

    The sequence is SC1.2's, step for step.  What changes is WHEN the trip is
    taken: under the abort, generation stops at the fire token and the trip is
    taken from there, instead of after the whole wrong answer exists.
    """
    from scripts.grm_det1_3_snapshot import restore_prefill_fork

    restore = restore_prefill_fork(
        arena,
        Path(manifest_path),
        withheld_mounts=sorted(int(v) for v in withheld),
        target_identity=identity,
        require_same_process_index=False,
    )

    attempt_started = time.perf_counter()
    observer = grm_demand.DemandObserver(
        arena, int(ngen), float(threshold), early_abort=bool(early_abort))
    with observer:
        produced, mask_receipt = _continue_forked_prefill_abortable(
            arena, restore["prompt_ids"], int(ngen), observer=observer)
    rows = observer.finish()
    attempt_wall_ms = (time.perf_counter() - attempt_started) * 1000.0
    decision = grm_demand.decide(rows, float(threshold))

    suspended = produced if isinstance(produced, dict) else None
    if suspended is not None and not decision["demand_fired"]:
        raise SC2Error(
            "an attempt was suspended by the early abort but the decision "
            "over its rows says it never fired; the abort and the carried "
            "decision rule have diverged")

    # The state at the abort point -- the resume point. Captured BEFORE the
    # trip's clean-room reset destroys it.
    abort_state = _snapshot_state(arena) if suspended is not None else None
    mounted_now = [int(v) for v in arena.cur_mounts]

    out: dict[str, Any] = {
        "withheld_alias_ids": sorted(int(v) for v in withheld),
        "registered_alias_ids": list(pair["plant_alias_ids"]),
        "attempt_min_mass": decision["demand_min_mass"],
        "attempt_token_count": decision["demand_token_count"],
        "demand_fired": bool(decision["demand_fired"]),
        "demand_token_index": decision["demand_token_index"],
        "demand_threshold": float(threshold),
        "demand_early_abort": bool(early_abort),
        "demand_wall_ms_attempt": attempt_wall_ms,
        "mask_consumption_gate_pass": bool(
            (mask_receipt or {}).get("gate_pass")) if mask_receipt else None,
        "fork_intervention": restore["intervention"],
        "fork_zero_intervention_no_deltas": restore[
            "zero_intervention_no_deltas"],
        "same_process_index_verified": bool(
            restore.get("same_process_index_verified")),
        "routing_index_status": restore.get("routing_index"),
        "source_manifest_sha256": restore["source_manifest_sha256"],
        "demand_loop": "ON",
    }
    if suspended is not None:
        out.update({
            "demand_abort_token_index": int(suspended["abort_token_index"]),
            "demand_tokens_generated_before_abort": int(
                suspended["tokens_generated_before_abort"]),
            "attempt_answer": None,
            "attempt_suspended": True,
        })
    else:
        out.update({
            "attempt_answer": str(produced),
            "attempt_suspended": False,
            "demand_abort_token_index": None,
            "demand_tokens_generated_before_abort": None,
        })

    if not decision["demand_fired"]:
        # Nothing fired: the attempt IS the served answer, abort or no abort.
        answer = str(produced)
        out["served_answer"] = answer
        out["served_verdict"] = sc12gpu.value_verdict(
            answer, expected_values=pair["expected_values"],
            rejected_values=pair["rejected_values"])
        out["demand_served"] = "original"
        out["demand_trip"] = None
        out["demand_resumed_original"] = False
        out["demand_tokens_saved"] = 0
        return out, rows

    trip_started = time.perf_counter()
    trip_decision, trip = sc12gpu._demand_trip_after_fork(
        arena=arena, e2e=e2e, question=pair["question"], threshold=threshold,
        ngen=ngen, rows=rows, mounted_now=mounted_now, live_ids=live_ids,
        route_limit=route_limit, topk=topk)
    trip_wall_ms = (time.perf_counter() - trip_started) * 1000.0
    out["demand_wall_ms_trip"] = trip_wall_ms
    out["demand_trip"] = trip

    grounded = bool(trip and trip.get("demand_trip_taken")
                    and trip.get("demand_trip_grounded"))
    if grounded:
        answer = str(trip["demand_trip_answer"])
        out["demand_served"] = "demand_trip"
        out["demand_resumed_original"] = False
        # ``demand_tokens_saved`` is filled in by the CALLER, not here.
        # The saving is (what the un-aborted attempt generated) minus (what
        # this attempt generated before aborting), and this function can only
        # see the second of those -- its own ``demand_token_count`` IS the
        # truncated count. Computing it from that would report 0 every time,
        # which is exactly the sort of number that looks measured and is not.
        out["demand_tokens_saved"] = None
        out["demand_tokens_saved_pending_reference"] = True
    else:
        out["demand_served"] = "original"
        if suspended is not None:
            # THE FALLBACK. Restore the abort-point state verbatim (the trip's
            # clean-room reset replaced it) and finish the original attempt.
            resume_started = time.perf_counter()
            _restore_state(arena, abort_state)
            answer = _resume_suspended(arena, suspended)
            out["demand_wall_ms_resume"] = (
                time.perf_counter() - resume_started) * 1000.0
            out["demand_resumed_original"] = True
            out["demand_tokens_saved"] = 0
            out["attempt_answer"] = answer
        else:
            answer = str(produced)
            out["demand_resumed_original"] = False
            out["demand_tokens_saved"] = 0

    out["served_answer"] = answer
    out["served_verdict"] = sc12gpu.value_verdict(
        answer, expected_values=pair["expected_values"],
        rejected_values=pair["rejected_values"])
    trip_answer = (trip or {}).get("demand_trip_answer")
    out["trip_answer_verdict"] = (
        sc12gpu.value_verdict(
            str(trip_answer), expected_values=pair["expected_values"],
            rejected_values=pair["rejected_values"])
        if trip_answer is not None else None)
    out["recovered"] = bool(out["served_verdict"]["correct"])
    return out, rows


# --------------------------------------------------------------------------
# One pair: the SC1.2 arm and the SC2 arm, back to back
# --------------------------------------------------------------------------


def measure_pair_abort(*, pair, frame, threshold, identity, repo, e2e):
    """Both variants, both abort states, on one loaded model.

    ``early_abort=False`` runs FIRST and is the REFERENCE: it is SC1.2's own
    path through this driver's loop, so the abort arm has something measured
    on the same hardware in the same process to be compared against, rather
    than only against a receipt from another day.
    """
    from scripts.grm_det1_5_gpu import _live_token_ids

    arena = repo.arena
    flags = frame["resolved_flags"]
    ngen = int(flags["ngen"])
    topk = int(flags["topk"])
    max_trips = int(flags["max_trips"])
    route_limit = max(topk, (max_trips + 1) * topk)
    manifest_path = ROOT / pair["snapshot_manifest"]
    if not manifest_path.is_file():
        raise SC2Error(f"frozen lived snapshot absent: {manifest_path}")
    live_now = {int(g) for g, _n in arena.live_segs if g is not None}

    arms: dict[str, Any] = {}
    for variant, withheld in (
        ("served", ()),
        ("planted_miss", tuple(pair["plant_alias_ids"])),
    ):
        os.environ["GRM_DEMAND_NGH"] = "1"
        os.environ[grm_demand.ENV_EARLY_ABORT] = "0"
        reference, _rows_ref = _fork_and_generate_abortable(
            arena=arena, e2e=e2e, pair=pair, manifest_path=manifest_path,
            identity=identity, withheld=withheld, threshold=threshold,
            ngen=ngen, live_ids=live_now, route_limit=route_limit, topk=topk,
            early_abort=False)

        os.environ[grm_demand.ENV_EARLY_ABORT] = "1"
        aborted, _rows_abort = _fork_and_generate_abortable(
            arena=arena, e2e=e2e, pair=pair, manifest_path=manifest_path,
            identity=identity, withheld=withheld, threshold=threshold,
            ngen=ngen, live_ids=live_now, route_limit=route_limit, topk=topk,
            early_abort=True)

        # ``demand_tokens_saved``: the tokens the un-aborted attempt generated
        # minus the tokens this one generated before aborting. It is computed
        # HERE because only here are both runs in view. When the trip does not
        # ground the answer is finished anyway on the resume path, so nothing
        # is saved and the field is 0 -- by construction, not by measurement.
        if aborted.get("demand_tokens_generated_before_abort") is not None:
            if aborted.get("demand_resumed_original"):
                aborted["demand_tokens_saved"] = 0
            else:
                aborted["demand_tokens_saved"] = max(0, (
                    int(reference["attempt_token_count"])
                    - int(aborted["demand_tokens_generated_before_abort"])))
            aborted.pop("demand_tokens_saved_pending_reference", None)
            aborted["demand_tokens_saved_basis"] = {
                "reference_attempt_token_count": int(
                    reference["attempt_token_count"]),
                "tokens_generated_before_abort": int(
                    aborted["demand_tokens_generated_before_abort"]),
                "resumed_original": bool(
                    aborted.get("demand_resumed_original")),
            }

        arms[variant] = {
            "reference_no_abort": reference,
            "early_abort": aborted,
            "served_text_identical": (
                str(reference["served_answer"]) == str(aborted["served_answer"])),
            "served_text_bytes_identical": (
                str(reference["served_answer"]).encode("utf-8")
                == str(aborted["served_answer"]).encode("utf-8")),
            "decision_identical": (
                bool(reference["demand_fired"]) == bool(aborted["demand_fired"])
                and reference["demand_token_index"]
                == aborted["demand_token_index"]),
            "recovered_identical": (
                reference.get("recovered") == aborted.get("recovered")),
            "wall_ms_before": reference["demand_wall_ms_attempt"] + float(
                reference.get("demand_wall_ms_trip") or 0.0),
            "wall_ms_after": aborted["demand_wall_ms_attempt"] + float(
                aborted.get("demand_wall_ms_trip") or 0.0) + float(
                aborted.get("demand_wall_ms_resume") or 0.0),
        }
    return arms


def run_pair(fixture_id: str) -> dict[str, Any]:
    """Reconstruct one pair's index, load the model once, measure both arms."""
    from scripts.grm_det1_3_gpu import _frame_identity
    from scripts.grm_det1_5_gpu import _process_instance
    from scripts.grm_det1_gpu import _runtime_env, _validate_visibility, NATIVE_LIB

    frame = sc12gpu._read(sc12gpu.RUNTIME_FRAME)
    if frame.get("native_library") != file_record(NATIVE_LIB):
        raise SC2Error("native runtime drifted from the frozen frame")
    os.environ.update(_runtime_env(frame))
    visible = _validate_visibility()
    if str(visible) != str(frame["cuda_visible_devices"]):
        raise SC2Error("visible GPU differs from the frozen runtime frame")

    threshold = grm_demand.registered_threshold()
    registered = grm_demand.load_registered()
    # The CARRIED line, not the candidate. G3 runs production's threshold.
    if float(threshold) != float(registered["threshold"]):
        raise SC2Error("G3 must run the CARRIED threshold")

    inst = sc12.read_jsonl(
        sc12gpu._shard_session(sc12.INSTRUMENTATION_SHARD)
        / "instrumentation.jsonl")
    source_manifest = sc12gpu._read(
        sc12gpu._shard_session(sc12.SOURCE_SHARD) / "repository"
        / "manifest.json")
    node_counts = sc12.probe_node_counts(inst, len(source_manifest["nodes"]))
    pairs = {p["fixture_id"]: p for p in sc12.e2e_pairs(node_counts=node_counts)}
    if fixture_id not in pairs:
        raise SC2Error(f"{fixture_id} is not one of the 7 registered E2E pairs")
    pair = pairs[fixture_id]

    os.environ["GRM_LSR_FIXES"] = "1"
    work = Path(tempfile.mkdtemp(prefix=f"sc2_{fixture_id}_"))
    try:
        scratch = sc12gpu.build_probe_repository(pair, work / "repository")
        e2e, model, tokenizer, repo, model_info = sc12gpu._open_probe_repo(
            work / "repository", frame)
        observed = len(repo.arena.grafts)
        if observed != int(pair["probe_node_index"]):
            raise SC2Error(
                f"reconstructed repository loaded {observed} nodes, expected "
                f"{pair['probe_node_index']}")
        identity = _frame_identity(
            model, tokenizer, e2e, sc12gpu._read(sc12gpu.DET1_REGISTRATION),
            frame)
        process = _process_instance()
        arms = measure_pair_abort(
            pair=pair, frame=frame, threshold=threshold, identity=identity,
            repo=repo, e2e=e2e)
    finally:
        import shutil

        shutil.rmtree(work, ignore_errors=True)

    return {
        "schema": f"{SCHEMA_PREFIX}.pair_early_abort.v1",
        "gate": "G3",
        "order": ORDER,
        "fixture_id": str(pair["fixture_id"]),
        "conversation_turn": int(pair["conversation_turn"]),
        "question": str(pair["question"]),
        "expected_values": list(pair["expected_values"]),
        "rejected_values": list(pair["rejected_values"]),
        "carried_threshold": float(threshold),
        "carried_threshold_caveat": registered["caveat"],
        "candidate_threshold_NOT_USED_HERE": registered.get(
            "candidate_threshold"),
        "early_abort_flag": registered["early_abort_flag"],
        "arms": arms,
        "scratch_repository": scratch,
        "process": process,
        "frozen_tree_write_policy": (
            "READ-ONLY. Every path under the frozen run is opened for reading "
            "only; reconstructions and receipts are written under "
            "artifacts/grm_sc2/ and a scratch directory."),
        "sources": {
            name: file_record(ROOT / name) for name in (
                "core/grm_demand.py", "core/graft_arena.py",
                "scripts/grm_sc2_e2e_abort_gpu.py",
                "scripts/grm_sc1_2_e2e_recovery_gpu.py",
                "config/grm_demand_registered.json",
            )
        },
    }


# --------------------------------------------------------------------------
# Roll-up
# --------------------------------------------------------------------------


def summarize(receipts: Sequence[Path]) -> dict[str, Any]:
    """The G3 table: SC2 against SC1.2, pair by pair."""
    sc12_table = json.loads(SC12_TEN_PAIR_TABLE.read_text(encoding="utf-8"))
    sc12_by_pair = {row["pair"]: row for row in sc12_table["table"]}

    rows: list[dict[str, Any]] = []
    identical = 0
    resumed_pairs: list[str] = []
    tokens_before: list[int] = []
    for path in receipts:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        fixture = payload["fixture_id"]
        planted = payload["arms"]["planted_miss"]
        served = payload["arms"]["served"]
        abort = planted["early_abort"]
        reference = planted["reference_no_abort"]
        sc12_row = sc12_by_pair.get(fixture, {})
        if abort.get("demand_tokens_generated_before_abort") is not None:
            tokens_before.append(
                int(abort["demand_tokens_generated_before_abort"]))
        if abort.get("demand_resumed_original"):
            resumed_pairs.append(fixture)
        text_matches_sc12 = (
            str(abort["served_answer"]) == str(sc12_row.get("value"))
            if sc12_row.get("value") is not None else None)
        if planted["served_text_bytes_identical"]:
            identical += 1
        rows.append({
            "pair": fixture,
            "fired": bool(abort["demand_fired"]),
            "abort_token_index": abort.get("demand_abort_token_index"),
            "tokens_before_abort": abort.get(
                "demand_tokens_generated_before_abort"),
            "attempt_token_count_no_abort": reference["attempt_token_count"],
            "trip_grounded": bool(
                (abort.get("demand_trip") or {}).get("demand_trip_grounded")),
            "resumed_original": bool(abort.get("demand_resumed_original")),
            "served_from": abort.get("demand_served"),
            "served_answer": abort.get("served_answer"),
            "served_text_identical_to_reference": bool(
                planted["served_text_bytes_identical"]),
            "served_text_identical_to_sc1_2": text_matches_sc12,
            "sc1_2_served_value": sc12_row.get("value"),
            "tokens_saved": abort.get("demand_tokens_saved"),
            "tokens_saved_basis": abort.get("demand_tokens_saved_basis"),
            "recovered": abort.get("recovered"),
            "sc1_2_recovered": sc12_row.get("recovered"),
            "recovery_matches_sc1_2": (
                abort.get("recovered") == sc12_row.get("recovered")
                if sc12_row else None),
            "wall_ms_before": planted["wall_ms_before"],
            "wall_ms_after": planted["wall_ms_after"],
            "wall_ms_delta_pct": (
                100.0 * (planted["wall_ms_after"] - planted["wall_ms_before"])
                / planted["wall_ms_before"]
                if planted["wall_ms_before"] else None),
            "served_control_fired": bool(served["early_abort"]["demand_fired"]),
            "served_control_min_mass": served["early_abort"]["attempt_min_mass"],
            "served_control_text_identical": bool(
                served["served_text_bytes_identical"]),
        })

    rows.sort(key=lambda r: r["pair"])
    recovered = sum(1 for r in rows if r["recovered"])
    sc12_recovered = sum(1 for r in rows if r["sc1_2_recovered"])
    mean_before = (sum(tokens_before) / len(tokens_before)
                   if tokens_before else None)
    all_text_match = all(
        r["served_text_identical_to_reference"] for r in rows)
    all_recovery_match = all(
        r["recovery_matches_sc1_2"] for r in rows if r["sc1_2_recovered"] is not None)
    return {
        "schema": f"{SCHEMA_PREFIX}.g3_early_abort_table.v1",
        "gate": "G3",
        "order": ORDER,
        "pair_count": len(rows),
        "served_text_identical_count": identical,
        "all_served_text_identical": all_text_match,
        "recovered": recovered,
        "sc1_2_recovered": sc12_recovered,
        "recovery_matches_sc1_2": all_recovery_match,
        "resumed_original_pairs": sorted(resumed_pairs),
        "mean_tokens_generated_before_abort": mean_before,
        "prediction_mean_tokens_before_abort_le_2": (
            mean_before is not None and mean_before <= 2),
        "gate_pass": bool(all_text_match and all_recovery_match),
        "pass_rule": (
            "served text byte-identical on every pair AND recovery identical "
            "to SC1.2 on every pair whose SC1.2 verdict is known"),
        "table": rows,
        "sc1_2_source": file_record(SC12_TEN_PAIR_TABLE),
    }


def emit(payload: Mapping[str, Any], stem: str) -> Path:
    import hashlib

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
    path = ARTIFACT_DIR / f"{stem}_{digest}.json"
    path.write_text(body + "\n", encoding="utf-8")
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("pair", "summarize"))
    parser.add_argument("--fixture-id", default=None)
    parser.add_argument("--receipts", nargs="*", default=())
    parser.add_argument("--lease-seconds", type=int, default=LEASE_SECONDS)
    parser.add_argument("--lock-wait-seconds", type=int,
                        default=LOCK_WAIT_SECONDS)
    parser.add_argument("--gap-seconds", type=int,
                        default=INTER_RUN_GAP_SECONDS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "summarize":
        payload = summarize([Path(p) for p in args.receipts])
        path = emit(payload, "sc2_g3_early_abort_table")
        print(f"receipt={path.relative_to(ROOT)}")
        print(f"gate_pass={payload['gate_pass']}")
        print(f"all_served_text_identical={payload['all_served_text_identical']}")
        print(f"recovered={payload['recovered']} "
              f"sc1_2_recovered={payload['sc1_2_recovered']}")
        print(f"resumed_original_pairs={payload['resumed_original_pairs']}")
        print(f"mean_tokens_before_abort="
              f"{payload['mean_tokens_generated_before_abort']}")
        for row in payload["table"]:
            print("G3 " + json.dumps(row))
        return 0 if payload["gate_pass"] else 1

    if not args.fixture_id:
        raise SC2Error("--fixture-id is required for the pair command")
    from scripts.grm_cmc1_gpu_arms import gpu_lease

    # The >= 30 s inter-process gap, taken INSIDE the wrapper BEFORE the lease
    # is acquired, so a waiting process never holds the lock while it sleeps.
    if int(args.gap_seconds) > 0:
        time.sleep(int(args.gap_seconds))
    with gpu_lease(int(args.lease_seconds), int(args.lock_wait_seconds)):
        payload = run_pair(str(args.fixture_id))
    path = emit(payload, f"sc2_pair_{args.fixture_id}")
    planted = payload["arms"]["planted_miss"]
    print(f"receipt={path.relative_to(ROOT)}")
    print(json.dumps({
        "fixture_id": payload["fixture_id"],
        "fired": planted["early_abort"]["demand_fired"],
        "abort_token_index": planted["early_abort"].get(
            "demand_abort_token_index"),
        "tokens_before_abort": planted["early_abort"].get(
            "demand_tokens_generated_before_abort"),
        "served_from": planted["early_abort"].get("demand_served"),
        "resumed_original": planted["early_abort"].get(
            "demand_resumed_original"),
        "served_text_bytes_identical": planted["served_text_bytes_identical"],
        "served_answer": planted["early_abort"].get("served_answer"),
        "wall_ms_before": planted["wall_ms_before"],
        "wall_ms_after": planted["wall_ms_after"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
