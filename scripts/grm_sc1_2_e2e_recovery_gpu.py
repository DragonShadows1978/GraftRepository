#!/usr/bin/env python3
"""GRM-SC1.2 G2/G3 — the seven E2E pairs, resumed from the frozen session.

ORDER: ``orders/GRM_SC1_2_E2E_PAIRS_SESSION_RESUME.md``.

THE QUESTION.  SC1 proved the demand loop recovers on the 3 registered pairs
its standalone harness could reach.  The flip decision needs 10.  The other 7
are ``certified_34_turn`` E2E fixtures whose lived rows were produced inside a
chained session by the campaign worker, whose ``_context`` refuses to run
outside the frozen run tree.  This driver measures those 7 WITHOUT writing
into the frozen tree, and without calling the campaign worker at all.

HOW THE SESSION IS RESUMED (registered path (a)).  Every one of the 7 pairs
has a per-probe-turn lived snapshot inside the frozen tree, at exactly the
boundary the race forked from: ``grm.det1_3.model_visible_snapshot.v2``,
label ``lived``, phase ``before_probe_prefill``, finalized.  So the driver
does not replay 34 turns; it hydrates the arena FROM that snapshot with
``restore_prefill_fork`` -- the same DET1.3 mechanism whose forks produced the
lived rows in the first place -- and continues the prefill.

  Arm 0  fork with NOTHING withheld  -> must reproduce the lived served row
         fork with the registered plant seats withheld -> must reproduce the
         lived planted-miss row
  Arm 1  the same two forks, demand ON, fixes ON -> detection / fetch /
         served-from / value / recovered

CROSS-PROCESS, AND SAID OUT LOUD.  The lived capture process is long gone, so
``require_same_process_index=False``.  That is not a new liberty: it is the
DET1.4 zero gate's own path (``scripts/grm_det1_4_gpu.py``), used here for the
same reason.  What it costs is stated in every receipt: ``restore`` returns
``routing_index = NOT_VERIFIED_CROSS_PROCESS_ZERO_GATE_ONLY``, so the full
D-LQR index projection is NOT proven identical to the lived process's.  D-LQR
is REFUTED-STRUCTURAL and unused here.  The index fidelity that the demand
trip DOES depend on is evidenced instead by (1) rebuilding the index as of the
probe turn rather than as of a later flush, (2) re-routing the lived question
through it and comparing against the lived ``ranking_ids`` recorded on that
same turn, and (3) the Arm 0 reproduction result itself.

WHAT restore_prefill_fork DOES NOT GIVE, AND WHY THE INDEX IS REBUILT.  The
snapshot carries MOUNTED PAYLOAD ONLY.  The demand trip's entire job is to
re-route into the UNMOUNTED index and fetch the withheld node, so that index
has to exist.  It is rebuilt from the frozen shard session's persisted
repository, truncated to the node prefix that existed at the probe turn, with
the mutable lineage back-edges RECOMPUTED rather than copied (an end-of-shard
flush says node 0 is retired at t05, when the lived t05 probe mounted it).
See ``grm_sc1_2_session`` for the derivation and its self-check.

FROZEN TREE IS READ-ONLY.  Every path under the frozen run is opened for
reading only.  The reconstructed repositories are written under a scratch
directory; the snapshots this driver itself captures go under
``artifacts/grm_sc1_2/session/``.

GPU DISCIPLINE (house rules, enforced here).
  * self-lease on /tmp/forge-gpu.lock via scripts.grm_cmc1_gpu_arms.gpu_lease;
  * ONE lease per process, <= 520 s, one pair per process by default;
  * the >= 30 s inter-process gap is taken INSIDE this wrapper BEFORE the
    lease is acquired, exactly as SC1.1 did;
  * the operator has absolute right of way.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import grm_demand  # noqa: E402
from scripts.grm_cmc1_mechanism import (  # noqa: E402
    canonical_json_bytes, sha256_bytes,
)
from scripts.grm_det1_common import contains_value, file_record  # noqa: E402
from scripts import grm_sc1_2_session as sc12  # noqa: E402

ARTIFACT_DIR = sc12.ARTIFACT_DIR
SESSION_DIR = ARTIFACT_DIR / "session"
SCHEMA_PREFIX = "grm.sc1_2"
ORDER = sc12.ORDER
REGISTRATION = sc12.REGISTRATION
RUNTIME_FRAME = sc12.RUNTIME_FRAME
DET1_REGISTRATION = sc12.DET1_REGISTRATION

LEASE_SECONDS = 520
LOCK_WAIT_SECONDS = 7200
INTER_RUN_GAP_SECONDS = 30

SC12Error = sc12.SC12Error


def _read(path: Path | str) -> Any:
    return sc12.read_json(path)


def value_verdict(
    answer: str,
    *,
    expected_values: Sequence[str],
    rejected_values: Sequence[str],
) -> dict[str, Any]:
    """The DET1 unified semantic comparator, with its negative guards.

    Identical to SC1's -- imported semantics, not a re-implementation, so a
    value scored correct here is correct by the race's own rule.
    """
    hit = [v for v in expected_values if contains_value(answer, v)]
    bad = [v for v in rejected_values if contains_value(answer, v)]
    return {
        "expected_hits": hit,
        "rejected_hits": bad,
        "correct": bool(hit and not bad),
    }


# --------------------------------------------------------------------------
# Index reconstruction on disk
# --------------------------------------------------------------------------
def _shard_session(shard: str) -> Path:
    return sc12.SHARDS / shard / "attempt_001" / "session"


def build_probe_repository(pair: Mapping[str, Any], dest: Path) -> dict[str, Any]:
    """Write a repository directory holding the index AS OF the probe turn.

    Reads (never writes) the frozen source session's repository, keeps the
    first ``N`` node payloads and index rows, and writes a manifest whose
    lineage back-edges were recomputed for that prefix.  The WAL directory is
    created EMPTY and the native checkpoint is dropped, so nothing can replay
    or reload a later state behind the truncation.
    """
    import numpy as np

    source = _shard_session(sc12.SOURCE_SHARD) / "repository"
    count = int(pair["probe_node_index"])
    source_manifest = _read(source / "manifest.json")
    no_fold_ids = {int(i) for i in pair["no_fold_ids"]}
    manifest = sc12.truncated_manifest(
        source_manifest, count,
        no_fold_at_probe={i: (i in no_fold_ids) for i in range(count)})

    dest.mkdir(parents=True, exist_ok=True)
    (dest / "nodes").mkdir(parents=True, exist_ok=True)
    # An EMPTY wal directory: manifest wal_lsn already equals the frozen WAL's
    # max lsn, and shipping no records makes that structural instead of
    # merely true.
    (dest / "wal").mkdir(parents=True, exist_ok=True)

    copied = []
    for index in range(count):
        name = f"{index:04d}.npz"
        src = source / "nodes" / name
        if not src.is_file():
            raise SC12Error(f"frozen repository lacks node payload {name}")
        shutil.copyfile(src, dest / "nodes" / name)
        copied.append(name)

    with np.load(source / "index.npz") as bundle:
        keys = [f"rkey_{index:04d}" for index in range(count)]
        missing = [k for k in keys if k not in bundle.files]
        if missing:
            raise SC12Error(f"frozen routing index lacks rows {missing}")
        np.savez(dest / "index.npz", **{k: bundle[k] for k in keys})

    (dest / "manifest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8")
    return {
        "schema": f"{SCHEMA_PREFIX}.probe_repository.v1",
        "path": str(dest),
        "node_count": count,
        "source_repository": str(source),
        "source_manifest": file_record(source / "manifest.json"),
        "source_index": file_record(source / "index.npz"),
        "node_payloads_copied": len(copied),
        "native_checkpoint_dropped": True,
        "wal_shipped_empty": True,
        "no_fold_ids": sorted(no_fold_ids),
        "no_fold_pinned": bool(pair["no_fold_pinned"]),
        "lineage_rule": (
            "superseded_by / active / retired RECOMPUTED as the closure of "
            "the prefix's own deposit-final `supersedes` edges; never copied "
            "from the end-of-shard flush, which records a LATER state"),
    }


def _open_probe_repo(repo_dir: Path, frame: Mapping[str, Any]):
    """A GraftRepository bound to a reconstructed probe-turn index.

    Every constructor argument matches ``scripts/grm_det1_2_gpu._load_model_repo``
    -- the frozen race's own loader -- except the repository path, which points
    at the reconstruction instead of an empty scratch dir.  Divergence here
    would silently change the arena the fork is validated against.
    """
    from transformers import AutoTokenizer
    from core.gpt_oss20b_tc import GptOss20B_TC, gpt_oss_grm_dialect_kwargs
    from core.graft_repository import GraftRepository
    from scripts import grm_e2e_session as e2e
    from scripts.grm_det1_gpu import NATIVE_LIB

    model_dir = str(frame["model"]["path"])
    model, model_info = GptOss20B_TC.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    dialect = gpt_oss_grm_dialect_kwargs(model.config)
    flags = frame["resolved_flags"]
    repo = GraftRepository(
        model,
        lambda text: tokenizer.encode(text, add_special_tokens=False),
        lambda ids: tokenizer.decode(ids, clean_up_tokenization_spaces=False),
        str(repo_dir),
        autosave=False,
        arena_cls=e2e.GptOssGQAArenaCache,
        native_lib_path=str(NATIVE_LIB),
        native_auto=False,
        vram_budget_mb=None,
        route_layer=int(dialect["route_layer"]),
        arena_width=int(flags["arena_width"]),
        topk=int(flags["topk"]),
        live_turns=int(flags["live_turns"]),
        max_live=int(flags["max_live"]),
        sink_text=e2e.HARMONY_SINK,
        prompt_template=e2e.harmony_turn,
        stop_sequences=e2e.HARMONY_STOPS,
        storage_bits=int(flags["graft_storage_bits"]),
        revision_resolution=bool(flags["sup_resolve"]),
        decisive_admission=bool(flags["adm_decisive"]),
    )
    return e2e, model, tokenizer, repo, model_info


# --------------------------------------------------------------------------
# The demand trip on a forked arena (SC1's sequence, carried)
# --------------------------------------------------------------------------
def _demand_trip_after_fork(
    *, arena, e2e, question, threshold, ngen, rows, mounted_now, live_ids,
    route_limit, topk,
):
    """The production demand trip, run on a forked arena.

    This is SC1's ``_demand_trip_after_fork`` sequence carried verbatim in
    behaviour -- decide, CLEAN-ROOM roll back, rebuild the query from the
    model's own partial output before the fire token, re-route excluding what
    is already mounted or live, fit, generate once, judge grounding, cap 1.
    It is re-expressed here rather than imported because SC1's copy is bound
    to that order's fixture shape; the SEQUENCE is not changed, and any change
    to it would be a silent law change across two orders.

    The clean-room reset (not a re-fork) is SC1's measured finding, carried:
    generating the trip on top of the failed attempt leaves the refusal in
    context in front of the new mount, and re-forking consumes the one-shot
    allowed-masks the trip's own routing probe needs.
    """
    from core.grm_admission import decisive_admission_profile

    decision = grm_demand.decide(rows, float(threshold))
    if not decision["demand_fired"]:
        return decision, None
    prefix = grm_demand.demand_prefix_text(
        arena, rows, int(decision["demand_token_index"]))
    query = grm_demand.demand_query_text(question, prefix)
    exclude = {int(v) for v in live_ids} | {int(v) for v in mounted_now}
    if getattr(arena, "decisive_admission", False):
        profile = decisive_admission_profile(
            arena, query, exclude=exclude, route_limit=int(route_limit))
        ranking = [int(v) for v in profile["ranking"]]
        plan = [int(v) for v in profile["rank_plan"]]
    else:
        ranking = [int(v) for v in (arena.route(
            query, exclude=exclude, limit=int(route_limit)) or [])]
        plan = ranking[:int(topk)]
    picks = sorted(int(v) for v in e2e._budget_fit_mounts(arena, plan))
    trip: dict[str, Any] = {
        "demand_query_text_sha256": grm_demand.demand_query_sha256(query),
        "demand_prefix": prefix,
        "demand_prefix_token_count": int(decision["demand_token_index"]),
        "demand_ranking": ranking,
        "demand_plan": plan,
        "demand_fetched": picks,
        "demand_fetch_source": grm_demand.DEMAND_FETCH_SOURCE,
        "demand_trip_cap": grm_demand.DEMAND_TRIP_CAP,
        "demand_excluded": sorted(exclude),
    }
    if not picks:
        trip.update({
            "demand_trip_taken": False,
            "demand_trip_answer": None,
            "demand_trip_grounded": None,
            "demand_refired": None,
            "demand_no_candidate": True,
        })
        return decision, trip
    arena.caches, arena.pos, arena.live_segs = None, 0, []
    arena.cur_mounts, arena.cur_mount_n = [], 0
    for layer in arena.m.layers:
        layer.self_attn.live_shift = arena.live_shift
    with grm_demand.DemandObserver(
            arena, int(ngen), float(threshold)) as observer:
        answer, _info = arena._attempt(
            question, picks, int(ngen), False,
            arena.stop_sequences or (), defer_memory=True)
    refire_rows = observer.finish()
    refire = grm_demand.decide(refire_rows, float(threshold))
    mounted_after = [int(v) for v in arena.cur_mounts]
    grounding_fields: dict[str, Any] = {}
    grounded, contributors = arena._grounding_attribution(
        answer, mounted_after, question)
    arena._grounding_receipt(answer, mounted_after, question, grounding_fields)
    legacy_grounded, _l = arena._grounding_verdict(
        answer, mounted_after, question, normalized=False)
    normalized_grounded, _n = arena._grounding_verdict(
        answer, mounted_after, question, normalized=True)
    trip.update({
        "demand_trip_taken": True,
        "demand_trip_answer": str(answer),
        "demand_trip_mounted_ids": mounted_after,
        "demand_trip_grounded": bool(grounded),
        "demand_trip_grounded_legacy": bool(legacy_grounded),
        "demand_trip_grounded_normalized": bool(normalized_grounded),
        "demand_trip_grounding_normalized": bool(
            grounding_fields.get("grounding_normalized")),
        "demand_trip_grounding_glyph_rescued": bool(
            grounding_fields.get("grounding_glyph_rescued")),
        "demand_trip_grounding_contributors": sorted(
            int(v) for v in (contributors or ())),
        "demand_trip_min_mass": refire["demand_min_mass"],
        "demand_refired": bool(refire["demand_fired"]),
        # NEVER LOOP: recorded, not acted on.  Cap 1 is registered.
        "demand_refire_acted_on": False,
        "demand_no_candidate": False,
    })
    return decision, trip


# --------------------------------------------------------------------------
# One pair
# --------------------------------------------------------------------------
def _fork_and_generate(
    *, arena, e2e, pair, manifest_path, identity, withheld, threshold, ngen,
    demand_on, live_ids, route_limit, topk,
):
    """One arm: hydrate from the frozen lived snapshot, generate, observe.

    ``withheld`` empty  -> the served control (a fire here is a FALSE fire)
    ``withheld`` set    -> the planted miss (the positive)

    Both arms hydrate from the SAME frozen snapshot, so the only difference
    between them is the withheld seats -- the race's own construction.
    """
    from scripts.grm_det1_3_snapshot import restore_prefill_fork
    from scripts.grm_det1_4_gpu import _continue_forked_prefill

    restore = restore_prefill_fork(
        arena,
        Path(manifest_path),
        withheld_mounts=sorted(int(v) for v in withheld),
        target_identity=identity,
        # The lived capture process is long gone.  This is the DET1.4 zero
        # gate's own path; what it does not verify is recorded below.
        require_same_process_index=False,
    )
    started = time.time_ns()
    with grm_demand.DemandObserver(
            arena, int(ngen), float(threshold)) as observer:
        answer, mask_receipt = _continue_forked_prefill(
            arena, restore["prompt_ids"], int(ngen))
    rows = observer.finish()
    elapsed = int(time.time_ns() - started)
    decision = grm_demand.decide(rows, float(threshold))
    mounted_now = [int(v) for v in arena.cur_mounts]
    verdict = value_verdict(
        answer,
        expected_values=pair["expected_values"],
        rejected_values=pair["rejected_values"])

    out: dict[str, Any] = {
        "withheld_alias_ids": sorted(int(v) for v in withheld),
        "registered_alias_ids": list(pair["plant_alias_ids"]),
        "attempt_answer": str(answer),
        "attempt_verdict": verdict,
        "attempt_mounted_ids": mounted_now,
        "attempt_min_mass": decision["demand_min_mass"],
        "attempt_token_count": decision["demand_token_count"],
        "demand_fired": bool(decision["demand_fired"]),
        "demand_token_index": decision["demand_token_index"],
        "demand_threshold": float(threshold),
        "elapsed_ns": elapsed,
        "mask_consumption_gate_pass": bool(mask_receipt.get("gate_pass")),
        "fork_intervention": restore["intervention"],
        "fork_source_mounts": restore["source_mounts"],
        "fork_mounts": restore["fork_mounts"],
        "fork_removed_seat_ranges": restore["removed_seat_ranges"],
        "fork_zero_intervention_no_deltas": restore[
            "zero_intervention_no_deltas"],
        "same_process_index_verified": bool(
            restore.get("same_process_index_verified")),
        "routing_index_status": restore.get("routing_index"),
        "source_manifest_sha256": restore["source_manifest_sha256"],
    }
    if not demand_on:
        # ARM 0.  The demand loop is OFF, so no trip is taken and the served
        # answer IS the attempt -- which is exactly what has to match lived.
        out["demand_loop"] = "OFF"
        out["served_answer"] = str(answer)
        out["served_verdict"] = verdict
        return out, rows

    out["demand_loop"] = "ON"
    trip_decision, trip = _demand_trip_after_fork(
        arena=arena, e2e=e2e, question=pair["question"], threshold=threshold,
        ngen=ngen, rows=rows, mounted_now=mounted_now, live_ids=live_ids,
        route_limit=route_limit, topk=topk)
    served_from = "original"
    final_answer = str(answer)
    if trip and trip.get("demand_trip_taken") and trip.get(
            "demand_trip_grounded"):
        served_from = "demand_trip"
        final_answer = str(trip["demand_trip_answer"])
    final_verdict = value_verdict(
        final_answer,
        expected_values=pair["expected_values"],
        rejected_values=pair["rejected_values"])
    trip_answer = (trip or {}).get("demand_trip_answer")
    trip_verdict = (
        value_verdict(
            str(trip_answer),
            expected_values=pair["expected_values"],
            rejected_values=pair["rejected_values"])
        if trip_answer is not None else None)
    is_planted = bool(withheld)
    out.update({
        "demand_trip": trip,
        "demand_served": served_from,
        "served_answer": final_answer,
        "served_verdict": final_verdict,
        "trip_answer_verdict": trip_verdict,
        # RECOVERY is exactly this: a planted-miss turn that ends serving the
        # expected value BECAUSE the trip went out.  Same definition as
        # SC1/SC1.1; not redefined here.
        "recovered": bool(
            is_planted and served_from == "demand_trip"
            and final_verdict["correct"]),
        # REPORTED alongside, never substituted: the trip fetched the right
        # node and the model read it, and only grounding stopped the serve.
        "recovery_blocked_by_grounding_only": bool(
            is_planted and served_from == "original"
            and trip_verdict is not None and trip_verdict["correct"]
            and not (trip or {}).get("demand_trip_grounded")),
    })
    _ = trip_decision
    return out, rows


def measure_pair(
    *, pair, frame, threshold, identity, repo, e2e, scratch,
) -> dict[str, Any]:
    """Both arms x both demand states for one registered E2E pair.

    Order matters and is deliberate: Arm 0 (demand OFF) runs FIRST for both
    variants, so the reproduction verdict is decided before anything the
    demand loop does can touch it.  A pair that fails Arm 0 is still measured
    in Arm 1 for the record, but ``arm0_reproduced`` False is what the table
    reads, and the roll-up excludes it.
    """
    from scripts.grm_det1_common import route_fixture_profile
    from scripts.grm_det1_e2e import _restore_counterfactual, _snapshot_counterfactual
    from scripts.grm_det1_5_gpu import _live_token_ids

    arena = repo.arena
    flags = frame["resolved_flags"]
    ngen = int(flags["ngen"])
    topk = int(flags["topk"])
    max_trips = int(flags["max_trips"])
    route_limit = max(topk, (max_trips + 1) * topk)
    manifest_path = ROOT / pair["snapshot_manifest"]
    if not manifest_path.is_file():
        raise SC12Error(f"frozen lived snapshot absent: {manifest_path}")

    # INDEX-FIDELITY CHECK, before any fork.  Re-route the lived question
    # through the reconstructed index and compare with the ranking the lived
    # turn recorded.  This is the evidence that stands in for the
    # cross-process routing-index verification the fork cannot do.
    live_now = {int(g) for g, _n in arena.live_segs if g is not None}
    base = _snapshot_counterfactual(arena)
    try:
        profile = route_fixture_profile(
            arena, str(pair["question"]),
            (pair["expected_values"] or [""])[0],
            live_excluded=live_now,
            route_limit=route_limit,
        )
    finally:
        _restore_counterfactual(arena, base)
    reconstructed_ranking = [int(v) for v in profile.get("raw_ranking", ())]
    lived_ranking = [int(v) for v in pair["lived"]["served_raw_ranking_ids"]]
    index_check = {
        "lived_raw_ranking_ids": lived_ranking,
        "reconstructed_raw_ranking_ids": reconstructed_ranking,
        "rank1_match": bool(
            reconstructed_ranking and lived_ranking
            and reconstructed_ranking[0] == lived_ranking[0]),
        "prefix_match": bool(
            reconstructed_ranking[:len(lived_ranking)] == lived_ranking),
        "node_count": int(pair["probe_node_index"]),
        "meaning": (
            "The reconstructed probe-turn index must route the lived question "
            "the way the lived turn did. rank1_match is the load-bearing leg "
            "(it is the node the turn mounts); prefix_match is reported "
            "alongside and a tail difference is named, not hidden."),
    }
    live_ids = _live_token_ids(arena)

    arms: dict[str, dict[str, Any]] = {}
    for variant, withheld in (
        ("served", ()),
        ("planted_miss", tuple(pair["plant_alias_ids"])),
    ):
        lived_answer = pair["lived"][
            "served_answer" if variant == "served" else "planted_answer"]
        lived_mass = pair["lived"][
            "served_min_mass" if variant == "served" else "planted_min_mass"]

        os.environ["GRM_DEMAND_NGH"] = "0"
        arm0, _rows0 = _fork_and_generate(
            arena=arena, e2e=e2e, pair=pair, manifest_path=manifest_path,
            identity=identity, withheld=withheld, threshold=threshold,
            ngen=ngen, demand_on=False, live_ids=live_now,
            route_limit=route_limit, topk=topk)
        arm0_verdict = sc12.reproduction_verdict(
            lived_answer=lived_answer,
            fork_answer=arm0["attempt_answer"],
            lived_min_mass=lived_mass,
            fork_min_mass=arm0["attempt_min_mass"],
            # A withheld fork is by construction NOT byte-identical to the
            # lived capture's arrays; the ZERO fork is.
            byte_exact_fork=(not withheld
                             and bool(arm0["fork_zero_intervention_no_deltas"])),
        )
        arm0.update(arm0_verdict)

        os.environ["GRM_DEMAND_NGH"] = "1"
        arm1, _rows1 = _fork_and_generate(
            arena=arena, e2e=e2e, pair=pair, manifest_path=manifest_path,
            identity=identity, withheld=withheld, threshold=threshold,
            ngen=ngen, demand_on=True, live_ids=live_now,
            route_limit=route_limit, topk=topk)

        arms[variant] = {
            "status": "MEASURED",
            "arm0": arm0,
            "arm1": arm1,
            "lived_answer": lived_answer,
            "lived_min_mass": lived_mass,
        }

    reproduced = sc12.pair_reproduced(arms)
    return {
        "fixture_id": str(pair["fixture_id"]),
        "conversation_turn": int(pair["conversation_turn"]),
        "shard": str(pair["shard"]),
        "question": str(pair["question"]),
        "expected_values": list(pair["expected_values"]),
        "rejected_values": list(pair["rejected_values"]),
        "plant_alias_ids": list(pair["plant_alias_ids"]),
        "plant_target_id": pair.get("plant_target_id"),
        "probe_node_index": int(pair["probe_node_index"]),
        "no_fold_ids": list(pair["no_fold_ids"]),
        "no_fold_pinned": bool(pair["no_fold_pinned"]),
        "session_resume_path": "a_frozen_per_probe_turn_snapshot_fork",
        "index_reconstruction": dict(scratch),
        "index_fidelity_check": index_check,
        "frozen_snapshot": file_record(manifest_path),
        "lived": dict(pair["lived"]),
        "arm0_reproduced": bool(reproduced),
        "arms": arms,
    }


# --------------------------------------------------------------------------
# Session driver
# --------------------------------------------------------------------------
def run_pair(fixture_id: str) -> dict[str, Any]:
    """Reconstruct one pair's index, load the model once, measure both arms."""
    from scripts.grm_det1_3_gpu import _frame_identity
    from scripts.grm_det1_5_gpu import _process_instance
    from scripts.grm_det1_gpu import _runtime_env, _validate_visibility, NATIVE_LIB

    frame = _read(RUNTIME_FRAME)
    if frame.get("native_library") != file_record(NATIVE_LIB):
        raise SC12Error("native runtime drifted from the frozen frame")
    os.environ.update(_runtime_env(frame))
    visible = _validate_visibility()
    if str(visible) != str(frame["cuda_visible_devices"]):
        raise SC12Error("visible GPU differs from the frozen runtime frame")

    threshold = grm_demand.registered_threshold()
    registered = grm_demand.load_registered()

    inst = sc12.read_jsonl(
        _shard_session(sc12.INSTRUMENTATION_SHARD) / "instrumentation.jsonl")
    source_manifest = _read(
        _shard_session(sc12.SOURCE_SHARD) / "repository" / "manifest.json")
    node_counts = sc12.probe_node_counts(inst, len(source_manifest["nodes"]))
    pairs = {p["fixture_id"]: p for p in sc12.e2e_pairs(node_counts=node_counts)}
    if fixture_id not in pairs:
        raise SC12Error(f"{fixture_id} is not one of the 7 registered E2E pairs")
    pair = pairs[fixture_id]

    # The reconstruction proofs, computed and RECORDED before the model loads.
    shard_manifests = {
        shard: _read(_shard_session(shard) / "repository" / "manifest.json")
        for shard in ("e2e-1", "e2e-2", "e2e-3", "e2e-4")
    }
    lineage = sc12.lineage_selfcheck(shard_manifests)
    if not lineage["all_match"]:
        raise SC12Error(
            "lineage derivation does not reproduce the frozen manifests; "
            "the index reconstruction is not trustworthy")
    agreements = {
        shard: sc12.prefix_agreement(
            source_manifest["nodes"], manifest["nodes"])
        for shard, manifest in shard_manifests.items()
    }
    disagreeing = [s for s, a in agreements.items() if not a["agree"]]
    if disagreeing:
        raise SC12Error(
            f"frozen shard manifests disagree on deposit-time fields: "
            f"{disagreeing}")
    ranking_check = sc12.ranking_within_prefix(pair)
    if not ranking_check["within"]:
        raise SC12Error(
            f"{fixture_id} lived ranking escapes its reconstructed prefix: "
            f"{ranking_check['outside_prefix']}")

    os.environ["GRM_LSR_FIXES"] = "1"

    work = Path(tempfile.mkdtemp(prefix=f"sc1_2_{fixture_id}_"))
    repo = model = tokenizer = None
    model_info: Any = None
    try:
        scratch = build_probe_repository(pair, work / "repository")
        e2e, model, tokenizer, repo, model_info = _open_probe_repo(
            work / "repository", frame)
        observed = len(repo.arena.grafts)
        if observed != int(pair["probe_node_index"]):
            raise SC12Error(
                f"reconstructed repository loaded {observed} nodes, expected "
                f"{pair['probe_node_index']}")
        identity = _frame_identity(
            model, tokenizer, e2e, _read(DET1_REGISTRATION), frame)
        process = _process_instance()
        result = measure_pair(
            pair=pair, frame=frame, threshold=threshold, identity=identity,
            repo=repo, e2e=e2e, scratch=scratch)
    finally:
        try:
            if repo is not None:
                repo.close()
        except Exception:  # noqa: BLE001 - teardown must not mask a result
            pass
        try:
            from core import kv_graft

            if model is not None:
                kv_graft.clear_injection(model)
        except Exception:  # noqa: BLE001
            pass
        del repo, model, tokenizer
        shutil.rmtree(work, ignore_errors=True)

    return {
        "schema": f"{SCHEMA_PREFIX}.pair_measurement.v1",
        "gate": "G2+G3",
        "fixture_id": fixture_id,
        "order": file_record(ORDER),
        "registration": file_record(REGISTRATION),
        "lsr_fixes_env": os.environ.get("GRM_LSR_FIXES"),
        "carried_threshold": float(threshold),
        "carried_threshold_caveat": str(registered["caveat"]),
        "carried_threshold_provenance": {
            "path": registered["provenance"]["thresholds_path"],
            "sha256": registered["provenance"]["thresholds_sha256"],
            "fit_turn_count": int(registered["fit_turn_count"]),
        },
        "session_resume_protocol": (
            "PATH (a): the certified session is RESUMED at the probe turn by "
            "hydrating the arena from the frozen per-probe-turn lived "
            "snapshot with restore_prefill_fork, the same DET1.3 mechanism "
            "that produced the lived rows. No 34-turn replay. The campaign "
            "worker's _context is never called and is not pointed at a copy: "
            "this driver imports the measurement functions instead."),
        "frozen_tree_write_policy": (
            "READ-ONLY. Every path under the frozen run is opened for reading "
            "only; reconstructions and receipts are written elsewhere."),
        "cross_process_fork_caveat": (
            "require_same_process_index=False (the lived process is gone). "
            "This is the DET1.4 zero gate's own path. restore() therefore "
            "reports routing_index=NOT_VERIFIED_CROSS_PROCESS_ZERO_GATE_ONLY: "
            "the full D-LQR index projection is NOT proven identical to the "
            "lived process's. D-LQR is REFUTED-STRUCTURAL and unused here; "
            "the index fidelity the demand trip depends on is evidenced by "
            "index_fidelity_check and by the Arm 0 reproduction itself."),
        "probe_node_counts": {str(k): int(v) for k, v in node_counts.items()},
        "lineage_selfcheck": lineage,
        "prefix_agreement": agreements,
        "ranking_within_prefix": ranking_check,
        "process": process,
        "result": result,
        "runtime_frame": file_record(RUNTIME_FRAME),
        "model": model_info,
        "sources": {
            "grm_sc1_2_e2e_recovery_gpu": file_record(Path(__file__).resolve()),
            "grm_sc1_2_session": file_record(
                ROOT / "scripts" / "grm_sc1_2_session.py"),
            "grm_demand": file_record(ROOT / "core" / "grm_demand.py"),
            "graft_arena": file_record(ROOT / "core" / "graft_arena.py"),
            "grm_text_norm": file_record(ROOT / "core" / "grm_text_norm.py"),
            "grm_e2e_session": file_record(
                ROOT / "scripts" / "grm_e2e_session.py"),
            "grm_det1_3_snapshot": file_record(
                ROOT / "scripts" / "grm_det1_3_snapshot.py"),
        },
    }


def summarize(receipts: Sequence[Path]) -> dict[str, Any]:
    """G2 reproduction table + G3 10-pair table from the per-pair receipts."""
    results = []
    for path in receipts:
        payload = _read(Path(path))
        results.append({**payload["result"], "_receipt": str(path)})

    g2_rows = []
    for row in results:
        served = (row["arms"].get("served") or {}).get("arm0") or {}
        planted = (row["arms"].get("planted_miss") or {}).get("arm0") or {}
        g2_rows.append({
            "pair": row["fixture_id"],
            "session_resume_path": row["session_resume_path"],
            "probe_node_index": row["probe_node_index"],
            "served_control_reproduced": served.get("reproduced"),
            "served_control_text_normalized_match": served.get(
                "text_normalized_match"),
            "served_control_text_bytes_match": served.get("text_bytes_match"),
            "served_control_lived_answer": served.get("lived_answer"),
            "served_control_fork_answer": served.get("fork_answer"),
            "served_control_lived_min_mass": served.get("lived_min_mass"),
            "served_control_fork_min_mass": served.get("fork_min_mass"),
            "served_control_mass_delta": served.get("min_mass_delta"),
            "planted_miss_reproduced": planted.get("reproduced"),
            "planted_miss_text_normalized_match": planted.get(
                "text_normalized_match"),
            "planted_miss_lived_answer": planted.get("lived_answer"),
            "planted_miss_fork_answer": planted.get("fork_answer"),
            "planted_miss_lived_min_mass": planted.get("lived_min_mass"),
            "planted_miss_fork_min_mass": planted.get("fork_min_mass"),
            "planted_miss_mass_delta": planted.get("min_mass_delta"),
            "pair_reproduced": row["arm0_reproduced"],
            "index_rank1_match": (row.get("index_fidelity_check") or {}).get(
                "rank1_match"),
            "index_prefix_match": (row.get("index_fidelity_check") or {}).get(
                "prefix_match"),
        })
    reproduced_count = sum(1 for r in g2_rows if r["pair_reproduced"])
    g2 = {
        "schema": f"{SCHEMA_PREFIX}.g2_reproduction.v1",
        "gate": "G2",
        "pair_count": len(g2_rows),
        "reproduced": reproduced_count,
        "prediction": ">= 6/7 pairs reproduce BOTH rows",
        "gate_pass": bool(reproduced_count >= 6),
        "not_reproduced": [
            {
                "pair": r["pair"],
                "served_control_reproduced": r["served_control_reproduced"],
                "planted_miss_reproduced": r["planted_miss_reproduced"],
                "served_control_mass_delta": r["served_control_mass_delta"],
                "planted_miss_mass_delta": r["planted_miss_mass_delta"],
            }
            for r in g2_rows if not r["pair_reproduced"]
        ],
        "not_reproduced_policy": (
            "A pair that does not reproduce is REPORTED and EXCLUDED from "
            "Arm 1. It is never patched into reproduction."),
        "table": g2_rows,
    }

    sc1_1 = _read(sc12.SC1_1_G2_SUMMARY)
    g3 = sc12.ten_pair_table(results, sc1_1)
    g3["gate"] = "G3"
    g3["sc1_1_source"] = file_record(sc12.SC1_1_G2_SUMMARY)
    g3["predictions"] = {
        "detection_recall": ">= 0.9 over the reproduced planted misses",
        "false_fires_on_served_controls": "<= 1",
        "recovery": ">= 5 of the reproduced planted misses (REPORTED)",
    }
    return {
        "schema": f"{SCHEMA_PREFIX}.summary.v1",
        "order": file_record(ORDER),
        "registration": file_record(REGISTRATION),
        "G2_arm0_reproduction": g2,
        "G3_arm1_recovery_ten_pairs": g3,
        "pair_receipts": [file_record(Path(p)) for p in receipts],
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
    parser.add_argument("command", choices=("plan", "pair", "summarize"))
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
    if args.command == "plan":
        inst = sc12.read_jsonl(
            _shard_session(sc12.INSTRUMENTATION_SHARD) / "instrumentation.jsonl")
        source_manifest = _read(
            _shard_session(sc12.SOURCE_SHARD) / "repository" / "manifest.json")
        counts = sc12.probe_node_counts(inst, len(source_manifest["nodes"]))
        pairs = sc12.e2e_pairs(node_counts=counts)
        print(json.dumps({
            "pair_count": len(pairs),
            "probe_node_counts": {str(k): v for k, v in counts.items()},
            "pairs": [
                {
                    "fixture_id": p["fixture_id"],
                    "conversation_turn": p["conversation_turn"],
                    "probe_node_index": p["probe_node_index"],
                    "plant_alias_ids": p["plant_alias_ids"],
                    "shard": p["shard"],
                    "no_fold_ids": p["no_fold_ids"],
                    "no_fold_pinned": p["no_fold_pinned"],
                    "snapshot_exists": (ROOT / p["snapshot_manifest"]).is_file(),
                }
                for p in pairs
            ],
        }, indent=1))
        return 0

    if args.command == "summarize":
        payload = summarize([Path(p) for p in args.receipts])
        path = emit(payload, "sc1_2_summary")
        g2 = payload["G2_arm0_reproduction"]
        g3 = payload["G3_arm1_recovery_ten_pairs"]
        print(f"receipt={path}")
        print(f"G2_reproduced={g2['reproduced']}/{g2['pair_count']} "
              f"gate_pass={g2['gate_pass']}")
        for row in g2["table"]:
            print("G2 " + json.dumps(row))
        print(f"G3_recall={g3['detection_recall']} "
              f"fp={g3['false_fires_on_served_controls']} "
              f"recovered={g3['recovered']}/{g3['measured_positives']} "
              f"gate_pass={g3['gate_pass']}")
        for row in g3["table"]:
            print("G3 " + json.dumps(row))
        return 0 if (g2["gate_pass"] and g3["gate_pass"]) else 1

    if not args.fixture_id:
        raise SC12Error("--fixture-id is required for the pair command")
    from scripts.grm_cmc1_gpu_arms import gpu_lease

    # The >= 30 s inter-process gap, taken INSIDE the wrapper BEFORE the
    # lease is acquired -- exactly as SC1.1 did, so a waiting process never
    # holds the lock while it sleeps.
    if int(args.gap_seconds) > 0:
        time.sleep(int(args.gap_seconds))
    with gpu_lease(int(args.lease_seconds), int(args.lock_wait_seconds)):
        payload = run_pair(str(args.fixture_id))
    path = emit(payload, f"sc1_2_pair_{args.fixture_id}")
    result = payload["result"]
    print(f"receipt={path}")
    print(json.dumps({
        "fixture_id": result["fixture_id"],
        "probe_node_index": result["probe_node_index"],
        "index_rank1_match": result["index_fidelity_check"]["rank1_match"],
        "index_prefix_match": result["index_fidelity_check"]["prefix_match"],
        "arm0_reproduced": result["arm0_reproduced"],
    }))
    for variant, arm in result["arms"].items():
        arm0, arm1 = arm["arm0"], arm["arm1"]
        print(json.dumps({
            "variant": variant,
            "arm0_reproduced": arm0["reproduced"],
            "arm0_text_normalized_match": arm0["text_normalized_match"],
            "arm0_lived_answer": arm0["lived_answer"],
            "arm0_fork_answer": arm0["fork_answer"],
            "arm0_lived_min_mass": arm0["lived_min_mass"],
            "arm0_fork_min_mass": arm0["fork_min_mass"],
            "arm0_mass_delta": arm0["min_mass_delta"],
            "arm1_fired": arm1["demand_fired"],
            "arm1_token_index": arm1["demand_token_index"],
            "arm1_min_mass": arm1["attempt_min_mass"],
            "arm1_fetched": (arm1.get("demand_trip") or {}).get("demand_fetched"),
            "arm1_served_from": arm1.get("demand_served"),
            "arm1_served_answer": arm1.get("served_answer"),
            "arm1_recovered": arm1.get("recovered"),
        }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
