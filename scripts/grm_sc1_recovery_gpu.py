#!/usr/bin/env python3
"""GRM-SC1 G3 — recovery on the race's planted misses, demand loop ON.

ORDER: ``orders/GRM_SC1_DEMAND_LOOP_NGH.md`` (gate G3).

THE QUESTION THIS ANSWERS.  The race proved D-NGH NOTICES a planted miss
(recall 1.00, FP 0.10 at ten pairs).  Noticing is not fetching.  G3 asks the
next question: when the detector fires on a planted miss, does the ONE
registered demand trip actually END the turn with the expected value?

  * detection recall on planted misses  -- prediction >= 0.9 (race 1.00)
  * false fires on served controls      -- prediction <= 1 of 10 (race 0.10)
  * RECOVERY (planted-miss turn serves the expected value via the trip)
    -- prediction >= 5/10, REPORTED not gated; it is the number David flips
    on.  PASS RULE for the mechanism = recall >= 0.9 AND false fires <= 1.

FORK-FROM-SNAPSHOT, NEVER REBUILD.  The planted miss is not re-derived here.
The registered plant target comes from the FROZEN DET1 race rows, and the
miss is produced the way the race produced it: capture the lived prefill
snapshot at the probe boundary, then ``restore_prefill_fork`` with the
registered alias seats WITHHELD from the mounted KV.  The withheld node stays
in the repository INDEX -- that is what makes recovery possible at all, and
what a rebuild would destroy.

WHY THE DET1.3 BOUNDARY IS FINE HERE.  P2A's G2 was BLOCKED because the
DET1.3 snapshot captures MOUNTED PAYLOAD ONLY at a boundary AFTER
routing/admission/fit, so a fork could not exercise the FIT stage.  SC1 is
not the fit stage: the demand loop acts at GENERATION time, strictly after
admission, and its trip re-routes through the live in-process repository
index that the same-process fork protocol keeps intact.  The boundary the
snapshot draws is therefore upstream of everything this gate measures.

PAIRS.  Each fixture contributes BOTH arms, exactly as the race scored them:
``served`` (the control -- a fire here is a FALSE fire) and ``planted_miss``
(the positive -- a fire here is a true positive, and the recovery question
follows).  Both arms fork from the SAME lived snapshot, so the only
difference between them is the withheld seats.

VALUE COMPARISON is the DET1 semantic comparator with its negative guards
(``contains_value`` over expected values, minus stale/wrong-fact values).

GPU DISCIPLINE (house rules, enforced here).
  * self-lease on /tmp/forge-gpu.lock via scripts.grm_cmc1_gpu_arms.gpu_lease;
  * ONE session per process, ONE lease per process, <= 580 s;
  * the caller inserts the >= 30 s inter-process gap;
  * the operator has absolute right of way.
"""
from __future__ import annotations

import argparse
import hashlib
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
from scripts.grm_cmc1_mechanism import (  # noqa: E402
    canonical_json_bytes, sha256_bytes,
)
from scripts.grm_det1_common import contains_value, file_record  # noqa: E402

ARTIFACT_DIR = ROOT / "artifacts" / "grm_sc1"
#: SC1.1 receipts land in their OWN directory; SC1's are frozen evidence.
SC1_1_ARTIFACT_DIR = ROOT / "artifacts" / "grm_sc1_1"
SCHEMA_PREFIX = "grm.sc1"
FROZEN_RUN = ROOT / "artifacts" / "grm_det1" / "run_20260831T160525Z_2"
RUNTIME_FRAME = FROZEN_RUN / "runtime_frame_28b3196f8fb04a41.json"
REGISTRATION = ARTIFACT_DIR / "grm_sc1_registration.json"
ORDER = ROOT / "orders" / "GRM_SC1_DEMAND_LOOP_NGH.md"
SC1_1_ORDER = ROOT / "orders" / "GRM_SC1_1_GROUNDING_GLYPHS.md"
SC1_1_REGISTRATION = SC1_1_ARTIFACT_DIR / "registration.json"
SUP_FIXTURES = ROOT / "tests" / "fixtures" / "supersession_battery"
EVAL_ROWS = FROZEN_RUN / "det1_4" / "campaign" / "eval" / "mechanistic_rows.jsonl"
#: The FROZEN race registration. Read-only here; it carries the model
#: snapshot inventory ``_frame_identity`` binds the arena frame against, so
#: the fork identity check is the race's own check, not a looser one.
DET1_REGISTRATION = FROZEN_RUN / "registration_62cb6c09cbec211d.json"

#: The supersession sessions, in frozen DET1.5 spec order (sup-1..sup-4).
SUP_SESSIONS = (
    "correction_then_restatement",        # sup-1
    "fresh_fact_controls",                # sup-2
    "multi_hop_a_b_c",                    # sup-3
    "short_correction_long_competitor",   # sup-4
)
LEASE_SECONDS = 580
LOCK_WAIT_SECONDS = 7200


class SC1Error(RuntimeError):
    """The SC1 recovery gate could not be performed as registered."""


def _read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in Path(path).read_text(
            encoding="utf-8").splitlines() if line.strip()
    ]


def race_rows() -> dict[str, dict[str, Any]]:
    """The race's own eval rows, keyed ``fixture_id:variant``.

    Read-only.  They supply the registered plant alias ids and the race's
    per-fixture numbers this gate is compared against.
    """
    return {str(row["row_id"]): dict(row) for row in _read_jsonl(EVAL_ROWS)}


def sup_eval_fixtures(session_id: str) -> list[dict[str, Any]]:
    """Every registered EVAL pair this supersession session carries.

    The plant alias ids are taken from the race's own rows -- the registered
    plant, never a fresh selection.
    """
    rows = race_rows()
    fixture = _read(SUP_FIXTURES / f"{session_id}.json")
    out: list[dict[str, Any]] = []
    for probe in fixture["probes"]:
        fixture_id = f"sup_{probe['probe_id']}"
        planted = rows.get(f"{fixture_id}:planted_miss")
        served = rows.get(f"{fixture_id}:served")
        if planted is None or served is None:
            continue
        out.append({
            "fixture_id": fixture_id,
            "session_id": session_id,
            "probe_id": str(probe["probe_id"]),
            "question": str(probe["question"]),
            "expected_values": [
                str(v) for v in probe.get("expected_values", ())],
            "rejected_values": [
                *[str(v) for v in probe.get("stale_values", ())],
                *[str(v) for v in probe.get("wrong_fact_values", ())],
            ],
            "plant_alias_ids": [
                int(v) for v in planted.get("plant_alias_ids", ())],
            "plant_target_id": planted.get("plant_target_id"),
            "race_planted_answer": str(planted.get("answer", "")),
            "race_planted_correct": bool(planted.get("answer_correct")),
            "race_served_answer": str(served.get("answer", "")),
            "race_served_correct": bool(served.get("answer_correct")),
            "race_planted_min_mass": min(
                (float(t["ngh"]["mounted_mass"])
                 for t in (planted.get("signals") or {}).get("tokens", ())),
                default=None),
            "race_served_min_mass": min(
                (float(t["ngh"]["mounted_mass"])
                 for t in (served.get("signals") or {}).get("tokens", ())),
                default=None),
        })
    return out


def value_verdict(
    answer: str,
    *,
    expected_values: Sequence[str],
    rejected_values: Sequence[str],
) -> dict[str, Any]:
    """The DET1 unified semantic comparator, with its negative guards."""
    hit = [v for v in expected_values if contains_value(answer, v)]
    bad = [v for v in rejected_values if contains_value(answer, v)]
    return {
        "expected_hits": hit,
        "rejected_hits": bad,
        "correct": bool(hit and not bad),
    }


def _demand_trip_after_fork(
    *, arena, e2e, question, threshold, ngen, rows, mounted_now, live_ids,
    route_limit, topk,
):
    """The production demand trip, run on a forked (post-fork) arena.

    This is the SAME sequence ``core.grm_demand`` drives inside ``step()``
    and ``_probe_ladder_chat``, expressed against the forked arena that the
    withholding produced: decide, ROLL BACK, rebuild the query from the
    model's own partial output before the fire token, re-route excluding what
    is already mounted or live, fit, generate once, judge grounding.  Cap 1:
    no second trip is ever taken, and a refire is recorded only.

    THE ROLLBACK IS THE CLEAN-ROOM RESET BELOW, not a re-fork.  Two things
    were measured getting here and both are recorded because they shaped the
    design:

      1. Generating the trip ON TOP of the failed attempt's cache does not
         work.  On sup_praxis_fresh the trip re-routed correctly and mounted
         the withheld node, and the model STILL refused, because the refusal
         it was meant to replace was sitting in context in front of the new
         mount (trip mounted_mass 0.198 against the served control's 0.353).
      2. Re-forking as the rollback does not work either: the fork arms
         ONE-SHOT allowed-masks that the very next forward must consume, and
         the trip's routing probe consumes them first ("DET1 fork allowed-mask
         does not match the next forward").

    The clean-room reset is what production rung 0 already does for an
    identifier probe, and it is the honest pre-attempt state: mounts cleared,
    live window drained, nothing of the failed attempt carried forward.
    """
    from core.grm_admission import decisive_admission_profile

    decision = grm_demand.decide(rows, float(threshold))
    if not decision["demand_fired"]:
        return decision, None
    prefix = grm_demand.demand_prefix_text(
        arena, rows, int(decision["demand_token_index"]))
    query = grm_demand.demand_query_text(question, prefix)
    exclude = set(int(v) for v in live_ids) | {int(v) for v in mounted_now}
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
    # CLEAN-ROOM, exactly as the production rung 0 for an identifier probe is
    # clean-room: the pre-attempt state the trip must start from is the one
    # BEFORE the failed attempt's prompt entered the cache. Leaving the failed
    # prefill in front of the new mount is what produced the measured
    # refusal-with-the-node-mounted (mounted_mass 0.198 vs the control's
    # 0.353): the answer the trip was replacing was still in context.
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
    # SC1.1: the glyph receipt for the verdict that blocked recovery 2/2 in
    # SC1. Both verdicts are computed (pure set arithmetic over already
    # generated text), so the receipt can say WHETHER the projection is what
    # let the trip serve, instead of leaving the reader to infer it.
    grounding_fields: dict[str, Any] = {}
    grounded, contributors = arena._grounding_attribution(
        answer, mounted_after, question)
    arena._grounding_receipt(
        answer, mounted_after, question, grounding_fields)
    legacy_grounded, _legacy_contributors = arena._grounding_verdict(
        answer, mounted_after, question, normalized=False)
    normalized_grounded, _norm_contributors = arena._grounding_verdict(
        answer, mounted_after, question, normalized=True)
    trip.update({
        "demand_trip_grounding_normalized": bool(
            grounding_fields.get("grounding_normalized")),
        "demand_trip_grounding_glyph_rescued": bool(
            grounding_fields.get("grounding_glyph_rescued")),
        "demand_trip_grounded_legacy": bool(legacy_grounded),
        "demand_trip_grounded_normalized": bool(normalized_grounded),
    })
    trip.update({
        "demand_trip_taken": True,
        "demand_trip_answer": str(answer),
        "demand_trip_mounted_ids": mounted_after,
        "demand_trip_grounded": bool(grounded),
        "demand_trip_grounding_contributors": sorted(
            int(v) for v in (contributors or ())),
        "demand_trip_mounted_texts": {
            str(i): str(arena.grafts[int(i)].get("text", ""))[:400]
            for i in mounted_after
        },
        "demand_trip_question_rare_tokens": sorted(
            str(t) for t in arena._rare_tokens(question)),
        "demand_trip_answer_rare_tokens": sorted(
            str(t) for t in arena._rare_tokens(str(answer))),
        "demand_trip_min_mass": refire["demand_min_mass"],
        "demand_refired": bool(refire["demand_fired"]),
        # NEVER LOOP: recorded, not acted on. Cap 1 is registered.
        "demand_refire_acted_on": False,
        "demand_no_candidate": False,
    })
    return decision, trip


def _measure_pair(
    *, repo, e2e, fixture, threshold, flags, snapshot_root, identity, process,
):
    """Fork BOTH arms of one registered pair from ONE lived snapshot."""
    from scripts.grm_det1_3_snapshot import (
        ARM_PROTOCOLS,
        finalize_snapshot_with_answer,
        load_snapshot,
        require_snapshot_member_coverage,
        restore_prefill_fork,
        stop_at_next_forward,
    )
    from scripts.grm_det1_4_gpu import _continue_forked_prefill
    from scripts.grm_det1_common import route_fixture_profile
    from scripts.grm_det1_e2e import (
        _restore_counterfactual, _snapshot_counterfactual,
    )
    from scripts.grm_det1_5_gpu import (
        _admission_snapshot, _attempt_schedule, _live_token_ids, _sha_array,
    )

    arena = repo.arena
    question = str(fixture["question"])
    ngen = int(flags["ngen"])
    topk = int(flags["topk"])
    max_trips = int(flags["max_trips"])
    route_limit = max(topk, (max_trips + 1) * topk)
    aliases = {int(v) for v in fixture["plant_alias_ids"]}

    live = {int(g) for g, _n in arena.live_segs if g is not None}
    base = _snapshot_counterfactual(arena)
    try:
        profile = route_fixture_profile(
            arena, question,
            (fixture["expected_values"] or [""])[0],
            live_excluded=live,
            route_limit=route_limit,
        )
    finally:
        _restore_counterfactual(arena, base)
    admission = profile.get("served_admission_profile") or {}
    planned = [int(v) for v in admission.get(
        "rank_plan", profile.get("raw_ranking", ()))]
    fitted = sorted(int(v) for v in e2e._budget_fit_mounts(arena, planned))
    if not fitted:
        raise SC1Error(f"no mount set fits for {fixture['fixture_id']}")
    # THE RUNG THE RACE MEASURED. ``_attempt_schedule`` is the production
    # ladder builder; its rung 0 for an identifier probe is CLEAN-ROOM, and
    # ``_measure_fixture_inline`` drains the live window before a clean rung.
    # That is not cosmetic: the clean rung makes ``_attempt`` take the
    # BOOTSTRAP/injection path, which is what sets ``graft_seats`` and gives
    # the snapshot the geometry ``restore_prefill_fork`` validates. Without
    # it the arena takes the ``swap`` path and the fork is refused (measured:
    # "fork graft-seat geometry disagrees at layer 0", graft_seats 0 vs 71).
    schedule = _attempt_schedule(
        arena, e2e, question, profile, topk=topk, max_trips=max_trips)
    rung_planned, rung_clean = schedule[0]
    rung_fitted = sorted(
        int(v) for v in e2e._budget_fit_mounts(arena, rung_planned))
    if rung_fitted:
        planned = [int(v) for v in rung_planned]
        fitted = rung_fitted

    provenance = {
        "schema": "grm.sc1.capture_provenance.v1",
        "arm": "lived",
        "protocol": ARM_PROTOCOLS["lived"],
        "protocol_source_sha256": file_record(
            Path(__file__).resolve())["sha256"],
        "run_id": FROZEN_RUN.name,
        "order_sha256": file_record(ORDER)["sha256"],
        "source_amendment_sha256": file_record(REGISTRATION)["sha256"],
        "registration_sha256": file_record(REGISTRATION)["sha256"],
        "runtime_frame_sha256": file_record(RUNTIME_FRAME)["sha256"],
        "fixture_sha256": hashlib.sha256(
            canonical_json_bytes(fixture)).hexdigest(),
        "fixture_id": str(fixture["fixture_id"]),
        "probe_id": str(fixture["probe_id"]),
        "probe_question_sha256": hashlib.sha256(
            question.encode("utf-8")).hexdigest(),
        "capture_boundary": "before_probe_prefill",
        "selection_boundary": "before_generation_and_grounding_selection",
        "probe_driver": "grm_sc1.recovery_inline_fork",
        "topk": topk,
        "max_trips": max_trips,
        "probe_ladder": True,
        "defer_memory": True,
        "attempt_ordinal": 0,
        **dict(process),
    }

    _restore_counterfactual(arena, base)
    if rung_clean:
        # RECENCY LAW, the lived instrument's own line: point lookups exclude
        # live/recency seats, and the drained live window is what puts the
        # attempt on the bootstrap path.
        arena.caches, arena.pos, arena.live_segs = None, 0, []
        arena.cur_mounts, arena.cur_mount_n = [], 0
    for layer in arena.m.layers:
        layer.self_attn.live_shift = arena.live_shift
    live_ids = _live_token_ids(arena)
    probe_key_sha256 = _sha_array(arena._probe_key(question))
    arms: dict[str, Any] = {}
    source_mounts_box: dict[str, Any] = {}

    def fork_callback(manifest_path, _prompt_ids, restore_fn):
        """Both arms, forked AT the captured prefill boundary.

        This is the race's own protocol and the only place the fork is
        geometrically valid: ``stop_at_next_forward`` hands control here with
        the arena still at the prefill state the snapshot describes, and
        restores the canonical zero fork afterwards so the lived prefill runs
        exactly once.  Forking after the lived answer instead would fail the
        graft-seat geometry check -- correctly, because the cache has moved.
        """
        snapshot_now = load_snapshot(Path(manifest_path))
        source_mounts_local = {
            int(v) for v in
            (snapshot_now.get("state") or {}).get("arena.cur_mounts", ())
        }
        source_mounts_box["value"] = source_mounts_local
        for variant in ("served", "planted_miss"):
            withheld = (
                sorted(aliases & source_mounts_local)
                if variant == "planted_miss" else [])
            if variant == "planted_miss" and not withheld:
                # The registered plant target had no mounted seat in this
                # rung: the withholding cannot apply. That is a RESULT about
                # this pair, not something to paper over with a substitute.
                arms[variant] = {
                    "status": "PLANT_NOT_MOUNTED_IN_THIS_RUNG",
                    "withheld_alias_ids": [],
                    "source_mounts": sorted(source_mounts_local),
                    "registered_alias_ids": sorted(aliases),
                }
                continue
            restore = restore_fn(withheld)
            started = time.time_ns()
            with grm_demand.DemandObserver(
                arena, ngen, float(threshold)) as observer:
                answer, _masks = _continue_forked_prefill(
                    arena, restore["prompt_ids"], ngen)
            rows = observer.finish()
            mounted_now = [int(v) for v in arena.cur_mounts]
            verdict = value_verdict(
                answer,
                expected_values=fixture["expected_values"],
                rejected_values=fixture["rejected_values"])
            decision, trip = _demand_trip_after_fork(
                arena=arena, e2e=e2e, question=question, threshold=threshold,
                ngen=ngen, rows=rows, mounted_now=mounted_now, live_ids=live,
                route_limit=route_limit, topk=topk)
            elapsed = int(time.time_ns() - started)
            served_from = "original"
            final_answer = str(answer)
            if trip and trip.get("demand_trip_taken") and trip.get(
                    "demand_trip_grounded"):
                served_from = "demand_trip"
                final_answer = str(trip["demand_trip_answer"])
            final_verdict = value_verdict(
                final_answer,
                expected_values=fixture["expected_values"],
                rejected_values=fixture["rejected_values"])
            # COUNTERFACTUAL, reported alongside the gated number and never
            # substituted for it. The order's rule is "serve the trip IFF it
            # passes GROUNDING", and grounding is arena grounding-v3, not the
            # DET1 value comparator. The two disagree on the SEPARATOR
            # ARTIFACT class the DET1.11 census already names: the model
            # emits U+2011 non-breaking hyphens ("Quartz-8-Jade"), the DET1
            # comparator normalizes them and scores the answer CORRECT, and
            # grounding-v3's rare-token overlap does not and rejects it.
            # This field says what the turn WOULD have served if grounding
            # agreed with the comparator, so the gap is visible instead of
            # being buried in a False.
            trip_answer = (trip or {}).get("demand_trip_answer")
            trip_verdict = (
                value_verdict(
                    str(trip_answer),
                    expected_values=fixture["expected_values"],
                    rejected_values=fixture["rejected_values"])
                if trip_answer is not None else None)
            arms[variant] = {
                "status": "MEASURED",
                "withheld_alias_ids": withheld,
                "registered_alias_ids": sorted(aliases),
                "attempt_answer": str(answer),
                "attempt_verdict": verdict,
                "attempt_mounted_ids": mounted_now,
                "attempt_min_mass": decision["demand_min_mass"],
                "attempt_token_count": decision["demand_token_count"],
                "demand_fired": bool(decision["demand_fired"]),
                "demand_token_index": decision["demand_token_index"],
                "demand_threshold": float(threshold),
                "demand_trip": trip,
                "demand_served": served_from,
                "served_answer": final_answer,
                "served_verdict": final_verdict,
                # RECOVERY is exactly this: a planted-miss turn that ends
                # serving the expected value BECAUSE the trip went out.
                "recovered": bool(
                    variant == "planted_miss"
                    and served_from == "demand_trip"
                    and final_verdict["correct"]),
                "trip_answer_verdict": trip_verdict,
                # The trip FETCHED the right node and the model READ it: the
                # trip's own answer carries the expected value. Only the
                # grounding gate stopped it from being served.
                "recovery_blocked_by_grounding_only": bool(
                    variant == "planted_miss"
                    and served_from == "original"
                    and trip_verdict is not None
                    and trip_verdict["correct"]
                    and not (trip or {}).get("demand_trip_grounded")),
                "elapsed_ns": elapsed,
                "same_process_index_verified": bool(
                    restore.get("same_process_index_verified")),
            }
        return {"arms": sorted(arms)}

    with stop_at_next_forward(
        arena,
        snapshot_root / str(fixture["fixture_id"]) / "rung_00",
        label="lived",
        provenance=lambda: provenance,
        fork_callback=fork_callback,
        question=question,
        # The FROZEN race's own admission-snapshot builder, imported not
        # reimplemented: the coverage validator is strict about this schema,
        # and a hand-rolled shape is exactly how a fork silently stops being
        # the race's fork.
        admission_plan=lambda: _admission_snapshot(
            arena=arena,
            profile=profile,
            schedule=schedule,
            ordinal=0,
            planned=planned,
            fitted=fitted,
            probe_key_sha256=probe_key_sha256,
        ),
        live_token_ids=live_ids,
        sink_text=e2e.HARMONY_SINK,
        sink_token_ids=arena.encode(e2e.HARMONY_SINK),
        explicit_identity=identity,
    ) as captured:
        lived_answer, _lived_info = arena._attempt(
            question, fitted, ngen, False,
            arena.stop_sequences or (), defer_memory=True)
    from scripts.grm_det1_3_gpu import _is_refusal

    lived_correct = value_verdict(
        str(lived_answer),
        expected_values=fixture["expected_values"],
        rejected_values=fixture["rejected_values"])["correct"]
    manifest_path = finalize_snapshot_with_answer(
        Path(captured["manifest_path"]),
        {
            "schema": "grm.sc1.inline_lived_answer.v1",
            "process_instance_sha256": process["process_instance_sha256"],
            "captured_attempt_ordinal": 0,
            "attempt_answer": str(lived_answer),
            "attempt_mounts": [int(v) for v in arena.cur_mounts],
            "attempt_answer_correct": bool(lived_correct),
            "attempt_refusal": bool(_is_refusal(str(lived_answer))),
            "probe_answer": str(lived_answer),
            "probe_selected_attempt": 0,
            "probe_mounts": [int(v) for v in arena.cur_mounts],
            "probe_answer_correct": bool(lived_correct),
            "probe_refusal": bool(_is_refusal(str(lived_answer))),
        },
    )
    snapshot = load_snapshot(manifest_path)
    require_snapshot_member_coverage(
        snapshot, f"{fixture['fixture_id']} lived source")
    source_mounts = set(source_mounts_box.get("value") or set())
    if not arms:
        raise SC1Error(
            f"fork callback never ran for {fixture['fixture_id']}")
    _restore_counterfactual(arena, base)
    return {
        "fixture_id": str(fixture["fixture_id"]),
        "session_id": str(fixture["session_id"]),
        "question": question,
        "expected_values": list(fixture["expected_values"]),
        "rejected_values": list(fixture["rejected_values"]),
        "plant_alias_ids": sorted(aliases),
        "plant_target_id": fixture.get("plant_target_id"),
        "plant_source": "FROZEN_DET1_RACE_EVAL_ROWS",
        "lived_answer": str(lived_answer),
        "lived_mounted_ids": sorted(source_mounts),
        "snapshot": file_record(manifest_path),
        "race": {
            "planted_answer": fixture["race_planted_answer"],
            "planted_correct": fixture["race_planted_correct"],
            "planted_min_mass": fixture["race_planted_min_mass"],
            "served_answer": fixture["race_served_answer"],
            "served_correct": fixture["race_served_correct"],
            "served_min_mass": fixture["race_served_min_mass"],
        },
        "arms": arms,
    }


def run_session(session_id: str) -> dict[str, Any]:
    """Install one lived session and measure every registered pair in it."""
    from scripts.grm_det1_2_gpu import _load_model_repo
    from scripts.grm_det1_3_gpu import _install_lived_nodes, _frame_identity
    from scripts.grm_det1_5_gpu import _process_instance

    frame = _read(RUNTIME_FRAME)
    flags = frame["resolved_flags"]
    threshold = grm_demand.registered_threshold()
    registered = grm_demand.load_registered()
    fixtures = sup_eval_fixtures(session_id)
    if not fixtures:
        raise SC1Error(f"no registered eval pair bound to {session_id}")
    fixture_source = _read(SUP_FIXTURES / f"{session_id}.json")

    # The demand loop is ON for this gate. Set it explicitly rather than
    # relying on inheritance, so the receipt can state what ran.
    os.environ["GRM_DEMAND_NGH"] = "1"
    # SC1.1: the fixes switch is what gates the glyph projection. G2 is the
    # fixes-ON arm, stated explicitly for the same reason.
    os.environ["GRM_LSR_FIXES"] = "1"

    repo_dir = Path(tempfile.mkdtemp(prefix=f"sc1_g3_{session_id}_"))
    snapshot_root = repo_dir / "snapshots"
    repo = model = tokenizer = None
    model_info: Any = None
    pairs: list[dict[str, Any]] = []
    try:
        e2e, model, tokenizer, repo, model_info = _load_model_repo(
            repo_dir, frame)
        node_to_idx, _ledgers = _install_lived_nodes(repo, e2e, fixture_source)
        identity = _frame_identity(
            model, tokenizer, e2e, _read(DET1_REGISTRATION), frame)
        process = _process_instance()
        for fixture in fixtures:
            pairs.append(_measure_pair(
                repo=repo, e2e=e2e, fixture=fixture, threshold=threshold,
                flags=flags, snapshot_root=snapshot_root, identity=identity,
                process=process))
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

    return {
        "schema": f"{SCHEMA_PREFIX}.g3_recovery_session.v1",
        "gate": "G3",
        "session_id": session_id,
        "order": file_record(ORDER),
        "registration": file_record(REGISTRATION),
        "sc1_1_order": file_record(SC1_1_ORDER),
        "sc1_1_registration": file_record(SC1_1_REGISTRATION),
        "lsr_fixes_env": os.environ.get("GRM_LSR_FIXES"),
        "demand_flag": os.environ.get("GRM_DEMAND_NGH"),
        "carried_threshold": float(threshold),
        "carried_threshold_caveat": str(registered["caveat"]),
        "carried_threshold_provenance": {
            "path": registered["provenance"]["thresholds_path"],
            "sha256": registered["provenance"]["thresholds_sha256"],
            "fit_turn_count": int(registered["fit_turn_count"]),
        },
        "fork_protocol": (
            "SAME-PROCESS lived capture then restore_prefill_fork with the "
            "REGISTERED plant alias seats withheld; the withheld node remains "
            "in the repository index, which is what makes recovery possible. "
            "Fork-from-snapshot, never rebuild."),
        "det1_3_boundary_note": (
            "The DET1.3 snapshot boundary sits AFTER routing/admission/fit. "
            "That blocked P2A's fit-stage gate but is FINE here: the demand "
            "loop acts at GENERATION time, strictly after admission."),
        "fixture_node_to_idx": {str(k): int(v) for k, v in node_to_idx.items()},
        "pairs": pairs,
        "runtime_frame": file_record(RUNTIME_FRAME),
        "model": model_info,
        "sources": {
            "grm_demand": file_record(ROOT / "core" / "grm_demand.py"),
            "graft_arena": file_record(ROOT / "core" / "graft_arena.py"),
            "grm_e2e_session": file_record(
                ROOT / "scripts" / "grm_e2e_session.py"),
            "grm_sc1_recovery_gpu": file_record(Path(__file__).resolve()),
            "grm_text_norm": file_record(ROOT / "core" / "grm_text_norm.py"),
        },
    }


def summarize(session_receipts: Sequence[Path]) -> dict[str, Any]:
    """Roll the per-session receipts into the G3 verdict."""
    rows: list[dict[str, Any]] = []
    for path in session_receipts:
        payload = _read(Path(path))
        for pair in payload["pairs"]:
            rows.append({**pair, "_receipt": str(path)})
    positives = [
        r for r in rows
        if (r["arms"].get("planted_miss") or {}).get("status") == "MEASURED"]
    negatives = [
        r for r in rows
        if (r["arms"].get("served") or {}).get("status") == "MEASURED"]
    tp = sum(1 for r in positives if r["arms"]["planted_miss"]["demand_fired"])
    fp = sum(1 for r in negatives if r["arms"]["served"]["demand_fired"])
    recovered = sum(1 for r in positives if r["arms"]["planted_miss"]["recovered"])
    blocked = sum(
        1 for r in positives
        if r["arms"]["planted_miss"].get("recovery_blocked_by_grounding_only"))
    recall = (tp / len(positives)) if positives else None
    fpr = (fp / len(negatives)) if negatives else None
    # PASS RULE, registered before the run: recall >= 0.9 AND false fires <= 1.
    # Recovery is REPORTED, never gated.
    gate_pass = bool(
        recall is not None and recall >= 0.9 and fp <= 1)
    return {
        "schema": f"{SCHEMA_PREFIX}.g3_summary.v1",
        # SC1.1 extends this receipt ADDITIVELY: every SC1 field keeps its
        # name, type and meaning, and the glyph block is new alongside them.
        "schema_extension": "grm.sc1_1.grounding_glyph_receipts.v1",
        "gate": "G3",
        "pair_count": len(rows),
        "measured_positives": len(positives),
        "measured_negatives": len(negatives),
        "true_positives": tp,
        "false_fires_on_served_controls": fp,
        "detection_recall": recall,
        "false_positive_rate": fpr,
        "recovered": recovered,
        "recovery_rate": (recovered / len(positives)) if positives else None,
        # REPORTED, never substituted for `recovered`. These are planted-miss
        # turns whose demand trip fetched the right node and produced the
        # EXPECTED VALUE, and which served the original anyway because arena
        # grounding-v3 rejected the answer on the SEPARATOR ARTIFACT class
        # (U+2011 non-breaking hyphens) that the DET1 value comparator
        # normalizes and grounding does not. The mechanism worked; the gate
        # in front of it disagreed with the comparator.
        "recovery_blocked_by_grounding_only": blocked,
        "recovery_if_grounding_matched_the_comparator": recovered + blocked,
        # SC1.1: how many planted-miss trips grounded ONLY because the glyph
        # projection landed. Registered prediction: 2 of 2.
        "recovery_glyph_rescued": sum(
            1 for r in positives
            if ((r["arms"]["planted_miss"].get("demand_trip") or {}).get(
                "demand_trip_grounding_glyph_rescued"))),
        "grounding_separator_artifact_note": (
            "Pre-existing class, named in the DET1.11 census "
            "(separator_artifact_rescued). NOT introduced by SC1 and NOT "
            "patched here: grounding-v3 is outside this order's file "
            "boundary and changing it would be a silent law change."),
        "pass_rule": "recall >= 0.9 AND false_fires <= 1",
        "gate_pass": gate_pass,
        "recovery_is_reported_not_gated": True,
        "predictions": {
            "recall": ">= 0.9 (race 1.00)",
            "false_fires": "<= 1 of 10 (race 0.10)",
            "recovery": ">= 5/10 (REPORTED, not gated)",
        },
        "table": [
            {
                "fixture_id": r["fixture_id"],
                "session_id": r["session_id"],
                "plant_alias_ids": r["plant_alias_ids"],
                "planted_fired": (r["arms"].get("planted_miss") or {}).get(
                    "demand_fired"),
                "planted_token_index": (r["arms"].get("planted_miss") or {}).get(
                    "demand_token_index"),
                "planted_min_mass": (r["arms"].get("planted_miss") or {}).get(
                    "attempt_min_mass"),
                "planted_fetched": ((r["arms"].get("planted_miss") or {}).get(
                    "demand_trip") or {}).get("demand_fetched"),
                "planted_served_from": (r["arms"].get("planted_miss") or {}).get(
                    "demand_served"),
                "planted_served_answer": (r["arms"].get("planted_miss") or {}).get(
                    "served_answer"),
                "recovered": (r["arms"].get("planted_miss") or {}).get(
                    "recovered"),
                "planted_trip_answer": ((r["arms"].get("planted_miss") or {}).get(
                    "demand_trip") or {}).get("demand_trip_answer"),
                "planted_trip_grounded": ((r["arms"].get("planted_miss") or {}).get(
                    "demand_trip") or {}).get("demand_trip_grounded"),
                # SC1.1: the glyph receipt for the trip's grounding verdict.
                "planted_trip_grounded_legacy": (
                    ((r["arms"].get("planted_miss") or {}).get(
                        "demand_trip") or {}).get(
                            "demand_trip_grounded_legacy")),
                "planted_trip_grounded_normalized": (
                    ((r["arms"].get("planted_miss") or {}).get(
                        "demand_trip") or {}).get(
                            "demand_trip_grounded_normalized")),
                "planted_trip_grounding_normalized": (
                    ((r["arms"].get("planted_miss") or {}).get(
                        "demand_trip") or {}).get(
                            "demand_trip_grounding_normalized")),
                "planted_trip_grounding_glyph_rescued": (
                    ((r["arms"].get("planted_miss") or {}).get(
                        "demand_trip") or {}).get(
                            "demand_trip_grounding_glyph_rescued")),
                "planted_trip_answer_correct": (
                    ((r["arms"].get("planted_miss") or {}).get(
                        "trip_answer_verdict") or {}).get("correct")),
                "recovery_blocked_by_grounding_only": (
                    r["arms"].get("planted_miss") or {}).get(
                        "recovery_blocked_by_grounding_only"),
                "served_fired": (r["arms"].get("served") or {}).get(
                    "demand_fired"),
                "served_min_mass": (r["arms"].get("served") or {}).get(
                    "attempt_min_mass"),
                "served_answer": (r["arms"].get("served") or {}).get(
                    "served_answer"),
                "served_correct": ((r["arms"].get("served") or {}).get(
                    "served_verdict") or {}).get("correct"),
                "race_planted_min_mass": r["race"]["planted_min_mass"],
                "race_served_min_mass": r["race"]["served_min_mass"],
            }
            for r in rows
        ],
        "session_receipts": [file_record(Path(p)) for p in session_receipts],
    }


def summarize_ten_pairs(
    sc1_1_summary_path: Path,
    sc1_2_pair_receipts: Sequence[Path],
) -> dict[str, Any]:
    """SC1.2: merge this order's 3 standalone pairs with SC1.2's 7 E2E pairs.

    This is the ONE extension SC1.2's file boundary permits in this module,
    and it is deliberately a DELEGATION: the merge rule, the exclusion rule
    and the floor rule all live in ``grm_sc1_2_session.ten_pair_table``, so
    there is exactly one implementation of "how the ten pairs are counted"
    rather than two that can drift.

    The three SC1/SC1.1 rows are read from the FROZEN SC1.1 summary, never
    re-measured here: they are already evidence.
    """
    from scripts import grm_sc1_2_session as sc12

    results = [
        _read(Path(path))["result"] for path in sc1_2_pair_receipts]
    table = sc12.ten_pair_table(results, _read(Path(sc1_1_summary_path)))
    table["gate"] = "G3"
    table["sc1_1_source"] = file_record(Path(sc1_1_summary_path))
    table["sc1_2_pair_receipts"] = [
        file_record(Path(p)) for p in sc1_2_pair_receipts]
    table["merge_owner"] = (
        "grm_sc1_2_session.ten_pair_table -- one counting rule, delegated to, "
        "never duplicated here")
    return table


def emit(payload: Mapping[str, Any], stem: str) -> Path:
    # SC1.1 stems are written under artifacts/grm_sc1_1/ so this order's
    # receipts never mix with the frozen SC1 evidence they are compared to.
    if stem.startswith("sc1_2_"):
        out_dir = ROOT / "artifacts" / "grm_sc1_2"
    elif stem.startswith("sc1_1_"):
        out_dir = SC1_1_ARTIFACT_DIR
    else:
        out_dir = ARTIFACT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    body = canonical_json_bytes(payload)
    digest = sha256_bytes(body)
    path = out_dir / f"{stem}_{digest[:16]}.json"
    path.write_bytes(body)
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("plan", "session", "summarize", "summarize10"))
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--receipts", nargs="*", default=())
    parser.add_argument(
        "--sc1-2-receipts", nargs="*", default=(),
        help="SC1.2 per-pair receipts for the merged 10-pair table")
    parser.add_argument("--lease-seconds", type=int, default=LEASE_SECONDS)
    parser.add_argument("--lock-wait-seconds", type=int,
                        default=LOCK_WAIT_SECONDS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "plan":
        plan = {s: sup_eval_fixtures(s) for s in SUP_SESSIONS}
        print(json.dumps({
            "sessions": list(SUP_SESSIONS),
            "pair_count": sum(len(v) for v in plan.values()),
            "pairs": {k: [f["fixture_id"] for f in v] for k, v in plan.items()},
        }, indent=1))
        return 0
    if args.command == "summarize10":
        if len(args.receipts) != 1:
            raise SC1Error(
                "summarize10 takes exactly one --receipts value: the SC1.1 "
                "3-pair summary to merge SC1.2's E2E pairs into")
        payload = summarize_ten_pairs(
            Path(args.receipts[0]),
            [Path(p) for p in args.sc1_2_receipts])
        path = emit(payload, "sc1_2_ten_pair_table")
        print(f"receipt={path}")
        print(f"rows={payload['rows_present']} "
              f"positives={payload['measured_positives']} "
              f"negatives={payload['measured_negatives']}")
        print(f"recall={payload['detection_recall']} "
              f"fp={payload['false_fires_on_served_controls']} "
              f"recovered={payload['recovered']} "
              f"gate_pass={payload['gate_pass']}")
        for row in payload["table"]:
            print(json.dumps(row))
        return 0 if payload["gate_pass"] else 1

    if args.command == "summarize":
        payload = summarize([Path(p) for p in args.receipts])
        path = emit(payload, "sc1_1_g2_summary")
        print(f"receipt={path}")
        print(f"gate_pass={payload['gate_pass']}")
        print(f"recall={payload['detection_recall']}")
        print(f"false_fires={payload['false_fires_on_served_controls']}")
        print(f"recovered={payload['recovered']}"
              f"/{payload['measured_positives']}")
        for row in payload["table"]:
            print(json.dumps(row))
        return 0 if payload["gate_pass"] else 1

    if not args.session_id:
        raise SC1Error("--session-id is required for the session command")
    from scripts.grm_cmc1_gpu_arms import gpu_lease

    with gpu_lease(int(args.lease_seconds), int(args.lock_wait_seconds)):
        payload = run_session(str(args.session_id))
    path = emit(payload, f"sc1_1_g2_{args.session_id}")
    print(f"receipt={path}")
    for pair in payload["pairs"]:
        planted = pair["arms"].get("planted_miss") or {}
        served = pair["arms"].get("served") or {}
        print(json.dumps({
            "fixture_id": pair["fixture_id"],
            "planted_status": planted.get("status"),
            "planted_fired": planted.get("demand_fired"),
            "planted_token_index": planted.get("demand_token_index"),
            "planted_min_mass": planted.get("attempt_min_mass"),
            "planted_fetched": (planted.get("demand_trip") or {}).get(
                "demand_fetched"),
            "planted_served_from": planted.get("demand_served"),
            "recovered": planted.get("recovered"),
            "served_fired": served.get("demand_fired"),
            "served_min_mass": served.get("attempt_min_mass"),
            "served_correct": (served.get("served_verdict") or {}).get(
                "correct"),
        }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
