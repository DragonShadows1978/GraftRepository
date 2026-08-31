#!/usr/bin/env python3
"""Opt-in DET1 wrapper around the certified 34-turn production driver.

With ``GRM_DET1_ENABLED`` false or unset, no callable is patched.  The wrapper
exists so DET-G1 can run the exact same entry point with instrumentation dark.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
import copy
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.grm_det1_common import (  # noqa: E402
    DetectorObserver,
    DETError,
    VERBAL_QUESTION,
    append_jsonl_once,
    contains_value,
    read_json,
    route_fixture_profile,
    verbal_result,
)


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().casefold() in ("1", "true", "yes", "on")


def _compact_info(info: Mapping[str, Any] | None) -> dict[str, Any]:
    info = dict(info or {})
    keys = (
        "trip", "clean_room", "driver_probe_ladder", "driver_topk",
        "point_lookup", "precise_first", "mount_plan", "mount_fitted",
        "mount_dropped_for_width", "ranking_ids", "ungrounded_kept_first",
        "no_mount_fit", "resident", "evicted", "live_tokens",
        "admission_policy", "admission_policy_branch",
        "admission_rank_plan", "admission_identifier_hit_count",
        "admission_identified_candidates", "admission_route_margin_1_2",
        "admission_route_margin_evaluated",
        "admission_margin_threshold", "admission_rule_sha256",
    )
    return {key: info.get(key) for key in keys if key in info}


def _answer_correct(answer: str, expected: Sequence[str], stale: Sequence[str]) -> bool:
    return bool(
        any(contains_value(answer, value) for value in expected)
        and not any(contains_value(answer, value) for value in stale)
    )


def _snapshot_counterfactual(arena) -> dict[str, Any]:
    attrs = {}
    for name in (
        "_s4_turn", "_deferred_route_key_token", "_deferred_route_keys",
        "mount_clock", "page_ins", "last_route_receipt", "last_route_backend",
        "last_route_policy_backend", "_last_route_cand_base_event",
    ):
        attrs[name] = (
            hasattr(arena, name),
            copy.deepcopy(getattr(arena, name, None)),
        )
    return {
        "caches": None if arena.caches is None else list(arena.caches),
        "pos": int(arena.pos),
        "live_segs": list(arena.live_segs),
        "cur_mounts": list(arena.cur_mounts),
        "cur_mount_n": int(arena.cur_mount_n),
        "graft_count": len(arena.grafts),
        "metadata_present": ["metadata" in graft for graft in arena.grafts],
        "metadata": [copy.deepcopy(graft.get("metadata")) for graft in arena.grafts],
        "last_used_present": ["last_used" in graft for graft in arena.grafts],
        "last_used": [copy.deepcopy(graft.get("last_used")) for graft in arena.grafts],
        "h_present": ["h" in graft for graft in arena.grafts],
        "h": [graft.get("h") for graft in arena.grafts],
        "route_receipt_history_len": len(
            getattr(arena, "route_receipt_history", ()) or ()),
        "attrs": attrs,
    }


def _restore_counterfactual(arena, snapshot: Mapping[str, Any]) -> None:
    from core import kv_graft

    arena.caches = snapshot["caches"]
    arena.pos = int(snapshot["pos"])
    arena.live_segs = list(snapshot["live_segs"])
    arena.cur_mounts = list(snapshot["cur_mounts"])
    arena.cur_mount_n = int(snapshot["cur_mount_n"])
    if len(arena.grafts) > int(snapshot["graft_count"]):
        del arena.grafts[int(snapshot["graft_count"]):]
        arena._bump_cuda_gqa_epoch()
    for index, graft in enumerate(arena.grafts):
        for key, present_key, values in (
            ("metadata", "metadata_present", "metadata"),
            ("last_used", "last_used_present", "last_used"),
        ):
            if snapshot[present_key][index]:
                graft[key] = copy.deepcopy(snapshot[values][index])
            else:
                graft.pop(key, None)
        if snapshot["h_present"][index]:
            graft["h"] = snapshot["h"][index]
        else:
            graft.pop("h", None)
    for name, (present, value) in snapshot["attrs"].items():
        if present:
            setattr(arena, name, copy.deepcopy(value))
        elif hasattr(arena, name):
            delattr(arena, name)
    history = getattr(arena, "route_receipt_history", None)
    if isinstance(history, list):
        del history[int(snapshot["route_receipt_history_len"]):]
    kv_graft.clear_injection(arena.m)
    # Keep an enabled native HostGraftStore's occupancy mirror aligned with
    # the restored Python/cache state.
    arena._commit_native_mount(arena.cur_mounts, arena.cur_mount_n)


@contextmanager
def _withheld_from_admission(e2e, aliases: set[int]) -> Iterator[None]:
    """Remove one logical graft only from A-DEC's completed mount plan.

    The production helper first performs its unmodified ranking, identifier,
    margin, and branch decision over the full eligible index.  DET1 then
    removes only the planted aliases from ``rank_plan``. All other planned
    mounts, the ladder, and L2 remain normal; D-LQR sees the full repository.
    """
    original = e2e.decisive_admission_profile
    original_fit = e2e._budget_fit_mounts
    withheld = {int(value) for value in aliases}

    def wrapped(arena, question, *, exclude, **kwargs):
        profile = dict(original(
            arena, question, exclude=exclude, **kwargs))
        unfiltered = [int(value) for value in profile["rank_plan"]]
        profile["rank_plan"] = [
            value for value in unfiltered if value not in withheld]
        profile["det1_unfiltered_rank_plan"] = unfiltered
        profile["det1_withheld_alias_ids"] = sorted(withheld)
        return profile

    def fit_without_withheld(arena, picks):
        return original_fit(
            arena,
            [int(value) for value in picks if int(value) not in withheld],
        )

    e2e.decisive_admission_profile = wrapped
    e2e._budget_fit_mounts = fit_without_withheld
    try:
        yield
    finally:
        e2e.decisive_admission_profile = original
        e2e._budget_fit_mounts = original_fit


class _PredictionObserver:
    """Collect the exact greedy token IDs for the separate verbal pass."""

    def __init__(self, arena, ngen: int):
        self.arena = arena
        self.ngen = int(ngen)
        self.original = None
        self.ids: list[int] = []

    def __enter__(self):
        self.original = self.arena._forward

        def wrapped(ids, last_only=True):
            logits = self.original(ids, last_only=last_only)
            self.ids.append(int(np.asarray(logits).argmax()))
            return logits

        self.arena._forward = wrapped
        return self

    def __exit__(self, exc_type, exc, tb):
        self.arena._forward = self.original
        return False

    def finish(self) -> list[int]:
        values = list(self.ids)
        if len(values) == self.ngen + 1:
            values.pop()
        if len(values) > self.ngen:
            raise DETError(
                f"verbal observer captured {len(values)} rows for ngen={self.ngen}")
        return values


class _LadderObserver:
    """Isolate signals per production ladder attempt, then select its winner."""

    def __init__(self, arena, aliases: set[int]):
        self.arena = arena
        self.aliases = {int(value) for value in aliases}
        self.original = None
        self.attempts: list[dict[str, Any]] = []

    def __enter__(self):
        self.original = self.arena._attempt
        outer = self

        def wrapped(user_text, picks, ngen, deposit, stops, *args, **kwargs):
            with DetectorObserver(outer.arena, outer.aliases, int(ngen)) as observer:
                answer, info = outer.original(
                    user_text, picks, ngen, deposit, stops, *args, **kwargs)
            outer.attempts.append({
                "call_ordinal": len(outer.attempts),
                "requested_picks": [int(value) for value in picks],
                "mounted_ids": [int(value) for value in outer.arena.cur_mounts],
                "answer": str(answer),
                "signals": observer.finish(),
            })
            return answer, info

        self.arena._attempt = wrapped
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.original is not None:
            self.arena._attempt = self.original
        return False

    def finish(
        self,
        *,
        answer: str,
        info: Mapping[str, Any] | None,
        mounted_ids: Sequence[int],
    ) -> dict[str, Any]:
        authoritative_mounts = [int(value) for value in mounted_ids]
        trip = int((info or {}).get("trip", 0) or 0)
        matches = [
            row for row in self.attempts
            if row["answer"] == str(answer)
            and row["mounted_ids"] == authoritative_mounts
        ]
        selected = None
        for row in matches:
            if int(row["call_ordinal"]) == trip:
                selected = row
                break
        if selected is None and matches:
            selected = matches[0]
        if selected is None:
            raise DETError(
                "could not map production ladder result to an isolated detector "
                f"attempt: trip={trip} mounts={authoritative_mounts} "
                f"attempts={[(r['call_ordinal'], r['mounted_ids']) for r in self.attempts]}")
        signals = dict(selected["signals"])
        signals["ladder_attempt_selection"] = {
            "production_trip": trip,
            "selected_call_ordinal": int(selected["call_ordinal"]),
            "attempt_count": len(self.attempts),
            "attempts": [{
                "call_ordinal": int(row["call_ordinal"]),
                "requested_picks": list(row["requested_picks"]),
                "mounted_ids": list(row["mounted_ids"]),
                "answer_prefix": str(row["answer"])[:160],
            } for row in self.attempts],
        }
        return signals


def _attempt_result(
    arena,
    answer: str,
    info: Mapping[str, Any] | None,
    signals: Mapping[str, Any],
    *,
    expected: Sequence[str],
    stale: Sequence[str],
    aliases: set[int],
) -> dict[str, Any]:
    mounted = [int(value) for value in arena.cur_mounts]
    full_mounted_text = "\n".join(
        str(arena.grafts[index].get("text", "") or "") for index in mounted)
    full_index = {int(value) for value in arena._route_cand_base()}
    return {
        "answer": str(answer),
        "answer_correct": _answer_correct(answer, expected, stale),
        "mounted_ids": mounted,
        "mounted_text_prefixes": [
            str(arena.grafts[index].get("text", "") or "")[:240]
            for index in mounted
        ],
        "mounted_contains_expected": any(
            contains_value(full_mounted_text, value) for value in expected),
        "full_index_contains_all_aliases": bool(
            aliases and {int(value) for value in aliases} <= full_index),
        "arena_info": _compact_info(info),
        "signals": dict(signals),
    }


def _mechanistic_counterfactual(
    repo,
    e2e,
    original: Callable[..., tuple[str, dict[str, Any]]],
    *,
    question: str,
    aliases: set[int],
    planted_miss: bool,
    topk: int,
    ngen: int,
    max_trips: int,
    turn_idx: int | None,
    expected: Sequence[str],
    stale: Sequence[str],
) -> dict[str, Any]:
    arena = repo.arena
    snapshot = _snapshot_counterfactual(arena)
    admission_context = (
        _withheld_from_admission(e2e, aliases)
        if planted_miss else nullcontext()
    )
    try:
        with admission_context:
            with _LadderObserver(arena, aliases) as observer:
                answer, info = original(
                    repo,
                    question,
                    topk=int(topk),
                    ngen=int(ngen),
                    defer_memory=True,
                    turn_idx=turn_idx,
                    probe_ladder=True,
                    max_trips=int(max_trips),
                )
            signals = observer.finish(
                answer=answer,
                info=info,
                mounted_ids=arena.cur_mounts,
            )
        if not bool((info or {}).get("driver_probe_ladder")):
            raise DETError("counterfactual escaped the production probe ladder")
        return _attempt_result(
            arena,
            answer,
            info,
            signals,
            expected=expected,
            stale=stale,
            aliases=aliases,
        )
    finally:
        _restore_counterfactual(arena, snapshot)


def _verbal_counterfactual(
    arena,
    *,
    prompt: str,
    mounts: Sequence[int],
    ngen: int,
    clean_room: bool,
) -> dict[str, Any]:
    snapshot = _snapshot_counterfactual(arena)
    try:
        if clean_room:
            arena.caches, arena.pos, arena.live_segs = None, 0, []
            arena.cur_mounts, arena.cur_mount_n = [], 0
        for layer in arena.m.layers:
            layer.self_attn.live_shift = arena.live_shift
        with _PredictionObserver(arena, int(ngen)) as observer:
            answer, info = arena._attempt(
                prompt,
                [int(value) for value in mounts],
                int(ngen),
                False,
                arena.stop_sequences,
            )
        token_ids = observer.finish()
        result = verbal_result(answer, token_ids, arena.decode)
        result.update({
            "mounted_ids": [int(value) for value in arena.cur_mounts],
            "arena_info": _compact_info(info),
            "prompt_suffix_exact": VERBAL_QUESTION,
            "clean_room_matched": bool(clean_room),
        })
        return result
    finally:
        _restore_counterfactual(arena, snapshot)


def _row(
    *,
    fixture: Mapping[str, Any],
    variant: str,
    stage: str,
    profile: Mapping[str, Any],
    result: Mapping[str, Any],
    verbal: Mapping[str, Any] | None,
    runtime_frame: Mapping[str, Any],
    registration_path: Path,
    started_ns: int,
) -> dict[str, Any]:
    aliases = {int(value) for value in profile["logical_alias_ids"]}
    mounted = {int(value) for value in result["mounted_ids"]}
    expected = [str(value) for value in fixture["expected_values"]]
    planted = variant == "planted_miss"
    served_admission = dict(profile.get("served_admission_profile") or {})
    adec_active = bool(served_admission)
    unfiltered_plan = [int(value) for value in (
        served_admission.get("rank_plan", ())
        if adec_active else profile.get("raw_ranking", ())
    )]
    expected_filtered_plan = [value for value in unfiltered_plan if value not in aliases]
    actual_info = dict(result["arena_info"])
    ladder_attempts = list(
        (result.get("signals") or {}).get("ladder_attempt_selection", {}).get(
            "attempts", ()))
    return {
        "schema": "grm.det1.probe_row.v1",
        "row_id": f"{fixture['fixture_id']}:{variant}",
        "fixture_id": str(fixture["fixture_id"]),
        "source_family": str(fixture["source_family"]),
        "split": str(fixture["split"]),
        "stage": str(stage),
        "variant": str(variant),
        "label": int(planted),
        "evaluation_started_unix_ns": int(started_ns),
        "question": str(fixture["question"]),
        "expected_values": expected,
        "stale_values": [str(value) for value in fixture.get("stale_values", fixture.get("old_values", []))],
        "wrong_fact_values": [str(value) for value in fixture.get("wrong_fact_values", [])],
        "rejected_answer_values": [
            str(value) for value in (
                list(fixture.get("stale_values", fixture.get("old_values", [])))
                + list(fixture.get("wrong_fact_values", []))
            )
        ],
        "raw_router_rank1": int(profile["raw_rank1"]),
        "logical_router_rank1": int(profile["logical_router_rank1"]),
        "production_admitted_rank1": int(profile["production_admitted_rank1"]),
        "router_rank1_admitted": bool(profile["router_rank1_admitted"]),
        "raw_ranking_ids": [int(value) for value in profile["raw_ranking"]],
        "post_l2_ranking_ids": [int(value) for value in profile["effective_ranking"]],
        "logical_alias_ids": sorted(aliases),
        "target_contains_expected": bool(profile["target_contains_expected"]),
        "served_admission_profile": served_admission,
        "served_effective_admission_plan": [
            int(value) for value in profile["served_effective_admission_plan"]],
        "expected_planted_admission_plan": expected_filtered_plan,
        "production_route_limit": int(profile["production_route_limit"]),
        "route_backend": str(profile["route_backend"]),
        "answer": str(result["answer"]),
        "answer_correct": bool(result["answer_correct"]),
        "mounted_ids": sorted(mounted),
        "mounted_text_prefixes": list(result["mounted_text_prefixes"]),
        "mounted_contains_expected": bool(result["mounted_contains_expected"]),
        "full_index_contains_all_aliases": bool(
            result["full_index_contains_all_aliases"]),
        "arena_info": actual_info,
        "signals": dict(result["signals"]),
        "verbal": None if verbal is None else dict(verbal),
        "plant_checks": {
            "applicable": planted,
            "withheld_aliases_absent": (not bool(aliases & mounted)) if planted else None,
            "logical_target_absent": (
                int(profile["logical_router_rank1"]) not in mounted
                if planted else None
            ),
            "raw_router_rank1_absent": (
                int(profile["raw_rank1"]) not in mounted
                if planted else None
            ),
            "expected_value_absent_from_mounted_text": (
                not bool(result["mounted_contains_expected"])
                if planted else None
            ),
            "withheld_aliases_remain_in_full_detector_index": (
                bool(result["full_index_contains_all_aliases"])
                if planted else None
            ),
            "admission_ranking_unchanged": (
                [int(value) for value in actual_info.get("ranking_ids", ())]
                == [int(value) for value in (
                    served_admission.get("ranking", ())
                    if adec_active else profile.get("raw_ranking", ())
                )][:len(actual_info.get("ranking_ids", ()))]
                if planted else None
            ),
            "admission_branch_unchanged": (
                (
                    actual_info.get("admission_policy_branch")
                    == served_admission.get("policy_branch")
                    if adec_active else
                    "admission_policy" not in actual_info
                )
                if planted else None
            ),
            "only_aliases_removed_from_admission_plan": (
                (
                    [int(value) for value in actual_info.get(
                        "admission_rank_plan", ())] == expected_filtered_plan
                    if adec_active else
                    all(
                        not bool(
                            aliases
                            & {int(value) for value in attempt.get(
                                "requested_picks", ())}
                        )
                        for attempt in ladder_attempts
                    ) and bool(ladder_attempts)
                )
                if planted else None
            ),
            "withheld_aliases_absent_from_every_ladder_attempt": (
                all(
                    not bool(
                        aliases
                        & ({int(value) for value in attempt.get("requested_picks", ())}
                           | {int(value) for value in attempt.get("mounted_ids", ())})
                    )
                    for attempt in ladder_attempts
                ) and bool(ladder_attempts)
                if planted else None
            ),
        },
        "runtime_frame": dict(runtime_frame),
        "registration_path": str(registration_path),
    }


def evaluate_registered_probe(
    repo,
    e2e,
    original: Callable[..., tuple[str, dict[str, Any]]],
    *,
    fixture: Mapping[str, Any],
    stage: str,
    rows_path: Path,
    runtime_frame: Mapping[str, Any],
    registration_path: Path,
    topk: int,
    ngen: int,
    max_trips: int,
    turn_idx: int | None,
    canonical_defer_memory: bool,
) -> tuple[str, dict[str, Any]]:
    arena = repo.arena
    if not bool(arena.revision_resolution):
        raise DETError("DET1 production frame requires L2 ON")
    frame_adm = bool(runtime_frame["resolved_flags"]["adm_decisive"])
    if bool(getattr(arena, "decisive_admission", False)) != frame_adm:
        raise DETError(
            "DET1 arena admission flag differs from frozen runtime frame: "
            f"arena={getattr(arena, 'decisive_admission', None)} frame={frame_adm}")
    live = {int(graft) for graft, _tokens in arena.live_segs if graft is not None}
    expected = [str(value) for value in fixture["expected_values"]]
    stale = [
        str(value) for value in (
            list(fixture.get("stale_values", fixture.get("old_values", [])))
            + list(fixture.get("wrong_fact_values", []))
        )
    ]
    profile_snapshot = _snapshot_counterfactual(arena)
    try:
        profile = route_fixture_profile(
            arena,
            str(fixture["question"]),
            expected[0],
            live_excluded=live,
            route_limit=max(int(topk), (int(max_trips) + 1) * int(topk)),
        )
    finally:
        _restore_counterfactual(arena, profile_snapshot)
    aliases = {int(value) for value in profile["logical_alias_ids"]}
    started_ns = time.time_ns()

    served = _mechanistic_counterfactual(
        repo, e2e, original,
        question=str(fixture["question"]),
        aliases=aliases,
        planted_miss=False,
        topk=int(topk),
        ngen=int(ngen),
        max_trips=int(max_trips),
        turn_idx=turn_idx,
        expected=expected,
        stale=stale,
    )

    planted = None
    served_verbal = None
    planted_verbal = None
    if stage == "eval":
        planted = _mechanistic_counterfactual(
            repo, e2e, original,
            question=str(fixture["question"]),
            aliases=aliases,
            planted_miss=True,
            topk=int(topk),
            ngen=int(ngen),
            max_trips=int(max_trips),
            turn_idx=turn_idx,
            expected=expected,
            stale=stale,
        )
        verbal_prompt = str(fixture["question"]) + "\n\n" + VERBAL_QUESTION
        served_verbal = _verbal_counterfactual(
            arena, prompt=verbal_prompt,
            mounts=served["mounted_ids"], ngen=int(ngen),
            clean_room=bool(
                served["arena_info"].get("clean_room")
                or served["arena_info"].get("point_lookup")))
        planted_verbal = _verbal_counterfactual(
            arena, prompt=verbal_prompt,
            mounts=planted["mounted_ids"], ngen=int(ngen),
            clean_room=bool(
                planted["arena_info"].get("clean_room")
                or planted["arena_info"].get("point_lookup")))

    # Advance the real certified path only after every counterfactual restored.
    canonical_answer, canonical_info = original(
        repo,
        str(fixture["question"]),
        topk=int(topk),
        ngen=int(ngen),
        defer_memory=bool(canonical_defer_memory),
        turn_idx=turn_idx,
        probe_ladder=True,
        max_trips=int(max_trips),
    )
    canonical_mounts = [int(value) for value in (canonical_info.get("mount_fitted") or arena.cur_mounts)]
    if (
        str(canonical_answer) != str(served["answer"])
        or canonical_mounts != [int(value) for value in served["mounted_ids"]]
    ):
        raise DETError(
            "instrumented served counterfactual differs from canonical production "
            f"path for {fixture['fixture_id']}")

    served_row = _row(
        fixture=fixture,
        variant="served",
        stage=stage,
        profile=profile,
        result=served,
        verbal=served_verbal,
        runtime_frame=runtime_frame,
        registration_path=registration_path,
        started_ns=started_ns,
    )
    append_jsonl_once(rows_path, served_row, key="row_id")
    if planted is not None:
        planted_row = _row(
            fixture=fixture,
            variant="planted_miss",
            stage=stage,
            profile=profile,
            result=planted,
            verbal=planted_verbal,
            runtime_frame=runtime_frame,
            registration_path=registration_path,
            started_ns=started_ns,
        )
        append_jsonl_once(rows_path, planted_row, key="row_id")
    print(json.dumps({
        "det1_fixture": fixture["fixture_id"],
        "stage": stage,
        "served_correct": served["answer_correct"],
        "served_mounts": served["mounted_ids"],
        "planted_mounts": None if planted is None else planted["mounted_ids"],
        "logical_aliases": sorted(aliases),
    }, sort_keys=True), flush=True)
    return canonical_answer, canonical_info


def install_patch(
    e2e,
    *,
    enabled: bool,
    registration_path: Path | None = None,
    runtime_frame_path: Path | None = None,
    rows_path: Path | None = None,
    stage: str | None = None,
) -> bool:
    if not enabled:
        return False
    if not all((registration_path, runtime_frame_path, rows_path, stage)):
        raise DETError("enabled DET1 wrapper lacks registration/frame/rows/stage")
    if stage not in ("calibration", "eval"):
        raise DETError(f"unsupported DET1 E2E stage: {stage}")
    registration = read_json(Path(registration_path))
    runtime_frame = read_json(Path(runtime_frame_path))
    selected = {
        int(row["turn"]): row
        for row in registration["fixtures"]
        if row["source_family"] == "certified_34_turn"
        and row["split"] == ("calibration" if stage == "calibration" else "eval")
    }
    original = e2e.probe_multimount_chat

    def wrapped(repo, user_text: str, *, topk: int, ngen: int,
                defer_memory: bool = False, turn_idx: int | None = None,
                probe_ladder: bool = True, max_trips: int = 1):
        if turn_idx is None or int(turn_idx) not in selected:
            return original(
                repo, user_text, topk=topk, ngen=ngen,
                defer_memory=defer_memory, turn_idx=turn_idx,
                probe_ladder=probe_ladder, max_trips=max_trips)
        if probe_ladder is not True:
            raise DETError("registered DET1 probe escaped ladder ON")
        fixture = selected[int(turn_idx)]
        if str(fixture["question"]) != str(user_text):
            raise DETError(f"turn {turn_idx} text drifted from registration")
        return evaluate_registered_probe(
            repo, e2e, original,
            fixture=fixture,
            stage=str(stage),
            rows_path=Path(rows_path),
            runtime_frame=runtime_frame,
            registration_path=Path(registration_path),
            topk=int(topk),
            ngen=int(ngen),
            max_trips=int(max_trips),
            turn_idx=int(turn_idx),
            canonical_defer_memory=bool(defer_memory),
        )

    e2e.probe_multimount_chat = wrapped
    return True


class _DETStageStop(BaseException):
    """Intentional persisted GPU-segment boundary."""


def install_stop_boundary(e2e, stop_after_turns: int | None) -> bool:
    """Persist an E2E session after N completed turns, then return to parent.

    This changes no turn implementation. It runs the production ``run_turn``,
    writes the production scorecard, flushes the repository, and emits the
    same restart receipt fields the driver consumes on ``--resume``.
    """
    if stop_after_turns is None:
        return False
    stop_after = int(stop_after_turns)
    if stop_after <= 0:
        raise DETError("GRM_DET1_STOP_AFTER_TURNS must be positive")
    original = e2e.run_turn

    def wrapped(repo, event, turn_idx, **kwargs):
        result = original(repo, event, turn_idx, **kwargs)
        if int(turn_idx) + 1 == stop_after:
            paths = kwargs["paths"]
            e2e.write_scorecard(paths, kwargs["probe_rows"])
            started = time.perf_counter()
            repo.flush_now()
            flush_ms = (time.perf_counter() - started) * 1000.0
            e2e.write_json(paths["restart"], {
                "schema": "grm_e2e_restart_v1",
                "after_turn": int(turn_idx),
                "flush_wall_ms": float(flush_ms),
                "vram_before_exec": e2e.vram_snapshot(),
                "mode": str(kwargs["args"].mode),
                "det1_segment_boundary": True,
            })
            print(f"det1_segment_stop_after_turns={stop_after}", flush=True)
            raise _DETStageStop()
        return result

    e2e.run_turn = wrapped
    return True


def selftest() -> dict[str, Any]:
    original = object()
    fake = type("Fake", (), {})()
    fake.probe_multimount_chat = original
    installed = install_patch(fake, enabled=False)
    if installed or fake.probe_multimount_chat is not original:
        raise DETError("flags-off wrapper changed the production callable")
    profile_fn = lambda _arena, _question, *, exclude, **_kwargs: {
        "ranking": [1, 2, 3],
        "rank_plan": [1, 2],
        "policy_branch": "frozen_branch",
    }
    fit_fn = lambda _arena, picks: list(picks)
    fake.decisive_admission_profile = profile_fn
    fake._budget_fit_mounts = fit_fn
    with _withheld_from_admission(fake, {1}):
        planted = fake.decisive_admission_profile(None, "q", exclude=set())
        fitted = fake._budget_fit_mounts(None, [1, 2, 3])
        if (
            planted["ranking"] != [1, 2, 3]
            or planted["policy_branch"] != "frozen_branch"
            or planted["rank_plan"] != [2]
            or fitted != [2, 3]
        ):
            raise DETError("planted admission isolation selftest failed")
    if (
        fake.decisive_admission_profile is not profile_fn
        or fake._budget_fit_mounts is not fit_fn
    ):
        raise DETError("planted admission hooks were not restored")
    return {
        "schema": "grm.det1.e2e_selftest.v1",
        "status": "PASS",
        "checks": {
            "flags_off_no_patch_identity": "PASS",
            "planted_plan_only_filter_and_restore": "PASS",
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if values == ["--selftest"]:
        print(json.dumps(selftest(), sort_keys=True))
        return 0
    from scripts import grm_e2e_session as e2e

    enabled = _truthy(os.environ.get("GRM_DET1_ENABLED"))
    if enabled:
        required = {
            "registration_path": os.environ.get("GRM_DET1_REGISTRATION"),
            "runtime_frame_path": os.environ.get("GRM_DET1_RUNTIME_FRAME"),
            "rows_path": os.environ.get("GRM_DET1_ROWS"),
            "stage": os.environ.get("GRM_DET1_STAGE"),
        }
        if not all(required.values()):
            raise DETError(f"missing DET1 environment: {required}")
        install_patch(e2e, enabled=True, **required)
    stop_raw = os.environ.get("GRM_DET1_STOP_AFTER_TURNS", "").strip()
    install_stop_boundary(e2e, int(stop_raw) if stop_raw else None)
    # Preserve this opt-in/off wrapper across the driver's registered restart.
    e2e.__file__ = str(Path(__file__).resolve())
    try:
        return int(e2e.main(values))
    except _DETStageStop:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
