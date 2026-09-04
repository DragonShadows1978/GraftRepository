#!/usr/bin/env python3
"""GRM-LSR-P2C G2 — lived-ORDER replay of the nine supersession probes.

Program: LSR (Lived-Serving Reliability), Phase 2C.
Order:   orders/GRM_LSR_P2C_UNSEATABLE_SPLIT_DESCENT.md

WHY THIS EXISTS AND WHAT P2A COULD NOT DO.
P2A's G2 was BLOCKED (``artifacts/lsr_p2a/lsr_p2a_g2_blocked_report.json``):
the DET1.3 snapshots capture MOUNTED PAYLOAD ONLY at a boundary AFTER
routing/admission/fit, so a fork cannot exercise the fit stage.  P2A's
fallback instrument used ``grm_det1_gpu._install_fixture_nodes`` — the DET1.3
"replay" arm, an isolated ``arena.deposit`` per node — and every probe served
a refusal, because that is NOT how the lived campaign built its repository.

WHAT THE LIVED CAMPAIGN ACTUALLY DID (read from the code, named here so the
claim can be checked):

  1. ``grm_det1_5_workers._run_sup`` loads model+repo via
     ``grm_det1_2_gpu._load_model_repo`` (arena_width=96, topk=3,
     live_turns=2, storage_bits=8, revision_resolution=True, A-DEC on).
  2. ``grm_det1_3_gpu._install_lived_nodes`` deposits every fixture node
     CHRONOLOGICALLY through ``arena.feed(harmony_turn(user, assistant),
     deposit=True)`` — each node passes through the LIVE cache.  With
     ``live_turns=2`` the two most recent fed nodes stay in ``live_segs``,
     and ``route()`` EXCLUDES live segments, which is why the lived
     adjudication records rankings shorter than the fixture node count
     (``lived_repository_size_at_probe`` 1..4).
  3. Each probe is then measured by ``grm_det1_5_gpu._measure_fixture_inline``:
     it builds the ladder with ``_attempt_schedule`` (which is
     ``e2e.build_probe_ladder_attempts`` merged behind the A-DEC rank_plan),
     restores the counterfactual base before EVERY rung, fits each rung with
     ``e2e._budget_fit_mounts`` (RANK-ORDER truncation — the pre-P2A law),
     generates with ``arena._attempt(..., defer_memory=True)``, and picks the
     served row with ``_select_ladder``: the FIRST grounded rung, else rung 0.

THIS MODULE REPRODUCES (2) AND (3) EXACTLY and adds nothing.  The heavy
snapshot/fork capture of ``_fork_attempt`` is deliberately not reproduced:
for the ``served`` variant it restores the same pre-prefill state and
regenerates greedily over the same mount set, which is the same computation
``arena._attempt`` performs from the same restored base.  That equivalence is
an ASSERTION THIS GATE TESTS, not an assumption it makes: Arm 0 either
reproduces the lived served values on all nine probes or it does not, and a
non-reproducing Arm 0 is a RESULT — reported, with Arm 1 not run as evidence.

ARMS.
  * ``--arm 0``  fixes OFF (``GRM_LSR_FIXES=0``): the pre-P2A fit stage.
    Must reproduce the lived served values on all 9 probes.
  * ``--arm 1``  fixes ON (default): P2A fit honesty + P2C split/descent.

GPU DISCIPLINE (house rules, enforced here).
  * self-lease on /tmp/forge-gpu.lock via scripts.grm_cmc1_gpu_arms.gpu_lease;
    the operator has absolute right of way;
  * ONE fixture per process, one lease per fixture;
  * <= 580 s per lease (MAX_LEASE_SECONDS 590 is the hard cap);
  * the caller inserts an inter-process gap.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.grm_frame import (  # noqa: E402
    ENV_NAME as PERSISTENT_BOAT_ENV,
    env_persistent_boat,
)
from core.grm_frame import (  # noqa: E402
    CAPTURE_PIN_ENV, CAPTURE_PIN_OFF, CAPTURE_PINS, SEAT_NEAR_LIVE_ENV,
    capture_pin_mode, seat_near_live_enabled,
)
from scripts.grm_cmc1_mechanism import canonical_json_bytes, sha256_bytes  # noqa: E402
from scripts.grm_det1_common import contains_value, file_record  # noqa: E402

ARTIFACT_DIR = ROOT / "artifacts" / "lsr_p2c"
SCHEMA_PREFIX = "grm.lsr_p2c"
RUNTIME_FRAME = (
    ROOT / "artifacts" / "grm_det1" / "run_20260831T160525Z_2"
    / "runtime_frame_28b3196f8fb04a41.json"
)
CENSUS = (
    ROOT / "artifacts" / "grm_det1" / "run_20260831T160525Z_2" / "det1_11"
    / "census" / "lived_serving_census_98ef71e88dec17a1.json"
)
SUP_FIXTURES = ROOT / "tests" / "fixtures" / "supersession_battery"
LEASE_SECONDS = 580
LOCK_WAIT_SECONDS = 7200

#: Every supersession fixture, in the frozen DET1.5 spec order (sup-1..sup-4).
SUP_SESSIONS = (
    "correction_then_restatement",        # sup-1
    "fresh_fact_controls",                # sup-2
    "multi_hop_a_b_c",                    # sup-3
    "short_correction_long_competitor",   # sup-4
)


class P2CError(RuntimeError):
    """The replay could not be performed as registered."""


def _read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def census_rows() -> dict[str, dict[str, Any]]:
    """The DET1.11 census rows, keyed by probe id (the LIVED column)."""
    payload = _read(CENSUS)
    return {str(row["probe_id"]): dict(row) for row in payload["rows"]}


def _probe_row(rows: Mapping[str, Any], probe_id: str, session_id: str,
               probe: Mapping[str, Any]) -> dict[str, Any]:
    row = rows[probe_id]
    return {
        "probe_id": probe_id,
        "session_id": session_id,
        "question": str(probe["question"]),
        "expected_values": [
            str(v) for v in probe.get("expected_values", ())],
        "rejected_values": [
            *[str(v) for v in probe.get("stale_values", ())],
            *[str(v) for v in probe.get("wrong_fact_values", ())],
        ],
        "role": str(row.get("role", "")),
        "lived_answer": str(row.get("served_answer", "")),
        "lived_correct": bool(row.get("served_answer_correct", False)),
        "lived_mounted_ids": [int(v) for v in row.get("mounted_ids", ())],
    }


def sup_probe_plan(session_id: str) -> list[dict[str, Any]]:
    """Every probe the LIVED ``_run_sup`` served for one fixture, IN ORDER.

    ``_run_sup`` serves ``fixture["probes"]`` first, then the frozen DET1.9
    reserve (``SUP_RESERVE_PROBES``) appended last.  Probe ORDER matters: the
    probes run sequentially against one repository, so reordering them is not
    a lived-equivalent replay.
    """
    from scripts.grm_det1_5_workers import SUP_RESERVE_PROBES

    rows = census_rows()
    fixture = _read(SUP_FIXTURES / f"{session_id}.json")
    plan: list[dict[str, Any]] = []
    for probe in fixture["probes"]:
        probe_id = f"sup_{probe['probe_id']}"
        if probe_id in rows:
            plan.append(_probe_row(rows, probe_id, session_id, probe))
    reserve = SUP_RESERVE_PROBES.get(session_id)
    if reserve is not None:
        probe_id = f"sup_{reserve['probe_id']}"
        if probe_id in rows:
            plan.append(_probe_row(rows, probe_id, session_id, reserve))
    if not plan:
        raise P2CError(f"no census probes bound to fixture {session_id}")
    return plan


def answer_verdict(
    answer: str,
    *,
    expected_values: Sequence[str],
    rejected_values: Sequence[str],
) -> dict[str, Any]:
    """G4: the DET1 unified semantic comparator with its negative guards."""
    hit = [v for v in expected_values if contains_value(answer, v)]
    bad = [v for v in rejected_values if contains_value(answer, v)]
    return {
        "expected_hits": hit,
        "rejected_hits": bad,
        "correct": bool(hit and not bad),
    }


def _lived_value_tokens(lived: str) -> list[str]:
    """Code-shaped tokens in a lived answer (``Word-N-Word`` and numerals).

    The battery's values are all ``Word-Digit-Word`` codes, and two lived
    wrong answers are bare numerals ("... is currently 0.", "... is 42.").
    Both are extracted here so the reproduction check has a needle; anything
    else falls back to exact text.
    """
    text = str(lived)
    codes = re.findall(
        r"[A-Za-z]+[‐‑\-][0-9]+[‐‑\-][A-Za-z‐‑\-]+",
        text)
    if codes:
        return list(codes)
    return list(re.findall(r"(?<![\w.\-])([0-9]+)(?![\w\-])", text))


def reproduction_verdict(served: str, lived: str) -> dict[str, Any]:
    """Arm-0 reproduction, judged by the SAME semantic comparator.

    "Verbatim by semantics": the replay reproduces the lived answer when the
    lived answer's VALUE is present in the replay's answer under
    ``contains_value`` (which normalizes Markdown emphasis and U+2010/U+2011),
    never by glyph equality.  A lived refusal reproduces a refusal.
    """
    from scripts.grm_det1_3_gpu import _is_refusal

    lived = str(lived)
    served = str(served)
    if _is_refusal(lived):
        return {"mode": "refusal",
                "reproduced": bool(_is_refusal(served)),
                "lived": lived, "served": served}
    values = _lived_value_tokens(lived)
    if not values:
        return {"mode": "exact_text",
                "reproduced": served.strip() == lived.strip(),
                "lived": lived, "served": served}
    return {"mode": "semantic_value",
            "lived_values": values,
            "reproduced": all(contains_value(served, v) for v in values),
            "lived": lived, "served": served}


_INFO_KEYS = (
    "fit_planned", "fit_seated", "fit_dropped_planned",
    "fit_dropped_filler", "fit_unseatable", "fit_shuttle",
    "fit_shuttle_trips", "served_without_plan_head",
    "fit_split_parent", "fit_split_children", "fit_split_ephemeral",
    "fit_descended_head", "fit_chunk_trips",
    "abstained", "abstain_reason", "abstain_identifier_tokens",
    "admission_policy_branch", "admission_rank_plan",
    "mount_plan", "mount_fitted", "ranking_ids", "trip",
    # GRM-EB1 frame receipts: which frame the turn ran under, what recency
    # nominated, what recency actually cost, and the live window at both ends
    # of the turn.  Receipts only -- this harness's serving path is unchanged.
    "frame_ephemeral", "frame_escape_active",
    "recency_mounted_ids", "recency_seats",
    "live_segments_inherited", "live_segments_carried_into_turn",
    "live_segments_after_turn",
    # GRM-EB1 G4: the demand-ON arm reports false fires and any recovery, so
    # the SC1 demand fields have to survive this filter to be reportable.
)

#: GRM-EB1 G4: the demand-ON arm must report false fires and any recovery, so
#: EVERY ``demand_`` field is captured by PREFIX rather than by an enumerated
#: list.  Guessing SC1's field names would silently drop whichever ones this
#: build actually emits, and an empty demand section would then read as "no
#: fire" when it really meant "not captured".
_INFO_PREFIXES = ("demand_",)


def _fit_fields(info: Mapping[str, Any]) -> dict[str, Any]:
    out = {key: info[key] for key in _INFO_KEYS if key in info}
    out.update({
        key: value for key, value in dict(info).items()
        if str(key).startswith(_INFO_PREFIXES)
    })
    return out


def _serve_probe_arm0(repo, e2e, question: str, flags: Mapping[str, Any]):
    """Arm 0: the LIVED serving mechanism, reproduced step for step.

    ``_measure_fixture_inline`` is the lived instrument.  Its serving core is:
    build the ladder from ``_attempt_schedule``, restore the counterfactual
    base before EVERY rung, fit each rung with ``_budget_fit_mounts``
    (rank-order truncation — the pre-P2A law), generate with
    ``arena._attempt(..., defer_memory=True)``, then ``_select_ladder``: the
    FIRST grounded rung, else rung 0.  Every one of those helpers is IMPORTED
    from the lived modules; none is reimplemented here.
    """
    from scripts.grm_det1_5_gpu import _attempt_schedule, _select_ladder
    from scripts.grm_det1_common import route_fixture_profile
    from scripts.grm_det1_e2e import (
        _restore_counterfactual, _snapshot_counterfactual)

    arena = repo.arena
    topk = int(flags["topk"])
    ngen = int(flags["ngen"])
    max_trips = int(flags["max_trips"])
    live = {int(g) for g, _n in arena.live_segs if g is not None}
    base = _snapshot_counterfactual(arena)
    try:
        profile = route_fixture_profile(
            arena, question, "",
            live_excluded=live,
            route_limit=max(topk, (max_trips + 1) * topk),
        )
    finally:
        _restore_counterfactual(arena, base)
    schedule = _attempt_schedule(
        arena, e2e, question, profile, topk=topk, max_trips=max_trips)
    rows: list[dict[str, Any]] = []
    for ordinal, (planned, clean) in enumerate(schedule):
        _restore_counterfactual(arena, base)
        if clean:
            arena.caches, arena.pos, arena.live_segs = None, 0, []
            arena.cur_mounts, arena.cur_mount_n = [], 0
        for layer in arena.m.layers:
            layer.self_attn.live_shift = arena.live_shift
        fitted = sorted(e2e._budget_fit_mounts(arena, planned))
        if not fitted and planned:
            continue
        answer, info = arena._attempt(
            question, [int(v) for v in fitted], ngen, False,
            arena.stop_sequences or (), defer_memory=True)
        mounts = [int(v) for v in arena.cur_mounts]
        grounded, _c = arena._grounding_attribution(answer, mounts, question)
        rows.append({"served": {
            "answer": str(answer),
            "attempt_ordinal": int(ordinal),
            "mounted_ids": mounts,
            "planned": [int(v) for v in planned],
            "requested_picks": [int(v) for v in fitted],
            "grounded": bool(grounded),
            "info": _fit_fields(info),
        }})
    if not rows:
        raise P2CError("no production ladder rung could be captured")
    selected, ordinal = _select_ladder(rows, "served")
    # NO trailing restore.  ``_measure_fixture_inline`` restores the
    # counterfactual base at the START of every rung and NOT after the last
    # one, so the arena is left in whatever state the final rung produced —
    # including a CLEARED LIVE WINDOW when that rung was clean-room.  That is
    # exactly why the lived receipts show probe 1 of a fixture routing over a
    # live-excluded repository (sup_praxis_fresh ranking [0], live={1,2}) and
    # probe 2 routing over the WHOLE repository (sup_solace_fresh ranking
    # [2,1,0], live={}): probe 1's clean-room rung drained the live window and
    # the driver never put it back.  Restoring here would break lived
    # equivalence for every probe after the first.
    return str(selected["answer"]), {
        **dict(selected["info"] or {}),
        "trip": int(ordinal),
        "mount_fitted": list(selected["requested_picks"]),
        "mount_plan": list(selected["planned"]),
        "admission_rank_plan": [
            int(v) for v in (
                (profile.get("served_admission_profile") or {})
                .get("rank_plan", ()))],
        "ranking_ids": [int(v) for v in profile.get("raw_ranking", ())],
        "ladder_rungs": [
            {
                "ordinal": int(r["served"]["attempt_ordinal"]),
                "requested_picks": list(r["served"]["requested_picks"]),
                "mounted_ids": list(r["served"]["mounted_ids"]),
                "grounded": bool(r["served"]["grounded"]),
                "answer_prefix": str(r["served"]["answer"])[:160],
            }
            for r in rows
        ],
    }, [int(v) for v in selected["mounted_ids"]]


def _serve_probe_arm1(repo, e2e, question: str, flags: Mapping[str, Any]):
    """Arm 1: the SAME repository, served through the production driver.

    ``_probe_ladder_chat`` is the production probe path and the site P2A/P2C
    changed.  ``defer_memory=True`` matches the lived
    ``canonical_defer_memory=True``, so the probe does not deposit its turn.

    STATE EVOLUTION MATCHES ARM 0 EXACTLY.  ``_measure_fixture_inline``
    leaves the arena in the final rung's state (see ``_serve_probe_arm0``),
    and ``_probe_ladder_chat`` likewise leaves it in the served rung's state.
    Arm 1 therefore does NOT restore a counterfactual base either: if it did,
    every probe after the first would route over a different repository than
    Arm 0 saw and the two arms would not be comparable.

    A P2C split that PERSISTS (the librarian seam) is a real, intended
    repository repair and is meant to survive into the next probe — that is
    the "repaired once, not re-chunked at every fit" property Part 1/Part 3
    exist for.  The receipt names the split, so a reader can see it happen.
    """
    arena = repo.arena
    answer, info = e2e._probe_ladder_chat(
        repo, question,
        topk=int(flags["topk"]),
        ngen=int(flags["ngen"]),
        max_trips=int(flags["max_trips"]),
        defer_memory=True,
    )
    return str(answer), _fit_fields(info), [int(v) for v in arena.cur_mounts]


def serve_fixture(session_id: str, *, arm: int,
                  capture_pin: str | None = None,
                  seat_near_live: bool | None = None) -> dict[str, Any]:
    """Replay one fixture's lived deposit order, then serve its probes.

    GRM-RS3 adds ``capture_pin`` and ``seat_near_live``: FLAG PLUMBING ONLY.
    Both default ``None``, which resolves through the (default-OFF, fail-
    closed) env, so a run that passes neither is the legacy run byte for byte.
    Passing them sets the process env for the whole replay, so every deposit
    and every bootstrap seating inside it runs under the named lever — which
    is what "run the battery with the winning pair ON" means.  Nothing else
    about the replay changes, and the resolved values land on the receipt.
    """
    from scripts.grm_det1_2_gpu import _load_model_repo
    from scripts.grm_det1_3_gpu import _install_lived_nodes

    if arm not in (0, 1):
        raise P2CError(f"arm must be 0 or 1, got {arm}")
    frame = _read(RUNTIME_FRAME)
    flags = frame["resolved_flags"]
    probes = sup_probe_plan(session_id)
    fixture = _read(SUP_FIXTURES / f"{session_id}.json")

    # Arm 0 turns the P2A+P2C fixes OFF for the whole process so the
    # reproduction arm cannot accidentally inherit one of them.
    os.environ["GRM_LSR_FIXES"] = "1" if arm else "0"
    # GRM-RS3 levers. An explicit value is validated and pinned for the whole
    # process; ``None`` CLEARS the variable so an ambient setting cannot steer
    # a run whose receipt would then not show it.
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

    repo_dir = Path(tempfile.mkdtemp(prefix=f"lsr_p2c_{session_id}_"))
    served: list[dict[str, Any]] = []
    repo = model = tokenizer = None
    node_to_idx: dict[str, int] = {}
    live_at_install: list[int] = []
    model_info: Any = None
    frame_ephemeral = True
    recency_mounts = 0
    try:
        e2e, model, tokenizer, repo, model_info = _load_model_repo(
            repo_dir, frame)
        # GRM-EB1: read the frame off the arena that will actually serve.
        # ``_load_model_repo`` is a READ-ONLY det1 module that passes no
        # ``ephemeral``, so it inherits the constructor default — which is now
        # the spec frame.  Recording the observed value keeps the receipt
        # honest if that ever stops being true.
        frame_ephemeral = bool(getattr(repo.arena, "ephemeral", False))
        recency_mounts = int(getattr(repo.arena, "recency_mounts", 0))
        # THE LIVED DEPOSIT SEQUENCE: chronological feed, turn by turn.
        node_to_idx, _ledgers = _install_lived_nodes(repo, e2e, fixture)
        live_at_install = [
            int(g) for g, _n in repo.arena.live_segs if g is not None]
        for probe in probes:
            started = time.time_ns()
            if arm == 0:
                answer, info, mounts = _serve_probe_arm0(
                    repo, e2e, str(probe["question"]), flags)
            else:
                answer, info, mounts = _serve_probe_arm1(
                    repo, e2e, str(probe["question"]), flags)
            elapsed_ns = int(time.time_ns() - started)
            served.append({
                **{k: probe[k] for k in (
                    "probe_id", "session_id", "question", "expected_values",
                    "rejected_values", "role", "lived_answer",
                    "lived_correct", "lived_mounted_ids")},
                "served_answer": answer,
                "verdict": answer_verdict(
                    answer,
                    expected_values=probe["expected_values"],
                    rejected_values=probe["rejected_values"]),
                "reproduction": reproduction_verdict(
                    answer, str(probe["lived_answer"])),
                "info": info,
                "mounted_ids": mounts,
                "arena_cur_mount_n": int(repo.arena.cur_mount_n),
                "elapsed_ns": elapsed_ns,
            })
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
        "schema": f"{SCHEMA_PREFIX}.fixture_replay.v1",
        "program": "LSR",
        "phase": "2C",
        "arm": int(arm),
        "arm_label": (
            "arm0_reproduction_fixes_off" if arm == 0
            else "arm1_p2a_p2c_fixes_on"),
        "lsr_fixes_env": os.environ.get("GRM_LSR_FIXES"),
        # GRM-RS3: the two levers this replay served under, RESOLVED (not the
        # raw env), plus the raw env beside them so an operator typo is
        # visible rather than swallowed by the fail-closed rule.
        "rs3_levers": {
            "capture_pin": resolved_pin,
            "seat_near_live": bool(resolved_seat),
            "capture_pin_env": os.environ.get(CAPTURE_PIN_ENV),
            "seat_near_live_env": os.environ.get(SEAT_NEAR_LIVE_ENV),
            "both_off_is_the_legacy_run": (
                resolved_pin == CAPTURE_PIN_OFF and not resolved_seat),
        },
        # GRM-EB1: the FRAME this replay served under, read off the arena that
        # actually served rather than assumed from the env.  A different frame
        # is a NEW baseline, so this field is what tells a reader whether a row
        # is comparable to a persistent-frame receipt (it is not).
        "frame": {
            "ephemeral": bool(frame_ephemeral),
            "escape_env": os.environ.get(PERSISTENT_BOAT_ENV),
            "escape_active": bool(env_persistent_boat()),
            "recency_mounts": int(recency_mounts),
            "spec": (
                "GRM-EB1: the chat log is not kept in memory context; any "
                "chat recall on facts is pulled via GRM"),
            "arm0_reproduction_claimed": False,
            "arm0_reproduction_note": (
                "A different frame is a NEW baseline. This run does NOT claim "
                "or attempt Arm-0 reproduction of persistent-frame rows; the "
                "lived/P2C columns are carried for CHANGE, not for identity."),
        },
        "session_id": session_id,
        "arena_width": int(flags["arena_width"]),
        "deposit_protocol": (
            "CHRONOLOGICAL_ARENA_FEED (grm_det1_3_gpu._install_lived_nodes), "
            "the LIVED arm — not the DET1.3 fixture-install replay arm"),
        "fixture_node_to_idx": {
            str(k): int(v) for k, v in node_to_idx.items()},
        "live_graft_ids_after_install": live_at_install,
        "probes": served,
        "reproduced": all(
            bool(row["reproduction"]["reproduced"]) for row in served),
        "declares_in_det_envelope": False,
        "runtime_frame": file_record(RUNTIME_FRAME),
        "census": file_record(CENSUS),
        "model": model_info,
        "sources": {
            "lsr_p2c_replay_gpu": file_record(Path(__file__).resolve()),
            "grm_admission": file_record(ROOT / "core" / "grm_admission.py"),
            "graft_arena": file_record(ROOT / "core" / "graft_arena.py"),
            "graft_repository": file_record(
                ROOT / "core" / "graft_repository.py"),
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
    parser.add_argument("command", choices=("plan", "fixture"))
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--arm", type=int, default=None, choices=(0, 1))
    # GRM-RS3 flag plumbing. Both default to the legacy run.
    parser.add_argument("--capture-pin", default=None, choices=CAPTURE_PINS,
                        help="GRM-RS3 capture geometry pin (default: off)")
    parser.add_argument("--seat-near-live", action="store_true",
                        default=False,
                        help="GRM-RS3 seat the plan head next to the live "
                             "band (default: off)")
    parser.add_argument("--lease-seconds", type=int, default=LEASE_SECONDS)
    parser.add_argument("--lock-wait-seconds", type=int,
                        default=LOCK_WAIT_SECONDS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "plan":
        out = {session: sup_probe_plan(session) for session in SUP_SESSIONS}
        print(json.dumps({
            "sessions": list(SUP_SESSIONS),
            "probe_count": sum(len(v) for v in out.values()),
            "probes": out,
        }, indent=1))
        return 0

    if not args.session_id:
        raise P2CError("--session-id is required for the fixture command")
    if args.arm is None:
        raise P2CError("--arm is required for the fixture command")

    from scripts.grm_cmc1_gpu_arms import gpu_lease

    with gpu_lease(int(args.lease_seconds), int(args.lock_wait_seconds)):
        payload = serve_fixture(
            str(args.session_id), arm=int(args.arm),
            capture_pin=args.capture_pin,
            seat_near_live=(True if args.seat_near_live else None))
    levers = payload["rs3_levers"]
    stem = f"lsr_p2c_arm{int(args.arm)}_{args.session_id}"
    if not levers["both_off_is_the_legacy_run"]:
        # A lever run gets its own receipt name so it can never be mistaken
        # for the legacy baseline it must be compared against.
        stem += f"_pin-{levers['capture_pin']}_seat-{int(levers['seat_near_live'])}"
    path = emit(payload, stem)
    print(f"receipt={path}")
    print(f"reproduced_all={payload['reproduced']}")
    for row in payload["probes"]:
        print(json.dumps({
            "probe_id": row["probe_id"],
            "lived": row["lived_answer"][:90],
            "served": row["served_answer"][:90],
            "reproduced": row["reproduction"]["reproduced"],
            "correct": row["verdict"]["correct"],
            "fit": row["info"],
        }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
