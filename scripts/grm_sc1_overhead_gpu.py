#!/usr/bin/env python3
"""GRM-SC1 G5 — observer overhead, ms/token, ON vs OFF on the same turn.

ORDER: ``orders/GRM_SC1_DEMAND_LOOP_NGH.md`` (Part 3, gate G5).

WHAT THIS MEASURES AND WHAT IT DOES NOT.  This is the cost of CARRYING the
detector, not the cost of a demand trip.  The observer runs on every token of
every served turn once the flag is on, so it is the price of the mechanism
even on turns that never fire; a trip is an occasional extra generation and
is priced separately by its own fire rate (G3/G4).

METHOD.  Three turns.  Each turn is generated TWICE from the SAME restored
counterfactual base -- once with the observer installed, once without -- and
both arms are wall-clocked over the same greedy token stream.  Same base,
same mounts, same prompt, same argmax decode, so the token counts match and
ms/token is a like-for-like comparison rather than two different generations.

The FIRST timed pair per process is discarded as a warm-up: the first attempt
after model load pays CUDA context and allocator costs that belong to neither
arm.  Which pairs were discarded is stated in the receipt.

NO BUDGET IS REGISTERED.  The order is explicit: report the number, register
nothing.  Whether the overhead is acceptable is David's call at flip time.

GPU DISCIPLINE: self-lease on /tmp/forge-gpu.lock, one process, <= 580 s, the
caller inserts the >= 30 s gap, operator has right of way.
"""
from __future__ import annotations

import argparse
import json
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
from scripts.grm_det1_common import file_record  # noqa: E402
from scripts import lsr_p2c_replay_gpu as p2c  # noqa: E402

ARTIFACT_DIR = ROOT / "artifacts" / "grm_sc1"
SCHEMA_PREFIX = "grm.sc1"
REGISTRATION = ARTIFACT_DIR / "grm_sc1_registration.json"
ORDER = ROOT / "orders" / "GRM_SC1_DEMAND_LOOP_NGH.md"
DEFAULT_SESSION = "fresh_fact_controls"
#: Warm-up pairs discarded before the reported table (see module docstring).
WARMUP_PAIRS = 1
LEASE_SECONDS = 580
LOCK_WAIT_SECONDS = 7200


def _time_attempt(arena, question, picks, ngen, observer_on, threshold):
    """One generation attempt, wall-clocked, observer optionally installed."""
    started = time.perf_counter_ns()
    if observer_on:
        with grm_demand.DemandObserver(
            arena, int(ngen), float(threshold)) as observer:
            answer, _info = arena._attempt(
                question, [int(v) for v in picks], int(ngen), False,
                arena.stop_sequences or (), defer_memory=True)
        rows = observer.finish()
        tokens = len(rows)
    else:
        answer, _info = arena._attempt(
            question, [int(v) for v in picks], int(ngen), False,
            arena.stop_sequences or (), defer_memory=True)
        rows = []
        tokens = None
    elapsed_ns = int(time.perf_counter_ns() - started)
    return str(answer), elapsed_ns, tokens, rows


def run_gate(session_id: str, *, turns: int = 3) -> dict[str, Any]:
    from scripts.grm_det1_2_gpu import _load_model_repo
    from scripts.grm_det1_3_gpu import _install_lived_nodes
    from scripts.grm_det1_common import route_fixture_profile
    from scripts.grm_det1_e2e import (
        _restore_counterfactual, _snapshot_counterfactual,
    )
    from scripts.grm_det1_5_gpu import _attempt_schedule

    frame = p2c._read(p2c.RUNTIME_FRAME)
    flags = frame["resolved_flags"]
    ngen = int(flags["ngen"])
    topk = int(flags["topk"])
    max_trips = int(flags["max_trips"])
    threshold = grm_demand.registered_threshold()
    fixture = p2c._read(p2c.SUP_FIXTURES / f"{session_id}.json")
    probes = p2c.sup_probe_plan(session_id)[:int(turns)]

    repo_dir = Path(tempfile.mkdtemp(prefix=f"sc1_g5_{session_id}_"))
    repo = model = tokenizer = None
    model_info: Any = None
    rows: list[dict[str, Any]] = []
    try:
        e2e, model, tokenizer, repo, model_info = _load_model_repo(
            repo_dir, frame)
        _node_to_idx, _ledgers = _install_lived_nodes(repo, e2e, fixture)
        arena = repo.arena
        base = _snapshot_counterfactual(arena)
        for probe in probes:
            question = str(probe["question"])
            live = {int(g) for g, _n in arena.live_segs if g is not None}
            _restore_counterfactual(arena, base)
            try:
                profile = route_fixture_profile(
                    arena, question, "",
                    live_excluded=live,
                    route_limit=max(topk, (max_trips + 1) * topk))
            finally:
                _restore_counterfactual(arena, base)
            schedule = _attempt_schedule(
                arena, e2e, question, profile, topk=topk, max_trips=max_trips)
            planned, clean = schedule[0]
            picks = sorted(
                int(v) for v in e2e._budget_fit_mounts(arena, planned))
            if not picks:
                continue

            def _prepare(clean=clean):
                _restore_counterfactual(arena, base)
                if clean:
                    arena.caches, arena.pos, arena.live_segs = None, 0, []
                    arena.cur_mounts, arena.cur_mount_n = [], 0
                for layer in arena.m.layers:
                    layer.self_attn.live_shift = arena.live_shift

            _prepare()
            answer_off, ns_off, _t, _r = _time_attempt(
                arena, question, picks, ngen, False, threshold)
            _prepare()
            answer_on, ns_on, tokens_on, observed = _time_attempt(
                arena, question, picks, ngen, True, threshold)
            _restore_counterfactual(arena, base)

            tokens = int(tokens_on or 0)
            decision = grm_demand.decide(observed, threshold)
            rows.append({
                "probe_id": str(probe["probe_id"]),
                "session_id": session_id,
                "question": question,
                "mounted_ids": [int(v) for v in picks],
                "clean_room": bool(clean),
                "tokens": tokens,
                "ns_off": int(ns_off),
                "ns_on": int(ns_on),
                "ms_off": ns_off / 1e6,
                "ms_on": ns_on / 1e6,
                "ms_per_token_off": (ns_off / 1e6 / tokens) if tokens else None,
                "ms_per_token_on": (ns_on / 1e6 / tokens) if tokens else None,
                "overhead_ms_per_token": (
                    (ns_on - ns_off) / 1e6 / tokens) if tokens else None,
                "overhead_ratio": (ns_on / ns_off) if ns_off else None,
                # Identical answers are the evidence that the two arms
                # generated the SAME token stream, so ms/token is like for
                # like and not two different generations compared.
                "answers_identical": answer_off == answer_on,
                "answer": answer_on,
                "min_mounted_mass": decision["demand_min_mass"],
                "demand_fired": decision["demand_fired"],
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

    reported = rows[int(WARMUP_PAIRS):]
    ratios = [r["overhead_ratio"] for r in reported if r["overhead_ratio"]]
    per_token = [
        r["overhead_ms_per_token"] for r in reported
        if r["overhead_ms_per_token"] is not None]
    return {
        "schema": f"{SCHEMA_PREFIX}.g5_overhead.v1",
        "gate": "G5",
        "session_id": session_id,
        "order": file_record(ORDER),
        "registration": file_record(REGISTRATION),
        "no_budget_registered": True,
        "budget_note": (
            "The order registers NO overhead budget. This gate reports the "
            "number; whether it is acceptable is David's call at flip time."),
        "measures": (
            "Cost of CARRYING the observer on every served token. A demand "
            "TRIP is an occasional extra generation and is priced separately "
            "by its fire rate (G3/G4), not here."),
        "warmup_pairs_discarded": int(WARMUP_PAIRS),
        "warmup_probe_ids": [r["probe_id"] for r in rows[:int(WARMUP_PAIRS)]],
        "ngen": ngen,
        "all_rows": rows,
        "reported_rows": reported,
        "mean_overhead_ratio": (
            sum(ratios) / len(ratios)) if ratios else None,
        "mean_overhead_ms_per_token": (
            sum(per_token) / len(per_token)) if per_token else None,
        "all_answers_identical": all(r["answers_identical"] for r in rows),
        "carried_threshold": float(threshold),
        "carried_threshold_caveat": str(
            grm_demand.load_registered()["caveat"]),
        "runtime_frame": file_record(p2c.RUNTIME_FRAME),
        "model": model_info,
        "sources": {
            "grm_demand": file_record(ROOT / "core" / "grm_demand.py"),
            "grm_sc1_overhead_gpu": file_record(Path(__file__).resolve()),
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
    parser.add_argument("--session-id", default=DEFAULT_SESSION)
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--lease-seconds", type=int, default=LEASE_SECONDS)
    parser.add_argument("--lock-wait-seconds", type=int,
                        default=LOCK_WAIT_SECONDS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    from scripts.grm_cmc1_gpu_arms import gpu_lease

    with gpu_lease(int(args.lease_seconds), int(args.lock_wait_seconds)):
        payload = run_gate(str(args.session_id), turns=int(args.turns))
    path = emit(payload, f"sc1_g5_overhead_{args.session_id}")
    print(f"receipt={path}")
    print(f"all_answers_identical={payload['all_answers_identical']}")
    print(f"mean_overhead_ratio={payload['mean_overhead_ratio']}")
    print(f"mean_overhead_ms_per_token={payload['mean_overhead_ms_per_token']}")
    for row in payload["all_rows"]:
        print(json.dumps({
            "probe_id": row["probe_id"],
            "tokens": row["tokens"],
            "ms_off": round(row["ms_off"], 1),
            "ms_on": round(row["ms_on"], 1),
            "ms_per_token_off": (
                None if row["ms_per_token_off"] is None
                else round(row["ms_per_token_off"], 3)),
            "ms_per_token_on": (
                None if row["ms_per_token_on"] is None
                else round(row["ms_per_token_on"], 3)),
            "overhead_ratio": (
                None if row["overhead_ratio"] is None
                else round(row["overhead_ratio"], 3)),
            "answers_identical": row["answers_identical"],
        }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
