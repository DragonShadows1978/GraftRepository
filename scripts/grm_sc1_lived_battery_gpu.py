#!/usr/bin/env python3
"""GRM-SC1 G4 — no regression on the lived battery, demand loop ON.

ORDER: ``orders/GRM_SC1_DEMAND_LOOP_NGH.md`` (gate G4).

WHAT THIS GATE ASKS.  P2C's G2 established what the nine supersession probes
serve under Arm 1 conditions: 4 CORRECTED + 5 unchanged, 0 regressions, 0
abstentions.  G4 re-runs exactly that with ``GRM_DEMAND_NGH=1`` and asks two
things:

  1. Do the nine probes still serve EXACTLY what P2C Arm 1 served?  A demand
     loop that quietly changes a correctly-served turn is a regression no
     matter how good its detector is.
  2. How many of the nine FIRE?  Prediction: <= 1.  Every fire on a
     CORRECTLY-SERVED probe is a FALSE FIRE and is listed by name -- this is
     the false-positive number on lived traffic, and it is the other half of
     what David needs at flip time.

WHY THIS IS A SEPARATE SCRIPT.  ``scripts/lsr_p2c_replay_gpu.py`` is the P2C
instrument and is READ-ONLY under this order.  Its ``_INFO_KEYS`` allowlist
projects ``fit_*``/``abstain_*`` and deliberately drops everything else, so a
demand-ON run through it reports no ``demand_*`` fields at all (measured).
This module IMPORTS its serving path unchanged and only widens the projection
-- it reimplements nothing.

GPU DISCIPLINE: self-lease on /tmp/forge-gpu.lock, one fixture per process,
<= 580 s, the caller inserts the >= 30 s gap, operator has right of way.
"""
from __future__ import annotations

import argparse
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
from scripts.grm_det1_common import file_record  # noqa: E402
from scripts import lsr_p2c_replay_gpu as p2c  # noqa: E402

ARTIFACT_DIR = ROOT / "artifacts" / "grm_sc1"
#: SC1.1 receipts land in their OWN directory; SC1's are frozen evidence.
SC1_1_ARTIFACT_DIR = ROOT / "artifacts" / "grm_sc1_1"
SCHEMA_PREFIX = "grm.sc1"
REGISTRATION = ARTIFACT_DIR / "grm_sc1_registration.json"
SC1_1_REGISTRATION = SC1_1_ARTIFACT_DIR / "registration.json"
ORDER = ROOT / "orders" / "GRM_SC1_DEMAND_LOOP_NGH.md"
SC1_1_ORDER = ROOT / "orders" / "GRM_SC1_1_GROUNDING_GLYPHS.md"
LEASE_SECONDS = 580
LOCK_WAIT_SECONDS = 7200

#: The P2C info projection PLUS every ``demand_`` field. Nothing P2C kept is
#: dropped; the point is strictly to stop dropping the SC1 block.
_INFO_KEYS = tuple(p2c._INFO_KEYS)


def _info_fields(info: Mapping[str, Any]) -> dict[str, Any]:
    out = {key: info[key] for key in _INFO_KEYS if key in info}
    out.update({
        key: value for key, value in info.items()
        # SC1.1 adds the ``grounding_`` block alongside SC1's ``demand_``
        # block; the P2C projection drops both, so widen it here rather than
        # touching the read-only P2C instrument.
        if str(key).startswith(("demand_", "grounding_"))
    })
    return out


def serve_fixture(session_id: str, demand: bool = True) -> dict[str, Any]:
    """P2C's Arm 1 replay, with the demand and grounding blocks kept.

    SC1.1 G3 runs BOTH demand arms (OFF and ON) with the fixes ON: the glyph
    projection sits in grounding, which the demand loop only consults, so the
    demand-OFF arm is the one that shows the projection's effect on the plain
    serving path with nothing else moving.
    """
    from scripts.grm_det1_2_gpu import _load_model_repo
    from scripts.grm_det1_3_gpu import _install_lived_nodes

    frame = p2c._read(p2c.RUNTIME_FRAME)
    flags = frame["resolved_flags"]
    probes = p2c.sup_probe_plan(session_id)
    fixture = p2c._read(p2c.SUP_FIXTURES / f"{session_id}.json")

    # Arm 1 conditions: the P2A+P2C fixes ON, exactly as P2C's G2 ran them.
    os.environ["GRM_LSR_FIXES"] = "1"
    # The one thing this gate changes (SC1.1 runs it both ways).
    os.environ["GRM_DEMAND_NGH"] = "1" if demand else "0"

    repo_dir = Path(tempfile.mkdtemp(
        prefix=f"sc1_1_g3_{session_id}_{int(bool(demand))}_"))
    served: list[dict[str, Any]] = []
    repo = model = tokenizer = None
    model_info: Any = None
    node_to_idx: dict[str, int] = {}
    try:
        e2e, model, tokenizer, repo, model_info = _load_model_repo(
            repo_dir, frame)
        node_to_idx, _ledgers = _install_lived_nodes(repo, e2e, fixture)
        for probe in probes:
            started = time.time_ns()
            # P2C's Arm 1 serving path, called directly and unmodified.
            answer, info = e2e._probe_ladder_chat(
                repo, str(probe["question"]),
                topk=int(flags["topk"]),
                ngen=int(flags["ngen"]),
                max_trips=int(flags["max_trips"]),
                defer_memory=True,
            )
            elapsed = int(time.time_ns() - started)
            fields = _info_fields(info)
            # SC1.1 G3: the per-probe legacy-vs-normalized grounding verdict,
            # recomputed on the SERVED answer against the mounts it was
            # actually read from. Pure set arithmetic over already generated
            # text -- no forward, no routing, no state mutation.
            served_mounts = [int(v) for v in repo.arena.cur_mounts]
            legacy_grounded, _lc = repo.arena._grounding_verdict(
                str(answer), served_mounts, str(probe["question"]),
                normalized=False)
            normalized_grounded, _nc = repo.arena._grounding_verdict(
                str(answer), served_mounts, str(probe["question"]),
                normalized=True)
            verdict = p2c.answer_verdict(
                answer,
                expected_values=probe["expected_values"],
                rejected_values=probe["rejected_values"])
            served.append({
                **{k: probe[k] for k in (
                    "probe_id", "session_id", "question", "expected_values",
                    "rejected_values", "role", "lived_answer",
                    "lived_correct", "lived_mounted_ids")},
                "served_answer": str(answer),
                "verdict": verdict,
                "info": fields,
                "demand_fired": bool(fields.get("demand_fired", False)),
                "demand_served": fields.get("demand_served"),
                "demand_min_mass": fields.get("demand_min_mass"),
                "demand_token_index": fields.get("demand_token_index"),
                "demand_fetched": fields.get("demand_fetched"),
                "demand_supported": fields.get("demand_supported"),
                # A fire on a probe that served CORRECTLY is a false fire.
                # Named here so the report does not have to infer it.
                "false_fire": bool(
                    fields.get("demand_fired", False) and verdict["correct"]),
                # SC1.1 G3 receipts.
                "grounding_normalized": fields.get("grounding_normalized"),
                "grounding_glyph_rescued": fields.get(
                    "grounding_glyph_rescued"),
                "grounded_legacy_recomputed": bool(legacy_grounded),
                "grounded_normalized_recomputed": bool(normalized_grounded),
                "glyph_rescued_recomputed": bool(
                    normalized_grounded and not legacy_grounded),
                "mounted_ids": served_mounts,
                "elapsed_ns": elapsed,
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
        "schema": f"{SCHEMA_PREFIX}.g4_lived_battery.v1",
        # SC1.1 extends this receipt ADDITIVELY (grounding_* per probe).
        # Every SC1 field keeps its name, type and meaning, so the schema id
        # stays and a reader of the SC1 receipts is not invalidated.
        "schema_extension": "grm.sc1_1.grounding_glyph_receipts.v1",
        "gate": "G4",
        "session_id": session_id,
        "arm": 1,
        "arm_label": "arm1_p2a_p2c_fixes_on_plus_sc1_demand_on",
        "lsr_fixes_env": os.environ.get("GRM_LSR_FIXES"),
        "demand_flag": os.environ.get("GRM_DEMAND_NGH"),
        "carried_threshold": grm_demand.registered_threshold(),
        "carried_threshold_caveat": str(
            grm_demand.load_registered()["caveat"]),
        "order": file_record(ORDER),
        "registration": file_record(REGISTRATION),
        "sc1_1_order": file_record(SC1_1_ORDER),
        "sc1_1_registration": file_record(SC1_1_REGISTRATION),
        "sources_grm_text_norm": file_record(
            ROOT / "core" / "grm_text_norm.py"),
        "probes": served,
        "demand_fired_count": sum(
            1 for row in served if row["demand_fired"]),
        "false_fire_count": sum(1 for row in served if row["false_fire"]),
        "fixture_node_to_idx": {str(k): int(v) for k, v in node_to_idx.items()},
        "runtime_frame": file_record(p2c.RUNTIME_FRAME),
        "model": model_info,
        "sources": {
            "grm_demand": file_record(ROOT / "core" / "grm_demand.py"),
            "graft_arena": file_record(ROOT / "core" / "graft_arena.py"),
            "grm_e2e_session": file_record(
                ROOT / "scripts" / "grm_e2e_session.py"),
            "lsr_p2c_replay_gpu": file_record(
                ROOT / "scripts" / "lsr_p2c_replay_gpu.py"),
            "grm_sc1_lived_battery_gpu": file_record(
                Path(__file__).resolve()),
        },
    }


def compare(demand_receipts: Sequence[Path],
            p2c_receipts: Sequence[Path]) -> dict[str, Any]:
    """The no-regression verdict: same served values as P2C Arm 1."""
    baseline: dict[str, dict[str, Any]] = {}
    for path in p2c_receipts:
        payload = p2c._read(Path(path))
        for row in payload["probes"]:
            baseline[str(row["probe_id"])] = {
                "answer": str(row["served_answer"]),
                "correct": bool(row["verdict"]["correct"]),
            }
    rows: list[dict[str, Any]] = []
    for path in demand_receipts:
        payload = p2c._read(Path(path))
        for row in payload["probes"]:
            probe_id = str(row["probe_id"])
            base = baseline.get(probe_id)
            same = (
                None if base is None
                else str(row["served_answer"]) == base["answer"])
            rows.append({
                "probe_id": probe_id,
                "session_id": row["session_id"],
                "served_answer": str(row["served_answer"]),
                "p2c_arm1_answer": None if base is None else base["answer"],
                "identical_to_p2c_arm1": same,
                "correct": bool(row["verdict"]["correct"]),
                "p2c_arm1_correct": None if base is None else base["correct"],
                "demand_fired": bool(row["demand_fired"]),
                "demand_min_mass": row.get("demand_min_mass"),
                "demand_token_index": row.get("demand_token_index"),
                "demand_served": row.get("demand_served"),
                "demand_fetched": row.get("demand_fetched"),
                "false_fire": bool(row["false_fire"]),
                # SC1.1 G3: legacy vs normalized grounded verdict per probe.
                "grounding_normalized": row.get("grounding_normalized"),
                "grounding_glyph_rescued": row.get("grounding_glyph_rescued"),
                "grounded_legacy_recomputed": row.get(
                    "grounded_legacy_recomputed"),
                "grounded_normalized_recomputed": row.get(
                    "grounded_normalized_recomputed"),
                "glyph_rescued_recomputed": row.get(
                    "glyph_rescued_recomputed"),
                "lived_answer": row["lived_answer"],
                "lived_correct": row["lived_correct"],
            })
    compared = [r for r in rows if r["identical_to_p2c_arm1"] is not None]
    fired = sum(1 for r in rows if r["demand_fired"])
    false_fires = [r["probe_id"] for r in rows if r["false_fire"]]
    all_identical = bool(
        compared and all(r["identical_to_p2c_arm1"] for r in compared))
    return {
        "schema": f"{SCHEMA_PREFIX}.g4_summary.v1",
        "schema_extension": "grm.sc1_1.grounding_glyph_receipts.v1",
        "gate": "G4",
        "probe_count": len(rows),
        "compared_count": len(compared),
        "uncompared_probe_ids": [
            r["probe_id"] for r in rows
            if r["identical_to_p2c_arm1"] is None],
        "all_identical_to_p2c_arm1": all_identical,
        "regressions": [
            r["probe_id"] for r in compared
            if not r["identical_to_p2c_arm1"]],
        "demand_fired_count": fired,
        "prediction_demand_fired": "<= 1",
        "demand_fired_within_prediction": bool(fired <= 1),
        "false_fires_on_correctly_served_probes": false_fires,
        "false_fire_count": len(false_fires),
        # SC1.1 G3: which probes grounded ONLY because of the glyph
        # projection. Registered prediction: the 4 P2C flips.
        "glyph_rescued_probe_ids": sorted(
            r["probe_id"] for r in rows if r.get("glyph_rescued_recomputed")),
        "glyph_rescued_count": sum(
            1 for r in rows if r.get("glyph_rescued_recomputed")),
        # The gate is the NO-REGRESSION claim. The fire count is REPORTED
        # against its prediction, exactly as the order words it.
        "gate_pass": all_identical,
        "table": rows,
        "demand_receipts": [file_record(Path(p)) for p in demand_receipts],
        "p2c_receipts": [file_record(Path(p)) for p in p2c_receipts],
    }


def emit(payload: Mapping[str, Any], stem: str) -> Path:
    # SC1.1 stems are written under artifacts/grm_sc1_1/ so this order's
    # receipts never mix with the frozen SC1 evidence they are compared to.
    out_dir = SC1_1_ARTIFACT_DIR if stem.startswith("sc1_1_") else ARTIFACT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    body = canonical_json_bytes(payload)
    digest = sha256_bytes(body)
    path = out_dir / f"{stem}_{digest[:16]}.json"
    path.write_bytes(body)
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fixture", "compare"))
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--demand", type=int, default=1, choices=(0, 1),
                        help="GRM_DEMAND_NGH arm: 0 = OFF, 1 = ON.")
    parser.add_argument("--demand-receipts", nargs="*", default=())
    parser.add_argument("--p2c-receipts", nargs="*", default=())
    parser.add_argument("--lease-seconds", type=int, default=LEASE_SECONDS)
    parser.add_argument("--lock-wait-seconds", type=int,
                        default=LOCK_WAIT_SECONDS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "compare":
        payload = compare(
            [Path(p) for p in args.demand_receipts],
            [Path(p) for p in args.p2c_receipts])
        path = emit(payload, "sc1_1_g3_summary")
        print(f"receipt={path}")
        print(f"gate_pass={payload['gate_pass']}")
        print(f"probes={payload['probe_count']}"
              f" compared={payload['compared_count']}")
        print(f"regressions={payload['regressions']}")
        print(f"demand_fired={payload['demand_fired_count']}")
        print(f"false_fires={payload['false_fires_on_correctly_served_probes']}")
        print(f"glyph_rescued={payload['glyph_rescued_probe_ids']}")
        for row in payload["table"]:
            print(json.dumps(row))
        return 0 if payload["gate_pass"] else 1

    if not args.session_id:
        raise RuntimeError("--session-id is required for the fixture command")
    from scripts.grm_cmc1_gpu_arms import gpu_lease

    with gpu_lease(int(args.lease_seconds), int(args.lock_wait_seconds)):
        payload = serve_fixture(str(args.session_id), bool(args.demand))
    path = emit(
        payload,
        f"sc1_1_g3_demand{int(bool(args.demand))}_{args.session_id}")
    print(f"receipt={path}")
    print(f"demand_fired_count={payload['demand_fired_count']}")
    print(f"false_fire_count={payload['false_fire_count']}")
    for row in payload["probes"]:
        print(json.dumps({
            "probe_id": row["probe_id"],
            "served": row["served_answer"][:90],
            "correct": row["verdict"]["correct"],
            "demand_fired": row["demand_fired"],
            "demand_min_mass": row["demand_min_mass"],
            "demand_served": row["demand_served"],
            "false_fire": row["false_fire"],
        }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
