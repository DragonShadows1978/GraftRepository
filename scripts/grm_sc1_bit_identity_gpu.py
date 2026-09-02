#!/usr/bin/env python3
"""GRM-SC1 G2 — bit-identity: production D-NGH observer vs the frozen race one.

ORDER: ``orders/GRM_SC1_DEMAND_LOOP_NGH.md`` (Part 1, gate G2).

WHAT THIS PROVES AND WHY IT IS NEEDED.  ``core/grm_demand.DemandObserver`` is
a REIMPLEMENTATION of the D-NGH capture that ``scripts/grm_det1_common
.DetectorObserver`` performs — the race module is frozen instrumentation and
also carries D-LQR/D-ENT hooks production must not pay for, so it could not
simply be imported into the serving path.  A reimplementation is a claim, and
the claim is that the production observer computes the SAME per-token
``mounted_mass``.  This gate tests that claim on real GPT-OSS attention
operands: one served fixture turn, generated TWICE from the SAME restored
counterfactual base, once under each observer, with EXACT float equality
required token for token.

Not "close".  Not ``allclose``.  The two observers run the same operations in
the same order on the same operands, so anything but exact equality means one
of them drifted, and a drifted detector cannot carry a CARRIED threshold.

DETERMINISM.  Both arms restore the identical counterfactual base before
generating and decode greedily (argmax), so the token stream is the same and
the comparison is per-position and total.

GPU DISCIPLINE (house rules, enforced here).
  * self-lease on /tmp/forge-gpu.lock via scripts.grm_cmc1_gpu_arms.gpu_lease;
    the operator has absolute right of way;
  * ONE process, ONE lease, <= 580 s (MAX_LEASE_SECONDS 590 is the hard cap);
  * the caller inserts the inter-process gap.
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

ARTIFACT_DIR = ROOT / "artifacts" / "grm_sc1"
SCHEMA_PREFIX = "grm.sc1"
RUNTIME_FRAME = (
    ROOT / "artifacts" / "grm_det1" / "run_20260831T160525Z_2"
    / "runtime_frame_28b3196f8fb04a41.json"
)
REGISTRATION = ARTIFACT_DIR / "grm_sc1_registration.json"
SUP_FIXTURES = ROOT / "tests" / "fixtures" / "supersession_battery"
#: The default served fixture. sup-2 is the cheapest lived session in the
#: battery (the race measured its probes at 12-13 s) and its first probe is a
#: LAWFUL served control in the DET1.11 census, so this is a SERVED turn --
#: the side the threshold was fit on -- exactly as the gate specifies.
DEFAULT_SESSION = "fresh_fact_controls"
LEASE_SECONDS = 580
LOCK_WAIT_SECONDS = 7200


class SC1Error(RuntimeError):
    """The SC1 bit-identity gate could not be performed as registered."""


def _read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _mounted_masses_production(arena, question, picks, ngen, threshold):
    """Generate one attempt under the PRODUCTION observer."""
    with grm_demand.DemandObserver(arena, int(ngen), float(threshold)) as obs:
        answer, _info = arena._attempt(
            question, [int(v) for v in picks], int(ngen), False,
            arena.stop_sequences or (), defer_memory=True)
    rows = obs.finish()
    return str(answer), [float(row["mounted_mass"]) for row in rows], rows


def _mounted_masses_frozen(arena, question, picks, ngen, alias_ids):
    """Generate one attempt under the FROZEN race observer.

    ``active_detectors=("D-NGH",)`` isolates the hook under test; the D-LQR
    and D-ENT hooks are not part of the claim and would add a route-capture
    seam that production deliberately does not have.
    """
    from scripts.grm_det1_common import DetectorObserver

    with DetectorObserver(
        arena, set(alias_ids), int(ngen), active_detectors=("D-NGH",),
    ) as obs:
        answer, _info = arena._attempt(
            question, [int(v) for v in picks], int(ngen), False,
            arena.stop_sequences or (), defer_memory=True)
    signals = obs.finish()
    tokens = list(signals["tokens"])
    return str(answer), [
        float(token["ngh"]["mounted_mass"]) for token in tokens], tokens


def run_gate(session_id: str, *, probe_index: int = 0) -> dict[str, Any]:
    """Serve one fixture turn twice and compare the two mass streams."""
    from scripts.grm_det1_2_gpu import _load_model_repo
    from scripts.grm_det1_3_gpu import _install_lived_nodes
    from scripts.grm_det1_e2e import (
        _restore_counterfactual, _snapshot_counterfactual,
    )
    from scripts.grm_det1_common import route_fixture_profile

    frame = _read(RUNTIME_FRAME)
    flags = frame["resolved_flags"]
    ngen = int(flags["ngen"])
    topk = int(flags["topk"])
    max_trips = int(flags["max_trips"])
    threshold = grm_demand.registered_threshold()
    registered = grm_demand.load_registered()
    fixture = _read(SUP_FIXTURES / f"{session_id}.json")
    probes = list(fixture["probes"])
    if not 0 <= int(probe_index) < len(probes):
        raise SC1Error(
            f"probe index {probe_index} out of range for {session_id}")
    probe = probes[int(probe_index)]
    question = str(probe["question"])

    repo_dir = Path(tempfile.mkdtemp(prefix=f"sc1_g2_{session_id}_"))
    repo = model = tokenizer = None
    model_info: Any = None
    try:
        e2e, model, tokenizer, repo, model_info = _load_model_repo(
            repo_dir, frame)
        node_to_idx, _ledgers = _install_lived_nodes(repo, e2e, fixture)
        arena = repo.arena

        support = grm_demand.arena_support(arena)
        if not support.get("demand_supported"):
            raise SC1Error(
                "production observer refused this arena: "
                f"{support.get('demand_unsupported_reason')}")

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
        admission = profile.get("served_admission_profile") or {}
        planned = [int(v) for v in admission.get(
            "rank_plan", profile.get("raw_ranking", ()))]
        picks = sorted(int(v) for v in e2e._budget_fit_mounts(arena, planned))
        if not picks:
            raise SC1Error(
                f"no mount set fits for {session_id} probe {probe_index}")
        alias_ids = {int(v) for v in profile["logical_alias_ids"]}

        # ARM A: production observer, from the restored base.
        _restore_counterfactual(arena, base)
        for layer in arena.m.layers:
            layer.self_attn.live_shift = arena.live_shift
        started_a = time.time_ns()
        answer_a, masses_a, rows_a = _mounted_masses_production(
            arena, question, picks, ngen, threshold)
        elapsed_a = int(time.time_ns() - started_a)

        # ARM B: frozen race observer, from the SAME restored base.
        _restore_counterfactual(arena, base)
        for layer in arena.m.layers:
            layer.self_attn.live_shift = arena.live_shift
        started_b = time.time_ns()
        answer_b, masses_b, tokens_b = _mounted_masses_frozen(
            arena, question, picks, ngen, alias_ids)
        elapsed_b = int(time.time_ns() - started_b)
        _restore_counterfactual(arena, base)
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

    same_length = len(masses_a) == len(masses_b)
    # EXACT equality, float for float. Not allclose.
    exact = bool(same_length and all(
        a == b for a, b in zip(masses_a, masses_b)))
    mismatches = [
        {"token_index": index, "production": a, "frozen_det1": b,
         "abs_delta": abs(a - b)}
        for index, (a, b) in enumerate(zip(masses_a, masses_b)) if a != b
    ]
    decision_a = grm_demand.decide(rows_a, threshold)
    frozen_decision_fired = any(
        float(token["ngh"]["mounted_mass"]) < threshold for token in tokens_b)

    return {
        "schema": f"{SCHEMA_PREFIX}.g2_bit_identity.v1",
        "gate": "G2",
        "order": file_record(ROOT / "orders" / "GRM_SC1_DEMAND_LOOP_NGH.md"),
        "registration": file_record(REGISTRATION),
        "session_id": session_id,
        "probe_id": str(probe.get("probe_id", "")),
        "probe_index": int(probe_index),
        "question": question,
        "variant": "served",
        "mounted_ids": [int(v) for v in picks],
        "ngen": int(ngen),
        "gate_pass": exact,
        "status": "PASS_EXACT_BIT_IDENTITY" if exact else "FAIL_DIVERGENT",
        "token_count_production": len(masses_a),
        "token_count_frozen_det1": len(masses_b),
        "token_counts_equal": same_length,
        "exact_equality": exact,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:32],
        "mounted_mass_production": masses_a,
        "mounted_mass_frozen_det1": masses_b,
        "answer_production": answer_a,
        "answer_frozen_det1": answer_b,
        "answers_equal": answer_a == answer_b,
        "elapsed_ns_production": elapsed_a,
        "elapsed_ns_frozen_det1": elapsed_b,
        "carried_threshold": float(threshold),
        "carried_threshold_caveat": str(registered["caveat"]),
        "carried_threshold_provenance": {
            "path": registered["provenance"]["thresholds_path"],
            "sha256": registered["provenance"]["thresholds_sha256"],
            "fit_turn_count": int(registered["fit_turn_count"]),
        },
        "decision_production": decision_a,
        "decision_frozen_det1_fired": bool(frozen_decision_fired),
        "decisions_agree": bool(
            decision_a["demand_fired"] == frozen_decision_fired),
        "fixture_node_to_idx": {str(k): int(v) for k, v in node_to_idx.items()},
        "runtime_frame": file_record(RUNTIME_FRAME),
        "model": model_info,
        "sources": {
            "grm_demand": file_record(ROOT / "core" / "grm_demand.py"),
            "grm_det1_common": file_record(
                ROOT / "scripts" / "grm_det1_common.py"),
            "grm_sc1_bit_identity_gpu": file_record(Path(__file__).resolve()),
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
    parser.add_argument("--probe-index", type=int, default=0)
    parser.add_argument("--lease-seconds", type=int, default=LEASE_SECONDS)
    parser.add_argument("--lock-wait-seconds", type=int,
                        default=LOCK_WAIT_SECONDS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    from scripts.grm_cmc1_gpu_arms import gpu_lease

    with gpu_lease(int(args.lease_seconds), int(args.lock_wait_seconds)):
        payload = run_gate(str(args.session_id),
                           probe_index=int(args.probe_index))
    path = emit(payload, f"sc1_g2_bit_identity_{args.session_id}")
    print(f"receipt={path}")
    print(f"gate_pass={payload['gate_pass']}")
    print(f"status={payload['status']}")
    print(f"tokens={payload['token_count_production']}"
          f"/{payload['token_count_frozen_det1']}")
    print(f"mismatch_count={payload['mismatch_count']}")
    print(f"answers_equal={payload['answers_equal']}")
    print(f"decisions_agree={payload['decisions_agree']}")
    return 0 if payload["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
