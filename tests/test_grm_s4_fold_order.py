"""G-FOLD S4 fold-order CPU contracts and lead-run GPU gate.

Pytest collection stays CPU-only.  Model imports and GPU work happen only
behind ``--run-gpu``.  Each GPU arm replays the early half of a repeat-probe
fixture, performs four deferred folds, then asks that fixture's late probes.
The extension adds twelve ordinary exchanges in three batches, so the late
probes are genuinely post-fold while the deferred hot path never needs its
2x backpressure escape hatch.

Lead commands (one process at a time)::

    python3 tests/test_grm_s4_fold_order.py --run-gpu --fold-order age
    python3 tests/test_grm_s4_fold_order.py --run-gpu --fold-order s4
    python3 tests/test_grm_s4_fold_order.py --analyze

Receipts default to ``/tmp/graftrepo_s4_fold_order_receipts``.  The fixture
tree is read only and the two arms overwrite only their own temporary
receipts, never a sealed fixture.
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time
from types import SimpleNamespace

import pytest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.graft_repository import GraftRepository
from tests.test_grm_importance_g1g2 import (
    next_probe_answer_turn,
    next_scripted_deposit_turn,
    scripted_deposit_text,
    wipe_workspace,
)


FIXTURES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fixtures",
    "importance_convos_repeat")
DEFAULT_RECEIPTS_DIR = "/tmp/graftrepo_s4_fold_order_receipts"
RECEIPT_SCHEMA = "grm_s4_fold_order_v1"
VERDICT_SCHEMA = "grm_s4_fold_order_verdict_v1"
FOLD_ORDERS = ("age", "s4")
BASELINE_FOLD_ORDER = "age"
CANDIDATE_FOLD_ORDER = "s4"
MIN_FOLD_EVENTS = 4
MAX_ADD_TURN_SECONDS = 0.27
NGEN = 48
MAX_TRIPS = 2
DRIVER_ID = "repeat_probe_prefix_four_deferred_folds_then_late_v1"

# The first repeat fixture supplies four early/late fact probes.  These
# ordinary exchanges extend it to 22 deposits (44 conversational turns):
# 10 fixture deposits + 12 extension deposits.  Every extension node is a
# filler, so no new relevance signal is planted after the early probes.
FOLD_EXTENSION = tuple(
    {
        "turn_id": 100 + index,
        "role": "user",
        "text": (
            f"Extension note {index}: the office supply cabinet was checked "
            f"at routine interval {index}; no service action is required."),
        "assistant": (
            f"Routine extension {index} recorded; the supply cabinet needs "
            "no action."),
        "node_class": "FILLER",
    }
    for index in range(1, 13)
)


# ---------------------------------------------------------------------------
# CPU: source-selection law.  This deliberately uses a small direct planner
# double rather than a model/ArenaCache: the tested methods are the same
# eligible-source, native-plan, and fallback-plan code used by both runtime
# librarian callers, with no model load or GPU dependency.
# ---------------------------------------------------------------------------

def _graft(*, n_grounded=0, kind="turn", h=None, no_fold=False,
           retired=False):
    return {
        "kind": kind,
        "h": object() if h is None else h,
        "host_payload": None,
        "no_fold": bool(no_fold),
        "retired": bool(retired),
        "metadata": {
            "active": True,
            "importance": {"s4": {"n_grounded": n_grounded}},
        },
    }


def _placeholder(*, n_grounded=0):
    """An M11-shaped node: text metadata but no resident/host/disk K/V."""
    graft = _graft(n_grounded=n_grounded)
    graft["h"] = None
    graft["payload_pending"] = True
    graft["recovered"] = True
    return graft


def _planner_repo(grafts, tmp_path, *, fold_order="age"):
    repo = object.__new__(GraftRepository)
    repo.path = str(tmp_path)
    repo.native_store = None
    repo.fold_order = fold_order
    repo.arena = SimpleNamespace(
        grafts=list(grafts), live_segs=[], ENABLE_ERA_FOLDING=True)
    return repo


def _digest_sources(repo, *, deferred_backpressure=False):
    jobs = repo._librarian_jobs(
        deferred_backpressure=deferred_backpressure)
    return jobs[0][1] if jobs else []


def test_fold_order_normalization_is_opt_in_and_strict():
    assert GraftRepository._normalize_fold_order(None) == "age"
    assert GraftRepository._normalize_fold_order("AGE") == "age"
    assert GraftRepository._normalize_fold_order("s4") == "s4"
    with pytest.raises(ValueError, match="fold order"):
        GraftRepository._normalize_fold_order("importance-score")


def test_s4_orders_hits_ascending_with_existing_age_tiebreak(tmp_path):
    # Index is the pre-existing age order.  The four zero-hit sources must
    # retain their age order rather than introducing any new score tiebreak.
    hits = [5, 0, 3, 0, 2, 0, 0, 4]
    repo = _planner_repo(
        [_graft(n_grounded=hits_i) for hits_i in hits], tmp_path,
        fold_order="s4")

    assert _digest_sources(repo) == [1, 3, 5, 6]


def test_s4_orders_eligible_digest_sources_with_the_same_key(tmp_path):
    hits = [4, 0, 2, 0, 1, 0]
    repo = _planner_repo(
        [_graft(n_grounded=hits_i, kind="digest") for hits_i in hits],
        tmp_path, fold_order="s4")

    assert repo._librarian_jobs() == [("era", [1, 3, 5])]


def test_age_default_takes_no_new_ordering_branch(tmp_path):
    hits = [5, 0, 3, 0, 2, 0, 0, 4]
    repo = _planner_repo(
        [_graft(n_grounded=hits_i) for hits_i in hits], tmp_path,
        fold_order="age")

    def unexpected_sort(_sources):
        pytest.fail("the default age arm must not invoke S4 ordering")

    repo._order_fold_sources_s4 = unexpected_sort
    assert _digest_sources(repo) == [0, 1, 2, 3]


@pytest.mark.parametrize(
    "deferred_backpressure,n_sources",
    [(False, GraftRepository.TURNS_HIGH),
     (True, GraftRepository.TURNS_HIGH * 2)],
    ids=["deferred", "backpressure"],
)
def test_s4_source_order_reaches_both_librarian_paths(
        tmp_path, deferred_backpressure, n_sources):
    hits = [6, 0, 4, 0, 2, 0, 0, 5]
    hits.extend([7] * (n_sources - len(hits)))
    repo = _planner_repo(
        [_graft(n_grounded=hits_i) for hits_i in hits], tmp_path,
        fold_order="s4")

    assert _digest_sources(
        repo, deferred_backpressure=deferred_backpressure) == [1, 3, 5, 6]


@pytest.mark.parametrize("fold_order", FOLD_ORDERS)
def test_m11_placeholder_stays_excluded_regardless_of_fold_order(
        tmp_path, fold_order):
    # The placeholder is index zero and a zero-hit source, so S4 would pick
    # it before every real node if eligibility moved behind ordering.
    hits = [5, 0, 3, 0, 2, 0, 0, 4]
    repo = _planner_repo(
        [_placeholder(n_grounded=0)]
        + [_graft(n_grounded=hits_i) for hits_i in hits],
        tmp_path, fold_order=fold_order)

    sources = _digest_sources(repo)
    assert 0 not in repo._foldable(("turn",))
    assert 0 not in sources
    if fold_order == "age":
        assert sources == [1, 2, 3, 4]
    else:
        assert sources == [2, 4, 6, 7]


def test_s4_exclusions_do_not_count_or_jam_later_progress(tmp_path):
    # 816a0a0's lesson: permanently excluded nodes cannot contribute to a
    # threshold or get a future window stuck retrying them.  Give both
    # exclusions attractive zero-hit metadata; S4 must still advance only
    # across real sources after enough real turns arrive.
    real = [_graft(n_grounded=(index % 3)) for index in range(
        GraftRepository.TURNS_HIGH - 1)]
    excluded_placeholder = _placeholder(n_grounded=0)
    excluded_no_fold = _graft(n_grounded=0, no_fold=True)
    repo = _planner_repo(
        [excluded_placeholder, excluded_no_fold] + real,
        tmp_path, fold_order="s4")

    assert _digest_sources(repo) == []

    repo.arena.grafts.append(_graft(n_grounded=0))
    first = _digest_sources(repo)
    assert len(first) == GraftRepository.TURNS_FOLD
    assert 0 not in first and 1 not in first
    for idx in first:
        repo.arena.grafts[idx]["retired"] = True

    # Replenish to the actual foldable threshold.  The second plan proves
    # source advancement rather than a count-only "due" loop.
    repo.arena.grafts.extend(
        _graft(n_grounded=0) for _ in range(GraftRepository.TURNS_FOLD))
    second = _digest_sources(repo)
    assert len(second) == GraftRepository.TURNS_FOLD
    assert 0 not in second and 1 not in second
    assert set(first).isdisjoint(second)


# ---------------------------------------------------------------------------
# CPU: receipt schema plus the registered, non-inferiority-only G-FOLD gate.
# ---------------------------------------------------------------------------

def _probe_contract(record):
    return (
        record["conversation_id"],
        int(record["probe_turn_id"]),
        int(record["target_turn_id"]),
        tuple(record["expected_answer_tokens"]),
    )


def _late_recall(records):
    hits = sum(bool(record["hit"]) for record in records)
    total = len(records)
    return {
        "hits": hits,
        "total": total,
        "rate": hits / total if total else None,
    }


def make_fold_order_receipt(
        fold_order, *, session_fingerprint, fixture_ids, n_add_turns,
        late_probes, folds, add_turn_latencies, backpressure_folds=0):
    if fold_order not in FOLD_ORDERS:
        raise ValueError(f"unknown fold order {fold_order!r}")
    late_probes = [dict(record) for record in late_probes]
    folds = [dict(fold) for fold in folds]
    add_turn_latencies = [float(value) for value in add_turn_latencies]
    if not add_turn_latencies:
        raise ValueError("receipt needs at least one add_turn latency")
    fidelity_aborts = sum(not bool(fold.get("accepted")) for fold in folds)
    return {
        "schema": RECEIPT_SCHEMA,
        "fold_order": fold_order,
        "session": {
            "driver_id": DRIVER_ID,
            "fingerprint": str(session_fingerprint),
            "repository_sessions": 1,
            "fixture_ids": list(fixture_ids),
            "n_add_turns": int(n_add_turns),
            "n_late_probes": len(late_probes),
            "minimum_fold_events": MIN_FOLD_EVENTS,
        },
        "late_probe_recall": _late_recall(late_probes),
        "fidelity_aborts": int(fidelity_aborts),
        "fold_events": len(folds),
        "folds": folds,
        "add_turn_latency": {
            "samples": len(add_turn_latencies),
            "max_seconds": max(add_turn_latencies),
            "seconds": add_turn_latencies,
            "backpressure_folds": int(backpressure_folds),
            "flat_limit_seconds": MAX_ADD_TURN_SECONDS,
        },
        "late_probes": late_probes,
    }


def validate_fold_order_receipt(receipt):
    if not isinstance(receipt, dict):
        raise ValueError("fold-order receipt must be a JSON object")
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise ValueError(
            f"receipt schema {receipt.get('schema')!r}, expected "
            f"{RECEIPT_SCHEMA!r}")
    if receipt.get("fold_order") not in FOLD_ORDERS:
        raise ValueError("receipt has invalid fold_order")
    session = receipt.get("session")
    if not isinstance(session, dict) or not session.get("fingerprint"):
        raise ValueError("receipt missing session fingerprint")
    if session.get("driver_id") != DRIVER_ID:
        raise ValueError("receipt driver does not match G-FOLD")
    if session.get("repository_sessions") != 1:
        raise ValueError("receipt must represent exactly one repository session")
    if session.get("minimum_fold_events") != MIN_FOLD_EVENTS:
        raise ValueError("receipt changed the registered fold prerequisite")
    late_probes = receipt.get("late_probes")
    recall = receipt.get("late_probe_recall")
    if not isinstance(late_probes, list) or not isinstance(recall, dict):
        raise ValueError("receipt missing late-probe records")
    if recall != _late_recall(late_probes):
        raise ValueError("late-probe recall summary disagrees with records")
    if session.get("n_late_probes") != len(late_probes):
        raise ValueError("late-probe count disagrees with session")
    for record in late_probes:
        _probe_contract(record)
        if int(record.get("fold_events_before_probe", 0)) < MIN_FOLD_EVENTS:
            raise ValueError("late probe was not measured after four folds")

    folds = receipt.get("folds")
    if not isinstance(folds, list) or len(folds) < MIN_FOLD_EVENTS:
        raise ValueError("receipt did not execute the registered fold minimum")
    if receipt.get("fold_events") != len(folds):
        raise ValueError("fold event count disagrees with composition")
    aborts = 0
    for fold in folds:
        sources = fold.get("sources")
        if not isinstance(sources, list) or not sources:
            raise ValueError("each fold needs source composition")
        if not bool(fold.get("accepted")):
            aborts += 1
        for source in sources:
            grounded = source.get("n_grounded")
            if (not isinstance(grounded, int)
                    or isinstance(grounded, bool) or grounded < 0):
                raise ValueError("fold source needs a nonnegative grounded count")
            if not isinstance(source.get("turns"), list):
                raise ValueError("fold source needs turn labels")
    if receipt.get("fidelity_aborts") != aborts:
        raise ValueError("fidelity-abort count disagrees with folds")

    latency = receipt.get("add_turn_latency")
    if not isinstance(latency, dict):
        raise ValueError("receipt missing add_turn latency")
    samples = latency.get("seconds")
    if (not isinstance(samples, list) or not samples
            or latency.get("samples") != len(samples)):
        raise ValueError("add_turn latency samples are malformed")
    if any(value < 0 for value in samples):
        raise ValueError("add_turn latency cannot be negative")
    if latency.get("max_seconds") != pytest.approx(max(samples)):
        raise ValueError("add_turn max latency disagrees with samples")
    if latency.get("flat_limit_seconds") != MAX_ADD_TURN_SECONDS:
        raise ValueError("receipt changed the registered flatness limit")
    if latency.get("backpressure_folds") != 0:
        raise ValueError("deferred G-FOLD arm used backpressure folding")
    if session.get("n_add_turns") != len(samples):
        raise ValueError("add_turn sample count disagrees with session")
    return receipt


def _receipt_path(receipts_dir, fold_order):
    return os.path.join(receipts_dir, f"grm_s4_fold_order_{fold_order}.json")


def write_fold_order_receipt(receipt, receipts_dir=DEFAULT_RECEIPTS_DIR):
    validate_fold_order_receipt(receipt)
    os.makedirs(receipts_dir, exist_ok=True)
    path = _receipt_path(receipts_dir, receipt["fold_order"])
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def load_fold_order_receipts(receipts_dir=DEFAULT_RECEIPTS_DIR):
    receipts = {}
    for fold_order in FOLD_ORDERS:
        path = _receipt_path(receipts_dir, fold_order)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                receipt = json.load(handle)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"missing {fold_order} receipt: {path}") from exc
        validate_fold_order_receipt(receipt)
        if receipt["fold_order"] != fold_order:
            raise ValueError(
                f"{path}: contains {receipt['fold_order']!r}, expected "
                f"{fold_order!r}")
        receipts[fold_order] = receipt
    return receipts


def analyze_fold_order_receipts(receipts):
    if set(receipts) != set(FOLD_ORDERS):
        raise ValueError("analysis requires exactly the age and s4 receipts")
    age = validate_fold_order_receipt(receipts[BASELINE_FOLD_ORDER])
    s4 = validate_fold_order_receipt(receipts[CANDIDATE_FOLD_ORDER])
    same_session = bool(
        age["session"]["fingerprint"] == s4["session"]["fingerprint"]
        and age["session"]["driver_id"] == s4["session"]["driver_id"]
        and age["session"]["fixture_ids"] == s4["session"]["fixture_ids"]
        and age["session"]["n_add_turns"] == s4["session"]["n_add_turns"])
    same_probes = bool(
        [_probe_contract(record) for record in age["late_probes"]]
        == [_probe_contract(record) for record in s4["late_probes"]])
    enough_folds = {
        fold_order: receipt["fold_events"] >= MIN_FOLD_EVENTS
        for fold_order, receipt in receipts.items()
    }
    deferred_only = {
        fold_order: receipt["add_turn_latency"]["backpressure_folds"] == 0
        for fold_order, receipt in receipts.items()
    }
    prerequisites = bool(
        same_session and same_probes and all(enough_folds.values())
        and all(deferred_only.values()))
    recall_noninferior = bool(
        s4["late_probe_recall"]["hits"]
        >= age["late_probe_recall"]["hits"]
        and s4["late_probe_recall"]["total"]
        == age["late_probe_recall"]["total"])
    aborts_noninferior = bool(
        s4["fidelity_aborts"] <= age["fidelity_aborts"])
    flatness = {
        fold_order: receipt["add_turn_latency"]["max_seconds"]
        <= MAX_ADD_TURN_SECONDS
        for fold_order, receipt in receipts.items()
    }
    hot_path_flat = all(flatness.values())
    gate_pass = bool(
        prerequisites and recall_noninferior and aborts_noninferior
        and hot_path_flat)
    equivalence = bool(
        gate_pass
        and s4["late_probe_recall"]["hits"]
        == age["late_probe_recall"]["hits"]
        and s4["fidelity_aborts"] == age["fidelity_aborts"])
    note = None
    if equivalence:
        note = ("PASS-by-equivalence: s4 is free but not yet advantageous "
                "at this scale.")
    elif not prerequisites:
        note = "INVALID gate prerequisites; G-FOLD is RED."
    return {
        "schema": VERDICT_SCHEMA,
        "comparison": {
            "baseline_fold_order": BASELINE_FOLD_ORDER,
            "candidate_fold_order": CANDIDATE_FOLD_ORDER,
        },
        "registered_gate": {
            "late_probe_recall": "s4 >= age",
            "fidelity_aborts": "s4 <= age",
            "add_turn_max_seconds": MAX_ADD_TURN_SECONDS,
        },
        "arms": {
            fold_order: {
                "late_probe_hits": receipt["late_probe_recall"]["hits"],
                "late_probe_total": receipt["late_probe_recall"]["total"],
                "fidelity_aborts": receipt["fidelity_aborts"],
                "fold_events": receipt["fold_events"],
                "add_turn_max_seconds": receipt["add_turn_latency"][
                    "max_seconds"],
            }
            for fold_order, receipt in receipts.items()
        },
        "prerequisites": {
            "same_session": same_session,
            "same_late_probes": same_probes,
            "at_least_four_folds": enough_folds,
            "no_backpressure_folds": deferred_only,
            "pass": prerequisites,
        },
        "conditions": {
            "late_probe_recall_noninferior": recall_noninferior,
            "fidelity_aborts_noninferior": aborts_noninferior,
            "hot_path_flat": hot_path_flat,
            "per_arm_flatness": flatness,
        },
        "margins": {
            "late_probe_hits_s4_minus_age": (
                s4["late_probe_recall"]["hits"]
                - age["late_probe_recall"]["hits"]),
            "fidelity_aborts_age_minus_s4": (
                age["fidelity_aborts"] - s4["fidelity_aborts"]),
        },
        "pass_by_equivalence": equivalence,
        "note": note,
        "pass": gate_pass,
    }


def _synthetic_probe(index, hit=True):
    return {
        "conversation_id": "convo_synthetic",
        "probe_turn_id": 200 + index,
        "target_turn_id": 10 + index,
        "expected_answer_tokens": [f"fact-{index}"],
        "answer": f"fact-{index}" if hit else "miss",
        "hit": bool(hit),
        "fold_events_before_probe": MIN_FOLD_EVENTS,
    }


def _synthetic_fold(index, *, accepted=True, grounded=0):
    return {
        "ordinal": index + 1,
        "kind": "digest",
        "accepted": bool(accepted),
        "digest_idx": 50 + index if accepted else None,
        "sources": [{
            "graft_index": index,
            "turns": [{"conversation_id": "convo_synthetic",
                       "turn_id": index + 1}],
            "n_grounded": grounded,
        }],
    }


def _synthetic_receipt(fold_order, *, hits=4, aborts=0, max_latency=0.2,
                       fingerprint="same-session"):
    probes = [_synthetic_probe(index, hit=index < hits) for index in range(4)]
    folds = [_synthetic_fold(
        index, accepted=index >= aborts, grounded=index % 2)
        for index in range(MIN_FOLD_EVENTS)]
    return make_fold_order_receipt(
        fold_order,
        session_fingerprint=fingerprint,
        fixture_ids=["convo_synthetic"],
        n_add_turns=4,
        late_probes=probes,
        folds=folds,
        add_turn_latencies=[max_latency] * 4,
    )


def test_fold_order_receipts_round_trip_and_reject_wrong_schema(tmp_path):
    receipts = {
        fold_order: _synthetic_receipt(fold_order)
        for fold_order in FOLD_ORDERS
    }
    for receipt in receipts.values():
        write_fold_order_receipt(receipt, str(tmp_path))
    assert load_fold_order_receipts(str(tmp_path)) == receipts

    path = _receipt_path(str(tmp_path), "s4")
    broken = dict(receipts["s4"])
    broken["schema"] = "wrong"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(broken, handle)
    with pytest.raises(ValueError, match="schema"):
        load_fold_order_receipts(str(tmp_path))


def test_registered_gate_accepts_equivalence_without_positive_margin():
    verdict = analyze_fold_order_receipts({
        "age": _synthetic_receipt("age", hits=3, aborts=1),
        "s4": _synthetic_receipt("s4", hits=3, aborts=1),
    })

    assert verdict["pass"] is True
    assert verdict["pass_by_equivalence"] is True


def test_registered_gate_rejects_mismatched_session_fingerprint():
    verdict = analyze_fold_order_receipts({
        "age": _synthetic_receipt("age", fingerprint="age-session"),
        "s4": _synthetic_receipt("s4", fingerprint="s4-session"),
    })

    assert verdict["pass"] is False
    assert verdict["prerequisites"]["same_session"] is False


@pytest.mark.parametrize(
    "candidate_kwargs, condition",
    [
        ({"hits": 2}, "late_probe_recall_noninferior"),
        ({"aborts": 2}, "fidelity_aborts_noninferior"),
        ({"max_latency": 0.28}, "hot_path_flat"),
    ],
    ids=["recall", "aborts", "latency"],
)
def test_registered_gate_rejects_each_required_condition(
        candidate_kwargs, condition):
    candidate = {"hits": 3, "aborts": 1}
    candidate.update(candidate_kwargs)
    verdict = analyze_fold_order_receipts({
        "age": _synthetic_receipt("age", hits=3, aborts=1),
        "s4": _synthetic_receipt("s4", **candidate),
    })

    assert verdict["pass"] is False
    assert verdict["conditions"][condition] is False


# ---------------------------------------------------------------------------
# Lead-run GPU arm.  Nothing in this section loads a model until a user
# explicitly passes --run-gpu to this file as a script.
# ---------------------------------------------------------------------------

def _fixture_paths():
    return sorted(glob.glob(os.path.join(FIXTURES_DIR, "convo_*.json")))


def _load_fold_fixture():
    paths = _fixture_paths()
    if len(paths) != 4:
        raise RuntimeError(
            f"expected four repeat-probe fixtures, found {len(paths)}")
    fixtures = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            fixtures.append(json.load(handle))
    # One fixture plus a registered 12-exchange extension is a 44-turn
    # session that creates exactly four scheduled turn folds before the late
    # probes.  The other three files remain part of the sealed style set and
    # are fingerprinted too, so any fixture-set drift invalidates an A/B.
    return fixtures[0], fixtures, paths


def _target_turn_id(probe):
    targets = [int(turn_id) for turn_id, grade in probe["relevance"].items()
               if grade == 3]
    if len(targets) != 1:
        raise ValueError(
            f"probe {probe['probe_turn_id']}: grade-3 targets {targets}")
    return targets[0]


def _probe_phases(fixture):
    seen = set()
    phases = {}
    counts = {"early": 0, "late": 0}
    for probe in fixture["probes"]:
        target = _target_turn_id(probe)
        phase = "late" if target in seen else "early"
        seen.add(target)
        phases[int(probe["probe_turn_id"])] = phase
        counts[phase] += 1
    if counts != {"early": 4, "late": 4}:
        raise ValueError(
            f"{fixture['conversation_id']}: expected 4 early/4 late probes, "
            f"got {counts}")
    return phases


def _session_fingerprint(fixture, style_fixtures, fixture_paths):
    contract = {
        "driver_id": DRIVER_ID,
        "fixture": fixture,
        "repeat_fixture_set": style_fixtures,
        "fixture_filenames": [os.path.basename(path) for path in fixture_paths],
        "fold_extension": FOLD_EXTENSION,
        "min_fold_events": MIN_FOLD_EVENTS,
        "ngen": NGEN,
        "max_trips": MAX_TRIPS,
    }
    encoded = json.dumps(
        contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_gpu_repository(workspace, fold_order):
    tensor_path = "/mnt/ForgeRealm/Project-Tensor/tensor_cuda"
    if tensor_path not in sys.path:
        sys.path.insert(0, tensor_path)
    import numpy as np
    import tensor_cuda as tc
    from tokenizers import Tokenizer as HFTok
    from core.minicpm3_tc import MiniCPM3_TC, _snap

    tokenizer = HFTok.from_file(os.path.join(_snap(), "tokenizer.json"))
    model, info = MiniCPM3_TC.from_pretrained()
    print(f"loaded: {info}", flush=True)
    with tc.no_grad():
        model(np.array([[1, 2, 3]], dtype=np.int64), kv_caches=None,
              position_offset=0, last_token_only=True)
    tc.synchronize()

    wipe_workspace(workspace)
    repo = GraftRepository(
        model,
        lambda text: tokenizer.encode(text).ids,
        lambda ids: tokenizer.decode(ids),
        workspace,
        autosave=False,
        librarian_mode="deferred",
        durability_mode="volatile_fast",
        fold_order=fold_order,
        native_auto=False,
        sink_text="<conversation>\n",
        arena_width=512,
        route_layer=44,
        topk=3,
        live_turns=2,
        ephemeral=True,
        recency_mounts=2,
    )
    assert not repo.arena.grafts, (
        f"workspace hygiene failure: {workspace} did not boot empty")
    assert repo.TURNS_HIGH == 8 and repo.TURNS_FOLD == 4
    for layer in model.layers:
        layer.self_attn.live_shift = repo.arena.live_shift
        layer.self_attn.absorbed_decode = True
    return repo, tc


def _s4_grounded_count(arena, idx):
    s4 = (arena.grafts[idx].get("metadata", {})
          .get("importance", {}).get("s4"))
    value = s4.get("n_grounded", 0) if isinstance(s4, dict) else 0
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else 0


def _timed_add_turn(repo, tc, user_text, assistant_text):
    tc.synchronize()
    started = time.perf_counter()
    repo.add_turn(user_text, assistant_text)
    tc.synchronize()
    return time.perf_counter() - started, repo.runtime.last_result


def _deposit_fixture_turn(repo, tc, fixture, user_turn, assistant_turn,
                          turn_to_graft, node_turns, latencies):
    before_len = len(repo.arena.grafts)
    if assistant_turn is None:
        # Preserve the repeat fixture's SOLO path but still time the complete
        # add_turn event boundary with CUDA work synchronized on both sides.
        tc.synchronize()
        started = time.perf_counter()
        before = repo._snapshot_state()
        repo.arena.feed(scripted_deposit_text(user_turn))
        repo._set_new_node_provenance(before, "exchange_span")
        extracted = repo._extract_from_new_turns(
            before,
            context={"event": "add_turn", "user_text": user_turn["text"],
                     "assistant_text": None})
        result = repo.runtime._finish_turn_event(
            "add_turn", before, extraction=extracted, autosave=False)
        repo._queue_s2_pending()
        tc.synchronize()
        elapsed = time.perf_counter() - started
    else:
        elapsed, result = _timed_add_turn(
            repo, tc, user_turn["text"], assistant_turn["text"])
    if (len(result.new_nodes) != 1
            or result.new_nodes[0] != before_len):
        raise RuntimeError(
            f"{fixture['conversation_id']} turn {user_turn['turn_id']}: "
            f"expected one node at {before_len}, got {result.new_nodes}")
    graft_idx = int(result.new_nodes[0])
    user_ref = {"conversation_id": fixture["conversation_id"],
                "turn_id": int(user_turn["turn_id"])}
    turn_to_graft[(fixture["conversation_id"], int(user_turn["turn_id"]))] = graft_idx
    node_turns[graft_idx] = [user_ref]
    if assistant_turn is not None:
        assistant_ref = {"conversation_id": fixture["conversation_id"],
                         "turn_id": int(assistant_turn["turn_id"])}
        turn_to_graft[(fixture["conversation_id"],
                       int(assistant_turn["turn_id"]))] = graft_idx
        node_turns[graft_idx].append(assistant_ref)
    latencies.append(elapsed)


def _deposit_extension_turn(repo, tc, extension, node_turns, latencies):
    before_len = len(repo.arena.grafts)
    elapsed, result = _timed_add_turn(
        repo, tc, extension["text"], extension["assistant"])
    if (len(result.new_nodes) != 1
            or result.new_nodes[0] != before_len):
        raise RuntimeError(
            f"extension {extension['turn_id']}: expected one node at "
            f"{before_len}, got {result.new_nodes}")
    graft_idx = int(result.new_nodes[0])
    node_turns[graft_idx] = [{
        "conversation_id": "g_fold_extension",
        "turn_id": int(extension["turn_id"]),
    }]
    latencies.append(elapsed)


def _drive_probe(repo, fixture, probe, phase, turn_to_graft,
                 fold_events_before_probe):
    target_turn = _target_turn_id(probe)
    target_node = turn_to_graft[(fixture["conversation_id"], target_turn)]
    before_grounded = _s4_grounded_count(repo.arena, target_node)
    answer, info = repo.arena.step(
        probe["question"], ngen=NGEN, deposit=False, max_trips=MAX_TRIPS)
    after_grounded = _s4_grounded_count(repo.arena, target_node)
    expected = [str(token).lower()
                for token in probe["expected_answer_tokens"]]
    hit = all(token in answer.lower() for token in expected)
    record = {
        "conversation_id": fixture["conversation_id"],
        "phase": phase,
        "probe_turn_id": int(probe["probe_turn_id"]),
        "target_turn_id": target_turn,
        "target_node": int(target_node),
        "expected_answer_tokens": expected,
        "answer": answer,
        "hit": bool(hit),
        "fold_events_before_probe": int(fold_events_before_probe),
        "target_n_grounded_before": before_grounded,
        "target_n_grounded_after": after_grounded,
        "mounts_one_based": list(info.get("mounts", ())),
        "trip": int(info.get("trip", 0)),
    }
    print(
        f"  {phase} {fixture['conversation_id']}#{probe['probe_turn_id']}: "
        f"{'HIT' if hit else 'MISS'} mounts={record['mounts_one_based']} "
        f"answer={answer[:100]!r}",
        flush=True)
    return record


def _fold_once_with_composition(repo, node_turns):
    jobs = repo._due()
    if not jobs:
        raise RuntimeError("expected a deferred fold job, found none")
    kind, idxs = jobs[0]
    sources = [{
        "graft_index": int(idx),
        "turns": list(node_turns.get(idx, [])),
        "n_grounded": _s4_grounded_count(repo.arena, idx),
    } for idx in idxs]
    history_before = len(repo.fold_history)
    done = repo.idle(max_jobs=1)
    if done != 1 or len(repo.fold_history) != history_before + 1:
        raise RuntimeError("deferred idle did not execute exactly one fold")
    event = repo.fold_history[-1]
    if event["kind"] != kind or event["sources"] != list(idxs):
        raise RuntimeError("fold history diverged from the planned source set")
    return {
        "ordinal": len(repo.fold_history),
        "kind": kind,
        "accepted": bool(event["accepted"]),
        "digest_idx": event["digest_idx"],
        "sources": sources,
    }


def _drive_fixture_prefix(repo, tc, fixture, turn_to_graft, node_turns,
                          latencies):
    probes = {int(probe["probe_turn_id"]): probe
              for probe in fixture["probes"]}
    phases = _probe_phases(fixture)
    early_records = []
    late_probes = []
    index = 0
    turns = fixture["turns"]
    while index < len(turns):
        turn = turns[index]
        if turn["node_class"] == "PROBE":
            probe = probes[int(turn["turn_id"])]
            _scripted_answer, index = next_probe_answer_turn(turns, index)
            if phases[int(turn["turn_id"])] == "early":
                early_records.append(_drive_probe(
                    repo, fixture, probe, "early", turn_to_graft,
                    fold_events_before_probe=0))
            else:
                late_probes.append(probe)
            continue
        assistant, index = next_scripted_deposit_turn(turns, index)
        _deposit_fixture_turn(
            repo, tc, fixture, turn, assistant, turn_to_graft, node_turns,
            latencies)
    return early_records, late_probes


def run_gpu_arm(fold_order, receipts_dir=DEFAULT_RECEIPTS_DIR):
    if fold_order not in FOLD_ORDERS:
        raise ValueError(f"unknown fold order {fold_order!r}")
    fixture, style_fixtures, fixture_paths = _load_fold_fixture()
    workspace = f"/tmp/graftrepo_s4_fold_order_{fold_order}"
    repo, tc = _load_gpu_repository(workspace, fold_order)
    turn_to_graft = {}
    node_turns = {}
    latencies = []
    fold_history_before_adds = len(repo.fold_history)
    early_records, late_probes = _drive_fixture_prefix(
        repo, tc, fixture, turn_to_graft, node_turns, latencies)
    if len(early_records) != 4 or len(late_probes) != 4:
        raise RuntimeError(
            f"expected four early/four late probes, got "
            f"{len(early_records)}/{len(late_probes)}")
    if len(repo.fold_history) != fold_history_before_adds:
        raise RuntimeError("fixture prefix used inline backpressure folding")

    folds = [_fold_once_with_composition(repo, node_turns)]
    for start in range(0, len(FOLD_EXTENSION), GraftRepository.TURNS_FOLD):
        for extension in FOLD_EXTENSION[start:start + GraftRepository.TURNS_FOLD]:
            _deposit_extension_turn(repo, tc, extension, node_turns, latencies)
        if len(repo.fold_history) != len(folds):
            raise RuntimeError("extension used inline backpressure folding")
        folds.append(_fold_once_with_composition(repo, node_turns))
    if len(folds) != MIN_FOLD_EVENTS:
        raise RuntimeError(
            f"registered four-fold schedule produced {len(folds)} folds")
    if any(fold["kind"] != "digest" for fold in folds):
        raise RuntimeError("G-FOLD schedule unexpectedly reached an era fold")

    late_records = [_drive_probe(
        repo, fixture, probe, "late", turn_to_graft,
        fold_events_before_probe=len(folds)) for probe in late_probes]
    receipt = make_fold_order_receipt(
        fold_order,
        session_fingerprint=_session_fingerprint(
            fixture, style_fixtures, fixture_paths),
        fixture_ids=[fixture["conversation_id"]],
        n_add_turns=len(latencies),
        late_probes=late_records,
        folds=folds,
        add_turn_latencies=latencies,
        backpressure_folds=0,
    )
    path = write_fold_order_receipt(receipt, receipts_dir)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    print(
        f"{fold_order.upper()} ARM COMPLETE: folds={receipt['fold_events']}, "
        f"late recall={receipt['late_probe_recall']['hits']}/"
        f"{receipt['late_probe_recall']['total']}, "
        f"fidelity aborts={receipt['fidelity_aborts']}, "
        f"add_turn max={receipt['add_turn_latency']['max_seconds']:.3f}s",
        flush=True)
    print(f"wrote {path}", flush=True)
    return receipt


def run_analysis(receipts_dir=DEFAULT_RECEIPTS_DIR):
    receipts = load_fold_order_receipts(receipts_dir)
    verdict = analyze_fold_order_receipts(receipts)
    for fold_order in FOLD_ORDERS:
        arm = verdict["arms"][fold_order]
        print(
            f"{fold_order.upper()}: late recall={arm['late_probe_hits']}/"
            f"{arm['late_probe_total']}; fidelity aborts="
            f"{arm['fidelity_aborts']}; folds={arm['fold_events']}; "
            f"add_turn max={arm['add_turn_max_seconds']:.3f}s",
            flush=True)
    conditions = verdict["conditions"]
    print(
        "S4 vs AGE conditions: recall(s4)>=recall(age)="
        f"{conditions['late_probe_recall_noninferior']}; "
        "aborts(s4)<=aborts(age)="
        f"{conditions['fidelity_aborts_noninferior']}; "
        f"hot-path-flat={conditions['hot_path_flat']}; "
        f"prerequisites={verdict['prerequisites']['pass']}; "
        f"PASS={verdict['pass']}",
        flush=True)
    if verdict["note"]:
        print(verdict["note"], flush=True)
    os.makedirs(receipts_dir, exist_ok=True)
    path = os.path.join(receipts_dir, "grm_s4_fold_order_verdict.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(verdict, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(verdict, sort_keys=True), flush=True)
    print(f"wrote {path}", flush=True)
    return verdict


def _main(argv=None):
    parser = argparse.ArgumentParser(
        description="G-FOLD S4 fold-order A/B GPU arms and CPU analysis")
    parser.add_argument(
        "--run-gpu", action="store_true",
        help="authorize one lead-run GPU fold-order arm")
    parser.add_argument(
        "--fold-order", choices=FOLD_ORDERS,
        help="fold-order arm to replay (requires --run-gpu)")
    parser.add_argument(
        "--receipts-dir", default=DEFAULT_RECEIPTS_DIR,
        help="temporary directory for arm receipts and verdict")
    parser.add_argument(
        "--analyze", action="store_true",
        help="CPU comparison of the sealed age and s4 receipts")
    args = parser.parse_args(argv)

    if args.run_gpu:
        if args.analyze:
            parser.error("--analyze is CPU-only; omit --run-gpu")
        if args.fold_order is None:
            parser.error("--run-gpu requires --fold-order {age,s4}")
        receipt = run_gpu_arm(args.fold_order, args.receipts_dir)
        return 0 if receipt["fold_events"] >= MIN_FOLD_EVENTS else 1
    if args.fold_order is not None:
        parser.error("--fold-order requires --run-gpu")
    if args.analyze:
        verdict = run_analysis(args.receipts_dir)
        return 0 if verdict["pass"] else 1
    print(
        "No action flag: run CPU units with `python3 -m pytest "
        "tests/test_grm_s4_fold_order.py -q`. GPU: --run-gpu --fold-order "
        "{age,s4}. Analysis: --analyze.",
        flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
