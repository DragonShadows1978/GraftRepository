"""S4 paging A/B CPU contracts and lead-run GPU harness.

Default pytest collection is CPU-only.  The GPU/model imports occur only
behind ``--run-gpu``.  Each policy invocation replays the same four repeat-
probe fixtures as one repository session; all fixture prefixes (including
the early probes) run before the 16 late probes so the registered recall
measurement happens under the long-session paging pressure.

Lead commands (one model process at a time)::

    python3 tests/test_grm_s4_demotion.py --run-gpu --policy lru \
        --vram-budget-mb 12
    python3 tests/test_grm_s4_demotion.py --run-gpu --policy s4_protect \
        --vram-budget-mb 12
    python3 tests/test_grm_s4_demotion.py --analyze \
        --baseline-policy lru --candidate-policy s4_protect

Receipts default to ``/tmp/graftrepo_s4_demotion_receipts`` so the gate does
not mutate the fixture tree.
"""
import argparse
import glob
import hashlib
import json
import os
import sys
from types import SimpleNamespace

import pytest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tests.test_grm_importance_g1g2 import (
    next_probe_answer_turn,
    next_scripted_deposit_turn,
    scripted_deposit_text,
    wipe_workspace,
)


FIXTURES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fixtures",
    "importance_convos_repeat")
DEFAULT_RECEIPTS_DIR = "/tmp/graftrepo_s4_demotion_receipts"
POLICY_RECEIPT_SCHEMA = "grm_s4_demotion_policy_v1"
VERDICT_SCHEMA = "grm_s4_demotion_verdict_v1"
POLICIES = ("lru", "s4", "s4_protect")
DEFAULT_BASELINE_POLICY = "lru"
DEFAULT_CANDIDATE_POLICY = "s4_protect"
MIN_OVERCOMMIT = 3.0
DRIVER_ID = "four_repeat_prefixes_then_late_v1"
NGEN = 48
MAX_TRIPS = 2


# ---------------------------------------------------------------------------
# CPU: policy ordering and pager progress
# ---------------------------------------------------------------------------

def _pager_double(hit_counts, last_mounted, *, keep, policy="s4"):
    """Make the smallest repository double that exercises the real _page."""
    from core.graft_repository import GraftRepository

    grafts = []
    for idx, (hits, mounted) in enumerate(zip(hit_counts, last_mounted)):
        grafts.append({
            "h": object(),
            "host_payload": {"resident": idx},
            "last_used": mounted,
            "metadata": {"importance": {"s4": {
                "n_grounded": hits,
            }}},
        })
    repo = object.__new__(GraftRepository)
    repo.vram_budget = int(keep)
    if policy is not None:
        repo.spill_policy = policy
    repo.arena = SimpleNamespace(grafts=grafts)
    repo._node_bytes = lambda _graft: 1
    repo._ensure_host_payload = lambda _idx, _graft: None
    repo._ensure_lifecycle = lambda _idx, _graft: None
    evicted = []
    repo._native_evict_device_copy = evicted.append
    return repo, evicted


def test_s4_spills_zero_hit_first_with_lru_tiebreak():
    repo, evicted = _pager_double(
        [8, 0, 1, 0], [1, 2, 3, 4], keep=1, policy="s4")

    assert repo._page() == 3
    # The two zero-hit nodes leave first in last-mounted order.  Positive
    # hit counts share one class, so 8 versus 1 does not become a score.
    assert evicted == [1, 3, 0]
    assert [i for i, graft in enumerate(repo.arena.grafts)
            if graft["h"] is not None] == [2]


def test_s4_protect_spills_unprotected_in_lru_order_before_protected():
    repo, evicted = _pager_double(
        [8, 0, 1, 0], [1, 2, 3, 4], keep=2, policy="s4_protect")

    assert repo._page() == 2
    # Node 0 is the oldest resident but grounded, so it survives while the
    # unprotected nodes leave in their own mount-LRU order.
    assert evicted == [1, 3]
    assert [i for i, graft in enumerate(repo.arena.grafts)
            if graft["h"] is not None] == [0, 2]


def test_s4_protect_all_protected_degrades_to_pure_lru():
    repo, evicted = _pager_double(
        [8, 1, 4, 2], [4, 1, 3, 2], keep=2, policy="s4_protect")

    assert repo._page() == 2
    assert evicted == [1, 3]


def test_s4_protect_spills_protected_after_unprotected_are_exhausted():
    repo, evicted = _pager_double(
        [2, 0, 1, 0], [3, 4, 1, 2], keep=0, policy="s4_protect")

    assert repo._page() == 4
    # The unprotected pair goes first; once none remains, the protected
    # residents spill in exactly the original LRU order.  No policy jam or
    # permanent protection is possible under pressure.
    assert evicted == [3, 1, 2, 0]
    assert all(graft["h"] is None for graft in repo.arena.grafts)


@pytest.mark.parametrize("policy", ["lru", None])
def test_flag_off_is_pure_last_mounted_lru(policy):
    repo, evicted = _pager_double(
        [9, 0, 7, 0], [1, 2, 3, 4], keep=2, policy=policy)

    assert repo._page() == 2
    assert evicted == [0, 1]


@pytest.mark.parametrize(
    "policy, expected",
    [("lru", [0, 1]), (None, [0, 1]), ("s4", [1, 3])],
    ids=["lru", "flag-absent", "existing-s4"],
)
def test_s4_protect_leaves_existing_policy_orders_unchanged(policy, expected):
    repo, evicted = _pager_double(
        [9, 0, 7, 0], [1, 2, 3, 4], keep=2, policy=policy)

    assert repo._page() == 2
    assert evicted == expected


@pytest.mark.parametrize(
    "hit_counts",
    [
        [0, 0, 0, 0, 0, 0],
        [9, 1, 4, 2, 8, 3],
    ],
    ids=["all-zero-hit", "all-hit"],
)
def test_s4_pager_makes_progress_for_single_class_populations(hit_counts):
    last_mounted = [6, 1, 5, 2, 4, 3]
    repo, evicted = _pager_double(
        hit_counts, last_mounted, keep=2, policy="s4")

    assert repo._page() == 4
    assert evicted == [1, 3, 5, 4]
    assert sum(graft["h"] is not None for graft in repo.arena.grafts) == 2


def test_spill_policy_normalization_is_opt_in_and_strict():
    from core.graft_repository import GraftRepository

    assert GraftRepository._normalize_spill_policy(None) == "lru"
    assert GraftRepository._normalize_spill_policy("LRU") == "lru"
    assert GraftRepository._normalize_spill_policy("s4") == "s4"
    assert GraftRepository._normalize_spill_policy("S4_PROTECT") == "s4_protect"
    with pytest.raises(ValueError, match="spill policy"):
        GraftRepository._normalize_spill_policy("grounded-score")


# ---------------------------------------------------------------------------
# CPU: receipt schema and the exactly registered comparison
# ---------------------------------------------------------------------------

def _probe_contract(record):
    return (
        record["conversation_id"],
        int(record["probe_turn_id"]),
        int(record["target_turn_id"]),
        tuple(record["expected_answer_tokens"]),
    )


def make_policy_receipt(
        policy, *, session_fingerprint, conversation_ids, budget_bytes,
        total_node_bytes, n_nodes, page_ins, late_probes, early_probes=(),
        page_ins_after_prefix=0, page_outs=0):
    if policy not in POLICIES:
        raise ValueError(f"unknown receipt policy {policy!r}")
    late_probes = [dict(record) for record in late_probes]
    early_probes = [dict(record) for record in early_probes]
    late_hits = sum(bool(record["hit"]) for record in late_probes)
    early_hits = sum(bool(record["hit"]) for record in early_probes)
    budget_bytes = int(budget_bytes)
    total_node_bytes = int(total_node_bytes)
    if budget_bytes <= 0:
        raise ValueError("budget_bytes must be positive")
    return {
        "schema": POLICY_RECEIPT_SCHEMA,
        "policy": policy,
        "session": {
            "driver_id": DRIVER_ID,
            "fingerprint": str(session_fingerprint),
            "repository_sessions": 1,
            "conversation_ids": list(conversation_ids),
            "n_conversations": len(conversation_ids),
            "n_nodes": int(n_nodes),
            "n_early_probes": len(early_probes),
            "n_late_probes": len(late_probes),
        },
        "overcommit": {
            "total_node_bytes": total_node_bytes,
            "budget_bytes": budget_bytes,
            "ratio": total_node_bytes / budget_bytes,
            "minimum": MIN_OVERCOMMIT,
        },
        "early_probe_recall": {
            "hits": early_hits,
            "total": len(early_probes),
            "rate": (early_hits / len(early_probes)
                     if early_probes else None),
        },
        "late_probe_recall": {
            "hits": late_hits,
            "total": len(late_probes),
            "rate": (late_hits / len(late_probes)
                     if late_probes else None),
        },
        # Registered page-in comparison is over the complete long-session
        # replay.  The phase split is report-only diagnostic detail.
        "page_ins": int(page_ins),
        "page_ins_by_phase": {
            "prefix_including_early": int(page_ins_after_prefix),
            "late_probes": int(page_ins) - int(page_ins_after_prefix),
        },
        "page_outs": int(page_outs),
        "paging_fired": int(page_ins) > 0,
        "early_probes": early_probes,
        "late_probes": late_probes,
    }


def validate_policy_receipt(receipt):
    if not isinstance(receipt, dict):
        raise ValueError("policy receipt must be a JSON object")
    if receipt.get("schema") != POLICY_RECEIPT_SCHEMA:
        raise ValueError(
            f"receipt schema {receipt.get('schema')!r}, expected "
            f"{POLICY_RECEIPT_SCHEMA!r}")
    if receipt.get("policy") not in POLICIES:
        raise ValueError(f"invalid receipt policy {receipt.get('policy')!r}")
    session = receipt.get("session")
    overcommit = receipt.get("overcommit")
    recall = receipt.get("late_probe_recall")
    probes = receipt.get("late_probes")
    if not isinstance(session, dict) or not session.get("fingerprint"):
        raise ValueError("receipt missing session fingerprint")
    if session.get("repository_sessions") != 1:
        raise ValueError("receipt must represent exactly one repository session")
    if not isinstance(overcommit, dict):
        raise ValueError("receipt missing overcommit object")
    total_bytes = overcommit.get("total_node_bytes")
    budget_bytes = overcommit.get("budget_bytes")
    if (not isinstance(total_bytes, int) or isinstance(total_bytes, bool)
            or not isinstance(budget_bytes, int)
            or isinstance(budget_bytes, bool) or budget_bytes <= 0):
        raise ValueError("receipt has invalid overcommit byte counts")
    expected_ratio = total_bytes / budget_bytes
    if float(overcommit.get("ratio", -1.0)) != pytest.approx(expected_ratio):
        raise ValueError("receipt overcommit ratio disagrees with byte counts")
    if overcommit.get("minimum") != MIN_OVERCOMMIT:
        raise ValueError("receipt changed the registered overcommit minimum")
    if not isinstance(recall, dict) or not isinstance(probes, list):
        raise ValueError("receipt missing late-probe recall records")
    hits = sum(bool(record.get("hit")) for record in probes)
    if recall.get("hits") != hits or recall.get("total") != len(probes):
        raise ValueError("late-probe recall summary disagrees with records")
    expected_rate = hits / len(probes) if probes else None
    if expected_rate is None:
        if recall.get("rate") is not None:
            raise ValueError("empty late-probe set must have null rate")
    elif float(recall.get("rate", -1.0)) != pytest.approx(expected_rate):
        raise ValueError("late-probe recall rate disagrees with records")
    page_ins = receipt.get("page_ins")
    if (not isinstance(page_ins, int) or isinstance(page_ins, bool)
            or page_ins < 0):
        raise ValueError("receipt page_ins must be a nonnegative integer")
    if bool(receipt.get("paging_fired")) != (page_ins > 0):
        raise ValueError("paging_fired disagrees with page_ins")
    for record in probes:
        _probe_contract(record)
    return receipt


def _receipt_path(receipts_dir, policy):
    return os.path.join(receipts_dir, f"grm_s4_demotion_{policy}.json")


def write_policy_receipt(receipt, receipts_dir=DEFAULT_RECEIPTS_DIR):
    validate_policy_receipt(receipt)
    os.makedirs(receipts_dir, exist_ok=True)
    path = _receipt_path(receipts_dir, receipt["policy"])
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def _policy_pair(baseline_policy=DEFAULT_BASELINE_POLICY,
                 candidate_policy=DEFAULT_CANDIDATE_POLICY):
    if baseline_policy not in POLICIES:
        raise ValueError(f"unknown baseline policy {baseline_policy!r}")
    if candidate_policy not in POLICIES:
        raise ValueError(f"unknown candidate policy {candidate_policy!r}")
    if baseline_policy == candidate_policy:
        raise ValueError("baseline and candidate policies must differ")
    return baseline_policy, candidate_policy


def load_policy_receipts(
        receipts_dir=DEFAULT_RECEIPTS_DIR,
        policies=(DEFAULT_BASELINE_POLICY, DEFAULT_CANDIDATE_POLICY)):
    policies = tuple(policies)
    if not policies:
        raise ValueError("at least one receipt policy is required")
    if len(set(policies)) != len(policies):
        raise ValueError("receipt policies must be distinct")
    unknown = [policy for policy in policies if policy not in POLICIES]
    if unknown:
        raise ValueError(f"unknown receipt policies {unknown!r}")
    receipts = {}
    for policy in policies:
        path = _receipt_path(receipts_dir, policy)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                receipt = json.load(handle)
        except FileNotFoundError as exc:
            raise RuntimeError(f"missing {policy} receipt: {path}") from exc
        validate_policy_receipt(receipt)
        if receipt["policy"] != policy:
            raise ValueError(
                f"{path}: contains policy {receipt['policy']!r}, expected "
                f"{policy!r}")
        receipts[policy] = receipt
    return receipts


def analyze_policy_receipts(
        receipts, *, baseline_policy=DEFAULT_BASELINE_POLICY,
        candidate_policy=DEFAULT_CANDIDATE_POLICY):
    baseline_policy, candidate_policy = _policy_pair(
        baseline_policy, candidate_policy)
    if set(receipts) != {baseline_policy, candidate_policy}:
        raise ValueError(
            "analysis requires exactly the named baseline and candidate "
            "receipts")
    baseline = validate_policy_receipt(receipts[baseline_policy])
    candidate = validate_policy_receipt(receipts[candidate_policy])

    baseline_contract = [
        _probe_contract(record) for record in baseline["late_probes"]]
    candidate_contract = [
        _probe_contract(record) for record in candidate["late_probes"]]
    same_session = bool(
        baseline["session"]["fingerprint"]
        == candidate["session"]["fingerprint"]
        and baseline["session"]["driver_id"]
        == candidate["session"]["driver_id"]
        and baseline["session"]["conversation_ids"]
        == candidate["session"]["conversation_ids"]
        and baseline["session"]["n_nodes"]
        == candidate["session"]["n_nodes"]
        and baseline["overcommit"]["total_node_bytes"]
        == candidate["overcommit"]["total_node_bytes"])
    same_budget = bool(
        baseline["overcommit"]["budget_bytes"]
        == candidate["overcommit"]["budget_bytes"])
    same_probes = baseline_contract == candidate_contract
    overcommit_ok = {
        policy: receipt["overcommit"]["ratio"] >= MIN_OVERCOMMIT
        for policy, receipt in receipts.items()
    }
    paging_fired = {
        policy: receipt["page_ins"] > 0
        for policy, receipt in receipts.items()
    }
    prerequisites = bool(
        same_session and same_budget and same_probes
        and all(overcommit_ok.values()) and all(paging_fired.values()))

    recall_noninferior = bool(
        candidate["late_probe_recall"]["hits"]
        >= baseline["late_probe_recall"]["hits"]
        and candidate["late_probe_recall"]["total"]
        == baseline["late_probe_recall"]["total"])
    page_ins_noninferior = candidate["page_ins"] <= baseline["page_ins"]
    gate_pass = bool(
        prerequisites and recall_noninferior and page_ins_noninferior)
    equivalence = bool(
        gate_pass
        and candidate["late_probe_recall"]["hits"]
        == baseline["late_probe_recall"]["hits"]
        and candidate["page_ins"] == baseline["page_ins"])
    note = None
    if equivalence:
        note = (f"PASS-by-equivalence: {candidate_policy} is free but not "
                "yet advantageous at this scale.")
    elif not prerequisites:
        note = ("INVALID gate prerequisites; "
                f"{candidate_policy} versus {baseline_policy} is RED.")

    def arm_summary(receipt):
        return {
            "late_probe_hits": receipt["late_probe_recall"]["hits"],
            "late_probe_total": receipt["late_probe_recall"]["total"],
            "late_probe_recall": receipt["late_probe_recall"]["rate"],
            "page_ins": receipt["page_ins"],
            "overcommit_ratio": receipt["overcommit"]["ratio"],
        }

    return {
        "schema": VERDICT_SCHEMA,
        "comparison": {
            "baseline_policy": baseline_policy,
            "candidate_policy": candidate_policy,
        },
        "registered_gate": {
            "minimum_overcommit": MIN_OVERCOMMIT,
            "late_probe_recall": (
                f"{candidate_policy} >= {baseline_policy}"),
            "page_ins": f"{candidate_policy} <= {baseline_policy}",
        },
        "arms": {
            baseline_policy: arm_summary(baseline),
            candidate_policy: arm_summary(candidate),
        },
        "prerequisites": {
            "same_session": same_session,
            "same_budget": same_budget,
            "same_probes": same_probes,
            "overcommit_at_least_3x": overcommit_ok,
            "paging_fired": paging_fired,
            "pass": prerequisites,
        },
        "conditions": {
            "late_probe_recall_noninferior": recall_noninferior,
            "page_ins_noninferior": page_ins_noninferior,
        },
        "margins": {
            (f"late_probe_hits_{candidate_policy}_minus_"
             f"{baseline_policy}"): (
                candidate["late_probe_recall"]["hits"]
                - baseline["late_probe_recall"]["hits"]),
            (f"page_ins_{baseline_policy}_minus_{candidate_policy}"):
                baseline["page_ins"] - candidate["page_ins"],
        },
        "pass_by_equivalence": equivalence,
        "note": note,
        "pass": gate_pass,
    }


def _synthetic_probe(index, hit):
    return {
        "conversation_id": "convo_synthetic",
        "probe_turn_id": 100 + index,
        "target_turn_id": index,
        "expected_answer_tokens": [f"token-{index}"],
        "answer": f"token-{index}" if hit else "miss",
        "hit": bool(hit),
    }


def _synthetic_receipt(policy, *, hits=2, page_ins=5):
    probes = [_synthetic_probe(i, i < hits) for i in range(4)]
    return make_policy_receipt(
        policy,
        session_fingerprint="same-session",
        conversation_ids=["convo_synthetic"],
        budget_bytes=100,
        total_node_bytes=300,
        n_nodes=12,
        page_ins=page_ins,
        late_probes=probes,
        page_ins_after_prefix=2,
        page_outs=10,
    )


def test_policy_receipts_round_trip_and_reject_wrong_schema(tmp_path):
    expected = {
        policy: _synthetic_receipt(policy, page_ins=5 - offset)
        for offset, policy in enumerate(POLICIES)
    }
    for receipt in expected.values():
        write_policy_receipt(receipt, str(tmp_path))
    assert load_policy_receipts(str(tmp_path), policies=POLICIES) == expected
    assert load_policy_receipts(
        str(tmp_path), policies=("lru", "s4_protect")) == {
            "lru": expected["lru"],
            "s4_protect": expected["s4_protect"],
        }

    path = _receipt_path(str(tmp_path), "s4")
    broken = dict(expected["s4"])
    broken["schema"] = "wrong"
    path_obj = tmp_path / os.path.basename(path)
    path_obj.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        load_policy_receipts(str(tmp_path), policies=POLICIES)


def test_registered_gate_has_no_unregistered_positive_margin():
    verdict = analyze_policy_receipts({
        "lru": _synthetic_receipt("lru", hits=3, page_ins=8),
        "s4_protect": _synthetic_receipt(
            "s4_protect", hits=3, page_ins=8),
    })
    assert verdict["pass"] is True
    assert verdict["pass_by_equivalence"] is True
    assert verdict["comparison"] == {
        "baseline_policy": "lru",
        "candidate_policy": "s4_protect",
    }

    red = analyze_policy_receipts({
        "lru": _synthetic_receipt("lru", hits=3, page_ins=8),
        "s4_protect": _synthetic_receipt(
            "s4_protect", hits=3, page_ins=9),
    })
    assert red["pass"] is False
    assert red["conditions"]["late_probe_recall_noninferior"] is True
    assert red["conditions"]["page_ins_noninferior"] is False


def test_registered_gate_accepts_any_named_policy_pair():
    verdict = analyze_policy_receipts(
        {
            "lru": _synthetic_receipt("lru", hits=3, page_ins=8),
            "s4": _synthetic_receipt("s4", hits=3, page_ins=7),
        },
        candidate_policy="s4",
    )

    assert verdict["pass"] is True
    assert verdict["comparison"] == {
        "baseline_policy": "lru",
        "candidate_policy": "s4",
    }


def test_cli_accepts_s4_protect_without_loading_a_model(monkeypatch, tmp_path):
    seen = {}

    def fake_run_gpu_policy(policy, vram_budget_mb, receipts_dir):
        seen.update({
            "policy": policy,
            "vram_budget_mb": vram_budget_mb,
            "receipts_dir": receipts_dir,
        })
        return _synthetic_receipt(policy, page_ins=1)

    monkeypatch.setattr(sys.modules[__name__], "run_gpu_policy", fake_run_gpu_policy)

    assert _main([
        "--run-gpu",
        "--policy", "s4_protect",
        "--receipts-dir", str(tmp_path),
    ]) == 0
    assert seen == {
        "policy": "s4_protect",
        "vram_budget_mb": 12,
        "receipts_dir": str(tmp_path),
    }


# ---------------------------------------------------------------------------
# Lead-run GPU arm.  No import in this section executes during pytest.
# ---------------------------------------------------------------------------

def _fixture_paths():
    return sorted(glob.glob(os.path.join(FIXTURES_DIR, "convo_*.json")))


def _load_fixtures():
    fixtures = []
    for path in _fixture_paths():
        with open(path, "r", encoding="utf-8") as handle:
            fixtures.append(json.load(handle))
    if len(fixtures) != 4:
        raise RuntimeError(
            f"expected four repeat-probe fixtures, found {len(fixtures)}")
    return fixtures


def _target_turn_id(probe):
    targets = [int(turn_id) for turn_id, grade in probe["relevance"].items()
               if grade == 3]
    if len(targets) != 1:
        raise ValueError(
            f"probe {probe['probe_turn_id']} has grade-3 targets {targets}")
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
    if counts != {"early": 4, "late": 4} or len(seen) != 4:
        raise ValueError(
            f"{fixture['conversation_id']}: expected four paired probes, "
            f"got phases={counts}, targets={sorted(seen)}")
    return phases


def _session_fingerprint(fixtures):
    contract = {
        "driver_id": DRIVER_ID,
        "ngen": NGEN,
        "max_trips": MAX_TRIPS,
        "fixtures": fixtures,
    }
    encoded = json.dumps(
        contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _iter_scripted_deposits(fixture):
    turns = fixture["turns"]
    index = 0
    while index < len(turns):
        turn = turns[index]
        if turn["node_class"] == "PROBE":
            _answer, index = next_probe_answer_turn(turns, index)
            continue
        assistant, index = next_scripted_deposit_turn(turns, index)
        yield turn, assistant


def _planned_node_bytes(repo, fixtures):
    total = 0
    for fixture in fixtures:
        for user_turn, assistant_turn in _iter_scripted_deposits(fixture):
            ntok = len(repo.arena.encode(
                scripted_deposit_text(user_turn, assistant_turn)))
            total += repo._node_bytes({"ntok": ntok})
    return total


def _load_gpu_repository(workspace, policy, vram_budget_mb):
    tensor_path = "/mnt/ForgeRealm/Project-Tensor/tensor_cuda"
    if tensor_path not in sys.path:
        sys.path.insert(0, tensor_path)
    import numpy as np
    import tensor_cuda as tc
    from tokenizers import Tokenizer as HFTok
    from core.graft_repository import GraftRepository
    from core.minicpm3_tc import MiniCPM3_TC, _snap

    tokenizer = HFTok.from_file(os.path.join(_snap(), "tokenizer.json"))
    model, info = MiniCPM3_TC.from_pretrained()
    print(f"loaded: {info}", flush=True)
    with tc.no_grad():
        model(np.array([[1, 2, 3]], dtype=np.int64), kv_caches=None,
              position_offset=0, last_token_only=True)

    wipe_workspace(workspace)
    repo = GraftRepository(
        model,
        lambda text: tokenizer.encode(text).ids,
        lambda ids: tokenizer.decode(ids),
        workspace,
        autosave=False,
        vram_budget_mb=vram_budget_mb,
        librarian_mode="deferred",
        durability_mode="volatile_fast",
        spill_policy=policy,
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
    # The paging gate isolates the consumer.  Prevent deferred librarian
    # backpressure from folding nodes during this one long replay.
    repo.TURNS_HIGH = 10 ** 9
    repo.DIGESTS_HIGH = 10 ** 9
    for layer in model.layers:
        layer.self_attn.live_shift = repo.arena.live_shift
        layer.self_attn.absorbed_decode = True
    return repo


def _s4_grounded_count(arena, idx):
    s4 = (arena.grafts[idx].get("metadata", {})
          .get("importance", {}).get("s4"))
    return int(s4.get("n_grounded", 0)) if isinstance(s4, dict) else 0


def _deposit_scripted(repo, fixture, user_turn, assistant_turn,
                      turn_to_graft):
    arena = repo.arena
    before_len = len(arena.grafts)
    if assistant_turn is not None:
        repo.add_turn(user_turn["text"], assistant_turn["text"])
        result = repo.runtime.last_result
    else:
        before = repo._snapshot_state()
        arena.feed(scripted_deposit_text(user_turn))
        repo._set_new_node_provenance(before, "exchange_span")
        extracted = repo._extract_from_new_turns(
            before,
            context={
                "event": "add_turn",
                "user_text": user_turn["text"],
                "assistant_text": None,
            },
        )
        result = repo.runtime._finish_turn_event(
            "add_turn", before, extraction=extracted, autosave=False)
        repo._queue_s2_pending()
    if len(result.new_nodes) != 1 or result.new_nodes[0] != before_len:
        raise RuntimeError(
            f"{fixture['conversation_id']} turn {user_turn['turn_id']}: "
            f"expected one node at {before_len}, got {result.new_nodes}")
    graft_idx = result.new_nodes[0]
    key = (fixture["conversation_id"], int(user_turn["turn_id"]))
    turn_to_graft[key] = graft_idx
    if assistant_turn is not None:
        assistant_key = (
            fixture["conversation_id"], int(assistant_turn["turn_id"]))
        turn_to_graft[assistant_key] = graft_idx
    return int(result.paged or 0)


def _drive_probe(repo, fixture, probe, phase, turn_to_graft):
    arena = repo.arena
    target_turn = _target_turn_id(probe)
    target_node = turn_to_graft[(fixture["conversation_id"], target_turn)]
    before_grounded = _s4_grounded_count(arena, target_node)
    before_page_ins = int(getattr(arena, "page_ins", 0))
    answer, info = arena.step(
        probe["question"], ngen=NGEN, deposit=False, max_trips=MAX_TRIPS)
    page_outs = int(repo._page() or 0)
    after_page_ins = int(getattr(arena, "page_ins", 0))
    after_grounded = _s4_grounded_count(arena, target_node)
    expected = [str(token).lower()
                for token in probe["expected_answer_tokens"]]
    answer_lower = answer.lower()
    hit = all(token in answer_lower for token in expected)
    record = {
        "conversation_id": fixture["conversation_id"],
        "phase": phase,
        "probe_turn_id": int(probe["probe_turn_id"]),
        "target_turn_id": target_turn,
        "target_node": int(target_node),
        "expected_answer_tokens": expected,
        "answer": answer,
        "hit": bool(hit),
        "page_ins_delta": after_page_ins - before_page_ins,
        "page_outs_after_probe": page_outs,
        "mounts_one_based": list(info.get("mounts", ())),
        "trip": int(info.get("trip", 0)),
        "target_n_grounded_before": before_grounded,
        "target_n_grounded_after": after_grounded,
    }
    print(
        f"  {phase} {fixture['conversation_id']}#"
        f"{probe['probe_turn_id']}: {'HIT' if hit else 'MISS'} "
        f"page_ins+={record['page_ins_delta']} mounts="
        f"{record['mounts_one_based']} answer={answer[:100]!r}",
        flush=True,
    )
    return record, page_outs


def _drive_fixture_prefix(repo, fixture, turn_to_graft):
    turns = fixture["turns"]
    probes = {int(probe["probe_turn_id"]): probe
              for probe in fixture["probes"]}
    phases = _probe_phases(fixture)
    early_records = []
    late_jobs = []
    page_outs = 0
    index = 0
    while index < len(turns):
        turn = turns[index]
        if turn["node_class"] == "PROBE":
            probe = probes[int(turn["turn_id"])]
            _scripted_answer, index = next_probe_answer_turn(turns, index)
            phase = phases[int(turn["turn_id"])]
            if phase == "late":
                late_jobs.append((fixture, probe))
            else:
                record, paged = _drive_probe(
                    repo, fixture, probe, phase, turn_to_graft)
                early_records.append(record)
                page_outs += paged
            continue
        assistant, index = next_scripted_deposit_turn(turns, index)
        page_outs += _deposit_scripted(
            repo, fixture, turn, assistant, turn_to_graft)
    return early_records, late_jobs, page_outs


def run_gpu_policy(policy, vram_budget_mb,
                   receipts_dir=DEFAULT_RECEIPTS_DIR):
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}")
    if int(vram_budget_mb) <= 0:
        raise ValueError("vram_budget_mb must be positive")
    fixtures = _load_fixtures()
    workspace = f"/tmp/graftrepo_s4_demotion_{policy}"
    repo = _load_gpu_repository(workspace, policy, int(vram_budget_mb))
    planned_bytes = _planned_node_bytes(repo, fixtures)
    planned_ratio = planned_bytes / repo.vram_budget
    print(
        f"policy={policy} planned_node_bytes={planned_bytes} "
        f"budget_bytes={repo.vram_budget} "
        f"overcommit={planned_ratio:.3f}x",
        flush=True,
    )
    if planned_ratio < MIN_OVERCOMMIT:
        raise RuntimeError(
            f"registered >=3x overcommit not met: {planned_ratio:.3f}x; "
            "rerun both arms with the same smaller --vram-budget-mb")

    turn_to_graft = {}
    early_records = []
    late_jobs = []
    page_outs = 0
    for fixture in fixtures:
        print(f"prefix: {fixture['conversation_id']}", flush=True)
        early, jobs, paged = _drive_fixture_prefix(
            repo, fixture, turn_to_graft)
        early_records.extend(early)
        late_jobs.extend(jobs)
        page_outs += paged
    if len(early_records) != 16 or len(late_jobs) != 16:
        raise RuntimeError(
            f"expected 16 early and 16 late probes, got "
            f"{len(early_records)} and {len(late_jobs)}")

    page_ins_after_prefix = int(getattr(repo.arena, "page_ins", 0))
    late_records = []
    print("late probes under full-session pressure", flush=True)
    for fixture, probe in late_jobs:
        record, paged = _drive_probe(
            repo, fixture, probe, "late", turn_to_graft)
        late_records.append(record)
        page_outs += paged

    total_node_bytes = sum(
        repo._node_bytes(graft) for graft in repo.arena.grafts
        if not graft.get("retired"))
    if total_node_bytes != planned_bytes:
        raise RuntimeError(
            f"session shape changed: planned {planned_bytes} node bytes, "
            f"observed {total_node_bytes}")
    receipt = make_policy_receipt(
        policy,
        session_fingerprint=_session_fingerprint(fixtures),
        conversation_ids=[fixture["conversation_id"] for fixture in fixtures],
        budget_bytes=repo.vram_budget,
        total_node_bytes=total_node_bytes,
        n_nodes=len(repo.arena.grafts),
        page_ins=int(getattr(repo.arena, "page_ins", 0)),
        late_probes=late_records,
        early_probes=early_records,
        page_ins_after_prefix=page_ins_after_prefix,
        page_outs=page_outs,
    )
    out_path = write_policy_receipt(receipt, receipts_dir)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    print(f"wrote {out_path}", flush=True)
    if not receipt["paging_fired"]:
        print(f"{policy.upper()} ARM INVALID: page-ins did not fire", flush=True)
    else:
        print(
            f"{policy.upper()} ARM COMPLETE: late recall "
            f"{receipt['late_probe_recall']['hits']}/"
            f"{receipt['late_probe_recall']['total']}, "
            f"page-ins={receipt['page_ins']}",
            flush=True,
        )
    return receipt


def run_analysis(receipts_dir=DEFAULT_RECEIPTS_DIR,
                 baseline_policy=DEFAULT_BASELINE_POLICY,
                 candidate_policy=DEFAULT_CANDIDATE_POLICY):
    baseline_policy, candidate_policy = _policy_pair(
        baseline_policy, candidate_policy)
    receipts = load_policy_receipts(
        receipts_dir, policies=(baseline_policy, candidate_policy))
    verdict = analyze_policy_receipts(
        receipts,
        baseline_policy=baseline_policy,
        candidate_policy=candidate_policy)
    for policy in (baseline_policy, candidate_policy):
        arm = verdict["arms"][policy]
        print(
            f"{policy.upper()}: late recall={arm['late_probe_hits']}/"
            f"{arm['late_probe_total']} ({arm['late_probe_recall']:.4f}), "
            f"page-ins={arm['page_ins']}, "
            f"overcommit={arm['overcommit_ratio']:.3f}x",
            flush=True,
        )
    print(
        f"{candidate_policy.upper()} vs {baseline_policy.upper()} conditions: "
        "recall(B)>=recall(A)="
        f"{verdict['conditions']['late_probe_recall_noninferior']}; "
        "page-ins(B)<=page-ins(A)="
        f"{verdict['conditions']['page_ins_noninferior']}; "
        f"prerequisites={verdict['prerequisites']['pass']}; "
        f"PASS={verdict['pass']}",
        flush=True,
    )
    if verdict["note"]:
        print(verdict["note"], flush=True)
    os.makedirs(receipts_dir, exist_ok=True)
    verdict_path = os.path.join(
        receipts_dir, "grm_s4_demotion_verdict.json")
    with open(verdict_path, "w", encoding="utf-8") as handle:
        json.dump(verdict, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(verdict, sort_keys=True), flush=True)
    print(f"wrote {verdict_path}", flush=True)
    return verdict


def _main(argv=None):
    parser = argparse.ArgumentParser(
        description="S4 paging A/B GPU arms and CPU analysis")
    parser.add_argument(
        "--run-gpu", action="store_true",
        help="authorize one lead-run GPU policy arm")
    parser.add_argument(
        "--policy", choices=POLICIES,
        help="pager arm to replay (requires --run-gpu)")
    parser.add_argument(
        "--baseline-policy", choices=POLICIES,
        help=("baseline arm for --analyze "
              f"(default {DEFAULT_BASELINE_POLICY})"))
    parser.add_argument(
        "--candidate-policy", choices=POLICIES,
        help=("candidate arm for --analyze "
              f"(default {DEFAULT_CANDIDATE_POLICY})"))
    parser.add_argument(
        "--vram-budget-mb", type=int, default=12,
        help="device-node budget; both arms must use the same value")
    parser.add_argument(
        "--receipts-dir", default=DEFAULT_RECEIPTS_DIR,
        help="directory for policy receipts and verdict")
    parser.add_argument(
        "--analyze", action="store_true",
        help="CPU comparison of the two sealed policy receipts")
    args = parser.parse_args(argv)

    if args.run_gpu:
        if args.analyze:
            parser.error("--analyze is CPU-only; omit --run-gpu")
        if (args.baseline_policy is not None
                or args.candidate_policy is not None):
            parser.error("--baseline-policy/--candidate-policy require --analyze")
        if args.policy is None:
            parser.error("--run-gpu requires --policy {lru,s4,s4_protect}")
        receipt = run_gpu_policy(
            args.policy, args.vram_budget_mb, args.receipts_dir)
        valid = bool(
            receipt["overcommit"]["ratio"] >= MIN_OVERCOMMIT
            and receipt["paging_fired"])
        return 0 if valid else 1
    if args.policy is not None:
        parser.error("--policy requires --run-gpu")
    if args.analyze:
        verdict = run_analysis(
            args.receipts_dir,
            baseline_policy=(args.baseline_policy or DEFAULT_BASELINE_POLICY),
            candidate_policy=(args.candidate_policy
                              or DEFAULT_CANDIDATE_POLICY))
        return 0 if verdict["pass"] else 1
    if args.baseline_policy is not None or args.candidate_policy is not None:
        parser.error("--baseline-policy/--candidate-policy require --analyze")
    print(
        "No action flag: run CPU units with `python3 -m pytest "
        "tests/test_grm_s4_demotion.py -q`. GPU: --run-gpu --policy "
        "{lru,s4,s4_protect}. Analysis: --analyze "
        "[--baseline-policy POLICY --candidate-policy POLICY].",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main())
