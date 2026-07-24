import pytest

from core.graft_arena import ArenaCache
from core.grm_three_pass import (
    MEMORY_LEDGER_SCHEMA,
    MemoryLedgerBuilder,
    Pass2ArenaMutationError,
    Pass2ReadOnlyGuard,
    StagedWorkingSetResolver,
    TurnStepIOTracker,
    arena_state_bytes,
)


class FakeArena:
    def __init__(self):
        self.grafts = []
        self._s4_turn = 0
        self._cuda_gqa_epoch = 0


class FakeRepository:
    def __init__(self):
        self.arena = FakeArena()


def _node(text, *, active=True):
    return {
        "text": text,
        "ntok": len(text.split()),
        "kind": "turn",
        "retired": not active,
        "metadata": {
            "active": active,
            "importance": {},
            "supersedes": [],
        },
        "provenance": [],
    }


def test_pass2_guard_allows_read_side_caches_and_rejects_persistent_mutation():
    repo = FakeRepository()
    repo.arena.grafts.append(_node("alpha"))
    with Pass2ReadOnlyGuard(repo) as guard:
        repo.arena.grafts[0]["rare"] = {"alpha"}
        repo.arena.grafts[0]["last_used"] = 7
    assert guard.read_only is True
    assert guard.before_sha256 == guard.after_sha256

    with pytest.raises(Pass2ArenaMutationError, match="pass 2 mutated"):
        with Pass2ReadOnlyGuard(repo):
            repo.arena.grafts[0]["metadata"]["active"] = False


def test_memory_ledger_receipts_are_frozen_schema_and_complete():
    repo = FakeRepository()
    builder = MemoryLedgerBuilder(
        repo,
        session_id="session-a",
        turn_id="0",
        request_text="remember alpha",
        output_text="ack",
    )

    def deposit():
        repo.arena.grafts.append(_node("alpha"))
        repo.arena._cuda_gqa_epoch += 1
        repo.arena._s4_turn += 1

    builder.record_operation(
        "deposit",
        reason_code="test_deposit",
        reason_detail="fixture",
        source_text="alpha",
        operation=deposit,
    )
    receipt, audit = builder.finalize()

    assert set(receipt) == {
        "schema", "session_id", "turn_id", "turn_pipeline", "pass",
        "provenance", "mutations", "mutation_count",
    }
    assert receipt["schema"] == MEMORY_LEDGER_SCHEMA
    assert receipt["turn_pipeline"] == "three_pass"
    assert receipt["pass"] == 3
    assert receipt["mutation_count"] == len(receipt["mutations"])
    assert [m["sequence"] for m in receipt["mutations"]] == list(
        range(receipt["mutation_count"]))
    assert {m["target"]["arena"] for m in receipt["mutations"]} == {
        "arena_control", "grafts"}
    assert all(len(m["provenance"]["source_sha256"]) == 64
               for m in receipt["mutations"])
    assert audit["complete"] is True
    assert audit["missing_targets"] == []


def test_deferred_turn_deposit_and_importance_match_legacy_blocks():
    arena = ArenaCache.__new__(ArenaCache)
    arena.cache_deposits = False
    arena.grafts = [_node("source ALPHA-1")]
    arena.live_segs = [(None, 4)]
    arena._s4_turn = 0
    arena._cuda_gqa_epoch = 0
    arena.s4_metadata_callback = None
    arena._rare_tokens = lambda text: {
        word.lower() for word in str(text).split() if "-" in word}

    def deposit(text):
        arena.grafts.append(_node(text))
        arena._cuda_gqa_epoch += 1
        return len(arena.grafts) - 1

    arena.deposit = deposit
    info = {
        "_deferred_memory": {
            "turn_text": "question answer",
            "user_text": "question",
            "seg_cache_ntok": 4,
            "picks": [0],
            "deposited": False,
            "importance_committed": False,
            "importance_bookkeeping": {
                "routed": [0],
                "mounted": [0],
                "grounded_mounts": [0],
                "turn": 1,
            },
        }
    }

    node_id = arena.deposit_deferred_turn(info)
    changed = arena.commit_deferred_importance(info)

    assert node_id == 1
    assert arena.live_segs == [(1, 4)]
    assert info["_deferred_memory"]["deposited"] is True
    assert info["_deferred_memory"]["importance_committed"] is True
    assert changed == (0,)
    s4 = arena.grafts[0]["metadata"]["importance"]["s4"]
    assert s4 == {
        "n_routed": 1,
        "n_mounted": 1,
        "n_grounded": 1,
        "last_grounded_turn": 1,
    }


def test_deferred_cache_deposit_consumes_prepared_route_key():
    arena = ArenaCache.__new__(ArenaCache)
    arena.cache_deposits = True
    arena.grafts = []
    arena.live_segs = [(None, 3)]
    route_key = object()
    arena._deferred_route_keys = {7: route_key}
    seen = {}

    def deposit_from_cache(text, ntok, route_key=None):
        seen.update(text=text, ntok=ntok, route_key=route_key)
        arena.grafts.append(_node(text))
        return 0

    arena.deposit_from_cache = deposit_from_cache
    info = {
        "_deferred_memory": {
            "turn_text": "complete turn",
            "user_text": "complete",
            "seg_cache_ntok": 3,
            "picks": [],
            "deposited": False,
            "importance_committed": False,
            "route_key_prepared": True,
            "route_key_token": 7,
        }
    }

    assert arena.deposit_deferred_turn(info) == 0
    assert seen == {
        "text": "complete turn",
        "ntok": 3,
        "route_key": route_key,
    }
    assert arena.live_segs == [(0, 3)]
    assert arena._deferred_route_keys == {}


def test_staged_working_set_resolves_l1_and_counts_repository_fallback():
    class Arena:
        def __init__(self):
            self.grafts = [{"h": object()}, {"h": object()}, {"h": object()}]
            self.last_route_backend = "python"
            self.route_fallbacks = 0
            self.ensure_calls = []

        def route(self, _text, _exclude, limit=None, **_kwargs):
            self.route_fallbacks += 1
            return [2, 1, 0][:limit]

        def _ensure_h(self, ids):
            self.ensure_calls.append(list(ids))

    arena = Arena()
    with StagedWorkingSetResolver(
        arena,
        probe_text="probe",
        ranking_ids=[2, 1, 0],
        staged_ids=[2, 1],
    ) as resolver:
        assert arena.route("probe", exclude=set(), limit=2) == [2, 1]
        arena._ensure_h([2, 1])
        assert arena.route("probe", exclude=set(), limit=3) == [2, 1, 0]

    receipt = resolver.receipt()
    assert receipt["route_l1_calls"] == 1
    assert receipt["payload_l1_resolutions"] == 2
    assert receipt["l2_miss_count"] == 1
    assert receipt["l2_misses"][0] == {
        "kind": "route_id_not_staged",
        "node_id": 0,
    }
    assert arena.route_fallbacks == 1
    assert arena.ensure_calls == [[2, 1]]


def test_step_io_tracker_attributes_payload_and_route_bank_uploads():
    class Store:
        def configure_cuda_gqa_route_bank(
                self, route_bank, node_ids=None, **_kwargs):
            return (route_bank, node_ids)

    class Arena:
        def __init__(self):
            self.grafts = [{"h": None, "host_payload": {"k": "fixture"}}]
            self.native_store = Store()
            self.node_loader = self.load

        def load(self, node_id):
            payload = object()
            self.grafts[int(node_id)]["h"] = payload
            return payload

    arena = Arena()
    with TurnStepIOTracker(arena) as tracker:
        tracker.set_step("1_prep")
        arena.node_loader(0)
        arena.native_store.configure_cuda_gqa_route_bank(
            [[1.0]], [11])
        tracker.set_step("2_inference")

    receipt = tracker.receipt()
    assert receipt["steps"]["1_prep"]["page_in_count"] == 1
    assert receipt["steps"]["1_prep"]["upload_count"] == 2
    assert receipt["steps"]["2_inference"]["page_in_count"] == 0
    assert receipt["steps"]["2_inference"]["upload_count"] == 0


def _deposit(repo, text):
    repo.arena.grafts.append(_node(text))
    repo.arena._cuda_gqa_epoch += 1
    repo.arena._s4_turn += 1


def _supersede(repo, old, new):
    for graft in repo.arena.grafts:
        if graft["text"] == old and not graft["retired"]:
            graft["retired"] = True
            graft["metadata"]["active"] = False
    replacement = _node(new)
    replacement["metadata"]["supersedes"] = [old]
    repo.arena.grafts.append(replacement)
    repo.arena._cuda_gqa_epoch += 1


def _importance(repo, text):
    for graft in repo.arena.grafts:
        if graft["text"] == text and not graft["retired"]:
            importance = graft["metadata"]["importance"]
            importance["n_routed"] = importance.get("n_routed", 0) + 1


def _active_value(repo):
    active = [g["text"] for g in repo.arena.grafts if not g["retired"]]
    return active[-1] if active else None


def test_scripted_g_equiv_and_g_compact_schedule():
    events = [
        ("deposit", "alpha"),
        ("supersede", ("alpha", "beta")),
        ("importance", "beta"),
    ]
    outputs = ["ack-alpha", "ack-beta", "beta"]

    single = FakeRepository()
    for kind, payload in events:
        if kind == "deposit":
            _deposit(single, payload)
        elif kind == "supersede":
            _supersede(single, *payload)
        else:
            _importance(single, payload)

    three = FakeRepository()
    receipts = []
    for turn, ((kind, payload), output) in enumerate(zip(events, outputs)):
        with Pass2ReadOnlyGuard(three):
            pass  # output is already fixed; pass 2 consumes memory only.
        builder = MemoryLedgerBuilder(
            three,
            session_id="scripted",
            turn_id=str(turn),
            request_text=repr(payload),
            output_text=output,
        )
        if kind == "deposit":
            builder.record_operation(
                "deposit", reason_code="scripted", reason_detail="fixture",
                source_text=payload, operation=lambda p=payload: _deposit(three, p))
        elif kind == "supersede":
            builder.record_operation(
                "supersession", reason_code="scripted",
                reason_detail="decisive L1 fixture", source_text=repr(payload),
                operation=lambda p=payload: _supersede(three, *p))
        else:
            builder.record_operation(
                "importance_bookkeeping", reason_code="scripted",
                reason_detail="fixture", source_text=payload,
                operation=lambda p=payload: _importance(three, p))
        receipt, audit = builder.finalize()
        assert audit["complete"] is True
        receipts.append(receipt)

    without_management = FakeRepository()
    assert outputs == ["ack-alpha", "ack-beta", "beta"]
    assert arena_state_bytes(three) == arena_state_bytes(single)
    assert _active_value(three) == "beta"  # decisive supersession in pass 3
    assert int(_active_value(three) == "beta") >= int(
        _active_value(without_management) == "beta")
    assert all(r["mutation_count"] > 0 for r in receipts)
