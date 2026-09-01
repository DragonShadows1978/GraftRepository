"""LSR-P2B G3 — fixture turn through the e2e ledger path, no model, no GPU.

A ``__new__``-style ``GQAArenaCache`` fixture is driven through the real
``scripts.grm_e2e_session`` probe path and the real step-3 memory ledger.
The gate: the persisted route receipt's ``ranking``, ``rank_plan``, and
``final_mounts`` match the arena ``info`` the serve actually produced.

Only the model boundary is stubbed (``_attempt`` generates text, and
``_grounding_attribution`` reads it).  Routing, identifier tokens, ladder
planning, budget fitting, and the ledger are the production code paths.
"""

import json

import pytest

from core.graft_arena import ArenaCache
from core.grm_three_pass import ROUTE_RECEIPT_SCHEMA, MemoryLedgerBuilder

import scripts.grm_e2e_session as e2e


ARENA_WIDTH = 96
# Two candidates; node 0 is the long competitor that cannot be seated
# alongside node 1 at width 96 -- the lived Phase 1 geometry.
NODES = [
    ("User: Record the lumen seal.\nAssistant: The lumen seal value is "
     "Birch-2-Beacon and the surrounding narrative runs long enough to "
     "consume most of the arena width on its own.", 70),
    ("User: What is the current lumen seal value?\nAssistant: The current "
     "lumen seal value is Auric-4-Alpha.", 40),
]


def _graft(text, ntok):
    return {
        "text": text,
        "ntok": int(ntok),
        "kind": "turn",
        "retired": False,
        "metadata": {"active": True, "importance": {}, "supersedes": []},
        "provenance": [],
    }


class FixtureRuntime:
    def __init__(self):
        self.last_result = None

    def _finish_turn_event(self, event, before, extraction=(), autosave=False):
        return None


class FixtureRepository:
    """Minimum repository surface the e2e probe + ledger path touches."""

    def __init__(self, arena):
        self.arena = arena
        self.runtime = FixtureRuntime()
        self._paging_tel = None

    def _snapshot_state(self):
        return list(self.arena.grafts)

    def _extract_from_new_turns(self, before, context=None):
        return []


def _fixture_arena(*, served_text, grounded):
    arena = ArenaCache.__new__(ArenaCache)
    arena.grafts = [_graft(text, ntok) for text, ntok in NODES]
    arena.width = ARENA_WIDTH
    arena.topk = 3
    arena.cur_mounts = []
    arena.cur_mount_n = 0
    arena.live_segs = []
    arena.caches = None
    arena.pos = 0
    arena._s4_turn = 0
    arena._cuda_gqa_epoch = 0
    arena.s4_metadata_callback = None
    arena.stop_sequences = ()
    arena.live_shift = 115
    arena.decisive_admission = False
    arena.revision_resolution = False
    arena.last_route_backend = "python"
    arena.last_route_receipt = None
    arena.route_receipt_history = []

    class _Layer:
        class _Attn:
            live_shift = 0
        def __init__(self):
            self.self_attn = _Layer._Attn()

    class _Model:
        layers = [_Layer()]

    arena.m = _Model()

    # Deterministic route: node 1 (the correction) ranks first, then node 0.
    def route(user_text, exclude=None, limit=None, **kwargs):
        excluded = {int(i) for i in (exclude or ())}
        order = [i for i in (1, 0) if i not in excluded]
        if limit is not None:
            order = order[:int(limit)]
        arena.last_route_backend = "python"
        return order

    arena.route = route

    # Model boundary only.
    def _attempt(user_text, picks, ngen, deposit, stops, defer_memory=False):
        arena.cur_mounts = list(picks)
        arena.cur_mount_n = sum(
            int(arena.grafts[i]["ntok"]) for i in picks)
        info = {
            "mounts": [i + 1 for i in picks],
            "resident": arena.cur_mount_n,
            "evicted": 0,
            "live_tokens": 0,
        }
        if defer_memory:
            info["_deferred_memory"] = {
                "turn_text": "User: %s\nAssistant: %s" % (
                    user_text, served_text),
                "user_text": user_text,
                "seg_cache_ntok": 4,
                "picks": [int(i) for i in picks],
                "deposited": False,
                "importance_committed": False,
                "route_key_prepared": True,
                "route_key_token": 1,
            }
        return served_text, info

    arena._attempt = _attempt
    arena._grounding_attribution = lambda ans, mounts, question: (
        bool(grounded and mounts), list(mounts))
    return arena


PROBE_IDENTIFIER = "What is the current lumen seal value?"
# No rare/identifier token: precise-first does NOT fire, so the ladder plans
# the whole top-k slice and budget fitting is what decides the mount set --
# the stage Phase 1 adjudicated as ADMISSION-PRUNE.
PROBE_TOPICAL = "what do the records say about that"


def _serve(*, served_text="The current lumen seal value is Auric-4-Alpha.",
           grounded=True, probe=PROBE_IDENTIFIER):
    arena = _fixture_arena(served_text=served_text, grounded=grounded)
    repo = FixtureRepository(arena)
    answer, info = e2e._probe_ladder_chat(
        repo,
        probe,
        topk=3,
        ngen=16,
        max_trips=1,
        defer_memory=False,
        turn_idx=0,
    )
    return repo, answer, info


def _receipt_from_serve(repo, info, answer):
    class _Args:
        topk = 3
        max_trips = 1

    return e2e._turn_route_receipt(
        repo,
        {"kind": "probe", "user": "What is the current lumen seal value?"},
        info,
        answer,
        turn_idx=0,
        session_id="lsr-p2b-fixture",
        prep_receipt=None,
        arena_before_sha256=None,
        args=_Args(),
    )


def test_fixture_turn_receipt_matches_arena_info():
    repo, answer, info = _serve()
    record = _receipt_from_serve(repo, info, answer)

    assert record["schema"] == ROUTE_RECEIPT_SCHEMA
    # ranking / rank_plan / final_mounts agree with the info the serve made.
    assert record["route"]["ranking_ids"] == [int(x)
                                              for x in info["ranking_ids"]]
    assert record["fit"]["planned"] == [int(x) for x in info["mount_plan"]]
    assert record["fit"]["seated"] == [int(x) for x in info["mount_fitted"]]
    assert record["fit"]["dropped"] == [
        int(x) for x in info["mount_dropped_for_width"]]
    # info["mounts"] is 1-based; the receipt is graft-indexed.
    assert record["fit"]["final_mounts"] == [
        int(x) - 1 for x in info["mounts"]]
    assert record["fit"]["final_mounts"] == list(repo.arena.cur_mounts)
    assert record["fit"]["cur_mount_n"] == repo.arena.cur_mount_n
    assert record["fit"]["width"] == ARENA_WIDTH
    assert record["provenance"]["repository_size_at_probe"] == len(
        repo.arena.grafts)
    # Ladder trips were observed, not reconstructed.
    assert record["fit"]["trip_count"] >= 1
    assert record["fit"]["trips"][0]["grounded"] is True
    assert record["serving_path"] == (
        "grm_e2e_session._probe_ladder_chat")


def test_fixture_width_drop_is_visible_in_the_receipt():
    """The Phase 1 failure shape: a planned node dropped at fit.

    The lived probes planned two nodes at width 96 and seated one.  This
    fixture reproduces that geometry and asserts the receipt NAMES the
    dropped node -- the observability Phase 1 lacked.
    """
    repo, answer, info = _serve(probe=PROBE_TOPICAL)
    record = _receipt_from_serve(repo, info, answer)
    planned = record["fit"]["planned"]
    seated = record["fit"]["seated"]
    dropped = record["fit"]["dropped"]
    # Non-vacuous: a real drop happened.
    assert planned == [1, 0]
    assert seated == [1]
    assert dropped == [0]
    assert set(planned) == set(seated) | set(dropped)
    assert record["fit"]["final_mounts"] == [1]
    assert record["fit"]["cur_mount_n"] + int(
        repo.arena.grafts[0]["ntok"]) > record["fit"]["width"]
    # And it matches the arena info the serve produced, field for field.
    assert planned == [int(x) for x in info["mount_plan"]]
    assert seated == [int(x) for x in info["mount_fitted"]]
    assert dropped == [int(x) for x in info["mount_dropped_for_width"]]


def test_ungrounded_turn_still_writes_a_receipt_into_the_ledger():
    repo, answer, info = _serve(served_text="I do not know.", grounded=False)
    record = _receipt_from_serve(repo, info, answer)
    builder = MemoryLedgerBuilder(
        repo, session_id="lsr-p2b-fixture", turn_id="0",
        request_text="What is the current lumen seal value?",
        output_text=answer)
    builder.attach_route_receipt(record)
    ledger_receipt, audit = builder.finalize()
    assert audit["complete"] is True
    assert ledger_receipt["mutation_count"] == 0
    assert ledger_receipt["route_receipt"]["route"]["ranking_ids"] == [1, 0]
    # Round-trips through the on-disk form the session driver writes.
    assert json.loads(json.dumps(ledger_receipt))["route_receipt"][
        "schema"] == ROUTE_RECEIPT_SCHEMA


def test_full_step3_path_writes_the_receipt_into_the_ledger():
    """The real run_pass3_memory_management call, with the receipt attached.

    Only the deferred-deposit cache machinery is stubbed (it needs a model);
    the ledger, the audit, and the receipt attach are production code.
    """
    repo2, answer2, info2 = _serve(probe=PROBE_TOPICAL)
    arena2 = repo2.arena

    def deposit_deferred_turn(payload):
        arena2.grafts.append(_graft("User: q\nAssistant: %s" % answer2, 8))
        arena2._cuda_gqa_epoch += 1
        payload["_deferred_memory"]["deposited"] = True

    arena2.deposit_deferred_turn = deposit_deferred_turn
    arena2.commit_deferred_importance = lambda payload: None
    info2["_deferred_memory"] = {
        "turn_text": "User: q\nAssistant: %s" % answer2,
        "user_text": "q",
        "seg_cache_ntok": 4,
        "picks": [1],
        "deposited": False,
        "importance_committed": False,
        "importance_bookkeeping": None,
        "route_key_prepared": True,
        "route_key_token": 1,
    }

    class _Args:
        topk = 3
        max_trips = 1

    event = {"kind": "probe", "user": PROBE_TOPICAL}
    record = e2e._turn_route_receipt(
        repo2, event, info2, answer2,
        turn_idx=4, session_id="lsr-p2b-fixture",
        prep_receipt=None, arena_before_sha256="deadbeef", args=_Args())

    class _Timers:
        supersession_ms = 0.0
        supersession_calls = 0

    (out_info, _chat_node_id, _correction, ledger_receipt,
     audit) = e2e.run_pass3_memory_management(
        repo2, event, answer2, info2, repo2._snapshot_state(),
        turn_idx=4, session_id="lsr-p2b-fixture", timers=_Timers(),
        route_receipt=record)

    assert audit["complete"] is True
    assert ledger_receipt["route_receipt"]["schema"] == ROUTE_RECEIPT_SCHEMA
    assert ledger_receipt["route_receipt"]["turn_id"] == "4"
    assert ledger_receipt["route_receipt"]["fit"]["planned"] == [1, 0]
    assert ledger_receipt["route_receipt"]["fit"]["dropped"] == [0]
    assert ledger_receipt["route_receipt"]["provenance"][
        "arena_state_sha256_before_serve"] == "deadbeef"
    # The frozen mutation rows are still there and still audited.
    assert ledger_receipt["mutation_count"] >= 1
    assert set(ledger_receipt) - {
        "schema", "session_id", "turn_id", "turn_pipeline", "pass",
        "provenance", "mutations", "mutation_count"} == {"route_receipt"}
    assert out_info is info2


# ------------------------------- P2A abstention through the receipt path

# A profile with identifier tokens and NO identified candidate is exactly the
# LSR-P2A Ruling 2 abstention trigger (core.grm_admission).
ABSTAIN_PROFILE = {
    "ranking": [1, 0],
    "identified_candidates": [],
    "identifier_tokens": ["lumen", "seal"],
    "rare_identifier_tokens": [],
    "identifier_hit_count": 0,
    "route_margin_1_2": 0.0,
    "route_margin_evaluated": False,
    "margin_threshold": 0.1385774091529802,
    "rank_plan": [],
    "policy_branch": "no_identifier_hit",
    "rule_sha256": (
        "c304609f81475bd2ae3399ad180d3bb00cc5d2b44b1a49db570387c810defb91"),
    "route_backend": "python",
}


def test_abstained_probe_turn_yields_a_receipt_with_abstained_true(
        monkeypatch):
    """LSR-P2A Ruling 2 abstention, through P2B's receipt path.

    The abstention short-circuit returns BEFORE any mount work, so it walks
    no rungs.  The gate: it still produces a route receipt, and the
    ``abstain_*`` fields reach ``fit.info_pass_through`` through the generic
    prefix rule -- no P2B enumeration of P2A's field names.
    """
    arena = _fixture_arena(served_text="unused", grounded=False)
    arena.decisive_admission = True
    # P2A Ruling 2.2 deposits the abstention turn; the ``__new__`` fixture
    # predates these two fields and the deposit is the model boundary.
    arena.prompt_template = None

    def _deposit(text):
        arena.grafts.append(_graft(str(text), 8))
        return len(arena.grafts) - 1

    arena.deposit = _deposit
    repo = FixtureRepository(arena)
    # The abstention decision is a pure function of the A-DEC profile, so the
    # profile is supplied directly; no model, no _probe_key, no GPU.
    monkeypatch.setattr(
        e2e, "decisive_admission_profile",
        lambda *a, **k: dict(ABSTAIN_PROFILE))

    answer, info = e2e._probe_ladder_chat(
        repo, "What is the current lumen seal value?",
        topk=3, ngen=16, max_trips=1, defer_memory=False, turn_idx=0)

    # P2A's branch actually fired.
    assert info["abstained"] is True
    assert info["abstain_reason"] == "identifier_unbound"
    assert info["abstain_identifier_tokens"] == ["lumen", "seal"]
    assert "abstain_deposited_graft" in info
    assert answer

    # P2B recorded it: the observation hand-off is present with no rungs.
    observation = info["_route_observation"]
    assert observation["trips"] == []
    assert observation["serving_path"] == (
        "grm_e2e_session._probe_ladder_chat:abstention")

    class _Args:
        topk = 3
        max_trips = 1

    record = e2e._turn_route_receipt(
        repo,
        {"kind": "probe", "user": "What is the current lumen seal value?"},
        info, answer,
        turn_idx=9, session_id="lsr-p2b-fixture",
        prep_receipt=None, arena_before_sha256=None, args=_Args())

    assert record["schema"] == ROUTE_RECEIPT_SCHEMA
    through = record["fit"]["info_pass_through"]
    assert through["abstained"] is True
    assert through["abstain_reason"] == "identifier_unbound"
    assert through["abstain_identifier_tokens"] == ["lumen", "seal"]
    assert "abstain_deposited_graft" in through
    # The route decision behind the abstention is on the record too.
    assert record["route"]["ranking_ids"] == [1, 0]
    assert record["admission"]["identifier_tokens"] == ["lumen", "seal"]
    assert record["admission"]["identified_candidates"] == []
    assert record["fit"]["trips"] == []
    assert record["fit"]["trip_count"] == 0
    # Persisting it through the ledger keeps the frozen rows frozen.
    builder = MemoryLedgerBuilder(
        repo, session_id="lsr-p2b-fixture", turn_id="9",
        request_text="What is the current lumen seal value?",
        output_text=answer)
    builder.attach_route_receipt(record)
    ledger_receipt, audit = builder.finalize()
    assert audit["complete"] is True
    assert ledger_receipt["route_receipt"]["fit"][
        "info_pass_through"]["abstained"] is True
