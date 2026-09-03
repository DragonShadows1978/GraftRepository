"""GRM-LSR-P2A — the SAME two rulings on the harness's own ladder path.

The lived probes went through ``scripts/grm_e2e_session._probe_ladder_chat``
(and the DET1.5 fork of the same ladder), not through ``ArenaCache.step``.
Ruling 1 and Ruling 2 are therefore required at BOTH fit sites, and this
module pins the driver site.

CPU-only: the arena and repository are stubs; no model, no GPU lease.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts import grm_e2e_session as e2e
from core.graft_arena import ArenaCache
from core.grm_admission import ABSTAIN_REASON_IDENTIFIER_UNBOUND


class _StubArena:
    # GRM-EB1: the driver's probe path is a PRODUCTION SERVING PATH and now
    # opens its turn through the arena's frame helpers, exactly as
    # ``ArenaCache.step()`` does.  The stub borrows the REAL methods rather
    # than reimplementing them, so these tests exercise the production frame
    # logic instead of a copy that could drift away from it.
    eb1_begin_turn = ArenaCache.eb1_begin_turn
    eb1_charge_recency = ArenaCache.eb1_charge_recency
    _eb1_frame_info = ArenaCache._eb1_frame_info

    def __init__(self, *, ntok, width, decisive, grounded=True,
                 ephemeral=False, recency_mounts=0):
        self.grafts = [
            {"text": f"node-{i}", "ntok": int(n), "kind": "fact"}
            for i, n in enumerate(ntok)
        ]
        self.width = int(width)
        self.decisive_admission = bool(decisive)
        # These fixtures pin the FIT stage, whose contract predates EB1 and is
        # frame-independent: every node is kind="fact", so recency nominates
        # nothing under either frame and the budget is the full width.  The
        # frame itself is pinned in tests/test_grm_eb1_ephemeral_frame.py.
        self.ephemeral = bool(ephemeral)
        self.recency_mounts = int(recency_mounts)
        self.live_segs = []
        self.caches = None
        self.pos = 0
        self.cur_mounts = []
        self.cur_mount_n = 0
        self.stop_sequences = ()
        self.live_shift = 9
        self.m = SimpleNamespace(
            layers=[SimpleNamespace(self_attn=SimpleNamespace(live_shift=None))])
        self._grounded = bool(grounded)
        self.served = []
        self.route_calls = []
        self.deposits = []

    # --- routing / lexical surface -------------------------------------
    def route(self, text, *, exclude, limit):
        self.route_calls.append((text, set(exclude), int(limit)))
        return list(range(len(self.grafts)))

    def _rare_tokens(self, _text):
        return set()

    def _query_lex_tokens(self, _text):
        return {"meridian", "docket"}

    def _node_text_tokens(self, text):
        return set(str(text).split())

    def _resolve_revision_mounts(self, picks):
        return [int(v) for v in picks]

    def _bump_cuda_gqa_epoch(self):
        return None

    def _format_step_turn(self, user_text, assistant_text):
        return f"User: {user_text}\nAssistant: {assistant_text}\n"

    def deposit(self, text):
        self.grafts.append({"text": text, "ntok": 3, "kind": "turn"})
        self.deposits.append(text)
        return len(self.grafts) - 1

    # --- serving --------------------------------------------------------
    def _attempt(self, _question, picks, _ngen, _deposit, _stops,
                 defer_memory=False):
        self.cur_mounts = list(picks)
        self.served.append(list(picks))
        return "answer", {"mounts": [v + 1 for v in picks]}

    def _grounding_attribution(self, _ans, picks, _q):
        return (self._grounded and bool(picks), list(picks))


class _StubRepo:
    def __init__(self, arena):
        self.arena = arena
        self._paging_tel = SimpleNamespace(
            snapshot_enabled=False, enabled=False)

    def _snapshot_state(self):
        return []


def _pin_profile(monkeypatch, profile):
    monkeypatch.setattr(
        e2e, "decisive_admission_profile",
        lambda _arena, _text, *, exclude, route_limit: dict(profile),
    )
    # The driver deposits through _probe_finish_deposit; stub it out so the
    # test exercises the ladder, not the repository's extraction machinery.
    monkeypatch.setattr(
        e2e, "_probe_finish_deposit",
        lambda _repo, _before, _u, _a, info, *, defer_memory: info,
    )


_LIVED_PROFILE = {
    "ranking": [3, 2, 1, 0],
    "identified_candidates": [3],
    "identifier_tokens": ["meridian", "docket"],
    "identifier_hit_count": 1,
    "route_margin_1_2": 1.036860985747615,
    "rank_plan": [3],
    "policy_branch": "exactly_one_identifier_decisive_rank1",
    "route_backend": "python",
    "margin_threshold": 0.1385774091529802,
    "rule_sha256": "c304609f81475bd2ae3399ad180d3bb00cc5d2b44b1a49db570387c810defb91",
    "route_margin_evaluated": True,
}


def test_probe_path_reports_the_unseatable_plan_head(monkeypatch):
    """The lived ADMISSION-PRUNE shape at the DRIVER fit site.

    ntok mirrors the lived receipt: node 3 is the 650-char competitor that
    consumes more than the whole 96-seat arena, node 2 is the 66-seat node
    the lived run actually mounted (arena.cur_mount_n = 66).
    """
    _pin_profile(monkeypatch, _LIVED_PROFILE)
    arena = _StubArena(ntok=[20, 25, 66, 130], width=96, decisive=True)
    _ans, info = e2e._probe_ladder_chat(
        _StubRepo(arena), "What is the current Meridian docket value?",
        topk=3, ngen=1, max_trips=1)
    assert info["fit_planned"] == [3]
    assert info["fit_unseatable"] == [3]
    assert info["fit_dropped_planned"] == [3]
    assert info["served_without_plan_head"] is True
    assert 3 not in info["fit_seated"]


def test_probe_path_seats_the_plan_head_ahead_of_cheaper_filler(monkeypatch):
    """Plan priority: the plan head seats even when filler is cheaper.

    Under the pre-P2A rank-order packing the widened rung ``[3, 2, 1]`` is
    packed in rank order, so node 3 (90) fits first here anyway; the
    discriminating case is a plan member that is NOT rank-1 in the rung.
    """
    _pin_profile(monkeypatch, {**_LIVED_PROFILE, "rank_plan": [1]})
    arena = _StubArena(ntok=[40, 40, 40, 40], width=96, decisive=True)
    _ans, info = e2e._probe_ladder_chat(
        _StubRepo(arena), "What is the current Meridian docket value?",
        topk=3, ngen=1, max_trips=1)
    assert 1 in info["fit_seated"]
    assert info["fit_dropped_planned"] == []
    assert info["served_without_plan_head"] is False


def test_probe_path_shuttles_a_non_cofitting_declared_synthesis_pair(monkeypatch):
    _pin_profile(monkeypatch, {
        **_LIVED_PROFILE,
        "identified_candidates": [0, 1],
        "identifier_hit_count": 2,
        "rank_plan": [0, 1],
        "ranking": [0, 1],
        "policy_branch": "declared_synthesis_identified_set",
    })
    arena = _StubArena(ntok=[60, 60], width=96, decisive=True, grounded=False)
    _ans, info = e2e._probe_ladder_chat(
        _StubRepo(arena), "What is the current Meridian docket value?",
        topk=3, ngen=1, max_trips=1)
    assert info["fit_shuttle"] is True
    assert info["fit_shuttle_trips"] == [[1]]
    seated_across_turn = {v for trip in arena.served for v in trip}
    assert seated_across_turn == {0, 1}
    assert info["fit_dropped_planned"] == []


def test_probe_path_shuttle_survives_a_grounded_first_trip(monkeypatch):
    """A wrong-but-grounded trip 0 must not end the turn (Ruling 1.2)."""
    _pin_profile(monkeypatch, {
        **_LIVED_PROFILE,
        "identified_candidates": [0, 1],
        "identifier_hit_count": 2,
        "rank_plan": [0, 1],
        "ranking": [0, 1],
        "policy_branch": "declared_synthesis_identified_set",
    })
    arena = _StubArena(ntok=[60, 60], width=96, decisive=True, grounded=True)
    _ans, info = e2e._probe_ladder_chat(
        _StubRepo(arena), "What is the current Meridian docket value?",
        topk=3, ngen=1, max_trips=1)
    assert info["fit_shuttle"] is True
    assert info["fit_shuttle_trips"] == [[1]]


def test_probe_path_legacy_is_unchanged(monkeypatch):
    """GRM_ADM_DECISIVE=0: legacy route call, legacy packing, no fit fields."""
    monkeypatch.setenv("GRM_ADM_DECISIVE", "0")
    monkeypatch.setattr(
        e2e, "_probe_finish_deposit",
        lambda _repo, _before, _u, _a, info, *, defer_memory: info,
    )
    arena = _StubArena(ntok=[1, 1, 1], width=16, decisive=False)
    _ans, info = e2e._probe_ladder_chat(
        _StubRepo(arena), "What is the current Meridian docket value?",
        topk=3, ngen=1, max_trips=1)
    assert arena.route_calls == [
        ("What is the current Meridian docket value?", set(), 6)]
    assert arena.served == [[0, 1, 2]]
    for key in (
        "fit_planned", "fit_seated", "fit_dropped_planned",
        "fit_dropped_filler", "fit_unseatable", "fit_shuttle",
        "fit_shuttle_trips", "served_without_plan_head",
        "abstained", "abstain_reason",
    ):
        assert key not in info, f"legacy info gained {key}"
    assert "admission_policy" not in info


def test_probe_path_abstains_on_an_unbound_identifier(monkeypatch):
    _pin_profile(monkeypatch, {
        **_LIVED_PROFILE,
        "identified_candidates": [],
        "identifier_hit_count": 0,
        "identifier_tokens": ["zephyr", "manifest"],
    })
    arena = _StubArena(ntok=[20, 25, 66, 40], width=96, decisive=True)
    ans, info = e2e._probe_ladder_chat(
        _StubRepo(arena), "What is the current Zephyr manifest value?",
        topk=3, ngen=1, max_trips=1)
    assert info["abstained"] is True
    assert info["abstain_reason"] == ABSTAIN_REASON_IDENTIFIER_UNBOUND
    assert ans == (
        "Not in memory: no stored record matches zephyr, manifest.")
    # Nothing confabulated: the topical nearest neighbour never mounted.
    assert arena.served == []
    assert info["mount_fitted"] == []
    # Ruling 2.2: the turn deposits, pinned kind="recall".
    assert len(arena.deposits) == 1
    assert arena.grafts[info["abstain_deposited_graft"]]["kind"] == "recall"


def test_probe_path_bound_identifier_never_abstains(monkeypatch):
    _pin_profile(monkeypatch, _LIVED_PROFILE)
    arena = _StubArena(ntok=[20, 25, 66, 40], width=96, decisive=True)
    _ans, info = e2e._probe_ladder_chat(
        _StubRepo(arena), "What is the current Meridian docket value?",
        topk=3, ngen=1, max_trips=1)
    assert "abstained" not in info
    assert arena.served
    assert 3 in info["fit_seated"]


@pytest.mark.parametrize("width,expect_head", [(96, False), (256, True)])
def test_probe_path_width_decides_the_lived_defect(monkeypatch, width,
                                                   expect_head):
    _pin_profile(monkeypatch, _LIVED_PROFILE)
    arena = _StubArena(ntok=[20, 25, 66, 130], width=width, decisive=True)
    _ans, info = e2e._probe_ladder_chat(
        _StubRepo(arena), "What is the current Meridian docket value?",
        topk=3, ngen=1, max_trips=1)
    assert (3 in info["fit_seated"]) is expect_head
