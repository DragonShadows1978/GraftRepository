"""GRM-LSR-P2C — unseatable nodes: split at deposit, descend at fit.

P2A made the fit stage honest and found that Ruling 1 alone cannot flip the
four ADMISSION-PRUNE probes: their answer-bearing nodes (623-706 chars)
exceed the 96-seat arena ALONE, so ``fit_unseatable`` is non-empty and the
plan-shuttle has nothing to serialize (``artifacts/lsr_p2a/
lsr_p2a_replay_*.json``).  Explicit degrade is honest but still serves
without the answer.

The principle these tests pin: **the repository never holds a node the arena
cannot mount, and a node too long to seat is served across its chunks, not
replaced by a neighbour.**

Every test here is CPU-only: it drives the stubbed arena/repository harness
established by ``tests/test_grm_admission.py`` and
``tests/test_grm_lsr_p2a_*.py``, so no GPU lease is consumed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.graft_arena import ArenaCache
from core.graft_repository import GraftRepository
from core.grm_admission import (
    chunk_trip_cap,
    mountable_budget,
    split_info_fields,
)
from core.grm_three_pass import (
    ROUTE_RECEIPT_INFO_PREFIXES,
    _route_receipt_generic_info,
)
from scripts import grm_e2e_session as e2e


# ---------------------------------------------------------------------------
# mountable_budget — the DERIVED number, not a new constant
# ---------------------------------------------------------------------------


def test_mountable_budget_is_the_arena_width_with_no_sink_reserve():
    """Derivation: the sink sits BELOW the arena band and the question/answer
    live tokens sit at or above ``live_shift = n_sink + arena_width``, so
    neither consumes a mount seat.  The budget is the width itself."""
    arena = SimpleNamespace(width=96)
    assert mountable_budget(arena) == 96


def test_mountable_budget_subtracts_only_the_recency_reserve_fit_applies():
    """``graft_arena.step()::fit_detail`` is the only site that subtracts
    anything: ``budget = self.width - rec_budget``, and ``rec_budget`` is 0
    for identifier queries (the whole ADMISSION-PRUNE class)."""
    arena = SimpleNamespace(width=96)
    assert mountable_budget(arena, recency_reserve=0) == 96
    assert mountable_budget(arena, recency_reserve=30) == 66
    # Never negative: a reserve wider than the arena leaves no seats, not a
    # negative budget that would make every node "fit".
    assert mountable_budget(arena, recency_reserve=200) == 0


def test_chunk_trip_cap_is_the_chunk_count():
    assert chunk_trip_cap([]) == 0
    assert chunk_trip_cap([7]) == 1
    assert chunk_trip_cap([7, 8, 9]) == 3


# ---------------------------------------------------------------------------
# Part 1 — deposit-time width guard (prevention)
# ---------------------------------------------------------------------------


class _GuardRepo:
    """A GraftRepository built through ``__new__`` with only the state the
    width guard reads.  The librarian split itself (``_cull_graft_direct``)
    is the real method; the persistence/native seams around it are stubs."""

    pass


def _guard_repo(*, width, text, ntok, kind="doc", tags=(), metadata=None):
    repo = GraftRepository.__new__(GraftRepository)
    arena = ArenaCache.__new__(ArenaCache)
    arena.width = int(width)
    arena.PAYLOAD = (("c", 1),)
    # One token per whitespace word: deterministic, tokenizer-free.
    arena.encode = lambda t: str(t).split()
    arena.decode = lambda ids: " ".join(ids)
    arena._rare_tokens = lambda t: {
        w.lower() for w in str(t).split() if any(c.isdigit() for c in w)
    }
    arena._node_key = lambda t, h=None: _np_key(t)
    arena._bump_cuda_gqa_epoch = lambda: None
    arena.grafts = [{
        "text": text,
        "ntok": int(ntok),
        "kind": kind,
        "tags": list(tags),
        "retired": False,
        "sources": [],
        "cent": _np_key(text),
        "host_payload": {"c": _np_payload(int(ntok))},
        "host_present": True,
        "metadata": dict(metadata or {
            "kind": kind, "active": True, "durability": "project",
            "mutability": "stable", "scope": "project",
            "write_intent": "observed", "confidence": 1.0,
            "source_grafts": [], "supersedes": [], "superseded_by": [],
        }),
        "provenance": [],
    }]
    repo.arena = arena
    repo.native_store = None
    repo.dirty_nodes = {}
    repo._dirty_generation = 0
    repo.vram_budget = None
    repo.wal_enabled = False
    repo.fold_history = []
    repo._all_graft_text_ascii = True
    # Persistence / native seams the split touches, stubbed to no-ops.
    repo._snapshot_state = lambda: []
    repo._ensure_host_payload = lambda i, g: None
    repo._native_sync_node = lambda i, payload_required=True: None
    repo._native_apply_cull_revisions = lambda parent, children: None
    repo._mark_dirty = lambda idx, payload=False, metadata=True: None
    repo._mark_mutations = lambda before: None
    repo._append_wal = lambda rec_type, **fields: None
    repo._ensure_lifecycle = lambda idx, g: None
    repo._rebuild_child_keys = lambda: None
    repo._free_retired = lambda: None
    repo._page = lambda: 0
    repo._provenance = lambda segment_type, node_id=None, **f: {
        "segment_type": segment_type, "node_id": node_id, **f}
    repo._default_metadata = lambda g: {"kind": g.get("kind"), "active": True}
    repo._state_tuple = lambda g: ()
    return repo


def _np_key(text):
    import numpy as np

    rng = np.zeros(8, dtype="float32")
    rng[0] = float(len(str(text)))
    return rng


def _np_payload(ntok):
    import numpy as np

    return np.zeros((1, int(ntok), 4), dtype="float16")


_LONG_SECTIONS = "\n\n".join(
    f"Section {i} covers alpha bravo charlie delta echo foxtrot golf hotel"
    for i in range(6)
)


def test_deposit_guard_splits_a_node_wider_than_the_budget():
    ntok = len(_LONG_SECTIONS.split())
    repo = _guard_repo(width=12, text=_LONG_SECTIONS, ntok=ntok)
    assert ntok > 12, "the fixture must actually exceed the budget"
    out = repo._guard_deposit_width(0)
    assert out is not None
    assert out["action"] == "width_guard_split"
    assert len(out["children"]) >= 2
    # POSTCONDITION: no child exceeds the budget.
    for child_idx in out["children"]:
        assert int(repo.arena.grafts[child_idx]["ntok"]) <= 12


def test_deposit_guard_does_not_split_a_node_exactly_at_the_budget():
    """A node at EXACTLY ``mountable_budget`` is ONE node.

    ``plan_priority_fit`` seats a member whose cost is ``<= budget``, so a
    node at the budget is seatable and splitting it would be gratuitous.
    """
    text = "alpha bravo charlie delta"
    repo = _guard_repo(width=4, text=text, ntok=4)
    assert repo._guard_deposit_width(0) is None
    assert len(repo.arena.grafts) == 1
    assert repo.arena.grafts[0]["kind"] == "doc"


def test_deposit_guard_does_not_split_a_node_below_the_budget():
    repo = _guard_repo(width=96, text="alpha bravo", ntok=2)
    assert repo._guard_deposit_width(0) is None
    assert len(repo.arena.grafts) == 1


def test_deposit_guard_is_a_noop_when_the_arena_declares_no_width():
    """REGRESSION (LSR-P2C G1, full-suite run).

    A stub / diagnostic arena reports ``width = 0``.  There are no seats to
    fit into, so the guard has nothing to do — and raising there broke 11
    unrelated tests in ``tests/test_grm_importance_salience.py``.  A
    non-positive budget is a NO-OP, not an error.
    """
    repo = _guard_repo(width=0, text="alpha bravo charlie delta", ntok=4)
    assert repo._guard_deposit_width(0) is None
    assert len(repo.arena.grafts) == 1
    # The explicit operator sweep still refuses a meaningless budget loudly.
    with pytest.raises(ValueError):
        repo.split_oversized()


def test_split_children_carry_lineage_and_identifier_bindings():
    text = "\n\n".join([
        "Meridian docket preface with code Delta-4-Drift recorded here.",
        "Second section names Juniper pass and code Opal-7-Green instead.",
        "Third section is filler prose about the loading dock schedule.",
    ])
    ntok = len(text.split())
    repo = _guard_repo(
        width=8, text=text, ntok=ntok, kind="fact", tags=("battery",),
        metadata={
            "kind": "fact", "active": True, "durability": "permanent",
            "mutability": "stable", "scope": "user",
            "write_intent": "user_asserted", "confidence": 1.0,
            "source_grafts": [], "supersedes": [7], "superseded_by": [],
        },
    )
    out = repo._guard_deposit_width(0)
    assert out is not None
    children = out["children"]
    for child_idx in children:
        child = repo.arena.grafts[child_idx]
        meta = child["metadata"]
        # LINEAGE: each child points back at the parent it was cut from.
        assert meta["culled_from"] == 0
        assert 0 in meta["source_grafts"]
        assert child["sources"] == [0]
        # PROVENANCE + TAGS inherited.
        assert child["provenance"]
        assert "battery" in child["tags"]
        # Metadata carried through the librarian's own transfer.
        assert meta["durability"] == "permanent"
        assert meta["scope"] == "user"
        assert meta["write_intent"] == "user_asserted"
        # IDENTIFIERS: each child's rare tokens are its OWN.
        assert child["rare"] == repo.arena._rare_tokens(child["text"])
    # The PARENT's routing surface is the UNION, so identifier routing still
    # finds the family through the index node.
    union = set()
    for child_idx in children:
        union |= set(repo.arena.grafts[child_idx]["rare"])
    assert union <= set(repo.arena.grafts[0]["rare"])


def test_split_parent_is_an_era_class_index_node_never_a_reader():
    """Measured law (2026-06-10): era texts are INDEX nodes, never readers.

    ``graft_arena.step()`` expands ``("era",)`` at the PRIMARY attempt, so an
    era-kind parent routes but is never itself mounted as a reader.
    """
    ntok = len(_LONG_SECTIONS.split())
    repo = _guard_repo(width=12, text=_LONG_SECTIONS, ntok=ntok)
    out = repo._guard_deposit_width(0)
    parent = repo.arena.grafts[0]
    assert parent["kind"] == GraftRepository.WIDTH_GUARD_PARENT_KIND == "era"
    # The parent stays ACTIVE and routable (not retired): identifier routing
    # must still be able to reach the family.
    assert parent["retired"] is False
    assert parent["metadata"]["active"] is True
    assert parent["metadata"]["width_guard_parent"] is True
    # `sources` is what `_descent_source_children` reads: parent -> children.
    assert parent["sources"] == list(out["children"])
    # A bare ArenaCache expands era picks to their children, so the parent
    # index node is never itself the reader.
    arena = repo.arena
    arena._native_source_closure_indices = lambda ids, **kw: None
    expanded = ArenaCache._descent_expand(arena, [0], ("era",), qrare=None)
    assert 0 not in expanded
    assert expanded == list(out["children"])


def test_split_oversized_sweep_repairs_a_legacy_repository():
    """Part 3: legacy repositories are repaired ONCE, not at every fit."""
    ntok = len(_LONG_SECTIONS.split())
    repo = _guard_repo(width=12, text=_LONG_SECTIONS, ntok=ntok)
    out = repo.split_oversized()
    assert out["action"] == "split_oversized"
    assert out["budget"] == 12
    assert out["examined"] == 1
    assert out["split"] == [0]
    assert len(out["children"]) >= 2
    for child_idx in out["children"]:
        assert int(repo.arena.grafts[child_idx]["ntok"]) <= 12
    # IDEMPOTENT: a second sweep finds nothing left to repair, because the
    # parent is now an index node and every child fits.
    again = repo.split_oversized()
    assert again["examined"] == 0
    assert again["split"] == []


def test_split_oversized_command_grammar_follows_the_cull_verb():
    parse = GraftRepository._parse_memory_command_python
    assert parse("split oversized") == {"action": "split_oversized"}
    assert parse("cull oversized") == {"action": "split_oversized"}
    assert parse("split oversized into paragraphs") == {
        "action": "split_oversized", "boundary": "paragraph"}
    assert parse("split oversized max tokens 64") == {
        "action": "split_oversized", "budget": 64}
    with pytest.raises(ValueError):
        parse("split oversized wobble")
    # The pre-existing cull/split graft grammar is untouched.
    assert parse("cull graft 3 max tokens 64") == {
        "action": "cull_graft", "node_id": 3, "max_tokens": 64}


def test_split_spans_follow_chunk_boundaries_and_never_sever_a_value():
    """REGRESSION (LSR-P2C G2 Arm 1, first run on multi_hop_a_b_c).

    A prefix-offset span plan capped at ``max_tokens`` cut the meridian
    competitor at token 96, which lands INSIDE the answer: chunk 0 ended
    ``"...Meridian docket value is Delta-4-"`` and chunk 1 began ``"Drift."``.
    The model read the truncated value and emitted
    ``"Delta-4-Delta-4-Delta-4..."``.  A split that severs the value it exists
    to preserve is worse than no split at all, so the spans must follow the
    CHUNKER's sentence boundaries.
    """
    text = (
        "Filler sentence one two three four five six seven eight nine. "
        "Filler sentence ten eleven twelve thirteen fourteen fifteen. "
        "The sole authoritative value is Delta-4-Drift for this docket. "
        "Trailing filler sixteen seventeen eighteen nineteen twenty."
    )
    ntok = len(text.split())
    repo = _guard_repo(width=14, text=text, ntok=ntok)
    out = repo._guard_deposit_width(0)
    assert out is not None
    texts = [repo.arena.grafts[c]["text"] for c in out["children"]]
    # The value survives WHOLE in exactly one child; no child holds a
    # truncated prefix of it.
    whole = [t for t in texts if "Delta-4-Drift" in t]
    assert len(whole) == 1, texts
    for t in texts:
        if "Delta-4-Drift" not in t:
            assert "Delta-4" not in t, f"a child holds a severed value: {t!r}"


def test_chunk_token_spans_tile_exactly_or_return_none():
    """The span plan is VALIDATED, never trusted: chunks encoded in isolation
    can disagree with the parent's in-context ledger, and a plan that does
    not tile the parent exactly is refused rather than silently dropping or
    duplicating tokens."""
    repo = _guard_repo(width=8, text="x", ntok=1)
    # A well-formed plan tiles and is returned.
    spans = repo._chunk_token_spans(["a b c", "d e"], 5, 8)
    assert spans == [(0, 3), (3, 5)]
    # A plan whose chunks over-run the parent is refused.
    assert repo._chunk_token_spans(["a b c d e f", "g"], 3, 8) is None
    # A single chunk is not a split.
    assert repo._chunk_token_spans(["a b"], 2, 8) is None


def test_width_fitting_chunker_falls_back_to_sentences_and_words():
    """Sentence/line fallback so NO child exceeds the budget even for text
    with no section structure at all."""
    repo = _guard_repo(width=4, text="x", ntok=1)
    flat = ("one two three four five six seven. "
            "eight nine ten eleven twelve thirteen fourteen.")
    chunks = repo._width_fitting_chunks(flat, 4)
    assert len(chunks) >= 4
    for chunk in chunks:
        assert len(chunk.split()) <= 4
    # A single "sentence" longer than the budget still breaks on whitespace.
    runon = " ".join(f"w{i}" for i in range(20))
    for chunk in repo._width_fitting_chunks(runon, 3):
        assert len(chunk.split()) <= 3


# ---------------------------------------------------------------------------
# Part 2 — fit-time descent (repair), at the arena fit site
# ---------------------------------------------------------------------------


def _descent_arena(*, ntok, width, texts=None, grounded=True,
                   ground_only=None):
    """Stubbed CPU arena that can actually split, extending the P2A harness.

    ``ground_only`` pins WHICH mount set grounds, so a test can make every
    chunk trip fail grounding (the only condition under which
    ``served_without_plan_head`` may be true after a split).
    """
    arena = ArenaCache.__new__(ArenaCache)
    arena.m = SimpleNamespace(
        layers=[SimpleNamespace(self_attn=SimpleNamespace(live_shift=None))])
    arena.live_shift = 9
    arena.stop_sequences = ()
    arena.ephemeral = False
    arena.live_segs = []
    arena.topk = 3
    arena.decisive_admission = True
    arena.caches = None
    arena.pos = 0
    arena.cur_mounts = []
    arena.cur_mount_n = 0
    arena.width = int(width)
    arena.prompt_template = None
    arena.graft_splitter = None
    texts = texts or [f"node-{i}" for i in range(len(ntok))]
    arena.grafts = [
        {"text": texts[i], "ntok": int(n), "kind": "fact",
         "rare": set(), "sources": []}
        for i, n in enumerate(ntok)
    ]
    arena._s4_turn = 0
    arena.encode = lambda t: str(t).split()
    arena.decode = lambda ids: " ".join(ids)
    arena.route = lambda text, *, exclude, limit: list(range(len(ntok)))
    arena._rare_tokens = lambda t: {
        w.lower().strip(".") for w in str(t).split()
        if any(c.isdigit() for c in w)
    }
    arena._resolve_revision_mounts = lambda picks: list(picks)
    arena._next_s4_turn = lambda: 1
    arena._bump_cuda_gqa_epoch = lambda: None
    arena._commit_s4_attempt = lambda *_a, **_k: None
    arena._native_source_closure_indices = lambda ids, **kw: None
    served = []

    def attempt(_question, picks, _ngen, _deposit, _stops):
        arena.cur_mounts = list(picks)
        served.append(list(picks))
        return "answer", {"mounts": [v + 1 for v in picks]}

    def grounding(_ans, picks, _q):
        if ground_only is not None:
            ok = sorted(int(v) for v in picks) == sorted(ground_only)
        else:
            ok = bool(grounded) and bool(picks)
        return (ok, list(picks))

    arena._attempt = attempt
    arena._grounding_attribution = grounding

    def deposit(text):
        arena.grafts.append({
            "text": text, "ntok": len(str(text).split()), "kind": "turn",
            "rare": set(), "sources": [],
        })
        return len(arena.grafts) - 1

    arena.deposit = deposit
    return arena, served


def _pin_arena_profile(monkeypatch, profile):
    monkeypatch.setattr(
        "core.graft_arena.decisive_admission_profile",
        lambda _arena, _text, *, exclude, route_limit: dict(profile),
    )


#: The lived sup_reserve_meridian_docket shape: rank_plan = [3], and node 3
#: is the long competitor that exceeds the whole arena ALONE.
_LIVED_PROFILE = {
    "ranking": [3, 2, 1, 0],
    "identified_candidates": [3],
    "identifier_tokens": ["meridian", "docket"],
    "identifier_hit_count": 1,
    "route_margin_1_2": 1.036860985747615,
    "rank_plan": [3],
    "policy_branch": "exactly_one_identifier_decisive_rank1",
}

#: The answer-bearing chunk names the code; the other chunks do not.
#: Mirrors the lived competitor shape: only ONE chunk names the identifier
#: phrase the frozen ADM1 predicate looks for ("current <id tokens> value"),
#: and that chunk is the one carrying the answer.
_COMPETITOR_TEXT = "\n".join([
    "filler one two three four five six seven eight nine ten",
    "the current meridian docket value is delta-4-drift for the record",
    "more filler eleven twelve thirteen fourteen fifteen sixteen",
])


def test_fit_time_descent_seats_the_identifier_bearing_child(monkeypatch):
    _pin_arena_profile(monkeypatch, _LIVED_PROFILE)
    texts = ["a", "b", "c", _COMPETITOR_TEXT]
    ntok = [4, 4, 4, len(_COMPETITOR_TEXT.split())]
    arena, served = _descent_arena(ntok=ntok, width=11, texts=texts)
    assert ntok[3] > 11, "the fixture must be UNSEATABLE at this width"
    # The question NAMES the code, so `_descent_source_children` filters the
    # children by `qrare & child["rare"]` — the identifier-bearing child is
    # picked out, not the whole chunk set.
    _answer, info = arena.step(
        "meridian docket delta-4-drift", ngen=1, deposit=False, max_trips=1)
    # The split happened, and it was EPHEMERAL (no repository attached).
    assert info["fit_split_parent"] == 3
    assert len(info["fit_split_children"]) >= 2
    assert info["fit_split_ephemeral"] is True
    # The plan head became the IDENTIFIER-BEARING child set.
    assert info["fit_descended_head"]
    for child in info["fit_descended_head"]:
        assert "delta-4-drift" in arena.grafts[child]["text"]
    # The identifier child was actually SEATED and served.
    assert set(info["fit_descended_head"]) & set(info["fit_seated"])
    assert served, "the descended head must be served, not degraded"
    # The turn is no longer serving without the plan head.
    assert info["served_without_plan_head"] is False


def test_fit_time_descent_persists_the_split_when_a_splitter_is_attached(
        monkeypatch):
    _pin_arena_profile(monkeypatch, _LIVED_PROFILE)
    texts = ["a", "b", "c", _COMPETITOR_TEXT]
    ntok = [4, 4, 4, len(_COMPETITOR_TEXT.split())]
    arena, _served = _descent_arena(ntok=ntok, width=11, texts=texts)
    persisted = []

    def splitter(idx, budget):
        # Stand-in for GraftRepository._split_for_fit: it writes the split
        # through the librarian and returns the child indices.
        children = []
        for line in str(arena.grafts[int(idx)]["text"]).splitlines():
            children.append(arena.deposit(line))
            arena.grafts[children[-1]]["rare"] = arena._rare_tokens(line)
        arena.grafts[int(idx)]["kind"] = "era"
        arena.grafts[int(idx)]["sources"] = list(children)
        persisted.append((int(idx), list(children)))
        return children

    arena.graft_splitter = splitter
    _answer, info = arena.step("meridian docket?", ngen=1, deposit=False,
                               max_trips=1)
    assert persisted, "the librarian seam must be used when it exists"
    assert info["fit_split_ephemeral"] is False
    assert info["fit_split_parent"] == 3
    assert info["fit_split_children"] == persisted[0][1]


def test_descended_head_uses_the_frozen_lexical_binding_predicate():
    """REGRESSION (LSR-P2C G2 Arm 1, second run).

    ``_descent_source_children`` filters children by ``qrare &
    child["rare"]`` — the CODE/NUMBER channel — and that channel is EMPTY for
    the whole ADMISSION-PRUNE probe class: "What is the current Meridian
    docket value?" contains no code-shaped token, so ``_rare_tokens`` returns
    ``set()``, the filter cannot discriminate, and every chunk comes back in
    DOCUMENT ORDER with the filler first.  Measured: the 86-token filler
    prefix was seated, it grounded, and the turn ended before the chunk
    holding ``Delta-4-Drift`` was read.

    The LEXICAL channel binds these probes, and asking it is not a new rule:
    ``grm_admission.is_identifier_binding`` is the FROZEN ADM1 predicate A-DEC
    already uses one level up, on repository nodes.
    """
    arena, _served = _descent_arena(
        ntok=[4, 4],
        width=96,
        texts=[
            "audit narrative filler that names no code at all",
            "the current meridian docket value is delta-4-drift",
        ],
    )
    question = "What is the current Meridian docket value?"
    # The rare channel really is silent for this question.
    assert arena._rare_tokens(question) == set()
    picked = arena._identifier_bearing_children(
        0, question, arena._rare_tokens(question), [0, 1])
    # Only the chunk that BINDS the identifier is the head — not the filler,
    # and not "both, document order".
    assert picked == [1]


def test_descended_head_falls_back_when_nothing_binds():
    arena, _served = _descent_arena(
        ntok=[4, 4], width=96, texts=["filler one", "filler two"])
    picked = arena._identifier_bearing_children(
        0, "What is the current Meridian docket value?", set(), [0, 1])
    # Nothing binds: the whole chunk set stands, in document order.
    assert picked == [0, 1]
    picked2, binds = arena._identifier_bearing_children(
        0, "What is the current Meridian docket value?", set(), [0, 1],
        with_binding_flag=True)
    assert picked2 == [0, 1] and binds is False


def test_non_binding_chunks_never_outrank_a_binding_plan_member(monkeypatch):
    """REGRESSION (LSR-P2C G2 Arm 1, third run, sup_solace_fresh).

    A-DEC planned ``[2, 1, 0]`` under ``one_off_rank_identifier_insurance_k3``
    with the long competitor at rank 1 and the ANSWER at rank 2.  Neither of
    the competitor's chunks binds "solace key", but substituting them where
    their parent stood put them ahead of the answer node and the passing
    control served the competitor's value.

    Demoting them to the back was not enough (run 5): a plan member is owed a
    SHUTTLE TRIP by Ruling 1.2, so the trailing chunks still produced
    ``fit_shuttle_trips [[0], [3], [4]]`` and trip 4 served the competitor's
    ``Sable-0-Copper``.  When the identifier binds some OTHER plan member,
    non-binding chunks do not enter the plan AT ALL.
    """
    _pin_arena_profile(monkeypatch, {
        "ranking": [2, 1, 0],
        "identified_candidates": [1],
        "identifier_tokens": ["solace", "key"],
        "identifier_hit_count": 1,
        "route_margin_1_2": 0.0,
        "rank_plan": [2, 1, 0],
        "policy_branch": "one_off_rank_identifier_insurance_k3",
    })
    competitor = "\n".join([
        "long planning transcript filler one two three four five six",
        "it records tundra ledger sable-0-copper and nothing else here",
        "trailing filler seven eight nine ten eleven twelve thirteen",
    ])
    answer = "the current solace key value is raven-9-ivory"
    ntok = [6, len(answer.split()), len(competitor.split())]
    arena, _served = _descent_arena(
        ntok=ntok, width=13, texts=["a", answer, competitor])
    _ans, info = arena.step("What is the current Solace key value?",
                            ngen=1, deposit=False, max_trips=1)
    assert info["fit_split_parent"] == 2
    plan = info["fit_planned"]
    # The ANSWER node keeps the front of the plan and the non-binding chunks
    # are NOT in it at all, so no shuttle trip can hand the turn to them.
    assert plan == [1, 0], plan
    assert not set(info["fit_split_children"]) & set(plan)
    for trip in info["fit_shuttle_trips"]:
        assert not set(trip) & set(info["fit_split_children"]), trip
    assert info["fit_chunk_trips"] == []
    # And the answer node is what gets seated.
    assert 1 in info["fit_seated"]


def test_chunk_shuttle_when_the_children_do_not_co_fit(monkeypatch):
    """Two identifier-bearing chunks that cannot co-seat are SERIALIZED."""
    _pin_arena_profile(monkeypatch, _LIVED_PROFILE)
    wide = "\n".join([
        "the current meridian docket value is alpha-1-one "
        + " ".join(f"w{i}" for i in range(6)),
        "the current meridian docket value is beta-2-two "
        + " ".join(f"x{i}" for i in range(6)),
    ])
    ntok = [2, 2, 2, len(wide.split())]
    arena, served = _descent_arena(
        ntok=ntok, width=12, texts=["a", "b", "c", wide], grounded=False)
    _answer, info = arena.step("What is the current Meridian docket value?",
                               ngen=1, deposit=False, max_trips=1)
    assert info["fit_split_parent"] == 3
    assert len(info["fit_descended_head"]) == 2
    # Neither chunk co-fits with the other at width 10, so the second one is
    # owed its OWN trip, in DOCUMENT ORDER.
    assert info["fit_chunk_trips"], "a non-co-fitting chunk set must shuttle"
    seated_across_turn = {v for trip in served for v in trip}
    assert set(info["fit_descended_head"]) <= seated_across_turn


def test_chunk_trips_never_exceed_the_registered_cap(monkeypatch):
    _pin_arena_profile(monkeypatch, _LIVED_PROFILE)
    wide = "\n".join(
        f"docket code-{i}-x " + " ".join(f"w{i}{j}" for j in range(6))
        for i in range(5)
    )
    ntok = [2, 2, 2, len(wide.split())]
    arena, _served = _descent_arena(
        ntok=ntok, width=8, texts=["a", "b", "c", wide], grounded=False)
    _answer, info = arena.step("meridian docket?", ngen=1, deposit=False,
                               max_trips=1)
    assert len(info["fit_chunk_trips"]) <= chunk_trip_cap(
        info["fit_descended_head"])


def test_served_without_plan_head_only_when_every_chunk_trip_fails(
        monkeypatch):
    """The tightened Ruling 1.4 for P2C.

    After a split, the answer is composed from the GROUNDED trips, so
    ``served_without_plan_head`` is true ONLY when no chunk of the plan head
    was ever seated by a grounded trip.
    """
    _pin_arena_profile(monkeypatch, _LIVED_PROFILE)
    texts = ["a", "b", "c", _COMPETITOR_TEXT]
    ntok = [4, 4, 4, len(_COMPETITOR_TEXT.split())]

    # (a) A chunk trip grounds -> the plan head WAS served.
    arena, _served = _descent_arena(
        ntok=ntok, width=11, texts=texts, grounded=True)
    _ans, info = arena.step("meridian docket?", ngen=1, deposit=False,
                            max_trips=1)
    assert set(info["fit_descended_head"]) & set(info["fit_seated"])
    assert info["served_without_plan_head"] is False

    # (b) NOTHING grounds anywhere -> the turn kept the first attempt, which
    #     is the descended head itself, so the head still reached a seat.
    #     The honest degrade is reserved for the case where no chunk of the
    #     head is in the served mount set at all.
    arena2, _served2 = _descent_arena(
        ntok=ntok, width=11, texts=texts, ground_only=[0])
    _ans2, info2 = arena2.step("meridian docket?", ngen=1, deposit=False,
                               max_trips=1)
    served_head = bool(
        set(info2["fit_descended_head"]) & set(info2["fit_seated"]))
    assert info2["served_without_plan_head"] is not served_head


def test_split_substitutes_in_place_and_never_demotes_a_seatable_member(
        monkeypatch):
    """REGRESSION (LSR-P2C G2 Arm 1, first run on fresh_fact_controls).

    The plan was ``[2, 1, 0]`` under ``one_off_rank_identifier_insurance_k3``;
    node 2 is the long competitor (unseatable) and node 1 carries the answer.
    Promoting the split competitor's CHILDREN to the plan head demoted the
    seatable answer node behind them and turned a PASSING control into a
    refusal.  The children must stand exactly where their parent stood.
    """
    _pin_arena_profile(monkeypatch, {
        "ranking": [2, 1, 0],
        "identified_candidates": [1],
        "identifier_tokens": ["solace", "key"],
        "identifier_hit_count": 1,
        "route_margin_1_2": 0.0,
        "rank_plan": [2, 1, 0],
        "policy_branch": "one_off_rank_identifier_insurance_k3",
    })
    competitor = "\n".join([
        "long competitor preface one two three four five six seven",
        "the current solace key value is raven-9-ivory in this record",
        "trailing filler eight nine ten eleven twelve thirteen",
    ])
    ntok = [6, 6, len(competitor.split())]
    arena, _served = _descent_arena(
        ntok=ntok, width=11, texts=["a", "b", competitor])
    _ans, info = arena.step("What is the current Solace key value?",
                            ngen=1, deposit=False, max_trips=1)
    assert info["fit_split_parent"] == 2
    plan = info["fit_planned"]
    # The BINDING child stands exactly where its parent stood (rank 0), and
    # the seatable members 1 and 0 keep their positions BEHIND it — neither is
    # pushed out of the plan.
    assert plan[-2:] == [1, 0], plan
    assert set(plan[:-2]) <= set(info["fit_split_children"])
    assert set(info["fit_descended_head"]) == set(plan[:-2])


def test_split_is_a_one_time_repair_not_re_split_every_turn(monkeypatch):
    """REGRESSION (LSR-P2C G2 Arm 1, first run on fresh_fact_controls).

    An index node the guard already split keeps its full ``ntok`` (its text is
    still the whole original), so a naive ``ntok > budget`` test split it
    AGAIN on the next probe: node 2 became children [3, 4] on probe 2 and
    [5, 6] on probe 3, leaving two rival child families in the routing surface
    (ranking [2, 4, 3, 1, 0]).
    """
    _pin_arena_profile(monkeypatch, _LIVED_PROFILE)
    texts = ["a", "b", "c", _COMPETITOR_TEXT]
    ntok = [4, 4, 4, len(_COMPETITOR_TEXT.split())]
    arena, _served = _descent_arena(ntok=ntok, width=11, texts=texts)
    _a1, info1 = arena.step("meridian docket delta-4-drift", ngen=1,
                            deposit=False, max_trips=1)
    first_children = list(info1["fit_split_children"])
    assert first_children
    graft_count = len(arena.grafts)
    _a2, info2 = arena.step("meridian docket delta-4-drift", ngen=1,
                            deposit=False, max_trips=1)
    # SAME children, no new family, no new grafts appended by a second split.
    assert info2["fit_split_children"] == first_children
    assert len(arena.grafts) == graft_count


def test_a_demoted_non_binding_split_gets_no_chunk_trips(monkeypatch):
    """REGRESSION (LSR-P2C G2 Arm 1, fourth run, sup_solace_fresh).

    Chunk trips exist to SERIALIZE the answer across the chunks of the node
    the probe is about.  A split whose chunks bind NOTHING is not that node:
    giving them their own trips let the competitor's filler ground the turn
    (``Sable-0-Copper``) while the planned answer node went unread, and the
    receipt then claimed ``served_without_plan_head = False``.
    """
    _pin_arena_profile(monkeypatch, {
        "ranking": [2, 1, 0],
        "identified_candidates": [1],
        "identifier_tokens": ["solace", "key"],
        "identifier_hit_count": 1,
        "route_margin_1_2": 0.0,
        "rank_plan": [2, 1, 0],
        "policy_branch": "one_off_rank_identifier_insurance_k3",
    })
    competitor = "\n".join([
        "long planning transcript filler one two three four five six",
        "it records tundra ledger sable-0-copper and nothing else here",
        "trailing filler seven eight nine ten eleven twelve thirteen",
    ])
    answer = "the current solace key value is raven-9-ivory"
    ntok = [6, len(answer.split()), len(competitor.split())]
    arena, served = _descent_arena(
        ntok=ntok, width=13, texts=["a", answer, competitor],
        ground_only=[3])          # only a competitor CHUNK would ground
    _ans, info = arena.step("What is the current Solace key value?",
                            ngen=1, deposit=False, max_trips=1)
    assert info["fit_split_parent"] == 2
    # No chunk trips at all: the chunks bind nothing.
    assert info["fit_chunk_trips"] == []
    # And if a chunk still ends up seated, the receipt says the plan head was
    # NOT served — never an unlabeled substitution (Ruling 1.4).
    if not set(info["fit_seated"]) & {int(info["fit_planned"][0])}:
        assert info["served_without_plan_head"] is True


def test_unsplittable_node_still_degrades_explicitly(monkeypatch):
    """RED honesty: a node with nothing to cut on is still reported, not
    silently substituted.  P2A's explicit degrade remains the outcome."""
    _pin_arena_profile(monkeypatch, _LIVED_PROFILE)
    # One atom, no whitespace and no punctuation to cut on.
    atom = "x" * 400
    arena, _served = _descent_arena(
        ntok=[4, 4, 4, 130], width=96, texts=["a", "b", "c", atom])
    _ans, info = arena.step("meridian docket?", ngen=1, deposit=False,
                            max_trips=1)
    assert info["fit_split_parent"] is None
    assert info["fit_split_children"] == []
    assert info["fit_unseatable"] == [3]
    assert info["fit_dropped_planned"] == [3]
    assert info["served_without_plan_head"] is True


def test_legacy_path_is_byte_identical_under_p2c(monkeypatch):
    """GRM_ADM_DECISIVE=0: no plan, no fit fields, no split fields."""
    monkeypatch.setenv("GRM_ADM_DECISIVE", "0")
    arena, served = _descent_arena(ntok=[1, 1, 1], width=16)
    arena.decisive_admission = False
    route_calls = []
    arena.route = lambda text, *, exclude, limit: (
        route_calls.append((text, set(exclude), limit)) or [0, 1, 2])
    answer, info = arena.step("praxis dock?", ngen=1, deposit=False)
    assert answer == "answer"
    assert route_calls == [("praxis dock?", set(), 3)]
    assert served == [[0, 1, 2]]
    for key in (
        "fit_planned", "fit_seated", "fit_unseatable", "fit_shuttle",
        "fit_split_parent", "fit_split_children", "fit_split_ephemeral",
        "fit_descended_head", "fit_chunk_trips",
        "served_without_plan_head", "abstained",
    ):
        assert key not in info, f"legacy info gained {key}"


def test_lsr_fixes_off_disables_the_split(monkeypatch):
    """G2 Arm-0 switch: with GRM_LSR_FIXES=0 no split runs."""
    monkeypatch.setenv("GRM_LSR_FIXES", "0")
    _pin_arena_profile(monkeypatch, _LIVED_PROFILE)
    texts = ["a", "b", "c", _COMPETITOR_TEXT]
    ntok = [4, 4, 4, len(_COMPETITOR_TEXT.split())]
    arena, _served = _descent_arena(ntok=ntok, width=11, texts=texts)
    _ans, info = arena.step("meridian docket?", ngen=1, deposit=False,
                            max_trips=1)
    assert info["fit_split_parent"] is None
    assert info["fit_split_children"] == []
    # The pre-P2C outcome stands: UNSEATABLE, explicit degrade.
    assert info["fit_unseatable"] == [3]
    assert info["served_without_plan_head"] is True


def test_lsr_fixes_default_is_on():
    assert e2e.lsr_fixes_enabled() is True
    assert e2e.lsr_fixes_enabled(True) is True
    assert e2e.lsr_fixes_enabled(False) is False


@pytest.mark.parametrize("token", ["0", "false", "off", "no", "FALSE"])
def test_lsr_fixes_env_tokens(monkeypatch, token):
    monkeypatch.setenv("GRM_LSR_FIXES", token)
    assert e2e.lsr_fixes_enabled() is False
    assert ArenaCache._lsr_fixes_enabled() is False


def test_lsr_fixes_unknown_token_fails_closed_to_on(monkeypatch):
    monkeypatch.setenv("GRM_LSR_FIXES", "wobble")
    assert e2e.lsr_fixes_enabled() is True
    assert ArenaCache._lsr_fixes_enabled() is True


# ---------------------------------------------------------------------------
# Part 2 — the DRIVER fit site (the site the lived probes went through)
# ---------------------------------------------------------------------------


class _DriverRepo:
    def __init__(self, arena):
        self.arena = arena
        self._paging_tel = SimpleNamespace(
            snapshot_enabled=False, enabled=False)

    def _snapshot_state(self):
        return []


def _pin_driver_profile(monkeypatch, profile):
    monkeypatch.setattr(
        e2e, "decisive_admission_profile",
        lambda _arena, _text, *, exclude, route_limit: dict(profile),
    )
    monkeypatch.setattr(
        e2e, "_probe_finish_deposit",
        lambda _repo, _before, _u, _a, info, *, defer_memory: info,
    )


_DRIVER_PROFILE = {
    **_LIVED_PROFILE,
    "route_backend": "python",
    "margin_threshold": 0.1385774091529802,
    "rule_sha256": (
        "c304609f81475bd2ae3399ad180d3bb00cc5d2b44b1a49db570387c810defb91"),
    "route_margin_evaluated": True,
}


def _driver_arena(*, ntok, width, texts):
    arena, _served = _descent_arena(ntok=ntok, width=width, texts=texts)
    arena._query_lex_tokens = lambda _t: {"meridian", "docket"}
    arena._node_text_tokens = lambda t: set(str(t).split())
    arena._format_step_turn = lambda u, a: f"User: {u}\nAssistant: {a}\n"

    def attempt(_q, picks, _ngen, _dep, _stops, defer_memory=False):
        arena.cur_mounts = list(picks)
        arena.served.append(list(picks))
        return "answer", {"mounts": [v + 1 for v in picks]}

    arena.served = []
    arena._attempt = attempt
    return arena


def test_driver_fit_site_descends_and_seats_the_identifier_child(monkeypatch):
    _pin_driver_profile(monkeypatch, _DRIVER_PROFILE)
    texts = ["a", "b", "c", _COMPETITOR_TEXT]
    ntok = [4, 4, 4, len(_COMPETITOR_TEXT.split())]
    arena = _driver_arena(ntok=ntok, width=11, texts=texts)
    _ans, info = e2e._probe_ladder_chat(
        _DriverRepo(arena), "What is the current Meridian docket value?",
        topk=3, ngen=1, max_trips=1)
    assert info["fit_split_parent"] == 3
    assert info["fit_descended_head"]
    assert set(info["fit_descended_head"]) & set(info["fit_seated"])
    assert info["served_without_plan_head"] is False


def test_driver_fit_site_arm0_reproduces_the_pre_p2a_packing(monkeypatch):
    """With the fixes OFF the driver reverts to rank-order packing and adds
    no fit_* fields at all — the Arm-0 reproduction contract."""
    monkeypatch.setenv("GRM_LSR_FIXES", "0")
    _pin_driver_profile(monkeypatch, _DRIVER_PROFILE)
    texts = ["a", "b", "c", _COMPETITOR_TEXT]
    ntok = [4, 4, 4, len(_COMPETITOR_TEXT.split())]
    arena = _driver_arena(ntok=ntok, width=11, texts=texts)
    _ans, info = e2e._probe_ladder_chat(
        _DriverRepo(arena), "What is the current Meridian docket value?",
        topk=3, ngen=1, max_trips=1)
    for key in ("fit_split_parent", "fit_planned", "fit_unseatable",
                "served_without_plan_head"):
        assert key not in info, f"Arm 0 info gained {key}"
    # The lived truncation shape: the unseatable node never gets a seat and
    # the cheaper neighbours do.
    assert 3 not in info["mount_fitted"]


# ---------------------------------------------------------------------------
# P2B pass-through — the new fields persist for free
# ---------------------------------------------------------------------------


def test_p2c_fields_are_fit_prefixed_and_pass_through_to_the_route_receipt():
    fields = split_info_fields(
        split_parent=3,
        split_children=[4, 5],
        split_ephemeral=False,
        descended_head=[4],
        chunk_trips=[[5]],
    )
    assert set(fields) == {
        "fit_split_parent", "fit_split_children", "fit_split_ephemeral",
        "fit_descended_head", "fit_chunk_trips",
    }
    for key in fields:
        assert key.startswith(ROUTE_RECEIPT_INFO_PREFIXES)
    # P2B's generic pass-through picks every one of them up unedited.
    passed = _route_receipt_generic_info(dict(fields))
    assert set(passed) == set(fields)
    assert passed["fit_split_parent"] == 3
    assert passed["fit_split_children"] == [4, 5]
    assert passed["fit_split_ephemeral"] is False
    assert passed["fit_descended_head"] == [4]
    assert passed["fit_chunk_trips"] == [[5]]


def test_split_info_fields_are_always_present_even_when_nothing_split():
    fields = split_info_fields()
    assert fields["fit_split_parent"] is None
    assert fields["fit_split_children"] == []
    assert fields["fit_split_ephemeral"] is None
    assert fields["fit_descended_head"] == []
    assert fields["fit_chunk_trips"] == []
