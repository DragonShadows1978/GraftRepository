"""GRM-EB1 — the ephemeral boat is the PRODUCTION frame.

THE SPEC (David, 2026-09-02).  "The chat log is not kept in memory context.
Any chat recall on facts is pulled via GRM, that way the chat can grow to any
length, limited only by RAM and NVMe."

WHAT THESE TESTS PIN.

  1. ``ephemeral=True`` is the DEFAULT on every constructor path, and the
     registered escape ``GRM_PERSISTENT_BOAT`` is the only way to get the old
     persistent live window back.  The escape FAILS CLOSED to the spec frame.

  2. Under the spec frame, after N turns the live cache holds only turn N's
     segments — no prior turn survives across a turn boundary.  This is the
     spec restated as a measurable postcondition, and it is measured on a
     real ``step()`` rather than asserted from the window arithmetic.

  3. Recency stays a MOUNT, and every served turn carries the registered
     frame receipt so a reader can see which frame a turn ran under and what
     recency cost it.

  4. The e2e driver's resume path does NOT re-feed transcript turns into the
     live cache under the spec frame (a chat log in context is the deviation),
     and the re-feed machinery is kept only for the escape.

Every test here is CPU-only: it drives the ``__new__``-built stub-arena
harness established by ``tests/test_grm_admission.py``, so no GPU lease is
consumed.
"""

from __future__ import annotations

import inspect
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.graft_arena import ArenaCache  # noqa: E402
from core.grm_frame import (  # noqa: E402
    ENV_NAME,
    env_persistent_boat,
    ephemeral_frame_enabled,
    frame_receipt,
)
from core.grm_three_pass import (  # noqa: E402
    ROUTE_RECEIPT_INFO_PREFIXES,
    _route_receipt_generic_info,
)
from scripts import grm_e2e_session as e2e  # noqa: E402


# ===========================================================================
# 1. The escape resolves, and it fails CLOSED to the spec frame
# ===========================================================================


def test_spec_frame_is_the_default_when_the_escape_is_unset():
    assert ephemeral_frame_enabled(None, {}) is True
    assert env_persistent_boat({}) is False


@pytest.mark.parametrize("token", ["1", "true", "yes", "on", "TRUE", " On "])
def test_true_tokens_select_the_persistent_frame(token):
    env = {ENV_NAME: token}
    assert env_persistent_boat(env) is True
    assert ephemeral_frame_enabled(None, env) is False


@pytest.mark.parametrize("token", ["0", "false", "no", "off", "", "OFF"])
def test_false_tokens_select_the_spec_frame(token):
    env = {ENV_NAME: token}
    assert env_persistent_boat(env) is False
    assert ephemeral_frame_enabled(None, env) is True


@pytest.mark.parametrize("token", ["maybe", "2", "persistent", "ephemeral",
                                   "tru", "yes please", "-1"])
def test_unknown_tokens_FAIL_CLOSED_TO_THE_SPEC_FRAME(token):
    """A mistyped escape must never re-enable the chat-log-in-context frame.

    The direction is deliberate and is the opposite of ``GRM_LSR_FIXES``
    (which fails closed to ON): in each case the unreadable token lands on
    the state that does NOT change what the stack serves.  Here that state is
    the spec frame.
    """
    env = {ENV_NAME: token}
    assert env_persistent_boat(env) is False
    assert ephemeral_frame_enabled(None, env) is True


def test_explicit_caller_outranks_the_escape_in_both_directions():
    """A harness must be able to pin either frame regardless of the ambient
    operator setting -- that is how frozen persistent-frame receipts get
    reproduced while the operator's shell is on the spec frame, and how a
    spec-frame gate runs while the escape is exported."""
    assert ephemeral_frame_enabled(False, {}) is False
    assert ephemeral_frame_enabled(True, {ENV_NAME: "1"}) is True


def test_frame_receipt_reports_the_frame_and_whether_the_escape_chose_it():
    assert frame_receipt(True, {}) == {
        "frame_ephemeral": True, "frame_escape_active": False}
    assert frame_receipt(False, {ENV_NAME: "1"}) == {
        "frame_ephemeral": False, "frame_escape_active": True}


# ===========================================================================
# 2. The CONSTRUCTOR default, and the escape at the constructor
# ===========================================================================


class _StubModel:
    def extend_rope(self, _length):
        pass


class _StubArena(ArenaCache):
    def _harvest(self, _ids):
        return ()


def _make_constructor_probe(**kwargs):
    return _StubArena(
        _StubModel(), encode=lambda _text: [1], decode=lambda _ids: "",
        **kwargs)


def test_arena_constructor_defaults_to_the_spec_frame(monkeypatch):
    monkeypatch.delenv(ENV_NAME, raising=False)
    assert _make_constructor_probe().ephemeral is True


def test_arena_constructor_honours_escape_and_explicit_pin(monkeypatch):
    monkeypatch.setenv(ENV_NAME, "1")
    assert _make_constructor_probe().ephemeral is False
    # An explicit pin beats the escape, in both directions.
    assert _make_constructor_probe(ephemeral=True).ephemeral is True
    monkeypatch.setenv(ENV_NAME, "0")
    assert _make_constructor_probe().ephemeral is True
    assert _make_constructor_probe(ephemeral=False).ephemeral is False


def test_arena_constructor_fails_closed_on_an_unknown_escape_token(monkeypatch):
    monkeypatch.setenv(ENV_NAME, "definitely-not-a-boolean")
    assert _make_constructor_probe().ephemeral is True


def test_recency_mounts_default_is_unchanged_by_the_frame_change(monkeypatch):
    """Recency grafts are pulled from the REPOSITORY and co-seated as mounts,
    so they are repository reads and not a retained chat log.  EB1 does not
    touch them; pinning the default here says so."""
    monkeypatch.delenv(ENV_NAME, raising=False)
    assert _make_constructor_probe().recency_mounts == 2
    monkeypatch.setenv(ENV_NAME, "1")
    assert _make_constructor_probe().recency_mounts == 2


# ===========================================================================
# 3. After N turns the live cache holds ONLY turn N's segments
# ===========================================================================


def _step_probe(*, ephemeral: bool, live_turns: int = 2):
    """A ``__new__``-built arena that runs the REAL ``step()``.

    Only the model forward, the router and the deposit seams are stubbed.
    The boat-clear, the recency nomination, the fit budget and the receipt
    assembly are the production code paths.
    """
    arena = ArenaCache.__new__(ArenaCache)
    arena.m = SimpleNamespace(
        layers=[SimpleNamespace(self_attn=SimpleNamespace(live_shift=None))])
    arena.live_shift = 9
    arena.stop_sequences = ()
    arena.ephemeral = bool(ephemeral)
    arena.recency_mounts = 2
    arena.live_segs = []
    arena.live_turns = int(live_turns)
    arena.topk = 3
    arena.decisive_admission = False
    arena.caches = None
    arena.pos = 0
    arena.cur_mounts = []
    arena.cur_mount_n = 0
    arena.grafts = [
        {"text": "alpha turn", "ntok": 4, "kind": "turn"},
        {"text": "bravo turn", "ntok": 4, "kind": "turn"},
        {"text": "charlie turn", "ntok": 4, "kind": "turn"},
    ]
    arena.width = 96
    arena.n_sink = 3
    arena._s4_turn = 0
    arena._clear_transients = lambda: None
    # ``evict()`` reslices real cache tensors; the stub's cache is an empty
    # tensor list, so the per-layer loop is a no-op and only the SEGMENT
    # bookkeeping -- the thing under test -- runs for real.
    arena._evict_cache_tensor = (
        lambda t, dim, head, drop_n: t)
    arena.route = lambda text, *, exclude, limit: [
        i for i in range(len(arena.grafts)) if i not in set(exclude)]
    arena._rare_tokens = lambda _text: set()
    arena._descent_expand = lambda picks, _kinds, qrare=None: list(picks)
    arena._resolve_revision_mounts = lambda picks: list(picks)
    arena._next_s4_turn = lambda: 1
    arena._bump_cuda_gqa_epoch = lambda: None
    arena._commit_s4_attempt = lambda *_args, **_kwargs: None
    arena._grounding_attribution = lambda _answer, picks, _question: (
        True, list(picks))

    counter = {"n": 0}

    def attempt(_question, picks, _ngen, _deposit, _stops, **_kw):
        # Stand in for the real attempt: seat the mounts and append THIS
        # turn's live segment, exactly as the production path does.
        arena.cur_mounts = list(picks)
        arena.cur_mount_n = sum(int(arena.grafts[i]["ntok"]) for i in picks)
        counter["n"] += 1
        # The real ``_attempt`` appends this turn's live segment and then
        # calls ``evict()``; the stub does both so the persistent-frame arm
        # measures the REAL window arithmetic and not a stub artifact.
        arena.live_segs.append((None, 7))
        arena.caches = arena.caches or []
        arena.evict()
        return f"answer-{counter['n']}", {
            "mounts": [v + 1 for v in picks],
            "live_tokens": sum(n for _g, n in arena.live_segs),
        }

    arena._attempt = attempt
    return arena


def test_spec_frame_live_cache_holds_only_the_CURRENT_turns_segments():
    """THE SPEC, restated as a postcondition and measured.

    Five turns are served.  After each one the live cache must hold exactly
    the segments that turn produced -- one -- and the turn must have INHERITED
    nothing from its predecessor.  That is "the chat log is not kept in
    memory context" as an assertion about the KV cache.
    """
    arena = _step_probe(ephemeral=True)
    for turn in range(1, 6):
        _answer, info = arena.step(f"question {turn}", ngen=1, deposit=False)
        assert info["frame_ephemeral"] is True
        # Nothing carried INTO this turn's model context ...
        assert info["live_segments_carried_into_turn"] == [], (
            f"turn {turn} started from a live window: "
            f"{info['live_segments_carried_into_turn']}")
        # ... and exactly this turn's own segment carried out.
        assert len(info["live_segments_after_turn"]) == 1, (
            f"turn {turn} left {len(info['live_segments_after_turn'])} live "
            "segments; the spec frame allows only the current turn's")
        assert len(arena.live_segs) == 1
        # From turn 2 on the previous turn's segment WAS there and WAS
        # discarded -- the clear is measured, not assumed.
        if turn > 1:
            assert len(info["live_segments_inherited"]) == 1


def test_persistent_frame_ACCUMULATES_a_live_window_across_turns():
    """The control arm, and the reason the spec frame is a change.

    Under the escape the live window is exactly what P2C.1 measured on t33: a
    prior turn's tokens sitting in the model's context, invisible to the
    router.  ``live_turns=2`` keeps two of them.
    """
    arena = _step_probe(ephemeral=False, live_turns=2)
    after = []
    carried_in = []
    for turn in range(1, 6):
        _answer, info = arena.step(f"question {turn}", ngen=1, deposit=False)
        assert info["frame_ephemeral"] is False
        after.append(len(info["live_segments_after_turn"]))
        carried_in.append(len(info["live_segments_carried_into_turn"]))
    # It grows to the window and then holds there.
    assert after == [1, 2, 2, 2, 2]
    # THE CONTRAST WITH THE SPEC FRAME: from turn 2 on, this frame carries a
    # prior turn's tokens INTO the model's context. Under the spec frame the
    # same list is [0, 0, 0, 0, 0].
    assert carried_in == [0, 1, 2, 2, 2]


def test_spec_frame_excludes_only_recency_from_routing_not_a_live_window():
    """Under the spec frame ``live_idx`` -- the routing exclusion set -- is
    the RECENCY nomination alone, because the live window is empty.  A fact a
    probe asks about is therefore routable unless recency itself mounted it,
    which is the mechanism the t33 prediction rests on."""
    arena = _step_probe(ephemeral=True)
    _answer, info = arena.step("question", ngen=1, deposit=False)
    assert info["recency_mounted_ids"] == [1, 2]
    assert info["live_segments_carried_into_turn"] == []


# ===========================================================================
# 4. The registered receipts, on every served turn
# ===========================================================================


def test_every_served_turn_carries_the_four_registered_receipt_fields():
    arena = _step_probe(ephemeral=True)
    _answer, info = arena.step("question", ngen=1, deposit=False)
    for field in ("frame_ephemeral", "recency_mounted_ids", "recency_seats",
                  "live_segments_after_turn"):
        assert field in info, f"missing registered receipt field {field}"
    assert info["frame_escape_active"] is False


def test_recency_seats_receipt_reports_the_width_and_the_actual_charge():
    """The recency COST table is read off this receipt, not re-derived.

    ``nominated_ntok`` is what the recency grafts would cost; ``charged_ntok``
    is what the fit stage actually subtracted from the arena width.  For a
    non-identifier query the charge is levied.
    """
    arena = _step_probe(ephemeral=True)
    _answer, info = arena.step("question", ngen=1, deposit=False)
    seats = info["recency_seats"]
    assert seats["arena_width"] == 96
    assert seats["nominated_ids"] == [1, 2]
    assert seats["nominated_ntok"] == 8           # two 4-token turn grafts
    assert seats["charged_ntok"] == 8
    assert seats["budget_after_charge"] == 96 - 8
    assert seats["charge_waived_reason"] is None


def test_identifier_queries_pay_ZERO_recency_seats_the_point_lookup_rule():
    """``rec_budget = 0 if qrare`` -- an identifier query is a point lookup,
    recency is already excluded from its mount set, and charging its seats too
    would be a double penalty.  The receipt NAMES the waiver so a reader can
    tell "paid nothing" from "recency nominated nothing"."""
    arena = _step_probe(ephemeral=True)
    arena._rare_tokens = lambda text: (
        {"marble-4-juliet"} if "polaris" in str(text) else set())
    _answer, info = arena.step("polaris mark?", ngen=1, deposit=False)
    seats = info["recency_seats"]
    assert seats["nominated_ids"] == [1, 2]
    assert seats["nominated_ntok"] == 8
    assert seats["charged_ntok"] == 0
    assert seats["budget_after_charge"] == 96
    assert seats["charge_waived_reason"] == "identifier_query_point_lookup"


def test_frame_and_recency_receipts_survive_into_the_route_receipt_record():
    """LSR-P2B persists ``info`` by PREFIX.  EB1 adds three prefixes so the
    frame a served turn ran under reaches the durable receipt."""
    assert "frame_" in ROUTE_RECEIPT_INFO_PREFIXES
    assert "recency_" in ROUTE_RECEIPT_INFO_PREFIXES
    assert "live_segments_" in ROUTE_RECEIPT_INFO_PREFIXES
    arena = _step_probe(ephemeral=True)
    _answer, info = arena.step("question", ngen=1, deposit=False)
    generic = _route_receipt_generic_info(info)
    for field in ("frame_ephemeral", "frame_escape_active",
                  "recency_mounted_ids", "recency_seats",
                  "live_segments_after_turn",
                  "live_segments_carried_into_turn",
                  "live_segments_inherited"):
        assert field in generic, f"{field} did not reach the route receipt"


# ===========================================================================
# 4b. BOTH production serving paths run ONE frame
# ===========================================================================


def test_the_driver_probe_path_opens_its_turn_through_the_shared_frame():
    """THE DEFECT THIS PINS, measured on the G2 sup battery.

    ``ArenaCache.step()`` and ``grm_e2e_session._probe_ladder_chat`` are BOTH
    production serving paths.  When only ``step()`` carried the ephemeral
    logic, the probe path run against an ephemeral arena served a THIRD frame:
    no live window (nothing feeds one) and no recency mounts either (it never
    nominated any), so the routing exclusion the persistent frame got for free
    from its live window simply vanished.  Three sup probes that the
    persistent frame served correctly regressed to refusals.

    The fix was to SHARE the turn-open, so this pins the sharing itself: the
    probe path must call the arena's own helper, and must derive its routing
    exclusion from that helper's recency nomination.
    """
    src = inspect.getsource(e2e._probe_ladder_chat)
    assert "arena.eb1_begin_turn()" in src
    assert "set(rec)" in src, (
        "the probe path must fold the recency nomination into live_idx")
    assert "arena.eb1_charge_recency(rec, id_tokens)" in src, (
        "the probe path must charge recency seats against its fit budget")


def test_the_probe_path_seats_recency_under_step_s_own_use_rec_rule():
    """``step()``'s rule is ``rec and not qrare and not clean and picks !=
    precise``.  The probe path must restate that rule, not invent one: an
    identifier lookup is a point read and a clean-room rung is the RECENCY LAW
    rung that exists to exclude these seats."""
    src = inspect.getsource(e2e._probe_ladder_chat)
    assert "use_rec = bool(rec) and not id_tokens and not clean" in src


def test_the_frame_receipt_is_stamped_at_the_probe_paths_single_funnel():
    """Every probe-path return passes through ``_probe_finish_deposit``, so
    the receipt is stamped there -- a new return site cannot ship a turn with
    no frame receipt."""
    src = inspect.getsource(e2e._probe_finish_deposit)
    assert "_eb1_frame_info" in src


def test_legacy_new_built_fixtures_keep_their_pre_EB1_contract():
    """``eb1_begin_turn`` must not explode on a ``__new__``-only fixture that
    never got an ``ephemeral`` field.  It falls back to the persistent frame
    for those, exactly as ``step()`` falls back for ``decisive_admission`` --
    a legacy fixture keeps its old contract without weakening the production
    default, which is set by the real constructor."""
    legacy = ArenaCache.__new__(ArenaCache)
    legacy.live_segs = []
    legacy.grafts = []
    legacy.width = 96
    assert legacy.eb1_begin_turn() == []
    info = legacy._eb1_frame_info()
    assert info["frame_ephemeral"] is False
    assert info["recency_mounted_ids"] == []


# ===========================================================================
# 5. ESCAPE-FRAME BYTE IDENTITY on a fixture turn
# ===========================================================================


def _pre_eb1_step_probe(*, live_turns: int = 2):
    """The arena as it behaved BEFORE EB1: ``ephemeral=False`` set directly.

    This is the reference the escape has to match.  It bypasses the frame
    resolver entirely -- the field is assigned, exactly as the pre-EB1
    constructor assigned it -- so the comparison below is against the old
    behaviour and not against a second reading of the new code.
    """
    arena = _step_probe(ephemeral=False, live_turns=live_turns)
    arena.ephemeral = False
    return arena


_FIXTURE_TURNS = ("What is the harbor token?",
                  "And the praxis dock?",
                  "What is the solace key?",
                  "Remind me of the harbor token.")


def _drive(arena, turns=_FIXTURE_TURNS):
    """Serve a fixed turn sequence and return the comparable receipt."""
    out = []
    for text in turns:
        answer, info = arena.step(text, ngen=1, deposit=False)
        out.append({
            "answer": answer,
            "mounts": list(info.get("mounts", ())),
            "cur_mounts": list(arena.cur_mounts),
            "cur_mount_n": int(arena.cur_mount_n),
            "live_tokens": info.get("live_tokens"),
            "recency_mounted_ids": list(info["recency_mounted_ids"]),
            "recency_seats": dict(info["recency_seats"]),
            "live_inherited": list(info["live_segments_inherited"]),
            "live_carried_in": list(info["live_segments_carried_into_turn"]),
            "live_after": list(info["live_segments_after_turn"]),
        })
    return out


def test_escape_reproduces_the_pre_EB1_frame_byte_for_byte_on_a_fixture_turn(
        monkeypatch):
    """G1: ``GRM_PERSISTENT_BOAT=1`` restores the OLD frame exactly.

    A frozen receipt taken before EB1 can only be reproduced if the escape
    gives back the same serve, not merely a similar one.  The whole four-turn
    fixture trace is compared -- answers, mount sets, seat counts and the
    live-window evolution -- because the frame's effect is on the SEQUENCE,
    and a per-turn spot check would miss a divergence that only shows up once
    the window has filled.
    """
    monkeypatch.setenv(ENV_NAME, "1")
    escaped = _step_probe(ephemeral=ephemeral_frame_enabled(), live_turns=2)
    assert escaped.ephemeral is False, "the escape did not select the frame"
    reference = _pre_eb1_step_probe(live_turns=2)

    assert _drive(escaped) == _drive(reference)


def test_the_spec_frame_and_the_escape_frame_actually_DIFFER(monkeypatch):
    """The identity test above is only meaningful if the two frames are not
    the same computation.  They are not: this pins the difference, so a bug
    that collapsed the escape into the default could not pass both tests."""
    monkeypatch.delenv(ENV_NAME, raising=False)
    spec = _drive(_step_probe(ephemeral=ephemeral_frame_enabled(),
                              live_turns=2))
    persistent = _drive(_pre_eb1_step_probe(live_turns=2))
    assert spec != persistent
    # and the difference is exactly the live window in the model's context
    assert [len(row["live_carried_in"]) for row in spec] == [0, 0, 0, 0]
    assert [len(row["live_carried_in"]) for row in persistent] == [0, 1, 2, 2]


# ===========================================================================
# 6. Resume is repository state, not transcript re-feed
# ===========================================================================


class _ResumeArena:
    """Records whether anything was pushed back through the live cache."""

    def __init__(self, ephemeral):
        self.ephemeral = bool(ephemeral)
        self.live_segs = []
        self.fed = []

    def feed(self, turn_text, deposit=True):
        self.fed.append((turn_text, deposit))
        self.live_segs.append((None, 5))


class _ResumeRepo:
    def __init__(self, ephemeral):
        self.arena = _ResumeArena(ephemeral)


_TRANSCRIPT = [
    {"turn": 1, "user": "u1", "assistant": "a1", "chat_node_id": 11},
    {"turn": 2, "user": "u2", "assistant": "a2", "chat_node_id": 12},
    {"turn": 3, "user": "u3", "assistant": "a3", "chat_node_id": 13},
]


def test_refeed_live_window_is_escape_only_machinery_and_still_works():
    """The function is KEPT -- frozen persistent-frame receipts have to stay
    reproducible -- so it is tested, not deleted.  What changed is that
    ``main()`` no longer calls it under the spec frame."""
    repo = _ResumeRepo(ephemeral=False)
    replayed = e2e.refeed_live_window(repo, _TRANSCRIPT, 2)
    assert [row["turn"] for row in replayed] == [2, 3]
    # deposit=False is the tell: it re-establishes CONTEXT without adding
    # anything to the repository -- a chat log in the cache.
    assert [deposit for _text, deposit in repo.arena.fed] == [False, False]
    assert repo.arena.live_segs[-1][0] == 13


def test_the_resume_path_under_the_spec_frame_feeds_NOTHING():
    """Resume = repository state.  Under the spec frame the driver must not
    push transcript turns back into the live cache; the turns are already
    grafts and recency-as-mount pulls them by ROUTING on the next step."""
    repo = _ResumeRepo(ephemeral=True)
    frame_ephemeral = bool(getattr(repo.arena, "ephemeral", False))
    assert frame_ephemeral is True
    # The production guard, restated: the re-feed runs only when NOT ephemeral.
    if not frame_ephemeral:  # pragma: no cover - the spec frame never enters
        e2e.refeed_live_window(repo, _TRANSCRIPT, 2)
    assert repo.arena.fed == []
    assert repo.arena.live_segs == []


def test_driver_source_pins_the_resume_guard_and_the_skip_reason():
    """The guard lives in ``main()``, which needs a model to run, so the
    contract is pinned against the driver SOURCE rather than executed here.
    This is a deliberate, named limitation: the behavioural leg is the G3
    census run, whose restart receipts carry ``refeed: []`` and the skip
    reason under the spec frame."""
    src = inspect.getsource(e2e.main)
    assert "frame_ephemeral" in src
    assert "refeed_skipped_reason" in src
    assert "spec_frame_resume_is_repository_state" in src
    assert "if frame_ephemeral:" in src


def test_driver_arena_kw_declares_the_spec_frame():
    """``load_model_and_repo`` must be EXPLICIT about the frame rather than
    inheriting a constructor default silently, so a reader of the driver can
    see which frame a session ran under."""
    src = inspect.getsource(e2e.load_model_and_repo)
    assert '"ephemeral": ephemeral_frame_enabled()' in src


def test_recency_augmented_probe_stays_exploratory_report_only():
    """EB1 does NOT change this construction.  It concatenates recent
    transcript turns into a ROUTING query only -- it is never fed to
    user-visible inference (``status: exploratory_report_only``), so it is not
    a chat log in context and the spec does not reach it."""
    text = e2e.recency_augmented_probe(
        "current polaris mark?", _TRANSCRIPT, live_turns=2)
    assert "current polaris mark?" in text
    assert "u3" in text and "a3" in text
    # The construction's only consumer stamps it exploratory-report-only.
    assert "exploratory_report_only" in inspect.getsource(e2e.run_step1_prep)
