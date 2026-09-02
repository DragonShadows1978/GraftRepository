"""GRM-SC1 — Stage C: D-NGH as the production demand detector.

ORDER: ``orders/GRM_SC1_DEMAND_LOOP_NGH.md``.

THE PRINCIPLE THESE TESTS PIN: **a turn that is missing a needed memory
notices mid-generation and fetches, exactly once, and says so** — and with
the flag OFF, nothing about serving changes at all.

The threshold is CARRIED from the GRM-DET1 race
(``thresholds_2149a44b6136fd1b.json``, D-NGH 0.3380523274342219, fit on TWO
calibration turns — a thin envelope).  Nothing here refits it, and one test
pins that the production constant equals the race constant byte-for-byte.

Every test in this module is CPU-only: it drives the stubbed arena/repository
harness established by ``tests/test_grm_admission.py`` and
``tests/test_grm_lsr_p2a_*.py``, so no GPU lease is consumed.  The
bit-identity of the observer against the frozen DET1 ``DetectorObserver`` is
G2 and needs real GPT-OSS attention operands; it lives in
``scripts/grm_sc1_bit_identity_gpu.py``, not here.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import grm_demand
from core.grm_three_pass import (
    ROUTE_RECEIPT_INFO_PREFIXES,
    _route_receipt_generic_info,
)
from scripts import grm_e2e_session as e2e


ROOT = Path(__file__).resolve().parents[1]
RACE_THRESHOLDS = (
    ROOT / "artifacts" / "grm_det1" / "run_20260831T160525Z_2" / "det1_4"
    / "campaign" / "calibration" / "thresholds_2149a44b6136fd1b.json"
)
THRESHOLD = 0.3380523274342219


# ---------------------------------------------------------------------------
# The carried constant
# ---------------------------------------------------------------------------


def test_registered_threshold_is_the_race_constant_not_a_refit():
    """SC1 registers NO new number: the config carries the race's own float.

    If this ever fails, someone refit a serving threshold — the exact thing
    the order forbids — and the demand loop must not run until it is put
    back.
    """
    payload = grm_demand.load_registered()
    assert payload["threshold"] == THRESHOLD
    assert payload["carried_not_refit"] is True
    assert payload["direction"] == "trigger_if_strictly_below"
    assert payload["score"] == "per-token full-layer mean mounted_mass"
    # The float is the race file's float, read from the race file.
    race = json.loads(RACE_THRESHOLDS.read_text(encoding="utf-8"))
    assert payload["threshold"] == race["D-NGH"]["threshold"]
    assert payload["direction"] == race["D-NGH"]["direction"]
    assert payload["score"] == race["D-NGH"]["score"]
    assert payload["selection"] == race["D-NGH"]["selection"]


def test_registered_config_states_the_two_turn_thin_envelope_caveat():
    """The envelope is thin and every receipt has to say so."""
    payload = grm_demand.load_registered()
    assert payload["fit_turn_count"] == 2
    assert len(payload["provenance"]["fit_fixture_ids"]) == 2
    assert "THIN ENVELOPE" in payload["caveat"]


def test_registered_config_refuses_a_threshold_that_lost_its_provenance(tmp_path):
    """Fail CLOSED: a config that drops ``carried_not_refit`` is refused."""
    payload = grm_demand.load_registered()
    payload.pop("carried_not_refit")
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(grm_demand.DemandError, match="CARRIED"):
        grm_demand.load_registered(path)


def test_registered_config_refuses_a_drifted_score_definition(tmp_path):
    payload = grm_demand.load_registered()
    payload["score"] = "per-token mean live_mass"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(grm_demand.DemandError, match="score drifted"):
        grm_demand.load_registered(path)


# ---------------------------------------------------------------------------
# The flag: DEFAULT OFF, fail closed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value,expected", [
    ("1", True), ("true", True), ("TRUE", True), ("yes", True), ("on", True),
    ("0", False), ("false", False), ("no", False), ("off", False),
    ("", False),
])
def test_flag_parses_the_registered_tokens(value, expected):
    assert grm_demand.demand_enabled(
        environ={"GRM_DEMAND_NGH": value}) is expected


def test_flag_defaults_off_when_unset():
    assert grm_demand.demand_enabled(environ={}) is False


@pytest.mark.parametrize("value", ["maybe", "2", "ON!", "enabled", "-1"])
def test_flag_fails_closed_to_off_on_an_unknown_token(value):
    """Opposite direction from GRM_LSR_FIXES, on purpose.

    The fixes switch fails closed to ON because an operator who mistypes it
    should keep the fixes.  This one is not the production default until
    David flips it, so a typo must never turn it ON.
    """
    assert grm_demand.demand_enabled(
        environ={"GRM_DEMAND_NGH": value}) is False


def test_explicit_argument_beats_the_environment():
    assert grm_demand.demand_enabled(
        True, environ={"GRM_DEMAND_NGH": "0"}) is True
    assert grm_demand.demand_enabled(
        False, environ={"GRM_DEMAND_NGH": "1"}) is False


# ---------------------------------------------------------------------------
# The decision rule (carried verbatim)
# ---------------------------------------------------------------------------


def _rows(*masses):
    """Observer rows, shaped exactly as ``DemandObserver.finish`` returns.

    ``prediction_token_id`` is part of that shape and is load-bearing: the
    demand query is rebuilt from the token IDS, never from re-tokenizing the
    decoded string (decode->encode is not a round trip on BPE).
    """
    return [
        {"token_index": i, "mounted_mass": float(m),
         "prediction_token_id": 1000 + i}
        for i, m in enumerate(masses)
    ]


def test_decision_fires_on_the_first_token_strictly_below():
    decision = grm_demand.decide(_rows(0.9, 0.5, 0.2, 0.1), THRESHOLD)
    assert decision["demand_fired"] is True
    # FIRST, not the deepest: token 2 is the first strictly-below position
    # even though token 3 is further under.
    assert decision["demand_token_index"] == 2
    assert decision["demand_min_mass"] == pytest.approx(0.1)


def test_decision_does_not_fire_at_exactly_the_threshold():
    """``trigger_if_strictly_below``: equality is NOT a fire.

    The threshold is the minimum of the served calibration turns' minima, so
    the fitting turn itself sits exactly ON it. A >= rule would have fired on
    its own calibration data.
    """
    decision = grm_demand.decide(_rows(0.9, THRESHOLD, 0.9), THRESHOLD)
    assert decision["demand_fired"] is False
    assert decision["demand_token_index"] is None
    assert decision["demand_min_mass"] == pytest.approx(THRESHOLD)


def test_decision_does_not_fire_just_above_and_does_just_below():
    eps = 1e-12
    assert grm_demand.decide(
        _rows(THRESHOLD + eps), THRESHOLD)["demand_fired"] is False
    assert grm_demand.decide(
        _rows(THRESHOLD - eps), THRESHOLD)["demand_fired"] is True


def test_decision_reports_the_minimum_even_when_it_did_not_fire():
    """An un-fired turn still says how close it came — the number David
    needs when he decides whether the envelope is too tight or too loose."""
    decision = grm_demand.decide(_rows(0.90, 0.42, 0.77), THRESHOLD)
    assert decision["demand_fired"] is False
    assert decision["demand_min_mass"] == pytest.approx(0.42)
    assert decision["demand_token_count"] == 3


def test_decision_on_no_tokens_is_not_a_fire():
    decision = grm_demand.decide([], THRESHOLD)
    assert decision["demand_fired"] is False
    assert decision["demand_min_mass"] is None


# ---------------------------------------------------------------------------
# Unsupported-arena refusal is LOUD
# ---------------------------------------------------------------------------


def test_unsupported_arena_without_layers_refuses_with_a_reason():
    support = grm_demand.arena_support(SimpleNamespace(m=None))
    assert support["demand_supported"] is False
    assert "no model layers" in support["demand_unsupported_reason"]


def test_unsupported_arena_without_full_attention_layers_refuses():
    arena = SimpleNamespace(m=SimpleNamespace(layers=[
        SimpleNamespace(self_attn=SimpleNamespace(
            layer_type="sliding_attention", attention_mode="standard",
            layer_idx=0)),
    ]))
    support = grm_demand.arena_support(arena)
    assert support["demand_supported"] is False
    assert "full-attention" in support["demand_unsupported_reason"]


def test_unsupported_non_standard_attention_mode_names_the_layers():
    arena = SimpleNamespace(m=SimpleNamespace(layers=[
        SimpleNamespace(self_attn=SimpleNamespace(
            layer_type="full_attention", attention_mode="apa", layer_idx=3)),
        SimpleNamespace(self_attn=SimpleNamespace(
            layer_type="full_attention", attention_mode="standard",
            layer_idx=5)),
    ]))
    support = grm_demand.arena_support(arena)
    assert support["demand_supported"] is False
    assert "[3]" in support["demand_unsupported_reason"]


def test_supported_arena_counts_only_the_full_attention_layers():
    arena = SimpleNamespace(m=SimpleNamespace(layers=[
        SimpleNamespace(self_attn=SimpleNamespace(
            layer_type="full_attention", attention_mode="standard",
            layer_idx=0)),
        SimpleNamespace(self_attn=SimpleNamespace(
            layer_type="sliding_attention", attention_mode="standard",
            layer_idx=1)),
        SimpleNamespace(self_attn=SimpleNamespace(
            layer_type="full_attention", attention_mode="standard",
            layer_idx=2)),
    ]))
    support = grm_demand.arena_support(arena)
    assert support["demand_supported"] is True
    assert support["demand_full_attention_layers"] == 2


def test_unsupported_info_is_a_refusal_not_a_quiet_no_demand():
    """The failure mode this guards: ``demand_fired=False`` with no reason is
    indistinguishable from "the memory was there". It must never be that."""
    info = grm_demand.unsupported_info("no GPT-OSS full-attention layers")
    assert info["demand_supported"] is False
    assert info["demand_unsupported_reason"]
    assert info["demand_fired"] is False
    assert info["demand_served"] == "original"


# ---------------------------------------------------------------------------
# The receipt block passes through the route receipt
# ---------------------------------------------------------------------------


def test_demand_prefix_is_registered_in_the_route_receipt_prefixes():
    assert "demand_" in ROUTE_RECEIPT_INFO_PREFIXES


def test_every_demand_field_survives_the_generic_route_receipt_passthrough():
    """P2B persists by PREFIX, so the whole block rides the one-line
    addition SC1 was authorized to make to ``core/grm_three_pass.py``."""
    fields = grm_demand.demand_info_fields(
        supported=True,
        decision=grm_demand.decide(_rows(0.9, 0.1), THRESHOLD),
        served="demand_trip",
        query_text_sha256="ab" * 32,
        ranking=[4, 2],
        fetched=[4],
        refired=False,
        refire_token_index=None,
        trip_taken=True,
        trip_grounded=True,
        threshold=THRESHOLD,
        prefix_token_count=1,
    )
    assert fields
    assert all(key.startswith("demand_") for key in fields)
    projected = _route_receipt_generic_info(fields)
    assert projected == fields


def test_refire_is_recorded_and_flagged_as_not_acted_on():
    fields = grm_demand.demand_info_fields(
        supported=True,
        decision=grm_demand.decide(_rows(0.1), THRESHOLD),
        served="demand_trip", refired=True, refire_token_index=0,
        trip_taken=True, trip_grounded=True)
    assert fields["demand_refired"] is True
    assert fields["demand_refire_acted_on"] is False
    assert fields["demand_trip_cap"] == grm_demand.DEMAND_TRIP_CAP == 1


def test_fetch_source_is_the_registered_reroute_not_d_lqr():
    """D-LQR was REFUTED-STRUCTURAL in the race. It is not a fetch source."""
    fields = grm_demand.demand_info_fields(supported=True)
    assert fields["demand_fetch_source"] == "question_plus_prefix_reroute"
    assert "lqr" not in json.dumps(fields).lower()


# ---------------------------------------------------------------------------
# Query augmentation
# ---------------------------------------------------------------------------


def test_prefix_is_the_output_before_the_fire_token():
    """The fire token is the first one the model produced while ADRIFT.

    Everything before it was produced while the model was still reading its
    mounts, so that is the part worth using as query augmentation; the fire
    token itself is deliberately excluded.
    """
    arena = SimpleNamespace(decode=lambda ids: "|".join(str(i) for i in ids))
    rows = [
        {"token_index": 0, "prediction_token_id": 11, "mounted_mass": 0.9},
        {"token_index": 1, "prediction_token_id": 22, "mounted_mass": 0.8},
        {"token_index": 2, "prediction_token_id": 33, "mounted_mass": 0.1},
    ]
    assert grm_demand.demand_prefix_text(arena, rows, 2) == "11|22"
    assert grm_demand.demand_prefix_text(arena, rows, 0) == ""


def test_query_is_question_plus_prefix_and_is_stable_under_hashing():
    assert grm_demand.demand_query_text("Q?", "partial") == "Q?\npartial"
    assert grm_demand.demand_query_text("Q?", "  ") == "Q?"
    first = grm_demand.demand_query_sha256("Q?\npartial")
    assert first == grm_demand.demand_query_sha256("Q?\npartial")
    assert len(first) == 64


# ---------------------------------------------------------------------------
# The driver hook: flag OFF byte-identity, and the loop when ON
# ---------------------------------------------------------------------------


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
    "rule_sha256":
        "c304609f81475bd2ae3399ad180d3bb00cc5d2b44b1a49db570387c810defb91",
    "route_margin_evaluated": True,
}

QUESTION = "What is the current Meridian docket value?"


class _DemandArena:
    """A stub arena that LOOKS like GPT-OSS to ``grm_demand.arena_support``.

    ``_attempt`` is scripted per mount set, so a test can say "the attempt
    that mounts [3] returns the wrong answer, the one that mounts [1] returns
    the right one" and then assert which one went out.  The observer itself
    is never installed here (there is no real attention to observe): the
    tests drive the DECISION by monkeypatching the rows, which keeps them
    honest about what is CPU-checkable and what is G2's job.
    """

    def __init__(self, *, ntok, width=96, answers=None, grounded=None):
        self.grafts = [
            {"text": f"node-{i}", "ntok": int(n), "kind": "fact"}
            for i, n in enumerate(ntok)
        ]
        self.width = int(width)
        self.decisive_admission = True
        self.live_segs = []
        self.caches = None
        self.pos = 0
        self.cur_mounts = []
        self.cur_mount_n = 0
        self.stop_sequences = ()
        self.live_shift = 9
        self.n_sink = 4
        self.topk = 3
        # A GPT-OSS-shaped layer stack, so arena_support says supported.
        self.m = SimpleNamespace(layers=[
            SimpleNamespace(self_attn=SimpleNamespace(
                live_shift=None, layer_type="full_attention",
                attention_mode="standard", layer_idx=0)),
            SimpleNamespace(self_attn=SimpleNamespace(
                live_shift=None, layer_type="sliding_attention",
                attention_mode="standard", layer_idx=1)),
        ])
        self._answers = dict(answers or {})
        self._grounded = dict(grounded or {})
        self.served = []
        self.route_calls = []

    def route(self, text, *, exclude, limit):
        self.route_calls.append((text, set(exclude), int(limit)))
        return [i for i in range(len(self.grafts)) if i not in set(exclude)]

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

    def decode(self, ids):
        return " ".join(str(i) for i in ids)

    def deposit(self, text):
        self.grafts.append({"text": text, "ntok": 3, "kind": "turn"})
        return len(self.grafts) - 1

    def _attempt(self, _question, picks, _ngen, _deposit, _stops,
                 defer_memory=False):
        key = tuple(sorted(int(v) for v in picks))
        self.cur_mounts = list(picks)
        self.served.append(list(key))
        return self._answers.get(key, "answer"), {"mounts": [v + 1 for v in key]}

    def _grounding_attribution(self, _ans, picks, _q):
        key = tuple(sorted(int(v) for v in picks))
        if key in self._grounded:
            return bool(self._grounded[key]), list(picks)
        return bool(picks), list(picks)


class _DemandRepo:
    def __init__(self, arena):
        self.arena = arena
        self._paging_tel = SimpleNamespace(
            snapshot_enabled=False, enabled=False)

    def _snapshot_state(self):
        return []


@pytest.fixture
def demand_harness(monkeypatch):
    """Pin the admission profile and neutralize the deposit/snapshot seams."""
    monkeypatch.setattr(
        e2e, "decisive_admission_profile",
        lambda _arena, _text, *, exclude, route_limit: {
            **_LIVED_PROFILE,
            "ranking": [i for i in _LIVED_PROFILE["ranking"]
                        if i not in set(exclude)],
            "rank_plan": [i for i in _LIVED_PROFILE["rank_plan"]
                          if i not in set(exclude)]
            or [i for i in _LIVED_PROFILE["ranking"]
                if i not in set(exclude)][:1],
        },
    )
    monkeypatch.setattr(
        e2e, "_probe_finish_deposit",
        lambda _repo, _before, _u, _a, info, *, defer_memory: info,
    )
    monkeypatch.setattr(
        e2e, "_probe_mount_snapshot",
        lambda *a, **k: None,
    )
    return monkeypatch


def _pin_rows(monkeypatch, sequence):
    """Drive the observer's verdict without real attention operands.

    ``sequence`` is a list of row-lists, consumed one per generation attempt.
    """
    queue = list(sequence)

    class _FakeObserver:
        # GRM-SC2 added the ``early_abort`` keyword to the real observer. This
        # stub accepts and RECORDS it but never aborts: these SC1 tests pin the
        # generate-then-trip behaviour, and an aborting stub would be testing
        # SC2's path under SC1's name. The SC2 tests drive the real observer.
        def __init__(self, arena, ngen, threshold, early_abort=False):
            self._rows = queue.pop(0) if queue else []
            self.early_abort = bool(early_abort)
            self.aborted_at = None

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def finish(self):
            return list(self._rows)

    monkeypatch.setattr(grm_demand, "DemandObserver", _FakeObserver)
    monkeypatch.setattr(e2e.grm_demand, "DemandObserver", _FakeObserver)


def test_flag_off_serves_byte_identically_and_adds_no_demand_fields(
    demand_harness, monkeypatch,
):
    """P2C parity: with the flag OFF, the served answer and every receipt
    field are what P2C produced, and no ``demand_*`` key appears at all."""
    monkeypatch.delenv("GRM_DEMAND_NGH", raising=False)
    arena = _DemandArena(ntok=[20, 25, 40, 30])
    ans, info = e2e._probe_ladder_chat(
        _DemandRepo(arena), QUESTION, topk=3, ngen=4, max_trips=1)
    assert ans == "answer"
    assert not [key for key in info if key.startswith("demand_")]
    served_off = list(arena.served)

    # Same harness, flag explicitly OFF: identical serving trace.
    arena2 = _DemandArena(ntok=[20, 25, 40, 30])
    monkeypatch.setenv("GRM_DEMAND_NGH", "0")
    ans2, info2 = e2e._probe_ladder_chat(
        _DemandRepo(arena2), QUESTION, topk=3, ngen=4, max_trips=1)
    assert ans2 == ans
    assert arena2.served == served_off
    assert not [key for key in info2 if key.startswith("demand_")]


def test_flag_on_but_no_fire_serves_the_original_and_records_the_decision(
    demand_harness, monkeypatch,
):
    monkeypatch.setenv("GRM_DEMAND_NGH", "1")
    _pin_rows(monkeypatch, [_rows(0.9, 0.8, 0.7)])
    arena = _DemandArena(ntok=[20, 25, 40, 30])
    ans, info = e2e._probe_ladder_chat(
        _DemandRepo(arena), QUESTION, topk=3, ngen=4, max_trips=1)
    assert ans == "answer"
    assert info["demand_supported"] is True
    assert info["demand_fired"] is False
    assert info["demand_served"] == "original"
    assert info["demand_trip_taken"] is False
    assert info["demand_min_mass"] == pytest.approx(0.7)
    # No trip means no second generation.
    assert len(arena.served) == 1


def test_flag_on_fire_takes_exactly_one_trip_and_serves_it_when_grounded(
    demand_harness, monkeypatch,
):
    """The whole mechanism, end to end: fire -> roll back -> re-route ->
    serve the trip's answer because it grounded."""
    monkeypatch.setenv("GRM_DEMAND_NGH", "1")
    # Attempt 1 fires at token 1; the demand trip does not fire.
    _pin_rows(monkeypatch, [
        _rows(0.9, 0.1, 0.2),
        _rows(0.9, 0.8),
    ])
    arena = _DemandArena(
        ntok=[20, 25, 40, 30],
        answers={(3,): "wrong", (0,): "RIGHT", (1,): "RIGHT", (2,): "RIGHT"},
    )
    ans, info = e2e._probe_ladder_chat(
        _DemandRepo(arena), QUESTION, topk=3, ngen=4, max_trips=1)
    assert info["demand_fired"] is True
    assert info["demand_token_index"] == 1
    assert info["demand_trip_taken"] is True
    assert info["demand_served"] == "demand_trip"
    assert ans == "RIGHT"
    # The trip fetched something the first attempt did not already have.
    assert info["demand_fetched"]
    assert 3 not in info["demand_fetched"]
    assert info["demand_query_text_sha256"]
    # ONE trip. Two generations total: the original attempt and the trip.
    assert len(arena.served) == 2


def test_the_demand_trip_excludes_what_was_already_mounted(
    demand_harness, monkeypatch,
):
    """A trip that re-fetches the nodes already read is not a fetch."""
    monkeypatch.setenv("GRM_DEMAND_NGH", "1")
    _pin_rows(monkeypatch, [_rows(0.1), _rows(0.9)])
    arena = _DemandArena(ntok=[20, 25, 40, 30])
    _ans, info = e2e._probe_ladder_chat(
        _DemandRepo(arena), QUESTION, topk=3, ngen=4, max_trips=1)
    assert info["demand_fired"] is True
    first_mount_set = arena.served[0]
    assert set(info["demand_fetched"]).isdisjoint(set(first_mount_set))


def test_an_ungrounded_demand_trip_serves_the_original_answer(
    demand_harness, monkeypatch,
):
    """Step 3 of the order: serve the trip IFF it grounds, else the original
    — and say which one went out."""
    monkeypatch.setenv("GRM_DEMAND_NGH", "1")
    _pin_rows(monkeypatch, [_rows(0.1), _rows(0.05)])
    arena = _DemandArena(
        ntok=[20, 25, 40, 30],
        answers={(3,): "original-answer", (0,): "trip-answer",
                 (1,): "trip-answer", (2,): "trip-answer"},
        grounded={(0,): False, (1,): False, (2,): False},
    )
    ans, info = e2e._probe_ladder_chat(
        _DemandRepo(arena), QUESTION, topk=3, ngen=4, max_trips=1)
    assert info["demand_trip_taken"] is True
    assert info["demand_trip_grounded"] is False
    assert info["demand_served"] == "original"
    assert ans == "original-answer"


def test_a_refire_on_the_demand_trip_is_recorded_and_not_acted_on(
    demand_harness, monkeypatch,
):
    """NEVER LOOP (cap 1, registered). A second fire is a receipt field."""
    monkeypatch.setenv("GRM_DEMAND_NGH", "1")
    # Both the original attempt and the demand trip fire.
    _pin_rows(monkeypatch, [_rows(0.1), _rows(0.05)])
    arena = _DemandArena(ntok=[20, 25, 40, 30])
    _ans, info = e2e._probe_ladder_chat(
        _DemandRepo(arena), QUESTION, topk=3, ngen=4, max_trips=1)
    assert info["demand_refired"] is True
    assert info["demand_refire_acted_on"] is False
    # Exactly TWO generations ever: the original and the ONE trip. A third
    # would mean the refire was acted on.
    assert len(arena.served) == 2


def test_an_unsupported_arena_refuses_loudly_instead_of_serving_silently(
    demand_harness, monkeypatch,
):
    monkeypatch.setenv("GRM_DEMAND_NGH", "1")
    arena = _DemandArena(ntok=[20, 25, 40, 30])
    # Strip the full-attention layers: a non-GPT-OSS arena.
    arena.m = SimpleNamespace(layers=[
        SimpleNamespace(self_attn=SimpleNamespace(
            live_shift=None, layer_type="sliding_attention",
            attention_mode="standard", layer_idx=0)),
    ])
    _ans, info = e2e._probe_ladder_chat(
        _DemandRepo(arena), QUESTION, topk=3, ngen=4, max_trips=1)
    assert info["demand_supported"] is False
    assert info["demand_unsupported_reason"]
    assert info["demand_fired"] is False
    # It served normally; it just did not pretend to have detected anything.
    assert len(arena.served) == 1


# ---------------------------------------------------------------------------
# The arena hook: ArenaCache.step()
#
# The lived probes go through _probe_ladder_chat, but step() is the OTHER
# production serving path and the order names it as the hook site, so the
# same four properties are pinned there too.
# ---------------------------------------------------------------------------


def _step_arena(*, ntok, width=96, answers=None, grounded=None):
    """Stubbed CPU arena for ``ArenaCache.step``, GPT-OSS-shaped.

    Extends the harness in ``tests/test_grm_lsr_p2a_fit_honesty.py`` with the
    layer metadata ``grm_demand.arena_support`` reads and a per-mount-set
    answer script.
    """
    from core.graft_arena import ArenaCache

    arena = ArenaCache.__new__(ArenaCache)
    arena.m = SimpleNamespace(layers=[
        SimpleNamespace(self_attn=SimpleNamespace(
            live_shift=None, layer_type="full_attention",
            attention_mode="standard", layer_idx=0)),
        SimpleNamespace(self_attn=SimpleNamespace(
            live_shift=None, layer_type="sliding_attention",
            attention_mode="standard", layer_idx=1)),
    ])
    arena.live_shift = 9
    arena.n_sink = 4
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
    arena.recency_mounts = 0
    arena.grafts = [
        {"text": f"node-{i}", "ntok": int(n), "kind": "fact"}
        for i, n in enumerate(ntok)
    ]
    arena._s4_turn = 0
    arena.route = lambda text, *, exclude, limit: [
        i for i in range(len(ntok)) if i not in set(exclude)]
    arena._rare_tokens = lambda _text: set()
    arena._descent_expand = lambda picks, _kinds, qrare=None: list(picks)
    arena._resolve_revision_mounts = lambda picks: list(picks)
    arena._next_s4_turn = lambda: 1
    arena._bump_cuda_gqa_epoch = lambda: None
    arena._commit_s4_attempt = lambda *_a, **_k: None
    arena.decode = lambda ids: " ".join(str(i) for i in ids)
    answers = dict(answers or {})
    grounded = dict(grounded or {})

    def _grounding(_ans, picks, _q):
        key = tuple(sorted(int(v) for v in picks))
        if key in grounded:
            return bool(grounded[key]), list(picks)
        return bool(picks), list(picks)

    arena._grounding_attribution = _grounding
    served: list[list[int]] = []

    def attempt(_question, picks, _ngen, _deposit, _stops,
                defer_memory=False):
        key = tuple(sorted(int(v) for v in picks))
        arena.cur_mounts = list(picks)
        served.append(list(key))
        return answers.get(key, "answer"), {"mounts": [v + 1 for v in key]}

    arena._attempt = attempt
    return arena, served


def _pin_step_profile(monkeypatch):
    monkeypatch.setattr(
        "core.graft_arena.decisive_admission_profile",
        lambda _arena, _text, *, exclude, route_limit: {
            **_LIVED_PROFILE,
            "ranking": [i for i in _LIVED_PROFILE["ranking"]
                        if i not in set(exclude)],
            "rank_plan": [i for i in _LIVED_PROFILE["rank_plan"]
                          if i not in set(exclude)]
            or [i for i in _LIVED_PROFILE["ranking"]
                if i not in set(exclude)][:1],
        },
    )


def test_step_flag_off_serves_byte_identically_and_adds_no_demand_fields(
    monkeypatch,
):
    monkeypatch.delenv("GRM_DEMAND_NGH", raising=False)
    _pin_step_profile(monkeypatch)
    arena, served = _step_arena(ntok=[20, 25, 40, 30])
    ans, info = arena.step(QUESTION, ngen=4, deposit=False, max_trips=1)
    assert ans == "answer"
    assert not [key for key in info if key.startswith("demand_")]

    monkeypatch.setenv("GRM_DEMAND_NGH", "0")
    arena2, served2 = _step_arena(ntok=[20, 25, 40, 30])
    ans2, info2 = arena2.step(QUESTION, ngen=4, deposit=False, max_trips=1)
    assert ans2 == ans
    assert served2 == served
    assert not [key for key in info2 if key.startswith("demand_")]


def test_step_fire_takes_one_trip_and_serves_it_when_grounded(monkeypatch):
    monkeypatch.setenv("GRM_DEMAND_NGH", "1")
    _pin_step_profile(monkeypatch)
    _pin_rows(monkeypatch, [_rows(0.9, 0.1), _rows(0.9, 0.8)])
    arena, served = _step_arena(
        ntok=[20, 25, 40, 30],
        answers={(3,): "wrong", (0,): "RIGHT", (1,): "RIGHT", (2,): "RIGHT"},
    )
    ans, info = arena.step(QUESTION, ngen=4, deposit=False, max_trips=1)
    assert info["demand_fired"] is True
    assert info["demand_token_index"] == 1
    assert info["demand_served"] == "demand_trip"
    assert info["demand_trip_taken"] is True
    assert ans == "RIGHT"
    assert len(served) == 2


def test_step_ungrounded_demand_trip_serves_the_original(monkeypatch):
    monkeypatch.setenv("GRM_DEMAND_NGH", "1")
    _pin_step_profile(monkeypatch)
    _pin_rows(monkeypatch, [_rows(0.1), _rows(0.05)])
    arena, served = _step_arena(
        ntok=[20, 25, 40, 30],
        answers={(3,): "original-answer", (0,): "trip", (1,): "trip",
                 (2,): "trip"},
        grounded={(0,): False, (1,): False, (2,): False},
    )
    ans, info = arena.step(QUESTION, ngen=4, deposit=False, max_trips=1)
    assert info["demand_trip_grounded"] is False
    assert info["demand_served"] == "original"
    assert ans == "original-answer"


def test_step_refire_is_recorded_and_the_cap_holds_at_one_trip(monkeypatch):
    monkeypatch.setenv("GRM_DEMAND_NGH", "1")
    _pin_step_profile(monkeypatch)
    _pin_rows(monkeypatch, [_rows(0.1), _rows(0.05)])
    arena, served = _step_arena(ntok=[20, 25, 40, 30])
    _ans, info = arena.step(QUESTION, ngen=4, deposit=False, max_trips=1)
    assert info["demand_refired"] is True
    assert info["demand_refire_acted_on"] is False
    assert len(served) == 2


def test_step_unsupported_arena_refuses_loudly(monkeypatch):
    monkeypatch.setenv("GRM_DEMAND_NGH", "1")
    _pin_step_profile(monkeypatch)
    arena, served = _step_arena(ntok=[20, 25, 40, 30])
    arena.m = SimpleNamespace(layers=[
        SimpleNamespace(self_attn=SimpleNamespace(
            live_shift=None, layer_type="sliding_attention",
            attention_mode="standard", layer_idx=0)),
    ])
    _ans, info = arena.step(QUESTION, ngen=4, deposit=False, max_trips=1)
    assert info["demand_supported"] is False
    assert info["demand_unsupported_reason"]
    assert len(served) == 1


def test_step_no_fire_serves_the_original_untouched(monkeypatch):
    monkeypatch.setenv("GRM_DEMAND_NGH", "1")
    _pin_step_profile(monkeypatch)
    _pin_rows(monkeypatch, [_rows(0.9, 0.8)])
    arena, served = _step_arena(ntok=[20, 25, 40, 30])
    ans, info = arena.step(QUESTION, ngen=4, deposit=False, max_trips=1)
    assert ans == "answer"
    assert info["demand_fired"] is False
    assert info["demand_trip_taken"] is False
    assert len(served) == 1


def test_demand_fields_reach_the_route_receipt_on_a_served_turn(
    demand_harness, monkeypatch,
):
    """Part 2 item 5: every ``demand_*`` field persists via the receipt."""
    monkeypatch.setenv("GRM_DEMAND_NGH", "1")
    _pin_rows(monkeypatch, [_rows(0.1), _rows(0.9)])
    arena = _DemandArena(ntok=[20, 25, 40, 30])
    _ans, info = e2e._probe_ladder_chat(
        _DemandRepo(arena), QUESTION, topk=3, ngen=4, max_trips=1)
    projected = _route_receipt_generic_info(info)
    demand_keys = {k for k in info if k.startswith("demand_")}
    assert demand_keys
    assert demand_keys <= set(projected)
    for key in demand_keys:
        assert projected[key] == info[key] or projected[key] == list(
            info[key]) if isinstance(info[key], (list, tuple)) else True
