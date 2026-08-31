"""GRM-ADM2 default-on A-DEC policy and byte-exact escape regressions."""

from types import SimpleNamespace

import pytest

from core.grm_admission import (
    FROZEN_RULE_SHA256,
    MARGIN_THRESHOLD,
    adm_decisive_cli_argv,
    adm_decisive_enabled,
    decisive_admission_profile,
    env_adm_decisive_override,
    is_identifier_binding,
    policy_plan,
)
from core.graft_arena import ArenaCache


def test_adm_decisive_default_on_and_escape_tokens(monkeypatch):
    monkeypatch.delenv("GRM_ADM_DECISIVE", raising=False)
    assert env_adm_decisive_override() is None
    assert adm_decisive_enabled() is True
    for token in ("0", "false", "off", "no", "", "malformed"):
        monkeypatch.setenv("GRM_ADM_DECISIVE", token)
        assert env_adm_decisive_override() is False
        assert adm_decisive_enabled() is False


def test_adm_decisive_explicit_choice_precedes_environment(monkeypatch):
    monkeypatch.setenv("GRM_ADM_DECISIVE", "0")
    assert adm_decisive_enabled(True) is True
    monkeypatch.setenv("GRM_ADM_DECISIVE", "1")
    assert adm_decisive_enabled(False) is False
    assert adm_decisive_cli_argv(True) == ["--adm-decisive"]
    assert adm_decisive_cli_argv(False) == ["--no-adm-decisive"]


def test_frozen_identifier_binding_rule_covers_rare_and_lowercase_labels():
    assert is_identifier_binding(
        candidate_text="The crate code is QB-120.",
        ordered_identifier_tokens=["qb-120"],
        rare_identifier_tokens={"qb-120"},
    )
    assert not is_identifier_binding(
        candidate_text="The crate code is QB-121.",
        ordered_identifier_tokens=["qb-120"],
        rare_identifier_tokens={"qb-120"},
    )
    assert is_identifier_binding(
        candidate_text="The current Praxis dock value is Quartz-8-Jade.",
        ordered_identifier_tokens=["praxis", "dock"],
        rare_identifier_tokens=(),
    )
    assert not is_identifier_binding(
        candidate_text="Praxis and dock occur without an assertion.",
        ordered_identifier_tokens=["praxis", "dock"],
        rare_identifier_tokens=(),
    )


def test_frozen_policy_branch_precedence():
    assert policy_plan(
        ranking=[4, 5, 6], identified_candidates=[], route_margin_1_2=1.0,
    ) == ([4, 5, 6], "ambiguous_zero_identifier_hits_k3")
    assert policy_plan(
        ranking=[4, 5, 6], identified_candidates=[4, 6],
        route_margin_1_2=0.0,
    ) == ([4, 6], "declared_synthesis_identified_set")
    assert policy_plan(
        ranking=[4, 5, 6], identified_candidates=[4, 9],
        route_margin_1_2=0.0,
    ) == ([4, 9], "declared_synthesis_identified_set")
    assert policy_plan(
        ranking=[4, 5, 6], identified_candidates=[4], route_margin_1_2=0.0,
    ) == ([4], "exactly_one_identifier_decisive_rank1")
    assert policy_plan(
        ranking=[4, 5, 6], identified_candidates=[6],
        route_margin_1_2=MARGIN_THRESHOLD + 1.0e-6,
    ) == ([4], "fit_margin_decisive_rank1")
    assert policy_plan(
        ranking=[4, 5, 6], identified_candidates=[6],
        route_margin_1_2=MARGIN_THRESHOLD,
    ) == ([4, 5, 6], "one_off_rank_identifier_insurance_k3")


class _ProfileArena:
    def __init__(self):
        self.grafts = [
            {"text": "The current Praxis dock value is Quartz-8-Jade."},
            {"text": "The current Solace key value is Raven-9-Ivory."},
            {"text": "unrelated filler"},
        ]
        self.last_route_backend = "python"

    def _route_cand_base(self):
        return [0, 1, 2]

    def _probe_key(self, _question):
        return [1.0]

    def route(self, _question, *, exclude, limit, probe_key):
        assert exclude == set()
        assert limit == 3
        assert probe_key == [1.0]
        return [0, 1, 2]

    def _vector_route_scores(self, _probe_key, _eligible):
        return {0: 1.0, 1: 0.8, 2: 0.1}

    def _length_debias_scores(self, base, _eligible):
        return base

    def _normalize_scores(self, base):
        return base

    def _query_lex_tokens(self, _question):
        return {"praxis", "dock"}

    def _lex_bonus(self, qlex, graft):
        assert qlex == {"praxis", "dock"}
        return 0.0

    def _rare_tokens(self, _question):
        return set()


def test_decisive_profile_reuses_route_laws_and_frozen_receipt():
    profile = decisive_admission_profile(
        _ProfileArena(), "What is the current Praxis dock value?", exclude=set())
    assert profile["ranking"] == [0, 1, 2]
    assert profile["identified_candidates"] == [0]
    assert profile["rank_plan"] == [0]
    assert profile["policy_branch"] == "exactly_one_identifier_decisive_rank1"
    assert profile["route_margin_1_2"] == pytest.approx(0.2)
    assert profile["rule_sha256"] == FROZEN_RULE_SHA256


class _StubModel:
    def extend_rope(self, _length):
        pass


class _StubArena(ArenaCache):
    def _harvest(self, _ids):
        return ()


def _make_constructor_probe(**kwargs):
    return _StubArena(
        _StubModel(), encode=lambda _text: [1], decode=lambda _ids: "", **kwargs)


def test_arena_constructor_resolves_default_escape_and_explicit_pin(monkeypatch):
    monkeypatch.delenv("GRM_ADM_DECISIVE", raising=False)
    assert _make_constructor_probe().decisive_admission is True
    monkeypatch.setenv("GRM_ADM_DECISIVE", "0")
    assert _make_constructor_probe().decisive_admission is False
    assert _make_constructor_probe(decisive_admission=True).decisive_admission is True
    monkeypatch.setenv("GRM_ADM_DECISIVE", "1")
    assert _make_constructor_probe(decisive_admission=False).decisive_admission is False


def _step_probe(*, decisive: bool):
    arena = ArenaCache.__new__(ArenaCache)
    arena.m = SimpleNamespace(
        layers=[SimpleNamespace(self_attn=SimpleNamespace(live_shift=None))])
    arena.live_shift = 9
    arena.stop_sequences = ()
    arena.ephemeral = False
    arena.live_segs = []
    arena.topk = 3
    arena.decisive_admission = bool(decisive)
    arena.caches = None
    arena.pos = 0
    arena.cur_mounts = []
    arena.cur_mount_n = 0
    arena.grafts = [
        {"text": "target", "ntok": 1, "kind": "fact"},
        {"text": "sibling-a", "ntok": 1, "kind": "fact"},
        {"text": "sibling-b", "ntok": 1, "kind": "fact"},
    ]
    arena.width = 16
    arena._s4_turn = 0
    route_calls = []
    arena.route = lambda text, *, exclude, limit: (
        route_calls.append((text, set(exclude), limit)) or [0, 2, 1])
    arena._rare_tokens = lambda _text: set()
    arena._descent_expand = lambda picks, _kinds, qrare=None: list(picks)
    arena._resolve_revision_mounts = lambda picks: list(picks)
    arena._next_s4_turn = lambda: 1
    arena._bump_cuda_gqa_epoch = lambda: None
    arena._commit_s4_attempt = lambda *_args, **_kwargs: None
    arena._grounding_attribution = lambda _answer, picks, _question: (
        True, list(picks))

    def attempt(_question, picks, _ngen, _deposit, _stops):
        arena.cur_mounts = list(picks)
        return "answer", {"mounts": [value + 1 for value in picks]}

    arena._attempt = attempt
    return arena, route_calls


def test_step_default_applies_plan_but_escape_retains_legacy_route(monkeypatch):
    profile = {
        "ranking": [0, 2, 1],
        "identified_candidates": [0],
        "identifier_hit_count": 1,
        "route_margin_1_2": 0.25,
        "rank_plan": [0],
        "policy_branch": "exactly_one_identifier_decisive_rank1",
    }
    monkeypatch.setattr(
        "core.graft_arena.decisive_admission_profile",
        lambda _arena, _text, *, exclude, route_limit: dict(profile),
    )

    default, default_route_calls = _step_probe(decisive=True)
    answer, info = default.step("Praxis dock?", ngen=1, deposit=False)
    assert answer == "answer"
    assert default.cur_mounts == [0]
    assert default_route_calls == []
    assert info["admission_policy"] == "A-DEC"
    assert info["admission_rank_plan"] == [0]

    legacy, legacy_route_calls = _step_probe(decisive=False)
    answer, legacy_info = legacy.step("Praxis dock?", ngen=1, deposit=False)
    assert answer == "answer"
    assert legacy.cur_mounts == [0, 1, 2]
    assert legacy_route_calls == [("Praxis dock?", set(), 3)]
    assert "admission_policy" not in legacy_info
