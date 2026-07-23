"""Unit tests for GRM3P-FIX-LADDER probe-path planning (no GPU).

Covers flag default-off, precise-first, point-lookup clean-room, and the
one-retry attempt ladder. Does not reimplement arena scoring.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

from scripts.grm_probe_ladder import (
    build_probe_ladder_attempts,
    env_probe_ladder_enabled,
    identifier_tokens_from_parts,
    probe_ladder_enabled,
    rank1_covers_identifiers,
)


def test_probe_ladder_default_off(monkeypatch):
    monkeypatch.delenv("GRM_PROBE_LADDER", raising=False)
    assert env_probe_ladder_enabled() is False
    assert probe_ladder_enabled(SimpleNamespace(probe_ladder=False)) is False
    assert probe_ladder_enabled(None) is False


def test_probe_ladder_cli_and_env(monkeypatch):
    monkeypatch.delenv("GRM_PROBE_LADDER", raising=False)
    assert probe_ladder_enabled(SimpleNamespace(probe_ladder=True)) is True
    monkeypatch.setenv("GRM_PROBE_LADDER", "1")
    assert probe_ladder_enabled(SimpleNamespace(probe_ladder=False)) is True
    monkeypatch.setenv("GRM_PROBE_LADDER", "off")
    assert probe_ladder_enabled(SimpleNamespace(probe_ladder=False)) is False


def test_identifier_tokens_prefer_rare_over_qlex():
    rare = {"auric-4-alpha"}
    qlex = {"orion", "pin", "auric-4-alpha"}
    assert identifier_tokens_from_parts(rare, qlex) == rare
    assert identifier_tokens_from_parts(set(), qlex) == qlex
    assert identifier_tokens_from_parts((), ()) == set()


def test_rank1_covers_identifiers_rare_and_text():
    ids = {"orion", "pin"}
    assert rank1_covers_identifiers(ids, {"orion", "pin"}, ()) is True
    assert rank1_covers_identifiers(
        ids, set(), {"orion", "pin", "auric-4-alpha"}) is True
    assert rank1_covers_identifiers(
        ids, {"vortex-3-sierra"}, {"cypher", "bridge"}) is False
    assert rank1_covers_identifiers(set(), {"orion"}, {"orion"}) is False


def test_point_lookup_precise_first_then_multi_clean():
    ranking = [0, 2, 1]
    attempts = build_probe_ladder_attempts(
        ranking=ranking,
        topk=3,
        precise=[0],
        point_lookup=True,
        max_trips=1,
    )
    assert attempts == [([0], True), ([0, 2, 1], True)]


def test_point_lookup_no_precise_multi_only_clean():
    ranking = [5, 6, 7, 4]
    attempts = build_probe_ladder_attempts(
        ranking=ranking,
        topk=3,
        precise=None,
        point_lookup=True,
        max_trips=1,
    )
    # Trip-0 multi top-k clean; retry is next slice (also clean).
    assert attempts == [([5, 6, 7], True), ([4], True)]


def test_topical_non_point_keeps_live_on_primary():
    ranking = [1, 2, 3]
    attempts = build_probe_ladder_attempts(
        ranking=ranking,
        topk=2,
        precise=None,
        point_lookup=False,
        max_trips=1,
    )
    # Non-point multi: first attempt not clean; next ranking slice as retry.
    assert attempts == [([1, 2], False), ([3], False)]


def test_topical_same_slice_clean_room_retry():
    ranking = [9]
    attempts = build_probe_ladder_attempts(
        ranking=ranking,
        topk=3,
        precise=None,
        point_lookup=False,
        max_trips=1,
    )
    assert attempts == [([9], False), ([9], True)]


def test_max_trips_zero_is_single_attempt():
    attempts = build_probe_ladder_attempts(
        ranking=[0, 1, 2],
        topk=3,
        precise=[0],
        point_lookup=True,
        max_trips=0,
    )
    assert attempts == [([0], True)]


def test_t5_shaped_precise_covers_orion_source():
    """t5 recipe: rank-1 is the true orion source → precise clean alone."""
    ids = identifier_tokens_from_parts(
        rare=(),
        qlex={"orion", "pin"},
    )
    assert ids == {"orion", "pin"}
    assert rank1_covers_identifiers(
        ids,
        rank1_rare={"auric-4-alpha"},
        rank1_text_tokens={"orion", "pin", "auric-4-alpha", "current", "value"},
    )
    attempts = build_probe_ladder_attempts(
        ranking=[0, 2, 1],
        topk=3,
        precise=[0],
        point_lookup=True,
        max_trips=1,
    )
    assert attempts[0] == ([0], True)
    assert attempts[0][1] is True  # live/recency excluded


def test_t13_shaped_rank1_distractor_skips_precise():
    """t13 recipe: rank-1 is a cypher/distractor → multi-mount clean."""
    ids = {"orion", "pin"}
    covers = rank1_covers_identifiers(
        ids,
        rank1_rare={"vortex-3-sierra"},
        rank1_text_tokens={"cypher", "bridge", "vortex-3-sierra"},
    )
    assert covers is False
    attempts = build_probe_ladder_attempts(
        ranking=[5, 6, 7],
        topk=3,
        precise=None,
        point_lookup=True,
        max_trips=1,
    )
    assert attempts[0] == ([5, 6, 7], True)
