"""C4 additive experiment adapter; importing this module changes nothing.

Prior art: house GraftRepository LSR-P2C split/descent (2026), WC1 harness
(2026), RS3 capture/near-live seating (2026), and EB1 session save/restore
(2026), verified in this checkout. Borrow their implementations unchanged.
C4 contributes independent experiment controls and token-seat observations;
no new chunking, routing, scoring, or persistence algorithm is claimed.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from unittest.mock import patch

CHUNK_ENV = "GRM_C4_DEPOSIT_CHUNK_TOKENS"
GEOMETRY_BAND = 96


def chunk_setting(environ=None):
    """Absent/0/off is OFF; explicit experimental values are 64 or 96."""
    value = (os.environ if environ is None else environ).get(CHUNK_ENV, "0")
    if str(value).strip().lower() in ("", "0", "off"):
        return None
    if str(value).strip() not in ("64", "96"):
        raise ValueError(f"{CHUNK_ENV}: expected OFF, 64, or 96")
    return int(value)


@contextmanager
def deposit_chunking(chunk=None, events=None):
    """Explicit-budget fit repair is untouched; only default deposit calls move.

    Prior art: LSR-P2C GraftRepository._guard_deposit_width (2026).
    Reuse section/sentence packing and exact parent spans; C4 selects its
    existing budget argument independently of admission width. No clamp to
    arena width: (96,64) MUST remain a distinct deposit treatment, with later
    fit-time repair observed separately. OFF preserves callable identity.
    """
    from core.graft_repository import GraftRepository
    if chunk is None:
        chunk = chunk_setting()
    if chunk is None:
        yield
        return
    if chunk not in (64, 96):
        raise ValueError("chunk must be 64 or 96")
    original = GraftRepository._guard_deposit_width

    def guard(repo, idx, *, boundary="section", budget=None):
        parent_text = repo.arena.grafts[int(idx)]["text"]
        explicit = budget is not None
        result = original(repo, idx, boundary=boundary,
                          budget=budget if explicit else chunk)
        assert repo.arena.grafts[int(idx)]["text"] == parent_text
        if events is not None:
            events.append({"event": "guard", "idx": int(idx),
                           "phase": "explicit_budget_repair" if explicit else "deposit",
                           "budget": int(budget) if explicit else chunk,
                           "parent_preserved": True, "split": result})
        return result

    with patch.object(GraftRepository, "_guard_deposit_width", guard):
        yield


def residency(arena):
    """Observed current mounts, not fitted proposal count or final manifest.

    Prior art: ArenaCache.cur_mounts/cur_mount_n (house, 2026); C4 sums
    actual node ntok at observation time and checks the arena's own counter.
    Physical live seats are counted from cache shape, separately from pos
    (a RoPE running position, which need not equal retained token count).
    """
    ids = list(map(int, arena.cur_mounts))
    tokens = sum(int(arena.grafts[i]["ntok"]) for i in ids)
    assert tokens == int(arena.cur_mount_n), (tokens, arena.cur_mount_n)
    assert tokens <= int(arena.width)
    cache_tokens = None
    if arena.caches:
        _, dim = arena.PAYLOAD[0]
        cache_tokens = int(arena.caches[0][0].shape[dim])
    live = None if cache_tokens is None else cache_tokens - int(arena.n_sink) - tokens
    assert live is None or live >= 0
    return {"mounted_ids": ids, "mounted_token_seats": tokens,
            "sink_token_seats": int(arena.n_sink), "live_token_seats": live,
            "physical_cache_token_seats": cache_tokens,
            "arena_width": int(arena.width), "live_shift": int(arena.live_shift)}


@contextmanager
def fixed_geometry(width, events):
    """Pin physical RoPE band at 96; vary admission capacity after arena init.

    Prior art: RS3 (house, 2026) separates live_shift from physical packed KV.
    C4 keeps its near-live placement formula and harvest pin unchanged, and
    holds live_shift fixed while changing width before repository native
    configuration. This is a harness-only intervention, not a kernel edit.
    """
    from core.graft_arena import ArenaCache
    from scripts.grm_wc1_common import split_census
    if width not in (64, 96):
        raise ValueError("width must be 64 or 96")
    arenas = []
    original_init = ArenaCache.__init__
    original_capture = ArenaCache._capture_geometry
    original_seat = ArenaCache._rs3_seat_plan
    original_attempt = ArenaCache._attempt
    original_step = ArenaCache.step

    def init(arena, *args, **kwargs):
        kwargs["arena_width"] = GEOMETRY_BAND
        kwargs["ephemeral"] = True
        original_init(arena, *args, **kwargs)
        arena.width = width
        arena._rs3_capture_pin_explicit = "live"
        arena._rs3_seat_explicit = True
        arenas.append(arena)
        events.append({"event": "geometry", "arena_width": width,
                       "n_sink": int(arena.n_sink), "live_shift": int(arena.live_shift),
                       "capture_pin": "live", "seat_near_live": True,
                       "live_turns": int(arena.live_turns), "ephemeral": bool(arena.ephemeral)})
        assert arena.live_shift == arena.n_sink + GEOMETRY_BAND
        assert arena.live_turns == 2 and arena.ephemeral

    @contextmanager
    def capture(arena, *args, **kwargs):
        with original_capture(arena, *args, **kwargs) as info:
            assert arena.live_shift == arena.n_sink + GEOMETRY_BAND
            assert info['capture_pin'] == 'live'
            assert info['capture_shift_observed'] == arena.n_sink + GEOMETRY_BAND
            events.append({"event": "capture", "observed": dict(info),
                           "c4_derivation": "fixed n_sink+96; inherited arena_width derivation label is overridden"})
            yield info

    def seat(arena, *args, **kwargs):
        order, info = original_seat(arena, *args, **kwargs)
        assert info["live_shift"] == arena.n_sink + GEOMETRY_BAND
        if order:
            assert info["plan_head_last_position"] == arena.n_sink + GEOMETRY_BAND - 1
        events.append({"event": "seat", "observed": info})
        return order, info

    def attempt(arena, *args, **kwargs):
        result = original_attempt(arena, *args, **kwargs)
        events.append({"event": "attempt_residency", **residency(arena)})
        return result

    def step(arena, *args, **kwargs):
        result = original_step(arena, *args, **kwargs)
        events.append({"event": "turn", "info": result[1], **residency(arena)})
        return result

    with patch.object(ArenaCache, "__init__", init), \
         patch.object(ArenaCache, "_capture_geometry", capture), \
         patch.object(ArenaCache, "_rs3_seat_plan", seat), \
         patch.object(ArenaCache, "_attempt", attempt), \
         patch.object(ArenaCache, "step", step):
        try:
            yield
        finally:
            for arena in arenas:
                events.append({"event": "split_census", **split_census(arena.grafts)})
