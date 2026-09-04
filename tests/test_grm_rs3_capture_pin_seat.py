"""GRM-RS3 — CPU tests for the capture pin and the seat-near-live lever.

ORDER: ``orders/GRM_RS3_CAPTURE_PIN_SEAT_NEAR_LIVE.md`` (gate G1).

Everything here is CPU-only: nothing loads a model, nothing takes the GPU
lease, nothing writes into ``artifacts/``.  The model is a stub carrying only
what the two seams actually touch — a layer list whose ``self_attn`` objects
own a ``live_shift`` attribute, and the rope tables.

WHAT IS PINNED, and why each one is load-bearing.

  * FLAG RESOLVERS FAIL CLOSED.  Both levers change what the stack SERVES when
    they are on (a different stored payload; a different mount position), so an
    unknown or mistyped token must land on OFF.  ``GRM_CAPTURE_PIN=banana`` and
    ``GRM_SEAT_NEAR_LIVE=banana`` are both required to resolve OFF, and so is
    ``GRM_CAPTURE_PIN=1`` — a true-ish token is NOT a pin name, and guessing
    which pin the operator meant is exactly the silent change the fail-closed
    rule exists to prevent.  An EXPLICIT bad pin from a caller raises instead,
    because that is a bug in the caller rather than an ambient typo.

  * FLAGS OFF ARE BYTE-IDENTICAL.  ``_capture_geometry(off)`` must not touch
    ``live_shift`` on any layer — not even to write back the value it found —
    so the legacy harvest runs on the legacy state operand for operand.
    ``_rs3_seat_plan(off)`` must return the picks unchanged with a zero delta,
    and ``_rs3_rotate_injection`` under that plan must return the caller's own
    object, not a copy: the legacy branch's injection block reaches
    ``_set_injection_host`` untouched.

  * PIN GEOMETRY IS DERIVED, NEVER TYPED.  ``mount`` is ``n_sink`` and
    ``live`` is ``live_shift``, read off the arena.  The test builds an arena
    with non-default numbers so a hard-coded 19/115 would fail.

  * THE PIN IS SCOPED AND RESTORES.  The pin holds for the harvest and is
    unwound afterwards, per layer, INCLUDING when the body raises — a pin that
    leaked into the serving path would silently re-position every later turn.

  * THE RELOCATABLE-KEYS INVARIANT (``docs/GRM_Methodology.md`` §4).  The same
    graft seated at both ends of the band carries the SAME K/V payload; only
    the RoPE-applied keys differ, and they differ by a ROTATION — same
    per-row norms.  This is the invariant Part 2 rests on, and it is checked
    directly rather than inferred.

  * THE CONSTANT-DELTA LAW.  A band relocation moves every row by the SAME
    amount.  ``_rope_tensor(x, delta)`` does NOT do that — it rotates row j by
    ``delta + j``, which composes with the layer's own ``n_sink + j`` to
    ``n_sink + delta + 2j``: a SHEAR that de-phases the block's rows against
    each other and stops the keys being the same keys.  This test pins the
    composition ``rotate_const(delta) then rotate_at(n_sink) ==
    rotate_at(n_sink + delta)`` and pins that the span-style rotation FAILS
    it, so the distinction cannot be lost to a later refactor.

  * THE SEAT ORDER AND ITS RECEIPT.  ON puts the plan head LAST in the block
    so its final token is at ``live_shift - 1``; the receipt's
    ``seat_offset_plan_head`` and ``seat_order`` say so, and a block too wide
    for the band FAILS CLOSED to the legacy seating rather than overrunning
    the sink.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import grm_frame  # noqa: E402
from core.grm_frame import (  # noqa: E402
    CAPTURE_PIN_ENV, CAPTURE_PIN_LIVE, CAPTURE_PIN_MOUNT, CAPTURE_PIN_OFF,
    CAPTURE_PINS, SEAT_NEAR_LIVE_ENV, capture_pin_mode, env_capture_pin,
    env_seat_near_live, rs3_receipt, seat_near_live_enabled,
)
from core.graft_arena import GQAArenaCache  # noqa: E402


# ======================================================================
# The stub arena — only what the two seams touch
# ======================================================================
HEAD_DIM = 8
TABLE_POSITIONS = 512

#: Deliberately NOT the lived 19 / 96 / 115: a geometry hard-coded from the
#: RS2 receipts would pass against those numbers and fail here.
STUB_N_SINK = 7
STUB_WIDTH = 40
STUB_LIVE_SHIFT = STUB_N_SINK + STUB_WIDTH      # 47


def _rope_tables():
    inv = 1.0 / (10000 ** (np.arange(0, HEAD_DIM, 2) / HEAD_DIM))
    angles = np.arange(TABLE_POSITIONS)[:, None] * inv[None, :]
    cos = np.concatenate([np.cos(angles), np.cos(angles)], -1)
    sin = np.concatenate([np.sin(angles), np.sin(angles)], -1)
    return cos.astype(np.float32), sin.astype(np.float32)


class _HostTensor:
    """The narrow slice of the tensor surface these two seams use."""

    def __init__(self, array):
        self._array = np.asarray(array)

    def float(self):
        return _HostTensor(self._array.astype(np.float32))

    def numpy(self):
        return self._array

    def astype(self, _dtype):
        return self


class _StubAttention:
    def __init__(self):
        self.live_shift = None


class _StubLayer:
    def __init__(self):
        self.self_attn = _StubAttention()


class _StubModel:
    def __init__(self, n_layers=3):
        cos, sin = _rope_tables()
        self.rope_cos = _HostTensor(cos)
        self.rope_sin = _HostTensor(sin)
        self.layers = [_StubLayer() for _ in range(n_layers)]


def _arena(cls=GQAArenaCache, n_layers=3):
    """A bare arena carrying only the geometry the two seams read."""
    arena = cls.__new__(cls)
    arena.m = _StubModel(n_layers)
    arena.n_sink = STUB_N_SINK
    arena.width = STUB_WIDTH
    arena.live_shift = STUB_LIVE_SHIFT
    arena.dt = np.float32
    arena.grafts = []
    return arena


def _rope_at(arena, array, pos0):
    """Reference RoPE at an absolute seat range, in host numpy.

    This is ``F.apply_rotary``'s own expression (``x*cos + rotate_half(x)*sin``
    with the table sliced at ``[pos0, pos0 + L)``), which is what
    ``ArenaCache._rope_tensor`` computes and what the attention layer applies
    to an injected block.  Written out here so the test does not depend on the
    GPU engine being present.
    """
    length = array.shape[-2]
    cos = arena.m.rope_cos.numpy()[pos0:pos0 + length]
    sin = arena.m.rope_sin.numpy()[pos0:pos0 + length]
    half = array.shape[-1] // 2
    rotated = np.concatenate([-array[..., half:], array[..., :half]], axis=-1)
    return array * cos + rotated * sin


# ======================================================================
# Flag resolvers: fail closed
# ======================================================================
def test_capture_pin_absent_resolves_off():
    assert env_capture_pin({}) == CAPTURE_PIN_OFF
    assert capture_pin_mode(None, {}) == CAPTURE_PIN_OFF


@pytest.mark.parametrize("token", ["mount", "MOUNT", "  Mount  "])
def test_capture_pin_mount_token_selects_mount(token):
    assert env_capture_pin({CAPTURE_PIN_ENV: token}) == CAPTURE_PIN_MOUNT


@pytest.mark.parametrize("token", ["live", "LIVE", " live "])
def test_capture_pin_live_token_selects_live(token):
    assert env_capture_pin({CAPTURE_PIN_ENV: token}) == CAPTURE_PIN_LIVE


@pytest.mark.parametrize(
    "token",
    ["", "off", "OFF", "banana", "1", "true", "yes", "on", "0", "moun",
     "livee", "mount,live", "none"],
)
def test_capture_pin_unknown_and_false_tokens_fail_closed_to_off(token):
    """A true-ish token is NOT a pin name.

    ``GRM_CAPTURE_PIN=1`` is the dangerous case: it reads as "on" to a human,
    but the flag has TWO on-states with different geometries and guessing one
    would silently change every stored graft.  It fails closed like any other
    unknown token.
    """
    assert env_capture_pin({CAPTURE_PIN_ENV: token}) == CAPTURE_PIN_OFF


def test_explicit_capture_pin_outranks_env():
    env = {CAPTURE_PIN_ENV: "live"}
    assert capture_pin_mode(CAPTURE_PIN_MOUNT, env) == CAPTURE_PIN_MOUNT
    assert capture_pin_mode(CAPTURE_PIN_OFF, env) == CAPTURE_PIN_OFF
    assert capture_pin_mode(None, env) == CAPTURE_PIN_LIVE


def test_explicit_unknown_capture_pin_raises_rather_than_failing_closed():
    """An explicit bad pin is a CALLER bug, not an ambient typo.

    Serving OFF for it would hide the bug behind a legacy path that looks like
    a legitimate arm, so this one direction is strict.
    """
    with pytest.raises(ValueError) as excinfo:
        capture_pin_mode("banana", {})
    assert "banana" in str(excinfo.value)
    for name in CAPTURE_PINS:
        assert name in str(excinfo.value)


def test_seat_near_live_absent_resolves_off():
    assert env_seat_near_live({}) is False
    assert seat_near_live_enabled(None, {}) is False


@pytest.mark.parametrize("token", ["1", "true", "TRUE", "yes", "on", " on "])
def test_seat_near_live_true_tokens(token):
    assert env_seat_near_live({SEAT_NEAR_LIVE_ENV: token}) is True


@pytest.mark.parametrize("token", ["", "0", "false", "no", "off", "banana",
                                   "2", "onn"])
def test_seat_near_live_unknown_and_false_tokens_fail_closed(token):
    assert env_seat_near_live({SEAT_NEAR_LIVE_ENV: token}) is False


def test_explicit_seat_near_live_outranks_env():
    env = {SEAT_NEAR_LIVE_ENV: "1"}
    assert seat_near_live_enabled(False, env) is False
    assert seat_near_live_enabled(True, {}) is True
    assert seat_near_live_enabled(None, env) is True


def test_rs3_receipt_carries_both_levers_and_the_raw_env():
    env = {CAPTURE_PIN_ENV: "banana", SEAT_NEAR_LIVE_ENV: "1"}
    receipt = rs3_receipt(env_capture_pin(env), env_seat_near_live(env), env)
    # The RESOLVED value is off (fail-closed) but the RAW token is still on
    # the receipt, so an operator typo is visible rather than swallowed.
    assert receipt["capture_pin"] == CAPTURE_PIN_OFF
    assert receipt["capture_pin_env"] == "banana"
    assert receipt["seat_near_live"] is True
    assert receipt["seat_near_live_env"] == "1"


def test_registered_env_names_are_the_orders_names():
    assert grm_frame.CAPTURE_PIN_ENV == "GRM_CAPTURE_PIN"
    assert grm_frame.SEAT_NEAR_LIVE_ENV == "GRM_SEAT_NEAR_LIVE"


# ======================================================================
# Part 1 — the capture pin
# ======================================================================
def test_capture_shift_geometry_is_derived_from_the_arena():
    arena = _arena()
    assert arena.capture_shift_for(CAPTURE_PIN_MOUNT) == STUB_N_SINK
    assert arena.capture_shift_for(CAPTURE_PIN_LIVE) == STUB_LIVE_SHIFT
    assert arena.capture_shift_for(CAPTURE_PIN_OFF) is None
    # Derived, not typed: move the arena and the geometry moves with it.
    arena.n_sink = 3
    arena.width = 11
    arena.live_shift = arena.n_sink + arena.width
    assert arena.capture_shift_for(CAPTURE_PIN_MOUNT) == 3
    assert arena.capture_shift_for(CAPTURE_PIN_LIVE) == 14


def test_capture_pin_off_does_not_touch_live_shift():
    """THE OFF-PATH BYTE-IDENTITY RECEIPT for Part 1.

    OFF is not "pin to the current value" — it must not write ``live_shift``
    at all, or a legacy harvest would run against state this seam invented.
    """
    arena = _arena()
    sentinel = object()
    for layer in arena.m.layers:
        layer.self_attn.live_shift = sentinel
    with arena._capture_geometry(None) as receipt:
        assert all(layer.self_attn.live_shift is sentinel
                   for layer in arena.m.layers)
    assert all(layer.self_attn.live_shift is sentinel
               for layer in arena.m.layers)
    assert receipt["capture_pin"] == CAPTURE_PIN_OFF
    assert receipt["capture_shift"] is None
    # The OFF branch REPORTS live_shift; it must never coerce or raise on it.
    # The sentinel is deliberately un-int()-able for exactly this reason.
    assert isinstance(receipt["capture_shift_observed"], str)


@pytest.mark.parametrize(
    "pin,expected",
    [(CAPTURE_PIN_MOUNT, STUB_N_SINK), (CAPTURE_PIN_LIVE, STUB_LIVE_SHIFT)],
)
def test_capture_pin_sets_every_layer_and_restores_every_layer(pin, expected):
    arena = _arena(n_layers=5)
    before = [3, None, 11, None, 115]
    for layer, value in zip(arena.m.layers, before):
        layer.self_attn.live_shift = value
    with arena._capture_geometry(pin) as receipt:
        assert [layer.self_attn.live_shift for layer in arena.m.layers] == \
            [expected] * len(arena.m.layers)
        assert receipt["capture_shift"] == expected
        assert receipt["capture_shift_observed"] == expected
    # Restored PER LAYER to exactly what was there, including the Nones.
    assert [layer.self_attn.live_shift for layer in arena.m.layers] == before


def test_capture_pin_restores_when_the_harvest_raises():
    """A pin that leaked would silently re-position every later turn."""
    arena = _arena()
    for layer in arena.m.layers:
        layer.self_attn.live_shift = 115
    with pytest.raises(RuntimeError):
        with arena._capture_geometry(CAPTURE_PIN_MOUNT):
            raise RuntimeError("harvest blew up")
    assert [layer.self_attn.live_shift for layer in arena.m.layers] == \
        [115] * len(arena.m.layers)


def test_capture_receipt_carries_the_geometry_it_used():
    arena = _arena()
    with arena._capture_geometry(CAPTURE_PIN_LIVE) as receipt:
        pass
    assert receipt["capture_pin"] == CAPTURE_PIN_LIVE
    assert receipt["capture_shift"] == STUB_LIVE_SHIFT
    assert receipt["n_sink"] == STUB_N_SINK
    assert receipt["arena_width"] == STUB_WIDTH
    assert receipt["live_shift"] == STUB_LIVE_SHIFT
    assert "arena.live_shift" in receipt["capture_shift_derived_from"]


def test_capture_pin_off_receipt_records_the_unpinned_geometry():
    """An OFF receipt still says which geometry the harvest actually got.

    This is the field that makes the RS2 finding legible on a legacy run: the
    same ``deposit`` call reports None on a virgin arena and 47 after a served
    turn, which is the accident the pin exists to remove.
    """
    arena = _arena()
    for layer in arena.m.layers:
        layer.self_attn.live_shift = None
    with arena._capture_geometry(None) as fresh:
        pass
    assert fresh["capture_shift_observed"] is None
    for layer in arena.m.layers:
        layer.self_attn.live_shift = STUB_LIVE_SHIFT
    with arena._capture_geometry(None) as served:
        pass
    assert served["capture_shift_observed"] == STUB_LIVE_SHIFT


def test_capture_pin_resolves_through_the_env_when_not_passed(monkeypatch):
    arena = _arena()
    monkeypatch.setenv(CAPTURE_PIN_ENV, "mount")
    assert arena.capture_shift_for() == STUB_N_SINK
    monkeypatch.setenv(CAPTURE_PIN_ENV, "live")
    assert arena.capture_shift_for() == STUB_LIVE_SHIFT
    monkeypatch.setenv(CAPTURE_PIN_ENV, "banana")
    assert arena.capture_shift_for() is None
    monkeypatch.delenv(CAPTURE_PIN_ENV)
    assert arena.capture_shift_for() is None


def test_both_dialects_share_the_capture_seam():
    """``deposit`` is not overridden by the GQA dialect, so one pin covers
    both.  Pinned here so a future dialect fork cannot quietly opt out."""
    assert "deposit" not in GQAArenaCache.__dict__
    assert "deposit_from_cache" not in GQAArenaCache.__dict__
    assert "_capture_geometry" not in GQAArenaCache.__dict__


# ======================================================================
# Part 2 — the seating lever
# ======================================================================
def _seed_grafts(arena, ntoks):
    arena.grafts = [{"ntok": int(n), "text": f"node-{i}"}
                    for i, n in enumerate(ntoks)]
    return list(range(len(ntoks)))


def test_seat_plan_off_returns_the_picks_unchanged():
    """THE OFF-PATH BYTE-IDENTITY RECEIPT for Part 2."""
    arena = _arena()
    picks = _seed_grafts(arena, [5, 6, 7])
    order, info = arena._rs3_seat_plan(picks, seat_near_live=False)
    assert order == picks
    assert info["seat_near_live"] is False
    assert info["delta_positions"] == 0
    assert info["mount_pos0"] == STUB_N_SINK
    assert info["seat_offset_plan_head"] == STUB_N_SINK
    assert info["seat_order"] == picks


def test_seat_plan_on_puts_the_plan_head_last_and_adjacent_to_live_shift():
    arena = _arena()
    picks = _seed_grafts(arena, [5, 6, 7])       # plan head is 0, 5 tokens
    order, info = arena._rs3_seat_plan(picks, seat_near_live=True)
    assert order == [1, 2, 0]                    # head moved to the END
    assert info["seat_order"] == [1, 2, 0]
    assert info["plan_head"] == 0
    mount_ntok = 5 + 6 + 7
    assert info["mount_ntok"] == mount_ntok
    assert info["mount_pos0"] == STUB_LIVE_SHIFT - mount_ntok
    assert info["delta_positions"] == \
        (STUB_LIVE_SHIFT - mount_ntok) - STUB_N_SINK
    # The head starts head_ntok rows before the band's end and its LAST token
    # sits at live_shift - 1: adjacent to the first live token.
    assert info["seat_offset_plan_head"] == STUB_LIVE_SHIFT - 5
    assert info["plan_head_last_position"] == STUB_LIVE_SHIFT - 1
    assert info["plan_head_adjacent_to_live_shift"] is True


def test_seat_plan_on_with_one_mount_still_lands_the_head_at_the_band_top():
    arena = _arena()
    picks = _seed_grafts(arena, [9])
    order, info = arena._rs3_seat_plan(picks, seat_near_live=True)
    assert order == [0]
    assert info["mount_pos0"] == STUB_LIVE_SHIFT - 9
    assert info["seat_offset_plan_head"] == STUB_LIVE_SHIFT - 9
    assert info["plan_head_last_position"] == STUB_LIVE_SHIFT - 1


def test_seat_plan_fails_closed_when_the_block_is_wider_than_the_band():
    """Top-down seating a block wider than the band would overrun the SINK.

    Production's own width law already forbids such a block; this lever must
    not be the thing that corrupts the sink when one arrives anyway.
    """
    arena = _arena()
    picks = _seed_grafts(arena, [STUB_WIDTH + 5])
    order, info = arena._rs3_seat_plan(picks, seat_near_live=True)
    assert order == picks
    assert info["seat_near_live"] is False
    assert info["seat_near_live_requested"] is True
    assert info["declined_reason"] == "block_wider_than_band"
    assert info["mount_pos0"] == STUB_N_SINK
    assert info["delta_positions"] == 0


def test_seat_plan_with_no_mounts_reports_that_it_moved_nothing():
    arena = _arena()
    arena.grafts = []
    order, info = arena._rs3_seat_plan([], seat_near_live=True)
    assert order == []
    assert info["seat_near_live"] is False
    assert info["declined_reason"] == "no_mounts_to_seat"


def test_seat_plan_resolves_through_the_env_when_not_passed(monkeypatch):
    arena = _arena()
    picks = _seed_grafts(arena, [4, 4])
    monkeypatch.setenv(SEAT_NEAR_LIVE_ENV, "1")
    _order, info = arena._rs3_seat_plan(picks)
    assert info["seat_near_live"] is True
    monkeypatch.setenv(SEAT_NEAR_LIVE_ENV, "banana")
    _order, info = arena._rs3_seat_plan(picks)
    assert info["seat_near_live"] is False
    monkeypatch.delenv(SEAT_NEAR_LIVE_ENV)
    _order, info = arena._rs3_seat_plan(picks)
    assert info["seat_near_live"] is False


def test_rotate_injection_off_returns_the_callers_own_object():
    """OFF must not even COPY the block — the legacy branch's object is what
    reaches ``_set_injection_host``."""
    arena = _arena()
    picks = _seed_grafts(arena, [3, 4])
    _order, info = arena._rs3_seat_plan(picks, seat_near_live=False)
    inj = [{"k": np.zeros((1, 2, 9, HEAD_DIM), np.float32),
            "v": np.zeros((1, 2, 9, HEAD_DIM), np.float32)}]
    assert arena._rs3_rotate_injection(inj, info) is inj


def test_rotate_injection_leaves_the_sink_and_the_value_payload_alone():
    arena = _arena()
    picks = _seed_grafts(arena, [3, 2])
    _order, info = arena._rs3_seat_plan(picks, seat_near_live=True)
    rng = np.random.default_rng(0)
    n_rows = STUB_N_SINK + 5
    k = rng.standard_normal((1, 2, n_rows, HEAD_DIM)).astype(np.float32)
    v = rng.standard_normal((1, 2, n_rows, HEAD_DIM)).astype(np.float32)
    out = arena._rs3_rotate_injection([{"k": k.copy(), "v": v.copy()}], info)
    # The SINK's rows are untouched: moving them would be a second variable.
    np.testing.assert_array_equal(out[0]["k"][:, :, :STUB_N_SINK],
                                  k[:, :, :STUB_N_SINK])
    # Only the KEY carries position; the value payload passes through.
    np.testing.assert_array_equal(out[0]["v"], v)
    # The mount's rows DID move, and the block's shape is unchanged (the
    # PHYSICAL cache rows stay a packed prefix after the sink).
    assert not np.array_equal(out[0]["k"][:, :, STUB_N_SINK:],
                              k[:, :, STUB_N_SINK:])
    assert out[0]["k"].shape == k.shape


# ======================================================================
# The invariants Part 2 rests on
# ======================================================================
def test_relocatable_keys_invariant_payload_same_rope_differs_by_rotation():
    """``docs/GRM_Methodology.md`` §4, checked directly.

    The SAME graft mounted at both ends of the band has the SAME stored K/V
    payload — the stored keys are PRE-RoPE and therefore position-free — while
    the RoPE-APPLIED keys differ, and differ only by a rotation: same per-row
    norms.  This is what makes Part 2 a positioning change and nothing else.
    """
    arena = _arena()
    rng = np.random.default_rng(3)
    payload = rng.standard_normal((1, 2, 6, HEAD_DIM)).astype(np.float32)
    before = payload.copy()

    near_end = STUB_N_SINK                         # adjacent to the sink
    far_end = STUB_LIVE_SHIFT - payload.shape[-2]  # adjacent to the live band

    near = _rope_at(arena, payload, near_end)
    far = _rope_at(arena, payload, far_end)

    # (1) THE STORED PAYLOAD IS UNCHANGED by being seated at either end.
    np.testing.assert_array_equal(payload, before)
    # (2) The RoPE-applied keys genuinely DIFFER — the arm is doing something.
    assert np.abs(near - far).max() > 1e-3
    # (3) ...but only by a ROTATION: per-row norms are preserved.
    np.testing.assert_allclose(
        np.linalg.norm(near, axis=-1), np.linalg.norm(far, axis=-1),
        rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        np.linalg.norm(payload, axis=-1), np.linalg.norm(far, axis=-1),
        rtol=1e-5, atol=1e-5)


def test_constant_delta_composition_is_the_relocation_and_span_style_is_not():
    """THE CONSTANT-DELTA LAW.

    A band relocation moves every row by the SAME amount.  Rotating the block
    "at position delta" instead rotates row j by ``delta + j``, which composes
    with the layer's own ``n_sink + j`` to ``n_sink + delta + 2j`` — a SHEAR
    that de-phases the block's rows against one another.  Both halves are
    pinned: the constant-delta composition MATCHES the direct rotation, and
    the span-style one does NOT.
    """
    arena = _arena()
    rng = np.random.default_rng(5)
    mount = rng.standard_normal((1, 2, 6, HEAD_DIM)).astype(np.float32)
    delta = (STUB_LIVE_SHIFT - mount.shape[-2]) - STUB_N_SINK
    assert delta > 0

    want = _rope_at(arena, mount, STUB_N_SINK + delta)

    # The implementation's rotation, then the layer's own.
    got = _rope_at(arena, arena._rs3_rotate_rows(mount, 2, delta), STUB_N_SINK)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)

    # The span-style pre-rotation this seam must NOT use.
    sheared = _rope_at(arena, _rope_at(arena, mount, delta), STUB_N_SINK)
    assert np.abs(sheared - want).max() > 1e-2


def test_constant_delta_rotation_preserves_per_row_norms():
    arena = _arena()
    rng = np.random.default_rng(7)
    mount = rng.standard_normal((1, 2, 5, HEAD_DIM)).astype(np.float32)
    rotated = arena._rs3_rotate_rows(mount, 2, 13)
    np.testing.assert_allclose(
        np.linalg.norm(mount, axis=-1), np.linalg.norm(rotated, axis=-1),
        rtol=1e-5, atol=1e-5)


def test_constant_delta_rotation_round_trips_through_its_inverse():
    arena = _arena()
    rng = np.random.default_rng(11)
    mount = rng.standard_normal((1, 2, 4, HEAD_DIM)).astype(np.float32)
    there = arena._rs3_rotate_rows(mount, 2, 9)
    back = arena._rs3_rotate_rows(there, 2, -9)
    np.testing.assert_allclose(back, mount, rtol=1e-5, atol=1e-5)


def test_constant_delta_rotation_of_zero_is_the_identity_object():
    arena = _arena()
    mount = np.zeros((1, 2, 3, HEAD_DIM), np.float32)
    assert arena._rs3_rotate_rows(mount, 2, 0) is mount


def test_rotated_injection_lands_the_block_at_the_requested_positions():
    """End to end for Part 2's mechanism, on the injection block itself.

    After the seam pre-rotates and the LAYER applies its own
    ``cos.slice(0, 0, graft_seats)`` rotation, the mount's net positions must
    be ``[mount_pos0, mount_pos0 + mount_ntok)`` and the sink's must still be
    ``[0, n_sink)``.
    """
    arena = _arena()
    picks = _seed_grafts(arena, [3, 2])
    _order, info = arena._rs3_seat_plan(picks, seat_near_live=True)
    mount_ntok = info["mount_ntok"]
    rng = np.random.default_rng(13)
    n_rows = STUB_N_SINK + mount_ntok
    k = rng.standard_normal((1, 2, n_rows, HEAD_DIM)).astype(np.float32)

    out = arena._rs3_rotate_injection(
        [{"k": k.copy(), "v": np.zeros_like(k)}], info)
    # The layer RoPEs the WHOLE injected block at [0, graft_seats).
    applied = _rope_at(arena, out[0]["k"], 0)

    # The sink still lands where production puts it.
    np.testing.assert_allclose(
        applied[:, :, :STUB_N_SINK],
        _rope_at(arena, k[:, :, :STUB_N_SINK], 0), rtol=1e-5, atol=1e-5)
    # The mount lands at the REQUESTED band offset.
    np.testing.assert_allclose(
        applied[:, :, STUB_N_SINK:],
        _rope_at(arena, k[:, :, STUB_N_SINK:], info["mount_pos0"]),
        rtol=1e-5, atol=1e-5)


def test_legacy_injection_lands_the_block_where_production_puts_it():
    """The OFF control for the test above: with the lever off the mount's net
    positions are ``[n_sink, n_sink + mount_ntok)``, production's own."""
    arena = _arena()
    picks = _seed_grafts(arena, [3, 2])
    _order, info = arena._rs3_seat_plan(picks, seat_near_live=False)
    rng = np.random.default_rng(17)
    n_rows = STUB_N_SINK + 5
    k = rng.standard_normal((1, 2, n_rows, HEAD_DIM)).astype(np.float32)
    out = arena._rs3_rotate_injection([{"k": k.copy(), "v": np.zeros_like(k)}],
                                      info)
    applied = _rope_at(arena, out[0]["k"], 0)
    np.testing.assert_allclose(
        applied[:, :, STUB_N_SINK:],
        _rope_at(arena, k[:, :, STUB_N_SINK:], STUB_N_SINK),
        rtol=1e-5, atol=1e-5)
