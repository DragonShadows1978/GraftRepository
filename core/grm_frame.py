"""GRM-EB1: the serving FRAME — ephemeral boat by spec, persistent by escape.

THE SPEC (David, 2026-09-02).  "The chat log is not kept in memory context.
Any chat recall on facts is pulled via GRM, that way the chat can grow to any
length, limited only by RAM and NVMe."

THE FRAME THAT SPEC NAMES is the EPHEMERAL BOAT: ``ArenaCache.step()`` clears
the live cache at the START of every turn, so each turn runs on
``[sink | mounts | this turn]`` alone.  No prior turn's tokens survive in the
model's live KV cache across a turn boundary, and every recall of an earlier
fact is a ROUTED GRAFT pulled from the repository.  Resident seats are then a
CONSTANT for a conversation of any length — the repository IS the context
window, which is exactly the property the spec asks for.

Recency is untouched by this module: ``recency_mounts`` grafts are pulled from
the repository and co-seated as MOUNTS.  They are repository reads, not a
retained chat log, so they are consistent with the spec and stay as they were.

THE ESCAPE, ``GRM_PERSISTENT_BOAT``, restores the old persistent-live-window
frame.  It exists to reproduce FROZEN receipts taken under that frame.  It is
NOT a production fallback: under this order the persistent live window is the
DEVIATION, and "keep it for safety" is precisely what spec-is-law forbids.

FAIL DIRECTION: CLOSED TO EPHEMERAL.  An unknown or mistyped token yields the
SPEC frame, never the persistent one.  Turning the persistent live window back
on changes what the stack serves, so a typo must never make that change.  This
is the same discipline as ``GRM_DEMAND_NGH`` (closed to OFF) and deliberately
the opposite polarity from ``GRM_LSR_FIXES`` (closed to ON) — in each case the
unreadable token lands on the state that does not change what is served.

An explicit boolean from the caller always wins over the env, so a harness can
pin either frame independently of the ambient operator setting.
"""

from __future__ import annotations

from collections.abc import Mapping
import os


#: The registered escape.  Presence with a true token selects the PERSISTENT
#: (pre-EB1) frame; everything else — absent, false token, unknown token —
#: selects the SPEC frame.
ENV_NAME = "GRM_PERSISTENT_BOAT"

_ENV_TRUE = frozenset(("1", "true", "yes", "on"))
_ENV_FALSE = frozenset(("0", "false", "no", "off", ""))


def env_persistent_boat(
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Resolve the escape from the environment alone.

    ``True`` only for an explicit true token.  Absent, explicitly false, and
    UNKNOWN all return ``False`` — the fail-closed-to-ephemeral rule.
    """
    env = os.environ if environ is None else environ
    if ENV_NAME not in env:
        return False
    value = str(env.get(ENV_NAME, "")).strip().casefold()
    if value in _ENV_TRUE:
        return True
    # Both the explicitly-false tokens and any unknown token land here.
    return False


def ephemeral_frame_enabled(
    explicit: bool | None = None,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Resolve the serving frame: explicit caller > escape > SPEC default.

    Returns ``True`` for the ephemeral boat (the spec frame, the default) and
    ``False`` only when a caller explicitly asked for the persistent frame or
    the registered escape is set to a true token.
    """
    if explicit is not None:
        return bool(explicit)
    return not env_persistent_boat(environ)


def frame_receipt(ephemeral: bool,
                  environ: Mapping[str, str] | None = None,
                  ) -> dict[str, object]:
    """The per-turn frame receipt fields shared by every serving path."""
    return {
        "frame_ephemeral": bool(ephemeral),
        "frame_escape_active": bool(env_persistent_boat(environ)),
    }


# ======================================================================
# GRM-RS3 — the two capture/seating levers, BOTH DEFAULT OFF
# ======================================================================
#
# THE MEASURED SEAM (RS2 amendment ``amendment_b1p_capture_position.json``,
# sha256 72f73f08…).  ``ArenaCache.deposit`` runs a harvest forward and stores
# the PRE-RoPE keys, which are position-free — but the forward's QUERIES are
# rotated at ``cos.slice(0, position_offset + shift, L)`` where ``shift`` is
# ``self_attn.live_shift`` (falling back to ``graft_seats`` when ``None``).
# The queries decide each layer's ATTENTION OUTPUT, which is the input to the
# next layer's K/V — so the capture-time query position propagates into every
# layer above 0.  Measured: layer 0 identical, layers 1..23 all changed.
#
# Nothing pins that position.  A virgin arena leaves ``live_shift = None`` and
# harvests at [0, L); any served turn leaves ``live_shift = n_sink +
# arena_width`` and the IDENTICAL ``deposit(text)`` call then harvests at
# [live_shift, live_shift + L) and yields a DIFFERENT graft.  Which graft you
# get depends on whether a turn has been served yet.  ``GRM_CAPTURE_PIN``
# makes it a registered geometry instead of an accident.
#
# FAIL DIRECTION: CLOSED TO OFF, for both flags.  ON changes what the stack
# serves (a different stored payload; a different mount position), so an
# unknown or mistyped token must never make that change.  Same polarity as
# ``GRM_DEMAND_NGH``; the opposite of ``GRM_LSR_FIXES``.  An explicit value
# from the caller always outranks the env, so a harness pins either state
# regardless of the ambient operator setting.

#: Part 1's registered switch.  ``mount`` / ``live`` select a pin; every other
#: token — absent, ``off``, unknown, mistyped — selects OFF (legacy, unpinned).
CAPTURE_PIN_ENV = "GRM_CAPTURE_PIN"

#: Part 2's registered switch.  A true token seats the mount block so the plan
#: head's LAST token is adjacent to ``live_shift``; everything else is OFF.
SEAT_NEAR_LIVE_ENV = "GRM_SEAT_NEAR_LIVE"

#: The pin geometries.  ``off`` is legacy: ``deposit`` does not touch
#: ``live_shift`` at all and the graft is byte-identical to today's.
CAPTURE_PIN_OFF = "off"
#: ``mount``: queries at ``n_sink`` — the geometry a graft is READ in when it
#: is mounted at the band start, which is where production's bootstrap branch
#: always seats it.  "A graft is the text to the model" literally requires the
#: capture and the read to share a geometry; this is that geometry.
CAPTURE_PIN_MOUNT = "mount"
#: ``live``: queries at ``live_shift`` (``n_sink + arena_width``) — the
#: geometry of text fed LIVE immediately before a question, and the geometry
#: RS2's B1p harvested at when it lifted harbor 0.184 -> 0.303 and flipped it.
CAPTURE_PIN_LIVE = "live"

CAPTURE_PINS = (CAPTURE_PIN_OFF, CAPTURE_PIN_MOUNT, CAPTURE_PIN_LIVE)

#: Only these two tokens select a pin.  Case-folded and stripped first.
_CAPTURE_PIN_TOKENS = {
    CAPTURE_PIN_MOUNT: CAPTURE_PIN_MOUNT,
    CAPTURE_PIN_LIVE: CAPTURE_PIN_LIVE,
}


def env_capture_pin(environ: Mapping[str, str] | None = None) -> str:
    """Resolve ``GRM_CAPTURE_PIN`` from the environment alone.

    Returns one of :data:`CAPTURE_PINS`.  ONLY the exact tokens ``mount`` and
    ``live`` (case-folded, stripped) select a pin; absent, empty, ``off``, and
    ANY unknown token all return ``off`` — the fail-closed rule.
    """
    env = os.environ if environ is None else environ
    if CAPTURE_PIN_ENV not in env:
        return CAPTURE_PIN_OFF
    value = str(env.get(CAPTURE_PIN_ENV, "")).strip().casefold()
    # Unknown tokens land on OFF with the false tokens: fail closed.
    return _CAPTURE_PIN_TOKENS.get(value, CAPTURE_PIN_OFF)


def capture_pin_mode(explicit: str | None = None,
                     environ: Mapping[str, str] | None = None) -> str:
    """Resolve the capture pin: explicit caller > env > OFF.

    An explicit value is validated STRICTLY — a caller naming an unknown pin is
    a bug in the caller, not an ambient typo, and silently serving ``off``
    would hide it.  The env path stays fail-closed.
    """
    if explicit is not None:
        value = str(explicit).strip().casefold()
        if value not in CAPTURE_PINS:
            raise ValueError(
                f"{CAPTURE_PIN_ENV} pin must be one of {list(CAPTURE_PINS)}, "
                f"got {explicit!r}")
        return value
    return env_capture_pin(environ)


def env_seat_near_live(environ: Mapping[str, str] | None = None) -> bool:
    """Resolve ``GRM_SEAT_NEAR_LIVE`` from the environment alone.

    ``True`` only for an explicit true token.  Absent, explicitly false, and
    UNKNOWN all return ``False`` — the fail-closed-to-OFF rule.
    """
    env = os.environ if environ is None else environ
    if SEAT_NEAR_LIVE_ENV not in env:
        return False
    value = str(env.get(SEAT_NEAR_LIVE_ENV, "")).strip().casefold()
    if value in _ENV_TRUE:
        return True
    return False


def seat_near_live_enabled(explicit: bool | None = None,
                           environ: Mapping[str, str] | None = None) -> bool:
    """Resolve the seating lever: explicit caller > env > OFF (the default)."""
    if explicit is not None:
        return bool(explicit)
    return env_seat_near_live(environ)


def rs3_receipt(capture_pin: str, seat_near_live: bool,
                environ: Mapping[str, str] | None = None,
                ) -> dict[str, object]:
    """The RS3 lever receipt fields, shared by every path that carries them."""
    return {
        "capture_pin": str(capture_pin),
        "seat_near_live": bool(seat_near_live),
        "capture_pin_env": (os.environ if environ is None
                            else environ).get(CAPTURE_PIN_ENV),
        "seat_near_live_env": (os.environ if environ is None
                               else environ).get(SEAT_NEAR_LIVE_ENV),
    }
