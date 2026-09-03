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
