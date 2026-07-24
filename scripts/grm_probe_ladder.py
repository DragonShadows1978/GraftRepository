"""Fork-A probe ladder planning helpers (stdlib only; no model deps).

Driver-level enforcement of registered production laws on the e2e probe
path. Scoring and grounding stay in ``core.graft_arena``; this module only
decides mount plans and clean-room flags.

GRM3P-LADDER-ON (2026-07-23): probe ladder is DEFAULT-ON permanently.
Escape to the legacy multimount path: CLI ``--no-probe-ladder`` or
env ``GRM_PROBE_LADDER=0`` (also false/off/no).
"""

from __future__ import annotations

import os
from typing import Any, Iterable

_ENV_TRUE = ("1", "true", "yes", "on")
_ENV_FALSE = ("0", "false", "no", "off", "")


def env_probe_ladder_override() -> bool | None:
    """Return True/False if ``GRM_PROBE_LADDER`` is set; None if unset.

    Unset means "use the permanent default" (ON). Explicit ``0``/false/off/no
    (or empty string) is the registered escape to the legacy path.
    """
    if "GRM_PROBE_LADDER" not in os.environ:
        return None
    v = os.environ.get("GRM_PROBE_LADDER", "").strip().lower()
    if v in _ENV_TRUE:
        return True
    if v in _ENV_FALSE:
        return False
    # Unknown token: treat as escape-off (do not silently enable).
    return False


def env_probe_ladder_enabled() -> bool:
    """True only when env explicitly requests ON (``1``/true/yes/on).

    Unset env is NOT enabled under this helper — callers that want the
    permanent default must use :func:`probe_ladder_enabled`.
    """
    return env_probe_ladder_override() is True


def probe_ladder_enabled(args: Any = None) -> bool:
    """Resolve probe-ladder: default ON; escape restores legacy path.

    Priority:
      1. Explicit CLI (``args.probe_ladder`` is True/False, not None)
      2. Explicit env ``GRM_PROBE_LADDER`` (0/false/off/no → off; 1/… → on)
      3. Permanent default ON
    """
    if args is not None and hasattr(args, "probe_ladder"):
        cli = getattr(args, "probe_ladder")
        if cli is not None:
            return bool(cli)
    override = env_probe_ladder_override()
    if override is not None:
        return override
    return True


def probe_ladder_cli_argv(enabled: bool) -> list[str]:
    """CLI tokens that re-lock the resolved value across restart re-exec.

    Always returns an explicit flag so re-exec does not re-resolve against a
    changed env (default-ON parent → ``--probe-ladder``; escape-off parent →
    ``--no-probe-ladder``).
    """
    return ["--probe-ladder"] if enabled else ["--no-probe-ladder"]


def identifier_tokens_from_parts(
    rare: Iterable[str], qlex: Iterable[str]
) -> set[str]:
    """Prefer rare keys; fall back to query-lex content words."""
    rare_set = set(rare or ())
    if rare_set:
        return rare_set
    return set(qlex or ())


def rank1_covers_identifiers(
    id_tokens: set[str],
    rank1_rare: Iterable[str],
    rank1_text_tokens: Iterable[str],
) -> bool:
    """True when rank-1 covers every probe identifier token."""
    if not id_tokens:
        return False
    have = set(rank1_rare or ())
    if id_tokens <= have:
        return True
    have |= set(rank1_text_tokens or ())
    return id_tokens <= have


def build_probe_ladder_attempts(
    *,
    ranking: list[int],
    topk: int,
    precise: list[int] | None,
    point_lookup: bool,
    max_trips: int,
) -> list[tuple[list[int], bool]]:
    """Build ``(planned_ids, clean_room)`` attempts. At most max_trips+1.

    Point lookups force clean_room on every attempt (RECENCY LAW). Trip-0 is
    precise-first when eligible; the one registered retry is multi-mount
    top-k (or the next ranking slice).
    """
    want = max(int(topk), 1)
    multi = [int(x) for x in ranking[:want]]
    clean = bool(point_lookup)
    attempts: list[tuple[list[int], bool]] = []
    if precise is not None:
        attempts.append((list(precise), clean))
        if int(max_trips) >= 1:
            if multi and multi != list(precise):
                attempts.append((multi, clean))
            elif not clean:
                attempts.append((list(precise), True))
    else:
        attempts.append((multi, clean))
        if int(max_trips) >= 1:
            nxt = [int(x) for x in ranking[want: want * 2]]
            if nxt:
                attempts.append((nxt, clean))
            elif not clean:
                attempts.append((multi, True))
    if not attempts:
        attempts = [([], clean)]
    return attempts[: max(1, int(max_trips) + 1)]
