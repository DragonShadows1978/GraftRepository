"""Fork-A probe ladder planning helpers (stdlib only; no model deps).

Driver-level enforcement of registered production laws on the e2e probe
path. Scoring and grounding stay in ``core.graft_arena``; this module only
decides mount plans and clean-room flags.
"""

from __future__ import annotations

import os
from typing import Any, Iterable


def env_probe_ladder_enabled() -> bool:
    v = os.environ.get("GRM_PROBE_LADDER", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def probe_ladder_enabled(args: Any = None) -> bool:
    """CLI ``--probe-ladder`` or env ``GRM_PROBE_LADDER=1`` (default OFF)."""
    if args is not None and bool(getattr(args, "probe_ladder", False)):
        return True
    return env_probe_ladder_enabled()


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
