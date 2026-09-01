"""Production GRM admission policy selected by the ADM2 operator decision.

``GRM_ADM_DECISIVE`` controls the frozen ADM1 A-DEC policy.  The policy is
permanently ON when the variable is absent; ``GRM_ADM_DECISIVE=0`` restores
the legacy fixed-``k=3`` admission path.  Explicit constructor/CLI values are
used by experiment frames and take precedence over the environment.

The rule below is the content of
``decisiveness_rule_c304609f81475bd2.json`` expressed as production code.  In
particular, the fitted margin is a frozen constant: production never refits it
from serving traffic.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import math
import os
import re
from typing import Any


ENV_NAME = "GRM_ADM_DECISIVE"
FROZEN_RULE_SHA256 = (
    "c304609f81475bd2ae3399ad180d3bb00cc5d2b44b1a49db570387c810defb91"
)
MARGIN_THRESHOLD = 0.1385774091529802

_ENV_TRUE = frozenset(("1", "true", "yes", "on"))
_ENV_FALSE = frozenset(("0", "false", "no", "off", ""))


class AdmissionPolicyError(RuntimeError):
    """The production route could not be represented by the frozen rule."""


def env_adm_decisive_override(
    environ: Mapping[str, str] | None = None,
) -> bool | None:
    """Return an explicit environment choice, or ``None`` when unset.

    As with the probe-ladder and L2 default-on switches, an unknown token
    fails closed to the legacy path rather than silently enabling a policy an
    operator may have meant to disable.
    """
    env = os.environ if environ is None else environ
    if ENV_NAME not in env:
        return None
    value = str(env.get(ENV_NAME, "")).strip().casefold()
    if value in _ENV_TRUE:
        return True
    if value in _ENV_FALSE:
        return False
    return False


def adm_decisive_enabled(
    explicit: bool | None = None,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Resolve A-DEC: explicit caller, then env, then permanent default ON."""
    if explicit is not None:
        return bool(explicit)
    override = env_adm_decisive_override(environ)
    return True if override is None else override


def adm_decisive_cli_argv(enabled: bool) -> list[str]:
    """Freeze a resolved A-DEC value across the E2E driver's restart exec."""
    return ["--adm-decisive"] if enabled else ["--no-adm-decisive"]


def normalized_words(text: str) -> list[str]:
    """ADM1's frozen word normalization, byte-for-byte in semantics."""
    return [
        token.rstrip(".,:;").casefold()
        for token in re.findall(r"[A-Za-z0-9][\w:.,\-]*", text or "")
        if token.rstrip(".,:;")
    ]


def ordered_identifier_tokens(arena: Any, question: str) -> tuple[list[str], set[str]]:
    """Return the production identifier sequence and its rare-token subset."""
    rare = {str(value).casefold() for value in arena._rare_tokens(question)}
    selected = rare or {
        str(value).casefold() for value in arena._query_lex_tokens(question)
    }
    ordered: list[str] = []
    seen: set[str] = set()
    for word in normalized_words(question):
        if word in selected and word not in seen:
            ordered.append(word)
            seen.add(word)
    ordered.extend(sorted(selected - seen))
    return ordered, rare


def is_identifier_binding(
    *,
    candidate_text: str,
    ordered_identifier_tokens: Sequence[str],
    rare_identifier_tokens: Iterable[str],
) -> bool:
    """Frozen ADM1 identifier-binding predicate."""
    words = normalized_words(candidate_text)
    have = set(words)
    rare = {str(value).casefold() for value in rare_identifier_tokens}
    if rare:
        return rare <= have
    ordered = [str(value).casefold() for value in ordered_identifier_tokens]
    if not ordered:
        return False
    needle = ["current", *ordered, "value"]
    width = len(needle)
    return any(
        words[index:index + width] == needle
        for index in range(len(words) - width + 1)
    )


def policy_plan(
    *,
    ranking: Sequence[int],
    identified_candidates: Sequence[int],
    route_margin_1_2: float,
    margin_threshold: float = MARGIN_THRESHOLD,
) -> tuple[list[int], str]:
    """Apply the frozen A-DEC branch precedence to one complete ranking."""
    ranked = [int(value) for value in ranking]
    identified = [int(value) for value in identified_candidates]
    if not ranked:
        return [], "empty_ranking"
    if not identified:
        return ranked[:3], "ambiguous_zero_identifier_hits_k3"
    if len(identified) >= 2:
        identified_set = set(identified)
        # The router may be asked for only the bounded attempt window while
        # identifier binding is enumerated over every eligible candidate.
        # Preserve router order for visible hits, then retain every identified
        # off-window member in the deterministic order supplied by the caller.
        # The branch contract is the complete identified set, not merely its
        # intersection with top-k.
        plan = [value for value in ranked if value in identified_set]
        plan_seen = set(plan)
        for value in identified:
            if value not in plan_seen:
                plan.append(value)
                plan_seen.add(value)
        return plan, "declared_synthesis_identified_set"
    if identified[0] == ranked[0]:
        return [ranked[0]], "exactly_one_identifier_decisive_rank1"
    if float(route_margin_1_2) > float(margin_threshold):
        return [ranked[0]], "fit_margin_decisive_rank1"
    return ranked[:3], "one_off_rank_identifier_insurance_k3"


def decisive_admission_profile(
    arena: Any,
    question: str,
    *,
    exclude: Iterable[int],
    route_limit: int | None = None,
) -> dict[str, Any]:
    """Compute the exact production ranking, score margin, and A-DEC plan.

    The helper deliberately uses the arena's existing routing/scoring laws.
    Binding candidates are enumerated from the eligible text surfaces, while
    the router is asked only for the production attempt window. This keeps a
    full-rank host rescore off the common rank-1 and declared-synthesis paths.
    The expensive exact score reconstruction runs only when an off-rank sole
    binding makes the fitted margin branch decision-relevant (and for small
    diagnostic banks where it is bounded).
    """
    excluded = {int(value) for value in exclude}
    eligible = [
        int(value) for value in arena._route_cand_base()
        if int(value) not in excluded
    ]
    if not eligible:
        plan, branch = policy_plan(
            ranking=(), identified_candidates=(), route_margin_1_2=0.0)
        return {
            "ranking": [],
            "identified_candidates": [],
            "identifier_tokens": [],
            "rare_identifier_tokens": [],
            "identifier_hit_count": 0,
            "route_margin_1_2": 0.0,
            "route_margin_evaluated": False,
            "margin_threshold": MARGIN_THRESHOLD,
            "rank_plan": plan,
            "policy_branch": branch,
            "rule_sha256": FROZEN_RULE_SHA256,
            "route_backend": None,
        }

    probe_key = arena._probe_key(question)
    want = min(
        len(eligible),
        max(3, int(route_limit) if route_limit is not None else 3),
    )
    ranking = list(arena.route(
        question,
        exclude=excluded,
        limit=want,
        probe_key=probe_key,
    ) or ())
    route_backend = str(getattr(arena, "last_route_backend", "unknown"))

    ordered, rare = ordered_identifier_tokens(arena, question)
    identified_set = {
        index for index in eligible
        if is_identifier_binding(
            candidate_text=str(arena.grafts[index].get("text", "") or ""),
            ordered_identifier_tokens=ordered,
            rare_identifier_tokens=rare,
        )
    }
    # Preserve router order for observed hits; any identified member beyond
    # the bounded attempt window is still admitted and receives deterministic
    # graft-id order. Physical assembly sorts the final set in either case.
    identified = [index for index in ranking if index in identified_set]
    identified.extend(sorted(identified_set - set(identified)))
    margin = 0.0
    margin_evaluated = False
    margin_relevant = bool(
        len(identified) == 1
        and ranking
        and identified[0] != ranking[0]
    )
    if len(eligible) <= 16 or margin_relevant:
        base = arena._vector_route_scores(probe_key, eligible)
        if base is None:
            base = {}
            for index in eligible:
                score = arena._cent_score(probe_key, arena.grafts[index])
                if math.isfinite(float(score)):
                    base[index] = float(score)
        base = arena._length_debias_scores(base, eligible)
        base = arena._normalize_scores(base) or {}
        qlex = arena._query_lex_tokens(question)
        scores = {
            int(index): float(base[index])
            + float(arena._lex_bonus(qlex, arena.grafts[index]))
            for index in eligible
            if index in base and math.isfinite(float(base[index]))
        }
        reference = sorted(scores, key=lambda index: (-scores[index], index))
        if reference[:len(ranking)] != ranking:
            raise AdmissionPolicyError(
                "production route ranking differs from frozen A-DEC score "
                f"reconstruction: backend={route_backend} "
                f"production={ranking} reference={reference[:len(ranking)]}"
            )
        if len(reference) >= 2:
            margin = float(scores[reference[0]] - scores[reference[1]])
        margin_evaluated = True
    plan, branch = policy_plan(
        ranking=ranking,
        identified_candidates=identified,
        route_margin_1_2=margin,
    )
    return {
        "ranking": [int(value) for value in ranking],
        "identified_candidates": [int(value) for value in identified],
        "identifier_tokens": list(ordered),
        "rare_identifier_tokens": sorted(rare),
        "identifier_hit_count": len(identified),
        "route_margin_1_2": float(margin),
        "route_margin_evaluated": bool(margin_evaluated),
        "margin_threshold": MARGIN_THRESHOLD,
        "rank_plan": [int(value) for value in plan],
        "policy_branch": str(branch),
        "rule_sha256": FROZEN_RULE_SHA256,
        "route_backend": route_backend,
    }


# ---------------------------------------------------------------------------
# GRM-LSR-P2A Ruling 1 — fit-stage honesty (SHUTTLE over fail-loud)
#
# Phase-1 FINAL adjudication (artifacts/lsr_p1/
# lsr_p1_adjudication_31e5c89453f4aa42.json) found four ADMISSION-PRUNE
# probes sharing one shape: A-DEC planned exactly the answer-bearing node
# (``rank_plan = [X]``), the ladder's later rung widened the mount plan to
# ``[X, Y, Z]``, and expansion-ordered budget truncation then seated the
# WRONG lineage end ``[Y]`` because ``X`` alone already exceeded the 96-seat
# arena.  The planned node never received a seat and nothing said so.
#
# The principle this implements: a planned node is never displaced by an
# unplanned one, and a planned set that cannot be co-seated is SERIALIZED,
# never truncated.
# ---------------------------------------------------------------------------


def plan_priority_fit(
    *,
    plan: Sequence[int],
    candidates: Sequence[int],
    ntok: Mapping[int, int] | Any,
    budget: int,
) -> dict[str, Any]:
    """Seat ``plan`` members first, in plan order, then filler.

    ``candidates`` is the full post-L2, post-expansion pick list for the
    attempt.  Members of ``plan`` that appear in it are seated FIRST in plan
    order; everything else is filler and consumes only the seats left over,
    in its own (expansion) order — the EXPANSION-ORDER truncation law of
    ``graft_arena.step()::fit`` is preserved for filler, which is the only
    population it was ever measured on (2026-06-11 score-order refutation).

    ``ntok`` may be a mapping or any object supporting ``ntok[index]``
    lookup of a per-node seat cost (the arena's ``grafts`` list satisfies
    this via ``grafts[i]["ntok"]`` only through the mapping adapter the
    callers build, so callers pass an explicit dict).

    Returns the complete fit receipt.  ``fit_dropped_planned`` is ``[]``
    unless a plan member is UNSEATABLE (its own ``ntok`` exceeds ``budget``
    even alone); every other unseated plan member lands in
    ``fit_shuttle_pending`` for the caller to serialize across trips.
    """
    budget = int(budget)
    plan_order = [int(value) for value in plan]
    candidate_set = {int(value) for value in candidates}
    plan_in_play = [value for value in plan_order if value in candidate_set]
    plan_set = set(plan_in_play)
    filler = [int(value) for value in candidates if int(value) not in plan_set]

    def cost(index: int) -> int:
        return int(ntok[int(index)])

    # A plan member whose own cost exceeds the whole budget can never be
    # seated by any trip.  That is an explicit degrade, not a substitution.
    unseatable = [value for value in plan_in_play if cost(value) > budget]
    unseatable_set = set(unseatable)
    seatable = [value for value in plan_in_play if value not in unseatable_set]

    seated: list[int] = []
    used = 0
    pending: list[int] = []
    for value in seatable:
        n = cost(value)
        if used + n <= budget:
            seated.append(value)
            used += n
        else:
            # Not a drop: this member is owed its own shuttle trip.
            pending.append(value)

    seated_filler: list[int] = []
    dropped_filler: list[int] = []
    for value in filler:
        n = cost(value)
        if used + n <= budget:
            seated_filler.append(value)
            used += n
        else:
            dropped_filler.append(value)

    return {
        "fit_planned": list(plan_order),
        "fit_seated": sorted(seated + seated_filler),
        "fit_seated_planned": list(seated),
        "fit_seated_filler": list(seated_filler),
        "fit_dropped_planned": list(unseatable),
        "fit_dropped_filler": list(dropped_filler),
        "fit_shuttle_pending": list(pending),
        "fit_unseatable": list(unseatable),
        "fit_used_seats": int(used),
        "fit_budget": int(budget),
    }


def shuttle_trip_cap(plan: Sequence[int]) -> int:
    """Registered hard cap on ADDITIVE shuttle trips: ``len(rank_plan)``.

    Shuttle trips are additive to ``max_trips``; this is the ceiling, fixed
    before the P2A gates ran and never tuned against a result.
    """
    return len(list(plan))


def fit_info_fields(
    receipt: Mapping[str, Any],
    *,
    shuttle: bool = False,
    shuttle_trips: Sequence[Sequence[int]] = (),
    served_without_plan_head: bool = False,
) -> dict[str, Any]:
    """Compact, always-present arena receipt fields for one fit decision.

    NEVER SILENT: every fit decision lands in ``info`` whether or not it
    dropped anything, so a later reader cannot mistake absence of a field
    for absence of a drop.
    """
    return {
        "fit_planned": [int(v) for v in receipt.get("fit_planned", ())],
        "fit_seated": [int(v) for v in receipt.get("fit_seated", ())],
        "fit_dropped_planned": [
            int(v) for v in receipt.get("fit_dropped_planned", ())],
        "fit_dropped_filler": [
            int(v) for v in receipt.get("fit_dropped_filler", ())],
        "fit_unseatable": [int(v) for v in receipt.get("fit_unseatable", ())],
        "fit_shuttle": bool(shuttle),
        "fit_shuttle_trips": [
            [int(v) for v in trip] for trip in shuttle_trips],
        "served_without_plan_head": bool(served_without_plan_head),
    }


# ---------------------------------------------------------------------------
# GRM-LSR-P2A Ruling 2 — not-in-memory abstention (structural, no thresholds)
#
# Scope, stated honestly: the two NOT-YET-DEPOSITED probes of Phase 1 are
# NOT this rule's targets.  Their identifiers DID bind a repository node —
# the node that was the repository's best knowledge at that lived turn — and
# the "wrong" expectation came from the harness's deposit ordering, not from
# a retrieval failure.  No rule here is engineered to make those two pass.
#
# The target is the general Stage C class: a point lookup whose identifier
# tokens bind NO repository node at all.  The trigger is purely structural
# (an empty binding set over the FULL eligible repository), so no threshold
# is registered and none can drift.
# ---------------------------------------------------------------------------

ABSTAIN_REASON_IDENTIFIER_UNBOUND = "identifier_unbound"
ABSTENTION_TEMPLATE = "Not in memory: no stored record matches {tokens}."


def abstention_text(tokens: Sequence[str]) -> str:
    """The fixed abstention string.  Constant template, deterministic order."""
    listed = ", ".join(str(token) for token in tokens)
    return ABSTENTION_TEMPLATE.format(tokens=listed)


def identifier_unbound_abstention(
    profile: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Decide abstention from an A-DEC profile alone.  ``None`` = serve.

    Abstain exactly when the question yielded identifier tokens AND the
    identifier-binding scan over the full eligible repository found nothing.
    ``decisive_admission_profile`` already enumerates ``identified_candidates``
    over every eligible candidate (``for index in eligible``), not merely the
    bounded ranking window, so the scan needs no widening here; its cost is
    one ``normalized_words`` pass per eligible node per turn.

    Non-identifier (ambiguous / topical) questions never abstain this round:
    ``ordered_identifier_tokens`` returns empty for them and the guard below
    falls through.  That class is an open David question, deliberately
    untouched.
    """
    if not profile:
        return None
    tokens = [str(value) for value in profile.get("identifier_tokens", ())]
    if not tokens:
        return None
    if list(profile.get("identified_candidates", ())):
        return None
    # An empty repository is a degenerate case, not an unbound identifier:
    # there is nothing to have failed to match.  Abstaining is still the
    # honest answer, and the same string says so.
    return {
        "abstained": True,
        "abstain_reason": ABSTAIN_REASON_IDENTIFIER_UNBOUND,
        "abstain_identifier_tokens": list(tokens),
        "abstain_text": abstention_text(tokens),
    }


def admission_info_fields(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Compact deterministic arena receipt fields for a selected plan."""
    return {
        "admission_policy": "A-DEC",
        "admission_policy_branch": str(profile["policy_branch"]),
        "admission_rank_plan": [int(value) for value in profile["rank_plan"]],
        "admission_identifier_hit_count": int(profile["identifier_hit_count"]),
        "admission_identified_candidates": [
            int(value) for value in profile["identified_candidates"]
        ],
        "admission_route_margin_1_2": float(profile["route_margin_1_2"]),
        "admission_route_margin_evaluated": bool(
            profile.get("route_margin_evaluated", False)),
        "admission_margin_threshold": MARGIN_THRESHOLD,
        "admission_rule_sha256": FROZEN_RULE_SHA256,
    }
