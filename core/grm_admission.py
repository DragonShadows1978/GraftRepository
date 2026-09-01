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
