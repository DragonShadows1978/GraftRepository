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

from core.grm_text_norm import normalize_glyphs


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


ADMISSION_RULE_ENV = "GRM_ADMISSION_RULE"

# ---------------------------------------------------------------------------
# GRM-F5 — sole-binder insurance.
#
# F3 proved (docs/GRM_F3_ROUTING_AT_DISTANCE_LEDGER.md, row 10) that BOTH
# shipping admission rules have a branch NAMED for an identifier insurance
# they do not provide:
#
#   * ``margin_first_plan`` -> ``margin_insurance_k3_identifier_tiebreak``
#     reorders only WITHIN rank-1's exact-score tie group, then returns
#     ``ranking[:3]``;
#   * ``policy_plan`` -> ``one_off_rank_identifier_insurance_k3`` fires
#     precisely when there is exactly one off-rank binder, then returns
#     ``ranking[:3]``.
#
# Either way a sole binder outside the top 3 -- or outside the route window
# entirely -- is silently dropped.  The asymmetry that makes this a defect
# rather than a design choice: with TWO OR MORE binders the frozen rule takes
# ``declared_synthesis_identified_set`` and admits EVERY identified candidate,
# explicitly including ones the router never ranked.  With exactly one, it
# admits none.
#
# The rule this flag installs is structural and carries NO new threshold and
# nothing refit: when the identifier scan yields exactly one binder, that
# binder IS in the plan.  It REORDERS/SUBSTITUTES; it never scores.
#
# Prior art
# ---------
# * RT1 (``demote_non_binding_split_members``, ~200 lines below in this same
#   file; GRM contributors, 2026).  TAKEN VERBATIM: the reorder-never-score
#   stance and the "stable, the only thing that moves is the one node the rule
#   is named for" discipline.  OURS: nothing of the stance; only its
#   application to the identifier-insurance branches.
# * Maximal Marginal Relevance -- Carbonell & Goldstein, SIGIR 1998.  TAKEN:
#   the general shape of reordering an already-scored ranked list under a
#   secondary criterion without rescoring.  NOT taken: MMR's diversity
#   objective and its lambda; there is no tunable here.  Cited by F3 for the
#   same reason.
# * Feathers, *Working Effectively with Legacy Code* (2004), ch. 13 --
#   characterization tests.  TAKEN: the F3 pinned-defect tests stay untouched
#   and the ON-arm counterparts are ADDED beside them, so the pin still
#   records what OFF does.
# * UNVERIFIED against the wider literature (no network in this sandbox) --
#   lead to check.  Search terms: "must-include constraint top-k retrieval",
#   "constrained re-ranking guarantee matched entity", "hard inclusion
#   constraint re-ranking", "maximal marginal relevance reorder".
#
# WHEN THIS RULE STOPS APPLYING (stated with the rule, as every guard here is):
# * EXACTLY ONE binder.  Zero binders -> nothing to insure.  Two or more ->
#   ``declared_synthesis_identified_set`` already admits them all and this
#   rule declines to touch it.
# * It never changes WHICH candidates bind; it consumes the identifier scan's
#   verdict and does not re-run or widen it.
# * It never rescores and never changes ``route_margin_1_2``: a margin-
#   decisive rank-1 plan is left alone, because a decisive margin means the
#   plan is a deliberate singleton, not an insurance window.
# * It costs the plan's LAST slot, never its head.  If a plan of three is the
#   fit stage's contract, the substituted-out node is the one the ranker
#   trusted least.
ROUTE_SOLE_BINDER_INSURANCE_ENV = "GRM_ROUTE_SOLE_BINDER_INSURANCE"

#: The names the two insurance branches take once they actually insure.  New
#: names, not amended old ones: a receipt reader must be able to tell an
#: insured plan from a pre-F5 one by the branch string alone.
SOLE_BINDER_BRANCH_MARGIN_FIRST = "margin_insurance_k3_sole_binder_inserted"
SOLE_BINDER_BRANCH_ALL_TOKENS_BIND = "one_off_rank_identifier_insurance_k3_sole_binder_inserted"


def route_sole_binder_insurance_enabled(
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Resolve the F5 switch.  Default OFF; unknown tokens fail CLOSED to OFF.

    OFF is the direction an unknown token falls because this flag is not yet
    a default: an operator who mistypes it gets the pre-F5 world, which is
    the world every frozen receipt on disk was recorded in.  (A-DEC's own
    ``env_adm_decisive_override`` fails closed the same way; RT1 fails closed
    to ON because RT1 *is* the default.)
    """
    env = os.environ if environ is None else environ
    value = str(env.get(ROUTE_SOLE_BINDER_INSURANCE_ENV, "")).strip().casefold()
    return value in _ENV_TRUE


def insure_sole_binder(
    *,
    plan: Sequence[int],
    identified_candidates: Sequence[int],
    enabled: bool,
) -> tuple[list[int], bool]:
    """Put the sole identifier binder in ``plan``; return ``(plan, inserted)``.

    STABLE and MINIMAL, RT1's stance: the binder takes the plan's LAST slot
    and every other member keeps both its membership and its relative order.
    Nothing is scored, nothing is appended beyond the incoming width, and an
    empty plan is left empty (there is no slot to spend).

    NO-OP CONDITIONS, all structural:

    * the flag is OFF -- the returned list is the caller's, unchanged;
    * the scan did not yield EXACTLY one binder;
    * the binder is already in the plan (at any position -- including rank 1,
      which is the control the F5 fixtures pin);
    * the plan is empty.
    """
    planned = [int(value) for value in plan]
    if not enabled:
        return planned, False
    identified = [int(value) for value in identified_candidates]
    if len(identified) != 1 or not planned:
        return planned, False
    binder = identified[0]
    if binder in planned:
        return planned, False
    return [*planned[:-1], binder], True


def admission_rule(environ: Mapping[str, str] | None = None) -> str:
    """FIX-6 is opt-in; unset/unknown values retain the frozen default."""
    env = os.environ if environ is None else environ
    return "margin_first" if env.get(ADMISSION_RULE_ENV) == "margin_first" else "all_tokens_bind"


def shaped_identifier_tokens(arena: Any, question: str) -> list[str]:
    # Prior art: LT1 offline shaped_tokens, GRM contributors (2026).
    # Exact port of the lexical proxy and stop list, not an entity recognizer.
    normalized = arena._norm_text(question)
    selected = set(arena._rare_tokens(question))
    for raw in re.findall(r"[A-Za-z0-9][\w:.,\-]*", normalized):
        raw = raw.rstrip(".,:;")
        low = raw.casefold()
        if low in arena._QUERY_LEX_STOP or len(low) < 2:
            continue
        if raw[:1].isupper() or "-" in raw:
            selected.add(low)
    return list(dict.fromkeys(w for w in normalized_words(normalized) if w in selected))


def margin_first_plan(*, ranking: Sequence[int], route_margin_1_2: float,
                      identified_candidates: Iterable[int],
                      scores: Mapping[int, float] | None = None,
                      sole_binder_insurance: bool | None = None,
                      ) -> tuple[list[int], str, list[int]]:
    # Prior art: LT1 offline Rule2 / A-DEC / RT1 stable partition, GRM (2026).
    # Borrow strict frozen margin and EXACT top-score tie-break verbatim.
    # New: shared production entrypoint; no prior art known for exact composition.
    ranked = list(ranking)
    if not ranked:
        return [], "empty_ranking", ranked
    if route_margin_1_2 > MARGIN_THRESHOLD:
        return ranked[:1], "fit_margin_decisive_rank1", ranked
    # GRM-F5: ``identified_candidates`` is declared Iterable and was consumed
    # exactly once before this item; it is now read TWICE (tie group, then the
    # insurance), so materialize it before the first read rather than letting a
    # generator argument silently yield an empty second pass.
    identified_candidates = list(identified_candidates)
    hits = set(identified_candidates)
    if scores:
        tied = [i for i in ranked if scores.get(i) == scores.get(ranked[0])]
        ranked = ([i for i in tied if i in hits] + [i for i in tied if i not in hits]
                  + [i for i in ranked if i not in tied])
    elif route_margin_1_2 == 0.0 and len(ranked) > 1 and hits.intersection(ranked):
        raise AdmissionPolicyError("MISSING_TIED_SCORE_GROUP")
    # GRM-F5: the branch is named for an insurance it did not provide.  Under
    # the flag the sole binder takes the last of the three slots; OFF, the
    # call is a no-op and both the plan and the branch string are pre-F5.
    plan, inserted = insure_sole_binder(
        plan=ranked[:3],
        identified_candidates=list(identified_candidates),
        enabled=route_sole_binder_insurance_enabled()
        if sole_binder_insurance is None else bool(sole_binder_insurance),
    )
    branch = (SOLE_BINDER_BRANCH_MARGIN_FIRST if inserted
              else "margin_insurance_k3_identifier_tiebreak")
    return plan, branch, ranked


def normalized_words(text: str) -> list[str]:
    """ADM1 word scan with the shared routing glyph projection."""
    # Prior art: SC1.1/DET1.4 (GRM contributors, 2026), Unicode UAX15/UCD17
    # (Unicode Consortium, 2025). FIX-8 applies the existing normalizer to
    # own-text evidence, including digest/split/recency nodes; no inherited
    # routing-key union is accepted as proof of identifier presence.
    return [
        token.rstrip(".,:;").casefold()
        for token in re.findall(r"[A-Za-z0-9][\w:.,\-]*", normalize_glyphs(text or ""))
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


# ---------------------------------------------------------------------------
# GRM-RT1 — a fit-time split child must not outrank the fact node it competes
# with.
#
# MEASURED (RS1 A0 fresh_fact_controls; RS3 reproduced it; the channel
# decomposition is artifacts/grm_rt1/grm_rt1_diagnosis.json): on
# ``sup_solace_fresh`` the router returns ``[2, 4, 3, 1, 0]`` — the width-guard
# INDEX PARENT 2 and both its children 4 and 3 sweep the top three ranks while
# graft 1, the solace FACT node and the ONLY node in the repository whose own
# text binds the question's identifier, is pushed to rank 4.
#
# The cause is not identifier evidence and not L2/lineage.  It is the LATENT
# channel amplified by the split:
#
#   * the lexical channel is a FOUR-WAY TIE (``lex_bonus = 1.000`` for 2, 4, 3
#     AND 1) because the sable competitor's own text names "the Praxis dock and
#     Solace key references are index context", so both query content words hit
#     every member of the split family as well as the fact node;
#   * ``_guard_deposit_width`` gives the index parent ``child_cents``, so
#     ``_cent_score(parent) = max(own centroid, each child's)`` — and each
#     child also competes on its own.  One 159-token topically-hot node
#     therefore gets THREE bites at the apple against a 31-token fact node's
#     one.  ``route_margin_1_2 == 0.0`` is the fingerprint: ranks 1 and 2 are
#     exactly tied because the parent's score IS its winning child's.
#
# A-DEC then sees ``identified_candidates = [1]`` with ``ranking[0] = 2``: the
# sole binder is off-rank-1 and the margin is not above the fitted threshold,
# so the branch is ``one_off_rank_identifier_insurance_k3`` and the plan is
# ``ranking[:3] = [2, 4, 3]`` — the only identifier-binding node in the
# repository is not in the plan at all, and the turn refuses.
#
# THE RULE (structural; NO new threshold, nothing refit):
#
#   A width-guard split child, or its index parent, that does not bind the
#   question's identifier tokens IN ITS OWN TEXT never outranks a candidate
#   that does; and a split member's identifier binding is always judged on its
#   own text, never on the union routing surface it inherited from its parent.
#
# The second clause is the one the index parent needs: ``_guard_deposit_width``
# sets ``parent['rare']`` to the UNION of its own and every child's rare
# tokens, so a parent can appear to bind an identifier only a child's text
# carries.  RT1 judges it on ``parent['text']`` alone.
#
# The rule REORDERS; it never scores.  Relative order within each group is the
# router's, untouched.  When nothing binds, or every binder is itself a split
# member, or no split member is ranked, it is a NO-OP and the router's order
# stands verbatim.  Fixed at ADMISSION, never at readout.
# ---------------------------------------------------------------------------

RT1_RULE = (
    "a width-guard split child (or its index parent) that does not bind the "
    "question's identifier tokens in its OWN text never outranks a candidate "
    "that does, and a split member's binding is judged on its own text, never "
    "on the union surface inherited from its parent"
)

#: The switch RT1 rides.  It is P2A/P2C's, not a new one: the same
#: ``GRM_LSR_FIXES`` that governs the fit-time split and the plan-priority
#: fit, default ON, failing CLOSED to ON on an unknown token.
RT1_ENV_NAME = "GRM_LSR_FIXES"

#: ISOLATION ESCAPE, for measurement only.  ``GRM_LSR_FIXES`` governs the
#: whole P2A+P2C+RT1 family, so turning it off to measure RT1 also removes the
#: fit-time split — and the RT1 arm would then differ from its control by TWO
#: changes instead of one.  This variable turns RT1 ALONE off while P2A/P2C
#: stay on, so the G2 OFF arm is a clean single-variable control against RS3's
#: 8/9.  It is ABSENT in production and, when absent, changes nothing:
#: ``rt1_enabled`` falls through to ``GRM_LSR_FIXES`` exactly as before.
RT1_ONLY_ENV_NAME = "GRM_RT1_RULE"

_RT1_OFF_TOKENS = ("0", "false", "no", "off")


def rt1_enabled(arena: Any = None) -> bool:
    """Resolve the RT1 rule's switch.

    Precedence, and why:

    1. ``GRM_RT1_RULE`` when it is set — the measurement-only isolation
       escape, so a gate can turn RT1 off WITHOUT also turning off the
       fit-time split it must be compared against.  Absent in production.
    2. otherwise ``GRM_LSR_FIXES``, the family switch, resolved through the
       arena's own ``_lsr_fixes_enabled`` when the object has one so RT1 and
       P2C can never disagree about it on a live arena; the environment is
       read with identical semantics for the stubbed arenas the CPU tests
       build through ``__new__``.

    Both fail CLOSED to ON on an unknown token, the direction A-DEC and P2C
    already fail: an operator who mistypes the value KEEPS the fix.
    """
    override = os.environ.get(RT1_ONLY_ENV_NAME)
    if override is not None:
        return str(override).strip().casefold() not in _RT1_OFF_TOKENS
    resolver = getattr(arena, "_lsr_fixes_enabled", None)
    if callable(resolver):
        return bool(resolver())
    value = os.environ.get(RT1_ENV_NAME, "").strip().casefold()
    return value not in _RT1_OFF_TOKENS


def demote_non_binding_split_members(
    *,
    ranking: Sequence[int],
    binding: Iterable[int],
    split_members: Iterable[int],
) -> tuple[list[int], list[int]]:
    """Move non-binding split-family members below every binding non-member.

    ``binding`` is the set of candidates whose OWN text satisfies the frozen
    ADM1 predicate; ``split_members`` is the width-guard family membership the
    arena reports.  Returns ``(reordered_ranking, demoted_ids)``.

    STABLE: the two groups keep the router's relative order internally, so the
    only thing that moves is a non-binding split member that was sitting above
    a binder.  ``demoted_ids`` names exactly those, in the order they were
    demoted from — an empty list means the rule did nothing.

    NO-OP CONDITIONS, all of them structural:

    * no candidate binds — nothing to protect, so nothing moves;
    * every binder is itself a split member — the rule only ever protects a
      NON-member against a member, so it declines to reorder a family against
      itself;
    * no split member outranks a binder — the order already obeys the rule.
    """
    ranked = [int(value) for value in ranking]
    binders = {int(value) for value in binding}
    members = {int(value) for value in split_members}
    # Only a binding NON-member is protected: the rule exists to stop a
    # competitor's chunk from outranking the fact node, not to re-sort one
    # split family against another.
    protected = [value for value in ranked if value in binders - members]
    if not protected:
        return ranked, []
    highest_protected = ranked.index(protected[0])
    demoted = [
        value for position, value in enumerate(ranked)
        if position < highest_protected
        and value in members
        and value not in binders
    ]
    if not demoted:
        return ranked, []
    demoted_set = set(demoted)
    kept = [value for value in ranked if value not in demoted_set]
    return [*kept, *demoted], demoted


def split_member_binds_own_text(
    *,
    text: str,
    ordered_identifier_tokens: Sequence[str],
    rare_identifier_tokens: Iterable[str],
) -> bool:
    """The frozen ADM1 predicate applied to a split member's OWN text.

    A thin, deliberately named wrapper: RT1 introduces no predicate of its
    own, and the name makes the "own text, never the inherited union" clause
    of the rule visible at every call site.
    """
    return is_identifier_binding(
        candidate_text=str(text or ""),
        ordered_identifier_tokens=ordered_identifier_tokens,
        rare_identifier_tokens=rare_identifier_tokens,
    )


def rt1_info_fields(
    *,
    demoted: Sequence[int] = (),
    split_members: Iterable[int] = (),
    ranking_before: Sequence[int] = (),
) -> dict[str, Any]:
    """RT1's receipt fields.  NEVER SILENT, the same law P2A/P2C obey.

    Present on every A-DEC profile whether or not the rule moved anything, so
    a later reader cannot mistake the absence of a field for the absence of a
    demotion.
    """
    return {
        "admission_split_child_demoted": bool(demoted),
        "admission_split_demoted_ids": [int(v) for v in demoted],
        "admission_split_family_ids": sorted(int(v) for v in split_members),
        "admission_ranking_before_demotion": [
            int(v) for v in ranking_before],
    }


def policy_plan(
    *,
    ranking: Sequence[int],
    identified_candidates: Sequence[int],
    route_margin_1_2: float,
    margin_threshold: float = MARGIN_THRESHOLD,
    sole_binder_insurance: bool | None = None,
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
    # GRM-F5: this branch fires precisely when there is EXACTLY ONE off-rank
    # binder, and pre-F5 it returned a window that need not contain it.
    plan, inserted = insure_sole_binder(
        plan=ranked[:3],
        identified_candidates=identified,
        enabled=route_sole_binder_insurance_enabled()
        if sole_binder_insurance is None else bool(sole_binder_insurance),
    )
    branch = (SOLE_BINDER_BRANCH_ALL_TOKENS_BIND if inserted
              else "one_off_rank_identifier_insurance_k3")
    return plan, branch


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
    rule = admission_rule()
    margin_first = rule == "margin_first"
    excluded = {int(value) for value in exclude}
    eligible = [
        int(value) for value in arena._route_cand_base()
        if int(value) not in excluded
    ]
    if not eligible:
        plan, branch = policy_plan(
            ranking=(), identified_candidates=(), route_margin_1_2=0.0)
        empty = {
            "admission_rule": rule,
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
        # NEVER SILENT: the RT1 fields are present even on the degenerate
        # empty-repository profile, so a reader cannot mistake a missing field
        # for a demotion that did not get recorded.
        empty.update(rt1_info_fields())
        return empty

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
    if margin_first:
        ordered = shaped_identifier_tokens(arena, question)
        rare = set(ordered)
    identified_set = {
        index for index in eligible
        if is_identifier_binding(
            candidate_text=str(arena.grafts[index].get("text", "") or ""),
            ordered_identifier_tokens=ordered,
            rare_identifier_tokens=rare,
        )
    }

    # GRM-RT1: a non-binding split-family member never outranks a binder.
    # Applied HERE — after the router returns and BEFORE identified / margin /
    # policy_plan are computed — so the demotion is visible to the A-DEC
    # branch decision itself, which is the whole point: on sup_solace_fresh it
    # turns the off-rank-1 sole binder into rank 1, and the branch from
    # ``one_off_rank_identifier_insurance_k3`` (plan = a competitor's chunks)
    # into ``exactly_one_identifier_decisive_rank1`` (plan = the fact node).
    #
    # Governed by the SAME switch P2A/P2C are, ``GRM_LSR_FIXES``.  With it OFF
    # this block does not run at all and the ranking stays the router's
    # verbatim, so the legacy path is byte-identical (test-pinned).
    ranking_before = list(ranking)
    rt1_demoted: list[int] = []
    rt1_members: set[int] = set()
    if not margin_first and rt1_enabled(arena) and ranking:
        member_probe = getattr(arena, "_split_family_members", None)
        if callable(member_probe):
            rt1_members = {int(value) for value in member_probe(ranking)}
        if rt1_members:
            # Binding is judged on each candidate's OWN text.  For a
            # non-member that is identical to ``identified_set``; for a split
            # member it deliberately is NOT, because the width guard handed
            # the index parent the union rare surface.
            own_text_binders = {
                index for index in ranking
                if split_member_binds_own_text(
                    text=arena.grafts[index].get("text", ""),
                    ordered_identifier_tokens=ordered,
                    rare_identifier_tokens=rare,
                )
            }
            ranking, rt1_demoted = demote_non_binding_split_members(
                ranking=ranking,
                binding=own_text_binders,
                split_members=rt1_members,
            )
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
    scores = None
    if margin_first or len(eligible) <= 16 or margin_relevant:
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
        # The reconstruction guard checks the ROUTER against the frozen score
        # law, so it must compare against the router's own output.  RT1's
        # demotion is an admission-stage REORDER applied after that output;
        # comparing the reordered list here would make the guard fire on the
        # very defect it is meant to be blind to.  ``ranking_before`` is
        # ``ranking`` verbatim whenever RT1 moved nothing (and always with
        # GRM_LSR_FIXES off), so the legacy check is unchanged.
        if reference[:len(ranking_before)] != ranking_before:
            raise AdmissionPolicyError(
                "production route ranking differs from frozen A-DEC score "
                f"reconstruction: backend={route_backend} "
                f"production={ranking_before} "
                f"reference={reference[:len(ranking_before)]}"
            )
        if len(reference) >= 2:
            margin = float(scores[reference[0]] - scores[reference[1]])
        margin_evaluated = True
    # GRM-RT1: the MARGIN is a property of the router's own top two, and the
    # ``one_off_rank_identifier_insurance_k3`` branch asks whether the sole
    # binder sits off rank 1.  After a demotion the binder IS rank 1, so that
    # branch is not reached and the stale (pre-demotion) margin cannot decide
    # anything; it is still reported verbatim so the receipt shows the tie
    # (0.0 on sup_solace_fresh) that the split family produced.
    plan, branch = policy_plan(
        ranking=ranking,
        identified_candidates=identified,
        route_margin_1_2=margin,
    )
    if margin_first:
        plan, branch, ranking = margin_first_plan(
            ranking=ranking_before, route_margin_1_2=margin,
            identified_candidates=identified, scores=scores)
    profile = {
        "admission_rule": rule,
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
    profile.update(rt1_info_fields(
        demoted=rt1_demoted,
        split_members=rt1_members,
        ranking_before=ranking_before,
    ))
    # GRM-F5 receipt.  NEVER SILENT *within the flag's own world*: whenever
    # GRM_ROUTE_SOLE_BINDER_INSURANCE is ON the key is present on EVERY
    # profile, true or false, so a reader cannot mistake a missing field for
    # an insurance that did not get recorded.  When the flag is OFF the key is
    # ABSENT, and that absence is deliberate: OFF must be byte-identical to
    # the pre-F5 world that every frozen receipt on disk (FIX-4's
    # tests/fixtures/grm_scout_fix4/*.json among them) was recorded in.  The
    # presence of the key is therefore itself the "flag was ON" marker.
    if route_sole_binder_insurance_enabled():
        profile["sole_binder_inserted"] = bool(
            branch in (SOLE_BINDER_BRANCH_MARGIN_FIRST,
                       SOLE_BINDER_BRANCH_ALL_TOKENS_BIND))
    return profile


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


def identifier_serving_decision(arena, question, profile, *, exclude=()):
    """Keep admission mount-exempt; distinguish an existing live binder.

    Prior art: GRM contributors (2026), ADM1 is_identifier_binding and EB1
    live exclusions. Reuse the frozen predicate, not routing or score changes.
    David's FIX-4 ruling (2026): live-only binding satisfies the lookup.
    No prior art known to me for this exact repair composition.
    """
    if profile is not None and not profile.get("identified_candidates"):
        # Empty eligible banks have an empty profile token list; scan the
        # question itself so an all-live bank still serves its binding source.
        ordered, rare = ordered_identifier_tokens(arena, question)
        bound = [int(i) for i in sorted(set(exclude))
                 if not arena.grafts[int(i)].get("retired")
                 and is_identifier_binding(
                     candidate_text=str(arena.grafts[int(i)].get("text", "") or ""),
                     ordered_identifier_tokens=ordered,
                     rare_identifier_tokens=rare)] if ordered else []
        if bound:
            return {"served_from": "recency_mount", "served_from_node_ids": bound}
    return identifier_unbound_abstention(profile)


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
    if not profile or profile.get("admission_rule") == "margin_first":
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


# ---------------------------------------------------------------------------
# GRM-LSR-P2C — unseatable nodes: split at deposit, descend at fit
#
# P2A made the fit stage honest and found that Ruling 1 alone cannot flip the
# four ADMISSION-PRUNE probes: their answer-bearing nodes (623-706 chars)
# exceed the 96-seat arena ALONE, so ``fit_unseatable`` is non-empty and the
# shuttle has nothing to serialize.  Explicit degrade is honest, but it still
# serves without the answer.
#
# The principle: the repository never holds a node the arena cannot mount,
# and a node too long to seat is served ACROSS ITS CHUNKS, not replaced by a
# neighbour.  Same shape as co-mount prevention: fix at deposit/admission,
# never at readout.
#
# NO NEW CONSTANT.  ``mountable_budget`` is DERIVED from the arena's own
# width and the recency reserve the arena already applies in ``fit``.
# ---------------------------------------------------------------------------


def mountable_budget(arena: Any, *, recency_reserve: int = 0) -> int:
    """Seats a single node may occupy and still be mountable.

    DERIVATION (registered before the P2C gates, artifacts/lsr_p2c/
    lsr_p2c_registration.json):

    * ``ArenaCache.__init__`` sets ``self.width = arena_width`` and
      ``live_shift = self.n_sink + arena_width``.  The SINK occupies its own
      ``n_sink`` positions BELOW the arena band and the question/answer live
      tokens occupy positions at or above ``live_shift``.  Neither consumes a
      mount seat, so the sink/question reserve against the MOUNT budget is
      structurally ZERO.
    * The only reserve the arena subtracts is in
      ``graft_arena.step()::fit_detail``:
      ``rec_budget = 0 if qrare else sum(grafts[i]["ntok"] for i in rec)``,
      ``budget = self.width - rec_budget`` — the ephemeral recency mounts.
      It is zero for identifier queries (the whole ADMISSION-PRUNE class) and
      zero for every non-ephemeral arena.
    * ``grm_e2e_session._probe_ladder_chat::_fit_for`` and
      ``_budget_fit_mounts`` use ``int(arena.width)`` with no reserve at all.

    So ``mountable_budget = arena.width - recency_reserve``, and the deposit
    guard (which has no question in hand, hence no recency reserve to know)
    uses the reserve-free form.
    """
    width = int(getattr(arena, "width", 0))
    return max(0, width - int(recency_reserve))


def chunk_trip_cap(chunks: Sequence[Any]) -> int:
    """Registered hard cap on ADDITIVE chunk-shuttle trips: ``len(chunks)``.

    A node split at fit time is served across its chunks in document order;
    this is the ceiling on how many such trips a turn may add, fixed before
    the P2C gates ran and never tuned against a result.  It is the exact
    analogue of ``shuttle_trip_cap`` one level down: the plan shuttles over
    plan members, the chunk shuttle over one member's chunks.
    """
    return len(list(chunks))


def split_info_fields(
    *,
    split_parent: int | None = None,
    split_children: Sequence[int] = (),
    split_ephemeral: bool | None = None,
    descended_head: Sequence[int] = (),
    chunk_trips: Sequence[Sequence[int]] = (),
) -> dict[str, Any]:
    """The P2C receipt fields, all ``fit_``-prefixed.

    NEVER SILENT, the same law P2A's ``fit_info_fields`` obeys: a turn that
    split-and-descended says so, naming the parent it split, the children it
    produced, whether that split was persisted or ephemeral, the child set
    that became the plan head, and every chunk trip it composed from.

    ``fit_``-prefixed by contract: ``grm_three_pass.ROUTE_RECEIPT_INFO_PREFIXES``
    is ``("fit_", "abstain")``, so P2B persists every field here for free and
    ``core/grm_three_pass.py`` needs no edit.
    """
    return {
        "fit_split_parent": (
            None if split_parent is None else int(split_parent)),
        "fit_split_children": [int(v) for v in split_children],
        "fit_split_ephemeral": (
            None if split_ephemeral is None else bool(split_ephemeral)),
        "fit_descended_head": [int(v) for v in descended_head],
        "fit_chunk_trips": [[int(v) for v in trip] for trip in chunk_trips],
    }


def admission_info_fields(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Compact deterministic arena receipt fields for a selected plan."""
    fields = {
        "admission_rule": profile.get("admission_rule", "all_tokens_bind"),
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
    # GRM-RT1, NEVER SILENT: a turn that demoted a split member says so, and a
    # turn that did not says THAT, rather than omitting the field.  Profiles
    # built by older frames (or by a test's hand-written dict) simply have no
    # RT1 keys and the receipt reports the rule as inert, which is honest.
    fields.update(rt1_info_fields(
        demoted=profile.get("admission_split_demoted_ids", ()),
        split_members=profile.get("admission_split_family_ids", ()),
        ranking_before=profile.get(
            "admission_ranking_before_demotion", profile.get("ranking", ())),
    ))
    return fields
