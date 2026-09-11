"""GRM-F2 — a chronicle fold must not rebind facts onto an alias, flag-gated.

THE DEFECT THIS ADDRESSES (LT1.1 r2 arm A, measured, not inferred).
``artifacts/grm_d1/REPORT.md`` §3.3-§3.4 records digest node 24 of the arm-A
session store, folded from turns ``[13, 14, 17, 18]``, reading:

    ARCHIVE NOTE. For the archive: the  maintenance crew of the Beacon is
    located at Iona Vale, and the map position of the Beacon is (-31, 48, 12).

The sources say something else.  Turn 13 is "Commtower's maintenance crew
will be Iona Vale."; turn 14 is "Breakwater's map position will be
(-31, 48, 12)."; turn 17 is "Let's call Kestrel 'the Hauler' from now on";
turn 18 is "Let's call Lantern 'the Beacon' from now on".  Commtower's crew
and Breakwater's coordinates were both rebound onto *the Beacon*, which is an
alias for a THIRD entity (Lantern) that owns neither fact.  Era node 61
inherited it.  The store is verbatim in
``artifacts/grm_f2/receipt_digest24.json`` (copied from the read-only r2
run_A cell manifest).

THE MECHANISM, in three measured parts (see ``artifacts/grm_f2/LEDGER.md``):

  1. **The alias turns contribute no facts and are not enumerated.**
     ``ArenaCache._fact_set`` over the four sources yields exactly
     ``{'12', '31', '48', 'iona', 'vale'}`` — five facts, all from turns 13
     and 14.  FIX-5's ``[source N]`` enumeration therefore lists TWO spans,
     both entity-correct ("Commtower's maintenance crew will be Iona Vale.",
     "Breakwater's map position will be (-31, 48, 12)."), and never mentions
     Lantern, Kestrel, the Beacon or the Hauler at all.  Dropping turns 17
     and 18 from the window leaves ``need`` and the enumerated span list
     BYTE-IDENTICAL.  Their entire contribution to the fold is a K/V mount.

  2. **But they ARE mounted.**  ``ArenaCache.consolidate`` calls
     ``_set_inject`` over every source in ``idxs``, so "the Beacon" and "the
     Hauler" are in the model's attention window while the instruction names
     only crews and coordinates.  The model bound the enumerated values to
     the most salient proper name it could see.  The alias turns are pure
     contamination surface: zero instructional weight, full attention weight.

  3. **The coverage gate cannot see it.**  ``_coverage`` scores the digest
     against ``need``, and ``need`` contains no entity name, because
     ``_fact_set``'s multi-word rule needs >=2 consecutive capitalized words
     ("Iona Vale" qualifies; "Commtower" alone does not) and ``_caps_tokens``
     matches ``^[A-Z][\\w\\-]+$``, which the POSSESSIVE surface "Commtower's"
     and "Breakwater's" both fail.  So digest 24 scores ``best_cov = 1.0``,
     ``hit_count = 5/5``, and is accepted at perfect fidelity while every
     fact in it is attributed to the wrong entity.  A coverage gate over
     values alone is blind to attribution by construction.

  And the cost of accepting it is not only a wrong digest: ``consolidate``
  RETIRES its sources, so turns 17 and 18 — the only record that "the
  Beacon" means Lantern — are destroyed to produce a node that misuses that
  very alias.  Grepping the arm-A store for "Breakwater" finds ONE node (the
  retired source); arm A+, whose digest 28 folded ``[13, 14, 21, 22]`` with
  no alias turn in the window, keeps three.

THE TREATMENT — BOTH (a) and (b), because they are cause and detector.

  (a) EXCLUSION.  An alias/rename turn is excluded from chronicle fold
      windows.  This is the CAUSAL fix and it is free: part 1 above measures
      that removing it changes neither ``need`` nor the enumeration, so the
      digest the fold is asked to write is unchanged, while the contaminating
      mount is gone.  The excluded turn stays an active, routable turn node —
      which is where A1's ``_alias_fold_jobs`` looks for it, so exclusion
      here and A1's alias merge compose rather than compete.  This is
      ``fold_window_excludes`` below.

  (b) ATTRIBUTION QC.  A digest whose facts are attributed to a different
      entity than their source is REJECTED, the same way a lossy digest is
      rejected: the fold aborts, sources stay unfolded and routable, and the
      receipt names the offending (entity, value, digest-entities) triple.
      This is the DETECTOR, and it is not redundant with (a): (a) removes one
      known contamination source, (b) is the standing check that the
      resulting digest actually says what the sources said.  A rebind can
      also arrive from a model that simply confuses two entities in the SAME
      window with no alias turn present at all, and (a) is silent there.
      This is ``attribution_violations`` below.

  Evidence for taking both rather than one: on the receipt, (a) alone would
  have prevented digest 24 (the window becomes ``[13, 14]``, enumeration
  unchanged), and (b) alone would have caught it (5 violations, listed in the
  ledger).  On the 13 FIX-5 GPU-contrast folds (real Sol digests, recorded
  under ``artifacts/grm_scout_fix5/gpu/*/receipt.json``) (b) reports ZERO
  violations, so the detector does not cost an accepted fold.  Taking only
  (a) would leave the QC blind spot of part 3 open; taking only (b) would
  keep paying a fold abort — and the retirement of the alias turns — for a
  contamination we can simply not create.

FLAG.  ``GRM_FOLD_ALIAS_GUARD`` — DEFAULT OFF.  With the flag OFF nothing in
this module is called from any serving path and behaviour is byte-identical
to the pre-F2 branch (proved by
``tests/test_grm_f2_alias_guard.py::test_flag_off_byte_identical``).  An
unknown token fails CLOSED to OFF, matching the A1 / L2 / A-DEC precedent in
``grm_alias_fold.py``, ``grm_supersession.py`` and ``grm_admission.py``: a
malformed operator escape never silently enables a behaviour the operator may
not have meant to enable.

WHEN THIS GUARD STOPS APPLYING (round-1 rule: every guard states its edge).
  * It is a LEXICAL proxy, exactly as ``grm_admission.shaped_identifier_tokens``
    and ``grm_alias_fold.parse_alias_edge`` are.  It recognizes an entity as a
    capitalized word (possessive-tolerant); it is not an entity recognizer and
    it has no model.  An entity written in lower case, or a language without
    capitalization, is invisible to it and the fold behaves exactly as it does
    with the flag OFF.
  * It judges ATTRIBUTION, not truth.  It cannot tell a digest that dropped a
    fact (that is ``MIN_FOLD_KEEP``'s job, unchanged) from one that never had
    it; a value the digest does not carry at all raises NO violation here.
  * It cannot adjudicate a fact with no entity in its source clause
    ("the spacing will be 9 voxels" with the subject in a previous sentence).
    Such a fact is skipped, not guessed.
  * The alias allowance is scoped to ONE entity: "the Beacon" is accepted in
    place of "Lantern" and of nothing else.  That scoping is the whole point —
    an unscoped alias allowance is precisely the defect.

PRIOR ART.
  * Maynez et al. (2020), "On Faithfulness and Factuality in Abstractive
    Summarization" (arXiv 2005.00661) — names EXTRINSIC hallucination (content
    not entailed by the source) as distinct from content loss, and shows that
    ROUGE-style overlap metrics do not detect it.  TAKEN: the framing that a
    summary can score perfectly on token overlap while being unfaithful, which
    is exactly part 3 above (``best_cov = 1.0`` on a rebound digest).  NOT
    taken: their entailment-model-based metric — this module uses no model.
    UNVERIFIED (no network in this sandbox) — lead to check; search terms
    ``Maynez 2020 faithfulness factuality abstractive summarization``,
    ``extrinsic hallucination summarization``.
  * Nan et al. (2021), "Entity-level Factual Consistency of Abstractive
    Summarization" (arXiv 2102.09130) — entity precision/recall between source
    and summary as a faithfulness metric.  TAKEN: the unit of measurement —
    the ENTITY, not the token — and the idea of checking it against the source
    mechanically.  NOT taken: their NER pipeline or their training-time
    filtering; our entity proxy is lexical and the check is a hard gate on one
    fold, not a corpus statistic.  UNVERIFIED — lead to check; search terms
    ``Nan 2021 entity-level factual consistency summarization``,
    ``entity hallucination precision summarization``.
  * Goodrich et al. (2019), "Assessing The Factual Accuracy of Generated Text"
    (KDD) — relation-triple extraction (subject, relation, object) from source
    and summary, compared as sets.  TAKEN: the (entity, attribute, value)
    triple as the comparison unit, which is the order's own entity check.
    NOT taken: their learned relation extractor.  UNVERIFIED — lead to check;
    search terms ``Goodrich 2019 factual accuracy generated text``,
    ``relation triple summarization factual consistency``.
  * GRM contributors (2026), LOCAL and verified, all reused UNCHANGED as the
    verbs this module composes: FIX-5 source-enumerated consolidation and its
    ``_fact_set`` fidelity vocabulary (``core/graft_arena.py``); FIX-3's
    "reject rather than deposit a lossy derivative" abort contract and its
    ``no_fold`` exemption (``GraftRepository._fold_once``); A1's
    ``parse_alias_edge`` / ``alias_scan_text`` / ``identifier_set``
    (``core/grm_alias_fold.py``) — the alias detector here is A1's, not a
    second one; SC1.1 / FIX-8 ``normalize_glyphs`` (``core/grm_text_norm.py``)
    so this scan sees the same projection routing and admission see.
  * SC1.1 grounding (``ArenaCache._grounding_verdict``) is the nearest local
    antecedent for "check the produced text against what was mounted".  TAKEN:
    the stance.  NOT taken: its pooled token coverage, which is a VALUE check
    and has the same attribution blind spot this module exists to close.
  * No prior art known to me for this exact composition: a possessive-tolerant
    lexical entity proxy, clause-scoped attribution comparison against the
    FIX-5 enumeration's own source spans, and an alias allowance scoped to the
    single entity its edge registered.
"""

from __future__ import annotations

from collections.abc import Mapping
import os
import re
from typing import Any

from core import grm_alias_fold as _alias_fold
from core.grm_text_norm import normalize_glyphs


ENV_NAME = "GRM_FOLD_ALIAS_GUARD"
_ENV_TRUE = frozenset(("1", "true", "yes", "on"))
_ENV_FALSE = frozenset(("0", "false", "no", "off", ""))

#: Receipt reasons.  Every guard decision names exactly one of these.
REASON_WINDOW_EXCLUDED = "fold_window_alias_turn_excluded"
REASON_ATTRIBUTION = "fold_attribution_entity_mismatch"


def env_fold_alias_guard_override(
    environ: Mapping[str, str] | None = None,
) -> bool | None:
    """Return the explicit environment choice, or ``None`` when unset.

    Unknown tokens fail CLOSED to OFF — the same contract as
    ``grm_alias_fold.env_alias_fold_override``, which also defaults OFF.
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


def fold_alias_guard_enabled(
    explicit: bool | None = None,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Resolve F2: explicit caller > env escape > permanent default OFF."""
    if explicit is not None:
        return bool(explicit)
    override = env_fold_alias_guard_override(environ)
    return False if override is None else override


# -------------------------------------------------------------- entity proxy

#: An entity surface: a capitalized word, with the POSSESSIVE clitic consumed
#: rather than breaking the match.  The possessive is the whole reason the
#: existing ``ArenaCache._caps_tokens`` misses this defect: its
#: ``^[A-Z][\w\-]+$`` is anchored on a whitespace-split token, so
#: "Commtower's" and "Breakwater's" — the exact surfaces of the two rebound
#: entities in the receipt — do not match, while "Beacon" does.  A check
#: built on that class would be blind to precisely the sources it must protect.
_ENTITY = re.compile(r"(?<![\w'’-])([A-Z][\w-]*)(?:['’]s)?\b")

#: Words that are capitalized by POSITION or by TRANSPORT rather than by being
#: an entity.  Sentence-initial function words and the Harmony frame's own
#: vocabulary would otherwise read as entities and make every clause "owned"
#: by "The" or "Assistant".  This is deliberately a small closed list of
#: structural words, NOT a general stopword list: a real entity must never be
#: silently dropped from the check, so anything not clearly structural stays.
#: ``ArenaCache._FACT_STOP``'s members that are also entity-shaped are
#: included for the same reason it excludes them from ``_fact_set``.
ENTITY_STOP = frozenset((
    # pronouns and determiners
    "i", "you", "we", "they", "he", "she", "it", "the", "a", "an", "this",
    "that", "these", "those", "there", "their", "his", "her", "its", "our",
    "my", "your",
    # sentence-initial function words seen in the fixtures' assistant turns
    "and", "but", "so", "if", "when", "what", "which", "who", "how", "why",
    "let", "all", "for", "from", "with", "as", "at", "by", "in", "on", "to",
    "of", "or", "not", "no", "yes", "do", "does", "did", "is", "are", "was",
    "were", "be", "been", "can", "could", "would", "should", "will", "shall",
    "may", "might", "must", "have", "has", "had",
    # Harmony transport / role vocabulary
    "user", "assistant", "system", "chatgpt", "reasoning", "valid", "final",
    "channel", "message", "start", "end", "low", "high",
    # archive scaffolding (``ArenaCache._FACT_STOP`` overlap)
    "archive", "note", "chronicle", "record", "recorded", "combined",
    "conversation", "facts", "period", "during", "actually", "correction",
    "previous", "okay", "noted", "heads", "sure", "right", "good", "sounds",
    "thanks", "great", "nice", "sometimes", "maybe", "still", "also",
))


def entity_tokens(text: str) -> frozenset[str]:
    """Entity surfaces in ``text``, casefolded, FIX-8 normalized.

    The projection is ``normalize_glyphs`` — the SAME one routing, admission
    and A1's parser use — so an entity written with U+2011 is the entity
    written with U+002D.  Purely numeric matches are dropped: those are
    values, and ``ArenaCache._fact_set`` already owns them.
    """
    out = set()
    for match in _ENTITY.finditer(normalize_glyphs(str(text or ""))):
        word = match.group(1).casefold()
        if word in ENTITY_STOP or word.isdigit():
            continue
        out.add(word)
    return frozenset(out)


#: Clause boundaries.  An attribution lives in a clause, not in a document:
#: "the crew of X is Iona Vale, and the map position of Y is (-31, 48, 12)"
#: carries TWO attributions and splitting on sentence punctuation alone would
#: pool them and accept either entity for either value.  The ", and " arm is
#: what separates the receipt's own two conjoined clauses.
_CLAUSE = re.compile(r"[.!?;\n]+|,\s+(?:and|while|but)\s+")


def clauses(text: str) -> list[str]:
    """Attribution-sized spans of ``text``."""
    return [c for c in _CLAUSE.split(str(text or "")) if c and c.strip()]


# --------------------------------------------------------------- alias table

def alias_bindings(source_texts) -> dict[str, str]:
    """``{alias_entity: base_entity}`` registered by the window's own sources.

    The parser is A1's ``parse_alias_edge`` — this module introduces no second
    alias detector, so a form A1 merges is a form this guard honours and a
    form A1 does not see is a form this guard does not invent.  Both sides are
    reduced to their entity tokens and the LAST one is taken as the name
    ("the Beacon" -> ``beacon``), because the determiner is in ``ENTITY_STOP``
    and a multi-word alias's final word is its head in every registered
    fixture form.
    """
    out: dict[str, str] = {}
    for text in source_texts:
        pair = _alias_fold.parse_alias_edge(str(text or ""))
        if pair is None:
            continue
        alias = sorted(entity_tokens(pair[0]))
        base = sorted(entity_tokens(pair[1]))
        if not alias or not base:
            continue
        if alias[-1] == base[-1]:
            continue
        out[alias[-1]] = base[-1]
    return out


def accepted_names(entity: str, aliases: Mapping[str, str]) -> frozenset[str]:
    """Names that may stand for ``entity`` — ITSELF and its registered alias.

    The allowance is bidirectional (a digest may name the base where the
    source named the alias, or the reverse) and it is scoped to the ONE
    binding that names this entity.  An alias registered for a different
    entity is NOT accepted here, which is the whole defect: "the Beacon" is
    Lantern's alias and Commtower's crew may not be filed under it.
    """
    entity = str(entity).casefold()
    out = {entity}
    for alias, base in aliases.items():
        if base == entity:
            out.add(alias)
        elif alias == entity:
            out.add(base)
    return frozenset(out)


# ------------------------------------------------------- (a) window exclusion

def fold_window_excludes(fact_set, source_texts) -> list[int]:
    """Indices of ``source_texts`` that are FACT-LESS alias/rename turns.

    ``fact_set`` is ``ArenaCache._fact_set`` (passed in, not imported, so the
    guard uses the caller's own vocabulary and cannot drift from it).

    Returns positions, not node indices: the caller owns the mapping.  An
    EMPTY result means the window is unchanged and the caller must do nothing
    — the default, and the case for every fold in the FIX-5 GPU contrast set.

    The exclusion never empties a window: if every source is a fact-less alias
    turn there is nothing to preserve and no digest worth writing, so the
    caller is handed the empty list and folds the window as it always did.
    """
    keep_alias = []
    for position, text in enumerate(source_texts):
        text = str(text or "")
        if _alias_fold.parse_alias_edge(text) is None:
            continue
        # Fact-bearing alias turns stay: dropping one would drop a fact.
        if fact_set([text]):
            continue
        keep_alias.append(position)
    if len(keep_alias) >= len(source_texts):
        return []
    return keep_alias


# ----------------------------------------------------- (b) attribution QC

def attribution_violations(fact_set, source_texts, digest_text) -> list[dict]:
    """(entity, attribute, value) triples the digest filed under a WRONG name.

    The entity check, stated exactly as the order states it: every
    ``(entity, attribute, value)`` in the sources must appear in the digest
    with the SAME entity, or with an alias registered FOR THAT ENTITY ONLY.

    Mechanically, per source clause that has both an owning entity and at
    least one value:

      * the OWNER is the clause's entity tokens minus its value tokens (a
        multi-word value like "Iona Vale" is entity-shaped AND a
        ``_fact_set`` fact; it is the value, not the owner, so it is removed
        from the owner set — otherwise every value would vacuously own
        itself);
      * for each VALUE in the clause, the digest clauses that carry that
        value are collected.  A value the digest does not carry at all is
        SKIPPED: that is coverage's question and ``MIN_FOLD_KEEP`` already
        answers it, and raising here would double-punish a lossy digest;
      * the value passes if ANY carrying digest clause names an accepted name
        for the owner.  It is a VIOLATION only when the digest does carry the
        value and NO carrying clause names the owner or its registered alias.

    A clause with no owner (the subject is in a previous sentence) raises
    nothing: the guard does not guess an owner it cannot read.

    ``fact_set`` is passed in for the same reason as above.  Returns a list of
    receipt dicts, empty when the digest is entity-faithful.
    """
    digest_clauses = [(clause, fact_set([clause]), entity_tokens(clause))
                      for clause in clauses(str(digest_text or ""))]
    aliases = alias_bindings(source_texts)
    violations: list[dict] = []
    for position, source in enumerate(source_texts):
        scan = _alias_fold.alias_scan_text(str(source or ""))
        for clause in clauses(scan):
            values = fact_set([clause])
            if not values:
                continue
            owners = entity_tokens(clause) - values
            if not owners:
                continue
            allowed = set()
            for owner in owners:
                allowed |= accepted_names(owner, aliases)
            for value in sorted(values):
                carrying = [(text, ents) for text, facts, ents
                            in digest_clauses if value in facts]
                if not carrying:
                    continue  # coverage's question, not attribution's
                if any(ents & allowed for _, ents in carrying):
                    continue
                violations.append({
                    "reason": REASON_ATTRIBUTION,
                    "source_position": int(position),
                    "source_clause": clause.strip(),
                    "entity": sorted(owners),
                    "value": value,
                    "accepted_names": sorted(allowed),
                    "digest_entities": sorted(
                        set().union(*(ents for _, ents in carrying))
                        if carrying else set()),
                })
    return violations


def guard_receipt(fact_set, source_texts, digest_text,
                  excluded=()) -> dict[str, Any]:
    """One receipt for a guarded fold: what was excluded, what was rejected.

    NEVER SILENT (SCOUT-FIX-9's rule, reused): the receipt is written whether
    or not the guard acted, so a reader cannot mistake an absent field for
    "nothing was checked".
    """
    violations = attribution_violations(fact_set, source_texts, digest_text)
    return {
        "excluded_positions": [int(v) for v in excluded],
        "alias_bindings": alias_bindings(source_texts),
        "violations": violations,
        "attribution_ok": not violations,
    }
