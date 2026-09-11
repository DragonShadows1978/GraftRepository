"""Opt-in, CPU-only C5 grounding experiments; no production flag installation.

Prior art: local SC1.1 glyph normalization and ADM1/RT1 binding (GRM, 2026)
are imported, not reimplemented. External leads: Thompson (1968), regular
expression search; Codd (1970), relational keys. Unverified — lead to check:
"Thompson 1968 regular expression search"; "Codd 1970 relational model".
Borrowed: exact spans and entity/relation binding. Ours: this restricted
Word-number-Word / proper-name grammar and four-arm offline comparison.
No prior art known to me for this exact composition; no novelty claim.
"""
from __future__ import annotations

import re

from core.graft_arena import ArenaCache
from core.grm_admission import is_identifier_binding, ordered_identifier_tokens
from core.grm_text_norm import normalize_glyphs

ARMS = ("0", "S", "N", "W")
DEFAULT_RULE = "0"


class TextArena(ArenaCache):
    """Prior art: SC1.1 _StubArena (GRM, 2026); same text-only constructor seam."""

    def __init__(self, texts):
        self.grafts = [{"text": text} for text in texts]


def _entity(question):
    # Prior art: ADM1 current <identifier> value binding (GRM, 2026).
    # Ours: full query grammar admits lowercase names without a dictionary.
    match = re.fullmatch(
        r"(?:Recall probe\.\s*)?What is the current ([A-Za-z]+(?: [A-Za-z]+)*)"
        r" value\?(?: Reply with only the value\.)?", question, re.I)
    return match.group(1).casefold() if match else None


def _bindings(text):
    # Prior art: Thompson (1968), regex extraction; unverified — lead to check.
    # Ours: only standalone affirmative current-entity-value clauses, with
    # an entire three-part code. No substring inside a negated/quoted clause.
    text = normalize_glyphs(text)
    clauses = re.split(r"[.!?\n]+|<\|[^>]*\|>", text)
    out = []
    for clause in clauses:
        match = re.fullmatch(
            r"\s*(?:(?:User:|Assistant:|Understood\s*[—–-])\s*)?"
            r"(?:The\s+)?current\s+([A-Za-z]+(?: [A-Za-z]+)*)\s+value\s+is\s+"
            r"([A-Za-z]+-\d+-[A-Za-z]+)\s*", clause, re.I)
        if match:
            out.append((match.group(1).casefold(), match.group(2).casefold()))
    return out


def binding_candidates(question, texts, *, rule=DEFAULT_RULE):
    """Compare eligible candidate binding only; does not route or admit a mount."""
    if rule not in ARMS:
        raise ValueError(f"unknown grounding rule: {rule}")
    arena = TextArena(texts)
    with arena._glyph_norm(True):
        ordered, rare = ordered_identifier_tokens(arena, question)
        original = {i for i, text in enumerate(texts) if is_identifier_binding(
            candidate_text=text, ordered_identifier_tokens=ordered,
            rare_identifier_tokens=rare)}
    if rule != "N":
        return original
    # Prior art: Codd (1970), keys; unverified — lead to check. Ours: full
    # entity phrase AND relation own the value in one affirmative clause.
    entity = _entity(question)
    return original | {i for i, text in enumerate(texts)
                       if any(name == entity for name, _ in _bindings(text))}


def _span_support(answer, texts, question, *, separator_fold):
    # Prior art: SC1.1/DET value comparator (GRM, 2026) and exact phrase
    # matching (Thompson 1968; unverified — lead to check).
    # Ours: extract source-bound values WITHOUT consulting expected answers;
    # full answer grammar preserves count, order, relation and name boundary.
    entity = _entity(question)
    if entity is None:
        return set()
    answer = normalize_glyphs(answer).strip().rstrip(".").strip()
    separator = r"[-\s]+" if separator_fold else "-"
    code = rf"([A-Za-z]+){separator}(\d+){separator}([A-Za-z]+)"
    envelope = rf"(?:(?:The )?(?:current )?{re.escape(entity)} (?:value )?is )?"
    match = re.fullmatch(envelope + code, answer, re.I)
    if match is None:
        return set()
    value = "-".join(match.groups()).casefold()
    return {i for i, text in enumerate(texts)
            if (entity, value) in _bindings(text)}


def grounding_verdict(answer, texts, question, *, rule=DEFAULT_RULE,
                      normalized=True):
    """Return the production-shaped (bool, contributor-index set) result.

    Expected values are deliberately absent from this interface. S and N
    are independent additive alternatives; no environment flag enables them.
    """
    if rule not in ARMS:
        raise ValueError(f"unknown grounding rule: {rule}")
    arena = TextArena(texts)
    baseline = arena._grounding_verdict(
        answer, list(range(len(texts))), question, normalized=normalized)
    if rule == "0" or baseline[0]:
        return baseline
    if any(hedge in answer.lower() for hedge in arena.HEDGES):
        return baseline
    if rule == "W":
        # Prior art: SC1.1's explicitly rejected global separator fold
        # (GRM, 2026). Ours: execute that negative control using the original
        # pooled-coverage predicate; no special adversarial acceptance path.
        def fold(text):
            return re.sub(r"[-\s]+", " ", normalize_glyphs(text))
        folded = TextArena([fold(text) for text in texts])
        return folded._grounding_verdict(
            fold(answer), list(range(len(texts))), fold(question), normalized=True)
    contributors = _span_support(answer, texts, question,
                                 separator_fold=rule == "S")
    return bool(contributors), contributors
