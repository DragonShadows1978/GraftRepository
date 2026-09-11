"""Second scoring column for LT1.1 (amendment 9): value-span, glyph-tolerant.

WHY A SECOND COLUMN AT ALL
--------------------------
The registered scorer (`scripts/grm_lt1.score`) is the primary verdict and is
NOT touched here.  It already does more than a naive comparison: it NFKC-folds
and it maps the Unicode dashes (U+2010..U+2014, U+2212) to "-".  So the common
story that LT1.1 misses are "U+2011 glyph misses" is, for the dash itself,
already false -- `normalize()` folds U+2011 before the span search runs.

What the registered scorer does NOT fold, and what this column adds:

  1. U+202F NARROW NO-BREAK SPACE and U+00A0 -> " ".  The model writes
     "-31,<U+202F>48,<U+202F>12"; NFKC leaves U+202F alone, so the span
     "-31, 48, 12" does not match.  THIS is the real glyph defect, not the dash.
  2. Surrounding parentheses on a coordinate triple, so "(-31, 48, 12)" is
     found in "**-31, 48, 12**".
  3. Unit singular/plural and the hyphen-for-space compound, so the expected
     "9 voxels" is found in "a 9-voxel spacing".

A row counts in column 2 only if the model actually produced the right VALUE.
Nothing here rescues a wrong number, a fabricated entity, or an abstention.

Prior art: token/span normalization before exact-match scoring is standard in
QA evaluation -- SQuAD's official `normalize_answer` (Rajpurkar et al. 2016)
lowercases, strips articles and punctuation, and collapses whitespace before
exact match.  Taken from it: the idea of a declared normalization pipeline
applied identically to prediction and reference.  Mine (not SQuAD's): the
Unicode-space folding, the optional-parenthesis rule, and the unit
singular/plural fold, which are specific to this fixture's value shapes.
SQuAD strips ALL punctuation, which would be wrong here -- it would erase the
minus signs that make "(-31, 48, 12)" the right answer and "(31, 48, 12)" the
wrong one.  Unverified against the current literature (no network in this
sandbox); lead to check.  Search terms: "SQuAD normalize_answer exact match",
"answer span normalization unicode", "numeric answer equivalence QA eval".
"""
from __future__ import annotations

import re
import unicodedata

#: Spaces NFKC does not fold: narrow no-break and no-break.
UNICODE_SPACES = '    '

#: Dashes the registered scorer already folds; repeated so this column is
#: self-contained and can be read without cross-referencing grm_lt1.
UNICODE_DASHES = '‐‑‒–—−'


def normalize_v2(value: str) -> str:
    """NFKC + dashes -> '-' + unicode spaces -> ' ' + whitespace collapse."""
    text = unicodedata.normalize('NFKC', str(value))
    text = text.translate(str.maketrans({c: '-' for c in UNICODE_DASHES}))
    text = text.translate(str.maketrans({c: ' ' for c in UNICODE_SPACES}))
    return ' '.join(text.split())


def _span_found(value: str, text: str) -> bool:
    """The registered scorer's word-boundary span search, on v2-normalized text."""
    if not value:
        return False
    return bool(re.search(r'(?<!\w)' + re.escape(value) + r'(?!\w)', text))


def _variants(value: str):
    """Accepted written forms of one expected value.

    Every variant must still contain the SAME value. These are spelling
    alternatives, never different answers.
    """
    seen, out = set(), []

    def add(v):
        if v and v not in seen:
            seen.add(v)
            out.append(v)

    add(value)
    # (1) optional surrounding parentheses on a coordinate triple
    if value.startswith('(') and value.endswith(')'):
        add(value[1:-1].strip())
    else:
        add('(' + value + ')')
    # (2) unit singular/plural + hyphen-for-space compound:
    #     "9 voxels" -> "9 voxel", "9-voxels", "9-voxel"
    m = re.fullmatch(r'(\d+)\s+([A-Za-z]+?)(s?)', value)
    if m:
        number, unit = m.group(1), m.group(2)
        for u in (unit, unit + 's'):
            add('%s %s' % (number, u))
            add('%s-%s' % (number, u))
    return out


def score_v2(answer: str, expected: str) -> dict:
    """Column 2. Same shape as the registered scorer's result.

    `exact_correct` here means: some accepted written form of the expected
    value appears as a bounded span in the v2-normalized answer.
    """
    text = normalize_v2(answer)
    hit = None
    for variant in _variants(normalize_v2(expected)):
        if _span_found(variant, text):
            hit = variant
            break
    return dict(exact_correct=hit is not None, matched_variant=hit,
                answer=str(answer), normalized=text)


# ---------------------------------------------------------------------------
# GRM-F3 item 4 (F4) — REGISTERED SECONDARY SCORER + ABSTENTION EXTENSION
#
# Registration (lead order GRM_F3_ROUTING_AT_DISTANCE.md item 4): this module's
# `score_v2` column becomes round 2's REGISTERED SECONDARY scorer.  The
# registered PRIMARY stays `scripts/grm_lt1.score`, untouched, and remains the
# verdict of record; nothing below changes it.
#
# ABSTENTION EXTENSION.  The primary's abstention regex did not cover the one
# abstention phrasing round 2 actually produced (arm A `recall_5_25`, turn 41:
# "We're still working on that. ..."), so the row was banked as `wrong_value`
# — `miss_causes.json` names the defect `abstention_uncovered_by_scorer_regex`.
# An abstention scored as a wrong value overstates the wrong-value error and
# understates the abstention error; both columns are wrong by one row.
#
# SCOPE DISCIPLINE (registered BEFORE this pass, applied to r2 rows too): the
# alternatives below were derived by scanning EVERY r2 answer in both arms for
# abstention-shaped phrasing (artifacts/grm_f3/f4_rescore.json records the
# scan).  Exactly ONE family was observed — "we're/we are still working on
# that" — so exactly that family plus its orthographic variants is added.  The
# straight and curly apostrophes both appear in the corpus; `normalize_v2`
# already folds Unicode spaces, and the caller casefolds and maps U+2019 to
# "'" before matching, so only the ASCII form needs to be written here.  No
# speculative phrasing is added: an unobserved alternative cannot be validated
# against this corpus, and a regex that fires on unseen text is a scorer that
# silently reclassifies future rows.
#
# WHEN THIS GUARD STOPS APPLYING: the alternation is pinned to round 2's
# observed corpus.  A later round that produces a different abstention wording
# must re-run the scan and re-register, not widen the regex in place.
#
# Prior art: SQuAD v1.1 official evaluation `normalize_answer` + exact match
# (Rajpurkar et al., 2016) — taken: a declared normalization pipeline applied
# identically to prediction and reference before span matching.  Unanswerable /
# abstention detection as a separate scored class is SQuAD 2.0's (Rajpurkar,
# Jia & Liang, 2018), where "no answer" is a first-class label rather than a
# wrong answer; taken: the principle that abstention must not be banked as a
# wrong value.  Mine (no prior art known to me): this specific alternation,
# which is a corpus-pinned literal list rather than a learned or heuristic
# no-answer classifier.  Unverified against the current literature (no network
# in this sandbox); lead to check.  Search terms: "SQuAD 2.0 unanswerable
# no-answer exact match", "abstention detection QA evaluation regex",
# "calibrated refusal vs wrong answer scoring".
# ---------------------------------------------------------------------------

#: The registered PRIMARY's abstention alternation, copied verbatim from
#: `scripts/grm_lt1.score` so this module is self-contained and auditable
#: without cross-referencing.  It is NOT imported: importing it would let a
#: later edit to the primary silently change the secondary column.
PRIMARY_ABSTAIN_ALTERNATION = (
    r"unknown|not in memory|not (?:provided|specified)"
    r"|(?:do not|don't|cannot|can't) (?:know|recall|find)"
    r"|no (?:stored )?record|not enough information"
)

#: F4 extension — round 2's observed abstention family, and only that family.
#: "we're still working on that" / "we are still working on it" / bare
#: "still working on that".
F4_ABSTAIN_EXTENSION = (
    r"(?:we(?:'re| are) )?still working on (?:that|it|this)"
)

ABSTAIN_RE_V2 = re.compile(
    r"\b(?:" + PRIMARY_ABSTAIN_ALTERNATION + r"|" + F4_ABSTAIN_EXTENSION + r")\b"
)


def _low(text: str) -> str:
    """Primary's casefold + curly-apostrophe fold, on v2-normalized text."""
    return normalize_v2(text).casefold().replace('’', "'")


def abstains_v2(answer: str) -> bool:
    """Column-2 abstention predicate: primary's alternation + F4 extension."""
    text = normalize_v2(answer)
    return bool(ABSTAIN_RE_V2.search(_low(text))) or not text


def score_v2_full(answer: str, expected: str) -> dict:
    """Column 2 with the full category ladder, so a row can be compared
    against the primary category-for-category.

    Precedence follows the registered primary exactly — correct, then refusal,
    then abstention, then wrong_value — so the ONLY behavioural differences
    against the primary are (a) the v2 value normalization and variants and
    (b) the F4 abstention extension.  Refusal uses the primary's alternation
    verbatim for the same reason the abstention one is copied.
    """
    value = score_v2(answer, expected)
    low = _low(answer)
    refusal = bool(re.search(
        r"\b(?:i (?:cannot|can't|won't) (?:help|assist|comply)"
        r"|not able to assist|against (?:my|the) policy|i must refuse)\b", low))
    abstain = abstains_v2(answer)
    if value['exact_correct']:
        category = 'correct'
    elif refusal:
        category = 'refusal'
    elif abstain:
        category = 'abstention'
    else:
        category = 'wrong_value'
    return dict(
        exact_correct=value['exact_correct'],
        matched_variant=value['matched_variant'],
        category=category,
        wrong_value_error=int(category == 'wrong_value'),
        abstention_error=int(category == 'abstention'),
        refusal_error=int(category == 'refusal'),
        refusal_style=refusal, abstention_style=abstain,
        answer=str(answer), normalized=value['normalized'])
