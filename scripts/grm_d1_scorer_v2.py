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
