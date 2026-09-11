"""GRM-SC1.1 — one glyph normalizer, shared by the comparator and grounding.

THE PRINCIPLE THIS ENFORCES: *value comparison is semantics, not glyphs.*
It was minted in the DET1 race across three incidents and lives in
``scripts/grm_det1_common.py::normalize_value_text`` (DET1.4), which is the
DET value COMPARATOR.  Grounding v3 (``ArenaCache._grounding_attribution``)
is the one comparator the principle never reached, and the gap was measured:
SC1's demand trip fetched the withheld node, mounted it, and the model said
the right value 2/2 -- and grounding rejected both, because the model emits
U+2011 (non-breaking hyphen) and ``_rare_tokens``' character class
``[A-Za-z0-9][\\w:.,\\-]*`` does not contain it.  "Quartz‑8‑Jade"
shatters into ``{"8"}`` while the mounted node's ASCII "Quartz-8-Jade"
tokenizes whole, so ``content <= have`` is False on a CORRECT answer.

FIX-8 extends this existing projection with NFKC and the Unicode Dash
property for routing, admission and lexical scans. The DET1.4 corpus remains
compatible; the frozen answer scorer is unchanged.
"""

from __future__ import annotations

import re
import unicodedata

#: DET1.4 class 1 -- Unicode hyphen presentation.  U+2010 HYPHEN and U+2011
#: NON-BREAKING HYPHEN are the ASCII hyphen rendered; the model emits U+2011
#: inside identifier-shaped values and the stored node text carries U+002D.
UNICODE_HYPHENS = ("‐", "‑")

# Prior art: Unicode Consortium, UAX #15 / UCD 17.0 (2025), NFKC and
# PropList Dash (31 code points, including non-Pd minus signs).
# https://www.unicode.org/Public/17.0.0/ucd/PropList.txt
# https://www.unicode.org/reports/tr15/
# GRM SC1.1/DET1.4 (GRM contributors, 2026) supplies the existing projection.
# FIX-8 reuses it at missed identifier boundaries; no new matching algorithm.
UNICODE_DASHES = (
    "-\u058a\u05be\u1400\u1806\u2010\u2011\u2012\u2013\u2014\u2015"
    "\u2053\u207b\u208b\u2212\u2e17\u2e1a\u2e3a\u2e3b\u2e40\u2e5d"
    "\u301c\u3030\u30a0\ufe31\ufe32\ufe58\ufe63\uff0d\U00010d6e\U00010ead"
)
_DASH_TRANSLATION = str.maketrans({glyph: "-" for glyph in UNICODE_DASHES})

#: DET1.4 class 2 -- Markdown emphasis delimiters.  Paired, same-line, and
#: iterated to a fixed point so nested bold/italic is projected too.  The
#: payload's byte order is never changed, only the delimiters are dropped.
_EMPHASIS = re.compile(r"(\*\*|__|\*|_)([^\n]+?)\1")


def normalize_glyphs(text: str) -> str:
    """Shared FIX-8 NFKC/dash projection, retaining SC1.1 emphasis handling.

    Folded:
      1. NFKC, then Unicode 17.0 Dash -> ``"-"``.
      2. Paired Markdown emphasis delimiters (``**``, ``__``, ``*``, ``_``)
         around a same-line payload, iterated to a fixed point.

    DELIBERATELY NOT FOLDED -- separator-to-space / space-to-separator.
    DET1.10 registers ``[-\\s]+`` as an equivalence class, but that is a
    COMPARATOR rule for scoring an answer against a KNOWN expected value:
    ``value_separator_regex`` keeps the token payloads and the token COUNT
    load-bearing, so "Cobalt-2-India" (wrong token) and "Cobalt India"
    (missing token) still fail.  Grounding has no expected value to anchor a
    token count against.  Folding the separator class here would let
    "Cobalt 1 India" ground against "Cobalt-1-India" AND against any text
    that merely contains "cobalt", "1" and "india" somewhere -- a strictly
    weaker grounding gate wearing a normalization's clothes.  Underscore is
    excluded for DET1.10's own reason: it participates in the word-boundary
    class that separates tokens.

    DELIBERATELY NOT FOLDED -- case.  ``normalize_value_text`` ends in
    ``.casefold()`` because a comparator only ever needs the projection.
    ``_caps_tokens``' proper-noun channel is CASE-BEARING: casefolding its
    input empties it (measured: "Harbor" -> nothing).  The tokenizers below
    already lowercase the tokens they EMIT, so case survives exactly as far
    as it does today and no further.
    """
    out = unicodedata.normalize("NFKC", str(text)).translate(_DASH_TRANSLATION)
    previous = None
    while previous != out:
        previous = out
        out = _EMPHASIS.sub(r"\2", out)
    return out
