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

``normalize_glyphs`` folds EXACTLY the two glyph classes DET1.4 registered
and nothing else.  It is deliberately not a second definition of them: the
test suite pins ``normalize_glyphs(s).casefold() == normalize_value_text(s)``
over a corpus covering both classes, so the two functions cannot drift.
"""

from __future__ import annotations

import re

#: DET1.4 class 1 -- Unicode hyphen presentation.  U+2010 HYPHEN and U+2011
#: NON-BREAKING HYPHEN are the ASCII hyphen rendered; the model emits U+2011
#: inside identifier-shaped values and the stored node text carries U+002D.
UNICODE_HYPHENS = ("‐", "‑")

#: DET1.4 class 2 -- Markdown emphasis delimiters.  Paired, same-line, and
#: iterated to a fixed point so nested bold/italic is projected too.  The
#: payload's byte order is never changed, only the delimiters are dropped.
_EMPHASIS = re.compile(r"(\*\*|__|\*|_)([^\n]+?)\1")


def normalize_glyphs(text: str) -> str:
    """Project the two REGISTERED glyph classes; fold nothing else.

    Folded (and only these):
      1. ``U+2010``/``U+2011`` -> ``"-"``.
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
    out = str(text)
    for glyph in UNICODE_HYPHENS:
        out = out.replace(glyph, "-")
    previous = None
    while previous != out:
        previous = out
        out = _EMPHASIS.sub(r"\2", out)
    return out
