"""Registered lexical realization only; does not establish semantic entailment.
Prior art: Unicode Consortium, UAX #15 (1998 onward), NFKC; Python Unicode
casefold/whitespace primitives. https://www.unicode.org/reports/tr15/
House X3/C5 (2026) supplies ordered value-span rule. Independent local duplicate
of the C5 arm S specification; no C5 imports. Lead must reconcile implementations.
Own conservative negation veto/boundaries: no prior art known to me; no novelty
claim. Negation anywhere outside a refusal phrase vetoes realization, including
unrelated negation; arbitrary semantic relation attribution is NOT solved.
"""
import re
import unicodedata

# Unicode Dash property (15.0 local UCD), plus Unicode 16 Garay hyphen and the
# legacy Hyphen property extras. Explicit repertoire is frozen in registration.
DASH_POINTS = (45, 1418, 1470, 5120, 6150, *range(8208, 8214), 8275, 8315,
               8331, 8722, 11799, 11802, 11834, 11835, 11840, 11869, 12316,
               12336, 12448, 65073, 65074, 65112, 65123, 65293, 69293,
               0x10D6E, 0x00AD, 0x30FB, 0xFF65)
DASH_MAP = {p: '-' for p in DASH_POINTS}
NEGATION = re.compile(r"\b(?:not|no|never|neither|nor|cannot|false|incorrect|wrong|without)\b|n['’]t\b")
REFUSALS = ("not in memory", "i don't know", "i do not know", "i cannot", "i can't", "unknown")


def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC', text).translate(DASH_MAP).split()).casefold()


def value_span(text, value, *, refusal=False):
    """Contiguous ordered normalized span, token boundaries, negation veto.
    Prior art and deliberate conservative limitation are documented above.
    No token reordering, token omission, punctuation deletion or fuzzy matching.
    """
    answer, target = normalize(text), normalize(value)
    if not target:
        return False
    for match in re.finditer(r'(?<![\w-])' + re.escape(target) + r'(?![\w-])', answer):
        remainder = answer[:match.start()] + ' ' + answer[match.end():] if refusal else answer
        if not NEGATION.search(remainder):
            return True
    return False


def outcome(text, fixture, truncated=False):
    from scripts.grm_x3_diagnostic import outcome as frozen_exact
    legacy = frozen_exact(text, fixture, truncated)
    correct = any(value_span(text, value) for value in fixture['expected_values'])
    decoy = value_span(text, fixture['decoy_value'])
    refusal = any(value_span(text, phrase, refusal=True) for phrase in REFUSALS)
    # An answer presenting multiple competing classes realizes none of them.
    unique = sum((correct, decoy, refusal)) == 1
    realized = unique and {'correct': correct, 'decoy': decoy, 'refusal': refusal}[fixture['intended_group']]
    return {**legacy, 'category_realized': realized,
            'exact_equality_realization': legacy['category_realized'],
            'value_span_correct': correct and unique, 'value_span_decoy': decoy and unique,
            'value_span_refusal': refusal and unique, 'value_span_error': not (correct and unique),
            'realization_scorer': 'grm.x3.r2.value_span.v1',
            'exact_equality_outcome': legacy}
