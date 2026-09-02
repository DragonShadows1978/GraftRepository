"""GRM-SC1.1 — grounding v3 obeys "value comparison is semantics, not glyphs".

ORDER: ``orders/GRM_SC1_1_GROUNDING_GLYPHS.md``.

WHAT THESE TESTS PIN.

1. ``core.grm_text_norm.normalize_glyphs`` folds EXACTLY the two glyph
   classes ``scripts.grm_det1_common.normalize_value_text`` registered
   (DET1.4: U+2010/U+2011, Markdown emphasis) and NOTHING else.  The
   byte-equality corpus is >= 50 strings covering both classes; equality is
   pinned as ``normalize_glyphs(s).casefold() == normalize_value_text(s)``
   because the comparator ends in a casefold that the tokenizers must NOT
   inherit (``_caps_tokens`` is case-bearing; casefolding empties it).
2. The separator/space class DET1.10 registers for the COMPARATOR is
   deliberately NOT folded, and the "cobalt" counterexample the order names
   is pinned as still-ungrounded.
3. The switch: ON normalizes, OFF is byte-identical legacy behaviour.
4. The receipts (``grounding_normalized``, ``grounding_glyph_rescued``) and
   the ``grounding_`` route-receipt prefix.
5. The measured SC1 case: "Quartz‑8‑Jade" (U+2011) against a mounted node
   carrying ASCII "Quartz-8-Jade" grounds ON and does not ground OFF.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.graft_arena import ArenaCache  # noqa: E402
from core.grm_text_norm import UNICODE_HYPHENS, normalize_glyphs  # noqa: E402
from core.grm_three_pass import (  # noqa: E402
    ROUTE_RECEIPT_INFO_PREFIXES,
    _route_receipt_generic_info,
)
from scripts.grm_det1_common import normalize_value_text  # noqa: E402


# --------------------------------------------------------------------------
# 1. One normalizer, shared: byte-equality with the DET1 comparator.
# --------------------------------------------------------------------------

def _equivalence_corpus() -> list[str]:
    """>= 50 strings covering BOTH registered glyph classes.

    Built from real shapes: the supersession battery's identifier values as
    the model emits them (U+2011), as the nodes store them (ASCII), wrapped
    in every Markdown emphasis marker, plus nesting and non-pairing edges.
    """
    values = [
        "Quartz-8-Jade", "Nacre-6-Blue", "Opal-7-Green", "Sable-0-Copper",
        "Delta-4-Drift", "Vortex-3-Sierra", "Raven-9-Ivory", "Auric-4-Alpha",
        "Birch-2-Beacon", "Cobalt-1-India",
    ]
    corpus: list[str] = []
    for value in values:
        for glyph in UNICODE_HYPHENS:
            hyphenated = value.replace("-", glyph)
            corpus.append(hyphenated)
            corpus.append(f"The current value is **{hyphenated}**.")
            corpus.append(f"__{hyphenated}__")
            corpus.append(f"*{hyphenated}*")
            corpus.append(f"_{hyphenated}_")
        corpus.append(f"**{value}**")
        corpus.append(f"The node records {value} for this key.")
    corpus.extend([
        # Nested emphasis -- the comparator iterates to a fixed point.
        "**_Quartz-8-Jade_**",
        "*__Nacre-6-Blue__*",
        "___Opal-7-Green___",
        # Unpaired / cross-line markers must NOT be stripped by either.
        "**unclosed emphasis",
        "a * b * c",
        "line one **bold\nline two bold**",
        # No glyph class present at all -- both must be identity-modulo-case.
        "plain prose with no markers whatsoever",
        "7,400 units and BX-44 and NIGHTJAR",
        "",
        "   ",
        # Underscore inside an identifier: a word-boundary character, and the
        # emphasis rule must not eat it (it is not a PAIRED delimiter here).
        "cobalt_1_india",
        "snake_case_token_here",
        # Both classes at once.
        "**Quartz‑8‑Jade** and __Nacre‑6‑Blue__ on one line",
    ])
    return corpus


def test_equivalence_corpus_is_large_enough_and_covers_both_classes():
    corpus = _equivalence_corpus()
    assert len(corpus) >= 50, len(corpus)
    assert any(g in s for s in corpus for g in UNICODE_HYPHENS)
    assert any("**" in s for s in corpus)
    assert any("__" in s for s in corpus)


@pytest.mark.parametrize("text", _equivalence_corpus())
def test_normalize_glyphs_matches_the_det1_comparator_byte_for_byte(text):
    """The ONE normalizer: the two functions cannot drift on either class.

    ``normalize_value_text`` ends in ``.casefold()``; ``normalize_glyphs``
    deliberately stops before it (see the module docstring), so the pinned
    identity carries the casefold explicitly.
    """
    assert normalize_glyphs(text).casefold() == normalize_value_text(text)


def test_normalize_glyphs_does_not_casefold():
    assert normalize_glyphs("Harbor") == "Harbor"
    assert normalize_value_text("Harbor") == "harbor"


def test_normalize_glyphs_is_idempotent():
    for text in _equivalence_corpus():
        once = normalize_glyphs(text)
        assert normalize_glyphs(once) == once


def test_both_registered_classes_are_actually_folded():
    assert normalize_glyphs("Quartz‑8‑Jade") == "Quartz-8-Jade"
    assert normalize_glyphs("Quartz‐8‐Jade") == "Quartz-8-Jade"
    assert normalize_glyphs("**Nacre-6-Blue**") == "Nacre-6-Blue"
    assert normalize_glyphs("__Nacre-6-Blue__") == "Nacre-6-Blue"
    assert normalize_glyphs("*Nacre-6-Blue*") == "Nacre-6-Blue"
    assert normalize_glyphs("_Nacre-6-Blue_") == "Nacre-6-Blue"


# --------------------------------------------------------------------------
# 2. The separator/space class is NOT folded.
# --------------------------------------------------------------------------

def test_separator_and_space_classes_are_not_folded():
    """DET1.10's ``[-\\s]+`` equivalence is a COMPARATOR rule, not a
    tokenizer rule.  Folding it here would weaken grounding, not normalize
    it -- grounding has no expected value to anchor the token COUNT against.
    """
    assert normalize_glyphs("Cobalt 1 India") == "Cobalt 1 India"
    assert normalize_glyphs("Cobalt-1-India") == "Cobalt-1-India"
    assert normalize_glyphs("Cobalt 1 India") != normalize_glyphs(
        "Cobalt-1-India")
    # DET1.10 excludes underscore from its SEPARATOR class, and so does this
    # normalizer: "_" is never rewritten to "-" or to a space.
    assert "-" not in normalize_glyphs("cobalt_1_india")


def test_snake_case_is_eaten_by_the_INHERITED_emphasis_rule():
    """RESIDUAL RISK, pinned rather than hidden.

    ``normalize_value_text``'s Markdown rule treats ``_`` as an emphasis
    delimiter, so ``cobalt_1_india`` -> ``cobalt1india``.  SC1.1 inherits
    that byte-for-byte on purpose (the order's rule is "EXACTLY as
    ``normalize_value_text`` does for those two classes"), and the two
    functions are pinned equal above.  It is SAFE for grounding only because
    the projection is applied SYMMETRICALLY to the answer and to every
    mounted source, so a snake_case identifier still matches itself.  It is
    NOT safe to assume snake_case survives the projection intact -- if a
    future value class needs that, the emphasis rule has to be narrowed in
    ``normalize_value_text`` FIRST, and both callers move together.
    """
    assert normalize_glyphs("cobalt_1_india") == "cobalt1india"
    assert normalize_glyphs("cobalt_1_india") == normalize_value_text(
        "cobalt_1_india")
    # Symmetry is what keeps grounding correct under it.
    arena = _stub_arena(["The registry value is cobalt_1_india."])
    grounded, _c = arena._grounding_verdict(
        "The value is cobalt_1_india.", [0],
        "What is the registry value?", normalized=True)
    assert grounded is True


def test_space_separated_value_does_not_ground_against_hyphenated_source():
    """The counterexample the order names, pinned as still-REJECTED.

    If the separator class were folded, "Cobalt 1 India" would ground
    against "Cobalt-1-India" -- and worse, against any text merely
    containing "cobalt", "1" and "india" separately.
    """
    arena = _stub_arena(["The Cobalt registry value is Cobalt-1-India."])
    grounded, _c = arena._grounding_verdict(
        "The value is Cobalt 1 India.", [0],
        "What is the Cobalt registry value?", normalized=True)
    assert grounded is False


# --------------------------------------------------------------------------
# 3. The switch: ON normalizes, OFF is byte-identical legacy.
# --------------------------------------------------------------------------

_U2011_ANSWER = "The current Praxis dock value is **Quartz‑8‑Jade**."
_ASCII_SOURCE = "The current Praxis dock value is Quartz-8-Jade."


def test_rare_tokens_switch_on_normalizes(monkeypatch):
    monkeypatch.setenv("GRM_LSR_FIXES", "1")
    assert ArenaCache._rare_tokens(_U2011_ANSWER) == {"quartz-8-jade"}


def test_rare_tokens_switch_off_is_legacy(monkeypatch):
    """The measured legacy shattering: "Quartz‑8‑Jade" -> {"8"}."""
    monkeypatch.setenv("GRM_LSR_FIXES", "0")
    assert ArenaCache._rare_tokens(_U2011_ANSWER) == {"8"}


def test_caps_tokens_switch_on_and_off(monkeypatch):
    text = "Recorded: **Nacre‑6‑Blue** for Harbor."
    monkeypatch.setenv("GRM_LSR_FIXES", "0")
    off = ArenaCache._caps_tokens(text)
    monkeypatch.setenv("GRM_LSR_FIXES", "1")
    on = ArenaCache._caps_tokens(text)
    assert "nacre-6-blue" not in off
    assert "nacre-6-blue" in on
    # The case-bearing channel survives BOTH ways: normalize_glyphs never
    # casefolds, so "Harbor" is still a proper noun on the ON path.
    assert "harbor" in off and "harbor" in on


@pytest.mark.parametrize("token", ("0", "false", "no", "off"))
def test_switch_off_tokens_all_take_the_legacy_path(monkeypatch, token):
    monkeypatch.setenv("GRM_LSR_FIXES", token)
    assert ArenaCache._rare_tokens(_U2011_ANSWER) == {"8"}


def test_switch_fails_closed_to_on_for_an_unknown_token(monkeypatch):
    monkeypatch.setenv("GRM_LSR_FIXES", "wobble")
    assert ArenaCache._rare_tokens(_U2011_ANSWER) == {"quartz-8-jade"}


def test_explicit_override_ignores_the_env_in_both_directions(monkeypatch):
    """The block-scoped override the grounding receipt needs.

    The legacy counterfactual must be computable WITH the switch ON, or the
    receipt would always report "not rescued".  It is carried by
    ``_glyph_norm``, NOT by a widened tokenizer signature: ``_rare_tokens``
    and ``_caps_tokens`` are one-argument stub seams across the suite.
    """
    monkeypatch.setenv("GRM_LSR_FIXES", "1")
    with ArenaCache._glyph_norm(False):
        assert ArenaCache._rare_tokens(_U2011_ANSWER) == {"8"}
    monkeypatch.setenv("GRM_LSR_FIXES", "0")
    with ArenaCache._glyph_norm(True):
        assert ArenaCache._rare_tokens(_U2011_ANSWER) == {"quartz-8-jade"}


def test_tokenizer_stub_seams_keep_their_one_argument_signature():
    """Seven fixtures replace these with one-argument lambdas. Widening the
    signature to carry the glyph flag would break every one of them, which is
    why the flag rides on ``_glyph_norm`` instead."""
    import inspect

    assert list(inspect.signature(ArenaCache._rare_tokens).parameters) == [
        "text"]
    assert list(inspect.signature(ArenaCache._caps_tokens).parameters) == [
        "text", "skip_sentence_initial"]


def test_glyph_norm_restores_the_previous_value_even_on_exception(
        monkeypatch):
    monkeypatch.setenv("GRM_LSR_FIXES", "1")
    assert ArenaCache._glyph_norm_override is None
    with pytest.raises(RuntimeError):
        with ArenaCache._glyph_norm(False):
            raise RuntimeError("boom")
    assert ArenaCache._glyph_norm_override is None
    assert ArenaCache._rare_tokens(_U2011_ANSWER) == {"quartz-8-jade"}


def test_glyph_norm_nests_and_unwinds_in_order(monkeypatch):
    monkeypatch.setenv("GRM_LSR_FIXES", "1")
    with ArenaCache._glyph_norm(False):
        assert ArenaCache._rare_tokens(_U2011_ANSWER) == {"8"}
        with ArenaCache._glyph_norm(True):
            assert ArenaCache._rare_tokens(_U2011_ANSWER) == {"quartz-8-jade"}
        assert ArenaCache._rare_tokens(_U2011_ANSWER) == {"8"}
    assert ArenaCache._rare_tokens(_U2011_ANSWER) == {"quartz-8-jade"}


def test_legacy_path_never_calls_the_normalizer(monkeypatch):
    """OFF is byte-identical: ``normalize_glyphs`` is not reached at all."""
    calls = []

    def _spy(value):
        calls.append(value)
        raise AssertionError("normalize_glyphs must not run on the OFF path")

    monkeypatch.setenv("GRM_LSR_FIXES", "0")
    monkeypatch.setattr("core.graft_arena.normalize_glyphs", _spy)
    assert ArenaCache._rare_tokens(_U2011_ANSWER) == {"8"}
    # The emphasis-wrapped value is NOT a caps token on the legacy path
    # (the "**" prefix fails ``^[A-Z][\\w\\-]+$``); "Praxis" is, and always
    # was -- the point of the spy is that no projection ran at all.
    assert ArenaCache._caps_tokens(_U2011_ANSWER) == {"praxis"}
    assert calls == []


# --------------------------------------------------------------------------
# 4. Grounding: the measured SC1 case, and the receipts.
# --------------------------------------------------------------------------

class _StubArena(ArenaCache):
    """Grounding-only arena: no model, no cache, no forward.

    ``_grounding_verdict`` reads exactly two things off ``self`` besides the
    class-level token machinery -- ``HEDGES`` and ``self.grafts`` -- so a
    grounding test needs nothing else, and building nothing else keeps this
    test off the GPU.
    """

    def __init__(self, texts):
        self.grafts = [{"text": str(t)} for t in texts]


def _stub_arena(texts):
    return _StubArena(texts)


def test_sc1_measured_case_grounds_on_and_fails_off():
    """The exact SC1 G3 blocker, from the receipts.

    ``artifacts/grm_sc1/sc1_g3_summary_ec63a419031110bd.json`` recorded
    ``recovery_blocked_by_grounding_only = 2``: the trip fetched the node,
    the model said the right value, grounding said no.
    """
    arena = _stub_arena([_ASCII_SOURCE])
    question = "What is the current Praxis dock value?"
    legacy, _lc = arena._grounding_verdict(
        _U2011_ANSWER, [0], question, normalized=False)
    normalized, _nc = arena._grounding_verdict(
        _U2011_ANSWER, [0], question, normalized=True)
    assert legacy is False
    assert normalized is True


def test_grounding_receipts_report_the_rescue(monkeypatch):
    monkeypatch.setenv("GRM_LSR_FIXES", "1")
    arena = _stub_arena([_ASCII_SOURCE])
    info: dict = {}
    question = "What is the current Praxis dock value?"
    grounded, _c = arena._grounding_attribution(_U2011_ANSWER, [0], question)
    arena._grounding_receipt(_U2011_ANSWER, [0], question, info)
    assert grounded is True
    assert info["grounding_normalized"] is True
    assert info["grounding_glyph_rescued"] is True


def test_grounding_receipts_do_not_claim_a_rescue_without_one(monkeypatch):
    """An answer that grounds under BOTH paths is not "rescued"."""
    monkeypatch.setenv("GRM_LSR_FIXES", "1")
    arena = _stub_arena([_ASCII_SOURCE])
    info: dict = {}
    ans = "The current Praxis dock value is Quartz-8-Jade."
    question = "What is the current Praxis dock value?"
    grounded, _c = arena._grounding_attribution(ans, [0], question)
    arena._grounding_receipt(ans, [0], question, info)
    assert grounded is True
    assert info["grounding_normalized"] is True
    assert info["grounding_glyph_rescued"] is False


def test_grounding_receipts_on_the_off_path(monkeypatch):
    monkeypatch.setenv("GRM_LSR_FIXES", "0")
    arena = _stub_arena([_ASCII_SOURCE])
    info: dict = {}
    question = "What is the current Praxis dock value?"
    grounded, _c = arena._grounding_attribution(_U2011_ANSWER, [0], question)
    arena._grounding_receipt(_U2011_ANSWER, [0], question, info)
    assert grounded is False
    assert info["grounding_normalized"] is False
    # Never claims a rescue it did not perform.
    assert info["grounding_glyph_rescued"] is False


def test_grounding_attribution_keeps_its_three_argument_signature(monkeypatch):
    """The stub seam. Fixtures across the suite replace
    ``_grounding_attribution`` with a THREE-argument callable; the receipt
    therefore lives in ``_grounding_receipt``, never in that signature."""
    import inspect

    params = list(inspect.signature(
        ArenaCache._grounding_attribution).parameters)
    assert params == ["self", "ans", "mount_idxs", "question"]
    monkeypatch.setenv("GRM_LSR_FIXES", "1")
    arena = _stub_arena([_ASCII_SOURCE])
    assert arena._grounded(
        _U2011_ANSWER, [0], "What is the current Praxis dock value?") is True


def test_grounding_receipt_with_no_info_mapping_is_a_no_op():
    arena = _stub_arena([_ASCII_SOURCE])
    assert arena._grounding_receipt(
        _U2011_ANSWER, [0], "What is the current Praxis dock value?",
        None) is None


def test_hedged_answer_is_still_ungrounded_on_both_paths(monkeypatch):
    """The projection widens token matching; it does not touch the hedge
    branch, which is the gate that keeps a refusal from grounding."""
    arena = _stub_arena([_ASCII_SOURCE])
    hedged = "I don't have that information."
    for normalized in (False, True):
        grounded, _c = arena._grounding_verdict(
            hedged, [0], "What is the current Praxis dock value?",
            normalized=normalized)
        assert grounded is False


def test_identifier_gate_still_rejects_the_wrong_sibling():
    """The ``qrare & mounted`` gate is unchanged by the projection: a
    question naming a code no mounted source carries stays ungrounded."""
    arena = _stub_arena(["The Harbor token value is Nacre-6-Blue."])
    grounded, _c = arena._grounding_verdict(
        "The value is Quartz-8-Jade.", [0],
        "What is the value for BX-44?", normalized=True)
    assert grounded is False


def test_driver_stamp_is_a_no_op_on_an_arena_that_cannot_produce_one():
    """The driver is called with FAKE arenas by several fixtures. A receipt
    is a receipt: an arena that cannot compute it gets no fields, and
    NOTHING about serving changes."""
    from scripts.grm_e2e_session import _stamp_grounding_receipt

    class _NoReceipt:
        pass

    info: dict = {}
    _stamp_grounding_receipt(_NoReceipt(), "a", [0], "q", info)
    assert info == {}


def test_driver_stamp_delegates_to_a_real_arena(monkeypatch):
    from scripts.grm_e2e_session import _stamp_grounding_receipt

    monkeypatch.setenv("GRM_LSR_FIXES", "1")
    arena = _stub_arena([_ASCII_SOURCE])
    info: dict = {}
    _stamp_grounding_receipt(
        arena, _U2011_ANSWER, [0],
        "What is the current Praxis dock value?", info)
    assert info == {
        "grounding_normalized": True,
        "grounding_glyph_rescued": True,
    }


# --------------------------------------------------------------------------
# 5. The route-receipt prefix (the one edit to core/grm_three_pass.py).
# --------------------------------------------------------------------------

def test_grounding_prefix_is_registered_and_persists_the_block():
    assert "grounding_" in ROUTE_RECEIPT_INFO_PREFIXES
    out = _route_receipt_generic_info({
        "grounding_normalized": True,
        "grounding_glyph_rescued": True,
        "not_a_receipt_field": 1,
    })
    assert out == {
        "grounding_normalized": True,
        "grounding_glyph_rescued": True,
    }
