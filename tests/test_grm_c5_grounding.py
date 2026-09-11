"""Prior art: SC1.1 CPU text arena and Scout C5 adversaries (GRM, 2026).
Borrowed: independent corpus, original predicate as reference, default pins.
Ours: registered four-arm default/parity and narrow-grammar checks.
"""
import inspect
import json

import pytest

from tests.conftest import campaign_receipt_module

from scripts.grm_c5_offline import load_fixtures, mutation_checks
from scripts.grm_c5_rules import TextArena, binding_candidates, grounding_verdict

# GRM-H2: WHOLE-MODULE campaign receipt, declared at import.
#
# ``load_fixtures()`` below calls ``scripts.grm_c5_offline.registration()``,
# which verifies C5's frozen registration against the ABSOLUTE paths it was
# registered at -- ``/mnt/ForgeRealm/wt/grm-c5/artifacts/grm_c5/fixtures.json``
# and the gitignored EB1 session artifacts under
# ``/mnt/ForgeRealm/GraftRepository/artifacts/``.  That is the worktree-path
# receipt class (docs/TESTS_CAMPAIGN_RECEIPTS.md): it cannot pass on any tree
# but grm-c5, by construction.
#
# It is declared at MODULE level rather than per function because the check
# runs at IMPORT time -- ``FIXTURES`` is the argument list of the
# ``parametrize`` decorators below, so it cannot be deferred into a fixture
# without changing what the campaign registered.  Left unguarded it does not
# merely fail: it aborts collection of the entire tree-wide run
# ("Interrupted: 1 error during collection"), and `-m campaign_receipt` aborts
# the same way, so the receipt gate could not be run either.
#
# The skip is therefore UNCONDITIONAL -- there is no invocation in which this
# import can succeed off the grm-c5 tree.  The receipt is not swallowed by it:
# tests/test_grm_c5_grounding_receipt.py carries one campaign_receipt-marked
# test that calls the same load_fixtures() binding, so the failure stays
# collectable and reproduces under -m campaign_receipt like every other
# receipt.  No assertion in this file was changed; only whether it is
# collected.
campaign_receipt_module(
    registration='artifacts/grm_c5/registration.json (pins absolute '
                 '/mnt/ForgeRealm/wt/grm-c5/ and gitignored '
                 '/mnt/ForgeRealm/GraftRepository/artifacts/grm_eb1/ inputs)',
    reason='scripts.grm_c5_offline.load_fixtures() verifies C5 registration '
           'at import time')

FIXTURES = load_fixtures()
SCORABLE = [row for row in FIXTURES if row["texts"] is not None]


def test_default_off_pin():
    assert inspect.signature(grounding_verdict).parameters["rule"].default == "0"
    assert inspect.signature(binding_candidates).parameters["rule"].default == "0"


@pytest.mark.parametrize("row", SCORABLE, ids=lambda row: row["id"])
@pytest.mark.parametrize("normalized", [True, False])
def test_default_matches_unmodified_grounding(row, normalized, monkeypatch):
    monkeypatch.setenv("GRM_C5_RULE", "W")  # no environmental opt-in exists
    expected = TextArena(row["texts"])._grounding_verdict(
        row["answer"], list(range(len(row["texts"]))), row["question"], normalized=normalized)
    assert grounding_verdict(row["answer"], row["texts"], row["question"], normalized=normalized) == expected
    assert grounding_verdict(row["answer"], row["texts"], row["question"], rule="0", normalized=normalized) == expected


@pytest.mark.parametrize("arm", ["S", "N"])
@pytest.mark.parametrize("row", [r for r in FIXTURES if r["class"] in ("S", "N")], ids=lambda r: r["id"])
def test_no_new_control_acceptances(row, arm):
    original = grounding_verdict(row["answer"], row["texts"], row["question"])[0]
    candidate = grounding_verdict(row["answer"], row["texts"], row["question"], rule=arm)[0]
    assert not candidate or original


def test_frozen_contract_checks():
    import scripts.grm_c5_rules as rules
    assert mutation_checks(rules) == []


def test_controls_are_nonvacuous_and_originals_present():
    kinds = {"changed_digit", "omitted_token", "swapped_relation", "negation", "scattered_word", "alias_collision"}
    for family in ("S", "N"):
        controls = [r for r in FIXTURES if r["class"] == family]
        assert len(controls) == 10
        assert kinds <= {r["kind"] for r in controls}
    census = [r for r in FIXTURES if r.get("campaign") == "EB1" and r.get("turn", 1000) < 34]
    assert len(census) == 10
    assert sum(r["recorded_correct"] for r in census if r["turn"] not in (30, 33)) == 8


def test_missing_rt1_inputs_are_explicit():
    rows = [r for r in FIXTURES if r.get("campaign") == "RT1"]
    assert len(rows) == 18
    assert all(r["answer"] and r["texts"] is None and r["blocked_reason"] for r in rows)


def test_missing_mounts_cannot_be_replaced_by_proper_name_binding():
    row = next(r for r in FIXTURES if r["id"] == "EB1:t33")
    assert row["infer_calls"] == 0
    assert row["texts"] == []
    assert row["abstain_reason"] == "identifier_unbound"
    assert not grounding_verdict(row["answer"], [], row["question"], rule="N")[0]
    assert binding_candidates(row["question"], [row["source_node_text"]]) == {0}
    assert binding_candidates(row["question"], [row["source_node_text"]], rule="N") == {0}


def test_invalid_arm_is_an_error():
    with pytest.raises(ValueError):
        grounding_verdict("a", [], "q", rule="SN")


def test_no_cross_mount_join_or_injected_clause():
    q = "What is the current atlas tone value?"
    for sources in (["The current atlas tone value is Cobalt-1", "India."],
                    ['Someone said "The current atlas tone value is Cobalt-1-India".'],
                    ["It is false that the current atlas tone value is Cobalt-1-India."]):
        assert not grounding_verdict("Cobalt 1 India", sources, q, rule="S")[0]
