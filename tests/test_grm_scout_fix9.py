"""SCOUT-FIX-9 treatment pins: degenerate width-guard children and the
one-rule route-key/eligibility contract.

Prior art: FIX-3/FIX-5 fold QC and coverage vocabulary, and the C7 CPU
numerical doubles (GRM contributors, 2026) — reused verbatim as the fact
predicate and the repository harness. Cycle-safe DFS with a visited set is
textbook (Tarjan 1972). What is ours: reading the P1 r2 receipt's split
family as a CYCLE in the descent-key walk, and pinning the native/Python
key-set unity that `decisive_admission_profile`'s integrity guard compares.
No prior art known to me for this exact defect pair; unverified against the
wider literature (no network in this sandbox) — lead to check. CPU author
evidence only: no GPU, no model-quality claim.
"""
import json
from pathlib import Path

import pytest

from scripts.grm_c7_diagnose import repository

ROOT = Path(__file__).resolve().parents[1]

# The receipt's shape: a fact-less "ARCHIVE NOTE." header section in front of
# a fact-bearing body, sized one token over the width budget.
RECEIPT_TEXT = (
    "ARCHIVE NOTE.\n\nThe facts to archive are: Hull-1 is a silver alloy; "
    "Harbor-8-Golf is the current terra port value; Auric-4-Alpha is the "
    "current Orion pin; Nadir-1-Delta is the current Hub value.")

# Same facts, no fact-less section: every child of this split carries facts.
LEGITIMATE_TEXT = (
    "Hull-1 is a silver alloy and Harbor-8-Golf is the current terra "
    "port value for the docking section of the ship.\n\n"
    "Auric-4-Alpha is the current Orion pin and Nadir-1-Delta is the "
    "current Hub value for the bridge section of the ship.")

# A trailing bare-newline token makes `_decode_token_span` fall through to its
# whole-parent placeholder for the one-token tail span.
PLACEHOLDER_TEXT = (
    "ARCHIVE NOTE.\nThe facts to archive are: Hull-1 is a silver alloy; "
    "Harbor-8-Golf is the current terra port value; Auric-4-Alpha is the "
    "current Orion pin; Nadir-1-Delta is the current Hub value.\n")


def era(repo, text):
    a = repo.arena
    idx = a.deposit(text)
    a.grafts[idx]['kind'] = 'era'
    a.grafts[idx].setdefault('rare', a._rare_tokens(text))
    a.grafts[idx].setdefault('metadata', repo._default_metadata(a.grafts[idx]))
    repo._ensure_lifecycle(idx, a.grafts[idx])
    return idx


@pytest.mark.parametrize('text,reason,cull_index', [
    (RECEIPT_TEXT, 'degenerate_child_no_parent_facts', 0),
    (PLACEHOLDER_TEXT, 'degenerate_child_whole_parent_text', 1),
])
def test_degenerate_split_rejected_sources_stay_active(
        tmp_path, monkeypatch, text, reason, cull_index):
    """Defect 1: a fact-less child is never deposited; the parent survives."""
    repo = repository(tmp_path / 'repo', monkeypatch)
    try:
        a = repo.arena
        idx = era(repo, text)
        budget = int(a.grafts[idx]['ntok']) - 1
        # The split PLAN exists — this is a QC rejection, not "nothing to cut".
        assert len(repo._width_guard_spans(idx, budget)) == 2
        assert repo._guard_deposit_width(idx, budget=budget) is None
        receipt = repo.last_width_guard_rejection
        assert receipt['action'] == 'width_guard_rejected'
        assert receipt['reason'] == reason
        assert receipt['cull_index'] == cull_index
        assert receipt['parent'] == idx
        assert receipt['parent_fact_count'] == 4
        # Sources stay active: no children, parent unretired and unre-classed.
        assert len(a.grafts) == 1
        assert not a.grafts[idx].get('retired')
        assert a.grafts[idx]['kind'] == 'era'
        assert not (a.grafts[idx].get('metadata') or {}).get('width_guard_parent')
        assert not a.grafts[idx].get('child_cents')
    finally:
        repo.close()


def test_legitimate_fold_split_unchanged(tmp_path, monkeypatch):
    """Defect 1 negative control: a split whose children all carry facts."""
    repo = repository(tmp_path / 'repo', monkeypatch)
    try:
        a = repo.arena
        idx = era(repo, LEGITIMATE_TEXT)
        budget = int(a.grafts[idx]['ntok']) - 1
        out = repo._guard_deposit_width(idx, budget=budget)
        assert out is not None and out['action'] == 'width_guard_split'
        assert repo.last_width_guard_rejection is None
        children = list(out['children'])
        assert len(children) == 2
        need = a._fact_set([LEGITIMATE_TEXT])
        for child in children:
            assert a._fact_set([a.grafts[child]['text']]) & need
            assert int(a.grafts[child]['ntok']) <= budget
        assert (a.grafts[idx].get('metadata') or {})['width_guard_parent']
        assert a.grafts[idx]['sources'] == children
    finally:
        repo.close()


def test_descent_keys_are_cycle_safe(tmp_path, monkeypatch):
    """Defect 2 (cause 1): the parent<->child edge must not re-enter.

    The width guard points the index parent AT its children while each child
    points back at the parent. Before the fix the descent-key walk followed
    that cycle, so every split-family member inherited the family's maximal
    key and a 4-token header routed as its 97-token parent — the exact block
    the P1 r2 integrity guard fired on.
    """
    repo = repository(tmp_path / 'repo', monkeypatch)
    try:
        a = repo.arena
        idx = era(repo, LEGITIMATE_TEXT)
        budget = int(a.grafts[idx]['ntok']) - 1
        children = list(repo._guard_deposit_width(idx, budget=budget)['children'])
        assert a.grafts[idx]['sources'] == children
        for child in children:
            assert a.grafts[child]['sources'] == [idx]
        repo._rebuild_child_keys()
        # Cycle closed: no node inherits a key more than once, and no node
        # inherits its own key through the parent edge.
        for node in (idx, *children):
            keys = a.grafts[node].get('child_cents') or ()
            assert len(keys) == len({id(k) for k in keys})
        # A child's descent set is bounded by the family, not the cycle's
        # unbounded re-entry (12/10/10 rows before the fix, 3 after).
        for child in children:
            assert len(a.grafts[child].get('child_cents') or ()) <= len(children) + 1
    finally:
        repo.close()


def test_keyless_node_is_ineligible_in_both_paths(tmp_path, monkeypatch):
    """Defect 2 (lead ruling): `cent is None` is not a route candidate.

    The native router drops a keyless entry outright (it never sets `have[]`
    in `route_gqa_raw`), so admitting it on the Python side gave the
    integrity guard two different candidate sets to compare.
    """
    repo = repository(tmp_path / 'repo', monkeypatch)
    try:
        a = repo.arena
        keyed = a.deposit('Hull-1 is a silver alloy in the docking section.')
        keyless = a.deposit('Harbor-8-Golf is the current terra port value.')
        a.grafts[keyless]['cent'] = None
        a._bump_cuda_gqa_epoch()
        base = list(a._route_cand_base())
        assert keyed in base
        assert keyless not in base
        # And it stays out of the router's own answer.
        assert keyless not in list(
            a.route('What is the terra port value?', exclude=(), limit=5) or ())
    finally:
        repo.close()


def test_c2_recorded_plans_byte_identical():
    """The FIX-6/FIX-8 replay gate: 132 recorded A-DEC plans must not move."""
    from scripts.grm_scout_fix8_cpu import c2_plans
    rows = c2_plans()
    assert len(rows) == 132
    assert sum(r['identical'] for r in rows) == 132
    frozen = json.loads(
        (ROOT / 'artifacts/grm_scout_fix8/green.json').read_text())
    assert frozen['c2_identical'] == 132
    assert rows == frozen['c2']


def test_receipt_registration_present():
    """The replay the lead runs is registered, not run here (no GPU)."""
    reg = json.loads(
        (ROOT / 'artifacts/grm_scout_fix9/registration.json').read_text())
    assert reg['status'] == 'NOT_MEASURED'
    assert reg['gpu_executed'] is False
    assert reg['profile'] == 'eb1_c2'
    assert reg['gpu_cap_hours'] <= 0.1
    assert reg['prediction']['raises'] is False
