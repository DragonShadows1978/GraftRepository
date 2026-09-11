"""GRM-D1 follow-up: gates for the A+ CPU alias check.

These hold the D1-side check of A1's claim to its receipts. The check is a
CPU fake session: it can prove the alias fold FIRES, that the digest NAMES
both entities, that routing ADMITS it and the ladder MOUNTS it. It cannot
prove language-model recall, and the tests say so.

Prior art: A1's `core/grm_alias_fold.py` public surface and the C7/LT1 CPU
doubles, both reused unchanged. Full annotation in
scripts/grm_d1_alias_cpu.py.
"""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import grm_d1_alias_cpu as ac  # noqa: E402


@pytest.fixture(scope='module')
def plus(tmp_path_factory):
    return ac.run(tmp_path_factory.mktemp('aplus'), alias_fold=True)


@pytest.fixture(scope='module')
def base(tmp_path_factory):
    return ac.run(tmp_path_factory.mktemp('a'), alias_fold=False)


# --------------------------------------------------- the pin, R1 contract

def test_the_arm_pin_is_read_back_not_assumed(plus, base):
    assert plus['pinned'] == dict(admission_rule='margin_first',
                                  alias_fold_merge=True)
    assert base['pinned'] == dict(admission_rule='margin_first',
                                  alias_fold_merge=False)


def test_a_failed_alias_pin_is_red(monkeypatch):
    """RED proof: the readback must catch a pin that did not take."""
    import core.grm_alias_fold as af
    monkeypatch.setattr(af, 'alias_fold_enabled', lambda *a, **k: False)
    patch = pytest.MonkeyPatch()
    try:
        with pytest.raises(ValueError, match='D1_ALIAS_PIN_FAILED'):
            ac.pin_arm(patch, alias_fold=True)
    finally:
        patch.undo()


def test_a_failed_rule_pin_is_red(monkeypatch):
    import core.grm_admission as adm
    monkeypatch.setattr(adm, 'admission_rule', lambda *a, **k: 'all_tokens_bind')
    patch = pytest.MonkeyPatch()
    try:
        with pytest.raises(ValueError, match='D1_RULE_PIN_FAILED'):
            ac.pin_arm(patch, alias_fold=True)
    finally:
        patch.undo()


# --------------------------------------------------- flag OFF is inert

def test_flag_off_fires_no_fold_at_all(base):
    """A1's default-OFF contract, observed from D1's side."""
    assert base['merges_executed'] == 0
    assert base['alias_fold_pending_before'] == 0
    assert base['alias_fold_history'] == []
    for row in base['rows']:
        assert row['fold_job_fired'] is False
        assert row['digest_nodes'] == []
        assert row['admitted'] is False


def test_flag_off_reproduces_the_lt1_alias_shape(base):
    """Arm A still mounts the BASE fact alone -- the D1 diagnosis."""
    for row in base['rows']:
        assert len(row['mounted_ids']) == 1
        # The mounted node is the base fact's node, not the alias edge's.
        assert row['mounted_ids'][0] != row['edge_node']


# --------------------------------------------------- flag ON does the work

def test_every_alias_probe_gets_a_fold(plus):
    assert len(plus['rows']) == 10          # 2 entities x 5 distances
    assert plus['merges_executed'] >= 2
    assert plus['alias_fold_pending_after'] == 0
    for row in plus['rows']:
        assert row['fold_job_fired'] is True, row['probe_id']


def test_the_digest_names_both_the_alias_and_the_base(plus):
    """A1's headline claim, checked with A1's OWN identifier projection."""
    from core.grm_alias_fold import identifier_set
    for row in plus['rows']:
        assert row['digest_names_both'] is True
        have = set(row['digest_identifiers'])
        for name in (row['alias'], row['entity']):
            assert identifier_set(name) <= have, (
                '%s: digest does not name %r' % (row['probe_id'], name))


def test_the_digest_is_admitted_and_mounted(plus):
    for row in plus['rows']:
        assert row['admitted'] is True, row['probe_id']
        assert row['digest_mounted'] is True, row['probe_id']
        assert set(row['digest_nodes']) & set(row['mounted_ids'])


def test_the_merge_receipt_names_a_reason_and_a_lineage(plus):
    from core import grm_alias_fold as af
    for row in plus['rows']:
        merged = [d for d in row['fold_decisions']
                  if d.get('reason') == af.REASON_MERGED]
        assert merged, row['probe_id']
        record = merged[0]
        assert record['lineage'] in (af.LINEAGE_EDGE_AND_BASE,
                                     af.LINEAGE_EDGE_ONLY)
        assert record['consolidation']['accepted'] is True
        assert record['consolidation']['best_cov'] == 1.0


def test_the_digest_fits_the_arena_width(plus):
    """RD2's width wall (110 > 96) is what a write-time join has to beat."""
    from core import grm_alias_fold as af
    for row in plus['rows']:
        record = [d for d in row['fold_decisions']
                  if d.get('reason') == af.REASON_MERGED][0]
        width = record['width']
        assert width['arena_width'] == 96
        assert width['fits'] is True
        assert width['digest_tokens'] <= width['arena_width']


def test_on_the_cpu_double_the_served_value_is_correct(plus, base):
    """Lineage + routing + mount are enough for the DOUBLE to answer.

    This is NOT a language-model claim: the reader is a regex stub. It says
    the right text reached the mount, which is the necessary condition the
    GPU run then tests for sufficiency.
    """
    assert all(r['served_correct'] for r in plus['rows'])
    assert not any(r['served_correct'] for r in base['rows'])


def test_the_two_arms_differ_only_in_the_flag(plus, base):
    assert {r['probe_id'] for r in plus['rows']} == {
        r['probe_id'] for r in base['rows']}
    assert plus['turns_replayed'] == base['turns_replayed']
    assert plus['pinned']['admission_rule'] == base['pinned']['admission_rule']
    assert (plus['pinned']['alias_fold_merge']
            != base['pinned']['alias_fold_merge'])


def test_evidence_class_refuses_a_model_quality_claim(plus):
    assert 'NOT a language-model quality measurement' in plus['evidence_class']


def test_emitted_receipt_matches_a_fresh_run(plus):
    path = ac.OUT / 'alias_cpu_aplus.json'
    if not path.exists():
        pytest.skip('receipt not emitted yet; run scripts/grm_d1_alias_cpu.py')
    saved = json.loads(path.read_text())['A+']
    assert saved['merges_executed'] == plus['merges_executed']
    assert [r['probe_id'] for r in saved['rows']] == [
        r['probe_id'] for r in plus['rows']]
    assert all(r['admitted'] for r in saved['rows'])
