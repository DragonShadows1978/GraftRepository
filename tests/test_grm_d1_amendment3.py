"""GRM-D1 follow-up 3: gates for LT1.1 amendment 3.

Amendment 3 rebinds the runner after the seam fix and records both the defect
and the open host blocker. Every claim it makes must be checkable on CPU.

Prior art: the SHA-chained amendment contract is LT1's / C7 r3's. Full
annotation in scripts/grm_d1_amendment3.py.
"""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import grm_d1_amendment3 as am3  # noqa: E402


@pytest.fixture(scope='module')
def doc():
    return am3.amendment()


# ------------------------------------------------------------------- chain

def test_amendment3_chains_to_amendment2(doc):
    assert doc['previous_amendment_sha256'] == am3.sha_path(am3.AMENDMENT2)
    recorded = (am3.OUT / 'amendment2.sha256').read_text().split()[0]
    assert doc['previous_amendment_sha256'] == recorded
    assert doc['schema'] == 'grm.lt1_1.amendment3.v1'
    assert doc['amendment'] == 3
    assert doc['order_sha256'] == am3.sha_path(am3.ORDER)


def test_amendment3_refuses_a_drifted_parent(monkeypatch, tmp_path):
    """RED proof for the chain check."""
    forged = tmp_path / 'amendment2.json'
    forged.write_text(json.dumps({'registration_sha256': 'x'}))
    (tmp_path / 'amendment2.sha256').write_text('deadbeef  amendment2.json\n')
    monkeypatch.setattr(am3, 'AMENDMENT2', forged)
    monkeypatch.setattr(am3, 'OUT', tmp_path)
    with pytest.raises(ValueError, match='LT11_AMENDMENT2_SHA_MISMATCH'):
        am3.amendment()


def test_amendment3_bytes_are_deterministic():
    assert am3.canonical(am3.amendment()) == am3.canonical(am3.amendment())


def test_written_amendment3_matches_its_sidecar():
    path = am3.OUT / 'amendment3.json'
    if not path.exists():
        pytest.skip('amendment 3 not emitted yet')
    sidecar = (am3.OUT / 'amendment3.sha256').read_text().split()[0]
    assert am3.sha_path(path) == sidecar


# ------------------------------------------------------------- the rebind

def test_the_runner_is_rebound_to_the_fixed_file(doc):
    runner = doc['runner']
    assert runner['sha256'] == am3.sha_path(ROOT / runner['path'])
    assert runner['sha256'] != runner['superseded_sha256'], (
        'the runner did not change; amendment 3 would be a no-op')
    parent = json.loads(am3.AMENDMENT2.read_text())
    assert runner['superseded_sha256'] == parent['runner']['sha256']


def test_both_runner_test_files_are_bound(doc):
    for name, digest in doc['runner']['tests_sha256'].items():
        assert (ROOT / name).is_file()
        assert digest == am3.sha_path(ROOT / name)
    assert 'tests/test_grm_lt1_1_resume_route.py' in doc['runner']['tests']


def test_the_new_entry_point_actually_parses(doc):
    from scripts import grm_lt1_1 as r
    assert '--resume --dry-lease' in doc['runner']['entry_points']
    args = r.parse_args(['--arm', 'A+', '--resume', '--dry-lease',
                         '--no-host-gate'])
    assert args.resume and args.dry_lease and args.no_host_gate


# ------------------------------------------------------------- the defect

def test_the_defect_record_names_all_three_causes(doc):
    causes = {c['id']: c for c in doc['defect']['causes']}
    assert set(causes) == {'D3-C1', 'D3-C2', 'D3-C3'}
    assert causes['D3-C1']['kind'] == 'signature drift'
    assert causes['D3-C2']['kind'] == 'seam too wide'
    assert causes['D3-C3']['kind'] == 'ordering'
    for cause in causes.values():
        assert cause['what'] and cause['fix']
    assert "KeyError: 'CPU'" in doc['defect']['symptom']
    assert 'grm_lt1_amendment4.apply' in doc['defect']['why_the_gates_missed_it']


def test_every_fix_the_defect_record_claims_is_in_the_source(doc):
    """The record must describe the code that exists, not an intention."""
    source = (ROOT / 'scripts/grm_lt1_1.py').read_text()
    assert 'def binding(label):' in source            # D3-C1
    assert 'def current_arm(' in source               # D3-C1
    assert 'worker.bind = binding' in source          # D3-C2
    assert 'lt.binding = binding' not in source       # D3-C2
    assert 'def host_preflight(' in source            # D3-C3


# --------------------------------------------------------- the seam audit

def test_the_seam_audit_lists_every_moved_and_unmoved_name(doc):
    audit = doc['seam_audit']
    assert set(audit['redirected']) == {'lt.FIX', 'worker.bind', 'worker.RUN'}
    assert set(audit['deliberately_not_redirected']) == {
        'lt.binding', 'lt.RUN', 'lt.REG'}
    for reason in audit['deliberately_not_redirected'].values():
        assert reason


def test_the_seam_audit_matches_what_the_runner_actually_moves(doc):
    """Cross-check the document against live behaviour."""
    from scripts import grm_lt1 as lt
    from scripts import grm_lt1_worker as worker
    from scripts import grm_lt1_1 as r
    before = (lt.FIX, lt.binding, worker.bind, worker.RUN, lt.RUN, lt.REG)
    with r.lt1_1_seams('A+'):
        moved = {
            'lt.FIX': lt.FIX != before[0],
            'worker.bind': worker.bind is not before[2],
            'worker.RUN': worker.RUN != before[3],
        }
        unmoved = {
            'lt.binding': lt.binding is before[1],
            'lt.RUN': lt.RUN == before[4],
            'lt.REG': lt.REG == before[5],
        }
    assert all(moved.values()), moved
    assert all(unmoved.values()), unmoved
    assert set(moved) == set(doc['seam_audit']['redirected'])
    assert set(unmoved) == set(doc['seam_audit']['deliberately_not_redirected'])


# ---------------------------------------------------------------- the gate

def test_the_gate_record_points_at_tests_that_exist(doc):
    gate = doc['gate']
    assert (ROOT / gate['tests']).is_file()
    source = (ROOT / gate['tests']).read_text()
    for name in ('test_the_shipped_seam_reproduces_the_leads_keyerror',
                 'test_resume_route_reaches_the_lease_boundary',
                 'test_only_the_documented_seams_move'):
        assert 'def %s' % name in source, name
    assert 'NOT a language-model quality measurement' in gate['evidence_class']


# ------------------------------------------------------- the host blocker

def test_the_open_host_blocker_is_recorded_not_hidden(doc):
    blocker = doc['host_blocker']
    assert blocker['status'].startswith('OPEN')
    assert 'INPUT_SHA_MISMATCH' in blocker['symptom']
    assert blocker['lead_decision_required'] is True
    assert 'pre-existing' in blocker['status']


def test_the_host_blocker_claim_was_true_and_is_now_resolved():
    """INVERTED by the lead's ruling of 2026-09-11, with the receipt.

    This test used to be `test_the_host_blocker_claim_is_true_right_now` and
    asserted that `r.host_preflight()` raised `INPUT_SHA_MISMATCH`. The ruling
    resolved that: LT1's registration is a frozen receipt, so LT1.1 validates
    its OWN chain. Both halves are still asserted, because the honest claim is
    not "the drift went away" -- it did not:

      1. the underlying LT1 condition is UNCHANGED (its gate still refuses);
      2. LT1.1's gate, which is the one that governs, is READY.

    Amendment 3's recorded blocker therefore stands as a true statement about
    its day, and amendment 4 records its resolution.
    """
    from scripts import grm_lt1 as lt
    from scripts import grm_lt1_1 as r
    # 1. Untouched: we did not paper over the drift.
    with pytest.raises(ValueError, match='INPUT_SHA_MISMATCH'):
        lt.preflight()
    # 2. Resolved: the governing gate passes.
    with r.pinned_arm('A+'):
        gate = r.host_preflight('A+')
    assert gate['status'] == 'READY', gate['reasons']
    assert not any('INPUT_SHA_MISMATCH' in reason for reason in gate['reasons'])


def test_amendment3_blocker_is_superseded_by_amendment4(doc):
    """Amendment 3 stays as written; amendment 4 records the resolution."""
    assert doc['host_blocker']['status'].startswith('OPEN')
    amendment4 = am3.OUT / 'amendment4.json'
    if not amendment4.exists():
        pytest.skip('amendment 4 not emitted yet')
    later = json.loads(amendment4.read_text())
    assert later['previous_amendment_sha256'] == am3.sha_path(
        am3.OUT / 'amendment3.json')
    assert later['resolves']['status'] == 'RESOLVED by ruling'
    assert 'INPUT_SHA_MISMATCH' in later['resolves']['blocker']


def test_the_blocker_does_not_stop_the_route_gate(tmp_path):
    """The LT1.1 route is green even while the host gate is red."""
    from scripts import grm_lt1_1 as r
    value = r.resume('A+', root=tmp_path, dry_lease=True, host_gate=False)
    assert value['status'] == 'PASS'
    assert value['stopped_at'] == 'worker.run_cell (lease boundary)'


# ------------------------------------------------------ carried-forward

def test_budget_and_predictions_carry_forward_unchanged(doc):
    parent = json.loads(am3.AMENDMENT2.read_text())
    assert doc['budget_gpu_seconds_per_arm'] == parent['budget_gpu_seconds_per_arm']
    assert doc['budget_gpu_hours_total'] == parent['budget_gpu_hours_total']
    for arm in ('A', 'A+'):
        assert doc['arms'][arm]['predictions'] == parent['arms'][arm]['predictions']


def test_process_safety_names_the_dry_lease_as_lease_free(doc):
    gpu = doc['process_safety']['gpu']
    assert '--dry-lease' in gpu
    assert 'never do' in gpu
