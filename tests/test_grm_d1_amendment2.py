"""GRM-D1 follow-up: gates for LT1.1 amendment 2.

Amendment 2 binds the runner and corrects the budget. Both claims must be
checkable without a GPU: the pinned runner sha must match the file on disk,
and the corrected ceiling must actually cover what the machinery reserves.

Prior art: the SHA-chained amendment contract is LT1's / C7 r3's. Full
annotation in scripts/grm_d1_amendment2.py.
"""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import grm_d1_amendment2 as am2  # noqa: E402


@pytest.fixture(scope='module')
def doc():
    return am2.amendment()


# ------------------------------------------------------------------- chain

def test_amendment2_chains_to_amendment1_and_the_order(doc):
    assert doc['previous_amendment_sha256'] == am2.AMENDMENT1_SHA
    assert am2.sha_path(am2.AMENDMENT1) == am2.AMENDMENT1_SHA
    assert doc['order_sha256'] == am2.sha_path(am2.ORDER)
    assert doc['schema'] == 'grm.lt1_1.amendment2.v1'
    assert doc['amendment'] == 2
    # It inherits, never restates, the registration identity.
    parent = json.loads(am2.AMENDMENT1.read_text())
    assert doc['registration_sha256'] == parent['registration_sha256']


def test_amendment2_refuses_a_drifted_parent(monkeypatch):
    """RED proof for the chain check."""
    monkeypatch.setattr(am2, 'AMENDMENT1_SHA', 'deadbeef' * 8)
    with pytest.raises(ValueError, match='LT11_AMENDMENT1_SHA_MISMATCH'):
        am2.amendment()


def test_amendment2_bytes_are_deterministic():
    assert am2.canonical(am2.amendment()) == am2.canonical(am2.amendment())


def test_written_amendment2_matches_its_sidecar():
    path = am2.OUT / 'amendment2.json'
    if not path.exists():
        pytest.skip('amendment 2 not emitted yet')
    sidecar = (am2.OUT / 'amendment2.sha256').read_text().split()[0]
    assert am2.sha_path(path) == sidecar


# ------------------------------------------------------------- the runner

def test_the_runner_is_bound_and_exists(doc):
    runner = doc['runner']
    assert runner['path'] == 'scripts/grm_lt1_1.py'
    assert (ROOT / runner['path']).is_file()
    assert runner['sha256'] == am2.sha_path(ROOT / runner['path'])
    assert (ROOT / runner['tests']).is_file()
    assert runner['tests_sha256'] == am2.sha_path(ROOT / runner['tests'])


def test_every_bound_entry_point_actually_parses(doc):
    """The whole point of amendment 2: the interface must be real."""
    from scripts import grm_lt1_1 as r
    for flag in doc['runner']['entry_points']:
        if flag in ('--worker', '--fake-cell'):
            args = r.parse_args(['--arm', 'A', flag, 'A-001-008'])
        else:
            args = r.parse_args(['--arm', 'A', flag])
        assert args.arm == 'A'


def test_the_reuse_claim_is_stated_and_true(doc):
    """'Parameterized, not forked' must be checkable, not just asserted."""
    assert 'NOT forked' in doc['runner']['reuse']
    source = (ROOT / 'scripts/grm_lt1_1.py').read_text()
    # The runner CALLS the existing machinery ...
    assert 'worker.execute(' in source
    assert 'worker.run_cell(' in source
    assert 'worker.pending(' in source
    # ... and does not redefine it.
    for banned in ('def execute(', 'def run_cell(', 'def pending('):
        assert banned not in source, 'runner re-implements %s' % banned


def test_the_arm_pin_contract_is_recorded(doc):
    pin = doc['runner']['arm_pin']
    assert 'pin_rule' in pin
    assert 'environment(flags)' in pin
    assert 'read' in pin and 'restored' in pin


# ------------------------------------------------------------- the budget

def test_the_corrected_budget_covers_the_reservation(doc):
    """A ceiling under the reservation sum rails mid-campaign. It must not."""
    parent = json.loads(am2.AMENDMENT1.read_text())
    cells = parent['arms']['A']['cells']
    lease = sum(c['lease_seconds'] for c in cells)
    assert doc['budget_gpu_seconds_per_arm'] == lease
    assert doc['budget_gpu_seconds_per_arm'] >= lease, 'ceiling below reservation'
    for arm in doc['arms'].values():
        assert arm['budget_gpu_seconds'] == lease


def test_the_correction_is_declared_as_an_increase(doc):
    """A budget that went UP must say so; a silent raise is not honest."""
    correction = doc['budget_correction']
    assert correction['superseded_gpu_seconds_per_arm'] == 6120
    assert correction['corrected_gpu_seconds_per_arm'] == 7410
    assert (correction['corrected_gpu_hours_per_arm']
            > correction['superseded_gpu_hours_per_arm'])
    assert 'budget increase' in correction['why']
    assert 'LEAD DECISION' in correction['why']
    assert 'dry-run' in correction['discovered_by']


def test_the_projection_is_reported_beside_the_ceiling(doc):
    """Expected spend and worst-case reservation are different numbers."""
    correction = doc['budget_correction']
    assert correction['projected_seconds'] < correction['reservation_seconds']
    assert correction['projected_gpu_hours'] == 1.25
    assert correction['reservation_gpu_hours'] == 2.06
    assert 'lease_seconds' in correction['basis']


def test_the_dry_run_agrees_with_the_registered_budget():
    """The runner's own report must match what amendment 2 registered."""
    from scripts import grm_lt1_1 as r
    value = r.dry_run('A')
    doc = am2.amendment()
    assert value['lease_seconds'] == doc['budget_gpu_seconds_per_arm']


def test_totals_are_two_arms(doc):
    assert doc['budget_gpu_seconds_total'] == 2 * doc['budget_gpu_seconds_per_arm']
    assert doc['budget_gpu_hours_total'] == 4.12


# --------------------------------------------------------------- the arms

def test_arms_keep_their_distinct_roots_and_flags(doc):
    a, aplus = doc['arms']['A'], doc['arms']['A+']
    assert a['out_dir'] != aplus['out_dir']
    assert a['alias_fold_merge'] is False and aplus['alias_fold_merge'] is True
    env_a = a['env_after_environment_flags']
    env_p = aplus['env_after_environment_flags']
    assert set(env_p) - set(env_a) == {am2.ALIAS_FLAG}


def test_predictions_carry_forward_unchanged(doc):
    """A runner fix must not quietly re-open a registered prediction."""
    parent = json.loads(am2.AMENDMENT1.read_text())
    for name in ('A', 'A+'):
        assert doc['arms'][name]['predictions'] == parent['arms'][name]['predictions']


def test_every_arm_has_a_runnable_invocation_for_each_mode(doc):
    for name, spec in doc['arms'].items():
        for mode in ('preflight', 'dry_run', 'resume', 'summary'):
            command = spec['invocation'][mode]
            assert command.startswith('python3 scripts/grm_lt1_1.py')
            assert '--arm %s' % name in command


# ----------------------------------------------------------------- proof

def test_the_proof_section_names_what_was_actually_run(doc):
    proof = doc['proof']
    for key in ('dry_run', 'fake_execution', 'summary', 'resume',
                'command_gate'):
        assert proof[key]
    assert 'NOT a language-model quality measurement' in proof['evidence_class']
    assert 'test_every_emitted_command_actually_runs' in proof['command_gate']


def test_process_safety_names_which_modes_take_the_lease(doc):
    safety = doc['process_safety']
    assert '--resume' in safety['gpu']
    assert 'never do' in safety['gpu']
    assert 'never run concurrently' in safety['arms']
