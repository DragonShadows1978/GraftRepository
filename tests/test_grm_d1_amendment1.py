"""GRM-D1 follow-up: gates for LT1.1 amendment 1.

The amendment makes three claims that must be checkable without a GPU:
the base registration is untouched, every core input that differs from the
LT1 run is named and correctly attributed, and the two arms differ by exactly
one environment variable.

Prior art: the SHA-chained amendment contract is LT1's / C7 r3's; the
pin-then-read-back contract is R1's `pin_rule`. Full annotation in
scripts/grm_d1_amendment1.py.
"""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import grm_d1_amendment1 as am  # noqa: E402


@pytest.fixture(scope='module')
def doc():
    return am.amendment()


# ----------------------------------------------------------- the chain

def test_base_registration_is_untouched_at_its_sha():
    """The amendment exists precisely so the base need not be rewritten."""
    assert am.sha_path(am.BASE_REGISTRATION) == am.BASE_SHA
    sidecar = (am.OUT / 'registration.sha256').read_text().split()[0]
    assert sidecar == am.BASE_SHA


def test_amendment_binds_to_the_registration_and_the_order(doc):
    assert doc['registration_sha256'] == am.BASE_SHA
    assert doc['order'] == 'orders/GRM_D1_LT1_RESIDUAL_DIAGNOSIS.md'
    assert doc['order_sha256'] == am.sha_path(am.ORDER)
    assert len(doc['order_sha256']) == 64
    assert doc['schema'] == 'grm.lt1_1.amendment1.v1'
    assert doc['amendment'] == 1
    # The follow-up that produced it is recorded, since it was not a file.
    assert 'GRM_ALIAS_FOLD_MERGE=1' in doc['follow_up_instruction']


def test_amendment_refuses_a_drifted_base(monkeypatch):
    """RED proof for the chain check."""
    monkeypatch.setattr(am, 'BASE_SHA', 'deadbeef' * 8)
    with pytest.raises(ValueError, match='LT11_BASE_REGISTRATION_SHA_MISMATCH'):
        am.amendment()


def test_amendment_bytes_are_deterministic():
    assert am.canonical(am.amendment()) == am.canonical(am.amendment())


def test_written_amendment_matches_its_sidecar():
    path = am.OUT / 'amendment1.json'
    sidecar = am.OUT / 'amendment1.sha256'
    if not path.exists():
        pytest.skip('amendment not emitted yet; run scripts/grm_d1_amendment1.py')
    assert am.sha_path(path) == sidecar.read_text().split()[0]
    assert json.loads(path.read_text())['registration_sha256'] == am.BASE_SHA


# ----------------------------------------------------------- the rebind

def test_every_rebound_input_matches_the_tree_right_now(doc):
    for name, change in doc['core_rebind']['changed'].items():
        assert am.sha_path(ROOT / name) == change['after_sha256'], (
            '%s rebound to a sha this tree does not have' % name)
        assert change['after_sha256'] != change['before_sha256']


def test_the_rebind_covers_every_drifted_core_input(doc):
    """Nothing that moved may be left unnamed -- that is the whole point."""
    pinned = {n: c['after_sha256'] for n, c in
              json.loads(am.LT1_BINDING.read_text())['core_shas'].items()}
    drifted = {n for n, before in pinned.items()
               if am.sha_path(ROOT / n) != before}
    assert set(doc['core_rebind']['changed']) == drifted
    assert len(drifted) + doc['core_rebind']['unchanged_count'] == len(pinned)


def test_a1_files_are_attributed_to_a1_and_the_others_are_not(doc):
    """A1 touches exactly the files that reference the alias fold module."""
    changed = doc['core_rebind']['changed']
    for name in am.A1_FILES:
        assert name in changed, '%s should have been rebound' % name
        assert changed[name]['attribution'].startswith('A1')
        # And the attribution is checkable: the file really uses alias_fold.
        assert 'alias_fold' in (ROOT / name).read_text()
    for name, change in changed.items():
        if name in am.A1_FILES:
            continue
        assert 'not attributable to A1' in change['attribution']
        # A non-A1 file must NOT reference the alias module.
        assert 'alias_fold' not in (ROOT / name).read_text(), (
            '%s references alias_fold but is attributed away from A1' % name)


def test_the_new_alias_module_is_registered_as_a_new_input(doc):
    new = doc['core_rebind']['new_inputs']
    assert 'core/grm_alias_fold.py' in new
    entry = new['core/grm_alias_fold.py']
    assert entry['sha256'] == am.sha_path(ROOT / 'core/grm_alias_fold.py')
    assert 'new module' in entry['attribution']
    # It genuinely is new: the LT1 binding never pinned it.
    pinned = json.loads(am.LT1_BINDING.read_text())['core_shas']
    assert 'core/grm_alias_fold.py' not in pinned


# ----------------------------------------------------------- the arms

def test_two_arms_share_everything_but_the_one_flag(doc):
    a, aplus = doc['arms']['A'], doc['arms']['A+']
    assert a['flags'] == aplus['flags']
    assert a['cells'] == aplus['cells'] and len(a['cells']) == 26
    assert a['fixture'] == aplus['fixture']
    assert a['admission_rule'] == aplus['admission_rule'] == 'margin_first'
    assert a['alias_fold_merge'] is False
    assert aplus['alias_fold_merge'] is True
    # The environments differ by EXACTLY the alias flag.
    env_a = a['env_after_environment_flags']
    env_p = aplus['env_after_environment_flags']
    assert set(env_p) - set(env_a) == {am.ALIAS_FLAG}
    assert set(env_a) - set(env_p) == set()
    assert env_p[am.ALIAS_FLAG] == '1'
    assert am.ALIAS_FLAG not in env_a
    for key in env_a:
        assert env_a[key] == env_p[key]


def test_both_arms_are_separately_resumable_with_distinct_outputs(doc):
    a, aplus = doc['arms']['A'], doc['arms']['A+']
    assert a['resumable'] is True and aplus['resumable'] is True
    assert a['out_dir'] != aplus['out_dir'], (
        'the arms must not write into the same directory or neither is '
        'separately resumable')


def test_budget_is_two_arms_of_one_point_seven_gpu_hours(doc):
    for arm in doc['arms'].values():
        assert arm['budget_gpu_seconds'] == 6120
        assert arm['budget_gpu_hours'] == 1.7
    assert doc['budget_gpu_seconds_total'] == 12240
    assert doc['budget_gpu_hours_total'] == 3.4


def test_the_pin_note_names_the_r1_idiom_and_restoration(doc):
    note = doc['arms']['A+']['pin_note']
    assert 'pin_rule' in note
    assert 'environment(flags)' in note
    assert 'READ BACK' in note
    assert 'RESTORE' in note


def test_predictions_are_registered_per_arm_before_any_run(doc):
    a = doc['arms']['A']['predictions']
    aplus = doc['arms']['A+']['predictions']
    assert a['corrections'].startswith('5/5')
    assert a['aliases'].startswith('0/5')
    assert a['fresh'] == '6/7'
    assert aplus['corrections'].startswith('5/5')
    assert aplus['fresh'] == '6/7'
    # I register 4/5 where the lead proposed >= 3/5; the disagreement must be
    # stated with its reason, not silently substituted.
    assert aplus['aliases'].startswith('4/5')
    assert '>= 3/5' in aplus['aliases']
    assert 'reservation' in aplus['prediction_note']
    assert '>= 3/5' in aplus['recap']


# ----------------------------------------------------------- the flag

def test_the_amendment_records_the_flag_as_present_with_its_reader(doc):
    flags = {f['name']: f for f in doc['named_flags']}
    entry = flags[am.ALIAS_FLAG]
    assert entry['present_on_tree'] is True
    assert 'grm_alias_fold.py' in entry['reader']
    assert entry['pinned_in_arm'] == 'A+ only'
    assert 'supersedes' in entry and 'present_on_tree' in entry['supersedes']
    # The claim is checkable against core.
    import core.grm_alias_fold as af
    assert af.ENV_NAME == am.ALIAS_FLAG


def test_amendment1_no_longer_owns_lead_commands():
    """Amendment 1's command generator described a runner that never existed.

    It emitted `grm_lt1.py --run --arm … --registration … --out …`; that
    script has no such flags, so the commands could not run and the only
    test on them checked a sha string. `lead_commands.txt` is now generated
    by amendment 2 from the ACTUAL argparse of `scripts/grm_lt1_1.py`, and
    gated by `test_every_emitted_command_actually_runs`, which executes each
    emitted command. This test pins that ownership so the dead generator
    cannot quietly come back.
    """
    text = (am.OUT / 'lead_commands.txt').read_text()
    # Whichever amendment currently owns the file, it is not amendment 1.
    latest = max(int(p.stem.replace('amendment', ''))
                 for p in am.OUT.glob('amendment[0-9]*.json'))
    assert 'amendment %d' % latest in text
    assert latest >= 2
    assert 'scripts/grm_lt1_1.py' in text
    # The broken shape must be gone.
    assert '--registration' not in text
    assert 'grm_lt1.py --run' not in text
    # And amendment 1's own generator is no longer the source of truth.
    stale = am.lead_commands('0' * 64)
    assert stale != text


def test_the_amendment1_budget_was_superseded_by_amendment_two(doc):
    """Amendment 1 registered a ceiling below the reservation sum.

    `--dry-run` against the real cell list showed 26 cells reserve 7410 s,
    and `run_cell` charges the lease. Amendment 1's 6120 s would have railed
    mid-campaign; amendment 2 corrects it. Amendment 1 is NOT rewritten -- it
    stays at its sha -- so this test records that its budget is superseded.
    """
    lease = sum(c['lease_seconds'] for c in doc['arms']['A']['cells'])
    assert doc['budget_gpu_seconds_per_arm'] == 6120
    assert lease > doc['budget_gpu_seconds_per_arm'], (
        'the defect this documents has disappeared; re-check the cell list')
    amendment2 = am.OUT / 'amendment2.json'
    if not amendment2.exists():
        pytest.skip('amendment 2 not emitted yet')
    corrected = json.loads(amendment2.read_text())
    assert corrected['budget_gpu_seconds_per_arm'] == lease
    assert corrected['previous_amendment_sha256'] == am.sha_path(
        am.OUT / 'amendment1.json')
