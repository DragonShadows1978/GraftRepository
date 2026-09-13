"""GRM-D1: gates for the LT1.1 fixture amendment and registration.

The claim LT1.1 makes is narrow and must be provable on CPU: the ONLY things
that changed from LT1 are the correction lineage and the recap battery. These
tests hold the fixture to that, and hold the registration to naming what it
does not fix.

Prior art: the SHA-chained fixture-amendment contract is C7 r3's
(`wt/grm-c7` scripts/grm_c7_register_r3.py, GRM contributors 2026), reused.
See scripts/grm_d1_register_lt11.py for the full prior-art note.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import grm_d1_register_lt11 as reg11  # noqa: E402
from scripts.grm_d1_supersession import correction_command  # noqa: E402


@pytest.fixture(scope='module')
def lt1():
    return json.loads(reg11.LT1_FIXTURE.read_text())


@pytest.fixture(scope='module')
def lt11():
    return json.loads(reg11.fixture_bytes().decode())


def test_turn_mix_accounts_for_every_turn(lt11):
    mix = Counter(t['kind'] for t in lt11['turns'])
    assert mix == dict(fact=60, supersede=15, alias=10, ordinary=75,
                       probe=35, recap_probe=5)
    assert sum(mix.values()) == 200
    assert [t['turn'] for t in lt11['turns']] == list(range(1, 201))


def test_all_fifteen_corrections_became_supersessions(lt1, lt11):
    before = [t for t in lt1['turns'] if t['kind'] == 'correction']
    after = [t for t in lt11['turns'] if t['kind'] == 'supersede']
    assert len(before) == len(after) == 15
    for t in after:
        assert t['old_value']
        assert t['correction_command'].startswith('correct memory: ')
        assert ' => ' in t['correction_command']
        # The command must name the value it retires and carry the new prose.
        assert t['old_value'] in t['correction_command'].split(' => ')[0]
        assert t['user'] in t['correction_command']
        assert t['old_value'] != t['value']


def test_every_supersede_names_the_value_actually_in_force(lt11):
    """The old_value must be the lineage's previous tip, not an invention."""
    last = {}
    for t in lt11['turns']:
        key = (t.get('entity'), t.get('attribute'))
        if t['kind'] == 'supersede':
            assert t['old_value'] == last[key], (
                'turn %d claims old_value %r; the value in force was %r'
                % (t['turn'], t['old_value'], last[key]))
        if t.get('value') is not None:
            last[key] = t['value']


def test_the_user_prose_is_byte_identical_to_lt1(lt1, lt11):
    """Only lineage metadata may change; not one word the user says."""
    before = {t['turn']: t for t in lt1['turns']}
    after = {t['turn']: t for t in lt11['turns']}
    recap_turns = {q['turn'] for q in lt11['recap_probes']}
    for turn, old in before.items():
        if turn in recap_turns:
            continue          # deliberately replaced by a recap probe
        assert after[turn]['user'] == old['user'], 'turn %d prose changed' % turn
        if 'assistant' in old:
            assert after[turn].get('assistant') == old['assistant']


def test_the_recall_probes_and_distances_are_unchanged(lt1, lt11):
    assert lt11['probes'] == lt1['probes']
    assert lt11['distances'] == lt1['distances']
    assert len(lt11['probes']) == 35


def test_the_five_decisions_are_unchanged(lt1, lt11):
    assert lt11['decisions'] == lt1['decisions']


def test_the_old_recap_turn_is_gone(lt1, lt11):
    assert any(t.get('user') == 'recap the five biggest decisions we made'
               for t in lt1['turns'])
    assert not any(t.get('user') == 'recap the five biggest decisions we made'
                   for t in lt11['turns'])
    probes = [t for t in lt11['turns'] if t['kind'] == 'recap_probe']
    assert len(probes) == 5
    assert {t['recap_probe_id'] for t in probes} == {
        'recap_1', 'recap_2', 'recap_3', 'recap_4', 'recap_5'}


def test_fixture_records_its_parent_sha(lt11):
    assert lt11['parent_fixture_sha256'] == hashlib.sha256(
        reg11.LT1_FIXTURE.read_bytes()).hexdigest()


def test_fixture_bytes_are_deterministic():
    assert reg11.fixture_bytes() == reg11.fixture_bytes()


def test_registration_is_arm_a_only_within_budget():
    digest = hashlib.sha256(reg11.fixture_bytes()).hexdigest()
    reg = reg11.registration(digest)
    assert set(reg['arms']) == {'A'}
    assert len(reg['cells']) == 26
    assert all(c['arm'] == 'A' for c in reg['cells'])
    assert reg['budget_gpu_seconds'] == 6120
    assert reg['budget_gpu_hours'] <= 1.7
    assert reg['admission_rule'] == 'margin_first'
    assert reg['required_env']['GRM_ADMISSION_RULE'] == 'margin_first'
    assert reg['fixture']['sha256'] == digest


def test_base_registration_is_frozen_at_its_pre_a1_wording():
    """The base LT1.1 registration is sha-bound; it is NOT edited in place.

    It was written before A1 merged and truthfully recorded the flag as
    absent-at-that-time. Amendment 1 (a separate, sha-chained document) is
    what flips the flag to present and pinned -- see
    tests/test_grm_d1_amendment1.py. Rewriting the base would break its
    sha `02b44d02…` and every receipt that cites it.
    """
    digest = hashlib.sha256(reg11.fixture_bytes()).hexdigest()
    reg = reg11.registration(digest)
    flags = {f['name']: f for f in reg['optional_named_flags']}
    assert reg11.ALIAS_FLAG in flags
    assert flags[reg11.ALIAS_FLAG]['present_on_tree'] is False
    assert 'NOT pinned' in flags[reg11.ALIAS_FLAG]['disposition']
    # The base arm never sets it; only the amendment's A+ arm does.
    assert reg11.ALIAS_FLAG not in reg['required_env']


def test_alias_flag_is_present_and_really_read_on_this_tree():
    """A1 landed: the flag now EXISTS and a real reader consumes it.

    This replaces D1's original `…_is_absent_from_this_tree`. That test was
    correct on the pre-A1 tree and correctly failed the moment A1 merged --
    which is the point of asserting an absence instead of assuming one.
    The obligation now inverts: the registration may only pin the flag
    because a reader for it genuinely exists, so prove the reader.
    """
    import core.grm_alias_fold as af

    # 1. The module and its env contract exist.
    assert af.ENV_NAME == reg11.ALIAS_FLAG
    assert (ROOT / 'core/grm_alias_fold.py').exists()

    # 2. Some file OUTSIDE D1's own is a real consumer, not a mention.
    consumers = []
    for sub in ('core', 'scripts'):
        for path in (ROOT / sub).rglob('*.py'):
            if path.name.startswith(('grm_d1_', 'test_grm_d1_')):
                continue                # D1's own registration bookkeeping
            if reg11.ALIAS_FLAG in path.read_text(errors='ignore'):
                consumers.append(path.name)
    assert 'grm_alias_fold.py' in consumers, (
        'the flag must be read by core; consumers=%r' % sorted(consumers))

    # 3. The reader actually switches on the value, in both directions, and
    #    an unknown token fails CLOSED to the SHIPPED default (A1's stated
    #    contract).  GRM-D2 (2026-09-11) made A1's shipped default ON, so
    #    the last two assertions are pinned to the pre-D2 default with
    #    `GRM_LEGACY_DEFAULTS=1` and are otherwise unchanged; the first two
    #    are explicit tokens and outrank any default, so they stand as-is.
    #    What D1 needs from this test -- that a real reader SWITCHES on the
    #    value -- is exactly what the explicit pair proves.
    legacy = {'GRM_LEGACY_DEFAULTS': '1'}
    assert af.alias_fold_enabled(environ={reg11.ALIAS_FLAG: '1'}) is True
    assert af.alias_fold_enabled(environ={reg11.ALIAS_FLAG: '0'}) is False
    assert af.alias_fold_enabled(environ=dict(legacy)) is False
    assert af.alias_fold_enabled(
        environ=dict(legacy, **{reg11.ALIAS_FLAG: 'maybe'})) is False
    # And the post-D2 side, stated rather than left implicit: with no
    # umbrella the same unset/unknown cases now resolve ON.
    assert af.alias_fold_enabled(environ={}) is True
    assert af.alias_fold_enabled(environ={reg11.ALIAS_FLAG: 'maybe'}) is True


def test_p1_profile_now_pins_the_flag_instead_of_recording_it_absent():
    """P1's named-flag surface must agree that the flag became real."""
    from scripts.grm_profile import _flag_is_read, resolve_profile
    assert _flag_is_read(reg11.ALIAS_FLAG) is True
    resolved = resolve_profile(selection='eb1_c2', pinned=[reg11.ALIAS_FLAG])
    assert resolved['env'].get(reg11.ALIAS_FLAG) == '1'
    assert any(reg11.ALIAS_FLAG in entry
               for entry in resolved['named_flags_applied'])
    assert not any('ABSENT' in note.upper() for note in resolved['notes'])


def test_registration_names_the_core_gap_it_does_not_fix():
    digest = hashlib.sha256(reg11.fixture_bytes()).hexdigest()
    reg = reg11.registration(digest)
    assert len(reg['not_addressed']) == 1
    gap = reg['not_addressed'][0]
    assert gap['cause_class'].startswith('alias-edge-without-base')
    assert gap['rows'] == 5
    assert 'STOPPED' in gap['statement']
    # And the prediction must match: aliases stay wrong.
    assert '0/5' in reg['predictions']['aliases']


def test_no_change_touches_core():
    digest = hashlib.sha256(reg11.fixture_bytes()).hexdigest()
    reg = reg11.registration(digest)
    assert reg['changes'], 'the registration must name its changes'
    for change in reg['changes']:
        assert change['core_change'] is False
        assert change['kind'] == 'fixture'
        assert change['proof']


def test_correction_command_helper_matches_the_fixture(lt11):
    """The two producers of the command string must agree."""
    for t in lt11['turns']:
        if t['kind'] == 'supersede':
            assert correction_command(t, t['old_value']) == t['correction_command']


def test_no_old_value_can_over_match_another_turn(lt11):
    """`correct_memory` matches by substring: each old_value must be unique.

    core/graft_repository.py:correct_memory retires EVERY active node whose
    text contains the query. If two turns shared a value string, one correction
    would retire an unrelated entity's fact. Prove that cannot happen here.
    """
    texts = [(t['turn'], t['user'].casefold()) for t in lt11['turns']]
    for t in lt11['turns']:
        if t['kind'] != 'supersede':
            continue
        hits = [turn for turn, text in texts if t['old_value'].casefold() in text]
        assert len(hits) == 1, (
            'old_value %r for turn %d also appears in turns %r'
            % (t['old_value'], t['turn'], hits))


def test_fixture_rejects_a_correction_with_no_prior_value(lt1):
    broken = json.loads(json.dumps(lt1))
    # Strip the originating fact so turn 3's lineage has no tip to replace.
    broken['turns'] = [t for t in broken['turns'] if t['turn'] != 1]
    broken['turns'].insert(0, dict(turn=1, kind='ordinary',
                                   user='nothing in particular',
                                   assistant='noted'))
    path = ROOT / 'artifacts/grm_d1/_broken_fixture_for_red_proof.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(broken))
    try:
        with pytest.raises(ValueError, match='LT11_CORRECTION_WITHOUT_PRIOR_VALUE'):
            reg11.fixture_bytes(path)
    finally:
        path.unlink()
