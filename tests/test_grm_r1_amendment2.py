"""GRM-R1 amendment 2: stale-session archiving and environment restoration.

Both defects were found by the lead on a run/tree this seat had reported
green.  The env leak in particular hid behind FILE ORDER: the seat's earlier
"91 passed" ran R1 LAST.  These tests therefore assert the invariants
directly rather than relying on a suite ordering.

Prior art: GRM C7/C2 amendment-chain tests and FIX8 resume tests (GRM
contributors, 2026) -- TAKEN: chain drift injection and the archive-not-
overwrite contract.  pytest ``monkeypatch`` (pytest contributors) -- used
here so no test can itself leak, which is the very defect under test.
OURS: the collision fixture and the env-restoration assertions.  No prior
art known to me for this exact composition.  No model-quality claim.
"""
from __future__ import annotations

import json
import os

import pytest

from scripts import grm_r1_replay as run


def _chain_into(tmp_path, rearm=None, links=(1, 2, 3, 4)):
    """Copy the real amendment chain into ``tmp_path``, optionally re-scoped.

    The chain grows as the arc goes on, so this plants EVERY link by default;
    ``links`` narrows it for tests that deliberately truncate the chain.
    """
    for index in links:
        source = run.OUT / f'amendment_{index}.json'
        a = json.loads(source.read_text())
        if rearm is not None:
            a['rearm'] = rearm
        if index > 1:
            a['previous_amendment_sha256'] = run.sha(
                tmp_path / f'amendment_{index - 1}.json')
        run.write(tmp_path / f'amendment_{index}.json', a)
        (tmp_path / f'amendment_{index}.sha256').write_text(
            run.sha(tmp_path / f'amendment_{index}.json')
            + f'  amendment_{index}.json\n')


def _failed_controller(ids):
    return {'status': 'FAILED',
            'error': "RuntimeError('cudaMalloc failed: out of memory')",
            'red': [], 'elapsed_seconds': 138.5, 'charged_seconds': 263.0,
            'reservation_seconds': 263, 'fake': True, 'attempt': 1,
            'amendment': None, 'cells_run': ids, 'cells_retained': [],
            'registration_sha256': run.sha(run.REG)}


def _rearm(batch_id, keep, drop):
    return {batch_id: {'retain': keep, 'reissue': drop, 'lease_seconds': 200,
                       'prior_status': 'FAILED', 'prior_error': 'oom',
                       'prior_charged_seconds': 263.0,
                       'measured_per_cell_seconds': 18.0,
                       'estimate_basis': 'test fixture'}}


# --------------------------------------------------------------------------
# DEFECT 1 -- a re-issued cell collided with its own stale scratch
# --------------------------------------------------------------------------

def test_archive_stale_sessions_moves_only_that_cells_scratch(tmp_path):
    c = run.cells()[0]
    root = tmp_path / 'sessions' / c['id']
    (root / 'off' / 'repository').mkdir(parents=True)
    (root / 'off' / 'repository' / 'manifest.json').write_text('{"stale":1}')
    (root / 'on').mkdir(parents=True)
    other = tmp_path / 'sessions' / 'some-other-cell' / 'off'
    other.mkdir(parents=True)

    archived = run.archive_stale_sessions(tmp_path, c)
    assert archived['attempt'] == 1
    assert sorted(archived['archived_arms']) == ['off', 'on']
    # The scratch is ARCHIVED, not deleted: still inspectable.
    assert (root / 'attempt_1' / 'off' / 'repository' / 'manifest.json'
            ).read_text() == '{"stale":1}'
    assert not (root / 'off').exists() and not (root / 'on').exists()
    # Nothing outside this cell's own session directory is touched.
    assert other.exists()


def test_archive_stale_sessions_is_a_noop_when_clean(tmp_path):
    c = run.cells()[0]
    assert run.archive_stale_sessions(tmp_path, c) is None


def test_archive_stale_sessions_numbers_successive_attempts(tmp_path):
    c = run.cells()[0]
    root = tmp_path / 'sessions' / c['id']
    for expected in (1, 2, 3):
        (root / 'off').mkdir(parents=True)
        assert run.archive_stale_sessions(tmp_path, c)['attempt'] == expected
    assert {p.name for p in root.iterdir()} == {
        'attempt_1', 'attempt_2', 'attempt_3'}


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_r1/amendment_2.json')
def test_reissue_over_stale_scratch_no_longer_collides(tmp_path):
    """The exact shape of batch_R1_a1.log: cell 6 kept off/ and on/.

    Reproduces the collision condition, then asserts the re-issue archives
    the scratch, runs, and leaves every RETAINED receipt byte-identical.
    """
    r = run.verify(tmp_path)
    batch_id = r['batch_ids'][0]
    ids = r['batches'][batch_id]
    run.batch(batch_id, fake=True, root=tmp_path)
    d = tmp_path / 'gpu' / batch_id

    keep, drop = ids[:5], ids[5:]
    kept = {c: (d / 'cells' / (c + '.json')).read_bytes() for c in keep}
    for cell_id in drop:
        (d / 'cells' / (cell_id + '.json')).unlink()
    # The first re-issued cell KEEPS its scratch -- that is the collision.
    stale = d / 'sessions' / drop[0]
    assert (stale / 'off').exists() and (stale / 'on').exists()

    (d / 'controller.json').unlink()
    run.write(d / 'controller.json', _failed_controller(ids))
    _chain_into(tmp_path, rearm=_rearm(batch_id, keep, drop))

    run.batch(batch_id, fake=True, root=tmp_path)      # must NOT raise

    row = json.loads((d / 'cells' / (drop[0] + '.json')).read_text())
    archived = row['archived_stale_sessions']
    assert archived is not None
    assert sorted(archived['archived_arms']) == ['off', 'on']
    # Archived scratch AND fresh arms both present.
    assert (stale / 'attempt_1').is_dir()
    assert (stale / 'off').exists() and (stale / 'on').exists()
    # Retained receipts untouched; every cell now has one.
    for cell_id, before in kept.items():
        assert (d / 'cells' / (cell_id + '.json')).read_bytes() == before
    for cell_id in ids:
        assert (d / 'cells' / (cell_id + '.json')).exists()


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_r1/amendment_2.json')
def test_a_cell_with_a_receipt_is_never_rerun(tmp_path):
    """Archiving only ever sees cells with no receipt, so evidence is safe."""
    r = run.verify(tmp_path)
    batch_id = r['batch_ids'][0]
    run.batch(batch_id, fake=True, root=tmp_path)
    d = tmp_path / 'gpu' / batch_id
    ids = r['batches'][batch_id]
    keep, drop = ids[:5], ids[5:]
    for cell_id in drop:
        (d / 'cells' / (cell_id + '.json')).unlink()
    (d / 'controller.json').unlink()
    run.write(d / 'controller.json', _failed_controller(ids))
    _chain_into(tmp_path, rearm=_rearm(batch_id, keep, drop))
    run.batch(batch_id, fake=True, root=tmp_path)
    controller = json.loads((d / 'controller.json').read_text())
    assert controller['cells_run'] == drop
    # No retained cell was re-run, so none of them archived anything.
    for cell_id in keep:
        row = json.loads((d / 'cells' / (cell_id + '.json')).read_text())
        assert row.get('archived_stale_sessions') is None


# --------------------------------------------------------------------------
# DEFECT 2 -- the rule pin leaked into the process environment
# --------------------------------------------------------------------------

def test_scoped_env_restores_an_absent_key_by_removing_it(monkeypatch):
    monkeypatch.delenv(run.RULE_ENV, raising=False)
    with run.scoped_env():
        os.environ[run.RULE_ENV] = 'margin_first'
        assert os.environ[run.RULE_ENV] == 'margin_first'
    # Absent before => REMOVED after, not set to ''.
    assert run.RULE_ENV not in os.environ


def test_scoped_env_restores_a_prior_value_exactly(monkeypatch):
    monkeypatch.setenv(run.RULE_ENV, 'pre_existing')
    with run.scoped_env():
        os.environ[run.RULE_ENV] = 'margin_first'
    assert os.environ[run.RULE_ENV] == 'pre_existing'


def test_scoped_env_drops_keys_the_body_added(monkeypatch):
    monkeypatch.delenv('GRM_PROBE_LADDER', raising=False)
    with run.scoped_env():
        os.environ['GRM_PROBE_LADDER'] = '1'
    assert 'GRM_PROBE_LADDER' not in os.environ


def test_scoped_env_restores_on_exception(monkeypatch):
    monkeypatch.delenv(run.RULE_ENV, raising=False)
    with pytest.raises(RuntimeError):
        with run.scoped_env():
            os.environ[run.RULE_ENV] = 'margin_first'
            raise RuntimeError('boom')
    assert run.RULE_ENV not in os.environ


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_r1/amendment_2.json')
def test_fake_batch_leaves_no_rule_in_the_environment(tmp_path, monkeypatch):
    """THE REGRESSION: a batch must not re-rule the rest of the process."""
    monkeypatch.delenv(run.RULE_ENV, raising=False)
    r = run.verify(tmp_path)
    run.batch(r['batch_ids'][0], fake=True, root=tmp_path)
    assert run.RULE_ENV not in os.environ
    assert not [k for k in os.environ if k.startswith('GRM_')]


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_r1/amendment_2.json')
def test_admission_rule_is_back_to_default_after_a_fake_batch(tmp_path,
                                                              monkeypatch):
    """admission_rule() reads os.environ at CALL time -- check the real call."""
    from core.grm_admission import admission_rule
    monkeypatch.delenv(run.RULE_ENV, raising=False)
    assert admission_rule() == 'all_tokens_bind'
    r = run.verify(tmp_path)
    run.batch(r['batch_ids'][0], fake=True, root=tmp_path)
    assert admission_rule() == 'all_tokens_bind'


@pytest.mark.campaign_receipt(registration='artifacts/grm_r1/amendment_2.json (cell checkpoints pinned under absolute /mnt/ForgeRealm/wt/grm-c2/artifacts/grm_c2/epochs/scout-fix-2/, pruned with the grm-c2 fork)')
def test_run_cell_restores_the_environment(tmp_path, monkeypatch):
    monkeypatch.delenv(run.RULE_ENV, raising=False)
    run.run_cell(run.cells()[0], tmp_path, None, fake=True)
    assert run.RULE_ENV not in os.environ


@pytest.mark.campaign_receipt(registration='artifacts/grm_r1/amendment_2.json (cell checkpoints pinned under absolute /mnt/ForgeRealm/wt/grm-c2/artifacts/grm_c2/epochs/scout-fix-2/, pruned with the grm-c2 fork)')
def test_both_arms_still_get_their_own_rule_despite_restoration(tmp_path):
    """Restoration must not defeat the pin: the arms must still differ."""
    value, _ = run.run_cell(run.cells()[0], tmp_path, None, fake=True)
    assert value['rule_pins'] == {'off': 'all_tokens_bind',
                                  'on': 'margin_first'}
    assert value['arms']['off']['admission_rule'] == 'all_tokens_bind'
    assert value['arms']['on']['admission_rule'] == 'margin_first'
    assert value['off_plan_parity'] is True


# --------------------------------------------------------------------------
# the amendment chain
# --------------------------------------------------------------------------

@pytest.mark.campaign_receipt(
    registration='artifacts/grm_r1/amendment_2.json')
def test_amendment_2_chains_to_amendment_1():
    # amendment() returns the LATEST link, so assert link 2's own contents
    # and its place in the chain rather than "latest == 2".
    a2 = run.read(run.OUT / 'amendment_2.json')
    assert a2['amendment'] == 2
    assert a2['registration_sha256'] == run.sha(run.REG)
    assert a2['previous_amendment_sha256'] == run.sha(
        run.OUT / 'amendment_1.json')
    a = run.amendment()
    assert run.sha(run.OUT / 'amendment_2.json') in a['_chain']
    assert len(a['_chain']) >= 2
    # Scope and acceptance carry through unchanged.
    assert a2['rearm_unchanged_from_amendment_1'] is True
    assert len(a2['rearm']['R1']['retain']) == 5
    assert len(a2['rearm']['R1']['reissue']) == 3
    assert a2['acceptance_unchanged'] is True
    r = run.verify()
    assert a['prediction'] == r['prediction']
    assert a['verdict_rule'] == r['verdict_rule']


def test_broken_chain_link_is_rejected(tmp_path):
    _chain_into(tmp_path)
    a2 = json.loads((tmp_path / 'amendment_2.json').read_text())
    a2['previous_amendment_sha256'] = 'not-amendment-1'
    (tmp_path / 'amendment_2.json').unlink()
    run.write(tmp_path / 'amendment_2.json', a2)
    (tmp_path / 'amendment_2.sha256').write_text(
        run.sha(tmp_path / 'amendment_2.json') + '  amendment_2.json\n')
    with pytest.raises(ValueError, match='R1_AMENDMENT_2_CHAIN_MISMATCH'):
        run.amendment(tmp_path)


def test_both_amendments_rebinds_accumulate():
    """A later link wins, but an earlier link's rebind is not dropped."""
    a = run.amendment()
    rebound = a['rebound_inputs']
    assert 'scripts/grm_r1_replay.py' in rebound
    assert 'scripts/grm_r1_amend_1.py' in rebound     # from link 1
    assert 'scripts/grm_r1_amend_2.py' in rebound     # from link 2
    assert rebound['scripts/grm_r1_replay.py'] == run.sha(
        run.ROOT / 'scripts/grm_r1_replay.py')


def test_amendment_2_documents_both_defects():
    a = run.read(run.OUT / 'amendment_2.json')
    ids = {d['id'] for d in a['defects']}
    assert ids == {'R1-D1-stale-session-collision', 'R1-D2-environment-leak'}
    for defect in a['defects']:
        assert defect['evidence'] and defect['cause'] and defect['fix']


# --------------------------------------------------------------------------
# accounting: an archived attempt's GPU time is never refunded
# --------------------------------------------------------------------------

@pytest.mark.campaign_receipt(
    registration='artifacts/grm_r1/amendment_2.json')
def test_archived_attempt_charge_stays_on_the_books(tmp_path):
    """A re-issue must ADD to the failed attempt's charge, never replace it.

    Found while verifying amendment 2 on the real receipts: after the lead's
    second R1 attempt, campaign_state reported 134 s for a batch that had
    actually spent 263 + 134 = 397 s, because only controller.json was read
    and the archived attempt-1 controller was ignored.
    """
    r = run.verify(tmp_path)
    batch_id = r['batch_ids'][0]
    ids = r['batches'][batch_id]
    run.batch(batch_id, fake=True, root=tmp_path)
    d = tmp_path / 'gpu' / batch_id
    keep, drop = ids[:5], ids[5:]
    for cell_id in drop:
        (d / 'cells' / (cell_id + '.json')).unlink()
    (d / 'controller.json').unlink()
    run.write(d / 'controller.json', _failed_controller(ids))
    _chain_into(tmp_path, rearm=_rearm(batch_id, keep, drop))

    before, _ = run.campaign_state(run.verify(tmp_path), tmp_path)
    assert before == 263.0

    run.batch(batch_id, fake=True, root=tmp_path)
    assert (d / 'controller_attempt_1.json').exists()
    after, complete = run.campaign_state(run.verify(tmp_path), tmp_path)
    # 263 s of real GPU time is NOT refunded by the re-issue.
    assert after > before
    archived = json.loads((d / 'controller_attempt_1.json').read_text())
    current = json.loads((d / 'controller.json').read_text())
    assert after == pytest.approx(archived['charged_seconds']
                                  + current['charged_seconds'])
    assert complete == [batch_id]
