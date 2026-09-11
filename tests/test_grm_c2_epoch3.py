"""Prior art: C2 amendment1 fake campaign tests (project, 2026), reused.
New: epoch isolation, fixed source integrity and retained historical charge.
No prior art known to me beyond these local systems. CPU author baseline only.
"""
import copy
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts import grm_c2_epoch3 as e
from scripts import grm_c2_amended as a
from scripts.grm_c2_profile import read, create, sha
from tests.test_grm_c2_amendment import finished


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_c2/epochs/scout-fix-2/ + orders/GRM_SCOUT_FIX_2.md')
def test_epoch_registration_all52_fresh_and_prior_red_preserved():
    epoch = e.verify_amendment()
    r = e.effective_registration()
    assert len(r['cells']) == 52 and r['cells'] == read(a.REGISTRATION)['cells']
    assert r['budget_seconds'] == 3600 and r['estimated_seconds'] == 3140
    assert epoch['prior_charged_seconds'] == 46.348182500805706
    assert read(e.ROOT / epoch['prior_controller'])['status'] == 'RED'
    assert e.EPOCH_OUT != e.ROOT / 'artifacts/grm_c2'
    assert epoch['order_sha256'] == sha(e.ROOT / 'orders/GRM_SCOUT_FIX_2.md')


@pytest.mark.parametrize('defect', ['budget', 'charge', 'order', 'sources', 'epoch'])
def test_forged_epoch_refused_with_recomputed_sidecar(tmp_path, monkeypatch, defect):
    value = read(e.AMENDMENT)
    key = {'budget':'budget_seconds', 'charge':'prior_charged_seconds',
           'order':'order_sha256', 'sources':'source_paths', 'epoch':'epoch_directory'}[defect]
    value[key] = 'forged'
    path = tmp_path / 'forged.json'; create(path, value)
    path.with_suffix('.sha256').write_text(sha(path))
    monkeypatch.setattr(e, 'AMENDMENT', path)
    with pytest.raises(ValueError, match='forged or stale amendment 3'):
        e.verify_amendment()


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_c2/epochs/scout-fix-2/ + orders/GRM_SCOUT_FIX_2.md')
@pytest.mark.parametrize('target', ['core/graft_repository.py', 'scripts/grm_c2_profile.py',
    'orders/GRM_SCOUT_FIX_2.md', 'artifacts/grm_c2/lead_commands.txt',
    'artifacts/grm_c2/cells/profile-sup-correction_then_restatement/controller.json',
    'artifacts/grm_c2/amendment_lead_2.json'])
def test_stale_epoch_input_refused(monkeypatch, target):
    original = e.sha
    monkeypatch.setattr(e, 'sha', lambda p: '0' * 64 if Path(p) == e.ROOT / target else original(p))
    with pytest.raises(ValueError, match='stale amendment 3 binding'):
        e.verify_amendment()


@pytest.fixture
def local_epoch(tmp_path, monkeypatch):
    for key in ('OUT', '__file__', 'effective_registration', 'charged_seconds', 'receipt_bindings', 'dry_run'):
        monkeypatch.setattr(a, key, getattr(a, key))
    monkeypatch.setattr(a.old, 'OUT', a.old.OUT)
    monkeypatch.setattr(e, 'EPOCH_OUT', tmp_path / 'epoch')
    e.install()
    monkeypatch.setattr(a.time, 'sleep', lambda t: None)
    monkeypatch.setattr(a.old, 'sources', lambda: {})
    return a.effective_registration()


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_c2/epochs/scout-fix-2/ + orders/GRM_SCOUT_FIX_2.md')
def test_resume_runs_all52_fresh_then_only_next_unstarted(local_epoch, monkeypatch):
    r = local_epoch; calls = []
    assert a.charged_seconds(r) == r['epoch']['prior_charged_seconds']
    assert set(a.dry_run()['states'].values()) == {'UNSTARTED'}
    def execute(command, **kw):
        assert command[-3] == str(Path(e.__file__))
        cell = a.old.cell_by_id(r, command[-1]); calls.append(cell['id'])
        assert all(a.cell_state(a.old.cell_by_id(r, d)) == 'COMPLETE' for d in cell['depends'])
        finished(cell)
        assert read(a.OUT / 'cells' / cell['id'] / 'worker.json')['epoch_amendment_sha256'] == sha(e.AMENDMENT)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(a.subprocess, 'run', execute)
    assert a.run_campaign(r, resume=True) == 0
    assert calls == [c['id'] for c in a.ordered_cells(r)] and len(calls) == 52
    assert calls[0] == 'profile-sup-correction_then_restatement'
    assert a.charged_seconds(r) == pytest.approx(520 + 46.348182500805706)
    assert a.run_campaign(r, resume=True) == 0 and len(calls) == 52
    assert a.summary(r) == 0


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_c2/epochs/scout-fix-2/ + orders/GRM_SCOUT_FIX_2.md')
def test_old_epoch_receipt_refused(local_epoch):
    cell = local_epoch['cells'][0]
    d = a.OUT / 'cells' / cell['id']; d.mkdir(parents=True)
    old = read(e.ROOT / local_epoch['epoch']['prior_controller'])
    create(d / 'controller.json', old)
    with pytest.raises(ValueError, match='stale receipt binding'):
        a.cell_state(cell)


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_c2/epochs/scout-fix-2/ + orders/GRM_SCOUT_FIX_2.md')
def test_prior_charge_counts_toward_reservation(local_epoch, monkeypatch):
    r = local_epoch; cells = a.ordered_cells(r)
    # 3300+285 fits3600 alone, but the historical46.348 charge makes it exceed.
    for c in cells[1:13]: finished(c, charge=275)
    monkeypatch.setattr(a.subprocess, 'run', lambda *x, **k: pytest.fail('worker launched'))
    assert a.run_cell(cells[0], r) == 1
    assert 'budget rail' in read(a.OUT / 'cells' / cells[0]['id'] / 'controller.json')['error']


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_c2/epochs/scout-fix-2/ + orders/GRM_SCOUT_FIX_2.md')
def test_new_red_stops_and_is_not_retried(local_epoch, monkeypatch):
    r = local_epoch; calls = []
    def execute(command, **kw):
        cell = a.old.cell_by_id(r, command[-1]); calls.append(cell['id'])
        finished(cell, status='RED')
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(a.subprocess, 'run', execute)
    assert a.run_campaign(r, resume=True) == 1 and len(calls) == 1
    with pytest.raises(ValueError, match='never retried'):
        a.run_cell(a.ordered_cells(r)[0], r)
    assert a.charged_seconds(r) == pytest.approx(10 + 46.348182500805706)


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_c2/epochs/scout-fix-2/ + orders/GRM_SCOUT_FIX_2.md')
def test_orphan_fully_charged_and_restart_dependency_fails_closed(local_epoch, monkeypatch):
    r = local_epoch
    first = a.ordered_cells(r)[0]
    (a.OUT / 'cells' / first['id']).mkdir(parents=True)
    assert a.charged_seconds(r) == pytest.approx(285 + 46.348182500805706)
    restart = next(c for c in r['cells'] if c['phase'] == 'restart')
    monkeypatch.setattr(a.subprocess, 'run', lambda *x, **k: pytest.fail('worker launched'))
    assert a.run_cell(restart, r) == 1
    assert 'BLOCKED_DEPENDENCY' in read(a.OUT / 'cells' / restart['id'] / 'controller.json')['error']
