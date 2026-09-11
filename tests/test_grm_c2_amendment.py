"""Prior art: C2 r1 fake-receipt gates and CMC1 (project, 2026), reused.
New cases exercise amendment substitution, cumulative reservations, scheduling
and explicit resume. Author-run CPU tests, not blind or GPU validation.
"""
import copy
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts import grm_c2_amended as a
from scripts.grm_c2_profile import read, create, sha
from tests.test_grm_c2_profile import synthetic_rows


def test_amended_budget_preserves_all_52_cells_and_registers_defaults96_nonfit():
    r = a.effective_registration(); old = read(a.REGISTRATION)
    assert r['cells'] == old['cells'] and len(r['cells']) == 52
    assert r['estimated_seconds'] == 3140 and r['budget_seconds'] == 3600
    assert old['budget_seconds'] == 2880
    d = r['amendment']['defaults96']
    assert d['status'] == 'NON_FIT' and d['extra_cells'] == 26
    assert d['extra_estimated_seconds'] == sum(c['estimate_seconds'] for c in old['cells'] if c['side'] == 'defaults') == 1570
    assert d['total_estimated_seconds'] == 4710
    assert d['flags']['arena_width'] == 96 and d['flags']['capture_pin'] == 'off' and d['flags']['seat_near_live'] is False
    assert a.dry_run()['status'] == 'READY_FOR_LEAD'


@pytest.mark.parametrize('defect', ['cap', 'r1', 'order', 'resume', 'width'])
def test_forged_amendment_refused_even_with_recomputed_sidecar(tmp_path, monkeypatch, defect):
    value = read(a.AMENDMENT)
    if defect == 'cap': value['budget_seconds'] = 7200
    elif defect == 'r1': value['registration_sha256'] = '0' * 64
    elif defect == 'order': value['bindings']['orders/GRM_C2_AMENDMENT_1.md'] = '0' * 64
    elif defect == 'resume': value['resume_rule'] = 'retry RED'
    else: value['defaults96']['status'] = 'FIT'
    path = tmp_path / 'forged.json'; create(path, value)
    path.with_suffix('.sha256').write_text(sha(path))
    monkeypatch.setattr(a, 'AMENDMENT', path)
    with pytest.raises(ValueError, match='forged or stale amendment'):
        a.effective_registration()


@pytest.mark.parametrize('target', ['orders/GRM_C2_AMENDMENT_1.md', 'artifacts/grm_c2/registration.json', 'scripts/grm_c2_cells.py'])
def test_stale_order_registration_or_worker_binding_refused(monkeypatch, target):
    original = a.sha
    monkeypatch.setattr(a, 'sha', lambda p: '0' * 64 if Path(p) == a.ROOT / target else original(p))
    with pytest.raises(ValueError, match='stale amendment binding'):
        a.effective_registration()


def test_missing_amendment_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(a, 'AMENDMENT', tmp_path / 'missing.json')
    with pytest.raises(FileNotFoundError): a.effective_registration()


@pytest.fixture
def local(tmp_path, monkeypatch):
    r = a.effective_registration()
    monkeypatch.setattr(a, 'OUT', tmp_path)
    monkeypatch.setattr(a.old, 'sources', lambda: {})
    monkeypatch.setattr(a.time, 'sleep', lambda t: None)
    return r


def finished(cell, status='COMPLETE', charge=10):
    d = a.OUT / 'cells' / cell['id']; d.mkdir(parents=True)
    value = dict(cell=cell, rows=[x for x in synthetic_rows() if
        (x['side'], x['battery'], x['phase']) == (cell['side'], cell['battery'], cell['phase'])],
        end_capture={'valid': True}, **a.receipt_bindings())
    if cell['phase'] == 'restart': ids = cell['probes']
    elif cell['battery'] == 'sup':
        ids = [p['probe_id'] for p in read(a.REGISTRY)['sup_plans'][cell['session']]]
    else:
        reg = read(a.REGISTRY)
        ids = [f'lh_t{t:03d}' if cell['battery'] == 'longhistory' else
               next(p['probe_id'] for p in reg['census_probes'] if p['turn'] == t)
               for t in range(cell['start'], cell['stop']) if reg['session_script'][t]['kind'] == 'probe']
    value['rows'] = [x for x in value['rows'] if x['probe_id'] in ids]
    create(d / 'worker.json', value)
    create(d / 'controller.json', dict(status=status, charged_seconds=charge,
        finished_at=0, cell=cell, worker_sha256=sha(d / 'worker.json'), **a.receipt_bindings()))


def test_order_scoring_and_outer_rails_for_all_52_cells(local, monkeypatch):
    r = local; events = []
    def execute(command, **kw):
        assert command[:5] == ['timeout', '--signal=TERM', '--kill-after=5', '585', a.sys.executable]
        assert command[-2] == '--cell' and kw['pass_fds']
        cell = a.old.cell_by_id(r, command[-1])
        assert all(a.cell_state(a.old.cell_by_id(r, dep)) == 'COMPLETE' for dep in cell['depends'])
        events.append(cell['id']); finished(cell)
        return SimpleNamespace(returncode=0)
    original = a.score_battery
    def score(r, battery):
        events.append('score:' + battery)
        return original(r, battery)
    monkeypatch.setattr(a.subprocess, 'run', execute)
    monkeypatch.setattr(a, 'score_battery', score)
    assert a.run_campaign(r) == 0
    assert len([e for e in events if not e.startswith('score:')]) == 52
    for b in a.BATTERIES:
        positions = [i for i, e in enumerate(events) if e in [c['id'] for c in r['cells'] if c['battery'] == b]]
        assert events[max(positions) + 1] == 'score:' + b
    assert a.summary(r) == 0


def test_stop_on_red_resume_next_unstarted_never_retry(local, monkeypatch):
    r = local; calls = []
    def execute(command, **kw):
        cell = a.old.cell_by_id(r, command[-1]); calls.append(cell['id'])
        finished(cell, status='RED' if len(calls) == 1 else 'COMPLETE')
        return SimpleNamespace(returncode=1 if len(calls) == 1 else 0)
    monkeypatch.setattr(a.subprocess, 'run', execute)
    assert a.run_campaign(r) == 1 and len(calls) == 1
    with pytest.raises(ValueError, match='use --resume'): a.run_campaign(r)
    assert a.run_campaign(r, resume=True) == 1  # sup score RED remains RED
    assert calls[1] == a.ordered_cells(r)[1]['id'] and len(calls) == len(set(calls))
    assert read(a.OUT / 'scores/sup.json')['status'] == 'RED'
    assert a.run_campaign(r, resume=True) == 0
    assert len(calls) == 52 and a.summary(r) == 1


def test_red_and_orphan_cells_never_retried_and_orphan_fully_charged(local):
    r = local; first, second = a.ordered_cells(r)[:2]
    finished(first, status='RED')
    (a.OUT / 'cells' / second['id']).mkdir(parents=True)
    assert a.charged_seconds(r) == 295
    for cell in (first, second):
        with pytest.raises(ValueError, match='never retried'): a.run_cell(cell, r)


def test_red_persist_blocks_restart_without_worker_or_recapture(local, monkeypatch):
    r = local; cell = next(c for c in r['cells'] if c['phase'] == 'restart')
    monkeypatch.setattr(a.subprocess, 'run', lambda *x, **k: pytest.fail('worker launched'))
    assert a.run_cell(cell, r) == 1
    receipt = read(a.OUT / 'cells' / cell['id'] / 'controller.json')
    assert 'BLOCKED_DEPENDENCY' in receipt['error'] and receipt['charged_seconds'] == 0


def test_amended_cap_used_for_reservation_and_worker_lease(local, monkeypatch):
    r = local; cells = a.ordered_cells(r)
    for c in cells[1:12]: finished(c, charge=280)
    # 3080 already charged: r1 cap would refuse; 3080+285 fits amended cap.
    assert a.charged_seconds(r) == 3080
    leases = []
    @contextmanager
    def lease(seconds, wait):
        leases.append((seconds, wait)); yield
    monkeypatch.setitem(a.sys.modules, 'scripts.grm_cmc1_gpu_arms', SimpleNamespace(gpu_lease=lease))
    def execute(command, **kw):
        assert kw['timeout'] == 280 and command[-2] == '--worker'
        return SimpleNamespace(returncode=42)  # CPU fake worker RED, no GPU
    monkeypatch.setattr(a.subprocess, 'run', execute)
    assert a.run_cell(cells[0], r) == 1 and leases == [(285, 240)]
    value = read(a.OUT / 'cells' / cells[0]['id'] / 'controller.json')
    assert 'returncode=42' in value['error']


def test_budget_reservation_refuses_over_cap_without_worker(local, monkeypatch):
    r = local; cells = a.ordered_cells(r)
    for c in cells[1:13]: finished(c, charge=285)
    assert a.charged_seconds(r) == 3420
    monkeypatch.setattr(a.subprocess, 'run', lambda *x, **k: pytest.fail('worker launched'))
    assert a.run_cell(cells[0], r) == 1
    assert 'budget rail' in read(a.OUT / 'cells' / cells[0]['id'] / 'controller.json')['error']


def test_cooldown_survives_resume(local, monkeypatch):
    r = local; cell = a.ordered_cells(r)[0]; finished(cell)
    path = a.OUT / 'cells' / cell['id'] / 'controller.json'
    value = read(path); value['finished_at'] = 100; path.write_text(json.dumps(value))
    monkeypatch.setattr(a.time, 'time', lambda: 112)
    waits = []; monkeypatch.setattr(a.time, 'sleep', waits.append)
    a.cooldown(r)
    assert waits == [18]


def test_summary_reports_denominators_width_seats_rt1_and_missing(local, capsys):
    r = local
    assert a.summary(r) == 1
    output = capsys.readouterr().out
    assert 'NOT_RUN' in output and 'UNKNOWN/MISSING' in output
    for c in r['cells']: finished(c)
    assert a.summary(r) == 0
    output = capsys.readouterr().out
    assert 'defaults(256)' in output and 'profile(96)' in output
    assert '9/9 | 9/9' in output and '9/10 | 9/10' in output and '13/14 | 13/14' in output
    assert '675/675 | YES/YES' in output
    assert 'shipped defaults vs proposed profile' in output and 'not a width-matched pair' in output
    for k in a.RT1_FIELDS: assert k in output


@pytest.mark.parametrize('defect', ['rt1', 'token_seats', 'metadata_retained', 'new_process', 'duplicate'])
def test_cell_evidence_refuses_missing_or_duplicate_fields(local, defect):
    cell = next(c for c in local['cells'] if c['phase'] == 'restart'); finished(cell)
    value = read(a.OUT / 'cells' / cell['id'] / 'worker.json')
    if defect == 'duplicate': value['rows'].append(copy.deepcopy(value['rows'][0]))
    else: value['rows'][0].pop(defect)
    with pytest.raises(ValueError): a.evidence_rows(cell, value)


def test_changed_completed_receipt_refused(local):
    cell = local['cells'][0]; finished(cell)
    path = a.OUT / 'cells' / cell['id'] / 'worker.json'; path.write_text(path.read_text() + ' ')
    with pytest.raises(ValueError, match='changed after completion'): a.collect(local)
