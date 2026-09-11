"""Prior art: C2 synthetic paired receipts/tests (project, 2026), inspected.
Reuse registered IDs and control winners; add A4 scope and immutable replay cases.
No prior art known to me beyond these local systems for this test adapter.
"""
import copy
import importlib.util
import os
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get('A4_SCORER_PATH', str(ROOT / 'artifacts/grm_c2/a4_staging/scripts/grm_c2_score_a4.py')))
spec = importlib.util.spec_from_file_location('a4_test_subject', SOURCE)
a4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a4)


def synthetic():
    registry = a4.read(ROOT / 'config/grm_eb1_profile_registered.json')
    rows = []
    for battery, ids in registry['probe_ids'].items():
        controls = registry['original_correct_controls'][battery]
        for side in ('profile', 'defaults'):
            winners = (controls + [i for i in ids if i not in controls])[:registry['prediction'][side][battery]]
            for phase in ('fresh', 'restart'):
                for i in ids:
                    rows.append(dict(battery=battery, side=side, phase=phase, probe_id=i,
                                     correct=i in winners, metadata_retained=True, new_process=True,
                                     token_seats=75, capture_valid=True,
                                     rt1={k: [] for k in a4.legacy().RT1_FIELDS}))
    return registry, rows


def target(rows, side='profile', phase='restart'):
    return next(r for r in rows if r['battery'] == 'sup' and r['side'] == side and r['phase'] == phase)


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_c2/a4_staging/ + artifacts/grm_c2/epochs/scout-fix-2/')
def test_epoch3_sup_original_red_corrected_green():
    _, registry, registration, bindings = a4.context()
    v = a4.evaluate(a4.EPOCH, registration, registry, bindings, 'sup')
    assert v['original_rule_verdict'] == v['recorded_original_verdict'] == 'RED'
    assert v['status'] == v['corrected_rule_verdict'] == 'GREEN'
    assert v['corrected_rule']['adoption_acceptance'] is True
    assert v['defaults_findings']['restart_retained'] is False
    for side, scores, seats in [('profile', [9, 9], [575, 575]), ('defaults', [4, 3], [1391, 1241])]:
        phases = v['original_rule']['arms'][side]['phases']
        assert [phases[p]['correct'] for p in ('fresh', 'restart')] == scores
        assert [sum(r['token_seats'] for r in phases[p]['rows'].values()) for p in ('fresh', 'restart')] == seats


def test_synthetic_profile_restart_loss_red_both():
    registry, rows = synthetic()
    target(rows)['correct'] = False
    v = a4.compare(rows, registry, 'sup')
    assert v['original_rule_verdict'] == v['corrected_rule_verdict'] == 'RED'
    assert v['corrected_rule']['adoption_acceptance'] is False


@pytest.mark.parametrize('defect', ['metadata_retained', 'new_process', 'capture_valid', 'rt1', 'token_seats', 'negative_seats', 'bool_seats'])
def test_profile_evidence_required(defect):
    registry, rows = synthetic()
    r = target(rows)
    if defect == 'rt1': r['rt1'] = {}
    elif defect == 'token_seats': r[defect] = None
    elif defect == 'negative_seats': r['token_seats'] = -1
    elif defect == 'bool_seats': r['token_seats'] = True
    else: r[defect] = False
    assert a4.compare(rows, registry, 'sup')['corrected_rule']['adoption_acceptance'] is False


@pytest.mark.parametrize('defect', ['score', 'metadata_retained', 'new_process', 'capture_valid', 'rt1', 'token_seats'])
def test_defaults_restart_and_evidence_are_findings(defect):
    registry, rows = synthetic()
    r = target(rows, 'defaults')
    if defect == 'score': r['correct'] = not r['correct']
    elif defect == 'rt1': r['rt1'] = {}
    elif defect == 'token_seats': r['token_seats'] = None
    else: r[defect] = False
    v = a4.compare(rows, registry, 'sup')
    assert v['original_rule_verdict'] == 'RED'
    assert v['corrected_rule_verdict'] == 'GREEN'


@pytest.mark.parametrize('side', ['profile', 'defaults'])
@pytest.mark.parametrize('defect', ['missing', 'duplicate'])
def test_exact_probe_coverage_required(side, defect):
    registry, rows = synthetic()
    r = target(rows, side)
    if defect == 'missing': rows.remove(r)
    else: rows.append(copy.deepcopy(r))
    assert a4.compare(rows, registry, 'sup')['corrected_rule']['adoption_acceptance'] is False


def test_no_vacuous_empty_pass():
    registry, _ = synthetic()
    v = a4.compare([], registry, 'sup')
    assert v['corrected_rule_verdict'] == v['original_rule_verdict'] == 'RED'


def test_original_controls_required_even_above_floor():
    registry, rows = synthetic()
    for phase in ('fresh', 'restart'):
        target(rows, phase=phase)['correct'] = False
        target(rows, 'defaults', phase)['correct'] = False
    assert a4.compare(rows, registry, 'sup')['corrected_rule']['adoption_acceptance'] is False


def test_measured_controls_required():
    registry, rows = synthetic()
    lost = next(r['probe_id'] for r in rows if r['battery'] == 'sup' and r['side'] == 'defaults'
                and r['correct'] and r['probe_id'] not in registry['original_correct_controls']['sup'])
    for r in rows:
        if r['battery'] == 'sup' and r['side'] == 'profile' and r['probe_id'] == lost:
            r['correct'] = False
    assert a4.compare(rows, registry, 'sup')['corrected_rule']['adoption_acceptance'] is False


def test_registered_baseline_floor_required():
    registry, rows = synthetic()
    keep = set(registry['original_correct_controls']['sup'])
    for r in rows:
        if r['battery'] == 'sup': r['correct'] = r['probe_id'] in keep
    v = a4.compare(rows, registry, 'sup')
    assert v['corrected_rule']['controls_unchanged'] is True
    assert v['corrected_rule']['adoption_acceptance'] is False


def test_exact_fix_validation_distinct_from_adoption():
    registry, rows = synthetic()
    extra = next(i for i in registry['probe_ids']['census'] if i not in registry['original_correct_controls']['census'])
    for r in rows:
        if r['battery'] == 'census' and r['side'] == 'profile' and r['probe_id'] == extra:
            r['correct'] = True
    v = a4.compare(rows, registry, 'census')
    assert v['corrected_rule']['adoption_acceptance'] is True
    assert v['corrected_rule']['fix_validation_exact'] is False
    assert v['corrected_rule_verdict'] == 'RED'


@pytest.fixture
def copied_sup(tmp_path):
    _, registry, registration, bindings = a4.context()
    for c in registration['cells']:
        if c['battery'] != 'sup': continue
        dest = tmp_path / 'cells' / c['id']
        dest.mkdir(parents=True)
        for name in ('controller.json', 'worker.json'):
            shutil.copyfile(a4.EPOCH / 'cells' / c['id'] / name, dest / name)
    (tmp_path / 'scores').mkdir()
    shutil.copyfile(a4.EPOCH / 'scores/sup.json', tmp_path / 'scores/sup.json')
    return tmp_path, registration, registry, bindings


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_c2/a4_staging/ + artifacts/grm_c2/epochs/scout-fix-2/')
def test_worker_digest_and_epoch_binding_checked(copied_sup):
    epoch, registration, registry, bindings = copied_sup
    path = next((epoch / 'cells').glob('*/worker.json'))
    original = path.read_bytes()
    path.write_bytes(original + b' ')
    with pytest.raises(ValueError, match='worker receipt changed'):
        a4.evaluate(epoch, registration, registry, bindings, 'sup')
    path.write_bytes(original)
    with pytest.raises(ValueError, match='stale receipt binding'):
        a4.evaluate(epoch, registration, registry, dict(bindings, epoch='wrong'), 'sup')


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_c2/a4_staging/ + artifacts/grm_c2/epochs/scout-fix-2/')
def test_original_red_immutable_and_repeat_score_identical(copied_sup):
    epoch, registration, registry, bindings = copied_sup
    before = (epoch / 'scores/sup.json').read_bytes()
    first = a4.score_a4(epoch, registration, registry, bindings, ['sup'])
    after = (epoch / 'scores_a4/sup.json').read_bytes()
    assert first == a4.score_a4(epoch, registration, registry, bindings, ['sup'])
    assert (epoch / 'scores/sup.json').read_bytes() == before
    assert (epoch / 'scores_a4/sup.json').read_bytes() == after
    path = epoch / 'scores_a4/sup.json'
    path.write_text('{"status": "RED"}\n')
    with pytest.raises(ValueError, match='immutable output differs'):
        a4.score_a4(epoch, registration, registry, bindings, ['sup'])
    assert path.read_text() == '{"status": "RED"}\n'


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_c2/a4_staging/ + artifacts/grm_c2/epochs/scout-fix-2/')
def test_incomplete_all_and_summary_do_not_publish(copied_sup):
    epoch, registration, registry, bindings = copied_sup
    with pytest.raises(ValueError, match='campaign incomplete'):
        a4.score_a4(epoch, registration, registry, bindings, a4.BATTERIES)
    assert not (epoch / 'scores_a4').exists()
    with pytest.raises(ValueError, match='summary incomplete'):
        a4.summary_a4(epoch, registration, registry, bindings)
    assert not (epoch / 'scores_a4/summary_a4.md').exists()


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_c2/a4_staging/ + artifacts/grm_c2/epochs/scout-fix-2/')
def test_install_refuses_incomplete_campaign(copied_sup, monkeypatch):
    epoch, registration, registry, bindings = copied_sup
    (epoch / 'campaign.lock').touch()
    destination = epoch / 'new_script.py'
    monkeypatch.setattr(a4, 'TARGET', destination)
    with pytest.raises(ValueError, match='campaign incomplete'):
        a4.install_after_campaign(epoch, registration, bindings)
    assert not destination.exists()


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_c2/a4_staging/ + artifacts/grm_c2/epochs/scout-fix-2/')
def test_registration_bound_and_original_path_set_unchanged():
    a4.verify_amendment()
    a3 = a4.read(a4.OUT / 'amendment_lead_3.json')
    paths = sorted(str(p.relative_to(ROOT)) for d in ('core', 'scripts', 'cpp', 'config')
                   for p in (ROOT / d).rglob('*') if p.is_file()
                   and p.suffix in ('.py', '.cpp', '.hpp', '.h', '.json'))
    # Postcampaign installation is the one registered additive source.
    paths = [p for p in paths if p != 'scripts/grm_c2_score_a4.py']
    assert paths == a3['source_paths']


def test_summary_all_batteries_and_red_exit(tmp_path, monkeypatch):
    registry, rows = synthetic()
    values = {}
    for b in a4.BATTERIES:
        result = a4.compare(rows, registry, b)
        values[b] = dict(battery=b, status=result['corrected_rule_verdict'], **result)
        a4.immutable(tmp_path / 'scores_a4' / f'{b}.json', a4.encoded(values[b]))
    monkeypatch.setattr(a4, 'evaluate', lambda e, r, reg, bindings, b: values[b])
    assert a4.summary_a4(tmp_path, {}, registry, {}) == 0
    report = (tmp_path / 'scores_a4/summary_a4.md').read_bytes()
    assert report.count(b'| profile (96) |') == 3
    assert report.count(b'| defaults (256) |') == 3
    assert a4.summary_a4(tmp_path, {}, registry, {}) == 0
    assert (tmp_path / 'scores_a4/summary_a4.md').read_bytes() == report
    # Separate synthetic RED score set, never rewrite a prior summary.
    red_epoch = tmp_path / 'red'
    values['sup']['status'] = values['sup']['corrected_rule_verdict'] = 'RED'
    for b, v in values.items():
        a4.immutable(red_epoch / 'scores_a4' / f'{b}.json', a4.encoded(v))
    assert a4.summary_a4(red_epoch, {}, registry, {}) == 1


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_c2/a4_staging/ + artifacts/grm_c2/epochs/scout-fix-2/')
def test_install_complete_campaign_is_create_only(copied_sup, monkeypatch):
    epoch, registration, registry, bindings = copied_sup
    (epoch / 'campaign.lock').touch()
    for cell in registration['cells']:
        p = epoch / 'cells' / cell['id'] / 'controller.json'
        if not p.exists():
            a4.immutable(p, a4.encoded(dict(cell=cell, status='COMPLETE', **bindings)))
    for b in a4.BATTERIES:
        p = epoch / 'scores' / f'{b}.json'
        if not p.exists(): a4.immutable(p, b'{}')
    destination = epoch / 'new_script.py'
    monkeypatch.setattr(a4, 'TARGET', destination)
    a4.install_after_campaign(epoch, registration, bindings)
    assert destination.read_bytes() == SOURCE.read_bytes()
    a4.install_after_campaign(epoch, registration, bindings)
    destination.write_bytes(b'existing unrelated source')
    with pytest.raises(ValueError, match='immutable output differs'):
        a4.install_after_campaign(epoch, registration, bindings)
    assert destination.read_bytes() == b'existing unrelated source'


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_c2/a4_staging/ + artifacts/grm_c2/epochs/scout-fix-2/')
def test_install_refuses_active_campaign(copied_sup, monkeypatch):
    epoch, registration, registry, bindings = copied_sup
    path = epoch / 'campaign.lock'
    path.touch()
    monkeypatch.setattr(a4, 'TARGET', epoch / 'new_script.py')
    with path.open('rb') as stream:
        a4.fcntl.flock(stream, a4.fcntl.LOCK_EX | a4.fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            a4.install_after_campaign(epoch, registration, bindings)
    assert not (epoch / 'new_script.py').exists()


def test_forged_amendment_refused(tmp_path, monkeypatch):
    value = a4.read(a4.AMENDMENT)
    value['acceptance'] = 'always GREEN'
    path = tmp_path / 'forged.json'
    path.write_bytes(a4.encoded(value))
    monkeypatch.setattr(a4, 'AMENDMENT', path)
    with pytest.raises(ValueError, match='stale or forged amendment 4'):
        a4.verify_amendment()
