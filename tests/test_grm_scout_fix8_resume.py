"""CPU author baseline. Prior art: GRM C7/FIX8 controller doubles (2026).
New boundary/failure cases for the exact registered successor; no prior art
known to me for this composition. No device queries or GPU execution.
"""
from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts import grm_scout_fix8_resume as run


def xml(used=275, processes='', total=12282):
    return f'''<nvidia_smi_log><gpu><uuid>GPU-test</uuid>
    <fb_memory_usage><used>{used} MiB</used><total>{total} MiB</total></fb_memory_usage>
    <processes>{processes}</processes></gpu></nvidia_smi_log>'''


def process(pid=42, memory=100, kind='G'):
    return f'''<process_info><pid>{pid}</pid><type>{kind}</type>
    <process_name>fixture</process_name><used_memory>{memory} MiB</used_memory></process_info>'''


@pytest.fixture(autouse=True)
def forbid_live_probe(monkeypatch):
    def forbidden(*a, **kw):
        raise AssertionError('LIVE_SUBPROCESS_FORBIDDEN_IN_CPU_TEST')
    monkeypatch.setattr(run.subprocess, 'run', forbidden)


@pytest.mark.parametrize('used,allowed', [(0, True), (999, True), (1000, True), (1001, False)])
def test_memory_boundary_and_graphics_pids(used, allowed):
    value = dict(run.parse_memory(xml(used, process()), '0', 99), status='OK')
    assert value['pids'] == value['other_process_pids'] == [42]
    assert value['processes'][0]['type'] == 'G'
    if allowed:
        run.memory_gate(value, 1000)
    else:
        with pytest.raises(ValueError, match=r'DEVICE_MEMORY_BUSY.*pids=\[42\]'):
            run.memory_gate(value, 1000)


@pytest.mark.parametrize('data,visible', [
    ('garbage', '0'), (xml().replace('275 MiB', 'N/A'), '0'),
    (xml().replace('<processes></processes>', ''), '0'),
    (xml(processes='N/A'), '0'), (xml(processes=process(memory='N/A')), '0'),
    (xml(13000), '0'), (xml(-1), '0'), (xml(), ''), (xml(), '1'),
    (xml(), '0,1'), (xml().replace('</nvidia_smi_log>', '<gpu/></nvidia_smi_log>'), '0')])
def test_malformed_unknown_or_wrong_device_fails(data, visible):
    with pytest.raises((ValueError, run.ET.ParseError)):
        run.parse_memory(data, visible, 99)


def test_self_allocation_rejected_and_uuid_supported():
    value = dict(run.parse_memory(xml(processes=process(pid=99)), 'GPU-test', 99), status='OK')
    with pytest.raises(ValueError, match='REQUIRES_FRESH_GPU_WORKER'):
        run.memory_gate(value, 1000)


@pytest.mark.parametrize('mode', ['ok', 'command_error', 'timeout', 'malformed'])
def test_probe_receipts_and_fail_closed(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(run, 'binding', lambda: {})
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    def fake(cmd, **kw):
        assert cmd == ['nvidia-smi', '-q', '-x'] and kw['timeout'] == 5
        if mode == 'timeout':
            raise run.subprocess.TimeoutExpired(cmd, 5)
        return SimpleNamespace(returncode=1 if mode == 'command_error' else 0,
                               stdout='bad' if mode == 'malformed' else xml(), stderr='fixture')
    monkeypatch.setattr(run.subprocess, 'run', fake)
    path = tmp_path / 'probe.json'
    value = run.snapshot(path, 'before_lease')
    assert run.read(path) == value
    if mode == 'ok':
        run.memory_gate(value, 1000)
    else:
        with pytest.raises(ValueError, match='DEVICE_MEMORY_PROBE_FAILED'):
            run.memory_gate(value, 1000)
    with pytest.raises(FileExistsError):
        run.snapshot(path, 'overwrite')


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    base = run.original.read(run.original.REG)
    original_out = tmp_path / 'original'
    original_out.mkdir()
    out = tmp_path / 'resume'
    out.mkdir()
    requests = run.read(run.original.OUT / 'requests.json')
    run.write(original_out / 'requests.json', requests)
    run.write(original_out / 'gpu/F5/controller.json', {'status': 'FAILED', 'error': 'original OOM'})
    r = {'batches': {
        'F5-R1': {'probe_ids': base['batches']['F5'], 'lease_seconds': 120},
        'F6': {'probe_ids': base['batches']['F6'], 'lease_seconds': 280}},
        'memory_limit_mib': 1000, 'historical_charged_seconds': 1400, 'gpu_cap_seconds': 1800}
    monkeypatch.setattr(run, 'OUT', out)
    monkeypatch.setattr(run.original, 'OUT', original_out)
    monkeypatch.setattr(run, 'verify', lambda: (r, base))
    monkeypatch.setattr(run, 'binding', lambda: {'registration_sha256': run.sha(run.original.REG),
                                               'resume_registration_sha256': 'fixture'})
    monkeypatch.setattr(run.shutil, 'disk_usage', lambda _: SimpleNamespace(free=100 * 1024**3))
    monkeypatch.setattr(run.time, 'sleep', lambda _: None)
    # Save/restore environment even though execute rewrites GRM variables.
    for k in list(os.environ):
        if k.startswith('GRM_'):
            monkeypatch.delenv(k)
    from scripts import grm_c2_cells as cells
    monkeypatch.setattr(cells, 'environment', lambda _: {})
    events = []
    def snap(path, stage):
        events.append(stage)
        value = dict(run.parse_memory(xml(), '0', 99), status='OK', stage=stage)
        run.write(path, value)
        return value
    monkeypatch.setattr(run, 'snapshot', snap)
    from scripts import grm_cmc1_gpu_arms as leases
    from scripts import grm_c7_middle_replay as middle
    @contextmanager
    def lease(seconds, wait):
        assert wait == 0
        events.append(('lease', seconds))
        yield
        events.append('release')
    monkeypatch.setattr(leases, 'gpu_lease', lease)
    repo = SimpleNamespace(close=lambda: events.append('close'))
    def open_copy(q, d, base, loaded):
        events.append('open_copy')
        return repo, ('model', 'tokenizer', {'fixture': True})
    monkeypatch.setattr(middle, 'open_copy', open_copy)
    def serve(repo, q, d):
        events.append('serve')
        run.write(d / (q['probe_id'] + '.json'), row(q['probe_id']))
    monkeypatch.setattr(run.original, 'serve', serve)
    return r, base, events


def row(pid):
    return {'probe_id': pid, 'registration_sha256': run.sha(run.original.REG),
            'admitted': True, 'identifier_unbound': False, 'mounted_ids': [0],
            'score': {k: int(k == 'exact_correct') for k in
                ('exact_correct', 'wrong_value_error', 'abstention_error', 'unsupported_answer_error')}}


def test_success_once_order_budget_and_original_untouched(campaign):
    r, base, events = campaign
    before = (run.original.OUT / 'gpu/F5/controller.json').read_bytes()
    with pytest.raises(ValueError, match='PRIOR_BATCH_INCOMPLETE'):
        run.batch('F6')
    run.batch('F5-R1')
    assert events.index('before_lease') < events.index(('lease', 120))
    assert events.index('under_lock_before_load') < events.index('open_copy')
    assert run.campaign_state(r) == (1520, ['F5-R1'])
    with pytest.raises(ValueError, match='SUCCESSOR_ALREADY_CONSUMED'):
        run.batch('F5-R1')
    run.batch('F6')
    assert run.campaign_state(r) == (1800, ['F5-R1', 'F6'])
    assert (run.original.OUT / 'gpu/F5/controller.json').read_bytes() == before
    assert not (run.original.OUT / 'replay.active').exists()
    for b in ('F1', 'F2', 'F3', 'F4'):
        for pid in base['batches'][b]:
            run.write(run.original.OUT / 'gpu' / b / (pid + '.json'), row(pid))
    result = run.summary()
    assert result['complete'] and result['rows'] == 24 and result['charged_seconds'] == 1800
    assert result['status'] == 'RED' and result['replay_status'] == 'PASS_ADMISSION'
    assert result['historical_RED_preserved'] and not result['historical_partial_F5_rows_used']


@pytest.mark.parametrize('under_lock', [False, True])
def test_busy_before_load_and_attempt_consumption(campaign, monkeypatch, under_lock):
    r, _, events = campaign
    previous = run.snapshot
    def busy(path, stage):
        value = previous(path, stage)
        if stage == ('under_lock_before_load' if under_lock else 'before_lease'):
            value['memory.used'] = 1001
        return value
    monkeypatch.setattr(run, 'snapshot', busy)
    with pytest.raises((ValueError, RuntimeError), match='DEVICE_MEMORY_BUSY'):
        run.batch('F5-R1')
    assert 'open_copy' not in events
    assert bool(list((run.OUT / 'probes').glob('*.json')))
    if under_lock:
        c = run.read(run.OUT / 'gpu/F5-R1/controller.json')
        assert c['charged_seconds'] == 120 and c['status'] == 'FAILED'
        with pytest.raises(ValueError, match='FAILED_SUCCESSOR_CAMPAIGN_STOP'):
            run.batch('F5-R1')
    else:
        assert ('lease', 120) not in events and not (run.OUT / 'gpu/F5-R1').exists()
        monkeypatch.setattr(run, 'snapshot', previous)
        run.batch('F5-R1')  # no lease was consumed by refusal


@pytest.mark.parametrize('failure', ['load', 'serve', 'lease'])
def test_failure_diagnostics_and_stop(campaign, monkeypatch, failure):
    r, _, events = campaign
    from scripts import grm_c7_middle_replay as middle
    from scripts import grm_cmc1_gpu_arms as leases
    def fail(*a, **kw):
        raise RuntimeError('cudaMalloc failed: out of memory')
    if failure == 'load':
        monkeypatch.setattr(middle, 'open_copy', fail)
    elif failure == 'serve':
        monkeypatch.setattr(run.original, 'serve', fail)
    else:
        monkeypatch.setattr(leases, 'gpu_lease', fail)
    with pytest.raises(RuntimeError, match='cudaMalloc failed: out of memory'):
        run.batch('F5-R1')
    c = run.read(run.OUT / 'gpu/F5-R1/controller.json')
    assert c['status'] == 'FAILED' and c['charged_seconds'] == 120
    assert c['rows_completed'] == [] and len(c['rows_attempted']) == (0 if failure == 'lease' else 1)
    assert 'in fail' in c['traceback'] and c['historical_RED_preserved']
    if failure == 'serve':
        assert events.index(c['stage'] + ':failure_before_close') < events.index('close')
    with pytest.raises(ValueError, match='FAILED_SUCCESSOR_CAMPAIGN_STOP'):
        run.batch('F6')


def test_owner_orphan_unknown_and_budget_stops(campaign):
    r, _, events = campaign
    owner = run.original.OUT / 'replay.active'
    owner.write_text('fixture other owner')
    with pytest.raises(FileExistsError):
        run.batch('F5-R1')
    assert owner.read_text() == 'fixture other owner'
    owner.unlink()  # fixture owns this file
    r['historical_charged_seconds'] = 1401
    with pytest.raises(ValueError, match='GPU_BUDGET_RAIL'):
        run.batch('F5-R1')
    assert events == []
    r['historical_charged_seconds'] = 1400
    (run.OUT / 'gpu/F5-R1').mkdir(parents=True)
    with pytest.raises(ValueError, match='ORPHAN_RESERVATION_STOP'):
        run.batch('F5-R1')
    (run.OUT / 'gpu/foreign').mkdir()
    with pytest.raises(ValueError, match='UNKNOWN_RESUME_BATCH'):
        run.batch('F5-R1')


@pytest.mark.parametrize('change,match', [
    ({'status': 'FAILED'}, 'FAILED_SUCCESSOR_CAMPAIGN_STOP'),
    ({'charged_seconds': 119}, 'INVALID_CHARGE'),
    ({'charged_seconds': float('nan')}, 'INVALID_CHARGE'),
    ({'resume_registration_sha256': 'foreign'}, 'RESUME_RESULT_BINDING_MISMATCH'),
    ({'rows_completed': []}, 'INCOMPLETE_BATCH_ROWS')])
def test_tampered_successor_fails(campaign, change, match):
    r, _, _ = campaign
    run.batch('F5-R1')
    p = run.OUT / 'gpu/F5-R1/controller.json'
    value = run.read(p)
    value.update(change)
    p.write_text(json.dumps(value))  # mutate only our CPU fixture
    with pytest.raises(ValueError, match=match):
        run.campaign_state(r)


def test_overrun_is_red_and_charged_in_full(campaign, monkeypatch):
    r, _, _ = campaign
    ticks = iter([0, 121, 122])
    monkeypatch.setattr(run.time, 'monotonic', lambda: next(ticks))
    with pytest.raises(RuntimeError, match='LEASE_OVERRUN_RED'):
        run.batch('F5-R1')
    c = run.read(run.OUT / 'gpu/F5-R1/controller.json')
    assert c['status'] == 'FAILED' and c['charged_seconds'] == 122
    with pytest.raises(ValueError, match='FAILED_SUCCESSOR_CAMPAIGN_STOP'):
        run.batch('F6')


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_scout_fix8/resume_amendment_1/registration.json')
def test_live_registration_and_old_red_stop():
    r, base = run.verify()
    assert r['historical_charged_seconds'] + sum(s['lease_seconds'] for s in r['batches'].values()) == 1800
    with pytest.raises(ValueError, match='FAILED_CAMPAIGN_STOP'):
        run.original.campaign_state(base)
    assert run.summary()['historical_RED_preserved']


@pytest.mark.campaign_receipt(
    registration='artifacts/grm_scout_fix8/resume_amendment_1/registration.json')
@pytest.mark.parametrize('suffix,match', [
    ('scripts/grm_scout_fix8_resume.py', 'RESUME_INPUT_SHA_MISMATCH'),
    ('gpu/F5/controller.json', 'HISTORICAL_RECEIPT_CHANGED')])
def test_registered_source_and_historical_drift_rejected(monkeypatch, suffix, match):
    original_sha = run.sha
    monkeypatch.setattr(run, 'sha', lambda p: 'drift' if str(p).endswith(suffix) else original_sha(p))
    with pytest.raises(ValueError, match=match):
        run.verify()
