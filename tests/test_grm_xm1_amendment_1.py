"""CPU regression of the GPU execute path, not GPU numerical evidence.

Prior art: GRM contributors (2026), RS4 recorded replay and LT1 receipts;
Python tempfile scratch lifetime management. Ours: amendment regressions.
"""
import json
import tempfile
import pytest
from scripts import grm_xm1_parity as xm
from scripts import grm_rs4_ceiling_gpu as rs4
from scripts.grm_xm1_registration import read, create

REG = read(xm.REG)
GPT = [c for c in REG['cells'] if c['model'] == 'gpt-oss']
QWEN = next(c for c in REG['cells'] if c['model'] == 'qwen35')


@pytest.fixture
def gpu_path_double(tmp_path, monkeypatch):
    out = tmp_path / 'artifacts/grm_xm1'
    (out / 'cells/gpu').mkdir(parents=True)
    monkeypatch.setattr(xm, 'OUT', out)
    calls, scratch, telemetry = [], [], []
    def replay(session_id, arm):
        calls.append((session_id, arm))
        path = xm.Path(tempfile.mkdtemp(prefix='inner_rs4_'))
        scratch.append(path)
        assert path.is_dir() and not path.is_relative_to(out / 'tmp')
        assert not (out / 'tmp').exists()
        (path / 'scratch.txt').write_text('CPU fixture scratch')
        expected = next(r for key, r in REG['rs4_references'].items()
                        if key.startswith(arm + ':') and r['session_id'] == session_id)
        return read(xm.ROOT / 'artifacts/grm_rs4' / expected['receipt'])
    def memory():
        telemetry.append('CPU stub')
        return {'kind': 'CPU stub, no device queried'}
    monkeypatch.setattr(rs4, 'run_arm', replay)
    monkeypatch.setattr(xm, 'device_memory', memory)
    assert not (out / 'tmp').exists()
    return out, calls, scratch, telemetry


def test_worker_scratch_without_pytest_basetemp(gpu_path_double):
    out, calls, scratch, _ = gpu_path_double
    got = xm.execute(GPT[0], REG, out, 'gpu')
    assert got['status'] == 'PASS', got
    assert len(calls) == len(scratch) == 1
    assert not scratch[0].exists() and not (out / 'tmp').exists()


def test_failed_reference_does_not_block_next_gpt(gpu_path_double):
    out, calls, _, _ = gpu_path_double
    create(xm.cell_path(out, GPT[0], 'gpu'),
           {'binding': xm.binding(GPT[0], 'gpu'), 'status': 'ERROR'})
    got = xm.execute(GPT[1], REG, out, 'gpu')
    assert got['status'] == 'PASS', got
    assert len(calls) == 1
    assert xm.barrier(out, REG)['status'] == 'FAIL'


# GRM-H4: sha-bound: registration source drift: core/grm_admission.py (D2 flip)
@pytest.mark.campaign_receipt(registration='artifacts/grm_xm1/registration.json + artifacts/grm_xm1/registration_amendment_1.json')
def test_ten_gpu_execute_cells_then_deleted_reference_stops_qwen(gpu_path_double):
    out, calls, scratch, telemetry = gpu_path_double
    for index, cell in enumerate(GPT):
        got = xm.execute(cell, REG, out, 'gpu')
        assert got['status'] == 'PASS', got
        assert xm.cell_path(out, cell, 'gpu').is_file()
        progress = sorted((out / 'barriers/gpu').glob('*.json'))
        assert len(progress) == index + 1
        gate = read(progress[-1])['barrier']
        assert gate['status'] == ('PASS' if index == 9 else 'BLOCKED')
        assert len(gate['missing']) == 9 - index and not gate['differences']
    assert len(list((out / 'cells/gpu/gpt-oss').glob('*.json'))) == 10
    assert len(calls) == 10 and len(telemetry) == 20
    assert all(not p.exists() for p in scratch)
    assert not (out / 'tmp').exists()
    # Deletion/mutation is restricted to isolated test receipts.
    xm.cell_path(out, GPT[0], 'gpu').unlink()
    for resumed in (False, True):
        if resumed:
            create(xm.cell_path(out, QWEN, 'gpu'),
                   {'binding': xm.binding(QWEN, 'gpu'), 'status': 'PASS'})
        with pytest.raises(xm.XM1Error, match='STOP: GPT-OSS RS4 parity barrier'):
            xm.execute(QWEN, REG, out, 'gpu',
                       lambda *_: pytest.fail('barrier reached Qwen loader'))
    assert len(calls) == 10 and len(telemetry) == 20
    # CLI's existing-receipt shortcut must also enforce the same barrier.
    with pytest.raises(xm.XM1Error, match='STOP: GPT-OSS RS4 parity barrier'):
        xm.main(['--model', QWEN['model'], '--arm', QWEN['arm'],
                 '--probe', QWEN['probe'], '--output', str(out)])


def test_mismatched_reference_stops_qwen_but_not_gpt(gpu_path_double):
    out, calls, _, telemetry = gpu_path_double
    for cell in GPT:
        xm.execute(cell, REG, out, 'gpu')
    path = xm.cell_path(out, GPT[0], 'gpu')
    receipt = read(path)
    receipt['result']['reference_row']['served'] = 'mismatch'
    path.write_text(json.dumps(receipt))
    with pytest.raises(xm.XM1Error, match='parity barrier'):
        xm.execute(QWEN, REG, out, 'gpu')
    xm.cell_path(out, GPT[1], 'gpu').unlink()
    assert xm.execute(GPT[1], REG, out, 'gpu')['status'] == 'PASS'
    assert len(calls) == 11 and len(telemetry) == 22


def test_worker_scratch_cleanup_on_error(gpu_path_double, monkeypatch):
    out, _, _, _ = gpu_path_double
    paths = []
    def fail(*_):
        paths.append(xm.Path(tempfile.mkdtemp()))
        raise RuntimeError('registered scratch cleanup failure double')
    monkeypatch.setattr(rs4, 'run_arm', fail)
    got = xm.execute(GPT[0], REG, out, 'gpu')
    assert got['status'] == 'ERROR'
    assert got['error'] == 'registered scratch cleanup failure double'
    assert paths and not paths[0].exists()


# GRM-H4: sha-bound: registration source drift: core/grm_admission.py (D2 flip)
@pytest.mark.campaign_receipt(registration='artifacts/grm_xm1/registration.json + artifacts/grm_xm1/registration_amendment_1.json')
def test_amendment_binds_worker_and_preserves_original_registration():
    amendment = read(xm.AMENDMENT)
    assert xm.sha(xm.REG) == '758fd3ac1231ed44937714a86009b557f554f026e9d85773cb730074e04439e8'
    assert amendment['registration_sha256'] == xm.sha(xm.REG)
    assert amendment['previous_implementation_sha256'] == xm.sha(xm.IMPL)
    # XM2 supersedes source pins in a separate immutable registration. Keep
    # proving the original amendment against its exact preserved source.
    assert amendment['pins']['scripts/grm_xm1_parity.py'] == xm.sha(xm.ROOT / 'artifacts/grm_xm2/sources/grm_xm1_parity_amendment_1.py')
    assert read(xm.XM2_REG)['pins']['scripts/grm_xm1_parity.py'] == xm.sha(xm.ROOT / 'artifacts/grm_xm2/amendment_1/sources/scripts/grm_xm1_parity.py')
    assert xm.xm2_registration()['pins']['scripts/grm_xm1_parity.py'] == xm.sha(xm.ROOT / 'scripts/grm_xm1_parity.py')
    assert xm.binding(GPT[0], 'gpu')['amendment_sha256'] == xm.sha(xm.AMENDMENT)
    assert xm.validate_registration() == REG
    commands = (xm.ROOT / 'artifacts/grm_xm1/lead_commands.txt').read_text()
    assert 'worker leases itself' in commands and 'DO NOT use an outer flock' in commands
    assert all('flock' not in line and '--output artifacts/grm_xm1/amendment_1_run' in line
               for line in commands.splitlines() if line and not line.startswith('#'))
