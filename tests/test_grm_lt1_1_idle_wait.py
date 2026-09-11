"""LT1.1 bounded idle wait — a busy card is not a cell failure.

The 2026-09-11 lead run launched `--arm A --resume` while the A1 contrast was
still leaving the card. The single-probe check refused, and the campaign
recorded a RED cell with a charged reservation for a card somebody else was
finishing with. That is the wrong verdict: a busy card is a resource another
process holds, not a fault in our work.

These fixtures drive the policy without touching a GPU: `device_snapshot` is
replaced by a scripted sequence of samples, so "busy -> wait -> idle" and
"busy past the bound" are both exercised deterministically.

The contract under test, from R1's `await_idle`:
  * keys on TOTAL framebuffer used, never the compute-process list;
  * the bound is STRUCTURAL -- a counted loop, fixed up front;
  * it declines and it waits; it never signals, kills or waits on a process.

Prior art: R1 `await_idle` / `idle_gate` / `device_snapshot`
(`scripts/grm_r1_replay.py:352-470`, GRM contributors 2026). Full annotation
in scripts/grm_d1_amendment6.py.
"""
import ast
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import grm_lt1_1 as runner          # noqa: E402
from scripts import grm_lt1_worker as worker     # noqa: E402


def _sample(used_mib, status='OK', pids=()):
    return dict(time_unix=0.0, status=status, memory_used_mib=used_mib,
                memory_free_mib=12288 - used_mib, memory_total_mib=12288,
                other_process_pids=list(pids), compute_list_empty=not pids,
                scope='fixture')


def _script(monkeypatch, samples):
    """Feed `await_idle` a scripted sequence of device samples."""
    queue = list(samples)
    seen = []

    def probe():
        value = queue.pop(0) if len(queue) > 1 else queue[0]
        seen.append(value)
        return value
    monkeypatch.setattr(runner, 'device_snapshot', probe)
    return seen


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    import time
    monkeypatch.setattr(time, 'sleep', lambda *_: None)


# ------------------------------------------------------------- the gate

def test_an_idle_card_passes_on_the_first_probe(monkeypatch):
    seen = _script(monkeypatch, [_sample(120)])
    ok, receipt = runner.await_idle()
    assert ok is True
    assert len(seen) == 1, 'an idle card must not be polled repeatedly'
    assert receipt['reason'].startswith('IDLE:')
    assert receipt['attempts'][0]['ok'] is True


def test_busy_then_idle_waits_and_then_proceeds(monkeypatch):
    """THE fixture: busy -> wait -> idle -> the cell runs."""
    seen = _script(monkeypatch, [_sample(4096, pids=[3336818]),
                                 _sample(3144, pids=[3336818]),
                                 _sample(96)])
    ok, receipt = runner.await_idle(wait_seconds=60, poll_seconds=15)
    assert ok is True, receipt['reason']
    assert len(seen) == 3, 'it must have waited through both busy samples'
    assert [a['ok'] for a in receipt['attempts']] == [False, False, True]
    assert receipt['attempts'][0]['memory_used_mib'] == 4096
    assert receipt['reason'].startswith('IDLE:')


def test_busy_past_the_bound_declines_with_a_snapshot(monkeypatch):
    """The other half: still busy after the bound -> RED, with evidence."""
    _script(monkeypatch, [_sample(4096, pids=[3336818])])
    ok, receipt = runner.await_idle(wait_seconds=45, poll_seconds=15)
    assert ok is False
    assert 'DEVICE_BUSY' in receipt['reason']
    assert receipt['final']['memory_used_mib'] == 4096
    assert receipt['final']['other_process_pids'] == [3336818]
    assert len(receipt['attempts']) == receipt['attempts_allowed']
    # The snapshot the lead asked for is in the receipt, not just a pid.
    assert 'memory.used=4096 MiB' in receipt['reason']


def test_a_failed_probe_declines_rather_than_assuming_idle(monkeypatch):
    _script(monkeypatch, [dict(status='ERROR', error='nvidia-smi exit 9')])
    ok, receipt = runner.await_idle(wait_seconds=0)
    assert ok is False
    assert 'DEVICE_PROBE_FAILED' in receipt['reason']


def test_it_keys_on_framebuffer_not_the_compute_list():
    """FIX-8's rule: an empty compute list never means an idle card."""
    ok, reason = runner.idle_gate(_sample(3144, pids=[]))
    assert ok is False, 'a display-side holder must still block the start'
    assert 'compute_list_empty=True' in reason
    ok, reason = runner.idle_gate(_sample(64, pids=[]))
    assert ok is True


# --------------------------------------------------------- the bound

def test_the_wait_is_bounded_and_counted(monkeypatch):
    _script(monkeypatch, [_sample(8192)])
    for wait, poll, expected in ((0, 15, 1), (15, 15, 2), (900, 15, 61)):
        ok, receipt = runner.await_idle(wait_seconds=wait, poll_seconds=poll)
        assert ok is False
        assert receipt['attempts_allowed'] == expected
        assert len(receipt['attempts']) == expected


def test_the_loop_is_structurally_bounded_not_a_break():
    """No `while True` in the wait: the probe count is fixed up front."""
    source = (ROOT / 'scripts/grm_lt1_1.py').read_text()
    node = next(n for n in ast.walk(ast.parse(source))
                if isinstance(n, ast.FunctionDef) and n.name == 'await_idle')
    assert not any(isinstance(inner, ast.While) for inner in ast.walk(node)), (
        'await_idle uses a while loop; the bound must be structural'
    )
    assert any(isinstance(inner, ast.For) for inner in ast.walk(node))


def test_the_wait_never_signals_anything():
    """It declines and it waits. It does not touch another process."""
    source = (ROOT / 'scripts/grm_lt1_1.py').read_text()
    tree = ast.parse(source)
    for name in ('await_idle', 'idle_gate', 'device_snapshot'):
        node = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == name)
        for inner in ast.walk(node):
            if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute):
                assert inner.func.attr not in ('kill', 'terminate',
                                               'send_signal', 'killpg'), name
            if isinstance(inner, ast.Attribute):
                assert inner.attr not in ('SIGKILL', 'SIGTERM'), name


# ------------------------------------------------------------ the seam

def test_the_seam_is_installed_and_restored():
    assert not hasattr(worker, 'await_idle'), 'the seam leaked out'
    with runner.lt1_1_seams('A+'):
        assert worker.await_idle is runner.await_idle
    assert not hasattr(worker, 'await_idle'), 'the seam was not restored'


def test_the_lt1_default_branch_is_byte_identical():
    """LT1's own campaign keeps its single-probe refusal, unchanged."""
    source = (ROOT / 'scripts/grm_lt1_worker.py').read_text()
    assert ("busy=subprocess.run(['nvidia-smi','--query-compute-apps=pid',"
            "'--format=csv,noheader'],capture_output=True,text=True)") in source
    assert ("if busy.returncode or busy.stdout.strip():"
            "raise ValueError('GPU_NOT_IDLE: '+busy.stdout.strip())") in source


def test_run_cell_raises_only_after_the_bound(monkeypatch, tmp_path):
    """Through `run_cell`: a busy card past the bound is a RED with evidence."""
    import scripts.grm_cmc1_gpu_arms as arms
    lock = tmp_path / 'lock'
    lock.touch()
    monkeypatch.setattr(arms, 'LOCK_PATH', lock)
    root = tmp_path / 'run_A'
    root.mkdir()
    reg = runner.registration('A')
    with runner.lt1_1_seams('A'), runner.pinned_arm('A'):
        monkeypatch.setattr(worker, 'RUN', root)
        monkeypatch.setattr(worker, 'accounting', lambda *a, **k: 0.0)
        monkeypatch.setattr(worker.time, 'sleep', lambda *_: None)
        monkeypatch.setattr(runner, 'device_snapshot',
                            lambda: _sample(4096, pids=[3336818]))
        monkeypatch.setattr(worker, 'await_idle',
                            lambda: runner.await_idle(wait_seconds=30,
                                                      poll_seconds=15))
        assert worker.run_cell(reg['cells'][0], reg) is False

    controller = json.loads(
        (root / 'cells/A-001-008/controller.json').read_text())
    assert controller['status'] == 'RED'
    assert 'GPU_NOT_IDLE' in controller['error']
    # The snapshot is attached, not just a bare pid as in the lead run.
    assert 'memory_used_mib' in controller['error']
    assert 'attempts_allowed' in controller['error']


# --------------------------------------------- the archived spurious RED

def test_the_spurious_red_cell_is_archived_and_arm_a_rearmed():
    archive = runner.OUT / 'archive' / 'A_A-001-008_GPU_NOT_IDLE'
    if not archive.exists():
        pytest.skip('archive not created yet; run scripts/grm_d1_amendment6.py')
    controller = json.loads((archive / 'controller.json').read_text())
    assert controller['status'] == 'RED'
    assert 'GPU_NOT_IDLE' in controller['error']
    # No worker.log: the child was never spawned, which is the tell that the
    # card refused before any work began.
    assert not (archive / 'worker.log').exists()
    live = runner.OUT / 'run_A' / 'cells' / 'A-001-008'
    # The re-arm assertion below describes the state right after the
    # archive was taken, not an invariant of the tree. The lead has since
    # run LT1.1 r2 to completion (26/26 both arms), so the live cell
    # legitimately exists again. Assert the re-arm only while the arm has
    # not been re-run; the archive check above is the part that must always
    # hold.
    if (live / 'controller.json').exists():
        pytest.skip('arm A has been re-run since the archive (r2 complete)')
    assert not live.exists(), 'arm A did not re-arm'
