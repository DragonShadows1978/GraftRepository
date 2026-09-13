"""LT1.1 child-spawn gates — the path `--dry-lease` stops short of.

WHAT WENT WRONG. A green parent produced a RED cell. `--arm A+ --resume`
pinned the arm, started A-001-008, and the child died:

    scripts/grm_lt1_worker.py:289  __main__ -> lt.verify()
    ValueError: INPUT_SHA_MISMATCH: core/graft_arena.py

`run_cell` hard-coded `Popen([... '-m', 'scripts.grm_lt1_worker', '--worker',
id])`, so the CHILD ran LT1's chain -- the gate the lead's ruling removed from
the parent.

WHY THE EXISTING GATES MISSED IT, and what this file does about each:

  * `--dry-lease` stops AT the lease boundary, before `run_cell` spawns.
    -> These tests call the REAL `run_cell`, with only the lease and the model
       stubbed.
  * the `--fake` proof used `--fake-cell`, which never enters `run_cell`.
    -> `test_the_fake_cell_path_is_not_the_run_cell_path` pins that
       distinction so the two can never again be confused for each other.
  * a docstring ASSERTED the child came back through the LT1.1 module.
    -> `test_run_cell_spawns_the_lt1_1_child` reads the argv `run_cell`
       actually builds, rather than trusting prose.

Prior art: the lease/child contract is `scripts/grm_lt1_worker.py`'s (GRM
contributors, 2026). Full annotation in scripts/grm_d1_amendment5.py.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import grm_lt1_1 as runner          # noqa: E402
from scripts import grm_lt1_worker as worker     # noqa: E402


def _stub_lease(monkeypatch, tmp_path):
    """Replace the shared GPU lease with a flock on a temp file.

    ONLY the lease and the GPU idle check are stubbed. `run_cell`'s reservation
    accounting, directory creation, argv construction, Popen, foreground wait,
    charge computation, checkpoint validation and controller receipt all run
    unchanged -- that is what makes this a gate on the real route.

    The shared lock is never opened and no GPU is touched.
    """
    lock = tmp_path / 'fake_gpu.lock'
    lock.touch()
    import scripts.grm_cmc1_gpu_arms as arms
    monkeypatch.setattr(arms, 'LOCK_PATH', lock)

    class Idle:
        returncode = 0
        stdout = ''
    real_run = subprocess.run

    def fake_run(argv, *a, **kw):
        if argv and 'nvidia-smi' in argv[0]:
            return Idle()
        return real_run(argv, *a, **kw)
    monkeypatch.setattr(worker.subprocess, 'run', fake_run)
    # The idle policy (amendment 6) probes the card through
    # `runner.device_snapshot`, not the old compute-app query, so report an
    # idle card there too. Stubbing the PROBE rather than `await_idle` keeps
    # the bounded-wait logic itself on the real path.
    monkeypatch.setattr(runner, 'device_snapshot',
                        lambda: dict(time_unix=0.0, status='OK',
                                     memory_used_mib=64, memory_free_mib=12224,
                                     memory_total_mib=12288,
                                     other_process_pids=[],
                                     compute_list_empty=True,
                                     scope='gate stub: idle card'))
    return lock


def _child_argv_using_the_cpu_double(cell):
    """The LT1.1 child argv with `--worker-cpu` instead of `--worker`.

    Identical route: the same `main()` branch shape, the same chain preflight,
    the same seams, the same `worker.execute` into the directory `run_cell`
    already created, the same create-only `worker.json`. The ONLY difference
    is the model, because this machine has no GPU for a gate to use.
    """
    return [sys.executable, '-m', 'scripts.grm_lt1_1',
            '--arm', runner.current_arm(), '--worker-cpu', cell['id']]


# ------------------------------------------------- the argv, read not trusted

def test_run_cell_spawns_the_lt1_1_child():
    """The claim a docstring got wrong: verify the argv `run_cell` builds."""
    with runner.lt1_1_seams('A+'):
        argv = worker.spawn_argv({'id': 'A-001-008'})
    assert argv[1:3] == ['-m', 'scripts.grm_lt1_1'], argv
    assert '--worker' in argv and 'A-001-008' in argv
    assert '--arm' in argv and 'A+' in argv
    assert 'scripts.grm_lt1_worker' not in argv


def test_the_default_argv_is_unchanged_for_lt1():
    """LT1's own campaign must be byte-identical without the seam."""
    assert not hasattr(worker, 'spawn_argv'), (
        'the seam leaked outside lt1_1_seams; LT1 campaigns would change')
    source = (ROOT / 'scripts/grm_lt1_worker.py').read_text()
    assert "[sys.executable,'-m','scripts.grm_lt1_worker','--worker',cell['id']]" \
        in source, 'the default child argv is no longer LT1 own'


def test_spawn_env_carries_the_arm_across_the_boundary():
    for arm in runner.ARMS:
        with runner.lt1_1_seams(arm):
            env = worker.spawn_env({}, {'id': 'A-001-008'})
        assert env[runner.ARM_ENV] == arm
        assert env[runner.RULE_ENV] == 'margin_first'
        # GRM-D2: both arms pin the flag EXPLICITLY.  Absence used to mean
        # OFF; A1 now ships ON, so an absent variable would carry the
        # treatment across the boundary into the control arm.
        assert env[runner.ALIAS_ENV] == ('1' if arm == 'A+' else '0')


def test_every_lt_verify_reach_point_is_covered():
    """The audit, re-derived from source rather than quoted."""
    from scripts import grm_d1_amendment5 as am5
    rows = am5.verify_reach_audit()
    assert rows, 'no lt.verify() call sites found; the audit is stale'
    for row in rows:
        assert row['disposition'].startswith(('covered', 'not reachable')), row
        assert 'UNCLASSIFIED' not in row['disposition'], (
            'a new lt.verify() call site appeared at line %d in %s; classify '
            'it before shipping' % (row['line'], row['function']))
    functions = {row['function'] for row in rows}
    assert functions == {'worker', 'resume', '__main__'}, functions

    # The audit counts CALLS, not mentions. An earlier substring version
    # reported a comment as a fourth site, so pin the distinction: the module
    # discusses lt.verify() in prose more often than it calls it.
    text = (ROOT / 'scripts/grm_lt1_worker.py').read_text()
    assert text.count('lt.verify()') > len(rows), (
        'the module no longer discusses lt.verify() in prose; this test '
        'should be re-examined rather than left silently passing')


def test_the_fake_cell_path_is_not_the_run_cell_path():
    """`--fake-cell` bypasses run_cell; that is why it proved nothing here."""
    source = (ROOT / 'scripts/grm_lt1_1.py').read_text()
    assert '--fake-cell' in source
    fake = source[source.index('def run_fake('):source.index('def resume(')]
    assert 'run_cell' not in fake, (
        'run_fake now enters run_cell; this test and its comment need '
        'rewriting rather than silently passing')


# ------------------------------------------- the child body runs OUR chain

def test_the_child_entry_point_verifies_our_chain_not_lt1s():
    source = (ROOT / 'scripts/grm_lt1_1.py').read_text()
    branch = source[source.index('if args.worker:'):
                    source.index('if args.fake_cell:')]
    assert 'lt1_1_preflight(args.arm)' in branch
    assert 'LT11_CHILD_PREFLIGHT_BLOCKED' in branch
    assert 'lt.verify' not in branch


def test_lt1_1_worker_substitutes_only_the_registration_source():
    """It must keep the lease-parent contract and the no-kill deadline."""
    import ast
    source = (ROOT / 'scripts/grm_lt1_1.py').read_text()
    tree = ast.parse(source)
    bodies = {node.name: node for node in ast.walk(tree)
              if isinstance(node, ast.FunctionDef)
              and node.name.startswith('lt1_1_worker')}
    assert set(bodies) == {'lt1_1_worker', 'lt1_1_worker_cpu'}, sorted(bodies)

    for name, node in bodies.items():
        # Strip the docstring: it EXPLAINS the substitution, so a text search
        # over it would match `lt.verify` and "no kill syscall".
        statements = [s for s in node.body
                      if not (isinstance(s, ast.Expr)
                              and isinstance(s.value, ast.Constant)
                              and isinstance(s.value.value, str))]
        code = '\n'.join(ast.unparse(s) for s in statements)
        assert 'WORKER_REQUIRES_LEASE_PARENT' in code, name
        assert "signal.alarm(cell['worker_seconds'])" in code, name
        assert 'registration(arm)' in code, name
        assert 'lt.verify' not in code, '%s calls LT1 verify' % name
        # No signalling, checked structurally.
        for inner in ast.walk(node):
            if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute):
                assert inner.func.attr not in ('kill', 'terminate',
                                               'send_signal'), (
                    '%s signals a process' % name)
            if isinstance(inner, ast.Attribute):
                assert inner.attr not in ('SIGKILL', 'SIGTERM'), (
                    '%s names a killing signal' % name)


@pytest.mark.parametrize('arm', ['A', 'A+'])
def test_the_child_entry_point_blocks_on_a_drifted_chain(arm):
    """A drifted LT1.1 chain must stop the CHILD, not just the parent."""
    env = dict(os.environ)
    env[runner.ARM_ENV] = arm
    env['GRM_LT1_LEASE_PARENT'] = str(os.getpid())
    env.pop(runner.ALIAS_ENV, None)
    # An unpinned rule is the cheapest drift to inject without touching files.
    env['GRM_ADMISSION_RULE'] = 'all_tokens_bind'
    result = subprocess.run(
        [sys.executable, '-m', 'scripts.grm_lt1_1', '--arm', arm,
         '--worker', 'A-001-008'],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode != 0
    assert 'LT11_CHILD_PREFLIGHT_BLOCKED' in result.stderr


# ----------------------------------------- THE gate: spawn through run_cell

@pytest.mark.parametrize('arm', ['A', 'A+'])
def test_run_cell_spawns_a_child_that_writes_receipts(arm, monkeypatch,
                                                      tmp_path):
    """Cells 1-2 through the REAL `run_cell`, lease and model stubbed.

    This is the gate the lead asked for: it exercises the spawn that
    `--dry-lease` stops before and that `--fake-cell` bypasses.
    """
    _stub_lease(monkeypatch, tmp_path)
    root = tmp_path / ('run_' + arm.replace('+', 'plus'))
    reg = runner.registration(arm)

    with runner.lt1_1_seams(arm), runner.pinned_arm(arm):
        monkeypatch.setattr(worker, 'RUN', root)
        # The child runs the CPU double; everything else is the real route.
        monkeypatch.setattr(worker, 'spawn_argv',
                            _child_argv_using_the_cpu_double)
        monkeypatch.setattr(worker, 'accounting', lambda *a, **k: 0.0)
        monkeypatch.setattr(worker.time, 'sleep', lambda *_: None)
        for cell in reg['cells'][:2]:
            assert worker.run_cell(cell, reg) is True, cell['id']

    for cell_id in ('A-001-008', 'A-009-016'):
        directory = root / 'cells' / cell_id
        for name in ('controller.json', 'worker.json', 'reservation.json'):
            assert (directory / name).is_file(), '%s/%s' % (cell_id, name)
        assert (directory / 'checkpoint/checkpoint.json').is_file()
        controller = json.loads((directory / 'controller.json').read_text())
        assert controller['status'] == 'COMPLETE', controller.get('error')

        # The defect, by name: the child log must be clean.
        log = (directory / 'worker.log').read_text()
        assert 'INPUT_SHA_MISMATCH' not in log, log[-2000:]
        assert 'Traceback' not in log, log[-2000:]

    # The receipt carries the right arm, written by the child.
    receipt = json.loads((root / 'cells/A-009-016/worker.json').read_text())
    assert receipt['binding']['campaign_arm'] == arm
    assert receipt['binding']['alias_fold_merge'] is (arm == 'A+')
    # RESTART_REQUIRES_NEW_PROCESS: the second cell ran in its own process.
    assert receipt['pid'] != receipt['previous_pid']
    assert receipt['process_id'] != receipt['previous_process_id']
    assert receipt['pid'] != os.getpid(), 'the child was not a subprocess'


def test_the_old_argv_reproduces_the_leads_worker_exit_1():
    """RED-before: the argv that shipped must still fail, by name.

    Spawns `-m scripts.grm_lt1_worker --worker <cell>` -- the exact command
    `run_cell` used to build -- and requires the child to die with
    INPUT_SHA_MISMATCH, reproducing `run_Aplus/cells/A-001-008/worker.log`.
    """
    env = dict(os.environ)
    env['GRM_LT1_LEASE_PARENT'] = str(os.getpid())
    env['GRM_ADMISSION_RULE'] = 'margin_first'
    result = subprocess.run(
        [sys.executable, '-m', 'scripts.grm_lt1_worker', '--worker',
         'A-001-008'],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    combined = result.stdout + result.stderr
    assert result.returncode != 0, 'the old argv no longer fails; re-check'
    assert 'INPUT_SHA_MISMATCH' in combined, combined[-2000:]
    assert 'core/graft_arena.py' in combined


def test_the_archived_red_cell_is_preserved():
    """The lead-run failure is kept as evidence, create-only."""
    archive = runner.OUT / 'archive' / 'Aplus_A-001-008_RED'
    if not archive.exists():
        pytest.skip('archive not created yet; run scripts/grm_d1_amendment5.py')
    log = (archive / 'worker.log').read_text()
    assert 'INPUT_SHA_MISMATCH' in log, 'the archived RED evidence was altered'
    controller = json.loads((archive / 'controller.json').read_text())
    assert controller['status'] == 'RED'
    # ... and the live cell was cleared so the arm re-arms from cell 1.
    # The re-arm assertion below describes the state right after the
    # archive was taken, not an invariant of the tree. The lead has since
    # run LT1.1 r2 to completion (26/26 both arms), so the live cell
    # legitimately exists again. Assert the re-arm only while the arm has
    # not been re-run; the archive check above is the part that must always
    # hold.
    live = runner.OUT / 'run_Aplus' / 'cells' / 'A-001-008'
    if (live / 'controller.json').exists():
        pytest.skip('arm A+ has been re-run since the archive (r2 complete)')
    assert not live.exists(), 'arm A+ did not re-arm'
