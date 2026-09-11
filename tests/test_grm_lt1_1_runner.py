"""LT1.1 runner gates — a registration is not runnable until a worker runs it.

The defect these tests exist for: D1 emitted `lead_commands.txt` describing
`scripts/grm_lt1.py --run --arm A --registration … --amendment … --out …`.
That script has no such flags. The commands were never executed, and the only
test on them checked a sha string. So the load-bearing test here is
`test_every_emitted_command_actually_runs`, which executes each emitted
command with `--dry-run` appended and requires exit 0.

Prior art: the one-subprocess-per-cell fake loop is LT1's own
(tests/test_grm_lt1_fix6.py, GRM contributors 2026). Full annotation in
scripts/grm_lt1_1.py.
"""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import grm_lt1_1 as runner  # noqa: E402


# ------------------------------------------------------------------- the CLI

def test_the_cli_accepts_what_the_lead_commands_emit():
    """Every flag the emitted commands use must actually parse."""
    for arm in runner.ARMS:
        for mode in ('--preflight', '--dry-run', '--summary', '--fake'):
            args = runner.parse_args(['--arm', arm, mode])
            assert args.arm == arm
    args = runner.parse_args(['--arm', 'A+', '--fake', '--limit', '2',
                              '--out', '/tmp/x'])
    assert args.limit == 2 and args.out == '/tmp/x'


def test_arm_is_required_and_validated():
    with pytest.raises(SystemExit):
        runner.parse_args(['--dry-run'])                 # no --arm
    with pytest.raises(SystemExit):
        runner.parse_args(['--arm', 'B', '--dry-run'])   # LT1.1 has no arm B
    with pytest.raises(SystemExit):
        runner.parse_args(['--arm', 'A'])                # no mode


def test_modes_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        runner.parse_args(['--arm', 'A', '--dry-run', '--resume'])


# ------------------------------------------------------ documents and binding

def test_preflight_is_ready_on_the_registered_documents():
    for arm in runner.ARMS:
        value = runner.preflight(arm)
        assert value['status'] == 'READY', value['reasons']
        assert value['registration_sha256'] == runner.sha(runner.REGISTRATION)
        assert value['amendment1_sha256'] == runner.sha(runner.AMENDMENT1)


def test_registration_refuses_a_broken_chain(monkeypatch, tmp_path):
    """RED proof: a drifted amendment must not silently run."""
    forged = tmp_path / 'amendment1.json'
    doc = json.loads(runner.AMENDMENT1.read_text())
    doc['registration_sha256'] = 'deadbeef' * 8
    forged.write_text(json.dumps(doc))
    (tmp_path / 'amendment1.sha256').write_text(
        runner.sha(forged) + '  amendment1.json\n')
    monkeypatch.setattr(runner, 'AMENDMENT1', forged)
    monkeypatch.setattr(runner, 'OUT', tmp_path)
    with pytest.raises(ValueError, match='LT11_AMENDMENT1_CHAIN_MISMATCH'):
        runner.registration('A')


def test_the_binding_separates_the_arms():
    a, aplus = runner.binding('A'), runner.binding('A+')
    assert a['arm'] == 'A' and aplus['arm'] == 'A+'
    assert a['alias_fold_merge'] is False
    assert aplus['alias_fold_merge'] is True
    assert a != aplus, 'a receipt must not validate under the other arm'
    for value in (a, aplus):
        assert value['registration_sha256'] == runner.sha(runner.REGISTRATION)
        assert value['fixture_sha256'] == runner.sha(runner.FIXTURE)
        assert 'core/grm_alias_fold.py' in value['rebound_core_shas']


def test_registration_supplies_what_the_lt1_worker_reads():
    """`execute` reads fields a delta registration does not restate."""
    for arm in runner.ARMS:
        reg = runner.registration(arm)
        assert reg['effective_admission_rule'] == 'margin_first'
        assert reg['restart_sentinels'] == list(runner.RESTART_SENTINELS)
        assert 'A' in reg['arms'], 'cells carry arm "A" in both variants'
        assert len(reg['cells']) == 26
        assert reg['lt1_1']['arm'] == arm


# -------------------------------------------------------------- the arm pin

def test_the_pin_is_read_back_and_restored():
    before = dict(os.environ)
    for arm, wanted in (('A', False), ('A+', True)):
        with runner.pinned_arm(arm) as pin:
            assert pin['admission_rule'] == 'margin_first'
            assert pin['alias_fold_merge'] is wanted
            from core.grm_alias_fold import alias_fold_enabled
            assert alias_fold_enabled() is wanted
    assert dict(os.environ) == before, 'the pin leaked out of its context'


def test_a_pin_that_does_not_take_is_red(monkeypatch):
    import core.grm_alias_fold as af
    monkeypatch.setattr(af, 'alias_fold_enabled', lambda *a, **k: False)
    with pytest.raises(ValueError, match='LT11_ALIAS_PIN_FAILED'):
        with runner.pinned_arm('A+'):
            pass


def test_arm_environment_pins_after_the_ambient_strip():
    """R1's contract: environment(flags) wipes GRM_*, so the pin comes after."""
    reg = runner.registration('A+')
    flags = reg['arms']['A']['flags']
    env_plus = runner.arm_environment('A+', flags)
    env_a = runner.arm_environment('A', flags)
    assert env_plus[runner.ALIAS_ENV] == '1'
    assert runner.ALIAS_ENV not in env_a
    assert env_plus[runner.RULE_ENV] == env_a[runner.RULE_ENV] == 'margin_first'
    assert set(env_plus) - set(env_a) == {runner.ALIAS_ENV}
    assert set(env_a) - set(env_plus) == set()


def test_ambient_pollution_cannot_reach_the_child():
    """A stale flag in the operator's shell must not leak into arm A."""
    reg = runner.registration('A')
    flags = reg['arms']['A']['flags']
    os.environ[runner.ALIAS_ENV] = '1'
    try:
        env = runner.arm_environment('A', flags)
        assert runner.ALIAS_ENV not in env
    finally:
        os.environ.pop(runner.ALIAS_ENV, None)


# ----------------------------------------------------------------- dry run

def test_dry_run_enumerates_twentysix_cells_per_arm():
    for arm in runner.ARMS:
        value = runner.dry_run(arm)
        assert value['status'] == 'PASS'
        assert value['gpu_executed'] is False
        assert value['cells'] == 26
        assert len(value['cell_ids']) == 26
        # `next_cell` is the FIRST INCOMPLETE cell, so 'A-001-008' holds
        # only while the arm has not been run. Both arms completed 26/26 in
        # the lead's r2 run, which makes it None. Assert the enumeration
        # (26 cells, ids, budget), and pin next_cell to whichever of the two
        # states the tree is actually in rather than to the un-run one.
        complete = value['next_cell'] is None
        assert value['next_cell'] == (None if complete else 'A-001-008')
        assert value['estimate_seconds'] > 0
        assert value['lease_seconds'] > value['estimate_seconds']
        assert value['pinned']['alias_fold_merge'] is (arm == 'A+')


def test_dry_run_reports_the_budget_honestly():
    """The lease sum is the reservation ceiling; report it, never hide it.

    This assertion is what caught the amendment-1 budget defect: a per-arm
    ceiling below the lease sum trips `run_cell`'s rail mid-campaign.
    """
    value = runner.dry_run('A')
    assert 'within_budget' in value
    assert value['within_budget'] is (
        value['lease_seconds'] <= value['budget_gpu_seconds'])


def test_the_arms_use_distinct_campaign_roots():
    assert runner.out_dir('A') != runner.out_dir('A+')
    assert runner.dry_run('A')['out_dir'] != runner.dry_run('A+')['out_dir']


# ------------------------------------------------- fake execution + summary

@pytest.mark.parametrize('arm', ['A', 'A+'])
def test_fake_path_writes_real_receipts_and_summary_reads_them(arm, tmp_path):
    root = tmp_path / ('run_' + arm.replace('+', 'plus'))
    result = runner.run_fake(arm, root=root, limit=2)
    assert result['gpu_executed'] is False
    assert len(result['cells_written']) == 2
    assert 'NOT a language-model quality measurement' in result['evidence_class']

    for cell_id in ('A-001-008', 'A-009-016'):
        directory = root / 'cells' / cell_id
        for name in ('controller.json', 'worker.json', 'reservation.json'):
            assert (directory / name).is_file(), '%s/%s' % (cell_id, name)
        assert (directory / 'checkpoint/checkpoint.json').is_file()
        assert (directory / 'checkpoint/repository/manifest.json').is_file()
        controller = json.loads((directory / 'controller.json').read_text())
        assert controller['status'] == 'COMPLETE'
        assert controller['binding'] == runner.binding(arm)

    value = runner.summary(arm, root=root)
    assert value['complete_cells'] == 2
    assert value['total_cells'] == 26
    assert value['complete'] is False
    assert value['completed_cell_ids'] == ['A-001-008', 'A-009-016']
    assert value['binding']['alias_fold_merge'] is (arm == 'A+')


def test_resume_after_interrupt_skips_completed_cells(tmp_path):
    root = tmp_path / 'resume'
    first = runner.run_fake('A', root=root, limit=2)
    assert len(first['cells_written']) == 2 and first['cells_skipped'] == []

    # Interrupt and resume: ask for three, two are already COMPLETE.
    second = runner.run_fake('A', root=root, limit=3)
    assert second['cells_skipped'] == ['A-001-008', 'A-009-016']
    assert len(second['cells_written']) == 1
    assert second['cells_written'][0].endswith('A-017-024')

    # The third cell spans the first probes, so rows now exist and score.
    value = runner.summary('A', root=root)
    assert value['complete_cells'] == 3
    assert value['measured_recalls'] == 3
    assert value['evidence_class'] == 'partial raw rows'
    assert value['by_class']['fresh']['n'] == 3


def test_a_resumed_cell_chains_to_its_predecessor_checkpoint(tmp_path):
    """The dependency contract is real: cell 2 loads cell 1's checkpoint."""
    root = tmp_path / 'chain'
    runner.run_fake('A', root=root, limit=2)
    second = json.loads((root / 'cells/A-009-016/worker.json').read_text())
    assert second['cell']['depends'] == 'A-001-008'
    assert second['previous_process_id'] is not None
    assert second['process_id'] != second['previous_process_id']
    assert second['pid'] != second['previous_pid']


def test_summary_of_an_empty_root_is_honest(tmp_path):
    value = runner.summary('A', root=tmp_path / 'nothing')
    assert value['complete_cells'] == 0
    assert value['measured_recalls'] == 0
    assert value['complete'] is False
    assert value['evidence_class'] == 'no rows'


# ------------------------------------------ the load-bearing test: it RUNS

def _emitted_commands():
    """Parse runnable `python3 scripts/...` invocations out of lead_commands."""
    path = ROOT / 'artifacts/grm_d1/lt1_1/lead_commands.txt'
    if not path.exists():
        return []
    joined = path.read_text().replace('\\\n', ' ')
    out = []
    for line in joined.splitlines():
        line = line.strip()
        if line.startswith('#') or not line or 'python3 scripts/' not in line:
            continue
        # Strip any leading `env …` / `flock …` prefix: those set the
        # environment or need a GPU lease, neither needed for a dry-run check.
        tokens = shlex.split(line)
        start = next(i for i, t in enumerate(tokens) if t.endswith('python3'))
        out.append(tokens[start:])
    return out


def test_lead_commands_emit_runnable_invocations():
    commands = _emitted_commands()
    assert commands, 'lead_commands.txt emitted no python3 script invocation'
    for tokens in commands:
        script = tokens[1]
        assert (ROOT / script).is_file(), (
            'command names a missing script: %s' % script)


def test_every_emitted_command_actually_runs():
    """THE regression test for this follow-up.

    Each emitted script invocation is executed with `--dry-run` appended (and
    any conflicting mode flag removed) and must exit 0. A command describing a
    runner that does not exist, or flags that do not parse, fails here instead
    of failing on the lead's terminal.
    """
    commands = _emitted_commands()
    assert commands
    modes = {'--resume', '--summary', '--fake', '--preflight', '--dry-run'}
    checked = 0
    for tokens in commands:
        cleaned, skip = [], False
        for token in tokens:
            if skip:
                skip = False
                continue
            if token in ('--out', '--limit'):
                skip = True
                continue
            if token in modes:
                continue
            cleaned.append(token)
        cleaned.append('--dry-run')
        result = subprocess.run([sys.executable] + cleaned[1:], cwd=ROOT,
                                capture_output=True, text=True, timeout=300)
        assert result.returncode == 0, (
            'emitted command failed:\n  %s\nstdout: %s\nstderr: %s'
            % (' '.join(cleaned), result.stdout[-2000:], result.stderr[-2000:]))
        checked += 1
    assert checked >= 2, 'expected at least one command per arm'


def test_the_old_broken_invocation_would_have_been_caught():
    """Proof the new gate has teeth: the shape D1 shipped must fail it."""
    result = subprocess.run(
        [sys.executable, 'scripts/grm_lt1.py', '--run', '--arm', 'A',
         '--registration', 'artifacts/grm_d1/lt1_1/registration.json',
         '--dry-run'],
        cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert result.returncode != 0
    assert 'unrecognized arguments' in (result.stderr + result.stdout)
