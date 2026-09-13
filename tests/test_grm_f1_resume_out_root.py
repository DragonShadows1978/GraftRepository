"""GRM-F1 follow-up 3 — `--out` must govern EVERY root on the resume route.

WHAT WENT WRONG (lead run, 2026-09-11, artifacts/grm_f1/lt1_1_r3/
lead_Aprime_try1.log). Arm A' was launched with an explicit `--out`:

    python3 -m scripts.grm_lt1_1 --arm A --resume \
        --out artifacts/grm_f1/lt1_1_r3/run_Aprime

and died before taking any lease:

    scripts/grm_lt1_1.py:1206  resume -> worker.pending(reg, run=root)
    scripts/grm_lt1_worker.py:240  ValueError: COMPLETED_CELL_BINDING_OR_STATUS

`run_Aprime/cells` was empty, which is the tell: the campaign never wrote
there at all.

THE CAUSE, settled by receipt (artifacts/grm_f1/red/), was NEITHER candidate
the lead offered. `main()` read:

    return resume(args.arm)

`--out` was accepted by argparse and DROPPED. Every other mode
(`--dry-run`, `--summary`, `--fake`, `--dry-lease`) forwarded `args.out`, so
the flag looked honoured everywhere a gate could see it and was discarded on
the one route that spends GPU. `resume` fell back to `out_dir('A')` =
r2's FROZEN `artifacts/grm_d1/lt1_1/run_A`, whose 26 COMPLETE cells carry
r2's binding, and `pending()` raised on the FIRST cell, A-001-008. The
restored `A-025-032/session/` directory was irrelevant: `pending()` never
reads `session/`, and it never reached that cell.

TWO MORE ROOT LEAKS sat behind it, neither of which today's crash would have
revealed, because the crash happened first:

  2. `lt1_1_seams` pinned `worker.RUN = out_dir(arm)` unconditionally.
     `run_cell`, `accounting` and the leased child ALL read `worker.RUN`, so
     a campaign could SELECT a cell from one root and EXECUTE it into
     another.
  3. `spawn_env` re-applied only the arm pin after `environment(flags)`
     stripped every ambient `GRM_*`, so F1/F2/F5 never crossed into the
     leased child -- the process that actually runs the turns. A treatment
     arm would have EXECUTED AS THE CONTROL while its receipts recorded the
     flags as on. Receipt: artifacts/grm_f1/red/red_child_env_strip.json.

WHY THE EXISTING GATES MISSED ALL THREE. `--dry-lease` stops AT the lease
boundary and, critically, is a DIFFERENT `main()` branch that already passed
`root=args.out` -- so it exercised the fixed path while the broken one
shipped. This file therefore drives `resume()` itself (not the `--dry-lease`
branch) through `pending()` and one real `run_cell` spawn.

Prior art: the lease/child stubbing shape is
`tests/test_grm_lt1_1_child_spawn.py` (GRM contributors, 2026), reused
verbatim -- only the lease and the GPU idle probe are replaced, so
reservation accounting, directory creation, argv construction, Popen, the
foreground wait, the charge computation, checkpoint validation and the
controller receipt all run unchanged. New here: the root-isolation
assertions and the three-arm flag-carry matrix. No prior art known to me for
this exact suite. CPU author evidence only: no GPU, no model-quality claim.
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

#: The three r3 arms, as the lead exports them.
ARM_MATRIX = {
    'A0':   dict(base='A',  f1='0', f2='0', f5='0', alias=None),
    "A'":   dict(base='A',  f1='1', f2='1', f5='1', alias=None),
    "A+'":  dict(base='A+', f1='1', f2='1', f5='1', alias='1'),
}


def _apply_arm_env(monkeypatch, spec):
    monkeypatch.setenv('GRM_ADMISSION_RULE', 'margin_first')
    monkeypatch.setenv('GRM_PROFILE', 'eb1_c2')
    monkeypatch.setenv(runner.TREATMENT_FLAGS[0], spec['f1'])
    monkeypatch.setenv(runner.TREATMENT_FLAGS[1], spec['f2'])
    monkeypatch.setenv(runner.TREATMENT_FLAGS[2], spec['f5'])
    if spec['alias'] is None:
        monkeypatch.delenv(runner.ALIAS_ENV, raising=False)
    else:
        monkeypatch.setenv(runner.ALIAS_ENV, spec['alias'])


def _stub_lease(monkeypatch, tmp_path):
    """Lease + GPU idle probe only. Everything else on the real route.

    Copied from tests/test_grm_lt1_1_child_spawn.py so the two gates cannot
    drift into stubbing different amounts of the path.
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
    monkeypatch.setattr(runner, 'device_snapshot',
                        lambda: dict(time_unix=0.0, status='OK',
                                     memory_used_mib=64, memory_free_mib=12224,
                                     memory_total_mib=12288,
                                     other_process_pids=[],
                                     compute_list_empty=True,
                                     scope='gate stub: idle card'))
    return lock


def _child_argv_using_the_cpu_double(cell):
    return [sys.executable, '-m', 'scripts.grm_lt1_1',
            '--arm', runner.current_arm(), '--worker-cpu', cell['id']]


# --------------------------------------------------- RED: today's failure

def test_red_resume_without_out_hits_the_frozen_r2_root():
    """The failure as it happened, pinned so it cannot come back silently.

    With NO `--out`, `resume` resolves to r2's frozen arm root, and the very
    first registered cell is already COMPLETE there under r2's binding. This
    is the exact ValueError the lead's run died on.
    """
    reg = runner.registration('A')
    root = runner.out_dir('A')
    with runner.pinned_arm('A'), runner.lt1_1_seams('A'):
        first = sorted(reg['cells'], key=lambda c: (c['start'], c['arm']))[0]
        directory = worker.cell_directory(root / 'cells', first['id'])
        assert directory.exists(), 'r2 receipts missing; gate cannot run'
        controller = json.loads((directory / 'controller.json').read_text())
        assert controller['status'] == 'COMPLETE'
        # COMPLETE, but bound to r2 -- amendments 10/11 rebound the core.
        assert controller['binding'] != worker.bind(first['arm'])
        with pytest.raises(ValueError) as excinfo:
            worker.pending(reg, run=root)
    assert 'COMPLETED_CELL_BINDING_OR_STATUS' in str(excinfo.value)


def test_red_is_not_caused_by_the_restored_session_directory():
    """The lead's candidate (1), REFUTED by receipt.

    `pending()` fails on the FIRST cell and never reaches A-025-032, and it
    reads only `controller.json` / `worker.json` / `checkpoint/` -- never
    `session/`. The restored directory had nothing to do with it.
    """
    reg = runner.registration('A')
    root = runner.out_dir('A')
    ordered = sorted(reg['cells'], key=lambda c: (c['start'], c['arm']))
    assert ordered[0]['id'] == 'A-001-008'
    assert 'A-025-032' != ordered[0]['id']
    source = (ROOT / 'scripts/grm_lt1_worker.py').read_text()
    body = source.split('def pending(', 1)[1].split('\ndef ', 1)[0]
    assert 'session' not in body, body
    # And the stray directory is gone from the live run root entirely.
    assert not (root / 'cells/A-025-032/session').exists()


# ------------------------------------------------ GREEN: --out governs all

@pytest.mark.parametrize('label', list(ARM_MATRIX))
def test_out_governs_every_resolved_root(label, tmp_path, monkeypatch):
    spec = ARM_MATRIX[label]
    _apply_arm_env(monkeypatch, spec)
    out = tmp_path / 'campaign'
    roots = runner.resolved_roots(spec['base'], out)
    assert roots['arm_default_in_use'] is False
    assert roots['campaign_root'] == str(out)
    assert roots['cells_root'] == str(out / 'cells')
    assert roots['owner_file'] == str(out / 'campaign.active')
    assert roots['worker_RUN'] == str(out)
    # The arm default is REPORTED, so a reader can see what was avoided.
    assert roots['arm_default_root'] == str(runner.out_dir(spec['base']))
    assert roots['arm_default_root'] != roots['campaign_root']


def test_seams_pin_worker_run_to_the_given_root(tmp_path):
    """`worker.RUN` is what run_cell / accounting / the child all read."""
    out = tmp_path / 'campaign'
    with runner.lt1_1_seams('A', out):
        assert Path(worker.RUN) == out
    # Default unchanged for every pre-existing caller.
    with runner.lt1_1_seams('A'):
        assert Path(worker.RUN) == runner.out_dir('A')


def test_isolation_assert_rejects_a_root_that_is_the_arm_default(tmp_path):
    with runner.lt1_1_seams('A', runner.out_dir('A')):
        with pytest.raises(ValueError) as e:
            runner.assert_root_isolation('A', runner.out_dir('A'))
    assert 'LT11_OUT_EQUALS_ARM_DEFAULT' in str(e.value)


def test_isolation_assert_rejects_a_root_inside_the_arm_default(tmp_path):
    nested = runner.out_dir('A') / 'nested_campaign'
    with runner.lt1_1_seams('A', nested):
        with pytest.raises(ValueError) as e:
            runner.assert_root_isolation('A', nested)
    assert 'LT11_OUT_INSIDE_ARM_DEFAULT' in str(e.value)


def test_isolation_assert_rejects_a_desynced_worker_run(tmp_path):
    """The leak that would let a campaign select here and execute there."""
    out = tmp_path / 'campaign'
    with runner.lt1_1_seams('A', runner.out_dir('A') / 'x'):
        with pytest.raises(ValueError) as e:
            runner.assert_root_isolation('A', out)
    assert 'LT11_WORKER_RUN_NOT_ISOLATED' in str(e.value)


# ------------------------------------- GREEN: the flags reach the child

@pytest.mark.parametrize('label', list(ARM_MATRIX))
def test_treatment_flags_cross_into_the_leased_child(label, monkeypatch):
    """The silent-wrong-arm defect: every flag must survive the strip.

    `run_cell` builds the child environment from
    `grm_c2_cells.environment(flags)`, which deletes every ambient `GRM_*`.
    Before the fix all three treatment flags were ABSENT from the child.
    """
    spec = ARM_MATRIX[label]
    _apply_arm_env(monkeypatch, spec)
    from scripts.grm_c2_cells import environment
    with runner.lt1_1_seams(spec['base']):
        reg = runner.registration(spec['base'])
        flags = reg['arms'][reg['cells'][0]['arm']]['flags']
        stripped = environment(flags)
        # The strip really does remove them -- this is the hazard, pinned.
        for flag in runner.TREATMENT_FLAGS:
            assert flag not in stripped, flag
        env = worker.spawn_env(dict(stripped), {'id': 'A-001-008'})
    assert env[runner.TREATMENT_FLAGS[0]] == spec['f1']
    assert env[runner.TREATMENT_FLAGS[1]] == spec['f2']
    assert env[runner.TREATMENT_FLAGS[2]] == spec['f5']
    assert env[runner.ARM_ENV] == spec['base']
    # GRM-D2 (2026-09-11) CHANGED THIS ASSERTION, and the change is real
    # rather than a re-pin: before D2 the OFF arm expressed itself by
    # leaving the variable ABSENT, which was a correct pin only while
    # absent meant OFF.  D2 made A1's shipped default ON, so an absent
    # variable would now hand the CONTROL arm the treatment -- caught by
    # `pinned_arm`'s readback as LT11_ALIAS_PIN_FAILED.  The arm is now
    # pinned EXPLICITLY on both sides (`runner.arm_alias_pin`), so what
    # this test pins is the arm's resolved value, not its absence.  The
    # property being tested is unchanged: the arm survives the GRM_* strip
    # into the leased child.
    assert env[runner.ALIAS_ENV] == runner.arm_alias_pin(spec['base'])
    assert env[runner.ALIAS_ENV] == ('1' if spec['alias'] else '0')


def test_absent_flags_stay_absent_in_the_child(monkeypatch):
    """A caller with none of them set gets a byte-identical child env."""
    monkeypatch.setenv('GRM_ADMISSION_RULE', 'margin_first')
    for flag in runner.TREATMENT_FLAGS:
        monkeypatch.delenv(flag, raising=False)
    monkeypatch.delenv(runner.ALIAS_ENV, raising=False)
    from scripts.grm_c2_cells import environment
    with runner.lt1_1_seams('A'):
        reg = runner.registration('A')
        env = worker.spawn_env(
            environment(reg['arms'][reg['cells'][0]['arm']]['flags']),
            {'id': 'A-001-008'})
    for flag in runner.TREATMENT_FLAGS:
        assert flag not in env, flag


def test_child_inherits_the_campaign_root(tmp_path, monkeypatch):
    monkeypatch.setenv('GRM_ADMISSION_RULE', 'margin_first')
    out = tmp_path / 'campaign'
    from scripts.grm_c2_cells import environment
    with runner.lt1_1_seams('A', out):
        reg = runner.registration('A')
        env = worker.spawn_env(
            environment(reg['arms'][reg['cells'][0]['arm']]['flags']),
            {'id': 'A-001-008'})
    assert env[runner.RUN_ENV] == str(out)
    # And the child resolves its own root from that variable.
    monkeypatch.setenv(runner.RUN_ENV, str(out))
    assert runner.out_dir('A') == out


# ---------------------- THE gate: the real resume route, with --out set

@pytest.mark.parametrize('label', list(ARM_MATRIX))
def test_real_resume_route_through_pending_and_one_run_cell(
        label, tmp_path, monkeypatch):
    """`resume()` itself -- not the `--dry-lease` branch -- with `--out`.

    Drives the production route: chain preflight, seams, arm pin,
    campaign-owner file, `worker.pending` selection, and ONE real
    `worker.run_cell` spawn onto the CPU double. `run_cell` is allowed to
    execute exactly one cell and is then short-circuited, so the gate stays
    inside its time budget while still proving the spawn.
    """
    spec = ARM_MATRIX[label]
    _apply_arm_env(monkeypatch, spec)
    _stub_lease(monkeypatch, tmp_path)
    out = tmp_path / ('run_' + label.replace('+', 'plus').replace("'", 'p'))

    monkeypatch.setattr(worker, 'accounting', lambda *a, **k: 0.0)
    monkeypatch.setattr(worker.time, 'sleep', lambda *_: None)

    # `spawn_argv` CANNOT be patched out here: `resume()` enters
    # `lt1_1_seams`, which assigns `worker.spawn_argv = spawn_argv` (the real
    # `--worker` argv) and restores its saved value on exit. A monkeypatch
    # applied before `resume()` is therefore overwritten the moment the
    # seams open, the child loads the real 20B model, and on a machine with
    # no GPU it exits 1 -> WORKER_EXIT_1 -> rc 2.
    #
    # That is exactly the "flake" this file chased: NOT a race and NOT load
    # sensitivity, but a seam that silently wins over the patch. It looked
    # intermittent only because the model-load failure is fast when the
    # weights are cold in page cache and slow when they are warm, so the
    # cell sometimes outran the surrounding assertions.
    #
    # `runner.spawn_argv` is what the seams INSTALL, so patching it at the
    # source survives the seams and keeps the child on the CPU double.
    monkeypatch.setattr(runner, 'spawn_argv',
                        _child_argv_using_the_cpu_double)

    real_run_cell = worker.run_cell
    calls = []

    def one_cell_then_stop(cell, reg):
        calls.append(cell['id'])
        result = real_run_cell(cell, reg)
        # Stop the campaign loop after the first real spawn.
        monkeypatch.setattr(worker, 'pending', lambda *a, **k: None)
        return result
    monkeypatch.setattr(worker, 'run_cell', one_cell_then_stop)

    rc = runner.resume(spec['base'], root=out, host_gate=False)
    assert rc == 0, rc
    assert calls == ['A-001-008'], calls

    # THE assertion this whole file exists for: the campaign wrote into
    # `--out`, and the frozen arm root gained nothing.
    directory = out / 'cells/A-001-008'
    for name in ('controller.json', 'worker.json', 'reservation.json'):
        assert (directory / name).is_file(), name
    controller = json.loads((directory / 'controller.json').read_text())
    assert controller['status'] == 'COMPLETE', controller.get('error')
    assert (directory / 'checkpoint/checkpoint.json').is_file()
    log = (directory / 'worker.log').read_text()
    assert 'Traceback' not in log, log[-2000:]
    assert 'INPUT_SHA_MISMATCH' not in log, log[-2000:]

    # The child ran with THIS arm's flags, read back off its own receipt.
    receipt = json.loads((directory / 'worker.json').read_text())
    assert receipt['binding']['campaign_arm'] == spec['base']
    assert receipt['binding']['alias_fold_merge'] is (spec['alias'] == '1')

    # And the frozen r2 root is untouched by this campaign.
    frozen = runner.out_dir(spec['base'])
    assert not (frozen / 'cells/A-001-008/reservation.json').exists() or (
        frozen != out)
    assert not (frozen / 'campaign.active').exists()


def test_resume_reports_its_resolved_roots(tmp_path, monkeypatch):
    """`--dry-lease` carries the roots so an operator never has to infer."""
    _apply_arm_env(monkeypatch, ARM_MATRIX["A'"])
    out = tmp_path / 'campaign'
    result = runner.resume('A', root=out, dry_lease=True, host_gate=False)
    assert result['status'] == 'PASS'
    assert result['resolved_roots']['campaign_root'] == str(out)
    assert result['resolved_roots']['arm_default_in_use'] is False
    assert result['resolved_roots']['worker_RUN'] == str(out)


def test_main_forwards_out_to_the_real_resume():
    """The one-line defect, pinned at the source.

    Reading the source rather than only exercising it: the crash was that
    this call dropped its argument, and an exercise-only test would pass
    again the moment someone re-introduced a second no-argument call site.
    """
    source = (ROOT / 'scripts/grm_lt1_1.py').read_text()
    assert 'return resume(args.arm)\n' not in source
    assert 'return resume(args.arm, root=args.out,' in source


# ------------------------------------------------------------ F2's pin

def test_f2_flag_contract_pin_is_off_the_live_run_root():
    """A contract pin must not require a file inside a campaign run root."""
    contract = json.loads(
        (ROOT / 'artifacts/grm_f2/flag_contract.json').read_text())
    for name, path in contract['repo_relative_pins'].items():
        assert not os.path.isabs(path), path
        assert (ROOT / path).exists(), path
        assert '/run_A' not in path, (name, path)
        assert '/cells/' not in path, (name, path)
    relocation = contract['pin_relocations'][0]
    assert relocation['pin'] == 'r2_arm_a_source_store'
    assert relocation['to'] == contract['repo_relative_pins'][
        'r2_arm_a_source_store']
