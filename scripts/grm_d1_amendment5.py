#!/usr/bin/env python3
"""GRM-D1 follow-up 5: LT1.1 amendment 5 — the leased CHILD runs our chain.

Amendment 4 made the PARENT validate LT1.1's chain per the lead's ruling. The
lead-run `--arm A+ --resume` then pinned the arm, started cell A-001-008, and
the cell went RED with `WORKER_EXIT_1`. The child's log:

    scripts/grm_lt1_worker.py line 289, in <module>
        if args.worker: worker(next(c for c in lt.verify()['cells'] ...))
    scripts/grm_lt1.py line 101, in verify
        ValueError: INPUT_SHA_MISMATCH: core/graft_arena.py

`run_cell` hard-coded `Popen([... '-m', 'scripts.grm_lt1_worker', '--worker',
id])`, so the CHILD re-entered LT1's module and re-ran LT1's chain
verification -- the exact gate the ruling removed from the parent. The parent
was green and the child was not.

WHY NOTHING CAUGHT IT, stated plainly because it is the third time a gate has
missed the real route:

  * `--dry-lease` stops AT the lease boundary, before `run_cell` spawns;
  * the `--fake` 2-cell proof used `--fake-cell`, a path that never enters
    `run_cell` at all, so it exercised a different child;
  * my follow-up-2 docstring ASSERTED "the worker subprocess is re-entered
    through THIS module". It was not. That was an unverified claim in a
    document, and it is corrected here.

THE FIX. Two more seams, five in total. `scripts/grm_lt1_worker.run_cell`
gains a `spawn_argv` / `spawn_env` seam whose DEFAULT is byte-identical to the
line it replaced; LT1's own campaign is unaffected. LT1.1 sets it so the child
is `scripts/grm_lt1_1.py --arm <arm> --worker <cell>`, which runs
`lt1_1_preflight` on OUR chain, redirects the same seams, re-pins the arm from
`GRM_LT1_1_ARM`, and calls a `lt1_1_worker` that is LT1's `worker` with its
`lt.verify()` replaced by `registration(arm)`.

AUDIT of every path from the runner to LT1's chain: `lt.verify()` is reachable
in `grm_lt1_worker` at three points -- `worker()` line 188, `resume()` line
273, and `__main__` line 289. The runner never calls `worker.resume`. Lines
188 and 289 are both covered: 188 by redirecting `worker.worker`, 289 by
redirecting the spawn argv so that entry point is never reached.

Prior art:
  * `scripts/grm_lt1_worker.py` `worker` / `run_cell` (GRM contributors,
    2026): the lease-parent contract, the cooperative SIGALRM deadline with no
    kill syscall, the flock lease, the create-only receipts. Reused; the new
    child body reproduces `worker`'s structure with ONE substitution.
  * Pin-across-a-process-boundary after the ambient strip: R1 `pin_rule`
    (`scripts/grm_r1_replay.py:242`, GRM contributors 2026), extended here
    from a variable in one process to a variable carried into a child.
  * Mine: the `spawn_argv`/`spawn_env` seam with a byte-identical default, and
    the spawning gate that runs the real `run_cell` with only the lease and
    the model stubbed.
  * No prior art known to me for this exact composition.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / 'artifacts/grm_d1/lt1_1'
AMENDMENT4 = OUT / 'amendment4.json'
ORDER = ROOT / 'orders/GRM_D1_LT1_RESIDUAL_DIAGNOSIS.md'

RUNNER = 'scripts/grm_lt1_1.py'
WORKER = 'scripts/grm_lt1_worker.py'
RUNNER_TESTS = ('tests/test_grm_lt1_1_runner.py',
                'tests/test_grm_lt1_1_resume_route.py',
                'tests/test_grm_lt1_1_preflight.py',
                'tests/test_grm_lt1_1_child_spawn.py')

ALIAS_FLAG = 'GRM_ALIAS_FOLD_MERGE'

#: The RED cell the lead-run produced, archived rather than deleted.
RED_CELL = 'A-001-008'
RED_ARM = 'A+'

FOLLOW_UP = (
    'Lead follow-up 5 to GRM-D1 (2026-09-11): `--arm A+ --resume` started '
    'cell A-001-008 and the cell went RED with WORKER_EXIT_1 because '
    'run_cell spawns `-m scripts.grm_lt1_worker --worker <cell>`, whose '
    '__main__ re-runs LT1 verify() in the child. Route the child through '
    'scripts/grm_lt1_1.py --worker with our own chain, seams and arm pin; '
    'audit every other path to LT1 __main__ / lt.verify(); add a gate that '
    'actually spawns the child through the REAL run_cell with the lease and '
    'model stubbed, both arms, cells 1-2, asserting no INPUT_SHA_MISMATCH in '
    'the child log; archive the RED cell and re-arm A+ from cell 1.')


def sha_path(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(obj):
    return (json.dumps(obj, sort_keys=True, indent=2) + '\n').encode()


def verify_reach_audit():
    """Every `lt.verify()` CALL in the worker module, and its cover.

    AST-based, not substring-based: this file and the worker module both
    DISCUSS `lt.verify()` in comments and docstrings, and an earlier
    substring version of this audit duly reported a comment as a fourth call
    site. A reach audit that miscounts is worse than none.
    """
    import ast
    source = (ROOT / WORKER).read_text()
    tree = ast.parse(source)
    lines = source.splitlines()

    #: line -> enclosing function name ('__main__' when at module level).
    owner = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for line in range(node.lineno, node.end_lineno + 1):
                owner.setdefault(line, node.name)

    cover = {
        'worker': 'covered: `worker.worker` is redirected to `lt1_1_worker`, '
                  'which uses `registration(arm)` instead',
        'resume': 'not reachable: the runner never calls `worker.resume`; it '
                  'runs its own `resume` loop over `worker.pending` / '
                  '`worker.run_cell`',
        'run_cell': 'not reachable: `run_cell` receives its registration as an '
                    'argument from the caller, which is the LT1.1 `resume`',
        '__main__': 'not reachable: `spawn_argv` routes the child to '
                    '`scripts/grm_lt1_1.py --worker`, so this entry point is '
                    'never executed by an LT1.1 campaign',
    }
    rows = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'verify'
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == 'lt'):
            continue
        where = owner.get(node.lineno, '__main__')
        rows.append(dict(line=node.lineno,
                         source=lines[node.lineno - 1].strip(),
                         function=where,
                         disposition=cover.get(
                             where, 'UNCLASSIFIED — audit needs updating')))
    return sorted(rows, key=lambda row: row['line'])


def amendment():
    if not AMENDMENT4.exists():
        raise ValueError('LT11_AMENDMENT4_MISSING')
    parent_sha = sha_path(AMENDMENT4)
    recorded = (OUT / 'amendment4.sha256').read_text().split()[0]
    if parent_sha != recorded:
        raise ValueError('LT11_AMENDMENT4_SHA_MISMATCH')
    parent = json.loads(AMENDMENT4.read_text())

    archive = OUT / 'archive' / ('%s_%s_RED' % (RED_ARM.replace('+', 'plus'),
                                                RED_CELL))
    return dict(
        schema='grm.lt1_1.amendment5.v1',
        amendment=5,
        previous_amendment='artifacts/grm_d1/lt1_1/amendment4.json',
        previous_amendment_sha256=parent_sha,
        registration='artifacts/grm_d1/lt1_1/registration.json',
        registration_sha256=parent['registration_sha256'],
        order='orders/GRM_D1_LT1_RESIDUAL_DIAGNOSIS.md',
        order_sha256=sha_path(ORDER),
        follow_up_instruction=FOLLOW_UP,
        purpose=('Route the leased child through the LT1.1 chain, so a green '
                 'parent cannot produce a child that re-runs LT1 own '
                 'verification.'),
        defect=dict(
            symptom='cell %s RED with WORKER_EXIT_1 on arm %s; child log: '
                    'INPUT_SHA_MISMATCH: core/graft_arena.py raised from '
                    'scripts/grm_lt1.py line 101 via '
                    'scripts/grm_lt1_worker.py line 289' % (RED_CELL, RED_ARM),
            receipt='artifacts/grm_d1/lt1_1/run_Aplus/cells/%s/worker.log'
                    % RED_CELL,
            cause='run_cell hard-coded the child argv as `-m '
                  'scripts.grm_lt1_worker --worker <cell>`; that module '
                  '__main__ resolves its cell with lt.verify(), which is LT1 '
                  'own chain and the gate the ruling removed from the parent',
            why_the_gates_missed_it=[
                '--dry-lease stops AT the lease boundary, before run_cell '
                'spawns anything',
                'the --fake 2-cell proof used --fake-cell, a path that never '
                'enters run_cell, so it exercised a different child',
                'the follow-up-2 docstring ASSERTED the child came back '
                'through the LT1.1 module; it did not, and that unverified '
                'claim is corrected here',
            ]),
        seams=dict(
            count=5,
            added=['worker.worker -> lt1_1_worker (covers lt.verify at '
                   'grm_lt1_worker.py:188)',
                   'worker.spawn_argv -> spawn_argv (covers grm_lt1_worker '
                   '__main__ at :289 by never reaching it)'],
            existing=['lt.FIX', 'worker.bind', 'worker.RUN'],
            worker_module_change=dict(
                file=WORKER,
                sha256=sha_path(ROOT / WORKER),
                what='run_cell consults optional module globals `spawn_argv` '
                     'and `spawn_env`; the default is byte-identical to the '
                     'line it replaced, so LT1 own campaign is unaffected',
                lt1_behaviour_unchanged=True)),
        verify_reach_audit=verify_reach_audit(),
        child=dict(
            argv='python3 -m scripts.grm_lt1_1 --arm <arm> --worker <cell>',
            runs='lt1_1_preflight (LT1.1 chain) -> lt1_1_seams -> '
                 'worker.execute with registration(arm)',
            arm_pin='carried across the process boundary in GRM_LT1_1_ARM by '
                    '`spawn_env`, re-applied after `environment(flags)` '
                    'strips every ambient GRM_*',
            preserved=['WORKER_REQUIRES_LEASE_PARENT contract',
                       'cooperative SIGALRM deadline, no kill syscall',
                       'create-only worker.json',
                       'RESTART_REQUIRES_NEW_PROCESS (one process per cell)']),
        gate=dict(
            tests='tests/test_grm_lt1_1_child_spawn.py',
            what='spawns the child through the REAL run_cell with the lease '
                 'stubbed to a no-op flock on a temp file and the model '
                 'stubbed to the CPU double; both arms, cells 1-2',
            asserts=['the child writes controller.json / worker.json / '
                     'checkpoint',
                     'the child log contains no INPUT_SHA_MISMATCH',
                     'the child process id differs from the parent '
                     '(RESTART_REQUIRES_NEW_PROCESS honoured)',
                     'the receipt binding carries the right arm'],
            red_before='test_the_old_argv_reproduces_the_leads_worker_exit_1 '
                       'spawns the ORIGINAL argv and asserts the child log '
                       'carries INPUT_SHA_MISMATCH and a non-zero exit',
            evidence_class='CPU child spawn (process and receipt plumbing; '
                           'NOT a language-model quality measurement)'),
        red_cell_archive=dict(
            arm=RED_ARM, cell=RED_CELL,
            archived_to=str(archive.relative_to(ROOT)),
            policy='create-only: the RED attempt is copied, never deleted or '
                   'overwritten, and the live cell directory is removed so '
                   'arm %s re-arms from cell 1' % RED_ARM,
            rearm='arm %s resumes at %s' % (RED_ARM, RED_CELL)),
        runner=dict(
            path=RUNNER,
            sha256=sha_path(ROOT / RUNNER),
            superseded_sha256=parent['runner']['sha256'],
            tests=list(RUNNER_TESTS),
            tests_sha256={name: sha_path(ROOT / name) for name in RUNNER_TESTS
                          if (ROOT / name).exists()}),
        ruling=parent['ruling'],
        arms=parent['arms'],
        budget_gpu_seconds_per_arm=parent['budget_gpu_seconds_per_arm'],
        budget_gpu_seconds_total=parent['budget_gpu_seconds_total'],
        budget_gpu_hours_total=parent['budget_gpu_hours_total'],
        process_safety=dict(
            gpu='only `--resume` without `--dry-lease` takes the shared lease, '
                'through LT1 own run_cell (flock + nvidia-smi idle check + '
                'no-kill child wait). The spawn gate stubs the lease to a '
                'temp-file flock and never touches the shared lock or the GPU.',
            children='the parent waits on its child and never signals it; the '
                     'cooperative deadline raises inside the child instead',
            arms='A and A+ never run concurrently on the same GPU',
            git='lead commits; the seat never runs git'),
        prior_art=[
            'Lease/child contract reused from scripts/grm_lt1_worker.py '
            'worker/run_cell (GRM contributors, 2026); the new child body '
            'reproduces worker structure with one substitution',
            'Pin-after-the-strip carried across a process boundary: R1 '
            'pin_rule, scripts/grm_r1_replay.py:242 (GRM contributors, 2026)',
            'spawn_argv/spawn_env seam with a byte-identical default, and a '
            'spawning gate over the real run_cell: no prior art known to me',
        ],
    )


def archive_red_cell():
    """Copy the RED cell aside (create-only) and clear it so A+ re-arms."""
    import shutil
    live = OUT / ('run_%s' % RED_ARM.replace('+', 'plus')) / 'cells' / RED_CELL
    archive = OUT / 'archive' / ('%s_%s_RED' % (RED_ARM.replace('+', 'plus'),
                                                RED_CELL))
    if not live.exists():
        return dict(status='NOTHING_TO_ARCHIVE', live=str(live))
    if archive.exists():
        return dict(status='ALREADY_ARCHIVED', archive=str(archive))
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(live, archive)
    shutil.rmtree(live)
    return dict(status='ARCHIVED', archive=str(archive),
                cleared=str(live),
                note='create-only: the attempt is preserved; the live cell is '
                     'cleared so the arm re-arms from cell 1')


def lead_commands(amend_sha):
    doc = amendment()
    lines = [
        '# LT1.1 — arm A and arm A+ (amendment 5)',
        '# registration sha256: %s' % doc['registration_sha256'],
        '# amendment 4  sha256: %s' % doc['previous_amendment_sha256'],
        '# amendment 5  sha256: %s' % amend_sha,
        '# runner            : %s (sha256 %s)'
        % (RUNNER, doc['runner']['sha256']),
        '#',
        '# 26 cells per arm. Reservation ceiling %d s (%.2f GPU-h) PER ARM;'
        % (doc['budget_gpu_seconds_per_arm'],
           doc['budget_gpu_seconds_per_arm'] / 3600.0),
        '# both arms: %.2f GPU-h ceiling. Separately resumable.'
        % doc['budget_gpu_hours_total'],
        '#',
        '# FIXED in amendment 5: the leased CHILD now runs the LT1.1 chain.',
        '# Previously run_cell spawned `-m scripts.grm_lt1_worker --worker`,',
        '# whose __main__ re-ran LT1 verify() and put cell %s RED with' % RED_CELL,
        '# WORKER_EXIT_1 / INPUT_SHA_MISMATCH. The child is now',
        '#   python3 -m scripts.grm_lt1_1 --arm <arm> --worker <cell>',
        '# The RED attempt is archived at %s' % doc['red_cell_archive']['archived_to'],
        '# and arm %s re-arms from cell 1.' % RED_ARM,
        '#',
        '# ---- CPU gates (no GPU, no lease) ----',
        'cd %s' % ROOT,
        'python3 -m pytest -q tests/test_grm_d1*.py tests/test_grm_lt1_1*.py',
        'python3 scripts/grm_d1_recap.py          # recap oracle 5/5 PASS',
        'python3 scripts/grm_d1_alias_cpu.py      # A+ alias fold/admit/mount',
        '',
    ]
    for arm in ('A', 'A+'):
        spec = doc['arms'][arm]
        label = ('alias fold OFF' if not spec['alias_fold_merge']
                 else '%s=1 pinned' % ALIAS_FLAG)
        lines += [
            '# ---- arm %s (%s) ----' % (arm, label),
            'python3 %s --arm %s --preflight' % (RUNNER, arm),
            'python3 %s --arm %s --dry-run' % (RUNNER, arm),
            '# Route check on CPU: walks the REAL resume path and stops at',
            '# the lease boundary.',
            'python3 %s --arm %s --resume --dry-lease' % (RUNNER, arm),
            '# GPU campaign (takes the shared lease; resumable -- run again',
            '# to continue):',
            'python3 %s --arm %s --resume' % (RUNNER, arm),
            'python3 %s --arm %s --summary' % (RUNNER, arm),
            '',
        ]
    lines += [
        '# The runner pins the arm itself, AFTER environment(flags) strips',
        '# every ambient GRM_*, and carries it into the leased child in',
        '# GRM_LT1_1_ARM. The child verifies the LT1.1 chain before any work:',
        '#   arm A   -> admission_rule=margin_first, alias_fold_merge=False',
        '#   arm A+  -> admission_rule=margin_first, alias_fold_merge=True',
        '# A pin that does not take is LT11_ALIAS_PIN_FAILED, never silent.',
        '# A drifted LT1.1 document stops the CHILD too:',
        '#   LT11_CHILD_PREFLIGHT_BLOCKED.',
        '',
        '# Registered predictions (before any run):',
    ]
    for arm in ('A', 'A+'):
        lines.append('#   %-3s %s' % (arm, json.dumps(
            doc['arms'][arm]['predictions'], sort_keys=True)))
    return '\n'.join(lines) + '\n'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    archived = archive_red_cell()
    payload = canonical(amendment())
    (OUT / 'amendment5.json').write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    (OUT / 'amendment5.sha256').write_text('%s  amendment5.json\n' % digest)
    (OUT / 'lead_commands.txt').write_text(lead_commands(digest))

    doc = amendment()
    print('amendment5 sha256 %s' % digest)
    print('parent amendment4 %s' % doc['previous_amendment_sha256'])
    print('runner %s' % doc['runner']['sha256'])
    print('  supersedes %s' % doc['runner']['superseded_sha256'])
    print('seams: %d' % doc['seams']['count'])
    for item in doc['seams']['added']:
        print('  + %s' % item)
    print('verify() reach points audited: %d' % len(doc['verify_reach_audit']))
    print('RED cell archive: %s' % archived['status'])


if __name__ == '__main__':
    main()
