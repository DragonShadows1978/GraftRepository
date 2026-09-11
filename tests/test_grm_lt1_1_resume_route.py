"""LT1.1 resume-route gates — the path that actually broke on the card.

WHAT WENT WRONG. `--arm A+ --resume` died before taking any lease:

    grm_lt1_amendment4.apply  ->  lt.binding('CPU')
    grm_lt1_1.binding         ->  ARM_ALIAS[arm]   ->  KeyError: 'CPU'

Two mistakes stacked. (1) The redirected `binding` took an ARM where LT1's
callers pass a BACKEND LABEL, so the signature drifted. (2) `lt.binding` was
the wrong seam to redirect at all: LT1 uses it to validate ITS OWN chain
(`apply4` compares a recorded `protocol_binding` against it), while receipts
are stamped through `worker.bind`. Redirecting the former broke LT1's
self-validation even after the signature was fixed.

Neither the fake path nor `--dry-run` traverses `apply4`, so nothing caught
it. These tests traverse the REAL resume route on CPU up to the lease
boundary, and pin the RED-before behaviour so the regression cannot return.

Prior art: the LT1 machinery under test is GRM contributors' (2026). The
lease-boundary stop is mine; no prior art known to me for it.
"""
import contextlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import grm_lt1 as lt                 # noqa: E402
from scripts import grm_lt1_1 as runner           # noqa: E402
from scripts import grm_lt1_worker as worker      # noqa: E402


# ------------------------------------------------- the signature contract

def test_binding_accepts_a_backend_label_like_lt1_callers_pass():
    """`lt.binding('CPU')` is what apply4/preflight/summary call. It must work."""
    for arm in runner.ARMS:
        with runner.lt1_1_seams(arm):
            value = worker.bind('CPU')           # the shape apply4 uses
            assert value['label'] == 'CPU'
            assert value['campaign_arm'] == arm
            assert value['alias_fold_merge'] is (arm == 'A+')


def test_binding_accepts_an_arm_like_worker_bind_passes():
    """`worker.bind(cell['arm'])` passes an ARM. It must also work."""
    for arm in runner.ARMS:
        with runner.lt1_1_seams(arm):
            value = worker.bind('A')             # the shape execute() uses
            assert value['label'] == 'A'
            assert value['campaign_arm'] == arm
            assert value['alias_fold_merge'] is (arm == 'A+')


def test_binding_never_raises_on_any_token_lt1_passes():
    """RED-before regression: every observed caller token must be accepted."""
    for token in ('CPU', 'GPU', 'A', 'B', 'A+'):
        with runner.lt1_1_seams('A+'):
            value = worker.bind(token)
            assert value['label'] == token
            assert value['campaign_arm'] == 'A+'


def test_the_arms_still_produce_different_receipt_bindings():
    """The signature fix must not collapse the arm split."""
    def bind_under(arm):
        with runner.lt1_1_seams(arm):
            return worker.bind('A')
    a, aplus = bind_under('A'), bind_under('A+')
    assert a != aplus
    assert a['alias_fold_merge'] is False
    assert aplus['alias_fold_merge'] is True
    assert a['campaign_arm'] == 'A' and aplus['campaign_arm'] == 'A+'


def test_binding_refuses_to_guess_an_unpinned_arm():
    """A binding computed under the wrong arm would mislabel a receipt."""
    import os
    saved = os.environ.pop(runner.ARM_ENV, None)
    try:
        with pytest.raises(ValueError, match='LT11_ARM_NOT_PINNED'):
            runner.binding('CPU')
    finally:
        if saved is not None:
            os.environ[runner.ARM_ENV] = saved


# ------------------------------------------------------- the seam is narrow

def test_lt_binding_itself_is_never_redirected():
    """LT1 must keep validating its own chain with its own binding."""
    original = lt.binding
    for arm in runner.ARMS:
        with runner.lt1_1_seams(arm):
            assert lt.binding is original, (
                'lt.binding was redirected; apply4 would compare its recorded '
                'protocol_binding against an LT1.1-shaped dict')
    assert lt.binding is original


def test_only_the_documented_seams_move():
    """Audit: exactly FIX, worker.bind and worker.RUN change, and restore."""
    before = (lt.FIX, lt.binding, worker.bind, worker.RUN, lt.RUN, lt.REG)
    with runner.lt1_1_seams('A+'):
        assert lt.FIX == runner.FIXTURE
        assert worker.bind is runner.binding
        assert worker.RUN == runner.out_dir('A+')
        # These must NOT move: they address LT1's own campaign and chain.
        assert lt.RUN == before[4]
        assert lt.REG == before[5]
        assert lt.binding is before[1]
    assert (lt.FIX, lt.binding, worker.bind, worker.RUN, lt.RUN, lt.REG) == before


def test_the_fake_cell_path_uses_the_same_narrow_seam():
    """The two seam sites must agree, or one route drifts from the other."""
    source = (ROOT / 'scripts/grm_lt1_1.py').read_text()
    assert 'lt.FIX, worker.bind, worker.RUN = saved' in source
    assert 'lt.binding = binding' not in source, (
        'a seam site still redirects lt.binding')


# ------------------------------------------------ RED-before reproduction

@contextlib.contextmanager
def _shipped_broken_seams(arm):
    """The seam exactly as it shipped, to prove the gate has teeth."""
    def broken_binding(name):
        return dict(arm=name,
                    alias_fold_merge=(runner.ARM_ALIAS[name] is not None))
    saved = (lt.FIX, lt.binding, worker.RUN)
    try:
        lt.FIX = runner.FIXTURE
        lt.binding = broken_binding
        worker.RUN = runner.out_dir(arm)
        yield
    finally:
        lt.FIX, lt.binding, worker.RUN = saved


def test_the_shipped_seam_reproduces_the_leads_keyerror():
    """RED-before: the exact failure from lead_Ap_resume.log."""
    with _shipped_broken_seams('A+'):
        with pytest.raises(KeyError) as exc:
            lt.preflight()
    assert exc.value.args[0] == 'CPU'
    # And it came from the line the lead's traceback names.
    frames = [f for f in exc.traceback if 'grm_lt1_amendment4' in str(f.path)]
    assert frames, 'the KeyError must surface through grm_lt1_amendment4.apply'


def test_the_fixed_seam_does_not_raise_keyerror():
    """GREEN-after on the same call."""
    for arm in runner.ARMS:
        with runner.lt1_1_seams(arm):
            try:
                lt.preflight()
            except KeyError:                      # pragma: no cover
                pytest.fail('KeyError returned on arm %s' % arm)
            except ValueError:
                pass    # host-tree drift is a separate, pre-existing RED


def test_dry_run_never_traversed_apply4_which_is_why_it_missed_this():
    """Name the blind spot, so the lesson is encoded and not just narrated."""
    value = runner.dry_run('A+')
    assert value['status'] == 'PASS'
    assert 'host_preflight' not in value, (
        'if --dry-run ever runs the host preflight, this test should be '
        'replaced by a stronger one rather than silently passing')


# ------------------------------------------------ the real route, on CPU

@pytest.mark.parametrize('arm', ['A', 'A+'])
def test_resume_route_reaches_the_lease_boundary(arm, tmp_path):
    """THE gate: the production resume path, CPU, stopping at the lease.

    Traverses amendment loading, apply4-bearing preflight, seam redirection,
    the arm pin, the campaign owner file and cell selection. Only
    `worker.run_cell` -- the lease/flock/model -- is not entered.
    """
    value = runner.resume(arm, root=tmp_path / arm.replace('+', 'plus'),
                          dry_lease=True, host_gate=False)
    assert value['status'] == 'PASS'
    assert value['mode'] == 'RESUME_DRY_LEASE'
    assert value['gpu_executed'] is False
    assert value['stopped_at'] == 'worker.run_cell (lease boundary)'
    assert value['next_cell'] == 'A-001-008'
    assert value['pinned']['admission_rule'] == 'margin_first'
    assert value['pinned']['alias_fold_merge'] is (arm == 'A+')
    # The binding a receipt WOULD have carried, computed on the real route.
    assert value['receipt_binding']['campaign_arm'] == arm
    assert value['receipt_binding']['alias_fold_merge'] is (arm == 'A+')


def test_the_resume_route_releases_its_campaign_owner(tmp_path):
    """`campaign.active` is exclusive; a dry-lease run must not leave one."""
    root = tmp_path / 'owner'
    runner.resume('A', root=root, dry_lease=True, host_gate=False)
    assert not (root / 'campaign.active').exists()
    # Provably re-runnable: a stale owner file would make this raise.
    runner.resume('A', root=root, dry_lease=True, host_gate=False)


def test_the_resume_route_selects_the_next_pending_cell(tmp_path):
    """Cell selection is `worker.pending`, not a reimplementation."""
    root = tmp_path / 'pending'
    runner.run_fake('A', root=root, limit=2)
    value = runner.resume('A', root=root, dry_lease=True, host_gate=False)
    assert value['next_cell'] == 'A-017-024', (
        'resume must continue after the completed cells, not restart')


def test_host_gate_is_on_by_default_for_the_gpu_path():
    """The real campaign always runs LT1's host preflight."""
    import inspect
    signature = inspect.signature(runner.resume)
    assert signature.parameters['host_gate'].default is True
    assert signature.parameters['dry_lease'].default is False


# --------------------------------------------------------- the CLI surface

@pytest.mark.parametrize('arm', ['A', 'A+'])
def test_resume_dry_lease_runs_from_the_command_line(arm, tmp_path):
    """End to end through argparse, the way the lead would invoke it."""
    result = subprocess.run(
        [sys.executable, 'scripts/grm_lt1_1.py', '--arm', arm, '--resume',
         '--dry-lease', '--no-host-gate', '--out',
         str(tmp_path / arm.replace('+', 'plus'))],
        cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    # stdout must be exactly one JSON document: progress goes to stderr, so a
    # caller can pipe this straight into jq.
    payload = json.loads(result.stdout)
    assert payload['mode'] == 'RESUME_DRY_LEASE'
    assert payload['gpu_executed'] is False
    assert payload['pinned']['alias_fold_merge'] is (arm == 'A+')
    assert 'pinned' in result.stderr, 'the operator progress line went missing'


def test_dry_lease_and_no_host_gate_parse():
    args = runner.parse_args(['--arm', 'A+', '--resume', '--dry-lease',
                              '--no-host-gate'])
    assert args.resume and args.dry_lease and args.no_host_gate
