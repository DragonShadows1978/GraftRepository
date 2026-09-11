"""GRM-F6: repo-relative receipt roots, and loud failure on a dead one.

What this module gates
----------------------
Round-1 seat worktrees ``/mnt/ForgeRealm/wt/grm-*`` were pruned on
2026-09-11.  Five scripts hard-coded absolute paths into them.  The
dangerous shape was SILENT: ``Path(dead).glob(...)`` yields nothing, the
census loop produces zero rows, and a gate phrased "every row agrees"
passes over nothing.

So this module asserts three things, per rebound script:

1. No script under ``scripts/`` binds a pruned ``/mnt/ForgeRealm/wt/``
   path as an executable path expression any more (comments and docstrings
   describing the history are fine and are excluded by parsing the AST).
2. Every rebound root resolves INSIDE the repository, from ``__file__``
   rather than the CWD, and the receipt data is actually there.
3. Planting a dead path (via the script's own override env var) makes the
   script fail LOUD -- naming the missing path, the env var and the
   receipt that pinned it -- and a zero-row glob is RED, never a pass.

Prior art
---------
* ``ast.parse`` + ``ast.walk`` over a source tree to assert a property of
  the CODE rather than of a grep is the standard lint approach (``ast``,
  Python 2.6, 2008; ``flake8``/``pylint`` checkers, 2010-).  Taken
  verbatim; using the AST is what lets rule 1 distinguish a live path
  constant from a comment recording the dead one.
* Subprocess-isolated import to observe a module-import-time failure is
  the standard pytest recipe for "the error happens at import" (pytest
  ``subprocess``/``pytester`` idiom, pytest-dev, 2009-).
* Zero-rows-is-an-error: dbt row-count tests (Fishtown Analytics, 2018),
  Great Expectations ``expect_table_row_count_to_be_between``
  (Superconductive, 2018), pytest exit code 5 for "no tests collected"
  (pytest-dev, 2019).  Taken: the principle.  Ours: wiring it into these
  particular receipt consumers.
* No prior art known to me for the specific "pruned seat worktree leaves a
  vacuously-passing receipt gate" pattern as a named failure mode.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import grm_repo_paths as rp  # noqa: E402

#: The five consumers F6 rebound, with the env var that overrides each root.
REBOUND = {
    'scripts/grm_lt1_offline.py': 'GRM_C2_EPOCH_ROOT',
    'scripts/grm_scout_fix8_cpu.py': 'GRM_C2_EPOCH_ROOT',
    'scripts/grm_r1_register.py': 'GRM_C2_EPOCH_ROOT',
    'scripts/grm_rd1.py': 'GRM_C7_SOURCE_ROOT',
    'scripts/grm_d1_cause_table.py': 'GRM_LT1_CELLS_ROOT',
}

#: Seat worktrees pruned on 2026-09-11. `grm-f5` is included deliberately:
#: it still exists today, and binding it would be the same bug one worktree
#: later (the F5 gate script did exactly that, and F6 removed it).
PRUNED_PREFIX = '/mnt/ForgeRealm/wt/'


# ---------------------------------------------------------------- rule 1

def _string_constants(path):
    """Every string literal in a module, from the AST -- comments excluded."""
    tree = ast.parse(Path(path).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.lineno, node.value


def _is_docstring_of(tree):
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc is not None:
                out.add(doc)
    return out


#: Keyword arguments whose value is a DIAGNOSTIC MESSAGE, not a path that
#: gets opened. ``receipt_root(dead_absolute=...)`` quotes the pruned path in
#: its RED so the failure is diagnosable without reading the script; that is
#: the fix reporting itself, not the bug. Everything else is a live path.
#: ``registration=`` is the campaign_receipt marker's PROSE describing which
#: registration a receipt is bound to -- including, deliberately, the fact
#: that the registration pins now-pruned absolute paths. That sentence is the
#: receipt staying honest about what it was true at; it is never opened.
MESSAGE_ONLY_KWARGS = {'dead_absolute', 'pinned_by', 'registration', 'reason'}


def _message_only_nodes(tree):
    """String constants passed as a message-only keyword, or 3rd/4th
    positional of ``receipt_root`` (``pinned_by``, ``dead_absolute``)."""
    out = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(
            func, 'id', None)
        for kw in node.keywords:
            if kw.arg in MESSAGE_ONLY_KWARGS and isinstance(kw.value, ast.Constant):
                out.add(id(kw.value))
        if name == 'receipt_root':
            for arg in node.args[2:4]:
                if isinstance(arg, ast.Constant):
                    out.add(id(arg))
    return out


def _live_worktree_literals(path):
    """Worktree paths used AS PATHS -- not docstring prose, not RED text.

    A path is "live" when the value could be opened: a module constant, a
    ``Path(...)`` argument, a glob root. It is NOT live when it is prose in
    a docstring, nor when it is the pruned path quoted back inside a
    :func:`receipt_root` failure message -- that string is never opened, and
    removing it would make the RED less diagnosable, not the code safer.
    """
    source = Path(path).read_text()
    tree = ast.parse(source)
    docstrings = _is_docstring_of(tree)
    exempt = _message_only_nodes(tree)
    hits = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        if PRUNED_PREFIX not in node.value:
            continue
        if node.value in docstrings or id(node) in exempt:
            continue
        hits.append((node.lineno, node.value))
    return sorted(hits)


@pytest.mark.parametrize('relative', sorted(REBOUND))
def test_rebound_script_has_no_live_worktree_literal(relative):
    """The rebind is real: no executable /mnt/ForgeRealm/wt/ path remains.

    The dead paths survive in COMMENTS (so the history is readable) and F6
    deliberately keeps them there; the AST walk ignores comments, so this
    asserts the code, not the grep.
    """
    hits = _live_worktree_literals(ROOT / relative)
    assert hits == [], (
        '%s still binds a seat-worktree path as live code: %s' % (relative, hits))


def test_f5_gate_no_longer_chdirs_into_a_seat_worktree():
    """The F5 gate used ROOT = Path('/mnt/ForgeRealm/wt/grm-f5') + os.chdir.

    That was the same bug one worktree later. It now resolves from __file__.
    """
    path = ROOT / 'scripts/grm_f5_c2_replay_gate.py'
    source = path.read_text()
    tree = ast.parse(source)
    assert "ROOT = Path(__file__).resolve().parents[1]" in source
    assert _live_worktree_literals(path) == []
    # No LIVE os.chdir call. The docstring still quotes the removed line, on
    # purpose -- the history is the point of the note.
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == 'chdir']
    assert calls == [], 'grm_f5_c2_replay_gate.py still chdirs at line(s) %s' % (
        [n.lineno for n in calls],)


# ---------------------------------------------------------------- rule 2

def test_repo_root_resolves_from_file_not_cwd(tmp_path, monkeypatch):
    """repo_root() must not depend on the process CWD."""
    here = rp.repo_root()
    monkeypatch.chdir(tmp_path)
    assert rp.repo_root() == here == ROOT


@pytest.mark.parametrize('relative', [
    'artifacts/grm_c2/epochs/scout-fix-2',
    'artifacts/grm_c7/r2',
    'artifacts/grm_lt1/amendment2/run_margin_first/cells',
])
def test_receipt_root_is_inside_the_repository(relative):
    resolved = rp.receipt_root(relative, 'GRM_F6_UNSET_FOR_THIS_TEST',
                               'test', None)
    assert resolved == ROOT / relative
    assert resolved.is_relative_to(ROOT)
    assert resolved.exists()


def test_receipt_root_env_override_appends_relative(tmp_path, monkeypatch):
    (tmp_path / 'artifacts/grm_c7/r2').mkdir(parents=True)
    monkeypatch.setenv('GRM_F6_TEST_ROOT', str(tmp_path))
    got = rp.receipt_root('artifacts/grm_c7/r2', 'GRM_F6_TEST_ROOT', 'test')
    assert got == tmp_path / 'artifacts/grm_c7/r2'


# ---------------------------------------------------------------- rule 3

def test_dead_receipt_root_fails_loud_naming_everything(tmp_path, monkeypatch):
    """A planted dead path is a RED that carries the whole diagnosis."""
    monkeypatch.setenv('GRM_F6_TEST_ROOT', str(tmp_path / 'no-such-worktree'))
    with pytest.raises(rp.DeadReceiptPath) as excinfo:
        rp.receipt_root('artifacts/grm_c2/epochs/scout-fix-2',
                        'GRM_F6_TEST_ROOT',
                        'artifacts/grm_scout_fix8/green.json',
                        '/mnt/ForgeRealm/wt/grm-c2/artifacts/grm_c2/'
                        'epochs/scout-fix-2')
    message = str(excinfo.value)
    assert 'GRM_F6_DEAD_RECEIPT_PATH' in message
    assert 'no-such-worktree' in message                  # the missing path
    assert 'GRM_F6_TEST_ROOT' in message                  # the override
    assert 'artifacts/grm_scout_fix8/green.json' in message   # the receipt
    assert '/mnt/ForgeRealm/wt/grm-c2/' in message        # the pruned path


def test_zero_rows_is_red_not_a_pass(tmp_path):
    """The vacuous-zero guard: an empty glob RAISES instead of returning []."""
    with pytest.raises(rp.DeadReceiptPath) as excinfo:
        rp.require_rows(sorted(tmp_path.glob('cells/*/worker.json')),
                        tmp_path, 'cells/*/worker.json', 'C2 worker census')
    message = str(excinfo.value)
    assert 'GRM_F6_VACUOUS_ZERO' in message
    assert 'C2 worker census' in message
    assert str(tmp_path) in message
    assert 'cells/*/worker.json' in message


def test_require_rows_passes_real_rows_through():
    rows = rp.require_rows(iter([1, 2, 3]), Path('/x'), 'p', 'w')
    assert rows == [1, 2, 3]          # materialised, so it cannot be re-consumed


@pytest.mark.parametrize('relative,env', sorted(REBOUND.items()))
def test_planted_dead_path_makes_the_script_fail_loud(relative, env, tmp_path):
    """Import each rebound script with its root pointed at a dead path.

    Runs in a SUBPROCESS: the failure is raised at module import time (by
    design -- failing at import is what stops a later glob from silently
    returning zero rows), so it cannot be observed by importing in-process.
    """
    dead = tmp_path / 'pruned-worktree'
    module = 'scripts.' + Path(relative).stem
    environment = dict(os.environ, **{env: str(dead)})
    environment.pop('PYTHONPATH', None)
    proc = subprocess.run(
        [sys.executable, '-c',
         'import sys; sys.path.insert(0, %r); import %s' % (str(ROOT), module)],
        cwd=str(ROOT), env=environment, capture_output=True, text=True,
        timeout=300)
    assert proc.returncode != 0, (
        '%s imported CLEANLY with %s=%s -- a dead receipt root must be loud, '
        'never a silent zero-row glob.\nstdout: %s' % (relative, env, dead,
                                                       proc.stdout))
    assert 'GRM_F6_DEAD_RECEIPT_PATH' in proc.stderr, proc.stderr
    assert str(dead) in proc.stderr
    assert env in proc.stderr


def test_c2_replay_census_is_red_on_a_dead_root(tmp_path, monkeypatch):
    """The F5 132/132 gate's own census: 0 rows must RAISE, not return [].

    Before F6 this returned an empty list against the pruned grm-c2 path and
    the caller's ``identical == len(rows) == 132`` was the only thing
    standing between that and a vacuous PASS.
    """
    from scripts import grm_lt1_offline as old
    from scripts import grm_lt1_offline_supplement as sup
    monkeypatch.setattr(old, 'C2', tmp_path / 'pruned-grm-c2')
    with pytest.raises(rp.DeadReceiptPath) as excinfo:
        sup.c2_replay()
    assert 'GRM_F6_VACUOUS_ZERO' in str(excinfo.value)
    assert 'C2 replay census' in str(excinfo.value)


def test_message_only_exemption_is_not_a_loophole(tmp_path):
    """The AST rule exempts RED text, and NOTHING else.

    Guards the guard: a module constant, a ``Path(...)`` argument and a glob
    root must all still be caught, while the same string inside a
    ``receipt_root`` failure message is allowed.
    """
    caught = tmp_path / 'caught.py'
    caught.write_text(
        "from pathlib import Path\n"
        "C2 = Path('/mnt/ForgeRealm/wt/grm-c2/artifacts')\n"
        "rows = list(Path('/mnt/ForgeRealm/wt/grm-c7').glob('*'))\n")
    assert len(_live_worktree_literals(caught)) == 2

    marker = tmp_path / 'marker.py'
    marker.write_text(
        "import pytest\n"
        "@pytest.mark.campaign_receipt(registration='artifacts/grm_c4/reg.json"
        " (pins absolute /mnt/ForgeRealm/wt/grm-c4/ paths)')\n"
        "def test_x():\n    pass\n")
    assert _live_worktree_literals(marker) == []

    allowed = tmp_path / 'allowed.py'
    allowed.write_text(
        "from scripts import grm_repo_paths as repo_paths\n"
        "C2 = repo_paths.receipt_root('artifacts/grm_c2', 'GRM_C2_EPOCH_ROOT',\n"
        "    'artifacts/grm_scout_fix8/green.json',\n"
        "    '/mnt/ForgeRealm/wt/grm-c2/artifacts')\n")
    assert _live_worktree_literals(allowed) == []


# ------------------------------------------------------- class (b) and (c)
#
# Not every dead absolute path can be rebound, and F6 does not pretend
# otherwise. These gates RECORD the classification so a later reader does not
# have to re-derive it, and so a path that silently acquires a surviving copy
# shows up as a failing assertion rather than as nothing.

def test_apamq_int4_leg_has_no_surviving_copy():
    """CLASS (b): the apa_int4 arm pins a compiled .so that no longer exists.

    ``scripts/apamq_fc_ppl.py`` and ``scripts/apamq_fbd1_diag.py`` default the
    ``apa_int4`` leg to ``/mnt/ForgeRealm/wt/apamq-fa/tensor_cuda``, and
    ``apamq_fc_ppl.resolve_tc_root``'s caller REFUSES to run that arm against
    any other root (apamq_fc_ppl.py, "apa_int4 must use FA worktree").  The
    receipt ``artifacts/apamq_fc/apa_int4.json`` pins
    ``engine.compiled_module`` to a ``.so`` INSIDE that pruned worktree.

    ``/mnt/ForgeRealm/wt/apamq-fa2`` exists, but it is a different
    Project-Tensor worktree, not that build, and the binding is to a compiled
    artifact -- so there is NO surviving copy and F6 does not invent one.
    Rebinding this would forge the receipt, and verifying a substitute needs a
    GPU run this seat does not have.  Reported, not repaired.
    """
    receipt = ROOT / 'artifacts/apamq_fc/apa_int4.json'
    if not receipt.exists():
        pytest.skip('apamq_fc receipts not mounted')
    import json
    engine = json.loads(receipt.read_text())['engine']
    pinned = Path(engine['compiled_module'])
    assert str(pinned).startswith('/mnt/ForgeRealm/wt/apamq-fa/'), engine
    assert not pinned.exists(), (
        'A copy of the pinned apa_int4 engine has appeared at %s. F6 declared '
        'this path class (b) "no surviving copy"; re-classify it.' % pinned)


def test_c7_registration_builder_keeps_its_absolute_c3_pin():
    """CLASS (c)-by-design: a registration BUILDER's pins are not rebound.

    ``scripts/grm_c7_register.py`` hashes
    ``/mnt/ForgeRealm/wt/grm-c3/orders/GRM_C3_DNGH_DECOY_CALIBRATION.md``
    into ``artifacts/grm_c7/registration.json``'s immutable inputs.  The
    order file DOES survive in-repo at ``orders/``, so this is rebindable in
    the narrow sense -- but rebinding it would change what a future
    registration hashes and under which key, i.e. it would change a RECEIPT,
    not a consumer's path resolution.  The order says: "Keep sha-bound
    registrations untouched -- they are receipts; only the consumer's path
    resolution changes."  So this is left alone, deliberately, and recorded
    here.
    """
    source = (ROOT / 'scripts/grm_c7_register.py').read_text()
    assert '/mnt/ForgeRealm/wt/grm-c3/orders/' in source
    assert (ROOT / 'orders/GRM_C3_DNGH_DECOY_CALIBRATION.md').exists(), (
        'the C3 order no longer survives in-repo either; the C7 registration '
        'builder can no longer be re-run at all')


def test_a1_checkpoint_manifest_read_is_repo_relative():
    """CLASS (a), rebound: the C7 r3 node-16 alias fixture.

    ``tests/test_grm_a1_gpu_contrast.py`` read the manifest through an
    absolute grm-c7 path. That failed LOUD (FileNotFoundError), not
    vacuously -- but the gate still could not run. The manifest is canonical
    in-repo, so the read is rebound and the U+2011 alias-parser assertion is
    live again.
    """
    manifest = (ROOT / 'artifacts/grm_c7/r3/cells/A-032-039'
                       '/checkpoint/repository/manifest.json')
    assert manifest.exists()
    assert _live_worktree_literals(
        ROOT / 'tests/test_grm_a1_gpu_contrast.py') == []
