"""GRM-H1: prove the GRM_* environment guard in tests/conftest.py.

The guard is only worth having if it can (a) actually FAIL the test that
leaked, naming the key, and (b) restore the environment so the NEXT test is
not mis-ruled.  Both are proved here by planting real leaks.

(a) is proved with an inner pytest run (``pytester``-style, but via a plain
subprocess so this file needs no plugin): a throwaway test module that leaks
``GRM_ADMISSION_RULE`` is run with this repository's own ``tests/conftest.py``
copied in, and the run must fail with ``GRM_ENV_LEAK`` naming the key.

(b) is proved in-process: a test marked ``grm_env_leak_expected`` plants a
leak (the fixture restores it without failing), and the next test asserts the
variable is gone.

Prior art: pytest's own ``pytester``/``testdir`` fixture (pytest-dev, 2009-)
is the established way to test a plugin by running pytest inside pytest --
reused here as a subprocess so no extra plugin has to be enabled.  The
planted-leak-then-assert-attribution shape is ours; no external prior art
known to me for this specific assertion.
"""
from __future__ import annotations

import os
import subprocess
import shutil
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFTEST = ROOT / "tests" / "conftest.py"

KEY = "GRM_H1_ISOLATION_PROBE"
RULE = "GRM_ADMISSION_RULE"


# -- (a) the guard FAILS the leaker, and names the key ----------------------

def _run_inner(tmp_path, body):
    """Run one throwaway test module under a copy of our conftest.py."""
    pkg = tmp_path / "inner"
    pkg.mkdir()
    shutil.copy(CONFTEST, pkg / "conftest.py")
    (pkg / "test_inner.py").write_text(textwrap.dedent(body))
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header", "-p",
         "no:cacheprovider", str(pkg / "test_inner.py")],
        cwd=str(tmp_path), capture_output=True, text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})


def test_a_bare_os_environ_set_is_attributed_to_its_own_test(tmp_path):
    out = _run_inner(tmp_path, """
        import os
        def test_leaks_the_admission_rule():
            os.environ['GRM_ADMISSION_RULE'] = 'margin_first'
            assert True          # the test body itself PASSES
    """)
    assert out.returncode != 0, out.stdout
    assert "GRM_ENV_LEAK" in out.stdout
    assert "GRM_ADMISSION_RULE" in out.stdout
    assert "unset -> 'margin_first'" in out.stdout
    # attributed to the leaker, not to some later victim
    assert "test_leaks_the_admission_rule" in out.stdout


def test_a_deleted_var_is_attributed_too(tmp_path):
    out = _run_inner(tmp_path, """
        import os
        os.environ['GRM_PREEXISTING'] = 'keepme'
        def test_deletes_a_var():
            del os.environ['GRM_PREEXISTING']
    """)
    assert out.returncode != 0, out.stdout
    assert "GRM_ENV_LEAK" in out.stdout
    assert "GRM_PREEXISTING" in out.stdout
    assert "-> unset" in out.stdout


def test_a_mutated_var_is_attributed_too(tmp_path):
    out = _run_inner(tmp_path, """
        import os
        os.environ['GRM_PREEXISTING'] = 'before'
        def test_mutates_a_var():
            os.environ['GRM_PREEXISTING'] = 'after'
    """)
    assert out.returncode != 0, out.stdout
    assert "GRM_ENV_LEAK" in out.stdout
    assert "'before' -> 'after'" in out.stdout


def test_the_next_test_in_the_inner_run_is_not_mis_ruled(tmp_path):
    """The restore happens even though the leaker is failed."""
    out = _run_inner(tmp_path, """
        import os
        def test_one_leaks():
            os.environ['GRM_ADMISSION_RULE'] = 'margin_first'
        def test_two_sees_a_clean_environment():
            assert 'GRM_ADMISSION_RULE' not in os.environ
    """)
    assert "1 failed, 1 passed" in out.stdout, out.stdout
    assert "error" not in out.stdout, (
        "the leak must be a FAILURE of the leaker, not a teardown ERROR")


def test_a_clean_test_is_not_flagged(tmp_path):
    out = _run_inner(tmp_path, """
        import os
        def test_uses_monkeypatch(monkeypatch):
            monkeypatch.setenv('GRM_ADMISSION_RULE', 'margin_first')
            assert os.environ['GRM_ADMISSION_RULE'] == 'margin_first'
    """)
    assert out.returncode == 0, out.stdout
    assert "GRM_ENV_LEAK" not in out.stdout


def test_non_grm_variables_are_not_policed(tmp_path):
    out = _run_inner(tmp_path, """
        import os
        def test_sets_an_unrelated_var():
            os.environ['SOME_OTHER_VAR'] = '1'
    """)
    assert out.returncode == 0, out.stdout


def test_a_wholesale_environ_clear_is_repaired_for_the_next_test(tmp_path):
    """GRM-H1 follow-up: `pin_flags()` does os.environ.clear().

    A test that reaches that on an exception path wipes PATH/HOME/DISPLAY for
    every later test in the process.  It is outside the GRM_* contract the
    guard ATTRIBUTES on -- so the wiping test is not failed for it -- but the
    repair must still happen, or one receipt takes the whole run down with it.
    """
    out = _run_inner(tmp_path, """
        import os
        os.environ['GRM_PREEXISTING'] = 'keepme'
        os.environ['NOT_A_GRM_VAR'] = 'keepme too'
        def test_wipes_everything():
            os.environ.clear()
        def test_two_still_has_its_environment():
            assert os.environ.get('NOT_A_GRM_VAR') == 'keepme too'
            assert os.environ.get('GRM_PREEXISTING') == 'keepme'
            assert 'PATH' in os.environ
    """)
    # The wiper is failed for the GRM_ key it deleted; the NEXT test passes,
    # which is the point.
    assert "1 failed, 1 passed" in out.stdout, out.stdout
    assert "GRM_PREEXISTING" in out.stdout


# -- (b) the restore really happens, in THIS process ------------------------

@pytest.mark.grm_env_leak_expected
def test_plant_a_leak_in_this_process():
    """Marked, so the fixture restores without failing us."""
    assert KEY not in os.environ
    os.environ[KEY] = "planted"


def test_the_planted_leak_did_not_reach_this_test():
    assert KEY not in os.environ


@pytest.mark.grm_env_leak_expected
def test_the_leak_was_recorded_for_attribution(grm_env_leak_record):
    _LAST_LEAK = grm_env_leak_record
    os.environ[KEY] = "planted-again"
    # _LAST_LEAK is written by the fixture's teardown, so at THIS point it
    # still holds the previous planted leak -- which is the receipt that the
    # fixture described the key rather than swallowing it.
    assert _LAST_LEAK, "the fixture recorded no leak at all"
    (nodeid, leaks), = _LAST_LEAK.items()
    assert "test_plant_a_leak_in_this_process" in nodeid
    assert any(KEY in line and "unset ->" in line for line in leaks), leaks


# -- the marker itself is registered ---------------------------------------

def test_entrypoint_owned_environments_are_restored_but_not_attributed(tmp_path):
    """`grm_env_owned_by_entrypoint` suppresses attribution, NOT the restore.

    `scripts/grm_a1_gpu_contrast.pin_flags()` clears and repopulates the whole
    environment by design (it normally runs as a one-shot worker process).
    Failing every test that drives it would be the guard mis-ruling working
    code, so a module can declare the ownership -- but the restore must still
    happen, or one such module poisons the rest of the run.
    """
    out = _run_inner(tmp_path, """
        import os, pytest
        pytestmark = pytest.mark.grm_env_owned_by_entrypoint(reason='proof')
        os.environ['GRM_PREEXISTING'] = 'keepme'
        def test_reshapes_the_environment():
            os.environ['GRM_ADMISSION_RULE'] = 'margin_first'
            del os.environ['GRM_PREEXISTING']
        def test_two_sees_the_restored_environment():
            assert 'GRM_ADMISSION_RULE' not in os.environ
            assert os.environ.get('GRM_PREEXISTING') == 'keepme'
    """)
    assert out.returncode == 0, out.stdout
    assert "2 passed" in out.stdout, out.stdout
    assert "GRM_ENV_LEAK" not in out.stdout


def test_the_ownership_marker_does_not_leak_to_other_modules(tmp_path):
    """It is per-module opt-in; an undeclared module is still policed."""
    out = _run_inner(tmp_path, """
        import os
        def test_undeclared_module_still_fails():
            os.environ['GRM_ADMISSION_RULE'] = 'margin_first'
    """)
    assert out.returncode != 0, out.stdout
    assert "GRM_ENV_LEAK" in out.stdout


def test_campaign_receipt_marker_is_registered(pytestconfig):
    names = pytestconfig.getini("markers")
    assert any(n.startswith("campaign_receipt(") for n in names), names
    assert any(n.startswith("grm_env_leak_expected") for n in names), names
    assert any(n.startswith("grm_env_owned_by_entrypoint(") for n in names), names
