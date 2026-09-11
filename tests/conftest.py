"""GRM-H1 suite hygiene: campaign-receipt markers + GRM_* environment isolation.

Two independent mechanisms live here.

1. ``@pytest.mark.campaign_receipt(registration=<path>)``
   A campaign-receipt test is a RECEIPT, valid only at the sha its campaign
   registered.  It either asserts ``INPUT_SHA_MISMATCH``-class binding against
   core/scripts shas frozen on the campaign's day, or reads gitignored campaign
   artifacts under ``artifacts/``.  Once the core moves, such a test fails on
   its OWN source branch too -- it is not a regression of the tree under test,
   and a tree-wide run must not report it as one.  Default collection SKIPS
   them with a reason naming the registration; ``-m campaign_receipt`` or
   ``--campaign-receipts`` runs them.

2. A ``GRM_*`` environment guard.
   Some tests pin ``GRM_ADMISSION_RULE`` (and friends) through the production
   code path (``scripts/grm_r1_replay.pin_rule`` writes ``os.environ``
   directly), and an exception escaping before the caller's cleanup leaves the
   variable set for every later module in the same pytest process.  The guard
   snapshots every ``GRM_*`` key at setup, restores it exactly (deleting keys
   that did not exist), and FAILS the test that leaked, naming the key -- so
   the leak is ATTRIBUTED to its author rather than silently mis-ruling a
   downstream module.  It runs at the END OF THE CALL PHASE (so the leak is a
   genuine FAILURE, not an "ERROR at teardown of a passing test") and excludes
   keys a live ``monkeypatch`` is still going to undo -- see
   ``_monkeypatch_owned``.  A second, silent restore wraps teardown in case a
   fixture finalizer leaks after the call phase.

Prior art
---------
* Custom markers registered via ``pytest_configure`` + deselection in
  ``pytest_collection_modifyitems``, and a CLI opt-in flag added in
  ``pytest_addoption`` -- the canonical pytest recipe documented under
  "Working with custom markers" / "Control skipping of tests" (pytest-dev,
  Holger Krekel et al., 2009-present).  Taken verbatim as an idiom; nothing
  about it is ours.
* The save/restore-environment idiom is the standard
  ``monkeypatch``/``mock.patch.dict(os.environ)`` pattern (pytest
  ``monkeypatch``, 2010-; ``unittest.mock.patch.dict``, Michael Foord, Python
  3.3, 2012).  Taken: snapshot-then-restore of ``os.environ``.
* Attributing a leak to the test that caused it, rather than only repairing
  it, is the idea behind ``pytest-env``/``pytest-randomly``'s (Adam Johnson,
  2016) order-dependence hunting and ``pytest --forked`` isolation.  Ours: the
  restore is paired with a FAILURE naming the leaked key, so a leak is a
  reported result rather than a silently swallowed one, and the leaked-key
  report survives into the test that caused it instead of surfacing in a
  victim module.
* The "receipt valid only at its registration sha" framing is this project's
  own (GRM campaign registrations, 2026); the marker is merely the transport.
  No external prior art known to me for that framing.
"""
from __future__ import annotations

import os

import pytest
from _pytest.outcomes import Failed


@pytest.fixture
def grm_env_leak_record():
    """The last leak the guard described, as ``{nodeid: [description, ...]}``.

    Exposed as a fixture (not an importable global) because a conftest is
    loaded by path, so ``from tests.conftest import _LAST_LEAK`` would bind a
    DIFFERENT module object than the one pytest is running.
    """
    return _LAST_LEAK


# --------------------------------------------------------------------------
# 1. campaign-receipt marker
# --------------------------------------------------------------------------

MARKER = "campaign_receipt"
_OPT = "--campaign-receipts"


def pytest_addoption(parser):
    parser.addoption(
        _OPT, action="store_true", default=False,
        help="run campaign-receipt tests (sha-bound to a campaign "
             "registration, or reading gitignored artifacts/). They are "
             "SKIPPED by default because they fail on any tree whose core "
             "moved past their registration sha -- including their own.")


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "campaign_receipt(registration=...): this test is a CAMPAIGN RECEIPT, "
        "valid only at the sha its campaign registered (INPUT_SHA_MISMATCH-"
        "class binding) or dependent on gitignored artifacts/. Skipped by "
        "default; run with -m campaign_receipt or --campaign-receipts.")
    config.addinivalue_line(
        "markers",
        "grm_env_leak_expected: this test deliberately leaves a GRM_* "
        "variable set; the isolation fixture restores it but does not fail "
        "the test. Used only to prove the fixture.")


def _requested(config):
    """True when the run explicitly asked for campaign receipts."""
    if config.getoption(_OPT):
        return True
    expr = config.getoption("-m", default="") or ""
    # A bare "-m campaign_receipt" (or any expression naming it positively)
    # is an explicit request. "not campaign_receipt" is not -- but pytest's
    # own -m evaluation already deselects those, so naming it at all is
    # enough to hand control back to -m.
    return MARKER in expr


def pytest_collection_modifyitems(config, items):
    if _requested(config):
        return
    for item in items:
        mark = item.get_closest_marker(MARKER)
        if mark is None:
            continue
        registration = mark.kwargs.get("registration", "unregistered")
        item.add_marker(pytest.mark.skip(reason=(
            "campaign receipt (registration=%s): sha-bound to its campaign; "
            "run with -m campaign_receipt" % registration)))


# --------------------------------------------------------------------------
# 2. GRM_* environment isolation + leak attribution
# --------------------------------------------------------------------------

PREFIX = "GRM_"

#: ``{nodeid: [description, ...]}`` for the last leak the guard described.
#: Read through the ``grm_env_leak_record`` fixture, never by import: a
#: conftest is loaded BY PATH, so ``from tests.conftest import _LAST_LEAK``
#: binds a different module object than the one pytest is running.
_LAST_LEAK: dict = {}

_SETUP_SNAPSHOT = "_grm_h1_env_before"


def _snapshot():
    return {k: v for k, v in os.environ.items() if k.startswith(PREFIX)}


def _describe(before, after):
    """Sorted human-readable per-key descriptions of every GRM_* difference."""
    out = []
    for key in sorted(set(before) | set(after)):
        was, now = before.get(key), after.get(key)
        if was == now:
            continue
        if was is None:
            out.append("%s: unset -> %r (set and not cleaned up)" % (key, now))
        elif now is None:
            out.append("%s: %r -> unset (deleted and not restored)" % (key, was))
        else:
            out.append("%s: %r -> %r (mutated and not restored)"
                       % (key, was, now))
    return out


def _restore(before, after):
    """Put the GRM_* environment back EXACTLY as it was at setup."""
    for key in list(after):
        if key not in before:
            os.environ.pop(key, None)
    for key, value in before.items():
        os.environ[key] = value


def _monkeypatch_owned(item):
    """GRM_* keys a ``monkeypatch`` fixture is still going to undo.

    The guard has to run at the END OF THE CALL PHASE so a leak is a genuine
    FAILURE of the leaking test rather than an "ERROR at teardown of <a test
    that passed>".  But ``monkeypatch`` -- the CORRECT way to pin a variable
    -- only undoes itself during teardown, which is AFTER that point.  So the
    guard asks the live ``monkeypatch`` object which keys it has recorded and
    excludes exactly those: a monkeypatched variable is pinned, not leaked.

    ``MonkeyPatch._setitem`` is a private list of ``(mapping, name, value)``
    undo records (pytest 3.x-9.x).  If a future pytest renames it, this
    returns the empty set and the guard becomes conservative in the SAFE
    direction -- monkeypatch users get flagged, loudly, rather than real
    leaks being silently let through.
    """
    owned = set()
    for value in getattr(item, "funcargs", {}).values():
        records = getattr(value, "_setitem", None)
        if not records:
            continue
        for record in records:
            try:
                mapping, name = record[0], record[1]
            except (TypeError, IndexError):        # pragma: no cover
                continue
            if mapping is os.environ and str(name).startswith(PREFIX):
                owned.add(str(name))
    return owned


LEAK_MESSAGE = (
    "GRM_ENV_LEAK: this test left the GRM_* environment modified. "
    "The variables have been restored so later tests are not mis-ruled, "
    "but the leak is attributed here, not hidden:\n  %s\n"
    "Pin GRM_* with monkeypatch.setenv / a try-finally, never a bare "
    "os.environ assignment that an exception can skip.")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item):
    """Snapshot GRM_* BEFORE any fixture of this test has run."""
    setattr(item, _SETUP_SNAPSHOT, _snapshot())
    yield


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """Restore GRM_*, and FAIL the test that leaked one -- naming the key."""
    outcome = yield
    before = getattr(item, _SETUP_SNAPSHOT, None)
    if before is None:                              # pragma: no cover
        return
    after = _snapshot()
    if before == after:
        return

    pinned = _monkeypatch_owned(item)
    leaks = [line for line in _describe(before, after)
             if line.split(":", 1)[0] not in pinned]

    # Restore EXACTLY, for every changed key -- including the monkeypatched
    # ones, which monkeypatch will then redundantly (and harmlessly) undo.
    # The restore is unconditional so a leak can never reach the next test,
    # even when the attribution below raises.
    _restore(before, after)

    if not leaks:
        return                                      # all of it was pinned

    _LAST_LEAK.clear()
    _LAST_LEAK[item.nodeid] = list(leaks)

    if item.get_closest_marker("grm_env_leak_expected") is not None:
        return
    if outcome.excinfo is not None:
        # Already failing on its own terms; that receipt is the useful one,
        # and the restore above happened either way.
        return

    outcome.force_exception(
        Failed(LEAK_MESSAGE % "\n  ".join(leaks), pytrace=False))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item, nextitem):
    """Belt and braces: a fixture finalizer can leak after the call phase."""
    yield
    before = getattr(item, _SETUP_SNAPSHOT, None)
    if before is None:                              # pragma: no cover
        return
    after = _snapshot()
    if before != after:
        _restore(before, after)
