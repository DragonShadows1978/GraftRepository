#!/usr/bin/env python3
"""GRM-F6: resolve campaign receipt roots REPO-RELATIVE, never into a worktree.

Why this module exists
----------------------
Round-1 seat worktrees ``/mnt/ForgeRealm/wt/grm-*`` were pruned on 2026-09-11.
Several consumer scripts hard-coded absolute paths into them.  The dangerous
failure is not the loud one: a ``Path(dead).glob(...)`` returns an EMPTY
iterator, so a census loop produces zero rows, and any gate phrased as
"every row agrees" passes VACUOUSLY over nothing.  ``scripts/grm_lt1_offline``
(F5's finding), ``scripts/grm_rd1`` and ``scripts/grm_scout_fix8_cpu`` (H2's)
each had exactly that shape.

Contract
--------
* :func:`repo_root` is the repository this file lives in, resolved from
  ``__file__`` -- so a checkout, a worktree or a moved clone all work, and no
  seat worktree name is ever baked into a script.
* :func:`receipt_root` returns ``repo_root() / relative``, overridable per
  script by an environment variable, and FAILS LOUD when the directory is
  absent, naming the missing path, the env var that overrides it, and the
  receipt that originally pinned the dead absolute path.
* :func:`require_rows` is the vacuous-zero guard: a glob/list that feeds a
  gate is passed through it, and ``0`` rows is a RED that names the path and
  the pattern, never a silent pass.

Registrations are NOT touched.  A ``registration.json`` that sha-binds
``/mnt/ForgeRealm/wt/grm-c7/...`` is a receipt of what was hashed on the day;
rewriting it would forge the receipt.  Only the CONSUMER's path resolution
moves.

Prior art
---------
* Resolving a package/repo root from ``Path(__file__).resolve().parents[n]``
  instead of the process CWD is the standard Python idiom (setuptools /
  pytest ``rootdir`` discovery, pytest-dev 2009-; ``Path.resolve`` semantics,
  PEP 428, Antoine Pitrou, 2012).  Taken verbatim; nothing about it is ours.
  Eleven scripts in this very repo already do it (e.g.
  ``scripts/grm_rd1.py:21``) -- F6 only extends it to the DATA roots those
  same scripts still pinned absolutely.
* Environment-variable override with a repo-relative default is the
  twelve-factor config idiom (Adam Wiggins, 2011) and matches this repo's own
  ``TENSOR_CUDA_ROOT`` / ``GRM_*`` convention.
* Treating an empty result set as an ERROR rather than a vacuous pass is the
  "vacuous truth" hazard named in formal-methods literature and operational-
  ised as, e.g., ``NOT NULL``/row-count assertions in dbt tests (Fishtown
  Analytics, 2018), Great Expectations' ``expect_table_row_count_to_be_
  between`` (Superconductive, 2018), and pytest's own
  ``--collect-only``-returns-nothing exit code 5 (pytest-dev, 2019).  Taken:
  the principle that "zero rows" must be distinguishable from "all rows
  agree".  Ours: nothing algorithmic -- only the wiring of that check into
  these particular receipt consumers, with the dead absolute path and the
  naming receipt carried in the message so the RED is diagnosable without
  reading the script.
* No prior art known to me for the specific "pruned seat worktree leaves a
  vacuously-passing receipt gate" failure mode as a named pattern; it is an
  instance of the vacuous-truth hazard above.
"""
from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """The repository root, resolved from THIS FILE, never from the CWD."""
    return Path(__file__).resolve().parents[1]


class DeadReceiptPath(RuntimeError):
    """A receipt root does not exist on disk.  Loud, with the whole story."""


def receipt_root(relative: str, env: str, pinned_by: str,
                 dead_absolute: str | None = None,
                 root: Path | None = None) -> Path:
    """Resolve a campaign receipt root repo-relative, or fail LOUD.

    Parameters
    ----------
    relative
        Path under the repository root, e.g. ``'artifacts/grm_c2/epochs/
        scout-fix-2'``.
    env
        Name of the per-script environment variable that overrides the
        ROOT (``relative`` is appended to whatever it names).
    pinned_by
        The receipt / registration that named the original absolute path, so
        a RED says where the binding came from.
    dead_absolute
        The pruned absolute path this replaced, quoted in the error.
    root
        Override for the repo root (tests).
    """
    override = os.environ.get(env)
    if override:
        # The override names a ROOT, and ``relative`` is appended to it, so a
        # caller that also needs the root back (RD1's ``source_root``) gets a
        # consistent answer from both. An override pointing straight at the
        # leaf directory would break that invariant, so it is not accepted.
        base = Path(override).expanduser()
        path = base / relative
        source = "%s=%s + %s" % (env, override, relative)
    else:
        base = root or repo_root()
        path = base / relative
        source = "repo-relative %s" % relative
    if not path.exists():
        raise DeadReceiptPath(
            "GRM_F6_DEAD_RECEIPT_PATH: %s resolved to %s, which does not "
            "exist.\n  resolution: %s\n  pinned by: %s%s\n"
            "  A campaign receipt root must resolve inside the repository. "
            "Set %s to an existing copy, or report the path as having no "
            "surviving copy -- do NOT let a glob over it return zero rows."
            % (env, path, source, pinned_by,
               ("\n  pruned absolute path it replaced: %s" % dead_absolute)
               if dead_absolute else "",
               env))
    return path


def require_rows(rows, path, pattern, what):
    """Vacuous-zero guard: 0 rows feeding a gate is RED, never a pass.

    ``rows`` may be any sized/iterable result; it is materialised to a list
    so a generator cannot be silently consumed to nothing downstream.
    """
    rows = list(rows)
    if not rows:
        raise DeadReceiptPath(
            "GRM_F6_VACUOUS_ZERO: %s matched 0 rows under %s (pattern %r). "
            "A gate over zero rows passes vacuously, so this is RED. Either "
            "the receipt root is dead/misresolved, or the pattern no longer "
            "matches the archive layout."
            % (what, path, pattern))
    return rows
