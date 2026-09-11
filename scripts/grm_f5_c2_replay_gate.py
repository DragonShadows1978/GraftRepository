#!/usr/bin/env python3
"""GRM-F5 OFF byte-identity gate: the C2 132-plan replay.

Runs the FIX-6 recorded-state replay (scripts/grm_scout_fix6_replay.py's
`c2_replay` source) over all 132 recorded C2 executions and asserts that
every OFF plan reproduces the GPU-recorded `rank_plan` byte-for-byte.

GRM-F6 (2026-09-11) rewrite of the path handling
------------------------------------------------
This file previously did::

    ROOT = Path('/mnt/ForgeRealm/wt/grm-f5')
    sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
    from scripts import grm_lt1_offline as old
    old.C2 = ROOT / 'artifacts/grm_c2/epochs/scout-fix-2'

i.e. it worked around `grm_lt1_offline.C2` being a dead absolute path into
the pruned `grm-c2` worktree by REBINDING the constant -- to another
absolute seat-worktree path (`grm-f5`), which is the identical bug one
worktree later: as soon as `grm-f5` is pruned, ROOT is dead, the chdir
raises, and (had it not) the glob would return zero rows again.

`grm_lt1_offline.C2` is now repo-relative in its own right (see
scripts/grm_repo_paths.py), so the rebind is DELETED rather than repointed,
and ROOT resolves from this file. The gate's arithmetic is unchanged:
132 rows, 132 byte-identical, or FAIL.

Prior art: repo-root-from-__file__ (setuptools / pytest ``rootdir``
discovery, pytest-dev 2009-; PEP 428 ``Path.resolve``, Antoine Pitrou,
2012) -- taken verbatim. The 132/132 byte-identity criterion and the
`c2_replay` source are GRM FIX-6 / F5's (GRM contributors, 2026), unchanged
here. No prior art known to me for this gate's exact composition; F6
changed only path resolution, no gate logic.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.grm_lt1_offline_supplement import c2_replay  # noqa: E402

rows = c2_replay()
identical = sum(1 for r in rows if r['baseline_plan_parity'])
flag = os.environ.get('GRM_ROUTE_SOLE_BINDER_INSURANCE', '<unset>')
print(json.dumps({
    'flag_GRM_ROUTE_SOLE_BINDER_INSURANCE': flag,
    'rows': len(rows),
    'off_byte_identical': identical,
    'verdict': 'PASS' if identical == len(rows) == 132 else 'FAIL',
}, indent=2))
sys.exit(0 if identical == len(rows) == 132 else 1)
