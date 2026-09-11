"""GRM-F5 OFF byte-identity gate: the C2 132-plan replay.

Runs the FIX-6 recorded-state replay (scripts/grm_scout_fix6_replay.py's
`c2_replay` source) over all 132 recorded C2 executions and asserts that
every OFF plan reproduces the GPU-recorded `rank_plan` byte-for-byte.

`scripts/grm_lt1_offline.C2` is a hard-coded absolute path into the
now-removed /mnt/ForgeRealm/wt/grm-c2 worktree; the same epoch data is
present in this worktree under artifacts/grm_c2/epochs/scout-fix-2, so the
module constant is REBOUND here (read-only; no source file is edited).
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path('/mnt/ForgeRealm/wt/grm-f5')
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from scripts import grm_lt1_offline as old
old.C2 = ROOT / 'artifacts/grm_c2/epochs/scout-fix-2'

from scripts.grm_lt1_offline_supplement import c2_replay

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
