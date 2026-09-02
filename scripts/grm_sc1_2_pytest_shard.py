#!/usr/bin/env python3
"""GRM-SC1.2 G1 — run one pytest shard under the house GPU lease.

ORDER: ``orders/GRM_SC1_2_E2E_PAIRS_SESSION_RESUME.md`` (gate G1).

WHY A SHARD RUNNER EXISTS.  Two house rules collide with a whole-suite
``pytest``: every Bash call must stay under ten minutes, and every process
that touches the card must hold the ``/tmp/forge-gpu.lock`` lease with a cap
(so the operator keeps absolute right of way).  A single suite run satisfies
neither.  This runs one named group of test FILES inside one lease and prints
its own ``SHARD_RC``, so the caller can compose the pre-existing-set
comparison from foreground calls that each finish well inside the budget.

It is a harness, not a gate: it decides nothing.  The pass/fail comparison
against SC1.1's baseline is made from the logs it writes.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: Below the house 580 s cap and below the 10-minute per-call budget, with
#: room for the lease to release and the log to flush.
LEASE_SECONDS = 520
LOCK_WAIT_SECONDS = 7200


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", help="test files, repo-relative")
    parser.add_argument("--lease-seconds", type=int, default=LEASE_SECONDS)
    parser.add_argument("--lock-wait-seconds", type=int,
                        default=LOCK_WAIT_SECONDS)
    parser.add_argument("--timeout-seconds", type=int, default=500)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    from scripts.grm_cmc1_gpu_arms import gpu_lease

    # FLAGS AFTER THE FILES, deliberately.  ``tests/test_scribe_floor_gates.py``
    # reads ``sys.argv[1]`` AT IMPORT TIME, so a flag sitting in that slot is
    # swallowed as a path and the file dies with a FileNotFoundError naming the
    # flag -- a harness artifact that looks exactly like a real collection
    # error.  Measured once; pinned here so it cannot recur.
    command = [
        sys.executable, "-m", "pytest", "-q", *args.files,
        "--continue-on-collection-errors",
    ]
    with gpu_lease(int(args.lease_seconds), int(args.lock_wait_seconds)):
        try:
            completed = subprocess.run(
                command, cwd=str(ROOT), timeout=int(args.timeout_seconds))
            rc = completed.returncode
        except subprocess.TimeoutExpired:
            # A shard that outruns the lease is a RESULT about that shard, not
            # a reason to widen the lease.  Report it as its own code.
            print("SHARD_TIMEOUT=1", flush=True)
            rc = 124
    print(f"SHARD_RC={rc}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
