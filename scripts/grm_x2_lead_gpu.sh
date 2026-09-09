#!/usr/bin/env bash
# Prior art: house GPU lease wrappers (2026), util-linux flock (year
# unverified — lead to check: flock manual). Ours: X2 command dispatch only.
set -euo pipefail
GRM_X2_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$GRM_X2_ROOT"
export PYTHONDONTWRITEBYTECODE=1
exec python3 scripts/grm_x2_gpu.py "$@"
