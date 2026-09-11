#!/usr/bin/env bash
# Prior art: util-linux flock (established advisory locking, year unverified —
# lead to check "util-linux flock"), house bounded leased runners (2026).
# New: X1 create-only cell controller. One foreground cell per run/resume.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT:/mnt/ForgeRealm/Project-Tensor/tensor_cuda${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python3}" scripts/grm_x1_campaign.py "$@"
