#!/usr/bin/env bash
# Prior art: house DET1 foreground flock runners (2026), util-linux flock.
# X3 owns no background workers and never sends a kill signal.
set -euo pipefail
X3_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$X3_ROOT"
X3_PY="${PYTHON:-python3}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$X3_ROOT:/mnt/ForgeRealm/Project-Tensor/tensor_cuda${PYTHONPATH:+:$PYTHONPATH}"
if [[ "${1:-}" == "_lease" ]]; then
    [[ "$#" -eq 3 ]]
    exec 9<>/tmp/forge-gpu.lock
    flock --wait 240 9
    export GRM_X3_LEASED=1
    exec "$X3_PY" scripts/grm_x3_r2_lead.py _leased "$2" "$3"
fi
exec "$X3_PY" scripts/grm_x3_r2_lead.py "$@"
