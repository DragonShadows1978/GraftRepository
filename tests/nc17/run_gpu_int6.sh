#!/usr/bin/env bash
# NC17-P3 foreground flock-wrapped GPU runner — FORK INT6 ENGINE.
# Identical to run_gpu.sh but exports PYTHONPATH to the Project-Tensor fork
# (branch int6-weights) so `import tensor_cuda` resolves the INT6 build. Every
# P3 GPU run goes through this so the engine build is unambiguous.
# Usage: run_gpu_int6.sh <poller_log> <python_cmd...>
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLL_LOG="${1:?poller log path required}"; shift
LOCK=/tmp/forge-gpu.lock
INT6_ENGINE=/mnt/ForgeRealm/Project-Tensor-int6/tensor_cuda
export PYTHONPATH="$INT6_ENGINE${PYTHONPATH:+:$PYTHONPATH}"

exec 9>"$LOCK"
echo "[run_gpu_int6] engine=$INT6_ENGINE"
echo "[run_gpu_int6] waiting for lock $LOCK ..."
flock -w 3600 9 || { echo "[run_gpu_int6] FAILED to acquire lock within 3600s"; exit 99; }
echo "[run_gpu_int6] lock acquired; starting poller -> $POLL_LOG"

setsid bash "$HERE/poller.sh" "$POLL_LOG" >/dev/null 2>&1 &
POLLER_PID=$!
cleanup() { kill -- -"$POLLER_PID" 2>/dev/null; kill "$POLLER_PID" 2>/dev/null; }
trap cleanup EXIT

echo "[run_gpu_int6] exec: timeout 590 $*"
timeout 590 "$@"
RC=$?
echo "[run_gpu_int6] python rc=$RC"
if [[ -s "$POLL_LOG" ]]; then
  PEAK=$(awk '{if($2+0>m)m=$2} END{print m+0}' "$POLL_LOG")
  echo "[run_gpu_int6] poller_peak_MiB=$PEAK"
fi
exit $RC
