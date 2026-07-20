#!/usr/bin/env bash
# NC17-P3 flock wrapper for a multi-probe DRIVER — FORK INT6 ENGINE.
# Identical to run_gpu_driver.sh but exports PYTHONPATH to the Project-Tensor
# fork so the driver AND every setsid'd probe subprocess it spawns import the
# INT6 build (env is inherited by the children). Holds /tmp/forge-gpu.lock once.
# Usage: run_gpu_driver_int6.sh <poller_log> <overall_timeout_s> <cmd...>
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLL_LOG="${1:?poller log path required}"; shift
OVERALL="${1:?overall timeout required}"; shift
LOCK=/tmp/forge-gpu.lock
INT6_ENGINE=/mnt/ForgeRealm/Project-Tensor-int6/tensor_cuda
export PYTHONPATH="$INT6_ENGINE${PYTHONPATH:+:$PYTHONPATH}"

exec 9>"$LOCK"
echo "[run_gpu_driver_int6] engine=$INT6_ENGINE"
echo "[run_gpu_driver_int6] waiting for lock $LOCK ..."
flock -w 3600 9 || { echo "[run_gpu_driver_int6] FAILED to acquire lock"; exit 99; }
echo "[run_gpu_driver_int6] lock acquired; poller -> $POLL_LOG; overall_timeout=${OVERALL}s"

setsid bash "$HERE/poller.sh" "$POLL_LOG" >/dev/null 2>&1 &
POLLER_PID=$!
cleanup() { kill -- -"$POLLER_PID" 2>/dev/null; kill "$POLLER_PID" 2>/dev/null; }
trap cleanup EXIT

echo "[run_gpu_driver_int6] exec: timeout $OVERALL $*"
timeout "$OVERALL" "$@"
RC=$?
echo "[run_gpu_driver_int6] driver rc=$RC"
if [[ -s "$POLL_LOG" ]]; then
  PEAK=$(awk '{if($2+0>m)m=$2} END{print m+0}' "$POLL_LOG")
  echo "[run_gpu_driver_int6] poller_peak_MiB=$PEAK"
fi
exit $RC
