#!/usr/bin/env bash
# Flock wrapper for a multi-probe DRIVER (holds /tmp/forge-gpu.lock ONCE while the
# driver spawns one setsid'd, timeout-capped probe per rung). The per-probe cap
# lives in the driver (--probe-timeout); the outer has an overall ceiling so a
# runaway driver still releases the lock. Poller runs 1s the whole time.
# Usage: run_gpu_driver.sh <poller_log> <overall_timeout_s> <cmd...>
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLL_LOG="${1:?poller log path required}"; shift
OVERALL="${1:?overall timeout required}"; shift
LOCK=/tmp/forge-gpu.lock

exec 9>"$LOCK"
echo "[run_gpu_driver] waiting for lock $LOCK ..."
flock -w 3600 9 || { echo "[run_gpu_driver] FAILED to acquire lock"; exit 99; }
echo "[run_gpu_driver] lock acquired; poller -> $POLL_LOG; overall_timeout=${OVERALL}s"

setsid bash "$HERE/poller.sh" "$POLL_LOG" >/dev/null 2>&1 &
POLLER_PID=$!
cleanup() { kill -- -"$POLLER_PID" 2>/dev/null; kill "$POLLER_PID" 2>/dev/null; }
trap cleanup EXIT

echo "[run_gpu_driver] exec: timeout $OVERALL $*"
timeout "$OVERALL" "$@"
RC=$?
echo "[run_gpu_driver] driver rc=$RC"
if [[ -s "$POLL_LOG" ]]; then
  PEAK=$(awk '{if($2+0>m)m=$2} END{print m+0}' "$POLL_LOG")
  echo "[run_gpu_driver] poller_peak_MiB=$PEAK"
fi
exit $RC
