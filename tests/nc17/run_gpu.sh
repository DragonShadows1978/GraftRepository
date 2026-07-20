#!/usr/bin/env bash
# Foreground, blocking, flock-wrapped GPU runner for NC17-P0.
# Usage: run_gpu.sh <poller_log> <python_cmd...>
# - Acquires /tmp/forge-gpu.lock (flock -w 3600): waiting on the lock is normal.
# - Starts the 1s nvidia-smi poller with setsid inside the flocked shell.
# - Runs the python command under `timeout 590` (<=10 min hard cap).
# - Always kills the poller and reports the poller peak MiB on exit.
# NEVER nohup/detach/background the GPU work itself. OOM is a measurement.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLL_LOG="${1:?poller log path required}"; shift
LOCK=/tmp/forge-gpu.lock

exec 9>"$LOCK"
echo "[run_gpu] waiting for lock $LOCK ..."
flock -w 3600 9 || { echo "[run_gpu] FAILED to acquire lock within 3600s"; exit 99; }
echo "[run_gpu] lock acquired; starting poller -> $POLL_LOG"

setsid bash "$HERE/poller.sh" "$POLL_LOG" >/dev/null 2>&1 &
POLLER_PID=$!

cleanup() {
  kill -- -"$POLLER_PID" 2>/dev/null
  kill "$POLLER_PID" 2>/dev/null
}
trap cleanup EXIT

echo "[run_gpu] exec: timeout 590 $*"
timeout 590 "$@"
RC=$?
echo "[run_gpu] python rc=$RC"

if [[ -s "$POLL_LOG" ]]; then
  PEAK=$(awk '{if($2+0>m)m=$2} END{print m+0}' "$POLL_LOG")
  echo "[run_gpu] poller_peak_MiB=$PEAK"
fi
exit $RC
