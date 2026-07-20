#!/usr/bin/env bash
# NC17-P3y flock wrapper — FORK INT6 ENGINE. Runs the single-process
# incremental-32k probe under flock -w 7200 with a setsid poller, timeout 3000
# per invocation (task rails). After the run, splices the poller peak into the
# probe's JSON out file. Holds /tmp/forge-gpu.lock once.
# Usage: p3y_driver.sh <poller_log> <out_json> <python_cmd...>
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLL_LOG="${1:?poller log path required}"; shift
OUT_JSON="${1:?out json path required}"; shift
LOCK=/tmp/forge-gpu.lock
INT6_ENGINE=/mnt/ForgeRealm/Project-Tensor-int6/tensor_cuda
export PYTHONPATH="$INT6_ENGINE${PYTHONPATH:+:$PYTHONPATH}"

exec 9>"$LOCK"
echo "[p3y_driver] engine=$INT6_ENGINE"
echo "[p3y_driver] waiting for lock $LOCK (flock -w 7200) ..."
flock -w 7200 9 || { echo "[p3y_driver] FAILED to acquire lock within 7200s"; exit 99; }
echo "[p3y_driver] lock acquired $(date +%T); poller -> $POLL_LOG; timeout=3000s"

setsid bash "$HERE/poller.sh" "$POLL_LOG" >/dev/null 2>&1 &
POLLER_PID=$!
cleanup() { kill -- -"$POLLER_PID" 2>/dev/null; kill "$POLLER_PID" 2>/dev/null; }
trap cleanup EXIT

echo "[p3y_driver] exec: timeout 3000 $*"
timeout 3000 "$@"
RC=$?
echo "[p3y_driver] python rc=$RC"

PEAK=""
if [[ -s "$POLL_LOG" ]]; then
  PEAK=$(awk '{if($2+0>m)m=$2} END{print m+0}' "$POLL_LOG")
  echo "[p3y_driver] poller_peak_MiB=$PEAK"
fi

# splice poller peak + driver rc into the out JSON (python, no jq dependency)
if [[ -f "$OUT_JSON" && -n "$PEAK" ]]; then
  python3 - "$OUT_JSON" "$PEAK" "$RC" <<'PY'
import json, sys
p, peak, rc = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
d = json.load(open(p))
d["poller_peak_mb"] = peak
d["_driver_rc"] = rc
json.dump(d, open(p, "w"), indent=2)
print(f"[p3y_driver] spliced poller_peak_mb={peak} driver_rc={rc} -> {p}")
PY
fi
exit $RC
