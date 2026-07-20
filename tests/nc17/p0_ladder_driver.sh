#!/usr/bin/env bash
# NC17-P0 OOM ladder driver (binary search), ONE PROBE PER SUBPROCESS.
# Each probe = a fresh flock-wrapped foreground run_gpu.sh call with its own
# poller. We orchestrate the binary search here but never hold GPU state
# between probes (Gemma law: fragmentation fakes cross-context OOMs).
#
# Usage: p0_ladder_driver.sh <prefill|decode> <lo> <hi>
#   lo must be known-solid-ish, hi known-or-suspected-OOM. Binary-searches to
#   the wall; records every probe to logs/nc17/ladder_<mode>.jsonl and prints
#   last-solid / first-oom with poller peaks.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
MODE="${1:?mode: prefill|decode}"
LO="${2:?lo}"
HI="${3:?hi}"
LOGDIR="$REPO_ROOT/logs/nc17"
JSONL="$LOGDIR/ladder_${MODE}.jsonl"
mkdir -p "$LOGDIR"

# one probe: run in a fresh flocked process; echoes back "rc <json> peak"
probe() {
  local S="$1"
  local plog="$LOGDIR/poll_${MODE}_S${S}.log"
  local rlog="$LOGDIR/probe_${MODE}_S${S}.log"
  echo "[driver] probe mode=$MODE S=$S" >&2
  bash "$HERE/run_gpu.sh" "$plog" python3 "$HERE/p0_oom_ladder.py" "$MODE" "$S" \
    > "$rlog" 2>&1
  local rc=$?
  local jline
  jline=$(grep '^RESULT ' "$rlog" | tail -1 | sed 's/^RESULT //')
  local peak
  peak=$(grep 'poller_peak_MiB=' "$rlog" | tail -1 | sed 's/.*poller_peak_MiB=//')
  [[ -z "$jline" ]] && jline='{}'
  # merge rc + poller peak into the json record
  python3 - "$MODE" "$S" "$rc" "$peak" "$jline" >> "$JSONL" <<'PY'
import json, sys
mode, S, rc, peak, jline = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
try:
    rec = json.loads(jline)
except Exception:
    rec = {}
rec.update({"mode": mode, "S": S, "rc": rc,
            "poller_peak_MiB": (float(peak) if peak not in ("", None) else None)})
if "oom" not in rec:
    rec["oom"] = (rc == 42)
if "solid" not in rec:
    rec["solid"] = (rc == 0)
print(json.dumps(rec))
PY
  echo "$rc"
}

echo "[driver] === $MODE binary search in [$LO, $HI] ===" >&2
: > "$JSONL"

# Confirm lo is solid; if lo OOMs, we still record it and narrow downward is
# impossible — report as-is.
rc=$(probe "$LO"); LO_RC=$rc
if [[ "$LO_RC" == "42" ]]; then
  echo "[driver] WARN: lo=$LO already OOM" >&2
fi
# Confirm hi (expected OOM or solid).
rc=$(probe "$HI"); HI_RC=$rc
if [[ "$HI_RC" == "0" ]]; then
  echo "[driver] NOTE: hi=$HI is SOLID (no OOM within range); wall is above $HI" >&2
fi

last_solid=""
first_oom=""
[[ "$LO_RC" == "0" ]] && last_solid=$LO
[[ "$HI_RC" == "42" ]] && first_oom=$HI
[[ "$HI_RC" == "0" ]] && last_solid=$HI

lo=$LO; hi=$HI
# Only search if we have a solid lo and an OOM hi to bracket.
if [[ "$LO_RC" == "0" && "$HI_RC" == "42" ]]; then
  while (( hi - lo > 512 )); do
    mid=$(( (lo + hi) / 2 ))
    # round mid to nearest 256 for tidy rungs
    mid=$(( (mid / 256) * 256 ))
    (( mid <= lo )) && mid=$(( lo + 256 ))
    (( mid >= hi )) && break
    rc=$(probe "$mid")
    if [[ "$rc" == "0" ]]; then
      last_solid=$mid; lo=$mid
    elif [[ "$rc" == "42" ]]; then
      first_oom=$mid; hi=$mid
    else
      echo "[driver] probe rc=$rc at S=$mid (non-OOM error) — stopping search" >&2
      break
    fi
  done
fi

echo "[driver] $MODE: last_solid=$last_solid first_oom=$first_oom" >&2
echo "LADDER_DONE mode=$MODE last_solid=$last_solid first_oom=$first_oom"
