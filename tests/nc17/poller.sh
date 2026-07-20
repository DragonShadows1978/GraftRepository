#!/usr/bin/env bash
# 1s nvidia-smi memory poller. Writes "epoch_ms used_MiB" lines to $1.
# Started with setsid inside a flocked shell; killed by the driver when the
# GPU run completes. A0 law: poller peaks are authoritative over framework
# counters.
OUT="${1:?poller output path required}"
: > "$OUT"
while true; do
  ts=$(date +%s%3N)
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  echo "$ts $used" >> "$OUT"
  sleep 1
done
