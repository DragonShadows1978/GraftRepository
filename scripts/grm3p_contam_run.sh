#!/usr/bin/env bash
# GRM3P-DIAG-CONTAM runner: default-smoke byte-identity, then 8/16/unbounded
# legs with paging telemetry + mount snapshots enabled.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export GRM_GQA_CUDA_ROUTE=1
export GRM_GRAFT_STORAGE_BITS=8

PY="${PYTHON:-python3}"
E2E="$ROOT/scripts/grm_e2e_session.py"
ART="$ROOT/artifacts/grm_three_pass"
LOGDIR="$ROOT/logs"
mkdir -p "$LOGDIR" "$ART"

REG_SHA="68da84b8824eb79a14f65a692a602215bc53d84232cd041002410f51657bba4f"

run_locked() {
  local name="$1"; shift
  echo "=== START $name $(date -Is) ==="
  flock -w 7200 /tmp/forge-gpu.lock "$@"
  local st=$?
  echo "=== END $name status=$st $(date -Is) ==="
  return $st
}

sha_transcript() {
  local f="$1"
  sha256sum "$f" | awk '{print $1}'
}

echo "===== 1) DEFAULT SMOKE (telemetry OFF) ====="
# Ensure env paths are unset for byte-identity.
unset GRM_PAGING_TELEMETRY_PATH GRM_MOUNT_SNAPSHOT_DIR || true
SMOKE_DIR="$ART/contam_default_smoke_postwire_single"
rm -rf "$SMOKE_DIR"
run_locked default_smoke \
  "$PY" "$E2E" \
    --mode smoke \
    --turn-pipeline single \
    --session-dir "$SMOKE_DIR" \
    --skip-gpu-idle-check \
  | tee "$LOGDIR/grm3p_contam_default_smoke.log"

SMOKE_SHA="$(sha_transcript "$SMOKE_DIR/transcript.jsonl")"
echo "default-smoke transcript sha: $SMOKE_SHA"
echo "registered sha:               $REG_SHA"
if [[ "$SMOKE_SHA" != "$REG_SHA" ]]; then
  echo "RED: default-smoke transcript SHA mismatch after wiring" >&2
  exit 2
fi
echo "default-smoke byte-identity: PASS"

run_leg() {
  local tag="$1"
  local budget_arg=()
  if [[ "$2" != "none" ]]; then
    budget_arg=(--vram-budget-mb "$2")
  fi
  local sdir="$ART/contam_${tag}_single"
  rm -rf "$sdir"
  export GRM_PAGING_TELEMETRY_PATH="{session}/paging_events.jsonl"
  export GRM_MOUNT_SNAPSHOT_DIR="{session}/mount_snapshots"
  run_locked "$tag" \
    "$PY" "$E2E" \
      --mode full \
      --turn-pipeline single \
      --session-dir "$sdir" \
      --skip-gpu-idle-check \
      "${budget_arg[@]}" \
    | tee "$LOGDIR/grm3p_contam_${tag}.log"
  echo "leg $tag scorecard: $(python3 -c "import json; d=json.load(open('$sdir/probe_scorecard.json')); print(f\"{d.get('passed')}/{d.get('total', len(d.get('probes',[])))} all_passed={d.get('all_passed')}\")")"
}

echo "===== 2) TELEMETRY LEGS ====="
run_leg "08mb" 8
run_leg "16mb" 16
run_leg "unbounded" none

echo "===== 3) ANALYSIS ====="
"$PY" "$ROOT/scripts/grm3p_contam_analyze.py" \
  --legs \
    "$ART/contam_08mb_single" \
    "$ART/contam_16mb_single" \
    "$ART/contam_unbounded_single" \
  --out "$ART/contam_analysis.json" \
  | tee "$LOGDIR/grm3p_contam_analysis.log"

echo "DONE contam runner $(date -Is)"
