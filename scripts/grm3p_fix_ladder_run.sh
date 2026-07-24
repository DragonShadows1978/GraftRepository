#!/usr/bin/env bash
# GRM3P-FIX-LADDER runner (unsandboxed). Mirrors DIAG-CONTAM handoff pattern.
#
# Gates:
#   G-FIX-DEFAULT  flag OFF smoke transcript sha == registered
#   G-FIX-FULL     flag ON  F-FULL single 9/9
#   G-FIX-COLD     flag ON  4MB full 9/9 + smoke 2/2
#   G-FIX-SUITE    pytest (single env; three_pass covered by unit tests)
#
# Usage:
#   cd /home/vader/GraftRepository-three-pass
#   bash scripts/grm3p_fix_ladder_run.sh
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
  sha256sum "$1" | awk '{print $1}'
}

scorecard_line() {
  local sdir="$1"
  "$PY" -c "
import json
d=json.load(open('${sdir}/probe_scorecard.json'))
probes=d.get('probes',[])
print(f\"{d.get('passed')}/{d.get('total', len(probes))} all_passed={d.get('all_passed')}\")
for p in probes:
    print(f\"  turn {p.get('turn')}: {'PASS' if p.get('pass') else 'FAIL'} "
          f\"{p.get('fact_id')} expected={p.get('expected')!r} answer={p.get('answer')!r} "
          f\"plan={p.get('mount_plan')} fitted={p.get('mounted_ids') or p.get('mount_fitted')}\")
"
}

echo "===== 0) G-FIX-SUITE (unit) ====="
run_locked suite \
  "$PY" -m pytest -q \
    tests/test_grm_*.py \
    tests/test_gqa_ragged_cuda_bank.py \
  | tee "$LOGDIR/grm3p_fix_ladder_suite.log"

echo "===== 1) G-FIX-DEFAULT (flag OFF smoke) ====="
unset GRM_PROBE_LADDER || true
SMOKE_DIR="$ART/fix_ladder_default_smoke_single"
rm -rf "$SMOKE_DIR"
run_locked default_smoke \
  "$PY" "$E2E" \
    --mode smoke \
    --turn-pipeline single \
    --session-dir "$SMOKE_DIR" \
    --skip-gpu-idle-check \
  | tee "$LOGDIR/grm3p_fix_ladder_default_smoke.log"

SMOKE_SHA="$(sha_transcript "$SMOKE_DIR/transcript.jsonl")"
echo "G-FIX-DEFAULT transcript sha: $SMOKE_SHA"
echo "registered sha:               $REG_SHA"
if [[ "$SMOKE_SHA" != "$REG_SHA" ]]; then
  echo "RED: G-FIX-DEFAULT smoke transcript SHA mismatch" >&2
  exit 2
fi
echo "G-FIX-DEFAULT: PASS"

echo "===== 2) G-FIX-FULL (flag ON F-FULL single unbounded) ====="
FULL_DIR="$ART/fix_ladder_full_on_single"
rm -rf "$FULL_DIR"
run_locked full_on \
  env GRM_PROBE_LADDER=1 \
  "$PY" "$E2E" \
    --mode full \
    --turn-pipeline single \
    --probe-ladder \
    --session-dir "$FULL_DIR" \
    --skip-gpu-idle-check \
  | tee "$LOGDIR/grm3p_fix_ladder_full_on.log"
echo "G-FIX-FULL scorecard:"
scorecard_line "$FULL_DIR"
FULL_PASS="$("$PY" -c "import json; d=json.load(open('$FULL_DIR/probe_scorecard.json')); print(int(bool(d.get('all_passed'))))")"
if [[ "$FULL_PASS" != "1" ]]; then
  echo "RED: G-FIX-FULL not 9/9" >&2
  exit 3
fi
echo "G-FIX-FULL: PASS"

echo "===== 3) G-FIX-COLD (flag ON 4MB full + smoke) ====="
COLD_DIR="$ART/fix_ladder_cold4mb_on_single"
rm -rf "$COLD_DIR"
run_locked cold4mb \
  env GRM_PROBE_LADDER=1 \
  "$PY" "$E2E" \
    --mode full \
    --turn-pipeline single \
    --probe-ladder \
    --vram-budget-mb 4 \
    --session-dir "$COLD_DIR" \
    --skip-gpu-idle-check \
  | tee "$LOGDIR/grm3p_fix_ladder_cold4mb.log"
echo "G-FIX-COLD full scorecard:"
scorecard_line "$COLD_DIR"
COLD_PASS="$("$PY" -c "import json; d=json.load(open('$COLD_DIR/probe_scorecard.json')); print(int(bool(d.get('all_passed'))))")"
if [[ "$COLD_PASS" != "1" ]]; then
  echo "RED: G-FIX-COLD full 4MB not 9/9" >&2
  exit 4
fi

SMOKE_ON_DIR="$ART/fix_ladder_smoke_on_single"
rm -rf "$SMOKE_ON_DIR"
run_locked smoke_on \
  env GRM_PROBE_LADDER=1 \
  "$PY" "$E2E" \
    --mode smoke \
    --turn-pipeline single \
    --probe-ladder \
    --session-dir "$SMOKE_ON_DIR" \
    --skip-gpu-idle-check \
  | tee "$LOGDIR/grm3p_fix_ladder_smoke_on.log"
echo "G-FIX-COLD smoke scorecard:"
scorecard_line "$SMOKE_ON_DIR"
SMOKE_ON_PASS="$("$PY" -c "import json; d=json.load(open('$SMOKE_ON_DIR/probe_scorecard.json')); print(int(d.get('passed')==2 and d.get('all_passed')))")"
if [[ "$SMOKE_ON_PASS" != "1" ]]; then
  echo "RED: G-FIX-COLD smoke not 2/2" >&2
  exit 5
fi
echo "G-FIX-COLD: PASS"

echo "===== DONE — all registered gates green ====="
echo "sessions:"
echo "  default_smoke=$SMOKE_DIR"
echo "  full_on=$FULL_DIR"
echo "  cold4mb=$COLD_DIR"
echo "  smoke_on=$SMOKE_ON_DIR"
