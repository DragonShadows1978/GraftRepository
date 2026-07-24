#!/usr/bin/env bash
# GRM3P-LADDER-ON re-gate runner (unsandboxed lead seat).
#
# Registered gates:
#   G-ON-ESCAPE   escape-off smoke transcript sha == 68da84b8… (legacy)
#   G-ON-BASELINE default (no env) smoke — record NEW sha; three_pass
#                 transcript byte-equal to single
#   G-ON-FULL     default (no env) F-FULL unbounded 9/9; 4MB 9/9
#   G-ON-SUITE    both-pipeline unit suites green
#
# Usage:
#   cd /home/vader/GraftRepository-three-pass
#   bash scripts/grm3p_ladder_on_run.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export GRM_GQA_CUDA_ROUTE=1
export GRM_GRAFT_STORAGE_BITS=8
# Default-on: ensure env does not force escape for baseline/full gates.
unset GRM_PROBE_LADDER || true

PY="${PYTHON:-python3}"
E2E="$ROOT/scripts/grm_e2e_session.py"
ART="$ROOT/artifacts/grm_three_pass"
LOGDIR="$ROOT/logs"
mkdir -p "$LOGDIR" "$ART"

LEGACY_SHA="68da84b8824eb79a14f65a692a602215bc53d84232cd041002410f51657bba4f"

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

echo "===== 0) G-ON-SUITE (unit, both pipelines via suite files) ====="
run_locked suite \
  "$PY" -m pytest -q \
    tests/test_grm_*.py \
    tests/test_gqa_ragged_cuda_bank.py \
  | tee "$LOGDIR/grm3p_ladder_on_suite.log"
echo "G-ON-SUITE: PASS (pytest exit 0)"

echo "===== 1) G-ON-ESCAPE (escape-off smoke == legacy 68da84b8…) ====="
ESCAPE_DIR="$ART/ladder_on_escape_smoke_single"
rm -rf "$ESCAPE_DIR"
run_locked escape_smoke \
  env GRM_PROBE_LADDER=0 \
  "$PY" "$E2E" \
    --mode smoke \
    --turn-pipeline single \
    --no-probe-ladder \
    --session-dir "$ESCAPE_DIR" \
    --skip-gpu-idle-check \
  | tee "$LOGDIR/grm3p_ladder_on_escape_smoke.log"

ESCAPE_SHA="$(sha_transcript "$ESCAPE_DIR/transcript.jsonl")"
echo "G-ON-ESCAPE transcript sha: $ESCAPE_SHA"
echo "registered legacy sha:      $LEGACY_SHA"
if [[ "$ESCAPE_SHA" != "$LEGACY_SHA" ]]; then
  echo "RED: G-ON-ESCAPE smoke transcript SHA mismatch (legacy path not byte-identical)" >&2
  exit 2
fi
echo "G-ON-ESCAPE: PASS"

echo "===== 2) G-ON-BASELINE (default smoke; single vs three_pass byte-equal) ====="
BASE_SINGLE="$ART/ladder_on_default_smoke_single"
BASE_TP="$ART/ladder_on_default_smoke_three_pass"
rm -rf "$BASE_SINGLE" "$BASE_TP"
run_locked baseline_single \
  "$PY" "$E2E" \
    --mode smoke \
    --turn-pipeline single \
    --session-dir "$BASE_SINGLE" \
    --skip-gpu-idle-check \
  | tee "$LOGDIR/grm3p_ladder_on_baseline_single.log"
run_locked baseline_three_pass \
  "$PY" "$E2E" \
    --mode smoke \
    --turn-pipeline three_pass \
    --session-dir "$BASE_TP" \
    --skip-gpu-idle-check \
  | tee "$LOGDIR/grm3p_ladder_on_baseline_three_pass.log"

BASE_SHA="$(sha_transcript "$BASE_SINGLE/transcript.jsonl")"
TP_SHA="$(sha_transcript "$BASE_TP/transcript.jsonl")"
echo "G-ON-BASELINE single sha:     $BASE_SHA"
echo "G-ON-BASELINE three_pass sha: $TP_SHA"
echo "REGISTERED_NEW_BASELINE_SHA=$BASE_SHA"
if [[ "$BASE_SHA" != "$TP_SHA" ]]; then
  echo "RED: G-ON-BASELINE three_pass transcript != single at new default" >&2
  exit 3
fi
# Scorecards must also be 2/2 under default-on.
for sdir in "$BASE_SINGLE" "$BASE_TP"; do
  ok="$("$PY" -c "import json; d=json.load(open('${sdir}/probe_scorecard.json')); print(int(d.get('passed')==2 and d.get('all_passed')))")"
  if [[ "$ok" != "1" ]]; then
    echo "RED: G-ON-BASELINE scorecard not 2/2 at $sdir" >&2
    scorecard_line "$sdir"
    exit 3
  fi
done
echo "G-ON-BASELINE: PASS (new registered baseline sha above)"

echo "===== 3) G-ON-FULL (default no-env F-FULL unbounded 9/9 + 4MB 9/9) ====="
FULL_DIR="$ART/ladder_on_full_default_single"
rm -rf "$FULL_DIR"
run_locked full_default \
  "$PY" "$E2E" \
    --mode full \
    --turn-pipeline single \
    --session-dir "$FULL_DIR" \
    --skip-gpu-idle-check \
  | tee "$LOGDIR/grm3p_ladder_on_full_default.log"
echo "G-ON-FULL unbounded scorecard:"
scorecard_line "$FULL_DIR"
FULL_PASS="$("$PY" -c "import json; d=json.load(open('$FULL_DIR/probe_scorecard.json')); print(int(bool(d.get('all_passed'))))")"
if [[ "$FULL_PASS" != "1" ]]; then
  echo "RED: G-ON-FULL unbounded not 9/9" >&2
  exit 4
fi

COLD_DIR="$ART/ladder_on_cold4mb_default_single"
rm -rf "$COLD_DIR"
run_locked cold4mb_default \
  "$PY" "$E2E" \
    --mode full \
    --turn-pipeline single \
    --vram-budget-mb 4 \
    --session-dir "$COLD_DIR" \
    --skip-gpu-idle-check \
  | tee "$LOGDIR/grm3p_ladder_on_cold4mb_default.log"
echo "G-ON-FULL 4MB scorecard:"
scorecard_line "$COLD_DIR"
COLD_PASS="$("$PY" -c "import json; d=json.load(open('$COLD_DIR/probe_scorecard.json')); print(int(bool(d.get('all_passed'))))")"
if [[ "$COLD_PASS" != "1" ]]; then
  echo "RED: G-ON-FULL 4MB not 9/9" >&2
  exit 5
fi
echo "G-ON-FULL: PASS"

echo "===== DONE — all registered G-ON gates green ====="
echo "sessions:"
echo "  escape_smoke=$ESCAPE_DIR"
echo "  baseline_single=$BASE_SINGLE"
echo "  baseline_three_pass=$BASE_TP"
echo "  full_default=$FULL_DIR"
echo "  cold4mb_default=$COLD_DIR"
echo "REGISTERED_NEW_BASELINE_SHA=$BASE_SHA"
echo "LEGACY_ESCAPE_SHA=$ESCAPE_SHA"
