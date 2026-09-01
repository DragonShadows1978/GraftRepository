#!/usr/bin/env bash
# LSR Phase 1 — lead GPU battery: the ONE decision input Phase 1 could not
# read off disk.
#
# INVOCATION (bare; the script cds to the repo root itself):
#     bash scripts/lsr_p1_lead_gpu.sh
#
# WHY THIS EXISTS
# ---------------
# Phase 1 is otherwise complete from persisted receipts. Exactly one input
# is missing at sufficient granularity, and it is load-bearing for the two
# ROUTE-RANK verdicts (sup_lumen_head, sup_orion_current):
#
#   The lived DET1.5 snapshots persist admission.ranking (the RAW output of
#   ArenaCache.route()), but NOT the router's own per-call receipt --
#   specifically `candidate_count` and the per-candidate score table. For
#   sup_lumen_head the lived ranking is [1, 0] over a 4-node fixture, and
#   for sup_orion_current it is [0] over a 3-node fixture. Both are SHORTER
#   than the eligible universe, and core/grm_admission.py bounds the ask at
#   want = min(len(eligible), max(3, route_limit)) with a lived route_limit
#   of 6 -- so the shortfall cannot be a window bound.
#
#   The DET1.5 EVAL stage does persist that receipt
#   (fork_restore.routing_index.candidate_count) and it reads 4 for the
#   sup_harbor_restatement control whose raw_ranking_ids is likewise [1, 0].
#   That single row already shows route() returning 2 ids from 4 candidates.
#   But the six FAILING probes never reached the eval stage -- they were
#   ruled UNPLANTABLE at plant registration -- so no such row exists for
#   them, and the SCORE TABLE that would name WHICH candidates were dropped
#   and why (the non-finite-score filter in ArenaCache.route(), the "M6
#   law") was never persisted for any of them.
#
# WHAT THIS MEASURES
# ------------------
# For each of the two ROUTE-RANK probes plus the harbor control, on the
# frozen DET1.5 runtime frame:
#   * len(_route_cand_base()) and the per-call exclude set;
#   * the full per-candidate score table (_vector_route_scores /
#     _cent_score, then _length_debias_scores, _normalize_scores, and the
#     _lex_bonus channel), with an explicit finite/non-finite flag per
#     candidate;
#   * the resulting ranking at limit=None (full rank) and at the lived
#     production route_limit of 6.
#
# This turns the ROUTE-RANK verdicts from "the node was absent from the
# ranking" into "the node was dropped by <named mechanism> at <named
# score>". It does NOT re-serve and does NOT change any verdict on its own:
# the stage attribution already stands on the persisted admission.ranking.
#
# GPU DISCIPLINE (standing house rules, identical to the Phase 0 battery):
#   * every probe runs in its OWN process under its OWN self-lease
#     (scripts.grm_cmc1_gpu_arms.gpu_lease on /tmp/forge-gpu.lock);
#   * lease cap 580 s, under the 590 s hard cap;
#   * a 30 s GAP separates consecutive leases so the operator (absolute
#     right of way) and any parallel seat can take the lock between probes;
#   * lock wait 7200 s: this battery yields to whatever already holds it.
#
# NOTE FOR THE LEAD: the Python driver referenced below
# (scripts/lsr_p1_route_probe.py) is NOT written by this seat -- writing it
# would mean authoring a GPU instrument this seat cannot run or verify.
# This file states the required measurement, the flags, and the discipline
# so the lead can commission it deliberately. Everything above the GPU
# section is already done and needs no lease.
#
# APPEND-ONLY: receipts belong under artifacts/lsr_p1/, content-addressed,
# with their own provenance (NOT a DET envelope).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
export PYTHONPATH="$ROOT:/mnt/ForgeRealm/Project-Tensor/tensor_cuda${PYTHONPATH:+:$PYTHONPATH}"

LEASE_SECONDS="${LSR_LEASE_SECONDS:-580}"
LOCK_WAIT_SECONDS="${LSR_LOCK_WAIT_SECONDS:-7200}"
GAP_SECONDS="${LSR_GAP_SECONDS:-30}"

echo "== LSR Phase 1 =="
echo "root=$ROOT lease=${LEASE_SECONDS}s gap=${GAP_SECONDS}s wait=${LOCK_WAIT_SECONDS}s"
echo

# ---------------------------------------------------------------- CPU legs
# No GPU, no lease. These are the complete Phase 1 archaeology and they
# already produce the six stage verdicts. Run them first: if the instrument
# is broken, no GPU time is spent discovering that.
echo "-- harvest lived DET1.5 admission/route state (no lease) --"
"$PY" scripts/lsr_p1_harvest.py

echo
echo "-- model-free replay of identifier / A-DEC / L2 stages (no lease) --"
"$PY" scripts/lsr_p1_stage_replay.py

echo
echo "-- per-probe stage adjudication (no lease) --"
"$PY" scripts/lsr_p1_adjudicate.py

# ---------------------------------------------------------------- GPU legs
# One lease per probe, 30 s gap between them. Only the probes whose verdict
# rests on an un-persisted router input are measured; the four
# ADMISSION-PRUNE verdicts need no GPU (their planned/fitted/dropped fields
# are fully persisted).
PROBES=(
  sup_lumen_head
  sup_orion_current
  sup_harbor_restatement
)

echo
echo "-- GPU route-score battery: ${#PROBES[@]} probes, one lease each --"
echo "   REQUIRES: scripts/lsr_p1_route_probe.py (to be commissioned)"
if [ ! -f scripts/lsr_p1_route_probe.py ]; then
  echo
  echo "   scripts/lsr_p1_route_probe.py is absent."
  echo "   The CPU legs above are complete and the six stage verdicts stand."
  echo "   Skipping the GPU battery rather than inventing a measurement."
  echo
  echo "== LSR Phase 1 CPU legs complete; receipts under artifacts/lsr_p1/ =="
  exit 0
fi

first=1
for probe in "${PROBES[@]}"; do
  if [ "$first" -eq 0 ]; then
    echo "   (inter-lease gap ${GAP_SECONDS}s)"
    sleep "$GAP_SECONDS"
  fi
  first=0
  echo
  echo "== route-score table $probe =="
  "$PY" scripts/lsr_p1_route_probe.py score-table \
    --probe "$probe" \
    --lease-seconds "$LEASE_SECONDS" \
    --lock-wait-seconds "$LOCK_WAIT_SECONDS"
done

echo
echo "== LSR Phase 1 complete; receipts under artifacts/lsr_p1/ =="
