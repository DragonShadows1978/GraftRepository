#!/usr/bin/env bash
# LSR Phase 0 — lead GPU battery: full-window witness for all 8 probes.
#
# INVOCATION (bare; the script cds to the repo root itself):
#     bash scripts/lsr_p0_lead_gpu.sh
#
# PYTHONPATH: set below to repo-root + tensor_cuda. The Python drivers are
# plain `python3 scripts/...` invocations with no wrapper, so a lead can copy
# any single line out of this file and run it standalone.
#
# GPU DISCIPLINE (standing house rules):
#   * every probe runs in its OWN process under its OWN self-lease
#     (scripts.grm_cmc1_gpu_arms.gpu_lease on /tmp/forge-gpu.lock);
#   * lease cap 580 s, under the 590 s hard cap; a probe that cannot finish
#     inside 580 s fails loudly rather than overrunning;
#   * a 30 s GAP separates consecutive leases so the operator (absolute right
#     of way) and any parallel seat can take the lock between probes;
#   * lock wait 7200 s: this battery yields to whatever already holds the GPU.
#
# The CPU legs (selftest, enumeration, survey) run FIRST and take no lease at
# all: if the instrument is broken, no GPU time is spent discovering that.
#
# APPEND-ONLY: every receipt is content-addressed under artifacts/lsr_p0/.
# Re-running never duplicates or overwrites; identical payloads collapse onto
# the same filename and differing ones raise.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
export PYTHONPATH="$ROOT:/mnt/ForgeRealm/Project-Tensor/tensor_cuda${PYTHONPATH:+:$PYTHONPATH}"

LEASE_SECONDS="${LSR_LEASE_SECONDS:-580}"
LOCK_WAIT_SECONDS="${LSR_LOCK_WAIT_SECONDS:-7200}"
GAP_SECONDS="${LSR_GAP_SECONDS:-30}"

echo "== LSR Phase 0 =="
echo "root=$ROOT lease=${LEASE_SECONDS}s gap=${GAP_SECONDS}s wait=${LOCK_WAIT_SECONDS}s"
echo

# ---------------------------------------------------------------- CPU legs
# No GPU, no lease. These must pass before any lease is taken.
echo "-- CPU selftest (no lease) --"
"$PY" scripts/lsr_p0_core.py --selftest --emit

echo
echo "-- tap selftest against a fake engine (no lease, no weights) --"
# Covers the model/layer attribute paths, layer attribution under the nested
# chunked sliding path, and the engine-supplied scale. This is the gate that
# would have caught the .cfg/.config shakedown failure without a GPU lease.
"$PY" scripts/lsr_p0_gpu.py selftest

echo
echo "-- probe enumeration (no lease) --"
"$PY" scripts/lsr_p0_core.py --enumerate --emit

echo
echo "-- frozen-snapshot survey (no lease) --"
"$PY" scripts/lsr_p0_survey.py --emit

# ---------------------------------------------------------------- GPU legs
# One lease per probe, 30 s gap between them. Order is the enumeration order:
# the six wrong-value probes, then the two lawful same-session controls.
PROBES=(
  sup_lumen_head
  sup_orion_current
  sup_reserve_falcon_registry
  sup_reserve_juniper_pass
  sup_reserve_meridian_docket
  sup_reserve_tundra_ledger
  sup_harbor_restatement
  sup_praxis_fresh
)

echo
echo "-- GPU witness battery: ${#PROBES[@]} probes, one lease each --"
first=1
for probe in "${PROBES[@]}"; do
  if [ "$first" -eq 0 ]; then
    echo "   (inter-lease gap ${GAP_SECONDS}s)"
    sleep "$GAP_SECONDS"
  fi
  first=0
  echo
  echo "== witness $probe =="
  "$PY" scripts/lsr_p0_gpu.py witness \
    --probe "$probe" \
    --lease-seconds "$LEASE_SECONDS" \
    --lock-wait-seconds "$LOCK_WAIT_SECONDS"
done

# ------------------------------------------------------------ adjudication
# Applies the plan's registered rule over whatever witnesses exist. If any
# probe is missing the status is NOT_MEASURED, never a silent NOT_CONVICTED.
echo
echo "-- registered adjudication (no lease) --"
"$PY" scripts/lsr_p0_gpu.py adjudicate

echo
echo "== LSR Phase 0 complete; receipts under artifacts/lsr_p0/ =="
