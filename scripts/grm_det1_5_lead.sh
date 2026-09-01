#!/usr/bin/env bash
# GRM-DET1.7 lived-admission registration campaign handoff.
#
# The Python parent first freezes the amended source authorization, captures
# served-only lived admission observations, and freezes the content-addressed
# plant registry before any detector or evaluation arm can run.  It also owns
# every GPU lease, the 30-second inter-process gap, ordered stage validation,
# and append-only receipts.  This shell is the one lead-seat entry point; do
# not invoke the hidden worker command directly.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
RUN_DIR="$ROOT/artifacts/grm_det1/run_20260831T160525Z_2"
DRIVER="$ROOT/scripts/grm_det1_5_gpu.py"

export PYTHONPATH="$ROOT:/mnt/ForgeRealm/Project-Tensor/tensor_cuda${PYTHONPATH:+:$PYTHONPATH}"

"$PY" "$DRIVER" selftest --run-dir "$RUN_DIR"
"$PY" "$DRIVER" author-det1-7-source --run-dir "$RUN_DIR"
# GRM-DET1.9: the campaign-r4 t30/t33 lived-control finding is a hard
# prerequisite of plant-registration, so it must be recorded before the
# registry can freeze.  Authoring is guarded (author-if-absent, else
# revalidate) because the receipt payload carries created_utc.
"$PY" "$DRIVER" author-det1-9-finding --run-dir "$RUN_DIR"
"$PY" "$DRIVER" inventory --run-dir "$RUN_DIR"
"$PY" "$DRIVER" det1-7-cpu-preflight --run-dir "$RUN_DIR"
"$PY" "$DRIVER" gpu-preflight --run-dir "$RUN_DIR" --write-receipt
"$PY" "$DRIVER" campaign \
  --run-dir "$RUN_DIR" \
  --lease-seconds 580 \
  --lock-wait-seconds 7200 \
  --gap-seconds 30
