# GRM-SC1.2 — Demand-loop recovery on the 7 E2E race pairs (session resume)

Lead-authored 2026-09-02 from the SC1/SC1.1 shortfall: only 3 of the
race's 10 registered pairs were reachable by the standalone harness.
The other 7 are `certified_34_turn` E2E fixtures whose lived rows were
produced inside the chained session by the campaign worker, whose
`_context` refuses to run outside the frozen run tree. This order
measures recovery on those 7 WITHOUT writing into the frozen tree.
The flip decision needs recovery at 10, not 3.

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/GraftRepository` (branch
`lc1-wip`, HEAD b527ba3) — edits, builds, test runs, and BOUNDED GPU
runs AUTHORIZED. A registered order IS the permission: run first,
report after.

## Context (read fully, in order)

1. `orders/GRM_SC1_DEMAND_LOOP_NGH.md`, `orders/GRM_SC1_1_GROUNDING_GLYPHS.md`
   and their receipts (`artifacts/grm_sc1/`, `artifacts/grm_sc1_1/`):
   the 3-pair G3 result, `scripts/grm_sc1_recovery_gpu.py` (`plan` /
   `session --session-id` / `summarize`), its RED-1 note on the 7 pairs.
2. The race substrate: plant registry (`scripts/grm_det1_7_registry.py`,
   `artifacts/grm_det1/run_20260831T160525Z_2/det1_4/campaign/det1_7/plant_registration/`),
   the 10 registered pairs and their `planted_miss` / `served` rows
   (`.../campaign/eval/mechanistic_rows.jsonl`), the runtime frame,
   and `scripts/grm_det1_5_workers.py::_context` (~202: the FROZEN_RUN
   guard), `_run_e2e`/session handling (~468–580: `run_config.json`,
   `det1_5_resume_*.json`, session_dir lineage checks).
3. `scripts/grm_det1_5_gpu.py::_measure_fixture_inline`,
   `_restore_counterfactual`, `restore_prefill_fork` (DET1.3) — the fork
   mechanism the lived E2E rows used (P2C's G3 blocked-report names
   the fork-ladder-vs-in-session mismatch; here we USE the fork ladder,
   so it is the right instrument).
4. `scripts/lsr_p2c_e2e_gpu.py` (sharded 34-turn replay with
   save/restore + `--resume`; 4 shards at 59–93 s) — reuse its
   sharding and state carry.

## Mission

1. **Session resume, read-only on the frozen tree.** Reconstruct the
   certified session's state at each of the 7 E2E probe turns by the
   SAME mechanism that produced the lived rows: either (a) load the
   race's saved session snapshots/resume records from the frozen tree
   (read-only; copy what you need under `artifacts/grm_sc1_2/session/`)
   and fork at the probe turn, or (b) if no per-turn snapshot exists
   for a probe turn, replay the certified session's deposit sequence
   turn by turn with `lsr_p2c_e2e_gpu.py`'s sharded carry to that turn,
   then fork. State per probe which path was used. The campaign
   worker's `_context` is not to be bypassed by pointing it at a copy:
   write your own thin driver that imports the measurement functions.
2. **Arm 0 reproduction first.** For each of the 7 pairs, the served
   control AND the planted miss must reproduce the lived row's served
   text (semantic comparator) and D-NGH signal (min mass within float
   equality if the fork is byte-exact; otherwise report the delta and
   the reason) with demand OFF. A pair whose Arm 0 does not reproduce
   is reported NOT-REPRODUCED and excluded from Arm 1 — never patched
   into reproduction.
3. **Arm 1 recovery.** Demand ON, fixes ON (SC1.1 grounding), on every
   reproduced pair: detection, fetch, served-from, value, recovered.
4. Roll the 3 standalone pairs (SC1.1 G2) and the 7 E2E pairs into one
   10-pair table with the race's registered pair ids.

## Gates (synchronous, FOREGROUND, in-call waits only; RED is a result)

Register before any gate run (`artifacts/grm_sc1_2/registration.json`):
predictions below; no numeric constants.

G1. `python3 -m pytest -q` sharded under 10 min per call — pre-existing
    set byte-identical to SC1.1's logs; new CPU tests for the driver's
    pure parts (pair selection, row matching, table assembly).
G2. **Arm 0 reproduction** (GPU, leased, bounded, sharded): prediction
    ≥ 6/7 pairs reproduce (both rows); any non-reproducing pair named
    with the mechanism.
G3. **Arm 1 recovery**: prediction (lead) detection recall ≥ 0.9 over
    the reproduced planted misses; false fires on served controls ≤ 1;
    recovery ≥ 5 of the reproduced planted misses. PASS RULE for the
    mechanism = recall ≥ 0.9 AND FP ≤ 1 (same as SC1); recovery is
    REPORTED. Final 10-pair table with recall / FP / recovery over the
    achieved count (floor: report the count; do not fabricate).
G4. Lived battery unchanged (`scripts/grm_sc1_lived_battery_gpu.py`,
    demand ON): 9/9 identical to SC1.1 G3 — only if any production
    file changed; if the order lands with scripts/tests only, say so
    and skip with that reason.

GPU conventions: `/tmp/forge-gpu.lock` lease via the repo's
`gpu_lease`, single GPU, every run ≤ 10 min wall, ≥30 s gaps (inside
the wrapper before lease, as SC1.1 did), operator has absolute right
of way. Fork-from-snapshot only; no rebuild-from-fixture-install.

## File boundary

Modify ONLY: `scripts/grm_sc1_2_*.py` (new), `scripts/grm_sc1_recovery_gpu.py`
(extend `summarize` to merge the 10 pairs), `tests/`,
`artifacts/grm_sc1_2/`, `logs/`. Production code (`core/`) is READ-ONLY
in this order — if a production change turns out to be required,
STOP and report it as a blocker with the receipt. Read-only: every
`scripts/grm_det1_*.py`, `scripts/lsr_*.py` (import only; the frozen
run tree is never written), `docs/`, `orders/`, all receipts.

## Principles (binding)

Thresholds carried, never refit. Determinism. NO git (lead commits).
NO subagents. **NO background processes for waiting**: no
`run_in_background`, no watcher/poll loops, no "block until X exits"
helpers; foreground commands only, wait in-call, every Bash call under
10 minutes. RED honesty: NOT-REPRODUCED is a result. Verify your own
claims against artifacts before writing them.

## Done (verbatim in the final message)

1. pytest summary line(s) + pre-existing set unchanged.
2. Registration block as written before gates.
3. G2 table (7 pairs × path used / served-control reproduced /
   planted-miss reproduced / mass delta); G3 10-pair table (pair ×
   fired / token index / fetched / served-from / value / recovered)
   with recall, FP, recovery over the achieved count; G4 status.
4. Files created/modified.
5. Ambiguities, deviations, residual risks, anything RED — and the
   flip numbers at 10 pairs.
