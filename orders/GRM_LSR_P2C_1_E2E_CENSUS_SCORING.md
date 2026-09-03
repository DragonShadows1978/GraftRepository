# GRM-LSR-P2C.1 — Score P2C on the 10 e2e census probes via session resume

Lead-authored 2026-09-02 evening. P2C (2001ada) closed the wrong-value
class on the supersession battery (4 flips, 0 regressions) but its G3
— the 10 `e2e_t*` census probes inside the certified 34-turn session —
was BLOCKED: the deposit-order session replay reproduced only 8/10
(fork-ladder vs in-session mechanism). SC1.2 (c585100) then built the
right instrument: fork the frozen per-probe lived snapshot and rebuild
the routing index to the state at that turn (truncate to [0,N),
recompute supersession closure; proven against all 4 frozen manifests
with 0 mismatches; Arm 0 reproduced 7/7 with mass delta exactly 0.0).
This order reuses that instrument to score P2A+P2C+SC1.1 (fixes ON) on
all 10 census probes. Measurement only. No production change. No
decision needed.

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/GraftRepository` (branch
`lc1-wip`, HEAD 3a7f7cd) — scripts, tests, artifacts, and BOUNDED GPU
runs AUTHORIZED. A registered order IS the permission: run first,
report after.

## Context (read fully, in order)

1. `orders/GRM_LSR_P2C_UNSEATABLE_SPLIT_DESCENT.md` and its G3
   blocked-report (`artifacts/lsr_p2c/lsr_p2c_g3_blocked_report_*.json`,
   Arm-0 receipt `lsr_p2c_g3_arm0_*.json`: 8/10 reproduced; t30 and
   t33 diverged).
2. `orders/GRM_SC1_2_E2E_PAIRS_SESSION_RESUME.md`,
   `scripts/grm_sc1_2_session.py` (the resume + index rebuild),
   `scripts/grm_sc1_2_e2e_recovery_gpu.py`, and the SC1.2 receipts
   (`artifacts/grm_sc1_2/`): the 7 probe turns it reached (t05, t13,
   t19, t22, t24, t26, t30) and how N was derived (`probe_node_counts`,
   backward walk over the frozen `instrumentation.jsonl`).
3. The census: `artifacts/grm_det1/run_20260831T160525Z_2/det1_11/census/`
   — the 10 e2e probes, their lived served values, and verdicts
   (including `e2e_t33_polaris_mark` = the lived REFUSAL and
   `e2e_t30_atlas_tone` = separator-rescued correct).
4. The fixes switch `GRM_LSR_FIXES` (P2C; SC1.1 grounding rides it) and
   `GRM_DEMAND_NGH` (keep OFF here — this scores serving, not the
   demand loop).

## Mission

1. Extend the SC1.2 session-resume path to all 10 census probe turns
   (the 7 it already reaches + t09, t16, t33 or whichever three are
   missing — derive N for each by the same registered backward-walk
   rule; if a probe turn has no frozen snapshot, say so and mark it
   UNREACHABLE rather than rebuilding).
2. **Arm 0 (fixes OFF, demand OFF):** reproduce the lived served text
   (DET1 semantic comparator) on every reachable probe. A probe that
   does not reproduce is NOT-REPRODUCED, excluded from Arm 1, never
   patched.
3. **Arm 1 (fixes ON: P2A+P2C+SC1.1; demand OFF):** serve each
   reproduced probe; report served value, correct?, and the fit/
   grounding receipts (`fit_*`, `grounding_*`).
4. Table: probe × (lived value, lived verdict, Arm 0 reproduced, Arm 1
   value, Arm 1 correct, change).

## Gates (synchronous, FOREGROUND, in-call waits only; RED is a result)

Register before any gate run (`artifacts/lsr_p2c_1/registration.json`):
the N-derivation rule (carried from SC1.2), reachability list, and the
predictions below. No numeric constants.

G1. `python3 -m pytest -q` sharded under 10 min per call — pre-existing
    set unchanged vs SC2's logs; new CPU tests for the driver's pure
    parts (probe selection, N derivation for the 3 new turns pinned
    against the frozen instrumentation, table assembly).
G2. **Arm 0 reproduction** (GPU, leased, bounded): prediction ≥ 9/10
    reachable and reproduced (SC1.2 got 7/7 on its 7; the lived t33
    refusal must reproduce as a refusal).
G3. **Arm 1 scoring**: prediction (lead): the 8 census probes that were
    lived-correct stay correct (0 regressions); t30 stays correct;
    t33 (lived refusal; census cause was a plant-era artifact, not a
    wrong value) — no prediction registered, report what it does.
    PASS RULE = 0 regressions on lived-correct probes. Everything else
    is reported.

GPU conventions: `/tmp/forge-gpu.lock` lease via the repo's
`gpu_lease`, single GPU, every run ≤ 10 min wall, ≥30 s gaps inside
the wrapper, operator has absolute right of way. Fork-from-snapshot
only. The frozen run tree is never written.

## File boundary

Modify ONLY: `scripts/grm_lsr_p2c_1_*.py` (new; may import from
`grm_sc1_2_session`), `tests/`, `artifacts/lsr_p2c_1/`, `logs/`.
Production code (`core/`, `config/`) READ-ONLY — if a production change
is required, STOP and report it as a blocker. Read-only: every
`scripts/grm_det1_*.py`, `scripts/lsr_*.py`, `scripts/grm_sc1_*.py`
(import only), `docs/`, `orders/`, all receipts.

## Principles (binding)

Determinism. NO git (lead commits). NO subagents. **NO background
processes for waiting**: no `run_in_background`, no watcher/poll loops,
no "block until X exits" helpers; foreground only, wait in-call, every
Bash call under 10 minutes. **Never kill, signal, or interrupt any
process you did not start** (multiple projects share this machine): no
kill/pkill/killall/fuser -k/nvidia-smi --gpu-reset/systemctl; processes
found via pgrep/ps are off-limits; wait on the GPU flock or report
blocked, never clear it. Acknowledge process safety in the final
report. RED honesty. Verify your own claims against artifacts before
writing them.

## Done (verbatim in the final message)

1. pytest summary line(s) + pre-existing set status.
2. Registration block as written before gates (incl. reachability).
3. G2 table (10 probes × N / reachable / reproduced / mass or text
   delta); G3 table (probe × lived / Arm 0 / Arm 1 / correct / change)
   with regression count.
4. Files created/modified.
5. Ambiguities, deviations, residual risks, anything RED; process-
   safety acknowledgement.
