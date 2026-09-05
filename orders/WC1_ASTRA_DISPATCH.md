DISPATCH CONTEXT (lead, 2026-09-04): You are the implementation seat for order GRM-WC1. YOUR WRITABLE TARGET is this worktree: /mnt/ForgeRealm/GraftRepository-wt-wc1-astra (branch wc1-astra, forked from lc1-wip @ 29c8ef8). Never touch /mnt/ForgeRealm/GraftRepository (main checkout) or /mnt/ForgeRealm/GraftRepository-wt-wc1-opus (a parallel seat running the SAME order). artifacts/ and logs/ are gitignored: read RS3/RT1/EB1 receipts read-only from /mnt/ForgeRealm/GraftRepository/artifacts/ via absolute paths; write yours under this worktree's artifacts/grm_wc1/. The native runtime library may need building in this worktree (cpp/build) as RT1 did; the frozen race tree is read-only. NO git. NO subagents. NO background waits (foreground with explicit timeout; every command under 10 minutes). Never kill a process you did not start; wait on /tmp/forge-gpu.lock via gpu_lease, never clear it. Registration file immutable. In your final report state your model id and reasoning effort. The order follows verbatim.

# GRM-WC1 — The arena width curve under the spec frame with seating on

Lead-authored 2026-09-04 night. Primer open question 2: "How wide
should the arena be?" has never been measured. 96 seats is the
reliability harness default (chosen because GPT-OSS/YaRN collapses at
width 384); the arena class defaults to 256. Co-mount blending argues
against more residents; splitting and shuttling argue against fewer.
With the spec frame (EB1) and seat-near-live (RS3) in place, the
question is finally well-posed. Measurement only. No production
change.

YOUR WRITABLE TARGET is the worktree you were dispatched into (stated
in your dispatch message; branch forked from `lc1-wip` @ 29d860b) —
scripts, tests, artifacts, and BOUNDED GPU runs AUTHORIZED. A
registered order IS the permission: run first, report after.
Production code (`core/`, `config/`) is READ-ONLY.

## Context (read fully, in order)

1. `docs/GRM_Methodology.md` §6 (residency), `docs/GRM_E2E_RECEIPT_LEDGER.md`
   (~249–260: the width-384 collapse and why 96), `docs/LSR_ADDENDUM_1.md`
   (P2C: splitting at width 96; chunk can take 96/96 seats).
2. `core/graft_arena.py::__init__` (`arena_width`, `live_shift =
   n_sink + arena_width`, `extend_rope`), `_guard_deposit_width` /
   `mountable_budget` (P2C: the width governs deposit-time splitting),
   `_rs3_seat_plan` (RS3 seating), `core/grm_frame.py` (flags).
3. Harnesses: `scripts/lsr_p2c_replay_gpu.py` (sup battery,
   deposit-order, `--arena-width` / flags), `scripts/lsr_p2c_e2e_gpu.py`
   (census, sharded), `scripts/grm_eb1_longhorizon_gpu.py` (104-turn).
   Receipts to compare against at width 96: RS3 Part 4
   (`artifacts/grm_rs3/grm_rs3_part4.json` in the MAIN checkout —
   `artifacts/` is gitignored, read it there read-only), RT1 G2/G3.
4. `core/grm_demand.py` / RS2's `LayerTypeMassObserver` for mounted
   mass at readout (optional column; do not let it block the gate).

## Mission

Sweep `arena_width ∈ {64, 96, 128, 192, 256}` under the spec frame
(`ephemeral` default), fixes ON, `GRM_SEAT_NEAR_LIVE=1`,
`GRM_CAPTURE_PIN=live`, demand OFF. Note that width changes the
deposit-time split budget too (P2C), so nodes split differently per
width — that is part of the measurement, not a confound; receipt the
split counts.

For each width: sup battery (9 probes), census (10 probes, sharded),
and the 4 long-horizon spot probes at distances 36–60 from EB1 G5
(reuse the generator; same seed). Report per width: correct counts,
regressions vs width 96, mean mounted mass at readout (if the observer
attaches cleanly), number of split nodes, mean resident seats per
turn, mean wall ms per turn.

## Registered predictions (lead, before any run)

- 96 is NOT the optimum: 128 and 192 match or beat 96 on all three
  batteries (fewer splits, no co-mount blending because admission is
  k=1 on these probes).
- 64 loses at least 2 sup probes (unseatable-after-split class).
- 256 shows the first sign of the YaRN wall: wall ms per turn rises
  and at least one census probe degrades; if it does not, say so —
  the 384 collapse may be a cliff, not a slope.
- Long-horizon 4/4 at every width ≥ 96.

## Gates (synchronous, FOREGROUND, in-call waits only; RED is a result)

Register before any gate run (`artifacts/grm_wc1/registration.json`,
IMMUTABLE; amendments in a separate file citing its sha256): widths,
flags, seeds, predictions. No numeric constants beyond the width grid.

G1. `python3 -m pytest -q` sharded under 10 min per call — pre-existing
    set unchanged vs the RS4 logs in the main checkout; new CPU tests
    for the sweep driver's pure parts (plan, table assembly,
    regression-vs-96 computation).
G2. **Width 96 reproduces RS3 Part 4 + RT1**: sup 9/9, census 9/10,
    long-horizon 4/4, served text identical (semantic comparator). A
    non-reproducing 96 stops the sweep — report it.
G3. The sweep table (5 widths × 3 batteries) + predictions hit/miss +
    a one-paragraph reading of the curve.

GPU conventions: `/tmp/forge-gpu.lock` lease via the repo's
`gpu_lease`, single GPU (ANOTHER SEAT IS RUNNING THE SAME ORDER IN A
PARALLEL WORKTREE — you WILL contend for the lock; wait on it, never
clear it), every run ≤ 10 min wall, ≥30 s gaps inside the wrapper,
operator has absolute right of way. Frozen run tree never written.

## File boundary

Modify ONLY: `scripts/grm_wc1_*.py` (new), `tests/`, `artifacts/grm_wc1/`,
`logs/`. Read-only: everything else, including `core/`, `config/`, all
other scripts (import only), `docs/`, `orders/`, all receipts (the main
checkout's `artifacts/` is read-only reference material).

## Principles (binding)

Determinism. NO git (lead commits). NO subagents. **NO background
processes for waiting**: no `run_in_background`, no watcher/poll loops,
no "block until X exits" helpers; foreground only with explicit
`timeout`, wait in-call, every Bash call under 10 minutes (shard the
census by turn range with save/restore as `lsr_p2c_e2e_gpu.py` does).
**Never kill, signal, or interrupt any process you did not start**
(multiple projects and a parallel seat share this machine; no
kill/pkill/killall/fuser -k/gpu-reset/systemctl; pgrep/ps hits are
off-limits; wait on the flock or report blocked). Acknowledge process
safety in the report. RED honesty: a width that fails is a result.
Verify your own claims against artifacts before writing them.

## Done (verbatim in the final message)

1. pytest summary line(s) + pre-existing set status.
2. Registration block as written before gates + its sha256; any
   amendment file.
3. G2 reproduction table; G3 sweep table; predictions hit/miss; the
   reading of the curve.
4. Files created; artifact paths with sha256.
5. Ambiguities, deviations, residual risks, anything RED; process-
   safety acknowledgement; seat model and effort used.
