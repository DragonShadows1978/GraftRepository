# GRM-SC2 — Demand loop: wider calibration (rule unchanged) + early abort at the fire token

David 2026-09-02 midday: "write it up and ship it out" (optimization
items 1+2). Two things, one order, measurement first:

- **Part A (CPU, receipts-only): calibrate the D-NGH threshold on every
  lived served control we hold, under the race's UNCHANGED envelope
  rule**, with a registered calibration/held-out split, and report the
  candidate line. Production keeps the carried race threshold; adopting
  the candidate is David's decision with these frame receipts.
- **Part B (GPU, bounded): abort generation at the fire token** instead
  of generating the whole wrong answer, with a suspended-attempt
  fallback so a failed trip still serves exactly what it served before.

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/GraftRepository` (branch
`lc1-wip`, HEAD c585100) — edits, builds, test runs, and BOUNDED GPU
runs AUTHORIZED. A registered order IS the permission: run first,
report after.

## Context (read fully, in order)

1. `orders/GRM_SC1_DEMAND_LOOP_NGH.md`, `GRM_SC1_1_GROUNDING_GLYPHS.md`,
   `GRM_SC1_2_E2E_PAIRS_SESSION_RESUME.md` and their receipts under
   `artifacts/grm_sc1/`, `artifacts/grm_sc1_1/`, `artifacts/grm_sc1_2/`
   (per-token `mounted_mass` rows for every served control and planted
   miss are PERSISTED there — Part A needs no GPU).
2. The race threshold file
   `artifacts/grm_det1/run_20260831T160525Z_2/det1_4/campaign/calibration/thresholds_2149a44b6136fd1b.json`
   and `scripts/grm_det1_common.py::fit_thresholds` (~795) — the rule:
   `threshold = min over served turns of (min over answer tokens of
   mounted_mass)`, direction `trigger_if_strictly_below`. THIS RULE IS
   NOT CHANGED. `detector_decision` (~840): first token strictly below.
3. `core/grm_demand.py`, `core/graft_arena.py::step()` demand block
   (~3080–3130 per SC1's report; `_demand_trip`, `_serve` wrapper),
   `_attempt` (the generation loop the observer wraps),
   `config/grm_demand_registered.json`.
4. `scripts/grm_sc1_2_e2e_recovery_gpu.py`, `scripts/grm_sc1_recovery_gpu.py`
   (the 10-pair harness; reuse), `scripts/grm_sc1_lived_battery_gpu.py`,
   `scripts/grm_sc1_overhead_gpu.py`.

## Part A — calibration widening (CPU only)

1. **Inventory every lived served-control turn with persisted per-token
   mass rows**: race calibration (2), race eval served controls (10),
   SC1.1 G3 lived battery served turns (9, demand-OFF arm), SC1.2 Arm 0
   served controls (7 — these overlap the race eval set: de-duplicate
   by probe id and prefer the row with the lived provenance; state the
   final count). Also inventory every planted-miss row (9 measurable).
2. **Register the split BEFORE computing anything**: calibration set =
   a deterministic subset chosen by a rule written in the registration
   (e.g. odd-indexed by sorted probe id, or "all race rows" vs "all
   post-race rows") — pick one, state it, never re-split. Held-out =
   the rest. Both sets non-empty; held-out must contain
   `sup_solace_fresh` (the current false fire) OR you must say why
   the rule excluded it.
3. **Compute the candidate threshold** = envelope rule over the
   calibration set. Report: candidate value, calibration set minima,
   the carried race value, and the **gap**: max planted-miss mass
   (expected 0.0) vs min served-control mass over ALL rows (expected
   ~0.291). Then, on the HELD-OUT set only: false-fire count at the
   carried threshold vs at the candidate; recall on the 9 planted
   misses at both (deterministic from rows via `detector_decision`
   semantics — reimplement nothing; import or copy the one-line rule
   and pin equality with a test against the DET1 function).
4. **Sensitivity, reported not gated**: the held-out false-fire count
   as the line sweeps from 0.05 to 0.45 in 0.01 steps, and recall
   likewise — one small table/JSON. No value from this sweep is
   registered or adopted; it is the picture David needs.
5. Write the candidate into `config/grm_demand_registered.json` as
   `candidate_threshold` with provenance (calibration set ids, rule
   sha, order id). **Production continues to read the carried race
   threshold** — a test pins that the detector uses `threshold`, not
   `candidate_threshold`, until a config field `adopted_by` is set (it
   is not set by this order).

## Part B — early abort at the fire token (GPU)

1. In the demand block: when D-NGH fires at token t (first strictly-
   below token), STOP generating the attempt at t (do not produce the
   rest of the wrong answer). Suspend the attempt: snapshot its cache/
   position/live segments (the ladder already snapshots; reuse) so it
   can be resumed.
2. Take the demand trip exactly as SC1 does (rollback, question +
   prefix-up-to-t re-route, plan-first admission, P2C fit, generate).
3. If the trip grounds: serve it (as today). If not: **resume the
   suspended attempt from token t** and finish generating it, then
   serve that — the served text must be byte-identical to what SC1.2
   served on that pair (t30 is the live test: it must still serve
   "0.0", not something new).
4. Receipts: `demand_abort_token_index`, `demand_tokens_generated_before_abort`,
   `demand_tokens_saved` (= original attempt length in SC1.2's receipt
   minus tokens generated before abort, when the trip served),
   `demand_resumed_original` (bool), `demand_wall_ms_attempt`,
   `demand_wall_ms_trip`, `demand_wall_ms_resume`.
5. Flag: `GRM_DEMAND_EARLY_ABORT` default ON when demand is ON (state
   it); OFF path = SC1 behavior byte-identical (test pins). Demand OFF:
   nothing changes (test pins).

## Gates (synchronous, FOREGROUND, in-call waits only; RED is a result)

Register before any gate run (`artifacts/grm_sc2/registration.json`):
the Part A split rule + set ids, the envelope rule sha, the predictions
below. No numeric constants other than the sweep grid.

G1. `python3 -m pytest -q` sharded under 10 min per call — pre-existing
    set unchanged vs SC1.2's logs (note SC1.2's finding that
    `test_graft_quant_format` failures are contention artifacts; report
    idle-card status if you can get one); new CPU tests: rule equality
    with DET1 `fit_thresholds`/`detector_decision`; split determinism;
    candidate not read by production; early-abort OFF byte-identity;
    resume path serves the original text (fixture-level).
G2. **Part A receipts** (`artifacts/grm_sc2/sc2_calibration_*.json`):
    prediction (lead): candidate threshold ≤ 0.30; held-out false
    fires at candidate = 0; recall at candidate 9/9; gap ≥ 0.25.
G3. **Part B on the 10 pairs** (reuse the SC1.2 + SC1.1 harnesses,
    demand ON, early-abort ON, CARRIED threshold): recovery identical
    to SC1.2 (8/9, same pairs), served text on every pair byte-
    identical to SC1.2's served text (including t30 via the resume
    path), `demand_resumed_original = True` on exactly t30; report
    tokens saved and wall-ms per pair. Prediction: mean tokens
    generated before abort ≤ 2 (fires at token 0 → abort after the
    first token); wall time per recovered turn drops by ≥ 30% vs SC1.2.
G4. **Lived battery** (`scripts/grm_sc1_lived_battery_gpu.py`, demand
    ON, early-abort ON): 9/9 served values identical to SC1.1 G3; the
    one false fire (sup_solace) now takes the resume path and serves
    the identical text.

GPU conventions: `/tmp/forge-gpu.lock` lease via the repo's
`gpu_lease`, single GPU, every run ≤ 10 min wall, ≥30 s gaps inside
the wrapper, operator has absolute right of way. Fork-from-snapshot
only. Frozen run tree never written.

## File boundary

Modify ONLY: `core/grm_demand.py`, `core/graft_arena.py` (demand block
+ `_attempt` abort/resume seam), `config/grm_demand_registered.json`,
`scripts/grm_e2e_session.py` (flag plumbing + probe-path abort/resume
mirror), `scripts/grm_sc2_*.py` (new), the SC1.x harness scripts
(receipt fields only), `tests/`, `artifacts/grm_sc2/`, `logs/`.
Read-only: `core/grm_three_pass.py` (demand_ prefix already persists),
every `scripts/grm_det1_*.py` and `scripts/lsr_*.py` (import only),
`docs/`, `orders/`, all receipts.

## Principles (binding)

The envelope rule is carried unchanged; the candidate is REPORTED and
recorded, never adopted here. Split registered before computation.
Determinism. NO git (lead commits). NO subagents. **NO background
processes for waiting**: no `run_in_background`, no watcher/poll loops,
no "block until X exits" helpers; foreground only, wait in-call, every
Bash call under 10 minutes. RED honesty. Verify your own claims
against artifacts before writing them.

## Done (verbatim in the final message)

1. pytest summary line(s) + pre-existing set status.
2. Registration block as written before gates.
3. Part A: inventory counts, split, candidate threshold, gap, held-out
   false fires + recall at carried vs candidate, the sweep table.
4. Part B: 10-pair table (pair × fired / abort idx / tokens before
   abort / trip grounded / resumed / served text identical / tokens
   saved / wall-ms before vs after); G4 status.
5. Files created/modified; new `demand_*` fields.
6. Ambiguities, deviations, residual risks, anything RED — and the
   two numbers David adopts or not: the candidate threshold and the
   early-abort default.
