# GRM-SC1 — Stage C demand loop: D-NGH as the production demand detector

David 2026-09-01 night: "Make it so" (Stage C wiring, option 1). This
order wires the race winner into the serving path so a turn that is
missing a needed memory NOTICES mid-generation and fetches. The
production default flip is a separate David decision with frame
receipts; this order lands the mechanism behind a flag and measures it.

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/GraftRepository` (branch
`lc1-wip`, HEAD 007efa6) — edits, builds, test runs, and BOUNDED GPU
runs AUTHORIZED. A registered order IS the permission: run first,
report after.

## Context (read fully, in order)

1. `docs/GRM_MOUNT_CURATION_DESIGN.md` — Stage C (the demand loop);
   `/mnt/Shared/GRM_Race_Report_2026-09-01.md` (verdict + queue).
2. The race receipt:
   `artifacts/grm_det1/run_20260831T160525Z_2/det1_4/campaign/analysis/analysis_receipt_89bcd331ae7f9ef4.json`
   (D-NGH recall 1.00, FP 0.10, F1 0.952 at 10 pairs; D-LQR
   REFUTED-STRUCTURAL — do not use live-query routing as the fetch
   source) and the frozen threshold
   `.../calibration/thresholds_2149a44b6136fd1b.json`:
   D-NGH `threshold = 0.3380523274342219`, score = per-token
   full-layer mean `mounted_mass`, `trigger_if_strictly_below`,
   selection = min of served-turn minima, **fit on 2 calibration
   turns** (thin envelope — carry it as the registered race constant,
   say so in every receipt, register no new number).
3. `scripts/grm_det1_common.py::DetectorObserver` (~540–800): the
   read-only D-NGH observer — wraps `gpt_oss20b_tc.sink_attention_tc`
   (full-attention layers only; sliding layers suppressed), computes
   `_full_mass` per token, `_summarize_mass` → `mounted_mass`;
   `detector_decision` (~840): fires on the FIRST token strictly below
   threshold. `fit_thresholds` (~795). This module is FROZEN race
   instrumentation: import/copy from it, never edit it.
4. The plant registry and planted-miss substrate:
   `scripts/grm_det1_7_registry.py`, `scripts/grm_det1_5_gpu.py`
   (`_measure_fixture_inline`, `_restore_counterfactual`, fork-from-
   snapshot), and the DET1.11 census. A planted miss = the needed
   sibling graft EXISTS in the repository and is UNMOUNTED — so
   recovery is possible in-turn.
5. `core/graft_arena.py::step()` as of P2C (plan-first fit, shuttle
   rungs, chunk descent, `attach_fit`, grounding-driven trips,
   snapshot/rollback), `core/grm_admission.py`, the P2C lived-order
   harness `scripts/lsr_p2c_replay_gpu.py` (Arm 0/1 pattern — reuse).
6. `core/grm_three_pass.py` route receipt: `ROUTE_RECEIPT_INFO_PREFIXES
   = ("fit_", "abstain")`. You MAY add `"demand_"` to that tuple (one
   line) and one test; nothing else in that file changes.

## Mission

### Part 1 — Production detector (`core/grm_demand.py`, new)

A production D-NGH observer usable inside `step()`: same capture
(full-attention layers, standard path only, sliding suppressed), same
per-token `mounted_mass`, same decision rule, threshold read from a
registered config entry (`config/grm_demand_registered.json`, new:
the race constant + its provenance sha + "fit on 2 turns" caveat).
Refuse loudly (not silently no-op) on a non-GPT-OSS or non-standard
attention arena — `demand_supported=False` in `info` with the reason.
**Bit-identity test vs the frozen DET1 observer**: on one fixture turn,
per-token `mounted_mass` from the production observer equals the DET1
`DetectorObserver` output exactly (GPU, short, leased).

### Part 2 — The demand trip in `step()` (and `_probe_ladder_chat`)

Behind `GRM_DEMAND_NGH` (env/flag, DEFAULT OFF; fail closed to OFF on
an unknown token). When ON:
1. Generate the attempt under the observer. If D-NGH fires at token
   index t (first strictly-below token), record
   `demand_fired=True`, `demand_token_index=t`, `demand_min_mass`.
2. Roll back to the pre-attempt snapshot (existing rollback path) and
   take ONE demand trip (cap = 1 per turn, registered): re-route using
   the question text PLUS the generated prefix up to token t (the
   model's own partial output as query augmentation — cheap, uses the
   existing `route()`/admission path, NOT D-LQR's live-query
   fingerprints), excluding already-mounted and live ids; admit
   plan-first (A-DEC on the augmented text), P2C fit rules apply.
   Record `demand_query_text_sha256`, `demand_ranking`,
   `demand_fetched` (ids newly mounted), `demand_fetch_source =
   "question_plus_prefix_reroute"`.
3. Serve the demand trip's answer iff it passes grounding; otherwise
   serve the original attempt's answer (state which was served:
   `demand_served = "demand_trip" | "original"`). Never loop: a
   second fire on the demand trip is recorded (`demand_refired`) and
   NOT acted on.
4. Flag OFF: served outputs byte-identical to P2C (test pins it).
5. All `demand_*` fields persist via the route receipt prefix.

### Part 3 — Overhead

Measure ms/token with the observer ON vs OFF on the same turn (three
turns, leased). Report; register no budget (David's call at flip time).

## Gates (synchronous, FOREGROUND, in-call waits only; RED is a result)

Register before any gate run (`artifacts/grm_sc1/grm_sc1_registration.json`):
demand-trip cap (1), the carried threshold + provenance, and the
predictions below.

G1. `python3 -m pytest -q tests/ --continue-on-collection-errors` —
    pre-existing failure/error set byte-identical to
    `logs/lsr_p2c_pytest_final.log`; new CPU tests: flag OFF byte-
    identity (fit/serve path unchanged); decision rule (first token
    strictly below; equal-to-threshold does NOT fire); one-trip cap;
    refire recorded not acted; unsupported-arena refusal; `demand_*`
    pass-through into the route receipt.
G2. **Bit-identity** (GPU, leased): production observer vs DET1
    `DetectorObserver` per-token `mounted_mass`, one served fixture
    turn, exact equality.
G3. **Recovery on the race's planted misses** (GPU, leased, bounded):
    the registered pairs from the race (planted-miss + served control
    per pair; take them from the plant registry / DET1.5 substrate via
    fork-from-snapshot exactly as the race did — never rebuild), demand
    ON. Registered predictions (lead, before the run):
    - detection recall on planted misses ≥ 0.9 (race: 1.00);
    - false fires on served controls ≤ 1 of 10 (race: 0.10);
    - **recovery** (planted-miss turn ends serving the expected value
      via the demand trip): prediction ≥ 5/10; PASS RULE for the
      mechanism = recall ≥ 0.9 AND FP ≤ 1; recovery is REPORTED, not
      gated (it is the number David flips on).
    Value comparison = DET1 semantic comparator with negative guards.
G4. **No regression on the lived battery**: P2C's lived-order replay
    (`scripts/lsr_p2c_replay_gpu.py`, Arm 1 conditions) with demand
    ON: the 9 sup probes serve exactly what P2C Arm 1 served
    (4 corrected + 5 unchanged), and `demand_fired` count on them is
    reported (prediction: ≤ 1; every fire on a correctly-served probe
    is a false fire — list it).
G5. Overhead table (Part 3).

GPU conventions: `/tmp/forge-gpu.lock` lease via the repo's
`gpu_lease`, single GPU, every run ≤ 10 min wall, ≥30 s gaps, operator
has absolute right of way. Fork-from-snapshot only; the DET1.3 boundary
(after admission) is FINE here because the demand loop acts at
generation time, after admission — say so in the report.

## File boundary

Modify ONLY: `core/grm_demand.py` (new), `core/graft_arena.py`,
`core/grm_admission.py`, `core/grm_runtime.py` (flag plumbing),
`core/grm_three_pass.py` (the one-line prefix addition + its test
ONLY), `config/grm_demand_registered.json` (new),
`scripts/grm_e2e_session.py` (flag + probe path), `scripts/grm_sc1_*.py`
(new), `tests/`, `artifacts/grm_sc1/`, `logs/`. Read-only: every
`scripts/grm_det1_*.py` and `scripts/lsr_*.py` (import only), `docs/`,
`orders/`, all receipts.

## Principles (binding)

Thresholds registered before gates, never adjusted after; the D-NGH
threshold is CARRIED, not refit. Determinism. NO git (lead commits).
NO subagents. **NO background processes of any kind for waiting**:
no `run_in_background` Bash, no watcher/poll loops, no "block until
pytest exits" helpers — every gate runs as a foreground command and
you wait in-call (the previous seat's leftover wait-loops fired
duplicate notices for an hour; that is a conduct violation here).
RED honesty. Verify your own claims against artifacts before writing
them.

## Done (verbatim in the final message)

1. pytest summary line + confirmation the pre-existing set is
   unchanged.
2. Registration block as written before gates.
3. G2 bit-identity receipt; G3 table (pair × fired / token index /
   fetched ids / served value / recovered?) with recall, FP,
   recovery; G4 table (9 probes, served value, demand_fired); G5
   overhead table.
4. Files created/modified; the `demand_*` fields and semantics; the
   exact production hook site in `step()`.
5. Ambiguities, deviations, residual risks, anything RED — and your
   recommendation on the default flip, with the numbers David needs.
