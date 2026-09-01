# GRM-LSR-P2C — Unseatable nodes: split at deposit, descend at fit, shuttle the chunks

Lead-authored 2026-09-01 evening from the P2A finding. David's ruling for
Phase 2 was "all three, shuttle over fail-loud"; P2A made the fit stage
honest and found that Ruling 1 alone cannot flip the four ADMISSION-PRUNE
probes: their answer-bearing nodes (623–706 chars) exceed the 96-seat
arena ALONE (`fit_unseatable` non-empty on all four,
`artifacts/lsr_p2a/lsr_p2a_replay_*.json`). Explicit degrade is honest but
still serves without the answer. This order supplies the missing lever.
Mechanism, not policy: the width stays 96 (frozen driver default,
`scripts/grm_e2e_session.py:263`); no threshold changes.

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/GraftRepository` (branch
`lc1-wip`, HEAD = the P2A+P2B merge) — edits, builds, test runs, and
BOUNDED GPU runs AUTHORIZED. A registered order IS the permission: run
first, report after.

## Context (read fully, in order)

1. `orders/GRM_LSR_P2A_FIT_HONESTY_SHUTTLE.md` + the P2A seat's receipts:
   `artifacts/lsr_p2a/lsr_p2a_registration.json`,
   `lsr_p2a_g2_blocked_report.json`, `lsr_p2a_replay_*.json`.
2. `core/graft_arena.py::step()` as it now stands (plan-first fit,
   shuttle rungs, `fit_unseatable`, `served_without_plan_head`,
   `_serve_abstention`); `core/grm_admission.py` (`plan_priority_fit`,
   `shuttle_trip_cap`, `fit_info_fields`).
3. Descent machinery: `core/graft_arena.py::_descent_source_children`
   (identifier-filtered children via `qrare & child["rare"]`),
   `_descent_expand`; era/digest/children semantics (comments ~2540–2551:
   eras are INDEX nodes, never readers; children are the readers).
4. Librarian split: `core/graft_repository.py::cull_graft` /
   `split_graft` (~900–1000), `_section_text_chunks` (~591–640),
   `add_document`, `remember`, `add_turn`; `core/graft_arena.py::deposit`
   / `deposit_from_cache` (~505–540).
5. `scripts/grm_det1_5_gpu.py` — the lived campaign's fixture
   definitions and deposit sequences for the `sup_*` fixtures (the
   supersession battery), and its serving path; `scripts/lsr_p2a_replay_gpu.py`
   (P2A's leased replay harness — extend, don't fork).
6. `core/grm_three_pass.py` route receipt (P2B): every new `info` field
   you add must be `fit_`-prefixed (or listed) so it persists for free.

## Principle being implemented

**The repository never holds a node the arena cannot mount, and a node
too long to seat is served across its chunks, not replaced by a
neighbor.** Same shape as co-mount prevention: fix at deposit/admission,
not at readout.

## Mission

### Part 1 — Deposit-time width guard (prevention)

Any deposit path (`add_turn`, `add_document`, `remember`, arena
`deposit`/`deposit_from_cache`, deferred-turn deposit) whose resulting
node has `ntok > mountable_budget` splits the node at deposit into
width-fitting children under an index parent, using the librarian's
existing split (`cull_graft`) and chunker (`_section_text_chunks`, with
a sentence/line fallback so no child exceeds the budget). The parent
keeps `kind` semantics of an index node (era-class: routable, never a
reader; descends to children), children inherit provenance, tags,
lineage/supersession membership, and identifier binding (each child's
`rare` tokens are its own; the parent's routing surface is the union so
identifier routing still finds the family).
`mountable_budget` = arena width minus the registered sink/question
reserve the arena already applies in `fit` — derive it from the code,
state the number, register it before gates. No new constant beyond that
derivation.

### Part 2 — Fit-time descent for legacy oversized nodes (repair)

When P2A marks a plan member `fit_unseatable`, `step()` (and the
driver's `_probe_ladder_chat` fit site) does NOT degrade first. It
splits-and-descends: the unseatable node is chunked on the fly (same
chunker; persisted via `cull_graft` if `deposit`/mutation is allowed on
this path, else an ephemeral in-turn split — state which, and receipt
it as `fit_split_ephemeral`), and the plan head becomes the
identifier-bearing child set (`_descent_source_children` with `qrare`).
If that child set co-fits: seat it. If not: SHUTTLE across the chunks in
document order (P2A's shuttle rungs; cap = number of chunks, register
it), composing the answer from grounded trips. `served_without_plan_head`
becomes true ONLY when every chunk trip fails grounding.

New `info` fields (all `fit_`-prefixed): `fit_split_parent`,
`fit_split_children`, `fit_split_ephemeral`, `fit_descended_head`,
`fit_chunk_trips`.

### Part 3 — Librarian sweep op

A librarian operation (`split_oversized` or similar, following the
existing cull/split verb grammar at `graft_repository.py` ~1050) that
applies Part 1 to an existing repository's oversized nodes, so legacy
repositories are repaired once rather than at every fit.

## Gates (synchronous, FOREGROUND; RED is a result)

Register before any gate run (`artifacts/lsr_p2c/lsr_p2c_registration.json`):
`mountable_budget`, chunk-trip cap, and the predictions below.

G1. `python3 -m pytest -q tests/` — full suite green modulo the
    documented pre-existing environment failures (list them, unchanged
    set vs P2A's log `logs/lsr_p2a_pytest_final.log`). New CPU tests:
    deposit guard splits at the budget (and NOT below it — a node at
    exactly `mountable_budget` is one node); children carry lineage +
    identifiers; era-class parent never seated as a reader; fit-time
    descent seats the identifier child; chunk shuttle when children
    don't co-fit; `served_without_plan_head` only on total grounding
    failure; legacy path (`GRM_ADM_DECISIVE=0`) byte-identical; P2B
    receipt carries the new fields (pass-through).
G2. **Lived-order replay (GPU) — the gate P2A could not run.** Build each
    `sup_*` fixture by replaying its DET1.5 deposit sequence turn by turn
    (same texts, same order, same repository state at the probe turn as
    the lived manifest's `lived_repository_size_at_probe`), then serve
    the probe. Two arms, both leased and bounded:
    - **Arm 0 (reproduction, fixes OFF via a flag you add, default ON):**
      must reproduce the lived served values on all 9 sup probes
      (4 wrong values verbatim-by-semantics, 2 NOT-YET-DEPOSITED wrong
      values, 3 controls correct). If Arm 0 does not reproduce, the
      replay is not lived-equivalent: STOP, report which probe diverged
      and how, do not run Arm 1 as evidence.
    - **Arm 1 (P2A+P2C ON):** registered prediction (lead): the 4
      ADMISSION-PRUNE probes serve the expected value; the 2
      NOT-YET-DEPOSITED serve the SAME older value as before (they are
      correct relative to memory; unchanged); the 3 controls unchanged;
      zero abstentions; `fit_unseatable` empty on all 9 (split happened
      at deposit).
    Pass rule: Arm 0 = 9/9 reproduction; Arm 1 = 4 flips + 5 unchanged,
    zero control regressions. Value comparison = the DET1 semantic
    comparator with negative guards.
G3. The 10 `e2e_t*` census probes: if a deposit-order replay of the
    certified 34-turn session fits in bounded runs (≤10 min each,
    sharded by turn range with repository state carried by
    save/restore), run Arm 0/Arm 1 the same way. If it does not fit the
    bounds, deliver the sharded script + a blocked-report; do not
    exceed the bounds and do not fake it.

GPU conventions: `/tmp/forge-gpu.lock` lease via the repo's `gpu_lease`
(P2A used it: longest run 71 s), single GPU, every run ≤ 10 min wall,
≥30 s gaps, operator has absolute right of way.

## File boundary

Modify ONLY: `core/graft_arena.py`, `core/graft_repository.py`,
`core/grm_admission.py`, `core/grm_runtime.py` (deposit wiring only),
`scripts/grm_e2e_session.py` (fit site + flag), `scripts/lsr_p2a_replay_gpu.py`
(extend for Arm 0/1) or new `scripts/lsr_p2c_*.py`, `tests/`,
`artifacts/lsr_p2c/`, `logs/`. Read-only: `core/grm_three_pass.py`
(P2B; if a field will not pass through, report it rather than editing),
`docs/`, `orders/`, all DET1/LSR receipts and scripts (import only).

## Principles (binding)

Thresholds/caps registered before gates, never adjusted after.
Determinism. NO git (lead commits). NO subagents. NO monitor/watcher
idle loops. RED honesty; a non-reproducing Arm 0 is a result, not a
nuisance. Verify your own claims against artifacts before writing them.

## Done (verbatim in the final message)

1. pytest summary line, plus the unchanged pre-existing failure set.
2. Registration block (budget, cap, predictions) as written before gates.
3. G2 table: probe × (expected, lived, Arm 0, Arm 1, fit receipt,
   verdict). G3 table or blocked-report.
4. Files created/modified; new `info` fields; the deposit paths guarded
   (every one, named) and any deposit path deliberately left unguarded
   with the reason.
5. Ambiguities, deviations, residual risks, anything RED.
