# GRM-RT1 — A fit-time split child must not outrank the fact node it competes with

Lead-authored 2026-09-04 from RS1/RS3. `sup_solace_fresh` routes to
graft 4 — a P2C fit-time split child of the sable COMPETITOR — instead
of graft 1, the solace FACT node. RS1 measured it; RS3 proved the cost:
with the fact node mounted the probe is correct at mounted mass 0.405
under the seating lever, with the split child it refuses. `sup_reserve_
tundra_ledger` also serves from graft 4 and happens to be right only
because the chunk carries its value. This is a routing/descent defect
introduced by P2C's split: the child inherits the parent's routing
surface and a chunk that is topically hot outranks the identifier-
bound fact node. Fix at admission, not at readout.

YOUR WRITABLE TARGET is the worktree
`/mnt/ForgeRealm/GraftRepository-wt-rt1` (branch `rt1-split-child`,
forked from `lc1-wip`) — edits, builds, test runs, and BOUNDED GPU
runs AUTHORIZED. A registered order IS the permission.

## Context

1. `artifacts/grm_rs1/grm_rs1_results.json` (solace/tundra mount ids),
   `artifacts/grm_rs3/grm_rs3_results.json` (solace-fact panel),
   `artifacts/grm_rs3/registration.json` (`fixture_node_to_idx`).
2. P2C: `core/graft_repository.py` split guard (`_guard_deposit_width`,
   `cull_graft`, children inherit `rare`/routing surface; parent kept
   as index), `core/graft_arena.py::_descent_source_children`,
   `_identifier_bearing_children`, `_split_unseatable`; A-DEC
   `core/grm_admission.py` (`identified_candidates`, `is_identifier_
   binding`, `policy_plan`); `docs/LSR_ADDENDUM_1.md` Phase 2.
3. The lived-order sup replay `scripts/lsr_p2c_replay_gpu.py` (EB1
   spec frame; pass `GRM_SEAT_NEAR_LIVE`/`GRM_CAPTURE_PIN` through as
   RS3 wired them).

## Mission

1. Diagnose from receipts: on solace, what ranks graft 4 above graft 1
   — topical score, identifier hits on the chunk (does the sable
   chunk bind "solace"/"key"?), or L2/lineage? Write the ranking table
   for the probe with both candidates' channel scores.
2. Fix at admission with a structural rule, no new threshold: a
   split child (or its index parent) that does not bind the question's
   identifier tokens never outranks a node that does; and when a child
   and a non-child both bind, the child's identifier binding must be
   its OWN text's, never inherited from the parent's union surface.
   Receipt the decision (`admission_split_child_demoted`, ids).
3. Tests: the solace ranking flips to graft 1 (CPU, from the
   registered fixture texts); tundra still routes to a chunk that
   carries its value (or to its fact node — report which); legacy path
   byte-identical with the rule OFF (`GRM_LSR_FIXES` governs it).

## Gates (synchronous, FOREGROUND, in-call waits only)

Register before any gate (`artifacts/grm_rt1/registration.json`,
immutable; amendments separate). G1: pytest sharded, pre-existing set
unchanged vs RS3. G2 (GPU, leased, bounded): sup battery under the spec
frame with RS3's pair ON, rule ON — registered prediction: **9/9**
(solace recovers, nothing regresses); rule OFF reproduces RS3's 8/9
bit-equal on served text. G3: census 10 probes, same levers, rule ON:
9/10, 0 regressions (t33 is admission-identifier, out of scope; report
whether the rule touches it).

File boundary: `core/grm_admission.py`, `core/graft_arena.py`
(descent/identifier-child helpers only), `core/graft_repository.py`
(split metadata only), `scripts/grm_rt1_*.py`, `tests/`,
`artifacts/grm_rt1/`, `logs/`. Read-only: everything else, all
receipts. NO git (lead commits; you are on a worktree branch the lead
created), NO subagents, NO background waits (foreground with explicit
`timeout`, every call < 10 min), never kill a process you did not
start; process-safety acknowledgement. Thresholds: none new. RED
honesty.

## Done

1. pytest lines + pre-existing set. 2. Registration + sha.
3. The solace ranking table before/after; G2 table 9 probes (rule
   OFF/ON); G3 table. 4. Files; the rule stated in one sentence.
5. Deviations/risks/RED; process safety; the number David flips on.
