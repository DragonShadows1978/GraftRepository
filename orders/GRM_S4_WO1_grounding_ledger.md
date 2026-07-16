# S4-WO1: Grounding-hit ledger — attribution split + counters + fixtures + gate harness

YOUR WRITABLE TARGETS — edits AUTHORIZED on exactly these paths in
/mnt/ForgeRealm/GraftRepository (main tree):
- core/graft_arena.py
- core/graft_repository.py
- tests/test_grm_s4_ledger.py (new)
- tests/fixtures/importance_convos_repeat/ (new directory)

Everything else is READ-ONLY reference. No git (the lead commits). No
subagents. No GPU runs / model loads — CPU verification only; the lead
runs all GPU gates serially. RED honesty: a finding that the design
can't work as specified is a deliverable, not a failure — report it,
don't paper it. No monitor-idling: when your work is done, finish.

## Law

docs/GRM_S4_GROUNDING_LEDGER_PLAN.md is the spec — read it FIRST, in
full, plus docs/GRM_IMPORTANCE_LEDGER.md (predecessor receipts: the
registered floors, the seating-epoch invariant, the measurement laws).
If anything below conflicts with the plan, the plan wins; report the
conflict.

## Build

1. **Attribution split** (core/graft_arena.py): refactor _grounded()'s
   pooled coverage so per-mount contribution is computable; a mounted
   node scores a grounded-hit iff the answer is grounded AND the node
   contributes ≥1 token to the content-coverage set (with the
   substantive-word fallback mirroring the existing ladder). HARD
   CONSTRAINT: the pooled grounded/not-grounded verdict must be
   bit-identical to current behavior — the split is a pure refinement.
   Structure it so the existing _grounded() call sites are untouched
   or trivially adapted.
2. **Counters** (arena + repository): per-node
   metadata["importance"]["s4"] = {n_routed, n_mounted, n_grounded,
   last_grounded_turn}. n_routed increments when a node enters the
   routed candidate set; n_mounted on mount in an ACCEPTED attempt;
   n_grounded per the attribution split. Rolled-back trips never
   count (mirror the S1 seating-epoch discipline — study how
   _attempt/step snapshot and roll back). Persistence rides the
   existing metadata path (the importance dict location is already
   program-approved; write ONLY the s4 key).
3. **Repeat-probe fixtures** (tests/fixtures/importance_convos_repeat/):
   4 conversations per the plan's G1-S4 spec — ≥4 facts each probed
   TWICE ≥8 turns apart + never-probed graded controls; same schema
   as tests/fixtures/importance_convos/ (reuse its README schema and
   validator pattern; probe→scripted-answer turn structure included —
   the G1/G2 driver consumes that shape). House fact style (identifier
   plantables + relational), no cross-file identifier collisions, and
   include SOLO user turns deliberately in at least one file (the
   driver now supports them; keep the surface exercised).
4. **Gate harness** (tests/test_grm_s4_ledger.py): CPU units for the
   attribution split (pooled-verdict bit-parity on synthetic cases,
   per-mount hit assignment, no-hit on hedges, rollback exclusion) +
   G0-S4 GPU smoke and G1-S4 driver behind --run-gpu flags the lead
   executes (import S3 machinery from
   tests/test_grm_importance_counterfactual.py and driver plumbing
   from tests/test_grm_importance_g1g2.py — import, never edit those
   files; if you need a change in one, STOP and report). Analysis
   mode computes the plan's inherited bars + the secondary lift
   metric from sealed artifacts.

## Verify

python3 -m py_compile on all touched files; run your CPU tests AND the
predecessor suites that share your files:
  python3 -m pytest tests/test_grm_s4_ledger.py tests/test_grm_importance_telemetry.py tests/test_grm_importance_salience.py tests/test_grm_fold_recovered_guard.py tests/test_grm_importance_g1g2.py -q
All green or explain precisely.

## Done

Final message MUST contain verbatim: diff summary per file; the pytest
summary line for the full command above; fixture validator output;
exact --run-gpu commands for the lead (G0-S4 smoke, G1-S4 legs,
analyze); every spec ambiguity you hit and how you resolved or
escalated it; confirmation "no git, no GPU, no files outside grant".
