# CAMPAIGN-E: fold_order="s4" (hit-ordered fold sources) + G-FOLD gate harness

YOUR WRITABLE TARGETS in /mnt/ForgeRealm/GraftRepository (main tree):
- core/graft_repository.py
- tests/test_grm_s4_fold_order.py (new)
Everything else READ-ONLY (a sibling seat works a worktree; another
may touch unrelated test files — never touch core/graft_arena.py,
tests/test_grm_importance_g1g2.py, or tests/test_grm_s4_demotion.py).
No git. No subagents. No GPU / no model loads. RED honesty; no
monitor-idling.

## Law
docs/GRM_S4_FOLD_ORDER_PLAN.md (immutable) IS the spec — read it in
full, plus docs/GRM_S4_LEDGER.md (the s4 counters you consume, the
G2-S4 FAIL mechanism the plan's thesis answers) and the deferred
librarian machinery in your target file (_librarian_jobs, _foldable,
_due, backpressure path — and the M11 payload-resolvability guard and
816a0a0 counter-jam lessons, both of which your change must not
disturb).

## Build
1. fold_order="s4" | "age" (default "age", byte-identical): within
   the librarian's EXISTING eligible fold-source set, selection order
   becomes (n_grounded ascending, then existing age order). Both
   deferred and backpressure paths. Eligibility, thresholds, fold
   count, digest prompts: UNCHANGED.
2. G-FOLD gate harness (tests/test_grm_s4_fold_order.py, --run-gpu
   legs lead-executed): A/B identical sessions (fingerprint-checked,
   pattern from test_grm_s4_demotion.py) driving enough turns that
   ≥4 folds fire per arm — extend the repeat-probe fixture style
   (tests/fixtures/importance_convos_repeat/) with probes AFTER folds
   so post-fold recall is measurable. Receipts per arm (schema
   grm_s4_fold_order_v1): late recall, fidelity-abort count, per-fold
   composition (which turns folded, their n_grounded at fold time),
   add_turn max latency. --analyze prints the registered verdict
   (PASS = recall ≥, aborts ≤, hot-path flatness unchanged).
3. CPU units: ordering key correctness (hits ascending, age
   tiebreak), default byte-identical, both-paths coverage, no
   counter jam, M11 placeholders still excluded regardless of order.

## Verify
py_compile; python3 -m pytest tests/test_grm_s4_fold_order.py tests/test_grm_s4_ledger.py tests/test_grm_s4_demotion.py tests/test_grm_fold_recovered_guard.py tests/test_grm_importance_salience.py -q — paste summary.

## Done
Verbatim: diff summary; pytest line; exact --run-gpu commands +
analyze; how the harness guarantees ≥4 folds per arm; ambiguities;
"no git, no GPU, no files outside grant".
