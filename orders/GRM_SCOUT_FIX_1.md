# GRM-SCOUT-FIX-1 — fix the Part B surface findings and land the Part C #1 edge-case fixtures (worktree wt-grm-fix1, branch grm-fix1)

Origin: `docs/GRM_SCOUT_2026-09-08.md` Part B (your own findings) and
Part C rank 1. David's standing goal: work the Scout report to
completion; where there is a decision, test both sides. Reasoning
effort: **high**.

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/wt/grm-fix1` (branch
`grm-fix1`, forked from `lc1-wip`). These ARE production changes to
`core/` and `scripts/`, so: minimal, each fix its own commit-sized
diff (the lead commits), each pinned by a test that fails before and
passes after, no behaviour change beyond the finding, existing
batteries untouched. NO GPU (all five items are CPU-provable; the EB1
E2E re-validation is a later lead-run gate). Prior Art Directive
applies (nothing new expected). No git, no subagents, no background
waits, never kill anything.

## Items (each with the line refs from the Scout report)

B1. **Final-forward demand abort can become an exception.**
    `core/grm_demand.py:401–404` sets `aborted_at` on the final
    cache-commit forward; `core/graft_arena.py:4396–4411` swallows the
    abort expecting the unused row to be dropped, but the demand path
    re-raises / leaves state. Reproduce with a CPU fixture (final flush
    only, first-token abort, stop token), fix so a final-forward fire is
    recorded and never raises, completed answer unchanged.
B2. **Receipt projection loses RT1 provenance.**
    `core/grm_admission.py:288–305` emits demotion/family/ranking
    fields; `core/grm_three_pass.py:450–465` does not pass the admission
    prefix and `:660–690` fixed projection drops them. Carry the fields
    through `route_receipt.v1` (additive keys only; existing keys and
    ROUTE_RECEIPT_INFO_PREFIXES untouched except for the additive
    prefix), pin with a receipt-shape test.
B3. **Capture provenance not durable on the ordinary manifest path.**
    `core/graft_arena.py:852–865` attaches capture metadata at graft
    top level; `core/graft_repository.py:4619–4635` omits it on
    serialize, `:4769–4784` on reload. Serialize and reload it
    (additive manifest key), pin with a save→reload round-trip test;
    old manifests without the key must load unchanged.
B5. **WC1's "resident seats" column counts grafts, not token seats.**
    `scripts/grm_wc1_results.py:112–114, 161–182`. Add the summed
    token-seat column beside the existing one (do not delete the old
    column; label both), regenerate WC1's table from the existing
    receipts, and state in the ledger what changes in the reading.
(B4, grounding ≠ relational correctness, is being tested by GRM-X2 in
another worktree; do not touch it.)
C1. **Serving/receipt edge-case fixtures**: a CPU fixture set covering
    flush, first-token abort, stop token, session resume, receipt
    projection, save/reload — the registered prediction from the Scout
    report is that B1 reproduces on the flush-only case and B2/B3 on
    the projections. Report which fixtures RED before your fixes and
    GREEN after.

## Done (verbatim)
1. Per item: file/lines changed, test name, RED-before/GREEN-after
   evidence.
2. The WC1 table before and after B5, with the reading.
3. Anything you found that the Scout report did not.
4. Prior art; deviations; RED; process safety; model id and effort.
