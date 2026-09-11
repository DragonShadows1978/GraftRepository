# GRM-LT1 amendment 4 (lead, 2026-09-09) — the run crashed in core at cell A-033-040: `graft 25 has no native_node_id`

Lead-run under margin_first: cells A/B 001-032 COMPLETE (8 cells),
then **A-033-040 RED `WORKER_EXIT_1`**: `worker.log` ends in
`core/graft_arena.py:4443 _attempt → :2373 _commit_native_mount →
:2365 _native_mount_ids → RuntimeError: graft 25 has no native_node_id`
(receipt `artifacts/grm_lt1/amendment2/run_margin_first/cells/A-033-040/`).
Diagnose from the receipts and a CPU replay of the cell's checkpoint:
what graft 25 is (a FIX-2 split child? a node deposited on a path that
skips native sync? a node restored from checkpoint without its native
id?), why margin_first picked it when the identifier rule would not
(the ranking/plan in the route receipt), and whether the defect is
(a) the LT1 harness/checkpoint path, (b) FIX-6's picker admitting a
node the native store never registered, or (c) a core deposit/restore
gap that today's rule merely never reached. Fix (a)/(b) here with
RED-before/GREEN-after tests; STOP and report with file/lines for (c).
Then register the resume: cells 001-032 stay valid iff the fix does not
change their executed path (rule it); resume at A-033-040;
`lead_commands.txt --resume` executable. Same worktree, same rules (no
GPU, no git, no subagents, foreground, never kill anything). Effort:
high.

## Done (verbatim)
1. Diagnosis with quoted receipt fields; classification; fix
   file/lines + tests, or the core STOP report.
2. Resume registration path + sha; validity ruling; exact lead command.
3. Prior art; deviations; RED; process safety; model id and effort.
