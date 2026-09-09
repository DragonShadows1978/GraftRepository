# GRM-C4 amendment 7 (lead, 2026-09-09) — the A6 runner crashes before its first lease (`KeyError: 'status'`)

Lead-run of `lead_commands_a6.txt`: exited after 94 s with zero
leases: `scripts/grm_c4_skip_a6.py:163 main` →
`grm_c4_split_a5.py:306 worker` → `:299 admission` → `:229 accounting`
→ `KeyError: 'status'` (a receipt in the accounting scan lacks
`status`; likely an `.attempt.json` or a partial file under
`lead_a2/runs` or `runs/`). Fix the accounting to read only completed
receipts (and to say which file it skipped and why), pin it with a
test that reproduces the exact on-disk layout that crashed, keep the
skip list / cap / projection of amendment 6, and regenerate
`lead_commands_a7.txt` (same content as a6). No GPU was used; nothing
to charge. Same worktree, same rules (no git, no subagents, no GPU,
foreground, never kill anything). Effort: high.

## Done (verbatim)
1. Root cause (the offending file quoted); fix file/lines; test name;
   projection unchanged.
2. Exact lead commands (`lead_commands_a7.txt`).
3. Prior art; deviations; RED; process safety; model id and effort.
