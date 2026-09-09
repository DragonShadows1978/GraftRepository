# GRM-SCOUT-FIX-8 amendment 1 (lead, 2026-09-09) — batch F5 hit a cudaMalloc OOM; re-arm it once with a device-memory probe

Lead-run of `lead_commands_fix8.txt`: the amendment-7 middle-prompt
contrast COMPLETE (80 row receipts, A0 parity true), FIX-8 replay
batches F1–F4 COMPLETE, **F5 FAILED `cudaMalloc failed: out of memory`**;
the re-run then refused with `FAILED_CAMPAIGN_STOP` (registered stop).
The card showed 275 MiB used and a free lock immediately after; the
OOM is most likely a transient from another process on the shared
card (David runs other sessions) or the batch's own peak. Register a
one-time re-arm of F5 (create-only successor receipt; the RED stays),
and add a device-memory probe before each lease that records
`memory.used` and refuses to start a batch if more than 1,000 MiB is
already in use by other processes (record the pids). If F5's own peak
is the cause (say how you can tell from its receipt/log), split F5
into two batches under the same registration instead. Same worktree,
same rules (no GPU, no git, no subagents, foreground, never kill
anything). Effort: high.

## Done (verbatim)
1. F5 receipt quoted (rows attempted, memory at failure if recorded);
   ruling transient vs own-peak; amendment path + sha; tests.
2. Exact lead command (`lead_commands_fix8_resume.txt`).
3. Prior art; deviations; RED; process safety; model id and effort.
