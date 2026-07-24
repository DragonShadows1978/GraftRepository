# ORDER GRM3P-DIAG-DOSE — budget dose-response + payload diff (Opus 4.8 seat)

WRITABLE TARGET /home/vader/GraftRepository-three-pass. Production +
merge-train worktrees READ-ONLY. No git. No subagents. GPU under
`flock -w 7200 /tmp/forge-gpu.lock`. Read
docs/GRM3P_DIAG_UNBOUNDED_REPORT.md fully first; append results there,
never edit prior sections. Diagnostic only — NO fixes.

## Registered prediction (frozen before any run)

Under H-REHYDRATE, per-probe recall is predicted by the source node's
lifecycle, NOT the budget size: a probe passes iff its source node was
packed→evicted→rehydrated at least once before the probe turn
("washed"). A miss on a washed node, or a pass on a never-washed node
at any budget, counts AGAINST H-REHYDRATE and must be reported as such.
(Baseline anchors already on disk: unbounded = 7/9 with misses t5/t13
= never-washed sources; 4MB = 9/9 all-washed.)

## ARM E — dose-response sweep (existing CLI only)

Frames: `--mode full --turn-pipeline single`, env
`GRM_GQA_CUDA_ROUTE=1 GRM_GRAFT_STORAGE_BITS=8`, all defaults except
`--vram-budget-mb {8, 16, 32}` — three sessions, one variable. For each:
probe scorecard line verbatim + a per-probe table: turn, fact, pass,
source node id, washed-before-probe (from step_io/page-in receipts and
pager state), source_rank. Then the cross-budget correlation table:
washed vs pass over all 27 probe rows (+ the 18 anchor rows from the
on-disk unbounded and 4MB sessions). State the verdict per the frozen
prediction — no re-interpretation after seeing rows.

## ARM F — payload diff (CPU-side, no GPU needed beyond one load)

For the turn-5 source node (node 0) in the unbounded ARM B session and
the C3 session (both on disk): compare the device-resident deposit
payload vs the rehydrated payload — dequantized value diff (max/mean
per layer) AND structural/metadata diff (shapes, dtypes, un-RoPE state,
positions, group/scale layout, any seating metadata). Classify:
NUMERIC-ONLY (quant-noise scale), STRUCTURAL, or BOTH — with the
receipts. If the on-disk artifacts are insufficient to reconstruct the
device-resident payload, say so and list exactly what a live capture
would need (do not build it).

## Honesty

Frames pinned; RED is a result; no post-hoc prediction changes; the
falsification arm is as reportable as the confirmation. Lead verifies
against disk.

## Done
Verbatim: three scorecard lines, the washed-vs-pass correlation table,
ARM F classification + diff numbers, sessions dirs, verdict per frozen
prediction, residuals.
