# GRM-C7 amendment 7 (lead, 2026-09-09) — r3 diagnosis: admission refusals returned, folded digests are ranked but not identified, and the plain prompt fabricates on controls

Lead-run r3 (`artifacts/grm_c7/r3/cells/`, 39/39 COMPLETE, folds 13/16
accepted, max seats 94, restarts retained). Answerable probes (expected
≠ UNKNOWN), memory exact / wrong / abstain / n:
fresh 2/0/8/10 · alias 0/6/4/10 · correction **9/1/0/10** · folded 0/4/6/10.
Controls (expected UNKNOWN): **10/20 answered instead of abstained**
(fresh 1/5, alias 5/5, correction 2/5, folded 2/5), e.g. "The
inspection password for C7-Fresh-2 is **Flint-511**."
Corrections are fixed (0 → 9/10: the lineage fix worked). But:
1. **Fresh 2/10 with 8 abstentions at 0 seats** — admission REFUSED
   ("Not in memory: no stored record matches …") where r2 mounted the
   same probes (r2: 30/30 mounted-but-unread; RD1 A2 then read fresh
   4/4). Something in r3 (the turn-2 supersession change, FIX-5 folds
   retiring sources, the prompt without the clause, FIX-4) changed
   admission. Diagnose per probe id against r2's receipts: mounted
   ids, identified candidates, ranking, retired flags of the source
   node at that turn.
2. **Folded 0/10**: the digest node ranks first (e.g. ranking [10, …]
   for C7-Archive-0) but `identified_candidates` is empty or names a
   different node; i.e. the identifier scan does not bind
   `C7-Archive-0` to the digest that contains it. Quote the digest
   text and the identifier tokens; say whether the digest's
   identifier set is built from its own text (FIX-5 output) or
   inherited, and whether the case/dash normalization differs.
3. **Controls fabricate** without the abstention clause. Register a
   middle prompt for the memory rows — instruct answering from the
   mounted records and saying unknown only when the records lack it,
   in the production prompt's shape — and a replay contrast from the
   r3 checkpoints on the 20 controls + the 10 fresh + the 10 folded
   rows (≤ 0.5 GPU-h) with the frozen scorer; predictions registered.
Same worktree, same rules (no core edits without STOP-and-report; no
GPU; no git; no subagents; foreground; never kill anything). Effort:
high.

## Done (verbatim)
1. Per-probe table for fresh and folded (r2 vs r3: mounted ids,
   identified, ranking, retired); the mechanism for each; harness/core
   ruling with file/lines.
2. Middle-prompt contrast registration path + sha; exact lead command.
3. Prior art; deviations; RED; process safety; model id and effort.
