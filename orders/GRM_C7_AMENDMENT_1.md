# GRM-C7 amendment 1 (lead, 2026-09-09) — bind the SCOUT-FIX-2 source delta before the campaign runs

The lead cherry-picked SCOUT-FIX-2 (split children inherit the parent's
capture provenance, `core/graft_repository.py` ~line 1005, commit
6b2d3c8 on grm-c2) onto this branch (see `git log -1`), because C7's
restart and paging grading depends on capture provenance surviving on
split children (C2 amendment 2 found it missing). Your registration
records the source state at registration time; the campaign has NOT
started. Same worktree, same rules (no git, no subagents, no GPU,
foreground, never kill anything). Effort: high.

## Mission
1. Sha-bound amendment JSON (this order + registration + the old and
   new `core/graft_repository.py` shas) that records the source delta
   and rules which registered items it touches: fixtures/probes/oracle
   unchanged; the split-children provenance now present, so the
   "restart retains metadata" grade applies to children too (state
   whether that changes any registered acceptance wording; it must not
   weaken any).
2. The runner/receipts read the amendment (a receipt must carry the
   amended source sha); forged / stale amendment refused (tests);
   `--dry-run` still enumerates the 39 cells; `lead_commands.txt`
   byte-unchanged (a chain is queued on it).
3. Re-run the C7 CPU gates plus `tests/test_grm_scout_fix2_capture.py`
   on this branch; paste counts.

## Done (verbatim)
1. Amendment path + sha; the two source shas; files/lines; tests.
2. Confirmation that `lead_commands.txt` is byte-unchanged.
3. Prior art; deviations; RED; process safety; model id and effort.
