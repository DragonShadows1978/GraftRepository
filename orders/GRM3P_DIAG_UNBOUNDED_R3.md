# ORDER GRM3P-DIAG r3 — finish the mechanism hunt (Opus 4.8 seat)

Base order: orders/GRM3P_DIAG_UNBOUNDED.md. Frozen pre-registration +
results so far: docs/GRM3P_DIAG_UNBOUNDED_REPORT.md — READ BOTH FIRST.
Append results to that report; never edit prior sections.

State on entry (all lead-committed):
- ARM B reproduced: F-FULL single 7/9, misses turns 5+13, sha 06ef1fd2…
- C1 --topk 1: 7/9 byte-identical — H-COMOUNT REFUTED (rank-1-alone
  still answers the cypher value). H-FIT/H-ROUTE dead at this
  granularity too (identical transcript).
- C2-TOPK2: identical. C2-LIVE1: turn 5 flips to PASS when the cypher
  turn leaves the live window, but the control is CONFOUNDED (route
  eligibility changed globally; 8 other probes broke).
- H-REHYDRATE untested.

Remaining registered arms (frozen decision rules in the report):
1. **C3-EARLY-REHYDRATE**: F-FULL single, defaults except
   --restart-after 5. Conviction requires 9/9 (or the two misses
   flipping) WITH invariant rankings/source-rank/mount ids per the
   frozen rule, plus the storage_bits=8 node receipt.
2. **CONFIRM**: repeat the first decisive arm once; require identical
   scorecard + transcript sha.
3. **ARM D witness** (only if C3+CONFIRM leave ambiguity): per-mounted-
   node vs live-window attention-mass readout on turns 5+13 — additive
   instrumentation, default-off, default smoke byte-identity re-verified
   (68da84b8…). Given C1's result, the sharpest witness question is:
   during the wrong answer, is attention mass on the LIVE-WINDOW cypher
   turn tokens rather than the mounted orion node?
4. If still ambiguous: verdict is "not identified under [instruments]",
   name the un-searched space, stop.

Rails: writable target /home/vader/GraftRepository-three-pass; production
+ merge-train worktrees READ-ONLY; no git; no subagents; flock -w 7200
/tmp/forge-gpu.lock (another seat shares the GPU); frames pinned; RED is
a result; no post-hoc rule changes; lead verifies against disk. Do NOT
discuss host device configuration in the report — irrelevant to the
mechanism and out of scope.

## Done
Verbatim: each arm's frame + scorecard line + session dir, the
conviction/ambiguity statement per the frozen rules, and residuals.
