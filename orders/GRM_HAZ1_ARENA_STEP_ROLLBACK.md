# ORDER GRM-HAZ1 — fix the ArenaCache.step shallow-rollback production hazard

WRITABLE TARGET `/mnt/ForgeRealm/GraftRepository`; standing rules
(no git/subagents/network; CPU self-runs; GPU legs self-lease).

Context: ADM1.1 named this while fixing its own harness: production
`ArenaCache.step` performs a SHALLOW rollback; a consuming backend
(GPT-OSS consumes cache lists in place, core/gpt_oss20b_tc.py:1255)
plus a non-clean descent retry can reach it — same defect class as
the ADM harness aliasing (corrupted cache surviving a rollback), in
production code. David delegated: fix now.

Work:
1. Convict precisely: file:line of the shallow rollback; enumerate
   the reachable paths (which backends consume in place; which retry/
   descent flows roll back non-cleanly).
2. Minimal fix mirroring the ADM1.1 pattern (copy the outer list;
   share immutable tensors). No semantics changes beyond the hazard.
3. Regression test pinning the consuming-backend + retry path (the
   ADM1.1 test is the template; this one must exercise PRODUCTION
   ArenaCache.step).
4. Gates: full GRM suites green; registered default-transcript
   baselines byte-identical (the fix must change behavior ONLY on
   the corruption path, which currently fails — nothing certified
   may move); new regression green.

Done: conviction file:line + reachability list; diff summary; gate
results with numbers; files; anything not done.
