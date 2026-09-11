# GRM-A1 — alias resolution by fold-merge, flag-gated (lead order, 2026-09-10)

Seat: Opus 5 (opus-max). YOUR WRITABLE TARGET is the git worktree
`/mnt/ForgeRealm/wt/grm-a1` (branch `grm-a1`, forked from `grm-merge` =
lc1-wip + FIX-1..6/8). Edits, builds and CPU runs inside it are
AUTHORIZED, including `core/` under the flag below. Read-only:
`/mnt/ForgeRealm/GraftRepository`, every other `/mnt/ForgeRealm/wt/grm-*`
worktree, `/mnt/Shared/LEAD_TODO_2026-09-10.md`. No git. No subagents.
Foreground only, every Bash call < 10 min, no background waits. NEVER
kill or signal a process you did not start. Do NOT run GPT-OSS-20B or
touch the GPU: CPU fake-model gates + a registered GPU contrast the lead
runs under the lease.

## Context (receipts)
- RD2 (`/mnt/ForgeRealm/wt/grm-rd1/artifacts/grm_rd2/REPORT.md`): 8/8
  C7 alias probes mounted the alias-EDGE node without its base; L2
  records no alias→base edge (empty `links`/`sources`/`supersedes`);
  GPT-OSS refuses. Edge + base payloads = 49 + 61 = 110 tokens > width 96.
- FIX-7 (co-mount) was STOPPED as impossible; the design memo
  `/mnt/ForgeRealm/wt/grm-c7/artifacts/grm_c7/r3/ALIAS_DESIGN_OPTIONS.md`
  recommends option (b) FOLD-MERGE first; two-hop read is the fallback
  and is NOT built in this order.
- FIX-5 (`core/graft_arena.py` consolidation prompt enumerates the N
  source facts; budget max(120, 24N)): 13/13 folds accepted on GPU.
- FIX-8 (`core/grm_text_norm.py`): one identifier normalization for
  routing/admission/lexical scan; digest identifier sets from
  normalized text.
- LT1 (200-turn natural conversation): aliases 5/10 exact under the
  profile; C7 r3 aliases 0/10.

## Mission
1. Flag `GRM_ALIAS_FOLD_MERGE` (default OFF; OFF = byte-identical
   behaviour, prove it). When ON: at alias deposit (and on the
   librarian's fold pass for already-stored aliases), pair the alias
   edge with its CURRENT base (resolve through supersession; if the
   base has been corrected, the current version) as ONE fold job whose
   FIX-5 enumeration lists every fact of the base plus the alias
   relation; the digest names alias, base and value together. Lineage:
   the digest supersedes the edge; the base is retired ONLY if the
   digest's coverage check proves every base fact survived, otherwise
   the base stays active and the edge alone is superseded (record
   which). Digest identifier set (FIX-8 normalized) must contain both
   names. Tokenize the digest against the arena width (96) BEFORE
   claiming single-mount fit; over-width digests go through the
   existing width guard/split path and the receipt says so.
2. Revision semantics with tests: a later correction of the base value
   supersedes the merged digest (the stale merged value must not be
   served); missing base → no merge, edge stays, receipt says
   `alias_base_missing`; alias cycle / alias-of-alias → resolve to the
   terminal base or refuse with a receipt; alias reassignment retires
   the old digest.
3. Fixtures RED→GREEN on the fake model: the 8 RD2 alias rows (copy
   their recorded texts/ids), the 10 C7 r3 alias probes, and LT1's 10
   alias questions (fixture at
   `/mnt/ForgeRealm/wt/grm-lt1/artifacts/grm_lt1/` — read the
   registration for the alias turn ids). Controls byte-identical with
   the flag OFF and for non-alias folds with it ON. Existing suites
   green: `python3 -m pytest -q tests/test_grm_scout_fix3*.py
   tests/test_grm_scout_fix5*.py tests/test_grm_scout_fix8*.py
   tests/test_grm_admission.py tests/test_grm_a1*.py` verbatim last.
4. GPU contrast registration (lead-run, ≤ 0.5 GPU-h): replay the 8
   RD2 alias rows + the 10 C7 r3 alias probes from the C7 r3
   checkpoints (`/mnt/ForgeRealm/wt/grm-c7/artifacts/grm_c7/r3/…`, sha
   pinned) with the merge applied to the loaded state, reads under the
   plain production prompt; registered prediction: aliases ≥ 5/8 exact
   from a single digest mount, 0 controls broken. Worker resumable,
   leased, create-only receipts; `artifacts/grm_a1/lead_commands.txt`.

## Done (verbatim in your final message)
1. Core diff (files + line ranges), the flag, the OFF byte-identity
   proof, the lineage rules as implemented.
2. Fixture RED-before/GREEN-after evidence per group (RD2 8, C7 10,
   LT1 10) and the control lines; width receipts for every digest
   (tokens vs 96).
3. GPU contrast registration path + sha + budget; exact lead commands.
4. Prior art (RAPTOR 2024, Press 2022, materialized-view maintenance
   — say what was taken vs yours), deviations, RED items, process-
   safety statement, model id and reasoning effort. The pytest line LAST.
