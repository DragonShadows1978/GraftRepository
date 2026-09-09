# GRM-X1 — Give memories virtual addresses (worktree wt-grm-x1, branch grm-x1)

**The idea (yours, Part D #1, 75 %):** route identifier questions to a
stable entity–relation ADDRESS, resolve that address to the current
version's native graft pages, and treat an address absent from the
mounted set as a PAGE FAULT before generation — instead of letting
duplicate split-children compete in a global ranking. Topical
discovery stays on the existing three-channel router.

**If it works:** point recall is insensitive to split-family
multiplicity and many demand decisions need no attention telemetry.
**Falsifier (as you wrote it):** 24 held-out alias/correction queries
against 1–100 duplicated decoys, exact-oracle address metadata first;
<= 0.5 GPU-h. **Kill:** ranking dependence persists even with correct
addresses, or natural alias resolution erases the benefit.

## Mission
1. Define the address: (entity, relation) → current version → graft
   page ids, as an additive index beside the repository (flag
   `GRM_X1_ADDRESSES`, default OFF). Exact-oracle metadata first (the
   deposit path tags entity/relation from the fixture), natural
   alias resolution as a registered second arm only if the oracle arm
   is positive.
2. Fixtures: 24 held-out queries (12 alias, 12 correction) over the
   existing sup/census documents; decoy multiplicity {1, 10, 100}
   duplicated children per target family; freeze with shas.
3. Arms: A router-only (today's EB1); B addresses ON with exact
   oracle; C addresses ON + page-fault-before-generation (an absent
   address returns a structural abstention, not a guess). Score exact
   answer, false-answer rate, abstention correctness, demand fires,
   wall per turn. Same seeded runs for all arms.
4. Lead predictions: P1 A's exact-answer rate falls with decoy
   multiplicity (>= 20 points from 1 to 100); P2 B is flat within 5
   points across multiplicity; P3 C's false-answer rate at 100 decoys
   is below A's by >= half; P4 wall per turn for B/C is within 10 % of
   A. Register yours beside them.

## Common rules (binding, all three X orders)

Origin: `docs/GRM_SCOUT_2026-09-08.md` Part D (your own proposal); David
2026-09-08: "I am fine with testing and looking into all 3." Three
sibling orders run IN PARALLEL in separate worktrees (X1 addresses, X2
witnesses, X3 lesions); do not touch the others.

YOUR WRITABLE TARGET is the worktree you were dispatched into (branch
named in the header), forked from `lc1-wip`. Writable: `scripts/grm_x{N}_*`,
`tests/test_grm_x{N}_*`, `artifacts/grm_x{N}/`, `logs/`,
`docs/GRM_X{N}_LEDGER.md`, and — only where the order says so — an
ADDITIVE, flag-guarded, default-OFF module under `core/` that changes no
existing behaviour when its flag is unset (pin that with a test). The
production frame is EB1 (`GRM_EB1` order, `core/grm_frame.py`): the boat
is ephemeral, every recall is a repository graft, model GPT-OSS-20B,
arena width 96, seat-near-live + capture pin as measured (RS3/RS4), RT1
split-child routing. Reuse the existing batteries and fixtures
(`tests/`, `config/grm_live_registered_baselines.json`,
`config/grm_demand_registered.json`); freeze new fixtures under
`artifacts/grm_x{N}/fixtures/` with shas BEFORE any gate. Registration
first (`artifacts/grm_x{N}/registration.json`, IMMUTABLE; amendments
separate) with the lead predictions below and yours beside them.

Your sandbox has NO GPU: build, CPU-verify (unit tests, fixture
checks, a `--dry-run` that enumerates every GPU cell), deliver a leased
runner `scripts/grm_x{N}_lead_gpu.sh list|run CELL|resume|summary`
(`flock --wait` on `/tmp/forge-gpu.lock`, every job <= 285 s worker /
590 s outer, 30 s cooldown, create-only fingerprinted receipts, never
kills anything) plus a blocked-report and exact `lead_commands.txt`; the
lead runs the card (RTX 4070 SUPER 12 GB, shared with other projects;
the operator has absolute right of way). Total GPU budget for your
order: <= 0.75 GPU-hour. Evidence classes stated on every number
(unit test / kernel gate / E2E session receipt). Prior Art Directive
(HOUSE_RULES): annotate at code site + ledger + report, or "none
known" / "unverified — lead to check". NO git. NO subagents. NO
background waits (foreground, every call < 10 min). Never kill a
process you did not start. RED honesty; a failed falsifier is a
result. Registration immutable.

## Done (verbatim in your final message)
1. Frozen fixtures (paths, counts, shas); registration sha; both
   prediction sets.
2. What changed (files/lines); the default-OFF pin test name if a
   core module was added.
3. CPU gate results; GPU blocked-report with exact lead commands and
   wall estimates.
4. Prior art; deviations; residual risks; RED; process safety; model
   id and reasoning effort.
