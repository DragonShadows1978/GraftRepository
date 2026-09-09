# GRM-X3 — Lesion the memory to measure whether it mattered (worktree wt-grm-x3, branch grm-x3)

**The idea (yours, Part D #5, 55 %):** fork a captured live attempt
and remove or exchange ONE mount while holding every other state
constant; use the change in the one-token answer distribution as a
CAUSAL diagnostic beside attention mass, instead of assuming that
attention paid to a memory means the memory supplied the fact. This
extends the house S3/S4 arbiter and the D-NGH observer; it is
diagnostic-only at first (no serving change).

**If it works:** wrong confident reads are distinguishable from unused
correct memories. **Falsifier (as you wrote it):** 20 snapshots (correct
mount / plausible decoy / refusal), plus SHAM interventions (remove a
mount that cannot matter) and SAME-PAYLOAD interventions (replace a
mount by an identical copy); <= 0.5 GPU-h. **Kill:** harmless cache
surgery moves logits as much as factual interventions, or the effect
does not predict exact-answer errors.

## Mission
1. Snapshot forking: from a captured live attempt (the RS/EB1 capture
   machinery), produce forks with one mount removed, one mount swapped
   for a decoy, one mount swapped for an identical copy (same-payload),
   and one irrelevant mount removed (sham); all other state bitwise
   identical — pin that with a test. Diagnostic scripts only; no
   serving-path change; flag not needed.
2. Fixtures: 20 snapshots over the existing batteries: 8 correct-mount
   answers, 6 plausible-decoy answers (the wrong-memory cases D-NGH is
   weakest on), 6 refusals. Freeze with shas.
3. Measure per fork: KL and top-1 change of the one-token answer
   distribution vs the unforked attempt; attention mass on the
   lesioned mount (the existing telemetry); exact-answer outcome.
   Report the four-way table: (mass high/low) × (lesion effect
   high/low) with the exact-answer error rate in each cell.
4. Lead predictions: P1 same-payload and sham forks move KL by less
   than 0.05 nats on >= 18 of 20 snapshots; P2 removing the answer's
   true source moves KL by > 1 nat on >= 6 of 8 correct-mount cases;
   P3 on the 6 decoy cases, attention mass is high on the decoy in
   >= 4 while the lesion effect predicts the exact-answer error better
   than mass does (state the scoring rule before running). Register
   yours beside them.

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
