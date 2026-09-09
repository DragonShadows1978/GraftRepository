# GRM-X2 — Make memory carry relational witnesses (worktree wt-grm-x2, branch grm-x2)

**The idea (yours, Part D #2, 60 %):** store source-bound
entity–relation–value spans as admission and verification metadata
beside the native K/V; require each asserted binding in an answer to
have a WITNESS from one source version; represent joins explicitly
when an answer needs two sources. Motivation: Part B finding 4 —
today's "grounded" accepts one overlapping substantive word and checks
neither entity–value association nor negation.

**If it works:** high lexical overlap can no longer certify a wrong
relation. **Falsifier (as you wrote it):** CPU-check 40 role-swapped,
negated and scattered-token answers, then <= 0.5 GPU-h on 24 live
questions with witnesses hidden from the model prompt. **Kill:**
extraction misses ordinary paraphrases, or rejection destroys correct
answers.

## Mission
1. Witness extraction at deposit (additive, flag `GRM_X2_WITNESSES`,
   default OFF): entity–relation–value spans with source version id;
   a verification step at answer time that checks each asserted
   binding against a witness, with explicit join records for
   two-source answers. Witnesses are NEVER placed in the model prompt.
2. CPU battery first: 40 constructed answers — 10 role-swapped, 10
   negated, 10 scattered-token (all words present, wrong relation),
   10 correct paraphrases — against the existing grounding guard
   (`graft_arena.py` ~2725–2743 + `grm_text_norm.py`) and against the
   witness check. Report both confusion matrices.
3. GPU: 24 live EB1 questions (12 correct-mount, 6 plausible-decoy
   mount, 6 unavailable) with arms A today's grounding, B witness
   check gating the answer (reject → structural abstention). Score
   exact answer, false answer, abstention correctness, and how many
   correct answers B rejects.
4. Lead predictions: P1 the existing guard accepts >= 60 % of the
   role-swapped and negated constructions; P2 the witness check
   rejects >= 90 % of them while accepting >= 90 % of correct
   paraphrases; P3 on the live set B's false-answer rate on decoys is
   at most half of A's and it rejects no more than 1 of 12 correct
   answers. Register yours beside them.

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
