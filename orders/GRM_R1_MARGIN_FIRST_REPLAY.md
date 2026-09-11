# GRM-R1 — margin-first regression replay worker (lead order, 2026-09-10)

Seat: Opus 5 (opus-max). YOUR WRITABLE TARGET is the git worktree
`/mnt/ForgeRealm/wt/grm-r1` (branch `grm-r1`, forked from `grm-merge`
= lc1-wip + FIX-1..6/8 merged). Edits, builds and CPU runs inside it are
AUTHORIZED. Read-only: `/mnt/ForgeRealm/GraftRepository` (canonical),
every other `/mnt/ForgeRealm/wt/grm-*` worktree (receipts you may copy
FROM), `/mnt/Shared/LEAD_TODO_2026-09-10.md` (the plan). No git (the
lead commits). No subagents. Foreground only, every Bash call < 10 min,
no background waits, no monitor idling. NEVER kill or signal any
process you did not start; multiple projects share this machine. Do NOT
run GPT-OSS-20B or touch the GPU: build, gate on the CPU fake model,
deliver a blocked report + exact lead commands; the lead runs GPU under
the lease (`/tmp/forge-gpu.lock`, 285 s worker / 590 s outer cells).

## Context
SCOUT-FIX-6 (`core/grm_admission.py`, flag `GRM_ADMISSION_RULE=
margin_first`, default unset = today's all-tokens-bind rule, C2 132/132
recorded plans byte-identical) registered a regression set but did NOT
run it: `artifacts/grm_scout_fix6/c2_replay_cells.json` (31 execution
cells over 9 distinct C2 question ids: plan OFF vs plan ON, e.g. [6,7]
→ [6,7,4]), `c2_states/`, `C2_QUESTION_IDS.md`, `replay_amendment.json`,
`c2_unresolved.json` (20 executions without exact margins — keep them
unresolved, do not impute). Canonical copies live in
`/mnt/ForgeRealm/wt/grm-lt1/artifacts/grm_scout_fix6/`; the C2
checkpoints the cells point at live under
`/mnt/ForgeRealm/wt/grm-c2/artifacts/grm_c2/epochs/…` (sha-pinned in
each cell). The FIX-6 report says: "the lead must import these explicit
recorded paths into its replay registration and pin the flag after
`environment(flags)`." That is this order.

Plan (immutable, on Shared): registered prediction ≤ 2 of 31
executions change their ANSWER; 0 correct→wrong on supersession.
Verdict rule for adopting margin_first as the profile default:
correct→wrong = 0 on sup and ≤ 1 elsewhere. Budget ≤ 1.0 GPU-h.

## Mission
1. `scripts/grm_r1_replay.py`: a resumable, leased, create-only-receipt
   worker (same shape as the C2/C7/LT1 workers — reuse their lease,
   receipt and sha-binding code; do not fork a third copy of the lease
   logic if a shared helper exists) that, per cell: loads the recorded
   checkpoint (verify every file hash in `checkpoint_files` first; a
   mismatch is a RED receipt, not a retry), runs the recorded question
   through the PRODUCTION ladder (`scripts/grm_e2e_session.py` path the
   batteries use) twice from the same state — rule OFF (today's) and
   rule ON (`margin_first`) — with the C2 frame flags of that cell's
   side (profile vs defaults exactly as recorded), and records: rank
   plan, mounts, seats, answer text, route receipt, wall. Score both
   answers with the value-span rule the LT1/C5 scorers use against the
   cell's recorded expected value; classify each execution
   unchanged-correct / unchanged-wrong / correct→wrong / wrong→correct
   / abstain transitions. The OFF plan must reproduce the recorded
   plan byte-for-byte (parity barrier, like RD1 A0); if it does not,
   the cell is RED and the run stops.
2. Registration JSON (immutable; amendments separate): cells, budget
   (≤ 1.0 GPU-h, per-cell estimate from the C2 receipts), sha-bound
   sources (core files + worker + cells file), prediction and verdict
   rule verbatim from the plan.
3. CPU gates: fake-model run of all 31 cells (`--fake`), hash-mismatch
   RED fixture, parity-barrier RED fixture, scorer fixtures (exact /
   value-span / abstain / wrong), `--dry-run` enumerating cells with
   estimates, resume-after-interrupt test. Existing suites must stay
   green: run `python3 -m pytest -q tests/test_grm_admission.py
   tests/test_grm_scout_fix4.py tests/test_grm_scout_fix6*.py
   tests/test_grm_r1*.py` verbatim last.
4. `artifacts/grm_r1/lead_commands.txt` (preflight / dry-run / resume /
   summary), `blocked_report.json`, `LEDGER.md`, `REPORT.md`.

## Done (verbatim in your final message)
1. Files changed (paths + line ranges); worker entry points; how the
   flag is pinned per arm and proof the two arms share one state.
2. Registration path + sha; per-cell estimates; total budget.
3. CPU gate lines verbatim (the pytest line LAST).
4. Prior art (paper/system/year; taken vs own; or "none known to me");
   deviations from this order; RED items; process-safety statement
   (nothing killed, no GPU touched); model id and reasoning effort.
