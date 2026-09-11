# GRM-P1 — the product surface: `scripts/grm_chat.py` (lead order, 2026-09-10)

Seat: Opus 5 (opus-max). YOUR WRITABLE TARGET is the git worktree
`/mnt/ForgeRealm/wt/grm-p1` (branch `grm-p1`, forked from `grm-merge`).
Edits under `scripts/`, `tests/`, `docs/`, `artifacts/grm_p1/` are
AUTHORIZED; `core/` is READ-ONLY (STOP with a receipt if the surface
needs a core change). Read-only: `/mnt/ForgeRealm/GraftRepository`, all
other `/mnt/ForgeRealm/wt/grm-*`, `/mnt/Shared/LEAD_TODO_2026-09-10.md`,
`/mnt/Shared/GRM_LT1_200Turn_Conversation_Result_2026-09-09.md`,
`docs/GRM_Methodology.md`, `docs/GRM_Primer*.md`. No git. No subagents.
Foreground only, Bash calls < 10 min, no background waits. NEVER kill
or signal a process you did not start. No GPU: the fake-model smoke
runs on CPU; a 10-turn GPU smoke is REGISTERED for the lead (≤ 0.2
GPU-h).

## What this is
David's spec (2026-09-02): the chat log is NEVER in the model's context;
every recall goes through GRM; the ephemeral boat (EB1) is the
production frame. The batteries (C2/C7/LT1) prove it on frozen
fixtures; nothing ships that a person can sit down at. This order
ships that thing — an END-TO-END RUNNABLE deliverable, not gates plus
screenshots.

## Mission
1. `scripts/grm_chat.py`: an interactive session on GPT-OSS-20B through
   the SAME production ladder the batteries use
   (`scripts/grm_e2e_session.py` path — reuse, do not fork a new
   serving path), EB1 frame, one repository directory per session
   (`--repo DIR`, created or resumed; `--resume` restores after a
   restart exactly as the batteries' restart cells do). Each user turn
   = the question + mounted grafts only (prove with a receipt that no
   prior turn text is fed live except the current turn); the turn's
   text is deposited per EB1 after the answer. Per-turn route receipt
   (`grm.route_receipt.v1`) appended to `REPO/session_ledger.jsonl`.
   Commands: `/status` (seats used / width / mounted ids / rule /
   flags), `/recap` (the LT1-style five-decision recap over memory),
   `/restart` (close + reopen the repo in-process), `/quit`.
   Batch mode `--transcript FILE` (one user turn per line) for gates.
2. `GRM_PROFILE=eb1_c2` — ONE switch that selects the C2 profile
   registry entry (width 96, capture pin, seat-near-live, RT1,
   `GRM_ADMISSION_RULE=margin_first`; add `GRM_ALIAS_FOLD_MERGE` as an
   optional named flag the lead can pin when it exists on the merged
   tree). Shipped defaults stay the defaults; unset = today's
   behaviour. Print the resolved flag set at start.
3. Gates: fake-model transcript smoke (20 turns: 6 facts, 2
   corrections, 1 alias, recalls at 5/10/15 turns back, one /restart,
   one /recap) with the value-span scorer — target on the fake model is
   PLUMBING (mounts happen, receipts written, restart retains, no live
   history leak), not answer quality; leak test (assert the live prompt
   never contains a prior turn's text); `--help` works; `python3 -m
   pytest -q tests/test_grm_chat*.py` verbatim last.
4. Registered GPU smoke (lead-run, ≤ 0.2 GPU-h): the same 20-turn
   transcript on GPT-OSS-20B under `GRM_PROFILE=eb1_c2`, leased; expected
   recall ≥ 3/4 on the fresh facts (LT1 fresh rate), corrections and
   alias reported not gated. `artifacts/grm_p1/lead_commands.txt`.
5. `docs/GRM_CHAT.md`: how to run it (one command), what the switch
   does, what the receipts mean, known residuals (corrections/aliases
   per LT1) — short, honest.

## Done (verbatim)
1. Files (paths + lines), the exact command a user runs, the resolved
   flag set printout, the leak-test assertion.
2. Fake smoke receipt lines; GPU smoke registration path + sha + lead
   commands.
3. Prior art (say what is reused from the ladder / EB1 and what is
   new), deviations, RED items, process-safety statement, model id and
   effort. The pytest line LAST.
