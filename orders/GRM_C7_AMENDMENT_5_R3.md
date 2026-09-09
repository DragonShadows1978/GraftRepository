# GRM-C7 amendment 5 — r3: fix the fixture's correction lineage, drop the abstention clause, add flag-gated alias-base co-mount (SCOUT-FIX-7), run 300 turns again (worktree wt/grm-c7, branch grm-c7)

David (2026-09-09): "move forward with the best options to resolve the
issues." RD1/RD2 findings (wt/grm-rd1, `artifacts/grm_rd1/`, `artifacts/grm_rd2/REPORT.md`):
1. The C7 probe prompt's clause "if unspecified, reply unknown" caused
   the reader abstentions: without it, oracle 16/16, memory fresh 4/4
   and folded 9/10. Production prompts (`scripts/grm_e2e_session.py`)
   carry no such clause.
2. Corrections: the fixture registered its turn-2 correction as an
   ordinary fact (`scripts/grm_c7_register.py:25-40`, no
   `kind='supersede'`/`correction_command`), so the original Mica node
   stayed active, L2 has no edge from Onyx to Mica, A-DEC ranks Mica
   first and RS3 seats it nearest the question → all 8 corrections
   served the stale value with BOTH nodes mounted.
3. Aliases: the alias-edge node ("C7-Signal-0 is an alias for
   C7-AliasBase-0") is mounted without its base node; GPT-OSS then
   refuses ("I'm sorry, but I can't provide that"). Retrieval never
   follows the alias edge to the base value.
Same rules as FIX-1..6 (minimal core diff, RED/GREEN, flag-gated
behaviour change, no default flip, no GPU, no git, no subagents,
foreground, never kill anything). Effort: high.

## Mission
1. **Fixture (harness):** register turn 2 as a real correction
   (supersede / correction_command as the production session does),
   keep every other turn byte-identical, re-register the fixture sha;
   drop the abstention clause from the probe prompt so it matches the
   production prompt shape (quote both). Scorer: keep the frozen
   exact rule but casefold the abstention comparison for the
   unanswerable controls (amendment 4's contract note).
2. **SCOUT-FIX-7 (core, flag-gated `GRM_ALIAS_FOLLOW=1`, default
   OFF):** when the identifier binds an alias-edge node, the mount plan
   also mounts the alias's base node (follow the alias edge L2 already
   records; if it does not record one, say so and STOP). Tests: the
   RD2 alias rows on a CPU replay mount both nodes with the flag ON and
   only the edge with it OFF (byte-identical); C2 supersession fixtures
   unchanged with the flag OFF.
3. **r3 registration:** 39 cells as r2, arm A only (the C2 profile),
   FIX-3/4/5 + FIX-7 ON, cap 2.2 GPU-h (lead authorization; total C7
   ≈ 6 GPU-h), receipts under `r3/`; oracle rows with the plain
   prompt; `lead_commands_r3.txt` executable, resumable, free-space
   preflight. Registered prediction: fresh ≥ 10/12, folded ≥ 10/15,
   corrections ≥ 6/13, aliases ≥ 6/12, residency bounded, restarts
   retained.

## Done (verbatim)
1. Fixture diff (turn 2, prompt) + new sha; FIX-7 file/lines, flag,
   tests with RED/GREEN evidence; scorer note.
2. r3 registration path + sha; exact lead command.
3. Prior art; deviations; RED; process safety; model id and effort.
