# GRM-C7 amendment 6 — r3 without FIX-7: fix the fixture lineage and the prompt, register and run; alias resolution becomes a design memo (worktree wt/grm-c7, branch grm-c7)

Your amendment-5 stop is accepted for FIX-7: L2 records no alias→base
edge and the two payloads (110 tokens) cannot co-mount in a 96-seat
arena. The stop clause applied to FIX-7 only; parts 1 and 3 of that
order stand. Same rules (no GPU, no git, no subagents, foreground,
never kill anything). Effort: high.

## Mission
1. **Fixture + prompt (harness), exactly as amendment 5 §1:** turn 2
   registered as a real correction (supersede / correction_command as
   production does), every other turn byte-identical, fixture sha
   re-registered; the abstention clause dropped from the probe prompt
   to match the production prompt shape (quote both); scorer casefolds
   the abstention comparison for the unanswerable controls.
2. **r3 registration and commands:** 39 cells, arm A (C2 profile),
   FIX-3/4/5 ON, no FIX-7; alias probes stay registered and are
   EXPECTED to fail (that is the residual being measured, not hidden);
   cap 2.2 GPU-h; receipts under `r3/`; oracle rows with the plain
   prompt; `lead_commands_r3.txt` executable, resumable, free-space
   preflight ≥ 20 GB. Registered prediction: fresh ≥ 10/12, folded
   ≥ 10/15, corrections ≥ 6/13, aliases ≤ 2/12, residency bounded,
   restarts retained.
3. **Alias design memo (`artifacts/grm_c7/r3/ALIAS_DESIGN_OPTIONS.md`,
   for David):** the three candidate mechanisms with what each costs
   and touches: (a) two-hop read — mount the edge, read the base
   identifier, re-route within the same turn (the demand loop's
   machinery; latency of one extra re-prefill); (b) fold-merge — the
   librarian's fold (FIX-5 now works) merges an alias-edge node with
   its base into one digest carrying both names and the value, so a
   single mount answers; (c) alias-in-base rewrite at deposit — when an
   alias turn is deposited, the base node's text is amended to carry
   the alias as an identifier (changes stored text; supersession
   semantics). Say which the receipts favour and why; no implementation.

## Done (verbatim)
1. Fixture diff + new sha; prompt before/after; scorer note; tests.
2. r3 registration path + sha; exact lead command; the memo path.
3. Prior art; deviations; RED; process safety; model id and effort.
