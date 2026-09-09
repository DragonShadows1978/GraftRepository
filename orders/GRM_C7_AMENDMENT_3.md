# GRM-C7 amendment 3 (lead, 2026-09-09) — r2 mounts now, but the oracle path crashes on the real model at the first distance-30 probe; fix and resume

Lead-run r2 (FIX-3 source): cells A-001-008, A-009-016, A-017-023
COMPLETE — and this time the harness mounts (residency `summed_token_seats`
46, mounted turns 8/8 in cell 2) and memory answers appear
(`c7_fresh_0_d005` → `Basalt-811`, expected `Basalt-811`). Then
**A-024-031 FAILED**: `scripts/grm_c7_run.py:137 oracle`:
`shifts = [layer.self_attn.live_shift for layer in arena.m.layers]` →
`AttributeError: 'GptOssAttentionTC' object has no attribute 'live_shift'`
(worker.log). Your r2 harness fix for the oracle's layer-position setup
used an attribute the real attention module does not have; the CPU
fake evidently does. Also visible in the completed cells: every oracle
answer is still the literal `unknown`, and the alias probe
`c7_alias_0_d005` served `Not in memory: no stored record matches
c7-signal-0` with 0 seats.

Same worktree (r2 committed 9158c2b), same rules (no git, no
subagents, no GPU, foreground, never kill anything). Effort: high.

## Mission
1. Fix the oracle layer-position setup against the REAL module
   surface (`core/gpt_oss20b_tc.py` `GptOssAttentionTC`; find where
   the arena reads/sets the live shift for that class and reuse it),
   and make the CPU fake expose exactly the real attribute surface so
   this class of bug is caught (test: the fake's attention object has
   no attribute the real class lacks; enumerate from the real class).
2. Diagnose why the oracle answered `unknown` on the completed
   distance-5 probes even before the crash (is the source text in the
   live window at the oracle call? quote the Harmony-wrapped prompt
   from a receipt), and why the alias probe's identifier does not bind
   (is the alias deposited as text the identifier scan can see?). Fix
   in the harness if harness; STOP and report if core.
3. Register r2 continuation: cells 1–3 receipts stay valid iff their
   code path is unaffected by the fix (rule it; the oracle rows in
   them are already wrong-by-harness, so say whether the summary must
   treat their oracle column as NOT_MEASURED), resume at A-024-031;
   `lead_commands_r2_resume.txt`.

## Done (verbatim)
1. The attribute fix (file/lines) and the surface-parity test name;
   the oracle and alias diagnoses with quoted receipts; harness/core
   rulings.
2. Continuation registration path + sha; validity ruling for cells
   1–3; exact lead command.
3. Prior art; deviations; RED; process safety; model id and effort.
