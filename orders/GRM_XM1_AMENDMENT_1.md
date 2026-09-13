# GRM-XM1 amendment 1 — two harness defects on the card (lead, 2026-09-12)

Same worktree `/mnt/ForgeRealm/wt/grm-xm1`, same seat (Codex Astra high),
same rules as `orders/GRM_XM1_HARNESS.md` (no git, no subagents, no GPU,
foreground, < 10 min per call, never kill anything, `--basetemp` under
`artifacts/grm_xm1/tmp` for tests with cleanup, repo-relative pins).

## Lead-run receipts (GPU, `artifacts/grm_xm1/lead_gptoss_*.log`)
1. `gpt-oss C3l sup_harbor_restatement`: lease acquired, released after
   0.056 s, cell receipt `status: ERROR`, error
   `[Errno 2] No such file or directory: '…/artifacts/grm_xm1/tmp/rs4_wo6tuzsv'`.
   The worker writes its scratch under `artifacts/grm_xm1/tmp/`, which
   is the pytest basetemp the lead removes after every gate run. The
   worker must create its own scratch directory (`mkdir -p`, or a
   `tempfile.mkdtemp` under a dir it creates) and must not share the
   pytest basetemp.
2. Every following GPT-OSS cell (`C5 sup_harbor_restatement`,
   `C3l sup_praxis_fresh`, …) raised
   `XM1Error: STOP: GPT-OSS RS4 parity barrier {"status": "FAIL", "missing": [...9 cells...]}`
   — `execute()` evaluates the ten-cell barrier BEFORE running a GPT-OSS
   cell, so the barrier can never be satisfied. The barrier gates the
   OTHER models; GPT-OSS cells must run unconditionally, and the barrier
   check runs (a) after each GPT-OSS cell as a progress receipt (partial
   ok) and (b) as a hard STOP only for a non-GPT-OSS model cell while any
   of the ten is missing or mismatched.

## Mission
1. Fix both; the CPU double must exercise the exact `execute()` path the
   GPU cell takes (a fake run of all ten GPT-OSS cells in order from an
   empty `cells/gpu/` directory, with `artifacts/grm_xm1/tmp` ABSENT at
   start, must produce ten receipts and a PASS/partial barrier receipt;
   then one Qwen cell must STOP on the barrier when a GPT-OSS receipt is
   deleted — RED→GREEN both). Tests for both defects.
2. Registration amendment (separate JSON, sha-bound to registration
   758fd3ac; worker rebound); `lead_commands.txt` refreshed (commands
   must be runnable WITHOUT an outer flock — the worker leases itself;
   say so in the header).
3. `python3 -m pytest -q --basetemp artifacts/grm_xm1/tmp tests/test_grm_xm1*.py tests/test_grm_rs4*.py`
   verbatim last, and remove the basetemp after.

## Done (verbatim)
1. Files:lines for both fixes; the RED→GREEN lines.
2. Amendment path + sha; refreshed commands.
3. Prior art (unchanged unless new), deviations, RED, process safety,
   model id + effort. pytest LAST.
