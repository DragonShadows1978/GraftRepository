# GRM-XM1 X2 amendment 1 — the XM2 Qwen loader fails where the XM1 loader succeeded (lead, 2026-09-13)

Same worktree `/mnt/ForgeRealm/wt/grm-xm2`, same seat (Codex Astra high),
same rules as `orders/GRM_XM1_X2_QWEN.md` (no git, no subagents, no GPU,
foreground, < 10 min per call, never kill anything; `--basetemp` under
`artifacts/grm_xm2/tmp` with cleanup; repo-relative pins).

## Lead-run receipt (GPU)
`python3 scripts/grm_xm1_x2_run.py --cell C3l__sup_harbor_restatement --run`
→ lease acquired, model layers 1..32 loaded, then after 115 s
`ERROR: QuantLinearTC INT4 init failed tensor=lm_head.weight shape=(248320,4096)`
(`artifacts/grm_xm2/run/cells/C3l__sup_harbor_restatement.json`,
`artifacts/grm_xm2/lead_C3l__sup_harbor_restatement.log`). The XM1 worker
(`scripts/grm_xm1_parity.py --model qwen35`) loaded the SAME model on the
SAME card 40 minutes earlier and completed all ten Qwen cells — so the X2
run script takes a different load path for `lm_head` (the Qwen3.5 adapter's
receipts use a chunked / host-side lm_head; see `core/qwen35_tc.py` and
`scripts/qwen35_*` for the path XM1 inherited).

## Mission
1. Make `scripts/grm_xm1_x2_run.py` load Qwen through the exact XM1 loader
   path (import and call it; do not re-implement), differing ONLY in the
   final-channel decode + final-token window; the CPU double must prove
   both scripts resolve to the same loader callable and the same lm_head
   mode (a test that compares the resolved loader identity / config keys).
2. Registration amendment (separate JSON, sha-bound to XM2 registration
   7d537ab2; run script rebound); `lead_commands.txt` refreshed; the
   cold-start fit caveat (129.7 s historical vs 110 s cap) resolved by
   registering the cap the XM1 Qwen cells actually needed (their receipts
   carry `elapsed_s`), never by relaxing after the fact.
3. `python3 -m pytest -q --basetemp artifacts/grm_xm2/tmp tests/test_grm_xm1*.py tests/test_grm_xm2*.py`
   verbatim last; cleanup.

## Done (verbatim)
1. Files:lines; the loader-identity proof line.
2. Amendment path + sha; refreshed commands; the registered per-cell cap with its receipt.
3. Prior art (unchanged), deviations, RED, process safety, model id + effort; pytest LAST.
