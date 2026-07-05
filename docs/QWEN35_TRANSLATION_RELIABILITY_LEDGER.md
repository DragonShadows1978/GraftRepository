# Qwen3.5 Translation Reliability Ledger

This ledger is the operational completion record for the Qwen3.5 2B-to-9B
Translation reliability track.

## House Rules

- Register protocol changes in
  `docs/QWEN35_TRANSLATION_RELIABILITY_PLAN.md` before running new gates.
- Append or update this ledger after every completed implementation, eval,
  artifact, or analysis step.
- Record exact commands, artifact paths, and hashes for every result-bearing
  run.
- Commit per completed phase or stable checkpoint.
- Keep unrelated dirty work out of Translation commits.

## Starting Evidence

- Original final write-up:
  `docs/QWEN35_TRANSLATION_FINAL_WRITEUP.md`
- Original G1/G2 eval:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/translator/eval_metrics.json`
  - sha256:
    `c42847747374bb28b5b033d2a203d91dd6e14cef03f572eca5a0ff54541bfa9a`
- Original G3 binding eval:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/gates/binding_eval_metrics.json`
  - sha256:
    `0d72858222abb8a2a23a0079fec087e0e6c53f8f29c009dfc80a820f47ae954f`
- Original final pipeline status:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/pipeline_status.json`
  - sha256:
    `f4d0cbfe03fced8aec3c634a81a670be20e15edb36d0d31c8aa4709b8c7931af`
  - status: `complete`

## Entries

### 2026-07-04 - Reliability Track Registered

**Status:** complete.

**Completed work:**

- Created the reliability refinement plan.
- Created this ledger.
- Registered the phase order:
  R0 miss/floor analysis, R1 binding-probe V2 generator, R2 amnesia floor,
  R3 full V2 binding gate, R4 translator tuning, R5 live G0 repair.
- Frozen the first V2 floor thresholds before implementation:
  `<= 12 / 32` for one-query mode or `<= 24 / 64` for flattened two-query
  mode.

**Evidence:**

- Plan:
  `docs/QWEN35_TRANSLATION_RELIABILITY_PLAN.md`
- Ledger:
  `docs/QWEN35_TRANSLATION_RELIABILITY_LEDGER.md`

**Remaining work:**

- Implement R0 miss/floor analysis CLI.
- Run R0 on the existing V1 binding artifact.
- Implement R1 V2 probe generator and tests.
