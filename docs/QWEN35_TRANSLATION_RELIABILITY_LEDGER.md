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

### 2026-07-04 - R0 Baseline Miss And Floor Analysis

**Status:** complete.

**Completed work:**

- Added `analyze-binding-eval` CLI to join binding eval rows back to the probe
  manifest.
- Added per-probe best-decoy reporting, score/margin summaries, translated
  miss list, amnesia success list, and translated-vs-amnesia margin comparison.
- Added unit coverage for R0 analysis semantics.
- Ran R0 on the existing V1 binding artifact.
- Updated `docs/QWEN35_TRANSLATION_RELIABILITY_PLAN.md` with the R0 result and
  next open queue.

**Evidence:**

- Focused test command:
  `PYTHONDONTWRITEBYTECODE=1 PYTEST_ADDOPTS='-p no:cacheprovider' python3 -m pytest tests/test_qwen35_translation_poc.py -q`
  - result: `28 passed, 2 warnings in 0.57s`
- R0 command:
  `PYTHONPATH=.:/mnt/ForgeRealm/Project-Tensor/tensor_cuda python3 scripts/qwen35_graft_translate_poc.py analyze-binding-eval --binding-eval /mnt/ForgeRealm/qwen35_graft_translation_poc/gates/binding_eval_metrics.json --out /mnt/ForgeRealm/qwen35_graft_translation_poc/gates/binding_eval_analysis_v1.json`
- R0 artifact:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/gates/binding_eval_analysis_v1.json`
  - sha256:
    `7b10e0e71787592e7aca313023fbf7d7717140d9293c23a491d2bd8af56e1825`
  - schema: `qwen35_graft_translation_binding_analysis_v1`
  - probe count: `32`
  - tokenizer source:
    `/home/vader/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc`
- Mode summary:
  - amnesia: `20 / 32` positive margins, mean margin
    `0.7479729879753355`, min margin `-2.897013226382356`
  - source-native: `32 / 32` positive margins, mean margin
    `19.93564477957448`, min margin `15.442655127356694`
  - target-native: `32 / 32` positive margins, mean margin
    `18.62211288181452`, min margin `15.489111058558944`
  - translated: `25 / 32` positive margins, mean margin
    `1.3904626497223576`, min margin `-2.2494257231745074`
- R0 lists:
  - translated misses: `bind-005`, `bind-009`, `bind-014`, `bind-016`,
    `bind-017`, `bind-021`, `bind-025`
  - amnesia successes: `bind-000`, `bind-001`, `bind-002`, `bind-003`,
    `bind-004`, `bind-008`, `bind-012`, `bind-013`, `bind-014`, `bind-015`,
    `bind-018`, `bind-019`, `bind-020`, `bind-024`, `bind-026`, `bind-027`,
    `bind-028`, `bind-029`, `bind-030`, `bind-031`
  - translated beats amnesia by margin: `28 / 32`
  - amnesia beats translated by margin: `4 / 32`
- Translated miss detail:
  - `bind-005` Fenn gold `FE-4786`, translated margin
    `-2.2494257231745074`, translated best decoy `NA-4666`
  - `bind-009` Juno gold `JU-5625`, translated margin
    `-0.2830320464233118`, translated best decoy `SA-6067`
  - `bind-014` Orion gold `OR-5587`, translated margin
    `-0.006499587170058163`, translated best decoy `XY-1964`
  - `bind-016` Quill gold `QU-6150`, translated margin
    `-0.9094927078225616`, translated best decoy `YA-9016`
  - `bind-017` Riven gold `RI-6379`, translated margin
    `-1.59460943221778`, translated best decoy `AR-4459`
  - `bind-021` Vega gold `VE-3345`, translated margin
    `-0.8032524382285153`, translated best decoy `DA-9696`
  - `bind-025` Zephyr gold `ZE-4004`, translated margin
    `-1.1466538344599186`, translated best decoy `AS-1081`

**Interpretation:**

- V1 is not a clean no-memory floor because amnesia succeeds on `20 / 32`.
- Translated mode still improves margin over amnesia on `28 / 32`, so the V1
  result is not just prompt leakage.
- R1 should generate less semantically obvious probes and more tightly matched
  decoys before further translator tuning.

**Remaining work:**

- Implement R1 V2 probe generator and tests.
- Run R2 V2 amnesia floor gate.

### 2026-07-04 - R1 Binding Probe V2 Generator

**Status:** complete.

**Completed work:**

- Added `make_binding_probe_set_v2` and `write_binding_probe_set_v2`.
- Added `--version v2` and `--templates` support to `make-binding-probes`.
- Kept V1 generation and existing evaluator compatibility intact.
- Added unit coverage for deterministic flattened V2 output, schema metadata,
  two query templates per binding, opaque code format, matched decoy count, and
  no duplicate candidate codes across bindings.
- Wrote the registered V2 probe artifact under the existing PoC gate directory.

**Evidence:**

- Focused test command:
  `PYTHONDONTWRITEBYTECODE=1 PYTEST_ADDOPTS='-p no:cacheprovider' python3 -m pytest tests/test_qwen35_translation_poc.py -q`
  - result: `29 passed, 2 warnings in 0.44s`
- R1 artifact command:
  `PYTHONPATH=.:/mnt/ForgeRealm/Project-Tensor/tensor_cuda python3 scripts/qwen35_graft_translate_poc.py make-binding-probes --version v2 --out /mnt/ForgeRealm/qwen35_graft_translation_poc/gates/binding_probes_v2.json --count 32 --templates 2`
- R1 artifact:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/gates/binding_probes_v2.json`
  - sha256:
    `4cc97ff2d230692e8c0b21bc265bd63132605d16147c5b5623c4ba224728f4a5`
  - schema: `qwen35_graft_translation_binding_probes_v2`
  - binding count: `32`
  - templates per binding: `2`
  - flattened probe rows: `64`
  - seed: `qwen35-binding-v2`

**Interpretation:**

- R2 can now run against a frozen V2 probe set.
- The next decision point is whether amnesia stays under the frozen flattened
  floor threshold of `<= 24 / 64` positive margins.

**Remaining work:**

- Run R2 V2 amnesia floor gate.
- Run R3 V2 full binding gate if the floor is clean.

### 2026-07-04 - R2 V2 Amnesia Floor Gate

**Status:** complete.

**Completed work:**

- Ran the frozen V2 probe set in amnesia-only mode.
- Confirmed the V2 floor is clean enough for full translated evaluation.
- Updated `docs/QWEN35_TRANSLATION_RELIABILITY_PLAN.md` with the R2 result and
  moved the open queue to R3.

**Evidence:**

- R2 command:
  `PYTHONPATH=.:/mnt/ForgeRealm/Project-Tensor/tensor_cuda python3 scripts/qwen35_graft_translate_poc.py eval-binding-probes --probes /mnt/ForgeRealm/qwen35_graft_translation_poc/gates/binding_probes_v2.json --out /mnt/ForgeRealm/qwen35_graft_translation_poc/gates/binding_eval_v2_amnesia.json --target-model-dir /home/vader/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a --modes amnesia --max-probes 64 --layers all`
- R2 artifact:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/gates/binding_eval_v2_amnesia.json`
  - sha256:
    `c96291f61a0001680f2c65fce9d0703e020187cf907da73b015c4278dc838234`
  - schema: `qwen35_graft_translation_binding_eval_v1`
  - probe set:
    `/mnt/ForgeRealm/qwen35_graft_translation_poc/gates/binding_probes_v2.json`
  - mode: `amnesia`
  - probe count: `64`
  - positive margins: `9 / 64`
  - mean margin: `-2.255242589061309`
  - min margin: `-6.48096410594205`
- Amnesia positive probe IDs:
  `bind-v2-000-q1`, `bind-v2-001-q0`, `bind-v2-003-q0`,
  `bind-v2-009-q0`, `bind-v2-010-q1`, `bind-v2-018-q0`,
  `bind-v2-024-q0`, `bind-v2-024-q1`, `bind-v2-025-q0`

**Interpretation:**

- V2 passes the frozen flattened floor threshold of `<= 24 / 64`.
- The old V1 floor problem is materially improved: V1 amnesia was `20 / 32`,
  while V2 amnesia is `9 / 64`.
- R3 can now test source-native, target-native, and translated on the same
  frozen V2 set.

**Remaining work:**

- Run R3 V2 full binding gate.
- Start R4 translator tuning only after the R3 result is known.
