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

### 2026-07-05 - R3 V2 Full Binding Gate

**Status:** complete.

**Completed work:**

- Ran the frozen V2 probe set across `amnesia`, `source-native`,
  `target-native`, and `translated`.
- Ran `analyze-binding-eval` on the full V2 result to capture translated
  misses and translated-vs-amnesia margin comparisons.
- Updated `docs/QWEN35_TRANSLATION_RELIABILITY_PLAN.md` with the R3 result and
  moved the open queue to R4 translator tuning.

**Evidence:**

- R3 command:
  `PYTHONPATH=.:/mnt/ForgeRealm/Project-Tensor/tensor_cuda python3 scripts/qwen35_graft_translate_poc.py eval-binding-probes --probes /mnt/ForgeRealm/qwen35_graft_translation_poc/gates/binding_probes_v2.json --out /mnt/ForgeRealm/qwen35_graft_translation_poc/gates/binding_eval_v2_full.json --source-model-dir /home/vader/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc --target-model-dir /home/vader/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a --translator-dir /mnt/ForgeRealm/qwen35_graft_translation_poc/translator --modes amnesia,source-native,target-native,translated --max-probes 64 --layers all`
- R3 artifact:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/gates/binding_eval_v2_full.json`
  - sha256:
    `e8157316292e88ab872144354e5f54ee68b409c81d7334f4a190b13d1b9a2df7`
  - schema: `qwen35_graft_translation_binding_eval_v1`
  - probe count: `64`
- R3 analysis command:
  `PYTHONPATH=.:/mnt/ForgeRealm/Project-Tensor/tensor_cuda python3 scripts/qwen35_graft_translate_poc.py analyze-binding-eval --binding-eval /mnt/ForgeRealm/qwen35_graft_translation_poc/gates/binding_eval_v2_full.json --out /mnt/ForgeRealm/qwen35_graft_translation_poc/gates/binding_eval_v2_full_analysis.json`
- R3 analysis artifact:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/gates/binding_eval_v2_full_analysis.json`
  - sha256:
    `123543a4377c3c8068b17215b47f97707904b074bcc4634549777c0990f3ba23`
  - schema: `qwen35_graft_translation_binding_analysis_v1`

**Mode summary:**

- amnesia: `9 / 64` positive margins, mean margin
  `-2.255242589061309`, min margin `-6.48096410594205`
- source-native: `64 / 64` positive margins, mean margin
  `38.7234434921245`, min margin `31.493913498901925`
- target-native: `64 / 64` positive margins, mean margin
  `34.43096542845171`, min margin `29.44889459450494`
- translated: `40 / 64` positive margins, mean margin
  `0.41503181978418846`, min margin `-5.668900343599823`

**R3 analysis:**

- translated misses: `24 / 64`
  - `bind-v2-002-q0`, `bind-v2-002-q1`, `bind-v2-005-q0`,
    `bind-v2-005-q1`, `bind-v2-007-q1`, `bind-v2-012-q0`,
    `bind-v2-012-q1`, `bind-v2-013-q1`, `bind-v2-015-q0`,
    `bind-v2-016-q0`, `bind-v2-017-q0`, `bind-v2-017-q1`,
    `bind-v2-019-q0`, `bind-v2-020-q0`, `bind-v2-020-q1`,
    `bind-v2-022-q0`, `bind-v2-023-q0`, `bind-v2-023-q1`,
    `bind-v2-026-q0`, `bind-v2-027-q0`, `bind-v2-027-q1`,
    `bind-v2-029-q1`, `bind-v2-030-q1`, `bind-v2-031-q0`
- amnesia successes: `9 / 64`
- translated beats amnesia by margin: `63 / 64`
- amnesia beats translated by margin: `1 / 64`
  - `bind-v2-027-q0`

**Interpretation:**

- R3 passes the frozen full-gate thresholds:
  source-native and target-native both reached `64 / 64`, translated exceeded
  `>= 28 / 64`, and translated beat amnesia by `+31` positive margins.
- Claim level: stronger binding transfer signal, but not yet high-reliability
  transfer. The translated mean margin is shallow and misses `24 / 64`, so R4
  should tune the translator against the frozen V2 set.

**Remaining work:**

- Start R4 translator tuning against the frozen V2 probe set.
- Keep R5 live G0 repair separate from translator quality.

### 2026-07-05 - R4.1 Ridge Sweep Protocol Registered

**Status:** complete.

**Completed work:**

- Registered the first translator-tuning protocol before running new
  candidates.
- Froze R4.1 scope to a ridge-only sweep over the existing paired capture
  corpus.
- Confirmed no corpus recapture, probe changes, or architecture changes are in
  scope for R4.1.

**Protocol:**

- Frozen probe set:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/gates/binding_probes_v2.json`
  - sha256:
    `4cc97ff2d230692e8c0b21bc265bd63132605d16147c5b5623c4ba224728f4a5`
- Baseline translator:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/translator`
  - ridge lambda: `1e-4`
  - V2 translated result: `40 / 64`, mean margin `0.41503181978418846`
- New ridge candidates:
  - `1e-5`
  - `3e-5`
  - `3e-4`
  - `1e-3`
- Candidate output pattern:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/translator_ridge_<lambda>`
- Required commands per candidate:
  - `fit-translator`
  - `eval-translator --split heldout --topk 16`
  - `eval-binding-probes --modes translated` against frozen V2

**Selection rule:**

- Prefer higher translated positive margins.
- Break ties by translated mean margin.
- Reject candidates that materially worsen held-out G1/G2 against baseline.

**Remaining work:**

- Run R4.1 ridge candidates.
- Summarize the winner and commit the result.

### 2026-07-05 - R4.1 Ridge Sweep Implementation And Results

**Status:** complete.

**Completed work:**

- Added CUDA/CuPy-capable ridge accumulation/solve support.
- Added `fit-translator-sweep` so multiple ridge lambdas share one capture
  accumulation pass.
- Added `--skip-fit-metrics` so candidate translators can be written without
  the expensive train-fit metrics rescan. The real R4 metrics are held-out
  geometry plus frozen V2 binding.
- Added `eval-translator-sweep` so several translator directories can share one
  held-out capture pass.
- Added `--max-pairs` for explicitly bounded held-out diagnostics.
- Ran the R4.1 ridge sweep candidates and the frozen V2 translated binding
  gate for each candidate.

**Implementation evidence:**

- Focused test command:
  `PYTHONDONTWRITEBYTECODE=1 PYTEST_ADDOPTS='-p no:cacheprovider' python3 -m pytest tests/test_qwen35_translation_poc.py -q`
  - result after sweep/eval changes: `32 passed, 2 warnings in 0.38s`
- Syntax check note:
  `python3 -m py_compile ...` was not useful in this environment because
  `core/__pycache__` is read-only; the focused pytest import/execute path is
  the validation used for this checkpoint.

**CUDA/backend findings:**

- CuPy import/device smoke succeeded:
  - `cupy 14.0.1`
  - `device_count 1`
  - matrix multiply succeeded.
- Naive CPU single-lambda fit was interrupted after roughly `28` minutes while
  still accumulating; host CPU was about `1100%`.
- CuPy single-lambda fit was interrupted after roughly `12` minutes; GPU
  allocation was about `328 MiB`, utilization was light, and the trace showed
  the bottleneck in compressed `.npz` reads/decompression.
- CuPy multi-lambda sweep without `--skip-fit-metrics` completed weight writes
  but stalled in the second metrics pass over compressed captures.
- Conclusion: CUDA math works, but the compressed capture format is the current
  limiter. The next speed fix is uncompressed/mmap capture shards or a
  Rust/C++/CUDA streaming path that avoids Python zip decompression.

**Ridge sweep command:**

- `PYTHONPATH=.:/mnt/ForgeRealm/Project-Tensor/tensor_cuda python3 scripts/qwen35_graft_translate_poc.py fit-translator-sweep --capture-dir /mnt/ForgeRealm/qwen35_graft_translation_poc/captures --out-root /mnt/ForgeRealm/qwen35_graft_translation_poc --out-prefix translator_ridge --ridge-lambdas 1e-5,3e-5,3e-4,1e-3 --split train --backend cupy --skip-fit-metrics`
  - status: complete
  - schema: `qwen35_graft_translation_ridge_sweep_v1`
  - compute backend: `cupy`
  - fit metrics computed: `false`
  - paired shards: `8855`
  - train tokens per layer/kind: `2245444`

**Ridge artifact hashes:**

- `translator_ridge_1e-5/translator_manifest.json`
  - sha256:
    `f8c6ea3714e0fde9b836786731d79b85d84ceb3be8396c8ee38e2520f748549d`
- `translator_ridge_1e-5/fit_metrics.json`
  - sha256:
    `3d51cae8f807f11f2f721327822ee3d047b26bc2f75b0dc3ad5e94a3bd9faaea`
- `translator_ridge_3e-5/translator_manifest.json`
  - sha256:
    `56ba707d70f55c7b73626fdfe43503c061bfc4fead939205a8a857125292da80`
- `translator_ridge_3e-5/fit_metrics.json`
  - sha256:
    `beb831c45a7d60ca306b59abcf12415800e55071e34a8e9b120530af40f88e13`
- `translator_ridge_3e-4/translator_manifest.json`
  - sha256:
    `5693a0e1fa2c3c036f7d2d382794c90b9d464b4979421513fc96a8dc69256d4b`
- `translator_ridge_3e-4/fit_metrics.json`
  - sha256:
    `42fbbd10be320149d7733b9888df0d67934a145aefab8440033f86f436319146`
- `translator_ridge_1e-3/translator_manifest.json`
  - sha256:
    `405e89c76d2e0d5c594e9415c03beb503c77ae993090815ddb9c4519c4ecaecb`
- `translator_ridge_1e-3/fit_metrics.json`
  - sha256:
    `a8fd0155933ae328c6cdcc60f04d722b0cb2246eb6a882588061836e74fd1bcc`

**Held-out diagnostic command:**

- Full held-out sweep was interrupted after `50 / 1006` shards because it
  projected to roughly `100+` minutes. The recorded held-out geometry result is
  therefore an explicitly bounded diagnostic:
  `PYTHONPATH=.:/mnt/ForgeRealm/Project-Tensor/tensor_cuda python3 scripts/qwen35_graft_translate_poc.py eval-translator-sweep --capture-dir /mnt/ForgeRealm/qwen35_graft_translation_poc/captures --translator-dirs /mnt/ForgeRealm/qwen35_graft_translation_poc/translator_ridge_1e-5,/mnt/ForgeRealm/qwen35_graft_translation_poc/translator_ridge_3e-5,/mnt/ForgeRealm/qwen35_graft_translation_poc/translator_ridge_3e-4,/mnt/ForgeRealm/qwen35_graft_translation_poc/translator_ridge_1e-3 --out-name eval_metrics_heldout_128.json --progress-out /mnt/ForgeRealm/qwen35_graft_translation_poc/gates/ridge_eval_sweep_heldout_128_progress.json --split heldout --topk 16 --max-pairs 128`
  - status: complete
  - paired shards: `128`
  - layers per candidate: `6`

**Held-out diagnostic hashes:**

- `translator_ridge_1e-5/eval_metrics_heldout_128.json`
  - sha256:
    `29dd9ae98232433e171f7fa74fc8ee4a2fbdfbc8832fba1e83e1b721f9b05203`
- `translator_ridge_3e-5/eval_metrics_heldout_128.json`
  - sha256:
    `d74c819835e8ce0a2793b76b22531fc22354d045cd44bb9f4f24ccd8ebbaeb0b`
- `translator_ridge_3e-4/eval_metrics_heldout_128.json`
  - sha256:
    `8b3334577ae2fb49351f751fdec3bfe37b6fe852028d2fdd9650fc75bcd94d54`
- `translator_ridge_1e-3/eval_metrics_heldout_128.json`
  - sha256:
    `cbb76bdd525f3c81860536eb390ebf2be2c0efc9f40b69c85ff5d08309f11975`
- `gates/ridge_eval_sweep_heldout_128_progress.json`
  - sha256:
    `dcaa3b009e6704b8332028bf21d22f5653832fb6abbb3c6a646975dfd129f452`

**Held-out diagnostic summary:**

- `1e-5`: mean key recall@16 `0.6367017381462475`,
  translated-output cosine `0.9044831601116691`, translated-output MSE
  `0.17619949460165593`
- `3e-5`: mean key recall@16 `0.6367017182780326`,
  translated-output cosine `0.9044831600217935`, translated-output MSE
  `0.17619949465473603`
- `3e-4`: mean key recall@16 `0.6367016586733878`,
  translated-output cosine `0.9044831606176721`, translated-output MSE
  `0.17619949546122562`
- `1e-3`: mean key recall@16 `0.636701770047325`,
  translated-output cosine `0.904483160559194`, translated-output MSE
  `0.17619949885449604`

**Frozen V2 translated binding commands:**

- `1e-5`:
  `PYTHONPATH=.:/mnt/ForgeRealm/Project-Tensor/tensor_cuda python3 scripts/qwen35_graft_translate_poc.py eval-binding-probes --probes /mnt/ForgeRealm/qwen35_graft_translation_poc/gates/binding_probes_v2.json --out /mnt/ForgeRealm/qwen35_graft_translation_poc/gates/binding_eval_v2_translated_ridge_1e-5.json --source-model-dir /home/vader/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc --target-model-dir /home/vader/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a --translator-dir /mnt/ForgeRealm/qwen35_graft_translation_poc/translator_ridge_1e-5 --modes translated --max-probes 64 --layers all`
- `3e-5`: same command with
  `--out .../binding_eval_v2_translated_ridge_3e-5.json` and
  `--translator-dir .../translator_ridge_3e-5`
- `3e-4`: same command with
  `--out .../binding_eval_v2_translated_ridge_3e-4.json` and
  `--translator-dir .../translator_ridge_3e-4`
- `1e-3`: same command with
  `--out .../binding_eval_v2_translated_ridge_1e-3.json` and
  `--translator-dir .../translator_ridge_1e-3`

**Frozen V2 translated binding hashes:**

- `gates/binding_eval_v2_translated_ridge_1e-5.json`
  - sha256:
    `880ebb5dcdc8e7b4ef846ccca1e7eb1df25d22a5499bdf26c463bca8afabb828`
- `gates/binding_eval_v2_translated_ridge_3e-5.json`
  - sha256:
    `d2e06e548f8698283daa246ec6276092a71da6935b03826a430ced0c989f8009`
- `gates/binding_eval_v2_translated_ridge_3e-4.json`
  - sha256:
    `dd757aa0d85a5ab4a703fd190ba1673a96ddc03dc43ef918f5c88b0a1bf7eac7`
- `gates/binding_eval_v2_translated_ridge_1e-3.json`
  - sha256:
    `d8f8022fd57ec9d8c46f8da8ef30dc68a754a479881e7878816b9dc947748833`

**Frozen V2 translated binding summary:**

- Baseline `1e-4`: `40 / 64`, mean margin `0.41503181978418846`,
  min margin `-5.668900343599823`
- `1e-5`: `38 / 64`, mean margin `0.4035796222422129`,
  min margin `-5.658042730095097`
- `3e-5`: `36 / 64`, mean margin `0.4045612544007525`,
  min margin `-5.607635710753115`
- `3e-4`: `39 / 64`, mean margin `0.42209077022321806`,
  min margin `-5.747188284219789`
- `1e-3`: `38 / 64`, mean margin `0.4057173672102292`,
  min margin `-5.668216921540541`

**Per-probe comparison against baseline:**

- `1e-5`: gained `0`, lost `2`
  - lost: `bind-v2-021-q1`, `bind-v2-025-q1`
- `3e-5`: gained `0`, lost `4`
  - lost: `bind-v2-008-q0`, `bind-v2-015-q1`,
    `bind-v2-021-q1`, `bind-v2-026-q1`
- `3e-4`: gained `0`, lost `1`
  - lost: `bind-v2-026-q1`
- `1e-3`: gained `0`, lost `2`
  - lost: `bind-v2-021-q1`, `bind-v2-026-q1`

**Interpretation:**

- Scalar ridge lambda tuning does not improve Qwen3.5 2B-to-9B graft
  translation reliability on the frozen V2 binding gate.
- The original `1e-4` translator remains the R4.1 winner because it has the
  best positive-margin count and no tested ridge candidate recovered a baseline
  miss.
- `3e-4` has a slightly higher mean margin but worse success count and a lower
  min margin, so it is not a better operating point.
- Next R4 work should move to layer policy, more paired corpus, or an objective
  tied directly to K score/top-k and V attention-output preservation.

**Remaining work:**

- Commit and push the R4.1 implementation/results checkpoint.
- Start R4.2 only after selecting the next registered intervention.

### 2026-07-05 - R4.2 Layer Policy Protocol Registered

**Status:** complete.

**Completed work:**

- Registered the second translator-tuning protocol before running new gates.
- Froze R4.2 scope to layer-policy filtering over the existing `1e-4`
  baseline translator.
- Confirmed no corpus recapture, ridge refit, objective change, or probe change
  is in scope for R4.2.

**Protocol:**

- Frozen probe set:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/gates/binding_probes_v2.json`
  - sha256:
    `4cc97ff2d230692e8c0b21bc265bd63132605d16147c5b5623c4ba224728f4a5`
- Baseline translator:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/translator`
  - ridge lambda: `1e-4`
  - V2 translated result: `40 / 64`, mean margin
    `0.41503181978418846`
- Candidate layer policies:
  - `drop_l3`: remove source `3` to target `3`
  - `drop_l11_l15`: remove source `11` to target `15`
  - `strong4`: keep `7->7`, `15->19`, `19->27`, `23->31`
  - `late3`: keep `15->19`, `19->27`, `23->31`
- Candidate output pattern:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/translator_layer_<policy>`
- Required commands per candidate:
  - write filtered translator from the baseline translator,
  - `eval-binding-probes --modes translated` against frozen V2,
  - compare gained/lost probes against baseline translated rows.

**Selection rule:**

- Prefer higher translated positive margins.
- Break ties by translated mean margin.
- Reject candidates that only lose baseline successes without recovering
  baseline misses.

**Remaining work:**

- Implement reproducible translator layer filtering.
- Run R4.2 candidates.
- Summarize the winner and commit the result.

### 2026-07-05 - R4.2 Layer Policy Implementation And Results

**Status:** complete.

**Completed work:**

- Added `filter-translator-layers` CLI.
- Added `filter_translator_layers` helper and unit coverage.
- Wrote four filtered translator manifests from the frozen `1e-4` baseline.
- Ran the frozen V2 translated binding gate for each R4.2 candidate.

**Implementation evidence:**

- Focused test command:
  `PYTHONDONTWRITEBYTECODE=1 PYTEST_ADDOPTS='-p no:cacheprovider' python3 -m pytest tests/test_qwen35_translation_poc.py -q`
  - result: `33 passed, 2 warnings in 0.40s`

**Filtered translator commands:**

- `drop_l3`:
  `PYTHONPATH=.:/mnt/ForgeRealm/Project-Tensor/tensor_cuda python3 scripts/qwen35_graft_translate_poc.py filter-translator-layers --translator-dir /mnt/ForgeRealm/qwen35_graft_translation_poc/translator --out-dir /mnt/ForgeRealm/qwen35_graft_translation_poc/translator_layer_drop_l3 --policy-name drop_l3 --drop-pairs 3:3`
- `drop_l11_l15`:
  `PYTHONPATH=.:/mnt/ForgeRealm/Project-Tensor/tensor_cuda python3 scripts/qwen35_graft_translate_poc.py filter-translator-layers --translator-dir /mnt/ForgeRealm/qwen35_graft_translation_poc/translator --out-dir /mnt/ForgeRealm/qwen35_graft_translation_poc/translator_layer_drop_l11_l15 --policy-name drop_l11_l15 --drop-pairs 11:15`
- `strong4`:
  `PYTHONPATH=.:/mnt/ForgeRealm/Project-Tensor/tensor_cuda python3 scripts/qwen35_graft_translate_poc.py filter-translator-layers --translator-dir /mnt/ForgeRealm/qwen35_graft_translation_poc/translator --out-dir /mnt/ForgeRealm/qwen35_graft_translation_poc/translator_layer_strong4 --policy-name strong4 --keep-pairs 7:7,15:19,19:27,23:31`
- `late3`:
  `PYTHONPATH=.:/mnt/ForgeRealm/Project-Tensor/tensor_cuda python3 scripts/qwen35_graft_translate_poc.py filter-translator-layers --translator-dir /mnt/ForgeRealm/qwen35_graft_translation_poc/translator --out-dir /mnt/ForgeRealm/qwen35_graft_translation_poc/translator_layer_late3 --policy-name late3 --keep-pairs 15:19,19:27,23:31`

**Filtered translator manifest hashes:**

- `translator_layer_drop_l3/translator_manifest.json`
  - sha256:
    `7ee72e037fc5ee66a04f3d1e7052c8b350fa0e6001fd5b2e1c7b3f3217b7496e`
- `translator_layer_drop_l11_l15/translator_manifest.json`
  - sha256:
    `79ad506273f150390e1080490976e599b962d61467ee81be2153ab729d79acde`
- `translator_layer_strong4/translator_manifest.json`
  - sha256:
    `5b7492d8f33712a62e5fd356d1abaee1b6f16294baf7b8037a5cb647b8f90e50`
- `translator_layer_late3/translator_manifest.json`
  - sha256:
    `dd84baab9875d9344d3c1ec76d2ca91fdec9bf6b29a02e0a0cf9c8c504a5ec51`

**Frozen V2 translated binding commands:**

- `drop_l3`:
  `PYTHONPATH=.:/mnt/ForgeRealm/Project-Tensor/tensor_cuda python3 scripts/qwen35_graft_translate_poc.py eval-binding-probes --probes /mnt/ForgeRealm/qwen35_graft_translation_poc/gates/binding_probes_v2.json --out /mnt/ForgeRealm/qwen35_graft_translation_poc/gates/binding_eval_v2_translated_layer_drop_l3.json --source-model-dir /home/vader/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc --target-model-dir /home/vader/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a --translator-dir /mnt/ForgeRealm/qwen35_graft_translation_poc/translator_layer_drop_l3 --modes translated --max-probes 64 --layers all`
- `drop_l11_l15`: same command with
  `--out .../binding_eval_v2_translated_layer_drop_l11_l15.json` and
  `--translator-dir .../translator_layer_drop_l11_l15`
- `strong4`: same command with
  `--out .../binding_eval_v2_translated_layer_strong4.json` and
  `--translator-dir .../translator_layer_strong4`
- `late3`: same command with
  `--out .../binding_eval_v2_translated_layer_late3.json` and
  `--translator-dir .../translator_layer_late3`

**Frozen V2 translated binding hashes:**

- `gates/binding_eval_v2_translated_layer_drop_l3.json`
  - sha256:
    `82a3c64a51b4ab7053c3d28755fe4a02b6c47b16ecb8ad31b2c6e0c476731409`
- `gates/binding_eval_v2_translated_layer_drop_l11_l15.json`
  - sha256:
    `e5eaa686fa4b19c3cc3e7ef37411953a79d67836ad115036e4e7bcb97ee83147`
- `gates/binding_eval_v2_translated_layer_strong4.json`
  - sha256:
    `016110e64aa7dcaac4cd91d3d4c47bc33cd7a73e051343a3eb750618ef1b4f44`
- `gates/binding_eval_v2_translated_layer_late3.json`
  - sha256:
    `2acae59503f64233e80aa4f1bcd42a8dad7c467bcaac19b6af9cc0a51c865b1d`

**Frozen V2 translated binding summary:**

- Baseline `1e-4`: `40 / 64`, mean margin `0.41503181978418846`,
  min margin `-5.668900343599823`
- `drop_l3`: `38 / 64`, mean margin `0.35193815798529215`,
  min margin `-5.564677949742489`
- `drop_l11_l15`: `38 / 64`, mean margin `0.4061522761956001`,
  min margin `-5.4681166409774065`
- `strong4`: `36 / 64`, mean margin `0.32240649378745834`,
  min margin `-5.3586791078400395`
- `late3`: `36 / 64`, mean margin `0.3056605327202693`,
  min margin `-5.266425911516741`

**Per-probe comparison against baseline:**

- `drop_l3`: gained `3`, lost `5`
  - gained: `bind-v2-002-q1`, `bind-v2-022-q0`, `bind-v2-030-q1`
  - lost: `bind-v2-003-q1`, `bind-v2-008-q0`,
    `bind-v2-015-q1`, `bind-v2-021-q1`, `bind-v2-030-q0`
- `drop_l11_l15`: gained `0`, lost `2`
  - lost: `bind-v2-021-q1`, `bind-v2-026-q1`
- `strong4`: gained `2`, lost `6`
  - gained: `bind-v2-007-q1`, `bind-v2-022-q0`
  - lost: `bind-v2-003-q1`, `bind-v2-008-q0`,
    `bind-v2-015-q1`, `bind-v2-021-q1`, `bind-v2-025-q1`,
    `bind-v2-030-q0`
- `late3`: gained `2`, lost `6`
  - gained: `bind-v2-007-q1`, `bind-v2-022-q0`
  - lost: `bind-v2-003-q1`, `bind-v2-008-q0`,
    `bind-v2-015-q1`, `bind-v2-021-q1`, `bind-v2-026-q1`,
    `bind-v2-030-q0`

**Interpretation:**

- Simple layer filtering does not improve aggregate Qwen3.5 2B-to-9B graft
  translation reliability.
- Some layer policies recover individual baseline misses, so layer contribution
  is real, but omission is too blunt: every tested policy loses more baseline
  successes than it gains.
- The all-layer `1e-4` baseline remains the R4.2 winner.
- Next R4 work should use more paired corpus or objective-level tuning rather
  than layer omission.

**Remaining work:**

- Commit and push the R4.2 implementation/results checkpoint.
- Register R4.3 after inspecting whether more corpus is already available or
  needs capture work.

### 2026-07-05 - R4.3 5M Corpus Protocol Registered

**Status:** complete.

**Completed work:**

- Inspected the current corpus/capture state.
- Confirmed the baseline capture corpus is complete but capped at `2.5M`
  tokens:
  - source: `9861` shards, `2500000` tokens
  - target: `9861` shards, `2500000` tokens
  - train: `8855` paired shards, `2245444` tokens
  - heldout: `1006` paired shards, `254556` tokens
- Registered R4.3 as a `5M` paired-corpus expansion with separate artifacts
  from the baseline.

**Protocol:**

- Baseline corpus plan:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/corpus_plan.json`
  - max total tokens: `2500000`
  - command line recorded in plan:
    `scripts/qwen35_graft_translate_poc.py plan-corpus --model-dir /home/vader/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc --corpus /mnt/ForgeRealm/scribe_mint_v1/manifest.jsonl --corpus /mnt/ForgeRealm/HumanBaselineCorpus/stories --out /mnt/ForgeRealm/qwen35_graft_translation_poc/corpus_plan.json --chunk-tokens 256 --heldout-fraction 0.10 --seed qwen35-translation-poc-0 --max-total-tokens 2500000 --min-doc-tokens 64 --source-label scribe-mint-plus-humanbaseline-v1`
- R4.3 corpus plan:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/corpus_plan_5m.json`
  - max total tokens: `5000000`
  - same corpus paths, chunk size, split fraction, seed, min-doc token floor,
    and source label as baseline.
- R4.3 capture directory:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/captures_5m`
- R4.3 translator directory:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/translator_corpus_5m`
- Frozen V2 binding output:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/gates/binding_eval_v2_translated_corpus_5m.json`

**Rules:**

- Do not mutate the completed baseline `2.5M` capture plan/artifacts.
- Reuse old shards in `captures_5m` only when source/target shard identity is
  exact under the new plan.
- Use the same frozen V2 probe set as R2/R3/R4.1/R4.2.

**Remaining work:**

- Generate `corpus_plan_5m.json`.
- Determine exact overlap with the baseline plan.
- Seed `captures_5m` from exact overlap if possible.
- Capture missing source/target shards.
- Fit and gate `translator_corpus_5m`.

### 2026-07-05 - R4.3 Corpus Plan And Capture Seed

**Status:** complete.

**Completed work:**

- Generated the R4.3 `5M`-cap corpus plan.
- Determined that the registered source set contains `3,408,241` usable tokens
  under the current filters, so R4.3 is a `2.5M` to `3.41M` expansion unless a
  later protocol adds new corpus sources.
- Compared the baseline `2.5M` plan and R4.3 plan for exact chunk overlap.
- Hardlinked exact-overlap source/target `.npz` and sidecar `.json` files into
  `captures_5m`.
- Refreshed the R4.3 capture manifest.

**Commands:**

- Corpus plan:
  `PYTHONPATH=.:/mnt/ForgeRealm/Project-Tensor/tensor_cuda python3 scripts/qwen35_graft_translate_poc.py plan-corpus --model-dir /home/vader/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc --corpus /mnt/ForgeRealm/scribe_mint_v1/manifest.jsonl --corpus /mnt/ForgeRealm/HumanBaselineCorpus/stories --out /mnt/ForgeRealm/qwen35_graft_translation_poc/corpus_plan_5m.json --chunk-tokens 256 --heldout-fraction 0.10 --seed qwen35-translation-poc-0 --max-total-tokens 5000000 --min-doc-tokens 64 --source-label scribe-mint-plus-humanbaseline-v1`
- Capture manifest refresh:
  `PYTHONPATH=.:/mnt/ForgeRealm/Project-Tensor/tensor_cuda python3 scripts/qwen35_graft_translate_poc.py capture-status --plan /mnt/ForgeRealm/qwen35_graft_translation_poc/corpus_plan_5m.json --out-dir /mnt/ForgeRealm/qwen35_graft_translation_poc/captures_5m`

**Artifacts:**

- R4.3 corpus plan:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/corpus_plan_5m.json`
  - sha256:
    `8cf20f167796742dca730271f0912e776c22b67beb0a5c26a40f6d582cb48a83`
  - documents: `1292`
  - chunks: `13820`
  - total tokens: `3408241`
  - train tokens: `3067954`
  - heldout tokens: `340287`
- Overlap seed summary:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/captures_5m_seed_overlap.json`
  - sha256:
    `38be2419533d21c36f44eb4cf6daf4e57fe57c07db771cb368d65b6348418418`
  - exact reusable chunks: `9860`
  - mismatched overlap chunks: `1`
  - new-only chunks: `3959`
  - hardlinked files: `39440`
  - missing old file count: `0`
  - mismatch: `doc_id=60ca70dec690f631`, `chunk_id=9860`,
    old token count `37`, new token count `256`
- R4.3 capture manifest after seed:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/captures_5m/capture_manifest.json`
  - sha256:
    `89afe6d924df2c7db4170995f5141be039baf93641f7241d5871c7dc92f24928`
  - source completed: `9860 / 13820`
  - target completed: `9860 / 13820`
  - remaining source chunks: `3960`
  - remaining target chunks: `3960`
  - paired shards: `9860`
  - paired tokens: `2499963`

**Interpretation:**

- The same registered source set cannot supply a full `5M` tokens after
  filtering; it supplies `3.41M`.
- Exact hardlink reuse avoided recapturing nearly all of the baseline corpus.
- The remaining R4.3 work is approximately `908278` additional tokens per role
  plus the one corrected overlap chunk.

**Remaining work:**

- Capture the remaining R4.3 source chunks.
- Capture the remaining R4.3 target chunks.
- Fit and gate `translator_corpus_5m`.

### 2026-07-05 - R4.3 Fast Single-Fit Path

**Status:** complete.

**Completed work:**

- Added `compute_fit_metrics` support to the single `fit_ridge_translator`
  path.
- Exposed `--skip-fit-metrics` on the `fit-translator` CLI, matching the
  existing sweep fast path.
- Wired the same skip policy through `pipeline-next` and recorded it in the
  pipeline stage status.
- Updated the corpus runbook cron command to include `--skip-fit-metrics`.
- Added focused coverage for the single-fit skip path.

**Reason:**

- The expanded `3.41M` token corpus is large enough that re-reading all train
  capture shards just to compute train-set fit metrics adds a second expensive
  decompression pass.
- R4.3's real selection gates are held-out G1/G2 plus frozen V2 binding
  probes, so the translator can be written immediately after solving and still
  keep explicit null train metrics in `fit_metrics.json`.

**Validation:**

- `PYTHONPATH=. PYTEST_ADDOPTS='-p no:cacheprovider' pytest -q tests/test_qwen35_translation_poc.py -q`
  - result: `34 passed`

**Remaining work:**

- Finish target capture for `captures_5m`.
- Fit `translator_corpus_5m` with:
  `fit-translator --backend cupy --skip-fit-metrics`.
- Run held-out G1/G2 and frozen V2 translated binding gates.
