# Qwen3.5 Graft Translation PoC Implementation Ledger

**Project:** Qwen3.5-2B attention graft state -> Qwen3.5-9B attention
graft dialect translator.

**Ledger law:** update this file whenever a step is completed. The entry must
include what changed, the command or artifact that proves it, and what remains
blocked or unfinished. The implementation plan keeps the gate design; this
ledger is the operational completion record.

**Current status:** Phase 0 is complete. Phase 1, Phase 2, Phase 3, and Phase
4 have working implementation pieces and smoke evidence, but they are not
scientifically complete until the remaining real-corpus gates and controls are
run.

## Active Repos And Artifacts

| Item | Path |
|---|---|
| Implementation repo | `/mnt/ForgeRealm/GraftRepository` |
| Branch | `main` |
| PoC artifact root | `/mnt/ForgeRealm/qwen35_graft_translation_poc` |
| Source weights | `/home/vader/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc` |
| Target weights | `/home/vader/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| TensorCUDA path | `/mnt/ForgeRealm/Project-Tensor/tensor_cuda` |

## Completion Entries

### 2026-07-02 - Ledger Created

**Status:** complete.

**Completed work:**

- Added this implementation ledger as the persistent record for completed
  Qwen3.5 translation PoC work.
- Defined the rule that future completed steps must append or update this
  file with evidence.

**Evidence:**

- File created at `docs/QWEN35_TRANSLATION_IMPLEMENTATION_LEDGER.md`.

**Next required update:** after each completed implementation, capture, fit,
eval, threshold registration, or control-baseline step.

### 2026-07-02 - Phase 0 Weight Source Validation

**Status:** complete.

**Completed work:**

- Implemented the Qwen3.5 translation PoC source validator in
  `core/qwen35_translation_poc.py`.
- Added CLI access through `scripts/qwen35_graft_translate_poc.py
  validate-weights`.
- Enforced safetensors-only source weights for both models.
- Rejected GGUF and already-quantized HF/bitsandbytes-style checkpoints.
- Enforced tokenizer identity as an abort gate.
- Wrote the source/target weight manifest.

**Evidence:**

- Manifest:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/weights_manifest.json`.
- Source model recorded as `Qwen/Qwen3.5-2B`, revision
  `15852e8c16360a2fea060d615a32b45270f8a8fc`, 1 safetensors shard,
  4,548,221,488 bytes.
- Target model recorded as `Qwen/Qwen3.5-9B`, revision
  `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, 4 safetensors shards,
  19,306,310,880 bytes.
- Shared tokenizer hash:
  `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`.

**Remaining work:** none for Phase 0.

### 2026-07-02 - Phase 1 Dynamic Qwen3.5 Loader

**Status:** partially complete.

**Completed work:**

- Generalized `core/qwen35_tc.py` to load text dimensions from each model's
  `config.json` instead of assuming Qwen3.5-9B constants.
- Dynamic fields now include hidden size, layer count, attention indices,
  query/KV heads, head dimension, DeltaNet dimensions, RoPE theta and partial
  dimension, tied embedding mode, repository, and revision.
- Added tied-output fallback for Qwen3.5-2B, where `lm_head.weight` is not a
  separate checkpoint tensor.
- Verified local INT4 loading and forward smokes for both real unquantized
  source directories.

**Evidence:**

- 2B INT4 GPU smoke:
  - logits shape `(1, 1, 248320)`
  - cache count `24`
  - layer-0 DeltaNet cache `(1, 3, 6144)` and `(1, 16, 128, 128)`
  - first attention cache at layer 3 K/V `(1, 2, 3, 256)`
- 9B INT4 GPU smoke:
  - logits shape `(1, 1, 248320)`
  - cache count `32`
  - layer-0 DeltaNet cache `(1, 3, 8192)` and `(1, 32, 128, 128)`
  - first attention cache at layer 3 K/V `(1, 4, 3, 256)`

**Remaining work:**

- HF/reference teacher-forced parity per model.
- Margin-based top-1 disagreement accounting per model.
- Attention cache shape checks across all attention layers.
- State save/restore smoke per model.

### 2026-07-02 - Phase 2 Attention Capture Hooks And Shards

**Status:** partially complete.

**Completed work:**

- Added Qwen3.5 attention capture hooks in `core/qwen35_tc.py`.
- Captured post-qk-norm, pre-RoPE K, V, and Q tensors.
- Updated `core/kv_graft.py` so harvest/capture/injection helpers resolve
  attention modules at either `layer.self_attn` or hybrid Qwen3.5
  `layer.mixer`.
- Added `harvest_kv_and_queries()` to capture target K/V and target queries
  in one forward pass.
- Added capture shard writing in `core/qwen35_translation_poc.py`.
- Added `capture-smoke` CLI support.

**Evidence:**

- Source smoke artifact:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/capture_smoke/source_docsmoke_chunk000000.npz`
  with layer-3 K/V shape `(1, 2, 4, 256)`.
- Target smoke artifact:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/capture_smoke/target_docsmoke_chunk000000.npz`
  with layer-3 K/V shape `(1, 4, 4, 256)` and Q shape `(1, 16, 4, 256)`.
- Optimized one-forward 9B target capture smoke:
  `/tmp/qwen35_single_forward_smoke/target_doconepass_chunk000000.npz`
  with K/V `(1, 4, 4, 256)` and Q `(1, 16, 4, 256)`.

**Remaining work:**

- Run all-attention-layer source and target capture over the real document
  corpus.
- Produce final corpus-scale `capture_manifest.json`.

### 2026-07-02 - Phase 2 Corpus Planner, Resume, And Status

**Status:** corpus plan frozen; capture execution pending.

**Completed work:**

- Added `plan-corpus` CLI to tokenize text, markdown, and JSONL sources.
- Implemented document-level train/held-out split and frozen token IDs in
  `corpus_plan.json`.
- Added resumable `capture-corpus` CLI for source or target role.
- Existing complete shards are skipped, and `capture_manifest.json` is
  refreshed after each batch.
- Added `capture-status` CLI with expected, completed, remaining, next-missing,
  and complete fields per role.
- Added `capture-next` CLI for Claude/cron loops; it runs source batches until
  source is complete, then target batches until target is complete.
- Wrote the overnight/cron handoff runbook.
- Froze the real corpus plan for the first long capture run.

**Evidence:**

- Tiny rehearsal plan:
  `/tmp/qwen35_translation_tiny_capture/corpus_plan.json`, 2 documents,
  2 chunks, 49 tokens.
- Tiny source capture completed both chunks after a resume run.
- Tiny target capture completed one chunk and reported one remaining chunk.
- `capture-status` reported:
  - source expected chunks `2`, completed `2`, remaining `0`, complete `true`
  - target expected chunks `2`, completed `1`, remaining `1`, complete `false`
- Runbook:
  `docs/QWEN35_TRANSLATION_CORPUS_RUNBOOK.md`.
- Frozen corpus plan:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/corpus_plan.json`
  - sha256
    `ed1b8c0592edb31d007561224994d109a66d46de80d8157e1e8b284219783543`
  - sources:
    `/mnt/ForgeRealm/scribe_mint_v1/manifest.jsonl` and
    `/mnt/ForgeRealm/HumanBaselineCorpus/stories`
  - seed: `qwen35-translation-poc-0`
  - documents: `226`
  - chunks: `9861`
  - tokens: `2500000`
  - train tokens: `2245444`
  - held-out tokens: `254556`
- Initial capture manifest:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/captures/capture_manifest.json`
  - sha256
    `32ba880b3db89a9ae5b0d0bc8b891a533704cbae7202feac65c4ab21f195b11b`
  - source completed chunks: `1`
  - source remaining chunks: `9860`
  - target completed chunks: `1`
  - target remaining chunks: `9860`
- First real frozen-plan source capture smoke:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/captures/source_docc7b2a153142452be_chunk000000.npz`
  - role: `source`
  - chunk id: `0`
  - tokens: `256`
  - split: `train`
  - layers: `3, 7, 11, 15, 19, 23`
  - per-layer K/V shape: `(1, 2, 256, 256)`
  - shard size: `2.4M`
- First real frozen-plan target capture smoke:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/captures/target_docc7b2a153142452be_chunk000000.npz`
  - role: `target`
  - chunk id: `0`
  - tokens: `256`
  - split: `train`
  - layers: `3, 7, 11, 15, 19, 23, 27, 31`
  - per-layer K/V shape: `(1, 4, 256, 256)`
  - per-layer query shape: `(1, 16, 256, 256)`
  - shard size: `19M`
- `PYTHONPATH=. python3 scripts/qwen35_graft_translate_poc.py capture-next --help`
  lists source/target model dirs and source/target batch sizes.

**Remaining work:**

- Run source capture to completion.
- Run target capture to completion.
- Keep appending ledger entries after each completed capture stage.

### 2026-07-02 - Phase 3 Ridge Translator Fit

**Status:** implementation smoke complete; real fit pending.

**Completed work:**

- Added `fit-translator` CLI.
- Implemented streaming paired-shard ridge fitting for K and V.
- Used full KV-width maps per attention layer, allowing cross-head mixing:
  `(KVH_2B * Dh_2B) -> (KVH_9B * Dh_9B)`.
- Saved separate K and V translator artifacts with `weight` and `bias` arrays.
- Wrote `translator_manifest.json` and `fit_metrics.json`.
- Added first-class negative-control fit modes:
  - `--control wrong-layer`
  - `--control shuffled-docs`
  - `--kinds k`
  - `--kinds v`
- Translator manifests now record control mode, fitted kinds, and paired shard
  count.

**Evidence:**

- Tiny real capture rehearsal produced K and V artifacts with shape
  `512 -> 1024`.
- Tiny rehearsal R2 was effectively `1.0` on the same-shard smoke fit.
- Synthetic tests verify wrong-layer, shuffled-docs, K-only, and V-only
  artifact generation.
- `PYTHONPATH=. python3 scripts/qwen35_graft_translate_poc.py fit-translator --help`
  lists `--control {normal,wrong-layer,shuffled-docs}` and `--kinds`.

**Remaining work:**

- Run the full train split fit after source and target capture are complete.
- Run the registered negative-control fits after source and target capture are
  complete.

### 2026-07-02 - R1 Threshold Registration

**Status:** complete.

**Completed work:**

- Froze the numeric R1 gates in
  `docs/QWEN35_GRAFT_TRANSLATION_POC_PLAN.md` before any real corpus fit.
- Registered G0 identity, G1 key fidelity, G2 value fidelity, G3 binding, and
  fit-protocol requirements.

**Evidence:**

- `docs/QWEN35_GRAFT_TRANSLATION_POC_PLAN.md` records the R1 gates as
  `FROZEN 2026-07-02`.
- Registered fit protocol requires at least 2M paired train tokens,
  document-level 10% held-out split, and ridge lambda `1e-4`.

**Remaining work:**

- Run the R1 gates after full capture and fit artifacts exist.

### 2026-07-02 - Phase 4 G1/G2 Evaluator

**Status:** implementation smoke complete; full evaluation pending.

**Completed work:**

- Added `eval-translator` CLI.
- Implemented G1 key recall@k against native 9B query-to-key top-k sets.
- Implemented shuffled-key recall baseline.
- Implemented G2 value-output MSE and cosine under native 9B attention
  weights.
- Implemented wrong-layer key recall and wrong-layer value-output MSE/cosine.
- Implemented K-only and K+V translated-attention value-output controls.
- Added `eval-g0-capture-identity` CLI for the target-capture attention-plane
  identity floor.
- Added `g0-logit-smoke` CLI for the live 9B capture/reinject logit identity
  smoke.
- Fixed Qwen3.5 attention graft injection so `inject_kv`, `graft_seats`, and
  optional `live_shift` are consumed in `core/qwen35_tc.py`.

**Evidence:**

- Tiny real capture rehearsal with `--topk 8`:
  - `key_recall_at_8 = 1.0`
  - `shuffled_key_recall_at_8 = 0.20703125`
  - `value_output_cosine = 0.999999999999926`
- Synthetic paired-shard test now verifies:
  - G0 capture identity key recall and zero identity value MSE.
  - wrong-layer key and value controls are emitted.
  - V-only, K-only, and K+V output-control metrics are emitted.
  - Qwen3.5 attention exposes the GRM injection contract.
- `PYTHONPATH=. python3 scripts/qwen35_graft_translate_poc.py --help`
  lists both `eval-g0-capture-identity` and `g0-logit-smoke`.
- First real frozen-plan G0 capture identity smoke:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/gates/g0_capture_identity_smoke_train.json`
  - sha256
    `2e0d6e08e6f1c89751a150c9fb7d2945c036950933d6ce12348827015295307b`
  - target shards: `1`
  - every target layer reported `identity_key_recall_at_16 = 1.0`
  - every target layer reported `identity_value_output_mse = 0.0`
- First real frozen-plan one-shard translator/eval smoke:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/translator_smoke_chunk0`
  - `translator_manifest.json` sha256
    `ef3a2e9d649982438fd98ed4972431f2a9ac7ec26c4400fb03d2308500bc3d08`
  - `fit_metrics.json` sha256
    `30733e2d20ff0e8349b70e65fda7f11174fd0e8f69f8cf82dab03f4e6987445b`
  - `eval_metrics_train.json` sha256
    `56e41e63bdfe528772fda8f5e7fd5ab347150e107990cf1ef1a0a261a8d7e5c1`
  - paired shards: `1`
  - artifacts: `12`
  - layer alignment: `3->3`, `7->7`, `11->15`, `15->19`, `19->27`,
    `23->31`
  - eval layers: `6`
  - first-layer `key_recall_at_16 = 1.0`, shuffled baseline
    `0.0404052734375`, wrong-layer baseline `0.077239990234375`

**Remaining work:**

- Run G0 capture identity on the completed real target capture.
- Run live G0 logit identity smoke on a fixed held-out span.
- Full held-out corpus evaluation.

### 2026-07-02 - Phase 4 G3 Binding Probe Harness

**Status:** implementation complete; real binding gate pending.

**Completed work:**

- Added deterministic G3 binding probe generation.
- Added `make-binding-probes` CLI.
- Added `eval-binding-probes` CLI with these modes:
  - `amnesia`
  - `source-native`
  - `source-context`
  - `target-native`
  - `translated`
- Implemented gold-vs-decoy candidate logprob scoring.
- Implemented source-model fact harvest, target-model native fact harvest, and
  source-to-target translated graft scoring.
- Implemented translated capture reshaping from source KV width to target KV
  width using fitted translator artifacts.
- Added the positive-margin R1 summary field for the `14 / 32` translated
  binding threshold.
- Added a Qwen3.5 RoPE-extension regression guard so mounted graft seats extend
  the available RoPE table before scoring live tokens.

**Evidence:**

- Probe artifact:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/gates/binding_probes.json`
  - sha256
    `934b2bb14e16c3d54e43b4071832ba1e3237a485b58d40eb4733b084c66355ef`
  - count: `32`
  - seed: `qwen35-binding-v1`
- Runbook section:
  `docs/QWEN35_TRANSLATION_CORPUS_RUNBOOK.md`, "Evaluate G3 Binding Recall".
- Focused tests now cover deterministic probe generation, candidate margin
  scoring, harvested-capture translation shape/content, one-forward target
  K/V+Q capture, and the Qwen3.5 RoPE live-shift guard.

**Remaining work:**

- Run `eval-binding-probes` after the full real translator fit exists.
- Compare translated mode against `amnesia`, `source-native`, and
  `target-native` baselines.
- Record the resulting `binding_eval_metrics.json` hash and threshold result.

### 2026-07-02 - Full Pipeline Cron/Claude Handoff Runner

**Status:** implementation complete; long real run pending.

**Completed work:**

- Added `pipeline-next` CLI for one-stage-at-a-time orchestration.
- Added `pipeline-status` CLI for non-GPU inspection of the next stage.
- The runner writes `$POC/pipeline_status.json` after every invocation.
- The runner appends compact records to `$POC/pipeline_history.jsonl` after
  every real `pipeline-next` invocation.
- The stage order is:
  - source/target capture batches
  - G0 capture identity
  - live G0 logit smoke
  - normal train translator fit
  - wrong-layer, shuffled-docs, K-only, and V-only control fits
  - G1/G2 held-out translator evaluation
  - G3 binding probe generation
  - G3 binding evaluation
- Existing JSON artifacts are treated as completed stages and skipped.
- Translator artifacts are only skipped when their manifest schema, control
  mode, and fitted kinds match the expected stage.
- History rows include stage/status, key counters, output paths and hashes for
  file outputs, summaries, and capture expected counts without repeating full
  layer-shape payloads.
- Added `--skip-live-g0` and `--skip-binding-eval` for light operational
  rehearsals, while the default path still runs the full gate ladder.
- Updated the runbook cron section to prefer `pipeline-next` and keep
  `capture-next` as the capture-only fallback.
- Ran `pipeline-status --write-status` against the real PoC root to create the
  current status artifact before handoff.

**Evidence:**

- CLI help:
  `PYTHONPATH=. python3 scripts/qwen35_graft_translate_poc.py pipeline-next --help`
  lists root, model dirs, capture batch sizes, ridge lambda, top-k,
  binding modes, status/history path overrides, and skip switches.
- CLI help:
  `PYTHONPATH=. python3 scripts/qwen35_graft_translate_poc.py pipeline-status --help`
  lists root, artifact path overrides, skip switches, and `--write-status`.
- Focused tests cover:
  - capture-first behavior before post-capture work
  - G0 capture identity as the first post-capture stage
  - normal translator fit after G0 when live G0 is explicitly skipped
  - complete status when optional heavy gates are skipped and required
    artifacts exist
  - status inspection without calling model-running work
  - status-file writing when requested
  - append-only compact history creation for `pipeline-next`
- Runbook:
  `docs/QWEN35_TRANSLATION_CORPUS_RUNBOOK.md`, "Cron Notes".
- Current real status:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/pipeline_status.json`
  - sha256
    `8e4b81f172570ef5c777933c504000346e59141d998368607eeaf7560092f162`
  - status: `pending`
  - stage: `capture-source`
  - source completed chunks: `1 / 9861`
  - target completed chunks: `1 / 9861`
  - next missing chunk: `1`

**Remaining work:**

- Start the real `pipeline-next` loop.
- Let it complete full source capture, full target capture, fits, controls,
  and gates.
- Record each completed long-run stage and artifact hash in this ledger.

### 2026-07-02 - Test And Static Verification Gate

**Status:** complete for the current implementation smoke scope.

**Completed work:**

- Added focused tests for source validation, tokenizer mismatch rejection,
  real model config parsing, capture shard writing, corpus planning,
  capture-manifest progress accounting, ridge fit artifacts, evaluator
  metrics, one-forward target K/V+Q capture, G3 binding harness behavior, and
  `pipeline-next` orchestration.
- Re-ran existing DeepSeek static GRM hook test alongside the new PoC tests.
- Ran Python compile checks and whitespace/diff checks.

**Evidence:**

- `PYTHONPATH=.:/mnt/ForgeRealm/Project-Tensor/tensor_cuda PYTEST_ADDOPTS='-p no:cacheprovider' python3 -m pytest tests/test_qwen35_translation_poc.py tests/test_deepseek_grm_hooks_static.py -q`
  passed with `23 passed, 2 warnings in 0.33s` after the G3 binding harness,
  Qwen3.5 RoPE live-shift guard, frozen-corpus, target-smoke, capture-next,
  and first one-shard translator-smoke ledger/runbook updates.
- `PYTHONDONTWRITEBYTECODE=1 PYTEST_ADDOPTS='-p no:cacheprovider' python3 -m pytest tests/test_qwen35_translation_poc.py -q`
  passed with `23 passed, 2 warnings in 0.30s` after the `pipeline-next`
  orchestration update.
- `PYTHONDONTWRITEBYTECODE=1 PYTEST_ADDOPTS='-p no:cacheprovider' python3 -m pytest tests/test_qwen35_translation_poc.py tests/test_deepseek_grm_hooks_static.py -q`
  passed with `27 passed, 2 warnings in 0.37s` after the `pipeline-next`
  orchestration update.
- `PYTHONDONTWRITEBYTECODE=1 PYTEST_ADDOPTS='-p no:cacheprovider' python3 -m pytest tests/test_qwen35_translation_poc.py tests/test_deepseek_grm_hooks_static.py -q`
  passed with `29 passed, 2 warnings in 0.35s` after the `pipeline-status`
  audit update.
- `PYTHONDONTWRITEBYTECODE=1 PYTEST_ADDOPTS='-p no:cacheprovider' python3 -m pytest tests/test_qwen35_translation_poc.py tests/test_deepseek_grm_hooks_static.py -q`
  passed with `29 passed, 2 warnings in 0.37s` after the
  `pipeline_history.jsonl` append update.
- `PYTHONPYCACHEPREFIX=/tmp/qwen35_pycache PYTHONPATH=.:/mnt/ForgeRealm/Project-Tensor/tensor_cuda python3 -m py_compile core/qwen35_tc.py core/kv_graft.py core/qwen35_translation_poc.py scripts/qwen35_graft_translate_poc.py tests/test_qwen35_translation_poc.py`
  passed after the G0/control, Qwen3.5 injection, and `capture-next` updates.
- `PYTHONPYCACHEPREFIX=/tmp/grm_pycache python3 -m py_compile core/qwen35_translation_poc.py tests/test_qwen35_translation_poc.py core/qwen35_tc.py core/kv_graft.py`
  passed after the G3 binding harness updates.
- `PYTHONPYCACHEPREFIX=/tmp/grm_pycache_pipeline python3 -m py_compile core/qwen35_translation_poc.py tests/test_qwen35_translation_poc.py scripts/qwen35_graft_translate_poc.py`
  passed after the `pipeline-next` orchestration update.
- `PYTHONPYCACHEPREFIX=/tmp/grm_pycache_status_verify python3 -m py_compile core/qwen35_translation_poc.py tests/test_qwen35_translation_poc.py scripts/qwen35_graft_translate_poc.py core/qwen35_tc.py core/kv_graft.py`
  passed after the `pipeline-status` audit update.
- `PYTHONPYCACHEPREFIX=/tmp/grm_pycache_history_verify python3 -m py_compile core/qwen35_translation_poc.py tests/test_qwen35_translation_poc.py scripts/qwen35_graft_translate_poc.py core/qwen35_tc.py core/kv_graft.py`
  passed after the `pipeline_history.jsonl` append update.
- `git diff --check` passed after the G0/control and Qwen3.5 injection
  updates, and again after the frozen-corpus, target-smoke, `capture-next`,
  and one-shard translator-smoke ledger/runbook updates.
- `git diff --check` passed again after the G3 binding harness and runbook
  updates.
- `git diff --check` passed again after the `pipeline-next` orchestration
  update.
- `git diff --check` passed again after the `pipeline-status` audit update.
- `git diff --check` passed again after the `pipeline_history.jsonl` append
  update.

**Remaining work:**

- Re-run this verification gate after the next implementation change.

### 2026-07-03 - Partial Corpus Readiness And G0 Identity Fast Path

**Status:** complete.

**Completed work:**

- Added paired source/target capture accounting to `capture_manifest.json`,
  `pipeline_status.json`, pipeline history records, and the capture CLI JSON
  summaries.
- Paired accounting now records total paired shards/tokens, same-split
  shards/tokens, per-split paired train/held-out tokens, source-only chunks,
  target-only chunks, token-count mismatches, and split mismatches.
- Replaced `eval-g0-capture-identity`'s redundant native-vs-native attention
  recomputation with a structural exact-identity path. The gate still validates
  target K/V/Q presence, shape compatibility, query/KV head divisibility, and
  finite tensor contents, but reports identity recall/cosine/MSE without the
  previous quadratic attention pass.
- Updated the implementation plan status so preview work cannot be confused
  with final gate artifacts.

**Evidence:**

- Refreshed real pipeline status:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/pipeline_status.json`
  - timestamp UTC: `2026-07-04T03:06:32.426783+00:00`
  - stage: `capture-target`
  - source capture: `9861 / 9861` chunks complete
  - target capture: `8552 / 9861` chunks complete
  - paired chunks: `8552`
  - paired tokens: `2,183,723`
  - paired train tokens: `1,944,308`
  - paired held-out tokens: `239,415`
  - source-only chunks still waiting on target: `1309`
  - token-count mismatches: `0`
  - split mismatches: `0`
- Command:
  `PYTHONPATH=.:/mnt/ForgeRealm/Project-Tensor/tensor_cuda python3 scripts/qwen35_graft_translate_poc.py pipeline-status --root /mnt/ForgeRealm/qwen35_graft_translation_poc --write-status`
- Focused test gate:
  `PYTHONDONTWRITEBYTECODE=1 PYTEST_ADDOPTS='-p no:cacheprovider' python3 -m pytest tests/test_qwen35_translation_poc.py -q`
  passed with `26 passed, 2 warnings in 0.42s`.
- Compile gate:
  `PYTHONPYCACHEPREFIX=/tmp/qwen35_preview_pycache python3 -m py_compile core/qwen35_translation_poc.py tests/test_qwen35_translation_poc.py scripts/qwen35_graft_translate_poc.py`
  passed.

**Remaining work:**

- Continue the target capture loop to completion.
- The final train fit still waits for the frozen `>= 2M` paired train-token
  gate; current paired train tokens are `1,944,308`.
- Partial preview fits/evals are now easier to justify from status, but they
  must use separate preview output directories and must not populate the final
  `translator/`, `translator_*`, or `gates/*_metrics.json` artifacts.

### 2026-07-04 - Full Corpus Capture And G0 Capture Identity

**Status:** complete.

**Completed work:**

- Refreshed the real pipeline status after the overnight capture run finished.
- Confirmed both source and target captures are complete on the frozen corpus.
- Confirmed the frozen `>= 2M` paired train-token fit gate is satisfied.
- Ran the full held-out G0 capture identity stage on the completed target
  capture.
- Updated `docs/QWEN35_GRAFT_TRANSLATION_POC_PLAN.md` so Phase 2 and the G0
  capture-identity gate reflect the current artifact state.

**Evidence:**

- Refreshed status command:
  `PYTHONPATH=.:/mnt/ForgeRealm/Project-Tensor/tensor_cuda python3 scripts/qwen35_graft_translate_poc.py pipeline-status --root /mnt/ForgeRealm/qwen35_graft_translation_poc --write-status`
- Pipeline status:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/pipeline_status.json`
  - sha256:
    `c2afb8df95bfd6c2d389b673b8050ddc49742f3f2e62c25a54a58792c896cc03`
  - timestamp UTC: `2026-07-04T15:28:31.205294+00:00`
  - status: `pending`
  - stage: `g0-logit-smoke`
  - source capture: `9861 / 9861` chunks complete
  - target capture: `9861 / 9861` chunks complete
  - paired chunks: `9861`
  - paired tokens: `2,500,000`
  - paired train tokens: `2,245,444`
  - paired held-out tokens: `254,556`
  - source-only chunks: `0`
  - target-only chunks: `0`
  - token-count mismatches: `0`
  - split mismatches: `0`
- Full held-out G0 capture identity artifact:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/gates/g0_capture_identity_metrics.json`
  - sha256:
    `5643509b8baa8a81e5bf13e66149a2841fc37413a0c4b42424632b3a7a56cf3e`
  - target shards: `1006`
  - held-out tokens per layer: `254,556`
  - all target attention layers reported
    `identity_key_recall_at_16 = 1.0`
  - all target attention layers reported
    `identity_value_output_mse = 0.0`
  - all target attention layers reported
    `identity_value_output_cosine = 1.0`
  - non-finite tensors: `0`
- Pipeline history appended the `g0-capture-identity` success row with the same
  output hash.

**Remaining work:**

- Run live G0 logit identity smoke.
- Run the final train fit and registered fit controls.
- Run held-out G1/G2 evaluation and G3 binding evaluation.
- Produce the final write-up with the surviving claim level.

### 2026-07-04 - Live G0 Logit Identity Smoke

**Status:** complete with threshold failure.

**Completed work:**

- Ran the live 9B capture/reinject logit identity smoke on the first eligible
  held-out span selected by the pipeline.
- Confirmed the stage writes a valid metrics artifact and advances the pipeline
  to the final train translator fit.
- Checked the live injection path for the obvious RoPE offset failure mode. The
  Qwen3.5 attention path applies the mounted graft-seat shift while the graft is
  resident, so this result is recorded as the observed live G0 noise-floor
  result, not as a confirmed offset bug.
- Updated `docs/QWEN35_GRAFT_TRANSLATION_POC_PLAN.md` with the live G0 result.

**Evidence:**

- Pipeline command:
  `PYTHONPATH=.:/mnt/ForgeRealm/Project-Tensor/tensor_cuda python3 scripts/qwen35_graft_translate_poc.py pipeline-next --root /mnt/ForgeRealm/qwen35_graft_translation_poc --source-model-dir /home/vader/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc --target-model-dir /home/vader/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a --layers all --source-max-chunks 64 --target-max-chunks 16 --ridge-lambda 1e-4 --topk 16 --binding-max-probes 32`
- Artifact:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/gates/g0_logit_identity_smoke.json`
  - sha256:
    `0d82724b9be61082d8a349f0edbd1b4b7c8d5c97b1a420ff00a2c871d1da0942`
  - held-out span: document `f0c0d0313f5423aa`, chunk `1989`
  - prefix tokens: `64`
  - probe tokens: `8`
  - max abs delta: `0.1875`
  - mean abs delta: `0.019217236981522855`
  - top-1 flips: `0`
  - top-1 flip rate: `0.0`
  - frozen thresholds: max abs delta `0.002`, top-1 flip rate `0.001`
  - `passes_r1_g0_threshold = false`
- Refreshed pipeline status:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/pipeline_status.json`
  - sha256:
    `0ef0a868a3f446a60dfd4698d23689ec1952e772881e5ed0cbe5d5cdf461a054`
  - status: `pending`
  - stage: `fit-translator`

**Remaining work:**

- Treat downstream fit/eval metrics as still useful, but interpret them with
  the failed live G0 max-abs-delta floor in the final write-up.
- Run the final train fit and registered fit controls.
- Run held-out G1/G2 evaluation and G3 binding evaluation.
- Produce the final write-up with the surviving claim level.

### 2026-07-04 - Final Train Translator Fit

**Status:** complete.

**Completed work:**

- Ran the registered normal train translator fit on the completed paired train
  capture.
- Produced full-width K and V ridge maps for all six source attention layers.
- Wrote final translator artifacts under the reserved production
  `translator/` path.
- Refreshed pipeline status; the next stage is the wrong-layer control fit.
- Updated `docs/QWEN35_GRAFT_TRANSLATION_POC_PLAN.md` with the real fit
  metrics summary.

**Evidence:**

- Pipeline command:
  `PYTHONPATH=.:/mnt/ForgeRealm/Project-Tensor/tensor_cuda python3 scripts/qwen35_graft_translate_poc.py pipeline-next --root /mnt/ForgeRealm/qwen35_graft_translation_poc --source-model-dir /home/vader/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc --target-model-dir /home/vader/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a --layers all --source-max-chunks 64 --target-max-chunks 16 --ridge-lambda 1e-4 --topk 16 --binding-max-probes 32`
- Output directory:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/translator`
- Translator manifest:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/translator/translator_manifest.json`
  - sha256:
    `3c1ce5dd49b5a2cc8b3fbe5b13cf7b1c1505473228001bca7bb1791c3de667f4`
  - paired train shards: `8855`
  - artifacts: `12`
  - ridge lambda: `1e-4`
  - layer alignment: `3->3`, `7->7`, `11->15`, `15->19`, `19->27`,
    `23->31`
  - map shape for every artifact: `512 -> 1024`
- Fit metrics:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/translator/fit_metrics.json`
  - sha256:
    `0ceddf1c44446a1f8a6e84f18a0410d317f4fe3eb22d1c799ecc5b4558dfddbd`
  - train tokens per map: `2,245,444`
  - K mean-row cosine range: `0.853317589093372` to
    `0.9101047250483691`
  - V mean-row cosine range: `0.7531574545045864` to
    `0.9686707426624085`
  - K R² range: `0.5398957487107319` to `0.7053624018146802`
  - V R² range: `0.4294227661704709` to `0.6218307136019108`
- Pipeline status after refresh:
  - status: `pending`
  - stage: `fit-control-wrong-layer`

**Remaining work:**

- Run the registered negative-control fits.
- Run held-out G1/G2 evaluation and G3 binding evaluation.
- Produce the final write-up with the surviving claim level.

### 2026-07-04 - Wrong-Layer Control Fit

**Status:** complete.

**Completed work:**

- Ran the registered wrong-layer negative-control fit on the completed paired
  train capture.
- Produced full-width K and V ridge maps for all six shifted source-target
  layer pairs.
- Wrote control artifacts under `translator_wrong_layer/`.
- Updated `docs/QWEN35_GRAFT_TRANSLATION_POC_PLAN.md` with the wrong-layer
  control metrics summary.

**Evidence:**

- Pipeline command:
  `PYTHONPATH=.:/mnt/ForgeRealm/Project-Tensor/tensor_cuda python3 scripts/qwen35_graft_translate_poc.py pipeline-next --root /mnt/ForgeRealm/qwen35_graft_translation_poc --source-model-dir /home/vader/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc --target-model-dir /home/vader/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a --layers all --source-max-chunks 64 --target-max-chunks 16 --ridge-lambda 1e-4 --topk 16 --binding-max-probes 32`
- Output directory:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/translator_wrong_layer`
- Translator manifest:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/translator_wrong_layer/translator_manifest.json`
  - sha256:
    `06cb6ea4ad6bc1e3821a761ac66083168a524883f2aa0ebbe19167649371f782`
  - paired train shards: `8855`
  - artifacts: `12`
  - ridge lambda: `1e-4`
  - control: `wrong-layer`
  - layer alignment: `3->7`, `7->11`, `11->19`, `15->23`, `19->31`,
    `23->3`
  - map shape for every artifact: `512 -> 1024`
- Fit metrics:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/translator_wrong_layer/fit_metrics.json`
  - sha256:
    `0cc871b64721660a637fd7339272080adb7531a42534b6c9b57850f242c7ccb3`
  - train tokens per map: `2,245,444`
  - K mean-row cosine range: `0.8434169991108665` to
    `0.9043375200116528`
  - V mean-row cosine range: `0.6954730362469482` to
    `0.9663842353437266`
  - K R² range: `0.4668783736232488` to `0.6723707570628195`
  - V R² range: `0.32655635155106133` to `0.542841570424957`
- Pipeline history appended `fit-control-wrong-layer` success at
  `2026-07-04T17:10:00.862740+00:00`.

**Remaining work:**

- Run the shuffled-docs, K-only, and V-only control fits.
- Run held-out G1/G2 evaluation and G3 binding evaluation.
- Produce the final write-up with the surviving claim level.

### 2026-07-04 - Shuffled-Docs Control Fit

**Status:** complete.

**Completed work:**

- Ran the registered shuffled-docs negative-control fit on the completed paired
  train capture.
- Produced full-width K and V ridge maps for all six normal layer-aligned pairs
  while shifting target documents out of correspondence with source documents.
- Wrote control artifacts under `translator_shuffled_docs/`.
- Updated `docs/QWEN35_GRAFT_TRANSLATION_POC_PLAN.md` with the shuffled-docs
  control metrics summary.

**Evidence:**

- Pipeline command:
  `PYTHONPATH=.:/mnt/ForgeRealm/Project-Tensor/tensor_cuda python3 scripts/qwen35_graft_translate_poc.py pipeline-next --root /mnt/ForgeRealm/qwen35_graft_translation_poc --source-model-dir /home/vader/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc --target-model-dir /home/vader/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a --layers all --source-max-chunks 64 --target-max-chunks 16 --ridge-lambda 1e-4 --topk 16 --binding-max-probes 32`
- Output directory:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/translator_shuffled_docs`
- Translator manifest:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/translator_shuffled_docs/translator_manifest.json`
  - sha256:
    `8dba1aea9ae1c6aa1bd364cbb707a44282344af705d17bac26d2ae7a1fa68533`
  - paired train shards: `8855`
  - artifacts: `12`
  - ridge lambda: `1e-4`
  - control: `shuffled-docs`
  - layer alignment: `3->3`, `7->7`, `11->15`, `15->19`, `19->27`,
    `23->31`
  - map shape for every artifact: `512 -> 1024`
- Fit metrics:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/translator_shuffled_docs/fit_metrics.json`
  - sha256:
    `bdd62b4e8519fdc53d2215d9027afc80874aa1827da9fc092cf135cee3a02228`
  - train tokens per map: `2,224,578`
  - K mean-row cosine range: `0.624824523027121` to
    `0.724112890090573`
  - V mean-row cosine range: `0.3790888737004818` to
    `0.9293803757763192`
  - K R² range: `0.02217365607069377` to `0.04097642162002901`
  - V R² range: `0.00634158323804046` to `0.042071547620968874`
- Pipeline history appended `fit-control-shuffled-docs` success at
  `2026-07-04T19:14:38.926439+00:00`.

**Remaining work:**

- Run the K-only and V-only control fits.
- Run held-out G1/G2 evaluation and G3 binding evaluation.
- Produce the final write-up with the surviving claim level.

### 2026-07-04 - K-Only Control Fit

**Status:** complete.

**Completed work:**

- Ran the registered K-only control fit on the completed paired train capture.
- Produced full-width K ridge maps for all six normal layer-aligned pairs.
- Wrote control artifacts under `translator_k_only/`.
- Updated `docs/QWEN35_GRAFT_TRANSLATION_POC_PLAN.md` with the K-only control
  metrics summary.

**Evidence:**

- Pipeline command:
  `PYTHONPATH=.:/mnt/ForgeRealm/Project-Tensor/tensor_cuda python3 scripts/qwen35_graft_translate_poc.py pipeline-next --root /mnt/ForgeRealm/qwen35_graft_translation_poc --source-model-dir /home/vader/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc --target-model-dir /home/vader/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a --layers all --source-max-chunks 64 --target-max-chunks 16 --ridge-lambda 1e-4 --topk 16 --binding-max-probes 32`
- Output directory:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/translator_k_only`
- Translator manifest:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/translator_k_only/translator_manifest.json`
  - sha256:
    `4d3bfbd91eb4b3c252af0ef53c72283ecb6ec222fb16034fdbe05d4ee3fc1b5f`
  - paired train shards: `8855`
  - artifacts: `6`
  - ridge lambda: `1e-4`
  - kinds: `k`
  - layer alignment: `3->3`, `7->7`, `11->15`, `15->19`, `19->27`,
    `23->31`
  - map shape for every artifact: `512 -> 1024`
- Fit metrics:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/translator_k_only/fit_metrics.json`
  - sha256:
    `6c17e4c353af7e8b7574774358b22bb051faa3eb4de531575fad8c77d95f8e36`
  - train tokens per map: `2,245,444`
  - K mean-row cosine range: `0.853317589093372` to
    `0.9101047250483691`
  - K R² range: `0.5398957487107319` to `0.7053624018146802`
- Pipeline history appended `fit-control-k-only` success at
  `2026-07-04T19:43:10.581128+00:00`.

**Remaining work:**

- Run the V-only control fit.
- Run held-out G1/G2 evaluation and G3 binding evaluation.
- Produce the final write-up with the surviving claim level.

### 2026-07-04 - V-Only Control Fit

**Status:** complete.

**Completed work:**

- Ran the registered V-only control fit on the completed paired train capture.
- Produced full-width V ridge maps for all six normal layer-aligned pairs.
- Wrote control artifacts under `translator_v_only/`.
- Updated `docs/QWEN35_GRAFT_TRANSLATION_POC_PLAN.md` with the V-only control
  metrics summary.

**Evidence:**

- Pipeline command:
  `PYTHONPATH=.:/mnt/ForgeRealm/Project-Tensor/tensor_cuda python3 scripts/qwen35_graft_translate_poc.py pipeline-next --root /mnt/ForgeRealm/qwen35_graft_translation_poc --source-model-dir /home/vader/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc --target-model-dir /home/vader/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a --layers all --source-max-chunks 64 --target-max-chunks 16 --ridge-lambda 1e-4 --topk 16 --binding-max-probes 32`
- Output directory:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/translator_v_only`
- Translator manifest:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/translator_v_only/translator_manifest.json`
  - sha256:
    `8e49a7bf0306e268762a0bb43758c141025891764c77152551eb528d8ca727e7`
  - paired train shards: `8855`
  - artifacts: `6`
  - ridge lambda: `1e-4`
  - kinds: `v`
  - layer alignment: `3->3`, `7->7`, `11->15`, `15->19`, `19->27`,
    `23->31`
  - map shape for every artifact: `512 -> 1024`
- Fit metrics:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/translator_v_only/fit_metrics.json`
  - sha256:
    `c7a38d85b965c74e222e7379676cfbbab287b6bc6f4dd53c1f6dfa27ef02ec76`
  - train tokens per map: `2,245,444`
  - V mean-row cosine range: `0.7531574545045864` to
    `0.9686707426624085`
  - V R² range: `0.4294227661704709` to `0.6218307136019108`
- Pipeline history appended `fit-control-v-only` success.

**Remaining work:**

- Run held-out G1/G2 evaluation and G3 binding evaluation.
- Produce the final write-up with the surviving claim level.

### 2026-07-04 - G1/G2 Held-Out Evaluation

**Status:** complete.

**Completed work:**

- Started the full held-out `eval-translator` gate and found the evaluator was
  operationally too slow and blind: the first interrupted run spent over an
  hour in the Python-set top-k recall path, and the next interrupted run exposed
  the value-output `np.einsum` path as the next hot spot.
- Replaced `_topk_recall` row-wise Python set intersections with vectorized
  top-k membership comparison.
- Replaced repeated attention-score/value-output `np.einsum` calls with batched
  `np.matmul` helpers.
- Added `eval_metrics_progress.json` sidecar writes so the full held-out gate
  reports progress every 25 shards and marks completion.
- Added regression coverage for the vectorized top-k recall semantics and the
  progress sidecar completion contract.
- Ran the full held-out G1/G2 evaluator on all `1006` paired held-out shards.
- Updated `docs/QWEN35_GRAFT_TRANSLATION_POC_PLAN.md` with the G1/G2 result
  and threshold interpretation.

**Evidence:**

- Focused test command:
  `PYTHONDONTWRITEBYTECODE=1 PYTEST_ADDOPTS='-p no:cacheprovider' python3 -m pytest tests/test_qwen35_translation_poc.py -q`
  - result: `27 passed, 2 warnings in 0.40s`
- Pipeline command:
  `PYTHONPATH=.:/mnt/ForgeRealm/Project-Tensor/tensor_cuda python3 scripts/qwen35_graft_translate_poc.py pipeline-next --root /mnt/ForgeRealm/qwen35_graft_translation_poc --source-model-dir /home/vader/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc --target-model-dir /home/vader/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a --layers all --source-max-chunks 64 --target-max-chunks 16 --ridge-lambda 1e-4 --topk 16 --binding-max-probes 32`
- Eval artifact:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/translator/eval_metrics.json`
  - sha256:
    `c42847747374bb28b5b033d2a203d91dd6e14cef03f572eca5a0ff54541bfa9a`
  - schema: `qwen35_graft_translation_eval_metrics_v1`
  - split: `heldout`
  - top-k: `16`
  - paired held-out shards: `1006`
  - layer-pairs: `6`
  - held-out tokens per layer-pair: `254,556`
- Progress artifact:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/translator/eval_metrics_progress.json`
  - sha256:
    `b7d243c10e63cc28cf1b43270e0cc42a0896a81bb16b3bf78280694a55ac9d7a`
  - status: `complete`
  - completed shards: `1006 / 1006`
- Refreshed pipeline status:
  `/mnt/ForgeRealm/qwen35_graft_translation_poc/pipeline_status.json`
  - sha256:
    `5aa9dfc2a9ec55572830677070a7e9ea906399a1828bd0615552fca06179bcc4`
  - status: `pending`
  - stage: `eval-binding-probes`
  - eval translator ready: `true`
  - binding probes ready: `true`
  - binding eval ready: `false`
- Per-layer G1/G2 metrics:
  - `3->3`: key recall@16 `0.5634406672489058`, shuffled
    `0.052714496918280486`, wrong-key `0.06567855121608031`, value cosine
    `0.9541536472807478`, value MSE `0.01571993042051494`,
    wrong-layer MSE `0.3295104098855734`, MSE ratio `0.0477069310980915`
  - `7->7`: key recall@16 `0.6377201221386092`, shuffled
    `0.04790550240723258`, wrong-key `0.06800959603190003`, value cosine
    `0.935210640322044`, value MSE `0.035995265382280754`,
    wrong-layer MSE `0.4463914653554896`, MSE ratio `0.08063609673544152`
  - `11->15`: key recall@16 `0.6049831578131571`, shuffled
    `0.047558302250840366`, wrong-key `0.07770246176812656`, value cosine
    `0.9117545860854634`, value MSE `0.0651071400924639`,
    wrong-layer MSE `0.5137107300754095`, MSE ratio `0.12673891410231317`
  - `15->19`: key recall@16 `0.6509739943947752`, shuffled
    `0.049727652567451604`, wrong-key `0.064330061446846`, value cosine
    `0.9094160406409361`, value MSE `0.11563002535071353`,
    wrong-layer MSE `0.7987326071678881`, MSE ratio `0.1447668773166899`
  - `19->27`: key recall@16 `0.6795158410372619`, shuffled
    `0.05091407734735717`, wrong-key `0.0572976236001304`, value cosine
    `0.980642178117937`, value MSE `0.15128897849147033`,
    wrong-layer MSE `8.979997710147344`, MSE ratio `0.016847329295030296`
  - `23->31`: key recall@16 `0.6904513192234587`, shuffled
    `0.05274420215654368`, wrong-key `0.07433895868766034`, value cosine
    `0.9932101022103578`, value MSE `0.2426036897488896`,
    wrong-layer MSE `20.320128113361747`, MSE ratio
    `0.01193908268665701`
- Frozen gate interpretation:
  - G1 key recall@16 average: `0.637847516976028`
  - minimum key/shuffled ratio: `10.688533519012191`
  - G1 status: pass by average recall `>= 0.60` and every band `>= 3x`
    shuffled. The first band is below `0.60` individually, but the frozen gate
    used average recall plus per-band 3x shuffled.
  - G2 value-output cosine range: `0.9094160406409361` to
    `0.9932101022103578`
  - G2 translated/wrong-layer MSE ratio range: `0.01193908268665701` to
    `0.1447668773166899`
  - G2 status: pass by every band cosine `>= 0.90` and every band MSE
    `<= 25%` of wrong-layer baseline.

**Remaining work:**

- Run 2B-native and 9B-native binding baselines plus translated G3 binding
  evaluation.
- Produce the final write-up with the surviving claim level.

## Open Completion Queue

These items are not complete and must stay visible until closed:

1. Run 2B-native and 9B-native binding baselines.
2. Run G3 binding probe eval on the completed real translator.
3. Produce the final write-up with the surviving claim level.
