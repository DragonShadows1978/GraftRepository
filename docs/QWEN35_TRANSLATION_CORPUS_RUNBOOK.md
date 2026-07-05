# Qwen3.5 Translation PoC Corpus Runbook

This runbook starts the corpus phase for the Qwen3.5-2B to Qwen3.5-9B
attention-graft translator. It is designed for resumable cron/Claude runs:
completed shards are skipped, and `capture_manifest.json` is refreshed after
each batch.

The easiest cron handoff is `capture-next`: it runs source batches until source
is complete, then target batches until target is complete, then exits with
`status: complete`.

The more complete cron handoff is `pipeline-next`: it runs one missing stage
per invocation. It starts with capture batches, then proceeds through G0,
translator fitting, control fits, G1/G2 evaluation, binding probe generation,
and G3 binding evaluation once prerequisites exist. It writes
`$POC/pipeline_status.json` after every invocation and appends a compact record
to `$POC/pipeline_history.jsonl` for every real `pipeline-next` run.

After each completed step or batch milestone, update
`docs/QWEN35_TRANSLATION_IMPLEMENTATION_LEDGER.md` with what completed, the
artifact paths, and the next unfinished item.

## Paths

```bash
cd /mnt/ForgeRealm/GraftRepository

export POC=/mnt/ForgeRealm/qwen35_graft_translation_poc
export Q35_2B=/home/vader/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc
export Q35_9B=/home/vader/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
export TCPATH=/mnt/ForgeRealm/Project-Tensor/tensor_cuda
```

## 1. Plan The Corpus

The current frozen plan is already written at `$POC/corpus_plan.json`.
Regenerate it only if intentionally changing the corpus. The split is
document-level, not token-level.

```bash
PYTHONPATH=. python3 scripts/qwen35_graft_translate_poc.py plan-corpus \
  --model-dir "$Q35_2B" \
  --corpus /mnt/ForgeRealm/scribe_mint_v1/manifest.jsonl \
  --corpus /mnt/ForgeRealm/HumanBaselineCorpus/stories \
  --out "$POC/corpus_plan.json" \
  --chunk-tokens 256 \
  --heldout-fraction 0.10 \
  --seed qwen35-translation-poc-0 \
  --max-total-tokens 2500000 \
  --min-doc-tokens 64 \
  --source-label scribe-mint-plus-humanbaseline-v1
```

Frozen corpus summary:

- `corpus_plan.json` sha256:
  `ed1b8c0592edb31d007561224994d109a66d46de80d8157e1e8b284219783543`
- documents: `226`
- chunks: `9861`
- tokens: `2500000`
- train tokens: `2245444`
- held-out tokens: `254556`

## 2. Capture Source 2B

Run this repeatedly until `source.complete` is `true` in
`$POC/captures/capture_manifest.json`.

Chunk `0` is already complete from the first live smoke. The command below is
resumable and will skip it.

```bash
PYTHONPATH=.:$TCPATH python3 scripts/qwen35_graft_translate_poc.py capture-corpus \
  --plan "$POC/corpus_plan.json" \
  --model-dir "$Q35_2B" \
  --role source \
  --out-dir "$POC/captures" \
  --layers all \
  --max-chunks 64
```

## 3. Capture Target 9B

Run after source capture is complete. Target shards include 9B queries.
Target capture records K/V and queries in one forward pass per chunk.

```bash
PYTHONPATH=.:$TCPATH python3 scripts/qwen35_graft_translate_poc.py capture-corpus \
  --plan "$POC/corpus_plan.json" \
  --model-dir "$Q35_9B" \
  --role target \
  --out-dir "$POC/captures" \
  --layers all \
  --max-chunks 16
```

## 4. Check Progress

```bash
PYTHONPATH=. python3 scripts/qwen35_graft_translate_poc.py capture-status \
  --plan "$POC/corpus_plan.json" \
  --out-dir "$POC/captures"
```

The corpus phase is ready for fitting when both are true:

- `expected.source.complete == true`
- `expected.target.complete == true`

The status output also reports `remaining_chunks` and `next_missing_chunk` for
each role. It also reports `paired`, which is the current source/target overlap
that fit/eval commands can use. The final registered train fit still waits for
both captures to complete and for the frozen `>= 2M` paired train-token gate.
Partial preview fits are allowed only in separate preview output directories;
do not write partial artifacts to `$POC/translator`, `$POC/translator_*`, or
`$POC/gates/*_metrics.json`.

Current frozen-plan status refreshed on 2026-07-04 UTC:

- source completed chunks: `9861 / 9861`
- target completed chunks: `8552 / 9861`
- paired chunks: `8552`
- paired train tokens: `1,944,308`
- paired held-out tokens: `239,415`
- source-only chunks still waiting on target: `1309`
- token-count mismatches: `0`
- split mismatches: `0`

## 5. Run G0 Capture Identity

This is the cheap target-capture identity floor for the attention-plane gates.
It should report identity key recall at `1.0`, identity value MSE at `0.0`,
and identity value cosine near `1.0`.

```bash
PYTHONPATH=. python3 scripts/qwen35_graft_translate_poc.py eval-g0-capture-identity \
  --capture-dir "$POC/captures" \
  --out "$POC/gates/g0_capture_identity_metrics.json" \
  --split heldout \
  --topk 16
```

Optional live logit smoke for the R1 G0 threshold:

```bash
PYTHONPATH=.:$TCPATH python3 scripts/qwen35_graft_translate_poc.py g0-logit-smoke \
  --model-dir "$Q35_9B" \
  --prefix-token-ids "1,2,3,4" \
  --probe-token-ids "5,6" \
  --layers all \
  --out "$POC/gates/g0_logit_identity_smoke.json"
```

Replace the toy token ids with a fixed short span from the held-out plan before
recording the real G0 result.

## 6. Fit The Translator

Before the first real fit, confirm the frozen numeric R1 gate thresholds in
`docs/QWEN35_GRAFT_TRANSLATION_POC_PLAN.md` are still unchanged. Also confirm
`pipeline_status.json` reports at least `2,000,000` paired train tokens and a
complete target capture before writing the final `$POC/translator` artifact.

```bash
PYTHONPATH=. python3 scripts/qwen35_graft_translate_poc.py fit-translator \
  --capture-dir "$POC/captures" \
  --out-dir "$POC/translator" \
  --ridge-lambda 1e-4 \
  --split train
```

This writes:

- `$POC/translator/translator_manifest.json`
- `$POC/translator/fit_metrics.json`
- `$POC/translator/translator_l*_to_l*_{k,v}.npz`

Fit the registered negative-control artifacts into separate directories:

```bash
PYTHONPATH=. python3 scripts/qwen35_graft_translate_poc.py fit-translator \
  --capture-dir "$POC/captures" \
  --out-dir "$POC/translator_wrong_layer" \
  --ridge-lambda 1e-4 \
  --split train \
  --control wrong-layer

PYTHONPATH=. python3 scripts/qwen35_graft_translate_poc.py fit-translator \
  --capture-dir "$POC/captures" \
  --out-dir "$POC/translator_shuffled_docs" \
  --ridge-lambda 1e-4 \
  --split train \
  --control shuffled-docs

PYTHONPATH=. python3 scripts/qwen35_graft_translate_poc.py fit-translator \
  --capture-dir "$POC/captures" \
  --out-dir "$POC/translator_k_only" \
  --ridge-lambda 1e-4 \
  --split train \
  --kinds k

PYTHONPATH=. python3 scripts/qwen35_graft_translate_poc.py fit-translator \
  --capture-dir "$POC/captures" \
  --out-dir "$POC/translator_v_only" \
  --ridge-lambda 1e-4 \
  --split train \
  --kinds v
```

## 7. Evaluate G1/G2 And Controls

```bash
PYTHONPATH=. python3 scripts/qwen35_graft_translate_poc.py eval-translator \
  --capture-dir "$POC/captures" \
  --translator-dir "$POC/translator" \
  --out "$POC/translator/eval_metrics.json" \
  --split heldout \
  --topk 16
```

This reports key recall against the native 9B attention top-k set, a shuffled
key baseline, wrong-layer key recall, native-attention V-only value fidelity,
translated-attention K-only value fidelity, translated K+V output fidelity, and
wrong-layer value-output MSE/cosine.

## 8. Evaluate G3 Binding Recall

The fixed R1 binding-probe set is already written:

- `$POC/gates/binding_probes.json`
- sha256:
  `934b2bb14e16c3d54e43b4071832ba1e3237a485b58d40eb4733b084c66355ef`
- count: `32`
- seed: `qwen35-binding-v1`

Regenerate it only if intentionally changing the registered probe set:

```bash
PYTHONPATH=. python3 scripts/qwen35_graft_translate_poc.py make-binding-probes \
  --out "$POC/gates/binding_probes.json" \
  --count 32 \
  --seed qwen35-binding-v1
```

Run this after the real translator fit exists:

```bash
PYTHONPATH=.:$TCPATH python3 scripts/qwen35_graft_translate_poc.py eval-binding-probes \
  --probes "$POC/gates/binding_probes.json" \
  --source-model-dir "$Q35_2B" \
  --target-model-dir "$Q35_9B" \
  --translator-dir "$POC/translator" \
  --out "$POC/gates/binding_eval_metrics.json" \
  --modes amnesia,source-native,target-native,translated \
  --max-probes 32 \
  --layers all
```

The G3 summary reports positive gold-minus-best-decoy margins. The frozen R1
threshold is at least `14 / 32` positive margins for translated grafts, with
native source and native target modes used as sanity baselines.

## Cron Notes

Use a lock if running from cron so a slow 9B batch or gate cannot overlap the
next interval.

Preferred full pipeline loop:

```bash
flock -n /tmp/qwen35_translation_pipeline.lock bash -lc 'cd /mnt/ForgeRealm/GraftRepository && PYTHONPATH=.:/mnt/ForgeRealm/Project-Tensor/tensor_cuda python3 scripts/qwen35_graft_translate_poc.py pipeline-next --root /mnt/ForgeRealm/qwen35_graft_translation_poc --source-model-dir /home/vader/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc --target-model-dir /home/vader/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a --layers all --source-max-chunks 64 --target-max-chunks 16 --ridge-lambda 1e-4 --topk 16 --binding-max-probes 32 --skip-fit-metrics'
```

For larger corpus runs, `--skip-fit-metrics` avoids a second full train-shard
read after solving the translator. Use held-out G1/G2 and binding gates as the
selection metrics.

The current `pipeline-next` stage can be inspected with:

```bash
PYTHONPATH=. python3 scripts/qwen35_graft_translate_poc.py pipeline-status \
  --root "$POC" \
  --write-status

cat "$POC/pipeline_status.json"
```

After the first `pipeline-next` invocation, the append-only stage history can
be inspected with:

```bash
tail -n 20 "$POC/pipeline_history.jsonl"
```

Current real-root status:

- next stage: `capture-target`
- source completed chunks: `9861 / 9861`
- target completed chunks: `8552 / 9861`
- next missing target chunk: `8552`
- paired train tokens: `1,944,308`
- paired held-out tokens: `239,415`
- `pipeline_history.jsonl` is created by the first real `pipeline-next`
  invocation; it is not created by passive `pipeline-status` inspection.

Capture-only fallback:

```bash
flock -n /tmp/qwen35_translation_capture.lock bash -lc 'cd /mnt/ForgeRealm/GraftRepository && PYTHONPATH=.:/mnt/ForgeRealm/Project-Tensor/tensor_cuda python3 scripts/qwen35_graft_translate_poc.py capture-next --plan /mnt/ForgeRealm/qwen35_graft_translation_poc/corpus_plan.json --source-model-dir /home/vader/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc --target-model-dir /home/vader/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a --out-dir /mnt/ForgeRealm/qwen35_graft_translation_poc/captures --layers all --source-max-chunks 64 --target-max-chunks 16'
```
