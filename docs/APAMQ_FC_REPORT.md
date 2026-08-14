# APAMQ-FC — Engaged-Scoring Perplexity Arms

Date: 2026-08-13. Evidence class: **MODEL PERPLEXITY WITH ENGAGED
SCORING**. This worktree has no GPU. The harness and CPU receipt gates are
complete; all four model arms remain **NOT RUN HERE**. No perplexity, NLL, or
latency number is claimed in this report.

## Delivered harness

`scripts/apamq_fc_ppl.py` runs exactly one measurement arm per process and
writes an atomic/checkpointed JSON receipt. Its four model arms are:

| Arm | Global attention | Decode dispatch | Engine |
|---|---|---|---|
| `standard` | `standard` | non-APA | canonical TensorCUDA |
| `apa_blend` | `apa_selective`, r0.15, bits4 | blend forced with `GEMMA4_APA_DECODE_FUSED=0` | canonical TensorCUDA |
| `apa_fused` | `apa_selective`, r0.15, bits4 | fused forced with `GEMMA4_APA_DECODE_FUSED=1` | canonical TensorCUDA |
| `apa_int4` | `apa_selective`, r0.15 | `GEMMA4_APA_INT4=1`, fused decode | FA worktree TensorCUDA |

The APA settings retain `apa_min_context=2048` and `fast_max_seq=4096`.
Consequently, the 8192-token APA prefills retain the port's existing adaptive,
mixed dispatch. Only L=1 outputs contribute to NLL.
Unrelated resident-storage quantization is pinned off explicitly with
`GEMMA4_QUANT_V=0` and `GEMMA4_QUANT_KV4=0` in every arm.

## Fixed corpus and scoring documents

The public corpus is the cached `wikitext/wikitext-2-raw-v1` test split,
joined with two newlines and tokenized by the local Gemma-4-12B-it tokenizer
with `add_special_tokens=False`:

- dataset rows: 4,358;
- full token count: 292,282;
- raw-text SHA-256:
  `696cca6b65a171b0a358a4be6732cdfdf2dd6164a32e20fd70e3c13fc4dfae83`;
- int64 token-stream SHA-256:
  `5a4bcb9f9cff5ca58f896f3cf809b277588f7cab30ebbae01ba47dd1feb486dc`;
- tokenizer JSON SHA-256:
  `cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f`.

The dataset is loaded directly from its cached Arrow file, offline. This
avoids `datasets.load_dataset` attempting a write lock in the read-only cache.
Seed 20260813 permutes the 28 available aligned, non-overlapping 10,241-token
blocks and selects the first 25. The block order is:

```text
13, 12, 26, 3, 7, 14, 1, 15, 23, 0, 16, 21, 6, 5, 8, 17, 9, 25,
18, 22, 24, 2, 20, 27, 10
```

The selected stream contains 256,025 tokens and has int64 SHA-256
`2bf6c9cb15ce9bdca3233c82fa50149eabc7050c8c6b941fb019a22e0cecddb8`.
Each scoring document consists of 8,192 prefill tokens, one first L=1 decode
input, and 2,048 L=1-scored targets. Thus every arm scores 51,200 targets.

These are fixed corpus token blocks rather than original WikiText articles.
Only 9 of the 63 source articles are at least 10,241 Gemma tokens, so article
boundaries cannot satisfy the registered 25-document/8192+2048 shape. The
blocks are non-overlapping and hash-bound in every receipt.

## Engagement receipts

The harness monkeypatches only live Python call sites for observation; scored
outputs still come from the original engine functions. For every global
attention invocation it records phase and sequence length. For every actual
blend, fused, or INT4 engine call it records the exact branch.

A completed APA arm must have exactly 409,600 global L=1 scoring calls:
25 documents × 2,048 targets × 8 global layers. All must reach only its
registered branch (`blend`, `fused`, or `int4`). `standard` must make the same
409,600 global L=1 calls and zero APA-operator calls. Every teacher-forced
model input is asserted to have shape `(1, 1)` and dtype `int64` before the
call. Chunked prefill logits never enter NLL.

After each document, all eight global caches are checked:

- raw K and V rings must be `bfloat16` and have count 10,240;
- `apa_blend` and `apa_fused` must have a `bfloat16` derived KQ ring with
  `kq_count == count`;
- `standard` and `apa_int4` must have no derived KQ ring and `kq_count == 0`.

The INT4 absence assertion is the F-A no-ring proof. The branch census plus
nontrivial r0.15 configuration and `S > apa_min_context` is the scoring
engagement proof; configuration flags alone cannot complete a receipt.

## Statistics and summary

Each arm receipt contains per-document NLL, perplexity, and decode wall time,
plus aggregate mean NLL, perplexity, and decode ms/token. Decode timing covers
the L=1 model call, required logit device-to-host transfer, and host NLL;
prefill is excluded and each document is synchronized.

`--arm summary` reads `standard.json`, `apa_blend.json`, `apa_fused.json`, and
`apa_int4.json` from the summary output directory. It rejects incomplete,
unengaged, wrong-branch, wrong-ring, wrong-token-count, protocol-mismatched, or
code-mismatched receipts. For every arm it emits:

- aggregate relative perplexity delta versus `standard` and `apa_blend`;
- paired per-document NLL-delta mean, sample standard deviation, and SEM;
- paired per-document relative-perplexity-delta mean, sample standard
  deviation, and SEM;
- decode ms/token and engagement/branch/ring receipts.

The printed and JSON table contains `adjudication=not_performed` and no
verdicts.

## Exact lead commands

Run one command per process, in this order:

```bash
python3 scripts/apamq_fc_ppl.py --arm standard --output artifacts/apamq_fc/standard.json
python3 scripts/apamq_fc_ppl.py --arm apa_blend --output artifacts/apamq_fc/apa_blend.json
python3 scripts/apamq_fc_ppl.py --arm apa_fused --output artifacts/apamq_fc/apa_fused.json
TENSOR_CUDA_ROOT=/mnt/ForgeRealm/wt/apamq-fa/tensor_cuda python3 scripts/apamq_fc_ppl.py --arm apa_int4 --output artifacts/apamq_fc/apa_int4.json
python3 scripts/apamq_fc_ppl.py --arm summary --output artifacts/apamq_fc/summary.json
```

Exactly 25 documents are used, the minimum registered valid count. Whether a
full arm remains below the approximately 45-minute 4070 SUPER budget cannot be
measured in this GPU-less sandbox; each receipt explicitly records elapsed
time and `runtime_budget_exceeded` without converting a budget miss into a
quality verdict.

## CPU checks run here

Commands:

```bash
python3 -m py_compile scripts/apamq_fc_ppl.py tests/test_apamq_fc_ppl.py scripts/apamq_fbd1_diag.py tests/gemma4_apa_decode_fused.py
python3 -m pytest -q tests/test_apamq_fc_ppl.py
python3 scripts/apamq_fc_ppl.py --arm cpu-check --output /tmp/apamq_fc_cpu.json
python3 tests/gemma4_apa_decode_fused.py --cpu-check
TENSOR_CUDA_ROOT=/mnt/ForgeRealm/wt/apamq-fa/tensor_cuda python3 tests/gemma4_apa_decode_fused.py --cpu-check
```

Observed output:

```text
py_compile: PASS
pytest: 6 passed
CPU OFFLINE CORPUS LOAD: PASS
CORPUS: WikiText-2 raw-v1 test rows=4358 tokens=292282 raw_sha256=696cca6b65a171b0a358a4be6732cdfdf2dd6164a32e20fd70e3c13fc4dfae83
CPU DOCUMENT SELECTION: PASS docs=25 tokens/doc=10241 selected=256025 scored=51200
CPU TRUE-L1 PROTOCOL SHAPE: PASS prefill=8192 decode=(1,1) int64
CPU ENGINE ROOT MATRIX: PASS canonical=3 arms FA=apa_int4
CPU ENGINE ENTRY STATIC CHECK: PASS canonical=fused FA=int4
CPU ARM ENVIRONMENT MATRIX: PASS
CPU PROTOCOL FINGERPRINT: 2b8f4d2d77aad23b54255c05c1128a95b09934ff95fda9fa239828187bba52be
canonical existing gate: CPU FLAG MATRIX PASS; root PASS; INT4 entry ABSENT
FA existing gate: CPU FLAG MATRIX PASS; root PASS; INT4 entry PRESENT
```

`ruff` is not installed (`ruff: command not found`), so no lint result is
claimed. No GPU arm was executed and no GPU number was fabricated.
