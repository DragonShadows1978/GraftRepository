# INT2/INT3 Weight Model PPL Ledger

This ledger records the actual model-validation work for low-bit weight
testing. The implementation plan is immutable after its initial commit.

## 2026-07-06 15:54 EDT

Action: Opened the real-model validation track.

Repo state:
- Repository: `/mnt/ForgeRealm/GraftRepository`
- Branch: `codex/intn-model-ppl-sweep`
- Starting point: `a8c2824 merge: split quant bench work`

Reason for this track:
- The prior Project-Tensor quant sweep only covered kernel correctness and a
  structured linear sweep.
- It did not load a model with INT2/INT3 weights.
- It did not run model memory checks.
- It did not run PPL.

Relevant repo findings:
- Qwen3.5 local unquantized snapshots exist for both 2B and 9B.
- `Qwen35_TC` loads model weights through `QuantLinearTC`.
- `QuantLinearTC` is currently hardwired to INT4.
- Project-Tensor already exposes native INT2/INT3 kernels through
  `tc.intn_linear`, `tc.intn_linear_fused`, and `tc.intn_dequant`.

Next action:
- Commit the house-rule baseline.
- Generalize the real model weight wrapper without changing INT4 defaults.

## 2026-07-06 16:04 EDT

Action: Added selectable INT4/INT3/INT2 real-model weight support.

Files changed:
- `core/mistral7b_tc.py`
- `core/qwen35_tc.py`
- `core/minicpm3_tc.py`
- `core/deepseek_v2_lite_tc.py`
- `tests/test_quant_linear_tc_intn.py`

Implementation notes:
- `QuantLinearTC` remains the shared real-model weight wrapper.
- INT4 remains the default.
- INT3 and INT2 route to Project-Tensor `quantize_affine_per_group` and
  `tc.intn_linear` / `tc.intn_linear_fused`.
- Qwen3.5 loading now supplies tensor context to `QuantLinearTC`, so a load
  failure can identify the tensor that failed.
- Model load metadata now includes `weight_bits`.

Regression command:
- `env PYTHONPATH=/mnt/ForgeRealm/GraftRepository:/mnt/ForgeRealm/Project-Tensor/tensor_cuda PYTHONDONTWRITEBYTECODE=1 PYTEST_ADDOPTS='-p no:cacheprovider' python3 -m pytest tests/test_quant_linear_tc_intn.py -q`

Regression result:
- `6 passed, 2 warnings in 0.48s`

## 2026-07-06 16:05 EDT

Action: Added real Qwen3.5 INTN PPL/OOM runner.

Files changed:
- `.gitignore`
- `scripts/qwen35_intn_weight_ppl_sweep.py`

Runner behavior:
- Loads a real Qwen3.5 model with selected INT4/INT3/INT2 weights.
- Runs standard attention plus APA refine sweeps.
- Records PPL, scored tokens, load memory, observed evaluation memory, load
  time, and error/OOM records.
- Writes local JSON artifacts under `artifacts/intn_weight_ppl/`.

Syntax command:
- `env PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 -m py_compile scripts/qwen35_intn_weight_ppl_sweep.py core/mistral7b_tc.py core/qwen35_tc.py core/minicpm3_tc.py core/deepseek_v2_lite_tc.py`

Syntax result:
- Pass.

## 2026-07-06 16:05-16:08 EDT

Action: Ran Qwen3.5-2B load-only gates for INT4, INT3, and INT2.

Model:
- `/home/vader/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc`

Commands:
- `env PYTHONUNBUFFERED=1 PYTHONPATH=/mnt/ForgeRealm/GraftRepository:/mnt/ForgeRealm/Project-Tensor/tensor_cuda PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/qwen35_intn_weight_ppl_sweep.py --bits 4 --load-only --window 256 --scored 128 --n-windows 1 --step 32`
- `env PYTHONUNBUFFERED=1 PYTHONPATH=/mnt/ForgeRealm/GraftRepository:/mnt/ForgeRealm/Project-Tensor/tensor_cuda PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/qwen35_intn_weight_ppl_sweep.py --bits 3 --load-only --window 256 --scored 128 --n-windows 1 --step 32`
- `env PYTHONUNBUFFERED=1 PYTHONPATH=/mnt/ForgeRealm/GraftRepository:/mnt/ForgeRealm/Project-Tensor/tensor_cuda PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/qwen35_intn_weight_ppl_sweep.py --bits 2 --load-only --window 256 --scored 128 --n-windows 1 --step 32`

Artifacts:
- `artifacts/intn_weight_ppl/qwen35_intn_ppl_20260706_160543.json`
- `artifacts/intn_weight_ppl/qwen35_intn_ppl_20260706_160643.json`
- `artifacts/intn_weight_ppl/qwen35_intn_ppl_20260706_160809.json`

Load results:

| Bits | Status | Load seconds | Memory before MiB | Memory after MiB | OOM |
| ---: | --- | ---: | ---: | ---: | --- |
| 4 | load_only_ok | 23.82 | 439 | 1433 | no |
| 3 | load_only_ok | 55.90 | 439 | 1209 | no |
| 2 | load_only_ok | 15.01 | 439 | 953 | no |

Interpretation:
- Qwen3.5-2B can load with all three weight widths.
- No OOM layer occurred in this 2B load-only pass.
- INT3 load is slower because INT3 packing is more expensive.

## 2026-07-06 16:08-16:11 EDT

Action: Ran first real Qwen3.5-2B model PPL gates for INT4, INT3, and INT2.

Protocol:
- Real Qwen3.5-2B TensorCUDA model.
- Real WikiText-2 raw cached dataset.
- Window: 512 tokens.
- Scored tail: 256 tokens.
- Actual scored tokens: 255 per setting.
- Settings: standard, APA r0.15, APA r0.10, APA r0.05.

Commands:
- `env PYTHONUNBUFFERED=1 PYTHONPATH=/mnt/ForgeRealm/GraftRepository:/mnt/ForgeRealm/Project-Tensor/tensor_cuda PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/qwen35_intn_weight_ppl_sweep.py --bits 4 --window 512 --scored 256 --n-windows 1 --step 64 --refines 0.15,0.10,0.05`
- `env PYTHONUNBUFFERED=1 PYTHONPATH=/mnt/ForgeRealm/GraftRepository:/mnt/ForgeRealm/Project-Tensor/tensor_cuda PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/qwen35_intn_weight_ppl_sweep.py --bits 3 --window 512 --scored 256 --n-windows 1 --step 64 --refines 0.15,0.10,0.05`
- `env PYTHONUNBUFFERED=1 PYTHONPATH=/mnt/ForgeRealm/GraftRepository:/mnt/ForgeRealm/Project-Tensor/tensor_cuda PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/qwen35_intn_weight_ppl_sweep.py --bits 2 --window 512 --scored 256 --n-windows 1 --step 64 --refines 0.15,0.10,0.05`

Artifacts:
- `artifacts/intn_weight_ppl/qwen35_intn_ppl_20260706_160848.json`
- `artifacts/intn_weight_ppl/qwen35_intn_ppl_20260706_160933.json`
- `artifacts/intn_weight_ppl/qwen35_intn_ppl_20260706_161057.json`

PPL results:

| Bits | Standard | APA r0.15 | APA r0.10 | APA r0.05 | Load MiB | Max eval MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 9.1755 | 9.1462 | 9.1232 | 9.1504 | 1433 | 1673 |
| 3 | 18.0926 | 18.1321 | 18.0971 | 18.0338 | 1209 | 1449 |
| 2 | 37280.8409 | 35133.2378 | 33977.4481 | 35744.1918 | 953 | 1225 |

Interpretation:
- INT4 is the working baseline on this gate.
- INT3 does not OOM, but quality drops sharply: roughly 2x PPL versus INT4.
- INT2 is functionally collapsed by PPL on this gate despite loading and
  executing.
- APA refine variation does not rescue INT3 or INT2 in this first 2B gate.
- This is a small 255-token scored gate; it is real model evidence, but it
  should be expanded before making final claims.
