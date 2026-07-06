# INT2/INT3 Weight Model PPL Plan

Status: immutable after the initial house-rule commit.
Created: 2026-07-06
Branch: `codex/intn-model-ppl-sweep`

## Objective

Run actual-model validation for low-bit Project-Tensor weight kernels. The
target is not another synthetic linear sweep. The target is model-level
evidence for INT4, INT3, and INT2 weights under APA at multiple refine
percentiles, with memory/OOM receipts and perplexity results on real text.

## House Rules

- This plan is the fixed source of intent after its initial commit.
- The ledger records the operational trail: code changes, commands, artifacts,
  results, failures, OOMs, and follow-up decisions.
- The synthesis explains the ledger in narrative form.
- Failure is a result. If INT3 or INT2 collapse in PPL, OOM during load, or
  fail on a specific layer class, that result is preserved.
- Do not call kernel smoke tests model validation.
- Do not call synthetic linear sweeps PPL validation.

## Target Models

Primary first target:
- Qwen3.5-2B from the local unquantized HF snapshot:
  `/home/vader/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc`

Scale target after the 2B path is proven:
- Qwen3.5-9B from the local unquantized HF snapshot:
  `/home/vader/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a`

The 2B target is the bring-up path because it can test the real model adapter,
attention modes, PPL loop, and memory reporting with less blast radius.

## Required Implementation

1. Add a selectable low-bit model-weight wrapper.
   - INT4 remains the default behavior.
   - INT3 and INT2 route to the native Project-Tensor `intn_*` kernels.
   - The wrapper reports actual packed/scales/zeros bytes.
2. Add a real-model PPL/OOM runner.
   - Load a real model with a selected weight bit width.
   - Run standard attention and APA selective attention.
   - Sweep refine percentiles, initially `0.15`, `0.10`, and `0.05`.
   - Score real token windows with teacher-forced NLL.
   - Record load memory, post-run memory, per-setting peak memory, PPL, and
     any OOM/error with enough context to identify the failed setting.
3. Run at least a small 2B gate first.
4. Only after the 2B gate works, scale window count/window size and consider
   the 9B target.

## Evidence Required

For each tested model/bit/refine setting:
- model path
- weight bit width
- attention mode
- refine percentile when APA is active
- scored token count
- PPL
- memory used after load
- peak or max observed memory during evaluation
- error/OOM text if the setting fails

## Non-Goals

- No claim that INT3 or INT2 are production viable until a real model PPL run
  proves it.
- No claim that a small 2B smoke settles 9B behavior.
- No replacement of model PPL with output RMSE or cosine from synthetic layers.
