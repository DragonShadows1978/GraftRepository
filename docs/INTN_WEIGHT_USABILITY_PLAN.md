# INT3 Weight Usability Plan

Status: immutable after the initial house-rule commit.
Created: 2026-07-06
Branch: `codex/intn-model-ppl-sweep`

## Objective

Decide whether the Qwen3.5-9B INT3 perplexity jump is worth using in practice.
PPL is necessary evidence, but it is not the whole answer. The operational
question is whether INT3 can still do deterministic, useful work that INT4 can
do, especially exact fact extraction under greedy decode.

## House Rules

- This plan records intent. The ledger records commands, artifacts, results,
  failures, and follow-up decisions.
- The synthesis explains the result in narrative form.
- Failure is a valid result.
- Do not call kernel smoke tests usability validation.
- Do not call PPL alone usability validation.

## Gate

Run Qwen3.5-9B with INT4 and INT3 weights using the same TensorCUDA adapter.
For each bit width, test:

- standard attention
- APA selective attention at refine `0.15`

The first gate uses short real prompts with planted facts and greedy decoding.
Each answer is scored by exact expected-value containment after normalization,
plus a simple degeneration check.

## Evidence Required

For each tested bit width and attention setting:

- model path
- weight bit width
- attention mode and refine percentile
- prompt case names
- expected values
- generated answers
- exact hit count
- degeneration/failure count
- load memory and observed evaluation memory
- artifact path

## Interpretation

INT3 is not production-usable merely because it fits in memory. It must preserve
basic behavior. If INT4 passes exact extraction and INT3 fails, the memory
savings are not worth using for serious runs on this quantization path. If INT3
matches INT4 on the gate, the next step is a larger task battery.
