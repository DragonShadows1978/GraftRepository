# ORDER QWEN38-LC1-F3 — --force-alloc: empirical OOM receipts past the paper wall

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/GraftRepository` — edits/builds/test
runs AUTHORIZED. Continuation of QWEN38_LC1 + F1 + F2 (boundaries unchanged).
Small, surgical order.

## Why

David's descent protocol wants the EMPIRICAL OOM point ("run 128K and SEE if
it OOMs and where"). The LC1.1 fail-loud budget validator predicts walls at
~9K (device bf16 KV) and ~17K (device INT8 KV) driven by the 1,185 MiB fixed
transient allowance — calibrated from the OLD code's live peaks, before LC1's
own lm_head/prefill trims. Predicted walls are not receipts. The validator
currently refuses before any real allocation is attempted.

## Work items

**F3.1** Add `--force-alloc` to `scripts/qwen38_longctx_gate.py` (and the
underlying loader path): budget is still computed and PRINTED, but a
MemoryError from `validate_qwen38_memory_budget` becomes a logged WARNING
(`BUDGET_OVERRIDE {json}`) instead of an exit, and loading/allocation
proceeds for real.

**F3.2** Graceful empirical-OOM handling under --force-alloc: any CUDA
allocation failure (cudaMalloc / engine OOM exception, whatever tensor_cuda
actually raises — cite the exception type from the engine source, READ-ONLY)
is caught at load AND at prefill/decode; on catch, print
`RECALL_RESULT {"status": "RED", "error": "CUDA_OOM: ...", "phase": <load|prefill|decode>, "vram": <sampler receipt>}`
and exit 1 cleanly (sampler stopped, no hang, no corrupt cache files —
verify the pack cache is opened read-only so an OOM can never damage it).

**F3.3** CPU tests: force-alloc flag plumbing, override-warning path, and
the OOM-catch control flow (exception faked CPU-side).

## Boundaries

Same as LC1/F1/F2. Do NOT loosen the default (non-forced) fail-loud
behavior — --force-alloc is an explicit descent-protocol tool, default OFF.

## Done — final message MUST contain verbatim
1. Diff hunks / file:line list.
2. The engine's actual OOM exception type(s) with citation.
3. Full CPU test suite output.
4. Exact lead commands for a forced 128K bf16 attempt and a forced 128K
   int8 attempt.
5. Residuals stated plainly.
