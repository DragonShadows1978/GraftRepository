# ORDER QWEN38-LC1-F1 — fix G-LC0 crash + unexercised-path audit

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/GraftRepository` — edits/builds/test
runs AUTHORIZED. Continuation of order QWEN38_LC1_LONG_CONTEXT.md (read it
first; its boundaries and gates all still apply). Lead runs GPU gates.

## What happened

G-LC0 (baseline invariance) crashed on the lead's GPU run — RED receipt in
`logs/lc1_g0_baseline.log`:

```
File "core/qwen38_tc.py", line 455, in __call__
    q = tc.cat([F.apply_rotary(q.slice(3, 0, R), cseg, sseg),
NameError: name 'F' is not defined
```

`F` is `tensor_cuda.functional` (see `core/mistral7b_tc.py:25`); the LC1
changes to `core/qwen38_tc.py` use it without importing it. Your CPU tests
passed because they never execute the GPU attention path.

## Work items

**F1.1** Fix the NameError with the same import convention the sibling files
use. Do not restructure anything else.

**F1.2 — Unexercised-path audit (the real work).** Every line you added or
modified in LC1 that the CPU tests do NOT execute is suspect. Audit ALL of it
for undefined names, wrong attribute paths, arity mismatches against the APIs
actually defined in `core/mistral7b_tc.py` / `core/qwen35_tc.py` /
tensor_cuda's exported surface (read those definitions; cite file:line for
each API you call). No network and no pyflakes on the box: do the audit by
building an explicit symbol table for each new/modified function — list every
non-local name it references and where that name is defined. Include the
table in your report.

**F1.3** Extend `tests/test_qwen38_cpu.py` so the pure-Python control flow of
the new paths (baseline-check loop, needle builder, budget validator branches,
kv-int8 pack/unpack, host-KV bookkeeping) is executed CPU-side to the maximum
extent possible without the GPU — at minimum, every new function gets
imported and its argument signature exercised. The GPU math itself stays
lead-verified.

## Boundaries

Same as LC1: no git, no subagents, no network, READ-ONLY engine/models/cache,
no attention-math or weight-quant changes beyond the import fix, RED honesty.

## Done

Final message MUST contain verbatim:
1. The exact diff hunks you applied (small).
2. The F1.2 symbol table (per function: names used → definition site).
3. CPU test output (all tests, run synchronously).
4. Any additional latent defects found and fixed, each with its receipt.
5. Residuals stated plainly.
