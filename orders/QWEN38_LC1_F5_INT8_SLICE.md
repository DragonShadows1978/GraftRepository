# ORDER QWEN38-LC1-F5 — INT8-KV block readback crashes: engine slice rejects int8

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/wt/lc1-f4` (git worktree of
GraftRepository, branch lc1-f4) — edits/builds/test runs AUTHORIZED.
Continuation of LC1+F1..F4 (boundaries unchanged). Lead runs GPU gates.

## The defect (live GPU receipt)

`/mnt/ForgeRealm/GraftRepository/logs/lc1_recall_int8_16k_d0.25.log`
(READ-ONLY):

```
File "core/qwen38_tc.py", line 438, in block
    return (self._unpack(self.kb.slice(2, lo, n), self.ks.slice(2, lo, n)),
RuntimeError: op supports float32/float16/bfloat16 only
```

The INT8-KV path stores packed int8 KV on device, but tensor_cuda's `slice`
op does not support int8 dtype — every block readback crashes. All three
INT8 16K recall runs died identically. The CPU tests faked the engine and
could not catch it.

## Work items

**F5.1** Fix the INT8 block readback WITHOUT changing quantization
semantics. Read the engine source (READ-ONLY:
/mnt/ForgeRealm/Project-Tensor) and cite file:line for which ops accept
int8/uint8 tensors. Acceptable approaches, in preference order: (a) slice
via a dtype the engine supports for slicing while preserving exact bytes
(e.g. view/reinterpret if available); (b) byte-offset addressed reads
without slice; (c) store per-block int8 segments pre-split so no slicing
is needed. Justify the choice in one paragraph.

**F5.2** Audit EVERY other engine op the INT8 and host-KV paths invoke on
non-float dtypes (pack/unpack, cat, transpose, copy, upload/download) —
same symbol-table discipline as F1, but this time each op call must cite
the engine's dtype support (file:line in Project-Tensor source). List every
call site with its verdict.

**F5.3** CPU tests for the new readback control flow; and if the engine
exposes any CPU-executable path for the op(s) you rely on, add a real
(non-faked) test of byte-exactness through slice/readback.

## Boundaries

Same as LC1..F4. Engine repo is READ-ONLY — if the correct fix truly
requires an engine change, STOP, write the finding with citations, and
report; do not work around by changing quantization semantics silently.

## Done — final message MUST contain verbatim
1. Diff hunks / file:line list.
2. The F5.2 dtype-support audit table.
3. Full CPU test suite output.
4. Exact lead commands: INT8 16K recall x3 depths rerun + INT8 4K
   kv-agreement (G-LC3).
5. Residuals stated plainly.
