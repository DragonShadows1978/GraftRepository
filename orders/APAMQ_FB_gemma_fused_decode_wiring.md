# APAMQ-FB — Gemma-4: Fused APA Decode Wiring + INT4 Ring-Drop Scaffold

YOUR WRITABLE TARGET is the git worktree you are launched in (a
checkout of GraftRepository on branch `apamq-fb`) — edits to `core/`,
`scripts/`, `tests/` there are AUTHORIZED. Run first, report after; a
registered order IS the permission.

HARD BOUNDARIES: never touch the canonical repo at
/mnt/ForgeRealm/GraftRepository or /mnt/ForgeRealm/Project-Tensor
(read-only references; live measurement cells are running against
them), no git (lead commits/merges), no subagents, no network,
/mnt/ForgeRealm/models read-only. Your sandbox has NO GPU: deliver
code + tests the lead runs; CPU-side syntax/py_compile/pytest-skip
checks only. RED honesty; no monitor-idling.

## Context (pre-nailed — do not re-litigate)

Read (canonical repo, read-only):
/mnt/ForgeRealm/Project-Tensor/docs/APA_MQA_FIX_PLAN.md — you are
workstream F-B. Facts you build on (all receipted today, APAMQ ledger):
- Gemma APA DECODE unconditionally dispatches `_cublas_blend_attention`
  (your worktree's core/gemma4_tc.py:562-579 area) even though the
  fused kernel is D=512-legal and prefill already uses it above
  fast_max_seq=4096. That is the H-D wiring gap.
- The kqb ring (KVRing.kqb + quantized_keys incremental machinery) is
  APA's memory tax: 144 MiB resident at S=16K.
- E1 measured the fused kernel's transient advantage fully present at
  kv=1; slower in wall time (that fix is another seat's engine order —
  NOT yours; do not touch Project-Tensor).

## Task 1 — fused decode dispatch (works with TODAY'S engine)

In the APA decode branch: when `S_all > self.fast_max_seq` (same lever
as prefill), dispatch `tc.apa_selective_attention(q, k, kq, v, 1.0,
z_, False)` on the UNEXPANDED MQA cache + the incremental kqb ring
slice, instead of `_cublas_blend_attention`. Blend remains the
fallback below the lever and behind `GEMMA4_APA_DECODE_FUSED=0`
(default ON above the lever). Respect the existing incremental
quantize contract (quantized_keys quantizes only new rows; do not
regress it to whole-cache requant — that was the June OOM class).

## Task 2 — INT4 ring-drop scaffold (activates when the engine lands)

Behind `GEMMA4_APA_INT4=1` (default OFF): call
`tc.apa_selective_attention_int4(q, k, v, scale, zthr, is_causal)` —
NO kq argument; the engine packs internally — for BOTH prefill-fused
and decode-fused sites, and when active: never allocate/grow/populate
kqb (the ring and its quantize calls are skipped entirely, including
the cold-start chunked quantize). Guard with
`hasattr(tc, "apa_selective_attention_int4")` so the flag is inert on
today's engine. The signature is pre-nailed by the plan; code to it.

## Task 3 — gates (write; lead runs GPU)

- Extend/mirror tests/gemma4_apa_incremental.py into a new
  tests/gemma4_apa_decode_fused.py: decode-path branch assertion (the
  fused site actually fires above the lever), greedy-token/logit
  parity gate blend-vs-fused over ≥64 decode steps at S≈8K (register
  your tolerance: fused reorders float sums), and a
  ring-not-allocated assertion under GEMMA4_APA_INT4=1 (skip cleanly
  when the engine entry is absent).
- Extend scripts/apamq_e3_attrib.py with `--decode-fused` and
  `--int4` toggles so the SAME attribution harness measures G-B1/G-B2
  (decode peak ≤ standard+20 MiB; int4-mode peak ≤ standard peak,
  S=16K, one-mode-per-process).

## Done

Final message MUST contain verbatim: files changed with the decode
dispatch diff summarized, the flag matrix (env × engine-capability →
behavior), exact CPU-side check commands + output, the registered
parity tolerance and why, and any deviation. No GPU numbers; do not
fabricate.
