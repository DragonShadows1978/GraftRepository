# KV-Graft: mounting documents into a frozen model as lossless off-context memory

**tensor_cuda engine — Qwen3-4B INT4 (36L, 32Q/8KV, per-head qk-norm), 8GB RTX 3070.**
2026-06-07. Generalizes the injection idea from "Prometheus Mind" (arXiv 2601.15324)
from single-token steering to whole-document memory — and lands somewhere stronger
than the paper: **lossless, multi-fact, query-addressable, zero-training.**

## The result

Harvest a document's per-layer attention K/V once (pre-RoPE), then splice it into
every layer's attention as a positional prefix at generation time. With the
document **absent from the prompt**:

> `' The north vault access code is'` → **`' Bluejay. The supply ship captain is
> named Marisol Vexley.'`**
> `' The supply ship captain is named'` → **`' Marisol Vexley. The emergency
> rendezvous point is dock forty-seven.'`**
> `' The emergency rendezvous point is dock'` → **`' forty-seven.'`**
> control (`' The password for the south gate is'`, not in doc) → **no leakage**

**3/3 facts, verbatim, including an alien multi-token name; clean control.**

At the logit level the graft is **indistinguishable from in-context**: top-1
identical (21.38 vs 21.38), max |logit diff| 0.234 (~bf16 noise), per-layer
residual rel-diff 0.0035–0.019. **A grafted document IS the document, to the
model — at zero prompt-token cost.** No training, no weight changes, fully
reversible (clear the injection → exact original model), and the harvested K/V is
a savable artifact: harvest once, mount at will.

## The mechanism (all of it necessary)

1. **Capture per-layer K/V pre-RoPE** (position-neutral); capture V as-is. The
   hook sits in `GQAAttentionTC.__call__` after qk-norm, before RoPE.
2. **Inject as a positional prefix**: RoPE the grafted keys at positions 0..Sg−1,
   shift the live tokens to Sg.., cat in front of each layer's K/V.
3. **Bottom-right-aligned causal mask** so the prefix is fully visible to every
   query token while causality holds among the queries (see The Bug below).
4. Scale 1.0, all layers, persistent. That's it — no targeting, no clearing.

## The bug that ate two days of "recipes" — and the lesson

Everything before the fix — targeted answer-keys, scale sweeps (0.5 magic point),
per-layer profiles, first-token-only, clear_after=3, repetition penalties — was
**compensation for an engine bug none of it could see**:

`_causal_mask` built `np.triu(..., k=1)` regardless of shape. Square masks
(normal prefill) are correct, but on the **rectangular** (L queries, S=Sg+L keys)
masks the graft creates, `triu(k=1)` is **top-left aligned**: query row i saw
only key columns 0..i — i.e. the first few graft tokens, never the answers, and
**never the query's own tokens**. Every "partial success" was the visible-window
sliding over an answer by luck (duplicated keys at the front; answers emerging
mid-generation as the window grew). Diagnosed by an **activation-level A/B**:
graft-vs-in-context logits diverged from layer 0 and the graft's top prediction
was `' briefing'` — the literal first visible graft tokens. Fix: `k = 1+(S−L)`
(bottom-right alignment; square case unchanged). Engine commit `63960a4`,
59/59 tests pass.

**Lessons.** (a) A "working" config found by sweeping can be an artifact of a bug
that a single activation-level A/B exposes in one run — diff the mechanism, not
just the behavior. (b) The single-fact "success" was the most dangerous kind of
result: real enough to validate the wrong recipe. (c) The equivalence prediction
("FULL@1.0 should equal in-context") was the falsifiable claim that cracked it —
when it failed, the defect had to be mechanical.

## What this opens (now that the baseline is lossless)

- **Persistent mountable memory**: harvested K/V saved to disk per document;
  mount/unmount per query. The "drop a folder in, it's part of the model"
  experience — implemented as harvest-on-add + inject-on-query. For the 580k-token
  guide corpus: harvest each guide once, mount relevant guides per generation.
- **Memory cost is the KV cache of the grafted tokens** (the engine's INT8-KV
  halves it; ~30MB/1000 tok measured earlier on 1.5B) — and the engine's measured
  ceilings (Qwen-family >49k tokens) bound how much can be mounted at once.
- **The compression frontier**: scale <1, targeting, key-dedup, INT8/INT4-quantized
  graft K/V, and notably **APA-selective attention over the graft** (the engine's
  own selective kernel is exactly a query-driven top-k key selector) — all now
  measurable as fidelity-vs-footprint *dials on a lossless baseline* instead of
  voodoo.
- **Open questions**: behavior at corpus scale (interference between many mounted
  docs); style/rule-following from grafted guides (does a mounted style guide
  change *how* the model writes, or only what it can recall? — the skin-graft vs
  reference-library question); cross-model transfer of the recipe.

Harness: `core/kv_graft.py` (harvest/set_injection/clear), hooks in
`core/mistral7b_tc.py` `GQAAttentionTC.__call__`; mask fix in engine
`tensor_cuda/functional.py` (`63960a4`). Adapter: `core/qwen3_tc.py` (Qwen3-4B
INT4, per-head qk-norm, 3.0GB loaded).

---

## Addendum 2026-06-10 — everything after this doc

This writeup covers the original Qwen3-4B GQA result. The work has moved
substantially; current state lives in
`GraftRepository_Memory_Architecture.md` (foundations table = the receipts).
Headlines since:

- **Multi-turn cached grafting proven** (prefill-only injection,
  `graft_seats` position law, teacher-forced equivalence at the cache's own
  noise floor). Seat 0 is sacred (attention sink); selective amnesia
  confirmed — haunting carries awareness, not facts.
- **E1 router recall PASSED on both models** — and routed top-3 BEATS
  mount-all (over-mounting is the other forgetting). Router law forks by
  model: layer-0 key-space max on QK-normed Qwen3; LATENT-centroid cosine
  on MiniCPM3 (key-space fails without qk-norm — outlier keys). Summary
  key for MLA = unit-norm mean latent, 512 bytes.
- **MLA latent grafting** (MiniCPM3): graft = (c_n, pre-RoPE k_pe), 288
  vals/tok/layer (~22× smaller artifacts); equivalence + cached-stream +
  recall gates all green.
- **E4 end-to-end conversation memory PASSED**: routed memory 6/6 =
  full-transcript baseline at ~25% residency (bounded); amnesia control
  0/6. Routing hygiene: route on bare user text; don't deposit
  retrieval-only turns.
- **Persistent arena BUILT** (`core/graft_arena.py`): mount/unmount as
  cache surgery on one never-rebuilt cache, [SINK|ARENA|LIVE] seating,
  fixed live_shift, recency eviction. E4-arena 6/6.
- **Decode 675→21.6 ms/tok (31×)** on MiniCPM3 (absorbed MLA decode, int4
  GEMV kernel `8501a5c`, no_grad/pool/fused-norm) — arena turns 1.5s.
  See `MiniCPM3-MLA_Results.md` §decode.
