# Gemma-4 `core/gemma4_tc.py` — Blind Bug Hunt

**Scope:** independent correctness-only audit. Not told what the Architect's
3 claimed bugs are. No fixes, no commits, no GPU runs. All checks below are
either direct code reading or cheap CPU-only Python re-derivations of the
mask/index formulas compared against the documented/intended semantics.

**File audited:** `core/gemma4_tc.py`, 1010 lines, read in full.

**Integrity check (required by the task):**
- Start: mtime `2026-07-02 14:13:39.708331115 -0400`, size 48018 bytes,
  sha256 `74c125a4183cd5da6381217ade5706238b4a4b262a17781251312e3edab3b2e4`
- End: **identical** mtime, size, and sha256.
- **The file was NOT modified during this hunt.** All findings below refer
  to this exact, single version of the file (same version already covered
  by `docs/GEMMA4_APA_AUDIT_A1.md`).

---

## Result: 0 confirmed correctness bugs found

I read the file in full, cross-checked every one of the 16 registered traps
in `docs/GEMMA4_PORT_LEDGER.md` against the code, re-derived every mask/index
formula named as a prime suspect in the task brief, and ran CPU-only Python
checks against each. **None of them turned up a bug.** Per the task's
instruction not to manufacture findings to hit a count: this report lists 0
CONFIRMED bugs and 0 SUSPECTs I'd stand behind. Below is the checked list
(what I verified and why each one is clean), not a padded candidate list.

If the Architect is confident 3 bugs exist in this exact file version, they
are either (a) in code paths I could reason about but not execute (this was
a static-only audit, same constraint as A1 — nothing here was run on a GPU),
(b) subtler than anything a single-pass reading + formula re-derivation
catches (e.g. a numerical/dtype-precision issue that only shows up under
real tensor values, not shape/index logic), or (c) I'm simply missing them.
I'd rather say that plainly than invent 3 findings.

---

## Checks performed (all clean — no bug found)

### A. Mask geometry, non-square L×S (bottom-right causal)
- `functional.py:_causal_mask(L, S, ...)` builds `triu(-1e4, k=1+(S-L))` —
  re-derived in numpy against the documented bottom-right semantics (row i =
  absolute query position `S-L+i`, visible keys `j <= S-L+i`). **Matches.**
- Engine `causal_softmax_kernel` (`kernels.cu:2338-2367`) computes
  `visible = S - L + i + 1` from the tensor's own shape (`i = row % L`).
  Re-derived by hand: at Gemma's decode call site
  (`gemma4_tc.py:519-520`, `sc.reshape([B, KV*rep, 1, cap])`), this collapses
  to `visible = cap` for every row (L=1) — full visibility, matching the
  "L==1 causal == full row" comment exactly. **Matches.**
- Fused `apa_selective_attention` kernel (`kernels.cu:1080` /
  `:1269` / `:1387`) uses the same `s_max = (S-L)+i+1` formula (the
  A1 audit already traced this; I independently re-read all three call
  sites of the macro/pattern in `kernels.cu` and confirm they agree).
  Gemma's call site (`gemma4_tc.py:577-578`) passes whole, unchunked-in-L
  `q`/`k` per outer prefill chunk with `is_causal = L > 1`, and I verified
  by hand that `is_causal=False` at `L=1` yields the same `s_max` as
  `is_causal=True` would (`(S-1)+0+1 == S`), so the `L>1` shortcut is
  coincidentally exact, not a bug.
- `_cublas_blend_attention` MQA arm (`mistral7b_tc.py:216-241`, Gemma's
  only reachable arm since KVH=1 always for its globals): builds
  `F._causal_mask(L, S, ...)` using the **outer** `L` (whole-call query
  length, i.e. `q.shape[2]` from the top of the function) and slices rows
  `[i, i+bl)` per block — re-derived in numpy that this equals the
  correct per-block sub-mask of the full `(L,S)` bottom-right mask.
  **Matches**, and is consistent with the non-MQA arm's identical pattern
  a few lines down (both call the single canonical `F._causal_mask`, no
  private duplicate construction — the class of bug this codebase has
  been bitten by twice, per the ledger, is not present here).

### B. Sliding-window band mask (`kv ∈ (q−1024, q]`)
- `_band_mask` (`gemma4_tc.py:324-339`): re-derived the visibility formula
  `(j <= i) & (j > i - window)` in numpy at `window=3` — produced exactly
  `window` visible keys per row, matching "includes self, `window` keys
  max" (trap 8). **Matches.**
- Checked the `S_all <= W` branch (`gemma4_tc.py:610-613`), which uses
  plain `is_causal=True` (no band mask at all) instead of `_band_mask`.
  Proved algebraically that when `S_all <= W`, the band's lower bound
  (`j > abs_pos - W`) is never binding (`abs_pos - W < 0 <= j` always), so
  plain causal is exactly equivalent to the band mask in this regime.
  Verified the boundary case `S_all == W` explicitly in numpy (arrays
  compared with `np.array_equal`, true). **Not a bug** — a valid
  optimization, not a missing restriction.
- Ring-trim (`gemma4_tc.py:620-625`, `keep = min(W-1, S_all)`): confirmed
  `keep + 1 == W` after the next append, i.e. trimming to `W-1` before
  appending the new key is exactly right to land back at window size `W`.
  **Matches.**

### C. Chunked-prefill state handling
- `last_token_only` passthrough (`gemma4_tc.py:734-741, 779-780`): traced
  that slicing `h` to 1 row before `lm_head` on every chunk (not just the
  final one) only affects logits, never the K/V caches propagated between
  chunks — the wasted GEMV on non-final chunks is discarded by construction
  (outer loop only keeps the last chunk's `lg`). **Not a bug**, matches the
  documented trap/fix.
- Adaptive chunk step-size formula (`gemma4_tc.py:729-732`): simulated a
  20,000-token prefill in Python — terminates in 229 iterations, exact
  full coverage, no skipped or double-counted tokens, `step` floor of 64
  prevents a zero-step infinite loop at any context length tested up to
  10,000,000. **Not a bug.**
- `KVRing.append`/`quantized_keys` (`gemma4_tc.py:231-304`) bookkeeping:
  traced the growth interaction between `kb`/`vb`/`kqb` (`_grow1` runs on
  all three, in order, copying the same `old_n` valid-row prefix into each
  new same-sized buffer) and confirmed `quantized_keys`'s incremental
  `[kq_count:count)` window stays aligned with `kb`'s logical positions
  across growth events, because `quantized_keys` is only ever reachable on
  **global** layers, which are constructed with `ring_cap=None` and
  therefore never wrap (`self.ring=False`) — positions are stable for the
  cache's entire lifetime, exactly as the docstring claims. **Not a bug.**
- Bias-mask (`-1e4` invalid-row padding) interaction with ring wraparound:
  simulated a small ring (`window=4`) through and past its first wrap in
  Python — once `full` becomes true the bias buffer is permanently all-zero
  and stays correct; before that, `bias[pos]` is unmasked exactly on first
  write of each row. **Not a bug.**
- Decode-APA path (`gemma4_tc.py:484-508`) sidesteps the invalid-row-bias
  question entirely by slicing `kk`/`vv`/`kq` to exactly `kv_cache.count`
  (not `cap`), and passes `causal=False` — verified this is exactly right
  for decode (single newest query, always fully visible to its own prefix).
  **Not a bug.**

### D. dtype handling
- Searched the whole repo for `_NP_DTYPE`: **zero hits**. This class does
  not exist in `core/gemma4_tc.py` or anywhere else in this repository —
  it is not applicable to this file (may be a trap class from a different
  codebase, e.g. the tensor_cuda engine itself, conflated in the brief).
- Traced `QuantLinearTC.__init__` (`mistral7b_tc.py:84-91`) to confirm the
  embed-scale in-place mutation (`gemma4_tc.py:813-814`,
  `self.lm_head = QuantLinearTC(emb); emb *= sqrt(hidden)`) cannot corrupt
  the already-quantized tied head: `_quantize_int4` runs synchronously
  inside `__init__` and produces independent new arrays: no reference to
  the caller's `emb` is retained. **Not a bug.**
- `_q40_repack` (`gemma4_tc.py:73-88`): built a synthetic q4_0 block by
  hand in Python (known `x` values in [-8,7], packed per the documented
  GGUF j/j+16 nibble-interleave layout) and confirmed the function's
  output is bit-exact equal to the engine's expected even/odd-pair packed
  layout after de-interleaving. **Matches, verified not trusted.**

### E. Registered traps implemented subtly wrong
Checked all 16 traps from the ledger against the code; all match:
1. Plain-w RMSNorm everywhere incl. qk-norms — `_head_rmsnorm` never adds 1.
2. `layer_scalar` multiplies the whole post-block stream including the
   residual — `Gemma4BlockTC.__call__` line 666, `return h * self.layer_scalar`
   where `h` already contains the residual sum. **Confirmed correct
   placement** (this is exactly the kind of thing the brief flagged as a
   suspect — "layer_scalar applied to the wrong stream" — and it is NOT
   wrong here).
3. Attention scale 1.0 — grepped every attention call site (7 total),
   all pass `scale=1.0` / `alpha=1.0`.
4. Embed scale `sqrt(3840)` in fp32, lm_head unscaled — confirmed order of
   operations (quantize-then-scale) is safe, see D above.
5. Global V = `v_norm(k_proj out)`, no RoPE — confirmed `vsrc = kraw`
   (pre-norm, pre-RoPE raw projection) is independently normalized with
   `w=None` (scale-free) and RoPE is applied only to `q`/`k`, never `v`.
   **Confirmed correct** (this is exactly the "RoPE applied to global V by
   accident" suspect from the brief, and it is NOT present).
6. p-RoPE: re-derived the 64-real+192-zero frequency construction in numpy
   against `theta^(-2i/512)` for `i<64` — exact match, including the
   full-512-width apply via `emb_g = concat([pos*inv_g, pos*inv_g])`.
7. Sliding RoPE full 256-dim, θ=1e4 — confirmed same code path, different
   config constants.
8. Sliding mask window — see B above.
9. Sandwich norms — traced `Gemma4BlockTC.__call__` order: input_ln → attn
   → post_attn_ln(on attn output) → +residual → pre_ffw_ln → mlp →
   post_ffw_ln(on mlp output) → +residual. Matches Gemma-2-style sandwich.
10. Logit softcap 30 — `(lg.float()/30).tanh()*30`, matches.
11. RMSNorm eps inside rsqrt — confirmed in both the fused kernel
    (`kernels.cu`: `powf(mean_sq + eps, -0.5f)`) and the Python fallback
    (`(ms + eps).pow(-0.5)`).
12. RoPE layout — not independently re-derived (would need the HF
    reference to diff against; out of scope for a static/CPU-only check).
13. qk-norm present, bulk_bits=4 — `self.bulk_bits = 4` at
    `gemma4_tc.py:403`, matches.
14. Inert machinery — not exercised (config surface, no live code path in
    this file references MoE/double-wide-MLP/bidirectional).
15. Chat template — not applicable to this file (lives elsewhere).
16. Engine gelu tanh-approx — used via `.gelu()` in `GegluTC`, not
    independently re-verified against the exact tanh formula (would need
    the kernel source's constants, outside this file).

---

## What I did NOT check (explicit scope limits)
- No GPU execution — every finding above is either a direct code read or a
  CPU-only Python re-derivation of shapes/indices/formulas, per the task's
  constraints.
- Items 12, 14, 15, 16 above were traced for presence/placement but not
  independently re-derived bit-for-bit against an external reference,
  since doing so needs files outside `core/gemma4_tc.py` (HF reference
  impl, engine kernel constants) beyond what a quick cross-check allows.
- Did not attempt to find performance or style issues — out of scope per
  the task.

## Bottom line
Ranked list of candidate bugs, most-confident first: **none.** I found 0
CONFIRMED and 0 SUSPECT correctness bugs in `core/gemma4_tc.py` as it
stands at sha256 `74c125a4...edab3b2e4`. Every prime-suspect area named in
the task brief, and every registered trap in the ledger, checks out
correct under direct reading plus CPU-only formula verification. This is
fewer than the stated 3 — reported as-is rather than padded.
