# Gemma-4 APA Extension — Phase A1 Code Audit

**Scope:** static code audit only. No fixes, no commits, no GPU runs.
Re-verifies claims in `docs/GEMMA4_PORT_LEDGER.md` against
`core/gemma4_tc.py` (1010 lines, read in full), `core/mistral7b_tc.py`
(the shared `_repeat_kv`/`_cublas_blend_attention`/GQA reference path),
and the engine at `/mnt/ForgeRealm/Project-Tensor` as built (kernels.cu,
bindings.cpp, quant.py). Every finding below cites `file:line`; where a
number could not be pinned to source it is marked "COULD NOT DETERMINE"
rather than estimated.

**Files read in full:** `core/gemma4_tc.py` (1010 lines). Files read in
targeted ranges with line numbers recorded per citation:
`core/mistral7b_tc.py`, `tensor_cuda/src/kernels.cu`,
`tensor_cuda/src/bindings.cpp`, `tensor_cuda/tensor_cuda/quant.py`.

**Engine build provenance (checked, not assumed):** two engine trees
exist on disk —`/mnt/ForgeRealm/Project-Tensor/tensor_cuda` (git HEAD
`e3f2388`, `d57b76e` "fused APA kernel accepts head_dim 512 + bottom-
right causal fix" confirmed an ancestor via `git merge-base
--is-ancestor d57b76e HEAD`) and a second, unrelated
`/mnt/ForgeRealm/Project-Tensor/tc-speed/tensor_cuda` clone whose
`kernels.cu:976` still reads `TC_APA_MAXD = 256` and whose compiled
`.so` is dated Jun 10. `core/mistral7b_tc.py:23` (`sys.path.insert(0,
"/mnt/ForgeRealm/Project-Tensor/tensor_cuda")`) hard-codes the import
to the FIRST tree — the `tc-speed` clone is never on the Gemma port's
import path. The first tree's compiled `.so`
(`tensor_cuda/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so`,
Jun 28) predates `kernels.cu`'s Jul 2 mtime, but `git log --oneline --
tensor_cuda/src/kernels.cu` shows the only commits after `d57b76e` are
`c583119` (INT4 KV-cache pack/unpack) and `fa988d9` (GRM arena
cache-surgery) — unrelated to APA dispatch. `strings` on the `.so`
confirms `_ZN2tc20apa_selective_kernelI...Li512EEE` and
`_ZN2tc24apa_selective_bwd_kernelI...Li512EEE` template instantiations
are present in the binary. **Conclusion: the running engine is the
D=512-capable build; the mtime skew is a checkout artifact, not an
unbuilt fix.**

---

## 1. Expansion census

`_repeat_kv` is defined once, at `core/mistral7b_tc.py:172-177`
(unsqueeze+expand+reshape — a real materializing copy, not a view: the
`.expand` creates a broadcast view but `.reshape` after it forces a
contiguous copy since the expanded dim has stride 0).

### 1a. `core/gemma4_tc.py` — direct call sites

| Line | Code | Path reached | Materializes? |
|---|---|---|---|
| `gemma4_tc.py:612` | `_repeat_kv(k, H // KV)` | **Sliding-window layers only** (`else:` branch of `if self.is_global:` at line 544, i.e. taken when `self.is_global` is False — 40 of 48 layers). Both prefill and decode reach this via `F.scaled_dot_product_attention` at S_all ≤ window (line 610-613). | YES — expands `(B, 8, S≤1024, 256)` to `(B, 16, S≤1024, 256)`, a real 2× copy. At S=1024 that's `16·1024·256·2B(bf16)` ≈ **8.4 MB** per tensor (k and v each), ≈17MB total — small because sliding layers cap at window=1024 by construction. |
| `gemma4_tc.py:617` | `_repeat_kv(k, H // KV)` / `_repeat_kv(v, H // KV)` | Sliding-window layers, `S_all > W` branch (line 614-619, the masked non-square case). Same 40 layers, same bound (S never exceeds 1024 in this branch by the ring-trim contract at lines 620-625). | YES, same size class as above — bounded by `sliding_window=1024` regardless of total context. **This is NOT a global/APA-path expansion and does not scale with 4K/8K/12K context** — the registration note's "STILL LIVE" finding is correct as a grep hit but the ticket's actual target (the GQA-era 16× MQA blend expansion on the 512-dim GLOBAL layers) is a *different* code path (see 1b) that has already been de-expanded. |

**No other `_repeat_kv` call site exists in `gemma4_tc.py`** (grep-verified:
only 3 hits total in the file — 1 import at line 36, 2 calls at 612/617,
both traced above).

### 1b. Global-layer (D=512, 1 KV head) attention paths — the ticket's actual target

Global layers (8 of 48, `Gemma4Config.is_global`, `i % 6 == 5`,
`gemma4_tc.py:61-63`) reach FOUR distinct branches inside
`Gemma4AttentionTC.__call__` (`gemma4_tc.py:419-628`), none of which
call `_repeat_kv` on the 512-dim global K/V:

1. **Decode, standard (non-APA)** — `gemma4_tc.py:509-527`. Builds
   `qg = q.reshape([B, KV, rep, D])` (line 511, KV=1, rep=16) — a
   **reshape, not an expansion**, since q already has all 16 heads;
   `tc.matmul(qg, kv_cache.kb, ...)` (line 512) broadcasts the single KV
   head against 16 query-head rows inside the GEMM, no K/V copy. Zero
   materializing expansion.
2. **Decode, APA** (`apa_active`, lines 484-508) — calls
   `_cublas_blend_attention(q, kk, kq, vv, H // KV, ...)` (line 505-506)
   with `kk = kv_cache.kb.slice(...)` and `vv = kv_cache._v_get(...)` —
   both **unexpanded** (KV=1) tensors passed straight through. Inside
   `_cublas_blend_attention` (`mistral7b_tc.py:205-267`), the guard at
   `mistral7b_tc.py:216` (`if kH.shape[1] == 1 and H > 1`) is hit
   (KV=1 for Gemma globals) and takes the **MQA de-expansion branch**
   (lines 217-241): q heads fold into rows (`Qf = q.slice(...).reshape([B,
   1, H*bl, D])`, line 228) against the unexpanded `kqT1`/`kT1`
   (transposed views only, lines 222-223) — **no `_repeat_kv` call in
   this branch.** Confirmed by grep: `_cublas_blend_attention`'s MQA arm
   (lines 216-241) contains no `_repeat_kv` reference; the ONLY
   `_repeat_kv` calls inside `_cublas_blend_attention` are at line 242-243,
   inside the **non-MQA** `else` arm (the GQA case, `kH.shape[1] > 1`),
   which Gemma globals never reach.
3. **Prefill, standard (non-APA)** — `gemma4_tc.py:587-607`. `qf =
   q.reshape([B, 1, H*L, D])` (line 593) — reshape, folds heads into
   rows against unexpanded `k` (`(B,1,S_all,D)`). Zero expansion.
4. **Prefill, APA** — `gemma4_tc.py:546-586`. Same
   `_cublas_blend_attention` MQA-branch path as (2) when `S_all ≤
   fast_max_seq` (4096), or the fused `tc.apa_selective_attention` kernel
   (line 577-578) above `fast_max_seq` — the fused kernel is natively
   GQA/MQA-aware (`kernels.cu:1056` `kv_h = h / group`, no expansion
   ever, by kernel design) when `S_all > 4096`.

**Transient sizes at Gemma-4 12B global-layer geometry (1 KV head × 16 q
heads × D=512), per context length** — all four branches above are
confirmed expansion-free on K/V. The dominant materializing transient on
the global/APA path is NOT a `_repeat_kv` expansion but `_quantize_keys`
(`tensor_cuda/quant.py:111-129`, traced in §2 below), which the ledger
already identifies as the OOM driver (ledger lines 402-410, 552-558) and
which this audit independently confirms materializes ~5-6 simultaneous
fp32 `(B,1,S,512)` tensors:

| S | one fp32 (1,1,S,512) tensor | ×~6 (kd, unit, rotated, centroids, recon, +rotation-matrix broadcast `Rb.expand([B,1,512,512])`≈1MB fixed) |
|---|---|---|
| 4K | 512·4096·4B ≈ 8.4 MB | ≈ 44-50 MB (matches ledger's "S=4096 ... ~44 MiB" line 450) |
| 8K | ≈ 16.8 MB | ≈ 80-96 MB (matches ledger's measured OOM figure, lines 407-409, 445) |
| 12K | ≈ 25.2 MB | ≈ 140 MB (matches ledger line 502, 536, 576) |

This arithmetic reproduces the ledger's own numbers from the source
(`quant.py:111-129`) rather than re-quoting them — **independently
verified, not re-stated on trust.**

### 1c. Mask materialization

`_band_mask` (`gemma4_tc.py:324-339`) is cached per `(L,S,window,device,
dtype)` key (`_band_cache` dict, line 329) — a real per-shape tensor
build but cached across calls, so it is NOT a per-token materializing
copy in steady state (only on first hit of a new `(L,S)` shape, which
adaptive chunking makes numerous — the ledger's own "maskless global
prefill" fix at ledger:353-356 addresses this exact churn on the GLOBAL
path by using `causal_softmax`'s built-in mask instead; sliding layers
still use `_band_mask`, unaudited by that ticket since sliding context
is bounded at 1024 by design). No private/duplicate mask-copy pattern
found in `gemma4_tc.py` itself.

**Verdict for §1:** the registration note ("`_repeat_kv` STILL LIVE at
~lines 612, 617") is a correct grep result but a mischaracterization of
scope — those two call sites are on the **sliding-window path**
(bounded, cheap, never touches the 512-dim global geometry the mission
is about). The global/APA path that the ticket actually targets (GQA-era
16× MQA blend expansion on Gemma's 1-KV-head globals) is **already
de-expanded** in the current code, confirmed by tracing all four global-
layer branches to their unexpanded call sites. The live cost driver on
the global path is `_quantize_keys`'s O(S·D) transient, not `_repeat_kv`.

---

## 2. Kernel cap verification

- **`TC_APA_MAXD` in the live engine:** `tensor_cuda/src/kernels.cu:1035`
  — `constexpr int TC_APA_MAXD = 512;   // bumped 256->512 for Gemma 4
  global`. Confirmed this is the tree actually imported (see provenance
  note above; `core/mistral7b_tc.py:23`).
- **Did the June d57b76e commit raise it?** Yes — `git show
  d57b76e --stat` scope and `git merge-base --is-ancestor d57b76e HEAD`
  both confirm; the commit message itself ("fused APA kernel accepts
  head_dim 512 + bottom-right causal fix") matches the code found.
- **Is D=512 actually compiled into the running `.so`?** Yes —
  `strings tensor_cuda/tensor_cuda/_tensor_cuda.cpython-312-x86_64-
  linux-gnu.so | grep apa_selective` shows
  `_ZN2tc20apa_selective_kernelI13__nv_bfloat16Li512EEE...` and the
  `Li256E`/`Li128E`/`Li64E` sibling instantiations — all four dispatch
  arms are present as compiled template instances, not just source text.
- **Dispatch condition, traced in code** (`kernels.cu:1197-1219`):
  ```
  int cap = D > VD ? D : VD;
  if (cap > TC_APA_MAXD) throw ...
  ...
  if (cap <= 64)        launch(std::integral_constant<int, 64>{});
  else if (cap <= 128)  launch(std::integral_constant<int, 128>{});
  else if (cap <= 256)  launch(std::integral_constant<int, 256>{});
  else                  launch(std::integral_constant<int, 512>{});
  ```
  For Gemma globals, `D = VD = 512`, so `cap = 512`, which falls through
  to the last arm (`launch<512>`) — **the fused kernel CAN dispatch at
  D=512, confirmed by the dispatch arithmetic, not inferred.**
- **Bottom-right causal fix, traced:** `kernels.cu:1069-1073` —
  `int s_max = is_causal ? ((S - L) + i + 1) : S;` with the comment
  explicitly contrasting this against the old `s_max=i+1` top-left bug.
  Present in the same file/build as the D=512 arm.
- **Does the Gemma port actually REACH the fused kernel, or does it
  fall back to the blend?** Traced at `gemma4_tc.py:570-586`: `if S_all >
  self.fast_max_seq:` (line 570, `fast_max_seq=4096`, set at
  `gemma4_tc.py:417`) dispatches `tc.apa_selective_attention(...)` (line
  577), ELSE falls to `_cublas_blend_attention` (line 585). This gate
  exists ONLY in the **prefill** APA branch (lines 544-586, inside `if
  self.is_global:` / `apa_active` at line 546). **The decode APA branch
  (lines 484-508) has NO `fast_max_seq` check and unconditionally calls
  `_cublas_blend_attention` (line 505)** — confirmed by grep: `fast_max_seq`
  appears exactly once as a comparison (`gemma4_tc.py:570`), nowhere in
  the decode branch. This matches the ledger's own claim (ledger:610-611,
  "decode stays on the blend") and the stated rationale (fused kernel is
  a decode loser by structure, ledger:685-690) — **verified consistent
  between ledger and code, not just asserted by the ledger.**

**Verdict for §2:** the D=512 fused-kernel claim is CONFIRMED at every
level checked (source constant, compiled symbol, dispatch arithmetic,
call-site gating). The Gemma APA path dispatches the fused kernel ONLY
in prefill above 4096 tokens; decode always uses the blend by design
(not a silent fallback bug — an explicit, documented architectural
choice traceable in both files).

---

## 3. June-fix reachability

| Claimed fix | Exists? | Reached from current call graph? | Evidence |
|---|---|---|---|
| **Incremental kq ring** (`kqb`/`kq_count`) | YES | YES | `KVRing.__slots__` includes `kqb`, `kq_count` (`gemma4_tc.py:164-165`), initialized `None`/`0` in `__init__` (lines 175-176). `quantized_keys()` method (lines 275-304) is the only writer of `kqb`/`kq_count`. Called from exactly one site: `gemma4_tc.py:496-497`, inside the decode `apa_active` branch (line 484), which is reached when `L==1 and kv_cache is not None` (line 474) and `apa_active` (line 481-483: `self.is_global and self.attention_mode=="apa_selective" and S_all > self.apa_min_context`). Live so long as a caller sets `attention_mode="apa_selective"` on a global layer past `apa_min_context=2048` tokens — confirmed reachable, not dead code (single call site, on the hot decode path). |
| **Chunked cold-start quantize** (CHUNK=512 in `quantized_keys`) | YES | YES | `gemma4_tc.py:295-302`: `CHUNK = 512` then `for s0 in range(self.kq_count, self.count, CHUNK):` — this loop is the body of `quantized_keys`, same single call site as above (line 496). Reached whenever `new = self.count - self.kq_count` exceeds one row (cold start) via the same decode-APA path. Also present: chunked whole-span quantize for PREFILL APA at `gemma4_tc.py:558-565` (`if S_all > 4096: for s0 in range(0, S_all, 2048): ...`) — a second, separate chunking site, both confirmed live inside the two `apa_active` branches (decode: line 484-508; prefill: line 546-586). |
| **Bounded `_grow_cap`** | YES | YES | Defined `gemma4_tc.py:128-139` (double-while-small, then `+2048` block growth, matching ledger:493-497's "8K cap 16384→10240" description of the same algorithm). Called from `KVRing.__init__` (line 183) and `KVRing.append` (lines 239-240) — both on the live decode-append path (`append` called from `gemma4_tc.py:479`, inside the `L==1` decode branch that runs on every decode step for every layer). Confirmed reached on every decode token, not just at construction. |
| **Canonical-mask slicing in the blend (no private mask copies)** | YES | YES | `mistral7b_tc.py:253-260` comment explicitly documents this as a FIX for a prior bug ("this path used to build its OWN triu(k=i+1) ... duplicated"); current code at line 261 calls `F._causal_mask(L, S, ...)` — the shared/canonical mask function, not a private construction. Reached from the non-MQA arm of `_cublas_blend_attention` (line 252-264); Gemma's MQA arm (lines 216-241) uses `F._causal_mask` too, at line 234, same function. Both arms of the one function Gemma calls use the canonical mask — confirmed, no duplicate mask-building code found anywhere in `gemma4_tc.py` or the reachable part of `mistral7b_tc.py`. |
| **`write_rows` ring machinery** | YES | YES | Engine op `tc.write_rows` called 8 times in `gemma4_tc.py` (lines 199, 207, 208, 210, 252, 269, 272, 300) — covers KVRing construction, V-quant write, growth (`_grow1`), decode append (K, V, bias-unmask), and kqb chunk-write. All on paths already traced live above (construction, decode append, incremental-kq). Engine-side: `write_rows` symbol confirmed to exist as a bound op (`bindings.cpp` — not re-grepped in this pass beyond confirming the Python-side call sites resolve to `tc.write_rows`, a `tensor_cuda` engine attribute used throughout `gemma4_tc.py` without a local shim, so it resolves to the compiled engine binding). |

**Verdict for §3:** every claimed June fix is BOTH present in source AND
reached from the live call graph via at least one concrete call site
traced to the hot decode or prefill path. No dead/bypassed fix found in
this file. This is a stronger result than the mission's DISTRUST prior
predicted going in — worth flagging as the audit's most surprising
finding (see report-back).

---

## 4. Verdict table — ticket by ticket

| Ticket (ledger: line ~306 order) | Status | Evidence (file:line) |
|---|---|---|
| **Ring buffers** | **DONE** | `KVRing` class (`gemma4_tc.py:142-321`) implements fixed sliding-window ring (`ring_cap` param, line 167) + capacity-doubling globals (`_grow_cap`, lines 128-139) + in-place `write_rows` (line 199, 269, etc.) + bias-masked invalid rows (`self.bias`, lines 192-194, 263-265, 271-272). Ownership contract documented and consumed-list pattern enforced at `gemma4_tc.py:768-775` (`caches[i] = None` after each layer consumes its cache). |
| **Blend de-expansion** | **DONE for the global/APA path that was the actual ticket target; the two remaining `_repeat_kv` sites are on the sliding-window path (bounded ≤1024, not in scope of the GQA-era 16×/512-dim complaint)** | `_cublas_blend_attention`'s MQA arm (`mistral7b_tc.py:216-241`) is expansion-free and is the arm Gemma's global/APA calls hit (`gemma4_tc.py:505-506, 585-586`). Standard (non-APA) global decode/prefill also expansion-free via reshape-not-copy (`gemma4_tc.py:511-512, 593-594`). Remaining `_repeat_kv` calls: `gemma4_tc.py:612, 617` — sliding layers only, per §1. **Grep-clean claim from the ledger (line 51-52, "no `_repeat_kv` materialization anywhere in the attention path") is FALSE AS LITERALLY STATED (grep is not clean) but TRUE IN SUBSTANCE for the global-layer path the mission is about.** This is a real discrepancy between the registration note's framing and the code, worth flagging explicitly. |
| **Re-probe readiness (expect APA > standard)** | **NOT YET MEASURED — this audit is static only, per Phase A1 scope. Phase A0 (re-probe on the 4070S) is a separate, not-yet-executed step per the plan.** | N/A — out of scope for A1; flagging that A0 must run before B3's gate has any numbers to compare against. |
| **V-side APA** | **PARTIAL — INT8 V-storage done and gated; 4-bit V (global) measured but NOT shipped as default; asymmetric K8+V4 measured best-of-table but explicitly held behind a confirmation run** | `KVRing.QUANT_V` flag (`gemma4_tc.py:162`, env `GEMMA4_QUANT_V`, default OFF per line 162 `bool(int(os.environ.get(...,"0")))`) — confirms ledger's "shipped default is V-only INT8, conservatively" (ledger:563) is accurate: the flag being OFF by default is verified in code, not just asserted in the ledger. `_v_put`/`_v_get` (lines 203-224) implement the quantize-on-write/dequantize-on-read path. No K8+V4 asymmetric mode found wired into `KVRing` itself (only uniform `QUANT_V` for V; K stays bf16 unconditionally — no `QUANT_K` flag or equivalent exists in `gemma4_tc.py`, confirmed by grep for "QUANT_K"/"quant_k": zero hits). **The asymmetric K8+V4 mode described as "the best mode in the entire table" (ledger:552-558) is NOT implemented in `core/gemma4_tc.py` — it exists only as a ppl-sweep measurement via the external `KV_STORE_HOOK` test harness (`gemma4_tc.py:383-386, 463-464`), not as a shippable code path.** This is a load-bearing gap: the ledger's own "CAVEAT" (line 559-563) already flags this as unconfirmed/unshipped, and this audit confirms it is additionally *unbuilt* as a real KVRing mode, not merely unconfirmed. |

---

## Report-back summary (for the calling agent)

**Verdict table:** ring buffers DONE; blend de-expansion DONE on the
global/APA path (the actual ticket target) but the registration note's
"grep clean" framing is technically false — two `_repeat_kv` calls
remain, both on the bounded sliding-window path, out of scope for the
512-dim global complaint; re-probe (B3) not yet run (A0 is a separate,
unexecuted step); V-side APA PARTIAL — INT8 V shipped OFF-by-default
and gated, but the measured-best asymmetric K8+V4 mode is not built into
`KVRing` at all, only measured via an external test hook.

**Three most load-bearing findings:**

1. **The registration note's "`_repeat_kv` STILL LIVE" finding
   (mission doc lines 16-19) is a real grep hit but points at the wrong
   code path.** `gemma4_tc.py:612,617` are sliding-window-layer calls
   (bounded ≤1024 context by the ring-trim contract at lines 620-625),
   not the GQA-era 16×/512-dim global expansion the ticket is about. All
   four global-layer branches (decode-standard, decode-APA,
   prefill-standard, prefill-APA — `gemma4_tc.py:509-527, 484-508,
   587-607, 546-586`) are confirmed expansion-free, tracing into
   `_cublas_blend_attention`'s MQA de-expansion arm
   (`mistral7b_tc.py:216-241`) or the natively-GQA-aware fused kernel.
   Practical effect: **Phase B2 ("blend de-expansion, TOTAL... grep
   clean") as literally scoped in the plan cannot pass a pure grep-for-
   `_repeat_kv` gate without also touching the sliding-window path**,
   which is architecturally out of scope for the mission's stated
   problem (Gemma's 1-KV-head globals). The plan's B2 gate wording needs
   reconciling with this distinction before it's run.

2. **D=512 fused-kernel dispatch is real and verified at every layer
   checked** (source constant `kernels.cu:1035`, compiled symbol via
   `strings` on the actual `.so`, dispatch arithmetic `kernels.cu:1197-
   1219`, and the Gemma-side gating condition `gemma4_tc.py:570`) — this
   was the highest-risk claim to re-verify (compiled-binary claims are
   the easiest to silently regress) and it holds up. The two engine
   trees on disk (`Project-Tensor/tensor_cuda` vs the stale
   `tc-speed/tensor_cuda` clone with `TC_APA_MAXD=256`) could have been a
   silent-fallback trap if the import path were wrong; confirmed
   `mistral7b_tc.py:23` hard-codes the correct tree.

3. **The asymmetric K8+V4 storage mode — the ledger's own "BEST mode in
   the entire table" (ledger:552-558, −3.56% ppl) and the mission plan's
   B4 ticket — does not exist as a code path in `KVRing`.** Only
   uniform V-only INT8 (`QUANT_V`) is wired; there is no `QUANT_K` or
   equivalent. It was measured through the external `KV_STORE_HOOK`
   test-only hook (`gemma4_tc.py:383-386`), which round-trips K/V for
   ppl measurement but is not the same code as the actual `KVRing`
   storage/dequant path used in real serving. This means B4 has a
   ppl number but zero implementation — don't read "measured" as "built."

**Audit file:** `/mnt/ForgeRealm/GraftRepository/docs/GEMMA4_APA_AUDIT_A1.md`
(this document — the only file written).
