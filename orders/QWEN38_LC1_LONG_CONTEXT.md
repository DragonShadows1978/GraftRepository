# ORDER QWEN38-LC1 — Qwen3.8-27B INT3 long-context enablement (StoryScope target)

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/GraftRepository` — edits/builds/test
runs AUTHORIZED. You are the implementation seat; the lead runs all GPU gates
(your sandbox has NO GPU — this is known and expected; do not report it as a
blocker, author the gates so the lead can run them).

## Mission

The Qwen3.8-27B INT3 adapter (`core/qwen38_tc.py`, over `core/qwen35_tc.py`,
demo `scripts/qwen38_generate.py`) currently runs at an effective ~4K context.
Checkpoint native window is 262,144. Mission: make context length a
first-class, configurable dimension and push the usable window to **≥16K
required, 32K target** on the single 12GB RTX 4070 SUPER (whole-device budget:
peak ≤ 12,000 MiB sampled by nvidia-smi; desktop uses ~350 MiB).
**Speed is explicitly NOT a goal** — David's directive. Slow and correct wins.

Downstream consumer (context sizing rationale, read-only):
`/mnt/ForgeRealm/storyscope` — worst-case single-call windows ~12–16K tokens
(a ~5,000-word story + aspect prompt + JSON output).

Facts from today's live run (receipts in `logs/qwen38_demo_r2.log` if the lead
saved it; numbers verified live 2026-08-19):
- Resident VRAM 9.78 GiB (INT3 body 9.20 + INT3 lm_head 0.48 + norms/aux).
- Decode peak 11,666 MiB whole-device at seq≈160; prefill peak 11,279.
- KV cost 64 KiB/token bf16 (16 full-attention layers × 4 KV heads × K+V ×
  head_dim 256); the 48 GatedDeltaNet layers hold a FIXED 0.146 GiB state.
- lm_head dequant transient 317,849,600 B (chunk rows 31,040, 8 chunks).
- `extend_rope` exists and grows from an initial 4096 alloc
  (`core/qwen38_tc.py:571-588`).

## Work items

**LC1.0 — Ceiling map (first, cheap).** Trace and document (in the report)
every context-length limiter in the current path: KV cache allocation policy,
prefill chunking behavior (does the Qwen3.5 port's chunker apply?), RoPE
extension, positions/masks, any hardcoded 4096s, transient peaks by phase.
No behavior claims without a code citation (file:line).

**LC1.1 — Context plumbing.** `--max-context N` on `scripts/qwen38_generate.py`
(and adapter config): pre-allocates/validates KV + RoPE for N; fail-loud if
N > 262144 or if the VRAM budget math says it cannot fit (print the per-
component budget before loading, same style as the existing memory map).

**LC1.2 — Transient trim.** (a) `--lm-head-chunk-rows` configurable; pick a
default that caps the dequant transient at ≤64 MiB. (b) Chunked/windowed
prefill if not already bounded: prefill transient peak must not scale with
total sequence length. Correctness over speed everywhere.

**LC1.3 — INT8 KV cache, flag `--kv-int8`, default OFF.** Per-token, per-head
scales (asymmetric or symmetric — justify the choice in one paragraph;
WARNING from APAMQ: naive per-vector symmetric quantizers amplified
massive-activation dims on Gemma; check K distributions before choosing).
Dequant to bf16 at use. Bit-exact pack/unpack unit test that runs CPU-side.

**LC1.4 — Host-RAM KV, flag `--kv-host`, default OFF.** KV lives in host
(pinned if easy, pageable acceptable — speed irrelevant), streamed to a fixed
VRAM ring per attention layer at use. VRAM cost must be O(1) in context
length. This is the 262K-capable rung; correctness gates only.

**LC1.5 — Gate harness, `scripts/qwen38_longctx_gate.py`.** Deterministic:
- Needle protocol: filler from a fixed seeded generator (no downloads),
  planted fact ("The vault access code is <SEED-DERIVED>") at depth fraction
  d ∈ {0.25, 0.5, 0.75}, question last, greedy decode, exact-substring check.
- `--length L --depth d --seed s` + JSON receipt line per run
  (RECALL_RESULT {...}) including the nvidia-smi peak sampler from the demo.
- Coherence smoke mode: continue a long document 64 tokens, print raw.
- A `--baseline-check` mode: re-run today's 3 demo prompts at default config
  and diff token_ids against the values embedded from today's receipt (lead
  will supply; leave a REPLACE-ME constant clearly marked).

## Registered gates (lead runs on GPU; thresholds fixed NOW, before results)

- **G-LC0 invariance**: demo token_ids at default 4K config identical to
  today's run (greedy). Any diff = RED, stop.
- **G-LC1 VRAM**: peak ≤ 12,000 MiB whole-device at every tested rung.
- **G-LC2 recall**: 8K: 3 depths × 2 seeds, pass ≥5/6. 16K: 3 depths × 1
  seed, pass ≥2/3. 32K: 3 depths × 1 seed, REPORT-ONLY (protocol-limited by
  prefill wall-clock; not a pass/fail gate this round).
- **G-LC3 INT8 KV**: at 4K, greedy top-1 agreement ≥99% over 64 decode steps
  vs bf16 KV, AND 8K recall count no worse than bf16 arm. RED → flag stays
  default-OFF and is reported RED (that is a valid result).
- **G-LC4 coherence**: smoke output at the max achieved context is coherent
  prose (lead adjudicates; raw output in receipts).

## Boundaries

- Writable: this repo only. READ-ONLY: `/mnt/ForgeRealm/Project-Tensor*`,
  `/mnt/ForgeRealm/models/`, `/mnt/ForgeRealm/storyscope`,
  `artifacts/qwen38_int3_cache` (read the cache, never regenerate/modify it).
- Do NOT touch attention math or the APA path — standard attention only;
  APA is a separate registered order. Do not change quantization of weights.
- NO git (lead commits). NO subagents. NO network. No monitor-idling: run
  your CPU-side unit tests synchronously to completion.
- RED honesty: a failed gate or an unreachable rung is a RESULT — report it
  with receipts, never soften it.

## Done

Final message MUST contain verbatim:
1. Files created/modified with one-line purpose each.
2. CPU-side unit test output (pack/unpack, harness dry-run at tiny length).
3. The exact commands (copy-paste ready) for the lead to run: G-LC0, the
   VRAM budget print, each recall rung (8K/16K/32K, both KV modes), the
   coherence smoke, and the 262K-capable `--kv-host` invocation.
4. LC1.0 ceiling map (the file:line citation list).
5. Residuals/risks you did not address, stated plainly.

## Addendum (registered 2026-08-19, before any gate results)

David's directive clarified: reliability/functionality only, wall-clock
irrelevant. G-LC2 32K is UPGRADED from report-only to a real gate:
3 depths × 1 seed, pass ≥2/3. If the day allows, lead deepens seeds at all
rungs (extra runs strengthen, never replace, the registered thresholds).
