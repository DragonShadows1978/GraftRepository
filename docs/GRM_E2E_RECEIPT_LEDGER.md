# GRM Composed End-To-End Receipt Ledger

Execution record for the composed end-to-end receipt work order.
Immutable plan: `docs/GRM_E2E_RECEIPT_PLAN.md`. Narrative continues in
`docs/GPT_OSS_20B_APA_GRM_SYNTHESIS.md`.

## 2026-07-08 (opening)

Action: Work order opened per David's session goal ("Once completed
[P3], begin implementing the composed End-To-End receipt"). Plan
committed immutable alongside this entry.

Inherited state:
- P3 packed format shipped and closed (630a581): INT8/INT6 at rest,
  bit-identical recall, measured 1.79×/2.33× zlib disk — and the honest
  seam this receipt must price live: packed mounts 3.76× slower/layer
  (CPU dequant). E4's stable-VRAM leg uses the packed store; the mount
  cadence cost is one of the seams P2's read is FOR.
- Both route dialects on CUDA (GQA bridge 1.26-1.44× direct; MLA 2.22 ms
  at 1M), epoch staleness law, supersession machinery, 96k envelope.

Next action: P0 composition map (Sonnet, flat, read-only) — what the
live loop needs vs what exists; the key unknown is live per-turn
witnessed deposit (all existing gates capture offline).

## 2026-07-08 (P0 complete — composition map; plan premise corrected)

Action: P0 map landed (Sonnet, read-only;
artifacts/grm_e2e/P0_COMPOSITION_MAP.md — relocated from the agent's
mis-pathed ~/artifacts).

Findings:
- PREMISE CORRECTION (the plan's "key unknown" was wrong): live
  witnessed deposit EXISTS and is proven — deposit_from_cache()
  (graft_arena.py:314-341) slices K/V from the live VRAM caches, is the
  DEFAULT deposit path in step(), and the DeepSeek-MLA + generic-GQA
  suites exercise the full live loop today. Offline capture was a
  GPT-OSS-gate idiosyncrasy only.
- REAL CENTRAL RISK: GPT-OSS-20B has never been driven live through
  ArenaCache/GRMRuntime.chat — its cache layout (full+sliding layer
  types, YARN RoPE) against the dialect-generic slicing math is the
  untested seam. Wire-and-verify, not build.
- Exists/wire: evict() IS the live-window hook; CUDA route + packed
  store are env flips already load-bearing; flush_now/load persists the
  repository (grafts, route index, native ids, epoch) but never live
  caches — restart re-seeds by re-feed()ing the transcript (driver
  responsibility, by design).
- Build (small): per-turn route-wall timer (one-line return or wrapper);
  probe scorecard reuses the LEXICAL grader pattern (_grounded + accept
  lists) — no logit-margin scorer exists in-repo and none is needed.

P1 architecture (from the map's sketch): scripts/grm_e2e_session.py —
GRMRuntime.chat + gpt_oss dialect kwargs; default live deposits;
scripted facts/supersessions/probes; evict() policy; flush_now →
process restart → load → re-feed for the durability leg;
GRM_GQA_CUDA_ROUTE=1 + GRM_GRAFT_STORAGE_BITS=8. FIRST LEG = the risk:
a 2-3 turn GPT-OSS live-deposit smoke proving deposited grafts
round-trip (mount back + recall) before the full session is attempted.

Next action: P1 (Sonnet, flat).

## 2026-07-08 (P1 Leg 1 — RED at diagnosis; prerequisite build ordered)

Action: Leg-1 agent STOPPED correctly at a static diagnosis, zero GPU
spent. P0-map correction accepted.

Findings:
- GPT-OSS-20B has NO full-model class in the repo. ArenaCache binds
  self.m and requires .layers/.rope_cos/.rope_sin/.extend_rope()/
  __call__(ids, last_token_only=True) (graft_arena.py:56,67,160-250,
  1465,1620,1760; kv_graft.py:45-77). core/gpt_oss20b_tc.py provides
  per-block primitives only; "GptOss20B_TC" exists solely as a dialect
  metadata STRING (:202). All existing gates subprocess-drive
  stream_forward_smoke.py's hand-written per-layer loop (:430-492, YARN
  tables computed outside any model object at :395).
- P0-map correction: the gap is one level below "wire, not build" — the
  model object itself must be built before the actual named risk
  (dialect-generic cache slicing vs full+sliding mix + YARN) is even
  testable.
- DECISION (lead): build GptOss20B_TC as the in-plan P1 prerequisite
  (the plan's P1 clause authorizes product code exactly where P0 names a
  genuine gap; the corrected P0 names this one). Scope: embeddings +
  .layers of existing blocks + YARN RoPE table ownership/extend_rope +
  incremental KV-cache forward across the full/sliding mix + MoE
  dispatch, from_pretrained. PARITY LAW: identical input must reproduce
  stream_forward_smoke.py's captures/logits (the smoke IS the reference
  implementation; deterministic engine ⇒ near-bit parity expected).
  Precedents: Qwen35_TC (hybrid cache), Gemma4_TC (sliding mix).

Next action: model-class build (Sonnet, flat), then Leg 1 re-run, then
the session driver.

## 2026-07-08 (prerequisite build complete — GPT-OSS runs the live loop)

Action: GptOss20B_TC built (Sonnet, flat, full validation ladder),
lead-verified, committed.

Findings:
- Class = ~215-line container around the EXISTING block primitives (no
  math reimplemented). Contract grep-verified against every ArenaCache/
  kv_graft touchpoint. Convention discovery: ArenaCache calls
  self.m(..., kv_caches=...) (Mistral/Qwen3 style) — Qwen35_TC/Gemma4_TC
  use caches= and have never been arena-driven; class accepts both,
  rejects both-given.
- PARITY LAW: ZERO deviation — all 24 layers' pre-RoPE K/V bit-identical
  to stream_forward_smoke on the real snapshot (max_abs_diff_overall=0),
  top-5 logits exact. Incremental decode == full-refeed token-for-token
  (16-tok greedy), 1.12s vs 6.76s.
- NEW ENVELOPE RECEIPT: all-24-layers RESIDENT load = 10852/12282 MiB
  (the smoke always streamed one layer at a time; full residency was
  untested). ~1.4 GB headroom — E4's long-session VRAM leg will stress
  it.
- LEG-1: HIT — first-ever live GPT-OSS drive through GQAArenaCache:
  deposit_from_cache well-formed on BOTH layer types, evict confirmed,
  route ranked the fact turn first, swap mounted it, and the model
  answered 'Zeta-7-Quebec' PURELY from the remounted graft. The seam the
  work order was stopped on is green.
- Pre-existing gap surfaced (false-negative en route, root-caused):
  ArenaCache.step()'s hardcoded "User:/Assistant:" template is not
  Harmony format; GPT-OSS misses through step() but hits through
  format-agnostic feed() + the same underlying machinery. DRIVER
  DECISION REGISTERED for Leg 2: prefer a minimal dialect-appropriate
  template hook in step() if it stays small and suite-green (production-
  representative receipt); else the driver implements the turn loop at
  feed()+route()+swap() level. GRMRuntime.chat() calls step() — the
  choice determines which entry the receipt exercises.
- Suites: 21/21 + 117/117 + 18/18 (5 new class tests).

Next action: Leg 2 (session driver) — dispatched to codex-shim (first
Codex-as-subagent work order; sandbox = GraftRepository).

## 2026-07-08 (Leg 2 delivered — Codex work order; four seams named)

Action: session driver built by CODEX (first Codex-as-subagent dispatch,
raw shim, sandboxed workspace-write, 38 min, 301k tokens), lead
spot-checked, committed. Suites 118+21+18+22 green.

- Template decision: minimal hook chosen (ArenaCache prompt_template +
  stop_sequences, defaults unchanged) — driver runs the REAL
  chat()→step() path in Harmony format. Registered criterion met.
- BONUS REAL BUG (P3 escape): repository RAM normalization wraps scalar
  packed npz fields into 1-element arrays; unpack_kv_arrays rejected
  them. P3's cycle test used direct npz and missed the normalization
  path. Fixed fail-closed (0-D/size-1 accepted, wider rejected).
- Smoke: 10 turns + restart completed; instrumentation and durability
  machinery work (flush 0.55s; VRAM flat 10.9GB across exec/refeed).
  PROBES 0/2 — with the memory machinery GREEN (eviction verified,
  route found the graft, mount applied): the failure is GENERATION
  drift (meta/refusal text instead of answer-first) through step()'s
  loop; Leg-1's manual generate loop got a clean answer on the same
  machinery — a loop-parameter delta (stops/length/template nuance),
  not a memory failure.
- SEAMS NAMED (E3 red; P2 full run HELD until seam 1 is diagnosed):
  1. Probe generation drift (above) — first diagnosis target.
  2. CUDA GQA route never engages on LIVE banks: turn grafts are
     RAGGED (per-turn token counts) and the bridge contract demands
     dense same-shape single-key banks — a capture-world assumption.
     Design input for the synthetic-centroids successor.
  3. Native route 609-1087 ms at ~10 nodes (harness does 10k in 6 ms) —
     suspect per-turn arena re-preparation after each deposit epoch
     bump. Needs its own profile.
  4. Packed resume load ~23 s at smoke scale — dequant × load path
     compounding; policy echo of P3's mount-cost finding.

Next action: seam-1 diagnosis (compare Leg-1's working generate loop vs
step()'s), then seams 2-3 as scoped follow-ups, then P2 full run.
