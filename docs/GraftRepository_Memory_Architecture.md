# Graft Repository: Routed, Hierarchical, Tokenless Memory on a Frozen Model

**Design document — 2026-06-10. Working name: Graft Repository (naming TBD; it is
the native-representation successor to AfterImage).**

One line: store every turn and document as harvested K/V ("grafts"), route into
them per turn with APA's own scoring, mount only the winners into a fixed
positional arena, and let a background librarian consolidate old memory into
hierarchical digests — unbounded memory at bounded residency, zero training,
on a frozen model.

What it kills, for the consumer and the agent alike:
- **The slow death** — KV cache growth with conversation length. The live
  context stays permanently small; history lives in the repository.
- **Silent amnesia** — eviction hacks that quietly drop old turns. Nothing is
  dropped; everything is routable.
- **The re-read toll** — text memory (RAG, AfterImage, system-prompt summaries)
  re-pays tokenization + prefill every session. Grafts mount as computed
  meaning at zero prompt cost.

---

## 1. Foundations (already measured — this design builds on receipts, not hope)

| Property | Status | Receipt |
|---|---|---|
| Graft ≡ in-context (lossless) | **Proven** | Logit-level equivalence: top-1 identical, max logit diff ~bf16 noise, per-layer residual rel-diff 0.0035–0.019 |
| Pre-RoPE capture → relocatable keys | **Proven** | Harvest hook post-qk-norm / pre-RoPE; re-RoPE at any seat |
| APA selection is free at depth | **Proven** | r0.25/r0.10/r0.05 @ 20,480 within ±0.02 ppl of each other |
| Memory ceiling = trained window, not card | **Proven (MLA)** | MiniCPM3: every rung 2,048→32,768, resident flat ~2,856MB |
| Mounted content readable at depth | **Proven** | Fixed-window test: same 512 tokens, ppl 9.80→9.40 as prefix 8K→20K |
| Position relocation via rotation composition | **Math** | RoPE(p+Δ) = R(Δ)·RoPE(p); one vectorized pass re-seats cached keys exactly |
| Secondhand memory ("haunting") | **Measured 2026-06-10: carries AWARENESS, not facts** | Unmount-with-sink-kept: model coherently reaches for "the previous interactions" but recalls 0/5 needles. Digests must spell facts in their own tokens; verbatim recall is the router's job (re-mount on demand) |
| Turn harvest is free | **Measured 2026-06-10: PAYLOAD yes, ROUTING KEY no** | Cache-sliced (c_n, un-RoPE'd k_pe) re-mounts with full verbatim recall — contextualized K/V is a valid graft payload, zero extra forwards. But a centroid from contextualized latents is polluted by the running conversation (early turns become routing attractors: E4-arena 5/6, mounts collapsed onto turn 1). Hybrid ships: payload from cache + key from a layers-0..44 partial forward → 6/6, feeds 0.48→0.41s |
| Graft survives a persistent cache | **Proven 2026-06-10** | Prefill-only injection + persistent `graft_seats` shift: 5/5 recall across turns; teacher-forced A-vs-C logit diff 0.5-1.5 = the plain cache's own bf16 noise floor (0.75-0.9), top-1 identical at every position |
| Seat 0 is sacred (attention sink) | **Measured 2026-06-10** | Unmounting a graft including its first seats → repetition collapse; keeping ~4 sink seats restores coherence. Seating plan is [SINK | arena | recency | live] — the arena must never start at seat 0 |
| Routing works — and BEATS mount-all (E1) | **Measured 2026-06-10, BOTH models** | Qwen3: layer-0 |q·k| router, routed top-3 10/10 vs mount-all 4/10. MiniCPM3: key-space router FAILS (no qk-norm → outlier keys); LATENT-centroid cosine router confirms at 9-10/10 recall@3, routed 7/10 = mount-all 7/10. Router format is per-model (part of the dialect) |
| MLA latent grafting (MiniCPM3) | **Proven 2026-06-10** | Graft = (c_n latent, pre-RoPE k_pe), 288 vals/tok/layer (51-tok graft = 3.6MB host, all 62 layers). G1 graft-vs-in-context max diff 0.41 top-1 IDENTICAL; G2 cached-stream at plain floor (flips only at exact-tie margins 0.0000); G3 recall 3/3 |
| End-to-end routed conversation memory (E4) | **PASSED 2026-06-10** | 20-turn conversation: baseline 6/6 (670-858 ctx tokens, growing); routed memory 5/6 pre-registered, 6/6 with routing hygiene (~25% residency, BOUNDED); amnesia control 0/6. Hygiene: route on bare user text; don't deposit retrieval-only turns |
| Consolidation end-to-end (E4-C) | **PASSED 2026-06-10** | Turns 1-8 (all six facts) consolidated into TWO QC'd digest grafts, sources retired: probes 6/6 through digests alone = direct memory. Routing pool 14→8 nodes; every probe's top-3 covered all six needles (consolidation widens coverage/seat). Hierarchical descent (digest scores by max over child centroids) routed perfectly even when round-1 digests were fact-free |
| Shuttling — trips beat a bigger arena | **PASSED 2026-06-10** | Arena starved to ONE mount: 5/6; + max_trips=2 with grounding checks: **6/6 = the 3-mount arena.** Failed attempts roll back entirely (immutable cache tensors make snapshot/restore free — old list + position). Grounding = hedge detection + content-emptiness (deflections) + content-tokens ⊆ mounted sources; false trips cost latency only (fallback = first attempt). Trips trade latency for seats — the design's overflow law, measured |
| FULL REPOSITORY: auto-librarian + persistence + cross-session resume | **BUILT + PASSED 2026-06-10** | `core/graft_repository.py` GraftRepository (chat / add_turn / add_document / save / load / stats). Build session: 14 turns + 2 docs, librarian auto-fired at the 8-active-turn threshold (2 digests formed mid-conversation, sources retired + VRAM-freed to disk cold storage); 18 nodes = 24.9MB. FRESH PROCESS resume (dialect-guarded, descent keys rebuilt from lineage, only active nodes re-uploaded): **6/7 recall from disk artifacts alone** incl. document fact; trips fired and recovered during resume. Miss = the no-identifier offsite probe (topical two-digest routing, grounded-but-wrong — the known residual). Live cache deliberately NOT persisted: history lives in the repository |
| Corpus scale + sibling confusability (CORPUS-100) | **PASSED 2026-06-10: 20/20** | 100 chunks = 10 families × 10 near-duplicate siblings (codes differ), 20 identifier-keyed probes, 416MB device grafts, 50KB index, 1.3s/probe. Took FOUR measured fixes: latent-only routing 4/20 @1 (centroids can't separate siblings) → +LEXICAL channel (probe identifier tokens vs source rare-tokens, exact match dominates) 20/20 routing; co-mounted siblings collapse reads (16/20) → PRECISE-MOUNT (identifier query = point lookup, mount rank-1 alone); live-window echo of the previous same-family Q&A beats the mounted doc and repeats across same-window retries → CLEAN-ROOM trip (fresh mini-cache); ladder order = precise → clean → siblings for identifier queries. End recall **20/20** |
| PERSISTENT ARENA (swap/evict as cache surgery) | **BUILT + PASSED 2026-06-10** | E4-arena: 6/6 on ONE never-rebuilt cache through 20 turns, 6 routed swaps, per-turn evictions. Seating [SINK 6 | ARENA 256 | LIVE ~130]; residency 268-316 seats flat. live_shift = fixed arena width (decoupled from mount size); mounts occupy an arena prefix, remainder is a positional hole; MLA swap re-RoPEs only the 32-d k_pe |

Key vocabulary carried forward: **seats** = position range inside the trained
window (the scarce, unbuyable resource); **boat** = the live context;
**arena** = a fixed positional partition reserved for mounts;
**dialect** = a model's private residual-stream basis (grafts are model-specific).

---

## 2. System overview

```
                        ┌─────────────────────────────────────────┐
                        │              COLD PATH                  │
                        │  Librarian (AtlasForge mission, async)  │
                        │  - clusters aging turns (text-level)    │
                        │  - infers graph edges                   │
                        │  - schedules consolidation              │
                        │  - serving model digests in idle time   │
                        └───────────────▲─────────────────────────┘
                                        │ deposits / promotions
   ┌────────────────────────────────────┴───────────────────────────────┐
   │                          REPOSITORY (disk)                          │
   │   turn grafts · doc grafts · digests · era grafts                  │
   │   each node: K/V artifact + summary keys + graph edges + tags      │
   └───────────────▲────────────────────────────────────▲───────────────┘
                   │ page in winners                     │ harvest deposit
   ┌───────────────┴───────────────┐      ┌──────────────┴───────────────┐
   │  QUANTIZED INDEX (resident)   │      │        HOT PATH (per turn)   │
   │  summary keys of every node   │◄─────│  prompt → early-layer fwd    │
   │  at 4/8-bit (APA bulk format) │ score│  → bulk-score index          │
   └───────────────────────────────┘      │  → mount winners in arena    │
                                          │  → full generate             │
                                          │  → harvest turn (free)       │
                                          │  → deposit                   │
                                          └──────────────────────────────┘
```

The router is **not a model call**. It is APA's bulk pass — quantized
query·key scoring — pointed at the repository's summary keys instead of the
live context. A graft's keys are its own routing interface: content is
self-describing, the address is the payload. This is why no router training
exists anywhere in this design, and why nodes can be added/removed freely —
the property MoE experts can never have.

---

## 3. The hot path (per-turn pipeline)

```
 prompt
   │
   ▼
 [1] partial forward (layers 0..k), only the RECENCY WINDOW mounted
   │        └── produces the prompt's own queries = routing signal
   ▼
 [2] bulk-score quantized index (era level → episode level → node level)
   │        └── hierarchical descent: open a level only if its parent wins
   ▼
 [3] mount: page winners' full K/V from disk → seat in ARENA (re-RoPE at
   │        arena positions) → recency window keeps its reserved seats
   ▼
 [4] full forward / generate
   │        └── new tokens attend over [arena | recency | live] — and absorb
   │            traces of everything mounted (haunting = continuity)
   ▼
 [5] harvest the turn pre-RoPE (free — it was computed anyway), tag it,
   │        deposit to repository with provisional edges
   ▼
 [6] unmount; async: librarian updates graph
```

Seating plan inside the trained window (example, 32,768-seat model):

```
 seat 0..~4                16,384        24,576              32,768
   |SINK|------- ARENA ---------|--RECENCY--|------ LIVE --------|
   |    | routed mounts, per turn| always-on |  prompt + output   |
   |    | (re-seated every turn) | last N    |                    |
```

- **SINK seats (0..~4) are permanent and never unmounted** — measured
  2026-06-10: removing a mount that includes seat 0 collapses generation into
  repetition (attention-sink destruction, the StreamingLLM effect). The arena
  starts after the sink zone.
- **Recency window = the shared expert.** Always mounted, never routed.
  Recency is load-bearing in conversation; the router gets no vote on it.
- Arena size, recency size, and refine fraction are all runtime dials.
- Routing is per-**turn**, not per-query (seating can't change mid-forward).

### Working-set overflow (the duck/fox/corn case)
When the router's honest answer exceeds the arena: **shuttle in batches.**
Mount batch 1 → attend → generate digest tokens against a fixed digest prompt
→ keep ONLY those tokens' K/V → unmount → mount batch 2 → repeat → final pass
attends over all digests together. Pays forward passes instead of seats.
Trips-per-turn is a dial. Digests are **seat compression** — the only way to
mint more of the unbuyable resource.

---

## 4. The cold path (librarian + consolidation)

Memory hierarchy (sleep consolidation, literally):

```
                     ERA grafts        (~10² turns' gist in ~10² seats)
                    /     |     \
             EPISODE   EPISODE   EPISODE     (clustered, digested)
             /  |  \    ...       ...
         turn turn turn                      (verbatim K/V, cold storage)
```

- Aging turns get clustered (graph edges: temporal adjacency, semantic
  similarity, **haunting lineage** — turn 12 was generated with turns 3 & 7
  mounted, so it depends on them), digested into episode grafts, then era
  grafts. Verbatim turns demote to cold storage; gist stays hot.
- Routing descends the hierarchy; most history arrives pre-compressed.
- **The dialect wall (hard constraint):** a smaller model may do all
  TEXT-level librarianship — labels, clustering, edge inference,
  consolidation *decisions* — but every K/V artifact must be digested by the
  serving model under its own weights. The librarian writes the card catalog;
  only the resident author writes the books.
- Riddle constraints encoded as graph rules: a digest must not be mounted
  without (or must link to) the sources it was haunted by; contradictory
  nodes (old plan / revised plan) carry "don't co-seat" edges; consolidation
  may require re-mounting old turns so a new digest is computed coherently.

---

## 5. Engineering prerequisites (existing codebase, known fixes)

1. **Injection becomes prefill-only.** ✅ DONE 2026-06-10: graft cat gated on
   `kv_cache is None`; the +Sg position shift persists on cached decode via a
   `graft_seats` attribute (deriving Sg from the injected tensor collapsed the
   shift to 0 on decode — found and fixed the same day). `clear_injection`
   gained `free_seats=False` for unmount-but-keep-the-positional-hole.
2. **Pre-RoPE capture during live generation.** ✅ DONE 2026-06-10 as
   deposit-FROM-CACHE (`ArenaCache.deposit_from_cache`): c_n sliced as-is,
   k_pe un-RoPE'd by rotation composition (apply_rotary with −sin).
   Measured split: the contextualized PAYLOAD re-mounts with full recall;
   the contextualized CENTROID is conversation-polluted (5/6, turn-1
   attractor) — routing keys come from a layers-0..route partial forward
   (`max_layers` early exit). E4-arena 6/6, mounts identical to standalone.
3. **Device-resident mounts.** ✅ DONE 2026-06-10: grafts stored as device
   tensors (deposit() uploads once; deposit_from_cache never leaves the
   card except the 256-float key); swap surgery cats device tensors.
   Paging/spill policy still TODO at corpus scale.
4. **Summary-key computation** at harvest time. ✅ DONE for MLA 2026-06-10:
   `kv_graft.latent_centroid` — unit-norm mean latent, 512B fp16, free.
5. **Arena seating + rotation-relocation** as first-class cache ops.
   ✅ DONE 2026-06-10: `core/graft_arena.py` ArenaCache (swap/evict as cache
   surgery, fixed `live_shift`, sink graft, recency eviction).
6. Stale-doc hygiene: harvest docstring says post-RoPE (it's pre-RoPE);
   multifact test still ships the superstition recipe. Fix before this travels.

---

## 6. The failure budget — measure BEFORE building big

Three numbers decide whether this architecture works. All run on existing
machinery; none requires the full system.

**E1 — Router recall (the keystone).** ✅ **PASSED 2026-06-10 — routed top-3
BEATS mount-all.** Two rounds on Qwen3-4B (10 chunks, planted alien needles,
64-token greedy probes, controls 0-1/10):
- Round 1 (pre-registered all-layer-mean router): Arm B 6/10 = Arm A 6/10 —
  pass at parity. Per-layer diagnostic: layer 0 alone routes 9/10 recall@1;
  every other layer ≤3/10.
- Round 2 (fresh needles, layer-0 router fixed in advance): **Arm B 10/10 vs
  Arm A 4/10.** Router recall@3 10/10, recall@1 8/10, margins 10–50× the
  all-layer scores (which collapsed to 3/10 recall@3 — deep layers are
  routing noise; round 1's all-layer pass was attractor-chunk luck).

Score = mean over q-heads of max over (probe-q, graft-k) pairs of |q·k|/√Dh,
pre-RoPE both sides (position-free). Three consequences: (1) routing is not a
residency optimization — mount-all LOSES to routed top-3 because co-mounted
grafts interfere (rumination spirals, digit corruption: "07:40" for 07:42,
"velvet-octopus-27" for -29); over-mounting is the other forgetting. (2) The
router costs ONE layer-0 q-projection — no partial forward through k layers;
the latency dial resolves to k=0. (3) Capture hook: `_capture_q` in
GQAAttentionTC + `kv_graft.capture_queries()`. Harness:
mission_b74b7906/test_graft_e1_router.py (+ _round1).

**E1 on MiniCPM3 (MLA latent grafts), same day — the router law forked.**
The key-space router FAILED outright: probe-INDEPENDENT scores, identical
top-3 for every probe, margin +0.000. Diagnosed: argmax key position is
constant per graft (mid-content outlier keys at positions 7–12) — MiniCPM3
has NO qk-norm, so |q·k| max measures key norm, not relevance (same root
cause as the bulk-bits-tracks-key-normalization law). Cosine and sink-drop
variants do NOT rescue it (≤5/10 recall@3). What does: **routing in the
model's own 256-d LATENT space** — cos(mean c_n_probe, mean c_n_graft).
Pre-registered confirmation on fresh needles: L10 router 8/10 @1, 9/10 @3
(L44 diag: 9/10, 10/10); arms B 7/10 = A 7/10 → pass. The graft's summary
key is the unit-norm mean of its own stored latent: 512 bytes fp16, zero
extra computation at harvest — E3's answer for MLA, measured early. Note
MiniCPM3 tolerates co-mounting far better than Qwen3 (mount-all 7-8/10 vs
4-6/10) and its residual failures are READING errors (species-for-name;
F-77/CF-33 tag confusion between co-mounted same-format needles), not
routing. Harness: test_graft_e1_mla.py (gates: test_graft_mla_gate.py).

**Router law (working):** route in a normalized representation space native
to the model. QK-normed keys qualify (Qwen3, layer 0); unnormalized MLA
keys do not — use the latent centroid (MiniCPM3, L10 cheap / L44 best).
The routing index is part of the dialect.

**E2 — Digest fidelity (the haunting coefficient).** ✅ **MEASURED
2026-06-10 (MiniCPM3, 10 chunks, chained two generations):**
D0 direct-mount 9/10 → D1 through-digest 8/10 → D2 through
digest-of-digest **8/10 — ZERO second-generation decay.** Decomposition:
generation fidelity 10/10 at BOTH levels (the verbatim-preservation prompt
works: "archive note... preserving every name, code, number, and time
verbatim"); the loss is retrieval-side, and it is structural, not token-
level. **Digests are fixed points under re-digestion** (8/10 second-gen
digests ≈ character-identical to their parents): consolidation costs once
(coefficient ~0.89) then plateaus — era grafts (digests-of-digests) are
viable as designed, inheriting D1's loss and adding none.

Failure anatomy → librarian QC rules: both D1 misses were DEGENERATE
LIST-DIGESTS — greedy decoding collapsed 3/10 digests into comma-list
repetition loops ("Vesper, F-77, 412 g, lure, F-77, …"), one of which
hallucinated a timestamp absent from the source. Narrative-form digests
retrieved 7/7; list-form 1/3. Lists preserve tokens but drop RELATIONS
(what the code was a code *for*), and probes traverse relations. QC:
reject repetition-looped digests (n-gram detectable), require sentence
form, regenerate on failure. Harness: mission_b74b7906/test_graft_e2_digest.py.

**E3 — Summary-key quality.** *(answered for MLA by E1's failure analysis,
2026-06-10)* For MiniCPM3 the routing index is the unit-norm MEAN LATENT
(per layer, 512 bytes fp16) — it routes 9-10/10 recall@3 where every
key-space candidate fails. For QK-normed GQA models the full layer-0 keys
route directly; centroid compression of those keys remains unmeasured
(matters only when the index outgrows residency).

**E4 — End-to-end conversation needle test.** ✅ **PASSED 2026-06-10
(MiniCPM3, MLA latent grafts).** 20 scripted turns, 6 format-distinct facts
in turns 1–8, filler 9–14, probes 15–20. Baseline (full transcript): 6/6 at
670–858 ctx tokens, growing. System (L44 latent router, top-3 turn-grafts +
2-turn live window): pre-registered run 5/6 at 171–255 resident tokens,
bounded — parity within one, ~30% of baseline residency. Amnesia control
(window only, no mounts): 0/6 — the grafts carry everything.

Both pre-registered runs missed exactly one probe, and both misses were the
same failure shape: **routing style attractors.** (a) Routing on the
wrapped probe ("User: …\nAssistant:") pulls the centroid toward other
Q&A-shaped turns — the budget turn ranked 4 wrapped, 1 bare (measured).
(b) Deposited retrieval-only probe turns ("User: <q> Assistant: <short
answer>") are low-content, high-style nodes that crowd the top-3.
Exploratory v3 with both hygiene rules — route on the BARE user message;
do NOT deposit retrieval-only turns (the boat doesn't deposit its own
wake; the answer's content lives in the source turn) — scored **6/6, full
parity at ~25% residency.** Phase-1 defaults. Longer-term the librarian
subsumes rule (b): retrieval turns are consolidation fodder, not nodes.
Harness: mission_b74b7906/test_graft_e4_conversation.py.

Gates inherit house rules: protocols fixed in advance, fresh processes,
diverse-token validation, doors ledger from day one.

---

## 7. Risks & open doors

- **Retrieval miss = forgetting.** Mitigated by the recency shared expert +
  hierarchical descent; quantified by E1/E4. The router's recall IS the
  system's memory quality. CORPUS-100 ANSWER: the routing index is a
  THREE-channel hybrid — latent centroid (topical) + rare-token lexical
  keys (identifiers; exact match dominates — centroids cannot separate
  near-duplicate siblings) + child centroids (descent). All three are
  bytes-cheap and harvest-free.
- **Digest dangling.** A digest mounted without its haunting sources may
  reference ungrounded context. Graph dependency edges + E2 decide policy.
- **Dialect wall.** No cross-model artifacts, ever. Repository is per-model;
  a model upgrade invalidates the K/V store (text/tags survive; re-digestion
  is a batch job).
- **Seat ceiling.** Arena + recency + live must fit the trained window.
  Consolidation density (seats per turn-of-history) is the long-run currency.
- **Routing latency.** ~~One partial forward per turn (layers 0..k). Measure k
  vs. routing quality; k is a dial.~~ RESOLVED by E1: k=0. Layer-0 queries
  alone route best (deep layers are noise); cost is embedding + one q_proj.
- ~~**Consolidation quality unmeasured.**~~ MEASURED: E2 chained shows a
  STEP decay curve — one-time ~0.89 coefficient at first digestion, zero
  loss at the second (digests are fixed points under re-digestion). The
  compounding-loss risk did not materialize; the real risk is degenerate
  list-digests (librarian QC catches them).
- **Literature positioning (what's prior art, what's ours).** Retrieval-into-
  attention exists: Memorizing Transformers (kNN attention over cached K/V,
  trained-in), Unlimiformer (inference-time retrieval into attention),
  kNN-LM (output-layer retrieval), PromptCache (modular KV reuse with
  positional compromises). **Ours:** certified-lossless mounting baseline
  (equivalence harness), router and reader unified in one kernel (APA bulk
  pass), precision-tiered storage hierarchy, free conversational harvest,
  haunting-lineage graph, seat-compressing consolidation — all runtime, all
  on consumer hardware, zero training.

---

## 8. Build phases

- **Phase 0 — prerequisites.** §5 items 1–3 + hygiene. Exit: cached-decode
  generation with a mounted graft, no duplication, equivalence harness passes.
- **Phase 1 — routed single-session memory.** Flat repository (no graph, no
  digests): harvest turns, route per the model's router law (Qwen3: layer-0
  full-key; MiniCPM3: L10/L44 latent centroid), mount top-k + recency window.
  Exit: E1 ✅ both models (2026-06-10; Qwen3 routed 10/10 vs mount-all 4/10,
  MiniCPM3 routed 7/10 = mount-all 7/10) and E4 ✅ (2026-06-10; 6/6 = baseline
  at ~25% residency with routing hygiene). **PERSISTENT ARENA BUILT same day**
  (core/graft_arena.py, ArenaCache: route/swap/feed/step/evict — all cache
  surgery, zero re-prefill): E4-arena 6/6 on one never-rebuilt cache,
  residency flat 268-316 seats. Decode amortization DONE same day: absorbed
  MLA decode + int4 GEMV kernel + no_grad/pool/fused-norm stack = 675→21.6
  ms/tok (31×, all parity-gated; see MiniCPM3-MLA_Results.md §decode) —
  arena turns now 1.5s end-to-end (route+swap+prefill+48-tok answer), 6/6
  recall unchanged. Remaining Phase-1 engineering: harvest-on-generate
  (deposits currently cost one extra forward), device-resident mounts
  (host→device upload per swap).
- **Phase 2 — digests + shuttling.** Digest-token generation, multi-pass
  reads, E2 measured. Exit: overflow handled by trips, not bigger arena.
  STATUS 2026-06-10: E2 measured (step decay, ~0.89 once then flat);
  `ArenaCache.consolidate()` built (QC'd digest, lineage, child-centroid
  descent, source retirement) and E4-C passed 6/6. TWO LIBRARIAN LAWS
  measured the hard way: (a) the ACKNOWLEDGMENT TRAP — over mounted
  dialogue turns the model answers the digest request like a chat turn
  ("I'll create an archive note...", zero facts; round 1 went 0/6 while
  routing was perfect) — primed-prefix prompts ("Assistant: The facts to
  archive are:") force content mode; (b) fluency QC is not enough —
  content QC must mechanically verify the digest keeps the sources'
  code/number-shaped tokens (the librarian holds the sources; ≥50%
  retention enforced, three-prompt retry ladder, best-keeper fallback).
  **EXIT MET 2026-06-10:** shuttling built (`step(max_trips=N)`: full-rank
  routing, per-trip arena reseat, grounding check — hedges + content-free
  deflections + content⊆mounts — and total rollback of failed attempts via
  cache-tensor immutability). Starved 1-mount arena + 2 trips = 6/6 = the
  3-mount arena; recovered probe cost one extra attempt (2.8s vs 1.5s).
  Overflow handled by trips, not bigger arena — measured.
- **Phase 3 — hierarchy + librarian.** Summary keys (E3 winner), episode/era
  consolidation, AtlasForge mission for the cold path, graph edges live.
  STATUS 2026-06-10: IN-PROCESS librarian built (`GraftRepository`):
  threshold-triggered auto-consolidation during chat (8 active turns →
  fold 4; 6 active digests → era graft), descent keys + lexical keys
  flatten through generations (an era inherits its leaves' identifiers),
  retired nodes auto-freed from VRAM (disk = cold storage), documents
  exempt from folding (reference, not history). Era path coded, not yet
  exercised at depth (needs a longer session than the gate's 14 turns).
  Remaining: librarian as a BACKGROUND AtlasForge mission (the cold path),
  graph edges beyond lineage (contradiction / don't-co-seat).
- **Phase 4 — persistence + product shape.** STATUS 2026-06-10: CORE DONE —
  repository directory (manifest.json + index.npz + nodes/NNNN.npz fp16),
  dialect wall enforced at load, route-layer guard, descent keys rebuilt
  from lineage, active-only device upload, autosave-per-turn. Fresh-process
  resume gate: 6/7 recall from disk alone (incl. a document fact; trips
  fired and recovered during resume). 18 nodes = 24.9MB. Live cache
  deliberately not persisted — history lives in the repository.
  Remaining: per-model repository management UX; the consumer story:
  conversations that
  never crash, never truncate, never forget — on an 8GB card.

---

*Companion docs: `KV-Graft_Document-Injection.md` (mechanism + equivalence),
`APA-Quant_CrossModel_Results.md` (dial + key-distribution law),
`MiniCPM3-MLA_Results.md` (ceiling at trained window; latent caching note —
on MLA the repository's artifacts shrink to latent size, and this design's
economics improve by the same ratio).*
