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

## 2026-07-08 (seam-1 DIAGNOSED — hypothesis refuted with receipts; fork registered)

Action: seam-1 diagnosis dispatched to Codex (first native codex-shim
transport; 20 min, 235k tokens). NO code change landed — attempted
driver patch honestly reverted, worktree verified clean. The
deliverable is the diagnosis, and it refutes the registered hypothesis.

Findings (evidence class: live probe reproduction + byte-level prompt
receipts; artifacts/grm_e2e/seam1_diag_current_20260708_150642/):
- HYPOTHESIS REFUTED: seam-1 is NOT a Harmony template delta. The
  reproduced failing probe sent the byte-identical forced-final prompt
  (244 bytes, SHA-256 9025cc17..., step_prompt_equals_leg1_forced_final
  _same_user: true). The step() loop's prompt construction is correct.
- NAMED DELTA — VALUE SHAPE: Leg-1's fact value was code-shaped
  ("Zeta-7-Quebec": rare tokens, distinctive); the smoke's facts are
  common lowercase words ("lantern", "copper"). Two independent
  failures follow: (a) ROUTING — lowercase common words do not survive
  _rare_tokens, so the route missed the source fact (r5 receipt:
  source_node_id 4, mounted [2]); (b) READOUT — with mounts FORCED
  correct via a route-key patch, free greedy generation still drifted
  (cypher mounted [7]/source 6 → "<|constrain|>..."; orion mounted
  [5]/source 4 → "0."). Probes 0/2 even with green mounts.
- DEEPER SEAM (named, not fixed): free greedy GPT-OSS generation does
  not reliably extract low-entropy lowercase values from mounted
  grafts. Fix class = probe readout policy (candidate/value scoring or
  constrained selection), not prompt-template tweaking.

DECISION (David, fork registered): proceed with FORK A — the receipt
runs with distinctive code-shaped fact values (Leg-1's proven regime;
standard needle practice). The receipt will carry the honest caveat:
recall demonstrated with distinctive values; realistic low-entropy
value recall requires the readout policy. FORK B (readout policy)
REGISTERED as a successor work order, alongside seams 2-4.

## 2026-07-08 (FORK A REFUTED — degenerate generation; mechanical loop hypothesis raised)

Action: fork-A executed by Grok (grok-shim, 197s): driver fact values
swapped to code-shaped (e.g. "Vortex-3-Sierra", "Kestrel-9-Tango"),
10-turn smoke re-run. Agent STOPPED at the registered rail (no knob
iteration) — correct.

Findings (artifacts/grm_e2e/smoke_session_20260708_154313/):
- PROBES 0/2 WITH code-shaped values. Route missed identically
  (mounted [2], source_node_id 4 — same pattern as the lantern run):
  value shape does not fix routing, because the route query is built
  from the probe QUESTION, which never contains the answer value.
  Leg-1 routed correctly for a different reason: a ~3-turn session has
  nearly nothing to discriminate.
- WORSE — generation now DEGENERATE, not merely off-target: "It seems
  to with you." / "We need to the user=with with with but I cannot."
  Combined with Codex's receipts (special token <|constrain|> leaking
  as text; "I apologize for the user..."), three flavors of broken
  text through step()'s loop vs clean English from Leg-1's manual loop
  on identical machinery and byte-identical prompts.
- FORK A REFUTED as a sufficient fix. New working hypothesis raised:
  MECHANICAL defect in the step() generate path (position/cache
  misalignment around mount/swap, stop/slice handling, or decode
  config delta) — a readout policy would paper over it, so diagnosis
  precedes any fork.

Next action: instrumented A/B loop-divergence diagnosis (manual loop vs
step() on identical mounted state; per-step position/cache-length/
token/logit dumps; first divergence = the receipt) — dispatched to Grok.

## 2026-07-08 (ROOT CAUSE — arena position hole collapses GPT-OSS generation)

Action: instrumented A/B loop diagnosis completed by Grok (624s).
Verdict verified against artifacts. BOTH prior hypotheses now refuted
with receipts; real cause confirmed with a sweep.

Findings (evidence class: shared-KV-snapshot A/B + width sweep;
artifacts/grm_e2e/step_generate_mech_diag_20260708_155956/ +
step_vs_manual_ab_shared_20260708_155615/ +
live_shift_width_ab_20260708_155826/):
- LOOP BODY EXONERATED: on a true shared KV snapshot, manual loop vs
  step()'s _attempt are TOKEN-IDENTICAL every step (same positions,
  cache lengths, chosen ids, top-5 logits) — BOTH answer
  "Vortex-3-Sierra" cleanly. (An earlier independent-rebuild A/B
  falsely split at step 1 on a logit knife-edge tie 15.5625==15.5625 —
  rebuild FP noise, caught and corrected by the agent. Measurement-law
  echo: rebuilds are not snapshots.)
- ROOT CAUSE CONFIRMED — candidate (a), at the ARENA GEOMETRY level:
  live_shift = n_sink + arena_width is the RoPE position hole the
  arena reserves ahead of live context. GPT-OSS (YARN) collapses under
  a large hole. Width sweep, same prompt bytes: width 96 (live_shift
  99-115) → clean English; width 384 (live_shift 387/403) → word salad
  ("User: 3-V> < 3-4>…"). Leg-1 ran width=96; the session driver
  defaults --arena-width 384.
- FAILURE CASCADE EXPLAINED: session plants facts via step() free
  generation under live_shift≈387 → the deposited grafts are POISONED
  (garbage text/K/V) → routing degrades and probes read back garbage
  even when mounts are forced. Value shape (fork A) and template
  (seam-1 hypothesis) were both downstream symptoms.
- Candidates (c) stops/slicing and (d) sampling: refuted as primary
  (shared-state greedy identical). (b) resume: secondary amplifier
  only (refeed of already-poisoned plants).

MINIMAL FIX (stated by diagnosis, to be implemented + smoked next):
driver --arena-width default 384→96; Harmony sink_text instead of
default "<conversation>\n"; prefer feed() complete turns for fact
plants. PRODUCT FOLLOW-UP REGISTERED: GPT-OSS YARN tolerance of the
arena hole (core/graft_arena.py:77, core/gpt_oss20b_tc.py:686-691) —
a width guard or dialect-aware live_shift cap; candidate LAW for the
board once the fix-smoke confirms: arena position holes are
dialect-bounded (GPT-OSS/YARN ≤~115 proven clean, 387 collapses).

Next action: implement driver fix + re-apply code-shaped values +
one bounded smoke (target 2/2); green → lead fires P2 full run.

## 2026-07-08 (width fix CONFIRMED for its class; next layer exposed — refusal plants)

Action: width 96 + Harmony sink + code-shaped values implemented and
smoked by Grok (driver-only). Probes 0/2 — but the geometry fix
worked: NO word-salad anywhere (grammatical English), confirming the
RoPE-hole root cause by treatment effect.

Findings (smoke artifacts per report; driver diff = deliverable, held
uncommitted):
- NEW LAYER: fact/supersede plant turns run through step() free
  generation, and GPT-OSS REFUSES them ("I'm sorry, but I can't help
  with that") → deposited grafts carry refusal K/V, not the fact.
  Tell: the single node planted via the memory-command path (no free
  gen) is pristine ("The current orion pin value is Kestrel-9-Tango.").
- Probes now answer coherent-but-wrong ("...is 0.") — readout of a
  wrong/poisoned mount, no longer geometry collapse. Route still
  mounts [2] vs source 4, but routing cannot be judged over
  refusal-poisoned deposits.
- Residuals: T8 post-restart garble (minor, watch); YARN live_shift
  product cap still registered.
- This was the diagnosis' fix #3 (feed() complete-turn plants),
  deliberately scoped out of the minimal order — now proven necessary.

Next action: convert plants to feed() complete turns (probes/filler
keep the production chat()→step() path), add probe route-ranking
diagnostics, one bounded smoke. 2/2 → P2; miss with clean deposits →
route/readout seam with honest data.

## 2026-07-08 (feed() plants — FIRST PROBE PASS; route seam isolated)

Action: plants converted to feed() complete turns (Grok, driver-only);
route-ranking diagnostics added; bounded smoke run.

Findings: PROBES 1/2 — first pass in the E2E effort. Cypher probe:
source ranked 1 (1.000), mounted, answered "Vortex-3-Sierra" exactly —
deposit→evict→route→mount→recall PROVEN end-to-end on the production
path. Orion supersession probe FAILED on ROUTING: cypher plant node
scored 1.000 on an orion question and was mounted; pristine correct
memory ranked 4th (0.866); readout faithfully answered from the wrong
mount (readout exonerated). Plants verified value-bearing (no refusal
deposits). Driver diff held uncommitted (+233/−47).

## 2026-07-08 (route-score decomposition — PRODUCT bias named; seam-2 confirmed)

Action: T6 ranking decomposed offline from persisted state (Grok, high
effort; stopped at the registered product fork — no patch, no smoke).

Findings (numeric, term-by-term):
- CAUSE = PRODUCT SCORING, two compounding effects: (1) MAX-POOL
  LENGTH BIAS — route ranks by max |q·k| over key pairs; long Harmony
  turn keys systematically dominate short fact keys. Smoking gun: a
  filler turn with ZERO word overlap with the probe (lex=0) still
  outranks the 15-token pristine fact memory. Δ(n4−n3) raw +0.829 =
  100% latent; lex channel contributed 0 to every candidate.
  (2) EMPTY LEX CHANNEL — lowercase labels ("orion pin") never enter
  _rare_tokens, so disambiguation rests entirely on the length-biased
  latent term. Score 1.000 = cosmetic max-normalization, not affinity.
- REFUTED: template aliasing (lex=0 everywhere), recency/mount
  affinity (terms absent), CUDA-vs-native divergence (CUDA never ran).
  Driver plant phrasing = secondary plant-plant near-alias only.
- SEAM-2 CONFIRMED with receipt: GRM_GQA_CUDA_ROUTE=1 set, but
  _cuda_route_bank_inputs() → None on ragged live banks
  (8,{51,92,15,86},64); backend fell through to native on both probes.
  As the Leg-2 ledger entry predicted.
- Product fix options stated (none implemented): (1) lex-extend fact
  labels into the rare/lex channel; (2) kind-prior for fact nodes
  within a latent near-tie window; (3) length-debias max|q·k| /
  route-key hygiene (strip Harmony scaffolding from turn keys).

DECISION PENDING (David): (A) driver mounts top-k 2-3 (production-
realistic; correct memory sits rank 2-4 → likely 2/2 receipt tonight;
bias stays registered with the three fix options as successor) vs
(B) product scoring fix first (real fix; suites + re-gates; receipt
waits). Lead recommendation: A for the receipt, B as the registered
successor work order.

## 2026-07-08 (fork A exhausted by receipts; David directs product fix)

Action: top-3 mounts implemented and gate-smoked (Grok, driver-only).
STILL 1/2 — fork A exhausted by two independent receipts: (1) correct
memory ranks 4th, outside the top-3 plan; (2) WIDTH NULLIFIES
MULTI-MOUNT — one Harmony-scaffolded turn graft (86-92 toks for
one-sentence turns) fills the GPT-OSS-safe arena_width=96 alone; the
mount plan [4,2,1] seats only [4]. The two laws of the night are in
tension: width 384 breaks generation, width 96 holds ~one turn graft.
The 15-token pristine fact memory would fit anywhere — it is the
scaffolding-bloated turns that hog both ranking and budget.

DECISION (David): fix the product. Registered fix = option 1
(query-side lex extension), options 2/3 remain successors.

## 2026-07-08 (PRODUCT FIX LANDED — route query-side lex extension; smoke 2/2)

Action: implemented by Grok (first product-code order to that seat),
all registered gates green, held uncommitted for lead review + commit.

- core/graft_arena.py: content-word tokens from the route query now
  participate in the lexical channel (stopwords dropped); HIT-GATED
  rescore — queries with no content-word hit keep the untouched
  native/CUDA path (backend-assertion suites unaffected). Query-side
  only: node keys/indexes/epoch semantics unchanged. Dialect-generic.
  Default ON; opt-out GRM_ROUTE_QUERY_LEX=0.
- Tests: +3 unit tests (extraction; short value node outranks long
  zero-overlap node under length-biased latent; disabled path restores
  rare-only). Suites: router baseline 21/21, native runtime 121/121,
  gpt-oss scaffold 18/18.
- Smoke 2/2 (artifacts/grm_e2e/smoke_session_query_lex_20260708_174252):
  cypher rank 1 (unchanged); orion source_rank 4→2, mounted node also
  value-bearing, answered "Kestrel-9-Tango". FIRST GREEN SMOKE.
- Honest residuals registered: latent length-bias still elevates long
  turns among label-sharing candidates (options 2/3 = successors);
  driver diagnostics recompute scores rare-only (rank order from
  arena.route() is authoritative); CUDA bank skipped when content
  hits force the Python rescore path (seam-2 family).

Next action: P2 FULL RUN FIRED (lead, 30+ turns, restart/durability
leg, artifacts/grm_e2e/full_session_20260708_P2). E1-E4 read follows.

## 2026-07-08 (P2 FULL RUN COMPLETE — E-gate read; receipt closes with two named residuals)

Action: full composed session executed by lead (34 turns, 8 facts, 1
supersession, restart-after-turn-16 durability leg, 37 end nodes;
artifacts/grm_e2e/full_session_20260708_P2). PROBES 7/9.

E-GATE READ (thresholds as registered in the plan):
- E1 (recall outside live window, from remounted grafts): 7/7 FRESH
  facts PASS at session scale (lyra/nova/mira/terra/ember/atlas/
  cypher — all answered exactly from remounted grafts). The 2 failures
  are BOTH the superseded fact (orion), failed on ROUTING under
  competition (t5: mounted [1] vs source 0; t13: mounted [6,8] vs
  source 7), answering the cypher value both times. E1 = GREEN for
  fresh facts, RED for the superseded fact's instances.
- E2 (supersession returns current, never stale): RED by letter —
  current value not returned under competition. Sharpest sub-mode
  ABSENT: the stale value NEVER resurfaced (contains_stale=false;
  no un-supersession — the graft-quant law's failure mode did not
  occur here). Failure is wrong-fact routing, not stale readback.
- E3 (route wall ≤5 ms production path): RED as long predicted —
  observed ~756 ms at 37 nodes on the python path. Composite of
  seam-3 (per-turn arena re-prep, named at Leg-2) + the new lex
  rescore path + CUDA bank non-engagement on ragged banks (seam-2).
  Registered successors own all three.
- E4 (stable VRAM): GREEN — 10848→10962 MiB across 34 turns incl.
  restart (no monotonic growth); flush 1.48 s; refeed re-seeded live
  turns (ntok 51+53) per design.

RECEIPT VERDICT: the composed system is PROVEN at session scale for
witnessed deposit → evict → route → mount → recall of fresh facts
(7/7 exact answers through the production chat()→step() path, across
a process restart). TWO NAMED RESIDUALS carry to successors:
(1) supersession-under-competition routing (superseded fact loses the
route to a value-bearing competitor; echoes the quant sweep's
"corrections are the first casualties" — different mechanism, same
moral); (2) route latency at session scale (seams 2/3 + lex rescore).
Fresh-fact recall did not regress at 37 nodes; no stale value ever
returned; VRAM envelope held.

Successor queue (registered): route length-debias / kind-prior
(options 2/3); seam-2 ragged-bank CUDA route (synthetic centroids
design input); seam-3 re-prep profile; packed resume cadence (seam-4);
supersession-under-competition as its own probe battery.

Next action: lead review + commit of the uncommitted work (product
lex fix + tests, driver, this ledger); synthesis update.

NOTE (process): the diagnosis order's clean-tree rule caused the agent
to restore this ledger and scripts/grm_e2e_session.py to HEAD, wiping
uncommitted WIP (this ledger's two prior entries — re-recorded above —
and the fork-A value edits, recoverable from the smoke artifacts'
run_config). Lead brief error; rule now scoped to product code only.
