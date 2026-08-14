# Frozen Compute, Liquid Memory: GRM on Etched Silicon, and the Real-Time Librarian

Status: **design note / vision** — not a plan, no gates, nothing dispatched.
Captured 2026-08-13 (evening) from a working conversation between David and
the lead session, same-day as the APAMQ investigation (Project-Tensor
`docs/APA_MQA_ROOTCAUSE_LEDGER.md`). Claims are tagged by evidence class:
**[receipt]** = gated result in this repo or Project-Tensor; **[measured]** =
same-day lead-run measurement; **[external]** = public announcement, not
locally verified; **[design]** = extrapolation, no evidence yet.

Related: `GRM_LOCAL_APPLICATION_IMPLEMENTATION_PLAN.md` (repo-local Inbox;
awaiting go/no-go — this note does not modify it), `ROUTER_SCALING_REPORT.md`,
`GRM_MLA_CUDA_ROUTE_LEDGER.md`, `GRM_GQA_EXACT_RAGGED_CUDA_LEDGER.md`,
`GRM_S4_LEDGER.md`, `GRM_GQA_ROUTE_CARD_DB_LEDGER.md`.

## 1. The trigger

AMD acquired Taalas (announced 2026-08-06) **[external]**. Taalas' HC1 etches
a specific model's weights permanently into silicon — Llama 3.1 8B in 53B
transistors, TSMC 6nm, ~17,000 tokens/s per user, claimed ~10× lower power
and ~20× lower build cost than GPU serving; AMD plans to deploy the ASICs
alongside Instinct GPUs under ROCm **[external]**. The universally cited
objections: one chip runs exactly one model, forever, and that model's
knowledge is frozen at mask time.

Both objections are the problems GRM was built to dissolve. This note is the
composition argument.

## 2. The partition principle

A transformer has two physically different halves:

- **Weight-bound half** — QKV/O projections, MLPs, embeddings. Static, same
  for every user and token. This is what Taalas etches; it is the only part
  that CAN be etched.
- **State-bound half** — the KV cache and attention over it. QK^T and P·V
  are weightless data-data operations over per-conversation state that does
  not exist until the user speaks. **By definition un-etchable**: the cache
  must live in ordinary writable, addressable memory somewhere in the
  system, whatever the vendor's packaging.

Everything in this repository — grafts, the arena, the routers, supersession,
importance — operates exclusively on the state-bound half. Nothing here
touches weights. Taalas monetizes the half that can be frozen; GRM is
infrastructure for the half that cannot. The two partition the transformer
along its actual physical joint, which is why the composition is structural
rather than opportunistic. **[design]**, resting on the architectural fact
that attention state is dynamic data.

Consequence: mounting a graft into an etched-model system is a write into
cache memory that must exist and be writable for the chip to function at
all. The integration question is the cache's address layout and driver
surface — an ABI detail — not an architectural concession. The one
load-bearing unknown is whether that surface is externally reachable on
shipping parts (§8).

## 3. Why models get upgraded — and what that means for frozen ones

Companies upgrade deployed models overwhelmingly to chase **knowledge**
(currency, domain coverage, "hallucinates less on our docs") and vendor
deprecation — not raw capability. In bounded domains (customer service is
the canonical case) the capability bar is cleared early and does not move;
what rots is knowledge. Weight-world and RAG-world couple knowledge to the
model, so the only lever is swapping the brain.

GRM severs that coupling: knowledge lives in the graft store; the model is
the interpreter. New product line → encode it; the deployment learned it
this afternoon with no training and no behavioral regression surface. The
"model-locked graft" limitation inverts into the design's strongest
property: **the reason to upgrade never fires**, because the only thing that
was ever going stale lives outside the weights. The frozen model becomes an
appliance. Nobody upgrades a CPU to learn new facts. **[design]**

### Even when you MUST upgrade (capability-bar moves)

- **Re-mint from provenance.** Every graft descends from a source (doc,
  transcript, session). Minting is deterministic prefill, so migration is
  an embarrassingly parallel batch job while the old stack keeps serving.
  Design rule this implies: **provenance is never discarded** — the source
  behind every graft is part of the asset (Graft Format v1 versioning
  already points this way).
- **Rollback is a pointer swap.** Old weights + old store remain byte-exact.
- **Migration ships with receipts.** Same provenance minted through both
  models → diff grounding on the certification battery before cutover.
  Upgrades become a red/green report, not a behavioral leap of faith.
- **Translation is the upside path.** G3 POSITIVE: real cross-model binding
  transport (2B→9B) at ~5% native readability **[receipt]** — early, but if
  the line matures, even the re-mint disappears: translate the store.

## 4. MLA is the architecture you want etched

On GPUs, MLA trades a tiny latent cache (~85× smaller than full-width MQA
cache per the Gemma-4 port measurements **[receipt]**) for extra per-token
compute: the latent re-expands through projection weights every step. On an
etched chip **those expansion projections are in the metal** — MLA's cost
term lands on the near-free etched half, while its benefit (minimal state)
lands on the scarce un-etchable half. Etched silicon makes MLA strictly
dominant: maximize what can be frozen, minimize what cannot. **[design]**

Scale of the state half under MLA: full-width MQA globals run ~16 KB/token;
MLA latents run a couple hundred bytes/token. A million tokens of stored
attention state: ~16 GB full-width vs a few hundred MB as latents — **whole
customer-history libraries resident in host RAM; mounting becomes a
memcpy.** [design arithmetic from receipted cache-shape numbers]

And the MLA lane is the *most mature* lane in this repository:

- MLA CUDA route: production 1M-node route through `ArenaCache.route()` =
  **2.22 ms p50** (curve 0.184/0.210/0.398/2.22 ms at 1k/10k/100k/1M;
  117/117 suites) **[receipt]** — memory search costs ~1/10th of one decoded
  token on a 12B; on a 17K tok/s part it is unmeasurable at any library size.
- Historical APA context extension on MLA (MiniCPM3 3K→32K) **[receipt]**.
- GQA is also covered where needed: exact ragged GQA CUDA router, 512 nodes
  at p50 1.591 ms, 175/175 four-backend parity, merged default-off
  **[receipt]** — relevant since HC1's Llama 3.1 8B is GQA.

## 5. The composed appliance (CS chatbot as the canonical workload)

Etched MLA chip + GRM store + router, against RAG on GPUs:

- **The re-read tax dies.** RAG re-prefills the same KB text every
  conversation (a 10K-conversation/day operation re-encodes ~10^8 tokens
  daily of already-understood text). GRM encodes once, mounts forever —
  Gemma receipt: 690-token mount, prefill 15.5 s → 0.2 s **[receipt]**. On
  the etched part, prefill was the only cost silicon didn't already crush;
  GRM removes it. The two technologies erase each other's residual
  bottleneck.
- **Consistency is structural.** A graft is the model's interpretation,
  frozen; every conversation mounts the identical understanding of the
  refund policy. RAG re-derives understanding per conversation and varies.
- **Customer memory that doesn't summarize.** Four independent lines —
  SCRIBE, sub-floor storage quant, route cards, BABEL (external) — found
  the same law: compressed summaries of contextualized K/V don't degrade,
  they **vanish** **[receipt ×3 local, 1 external]**. RAG-style memory is a
  paraphrase layer; GRM mounts the March session's actual state in June.
- **Supersession with receipts.** SUP L2: superseded state cannot leak back
  into readback **[receipt]** — the compliance story for policy changes.
- **Free curation.** S4 grounding-hits importance: median Spearman 0.7556,
  top-1 0.875 (16/16), zero extra forwards **[receipt]** — per-node
  analytics on which knowledge actually resolves tickets, as a byproduct.
- **Bilateral model-lock cancels.** Grafts are model-locked; the chip IS one
  model. Silicon determinism collapses the plan's per-MACHINE certification
  toward per-SKU certification — certify the graft format against the SKU
  once, valid for every chip off that mask. **[design]**
- **Rack roles fall out naturally**: Instinct GPUs = scribes (minting,
  re-minting, router), etched ASICs = readers (serving with mounts), the
  graft store = the library. **[design]**

## 6. The real-time librarian

The utilization asymmetry is the whole argument: a human conversation needs
~15 tok/s to feel instant; the etched part does ~17,000. **The visible
conversation uses <0.1% of the machine.** A three-second typing pause is
~50K tokens of invisible capacity. That capacity hosts second and third
passes — in real time, inside the conversation:

- **Minting is nearly free.** Pure-KV mint discovery: the graft is a SLICE
  of the caches the visible pass already computed **[receipt]** — pass 2
  starts from state, not from text. Remembering is a byproduct of answering.
- **Curation rides the free signal.** S4 counters update as the visible
  pass grounds its answers **[receipt]**; promote/demote/supersede decisions
  consume data that serving generates anyway.
- **Supersession is the write path.** "Actually I moved last month" → the
  correction is minted and the stale fact SUP-killed before the next turn.
  Sleep-consolidation running between keystrokes. **[design]**
- **Pre-verification hides inside perception.** ~300 ms of human threshold
  is ~5K tokens on the etched part: draft, route against mounted policy,
  check grounding, revise, then speak. The customer sees pass N. **[design]**

### The honest RED that names the regime

The librarian already ran once and failed: the route-card line
(**LIBRARIAN-YIELD-RED [receipt]** — 42 accepted aliases on 12/32 nodes, 14
nodes starved on a frozen 384-token rail; cards lost sources at every
operating point). Read correctly, the receipt kills the **starved,
compressing** librarian — and compression was independently dead under the
irreducibility law. The librarian this note describes is a
**re-contextualizer**: it re-encounters sources in richer contexts, mints
additional route keys, builds supersession chains, and indexes — all
state-preserving operations, none of them summaries. The 384-token rail was
the binding constraint; 17K tok/s is the regime where the rail does not
exist. The RED receipt is thus a *boundary* result: it locates the failure
in the throughput-starved corner, exactly the corner etched silicon
eliminates. **[design, anchored on the receipt]**

## 7. One-sentence stack

Weights in metal (Taalas), state in RAM (MLA latents), library on disk
(GRM), search in 2 ms (router), curation for free (S4), supersession as the
write path (SUP), and a librarian running thousands of invisible passes per
conversation in the 99.9% of the machine the human cannot perceive — on a
chip that never changes and never needs to.

## 8. Load-bearing unknowns

1. **Cache reachability on etched parts** — is the KV/latent cache's address
   layout and write path exposed (ROCm-level ABI)? The state half must be
   conventional memory (§2), but "exists" ≠ "documented and reachable."
   This is the seam question, and the right place to meet the technology
   per the house MCP-boundary doctrine.
2. **Mount semantics on a fixed pipeline** — etched parts may hard-schedule
   prefill; mounting needs a prefill-bypass (cache-write + position/RoPE
   bookkeeping). All GRM mount machinery here assumes that seam.
3. **Translation maturity** — G3 is positive but early (~5% native
   readability); the guaranteed migration floor remains re-mint-from-
   provenance.
4. **Librarian yield at throughput** — the RED receipt bounds the starved
   corner; the rich-regime librarian is undemonstrated. It would need its
   own plan with pre-registered gates before any claim is made.
5. **Per-SKU certification** — assumes silicon-level determinism across
   chips of one mask; plausible, unverified.
