# Grafted Memory
## GRM: A Routed, Tokenless K/V Memory Runtime for Frozen Language Models
### Durable, Revisable Attention-State Memory Across Attention Architectures

**David Perry** — Independent Researcher (no institutional affiliation)
`dave@ai-storyforge.com`

*Preprint v0.9-draft — 2026-07-02. Quantitative claims trace to gated,
registered evaluations in the project's research logs (§7). All reference
identifiers verified against the arXiv record on 2026-07-02. Text licensed CC BY 4.0; the implementations are separately licensed
AGPL-3.0 with commercial licensing available.*

---

## Author's Note and Disclosure

This paper is the work of an independent researcher without academic
affiliation, produced with substantial assistance from AI systems (Anthropic's
Claude models and OpenAI's Codex as implementation, review, and drafting
collaborators under the author's direction; an autonomous research platform,
AtlasForge, for experiment execution — the AtlasForge platform is itself
open source (MIT): https://github.com/DragonShadows1978/AI-AtlasForge, PyPI
`ai-atlasforge`; the specific mission workspaces cited in this program are
project-local, not part of that repository. The platform's stage-gated
mission loop builds falsification in — adversarial review, mutation
testing, and post-fix re-runs are inspectable platform stages, not
narrative claims about method). The architecture, the hypotheses, the
acceptance gates — registered before results were seen — and all editorial
decisions are the author's, and the author accepts full responsibility for the
content. All experiments ran on consumer hardware (RTX 3070 8 GB, RTX 4070
SUPER 12 GB, 12 GB-class cards). This is the third artifact of a research
program released together: a precision-decay law and its origins [1], the
attention kernel built on it [2], and — here — the memory system built beside
them.

---

## Abstract

Language-model memory systems overwhelmingly store *text* and pay for every
recall by re-reading it: retrieval-augmented generation re-prefills retrieved
passages, and agent memory frameworks re-inject notes into the context window.
The expensive artifact — the model's own contextualized attention state over
that text — is discarded and recomputed on every use. We present **GRM (Graft
Repository Memory)**, a memory runtime for *frozen* language models that
stores the attention states themselves: position-free K/V payloads
("grafts") harvested from live cache, routed by content, and re-seated
(re-positioned via inverse/forward rotary transforms) into future contexts
without re-reading. GRM organizes memory into four planes — VRAM as a
disposable mount arena (the model's live context window: the K/V cache it
actually attends over during generation), RAM as the authoritative live
repository, NVMe as write-ahead-logged durable storage, and a policy plane that treats
conversation turns as *evidence* and fact records as *memory*, with
revision-instead-of-overwrite semantics, a review buffer for uncertain
writes, and fidelity-gated consolidation (folds that lose facts must abort).
A dialect layer makes the same repository semantics run across attention
architectures — MLA latent caches, GQA, MQA with sliding windows — with a
*graftability profile* that enforces, at construction time, which cache
families may be re-seated at all. On consumer GPUs, gated evaluations show:
50-turn conversational recall through routed mounts with the live context
cleared every turn (DeepSeek-V2-Lite INT4); two-generation consolidation (40
turns → 10 digests → 2 era nodes) with recall through the folded structure;
42-turn infinite-context gates at flat device memory on MiniCPM3 (8/8 probes,
40 MB active device) and Qwen3-4B (8/8, with descent through digest
lineage); and warm-memory serving on Gemma-4 12B that replaces a 15.2 s
cold prefill with a 0.2 s mount (690 of 691 prompt tokens restored from
memory) and carries a 10-template job at 4.7 s that took the program's
previous production stack (Qwen3.5-9B, cold) 38 s — an 8× cross-stack
job-throughput gain. We position
GRM against prefix caching, position-independent cache fusion, and trained
KV-retrieval, and argue the conjunction it occupies — content-routed,
re-seatable, durable, *revisable* attention-state memory on frozen models,
across architectures — has, to our knowledge, no prior occupant.

---

## 1. Introduction

### 1.1 Text memory re-reads; state memory remembers

A transformer [3] that reads a document performs expensive, non-recoverable
work: every layer contextualizes every token against everything before it. All
existing mainstream memory approaches throw that work away. RAG [12] stores
text and re-prefills it on retrieval; agent memory systems [11] curate text
notes and re-read them; even prefix caches [4, 5] — which do keep K/V — can
only reuse it when the *same tokens reappear in the same positions*, making
them accelerators for repeated prompts, not memories.

GRM's premise: the contextualized K/V state is the valuable artifact. Store
*it*. The premise has empirical teeth from a negative result in this
project's own history: an attempt to train a small "dialect student" to
generate graft payloads from text — to flatten contextualization into a
cheap function — failed decisively (style transferred; factual *bindings*
never did: 0/10 across three training regimes). Contextualization is not
distillable at low cost; the process is the product. A system that wants the
product must therefore *cache and reuse* it, and everything in GRM follows
from taking that seriously.

### 1.2 What makes K/V state hard to treat as memory

Four problems separate "a cache" from "a memory," and they define GRM's
design surface:

1. **Position.** K/V entries are position-encoded; naive reuse at new
   offsets is wrong. GRM harvests grafts *position-free* (inverse-rotary
   export) and re-seats them at mount time (forward rotary at the target
   seat). The re-seating transform is exact as a mathematical identity for
   rotary-family models — and *only* for them, which is why remountability
   is a per-architecture contract (§2.4), not an assumption; in practice
   the full pipeline carries floating-point and quantization noise,
   measured end-to-end by the §4 parity gates (max abs Δlogit 0.195, zero
   top-1 flips, on INT4 DeepSeek).
2. **Addressing.** A memory must be found by content, not by prefix
   identity. GRM routes over per-graft keys (latent centroids for MLA
   dialects; raw `|q·k|` scoring for GQA) plus a lexical channel for exact
   identifiers.
3. **Lifecycle.** A memory system must know what is true *now*: facts vs
   evidence, revision, expiry, conflict, consolidation. GRM gives grafts a
   full metadata model and never overwrites — it supersedes (§3).
4. **Durability.** RAM-resident memory must survive crashes without
   blocking the hot path: write-ahead logging, atomic checkpoints, and
   recovery that tolerates the crash artifacts it will actually see (§3.3).

### 1.3 Lineage

GRM did not begin as a caching project; it began as a memory-manager sketch
and passed through a published near-miss. We record the lineage because
provenance matters for a system claiming an unoccupied design point:

1. **The intuition.** The author's original design — predating this
   program's formal work, and predating his understanding of transformer
   internals — was a "memory manager": a cheap process that reads the full
   context and tells each attention head what to focus on. (In retrospect,
   that is exactly the architecture of this program's attention kernel:
   APA's low-bit bulk pass scores everything and directs full precision to
   the fraction that matters [2].)
2. **The catalyst.** Wind's *Prometheus Mind* [15] demonstrated the
   mechanism the sketch was missing. It retrofits memory to a frozen
   Qwen3-4B as *"a removable module for frozen language models"*, injecting
   state *"directly into the model's attention mechanism"* so that *"the
   model attends to the memory through its native attention computation
   rather than reading it as text."* Its one-sentence critique of RAG is
   this paper's premise stated first and best: **"The model does not think
   with the memory; it reads it."** Prometheus Mind proved that frozen
   attention will *accept* injected state — no architectural change, no
   weight updates, fully reversible.
3. **The wall, and the question.** Prometheus Mind *synthesizes* its
   injected state: per-fact semantic directions discovered contrastively,
   with `lm_head` weight rows as value vectors. This works at single-fact
   granularity — its reported figures are 94.4% retrieval on clean inputs,
   degrading to 19.4% on informal inputs "with ellipsis, filler words, or
   implicit subjects," with relation classification (47.3% accuracy) named
   as the primary bottleneck — precisely because single facts are
   representationally simple. Our own attempt to
   scale synthesis — training a "dialect student" to generate
   whole-document K/V state from text — failed decisively (§1.1): style
   transferred, bindings never did. The synthesis road ends at simple
   facts. The question that became GRM: if the transformer already
   computes the memory state we want every time it reads, why generate it
   at all? **Capture it.** Harvest the model's own K/V from live cache,
   and the *synthesis* fidelity question vanishes — the payload is exactly
   what the model computed, unchanged. (Fidelity re-enters exactly once,
   where generation re-enters: consolidation digests are gated for fact
   coverage and abort below threshold, §3.2.)
4. **Proof before system.** The capture question was answered as a bare
   mechanism first: whole-document K/V injection ("KV-grafts") with routed
   mounts, gated for parity and recall on live models — the repository's
   KV-graft document-injection and router primers record this stage (§7).
   Only after grafts were proven did GRM become a *system*: the
   repository, durability, folding, and policy planes grew around a
   working graft, not the reverse.

(A parallel thread of the same program — a number-theoretic
precision-decay law and its direct measurement in attention — produced the
attention kernel, APA; that lineage is told in the companion papers
[1, 2]. The two threads meet in practice: APA is what lets the same
consumer GPU afford the long contexts GRM fills.)

### 1.4 Contributions

1. **The graft abstraction**: position-free K/V payloads with text,
   metadata, routing keys, and lineage — memory as the model's own state
   (§2.1–2.3).
2. **A cross-architecture dialect system** with enforced graftability
   profiles: the same repository semantics on MLA latent caches
   (DeepSeek-V2-Lite, MiniCPM3), GQA (Qwen3-4B), and MQA+sliding-window
   (Gemma-4 12B); fixed/learned-absolute position families are structurally
   refused re-seating at construction time (§2.4).
3. **A four-plane runtime** — VRAM disposable / RAM authoritative / NVMe
   durable / policy — with an opt-in native (C++) mirror whose boundary
   discipline we state as a design law: *plans cross the ABI; policy does
   not; Unicode and time never do* (§2.5, §3.3).
4. **Memory semantics for state, not text**: evidence/fact split, temporal
   validity, authority-gated supersession, a review buffer, explicit
   chat-level memory commands, and **fidelity-gated two-generation folding**
   (turn → digest → era) with descent back to sources — a fold that loses
   facts aborts, and the abort is persisted (§3).
5. **Gated evaluations on consumer hardware** across four model families,
   including 50-turn recall with the live context cleared every turn,
   recall *through* two generations of folded memory, and warm-memory
   serving at 8× the job throughput of the program's prior cold-prefill
   production stack (§4).

---

## 2. The Runtime

### 2.1 Grafts

A graft is the unit of memory: named K/V tensors (payload), the source text,
routing keys (dense + lexical), and metadata (kind, scope, durability,
mutability, temporal validity, provenance edges). Payloads are stored
*position-free*: at harvest, the live cache span is exported through the
model's inverse positional transform, so the stored state is
seat-independent. Payload layout is dialect-specific — an MLA graft stores
the latent (`c`, plus rope-carrying keys) [14]; a GQA graft stores per-head
K/V — behind one repository interface.

### 2.2 Harvest and mounting

**Harvest** is hybrid: the payload is sliced from the already-computed live
cache (cheap; captures what the model actually contextualized), while
routing keys are generated standalone so retrieval is not polluted by
surrounding context. **Mounting** re-seats grafts into an arena laid out as
`[attention sinks | mounted grafts | live tail]`: payloads are re-positioned
by forward rotary at their assigned seats and spliced into the cache in
fused device operations (splice/evict/re-seat/export primitives; a
functional multi-layer cache transaction where the dialect permits). Mounted
grafts *compose*: multiple grafts from different sources co-reside in one
arena — the property that pure-KV architectures grant and recurrent-state
hybrids do not (§5).

### 2.3 Routing and descent

Routing is per-dialect but law-compatible: MLA dialects rank by centroid
cosine over latent keys, max-over-keys for multi-key nodes (digest children
route on child centroids); GQA dialects score raw `|q·k|` per head,
rescaled per routing call by the maximum absolute score across the
candidate set; both dialects add a lexical bonus for
rare-token/identifier hits, which carries exact-identifier recall. Consolidated nodes carry their
source lineage, and **descent** expands a routed digest/era into its
underlying sources on demand — recall through folded memory reaches the
original evidence, not a summary of a summary.

### 2.4 Dialects and the graftability contract

A dialect descriptor is derived from the model (layers, KV geometry, payload
kind, position law, state kind, composition) and carries a **graftability
profile**: a profile claiming remountability must declare a rotary/relative
position law; fixed or learned-absolute position families are accepted only
as same-position restores. The profile is validated at construction on the
native plane (in both the Python wrapper around native initialization and
the C++ host store itself); in the pure-Python default configuration the
profile is computed and carried but not construction-validated — a known
asymmetry on the runtime's tracked issue queue (§5). Where enforced, it
turns a silent correctness hazard (re-seating a cache the architecture
cannot re-seat) into a refused configuration. Checkpoints persist the dialect and profile
identity and refuse cross-dialect loads.

### 2.5 The four planes

- **VRAM (disposable).** The arena *is the model's context window* — the
  live K/V cache the model attends over during generation, laid out as
  `[sinks | mounted grafts | conversation tail]`. Mounted memory therefore
  occupies context-window seats exactly as if the model had just read the
  content — that is what "the model thinks with the memory" means
  mechanically. Everything here is a disposable copy: eviction is safe by
  construction because nothing in VRAM is authoritative. A device-byte
  budget pages least-recently-mounted payloads out; descent reloads on
  demand.
- **RAM (authoritative).** The live repository: host payloads, metadata,
  routing index, review buffer, dirty tracking. A new memory is a memory
  the moment it exists in RAM.
- **NVMe (durable).** An LSN'd write-ahead log plus atomic checkpoints
  (write-temp, fsync, rename, fsync-directory), with recovery that
  rehydrates text/metadata/active-state from WAL alone, replays semantic
  revisions, and tolerates torn tails (§3.3).
- **Policy.** Everything that decides what memory *means* (§3), kept in
  the Python plane; an opt-in C++ mirror owns deterministic mechanics
  (host store, routing index, span planning, command parsing) as
  side-effect-free plans. Where the two planes could disagree — Unicode
  case folding, timestamp parsing — the boundary rule is that such values
  never cross the ABI: native prunes by exactly-comparable state, Python
  decides.

## 3. Memory Semantics

### 3.1 Evidence and memory

Conversation turns are evidence; fact records are memory. Facts carry
identity (subject/predicate/value + scope), durability
(volatile/session/project/permanent), mutability, write intent
(observed/inferred/user-asserted/system-asserted/imported), confidence, and
temporal validity (`valid_from`/`expires_at`). Conflicts are scope-aware
and time-aware: an expired fact does not block a new one; a same-scope
contradiction from a non-authoritative source is diverted to a review
buffer rather than written; duplicate facts *reinforce* the existing node
(confidence, intent rank, reinforcement count) instead of forking. Mutable
memory is never edited in place: a correction writes a new active node and
marks the old one superseded, preserving audit lineage.

The same semantics are user-reachable as explicit chat-level commands,
parsed by a deterministic grammar: `remember permanently: …` / `remember
this for the project: …` (durability-scoped writes), `forget: …` (tombstone
by content match — an empty match never means "everything"),
`correct memory: X => Y` (supersession), pin/unpin and mutability marking,
`approve/reject/edit review <id>` (review-buffer control), `cull graft <id>
into sections` (§2.1 spans), `show memory about: …` / `why do you remember:
…` (introspection with provenance), and durability-mode switches. Explicit
user commands outrank automatic extraction by design rule.

### 3.2 Folding with a fidelity gate

Old turns fold into digest nodes, and digests fold into era nodes — but a
fold must prove it kept the facts: generated digests are checked for fact
coverage against their sources and **abort below threshold (0.70)**, with
the abort exemption persisted so a rejected window is not retried in a
loop. Era nodes for extractive dialects are index nodes over child digests
(no generative re-summarization), leaning on descent for recall. In the
gated DeepSeek run, 40 raw turns retire under 10 digests and 6 digests
under 2 era nodes — with one fold aborted by the fidelity gate and four
no-fold exemptions recorded — and a fact from a retired turn is still
recalled through the folded path (§4).

### 3.3 Durability that expects to crash

Five durability modes (volatile → strict) trade latency for guarantees;
mode changes are themselves WAL-logged configuration records. The commit
order is fixed: payloads, index, native checkpoint, manifest, then durable
marks — nothing is marked durable before the checkpoint boundary commits.
Recovery replays WAL over the manifest bounded by LSN; a torn final WAL
record (the signature of a crash mid-append) is dropped and truncated so
the repository opens, while corruption *before* intact records refuses
recovery rather than guessing. An empty query can never mean "every node"
— destructive commands require a non-empty match, a law adopted after
red-teaming our own runtime.

*(Honesty note: the crash-safety surface was adversarially reviewed in July
2026, and three critical durability bugs were found and fixed **before**
the §4 test-suite result was produced at the current head: an empty forget
query that retired every node, a torn WAL tail that made recovery itself
crash, and a mark-durable-before-checkpoint ordering that silently staled
the native checkpoint. A queue of verified lower-severity issues — e.g. an
async-flush LSN race — remains open and is tracked in the project's
research board. §5.)*

## 4. Evaluation (gated, registered)

All results are from the project's registered gates; each gate was run on
the named commit and hardware, with pass criteria fixed before the run.
Non-GPU runtime suite: **133/133** at the current head (lifecycle,
durability, native-mirror parity, crash simulation).

| Gate | Model (quant) | Result |
|---|---|---|
| Smoke parity | DeepSeek-V2-Lite (INT4) | graft-vs-in-context last-token max abs Δlogit 0.195, 0 top-1 flips; greedy recall hit |
| Full paging | DeepSeek-V2-Lite (INT4) | 5 docs + 8 turns under a **2 MB** device graft budget: 4/4 open-ended exact-fact recalls after fresh-process resume, 4 RAM page-ins |
| Turn-50 boat | DeepSeek-V2-Lite (INT4) | 50 stored turns, live context cleared each turn; turn-1 needle recalled at turn 50 via routed mount |
| Folding | DeepSeek-V2-Lite (INT4) | 40 turns→10 digests→2 eras (1 fold fidelity-aborted, 4 no-fold); needle + retired-turn fact recalled through folded memory |
| Infinite context | MiniCPM3-4B (INT4, MLA) | 42-turn gate **8/8 probes**; max resident 341 seats; ~40 MB active device; 91 MB RAM payload |
| Descent | Qwen3-4B (BF16, GQA) | 42-turn gate **8/8 probes** incl. digest descent; 429 seats; 266 MB device; 13 RAM page-ins |
| Warm serving | Gemma-4 12B (QAT q4_0) | 690/691 prompt tokens restored from memory at 0.2 s prefill (same-model cold: 15.2 s); warm 10-template job 4.7 s vs 38 s on the prior Qwen3.5 production stack — **8×** cross-stack job throughput |

Three observations the table compresses:

- **Memory decouples context from residency.** The MiniCPM3 gate holds
  device memory flat (~40 MB active) while effective conversational
  context grows across 42 turns; the DeepSeek paging gate answers
  correctly with a device budget (2 MB) below any single graft's size,
  paging from RAM on demand. Context length becomes a *storage* question,
  not a VRAM question.
- **Recall survives consolidation.** The folding gate's needle is
  recalled *after* its turn has been retired into a digest, through
  descent — the property that makes folding a memory policy rather than
  lossy deletion.
- **Memory is a throughput feature.** The Gemma-4 result is not a recall
  gate but a serving one: re-seating remembered K/V replaces prefill on
  the same model (15.2 s cold → 0.2 s mounted), and the resulting warm
  stack carries templated jobs at 8× the throughput of the program's
  prior production stack — a smaller model (Qwen3.5-9B) running cold.
  The comparison is cross-stack by design: it measures what memory does
  to serving economics, not a same-model ablation. Reuse of
  contextualization is where the economics of state memory show up first.

## 5. Limitations

GRM is an actively developed system, and its open items divide into two
kinds that should not be read with equal weight. Most are **engineering
debt with known solutions from adjacent, mature fields**: routing scale is
an ANN-indexing problem (decades of solved art), and retention, eviction
policy, encryption, and access control are standard database practice —
none required new ideas, so none were built first. Two are **genuinely
open problems**: extraction quality and recurrent-state composition. The
research contribution of this paper is the part that had no adjacent field
to borrow from — position-free harvest, content routing over attention
state, and re-seating — and that part is built and gated.

1. **Routing scale is unproven.** Centroid + lexical routing is exact at
   the gated scales (hundreds of nodes); behavior at 10⁵+ nodes is
   unmeasured, and the routing index is a linear scan today.
2. **Storage scaling and retention are unspecified.** The RAM-authoritative
   plane grows with every graft; folding reduces node count but fidelity
   aborts mean not all growth is reclaimed, and no retention/pruning policy
   is yet specified for the authoritative plane or the NVMe archive. The
   persisted planes also carry conversation-derived content with no
   encryption or access-control story — durable memory inherits the
   security obligations of a database, and GRM does not yet meet them.
3. **Extraction is plumbing without a brain.** The fact-extraction
   interface, policy, and review buffer are built and gated; no learned
   extractor model is attached yet — extraction quality is future work.
4. **Composition does not extend to recurrent state.** Pure-KV
   architectures compose multi-graft arenas; hybrid recurrent-state
   models (e.g. DeltaNet-family) support single-prefix restore but their
   states do not concatenate — an open design problem, honestly outside
   the current system.
5. **A verified issue queue is open.** Adversarial review of the runtime
   confirmed and fixed three critical durability bugs (July 2026) and
   left a tracked queue of majors (async-flush LSN race, recovery
   placeholder edge cases, native command-parser edge cases). The gates
   above pass at head; the queue is disclosed rather than hidden.
6. **GPU gates are per-commit snapshots.** Cross-architecture GPU gates
   were last run on the commits recorded in the ledgers; the non-GPU
   suite runs at head. (Sustained multi-GPU re-runs are paused for a
   facility electrical issue — disclosed in the spirit of this program's
   documentation.)
7. **Single-process, single-writer.** The production daemon (shared
   memory pool, multi-agent handles) is designed but deliberately
   deferred until the API boundary stops moving.

## 6. Related Work

**Memory as injected state.** Prometheus Mind [15] is GRM's closest
ancestor in intent and the direct catalyst of this work (§1.3). It shares
GRM's two core commitments — frozen weights, and memory entering through
the model's *native attention* rather than the token stream — and differs
in the injected state's origin, granularity, and depth: Prometheus Mind
synthesizes per-fact direction vectors (contrastive discovery; `lm_head`
rows as values) injected through adapters at a single late layer of one
model, whereas GRM harvests multi-thousand-token K/V spans that carry the
model's own contextualization, mounts them at every attention layer, and
needs no adapters or training. GRM is, in one sentence, Prometheus Mind's
premise — think *with* the memory, don't read it — scaled past the
synthesis wall by extraction.

**Prefix and prompt caching.** vLLM's PagedAttention/automatic prefix
caching [4] and SGLang's RadixAttention [5] reuse K/V across requests
keyed by *exact token-prefix identity* at fixed positions; PromptCache [6]
generalizes to structured prompt modules with predeclared position
schemas. These are throughput systems: nothing is routed by content,
nothing survives the process durably as a first-class memory, nothing can
be revised or superseded. GRM shares the "K/V is worth keeping" premise
and none of the addressing model.

**Position-independent cache fusion.** CacheBlend [7] and EPIC's LegoLink
[8] reuse K/V of retrieved *text chunks* regardless of position by fusing
caches and selectively recomputing a small fraction of tokens — CacheBlend
to repair cross-attention, LegoLink to neutralize the spurious
attention-sink each chunk boundary acquires — the closest neighbors to
GRM's re-seating. The differences are structural: fusion systems are a fast
path *under RAG* (the text index does the addressing; caches are keyed by
chunk text), and their persistence is storage of opaque blobs — CacheBlend's
LMCache can spill K/V to disk — with no memory semantics: no facts, no
revision, no folding, no crash-recovery contract. GRM is the memory
system those fast paths lack: routed by the memory's own keys, durable
and crash-recoverable, revisable, and consolidating. The approaches are
complementary rather than competing — a fusion-style partial-recompute
repair pass is a plausible future addition to GRM's mount path.

**KV-cache retrieval with trained components.** Memorizing Transformers
[9] attend over stored K/V via kNN — but require *training* dedicated
retrieval attention; GRM's contract is frozen models only. KV eviction
and compression work (StreamingLLM's attention sinks [10], H2O [13],
cache quantization) manages the *live* cache within a window; GRM borrows
the sink insight in its arena layout but addresses a different problem —
persistent, addressable memory, not window management.

**Agent memory frameworks.** MemGPT [11] and successors implement memory
*operating systems* over text: paging, tiers, self-editing notes. GRM
agrees with the OS framing and moves it down a level of representation —
the pages are attention states, so recall costs a mount, not a re-read.
RAG [12] remains the text-plane baseline throughout.

**The unoccupied conjunction.** Each neighbor holds pieces: reuse
(prefix caches), position independence (fusion), retrieval over K/V
(Memorizing Transformers), memory semantics (MemGPT). The deeper split is
in what the cached state *is*. Every prior system treats stored K/V as an
immutable blob — keyed by the text that produced it, reusable only whole,
never inspected again. GRM treats it as **material**: a stored graft can be
sliced into child grafts along token spans (each child carrying its own
payload slice, routing keys, and lineage), reclassified along the
evidence/fact and write-intent axes, superseded by a correction, folded
into digests, and audited back to its source turns. Caching *reuses*
state; GRM *operates on* it. To our knowledge,
no prior system combines *content-routed addressing over position-free
K/V state, durable crash-safe storage, revision/consolidation semantics,
frozen-model operation, and cross-architecture support* in one runtime.
That conjunction — not any single ingredient — is GRM's claim.

## 7. Reproducibility

The runtime (Python policy layer + C++ host mirror + CUDA cache-surgery
primitives), the gate scripts, and the per-model port ledgers live in the
project repositories (private at submission; commit hashes for every gated
result are recorded in the project research board and ledgers, e.g. the
runtime head `964439b` for the non-GPU suite). The repository also carries
the system's primers and design records — `GRM_Primer.md`,
`GraftRepository_Memory_Architecture.md`, `GRM_Methodology.md`,
`KV-Graft_Document-Injection.md` (the proof-stage record of §1.3),
`ROUTER_PRIMER.md`, and the RAM-tiered runtime build plan and completion
audit — which document each design decision at the time it was made. All models are open-weight
releases; all hardware is consumer-grade. The evaluation harness clears
probe caches between independent probes, uses fresh-process resume for
durability gates, and registers pass criteria before runs.

## References

1. Perry, D. *Ghost Geometry: A Precision-Collapse Framework for the
   Collatz Conjecture, and Its Measured Transfer to Transformer
   Attention.* 2026. Zenodo `[DOI on release]`.
2. Perry, D. *Selective Attention Is All You Need: Adaptive Precision
   Attention.* 2026. Zenodo `[DOI on release]`.
3. Vaswani, A., et al. *Attention Is All You Need.* NeurIPS 2017.
   arXiv:1706.03762.
4. Kwon, W., et al. *Efficient Memory Management for Large Language Model
   Serving with PagedAttention.* SOSP 2023. arXiv:2309.06180. (vLLM)
5. Zheng, L., et al. *SGLang: Efficient Execution of Structured Language
   Model Programs.* 2024. arXiv:2312.07104. (RadixAttention)
6. Gim, I., et al. *Prompt Cache: Modular Attention Reuse for Low-Latency
   Inference.* MLSys 2024. arXiv:2311.04934.
7. Yao, J., et al. *CacheBlend: Fast Large Language Model Serving for RAG
   with Cached Knowledge Fusion.* 2024. arXiv:2405.16444.
8. Hu, J., et al. *EPIC: Efficient Position-Independent Caching for
   Serving Large Language Models.* 2025. arXiv:2410.15332. (the LegoLink
   algorithm)
9. Wu, Y., et al. *Memorizing Transformers.* ICLR 2022. arXiv:2203.08913.
10. Xiao, G., et al. *Efficient Streaming Language Models with Attention
    Sinks.* 2023. arXiv:2309.17453. (StreamingLLM)
11. Packer, C., et al. *MemGPT: Towards LLMs as Operating Systems.* 2023.
    arXiv:2310.08560.
12. Lewis, P., et al. *Retrieval-Augmented Generation for
    Knowledge-Intensive NLP Tasks.* NeurIPS 2020. arXiv:2005.11401.
13. Zhang, Z., et al. *H2O: Heavy-Hitter Oracle for Efficient Generative
    Inference of Large Language Models.* NeurIPS 2023. arXiv:2306.14048.
14. DeepSeek-AI. *DeepSeek-V2.* 2024. arXiv:2405.04434. (MLA) — plus the
    evaluated open-weight models: MiniCPM3-4B (Hu et al.,
    arXiv:2404.06395), Qwen3-4B (Qwen Team, 2026), Gemma-4 12B (Gemma
    Team, June 2026), DeepSeek-V2-Lite.
15. Wind, M. *Prometheus Mind: Retrofitting Memory to Frozen Language
    Models.* 2026. arXiv:2601.15324.

---

## Suggested Citation

Perry, D. (2026). *Grafted Memory — GRM: A Routed, Tokenless K/V Memory
Runtime for Frozen Language Models.* Zenodo. `[DOI on release]`
