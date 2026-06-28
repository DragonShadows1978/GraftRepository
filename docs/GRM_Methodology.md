# GRM — Graft Repository Memory: Methodology, Mechanism, and Experiments

**A routed, hierarchical, tokenless memory for frozen LLMs.**

This document explains how GRM works, why it gives effectively unbounded
context at bounded VRAM, and the full record of experiments run against
it. It is a synthesis of the primary sources — do not treat it as the
authority where it disagrees with them:

- `core/graft_repository.py` — the implementation (329 lines)
- `docs/GraftRepository_Memory_Architecture.md` — the architecture +
  recorded receipts (the "Foundations" table is the result ledger)
- `docs/KV-Graft_Document-Injection.md` — the mounting-equivalence proof
- `tests/test_graft_*.py` — the 22-file experiment suite (protocols and
  gates; results are recorded in the architecture doc, not embedded in
  the tests)

Code references are `file:line`. Quoted material is verbatim from the
sources.

---

## 1. The one-line thesis

> "store every turn and document as harvested K/V ('grafts'), route into
> them per turn with APA's own scoring, mount only the winners into a
> fixed positional arena, and let a background librarian consolidate old
> memory into hierarchical digests — unbounded memory at bounded
> residency, zero training, on a frozen model."
> — `GraftRepository_Memory_Architecture.md:8-11`

Every clause is a mechanism. The rest of this document is those clauses.

---

## 2. The problem it solves: the re-read toll

Text memory — RAG, system-prompt summaries, an external KB — re-pays
**tokenization + prefill every session**. A 2,300-token transcript costs
2,300 tokens of context and the compute to attend over them, every time,
and it grows without bound. The model's usable history is capped by its
trained context window, and the window is capped by VRAM.

GRM's claim: history does not have to live in the window as text. What a
transformer actually *uses* from prior context is the **post-attention
K/V state** those tokens produced. If you harvest that state once and can
re-inject it later at the right positions, you get the memory **without
re-paying the tokens**. That is what "tokenless" means here:

> "Grafts mount as computed meaning at zero prompt cost."
> — `GraftRepository_Memory_Architecture.md:18-19`

---

## 3. What a graft is

A **graft is harvested K/V-cache state** — not text, not tokens. It is the
computed meaning of a turn or document, captured from the model's forward
pass and stored so it can be re-mounted later.

**Physical payload** (`graft_repository.py:18-19`): per-node on disk,
`nodes/NNNN.npz` holds the latent graft —
- **MLA models:** the latent pair `(c_n latent (L,S,256), pre-RoPE k_pe
  (L,S,32))`, fp16 — 288 values/token/layer.
- **GQA models:** standalone pre-RoPE full K/V.

The geometry is the model's **dialect** (§9). A node also carries metadata
in `self.arena.grafts[i]`: `kind`, `text`, `ntok`, `sources` (lineage),
`retired`, `no_fold`, `tags`, `rare` (lexical keys), `cent` (routing
centroid), `h` (device handle, `None` when paged out), `child_cents`.

**The certification that a graft IS the text, to the model**
(`GraftRepository_Memory_Architecture.md:27`):

> "Logit-level equivalence: top-1 identical, max logit diff ~bf16 noise,
> per-layer residual rel-diff 0.0035–0.019."

A mounted graft is logit-indistinguishable from having had that text in
context — at zero prompt-token cost. This is the load-bearing result the
whole design rests on (`KV-Graft_Document-Injection.md`).

**The four kinds** (`graft_repository.py`):
- **`turn`** — a verbatim conversation turn (default; `add_turn`, `:97`).
- **`doc`** — knowledge ingest (`add_document`, `:101`). *Never folded* —
  "reference material, not history" (`:118-119`).
- **`digest`** — a consolidation of folded turns (`:124`).
- **`era`** — a digest-of-digests, a consolidation of folded digests
  (`:128`).

---

## 4. The core trick: mount by cache surgery, not prefill

Mounting = inserting a stored graft's K/V into the live cache at reserved
positions **without re-running the prompt that produced it.** The
architecture doc calls this **cache surgery**.

The math that makes a position-free graft seatable anywhere
(`GraftRepository_Memory_Architecture.md:32`):

> "RoPE(p+Δ) = R(Δ)·RoPE(p); one vectorized pass re-seats cached keys
> exactly."

So grafts are **harvested pre-RoPE** (position-free) and **re-RoPE'd at
whatever arena seat they land in**. The harvest hook is "post-qk-norm /
pre-RoPE" (`:29`). For MLA, only the 32-dim `k_pe` is re-rotated; the
latent `c_n` is position-free (`:49`, `:191`).

**Why there is no re-prefill** (`:180-184`): injection is prefill-only —
the graft cat is gated on `kv_cache is None`, and the positional offset
persists across decode via a `graft_seats` attribute. The mounted graft
reads correctly at every later decode step without being re-attended.

**Certification of the seated graft** (`:35`):

> "Prefill-only injection + persistent `graft_seats` shift: 5/5 recall
> across turns; teacher-forced A-vs-C logit diff 0.5-1.5 = the plain
> cache's own bf16 noise floor … top-1 identical at every position."

Eviction/swap is also cache surgery, and free to roll back because cache
tensors are immutable: "Failed attempts roll back entirely (immutable
cache tensors make snapshot/restore free)" (`:41`).

### The seating plan (the arena)

The live window is partitioned (`:124-136`):

```
 seat 0..~4                16,384        24,576              32,768
   |SINK|------- ARENA ---------|--RECENCY--|------ LIVE --------|
   |    | routed mounts, per turn| always-on |  prompt + output   |
   |    | (re-seated every turn) | last N    |                    |
```

- **SINK (seats 0..~4) is sacred and never unmounted.** Removing a mount
  that includes seat 0 collapses generation into repetition (the
  StreamingLLM attention-sink effect). The arena *starts after* the sink.
- **ARENA** holds the per-turn routed mounts, re-seated every turn.
- **RECENCY** is the shared expert — last N turn-grafts, always mounted,
  never routed (for anaphora).
- **LIVE** is the current prompt + output.

An engine `live_shift` hook keeps the arena a fixed width regardless of
how big the mounted set is: mounts occupy an arena prefix, the remainder
is a positional hole (`:49`).

---

## 5. Routing: which grafts to bring in

**The router is not a model call.** It is APA's own bulk pass — quantized
query·key scoring — pointed at the repository's summary keys instead of
the live context (`:87-88`). There is **no router training anywhere in
the design** (`:92`); a graft's keys *are* its routing interface ("the
address is the payload", `:88-92`).

**The per-turn signal:** the prompt's own queries. E1 resolved the depth
dial to **layer 0 only** — "The router costs ONE layer-0 q-projection"
(`:228`). The scoring law for QK-normed GQA models (`:222`):

> "Score = mean over q-heads of max over (probe-q, graft-k) pairs of
> |q·k|/√Dh, pre-RoPE both sides (position-free)."

The law **forks by dialect**: MLA models have no qk-norm, so the key-space
router fails (outlier-key-norm pollution) and routing switches to
**latent-centroid cosine** — `cos(mean c_n_probe, mean c_n_graft)`
(`:238-252`). "The routing index is part of the dialect."

**Routed beats mount-all.** This is not just a residency saving — over-
mounting actively hurts (`:37`, `:222-226`): co-mounted grafts interfere
("rumination spirals, digit corruption: '07:40' for 07:42"). E1 measured
**routed top-3 = 10/10 vs mount-all = 4/10** on Qwen3. "Over-mounting is
the other forgetting."

**The three-channel hybrid index** (from CORPUS-100, `:310-316`):

> "the routing index is a THREE-channel hybrid — latent centroid
> (topical) + rare-token lexical keys (identifiers; exact match dominates
> — centroids cannot separate near-duplicate siblings) + child centroids
> (descent). All three are bytes-cheap and harvest-free."

The lexical channel exists because topical centroids cannot tell apart ten
sibling documents that differ only by a code (the CORPUS-100 failure mode:
4/20 routing on centroids alone → 20/20 once the lexical channel was
added).

---

## 6. Why context is effectively infinite

Unboundedness here is **structural, not a bigger buffer.** The mechanism
is the **ephemeral boat** (`test_graft_infinite.py`, verbatim):

> "INFINITE-CONTEXT gate: ephemeral boat ('clear the memory window at the
> start of each turn'). Every turn runs on [sink | mounts | turn] alone —
> resident seats CONSTANT for any conversation length; history exists only
> as repository nodes; recency is a MOUNT (last 2 turn-grafts) for
> anaphora."

Each turn the live cache is cleared and rebuilt as `[SINK | routed mounts
| current turn]`. History is **never** retained in the live window — it
lives entirely as off-context repository nodes, re-mounted on demand by
routing. Therefore residency is **constant for any conversation length.**

The measured proof (`GraftRepository_Memory_Architecture.md:44`):

> "42-turn history at ≤456 resident seats FLAT (transcript would be 2,300+
> and growing); the live-window echo failure class structurally
> eliminated."

Four supporting pillars:
1. **Lossless mounting** (§4) — a graft *is* the text to the model.
2. **Pre-RoPE relocatable keys** (§4) — re-seat exactly at any position.
3. **Memory ceiling = trained window, not VRAM** — "MiniCPM3: every rung
   2,048→32,768, resident flat ~2,856MB."
4. **VRAM decoupled from corpus size** — the paging result (§8).

The infinite gate result: **8/8 recall at ≤456 resident seats, flat,**
across 42 turns — including era-folded facts (recall through *double*
consolidation, turn→digest→era) and an anaphora probe ("And what time
exactly?" → 10:30).

---

## 7. The librarian: folding, digests, eras, descent

Old memory is consolidated by a background **librarian** so the routing
pool stays small and old turns compress — without losing facts.

**Fold thresholds** (`graft_repository.py:51-52`): `TURNS_HIGH,
TURNS_FOLD = 8, 4` and `DIGESTS_HIGH, DIGESTS_FOLD = 6, 3`. 8 active turns
→ fold 4 into a digest; 6 active digests → fold 3 into an era. The plan is
computed **statelessly** (`_due()`, `:116-129`) and executed by
`_fold_once()` (`:134-149`). Folded sources are **retired** — their VRAM
is freed; disk is their cold storage (`:207-218`), reloaded on demand by
descent.

**Digests are fixed points** (E2): consolidation costs once (coefficient
~0.89) then plateaus — D0 direct 9/10 → D1 through-digest 8/10 → D2
digest-of-digest 8/10, **zero second-generation decay** (`:254-264`).
Hard rule: "Digests must spell facts in their own tokens" (`:33`).

### Fidelity-gated folding — "recall > compression"

The single most important safety rule (`:46`):

> "FIDELITY GATE on EVERY fold: fact set = identifier tokens + multi-word
> named entities; the best candidate must cover >=0.70 of it or the fold
> ABORTS and sources are marked `no_fold` (persisted), permanently
> resident — root cause: a digest dropped '$7,400'+'Lake Arrowhead' at
> GENERATION; the facts then existed in NO node text -> unroutable,
> unrecoverable (recall > compression)."

Code (`:139-146`): if `consolidate()` returns no digest, the sources are
flagged `no_fold` (persisted, restored on load) and the planner moves on.
A fold that would lose facts is **refused** — the sources stay directly
routable forever rather than compress into something unrecoverable.

### Eras as index nodes, never readers

The central era rule (`:43-50`):

> "Era folding is ON and safe BY CONSTRUCTION: eras are INDEX nodes — the
> trips ladder expands them to their child digests at the primary attempt,
> so era text is routed into but never read."

Era *text* is unsafe to read because multi-source re-synthesis either
strips relations (list-form) or invents them (prose-form: "Project
NIGHTJAR was conducted by Priya Raghunathan" — fact fusion). So an era
node exists only to be *routed into*, then **expanded to its child
digests** (reloaded from cold storage, budget-fitted to the arena) on the
retry. This is **hierarchical descent**: a parent scores by max over its
child centroids; descent keys rebuild recursively from lineage up to depth
3 (`_rebuild_child_keys()`, `:269-281`).

### Inline vs deferred librarian (hot-path-flat)

Two modes (`:54-69`):
- **inline** — folds run inside the turn (simple; a fold stalls ~3s).
- **deferred** — the hot path *never* folds; due work is computed
  statelessly and drained by `idle()` between turns; a 2× backpressure
  threshold folds inline only as a last resort.

Measured (`:46`): deferred = max `add_turn` **0.27s FLAT** vs inline
spikes of 3-9s; recall **8/8** unchanged; 9 folds ~3.9s each, off-turn.

---

## 8. Experiment ledger

> **Provenance:** the `test_graft_*.py` files hold the protocols and gates
> (docstrings + assert/print logic); the *recorded results* live in
> `GraftRepository_Memory_Architecture.md` (the "Foundations" receipts).
> Numbers below are quoted from those receipts. Two gates are explicitly
> **stale or never-run** — flagged as such; do not cite them as landed.

| Experiment | Hypothesis | Result (recorded) |
|---|---|---|
| **E1 — router** | routed top-k recall ≈ (or beats) mount-all | **Qwen3: routed top-3 10/10 vs mount-all 4/10**; layer-0 \|q·k\| router. MiniCPM3 (latent, L10): routed 7/10 = mount-all 7/10 |
| **E2 — digest fidelity** | measure consolidation decay over a 2-gen chain | D0 9/10 → D1 8/10 → D2 8/10 — **zero 2nd-gen decay**; coefficient ~0.89, then fixed point |
| **E4 — conversation** | graft-routed memory at parity with full transcript | **6/6 at ~25% residency** (baseline 6/6 growing; amnesia control 0/6) |
| **E4-ARENA** | swap/evict as cache surgery on ONE never-rebuilt cache | **6/6** through 20 turns, 6 routed swaps, ~20 evictions; residency 268-316 seats flat |
| **E4-C — consolidation** | recall through digests after sources retired | **6/6** through digests alone; routing pool 14→8 nodes |
| **E4-TRIPS — shuttling** | small arena + trips ≈ topk=3 | arena starved to ONE mount 5/6; +max_trips=2 **6/6** = the 3-mount arena |
| **DESCENT** | era-folded recall recovers via child-digest expansion | **8/8** (from 3/8 without descent — see §9) |
| **LIBRARIAN** | deferred folding keeps hot path flat, recall intact | hot path **0.27s FLAT**, recall **8/8** = inline reference |
| **INFINITE** | constant residency over long history | **8/8 at ≤456 seats FLAT** over 42 turns incl. era-folded + anaphora |
| **CORPUS-100** | routing under 100 near-duplicate siblings | **20/20**; 416MB grafts, 50KB index, 1.3s/probe (took 4 fixes incl. lexical channel) |
| **PAGING** | recall under a tiny VRAM budget | **20/20** at a **64MB** budget over 100 docs; ~66MB resident, +0.1s/probe |
| **MLA-GATE** | harvested graft ≡ in-context (falsifiable) | G1 graft-vs-in-context max diff **0.41, top-1 IDENTICAL**; G3 recall 3/3 |
| **GQA-ARENA** | second-dialect (Qwen3) arena parity | **6/6** unified arena; starved-trips 6/6; E4-C 6/6 |
| **GQA-DESCENT** | flagship 42-turn on the GQA dialect | ⚠ **5/8, measured pre-early-stop-fix only — NOT re-gated** |
| **MIGRATE** | cross-model migration (text crosses, K/V never) | ⚠ tool + gate **written, NEVER RUN** — no pass number exists |

---

## 9. The 3/8 → 6/8 → 8/8 descent story

The clearest illustration of how the era machinery was hardened — the
era-folded 42-turn recall climbing as each fix landed (`:391-417`):

1. **3/8 (prose-era) / 4/8 (list-era) — no descent.** The era-depth
   negative result: folding multiple digests into one era text fails both
   ways — list-form eras strip relations, prose eras *invent* them (fact
   fusion).
2. **3/8 → 6/8 — descent added.** Eras become index nodes expanded to
   their child digests on the primary attempt; children identifier-
   filtered, cold-storage reloaded, budget-fitted to the arena.
3. **6/8 → 8/8 — relational first-gen digests.** Digest prompts demand
   complete sentences naming each fact's referent (bare-bullet QC on all
   folds). 8/8 became the standing reference for the infinite, descent,
   and librarian gates.

Two "improvements" were tested and **REFUTED** (each regressed 8/8 →
5-6/8): exempting era folds from the fidelity gate, and score-ordered
`fit()` truncation. Both are documented cautions, not open ideas.

---

## 10. Frozen, tokenless, dialect-bound

- **Frozen / zero training.** Model weights are never touched. The router
  is APA's bulk quant pass, not a learned component — "no router training
  exists anywhere in this design" (`:92`).
- **Tokenless.** The hot memory path pays no token cost — no
  tokenization, no prefill — because what is stored *is* the post-attention
  K/V, mounted by cache surgery (§4).
- **The dialect wall.** A graft's K/V is in the model's private residual-
  stream basis. The dialect string is built from K/V geometry
  (`:73-78`: `{ModelName}:{layers}x{hidden}:{r<rank>|g<heads>x<dim>}`),
  and `load()` refuses a repository harvested on a different model
  (`:246-254`): "K/V artifacts never transfer across models (texts
  survive; re-harvest to migrate)." `migrate()` (`:283-315`) is the escape
  hatch — it carries **texts only** and re-harvests every node under the
  new model's weights; lineage, kinds, tags, and fold-exemption flags are
  preserved so descent keys rebuild exactly. (This path is the
  never-run gate above.)

---

## 11. Open edges (honest status)

The deepest 42-turn / era-fold story is gated green on **MLA (MiniCPM3)**.
On the **GQA (Qwen3) dialect**, the arena/trips/E4-C gates pass (6/6), but:

- **GQA descent re-gate is stale at 5/8** — measured before the
  early-stop fix, never re-run. Not a landed number.
- **Cross-model migrate gate was never run** — no pass number exists.

These are the two places the "infinite context across any model" claim is
not yet closed by measurement. Everything else in the ledger is recorded
green.

---

*Sources: `core/graft_repository.py`,
`docs/GraftRepository_Memory_Architecture.md`,
`docs/KV-Graft_Document-Injection.md`, `tests/test_graft_*.py`.
Synthesized 2026-06-13. Where this document and a primary source
disagree, the primary source wins.*
