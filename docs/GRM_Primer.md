# GRM — A Primer

**Graft Repository Memory: routed, hierarchical, tokenless memory for a
frozen language model.**

This is the plain-language, end-to-end explanation: what GRM is, how it
works, what it unlocks, and the hypothesis behind it. No code.

---

## 1. The problem it solves

Every language model has the same memory problem. What a model "remembers"
during a conversation lives in its **context window** — the working span of
tokens it can attend to at once. That window sits in fast, scarce GPU memory,
and it is finite. So as a conversation or a body of reference material grows:

- the window fills up,
- and the model must either **forget** the oldest material (truncate) or
  **compress** it (summarize, which loses detail).

This is why long conversations degrade, why assistants "forget" what you said
an hour ago, and why everyone treats *infinite memory with no degradation* as
impossible. Within the usual frame, it **is** impossible — because that frame
assumes **memory = the context window**.

The other common fix, retrieval (RAG), keeps a library of text and pastes the
relevant bits back into the prompt each turn. But that text must be
re-tokenized and re-read every single time, it competes with reasoning for the
window, and the model treats it as *foreign text it has to interpret* rather
than knowledge it actually holds.

GRM breaks the assumption underneath all of this: **memory does not have to
live in the context window.**

---

## 2. The core idea: store computed meaning, not text

When a model reads a span of text, it produces internal state for it — the
"keys and values" each attention layer computes. That state, not the raw text,
is what the model actually *uses* to think. It is the **computed meaning** of
that text.

A **graft** is exactly that: the harvested internal state of a piece of
content — a conversation turn, a document, a fact — captured once and stored.

The key property, which is measured and not assumed: **a mounted graft is
indistinguishable from having read the original text.** When a stored graft is
placed back into the model's working state, the model's outputs match what it
would have produced with the text in context — down to the floating-point
noise floor of the hardware. A grafted document *is* the document, to the
model, at no re-reading cost.

So GRM stores history and knowledge as grafts — **out of the context window,
on disk or in RAM** — and brings only the relevant pieces back in when needed.

---

## 3. How a turn works (the whole loop)

Picture the working window divided into regions: a small permanent **anchor**
zone, an **arena** for mounted memory, the **recent** turns, and the **live**
prompt and response.

On each turn:

1. **Fingerprint the request.** As the model begins processing the incoming
   message, it produces a compact summary of what is being asked — drawn from
   its own internal representation. This is the routing query.

2. **Search the repository.** That fingerprint is scored against the summary
   key of every stored graft. The grafts whose meaning is closest to the
   request rank highest. (This is the "router" — explained in §6; it costs no
   extra tokens, only the time to look up and load.)

3. **Mount the winners.** The top few matching grafts are placed into the
   arena — their stored state is inserted directly into the working window,
   repositioned to sit at the right place. The model now has that knowledge
   present, as if it had just read it.

4. **Reason over everything.** The rest of the forward pass runs across the
   whole window — the live prompt *plus* the freshly mounted grafts — and
   produces the response.

5. **Harvest the turn.** The new exchange is itself turned into a graft and
   written back to the repository, so it can be recalled later.

Crucially, the working window only ever holds the **anchor + a few routed
grafts + the current turn** — never the whole history. History lives in the
repository. So the window stays small and constant no matter how long the
conversation runs, while the memory behind it grows without bound.

---

## 4. The librarian: keeping old memory usable

Left alone, a repository of every turn would grow unwieldy to search. So a
background **librarian** consolidates old material into a hierarchy:

- raw turns →
- **digests** (consolidations of several turns) →
- **eras** (consolidations of several digests).

This keeps the searchable index small while old memory still exists. When the
search lands on a digest or era, the system can **expand it back into its
constituent pieces** — pulling the original turns from cold storage and
mounting the exact fact underneath the summary.

The non-negotiable rule of consolidation is **recall over compression**: a
consolidation that would drop a fact is **refused**, and the originals stay
directly retrievable instead. The system would rather keep a thing
permanently than risk summarizing it into something it can no longer recover.
This is what lets old, folded-away memory still be recalled *precisely*,
rather than as a vague gist — the failure mode that makes ordinary
summarization-based memory degrade.

---

## 5. What it unlocks

- **Effectively infinite conversation, with no degradation.** History lives
  in the repository, so the window is constant for any length, and old facts
  remain exactly retrievable rather than summarized away. The thing everyone
  wants and assumes is impossible — because they assume memory lives in the
  window. It doesn't have to.

- **Knowledge as an addressable, growable resource.** Run any corpus through
  the model once and it harvests grafts for it — a coding library, a domain,
  a case file becomes mounted-able knowledge without retraining. Knowledge
  becomes something you *add by reading*, on disk, unbounded, rather than
  something baked into weights by training.

- **A memory hierarchy.** Grafts can live in GPU memory (active), RAM (the hot
  working set — recent conversation, always-mounted anchors), or disk (the
  cold archive). The cost of recall becomes seek time on a tiered store you
  can engineer — not a token cost competing with reasoning. The conversation
  runs at RAM speed; only deep, rare recall touches disk.

- **Anchored, always-present constraints.** Material placed in the permanent
  anchor zone is mounted every single turn and never evicted — a natural home
  for a system prompt, persona, or alignment constraints. Because the anchor
  positions are structurally required by the attention mechanism, those
  constraints are present *by construction*, not by polite instruction.

- **Cross-session and cross-population sharing.** The repository persists
  between sessions (resume picks up where it left off), and because it is a
  store separate from the model, many running instances of the same model can
  share one growing memory.

---

## 6. Why the search is essentially free — and works on any model

The "router" that searches the repository is **not a separate retrieval
model.** Scoring how relevant a stored graft is to the current request is the
*same operation a transformer already runs inside attention* — comparing the
query against keys. Attention does this constantly to decide which parts of
its own context to attend to; GRM points that exact same comparison at the
stored grafts instead of the live context.

So the search is not bolted on. It is the attention mechanism, used at a
second scale: **triaging keys inside a turn, and searching grafts across the
repository, are the same operation.** A model good at one is, by the same
fact, good at the other.

### On APA — and why it is not required

The system was first built alongside **APA** (an efficient selective-attention
technique). APA makes two things cheap: the relevance scoring (it runs a fast,
quantized version of the comparison) and very long contexts (its memory cost
stays flat as the window grows). Because of how they were developed together,
it is easy to assume GRM *needs* APA.

It does not. The two things GRM actually requires are present in essentially
every modern transformer:

- **Attention** — which provides the relevance score for free, as above.
- **Rotary position encoding** — which is what lets a stored graft be
  *repositioned* and inserted at any point in the window. Grafts are captured
  in a position-independent form and rotated into place when mounted; this is
  a property of standard rotary positioning, not of APA.

So GRM runs on any standard rotary transformer. The relevance score comes from
its ordinary attention; the mounting comes from its ordinary positioning. APA
is an **accelerator** — it makes the routing cheap and the context long enough
to hold a great deal of mounted memory at once — but it is the multiplier, not
the prerequisite. Remove APA and GRM still works; it simply costs more per
turn.

---

## 7. The hypothesis

GRM is the memory half of a larger bet, and it carries a hypothesis of its
own:

> **A model's knowledge does not have to live in its weights. It can live
> outside the model entirely — as addressable, mountable computed state —
> and the model can read it as well as if it had been trained on it.**

If that holds, several long-standing limits dissolve at once:

- Context length stops being a memory ceiling (history moves out of the
  window).
- Knowledge stops being fixed at training time (it is added by reading, and
  removed by deletion).
- Memory stops degrading (originals are kept, not summarized away).
- And knowledge becomes **inspectable and governable** — it is a store of
  readable artifacts you can audit, version, and delete, rather than an opaque
  blend inside billions of weights.

The deeper, paired bet — that a model can be trained to *reason* without
holding facts in its weights, leaving the facts to grafts — is what makes GRM
more than an efficiency trick. If reasoning can be separated from knowledge,
then GRM is not a cache bolted onto a model: it is **where the knowledge
lives**, and the model is **pure reasoning that reads it.**

That separation is the open question the broader project exists to test. GRM
is the mechanism that makes the answer *useful* either way: even on an ordinary
model that holds knowledge in its weights, GRM still delivers unbounded,
non-degrading, addressable memory. On a model built for separation, it becomes
the entire knowledge substrate.

---

## In one breath

> Store the meaning of what's been said as computed state, out of the window,
> on disk or in RAM. Each turn, let the model's own attention search that
> store, mount the few relevant pieces back into a small constant window, and
> reason over them — folding old memory into a hierarchy that can always be
> expanded back to the exact original fact. The window never fills, the
> history never ends, and nothing is forgotten. It needs only attention and
> rotary positioning — every modern model already has both.

---

*Companion documents: `GraftRepository_Memory_Architecture.md` (the full
architecture and measured results), `ROUTER_PRIMER.md` (the routing
mechanism in detail), `GRM_Methodology.md` (the experiment record).*
