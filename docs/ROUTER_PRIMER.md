# GraftRepository Routing — Short Primer

How the repository decides *which* stored grafts to bring into the model's
working context for a given turn — and why that search costs almost nothing.

---

## The one-line version

**The APA bulk-score pass is the router.** The same cheap, quantized
key-scoring the model already runs inside attention — to decide which keys
in its live context deserve full attention — is, pointed at the stored
grafts instead of the live context, the search over the entire repository.
One mechanism, two scales. There is no separate retrieval model and no
trained router. A graft's own keys are its address.

---

## 1. What a graft is indexed by

When a turn or document is stored, the model harvests a tiny **summary key**
for it — a compact fingerprint of that content's meaning, drawn from the
model's own internal representation. This key is kept in a small index
alongside the graft.

The exact form of the key depends on the model family (it is part of the
model's "dialect"), but the principle is constant: **the key is derived
from the content itself.** The content describes itself; the address is the
payload. That is why no router ever has to be trained — the grafts come
pre-addressed by the same representations the model uses to think.

---

## 2. How a turn gets routed

On each turn, using the bare user message as the query:

1. **Fingerprint the query.** The model produces the same kind of summary
   key for the incoming turn that the index stores for every graft.

2. **Score every graft.** Each stored graft's summary key is scored against
   the query's key — a similarity measure. This is the bulk-score pass:
   cheap, quantized, the same operation attention uses to triage keys.

3. **Add an exact-identifier bonus.** Similarity alone cannot tell apart two
   near-identical grafts that differ only by a specific code, name, or
   number. So an exact match on those identifier tokens between the query
   and a graft adds a decisive bonus that outranks raw similarity. This is
   what lets the router pick the *right instance* out of many lookalikes,
   not just the right topic.

4. **Mount the top few.** The best-scoring grafts (a small number — three by
   default) are mounted into the working context. Mounting *more* than
   needed actively hurts: co-present grafts interfere with each other, so a
   tight, well-chosen set beats mounting everything.

So the routing signal has three complementary channels: a **topical**
similarity (the summary-key score), an **exact-identifier** match (for
distinguishing lookalikes), and **lineage** keys (for reaching consolidated
memory — see below). All three are cheap and require no extra training.

---

## 3. Why the search is essentially free

This is the load-bearing point. A normal retrieval system bolts a *separate*
search engine onto the model — its own embeddings, its own index, often its
own trained components. Here, the search engine is **the attention mechanism
itself.**

Every time the model attends, it runs a cheap quantized scoring pass to
decide which keys in its current context matter most. Routing runs that
*identical* pass against the repository's stored keys instead of the live
context. Triaging keys inside a turn, and searching grafts across the whole
repository, are the same operation at two scales.

The consequence: a model that is good at the in-context triage is, by the
very same fact, good at repository search. One trained behavior delivers
both. There is nothing extra to build, train, or maintain — the search was
always there, hiding inside attention.

---

## 4. When the first pick is wrong

Routing produces a *ranking*, not a single answer, and the model is allowed
more than one attempt per turn.

If the first mounted grafts don't actually let the model ground its answer,
it takes another **trip**: the next-best grafts are mounted instead. And if
the router landed on a *consolidated* memory — a digest or an "era" that
summarizes many older turns — that consolidated node is **expanded into its
constituent pieces**, with the originals reloaded from cold storage and
fitted into the working context. The lineage keys are what make those
constituent pieces reachable.

This is what lets old, folded-away memory still be recalled precisely: the
router can land on a summary, and the trip mechanism descends from the
summary to the exact original fact underneath it.

---

## In one breath

Fingerprint the turn, score it against every graft with the model's own
quantized attention-scoring pass, break ties with exact-identifier matches,
mount the best few — and if that misses, take another trip and let
consolidated memories expand into their originals. No trained router, no
separate search engine: the bulk-score pass that triages attention is the
same pass that searches the repository.
