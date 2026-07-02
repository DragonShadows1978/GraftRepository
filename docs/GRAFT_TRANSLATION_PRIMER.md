# Graft Translation Primer — Cross-Model Memory Portability

**Status: REGISTERED HYPOTHESIS — no experiments run. Nothing in this
document is a result.**

Idea: David Perry, 2026-07-02 (the evening the GRM/APA/Ghost Geometry
trifecta published). Primer drafted same session. Successor question to
SCRIBE (closed 2026-06-11, premise refuted — see §2, and
`project_kv_graft` closure notes).

---

## 1. The Question

Can grafts harvested from one model be made READABLE by a different model
of a like architecture — e.g., within one model family across sizes
(2B → 9B), or across tiers of a deployed fleet (a Haiku-class model's
grafts mounted by an Opus-class model)?

Raw cross-model grafts are dead on arrival by construction: K/V states
live in a model-specific learned basis (different weights → different
representational coordinates, head structure, layer depth, norms). GRM's
dialect system exists precisely to refuse such mounts. The question is
whether a **trained translation map** between two specific models'
graft dialects can carry enough of the state to survive GRM's gates.

David's framing: "I don't think it's possible... BUT that would be a
training thing." Correct on both counts — it is a training thing, and
whether it is possible is exactly what the gates below decide.

## 2. Why This Is NOT SCRIBE (the load-bearing distinction)

SCRIBE asked a shallow student to **create** contextualization from text:
synthesize the K/V payload a deep model would have computed. Refuted
decisively: texture/style transfers (G2 ≈ 88.9%), bindings never do
(G3 = 0/10 across three regimes). The autopsy's law: attention is the only
cross-position channel; depth buys composition; the process IS the product.
You cannot flatten the deep circuit into a shallow one.

Translation asks something categorically weaker: the deep circuit
**already ran** — in model A. The bindings exist, paid for, encoded in A's
states. A translator A→B does not re-do contextualization; it performs a
**change of basis** between two learned coordinate systems. SCRIBE's
refutation says nothing about this. The question is open, and the
literature leans favorable:

- **Model stitching** (Lenc & Vedaldi; Bansal et al.): independently
  trained networks connect through learned affine "stitching layers" with
  modest loss — representations are more alignable than naive intuition
  predicts.
- **Representational convergence** ("Platonic Representation Hypothesis,"
  Huh et al. 2024): models trained on similar data converge toward similar
  geometry up to rotation, and alignment improves with scale.
- **Cross-lingual embedding alignment** (Artetxe, Conneau et al.): whole
  embedding spaces align through near-linear maps fit on small
  dictionaries.

Within one family — same tokenizer, same data lineage, adjacent sizes —
conditions are maximally favorable. The null hypothesis is not "needs a
deep student"; it is "**a per-layer linear map might be enough**."

## 3. Protocol (registered, escalation-laddered)

### 3.1 Candidate pairs (in order of preference)

1. **Qwen3 dense family, small → large (e.g., Qwen3-1.7B → Qwen3-4B).**
   Pure GQA, shared tokenizer; the 4B is already ported to tensor_cuda
   with GRM descent gates green. Cleanest first experiment; requires a
   small-model port on existing GQA dialect infrastructure.
2. **Qwen3.5-2B → Qwen3.5-9B.** Same-family hybrid; 9B already ported and
   GRM-entry-gated. CAVEAT: hybrid recurrent state is a priori
   untranslatable by this method — attention-layer KV only. Use only if
   the dense pair is unavailable.
3. Cross-family (Gemma ↔ Qwen): explicitly out of scope for round one.
   Different tokenizers likely fatal; revisit only if within-family
   passes.

### 3.2 Pair corpus

Run both models over identical text; harvest both KV sets **pre-RoPE**
(GRM's position-free export convention — inverse rotary already removes
the position confound). Every document yields aligned (A-state, B-state)
pairs at every layer. The SCRIBE pair-corpus machinery (20GB corpus,
INT4-backprop training loop in tensor_cuda) is the reusable lab equipment.

### 3.3 Escalation ladder — STOP at the first rung that passes

1. **Linear (the null hypothesis).** Per-layer least-squares / Procrustes
   maps on K and V separately, full KV-space (allow head mixing — heads do
   not correspond 1:1 across models). Fractional-depth layer alignment for
   depth mismatch (layer ℓ_A → layer round(ℓ_A · L_B/L_A)).
2. **Low-rank + mild nonlinearity** (small MLP per layer) only if linear
   fails gates but shows signal (partial G3, high attention-pattern
   correlation).
3. **STOP CONDITION (SCRIBE's lesson):** if required translator capacity
   approaches "re-run the model," the premise is dead — document and
   close. A translator that must do deep-circuit work is SCRIBE wearing a
   trench coat.

### 3.4 Gates (reuse the SCRIBE G-suite + GRM dialect gates)

- **G0** — pipeline integrity: translate B→B via identity config,
  bit-exact.
- **Attention-pattern fidelity**: translated-A keys vs B-native keys
  against B's live queries — max-abs-Δlogit ceilings, same style as the
  dialect surface gates.
- **G2** — texture: does mounted translated state read as B-coherent
  (style, fluency)?
- **G3 — THE decisive gate**: bindings. Mount translated A-grafts into B;
  probe facts contextualized in A's reading (the 10-probe battery). Score
  >0/10 and the door is open; SCRIBE scored 0/10 on creation — any
  positive binding transfer is a categorical improvement.
- **GRM arena end-to-end**: route → mount → decode with translated grafts
  in a live B session (the turn-recall gates, reduced battery).

### 3.5 Outcome tree (all outcomes are publishable)

- **G3 passes at high rate** → memory portability is real within families.
  Short paper; product implication immediate (see §4).
- **G2 passes, G3 fails** → translation carries texture but not bindings
  (SCRIBE-echo). CONSOLATION PRIZE IS REAL: translated grafts still work
  as **routing keys** — cross-model retrieval indices with native
  re-harvest on promotion (cheap cold-index across a fleet).
- **Everything fails at linear AND low-rank** → within-family translation
  requires deep capacity → closed as SCRIBE-class refutation, documented
  with the same honesty. The negative result still bounds what
  "representational convergence" buys in the KV plane — nobody has
  published that boundary.

## 4. Why It Matters

**For GRM (open models, David's stack):** memory portability across model
upgrades. The repository outlives any single model — swap Qwen3.5 for
Qwen4, keep the memories. This is the question every eventual user of a
GRM-backed assistant will ask ("new model — do I lose everything?"), and
the feature that turns GRM from a memory system for *a* model into a
memory **format**.

**For weights-holders (the "DOA for me, NOT for Anthropic" point):** a
frontier lab with a model family holds both the weights and the training
budget. Two escalating plays:

1. **Trained translators between tiers** — the cheap tier harvests
   (reads documents, holds conversations), the expensive tier mounts
   (reasons over the pre-contextualized state). Tier-hopping stops paying
   re-prefill; mid-conversation reroutes stop being fidelity cliffs;
   prefill cost amortizes across the fleet and across model versions.
2. **The stronger move — train the family to READ A SHARED GRAFT DIALECT
   natively** (an interlingua; graft-native training à la GRAPA, at
   frontier scale). No translators at all: any tier reads any tier's
   memory, by construction.

Either play converts serving-cost prefill into a one-time harvest expense.
That economic argument is why this primer exists as a timestamped document
rather than a conversation.

## 5. Honest Obstacles

- **Head-structure mismatch**: no 1:1 head correspondence; maps must be
  full-KV-space per layer (head-mixing), which grows parameter count —
  watch the §3.3 stop condition.
- **Two pathways must BOTH land**: keys must produce correct attention
  patterns against B's queries AND values must decode correctly under B's
  output circuitry. A map good for one may be poor for the other; gate
  them separately.
- **qk-norm / scaling differences** between family members change the
  score geometry the translated keys must survive.
- **KV-group count mismatch** (GQA group structure differs across sizes).
- **Dimension mismatch direction**: small→large is the friendly direction
  (room to spare); large→small provably loses information — expect
  asymmetric results and report both.
- **Hybrid architectures**: recurrent-plane state is out of scope;
  attention-layer KV only.

## 6. Provenance

Born 2026-07-02, hours after the trifecta published, from the exchange:
"Is it POSSIBLE that we could make LIKE architectures — Haiku Grafts,
Sonnet Grafts, Opus and Fable Grafts — READABLE? I don't think it's
possible... BUT that would be a training thing." The distinction that
makes it worth trying — translation transports contextualization that
already exists; SCRIBE died trying to create it — is the entire bet.

If pursued to a result (either direction), consider a short Zenodo note:
*"Cross-Model Graft Translation: Memory Portability Across Like
Architectures"* — citing GRM (doi:10.5281/zenodo.21138607) and the SCRIBE
closure as the constraint surface.
