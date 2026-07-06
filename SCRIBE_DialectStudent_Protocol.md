# Experiment Protocol — Dialect Student: Training a Predictor of a Frozen Model's KV-Grafts

**Status: pre-registered, not yet run. Every number below is a hypothesis or a dial,
not a result. Target hardware: one RTX 3070 (8GB). Stack: existing harvest/mount
harness + equivalence harness as the measurement instruments.**

Working name placeholder: **SCRIBE** (naming is the Architect's job).

---

## 0. The question

Can a small student model, trained on (text → exact harvested K/V) pairs from a
frozen target model, produce grafts in the target's dialect cheaply enough and
faithfully enough to serve as the repository's cold ingestion tier — with exact
harvest reserved for promotion?

**Prior art (for positioning):** Apple's "KV Prediction" (2024) trained an auxiliary
transformer to predict a base model's KV cache from the prompt, for time-to-first-token
reduction. Mechanism proven viable; quality loss real. Unoccupied ground this protocol
claims: persistent-repository ingestion with a **logit-level fidelity certificate
against a lossless tier** — they had no exact baseline to measure degradation
against; we do, and the equivalence harness is the instrument.

## 0.1 Registered hypotheses (stated before any run)

- **H1:** Router recall over predicted grafts survives nearly intact (layer-0 keys
  are near-embedding; the router lives at layer 0).
- **H2:** Verbatim needle readback degrades first and most (deep-layer values carry
  it; deep layers are hardest to predict).
- **H3:** Per-layer prediction error grows monotonically with depth.
- **H4:** Digests authored from predicted grafts are within noise of digests
  authored from exact grafts (digestion is already lossy; it should mask
  upstream prediction error).
- **H1+H2 jointly:** addressability is cheaper than content.

---

## 1. Target and prediction objective

- **Target model:** MiniCPM3-4B (MLA), INT4 weights, existing adapter — chosen
  because MLA collapses the prediction target to the **compressed latent** per
  token per layer (kv_lora_rank as configured in the adapter), an order of
  magnitude smaller than per-head K/V, and because the repository already stores
  MLA grafts latent-side. Predicted artifacts drop into the existing graft format
  unchanged. (GQA targets are a follow-on, not in scope.)
- **Prediction target:** pre-RoPE latents, exactly as the harvest path captures
  them — the student mints artifacts for the *existing* mount machinery; nothing
  downstream changes.
- **Information structure:** target latents at token t are functions of tokens ≤ t
  (causal prefill). The student is **causal** to mirror this. (A bidirectional
  variant is a permitted ablation, not the main arm.)

---

## 2. Arms

- **ARM-L (floor):** linear/MLP probe from the target's token embeddings to all
  layers' latents. Prices what a transformer student buys over a trivial map.
  If ARM-L passes gates, the whole problem is easier than believed — that's a
  finding, not a failure.
- **ARM-S (standalone student):** small causal transformer trained from scratch,
  per-layer prediction heads. Param count is a dial set by the Phase 0 fit probe.
- **ARM-T (truncated-target student):** the target's own first N layers (frozen
  or lightly tuned; N a dial) + prediction heads for the remaining layers'
  latents. Equivalent to "partial prefill + predict the rest." Registered
  expectation: ARM-T beats ARM-S at equal cost; if not, that's informative.

---

## 3. Training data and losses

- **Pair minting:** prefill a training corpus through the frozen target using the
  existing harvest path; store (text, latents) pairs. **The teacher is a function,
  not a dataset — supervision is unbounded at one prefill per document, and the
  existing repository is already a minted seed set.** Corpus: research-corpus
  documents + general text mix; **held-out documents (not just held-out spans)**
  for every gate. Domain composition logged — distribution shift is a registered
  door (§7).
- **Minting policy:** volume is free; coverage is the constraint. Phase 1+ minting
  is **coverage-weighted, then error-directed**: after each training round, mint
  new pairs preferentially where the student's per-layer/per-domain error is
  highest. The teacher answers any question; ask it the ones the student keeps
  failing. (Consequence for §5: T_train is adaptive, not fixed — report its
  trajectory, not just its total.)
- **Loss schedule:**
  1. **L-warm:** Huber on latents, per-layer normalized (depth-weighted variant
     is a dial), until plateau.
  2. **L-func (the real loss):** mount the predicted cache in the frozen reader;
     KL on logits against the exact-graft run over probe continuations. The
     equivalence harness repurposed as the training signal. Functional phase may
     subsample layers/positions per step to keep the inner target-forward
     affordable; subsampling schedule logged.
- **Checkpoint/resume spec (standing rule):** a checkpoint restores ALL of:
  student weights, fp32 master weights, optimizer moments, RNG state, data-stream
  position, loss-phase position (warm vs functional, subsampling schedule state).
  **Resume-from-kill is a Phase 0 gate:** checkpoint, kill, resume, verify loss
  trajectory continuity before any long run is trusted.

---

## 4. Pre-registered gates

All gates: fresh process, held-out documents, exact-graft and in-context runs as
the two baselines, bf16 noise floor as previously established by the equivalence
harness.

| Gate | Question | Measurement | Pass | Kill |
|---|---|---|---|---|
| **G0** Instrument check | Does the pair-minting pipeline produce valid exact grafts? | Minted exact grafts pass the existing equivalence harness | Logit-identical at noise floor | Any drift → fix pipeline before training |
| **G1** Latent error profile | How does prediction error distribute? | Per-layer normalized error, per arm | (descriptive — feeds H3) | — |
| **G2** Logit fidelity | How far is predicted from exact at the output? | Logit KL / top-1 agreement: predicted vs exact graft mounts, same seats | Pre-registered threshold set after G0 establishes noise floor; recorded before G2 runs | Top-1 agreement at chance → arm dead |
| **G3** Needle readback | Does content survive? | Verbatim needle probes: in-context vs exact vs predicted | Predicted within pre-registered Δ of exact | Collapse → predicted tier is router-fodder only (H2 strong form) |
| **G4** Router recall | Does addressability survive? | recall@1/@3 over a repository of predicted grafts vs same repository exact | Within Δ of exact recall | Below exact by > Δ → H1 falsified |
| **G5** Digest tolerance | Can the librarian eat lossy input? | Digests authored from predicted vs from exact grafts; probe accuracy through digests | Within noise of exact-sourced digests | Degradation compounds → predicted tier excluded from consolidation path |
| **G6** Tier integration | Does cold-predicted / hot-exact work end-to-end? | E4-style conversation gate: corpus ingested via student, promotion-on-access triggers exact re-harvest | Conversation gate parity with all-exact baseline | Failures trace to predicted tier → promotion policy tightened, gate rerun |

**Threshold discipline:** every Δ above is written down after G0 (which establishes
the instrument's noise floor) and before the gate it governs runs. No threshold is
set or adjusted after seeing the result it judges.

---

## 5. Amortization accounting (the economics gate, expressed as ratios — no time)

The student only pays if ingestion volume dwarfs training volume. Report:

- **T_train:** tokens prefilled through the target to mint training pairs.
- **C_student / C_target:** measured per-token compute cost ratio, student forward
  vs target prefill, same hardware.
- **Break-even ratio:** ingested tokens at which (T_train × target cost) is
  recovered, as a pure ratio. The repository's projected corpus sizes decide
  whether the ratio is reachable; the protocol only reports the number.

---

## 6. Phases

- **Phase 0 — Instruments.** Pair-minting pipeline + G0; VRAM fit probe (target
  INT4 + student + activations co-resident for the functional-loss phase — if
  they don't fit, functional loss runs in alternating swap mode, and that mode's
  correctness is itself verified); throughput probe sets ARM-S/ARM-T param dials;
  checkpoint/resume gate. *Exit: G0 green, fit mode chosen and verified,
  resume-from-kill green, all Δ thresholds for G2–G5 registered in writing.*
- **Phase 1 — Floor.** ARM-L trains and runs G1–G4. *Exit: floor recorded.*
- **Phase 2 — Students.** ARM-S and ARM-T train (warm → functional). G1–G4 per
  arm. *Exit: H1/H2/H3 each carry a verdict.*
- **Phase 3 — Integration.** Best arm feeds G5 and G6; amortization accounting
  reported. *Exit: predicted tier admitted to the repository, restricted to
  router-fodder, or rejected — with receipts.*
- **Phase 4 — Writeup.** Doors ledger, results doc in house format, raw transcripts
  and per-gate JSON retained.

---

## 7. Doors (declared now)

- **Distribution shift.** Student trained on one domain mix, repository ingests
  others. G2–G4 run on held-out *domains*, not just held-out documents; per-domain
  deltas reported separately.
- **Error compounding.** Predicted grafts feed real queries; small key errors
  shift attention nonlinearly. The functional loss targets this but the gates,
  not the loss, decide whether it worked.
- **Functional-loss cost.** The inner target-forward makes L-func expensive;
  subsampling may undertrain deep layers. Subsampling schedule is logged so the
  per-layer error profile (G1) can be checked against it.
- **Per-target student.** The student speaks one dialect; a target-model upgrade
  invalidates it along with the repository. The amortization section prices this
  honestly — the student is a depreciating asset on the same schedule as the
  repository itself.
- **Latent-space degeneracy.** Multiple latent values may decode to similar
  behavior (the up-projection's null directions). Huber loss penalizes
  differences the reader can't see; the functional phase exists precisely to
  stop optimizing those. If warm-phase error plateaus high but G2 passes anyway,
  that's the null directions showing themselves — record it, don't fight it.
- **Apophenia guard.** H1–H4 are written above, before any run. Results that
  confirm an unstated hypothesis get the skeptical treatment; results that
  falsify a stated one get reported with the same prominence as passes.

---

## 8. What exists if the gates go green

A student that writes the target's dialect at a fraction of the target's read
cost; a repository whose cold tier ingests at student prices and promotes to
exact on access; and a measured answer — the first anywhere — to what a forward
pass is worth: the fidelity gap between computed meaning and predicted meaning,
layer by layer, logit by logit, with addressability and content priced
separately.
