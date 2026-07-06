# Open Question: Does pre-RoPE capture make TRANSLATION easier (not just injection)?

**Status:** PROMOTED TO ACTIVE PIVOT, ablation still untested. Registered
2026-07-05; promoted during Qwen3.5 R4.6 reliability planning.
**Origin:** David + Fable design decision (Fable's call, early KV-Graft):
capture grafts PRE-RoPE instead of post-RoPE.

## The established design fact — and WHY (David's actual rationale)
KV-Graft captures grafts PRE-RoPE. **The reason is POSITIONAL
PORTABILITY / RESEATABILITY (David's design hypothesis):** capture the
graft position-naive so it can be RESEATED at ANY position upon
injection. A post-RoPE graft is frozen at its mint-time positions; a
pre-RoPE graft carries no committed position, so the host applies RoPE
in its own live frame and the graft can be dropped anywhere in the
target context. (This ALSO sidesteps the KVLink/positional re-encoding
problem, arXiv 2502.16002, by construction — but that is a side effect,
NOT the reason. The reason is reseatability.)

## The design pivot

The production translator path now assumes pre-RoPE capture is not a mere
runtime convenience. It is the cleanest formulation of the translation problem:
map source-model pre-RoPE graft state into target-model pre-RoPE graft state,
then let the target runtime apply its own live positional frame.

This does **not** prove pre-RoPE capture is why the linear/residual maps work.
That remains an ablation question. But it does change implementation priority:
do not train translator objectives against post-RoPE targets unless explicitly
running the ablation below.

## Operational evidence from Qwen3.5 R4.6

The R4.6 residual/KV split sweep strengthened the production-side case for
staying in the pre-RoPE plane:

- The best current translator, `s0p5_kv`, maps source pre-RoPE graft state into
  target pre-RoPE graft state.
- Frozen V2 binding improved from the R4.5 residual baseline `53 / 64` to
  `63 / 64`.
- Fresh holdout improved from `54 / 64` to `58 / 64`, with no lost successes
  versus the R4.5 residual baseline.
- The diagnostic oracle over residual candidates reached `64 / 64` frozen V2.

This is not the post-RoPE ablation, so it does not close the research question.
It does justify treating pre-RoPE translation as the active implementation path
while keeping post-RoPE comparison as a deliberate ablation rather than the
default objective.

## The open ablation (David: "I do not actually know the answer")
Does pre-RoPE capture also make the TRANSLATION (2B->9B map) easier,
independent of the injection benefit?

Hypothesis: post-RoPE, the translator must learn a map through TWO
entangled positional encodings (source rotations + target rotations)
layered on top of the raw representation difference. Pre-RoPE, it learns
a clean representation->representation map with NO positional confound.
If true, this could be PART OF WHY the linear ridge maps work as well as
they do — where the information-theoretic corpus (Platonic /
rep-alignment / rate-distortion, GraftRepository/inv_388b7d90) expected
linear maps to saturate and require nonlinearity.

## The test (frozen before running, to avoid post-hoc rationalization)
Capture a matched small graft set BOTH ways (pre-RoPE and post-RoPE).
Train the SAME linear ridge translator on each. Compare on the frozen
V2 binding gate + per-layer key-recall / value-cosine.
- If pre-RoPE translates measurably better (not just injects better):
  real methodological finding — "pre-RoPE capture decouples the
  translation problem from the positional problem," a strength to state
  in the paper, and a partial explanation for linear-map success.
- If no translation difference: pre-RoPE is purely an injection
  convenience; the linear-map success is explained by something else
  (feature universality, the ceiling being higher than estimated).
- Either result is publishable and honest.

## Why it matters
This is the difference between "we capture pre-RoPE for clean injection"
(engineering note) and "pre-RoPE capture is why cross-model KV
translation is linearly tractable" (a claim). Only the experiment
decides which. Hold as a QUESTION until measured.
