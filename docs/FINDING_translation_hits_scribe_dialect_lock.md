# FINDING: Translation binding is the SCRIBE dialect lock, not an optimization gap

**Registered:** 2026-07-06 (David's empirical result + SCRIBE mapping).
**Status:** load-bearing reinterpretation of the Qwen3.5 2B->9B translation track.

## The empirical result (David)
A 2B graft run through the translator DOES NOT transfer extractable data to
the 9B: when the 9B tries to EXTRACT / read the fact out of the translated
graft, the data is not there. Transfer fails at extraction.

## Why this coexists with the 63/64 binding gate (mechanism, code-confirmed)
The binding gate (core/qwen35_translation_poc.py, gold-vs-decoy) is a
SCORING gate: it ranks candidate answers by logprob and checks the gold-
minus-best-decoy MARGIN. It never asks the 9B to GENERATE/extract the fact.
Transported keys (key recall ~0.65) can tilt a 4-way logit margin toward
the gold answer WITHOUT the bound content being readable. So:
- 63/64 gate PASS  = routing/association survived (margin moved)
- extraction FAIL   = the bound content requiring the native dialect did not
These are the SAME divergence SCRIBE measured: "content unreadable while
logits 80% right" (H2). The gate measured the wrong axis.

## The SCRIBE connection (this is the same wall, not an analogy)
SCRIBE (project_kv_graft.md, CLOSED 2026-06-11, premise refuted) proved by
composition argument + 3-regime G3 0/10: "attention is the only cross-
position channel; deep latents = transcripts of the computation; a shallow
student can't flatten the circuit -- texture transfers, bindings don't. The
process IS the product." A linear/residual 2B->9B translator is the SAME
CLASS of object SCRIBE refuted: a shallow map trying to reproduce another
model's contextualization. The GraftRepository loader already enforces a
"DIALECT WALL" guard at load; SCRIBE proved why it can't be crossed shallowly.
The dialect is HARD-LOCKED to the model that produced it.

## Consequences
- The 63/64 / R4.6 oracle 64/64 do NOT indicate "one refinement from solved."
  They indicate the margin proxy is saturating while extraction stays at floor.
- Codex's "easily refined" optimism applies to the MARGIN gate, not to
  extraction. Refining the margin further does not cross the dialect lock.

## The deciding test (do this before any more margin optimization)
Add an EXTRACTION gate distinct from the margin gate: mount the translated
graft, have the 9B GENERATE the answer (teacher-forced readback or free
generation), measure verbatim/content recall vs an amnesia floor. Predict
(SCRIBE): extraction stays at/near floor even where the margin gate passes.
If extraction is at floor -> dialect lock confirmed; translation of
extractable content via shallow maps is refuted, same as SCRIBE.
If extraction transfers -> the lock does NOT apply here and the margin gate
was merely under-powered; a genuinely new result. Either way the EXTRACTION
gate, not the margin gate, is the real metric from here.

## The refit (SCRIBE's own conclusion, applies here)
SCRIBE closed with: "graft-native = reads its own dialect from birth" ->
GRAPA. If translation-of-dialect is hard-locked, the durable path is not a
better 2B->9B map; it is grafts minted and read within a shared dialect by
construction. Keys/routing DO transport (that survives) -- so cross-model
ROUTING/addressing may be viable even where cross-model CONTENT extraction
is not. That distinction is the salvage.
