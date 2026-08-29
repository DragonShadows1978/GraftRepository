# ORDER MOE-E3 — episodic consolidation on OLMoE: distill "the model has already read this"

## Grant

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/GraftRepository` — edits + CPU
runs AUTHORIZED (self-run CPU stages as in E2; OLMoE fits your sandbox).
No GPU: emit scripts for heavy stages, lead runs. READ-ONLY: the OLMoE
base snapshot and wikitext cache as in E2; the E2 negative-pool inputs;
NEW read grant: `/mnt/Shared/01 - Narrative Stories/` — but ONLY files
matching `*_COMPLETE.txt` (plain text novels), and NOTHING whose path
contains "RCFT", "Consciousness", or "Thesis". All moe_e2 artifacts
append-only. New code `scripts/olmoe_e3_*.py` (import the E2 harness
freely; modify nothing existing). Artifacts `artifacts/moe_e3/`.
FORBIDDEN: git, subagents, network.

## Context (one paragraph)

E2's verdict: the bolt-on pipeline is mechanism-GREEN end-to-end
(healthy testbed, bit-exact ABI, L10 install, 90.7% training-signal
capture), but the guides-as-teacher operationalization carried ≈ nil
signal (gap 0.10 ppl) and the faithful adapter amplified nothing into
harm (G4/G5 RED). E3 runs the §15-faithful experiment from the GRAPA
architecture note: consolidation of EPISODIC content — teacher = the
frozen base re-reading a document's own earlier text; expert = a
detachable module that carries that document so the model no longer
needs to re-read it. Here the teacher gap is real by construction
(same-model bring-up receipt: same-stream context lifts prediction by
−0.393 nats on wikitext; a NOVEL document should lift more).

## Corpus (registered)

- DOC-A (the episode to consolidate) and DOC-B (cross-probe): the two
  LARGEST eligible `*_COMPLETE.txt` novels by byte size (deterministic;
  record paths + sha256 + token counts). These are 2026-authored
  novels — outside OLMoE pretraining.
- Windowing per doc: sequential 512-token windows. DOC-A regions:
  PAIR region = windows 4..67 (64 pairs; skip the first 4 windows so
  every teacher prefix is a full 2,048 true preceding tokens);
  HELDOUT region = the last 16 windows of the doc (disjoint from PAIR;
  also ≥ window 4). Teacher arm for ANY window = its true preceding
  2,048 tokens (may overlap other regions — the teacher is allowed to
  re-read; that is the point). Student arm = window alone.
- Negatives for the key: wikitext + code + GRM docs + the E2 guides
  windows (four corpora, FIT/EVAL parity discipline as E2).

## Stages (reuse the E2 machinery/modes; content-addressed provenance)

1. `bringup` (inherit E2-G-1 receipt if content-valid; else re-run).
2. Key: capture router-input h on DOC-A windows; K4 LDA vs the pooled
   negatives; frozen fit-side τ. E3-G2 rule: recall ≥ 0.50 on DOC-A
   eval windows, code FPR ≤ 0.05, GRM FPR ≤ 0.05, guides FPR ≤ 0.10,
   wikitext fire DESCRIPTIVE; plus DESCRIPTIVE: fire rate on DOC-B
   (cross-episode selectivity — do not gate on it, report it).
3. Pairs at L\* from PAIR region (64 teacher/student pairs), r=64
   zero-init adapter, E3-G3 = val MSE beats zero-predictor ≥ 10%.
4. Behavioral eval on DOC-A HELDOUT windows:
   - PRECONDITION (registered): teacher gap on heldout ≥ 0.5 ppl
     (ppl_student − ppl_teacher). If smaller, STOP with "episodic
     teacher signal below registered floor" — that is a finding.
   - E3-G4: expert (mounted, gate live, NO context) recovers ≥ 25% of
     the teacher gap, bootstrap 95% CI low > 0.
   - E3-G5: wikitext ppl delta ≤ 0.5% with gate live; code fire
     ≤ 0.05; DESCRIPTIVE: DOC-B ppl delta and fire rate (an episodic
     expert should be near-silent on a different novel — report it).
5. `analyze`: report `artifacts/moe_e3/MOE_E3_REPORT.md`, ExpertPack
   `expertpack_docA_olmoe_v0/`, all tables with CIs.

## Constraints

Everything from E2 carries: bf16 only, content-addressed provenance,
idempotent prepare, fail-loud non-finite writer, GPU discipline in
emitted scripts (flock, ≤590 s, gaps), RED honesty, no post-hoc
widening. Adult content may appear in the novels — it is David's own
corpus; process it as data, excerpt nothing into reports beyond
provenance hashes and token counts.

## Done

Final message verbatim: E3-G-1/G2/G3/G4/G5 statuses with key numbers
(or NOT_MEASURED + lead script paths); DOC-A/DOC-B identities with
sha256 + token counts; the install row; the G4 row (ppl_student /
ppl_teacher / ppl_expert / recovery% + CI) once measured; the DOC-B
selectivity numbers; files created/modified; anything you could not do.
