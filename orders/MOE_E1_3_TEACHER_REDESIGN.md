# ORDER MOE-E1.3 — teacher-arm redesign: contiguity control + prefix-construction sweep

## Grant

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/GraftRepository` — edits and CPU
runs AUTHORIZED; no GPU in your sandbox (proven): build modes + emit the
GPU script, lead runs it. READ-ONLY as in `orders/MOE_E1_NARRATIVE_EXPERT.md`;
all moe_e1* artifacts append-only. New artifacts: `artifacts/moe_e1_3/`.
Code: extend `scripts/gpt_oss20b_e1_diag.py` / `_expert_e1.py` behind new
modes; original paths untouched. FORBIDDEN: git, subagents, network.

## Context (one paragraph)

E1.2-DIAG adjudicated no code defect (established path reproduces E1
teacher numbers exactly) — but the lead's follow-up on the receipts
localized the real mechanism: teacher degradation is CONTENT-DRIVEN by
prefix construction, not cognition-driven by guides. Severity tracks
prefix formatting density: w11 prefix = Active_Guides/07_ACCENT_REFERENCE
(phonetic table) → ppl 186,382; w13 = story-structure outline → 6,864;
w15 = prose-heavy guide → 98. Per-position NLL is flat-high with
degenerate top-1 ("…") — raw-concatenated reference-table prefixes push
the model into degenerate continuation. The registered E1 teacher
(2,048-token raw-file excerpts, seeded rotation) is therefore a broken
instrument; the adapter faithfully distilled a poisoned target, which
also explains the expert arm's uniform harm. E1.3 finds a teacher
construction with a genuinely positive gap, after one forward-sanity
control the diag lacked.

## Work (registered)

C0 contiguity control (forward sanity): base-arm ppl scoring the last
   511 targets of a CONTIGUOUS 2,560-token stream from one HELDOUT
   guide file (no splice). Sane result (same order of magnitude as the
   512-token base ppl on that file) proves scoring past position 2048
   is healthy; explosion here would reopen MECHANISM=BUG — report and
   stop.

P-sweep teacher constructions, each on heldout windows {11, 13, 15, 1}
   (worst→mild spread), gap = mean_nll_base − mean_nll_teacher per
   window (positive = prefix helps):
   - P1 prose-only: prefix drawn from TRAIN guide files, sentences
     only (strip headers/tables/lists; deterministic filter, document
     it), 2,048 tokens.
   - P2 framed: same P1 content wrapped in an instruction frame
     ("The following are writing-craft guidelines...\n\n<prefix>\n\n
     Passage:\n") — exact template recorded.
   - P3 same-file: prefix = the 2,048 tokens PRECEDING the window's
     own file content where available (contiguous same-document
     conditioning; falls back to same-folder file). Ceiling arm for
     what in-context conditioning can do.
   - P4 short-framed: P2 at 512 prefix tokens.
   Registered selection: the construction with the highest mean gap
   across the four windows, provided that mean gap > 0, becomes the
   E1.3 teacher. If NO construction achieves positive mean gap, the
   E1.1 premise finding STANDS for this model+corpus — state it in
   those words and stop.

## Gates

- CG0: C0 receipt (sane/explosion, numbers).
- CG1: 4 constructions × 4 windows = 16 teacher scores + gaps table.
- CG2: registered selection applied; winning construction named with
  its exact deterministic recipe (filter/template/seed), or the
  premise-stands sentence.
- CG3: report `artifacts/moe_e1_3/MOE_E1_3_REPORT.md`; if a winner
  exists, emit `GPU_E13_RESUME_COMMANDS.sh` that re-runs the E1.1
  chain from capture-pairs onward under the new teacher (append-only
  `_e13` receipts; G3/G4/G5' semantics UNCHANGED — only the teacher
  construction is re-registered by this order).

## Constraints

Standing GPU laws in the emitted script (flock, ≤590 s, sleeps, single
GPU). ~17 bounded GPU runs for the sweep + control. RED honesty.
Evidence class: behavioral inference measurement, one model, one
corpus. ## Done: CG verdicts, the 16-cell gap table verbatim, C0
numbers, winner + recipe (or premise-stands sentence), file list,
anything you could not do.
