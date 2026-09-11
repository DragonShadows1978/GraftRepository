# GRM-SCOUT-FIX-3 + C7 r2 — fold consolidation must generate through the Harmony wrapper with stop tokens; then fix the C7 harness and register r2 (worktree wt/grm-c7, branch grm-c7)

Origin: C7 amendment 2 diagnosis (committed 113d197,
`artifacts/grm_c7/diagnosis_lead_2/REPORT.md`). The core finding: the
librarian's fold/consolidation digest generation
(`core/graft_repository.py:3261` region) bypasses the Harmony chat
formatting and stop tokens the serving path uses
(`core/graft_arena.py:4348-4351` `harmony_turn`), and generates 120
tokens; on GPT-OSS this yields degenerate digests (ellipsis runs) and
every fold's coverage fails the 0.70 rule (r1: 13/13 folds rejected,
best coverage 0.375). This is the same class as B1/B3/FIX-2: a
production correctness gap, fixed minimally, pinned by a test that
fails before and passes after. Same rules as SCOUT-FIX-1/2 (no GPU,
no git, no subagents, foreground, never kill anything). Effort: high.

## Part 1 — SCOUT-FIX-3 (core, minimal)
1. Route digest generation through the same Harmony turn wrapper and
   stop-token set as serving (reuse `harmony_turn`; do not duplicate
   it), keeping the registered digest budget as a parameter (state the
   current 120 and whether the methodology doc registers a different
   number; do not change the number silently). Additive: a flag is NOT
   wanted here if the old behaviour is simply wrong on Harmony models;
   if a non-Harmony model path exists, keep it byte-identical and say
   how the dispatch is decided.
2. Pin: a CPU fake-model test that asserts the fold prompt is Harmony
   wrapped and generation stops at the stop token (RED before, GREEN
   after); the existing fold-coverage tests still pass; a digest QC
   rejecting degenerate text (ellipsis runs / repeated punctuation)
   before the coverage check, with its own test.
3. Audit any other generation site that bypasses the wrapper
   (era folds, digest re-folds, restore paths) and list rulings.

## Part 2 — C7 harness fixes (from your diagnosis)
- Admission: the uppercase `UNKNOWN` instruction word must not be an
  identifier candidate (lowercase the instruction in the C7 prompt, or
  exclude instruction tokens from the identifier scan in the C7
  harness — say which and why; if the right fix is in core
  admission, STOP and report as a further finding);
- oracle: apply the layer-position setup and record the wrapped token
  ids; add a CPU gate that the oracle answers from the live window on
  the fake model; the GPT-OSS `UNKNOWN` cause stays RED-unresolved and
  is measured again in r2;
- probe rows carry `question` and `expected`.
CPU gates: a fake-model session where the fixture identifiers MUST
mount and the oracle MUST answer.

## Part 3 — r2 registration
Sha-bound amendment binding the new core sha and harness fixes; same
fixtures/probes/oracle/39-cell layout; receipts under an r2 directory;
r1 receipts untouched; cap 2 GPU-h (lead authorization; total C7
3.9 GPU-h). `lead_commands_r2.txt`.

## Done (verbatim)
1. FIX-3 file/lines, test names, RED-before/GREEN-after evidence,
   site audit rulings.
2. Harness fixes file/lines + gate names; r2 amendment path + sha;
   exact lead commands.
3. Prior art; deviations; RED; process safety; model id and effort.
