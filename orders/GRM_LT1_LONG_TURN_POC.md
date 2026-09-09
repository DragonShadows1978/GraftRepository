# GRM-LT1 — a real 200-turn conversation as proof of concept (worktree wt/grm-lt1, branch grm-lt1, forked from grm-c7)

David (2026-09-09): "we need a real long turn test ... something like 200
turns ... proof of concept / real world example." This branch carries
SCOUT-FIX-1/2/3, the C2 profile registry, and the C7 harness with its
amendment-3 oracle fix. Reasoning effort: high. Prior Art Directive.

## What this is NOT
Not needle planting with synthetic identifiers. C7 r1 died on
`c7-fresh-N` ids colliding with the uppercase `UNKNOWN` instruction
word; C7 r2 stopped on the recency-exclusion rule when an alias probe
targeted a node still in the recency window (core rule, David's
decision, do not harness around it: keep every recall question's
source at least 10 turns back and never target a node that is a
recency nominee at probe time; assert that in the fixture gate).

## The conversation (fixture, frozen before any GPU)
A 200-turn working dialogue in natural prose between a user and the
assistant designing an expansion for a voxel space game (use the
domain vocabulary of `/mnt/ForgeRealm/Project-Frontier/docs` if it
exists; otherwise invent a coherent world; the model must not be able
to know the facts). Turn mix, registered:
- ~60 fact-bearing turns (station names, coordinates, resource counts,
  crew assignments, prices, dates) stated once in ordinary sentences;
- ~15 corrections ("actually make that 14, not 12"), some correcting
  an earlier correction;
- ~10 aliases ("call the Vega station 'the Hub' from now on");
- ~80 ordinary turns (discussion, digressions, small talk about the
  design) with no facts to recall;
- ~35 recall questions phrased the way a person would ("what did we
  settle on for the Hub's docking fee?"), at distances 10, 25, 50, 100,
  150 turns back, covering fresh facts, corrected facts (the CURRENT
  value is the answer), and aliases;
- two restarts (persist, new process, reload) at turns 70 and 140;
- turn 200: "recap the five biggest decisions we made" (scored by how
  many of the five registered decisions appear with their current
  values).
Answers are values a value-span match can score (C5 arm S rule:
contiguous ordered span after NFKC/dash/whitespace normalization);
report wrong-value and abstention errors separately, and refusal-style
texts separately again.

## Arms (both sides, same script)
- **A: C2 profile** (width 96, live capture pin, seat-near-live, RT1;
  EB1 frame; demand OFF);
- **B: shipped defaults** (width 256, pin OFF, near-live OFF).
Plus the **full-information oracle** per recall question (sources in
the live window) as the ceiling, with the amendment-3 attribute fix;
if GPT-OSS answers `unknown` with the fact in the prompt, record it,
do not "fix" it.

## Cells and budget
8-turn resumable leased cells per arm (the C7 machinery: bound session
state at cell boundaries, restart cells where registered), ≤ 285 s
worker / 590 s outer / 30 s cooldown; ~25 cells per arm. Budget
≤ 2.5 GPU-h total (lead authorization); if the projection exceeds it,
say so and register NON_FIT rather than trimming. Free-space preflight
≥ 20 GB.

## Registered predictions (yours to write before the run; the lead's
prior, for calibration only)
Arm A ≥ 80% exact on fresh and corrected recalls at every distance,
restart retained, residency bounded (≤ 2× arena + recency); arm B
loses corrected values on restart and degrades with distance.

## Deliverables
1. Fixture (`fixtures/lt1/`) with a manifest sha, the turn plan, and a
   CPU gate that proves: every recall question's source is ≥ 10 turns
   back and not a recency nominee at probe time (simulate the recency
   window from the plan); no probe wording contains an uppercase
   instruction token that the identifier scan would bind; the oracle
   prompt for each probe contains its sources; the fake-model session
   mounts and answers for the fixture (the r2/a3 parity fake).
2. Registration (immutable, sha-bound) with cells, estimates, budget,
   predictions, acceptance; `lead_commands.txt` (executable, resumable,
   stop-on-RED, per-arm score, summary with a per-distance table, the
   both-sides comparison, and the recap score).
3. CPU gates, blocked report, exact lead commands.
No GPU in the sandbox, no git, no subagents, foreground, never kill
anything.

## Done (verbatim)
1. Fixture manifest sha; turn-mix counts; gate test names and results.
2. Registration path + sha; cell list and projection; predictions.
3. Exact lead commands; prior art; deviations; RED; process safety;
   model id and effort.
