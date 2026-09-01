# LSR — Lived-Serving Reliability Program (PLAN, IMMUTABLE)

Opened 2026-09-01 (David: continue). Origin: the DET1 campaign's
census — 7/19 lived probes fail served controls with correct mounts;
6 serve confidently WRONG values on correct SINGLE mounts; displaced
values across supersession fixtures (meridian serves lumen's value;
orion serves its own superseded one). NEW defect class, distinct from
co-mount blending (which requires co-mounted siblings; these have
none). Census receipt: artifacts/grm_det1/.../det1_11/census/.

## Registered hypotheses (Phase 0 discriminates)

- **H-LSR-1 (prime): recent-turns window contamination.** The visible
  window = anchor + arena + RECENT TURNS + live. Multi-fixture lived
  sessions leave fixture N−1's values as plain text in recent turns
  at fixture N's probe; the model reads the displaced value from
  recent-turn tokens, not the (correct) arena. Predicts: wrong-value
  answers' attention mass concentrates on recent-turn tokens
  containing the served value; e2e-vs-supersession lawfulness split
  tracks recent-turn value density; the served wrong value appears
  verbatim in the window's recent-turns region.
- **H-LSR-2: repository route bleed** — despite mount receipts, some
  other graft's content reaches the window (would contradict the
  mount receipts; check anyway).
- **H-LSR-3: confabulation** — the value is nowhere in the window;
  the model invents it (predicts: served value absent from all window
  text/KV; mass diffuse).

## Phase 0 — where does the wrong value physically come from?

For each of the 6 wrong-value probes + 2 lawful same-session controls:
full-window witness (the DET instrument, scope widened beyond arena to
recent-turns and live regions) at answer-readout positions; per-region
mass split (arena / recent / anchor / live); locate the served value's
token occurrences across the window; report value-provenance verdict
per probe (RECENT-TURNS / ARENA / ABSENT). Adjudication: H-LSR-1
CONVICTED iff ≥4/6 wrong-value probes show the served value present in
recent turns AND plurality readout mass on those tokens. Registered
before any run.

## Phase 1 (entered only on conviction) — fix candidates, gated

Candidates (choose by Phase-0 receipts, gate semantics registered then):
recent-turns eviction/summarization for value-bearing turns already
deposited as grafts (the turn IS in the repo — carrying it verbatim in
the window is double residency); probe-ladder-style precise-first
discipline extended to the recent region; or window-layout change
(David-visible). Regression anchors: full supersession battery, E2E
sessions, census re-run (target: unlawful rate materially down, no
new failures).

## Evidence classes & roles

Inference measurement on one model (GPT-OSS lived sessions); census
numbers are the baseline. Fable = plan/verify/commits; Opus 5 seats =
implementation (Codex resting to ~Sep 7). Standing GPU discipline.
