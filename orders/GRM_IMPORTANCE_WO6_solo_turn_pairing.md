# WO-6: G1/G2 driver — handle solo scripted user turns (convo_02 leg failure)

YOUR WRITABLE TARGET is /mnt/ForgeRealm/GraftRepository/tests/test_grm_importance_g1g2.py
— edits AUTHORIZED on that single file. Everything else in the repository
is READ-ONLY reference. No git operations (the lead commits). No
subagents. No GPU runs, no model loads — CPU verification only.

## Context

GRM importance-weighting program (read docs/GRM_IMPORTANCE_PLAN.md and
docs/GRM_IMPORTANCE_LEDGER.md in full — plan and registered thresholds
are law). The G1/G2 driver in your writable target runs fixture
conversations (tests/fixtures/importance_convos/) through a live
GraftRepository. Legs 1,3,4,5,6 are GREEN with artifacts committed.
Leg 2 fails BEFORE running:

    AssertionError: turn 19 (user) not followed by an assistant turn —
    fixture pairing assumption violated at index 18

convo_02_travel_logistics.json turn 19 is a SOLO scripted user FILLER
turn ("Gate agent said the flight might be delayed...") immediately
followed by turn 20, a PROBE (user role). The driver's scripted-turn
walk assumes strict (user, assistant) pairs. Fixtures are
gate-registered data and MUST NOT be edited — the driver adapts.

## The fix

1. In the scripted-turn walk: when a user turn is NOT followed by an
   assistant turn (next is user/PROBE/EOF), deposit it as a SOLO user
   turn — same deposit path as pairs but with the assistant half absent
   (study how the repository's add_turn/feed composes
   "User: {u}\nAssistant: {a}\n" and produce the natural solo form
   "User: {u}\n"; if add_turn's signature cannot express a solo turn,
   report what you found and use the lowest-level public path that
   can, without touching core/).
   Solo turns MUST still deposit — they can be graded candidates in
   probe relevance maps (do not skip them).
2. The PROBE walk (probe user turn + following scripted assistant
   expected-answer turn, never deposited) is already correct — do not
   change its semantics.
3. CPU unit tests: extend the fixture-walk tests to cover solo user
   turns (followed by probe, followed by user, at EOF). The existing
   test_real_fixtures_satisfy_probe_answer_invariant walks all 6 real
   fixtures — extend or add a sibling test asserting the full walk of
   ALL SIX fixtures raises nothing and classifies every turn
   (pair / solo / probe / probe-answer), so any remaining unmodeled
   structure fails CPU tests, not a GPU leg.

## Rails

- RED honesty: if you find any other unmodeled fixture structure,
  handle it under the same classification test or, if ambiguous,
  STOP and report rather than guessing.
- Do not refactor unrelated driver code; minimal diff.
- Verify: python3 -m py_compile on the file; run
  `python3 -m pytest tests/test_grm_importance_g1g2.py -q` and paste
  the summary line. Do NOT execute --run-gpu.

## Done

Your final message MUST contain verbatim:
- The diff summary (functions touched, +/- lines).
- The pytest summary line (all tests, count must be >= 49 previous + new).
- Explicit confirmation: "fixtures untouched, core/ untouched, no git".
- Any other fixture-structure variants you discovered in the six files.
