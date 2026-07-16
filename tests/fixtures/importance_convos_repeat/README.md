# importance_convos_repeat — GRM S4 repeat-probe fixtures

Static, CPU-validatable fixtures for G1-S4 in
`docs/GRM_S4_GROUNDING_LEDGER_PLAN.md`.  The JSON shape is the same as
`../importance_convos/README.md`: each file has `conversation_id`,
`description`, ordered `turns`, and `probes`; turn and probe objects use the
same keys, node classes, 0-3 relevance grades, and probe-to-scripted-answer
pairing convention.

This arm changes the experimental shape, not the schema:

- Four target fact nodes are each probed exactly twice.
- The late probe for a fact is at least eight fixture turns after its early
  probe.  The G1-S4 driver freezes every candidate's `n_grounded` value just
  after the early probe and compares those frozen values with S3 at the late
  probe.
- Every probe grades the same six pre-probe candidates: four target facts and
  two never-targeted controls.  Controls receive non-top 0-2 grades but are
  never the unique grade-3 answer to a probe.
- Every PROBE user turn is immediately followed by one scripted assistant
  answer.  Probe/answer pairs are reference data only and are never deposited
  by the driver.
- `convo_02_lab_handoff.json` deliberately includes four consecutive SOLO
  user turns.  The production driver classifies and deposits them as natural
  `User: ...\n` nodes, preserving coverage of the predecessor's SOLO path.

Facts follow the repository house style: identifier-shaped plantables mixed
with relational facts.  Identifier tokens are collision-checked across all
four files by the validator.

## Validation

The validator is stdlib-only and performs no model or engine import:

```bash
python3 tests/fixtures/importance_convos_repeat/validate_fixtures.py \
  tests/fixtures/importance_convos_repeat
```

It checks the inherited schema, graded candidate maps, scripted-answer shape,
four two-probe fact pairs with the eight-turn minimum gap, never-probed graded
controls, consistent candidate sets, cross-file identifier uniqueness, and
directory-level SOLO coverage.
