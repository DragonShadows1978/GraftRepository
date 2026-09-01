# ORDER GRM-DET1.10 — separator normalization; reserve-selection diagnosis

WRITABLE TARGET `/mnt/ForgeRealm/GraftRepository` — edits/builds/CPU runs
AUTHORIZED. Rules as DET1.x: no git (lead commits), no subagents, no
network, no GPU (CPU self-runs fine; the lead runs
`./scripts/grm_det1_5_lead.sh`). Append-only artifacts. RED honesty.

State (campaign r5, `logs/grm_det1_5_gpu_lead_r5.log`) fail-closed:
`no unused lawful certified campaign-session reserve for
e2e_t30_atlas_tone: primary_reason='LIVED_SERVED_CONTROL_INCORRECT_OR_REFUSAL'`.

Two threads, kept separate.

## 1. SEPARATOR NORMALIZATION (primary fix)

t30's lived served control answered `Cobalt 1 India` against expected
`Cobalt-1-India` — the correct VALUE, spaces where the fixture writes
hyphens. The house has adjudicated this class twice already:

- ADM2.2 — orion value-assertion.
- DET1.4 — U+2010/U+2011 registered as glyph presentation, not value
  difference (`normalize_value_text`).

Comparison is value-semantics, not glyphs. Extend the whole-value
comparator so separator variants — ASCII hyphen `-`, U+2010, U+2011,
and space — are equivalent WITHIN a value token sequence, remaining
case-insensitive as already registered. Apply it consistently at EVERY
site where the DET campaign compares expected values: served controls,
planted-miss verification, and reserve lawfulness.

Guard (mandatory): normalization must NOT let a WRONG value pass. Add
explicit negative tests — `Cobalt-2-India` (wrong token) and
`Cobalt India` (missing token) must still FAIL against
`Cobalt-1-India`. Token identity and token count are load-bearing; only
the separator glyph is free.

## 2. RESERVE SELECTION DIAGNOSIS

All four `sup_reserve_*` lived snapshots were collected (manifests under
`det1_4/campaign/det1_7/plant_registration/shards/*/attempt_002/`), yet
none was selectable for the t30 slot. Determine why — family/split
scoping in the frozen rule
`DET1_9_PRIMARY_THEN_CANONICAL_UNUSED_LAWFUL_RESERVE_V1`, consumption by
t33, or their own lawfulness checks.

If the frozen selection rule itself over-constrains (e.g. family
matching making sup reserves unusable for e2e slots — which would make
DET1.9's widening vacuous for exactly the slot that motivated it),
report it plainly and implement the minimal correction AS an explicit
amendment note in the registry policy record. DET1.9's order language
governs: the widening's intent was cross-session substitution for e2e
slots; if the rule text contradicts that intent, the intent wins.

## Done

Final message carries receipts, verbatim:

- Thread 2's diagnosis: the exact reason, per reserve.
- The normalization diff summary.
- Negative-test receipts.
- Test counts (GRM modules touched + campaign/analyzer selftests).
- Files modified.
- Anything not done.
