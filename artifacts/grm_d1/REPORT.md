# GRM-D1 — LT1 residual diagnosis + recap battery redesign

Seat: Opus 5 (`claude-opus-5[1m]`), effort MAX. Worktree
`/mnt/ForgeRealm/wt/grm-d1` (branch `grm-d1`). `core/` untouched.
Date: 2026-09-10.

## Verdict in one line

All 11 arm-A wrong rows are **structural, not memory-quality**: 5 are a
fixture-lineage trap the harness created (corrections fed as ordinary
prose, nothing retired), 5 are a core-composition gap (no alias->base
binding exists in core at all), and 1 is a probe-ladder drop in which
A-DEC ranked the correct node **first** and the ladder threw the mount
away. The recap 0/5 is likewise structural: the question carried no
identifier token, so routing had nothing to key on and every ladder trip
returned ungrounded.

## 1. The cause table

Full table: `artifacts/grm_d1/cause_table_A.md` and `cause_table_A.json`.
Arm B contrast: `cause_table_B.md` / `.json`.

Class counts, arm A (11 rows):

| class | rows | ownership |
|---|---|---|
| alias-edge-without-base (core composition) | 5 | **STOPPED — core, A1 owns it** |
| fixture-lineage (harness) | 5 | **FIXED here** (LT1.1) |
| stale-first ranking (core A-DEC) [arm-A instance is ladder-drop, not mis-rank] | 1 | **STOPPED — core, reported below** |
| reader wrong value with the right mount | 0 | — |
| other | 0 | — |

Arm B (30 rows, for contrast): alias-edge-without-base (core composition) 10, fixture-lineage (harness) 5, reader wrong value with the right mount 12, stale-first ranking (core A-DEC) [arm-A instance is ladder-drop, not mis-rank] 3.
Arm B's extra `reader wrong value with the right mount` class is absent
from arm A entirely — a reader/geometry difference (w=256, defaults),
not a lineage one. That is evidence *for* the profile, and means the
fixture fix is not what separates the arms.

### The five correction rows — `recall_4_{10,25,50,100,150}`

Every one is the Vega docking-fee lineage: turn 1 = `17 credits`,
turn 3 = `23 credits`, turn 15 = `29 credits` (expected). The receipt is
identical at all five distances:

```
ranking_ids                     [0, 2, 14]   # = turns [1, 3, 15]
admission_identified_candidates [0, 2, 14]
admission_rank_plan             [0, 2, 14]
admission_route_margin_1_2      0.0005005675763460893   # threshold 0.1385774091529802
admission_policy_branch         margin_insurance_k3_identifier_tiebreak
fit_seated                      [0]   # = turn 1, '17 credits'
```

Served answer at all five distances: *'We agreed that Vega’s docking fee would be 17 credits.'*

**Why this is the harness, not core.** Core excludes retired nodes from
the route-candidate base outright:

```python
# core/graft_arena.py:1450-1454  (_route_cand_base)
base = [i for i, g in enumerate(self.grafts)
        if not g.get("retired")
        and g.get("kind", "turn") != "recall"]
```

Had the correction been a real supersession, nodes 0 and 2 would carry
`retired=True` and would **never have entered `ranking_ids` at all**.
Instead the LT1 worker feeds every turn identically:

```python
# scripts/grm_lt1_worker.py:146-148
# User corrections are ordinary prose, not hidden supersede calls.
idx=a.feed(e2e.harmony_turn(event['user'],event['assistant']))
a.grafts[idx]['kind']='turn';state['turn_nodes'][str(turn)]=idx
```

So the whole family stays active, the three near-identical scores tie
(margin 0.0005 << threshold 0.1386), and the k3 tiebreak preserves the
**deposit order** — oldest first.

Contrast that proves the mechanism is order, not "corrections are hard":
`recall_5_*` (Medibay, the *same* three-step correction shape) ranked
`[15, 3, 1]` = turns 16, 4, 2 — **newest first** — seated node 15
(`16 beds`) and scored **correct at all five distances**. Same policy
branch, same margin magnitude, opposite order, opposite outcome. The
ranking order within an un-retired family is arbitrary; the fixture,
not the policy, decides whether you win.

### The five alias rows — `recall_7_{10,25,50,100,150}`

Beacon/Lantern launch date. Receipt at d=10:

```
admission_identified_candidates [17]        # = turn 18, the ALIAS edge
admission_rank_plan             [5]         # = turn 6, the BASE fact
fit_seated                      [5]
identifier_tokens               ['beacon']
```

The identifier **did** find the alias edge. The mount seated only the
base fact. Nothing composes the two, so the reader receives
*"Lantern's launch date will be 18 October 2196"* while being asked
about "the Beacon", and answers *'We’re still working on that.'*.

**Why this is core, and STOPPED.** There is no alias mechanism in core:

```
$ grep -rn 'alias' core/graft_arena.py
core/graft_arena.py:1946:  # spans verbatim (including corrections and alias edges).   <- a comment
$ grep -rn 'alias' core/graft_repository.py
core/graft_repository.py:4767: """Compatibility alias for the old persistence entry point."""  <- unrelated
$ grep -rn 'GRM_ALIAS_FOLD_MERGE' core/ scripts/ tests/
(no match)
```

One comment and one unrelated docstring. The named optional flag
`GRM_ALIAS_FOLD_MERGE` does **not exist on this tree** — recorded in the
LT1.1 registration as `present_on_tree: false`, never silently claimed.

Note `recall_6_*` (Hauler/Kestrel) has the *identical* receipt shape
(identified `[16]`, rank_plan `[4]`, base-only mount) and scores
**correct** at all five distances. So the base-only mount is not
deterministically fatal — whether the reader can bridge the name gap
unaided is luck. The composition gap is real; its expression is
stochastic.

### The one fresh row — `recall_3_150`

Breakwater map position, `(-31, 48, 12)`, asked at turn 164 about turn 14.

```
ranking_ids              [13, 108, 80, 122, 37, 94]   # node 13 = turn 14 = CORRECT
admission_rank_plan      [13]
fit_seated               [122]                        # = turn 151
served_without_plan_head True
trip 0  planned=[13]           mount_set=[13]     grounded=False
trip 1  planned=[122, 37, 94]  mount_set=[122]    grounded=True
```

**A-DEC ranked the correct node first and made it the entire rank plan.**
The ladder's first trip mounted it, came back `grounded: false`, and a
second trip seated a *different* entity's map position, which read as
grounded. The answer was generic filler.

I keep the order's class label `stale-first ranking (core A-DEC)` so the
lead can map classes 1:1, but the receipt does not support "stale-first
ranking" for this row: the ranking was correct. The honest name is
**probe-ladder drop of a correctly-ranked plan head**, which is why the
class string carries that qualifier. This is core, and STOPPED.

The four correct `recall_3` rows at d=10/25/50/100 all ran two trips in
which **both** trips read `grounded: false`, and all four kept node 13
seated and answered correctly:

```
recall_3_10   correct  trips=2 seated [13]  grounded [False, False]
recall_3_25   correct  trips=2 seated [13]  grounded [False, False]
recall_3_50   correct  trips=2 seated [13]  grounded [False, False]
recall_3_100  correct  trips=2 seated [13]  grounded [False, False]
recall_3_150  WRONG    trips=2 seated [122] grounded [False, True]
```

So the displacement is not caused by trip 0 being ungrounded — that
happens in all five. It is caused by trip 1 *becoming* grounded on an
unrelated node at d=150 and being preferred over the plan head. The
grounding signal, not the ladder depth, is what changed. One row is not
enough to rule on the mechanism; registered as an observation, not a
finding.

## 2. What I changed, and what I stopped on

### FIXED (harness/fixture only — zero core edits)

| file | what |
|---|---|
| `scripts/grm_d1_supersession.py` | the two correction arms on CPU: `feed_only_arm` (LT1 lineage) vs `supersede_arm` (production path) |
| `scripts/grm_d1_recap.py` | the five-question recap battery, its two fixture gates and the oracle gate |
| `scripts/grm_d1_register_lt11.py` | the LT1.1 fixture amendment and immutable registration |
| `scripts/grm_d1_cause_table.py` | the per-row diagnosis |
| `scripts/grm_d1_emit_report.py` | this report and the ledger |
| `tests/test_grm_d1_alias_cpu.py` | 14 tests |
| `tests/test_grm_d1_amendment1.py` | 17 tests |
| `tests/test_grm_d1_amendment2.py` | 18 tests |
| `tests/test_grm_d1_amendment3.py` | 18 tests |
| `tests/test_grm_d1_cause_table.py` | 12 tests |
| `tests/test_grm_d1_recap.py` | 10 tests |
| `tests/test_grm_d1_registration.py` | 18 tests |
| `tests/test_grm_d1_supersession_fixture.py` | 4 tests |

**RED -> GREEN proof for the supersession fix.** Both arms run the SAME
three LT1 turns (Vega 17 -> 23 -> 29) through the same production code,
differing only in lineage. Receipt:
`artifacts/grm_d1/supersession_cpu_receipt.json`.

```
RED  (feed_only — LT1's actual lineage):
  retired_nodes    : []
  route_cand_base  : [0, 1, 2]
  ranking          : [0, 1, 2]        <- node 0 = "17 credits" ranks FIRST
  stale reachable  : ['17 credits', '23 credits']

GREEN (supersede — the C7 r3 shape):
  retired_nodes    : [0, 1]
  route_cand_base  : [2]              <- stale nodes GONE from the base
  ranking          : [2]              <- only the current value routable
  stale reachable  : []
  superseded_by    : {"0": [1], "1": [2]}
```

That is the whole fix: the stale node is not out-ranked, it is **not a
candidate**. `test_the_supersede_assertions_actually_fail_on_the_old_
lineage` runs all three GREEN assertions against the RED arm and
requires each to fail, so the gate is proven capable of going red.

**RED -> GREEN proof for the recap battery.** The old question fails the
new gate outright (`test_instruction_identifier_gate_can_go_red`:
`RECAP_INSTRUCTION_IDENTIFIER_COLLISION` on *"Recap the five biggest
decisions we made"*), and the oracle gate goes RED on a deliberately
unanswerable question. The new battery scores **oracle 5/5 PASS**.

### STOPPED (core — proposed fix, NOT implemented)

#### Proposed core fix — alias->base composition (5 rows, A1 target)

**Receipt of the gap:** `admission_identified_candidates=[17]` (the
alias edge) but `admission_rank_plan=[5]` (the base fact), and
`grep -rn 'alias' core/` returns one comment and one unrelated docstring.

**Proposed shape (for A1 to judge, not a patch):** the alias edge is
already a first-class node; what is missing is that matching it should
*pull its base in with it* rather than being replaced by the base.
Concretely, an alias node would carry a lineage pointer — the same
metadata slot supersession already uses (`metadata.supersedes` /
`superseded_by`, core/graft_repository.py:1953-1998) — and the admission
plan would treat an alias hit as a **two-node co-mount requirement**
rather than a one-node substitution: if node *a* is an alias edge naming
entity *E*, and the plan would seat *E*'s base fact *b*, seat `{a, b}`
together so the reader sees both *"Let's call Lantern 'the Beacon'"*
and *"Lantern's launch date will be 18 October 2196"* in one mount. The
existing multi-node plan machinery already supports this (`recall_5_10`
seats a 3-node `rank_plan`), so the change is to *plan construction*,
not the mount path.

**Why I would not implement it even if core were writable:**
`recall_6_*` scores correct with the base-only mount at all five
distances, so a co-mount is not obviously necessary and may cost width.
That is an A1 decision, on evidence, not a D1 one.

#### Reported core observation — probe-ladder drop (1 row)

`recall_3_150`: the ladder discarded a correctly-ranked, correctly-
planned mount because trip 0 read `grounded: false`, then accepted a
*different entity's* node from trip 1 because it read `grounded: true`.
The grounding check appears to test "did the mount produce a groundable
answer" rather than "did the mount produce an answer grounded **in the
planned node**". No fix proposed — I have one row, and four
counterexamples at shorter distances (d=10/25/50/100) whose trip 0 was
equally ungrounded and which scored correct. Registered as an
observation for the lead, not a finding.

## 3. LT1.1

- registration: `artifacts/grm_d1/lt1_1/registration.json`
  sha256 `02b44d02e3fbb82ca9809c4a5e08597cf2d195e5a6fe4df8e4e81f3f760d80e8`
- sidecar: `artifacts/grm_d1/lt1_1/registration.sha256`
- fixture: `artifacts/grm_d1/lt1_1/dialogue.json`
  sha256 `470da44b3a570684bd24c9f3b96f434468d7914094222ed5867c1382fbc94d26`
  (parent = LT1 `ad61c9488f722ae7ce248aee34cda688c13a5f0d3ec47b62954da2a3c146855c`)
- commands: `artifacts/grm_d1/lt1_1/lead_commands.txt`
- shape: arm A only, 26 cells, same frozen conversation, same 52-cell
  schedule, margin_first + profile. Budget **6120 s = 1.70 GPU-h**.
- turn mix: fact 60, supersede 15, alias 10, ordinary 75, probe 35,
  recap_probe 5 = 200. All 200 user/assistant prose strings, all 35
  recall probes, all distances and all 5 decisions are byte-identical to
  LT1 (`test_the_user_prose_is_byte_identical_to_lt1`).

**Registered predictions (before the run, per house rules):**
- aliases: 0/5 correct (unchanged) -- core gap, not addressed
- corrections: 5/5 correct (was 0/5) -- the supersession fix retires the stale nodes so they cannot be ranked
- evidence_class: prediction registered BEFORE the run, per house rules
- fresh: 6/7 unchanged; recall_3_150 is a ladder-drop, not addressed by either change
- recap: 5/5 on the oracle (gated on CPU); memory side unpredicted

**The optional alias flag.** `GRM_ALIAS_FOLD_MERGE` is **absent from this tree**. It is
registered as `present_on_tree: false` with the grep as evidence and
`disposition: "NOT pinned"`, and
`test_alias_flag_really_is_absent_from_this_tree` re-verifies that claim
against the source. The lead can pin it if a later merge introduces it;
LT1.1 must not claim it was applied.

### 3.1 LT1.1 r2 results (lead-run on the shipping core, amendment 9)

Receipts: `artifacts/grm_d1/lt1_1/lead_Aplus_r2_summary.json`,
`lead_A_r2_summary.json`; per-row causes in
`artifacts/grm_d1/lt1_1/miss_causes.json`. Both arms **26/26 cells COMPLETE**.
Binding in every cell carries amendment 8's rebound core shas, so these
numbers are measured on the core that ships.

**Two columns.** Column 1 is the registered scorer (`scripts/grm_lt1.score`),
which stays the **primary verdict** and is unmodified. Column 2 is
`value_span_glyph_tolerant_v2` (`scripts/grm_d1_scorer_v2.py`), registered and
sha-bound in amendment 9, applied identically to all three row sets. Column 2
is a verified **strict superset**: 0 rows pass column 1 and fail column 2.

| class | A+ col1 | A+ col2 | A col1 | A col2 | LT1 parent col1 | LT1 parent col2 |
|---|---|---|---|---|---|---|
| fresh | 8/15 | **10/15** | 10/15 | **13/15** | 16/17 | 16/17 |
| correction | 10/10 | 10/10 | 9/10 | 9/10 | 5/10 | 5/10 |
| alias | 10/10 | 10/10 | 10/10 | 10/10 | 5/10 | 5/10 |
| recap | 4/5 | 4/5 | 4/5 | 4/5 | n/a | n/a |
| **total** | **32/40** | **34/40** | **33/40** | **36/40** | 26/37 | 26/37 |

LT1 parent rows were rescored by this seat from
`artifacts/grm_lt1/amendment2/run_margin_first/cells/A-*/probes.jsonl` under
the LT1.1 class map. Corrections 5/10 and aliases 5/10 reproduce the lead's
figures exactly; the parent fixture holds 17 fresh rows and no recap probes,
so the lead's 14/15 fresh and 0/5 recap are **not** reproduced here and stand
as the campaign record for those two cells.

**What column 2 changes.** It moves **5 rows, all in LT1.1** (A+:
`recall_1_10`, `recall_3_100`; A: `recall_1_10`, `recall_1_25`,
`recall_1_50`) and **0 rows in the LT1 parent** — so it does not flatter the
new arms against the baseline. It changes no class verdict but fresh, and it
does not change the A-vs-A+ ordering.

**Correction to the "U+2011" framing.** For the *dash* this is already false:
`grm_lt1.normalize` NFKC-folds **and** maps U+2010..U+2014/U+2212 to `-`
before the span search. `9‑voxel` therefore arrives as `9-voxel`. The miss is
`9-voxel` vs expected `9 voxels` — hyphen-for-space plus singular/plural. The
genuinely unfolded glyph is **U+202F** (narrow no-break space) inside
`-31, 48, 12`. Column 2 folds U+202F/U+00A0, optional parentheses, and the
unit plural; it never rescues a wrong number, a fabricated entity, or an
abstention.

### 3.2 Per-row cause table (every miss, both arms)

| arm | probe | class | cause | served node actually mounted |
|---|---|---|---|---|
| A+ | `recall_1_10` | fresh | glyph/scorer miss, **content correct** | turn 24 (salvage loop) |
| A+ | `recall_1_25` | fresh | fabrication ("12 m lantern spacing") | turns 13, 29 |
| A+ | `recall_1_50` | fresh | fabrication ("12 m lantern spacing") | turns 13, 29 |
| A+ | `recall_2_100` | fresh | fabrication (Commtower crew = "local kids") | turn 65 (Morrow→"the Gate") |
| A+ | `recall_3_100` | fresh | glyph/scorer miss, **content correct** | turn 65 |
| A+ | `recall_2_150` | fresh | fabrication (same crew invention) | turn 65 |
| A+ | `recall_3_150` | fresh | fabrication `(0, 0, 0)` | turn 153 (Nacre ticket price) |
| A+ | `recap_3` | recap | fabrication `(-9, -44, 71)` | turn 153 |
| A | `recall_1_10` | fresh | glyph/scorer miss, **content correct** | turn 20 (comm tower silhouette) |
| A | `recall_1_25` | fresh | glyph/scorer miss, **content correct** | turn 20 |
| A | `recall_5_25` | correction | **abstention** ("still working on that") | turn 20, fact 28 |
| A | `recall_1_50` | fresh | glyph/scorer miss, **content correct** | turn 20 |
| A | `recall_3_100` | fresh | fabrication `(64, -17, 8)` | turn 42 (Saffron galley stock) |
| A | `recall_3_150` | fresh | fabrication `(-9, -44, 71)` | turn 141 (Fallow steward) |
| A | `recap_3` | recap | fabrication `(19, 83, -22)` | turn 125 (Crag tug price) |

`recall_5_25` asserts no value; the registered scorer's abstention regex does
not cover "we're still working on that", so it is counted `wrong_value`. The
registered category is left exactly as measured — the reclassification lives
only in `miss_causes.json`.

**Source nodes are never mounted — for hits either.** Across all 80 rows the
probe's own `source_ids` appear in `route_info.mounts` in **0 of 28** A+ hits
and **0 of 29** A hits. Correct answers are served through fold digests and
era nodes, never by mounting the original turn. "Source node not mounted"
therefore explains nothing on its own: it is the normal path for a correct
answer too.

### 3.3 Alias attribution — did A1's merge do anything measurable?

**Yes, structurally — but not on the alias column, and not in the direction
hoped for.**

- Alias probes score **10/10 in both arms**. On that column A1 changed nothing.
- Arm A mounted only **turn** nodes (13, 62) across its ten alias probes — *not*
  a librarian chronicle digest. A+ mounted turns 13, 19, 21, 56. Neither arm
  needed a fold: both answer by reading the alias turn itself.
- The flag was genuinely live: `binding.alias_fold_merge` is `true` in every
  A+ cell, `false` in every A cell.
- The stores diverge: **A+ 209 nodes (38 digests, 11 eras)** vs
  **A 199 nodes (31 digests, 8 eras)**.
- **Receipt gap:** `repository.alias_fold_history` is never persisted into
  `worker.json`, `probes.jsonl` or the manifest, so the GPU receipts cannot
  itemize fold decisions. The only itemization is the CPU replay
  `artifacts/grm_d1/alias_cpu_aplus.json`: **A+ 10 merges, A 0**, over the same
  164 turns — the Hauler=Kestrel, **the Beacon=Lantern**, the Forge=Foundry,
  the Cistern=Orchard, the Lookout=Aster, the Workshop=Tern, the Gate=Morrow,
  the Needle=Spindle, the Pit=Quarry, the Kitchen=Saffron.

**The Beacon capture — the substantive finding.** In **arm A**, digest node 24
folds sources `[13, 14, 17, 18]`, which *include* turn 18 ("Let's call Lantern
'the Beacon'"), and reads:

> the maintenance crew of **the Beacon** is located at Iona Vale, and the map
> position of **the Beacon** is (-31, 48, 12).

The fold rebound Commtower's crew and Breakwater's coordinates onto the wrong
entity — an alias for *Lantern*. Era node 61 inherits the error. In **arm A+**,
digest 28 folds `[13, 14, 21, 22]` (no alias turn) and reads "the maintenance
crew of **the comm tower** … the map position of **the breakwater**", and era
64 preserves it. Consequence: grepping the stores for "Breakwater" finds
**3 nodes in A+** and **1 in A** (the retired source turn alone). A+ holds an
intact retrievable record; A does not.

**Why both arms still miss Breakwater anyway.** Holding the record is not
serving it. In every Breakwater miss, in both arms, the mounted node is
unrelated to the question — Nacre's ticket price, Saffron's galley stock,
Crag's tug price. The correct digest/era is never ranked. This is a **routing**
failure, not a storage failure, and it is the top residual.

### 3.4 The fresh regression

A+ is **worse** on fresh than A: 8/15 vs 10/15 (10 vs 13 on column 2). The
separating rows are `recall_1_25`/`recall_1_50` (A+ fabricates "12 m lantern
spacing"; A answers "9-voxel intervals", correct content) and
`recall_2_100`/`recall_2_150` (A+ fabricates a Commtower crew).

This is **not** prose-feed vs production funnel — that difference was retired
in amendment 7, and both arms here run the same worker through the same
funnel. The remaining difference between the arms *is* the alias fold, which
enlarged the digest/era population (38/11 vs 31/8) and so changed what the
ranker had to choose among. **Fresh-recall regression under a larger digest
population is the honest reading, and it is a negative result for A1 on this
axis** — recorded as such, not smoothed.

### 3.5 Residuals

1. **Breakwater routing** — the correct digest/era exists in A+ and is never
   ranked. Top residual.
2. **Arm A fold provenance** — "the Beacon" captures Commtower/Breakwater
   facts. A correctness defect in the A arm, found here.
3. **`alias_fold_history` is not persisted** to campaign receipts; only the
   CPU replay itemizes folds.
4. **Abstention regex gap** — "we're still working on that" is scored
   `wrong_value`.

## 4. The five recap questions

Receipt: `artifacts/grm_d1/recap_battery.json`. Oracle **5/5 PASS**.

| id | turn | dist | question | expected span | decision targeted |
|---|---|---|---|---|---|
| `recap_1` | 196 | 184 | For Promenade, which lamp spacing did we go with? | `9 voxels` | the lamp spacing we chose for Promenade (turns 12) |
| `recap_2` | 197 | 184 | For Commtower, which maintenance crew did we go with? | `Iona Vale` | the maintenance crew we chose for Commtower (turns 13) |
| `recap_3` | 198 | 184 | For Breakwater, which map position did we go with? | `(-31, 48, 12)` | the map position we chose for Breakwater (turns 14) |
| `recap_4` | 199 | 184 | For Vega, which docking fee did we go with? | `29 credits` | the docking fee we chose for Vega (turns 1, 3, 15) |
| `recap_5` | 200 | 184 | For Medibay, which bed count did we go with? | `16 beds` | the bed count we chose for Medibay (turns 2, 4, 16) |

Gate rules kept: minimum source distance 184 turns (window is 10); no
instruction-word identifier in any question; scored with the existing
C5 arm S value-span scorer, unchanged. `recap_4` and `recap_5`
deliberately target the two correction lineages, so the recap battery
independently re-tests the supersession fix.

**Why the old one was 0/5.** `recap.json` shows
`admission_identified_candidates: []` — "recap / five / biggest /
decisions / made" are all instruction or stop words, so routing had no
key, ranked arbitrary nodes, every one of four ladder trips returned
`grounded: false`, and the reader free-associated ("89 iron ingots... 5
mithril ingots"). The battery measured whether an un-keyed query can be
routed. It cannot, by construction.

## 5. Prior art

- C7 r3 supersede fixture shape: wt/grm-c7 scripts/grm_c7_register_r3.py:41-49 (GRM contributors, 2026)
- Production supersede turn path: scripts/grm_e2e_session.py:2590-2597
- correct_memory retirement: core/graft_repository.py:1953-1998
- Route eligibility excludes retired: core/graft_arena.py:1450-1454
- Value-span scorer: C5 arm S via scripts/grm_lt1.py:score
- Recap decomposition: no specific prior art known to me; unverified -- lead to check (search: aggregate question decomposition, list-recall vs point-recall evaluation)
- **RD2's C7 finding** (GRM contributors, 2026) named both traps before
  I looked: correction-registered-as-ordinary-fact and
  alias-edge-mounted-without-base. This report confirms both on LT1
  receipts and adds the mechanism (candidate-base eligibility) plus the
  two counterexample lineages (`recall_5_*`, `recall_6_*`) showing the
  failures are order- and luck-dependent rather than categorical.
- **Cause-table shape and the per-row receipt join** — no prior art
  known to me.

## 5b. Follow-up (A1 merged): amendment 1 and the A+ CPU check

A1 landed `GRM_ALIAS_FOLD_MERGE` in core, so the alias class this
report STOPPED on now has a mechanism. Three things changed.

**(1) The absent-flag test became a present-flag test.** D1 asserted
the absence rather than assuming it, so the assertion failed the
moment A1 merged — which is the point. It is replaced by
`test_alias_flag_is_present_and_really_read_on_this_tree` (the reader
exists, switches in both directions, and an unknown token fails
CLOSED to OFF) and `test_p1_profile_now_pins_the_flag_instead_of_
recording_it_absent`. The BASE registration is deliberately NOT
rewritten: it is sha-bound at `02b44d02…` and truthfully records
what was true when it was written; the amendment supersedes it.

**(2) LT1.1 amendment 1** — `artifacts/grm_d1/lt1_1/amendment1.json`,
sha256 `0a5754cd3e5bc91c4c5428228ed4164534c5d281d2aa0c7410e8c7f8b145ce0e`,
chained to registration `02b44d02e3fbb82c…` and order `6ae0f52af0059f0a…`.

*Core rebind.* The lead named two changed files; the receipts show
**five** core inputs differ from the LT1 run, and all five are
rebound with attribution — pinning only two would silently accept
the other three:

| core input | before | after | attribution |
|---|---|---|---|
| `core/graft_arena.py` | `83a2d4a2bc0e…` | `918f5d0b202b…` | pre-A1 drift on grm-merge, not attributable to A1 |
| `core/graft_repository.py` | `fc6b9448efb4…` | `591657233106…` | A1 (alias fold-merge hooks) |
| `core/grm_admission.py` | `d1afc26a68b4…` | `ebdfd84af435…` | pre-A1 drift on grm-merge, not attributable to A1 |
| `core/grm_runtime.py` | `39f823cbdb98…` | `973addc491a2…` | A1 (alias fold-merge hooks) |
| `core/grm_text_norm.py` | `c4e496848b01…` | `e968d6879195…` | pre-A1 drift on grm-merge, not attributable to A1 |
| `core/grm_alias_fold.py` | — (new) | `147ea9687112…` | A1 (new module; did not exist at the LT1 run) |

21 of the 26 pinned core inputs are unchanged. The A1 attribution is
checkable and checked: `test_a1_files_are_attributed_to_a1_and_the_
others_are_not` requires every A1-attributed file to reference
`alias_fold` and every other changed file NOT to.

*Two arms.* Same frozen conversation, same 26-cell schedule,
separately resumable into distinct output directories:

| arm | alias flag | env after `environment(flags)` | out dir | budget |
|---|---|---|---|---|
| `A` | OFF | `GRM_ADMISSION_RULE=margin_first` | `artifacts/grm_d1/lt1_1/run_A` | 1.70 GPU-h |
| `A+` | **ON** | `GRM_ADMISSION_RULE=margin_first GRM_ALIAS_FOLD_MERGE=1` | `artifacts/grm_d1/lt1_1/run_Aplus` | 1.70 GPU-h |

Total 3.40 GPU-h; the lead may run A+ alone (1.70) and read A off
the existing LT1 arm-A receipts, or run both for a same-tree
contrast — running both is the only way to attribute a change to
the flag rather than to the core rebind.

*Registered predictions, before any run.* I disagree with one of
the lead's numbers and say so rather than substituting silently:
the lead proposed A+ aliases **≥ 3/5**; I register **4/5** (which
also meets ≥ 3/5). Reason: the CPU check below is unambiguous at
every distance, but the CPU reader is a regex double and LT1
already showed one long-distance ladder drop (`recall_3_150`) that
no lineage change addresses. 4/5 prices exactly one such drop.

**(3) The A+ CPU alias check** — `artifacts/grm_d1/alias_cpu_aplus.md`
/ `.json`. Flag and rule pinned and **read back** (R1 `pin_rule`
idiom); 164 turns replayed; 10 merges executed.

| probe | d | alias -> base | LT1 arm A | fold fired? | digest | admitted? | mounted | served (CPU double) |
|---|---|---|---|---|---|---|---|---|
| `recall_6_10` | 10 | the Hauler -> Kestrel | correct | yes | [164] | yes | [164] | `37 crates` |
| `recall_7_10` | 10 | the Beacon -> Lantern | WRONG | yes | [165] | yes | [165] | `18 October 2196` |
| `recall_6_25` | 25 | the Hauler -> Kestrel | correct | yes | [164] | yes | [164] | `37 crates` |
| `recall_7_25` | 25 | the Beacon -> Lantern | WRONG | yes | [165] | yes | [165] | `18 October 2196` |
| `recall_6_50` | 50 | the Hauler -> Kestrel | correct | yes | [164] | yes | [164] | `37 crates` |
| `recall_7_50` | 50 | the Beacon -> Lantern | WRONG | yes | [165] | yes | [165] | `18 October 2196` |
| `recall_6_100` | 100 | the Hauler -> Kestrel | correct | yes | [164] | yes | [164] | `37 crates` |
| `recall_7_100` | 100 | the Beacon -> Lantern | WRONG | yes | [165] | yes | [165] | `18 October 2196` |
| `recall_6_150` | 150 | the Hauler -> Kestrel | correct | yes | [164] | yes | [164] | `37 crates` |
| `recall_7_150` | 150 | the Beacon -> Lantern | WRONG | yes | [165] | yes | [165] | `18 October 2196` |

- A+ fold fired **10/10**, digest names both names **10/10**, admitted **10/10**, mounted **10/10**, served-correct **10/10**.
- A (flag OFF) fold fired **0/10**, served-correct **0/10** — the
  default-OFF contract holds and the LT1 failure reproduces.

The digest for `recall_7_*` reads: *"For the archive: the Let's call
Lantern 'the Beacon' from now on… Lantern's launch date will be 18
October 2196. the Beacon is an alias for Lantern."* — coverage 1.0,
lineage `digest_supersedes_edge_and_base`, **78 tokens inside the 96**
arena width. That is RD2's width wall (49+61=110 > 96, which killed
the FIX-7 co-mount) solved by moving the join to write time.

**RED I hit and had to work through, reported plainly.** My first
A+ run showed **0/10 folds fired**. The cause was mine, not A1's:
`ArenaCache.consolidate` gates the digest on fact coverage, and the
C7 CPU double answers every generation with the probe reader, which
returns "unknown" for a summarization request — coverage 0.0, so
every fold correctly aborted with `alias_fold_fidelity_abort`. The
fix was to route the two request kinds to two readers (probe reader
for probes, faithful extractive digest for consolidation), which
exercises the coverage gate rather than bypassing it. Worth
recording because a seat that stopped at the first run would have
reported A1 as non-functional on entirely harness-side grounds.

**What this does and does not show.** It shows the fold fires, the
digest names both entities under the same FIX-8 projection routing
uses, routing admits it, the ladder mounts it, and it fits the
width. It does **not** show language-model recall: the reader is a
regex stub. The GPU A+ arm is the test of sufficiency, and it is
registered, not run.

## 5c. Follow-up 2: the runner (the registration was not runnable)

**The defect.** The `lead_commands.txt` this report previously
described did not run. It named
`scripts/grm_lt1.py --run --arm A --registration … --out …`; that
script's CLI is `[--preflight] [--summary] [--cell] [--resume]
[--dry-run]` and it reads its registration from a fixed path. I
emitted an interface that does not exist and never executed it;
the only test on those commands compared a sha string. A
registration is not runnable until a worker executes it.

**The runner.** `scripts/grm_lt1_1.py`, sha256
`78fd7699acda6e0bd90daba512bbc6c1e68c05d7b57ca6066cd3ede34c3da3d5`,
one arm per invocation, bound as a registered input by amendment 2.

*Parameterized, not forked.* `grm_lt1_worker.execute` is already
argument-driven (cell, directory, registration, loader, run, fake)
-- `scripts/grm_lt1_worker_cpu.py` already reuses it that way.
Only three things in that module are bound to LT1 module state:
`lt.FIX`, `lt.binding` and `RUN`. The runner redirects exactly
those three and calls `execute` / `run_cell` / `pending`
unchanged. `test_the_reuse_claim_is_stated_and_true` greps the
runner for `def execute(` / `def run_cell(` / `def pending(` and
fails if any reappears, so "not forked" is checked, not asserted.

*Proof it runs* (`artifacts/grm_d1/lt1_1/proof/`):

- `--arm A --dry-run` -> exit 0, 26 cells, next `A-001-008`, estimate 4508 s, reservation 7410 s, budget 7410 s, within_budget True, alias pin False
- `--arm A+ --dry-run` -> exit 0, 26 cells, next `A-001-008`, estimate 4508 s, reservation 7410 s, budget 7410 s, within_budget True, alias pin True
- `--arm A --fake --limit 2` and `--arm A+ --fake --limit 2` -> 2
  cells each, one subprocess per cell, writing real
  `controller.json` / `worker.json` / `reservation.json` /
  `checkpoint/` under `proof/fake/{A,Aplus}/cells/`.
- `--summary` reads those receipts: 2/26 complete per arm, arm
  bindings differ (`alias_fold_merge` False vs True).
- resume: a third `--fake --limit 3` skipped `A-001-008`,
  `A-009-016` and ran only `A-017-024`; `--summary` then reports
  3/26 complete, 3 measured recalls, `partial raw rows`.

**Amendment 2** — `artifacts/grm_d1/lt1_1/amendment2.json`, sha256
`3bf21605058c86b15d3cb91784ea39409844838f20130b262dbadf1948417380`, chained to amendment 1
`0a5754cd3e5bc91c…`. It binds the runner and
corrects the budget.

**A second defect the dry-run caught — a budget that would have
railed.** Amendment 1 registered 6120 s (1.70 GPU-h) per arm. The
26 cells reserve `sum(lease_seconds) = 7410 s` (2.06 GPU-h), and
`run_cell` charges the LEASE, not the estimate, railing on
`accounting()+lease`. The old ceiling sat below the reservation
sum and would have tripped `COMBINED_GPU_BUDGET_RAIL` partway
through a campaign that was going to finish. Amendment 2 raises
the ceiling to the reservation sum.

| | per arm | both arms |
|---|---|---|
| amendment 1 ceiling (superseded) | 1.70 GPU-h | 3.40 GPU-h |
| amendment 2 ceiling (reservation) | **2.06 GPU-h** | **4.12 GPU-h** |
| projected actual spend | 1.25 GPU-h | 2.50 GPU-h |

**This is a budget INCREASE and needs the lead's eye.** Expected
spend is unchanged at 1.25 GPU-h per arm; only the ceiling moves,
to a number the machinery can honour. Registered before any run.

**The gate that would have caught the original mistake.**
`tests/test_grm_lt1_1_runner.py::test_every_emitted_command_
actually_runs` parses every `python3 scripts/…` line out of
`lead_commands.txt`, appends `--dry-run`, executes it, and
requires exit 0. A companion test
(`test_the_old_broken_invocation_would_have_been_caught`) runs the
exact shape I shipped and asserts it fails with "unrecognized
arguments", so the gate is proven to have teeth.

## 5d. Follow-up 3: the resume route (the seam had drifted)

**The defect.** `--arm A+ --resume` died on the card before taking
any lease:

```
grm_lt1_amendment4.apply -> lt.binding('CPU') -> ARM_ALIAS['CPU']
KeyError: 'CPU'
```

Two stacked mistakes, both mine:

- **D3-C1 (signature drift).** the redirected `binding` took an ARM; LT1 callers pass a BACKEND LABEL ('CPU') as well as an arm, and LT1's own `binding` is label-agnostic
  *Fix:* `binding(label)` keeps the LT1 signature and reads the arm from pinned runner state (`current_arm`)
- **D3-C2 (seam too wide).** `lt.binding` was redirected, but LT1 uses it to validate ITS OWN chain (`apply4` compares a recorded `protocol_binding` against it); receipts are stamped through `worker.bind`
  *Fix:* redirect `worker.bind` only; `lt.binding` stays LT1
- **D3-C3 (ordering).** LT1's host preflight ran INSIDE the redirected window, so `lt.binding` hashed the LT1.1 fixture and failed AMENDMENT4_PROTOCOL_MISMATCH
  *Fix:* `host_preflight()` runs before any seam moves

**Why 115 passing tests said nothing.** Neither `--dry-run` nor the fake path traverses `grm_lt1_amendment4.apply`; only the resume route does. 115 passing tests covered everything except the one path the lead actually ran.

**The seam audit.** Every redirected name was checked for arity and
argument meaning against every call site:

| name | moved? | why |
|---|---|---|
| `lt.FIX` | **yes** | PosixPath constant; same type, no signature. Read by `execute` for turns/probes/decisions AND by `lt.binding` for its fixture sha -- which is why the LT1 host preflight must run outside the window. |
| `worker.RUN` | **yes** | PosixPath constant; per-arm campaign root. |
| `worker.bind` | **yes** | callable(label) -> dict. Keeps the LT1 signature; stamps LT1.1 receipts. |
| `lt.REG` | no | LT1's own registration; `verify` must validate it. |
| `lt.RUN` | no | LT1's own campaign root; the runner has its own `summary` and never calls `lt.summary`. |
| `lt.binding` | no | LT1's self-validation (verify/preflight/apply4). Redirecting it breaks the amendment-4 chain. |

The two Path constants carry no signature. The one callable now
matches LT1 exactly: `binding(label)` echoes its argument the way
LT1's does, and reads the ARM from pinned runner state, which is
where the arm actually lives.

**The gate.** `--resume --dry-lease` walks the production route --
amendment load, apply4-bearing preflight, seam redirection, arm
pin, campaign owner file, `worker.pending` cell selection -- and
stops at `worker.run_cell`, the lease boundary. Nothing before that
point is stubbed.

- `--arm A --resume --dry-lease` -> rc 0, status PASS, next `A-001-008`, stopped at worker.run_cell (lease boundary), alias pin False, receipt `campaign_arm=A alias_fold_merge=False`
- `--arm A+ --resume --dry-lease` -> rc 0, status PASS, next `A-001-008`, stopped at worker.run_cell (lease boundary), alias pin True, receipt `campaign_arm=A+ alias_fold_merge=True`

RED-before / GREEN-after are both gated:
`test_the_shipped_seam_reproduces_the_leads_keyerror` reconstructs the shipped seam and asserts the exact
`KeyError: 'CPU'` surfaces through `grm_lt1_amendment4.apply`;
`test_resume_route_reaches_the_lease_boundary[A]` and `[A+]` prove
the fixed route. `test_only_the_documented_seams_move` cross-checks
the audit table against live behaviour.

**Amendment 3** — `artifacts/grm_d1/lt1_1/amendment3.json`, sha256
`09680a0567d2d74f0a9a491823cc627c1a8d609451fdf165b36384dc97e89a08`, chained to amendment 2
`3bf21605058c86b1…`. Runner rebound: `17f1577006e04494…`
supersedes `78fd7699acda6e0b…`.

**OPEN BLOCKER — needs your decision.** With the seam fixed, the
real `--resume` now reaches the genuine host check and stops there:

```
ValueError: INPUT_SHA_MISMATCH: core/graft_arena.py
```

T.his worktree forks grm-merge, whose core has drifted past the SHAs the LT1 registration pins; `lt.preflight()` fails identically with or without the LT1.1 redirection
`lt.preflight()` fails **identically with or without** the
LT1.1 redirection, so this is not something the runner introduced —
it is the same pre-existing core drift this report records in its
RED items. Consequence: the GPU campaign cannot start on this tree until the lead decides how the LT1 core pins are rebound for the host preflight. The LT1.1 route itself is proven green up to the lease boundary.

## 5e. Follow-up 4: LT1.1 validates its own chain (the ruling)

**The ruling** (lead, 2026-09-11), recorded verbatim in the
amendment and in the runner:

> LT1's original registration is a frozen receipt of its day; it is NOT re-validated against today's core. LT1.1's host preflight must validate LT1.1's own chain — registration 02b44d02 + amendments 1–3, whose core pins were rebound to this tree with attribution — using the same verification functions (`grm_lt1.verify`-class checks: sha-bound inputs, fixture sha, cell schedule, budget) but pointed at LT1.1's registration/amendment set. LT1's registration sha and its recorded core pins are carried as `parent` lineage in the LT1.1 receipt (recorded, with the drift table you already attributed), not as a gate.

**What changed.** `scripts/grm_lt1_1.py` replaced its
`lt.preflight()` call with `lt1_1_preflight(arm)`, which applies
the same verification classes to LT1.1's documents:

- sha-bound documents: registration + amendments 1-4 vs sidecars
- chain continuity: each amendment names its parent sha
- sha-bound inputs: every rebound core pin + the bound runner vs the file on disk
- fixture sha: dialogue.json vs the registered digest
- cell schedule: 26 arm-A cells matching the amendment
- budget: reservation sum within the registered ceiling
- host readiness: free space, pinned admission rule

Not checked, by the ruling: LT1 day-of core pins -- frozen receipt, see parent_lineage (this is the ruling)

**The `apply4` question you asked — measured, not assumed.**
`grm_lt1_amendment4.apply`'s protocol-binding check
(`a['protocol_binding'] != lt.binding('CPU')`) **passes on this
tree, unchanged and untouched** — recorded equals live: True.
`lt.binding` reports the core shas RECORDED in LT1 amendment 3, not live ones, so it is already immune to core drift. The whole of lt.verify() failure on this tree was its final input loop (INPUT_SHA_MISMATCH: core/graft_arena.py), which is exactly the re-validation the ruling removes.
So no second ruling is needed: amendment 4 never gated on drifted
core. Its recorded binding is reported as parent lineage, and
LT1.1 stamps its own through `worker.bind`.

**Parent lineage, recorded not gated.** LT1 registration
`e1913b144087ea79…`, 26 recorded core pins. Drift table:

| input | LT1 recorded | LT1.1 rebound = on tree now | attribution |
|---|---|---|---|
| `core/graft_arena.py` | `83a2d4a2bc0e…` | `918f5d0b202b…` | pre-A1 drift on grm-merge, not attributable to A1 |
| `core/graft_repository.py` | `fc6b9448efb4…` | `591657233106…` | A1 (alias fold-merge hooks) |
| `core/grm_admission.py` | `d1afc26a68b4…` | `ebdfd84af435…` | pre-A1 drift on grm-merge, not attributable to A1 |
| `core/grm_runtime.py` | `39f823cbdb98…` | `973addc491a2…` | A1 (alias fold-merge hooks) |
| `core/grm_text_norm.py` | `c4e496848b01…` | `e968d6879195…` | pre-A1 drift on grm-merge, not attributable to A1 |
| `core/grm_alias_fold.py` | — (new) | `147ea9687112…` | A1 (new module; did not exist at the LT1 run) |

**The gate still has teeth.** The risk in "stop checking X" is
that it becomes "stop checking".
`test_a_planted_drift_in_our_own_amendment_goes_red` plants a bad
core sha in LT1.1's OWN amendment 1 — sidecar kept consistent, so
the failure is the input check and not a document mismatch — and
requires `INPUT_SHA_MISMATCH` for that input. Five more tamper
tests cover the new-input branch, chain continuity, a tampered
document, a tampered fixture and an unpinned admission rule.

**Both arms, host gate ON, no escape hatch:**

- `--arm A --resume --dry-lease` -> rc 0, host_preflight **READY**, status PASS, next `A-001-008`, stopped at worker.run_cell (lease boundary), alias pin False — **no INPUT_SHA_MISMATCH**
- `--arm A+ --resume --dry-lease` -> rc 0, host_preflight **READY**, status PASS, next `A-001-008`, stopped at worker.run_cell (lease boundary), alias pin True — **no INPUT_SHA_MISMATCH**

**Amendment 3's blocker is resolved, and the test is inverted
with its receipt.** `test_the_host_blocker_claim_is_true_right_now`
became `test_the_host_blocker_claim_was_true_and_is_now_resolved`,
which asserts BOTH halves: LT1's own gate still refuses (the drift
is real and we did not paper over it) AND LT1.1's gate is READY.
Amendment 3 stays as written; amendment 4 records the resolution.

**Amendment 4** — `artifacts/grm_d1/lt1_1/amendment4.json`, sha256
`0b5da2387791e871759cf1d11c5ad27c4e4559642d81200a01bb2d9e7c85659d`, chained to amendment 3
`09680a0567d2d74f…`. Runner rebound `29c916601d4a3775…`.

## 5f. Follow-up 5: the leased CHILD runs our chain

**A green parent produced a RED cell.** `--arm A+ --resume`
pinned the arm, started A-001-008, and the child died:

```
scripts/grm_lt1_worker.py:289  __main__ -> lt.verify()
ValueError: INPUT_SHA_MISMATCH: core/graft_arena.py
```

run_cell hard-coded the child argv as `-m scripts.grm_lt1_worker --worker <cell>`; that module __main__ resolves its cell with lt.verify(), which is LT1 own chain and the gate the ruling removed from the parent

**Why the gates missed it — the third time, so stated plainly:**

- --dry-lease stops AT the lease boundary, before run_cell spawns anything
- the --fake 2-cell proof used --fake-cell, a path that never enters run_cell, so it exercised a different child
- the follow-up-2 docstring ASSERTED the child came back through the LT1.1 module; it did not, and that unverified claim is corrected here

The third is the one that matters: I wrote a docstring asserting
the child came back through the LT1.1 module and never checked.
`test_run_cell_spawns_the_lt1_1_child` now reads the argv
`run_cell` actually builds.

**The fix: five seams, not three.**

- worker.worker -> lt1_1_worker (covers lt.verify at grm_lt1_worker.py:188)
- worker.spawn_argv -> spawn_argv (covers grm_lt1_worker __main__ at :289 by never reaching it)

`scripts/grm_lt1_worker.py:run_cell` gained a `spawn_argv` /
`spawn_env` seam whose **default is byte-identical** to the line
it replaced, so LT1's own campaign is unaffected
(`test_the_default_argv_is_unchanged_for_lt1`). The child is now
`python3 -m scripts.grm_lt1_1 --arm <arm> --worker <cell>`,
which runs lt1_1_preflight (LT1.1 chain) -> lt1_1_seams -> worker.execute with registration(arm).

**Audit — every `lt.verify()` reach point in the worker module:**

| line | function | disposition |
|---|---|---|
| 188 | `worker` | covered: `worker.worker` is redirected to `lt1_1_worker`, which uses `registration(arm)` instead |
| 285 | `resume` | not reachable: the runner never calls `worker.resume`; it runs its own `resume` loop over `worker.pending` / `worker.run_cell` |
| 301 | `__main__` | not reachable: `spawn_argv` routes the child to `scripts/grm_lt1_1.py --worker`, so this entry point is never executed by an LT1.1 campaign |

**The gate: a real `run_cell` spawn, both arms, cells 1-2.**
Stubbed: lease stubbed to a no-op flock on a temp file and the model stubbed to the CPU double; both arms, cells 1-2. Unstubbed: reservation accounting, directory creation, argv construction, Popen + foreground wait, charge, checkpoint validation, controller receipt.

| arm | cell | run_cell | status | child pid | new process | arm in receipt | INPUT_SHA_MISMATCH |
|---|---|---|---|---|---|---|---|
| A | `A-001-008` | True | COMPLETE | 3347931 | True | A (alias False) | **False** |
| A | `A-009-016` | True | COMPLETE | 3347966 | True | A (alias False) | **False** |
| A+ | `A-001-008` | True | COMPLETE | 3348012 | True | A+ (alias True) | **False** |
| A+ | `A-009-016` | True | COMPLETE | 3348069 | True | A+ (alias True) | **False** |

RED-before: `test_the_old_argv_reproduces_the_leads_worker_exit_1`
spawns the ORIGINAL argv and requires the child to die with
`INPUT_SHA_MISMATCH: core/graft_arena.py` — the lead-run failure,
reproduced on demand. The `--dry-lease` gate is kept alongside.

**The RED cell is archived, not deleted.** `A-001-008` -> `artifacts/grm_d1/lt1_1/archive/Aplus_A-001-008_RED`; create-only: the RED attempt is copied, never deleted or overwritten, and the live cell directory is removed so arm A+ re-arms from cell 1
`test_the_archived_red_cell_is_preserved` asserts the archived log
still contains `INPUT_SHA_MISMATCH` (the evidence was not
sanitised) and that the live cell is gone so arm A+ re-arms.

**Amendment 5** — `artifacts/grm_d1/lt1_1/amendment5.json`, sha256
`c823e9e02b326c78434f589e4c996beb797450d65f19c246555cd7e26d0bbd0d`, chained to amendment 4
`0b5da2387791e871…`. Runner rebound `72fa0a79c174e313…`.

## 5g. Follow-up 6: a busy card is not a cell failure

**What happened.** The queue launched `--arm A --resume` while
the A1 contrast was still leaving the card. The single-probe idle
check refused and wrote `run_A/cells/A-001-008` RED with
`GPU_NOT_IDLE: 3336818` — controller and reservation only, no
`worker.log`, because no child was ever spawned. That leftover
then made `--arm A --resume --dry-lease` fail with
`PRIOR_CELL_RED`. Both effects were lead-caused; the second was a
real defect in the policy and is fixed here.

**The seam-restore question, answered: NO — state-caused.**
the assertion tested that the seam CHANGED the module global. When an earlier test had already left `worker.RUN` equal to the target, nothing changed and a correct seam read as broken. Restoration was verified separately by an identity check and was always correct.
The fix: the assertion now tests the invariant that matters -- the seam HOLDS the LT1.1 value inside the window -- plus an identity check on restore
I also checked the relaxed assertion still has teeth — with a
deliberately broken seam it reports `(False, False, False)` and
fails, so this is a sharper test, not a weaker one.

**The policy change.**

| | before | after |
|---|---|---|
| busy card | `one nvidia-smi compute-app probe` | `bounded wait for the card to fall below a framebuffer limit, then proceed` |
| keys on | compute-process list | TOTAL framebuffer used, never the compute-process list (FIX-8: never infer an idle card from an empty compute list) |
| bound | none (single probe) | 61 probes at 15 s, 900 s total |
| limit | any process at all | <= 512 MiB framebuffer used |
| on expiry | — | RED **with the memory snapshot** |

A busy card is a resource another process holds, not a fault in our cell. It NEVER signals, kills or waits on another process; it declines and it waits, nothing else
The bound is structural: attempts_allowed caps the probe count up front, so there is no unbounded loop and nothing re-runs a cell.

Installed as a sixth seam, `worker.await_idle`, with the same
byte-identical-default discipline used for `spawn_argv`:
`test_the_lt1_default_branch_is_byte_identical` pins LT1's own
single-probe refusal verbatim.

**Fixtures** (`tests/test_grm_lt1_1_idle_wait.py`, 12 tests):

- busy -> wait -> idle: the cell runs
- busy past the bound: RED with the memory snapshot
- probe failure: declines, with the reason
- the wait is bounded and counted, never unbounded
- nothing is ever signalled

The two the lead named: `test_busy_then_idle_waits_and_then_proceeds`
feeds 4096 MiB -> 3144 MiB -> 96 MiB and requires the wait to
proceed on the third probe; `test_busy_past_the_bound_declines_with_a_snapshot`
holds the card busy and requires a decline carrying
`memory.used=4096 MiB` and the holder pid.
`test_run_cell_raises_only_after_the_bound` drives the same
through the real `run_cell` and asserts the controller error
carries the snapshot, not the bare pid the lead run recorded.

**The spurious RED is archived, not deleted.** `A-001-008` -> `artifacts/grm_d1/lt1_1/archive/A_A-001-008_GPU_NOT_IDLE`; create-only: the attempt is copied, never deleted or overwritten, and the live cell directory is removed so arm A re-arms from cell 1
The archived controller still reads RED with `GPU_NOT_IDLE`, and
the absence of `worker.log` is itself the tell that the card
refused before any work began.

**Gate lines, both arms:**

- `--arm A --resume --dry-lease` -> rc 0, host_preflight **READY**, status PASS, **next=A-001-008**, stopped at worker.run_cell (lease boundary)
- `--arm A+ --resume --dry-lease` -> rc 0, host_preflight **READY**, status PASS, **next=A-001-008**, stopped at worker.run_cell (lease boundary)

**Amendment 6** — `artifacts/grm_d1/lt1_1/amendment6.json`, sha256
`89d26556ade25ea05de2101929c22ac0d7374c8eb59c6f057119b776d393a0ba`, chained to amendment 5
`c823e9e02b326c78…`. Runner rebound `664ad608468c8703…`.

## 6. Deviations, RED items, process safety

**Deviations from the order**

1. The order's class name `stale-first ranking (core A-DEC)` does not
   fit its one arm-A row: A-DEC ranked correctly and the *ladder*
   dropped the mount. I kept the order's name for 1:1 mapping and
   appended `[arm-A instance is ladder-drop, not mis-rank]` rather than
   silently reclassifying or silently mislabelling.
2. The order asked for the LT1.1 fixture under `fixtures/lt1_1/`. I
   wrote it to `artifacts/grm_d1/lt1_1/dialogue.json` instead, because
   the LT1 fixture manifest is SHA-frozen and a sibling under
   `fixtures/lt1*` risks tripping `FIXTURE_SHA_MISMATCH` in the existing
   verify chain. The registration's `fixture.path` points at the
   artifacts location. Trivial for the lead to move if preferred.
3. Arm B rows were built "where cheap", as permitted — full table, same
   classifier.

**RED items (honest)**

1. **`tests/test_grm_lt1*.py` is 19 failed / 58 passed on this branch,
   and this is PRE-EXISTING, not caused by my changes.** Cause: `grm-d1`
   forks `grm-merge`, whose `core/graft_arena.py` and
   `core/grm_admission.py` have drifted past the SHAs LT1's registration
   is bound to. The failure is `INPUT_SHA_MISMATCH: core/graft_arena.py`
   — the SHA-binding working correctly. Receipts:
   `artifacts/grm_d1/red_before_lt1.log`,
   `core_drift_graft_arena.diff` (50 lines),
   `core_drift_grm_admission.diff` (28 lines). The failing set is
   IDENTICAL before and after my work (19 = 19, zero new, zero D1).
2. **Worktree artifact restoration.** `grm-d1` was created without the
   gitignored `artifacts/` tree, so LT1's `.sha256` sidecars and the
   `amendment1`/`amendment3`/`grm_scout_fix4`/`grm_scout_fix6`
   directories were missing (baseline was 49 failed / 16 passed). I
   restored them with `rsync -a --ignore-existing
   /mnt/ForgeRealm/wt/grm-lt1/artifacts/ artifacts/`. This copied
   existing receipts; it fabricated nothing. Flagged because it changed
   the test baseline mid-session.
3. **No GPU was used and none is claimed.** Every number here is read off
   a frozen receipt or produced by a CPU double. The CPU supersession
   proof is a *lineage* claim (which nodes are route-eligible),
   explicitly not a model-quality claim. LT1.1 needs a real GPU run to
   test the prediction; it is **registered, not run** — 1.70 GPU-h,
   lead-run, commands in `lt1_1/lead_commands.txt`.
4. **The alias fix is unresolved.** 5 rows stay wrong by design in LT1.1.

**Process safety**

- No `git` was run.
- No subagents were spawned.
- No process this session did not start was killed or signalled.
- No GPU was touched; no GPU lease was taken.
- Every Bash call ran in the foreground, under 10 minutes, no background
  waits.
- `core/` was read but never written: `find core -newer
  artifacts/grm_d1/red_before_lt1.log` returns nothing.
- Writes were confined to
  `/mnt/ForgeRealm/wt/grm-d1/{scripts,tests,artifacts}`.
- `/mnt/ForgeRealm/wt/grm-lt1`, `/mnt/ForgeRealm/wt/grm-c7` and
  `/mnt/ForgeRealm/GraftRepository` were read-only.

