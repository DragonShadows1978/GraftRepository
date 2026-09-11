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

