# GRM-F3 ledger — routing at distance

Order is immutable: `orders/GRM_F3_ROUTING_AT_DISTANCE.md`, SHA256
`25876742f7a50aeb43b801606d37446d44ceadeec9839007504e4644cde51045`.
Receipts under test: LT1.1 r2, `artifacts/grm_d1/lt1_1/run_A` and
`artifacts/grm_d1/lt1_1/run_Aplus` (registration
`artifacts/grm_d1/lt1_1/registration.json`).

Seat: Opus 5 (model id `claude-opus-5[1m]`), reasoning effort max.
Writable target: `/mnt/ForgeRealm/wt/grm-f3` (branch `grm-f3`).
No git. No subagents. No GPU run. No process signalled or killed.
Every command foreground and under 10 minutes. `--basetemp` under
`artifacts/grm_f3/tmp`, removed after every pytest invocation.

## Verdict in one line

**NO FIX LANDS.** The order's item-3 gate ("a single mechanism explains
>= 5 of 7 rows") is NOT met — the largest class is 3/7 — so this delivers
the 10-row route table, the per-row cause, and a registered proposal.
`core/` is byte-identical to the r2 binding (all six files verified).

Three of the order's seven "miss" rows are not routing misses at all, and
the order's framing ("the ranker never reaches it at d >= 100") is refuted
for four of the seven. Those are RED findings against the order's premise
and are stated plainly below.

## Prior art

- **Characterization tests** — Feathers, *Working Effectively with Legacy
  Code* (2004), ch. 13. TAKEN: the practice of pinning current behaviour,
  defects included, before changing it, and the discipline of labelling a
  pinned defect as a defect. OURS: nothing; the technique is used as-is.
  Applied in `tests/test_grm_f3_routing_at_distance.py`, whose three
  `test_identifier_insurance_branch_*` tests assert WRONG behaviour on
  purpose so a later fix must break them.
- **Must-include / constrained re-ranking** — the registered proposal below
  is the standard "a candidate satisfying a hard constraint is never
  displaced by unconstrained scoring" shape. Closest prior art I know:
  Maximal Marginal Relevance (Carbonell & Goldstein, 1998) reorders a
  ranked list under a secondary criterion without rescoring, and the frozen
  RT1 rule already in `core/grm_admission.py` is this project's own
  instance of it. TAKEN: the reorder-never-score stance, from RT1
  verbatim. OURS: the observation that `policy_plan`'s
  `one_off_rank_identifier_insurance_k3` branch and `margin_first_plan`'s
  `margin_insurance_k3_identifier_tiebreak` branch are BOTH named for an
  insurance they do not provide.
- **SQuAD v1.1 `normalize_answer` + exact match** (Rajpurkar et al., 2016)
  and **SQuAD 2.0 unanswerable questions** (Rajpurkar, Jia & Liang, 2018).
  TAKEN: a declared normalization pipeline applied identically to
  prediction and reference (already the basis of `grm_d1_scorer_v2`), and
  the principle that abstention is a first-class scored class rather than a
  wrong answer — which is precisely the F4 defect. OURS: the corpus-pinned
  literal alternation; no learned or heuristic no-answer classifier.
- **The harness** — `scripts/grm_c7_diagnose.py` CPU doubles and the
  `campaign_receipt` marker convention in `tests/conftest.py` (GRM
  contributors, 2026), reused wholesale.
- UNVERIFIED against the wider literature (no network in this sandbox) —
  lead to check. Search terms: "must-include constraint top-k retrieval",
  "constrained re-ranking guarantee matched entity", "maximal marginal
  relevance reorder", "SQuAD 2.0 unanswerable no-answer exact match",
  "abstention detection QA evaluation", "characterization test legacy
  defect pin".

## What is exactly reproducible on the CPU, and what needs the GPU

Stated up front because the order asks for it.

REPRODUCED EXACTLY, host-only, against the frozen session repositories:

- the identifier scan (`shaped_identifier_tokens` -> `is_identifier_binding`
  over `_route_cand_base`'s eligibility law). All 10 rows' CPU scans equal
  their GPU `identified_candidates` verbatim — pinned by
  `test_cpu_identifier_scan_reproduces_every_gpu_route_receipt`.
- the admission branch and plan (`margin_first_plan`, `policy_plan`) from
  the receipt's recorded `ranking` + `route_margin_1_2`. Recomputed plans
  equal the receipts' `rank_plan` verbatim.
- the grounding predicate (`_grounding_verdict_inner`'s set algebra) from
  literal node texts — pinned by
  `test_grounding_rejects_a_correct_answer_against_an_alias_captured_record`.
- the eligibility law, the `retired` state of every value-carrying node, and
  the presence of a route key for every ranked node (`index.npz` is 1:1 with
  `manifest.json` in both arms: 106/106 and 118/118).

NEEDS THE GPU (not claimed here):

- the score vector itself. `route_margin_1_2` depends on `_probe_key`, which
  is a model forward. The margins in the table are the receipts' recorded
  values, not recomputed. `decisive_admission_profile`'s own reconstruction
  guard already proved the recorded ranking matches the frozen score law at
  run time, so the recorded margin is trustworthy — but it is a GPU receipt,
  not a CPU result.
- the per-trip answer TEXT. The receipts record each trip's `mount_set` and
  `grounded` verdict but not the string the model produced on that trip, so
  the arm-A trip-0 answer is reconstructed as a representative correct
  phrasing, not quoted. The STRUCTURAL claim (trip 0 mounted node 61 and
  grounding returned false; trip 3 mounted node 41 and grounding returned
  true) is receipt-verbatim.

## The 10-row route table

Columns: rank of the intact record / whether it entered the plan / whether
it was seated / branch / margin vs the 0.1385774091529802 threshold. The
"intact record" is the LIVE (non-retired) node carrying the expected value.

| # | arm | probe | t | ident tok | scan hits (CPU == GPU) | ranking | margin | branch | plan | intact node | its rank | in plan | seated | width/seats | recency excl | verdict | cause |
|---|-----|-------|---|-----------|------------------------|---------|--------|--------|------|-------------|----------|---------|--------|-------------|--------------|---------|-------|
| 1 | A | recall_3_50 | 64 | breakwater | `[]` == `[]` | 24,41,12,19,31,29 | 0.0748 | margin_insurance_k3_identifier_tiebreak | 24,41,12 | 24 (digest) | 1 | yes | 24,41 | 96 / 62 | 44,49 | **correct** | control: intact ranked 1 and seated; every trip ungrounded, ladder exhausted, keep-first served trip 0 = the right value |
| 2 | A | recall_3_100 | 114 | breakwater | `[]` == `[]` | 61,102,41,92,103,97 | 0.0615 | margin_insurance_k3_identifier_tiebreak | 61,102,41 | 61 (era) | **1** | yes | 41 | 96 / 23 | 104,105 | wrong_value | **grounding false-negative on the intact record.** Trip 0 mounted node 61 (holds `(-31, 48, 12)`), `grounded=false`; ladder descended; trip 3 mounted node 41 (Spindle `(64, -17, 8)`), `grounded=true`, served. |
| 3 | A | recall_3_150 | 164 | breakwater | `[]` == `[]` | 156,140,124,61,155,41 | 0.0133 | margin_insurance_k3_identifier_tiebreak | 156,140,124 | 61 (era) | 4 | no | 140 | 96 / 91 | 162,163 | wrong_value | same grounding false-negative; trip 1 mounted 61 anyway, `grounded=false`; all 6 trips ungrounded, keep-first served trip 0's node 140 (Ember/Willow era) |
| 4 | A | recap_3 | 198 | breakwater | `[]` == `[]` | 140,156,172,124,61,177 | 0.0032 | margin_insurance_k3_identifier_tiebreak | 140,156,172 | 61 (era) | 5 | no | 124 | 96 / 91 | 196,198 | wrong_value | same grounding false-negative; trip 1 `grounded=true` on node 124 (Cove `(19, 83, -22)`) and served it — a grounding FALSE-POSITIVE ends the ladder before 61 is tried |
| 5 | A+ | recall_2_50 | 63 | commtower | `[]` == `[]` | 28,18,20,23,12,46 | 0.4907 | fit_margin_decisive_rank1 | 28 | 28 (digest) | 1 | yes | 28 | 96 / 40 | 31,51 | **correct** | control: margin decisive, plan = the intact record, one grounded trip |
| 6 | A+ | recall_2_100 | 113 | commtower | `[]` == `[]` | 64,72,116,84,100,55 | 0.4562 | fit_margin_decisive_rank1 | 64 | 64 (era) | **1** | yes | **64** | 96 / 96 | 114,117 | wrong_value | **NOT A ROUTING MISS.** Node 64 ("the maintenance crew of the comm tower, named Iona Vale") ranked 1, was the whole plan, was seated, `grounded=true`. Reader confabulated with the answer in context. |
| 7 | A+ | recall_2_150 | 163 | commtower | `[]` == `[]` | 64,168,72,132,116,84 | 0.4460 | fit_margin_decisive_rank1 | 64 | 64 (era) | **1** | yes | **64** | 96 / 96 | 171,172 | wrong_value | identical to row 6, same served node, byte-identical answer |
| 8 | A+ | recall_3_50 | 64 | breakwater | `[28]` == `[28]` | 28,42,18,20,46,12 | 0.3221 | fit_margin_decisive_rank1 | 28 | 28 (digest) | 1 | yes | 28 | 96 / 40 | 31,51 | **correct** | control: sole binder, margin decisive |
| 9 | A+ | recall_3_100 | 114 | breakwater | `[64]` == `[64]` | 116,110,42,84,64,55 | 0.0673 | margin_insurance_k3_identifier_tiebreak | 116,110,42 | 64 (era) | 5 | **no** | **64** | 96 / 96 | 114,117 | wrong_value | **NOT A ROUTING MISS (scorer).** The plan dropped the sole binder, but trip 1 mounted 64 anyway and grounded; the served answer IS `-31, 48, 12` (no parentheses, U+2011/U+202F) — F4 column 2 scores it **correct**. |
| 10 | A+ | recall_3_150 | 164 | breakwater | `[64]` == `[64]` | 168,132,116,152,167,42 | 0.0077 | margin_insurance_k3_identifier_tiebreak | 168,132,116 | 64 (era) | **not ranked** | no | 152 | 96 / 90 | 171,172 | wrong_value | **the one true routing miss.** Node 64 is eligible and IS the sole identified binder, yet falls outside the 6-wide route window; the plan is three non-binders; the binder is never mounted. |

Per-row facts common to all ten: `admission_rule = margin_first`,
`route_backend = python`, `route_limit = 6`, arena `width = 96`,
`recency_seats.charge_waived_reason = identifier_query_point_lookup`
(so recency never charged a seat and never displaced a candidate),
`admission_split_child_demoted = false` with empty
`admission_split_family_ids` (so RT1 is a no-op on every row and no
split/width demotion is implicated), and every ranked node has a route key
in `index.npz` (so FIX-9's stale/absent-key path is not implicated either).

## Per-row cause line (the order's item 2)

One cause per row, with the receipt line that proves it.

1. `A/recall_3_50` — **control, correct.** No cause. All trips ungrounded;
   keep-first served trip 0 (`fit_seated = [24, 41]`, node 24 holds the value).
2. `A/recall_3_100` — **grounding false-negative from FIX-5/F2 alias capture.**
   `trips[0] = {mount_set: [61], grounded: false}`;
   `trips[3] = {mount_set: [41], grounded: true}`; `served_without_plan_head: true`.
   Node 61's text says "the Beacon's map position is (-31, 48, 12)", so
   `normalized_words(node 61)` contains `beacon` and not `breakwater`.
3. `A/recall_3_150` — **same alias-capture grounding false-negative.**
   `trips[1] = {mount_set: [61], grounded: false}`, six trips, none grounded.
4. `A/recap_3` — **same alias-capture grounding false-negative, compounded by
   a grounding false-positive.** `trips[1] = {mount_set: [124], grounded: true}`
   ends the ladder; node 61 is `planned` on that same trip but not mounted.
5. `A+/recall_2_50` — **control, correct.** No cause.
6. `A+/recall_2_100` — **reader failure, not routing.** `rank_plan: [64]`,
   `fit_seated: [64]`, `trips = [{mount_set: [64], grounded: true}]`, and node
   64's text contains "Iona Vale" verbatim. RED against the order's premise.
7. `A+/recall_2_150` — **reader failure, not routing.** Identical receipt shape
   to row 6. RED against the order's premise.
8. `A+/recall_3_50` — **control, correct.** No cause.
9. `A+/recall_3_100` — **scorer miss, not routing.** `trips[1] = {mount_set:
   [64], grounded: true}`; `memory.answer` = "The breakwater's coordinates are
   **-31, 48, 12**"; F4 column 2 `matched_variant = "-31, 48, 12"`.
   RED against the order's premise.
10. `A+/recall_3_150` — **identifier-insurance branch drops the sole binder.**
    `identified_candidates: [64]`, `route_margin_1_2: 0.0077 < 0.1386`,
    `policy_branch: margin_insurance_k3_identifier_tiebreak`,
    `rank_plan: [168, 132, 116]`. Node 64 is absent from both the plan and the
    6-wide ranking; no trip ever mounts it.

Classification tally over the order's 7 miss rows (rows 2, 3, 4, 6, 7, 9, 10):

| cause class | rows |
|---|---|
| alias-capture grounding false-negative (F2's consequence) | **3** |
| reader dilution, intact record mounted and grounded | 2 |
| scorer miss, answer glyph-correct (F4's consequence) | 1 |
| identifier-insurance branch drops the sole binder | 1 |

Max class = 3/7 < 5/7. **Item-3 gate not met: proposal only.**

## RED findings against the order's premise

Stated plainly, because they change what round 3 should do.

1. **"The mounted node is an unrelated fact each time" is a diagnostic
   artifact, not a router behaviour.** `core/graft_arena.py` writes
   `info["mounts"] = [i + 1 for i in picks]` — a 1-BASED display
   convention. `miss_causes.json` resolved its `served_nodes` from
   `mounts`, so it reported graft `i+1` (an unrelated turn) in **all 7**
   miss rows. The correct field, `mounted_ids` / `fit_seated`, was right
   all along. Verified 7/7 (`mounts == [seated+1]`), pinned by
   `test_miss_rows_mounts_field_is_seated_plus_one` and, at its source, by
   `test_mounts_receipt_field_is_one_based_and_is_not_a_graft_index`.
   The producing script is lead-side (not in this worktree); it is NOT
   edited here. Lead action: fix the `served_nodes` derivation before
   round 3 reuses `miss_causes.json`.
2. **"The ranker never reaches it at d >= 100" is refuted for 4 of 7 rows.**
   The intact record ranked **1** in rows 2, 6 and 7, and was **seated and
   grounded** in rows 6, 7 and 9. Only row 10 is a genuine failure to reach.
3. **2 of the 7 rows (A+ `recall_2_100`/`recall_2_150`) cannot be fixed in
   admission or routing at all.** The answer was in the mounted context and
   grounding passed. These belong to the reader/dilution track, not F3.
4. **1 of the 7 rows (A+ `recall_3_100`) is already correct** under the
   scorer this very order registers in item 4. It should be removed from the
   miss set.
5. **The oracle control is unreliable for the Commtower question.** The
   oracle refuses ("I'm sorry, but I can't help with that.") at d=50 AND at
   d=100/150, on a prompt containing the source verbatim. Any round-3
   comparison against the oracle on `recall_2_*` is comparing against a
   refusal, not a ceiling.

After removing rows 6, 7 and 9, the F3 residual is **4 rows**: three
alias-capture grounding false-negatives (rows 2, 3, 4) and one true routing
miss (row 10). That is the real target for round 3, and F2 already owns
three of the four.

## The proven defect (registered, NOT landed)

`core/grm_admission.py` has two admission branches named for an "identifier
insurance" they do not provide:

- `margin_first_plan` (FIX-6, the rule these runs used), branch
  `margin_insurance_k3_identifier_tiebreak`: when the margin is below
  threshold it reorders ONLY within the exact-score tie group of rank 1,
  then returns `ranking[:3]`. A sole binder whose score is not tied with
  rank 1 is never moved and, if it sits at rank >= 4, is silently dropped.
- `policy_plan` (the frozen `all_tokens_bind` rule), branch
  `one_off_rank_identifier_insurance_k3`: fires precisely when there is
  exactly one binder and it is off rank 1, and then returns `ranking[:3]`
  — which need not contain the binder at all.

Both are pinned as characterization tests. The asymmetry that makes this a
defect rather than a design choice: with TWO OR MORE binders the frozen rule
takes `declared_synthesis_identified_set` and admits **every** identified
candidate, explicitly including ones the router never ranked. With exactly
one, it admits none. Pinned by
`test_two_or_more_binders_are_all_admitted_regardless_of_rank`.

Receipt: row 10 — the arena held exactly one node whose own text binds
"breakwater", A-DEC identified it, and the plan was three nodes that do not.

### Registered proposal (for the lead to accept or decline)

**Rule (structural; no new threshold, nothing refit):** when the identifier
scan yields exactly one binder, that binder is in the plan. Concretely, in
both `margin_first_plan` and `policy_plan`, the `k3` insurance branches
become `ranking[:3]` with the sole identified candidate APPENDED if it is
not already present (or substituted for the last slot if a fixed k=3 is
required by the fit stage). Relative order is otherwise the router's,
untouched. It REORDERS/EXTENDS; it never scores — the same stance RT1 takes
two hundred lines above it in the same file.

- Flag: `GRM_ROUTE_SOLE_BINDER_INSURANCE`, default OFF, OFF byte-identical.
- Expected effect on the r2 receipts: **row 10 only** (1 of 7). Row 9 already
  reaches the binder via the ladder; rows 2/3/4 have no binder to insure
  (the alias capture removed the identifier), and rows 6/7 already seat it.
- This is why it is a proposal and not a landing: a one-row fix does not meet
  the order's own >= 5/7 bar, and landing it would spend a flag and a default
  decision on a mechanism that the F2 work may make moot.

### Second proposal (larger, out of F3's authorized scope)

The 3/7 class is the grounding gate, not the ranker. `_grounding_verdict_inner`
computes `content <= have` where `content` includes the entity name the answer
correctly used. When a fold has renamed the entity, **a correct answer can
never ground against the node that holds its value**, and the ladder's
first-grounded-wins then prefers any node whose numbers happen to be
self-consistent. Rows 2 and 4 show both halves: a false negative on the right
node and a false positive on a wrong one.

Two candidate shapes, both needing their own order and their own RED->GREEN:

1. Make grounding alias-aware — admit an identifier that the mounted node's
   fold provenance shows was rewritten (the alias/rename turn is in the
   node's `sources`). Narrow, but depends on F2's alias bookkeeping existing.
2. Make the ladder prefer the TRIP THAT MOUNTED A PLANNED NODE over a later
   grounded trip that mounted filler, instead of strict first-grounded-wins.
   Broader blast radius; would need the full battery.

F3 registers both and lands neither. The honest ordering is F2 first: if the
chronicle fold stops rebinding facts onto aliases, rows 2, 3 and 4 lose their
cause without any change to grounding or routing.

## F4 — scorer (the order's item 4)

**Registration path:** `scripts/grm_d1_scorer_v2.py`. Round 2's REGISTERED
SECONDARY scorer is that module's column, exposed as `score_v2_full` (the
existing `score_v2` value column plus the primary's category ladder, so rows
compare category-for-category). The registered PRIMARY remains
`scripts/grm_lt1.score`, untouched and still the verdict of record —
verified by `test_f4_extension_does_not_rescue_a_wrong_value_or_move_the_primary`.

**Abstention extension.** Registered BEFORE the rescore: every r2 answer in
both arms was scanned for abstention-shaped phrasing (the scan is recorded in
`artifacts/grm_f3/f4_rescore.json`). Exactly ONE family was observed — arm A
`recall_5_25`, "We're still working on that." — so exactly that family and its
orthographic variants were added:

```
(?:we(?:'re| are) )?still working on (?:that|it|this)
```

No speculative phrasing was added; an unobserved alternative cannot be
validated against this corpus, and a regex that fires on unseen text is a
scorer that silently reclassifies future rows. **When this guard stops
applying:** the alternation is pinned to round 2's observed corpus; a later
round with a different abstention wording must re-run the scan and
re-register, not widen the regex in place. The primary's own alternation is
COPIED into the module rather than imported, so a later edit to the primary
cannot silently change the secondary — staleness is caught by
`test_f4_primary_alternation_is_a_verbatim_copy_of_the_registered_scorer`.

**r2 two-column rescore** (n = 40 probes per arm):

| arm | column 1 (registered primary `grm_lt1.score`) | column 2 (registered secondary `score_v2_full`) |
|-----|-----------------------------------------------|--------------------------------------------------|
| A   | correct 33, wrong_value 7                     | correct 36, abstention 1, wrong_value 3           |
| A+  | correct 32, wrong_value 8                     | correct 34, wrong_value 6                         |

Rows that moved (6 total, all col1 `wrong_value`):

| arm | probe | t | col1 | col2 | expected | matched variant |
|-----|-------|---|------|------|----------|-----------------|
| A  | recall_1_10  | 22  | wrong_value | correct    | 9 voxels        | `9-voxel` |
| A  | recall_1_25  | 37  | wrong_value | correct    | 9 voxels        | `9-voxel` |
| A  | recall_5_25  | 41  | wrong_value | **abstention** | 16 beds     | — (F4 extension) |
| A  | recall_1_50  | 62  | wrong_value | correct    | 9 voxels        | `9-voxel` |
| A+ | recall_1_10  | 22  | wrong_value | correct    | 9 voxels        | `9-voxel` |
| A+ | recall_3_100 | 114 | wrong_value | **correct** | (-31, 48, 12) | `-31, 48, 12` |

The last row is the one that also removes a member of F3's own miss set.

## Commands run (all foreground, all < 10 min)

```
python3 -m pytest -q --basetemp /mnt/ForgeRealm/wt/grm-f3/artifacts/grm_f3/tmp \
  tests/test_grm_admission.py tests/test_grm_scout_fix4.py tests/test_grm_scout_fix8.py
      -> 55 passed (baseline, before any file was written)

python3 -m pytest -q --basetemp /mnt/ForgeRealm/wt/grm-f3/artifacts/grm_f3/tmp \
  -m campaign_receipt tests/test_grm_f3_routing_at_distance.py
      -> 22 passed, 12 deselected

python3 -m pytest -q --basetemp /mnt/ForgeRealm/wt/grm-f3/artifacts/grm_f3/tmp \
  tests/test_grm_f3*.py tests/test_grm_admission.py \
  tests/test_grm_scout_fix4.py tests/test_grm_scout_fix8.py
      -> the order's verbatim gate; result in the report
```
`artifacts/grm_f3/tmp` was removed after every invocation.

## Files changed

- `scripts/grm_d1_scorer_v2.py` — APPEND only (F4 registration block;
  `PRIMARY_ABSTAIN_ALTERNATION`, `F4_ABSTAIN_EXTENSION`, `ABSTAIN_RE_V2`,
  `abstains_v2`, `score_v2_full`). The pre-existing `normalize_v2`,
  `_span_found`, `_variants` and `score_v2` are untouched.
- `tests/test_grm_f3_routing_at_distance.py` — NEW. 12 pure-CPU pins +
  22 campaign-receipt pins.
- `docs/GRM_F3_ROUTING_AT_DISTANCE_LEDGER.md` — NEW (this file).
- `artifacts/grm_f3/route_rows.json` — NEW, the 10-row table's raw data.
- `artifacts/grm_f3/f4_rescore.json` — NEW, the two-column rescore and the
  abstention-phrasing scan.
- `core/` — **UNCHANGED.** All six files byte-identical to the r2 binding's
  `rebound_core_shas`.
