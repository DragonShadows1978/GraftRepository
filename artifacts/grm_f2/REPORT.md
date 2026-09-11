# GRM-F2 — chronicle folds must not rebind facts onto an alias

Seat: Opus 5 (`claude-opus-5[1m]`), reasoning effort MAX.
Worktree `/mnt/ForgeRealm/wt/grm-f2` (branch `grm-f2`, forked from `lc1-wip`
8edfae4). Date: 2026-09-11.
Order: `orders/GRM_F2_CHRONICLE_ALIAS_GUARD.md`.
Receipts, as they happened: `artifacts/grm_f2/LEDGER.md`.

## Verdict in one line

The rebind is **not a model failure the fold could not have avoided** — it is
a window the fold never needed: the two alias turns in digest 24's window
contribute **zero** facts and are **never enumerated**, so they change nothing
the fold was asked to write and only add "the Beacon" to the mounted K/V; and
the coverage gate scored that digest **1.0 (5/5)** because both rebound
entities appear in POSSESSIVE form, which the existing `_caps_tokens` class
cannot match. Both halves of the defect are now closed under
`GRM_FOLD_ALIAS_GUARD`, default OFF, OFF byte-identical.

## 1. The mechanism

Full measurements and the commands that produced them: `LEDGER.md` §2.

1. **The alias turns contribute no facts and are not enumerated.**
   `_fact_set([13,14,17,18]) = {'12','31','48','iona','vale'}` — five facts,
   all from turns 13 and 14. FIX-5's `[source N]` enumeration lists two spans,
   both entity-correct, and never names Lantern, Kestrel, the Beacon or the
   Hauler. Dropping turns 17 and 18 leaves `need` and the enumerated span list
   **byte-identical**.
2. **They are mounted anyway.** `consolidate` mounts every source in `idxs`,
   so "the Beacon" is in the attention window while the instruction names only
   crews and coordinates. Zero instructional weight, full attention weight.
3. **The coverage gate is blind to attribution.** `need` contains no entity
   name: `_fact_set`'s multi-word rule needs two consecutive capitalized
   words, and `_caps_tokens`' `^[A-Z][\w\-]+$` fails on `Commtower's` and
   `Breakwater's` while passing `Beacon`. So the rebound digest scores
   `best_cov = 1.0`, `hit_count = 5/5`, and is accepted at perfect fidelity.
4. **And it costs the alias record.** `_deposit_consolidation` retires all
   four sources, so the only record that "the Beacon" means Lantern is
   destroyed to produce a node that misuses that alias.

## 2. The entity check

> Every (entity, attribute, value) in the sources must appear in the digest
> with the SAME entity, or with an alias registered for THAT entity only.

Per source clause with both an owning entity and >= 1 value: the **owner** is
the clause's entity tokens minus its value tokens; each **value** is looked up
in the digest's clauses; the value **passes** if any carrying digest clause
names the owner or an alias registered for that owner. A value the digest does
not carry is **skipped** (coverage's question). A clause with no owner raises
nothing. Accepted names are scoped to **one** binding — "the Beacon" stands for
Lantern and for nothing else.

Entity proxy: capitalized word, **possessive-tolerant**, FIX-8/SC1.1
glyph-normalized, minus a small closed list of structural words. The
possessive tolerance is load-bearing — see §1.3. Alias detection is **A1's
`parse_alias_edge`, unchanged**.

Stated limits (`core/grm_fold_alias_guard.py`, "WHEN THIS GUARD STOPS
APPLYING"): it is a lexical proxy, not an entity recognizer; it judges
attribution, not truth or completeness; it never guesses an owner it cannot
read; the alias allowance is single-entity scoped.

## 3. The choice — (a) AND (b)

The order asks for evidence and both stated. Both were prototyped against the
receipt before either was written into core.

| arm | digest 24 | 13 FIX-5 GPU-contrast digests |
|---|---|---|
| **(b)** attribution QC | **5 violations** | **0 violations** |
| **(a)** window exclusion | window `[13,14,17,18]` -> `[13,14]` | **0 sources dropped** |

* **(a) is the causal fix and it is free** — §1.1 measures that removing the
  alias turns changes nothing the fold was asked to preserve.
* **(b) is the detector and it is not redundant** — a rebind can arrive with
  no alias turn in the window at all (a model confusing two entities), where
  (a) is silent; §1.3 is the standing blind spot (b) closes.
* Taking only (a) leaves that blind spot open. Taking only (b) keeps paying a
  fold abort, and the retirement of the alias turns, for a contamination we
  can simply not create.

## 4. Core diff

`core/grm_fold_alias_guard.py` — **NEW**, 440 lines. Flag resolution
(`GRM_FOLD_ALIAS_GUARD`, default OFF, unknown token fails CLOSED), the
possessive-tolerant entity proxy, the alias table, `fold_window_excludes` (a),
`attribution_violations` / `guard_receipt` (b).

`core/graft_repository.py` — lines 54-55 (imports), 262 (ctor kwarg), 332-344
(flag resolved once + `fold_guard_history`), 359-364 (install the hook only
when ON), 4104-4107 and 4115 (both `_librarian_jobs` return paths), 4118-4175
(`_guard_fold_windows`), 4177-4209 (`_fold_attribution_receipt`), 4211-4229
(`_record_fold_guard`).

`core/graft_arena.py` — lines 1934-1948 (`fold_attribution_guard = None`
class default + why the seam is here), 2249-2250 and 2359-2365 (the two
deposit sites), 2369-2386 (`_fold_attribution_ok`).

The hook runs **immediately before `_deposit_consolidation`**, the method that
retires the sources — a check after it would have to reverse a retirement and
a deposit. Rejection returns through `consolidate`'s **existing**
`(None, None)` abort path, so `_fold_once` takes the branch it already had:
no new failure mode.

## 5. Gates

RED -> GREEN on digest 24 (RED produced by neutering both treatments in
`grm_fold_alias_guard.py` and re-running; the module was then restored and
verified byte-identical to its backup):

```
RED   artifacts/grm_f2/cpu_red.log     6 failed, 28 passed
GREEN artifacts/grm_f2/cpu_green.log   35 passed
```

The six RED rows: `test_red_flag_off_deposits_the_rebound_digest`,
`test_green_flag_on_b_rejects_the_rebound_digest`,
`test_green_flag_on_a_excludes_the_alias_turns_from_the_window`,
`test_green_end_to_end_through_fold_once`, `test_entity_check_definition`,
`test_exclusion_keeps_a_fact_bearing_alias_turn`.

Controls:

* **Non-alias folds byte-identical ON vs OFF** —
  `::test_control_non_alias_folds_are_byte_identical_on_vs_off` compares the
  node table (with metadata), fold history, consolidation receipt and the
  model's whole call log across the two arms.
* **OFF byte-identical to the pre-F2 world** —
  `::test_flag_off_byte_identical` compares flag-ABSENT against explicit `0`
  over the same observables, and asserts OFF still deposits the defective
  digest (the RED baseline).
* **The 13 FIX-5 GPU-contrast folds' recorded digests still pass QC** —
  `::test_control_fix5_gpu_contrast_digests_still_pass`, 13 parametrized rows
  over the real GPT-OSS-20B digests in
  `artifacts/grm_scout_fix5/gpu/*/receipt.json`: zero attribution violations
  and zero sources dropped by exclusion on every one.
* **Regression surface** (fold/librarian/repository/arena suites):
  `artifacts/grm_f2/regression.log` — **32 passed, 1 skipped**.

## 6. RED I hit and had to work through, reported plainly

My first regression run FAILED 8 tests in `tests/test_grm_s4_fold_order.py`
with `AttributeError: 'GraftRepository' object has no attribute
'fold_alias_guard'` (saved verbatim at
`artifacts/grm_f2/regression_red.log`). The cause was mine: those tests build
a planner-only stub with `object.__new__(GraftRepository)` and a minimal
attribute set, bypassing `__init__` entirely — an existing idiom that
`_librarian_jobs` already honours for `fold_order` via `getattr`. My guard read
`self.fold_alias_guard` directly.

Fixed by using the same `getattr(..., False)` idiom in `_guard_fold_windows`
and tolerating a missing history list in `_record_fold_guard`, and pinned so
it cannot regress: `::test_planner_only_stub_keeps_the_pre_f2_plan`. Worth
recording because a seat that ran only its own fixtures would have shipped a
green report over eight broken tests in another file.

## 7. What this does and does not show

**Does.** The defect reproduces on the CPU double through the real production
consolidation path (FIX-3 Harmony wrapper, FIX-5 enumeration, real coverage
gate); the failure is on the entity check and not on the stub (the recorded
digest clears shape QC, list QC and coverage at 5/5 — pinned by
`::test_recorded_digest24_passes_coverage_and_qc`); an extractive double
reaches an entity-correct digest on the same window, so the RED is not the
harness; both treatments fire; the flag is OFF by default and OFF is
byte-identical.

**Does not.** No model-quality claim. No claim that GPT-OSS-20B writes an
entity-correct digest once the alias turn is gone — that is the lead's
registered GPU contrast. No claim about fresh recall, corrections, aliases or
recap: no CPU fixture here measures a scorer column. No routing claim — the
Breakwater routing miss (round-1 top residual, F3) is untouched; F2 changes
what is STORED, not what is ranked.

**Named before the run, not after** (`flag_contract.json`): the attribution QC
*can* abort a fold coverage would have accepted, and each abort marks its
sources `no_fold`, so a run with many aborts compresses less. Expected rare —
zero on all 13 GPU-contrast folds. A high abort count in the r3 arm is a
result, not a threshold to move afterwards.

## 8. Flag contract

`artifacts/grm_f2/flag_contract.json` — env name, constructor kwarg, default,
resolution order, fail-closed behaviour, what the lead pins on the LT1.1 r3
registration F1 owns, the registered predictions, the named abort risk, the
explicit not-claimed list, the guard's stated limits, and repo-relative pins
only (asserted by `::test_control_flag_contract_is_registered`). **No GPU run
of my own.**

Interaction note for the lead: F1 (`GRM_FOLD_RETAIN_SOURCES`) and F2 are
independent env names, but F2 changes the fold WINDOW and F1 changes what a
fold does to its sources afterwards. Both ON in one arm gives a joint reading
only.

## 9. Prior art

Annotated at the code site (`core/grm_fold_alias_guard.py` PRIOR ART block),
in `LEDGER.md` §7, and here.

**External — all UNVERIFIED (no network in this sandbox), lead to check.**

* **Maynez et al. (2020), arXiv 2005.00661**, "On Faithfulness and Factuality
  in Abstractive Summarization". TAKEN: the framing that a summary can score
  perfectly on token overlap while being unfaithful (EXTRINSIC hallucination)
  — exactly §1.3. NOT taken: their entailment-model metric; this module uses
  no model. Search: `Maynez 2020 faithfulness factuality abstractive
  summarization`, `extrinsic hallucination summarization`.
* **Nan et al. (2021), arXiv 2102.09130**, "Entity-level Factual Consistency
  of Abstractive Summarization". TAKEN: the unit of measurement — the ENTITY,
  not the token — checked mechanically against the source. NOT taken: their
  NER pipeline or training-time filtering. Search: `Nan 2021 entity-level
  factual consistency summarization`, `entity hallucination precision
  summarization`.
* **Goodrich et al. (2019), KDD**, "Assessing The Factual Accuracy of
  Generated Text". TAKEN: the (subject, relation, object) triple as the
  comparison unit — the order's own entity check. NOT taken: their learned
  relation extractor. Search: `Goodrich 2019 factual accuracy generated
  text`, `relation triple summarization factual consistency`.

**Local, verified, reused UNCHANGED as the verbs this composes**: FIX-5
source-enumerated consolidation and its `_fact_set` vocabulary; FIX-3's
"reject rather than deposit a lossy derivative" abort contract and its
`no_fold` exemption; A1's `parse_alias_edge` / `alias_scan_text` /
`identifier_set`; SC1.1/FIX-8 `normalize_glyphs`; SCOUT-FIX-9's NEVER-SILENT
receipt rule and `_record_alias_decision`'s dedup idiom; `ArenaCache`'s own
optional-seam idiom (`node_loader`, `native_store`) for the hook shape; the
fixture idioms of `tests/test_grm_scout_fix5.py` and
`tests/test_grm_a1_alias_fold.py`.

SC1.1 grounding (`_grounding_verdict`) is the nearest local antecedent for
"check the produced text against what was mounted". TAKEN: the stance. NOT
taken: its pooled token coverage — a VALUE check with the same attribution
blind spot this module exists to close.

**No prior art known to me** for this exact composition: a
possessive-tolerant lexical entity proxy, clause-scoped attribution comparison
against the FIX-5 enumeration's own source spans, and an alias allowance
scoped to the single entity its edge registered.

## 10. Deviations from the order

None. The order's item 2 says "pick by evidence and state both (a) and/or
(b)"; I took **both**, with the evidence for each in §3 — that is inside the
"and/or", stated rather than assumed.

One thing the order did not anticipate, recorded rather than smoothed: the
receipt's window holds **two** alias turns (17 Kestrel/the Hauler as well as
18 Lantern/the Beacon), and the Kestrel pair was dropped from digest 24
entirely. Both are excluded by (a).

## 11. Process safety

Writable target `/mnt/ForgeRealm/wt/grm-f2` only; the canonical
`/mnt/ForgeRealm/GraftRepository` and the `artifacts/grm_d1/lt1_1/run_A`
session stores were READ only. No git command run — the lead commits. No
subagents. No process killed or signalled, none started that outlives the
session. No GPU used or reserved, no flock taken. Every pytest run used
`--basetemp /mnt/ForgeRealm/wt/grm-f2/artifacts/grm_f2/tmp`, removed
afterwards. Every path in `flag_contract.json` repo-relative.
