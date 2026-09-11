# GRM-F2 — LEDGER (receipts, as they happened)

Seat: Opus 5 (`claude-opus-5[1m]`), reasoning effort MAX.
Worktree `/mnt/ForgeRealm/wt/grm-f2` (branch `grm-f2`). Date: 2026-09-11.
Order: `orders/GRM_F2_CHRONICLE_ALIAS_GUARD.md`.
No git run by this seat. No subagents. No GPU. No foreign process signalled.

---

## 1. Reading the receipt off disk

The r2 arm-A session repositories were copied into the worktree at
`artifacts/grm_d1/lt1_1/run_A/cells/*/session/repository`. Digest node 24
lives in the `A-025-032` cell manifest (the `A-017-024` cell has only 23
nodes and the fold had not yet happened there).

```
$ python3 -c "... manifest.json nodes[24] ..."
=== node 24 digest ntok 39 sources [13, 14, 17, 18]
ARCHIVE NOTE. For the archive: the  maintenance crew of the Beacon is
located at Iona Vale, and the map position of the Beacon is (-31, 48, 12).
```

Its four sources, verbatim (user spans):

| node | ntok | user span |
|---|---|---|
| 13 | 49 | `Commtower's maintenance crew will be Iona Vale.` |
| 14 | 53 | `Breakwater's map position will be (-31, 48, 12).` |
| 17 | 66 | `Let's call Kestrel 'the Hauler' from now on; …` |
| 18 | 63 | `Let's call Lantern 'the Beacon' from now on; …` |

**Note the window holds TWO alias turns, not one.** The round-1 report names
turn 18; turn 17 is an alias turn as well and its pair (Kestrel / the Hauler)
was dropped from the digest entirely.

Frozen for the fixtures at `artifacts/grm_f2/receipt_digest24.json`
(source texts + digest text + ntok, copied — the canonical store is read-only
and was never written to).

---

## 2. The mechanism, in three measurements

### 2.1 The alias turns contribute no facts and are not enumerated

```
$ python3 -c "ArenaCache._fact_set(srcs)"
need ['12', '31', '48', 'iona', 'vale']   (5 facts)

$ FIX-5 enumeration over the window [13,14,17,18]:
[source 1] Commtower's maintenance crew will be Iona Vale.
[source 2] Breakwater's map position will be (-31, 48, 12).

$ FIX-5 enumeration over [13,14] only:
[source 1] Commtower's maintenance crew will be Iona Vale.
[source 2] Breakwater's map position will be (-31, 48, 12).
```

**Identical.** `_fact_set([node 17])` and `_fact_set([node 18])` are both the
EMPTY set, so `_consolidation_prompts`' own filter
(`if self._fact_set([span]) & need`) drops both alias spans. The instruction
the model receives never mentions Lantern, Kestrel, the Beacon or the Hauler.

Gate: `tests/test_grm_f2_alias_guard.py::test_alias_turns_contribute_no_facts_and_no_enumeration`.

### 2.2 They are mounted anyway

`ArenaCache.consolidate` (core/graft_arena.py:2241-2248) mounts EVERY source
in `idxs` through `_set_inject` before generating. So "the Beacon" is in the
model's attention window while the instruction names only crews and
coordinates. Zero instructional weight, full attention weight — the alias
turns are pure contamination surface in this window.

### 2.3 The coverage gate is blind to it

```
$ ArenaCache._coverage(digest24_text, need)
1.0
$ sorted(need - (_rare_tokens(d) | _caps_tokens(d, False)))
[]
```

`best_cov = 1.0`, `hit_count = 5/5`, accepted at perfect fidelity — while
every fact in the digest is filed under the wrong entity.

**Why `need` holds no entity name.** Two independent blind spots:

* `_fact_set`'s multi-word rule needs >= 2 consecutive capitalized words.
  "Iona Vale" qualifies (and is therefore a VALUE); "Commtower" alone does not.
* `_caps_tokens` matches `^[A-Z][\w\-]+$` against a whitespace-split token:

```
$ re.match(r'^[A-Z][\w\-]+$', w.rstrip('.,;:'))
Commtower's   False      <- the possessive clitic breaks the match
Breakwater's  False
Kestrel       True
Lantern       True
Beacon        True
```

Both rebound entities appear in POSSESSIVE form in their source spans and are
therefore invisible to the existing vocabulary, while "Beacon" — the wrong
name — is visible. A coverage gate over values alone cannot see attribution.

Gate: `::test_coverage_is_blind_to_attribution`.

### 2.4 The cost of accepting it

`_deposit_consolidation` retires all four sources. So turns 17 and 18 — the
only record that "the Beacon" means Lantern — were destroyed to produce a node
that misuses that alias. Round-1 receipt: grepping the arm-A store for
"Breakwater" finds 1 node (the retired source); arm A+, whose digest 28 folded
`[13, 14, 21, 22]` with no alias turn in the window, holds 3.

---

## 3. The choice: (a) AND (b), and the evidence for each

The order says "pick by evidence and state both". Both were prototyped
against the receipt before either was written into core.

**(a) exclusion is the CAUSAL fix and it is free.** §2.1 measures that
removing the alias turns changes neither `need` nor the enumeration — the
digest the fold is asked to write is unchanged — while the contaminating
mount is gone. Cost: nothing the fold was asked to preserve.

**(b) attribution QC is the DETECTOR and it is not redundant.** (a) removes
one known contamination source; (b) is the standing check that the digest
says what the sources said. A rebind can also arrive with no alias turn in the
window at all (a model simply confusing two entities), and (a) is silent
there. §2.3 is the blind spot (b) closes.

**Prototype results, before any core edit:**

| arm | digest 24 | 13 FIX-5 GPU-contrast digests |
|---|---|---|
| (b) attribution check | **5 violations** | **0 violations** |
| (a) window exclusion | window `[13,14,17,18]` -> `[13,14]` | **0 sources dropped** |

The five violations:

```
{'source': 0, 'entity': ['commtower'],  'value': 'iona', 'digest_entities': ['beacon','iona','vale']}
{'source': 0, 'entity': ['commtower'],  'value': 'vale', 'digest_entities': ['beacon','iona','vale']}
{'source': 1, 'entity': ['breakwater'], 'value': '12',   'digest_entities': ['beacon']}
{'source': 1, 'entity': ['breakwater'], 'value': '31',   'digest_entities': ['beacon']}
{'source': 1, 'entity': ['breakwater'], 'value': '48',   'digest_entities': ['beacon']}
```

Zero false positives across all 13 real GPT-OSS-20B digests recorded under
`artifacts/grm_scout_fix5/gpu/*/receipt.json` — the detector does not cost an
accepted fold. Taking only (a) would leave §2.3's blind spot open; taking only
(b) would keep paying a fold abort, and the retirement of the alias turns, for
a contamination we can simply not create.

---

## 4. The entity check, defined

> Every (entity, attribute, value) in the sources must appear in the digest
> with the SAME entity, or with an alias registered for THAT entity only.

Mechanically, per source clause that has both an owning entity and >= 1 value:

1. **Owner** = the clause's entity tokens MINUS its value tokens. (A
   multi-word value like "Iona Vale" is entity-shaped AND a `_fact_set` fact;
   it is the value, not the owner, so removing it stops every value from
   vacuously owning itself.)
2. For each **value** in the clause, collect the digest clauses carrying it.
   A value the digest does not carry at all is **SKIPPED** — that is coverage's
   question and `MIN_FOLD_KEEP` already answers it.
3. The value **passes** if ANY carrying digest clause names an accepted name
   for the owner. **Accepted names** = the entity itself plus aliases
   registered for that entity, scoped to the one binding — "the Beacon" stands
   for Lantern and for nothing else.
4. A clause with **no owner** (subject in a previous sentence) raises nothing.

Entity proxy: a capitalized word, POSSESSIVE-TOLERANT
(`(?<![\w'’-])([A-Z][\w-]*)(?:['’]s)?\b`), FIX-8/SC1.1 glyph-normalized,
minus a small closed list of structural words (`ENTITY_STOP`). The possessive
tolerance is load-bearing — see §2.3; a check built on the existing
`_caps_tokens` class would be blind to exactly the two sources it must protect.

Alias detection is **A1's `parse_alias_edge`, unchanged** — F2 introduces no
second alias detector.

Clause granularity is `[.!?;\n]+` plus `,\s+(?:and|while|but)\s+`. That last
arm is what separates the receipt's own two conjoined clauses; without it the
digest's entities pool and either name would be accepted for either value.

---

## 5. Core diff

### `core/grm_fold_alias_guard.py` (NEW, 440 lines)

Flag resolution (`ENV_NAME = GRM_FOLD_ALIAS_GUARD`, default OFF, unknown
token fails CLOSED), the entity proxy, the alias table, `fold_window_excludes`
(a), `attribution_violations` / `guard_receipt` (b). Full defect narrative,
"when this guard stops applying", and the prior-art block live in its module
docstring.

### `core/graft_repository.py`

| lines | change |
|---|---|
| 54-55 | import `grm_fold_alias_guard` + `fold_alias_guard_enabled` |
| 262 | ctor kwarg `fold_alias_guard=None` |
| 332-344 | resolve the flag ONCE (`self.fold_alias_guard`), init `self.fold_guard_history = []` |
| 359-364 | install `arena.fold_attribution_guard` **only when the flag is ON** |
| 4104-4107 | route the FALLBACK librarian plan through `_guard_fold_windows` |
| 4115 | route the NATIVE librarian plan through `_guard_fold_windows` |
| 4118-4168 | `_guard_fold_windows` (a) — returns `jobs` unchanged by identity at line 4148 when OFF |
| 4170-4202 | `_fold_attribution_receipt` (b) — the hook body |
| 4204-4217 | `_record_fold_guard` — dedup'd receipt, same shape as `_record_alias_decision` |

Both `_librarian_jobs` return paths go through the guard, so the treatment
cannot depend on which planner ran.

### `core/graft_arena.py`

| lines | change |
|---|---|
| 1934-1948 | class attribute `fold_attribution_guard = None` (permanent default) + why the seam is here and not in `_fold_once` |
| 2249-2250 | extractive-era deposit site: check before deposit |
| 2359-2365 | generated deposit site: check before deposit |
| 2369-2386 | `_fold_attribution_ok` — returns True when no hook is installed |

The hook runs **immediately before `_deposit_consolidation`**, which is the
method that retires the sources. A check after it would have to reverse a
retirement and a deposit; here a rejected digest was never written and no node
ever changed state. Rejection returns through `consolidate`'s EXISTING
`(None, None)` abort path, so `_fold_once` takes the branch it already had —
no new failure mode.

---

## 6. Gates

```
$ python3 -m pytest -q --basetemp .../artifacts/grm_f2/tmp \
    tests/test_grm_f2_alias_guard.py
2 failed, 32 passed     # first run: a double space in the receipt text and a
                        # not-yet-written LEDGER.md path — both fixture-side
```

### 6.1 RED proof

Both treatments neutered in place (an early `return []` in
`fold_window_excludes` and in `attribution_violations`, module still
importable), then re-run:

```
6 failed, 28 passed        -> artifacts/grm_f2/cpu_red.log
```

The six: `test_red_flag_off_deposits_the_rebound_digest`,
`test_green_flag_on_b_rejects_the_rebound_digest`,
`test_green_flag_on_a_excludes_the_alias_turns_from_the_window`,
`test_green_end_to_end_through_fold_once`, `test_entity_check_definition`,
`test_exclusion_keeps_a_fact_bearing_alias_turn`.

Module restored from backup and verified BYTE-IDENTICAL (`diff` clean, zero
`RED PROOF` markers remaining), then re-run:

```
35 passed                  -> artifacts/grm_f2/cpu_green.log
```

### 6.2 The regression RED I caused, and the fix

First run of the fold/librarian/repository/arena regression set:

```
8 failed, 24 passed, 1 skipped   -> artifacts/grm_f2/regression_red.log
AttributeError: 'GraftRepository' object has no attribute 'fold_alias_guard'
  at core/graft_repository.py:4148
```

Cause, mine: `tests/test_grm_s4_fold_order.py::_planner_repo` builds a
planner-only stub with `object.__new__(GraftRepository)` and a minimal
attribute set, never running `__init__` — an existing idiom `_librarian_jobs`
already honours for `fold_order` via `getattr`. My guard read
`self.fold_alias_guard` directly.

Fix: the same `getattr(self, "fold_alias_guard", False)` idiom in
`_guard_fold_windows`, plus a missing-history tolerance in
`_record_fold_guard`. Pinned by
`::test_planner_only_stub_keeps_the_pre_f2_plan`. Re-run:

```
32 passed, 1 skipped             -> artifacts/grm_f2/regression.log
```

After correcting the two fixture-side assertions, the full battery:

```
$ python3 -m pytest -q --basetemp /mnt/ForgeRealm/wt/grm-f2/artifacts/grm_f2/tmp \
    tests/test_grm_f2*.py tests/test_grm_scout_fix3*.py \
    tests/test_grm_scout_fix5*.py tests/test_grm_a1_alias_fold.py
```

Result line recorded in `artifacts/grm_f2/REPORT.md` §Gates.

`--basetemp` removed after every run.

---

## 7. Prior art

Annotated at the code site (`core/grm_fold_alias_guard.py` PRIOR ART block),
here, and in `REPORT.md`.

**External — all UNVERIFIED (no network in this sandbox), lead to check.**

* **Maynez et al. (2020), "On Faithfulness and Factuality in Abstractive
  Summarization" (arXiv 2005.00661).** TAKEN: the framing that a summary can
  score perfectly on token overlap while being unfaithful (EXTRINSIC
  hallucination), which is exactly §2.3 — `best_cov = 1.0` on a rebound
  digest. NOT taken: their entailment-model metric; this module uses no model.
  Search terms: `Maynez 2020 faithfulness factuality abstractive
  summarization`, `extrinsic hallucination summarization`.
* **Nan et al. (2021), "Entity-level Factual Consistency of Abstractive
  Summarization" (arXiv 2102.09130).** TAKEN: the unit of measurement — the
  ENTITY, not the token — and checking it mechanically against the source.
  NOT taken: their NER pipeline or training-time filtering; our proxy is
  lexical and the check is a hard gate on one fold, not a corpus statistic.
  Search terms: `Nan 2021 entity-level factual consistency summarization`,
  `entity hallucination precision summarization`.
* **Goodrich et al. (2019), "Assessing The Factual Accuracy of Generated
  Text" (KDD).** TAKEN: the (subject, relation, object) triple as the
  comparison unit — which is the order's own entity check. NOT taken: their
  learned relation extractor. Search terms: `Goodrich 2019 factual accuracy
  generated text`, `relation triple summarization factual consistency`.

**Local, verified, reused UNCHANGED as the verbs this composes.**

* FIX-5 source-enumerated consolidation and its `_fact_set` fidelity
  vocabulary (`core/graft_arena.py:2142-2155, 1955-1978`).
* FIX-3's "reject rather than deposit a lossy derivative" abort contract and
  its `no_fold` exemption (`GraftRepository._fold_once`).
* A1's `parse_alias_edge` / `alias_scan_text` / `identifier_set`
  (`core/grm_alias_fold.py`) — the alias detector here is A1's, not a second
  one.
* SC1.1 / FIX-8 `normalize_glyphs` (`core/grm_text_norm.py`) so this scan sees
  the projection routing and admission see.
* SCOUT-FIX-9's NEVER-SILENT receipt rule and `_record_alias_decision`'s
  consecutive-dedup idiom.
* SC1.1 grounding (`ArenaCache._grounding_verdict`) is the nearest local
  antecedent for "check the produced text against what was mounted". TAKEN:
  the stance. NOT taken: its pooled token coverage, which is a VALUE check
  with the same attribution blind spot this module exists to close.
* The hook shape is `ArenaCache`'s own optional-seam idiom (`node_loader`,
  `native_store`), reused.
* Fixture idioms from `tests/test_grm_scout_fix5.py` (the `EnumeratedModel`
  prompt-reading stimulus, the byte-identity record) and
  `tests/test_grm_a1_alias_fold.py` (`seed` + `_sync_lifecycle`, the
  flag-off byte-identity comparison shape).

**No prior art known to me** for this exact composition: a possessive-tolerant
lexical entity proxy, clause-scoped attribution comparison against the FIX-5
enumeration's own source spans, and an alias allowance scoped to the single
entity its edge registered.

---

## 8. Process safety

* Writable target: `/mnt/ForgeRealm/wt/grm-f2` only. The canonical
  `/mnt/ForgeRealm/GraftRepository` and the `artifacts/grm_d1/lt1_1/run_A`
  session stores were READ only.
* No git command run.
* No subagents spawned.
* No process killed or signalled; none started that outlives the session.
* No GPU used, none reserved, no flock taken.
* Every pytest run used
  `--basetemp /mnt/ForgeRealm/wt/grm-f2/artifacts/grm_f2/tmp`, removed
  afterwards.
* Every Bash call foreground, under 10 minutes.
* Every path in `flag_contract.json` repo-relative
  (`::test_control_flag_contract_is_registered` asserts it).
