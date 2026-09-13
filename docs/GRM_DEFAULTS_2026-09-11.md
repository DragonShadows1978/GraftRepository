# GRM round-2 defaults — the decision, the evidence, the rollback

**Date:** 2026-09-11 · **Order:** `orders/GRM_D2_DEFAULTS.md` (GRM-D2) ·
**Branch:** `grm-d2`, forked from `lc1-wip`

David, 2026-09-11: *"You can make the best decision in your judgement. I can
roll it back if I have to."*

The lead's decision, executed here: **six settings that were measured
better become the shipped defaults, and every one of them keeps its old
value as an explicit setting, so rollback is one environment variable.**

---

## The one-line rollback

```bash
GRM_LEGACY_DEFAULTS=1
```

That restores EVERY pre-round-2 default at once: profile `legacy_256`,
`all_tokens_bind`, and F1 / F2 / F5 / A1 all OFF. It is implemented once, in
`core/grm_legacy_defaults.py`, and read by every resolver below.

It is a **default selector, not an override**. An explicitly set flag still
wins over it:

```bash
GRM_LEGACY_DEFAULTS=1 GRM_ALIAS_FOLD_MERGE=1   # legacy everything EXCEPT A1
```

That is deliberate — it is what makes the umbrella usable for bisecting
which of the six flips caused a regression, which is the reason an operator
reaches for it at all.

**Precedence, obeyed by every resolver:**

1. an explicit caller value (constructor / CLI);
2. the flag's own environment variable, when the operator set it;
3. `GRM_LEGACY_DEFAULTS`;
4. the round-2 shipped default.

---

## The flips

| # | resolver | pre-round-2 | round-2 default | old value still selectable as |
|---|---|---|---|---|
| 1 | `core/grm_admission.admission_rule` | `all_tokens_bind` | **`margin_first`** | `GRM_ADMISSION_RULE=all_tokens_bind` |
| 2 | `scripts/grm_profile.resolve_profile` | registry `default_flags` (unnamed) | **`eb1_c2`** | `GRM_PROFILE=legacy_256` |
| 3a | `core/grm_fold_retain.retain_sources_enabled` (F1) | OFF | **ON** | `GRM_FOLD_RETAIN_SOURCES=0` |
| 3b | `core/grm_fold_alias_guard.fold_alias_guard_enabled` (F2) | OFF | **ON** | `GRM_FOLD_ALIAS_GUARD=0` |
| 3c | `core/grm_admission.route_sole_binder_insurance_enabled` (F5) | OFF | **ON** | `GRM_ROUTE_SOLE_BINDER_INSURANCE=0` |
| 3d | `core/grm_alias_fold.alias_fold_enabled` (A1) | OFF | **ON** | `GRM_ALIAS_FOLD_MERGE=0` |

### Flip 1 — `margin_first` becomes the admission rule

**Evidence class: GPU campaign receipt (R1) + GPU campaign receipt (LT1/LT1.1).**

* `artifacts/grm_r1/summary_lead.json`: **status `ADOPT`**, `cells_measured`
  31, `answers_measured: true`, `off_plan_parity: true`,
  `correct_to_wrong: 0`, `wrong_to_correct: 1`, `unchanged_correct: 30`.
  The registered verdict rule was *"margin_first is adoptable as the profile
  default iff correct→wrong = 0 on sup and ≤ 1 elsewhere"* — it is met, and
  it was registered **before** the run, not chosen after seeing it.
* Against that: the old rule admits **0 of LT1's 35 natural questions**
  (`docs/GRM_CHAT.md` §measured). A rule that admits nothing on natural
  traffic is not a conservative default; it is a broken one.

The replay is the receipt that flipping costs no historical plan; the 0/35
is the receipt that not flipping costs every natural one.

### Flip 2 — the C2 profile becomes the shipped profile

**Evidence class: GPU battery (C2) + every LT1/LT1.1 campaign.**

`eb1_c2` is registry entry `gpt-oss-20b-eb1-w96-live-rt1`
(`config/grm_eb1_profile_registered.json`): arena width 96, capture pin
`live`, seat-near-live ON, RT1 ON. C2 beat the shipped defaults on every
battery and survived a restart, and **every LT1 and LT1.1 run — including
the r3 campaign flip 3 rests on — was taken under this profile.**

The old default set was never the measured configuration. It was the
un-measured one. After this order it has a name (`legacy_256`) instead of
being the unnamed fallback, because a default you can no longer reach by
doing nothing must be reachable by asking for it.

### Flip 3 — F1 + F2 + F5 + A1, **as a set**

**Evidence class: GPU campaign receipt (LT1.1 r3), three arms, same frozen
conversation and fixture sha.**

| arm | flags | primary total (recomputed from `run_*/cells/*/probes.jsonl`) |
|---|---|---|
| A0 (control) | all four OFF | **33/40** |
| A′ | F1+F2+F5, A1 OFF | **32/40** — a NET REGRESSION |
| A+′ | all four ON | **37/40** |

Per class (`artifacts/grm_f1/REPORT.md` §Scores):

| class | A0 | A′ | A+′ |
|---|---|---|---|
| fresh | 10/15 | 11/15 | **13/15** |
| corrections | 9/10 | 9/10 | **10/10** |
| aliases | 10/10 | **7/10** | 10/10 |
| recap | 4/5 | **5/5** | 4/5 |

**Never ship a subset.** A′ — three of the four, without the alias merge —
regresses aliases 10/10 → 7/10 and is a net regression against the control
on the verifiable column. The four are shipped together or not at all,
which is precisely what the single `GRM_LEGACY_DEFAULTS` switch enforces.

**A0 reproduces r2 arm A exactly on every class**, so the registration's
`attribution_rule` is satisfied: core drift is not in play and the movement
belongs to the flags. Within an arm four flags move TOGETHER — a moved
column belongs to the SET, never to one flag; that caveat is in the
registration and is repeated here rather than left to the reader.

#### A number in the order that this document does not repeat

The order (`orders/GRM_D2_DEFAULTS.md`, flip 3) states *"control 36/40 →
all four 38/40"*. Those are **amendment 9's second scoring column (c2),
reported by the lead and NOT persisted in any r3 artifact** — `probes.jsonl`
score objects carry `exact_correct` / `category` / the abstention fields and
no `col2_*` key, and `lead_*_summary.json` carries only `correct` / `n` /
`expected_n` / `exact_rate` per class. The F1 seat recorded that gap and
opened successor **F1-R3-S4** for it.

The tables above are therefore the **primary** column, which this tree can
recompute from the receipts. Both columns agree on direction and on the
subset regression; they disagree on two points of absolute total. The
decision rests on the column that has receipts.

---

## What "fail closed" means after this order

Every resolver still fails closed on an unknown token, and the rule it obeys
is unchanged: *a malformed operator escape must never silently select a
behaviour the operator may not have meant to select.*

What moved is **what it falls closed TO**: the SHIPPED default. Before D2
that was OFF / `all_tokens_bind`; after D2 it is ON / `margin_first`, and
under `GRM_LEGACY_DEFAULTS=1` it is OFF / `all_tokens_bind` again.

Concretely, `GRM_FOLD_RETAIN_SOURCES=banana` now resolves ON, where before
it resolved OFF. The three `env_*_override` helpers were changed to return
`None` (decline to decide) on an unknown token rather than `False` (pin
OFF) — pinning OFF was fail-closed only while OFF was what shipped.

This reversal is stated in every affected docstring, at the code site, and
not left to be discovered.

---

## Byte identity under the umbrella

The order required these to reproduce their recorded values with
`GRM_LEGACY_DEFAULTS=1`. They do:

| gate | result |
|---|---|
| C2 132-plan replay (`scripts/grm_f5_c2_replay_gate.py`) | `rows: 132`, `off_byte_identical: 132`, `verdict: PASS` |
| FIX-4 frozen whole-receipt control (`tests/test_grm_scout_fix4.py::test_nonrecency_existing_fixture_receipt_byte_identical`) | PASS against `artifacts/grm_scout_fix4/nonrecency_before.json` |
| F1 all-off fingerprint (`tests/test_grm_f1_fold_retain.py::test_flag_off_byte_identical`) | PASS |
| A1 all-off fingerprint (`tests/test_grm_a1_alias_fold.py::test_flag_off_byte_identical`) | PASS |
| OFF arms of every F1 / F2 / F5 / A1 flag test | PASS |

No frozen receipt on disk was re-blessed. Not one byte of
`nonrecency_before.json`, the C2 checkpoints, or any recorded plan moved.

---

## Tests that were re-pinned

These encoded the OLD defaults in their expectations. Per the order, each
was changed **only** by pinning `GRM_LEGACY_DEFAULTS=1` — their assertions
stay. The full list is in the D2 seat report and in each file's
`GRM-D2 re-pin` comment block.

Two assertions did change, and both are flagged in-place as real behavioural
consequences rather than re-pins:

1. `tests/test_grm_f1_resume_out_root.py::test_treatment_flags_cross_into_the_leased_child`
   — the LT1.1 OFF arm used to express itself by leaving
   `GRM_ALIAS_FOLD_MERGE` ABSENT. After D2, absent means the treatment, so
   the arm is now pinned EXPLICITLY (`grm_lt1_1.arm_alias_pin`). The
   property tested (the arm survives the `GRM_*` strip into the leased
   child) is unchanged; only its expression moved from absence to `'0'`.
   **An arm is a pin, never an absence: a campaign arm that inherits a
   default is not a controlled arm.**
2. `tests/test_grm_chat.py::test_profile_unset_is_todays_behaviour` — split
   into a `legacy_256`-under-the-umbrella test (every flag assertion
   unchanged) plus a new one pinning the post-D2 side. `GRM_ADMISSION_RULE`
   is now pinned rather than popped on the legacy side, because popping it
   would put a margin-first rule inside a legacy frame.

---

## Known RED at the time of this decision

**Campaign-receipt tests that are sha-bound to the LT1.1 r3 registration
fail on this branch, because D2 moves the core files that registration
pinned.** They are receipts of their campaign's day, not regressions of the
tree: `tests/test_grm_f1_fold_retain.py::test_governing_amendment_rebinds_every_core_pin`
and the three
`tests/test_grm_f1_resume_out_root.py::test_real_resume_route_through_pending_and_one_run_cell`
arms, the latter blocked by `LT11_CHILD_PREFLIGHT_BLOCKED:
INPUT_SHA_MISMATCH` on the four edited core files plus `RUNNER_SHA_MISMATCH`.
A further 16 in `tests/test_grm_f1_r3_receipts.py` were already RED before
this order (the gitignored `run_*` checkpoint trees are absent from this
worktree). None was touched: the order's rule is campaign receipts untouched.

**Re-binding them is a registered campaign act (a new amendment), not a
default flip, and it is the lead's to make.**

---

## Prior art

* The umbrella is the ordinary "one switch restores the previous release's
  behaviour" pattern — Django transitional settings (e.g.
  `DEFAULT_AUTO_FIELD`, 2020-), glibc `GLIBC_TUNABLES`, Linux
  `CONFIG_COMPAT_*` / `sysctl` compatibility toggles, Windows AppCompat
  shims. TAKEN: the shape (group a release's behaviour changes behind one
  named rollback; let an individually-set flag still win, so the rollback
  can bisect). NOT taken: any mechanism. UNVERIFIED (no network in the
  seat sandbox) — lead to check; search terms: `feature flag kill switch
  rollback default`, `compatibility switch restore previous release
  defaults`, `Django transitional settings`.
* The precedence ladder and the fail-closed-on-unknown-token rule are TAKEN
  VERBATIM from this tree's own resolvers —
  `grm_supersession.env_sup_resolve_override` (L2, default ON),
  `grm_admission.env_adm_decisive_override` (A-DEC, default ON),
  `grm_alias_fold.env_alias_fold_override` (A1, previously default OFF) —
  GRM contributors, 2026. Only the fourth rung is new.
* `arm_alias_pin`'s pin-then-read-back stance is R1 `pin_rule`
  (`scripts/grm_r1_replay.py`, GRM contributors 2026), taken verbatim.
  "Controls are set, not defaulted" is ordinary experimental practice; no
  specific prior art known to me.
* No new routing, admission, folding or scoring algorithm is introduced by
  this order. Nothing here is a model-quality claim by the implementing
  seat: every number above is read from a prior campaign's receipts and is
  labelled with the receipt it came from.
