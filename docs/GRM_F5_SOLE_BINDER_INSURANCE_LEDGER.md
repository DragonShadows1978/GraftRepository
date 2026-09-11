# GRM-F5 ledger — sole-binder insurance in the admission plan

Order is immutable: `orders/GRM_F5_SOLE_BINDER_INSURANCE.md`.
Base: F3's finding, `docs/GRM_F3_ROUTING_AT_DISTANCE_LEDGER.md`, "The proven
defect (registered, NOT landed)" and its registered proposal.

Seat: Opus 5 (model id `claude-opus-5[1m]`), reasoning effort max.
Writable target: `/mnt/ForgeRealm/wt/grm-f5` (branch `grm-f5`).
No git. No subagents. No GPU run. No process signalled or killed.
Every command foreground and under 10 minutes. `--basetemp` under
`artifacts/grm_f5/tmp`, removed after every pytest invocation.

## Verdict in one line

**THE FIX LANDS, flag-gated and default OFF.** F3's registered proposal is
implemented in both admission rules behind
`GRM_ROUTE_SOLE_BINDER_INSURANCE`; OFF is byte-identical on all three gates
the order names (C2 132/132, FIX-4/FIX-6, F3's pinned-defect tests, all
unedited); F3 row 10's RED→GREEN is a fixture.

## Prior art

- **RT1** — `core.grm_admission.demote_non_binding_split_members` (GRM
  contributors, 2026), 200 lines below the change in the same file.
  TAKEN VERBATIM: the reorder-never-score stance, and its structural
  discipline of enumerating the no-op conditions with the rule and moving
  only the node the rule is named for. OURS: nothing of the stance; only its
  application to the two identifier-insurance branches.
- **Maximal Marginal Relevance** — Carbonell & Goldstein, SIGIR 1998.
  TAKEN: the general shape of reordering an already-scored ranked list under
  a secondary criterion without rescoring. NOT TAKEN: MMR's diversity
  objective and its λ — F5 introduces no tunable and no threshold. F3 cited
  the same paper for the same reason when it registered the proposal.
- **Characterization tests** — Feathers, *Working Effectively with Legacy
  Code* (2004), ch. 13. TAKEN: the practice, and specifically the rule that
  a fix ADDS a counterpart rather than editing the pin. Applied literally:
  `tests/test_grm_f3_routing_at_distance.py` is not touched, and the
  ON-arm counterparts live in the F5 module named after the tests they
  answer.
- **The frozen A-DEC rule and FIX-6's margin-first tie-break** (GRM
  contributors, 2026) — consumed unchanged. F5 runs AFTER FIX-6's exact
  tie-group reorder, never instead of it; pinned by
  `test_control_tie_group_reorder_still_runs_and_then_insures`.
- **The FIX-6 C2 replay harness** — `scripts/grm_scout_fix6_replay.py` and
  `scripts/grm_lt1_offline_supplement.c2_replay` (GRM contributors, 2026),
  reused wholesale for the OFF byte-identity gate; the only new code is the
  path rebinding described under "Deviations" and the receipt shape.
- **The arena double** — `_ProfileArena` in `tests/test_grm_admission.py`
  (GRM contributors, 2026); shape reused, scores new.
- **UNVERIFIED** against the wider literature (no network in this sandbox) —
  lead to check. Search terms: "must-include constraint top-k retrieval",
  "constrained re-ranking guarantee matched entity", "hard inclusion
  constraint re-ranking", "maximal marginal relevance reorder",
  "characterization test legacy defect pin".

## The core diff

One file, `core/grm_admission.py`, four sites. `git` was not run; line
numbers are from the post-edit file.

| site | lines | what |
|---|---|---|
| new block after `ADMISSION_RULE_ENV` | 78–186 | the rationale + prior-art comment (78–133), `ROUTE_SOLE_BINDER_INSURANCE_ENV` (135), the two branch-name constants (140–141), `route_sole_binder_insurance_enabled()` (144), `insure_sole_binder()` (160) |
| `margin_first_plan` | 214–250 | signature + new `sole_binder_insurance` keyword (214–220); `identified_candidates` materialized before its first read (227–231); the `margin_insurance_k3_identifier_tiebreak` return routed through `insure_sole_binder` (239–250) |
| `policy_plan` | 493–537 | signature + new `sole_binder_insurance` keyword (493–500); the `one_off_rank_identifier_insurance_k3` return routed through `insure_sole_binder` (527–537) |
| `decisive_admission_profile` | 733–744 | `sole_binder_inserted` emitted ONLY when the flag is ON |

The rule itself, in full:

```python
def insure_sole_binder(*, plan, identified_candidates, enabled):
    planned = [int(value) for value in plan]
    if not enabled:
        return planned, False
    identified = [int(value) for value in identified_candidates]
    if len(identified) != 1 or not planned:
        return planned, False
    binder = identified[0]
    if binder in planned:
        return planned, False
    return [*planned[:-1], binder], True
```

No threshold, nothing refit, no score read. It substitutes for the plan's
LAST slot — the node the ranker trusted least of the three — and never
widens a plan.

### Two decisions worth naming

1. **`identified_candidates` is now read twice** in `margin_first_plan`
   (once into `hits` for the tie group, once by the insurance). Its declared
   type is `Iterable`, so a generator argument would have yielded an empty
   second pass and made the flag a silent no-op for such a caller. It is
   materialized before the first read. Pinned by
   `test_generator_identified_candidates_are_not_consumed_twice`.
2. **The `sole_binder_inserted` receipt key is emitted only when the flag is
   ON.** The house law is NEVER SILENT (P2A/P2C/RT1), and the order asks for
   the field; but `tests/test_grm_scout_fix4.py::test_nonrecency_existing_
   fixture_receipt_byte_identical` compares EVERY legacy receipt byte against
   a frozen fixture, so an unconditional additive key would have broken OFF
   byte-identity. The resolution: NEVER SILENT *within the flag's own world*
   — ON, the key is on every profile, true or false; OFF, it is absent, and
   its absence is itself the "flag was OFF" marker. Pinned by
   `test_receipt_off_is_byte_identical_and_carries_no_f5_key` and
   `test_receipt_on_is_never_silent_when_nothing_was_inserted`.

## OFF byte-identity (the order's item 1)

| gate | result |
|---|---|
| C2 132-plan replay, flag UNSET | `off_byte_identical: 132 / 132`, verdict PASS — `artifacts/grm_f5/c2_replay_off.json` |
| C2 132-plan replay, flag `=0` | `off_byte_identical: 132 / 132`, verdict PASS |
| C2 132-plan replay, PRE-EDIT baseline | `off_byte_identical: 132 / 132` — `artifacts/grm_f5/c2_replay_pre_edit.json` |
| FIX-4 suites (`tests/test_grm_scout_fix4*.py`) incl. the frozen-receipt-bytes test | pass, unedited |
| FIX-6 suites (`tests/test_grm_scout_fix6*.py`, `tests/test_grm_lt1_fix6.py` collected by the glob) | pass, unedited |
| F3 characterization suite incl. the three DEFECT-PINNED tests | pass, unedited |
| `tests/test_grm_admission.py` | pass, unedited |

`core/grm_admission.py` sha256 before the edit:
`ebdfd84af435a163dfe4a37281d489047acd974bf76190b79622340c29af7ab6`
(`artifacts/grm_f5/core_sha_before.txt`).

## The registered fixture (the order's item 2)

F3 row 10 = LT1.1 r2 arm A+ `recall_3_150`, turn 164, verbatim from
`artifacts/grm_f3/route_rows.json[9]`. Pinned against that file by
`test_f3_row_fixture_values_match_the_frozen_route_table`, so the fixture
cannot drift from the receipt and still pass.

| arm | flag | `margin_first_plan` | `policy_plan` |
|---|---|---|---|
| RED | OFF (unset, `0`, or an unknown token) | `[168, 132, 116]`, branch `margin_insurance_k3_identifier_tiebreak`, node 64 ABSENT | `[168, 132, 116]`, branch `one_off_rank_identifier_insurance_k3`, node 64 ABSENT |
| GREEN | ON (`1`) | `[168, 132, 64]`, branch `margin_insurance_k3_sole_binder_inserted` | `[168, 132, 64]`, branch `one_off_rank_identifier_insurance_k3_sole_binder_inserted` |

Row 9 (`recall_3_100`, binder at rank 5 rather than unranked) moves the same
way: `[116, 110, 42]` → `[116, 110, 64]`.

Controls, all with the flag ON, all asserting ON == OFF:

| control | why it must not move |
|---|---|
| sole binder already at rank 1 | `policy_plan` takes `exactly_one_identifier_decisive_rank1` — a SINGLETON the flag must not widen; `margin_first_plan` has it in the window already |
| sole binder already the plan's LAST slot | substituting would be a self-eviction |
| ≥ 2 binders, `policy_plan` | `declared_synthesis_identified_set` already admits them all |
| ≥ 2 binders, `margin_first_plan` | out of scope — see the residual below |
| zero binders | nothing to insure |
| decisive margin (0.322), binder on- or off-rank | a decisive margin is a deliberate singleton, not an insurance window |
| empty ranking | no slot to spend |
| exact-score tie group with `scores` supplied | FIX-6's tie-break runs FIRST and is byte-identical; here it already seats the binder, so F5 adds nothing |

And the mechanism test that separates the two rules:
`test_tie_break_cannot_reach_an_untied_binder_and_f5_can` — with node 64
scored 0.1 against a rank-1 of 1.0 there is no tie to break, so the branch
named `identifier_tiebreak` provably cannot move it OFF, and does ON.

ON-arm counterparts to F3's three pinned tests are in the F5 module, named
`test_counterpart_to_f3_*`. The order makes
`tests/test_grm_f3_routing_at_distance.py` READ-ONLY and item 2 asks for
counterparts "rather than editing the pinned ones"; both point the same way,
and the F3 file is byte-unchanged. Each counterpart re-executes the pinned
OFF assertion verbatim before asserting the ON one, so a drift in the pin is
caught in this module too.

## Blast radius, measured before any GPU run

- **C2, the 132 recorded executions**: 112 evaluable (20 left UNRESOLVED, not
  imputed — FIX-6's own discipline). ON changes **0** of 112 plans.
  `artifacts/grm_f5/c2_replay_on_risk.json`.
- **The 10 F3 route rows**: exactly **2** change (rows 9 and 10, both arm A+
  Breakwater probes). F3's F4 rescore already scored row 9's served answer
  **correct**, so row 10 is the single row with something to gain — which is
  precisely what F3 predicted when it registered the proposal ("Expected
  effect on the r2 receipts: row 10 only").
- **Risk named before the run**: the rule SPENDS the plan's last slot. On a
  row where the ranker's third choice would have been served and the binder
  would not, F5 makes the answer worse. Zero such rows exist in the 112
  evaluable C2 executions or the 10 F3 rows, but the mechanism is real and a
  regression on r3 is a RESULT, not something to tune away afterwards.

## RED / honest limits

1. **This is a PLAN claim, not an answer claim.** Being in `rank_plan` is
   not being seated, mounted, grounded, or served. F3 row 10 recorded
   `width 96`, `summed_seats 90`, `unseatable [132, 116]`; a plan of
   `[168, 132, 64]` may still fail to co-seat node 64. Every gate in this
   item is a pure-CPU replay of the admission rule over frozen rankings and
   margins. No router, reader or scorer was re-run, and no model-quality
   claim is made.
2. **The ≥ 2 binder asymmetry survives in `margin_first_plan`.** That rule
   has no `declared_synthesis_identified_set` branch at all: with two
   binders it returns `ranking[:3]` and can admit NEITHER. F3's
   `test_two_or_more_binders_are_all_admitted_regardless_of_rank` measured
   the asymmetry on `policy_plan`, where it is now closed; one rule higher
   it is not. F5 leaves it exactly as found — the order scopes this flag to
   the SOLE-binder case, and widening `margin_first` to admit an arbitrary
   identified set is a different rule with a different blast radius (it can
   grow a plan past three) needing its own order and its own RED→GREEN.
   REGISTERED, not landed:
   `artifacts/grm_f5/flag_contract.json:known_residual_not_fixed_by_this_item`,
   pinned by `test_control_two_binders_under_margin_first_is_also_unchanged`.
3. **F5 cannot help the rows F2 owns.** It consumes the identifier scan's
   verdict and never widens it, so an identifier the scan misses — which is
   exactly F3's alias-capture class, rows 2/3/4 — is invisible to it. F5
   fixes 1 of the 7 r2 miss rows and claims nothing about the other 6.
4. **The lead's r3 arm choice matters.** The lead's plan (L1) pins LT1.1 r3
   to **arm A only**. Row 10 is an **arm A+** row, and on arm A the r2
   receipts do not exercise the sole-binder case at all. A lead who wants
   F5's registered prediction tested must also run the A+ probes, or accept
   that arm A alone can neither confirm nor refute it. Named in the flag
   contract under `what_the_lead_pins.arm`.
5. **The C2 gate needed a path rebinding to run at all** — see Deviations.
   The 132/132 number is real and reproduced twice (pre- and post-edit), but
   it was produced by a harness this seat had to repair, not by running the
   FIX-6 script untouched.

## Deviations from the order

1. **The C2 replay harness points at a deleted worktree.**
   `scripts/grm_lt1_offline.py:18` hard-codes
   `C2 = Path('/mnt/ForgeRealm/wt/grm-c2/artifacts/grm_c2/epochs/scout-fix-2')`.
   That worktree no longer exists, so `c2_replay()` silently returns **0
   rows** — a gate that passes vacuously. The same epoch data is present in
   this worktree at `artifacts/grm_c2/epochs/scout-fix-2` (52 worker.json
   cells, both checkpoint sides). Rather than edit a module the order does
   not authorize, the gate script `scripts/grm_f5_c2_replay_gate.py` REBINDS
   the module constant in memory, read-only, and asserts `rows == 132`
   explicitly so the vacuous-zero failure mode cannot recur silently.
   **Lead action**: `scripts/grm_lt1_offline.py:18` should become a
   repo-relative path before round 3 reuses it; every other consumer of that
   constant is currently reading a dead absolute path.
2. **`artifacts/grm_f5/` did not exist**, so the first pytest invocation
   errored at setup (`--basetemp` cannot create a missing parent). Created
   with `mkdir -p`; the directory is a deliverable anyway.
3. **No new `campaign_receipt`-marked tests.** Every F5 fixture is pure CPU
   on literal values plus two committed JSON receipts, so none of them reads
   a gitignored r2 campaign receipt and none needs the marker or the
   sha-binding. The two receipt-pinning tests read files this seat wrote.

## Commands run (all foreground, all < 10 min)

```
python3 -m pytest -q --basetemp /mnt/ForgeRealm/wt/grm-f5/artifacts/grm_f5/tmp \
  tests/test_grm_f3_routing_at_distance.py tests/test_grm_admission.py \
  tests/test_grm_scout_fix4.py tests/test_grm_scout_fix6*.py
      -> 40 passed, 22 skipped   (BASELINE, before core was touched)

python3 scripts/grm_f5_c2_replay_gate.py
      -> 132/132 PASS            (PRE-EDIT baseline)

env -u GRM_ROUTE_SOLE_BINDER_INSURANCE python3 scripts/grm_f5_c2_replay_gate.py
      -> 132/132 PASS            (POST-EDIT, flag unset)
env GRM_ROUTE_SOLE_BINDER_INSURANCE=0 python3 scripts/grm_f5_c2_replay_gate.py
      -> 132/132 PASS            (POST-EDIT, flag 0)

(ON-arm blast radius, inline replay over the same 132 rows)
      -> 112 evaluable, 20 unresolved, 0 plans changed

python3 -m pytest -q --basetemp /mnt/ForgeRealm/wt/grm-f5/artifacts/grm_f5/tmp \
  tests/test_grm_f5*.py tests/test_grm_f3_routing_at_distance.py \
  tests/test_grm_admission.py tests/test_grm_scout_fix4.py \
  tests/test_grm_scout_fix6*.py
      -> the order's verbatim gate; result in the report
```
`artifacts/grm_f5/tmp` was removed after every pytest invocation.

## Files changed

- `core/grm_admission.py` — the only core edit; four sites, tabled above.
  All behaviour changes are behind `GRM_ROUTE_SOLE_BINDER_INSURANCE`.
- `tests/test_grm_f5_sole_binder_insurance.py` — NEW. 27 pure-CPU tests:
  the flag, the RED→GREEN fixture on both rules, 8 controls, the helper's
  contract, 3 receipt tests through `decisive_admission_profile`, 3
  counterparts to F3's pinned tests, and 3 registration pins.
- `scripts/grm_f5_c2_replay_gate.py` — NEW. The OFF byte-identity gate.
- `artifacts/grm_f5/flag_contract.json` — NEW. The named-optional
  registration for LT1.1 r3, same shape as F2's.
- `artifacts/grm_f5/c2_replay_off.json`,
  `artifacts/grm_f5/c2_replay_pre_edit.json`,
  `artifacts/grm_f5/c2_replay_on_risk.json`,
  `artifacts/grm_f5/core_sha_before.txt` — NEW receipts.
- `docs/GRM_F5_SOLE_BINDER_INSURANCE_LEDGER.md` — NEW (this file).
- `tests/test_grm_f3_routing_at_distance.py` — **UNCHANGED** (read-only per
  the order; its three DEFECT-PINNED tests still pass).
- Every other `core/` file — **UNCHANGED**.
