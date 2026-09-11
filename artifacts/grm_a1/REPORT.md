# GRM-A1 — alias fold-merge: the GPU contrast is RED-by-fixture

**Verdict: RED. Prediction not met. But the treatment never fired, so this is
a NULL on the C7 r3 checkpoints, not a measurement of fold-merge.**

Registered prediction (frozen before any GPU run, unchanged by amendments
1-4): *aliases >= 5/8 exact from a single merged-digest mount; 0 controls
broken.* Measured: **aliases 0/8 exact**,
**merge digests 0 on every row**, controls 0 broken
(none of the 5 control rows ran).

Evidence classes: **existing GPU receipts** for answers, mounts and timings;
**checkpoint manifest inspection** (CPU, no model) for the node-lifecycle
finding; **CPU gates** for plumbing; **reasoning** for the interpretation. No
model-quality claim is made from any CPU receipt.

---

## The numbers

Receipts: `artifacts/grm_a1/gpu/rows/*.json` (8 create-only row
receipts), `artifacts/grm_a1/gpu/summary.json`,
`artifacts/grm_a1/gpu/probes.jsonl`. Lead-run in 3-row leases
(`--limit 3 --lease-seconds 560`).

| row | mounted | merge digests | served | expected | exact |
|---|---|---|---|---|---|
| rd2::c7_alias_0_d030 | [16] | 0 | Basalt‑812 | Jasper-711 | no |
| rd2::c7_alias_0_d060 | [16] | 0 | Basalt‑812 | Jasper-711 | no |
| rd2::c7_alias_0_d120 | [16] | 0 | Basalt‑812 | Jasper-711 | no |
| rd2::c7_alias_0_d250 | [16] | 0 | Basalt‑812 | Jasper-711 | no |
| rd2::c7_alias_1_d030 | [22] | 0 | Onyx-911 | Jasper-712 | no |
| rd2::c7_alias_1_d060 | [22] | 0 | Onyx-911 | Jasper-712 | no |
| rd2::c7_alias_1_d120 | [22] | 0 | Onyx-911 | Jasper-712 | no |
| rd2::c7_alias_1_d250 | [22] | 0 | Onyx-911 | Jasper-712 | no |

`rows_measured 8` · `rows_non_fit 0` ·
`aliases_exact 0` ·
`aliases_single_mount 8/8` ·
`digests_over_width 0` · `verdict RED`.
Per-row wall 20.0-32.9 s (mean
24.5 s, total 195.6 s).

**8 of 23 planned rows ran.** The other 15 (10 C7 answerable + 5
controls) were never executed: the campaign was killed by an outer timeout
partway through row 16 (`artifacts/grm_a1/gpu_contrast_a4_1.log`, which shows
`rows total=23 pending=16`). This was **not** a re-scope — the registration
and every amendment still carry 23 rows. Amendment 4 records the shortfall
rather than shrinking the battery to match what happened to run. The 15
unrun rows are deliberately NOT being run: they are the same fixture null.

---

## Why it is RED-by-fixture

`merge digests 0` on every row is the whole story. The merge did not fail —
**it correctly found nothing to do.**

By distance >= 30 the ordinary librarian has **already folded the alias edge
and its base away**:

- the raw Signal edges (nodes 9/11/12) are `retired=True, active=False`;
- node **16** is an ACTIVE digest that already states the relation
  ("... and C7-Signal-0 is an alias for C7-AliasBase-0.");
- nodes **27/89** are active digests/eras holding the Jasper values.

A1's `_alias_active_nodes` correctly excludes a retired edge, so there is no
edge to pair with a base and no job to plan. On **20 of the 23 registered
rows the ON and OFF arms are the same repository** — the contrast compares a
repository with itself. Only `A-009-016` (d005) still holds live alias edges.

| cell | distance | live Signal edges |
|---|---|---|
| A-009-016 | d005 | **3** |
| A-032-039 | d030 | 0 (relation lives in digest 16) |
| A-040-047 | d030 | 0 |
| A-063-070 | d060 | 0 |
| A-125-132 | d120 | 0 |
| A-257-264 | d250 | 0 |

The served answers confirm it: the model mounted the librarian's own relation
digests ([16], [22]) and answered from the values those digests carried
(Basalt-812, Onyx-911), not the Jasper values the probes expect. That is a
correct read of the wrong evidence — a fixture outcome, not a fold-merge one.

**RED is a true statement about this battery and a false statement about the
treatment.** The registered prediction is untestable on these checkpoints as
they stand.

---

## What the null does and does not license

- **Does not** refute alias fold-merge: the mechanism never ran on the GPU.
- **Does not** confirm it: the contrast produced no evidence about A1 at all.
- **Does** establish, with receipts, that the C7 r3 checkpoints are the wrong
  instrument for this question, and why.
- **Does** leave the CPU evidence standing on its own terms: 125 gates in
  `tests/test_grm_a1_alias_fold.py` (evidence-sufficiency and plumbing, never
  model quality), 6/6 merged digests single-mount-claimable under width 96
  (`width_receipts.json`), and OFF byte-identical to the parent branch across
  two code trees (`off_parity.json`).

**The live measurement is LT1.1 A+** — aliases deposited during the run with
A1 ON from turn 1, on a natural conversation. The lead declined replaying the
C7 fixture with A1 ON from turn 1 (registered option 2): LT1.1 answers the
same question at about half the cost.

---

## Two defects the completed run exposed (fixed in amendment 4)

1. **Lease overrun.** With `--lease-seconds 560` and no `--limit` the run held
   the flock past 900 s and was killed by the lead's outer timeout. Rows were
   only checked *between* rows, via `gpu_lease`'s SIGALRM; the loop never
   budgeted the *next* row against the remaining lease. Fixed: before starting
   a row the loop stops (exit 4, resumable) when `elapsed + reserve > lease`,
   where the reserve is the worst row wall seen so far, or
   `--load-reserve-seconds` (default 200 s, >= the 163 s the cold a3_1 lease
   held) until a row has completed. Per-row wall is recorded; `--limit`
   remains an override; an already-done row is never charged to the budget.
2. **A partial campaign read as a finished one.** The summary said
   `rows: 8, complete: true` because `rows` counted receipts on disk, not the
   plan. Fixed: `rows_planned` and `rows_never_run` are reported alongside,
   and `lease_budget_stop` records whether the loop stopped itself and where.

---

## Prior art

- **Sarthi et al. (2024), RAPTOR** (arXiv 2401.18059) — combining sources into
  one retrievable summary; the antecedent for fold-merge. Not its clustering
  or tree construction.
- **Press et al. (2022)** (arXiv 2210.03350) — self-ask, antecedent for the
  *rejected* two-hop alternative. Nothing taken.
- **Gupta & Mumick (1995), materialized-view maintenance** — a write-time join
  owes an invalidation rule. **UNVERIFIED — lead to check.**
- **GRM contributors (2026), local and verified**: FIX-5 source-enumerated
  consolidation; FIX-8 / SC1.1 `normalize_glyphs`; M5/L2 supersession;
  the LSR-P2C width guard; `grm_c7_middle_replay.open_copy`'s `loaded` carry
  (the model-reuse loader that fit 16/24 rows from these checkpoints);
  `grm_cmc1_gpu_arms.gpu_lease`; `grm_c2_cells.environment`;
  `grm_scout_fix6_replay`'s pin-after-`environment` discipline.
- No prior art known to me for this exact composition.

## Receipts index

| what | path |
|---|---|
| GPU row receipts (8) | `artifacts/grm_a1/gpu/rows/*.json` |
| GPU summary | `artifacts/grm_a1/gpu/summary.json` |
| GPU probes stream | `artifacts/grm_a1/gpu/probes.jsonl` |
| Lead run logs | `artifacts/grm_a1/gpu_contrast_a3_1.log`, `..._a4_1.log` |
| Payload accounting | `artifacts/grm_a1/payload_accounting.json` |
| Width receipts | `artifacts/grm_a1/width_receipts.json` |
| OFF byte-identity | `artifacts/grm_a1/off_parity.json` |
| LT1 margin_first correction | `artifacts/grm_a1/lt1_margin_first.json` |
| Registration + amendments 1-4 | `artifacts/grm_a1/gpu_contrast_*.json` |
| Ledger | `artifacts/grm_a1/LEDGER.md` |

*Generated by `scripts/grm_a1_report.py` from the committed receipts.*
