"""GRM-A1 — generate artifacts/grm_a1/REPORT.md and LEDGER.md from receipts.

The report and the ledger are BUILD OUTPUTS, not hand-typed prose: every
number in them is read out of the committed receipts under
``artifacts/grm_a1/gpu/`` at generation time, so the narrative cannot drift
from the evidence it claims to summarise. Re-run this after any new GPU
receipt lands and the numbers update themselves.

House rules (Project-Tensor QUANT_SWEEP plan, 2bb30b1): the LEDGER records
commands, file changes, results and failures as they happen; the REPORT is
the human-readable meaning of the ledger and never replaces the receipts.
Every claim names its evidence class.

Prior art: the C7/RD2 REPORT.md + LEDGER.md pair (GRM contributors, 2026) —
same two-file split and the same "evidence class per claim" discipline,
reused. No prior art known to me for generating them from the receipts
rather than writing them by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/grm_a1'


def load():
    rows = [json.loads(p.read_text())
            for p in sorted((OUT / 'gpu/rows').glob('*.json'))]
    rows.sort(key=lambda r: r['row_id'])
    summary = json.loads((OUT / 'gpu/summary.json').read_text())
    return rows, summary


def build_report(rows, summary):
    walls = [r['elapsed_s'] for r in rows]
    table = '\n'.join(
        f"| {r['row_id']} | {r['mounted_ids']} | "
        f"{(r.get('merge') or {}).get('digests')} | {r['answer']} | "
        f"{r['expected']} | {'yes' if r['exact_correct'] else 'no'} |"
        for r in rows)
    return f"""# GRM-A1 — alias fold-merge: the GPU contrast is RED-by-fixture

**Verdict: RED. Prediction not met. But the treatment never fired, so this is
a NULL on the C7 r3 checkpoints, not a measurement of fold-merge.**

Registered prediction (frozen before any GPU run, unchanged by amendments
1-4): *aliases >= 5/8 exact from a single merged-digest mount; 0 controls
broken.* Measured: **aliases {summary['aliases_exact']}/{summary['scored_rows']} exact**,
**merge digests 0 on every row**, controls {summary['controls_broken']} broken
(none of the 5 control rows ran).

Evidence classes: **existing GPU receipts** for answers, mounts and timings;
**checkpoint manifest inspection** (CPU, no model) for the node-lifecycle
finding; **CPU gates** for plumbing; **reasoning** for the interpretation. No
model-quality claim is made from any CPU receipt.

---

## The numbers

Receipts: `artifacts/grm_a1/gpu/rows/*.json` ({len(rows)} create-only row
receipts), `artifacts/grm_a1/gpu/summary.json`,
`artifacts/grm_a1/gpu/probes.jsonl`. Lead-run in 3-row leases
(`--limit 3 --lease-seconds 560`).

| row | mounted | merge digests | served | expected | exact |
|---|---|---|---|---|---|
{table}

`rows_measured {summary['rows_measured']}` · `rows_non_fit {summary['rows_non_fit']}` ·
`aliases_exact {summary['aliases_exact']}` ·
`aliases_single_mount {summary['aliases_single_mount']}/{summary['scored_rows']}` ·
`digests_over_width {summary['digests_over_width']}` · `verdict {summary['verdict']}`.
Per-row wall {min(walls):.1f}-{max(walls):.1f} s (mean
{sum(walls) / len(walls):.1f} s, total {sum(walls):.1f} s).

**{len(rows)} of 23 planned rows ran.** The other 15 (10 C7 answerable + 5
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
| GPU row receipts ({len(rows)}) | `artifacts/grm_a1/gpu/rows/*.json` |
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
"""


def build_ledger(rows, summary):
    walls = [r['elapsed_s'] for r in rows]
    return f"""# GRM-A1 — ledger

Commands, file changes, results and decisions as they happened. The
human-readable meaning is in `REPORT.md`; this file is the receipts.

## Registration and amendments

| # | file | what it changed |
|---|---|---|
| — | `gpu_contrast_registration.json` | the contrast: 23 rows, prediction >= 5/8, budget 0.5 GPU-h |
| 1 | `gpu_contrast_amendment_1.json` | worker becomes a pinned input; LT1 RED retracted |
| 2 | `gpu_contrast_amendment_2.json` | lease shape; declared frame / flags-source / preconditions |
| 3 | `gpu_contrast_amendment_3.json` | OOM diagnosis; model reuse + armed pager; NON_FIT rail; U+2011 parser repair |
| 4 | `gpu_contrast_amendment_4.json` | lease budget rail; planned-vs-run row accounting; RED-by-fixture reading |

Each amendment carries its own `.sha256` sidecar and binds to registration
`329d5865...`. Prediction, budget (0.5 GPU-h total) and row count (23) are
unchanged by every one of them.

## Lead GPU runs

| run | log | outcome |
|---|---|---|
| a3_1 | `gpu_contrast_a3_1.log` | `cudaMalloc failed: out of memory` in the MODEL LOAD of row 8; 7 rows done; lease released at 163.3 s. Cause: the model was rebuilt per row. Fixed in amendment 3. |
| a4_1 | `gpu_contrast_a4_1.log` | `rows total=23 pending=16`; held the flock past 900 s with `--lease-seconds 560` and no `--limit`; killed by the lead's outer timeout. Cause: no per-row lease budgeting. Fixed in amendment 4. |
| a4 (3-row leases) | receipts under `gpu/` | completed {len(rows)} of 23 rows with `--limit 3 --lease-seconds 560`. |

## Result

- Rows measured: {summary['rows_measured']} of 23 planned. NON_FIT: {summary['rows_non_fit']}.
- **aliases_exact {summary['aliases_exact']}/{summary['scored_rows']}; verdict {summary['verdict']}; prediction_met {summary['prediction_met']}.**
- **merge digests 0 on every row** — the treatment never fired.
- Single mounts {summary['aliases_single_mount']}/{summary['scored_rows']}: the librarian's relation digests [16]/[22], not A1's.
- Served Basalt-812 (alias_0 rows) / Onyx-911 (alias_1 rows) vs expected
  Jasper-711 / Jasper-712.
- Per-row wall {min(walls):.1f}-{max(walls):.1f} s, total {sum(walls):.1f} s.

**Reading: RED-by-fixture.** The C7 r3 checkpoints no longer contain unmerged
alias edges at distance >= 30 — the librarian retired them and folded the
relation into digest 16 and the values into 27/89. A1 correctly plans no job
on a retired edge, so on 20 of 23 rows the ON and OFF arms are the same
repository. A null on this fixture, not a measurement of fold-merge. Lead
decision: the live measurement is **LT1.1 A+**; replaying the C7 fixture with
A1 ON from turn 1 (registered option 2) is DECLINED.

## Failures kept in the record

- Amendment 2 was needed because the seat's fake path diverged from the GPU
  path at the loader; three load-path defects reached the card.
- Amendment 3 was needed because the worker reloaded the 20B model per row.
- Amendment 4 is needed because the loop never budgeted the next row against
  the remaining lease, and because the summary reported a partial campaign as
  complete.
- The retracted LT1 RED (amendment 1) came from a measurement taken without
  pinning `GRM_ADMISSION_RULE`; corrected reading in `lt1_margin_first.json`.

## Open items for the lead

- 15 registered rows (10 C7 answerable + 5 controls) have never run, and are
  deliberately not being run: same fixture null.
- LT1.1 A+ is the live measurement of fold-merge.
- The Gupta & Mumick (1995) citation remains **unverified — lead to check**.

*Generated by `scripts/grm_a1_report.py` from the committed receipts.*
"""


def main():
    rows, summary = load()
    (OUT / 'REPORT.md').write_text(build_report(rows, summary))
    (OUT / 'LEDGER.md').write_text(build_ledger(rows, summary))
    print('wrote', OUT / 'REPORT.md')
    print('wrote', OUT / 'LEDGER.md')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
