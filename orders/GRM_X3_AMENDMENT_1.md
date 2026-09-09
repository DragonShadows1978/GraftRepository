# GRM-X3 amendment 1 (lead, 2026-09-08) — the four cells ran and wrote full results; the summary saw none of them

Your r1 is committed (49181e1). Lead-run: `cell_00..cell_03` all
completed on the card (234 s each), receipts under
`artifacts/grm_x3/runs/e59ce9ad…/cell_0N/x3_NN/result.json` with real
measurements (e.g. x3_00: base mass on lesioned mount 0.439, sham mass
0.039, remove-fork KL 2.908 nats, top-1 976 → 40; snapshot manifests
complete). Then `summary` produced
`artifacts/grm_x3/summaries/b80b90a2….json` with
`observed_snapshots 0, realized_strata false, status RED_STRATA_OR_INCOMPLETE`
and every prediction null. The summarizer is not reading the run it
should: the summary fingerprint (b80b90a2…) differs from the run
fingerprint (e59ce9ad…), so either the summary keys on the wrong
fingerprint or it requires a field the cells do not write (e.g. an
`observed_group`/exact-outcome that `grm_x3_diagnostic.py:144-147`
computes but the worker never persists). Diagnose, fix in your scripts
only, add a CPU test that summarizes a synthetic run directory laid out
exactly like the real one, re-run `summary` on the EXISTING receipts
(no GPU re-run; the receipts are the evidence), and deliver the
four-way table (mass high/low × lesion effect high/low with exact-error
rates), the P1–P3 verdicts for both prediction sets, and the sham /
same-payload control statistics. Same rules (no git, no subagents, no
background waits, never kill). Reasoning effort: **high**.

## Done (verbatim)
1. Root cause (file/lines) and the fix; the new test name.
2. The summary on the existing receipts: four-way table, controls,
   verdicts.
3. RED; process safety; model id and effort.
