# GRM-C4 lead amendment5 ledger

2026-09-09 — registration before CPU gates. Immutable order:
orders/GRM_C4_AMENDMENT_5.md, SHA256
8eb729c2454a1ee1da1f43ff9a288472c74541720911623455037a4f9646f0e4.
Read HOUSE_RULES.md and AGENTS.md. No git or subagents used.

Diagnosis (recorded E2E receipts and instrumentation): A2 lh06..11 walls
74.77318456,78.85257644,81.91498956,190.94023734,84.73291107,256.31871599 s.
A2 lh12 RED is AssertionError: Worker rail exceeded: NON_FIT,306.07129970 s
actual charged wall against285 s rail. The receipt completed turns88..95;
turn90 alone225.512 s. This contradicts a uniformly distributed split estimate:
no turn-range split containing turn90 can honestly estimate below200 s.
Overhead (wall minus turn walls) rises30.4999..38.7558 s. Register ceil39 s.
Halves88..92,92..96,96..100,100..104 estimate284,63,66,348 s. First/last
registered NON_FIT. Probe100 extrapolates probe90 plus max observed70->80 or
80->90 increment; ordinary turns use max observed non-probe wall. This is
planning reasoning, not measured lh13 evidence. It is not a rail increase.
Full same23-unit plan for each remaining cell; wide recovery has4 new units.
Retain all50 units and all14 probes. Refresh lh06..11 estimates from evidence.
Recorded2157.6785489856265 + registered5741 =7898.6785489856265 s, overflow
2498.6785489856265 s beyond5400. Register all overflow; do not trim or recycle
hypothetical non-execution savings into authorizations. Budget admission also
retains285 s reserve, lease285, cooldown30, outer590. Original receipts unchanged.

Implementation: new scripts/grm_c4_split_a5.py succeeds the A3 entrypoint;
no earlier source files changed. Within-cell dependencies, state inventory,
prior receipt links and complete scoring remain checked. No prior-cell score
required. Planning/dependency/cap NON_FIT receipts have no GPU claim or elapsed
work; partial score records contain no scores. Summary prints NON_FIT with
completed segments and INCONCLUSIVE prediction. Integrity faults still refuse.
New scripts/grm_c4_register_a5.py creates amendment_a7 and exact command list.
No GPU command file is run by this seat.

Prior art: verified local C4/A2/A3/A4, EB1 and DET1 (house,2026). Reused SHA
source closure, immutable registration, create-only claims, saved-state copies,
lease accounting and original scoring arithmetic. Added half-range schedule,
independent dispatch and partial-status assembly; no novel algorithm claimed.
Specific timing extrapolation heuristic: no prior art known to me.

Pre-registration: pre_registration.json;2567 existing run files hashed in
before.json. CPU author gates cover resume, forgery/staleness, immutable bytes,
cap accounting, decoupling and summary honesty. Synthetic timing admissions
explicitly permit mechanics testing; live NON_FIT units remain refused.
Deviation/RED: requested <200 s per split impossible with measured indivisible
turn90; report honest NON_FIT, no fabricated fit. This is not claimed fixed.
Model: GPT-6 (Codex; exact deployment identifier not exposed). Effort: high.

2026-09-09 — sealed amendment and gate receipts.
Amendment_a7.json SHA256:
e3ab21e4974a11b2d8499233b220ed02dbd2778b9e2023ca1272f1ea416a6d65,36076 bytes.
Foreground registration command exit0. Full pytest command (see FINAL_REPORT)
exit0:181 passed,2 warnings,40.66 s;35 new cases and146 existing cases.
No source edits after sealing. Dry-run and bash -n exit0. validation.json
records44 unchanged receipt/claim hashes and2567 unchanged run files; suite
hashed all2567 files. No new real workers. All old bindings still validate.

The full projection marks24 units cap NON_FIT (c96_w64/lh13b, all23 c64_w64).
The separate planning limit marks both probe-containing halves NON_FIT for
all cells. Independent starts tested synthetically, including c64_w64 without
any earlier-cell score. Actual c64_w64 admission remains cap NON_FIT. Both
limits are reported explicitly; do not infer actual completion from CPU gates.
Exact lead_commands_a5.txt contains all50 units,3 score commands and summary.
It is sealed, syntax-checked and unexecuted. Historical command lists unchanged.
Final report contains prior art, deviations, RED and safety statement. No
blind gate or GPU evidence claimed. Model GPT-6/Codex, effort high.
