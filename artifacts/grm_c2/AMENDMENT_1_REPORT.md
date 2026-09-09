# GRM-C2 lead amendment 1 — CPU preparation complete; GPU NOT_RUN

Amendment: `artifacts/grm_c2/amendment_lead_1.json`  
SHA-256: `91df018adaa90e9db3fd32981ad72a5564cee2cc8360b462775248945859807c`

The amended executable runner accepts the authorized **3600 GPU-second cap**.
The immutable r1 registration remains at2880s, SHA
`8a3a27302c41f2980f883666657f7d871c5ad01d195953e017804717d3644eb0`. All52 original cells, probes,
dependencies and estimates remain unchanged: **3140 estimated GPU seconds**,
460s headroom. No arm dropped or cell shortened. The executable wrapper invokes
`scripts/grm_c2_amended.py`; the historical `grm_c2_cells.py` CLI retains its
original refusal. Its worker is reused by import unchanged.

The comparison is **shipped defaults vs proposed profile**: defaults width256,
profile width96. It is **not a width-matched pair**. `defaults96` (width96,
capture pin OFF, seat-near-live OFF, RT1 unchanged ON) is registered **NON_FIT**:
26 extra cells,1570 extra estimated seconds,78 cells/4710s total,1110s over cap.
This is planning reasoning using unchanged r1 estimates, not measured timing.
A3's historical timing caveat remains:3140s is not a completion guarantee.

## Exact lead commands

Fresh execution (the wrapper dry-runs first, then runs cells, then prints summary):

```bash
cd /mnt/ForgeRealm/wt/grm-c2
./artifacts/grm_c2/lead_commands.txt
```

CPU-only preview:

```bash
./artifacts/grm_c2/lead_commands.txt --dry-run
```

Explicit resume after a stop, preserving all started/RED cells:

```bash
./artifacts/grm_c2/lead_commands.txt --resume
```

Print the current table independently:

```bash
python scripts/grm_c2_amended.py summary
```

The full52 commands and three scoring points are listed in
`lead_commands_expanded.txt` for audit. The executable wrapper provides
stop/resume; do not substitute an unconditional replay of that audit listing.
Each cell is one foreground CMC1 flock lease on `/tmp/forge-gpu.lock`:
wait<=240s, lease<=285s, worker timeout280s, outer TERM585s and hard bound590s.
The timeout wrapper targets only its own process group. A separate campaign
flock serializes budget and start decisions. A persisted timestamp enforces
30s foreground cooldown before the next cell, including after resume. The full
plan adds51*30=1530s cooldown: at least4670s estimated foreground wall time,
excluding lock wait, scoring and other CPU overhead. These are not GPU charges.

Cells execute sup, census, longhistory, preserving r1 order within each battery;
all restart dependencies have completed persistence first. Each battery scores
immediately after its final restart cell. First new RED stops. Explicit resume
selects the next unstarted cell, never retries COMPLETE/RED/unfinished cells,
and preserves immutable RED scoring results. A missing/RED persistence dependency
produces RED without a GPU worker or recapture. Unfinished starts remain charged
285s; no next worker starts unless its full285s reservation fits remaining cap.
The final wrapper exit remains nonzero when any RED or incomplete evidence remains.

## CPU gates and evidence

Evidence class: author-run CPU unit/suite tests and seeded behavioral mutations.
**45 passed,2 warnings in1.44s**:24 new amendment cases plus the unchanged21 C2
r1 cases. Receipt `amendment_lead_1_cpu_initial.log`.

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/test_grm_c2_amendment.py tests/test_grm_c2_profile.py tests/test_grm_c2_capture.py tests/test_grm_c2_payload.py
```

Amendment test function names (parameterization expands to24 cases):

- `test_amended_budget_preserves_all_52_cells_and_registers_defaults96_nonfit`
- `test_forged_amendment_refused_even_with_recomputed_sidecar`
- `test_stale_order_registration_or_worker_binding_refused`
- `test_missing_amendment_refused`
- `test_order_scoring_and_outer_rails_for_all_52_cells`
- `test_stop_on_red_resume_next_unstarted_never_retry`
- `test_red_and_orphan_cells_never_retried_and_orphan_fully_charged`
- `test_red_persist_blocks_restart_without_worker_or_recapture`
- `test_amended_cap_used_for_reservation_and_worker_lease`
- `test_budget_reservation_refuses_over_cap_without_worker`
- `test_cooldown_survives_resume`
- `test_summary_reports_denominators_width_seats_rt1_and_missing`
- `test_cell_evidence_refuses_missing_or_duplicate_fields`
- `test_changed_completed_receipt_refused`

The five registered mutants were killed **5/5=1.00**, exceeding the>=0.80 rail:
raised cap +1, orphan charge removed, summary green forced false, token-seat
requirement removed, and RED exit changed to success. Registration and result:
`amendment_lead_1_mutation_registration.json`,
`amendment_lead_1_mutation_results.json`; each includes or references raw logs.
Mutant execution deliberately retained original source identity for the independent
hash guard, so kills came from behavioral assertions. Production/source originals
were unchanged; mutations were temporary CPU copies. This is not blind verification.

`bash -n artifacts/grm_c2/lead_commands.txt` passed; executable `--dry-run` exited0
and enumerated all52 cells. Receipt `amendment_lead_1_dry_run.json`.
The actual unmeasured `summary` exited1 and printed all six arm/battery rows,
pre/post denominators9/10/14, UNKNOWN token-seat sums, UNKNOWN/MISSING RT1 fields,
explicit widths and defaults96 NON_FIT. Receipt
`amendment_lead_1_summary_not_run.txt`. Synthetic complete fixtures verify scores,
pre/post token-seat sums and the presence of all four named RT1 fields. Missing
or duplicated probes, missing seats/RT1, stale worker receipts, missing restart
metadata and same-process evidence fail. No synthetic score is a model result.

The amendment digest is pinned in trusted verifier code, bindings cover both
orders, r1 registration, unchanged worker, tests and executable commands; normalized
verifier source is also SHA-bound. A replaced JSON plus recomputed SHA sidecar
is refused. This provides content integrity relative to trusted source, **not**
a digital signature or protection against an attacker replacing the entire
trusted checkout. Every new worker/controller/score receipt carries amendment,
r1 and registry digests; workers/controllers also fingerprint sources.

## Prior art

C2 r1, CMC1, WC1/RT1.1 and SCOUT-FIX-1, project contributors (2026), verified by
local source inspection. Reused unchanged worker, existing battery comparator,
checkpoint persistence, reservation discipline and foreground flock lease.
New work: amendment verification, stable battery scheduling, explicit resume,
and presentation of existing measurements. SHA-256 content binding is established
practice, not a novel algorithm. House Rules/AtlasForge (project,2026) supplied
seeded-mutation discipline and>=0.80 threshold; the five defects are C2-specific.
No prior art known to me for this particular adapter beyond those local systems.
No external literature or novelty claim is made.

## Deviations, RED, process safety, identity

Additive new runner preserves the r1 script and historical tests; only the lead
command artifact is replaced, with original bytes archived in `lead_commands_r1.txt`.
Order, registry, immutable r1 registration, product flags/kernels and batteries are
unchanged. The requested extra arm is explicitly NON_FIT, not silently omitted.

**RED/NOT_RUN:** all GPU and adoption/fix-validation gates remain unmeasured.
Not claimed fixed: r1's broader CPU RED (1480 passed,58 failed,148 skipped,20 errors),
including missing receipts/calibration and CUDA-dependent inventory mistakes.
Those full suites were not rerun for this additive amendment; these numbers are
historical local r1 receipts, not current revalidation. No GPU regression absence,
profile adoption, blind verification or successful full-budget completion claimed.

This seat used no GPU, git, subagents, background waits, process kills/signals,
lock clearing, live services or external messages. CPU subprocesses ran sequentially
or independent CPU read-only previews; no heavy runs overlapped. Model identity:
**Codex / GPT-6; exact serving API model ID unavailable; reasoning effort high**.
