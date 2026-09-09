# GRM-C4 lead amendment 6 ledger

2026-09-09 — read HOUSE_RULES.md, AGENTS.md and immutable order GRM_C4_AMENDMENT_6.md.
Diagnosis: A5 already rejects planning NON_FIT without a lease, but its full-plan
projection charges all estimates and sets cap overflow at5400. A6 excludes six
registered skips, retains all44 fitting estimates, and overrides only cap6600.
Recorded2157.6785489856265 + fitting3845 =6002.6785489856265s; headroom597.3214510143735s.
Skip list: c4-lh-12a284s and c4-lh-13b348s in c64_w96, c96_w64, c64_w64.
A5 planning rail is <200s; lh12a284s does not exceed the285s lease. Preserve
the lead's explicitly named NON_FIT units, no new classification heuristic.
Residual: lh12b63s and lh13a66s in each cell depend on skipped lh12a. No
authorized path can execute these six fitting segments with unchanged state
rules. Register them honestly as dependency-blocked, retain their387s in the
projection, and dispatch for zero-work admission status.38 remaining workers
are reachable if all prior units pass; CPU preparation is not execution evidence.

Implementation: additive scripts/grm_c4_skip_a6.py overlays the unchanged A5
executor and uses its receipt root lead_a5/runs; new receipts bind amendment_a8.
New create-only registrar and tests. A5 and all historical registrations, source
files and command lists remain byte-unchanged. Registration and tests are pinned
before any gate. pre_registration.json defines gates; before.json snapshots
historical artifacts and source bytes. Only narrative synthesis is appendable.

Prior art: local C4/A4/A5 and DET1 (house,2026), verified source: exact SHA
closure, runtime overlays, create-only status receipts, saved-state dependencies,
full-worker reserve accounting, partial summaries and unchanged scorers. A6 adds
lead-directed fitting-only projection and explicit CPU skip dispatch; no novel
algorithm claimed. Inherited timing heuristic: no prior art known to me.
No external algorithm introduced or literature claim made.

RED: existing lh12 rail failure and incomplete batteries are not claimed fixed.
Deviation: no policy change beyond order; requested execution of every fitting
unit is blocked for six by unchanged dependencies, explicitly registered.
Safety: CPU only; no git, subagents, GPU, background jobs/waits, process kills,
lock or live service changes. Generated GPU commands are for lead, unexecuted.
Model: GPT-6 (Codex; exact deployment identifier not exposed). Effort: high.

2026-09-09 — sealed registration and completed CPU gates.
Amendment_a8.json4343189f992408e57797dfc0ebd13dc6df2a7e086e28d874823bdaa546f2c78e,42570 bytes.
Registration exit0; no source edits after sealing.31 new tests PASS21.86s;181
legacy tests PASS38.14s; both exit0, two Swig warnings plus exit warning each.
Exact commands and test/function locations in FINAL_REPORT.md; gate logs retained.
Dry-run and bash -n exit0. validation.json records2742 byte-unchanged historical
files, including2567 run files and44 receipts/claims; all prior commands unchanged.
No real A5/A6 runs root. Six skip tests prove no readiness, accounting, harness,
lease or attempt; forbidden executed-skip forgery refused. Synthetic dispatch
uses38 leases, six fitting dependency refusals, and three19-completed-unit cells
reported NON_FIT with no battery scores; INCONCLUSIVE and null prediction.
Actual reserve boundary6315 accepted,6315.001 refused under6600.
Peak projected reservation6221.678548985627s; fitting projection6002.678548985627s.
lead_commands_a6.txt70b9935128af97766e6f4d6b86a62b40a44ac147bcfb937d9e891b53667382bb,6699 bytes;
six CPU skip commands,44 fitting dispatches,three scores,summary. Unexecuted.
No deviations beyond the explicit inability to execute the six dependent fitting
units under unchanged state rules. Existing REDs not claimed fixed. Safety and
prior-art statements above remain unchanged. Model GPT-6/Codex, effort high.
