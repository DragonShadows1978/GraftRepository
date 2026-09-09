# GRM-SCOUT-FIX-2 ledger

Order is immutable: orders/GRM_SCOUT_FIX_2.md. Gates registered before execution in artifacts/grm_scout_fix_2/registration.json.

Prior art: B3/RS3 capture projection, LSR-P2C cull slicing, C2 amendment1 scheduler/receipts, project contributors (2026), inspected locally. Taken: existing field projection, payload slicing and scheduling. New: additive inheritance markers and isolated epoch with retained charge. No prior art known to me beyond these local systems; no novelty claim.

Initial audit: persistent child creation converges at _cull_graft_direct. Bare-arena fallback harvests via deposit (direct capture); manifest reload restores capture; WAL placeholders have no attested payload. Archived baseline source and lead command bytes before edits. No git, subagents, GPU, services, signals or background waits.

Initial test run (red_before.log) had a fixture defect: FakeSliceArena omits rare until save. Added rare at fake deposit; this is fixture repair, not treatment evidence. Production unchanged.

Treatment evidence: corrected pre-fix fixture log red_before_corrected.log: 8 failed (capture None), 2 passed. Unchanged test file after production treatment: green_after.log, 10 passed. Production diff: deepcopy B3 projection at cull child construction plus capture_inherited_from_parent and capture_parent_graft_id. Parent geometry is retained even when the current arena width differs.

Epoch implementation also needs r1 input supersession: r1 binds core/graft_repository.py and scripts/grm_c2_profile.py. New verifier checks amendment3 before accepting exact registered before/after replacements. Old orders/registrations untouched. Historical command and verifier bytes retained; old CLI redirects to epoch3.

Amendment3 registered before epoch gates: artifacts/grm_c2/amendment_lead_3.json SHA256 218cbc8cff58762dee57f4fe399d4440f13db74956d6a5cc18bd6859708d9b22. All52 cells unchanged; estimate3140 + historical46.348182500805706 = 3186.3481825008057 seconds; cap3600. Historical receipts/checkpoints bound byte-for-byte.

## Final CPU gates and receipt audit

- `CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/test_grm_c2_*.py tests/test_grm_scout_fix1_*.py tests/test_grm_scout_fix2_capture.py` -> 95 passed, 2 warnings, 8.76s; artifacts/grm_scout_fix_2/cpu_full_initial.log. Existing C2/SCOUT-FIX-1 tests unchanged; B3 old-manifest control included.
- `CUDA_VISIBLE_DEVICES='' ./artifacts/grm_c2/lead_commands.txt --dry-run` -> READY_FOR_LEAD; artifacts/grm_scout_fix_2/dry_run.json, empty stderr. All52 UNSTARTED, prior/current charge46.348182500805706, cap3600, estimate including prior3186.3481825008057. Epoch directory not created.
- Additional gates registered separately before execution: artifacts/grm_scout_fix_2/audit_gate_registration.json.
- `CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/test_grm_lsr_p2c_split_descent.py tests/test_grm_rt1_split_child_routing.py tests/test_grm_runtime_lifecycle.py` -> 178 passed, 2 warnings, 134.09s; audit_cpu.log. CPU native helpers compile main-repo C++ into pytest tmp; compared .cpp/.hpp SHA256 to worktree, both identical (native_test_source_identity.json).
- `CUDA_VISIBLE_DEVICES='' python scripts/grm_c2_amended.py --dry-run` -> byte-equivalent JSON value to new executable preflight; legacy_entry_dry_run.json.
- verify_amendment plus historical amendment2 binding checks -> final_integrity.json PASS. Historical RED/controller/worker/checkpoint bytes, original orders and amendments1/2 preserved. Refreshed command executable. Source patches generated using Python difflib, no git.

Full audit and exact commands: docs/GRM_SCOUT_FIX_2_REPORT.md. Core change only at cull child construction + deepcopy import. Bare-arena ephemeral children have direct capture via deposit; RT1 only routes; ordinary load already restores capture; WAL placeholders lack attested capture and remain explicitly unattested. No new epoch GPU cell executed; historical RED not rewritten or claimed green. All foreground commands completed; no process was killed. No blind verifier dispatched (no-subagents rule).

Exact next lead command: `./artifacts/grm_c2/lead_commands.txt --resume` (preflight included); CPU-only preflight `./artifacts/grm_c2/lead_commands.txt --dry-run`.

Model: Codex / GPT-6, exact serving API ID unavailable; requested effort high. Prior art as registered above; no novel algorithm claim. Initial fake rare-field repair is the only failed author setup; corrected RED-before/GREEN-after evidence is preserved separately. No orders/registrations amended in place.

Order SHA256 verified: bc96403679d4481e842ba0e90ffc9b50757a9fceee13a864887e5a911e24cdbb.
