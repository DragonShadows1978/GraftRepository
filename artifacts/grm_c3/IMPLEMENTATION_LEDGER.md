# GRM-C3 implementation ledger (append-only)

## 2026-09-08 — preparation, GPT-6, requested effort high

- Read /mnt/Shared/HOUSE_RULES.md, local AGENTS.md, immutable orders/GRM_C3_DNGH_DECOY_CALIBRATION.md, Scout Part C and demand config. Used a lightweight memory registry search for GraftRepository evidence-scope conventions; all task facts rechecked in current sources. No git/subagents.
- Inspected DET1.5 workers/campaign, DET1.2 loader, DET1.3 planting, DET1 observer, SC2 first crossing and SC1.2 minimum helper, EB1 reset and production attempt. Read-only worktree HEAD metadata names grm-c3. No ancestry claim.
- Added scripts/grm_c3_register.py; ran `python scripts/grm_c3_register.py` BEFORE any gate. Registration SHA `048e4c7e786be8bb26685abbdd21ce710c9d0f4f2f8fa5e6b3722538820c5666`; case-set SHA `c46535aec23824a876f9198eebf58a67e73cab7faa12c42f45980e07938c6f30`. Immutable creation uses exclusive writes.
- Added scripts/grm_c3.py and tests/test_grm_c3.py. Controlled mounts, four equal offline rules, one answer per case, 12 x 225-second total caps, 590-second outer cap, 30-second foreground cooldown. Model reload is in a fresh child for each case. Near-zero fixed at strict 1e-8. No production code edits.
- Command `PYTHONDONTWRITEBYTECODE=1 python scripts/grm_c3.py --pin` completed, source/model manifest SHA `f120564fa2e5224f3a5a86a883d2016d239571e3e92c3a7cb6e175eb3be0a9c3`. Fingerprints 346 files / 13,808,566,585 bytes. Fingerprinted the existing external native runtime because this worktree has none. No GPU initialized.
- Command `PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python scripts/grm_c3.py --cpu-gate` exit 0: **29 passed in 0.28s**. Receipt cpu_gate.json contains exact stdout/stderr plus fixture novelty source fingerprints. No test failures occurred.
- Command `PYTHONDONTWRITEBYTECODE=1 python scripts/grm_c3.py --dry-run > artifacts/grm_c3/dry_run.json` exit 0. Enumerates every case/cell/estimate. Estimated 2400 GPU seconds, cap 2700 GPU seconds; estimates are reasoning, not measured.
- Added case_set.md, blocked_report.md, lead_commands.txt and this ledger. GPU rows = 0; all empirical metrics UNRUN. No demand adoption claim. Trace-only preservation is limited to correct answers at risk; demand-ON preservation remains unmeasured.

## Prior art (code-site annotations also present)

Verified local source: DET1/SC2/SC1.2 (GraftRepository contributors, 2026): planting, mass observation, strict crossing/minimum, comparator, model loader and cleanup; imported, not copy-edited. The new material is case text, controlled-mount campaign design, 1e-8 tolerance and bounded offline orchestration. Fisher 1935 blocking; Chow 1970 rejection; NIST FIPS 180-4 SHA-256 (2015); POSIX flock and Python subprocess/GNU timeout are known prior-art leads, **unverified — lead to check** “Fisher Design of Experiments randomized blocks”, “Chow optimum recognition error reject tradeoff”, “attention mass missing knowledge detection”, “NIST FIPS 180-4 SHA-256”, “POSIX flock Python subprocess timeout”. No algorithmic novelty claim.

## RED / process safety

No GPU in authorized sandbox; empirical four-arm verdict remains blocked. Natural routing, achieved weak reads, runtime native compatibility, actual wall and demand-ON preservation untested. Author baseline is not blind verification. No process was signaled/killed, no lock touched, no background job launched, no git/subagent/service/config action. Registration/order unchanged after creation; no amendments. Expected writable outputs are only new C3 scripts, test and artifacts.

## 2026-09-09 — final artifact verification

- `bash -n artifacts/grm_c3/lead_commands.txt` exit 0. `sha256sum -c artifacts/grm_c3/registration.sha256` reports registration.json OK. Source/native manifest revalidation (`check_pins(PINS, include_model=False)`) passed after documentation writes; model bytes had already been pinned in the initial CPU inventory. Dry run still has 12 cells / 48 unique assignments; CPU passed; GPU traces remain zero. No additional model tests or GPU commands executed.
