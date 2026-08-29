# ORDER MOE-E2-F2 — micro fix: CUDA peak-stats reset before context init

WRITABLE TARGET `/mnt/ForgeRealm/GraftRepository`; same rules as E2-F1.

Defect (lead-verified on GPU): `scripts/olmoe_e2_experiment.py:1278`
calls `torch.cuda.reset_peak_memory_stats(0)` before any CUDA
allocation; torch 2.11 raises `RuntimeError: Invalid device argument`
pre-init (lead repro: pre-init FAILS, post-init OK). Chain died at
first load.

Fix: ensure CUDA context is initialized before ANY cuda.memory
telemetry call in the harness (e.g. `torch.cuda.init()` or a tiny
device allocation), audit the file for other pre-init cuda telemetry
calls, keep everything else byte-identical. Self-run your synthetic
contracts + compile check. No script regeneration needed unless the
audit finds more sites.

Done: the diff summary, audit result (other sites or none), receipts.
