# CONTAM HANDOFF — unsandboxed re-run required

The Grok seat for `orders/GRM3P_DIAG_CONTAM.md` landed the instrumentation
but could not execute GPU legs under `--sandbox strict` (site-packages,
HF weights, and tensor_cuda are unreadable).

## Already done (in worktree, uncommitted)
- `core/paging_telemetry.py` (new)
- hooks in `core/graft_repository.py`
- probe snapshot + env expand in `scripts/grm_e2e_session.py`
- `scripts/grm3p_contam_run.sh`
- `scripts/grm3p_contam_analyze.py`
- report append in `docs/GRM3P_DIAG_UNBOUNDED_REPORT.md` (RED residual)

## Run this outside sandbox
```bash
cd /home/vader/GraftRepository-three-pass
bash scripts/grm3p_contam_run.sh
```

Then append real Done lines (scorecards, P1, P2, P3, verdict) to
`docs/GRM3P_DIAG_UNBOUNDED_REPORT.md` (append only).

Registered default-smoke SHA gate:
`68da84b8824eb79a14f65a692a602215bc53d84232cd041002410f51657bba4f`
