# FIX-LADDER HANDOFF — unsandboxed re-run required

The Grok seat for `orders/GRM3P_FIX_LADDER.md` landed the flagged driver
wiring + unit tests, but this sandbox blocks site-packages / model /
GPU execution (same pattern as DIAG-CONTAM).

## Already done (worktree, uncommitted — no git from this seat)

- `scripts/grm_probe_ladder.py` — pure planning helpers (stdlib only)
- `scripts/grm_e2e_session.py` — `--probe-ladder` / `GRM_PROBE_LADDER=1`
  (default OFF); Fork-A probe path enforces:
  1. point-lookup → clean-room (exclude live/recency)
  2. precise-first when rank-1 covers probe identifiers
  3. `_grounding_attribution` + one retry trip
- `tests/test_grm_probe_ladder.py` — unit coverage of flag + plans
- `scripts/grm3p_fix_ladder_run.sh` — one-command gate runner

## Reused (not reimplemented)

- `arena.route`, `_rare_tokens`, `_query_lex_tokens`, `_node_text_tokens`
- `arena._attempt`, `arena._grounding_attribution`, `_bump_cuda_gqa_epoch`
- existing budget fit + contam mount snapshot helpers

## Core arena

Zero production logic edits. Driver-only wiring.

## Run this outside sandbox

```bash
cd /home/vader/GraftRepository-three-pass
bash scripts/grm3p_fix_ladder_run.sh
```

Registered default-smoke SHA gate (flag OFF):
`68da84b8824eb79a14f65a692a602215bc53d84232cd041002410f51657bba4f`

Flag-on expectations:
- F-FULL single unbounded: 9/9 (t5+t13 flips)
- F-COLD 4MB: 9/9
- smoke flag-on: 2/2
