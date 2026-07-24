# ORDER GRM3P-LADDER-ON — make the probe ladder the permanent default (Grok seat)

WRITABLE TARGET /home/vader/GraftRepository-three-pass. No git. No
subagents. Sandbox GPU expected blocked — implement + unit-check +
hand the lead the exact re-gate commands (proven pattern).

OPERATOR DECISION (David, 2026-07-23): the probe ladder
(@e26f98f, all gates green) becomes DEFAULT-ON permanently. The old
byte-identity baseline (68da84b8…) is SUPERSEDED by this registered
decision — record old→new in docs/GRM3P_DIAG_UNBOUNDED_REPORT.md
(append a "DEFAULT FLIP" section; never edit prior sections).

## Work
1. Flip the default: GRM_PROBE_LADDER / --probe-ladder ON by default;
   add an explicit escape (--no-probe-ladder / GRM_PROBE_LADDER=0)
   that restores the legacy path exactly.
2. Update run_config/restart plumbing so the resolved value persists
   across restart re-exec.
3. Unit tests: default-on resolution, escape-off resolution, restart
   persistence (extend tests/test_grm_probe_ladder.py).

## Registered gates (lead runs GPU; you hand commands)
- G-ON-ESCAPE: escape-off smoke transcript sha == 68da84b8… (legacy
  path byte-identical under the escape).
- G-ON-BASELINE: default smoke — record NEW sha as the registered
  baseline; `three_pass` transcript byte-equal to `single` at the new
  default (the pipelines must still agree).
- G-ON-FULL: default (no env) F-FULL unbounded 9/9; 4MB 9/9.
- G-ON-SUITE: both-pipeline suites green.

## Done
Verbatim: what changed, unit lines, the four gate commands for the
lead, residuals.
