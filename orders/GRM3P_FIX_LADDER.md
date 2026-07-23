# ORDER GRM3P-FIX-LADDER — enforce the registered laws on the Fork-A probe path (Grok seat)

WRITABLE TARGET /home/vader/GraftRepository-three-pass. Production +
merge-train READ-ONLY. No git. No subagents. GPU under
`flock -w 7200 /tmp/forge-gpu.lock`. Your sandbox may block GPU/model
access — if so, implement + unit-test, then STOP and hand the lead a
one-command runner (the DIAG-CONTAM handoff pattern; it worked).

## Context (read first)

docs/GRM3P_DIAG_UNBOUNDED_REPORT.md — full diagnostic incl. the lead
analysis section "MECHANISM IDENTIFIED". Root cause: the Fork-A probe
path in scripts/grm_e2e_session.py bypasses Arena.step, so three
registered production laws never run: precise-first mounting
(Corpus-100), RECENCY LAW (point lookups exclude recency — "the
previous turn IS the echo source", 2026-06-11), identifier-aware
grounding rejection (core/graft_arena.py ~line 1971).

## The fix (flagged, default-off)

One env/CLI flag (e.g. GRM_PROBE_LADDER=1 / --probe-ladder) that, on
probe turns in the Fork-A path, enforces:
1. Point-lookup detection (identifier-shaped probe): EXCLUDE live/
   recency seats from the attempt (clean context per the RECENCY LAW).
2. Precise-first: if rank-1 covers the probe's identifier tokens,
   mount rank-1 alone; else multi-mount top-k as today.
3. Grounding gate on the answer: identifier-aware rejection + ONE
   retry trip per the existing ladder conventions (reuse arena
   machinery; do not reimplement scoring).
Flag OFF = byte-identical current behavior. No changes to core arena
logic beyond what re-USING its existing functions requires (prefer
zero core edits; driver-level wiring only).

## Gates (registered now)

- G-FIX-DEFAULT: flag OFF — default smoke transcript sha equals
  68da84b8824eb79a14f65a692a602215bc53d84232cd041002410f51657bba4f.
- G-FIX-FULL: flag ON — F-FULL single (`--mode full`, no budget)
  recall 9/9 (t5+t13 flips are the whole point).
- G-FIX-COLD: flag ON — 4MB leg stays 9/9; smoke stays 2/2.
- G-FIX-SUITE: `pytest -q tests/test_grm_*.py tests/test_gqa_ragged_cuda_bank.py`
  green under single AND three_pass.
- Honest RED if any gate fails; no threshold/frame adjustment.

## Done
Verbatim: gate lines with shas/scores, what was wired vs reused,
per-probe table for F-FULL flag-on, session dirs, residuals. If
sandbox-blocked: unit results + runner handoff instead of GPU lines.
