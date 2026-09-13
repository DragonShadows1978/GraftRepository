# GRM-XM1 X2 — Qwen3.5-9B fails parity on the fresh-fact probes: diagnose (lead order, 2026-09-12)

Seat: Codex Astra (`gpt-6-astra`, high). YOUR WRITABLE TARGET is
`/mnt/ForgeRealm/wt/grm-xm2` (branch `grm-xm2`, forked from grm-xm1 after
amendment 1; the GPU receipts of the barrier and the Qwen cells are copied
in under `artifacts/grm_xm1/amendment_1_run/`). `core/qwen35_tc.py` and
`scripts/grm_xm1_*` edits AUTHORIZED, flag-gated where behaviour changes.
Read-only: canonical `/mnt/ForgeRealm/GraftRepository`,
`/mnt/ForgeRealm/wt/grm-xm1` (a GPU queue is RUNNING there — never touch
it). No git, no subagents, no GPU, foreground, < 10 min per call, never
kill anything; `--basetemp` under `artifacts/grm_xm2/tmp` with cleanup;
repo-relative pins.

## The receipts (lead-run, `artifacts/grm_xm1/amendment_1_run/cells/gpu/qwen35/*.json`)
Full-attention layers, mean over answer positions, C5 fed-text mass vs C3l
mounted-band mass: harbor 0.491/0.513, tundra 0.520/0.537, meridian
0.470/0.531 (parity); **praxis 0.549/0.348 (+0.201), solace 0.505/0.337
(+0.168) — the registered falsifier (> 0.10) fires on both fresh-fact
probes.** GPT-OSS on the same five: all within ±0.06. Also: Qwen's served
text is its THINKING channel in every arm ("Okay, the user is asking…";
value_span correct on 0/10 C5 rows), so the answer-position window is
reasoning tokens.

## Mission
1. Decode: make the Qwen adapter serve the FINAL channel (strip/skip the
   thinking block per the model's chat template; flag `GRM_QWEN35_FINAL_CHANNEL`,
   default ON for XM1 cells, byte-identical OFF), and make "answer
   positions" the final-answer tokens only. CPU double proves the window
   selection on a recorded trace.
2. Diagnose the fresh-vs-superseded split from the receipts BEFORE any
   re-run: per-layer profiles for praxis/solace vs harbor/tundra/meridian
   (where in depth does mounted mass fall short?); the mounted node's
   capture geometry (which node was mounted, its `ntok`, seat position vs
   `live_shift`, whether the fresh-fact node was captured under a
   different pin than the superseded one — RS3's capture-pin lever);
   identifier/scaffold differences between the fresh and sup probe texts
   on Qwen's tokenizer. Register the hypotheses with a falsifier each.
3. Register (do not run) the Qwen re-measurement: the five probes × C3l/C5
   under the final-channel decode, plus for praxis/solace the RS3 lever
   arms on Qwen (capture pin off/live; seat-near-live off/on) — ≤ 0.5
   GPU-h, repo-relative pins, `artifacts/grm_xm2/lead_commands.txt`
   dry-run gated, no outer flock (the worker leases itself).
4. `python3 -m pytest -q --basetemp artifacts/grm_xm2/tmp tests/test_grm_xm1*.py tests/test_grm_xm2*.py`
   verbatim last; ledger + report; prior art (Qwen3.5 chat template /
   thinking mode — cite the source you used, mark unverified).

## Done (verbatim)
1. Files:lines; the final-channel mechanism and OFF byte-identity.
2. The diagnosis table (per-layer, geometry, tokenizer) + registered hypotheses/falsifiers.
3. Registration path + sha; lead commands.
4. Prior art, deviations, RED, process safety, model id + effort; pytest LAST.
