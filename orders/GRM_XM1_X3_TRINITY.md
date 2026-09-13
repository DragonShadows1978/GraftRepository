# GRM-XM1 X3 — Trinity-Nano serves no answer in either arm: diagnose the adapter (lead order, 2026-09-12)

Seat: Codex Astra (`gpt-6-astra`, high). YOUR WRITABLE TARGET is
`/mnt/ForgeRealm/wt/grm-xm3` (branch `grm-xm3`, forked from grm-xm1 after
the GPU run; the 40 GPU receipts are under `artifacts/grm_xm1/amendment_1_run/`).
Trinity adapter code and `scripts/grm_xm1_*` edits AUTHORIZED, flag-gated
where behaviour changes. Read-only: canonical `/mnt/ForgeRealm/GraftRepository`,
`/mnt/ForgeRealm/wt/grm-xm1`, `/mnt/ForgeRealm/wt/grm-xm2` (another seat is
diagnosing Qwen there — never touch it; do not edit `core/qwen35_tc.py`).
No git, no subagents, no GPU, foreground, < 10 min per call, never kill
anything; `--basetemp` under `artifacts/grm_xm3/tmp` with cleanup;
repo-relative pins.

## The receipts (lead-run, `artifacts/grm_xm1/amendment_1_run/cells/gpu/trinity/*.json`)
Full-attention layers, mean over answer positions, C5 fed vs C3l mounted:
harbor 0.816/0.712, praxis 0.803/0.459, solace 0.783/0.672, tundra
0.803/0.655, meridian 0.797/0.602 — the falsifier fires 5/5 — AND the
served text is wrong in EVERY cell of BOTH arms (value_span 0/10),
including C5 where the fact text is fed live. GPT-OSS and MiniCPM3 on the
same panel: parity and correct answers. Your own X0 table says Trinity's
NoPE full-layer keys are unchanged while local-layer keys are relocated,
and flagged Trinity as a fit risk.

## David's principle (binding)
The model cannot tell a grafted file from a tokenized one, so attention to
it cannot differ; a measured gap is an adapter or measurement defect
(position, tokens, capture geometry, band sum, decode window). Find which.

## Mission
1. From the receipts and the adapter code, before any re-run: what did
   Trinity actually serve (raw text per cell)? Is the probe scaffold /
   chat template right for Trinity-Nano (cite the model's template
   source, mark unverified)? Which layers are NoPE vs local-RoPE in this
   model, and how does the harness seat a graft in each (the relocation
   of local-layer keys: are they re-rotated to the seat position, or
   left at capture position)? Does the "answer positions" window fall on
   an actual answer? Does the band decomposition sum to S on Trinity
   rows (it should — check)?
2. Register hypotheses with a falsifier each (template; local-layer
   relocation; fit/width at 96 seats with Trinity's tokenization; capture
   pin), and fix what the CPU double can prove (flag-gated); the RS3
   lever arms (capture pin off/live, seat-near-live off/on) registered
   for Trinity on praxis + harbor.
3. Register (do not run) the Trinity re-measurement: ≤ 0.5 GPU-h,
   repo-relative pins, `artifacts/grm_xm3/lead_commands.txt` dry-run
   gated, worker leases itself (no outer flock).
4. `python3 -m pytest -q --basetemp artifacts/grm_xm3/tmp tests/test_grm_xm1*.py tests/test_grm_xm3*.py`
   verbatim last; ledger + report; prior art.

## Done (verbatim)
1. Served-text table (10 cells) + template/geometry findings (file:line).
2. Hypotheses + falsifiers; fixes landed (flag, OFF byte-identity).
3. Registration path + sha; lead commands.
4. Prior art, deviations, RED, process safety, model id + effort; pytest LAST.
