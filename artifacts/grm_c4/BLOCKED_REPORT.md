# GRM-C4 lead amendment 1 — CPU handoff ready; GPU RED / INCONCLUSIVE

The complete fixed-geometry cross is registered. **66 CPU tests passed; no GPU
run occurred.** Three cells are scheduled at an unvalidated total estimate of
4800s under the lead-authorized 4800s cap. This registers the experiment; it
neither completes GPU execution nor establishes the chunking prediction.

## 1. Amendment path + sha; cells and estimates; the (96,96) ruling

- Amendment: `artifacts/grm_c4/amendment_a3.json` (lead amendment **1**; r1
  already used artifact A1/A2 for implementation corrections).
- SHA-256: `168acfe429f3b1a288ea74c0bc491ae3359c297a4e363d50fe4eedf7a2677c42`.
- Bound order: `orders/GRM_C4_AMENDMENT_1.md`, SHA-256
  `2bb3e59d54f62895a072718613dc4768aae6513b49e7203c94310f83936d03f7`.
- Base registration unchanged: `registration.json`, SHA-256
  `7fb561c70dd8c04abc18ef99a7a2a5ec4d37cbc2a22b1de1f89ab2119bd285e8`.
- A3 also binds A2, which binds A1; active sources and lead_commands are SHA
  checked. Original order, plan, registration and A1/A2 were not edited.

| Cell | Chunk | Width | Treatment | New GPU estimate |
|---|---:|---:|---|---:|
| c64_w96 | 64 | 96 | NEW, run first | 1600s |
| c96_w64 | 96 | 64 | NEW, run second | 1600s |
| c64_w64 | 64 | 64 | NEW fixed-geometry diagonal, run third | 1600s |
| c96_w96 | 96 | 96 | CITE_WC1, geometry pin matches | 0s |

All new cells retain the base capture/seat/live geometry and three batteries:
supersession 9, census 10, full long-history 14. Each has 4 supersession
fixtures, 4 census shards and 13 eight-turn LH segments (104 turns), 21 leases.
Worker285s / outer590s / cooldown30s are unchanged. Planning total:
**4800s = 1.333 GPU-h**, not measured timing. The unchanged full285s reservation
before each worker means exact estimated timings would require a cumulative
reservation of 5005s at the last worker: completion at this cap is not assured.
Stop NON_FIT at the cap/reservation rail, never retry or extend. Cooldowns add
1890s foreground wall time outside GPU budget; total estimate including those
cooldowns is 6690s, excluding CPU scoring and operator overhead.

**(96,96) ruling: reuse is valid for the fixed-geometry quality diagonal.**
Evidence class: archived E2E receipts plus source/registration reasoning.
RS3 registration `geometry_REGISTERED_BEFORE_ANY_GATE` derives sink19,
arena96, live115; its live capture pin is115 and near-live rule puts the plan
head's last token at114. WC1 width96 uses that geometry by construction.

Exact pin: `/mnt/ForgeRealm/GraftRepository/artifacts/grm_wc1_opus/frames/runtime_frame_w96_590846a9fbab95c0.json`,
SHA-256 `590846a9fbab95c083a4f5080d05cb5b04e6851713ba53bb6278e78cd6194fb4`.
All 13 WC1 worker receipts bind this frame, the frozen DET1 frame
`28b3196f8fb04a4123fcf21b1e61f8bb1b8a0e64fadcfc6643801f3e19c4ac02`,
and the live/near-live levers. All 13 historical execution-source fingerprints
match. A3 `overrides.diagonal_ruling` records the RS3 registration SHA, every
worker pin and the derivation; its source inventory additionally binds run
configs and quality receipts. Historical width96 scores are **9/9,9/10,13/14**.
Historical actual token residency remains missing; the WC1 graft-count column
is not reused as token seats. This is not a new GPU parity measurement.

Prediction and rejection are unchanged: **c64_w96 recovers t33, reaches14/14,
and keeps Juniper; reject the chunking claim if improvement needs width64.**
A3 registers c64_w64 against WC1-historical c64_w64 at **8/9,10/10,14/14**,
whose capture/live83 and head82 differ from fixed capture/live115 and head114.
This asks whether the historical benefit survives fixed geometry at all. The
scorer adds componentwise score deltas; all probe rows remain present for
t33/Juniper comparison. This diagonal comparison alone cannot support chunking.
Per-cell scoring leaves the factorial conclusion to the lead after all cells.

## 2. CPU gate results; exact lead commands

- `python -m pytest -q tests/test_grm_c4_campaign.py tests/test_grm_lsr_p2c_split_descent.py`
  -> **66 passed, 2 warnings in 5.63s**, exit0; `cpu_gate_lead_a1.txt`.
  The two warnings and exit-time swigvarlink warning are preserved verbatim.
- `python scripts/grm_c4_campaign.py preflight --dry-run` -> exit0;
  `dry_run_lead_a1.json` lists 4 logical cells, 3 executable cells, 63 worker
  units and 4800s estimate, with active A3 binding and diagonal evidence.
- `bash -n artifacts/grm_c4/lead_commands.txt` -> exit0. The CPU suite checks
  all 63 worker/spec dependencies, cooldowns, rails and immediate cell scores.

New-cell coverage includes synthetic 64-token splitting/parent preservation,
fixed numeric geometry, missing-result refusal, synthetic timeout RED/no retry,
exact unchanged prediction/fixtures/rails, amendment/order SHA drift refusal,
and refusal to launch the cited width96 diagonal. Existing OFF manifest parity,
residency sums and saved-state coverage remain passing. These are author-run
CPU baselines, not blind review or GPU E2E acceptance.

**Exact lead commands:** `artifacts/grm_c4/lead_commands.txt`. It contains the
preflight and CPU gate commands above, then every unit explicitly expanded in
this order: c64_w96 → score, c96_w64 → score, c64_w64 → score. c96_w96 is explicitly
cited with no GPU command. Example first unit and each terminal score:

```sh
timeout --signal=KILL 590s python scripts/grm_c4_campaign.py worker --cell c64_w96 --battery sup --spec correction_then_restatement
sleep 30
# Remaining explicit units are in lead_commands.txt, not omitted from the file.
python scripts/grm_c4_campaign.py score --cell c64_w96
python scripts/grm_c4_campaign.py score --cell c96_w64
python scripts/grm_c4_campaign.py score --cell c64_w64
```

Each score is placed after its own complete cell in the file. Execute individual
commands in separate foreground invocations; stop on any nonzero exit. Do not
execute the whole 63-unit file as one call. The unchanged r1 outer timeout is
for the lead's own started worker only. No GPU/timeout/sleep command ran here.

## Prior art

Verified local house **RS3, WC1, EB1, DET1 and C4 (2026)**: borrowed numeric
geometry derivation, immutable SHA amendment chain, registered cell enumeration,
per-battery score deltas, session resume and foreground lease. This amendment
adds the lead-authorized diagonal, cap, pins and completion schedule. No novel
algorithm claimed. The original external **Fisher, The Design of Experiments
(1935)** lead remains **unverified — lead to check**; search terms: `Fisher
factorial experiments 1935`. No external literature conclusion is asserted.
Annotations are at changed code sites and in IMPLEMENTATION_LEDGER.md.

## Deviations; RED; process safety; model id and effort

The lead's cap increase and added diagonal are authorized scope changes. Artifact
A3 means lead amendment1, preserving prior A1/A2. Source-only amendment overlays
and per-cell command order changed; core, kernels, batteries, registries, config,
flag defaults and the adapter did not. A registration-writer schema selection
error (`KeyError: 'shard'`) occurred before A3 existed; corrected by selecting
worker receipt schemas, recorded in the ledger. No failed CPU gate was hidden.

RED: no GPU results exist; t33 recovery, Juniper retention, geometry parity on
GPU, historical benefit survival and chunking causality are **not claimed fixed**.
Full-cross GPU verdict remains INCONCLUSIVE pending execution. The scheduling
gap is resolved; measured budget fit and historical token residency remain open.
No existing r1 GPU receipt was reused or invalidated. Blind review is lead-owned.

No git commands, subagents, GPU/model load, background waits/jobs, process
signals/kills, lock access/clearing, live-service changes or external delivery
occurred. All work ran in bounded foreground CPU calls. Preserved old handoff
and reports as *_r1 files; current evidence is this report and A3 receipts.
Model: **GPT-6 (Codex; finer deployment id not exposed)**. Effort: **high** per order.
