# GRM-RS1 — Read-strength probe on the three spec-frame refusers

David, 2026-09-03 evening: "run the read-strength probe on the three
refusers." Context: EB1 (68ce3c4) made the ephemeral boat the production
default. Under it, three sup-battery probes that were CORRECT under the
persistent frame now REFUSE with the correct node mounted (identical
`mounted_ids` to the lived run; `live_graft_ids_after_install` went
[2,3] → []): `sup_harbor_restatement`, `sup_praxis_fresh`,
`sup_solace_fresh`. Finding to be explained: the live window, not the
mounted graft, had been carrying those reads. This order measures WHAT
the live window contributed that the graft alone does not. Measurement
only. No production change. No decision needed.

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/GraftRepository` (branch
`lc1-wip`, HEAD 68ce3c4) — scripts, tests, artifacts, and BOUNDED GPU
runs AUTHORIZED. A registered order IS the permission: run first,
report after.

## Context (read fully, in order)

1. `orders/GRM_EB1_EPHEMERAL_BOAT_FRAME.md` and the EB1 receipts:
   `artifacts/grm_eb1/grm_eb1_summary.json`, the G2 arm receipts
   `artifacts/lsr_p2c/lsr_p2c_arm1_*_{cd12f724a2c883f2,329da2bfb86d6937,f239ef35082dc501,642b12e23ae50e19}.json`
   (per-probe `mounted_ids`, `live_graft_ids_after_install`, served
   text, `demand_min_mass` where present), and the G4 demand-ON arm
   (all nine fired — mounted mass is LOW across the board under the
   spec frame).
2. The sup fixtures and installer: `scripts/grm_det1_3_gpu.py::_install_lived_nodes`
   (chronological `arena.feed`; every node pinned `kind="fact"`),
   fixture texts in `scripts/grm_det1_5_gpu.py` (families
   `correction_then_restatement`, `fresh_fact_controls`,
   `multi_hop_a_b_c`, `short_correction_long_competitor`), and
   `scripts/lsr_p2c_replay_gpu.py` (the EB1-frame deposit-order build).
3. The frame seams: `core/grm_frame.py`, `ArenaCache.eb1_begin_turn`,
   `eb1_charge_recency`, `feed(text, deposit=False)` (live-feed
   without deposit), `storage_bits` on the repository/arena
   (`grm_e2e_session.py` arena_kw `"storage_bits": 8`).
4. The mass instrument: `core/grm_demand.py` (production D-NGH
   observer: per-token mounted mass over full-attention layers) and
   the frozen `scripts/grm_det1_common.py::DetectorObserver` (also
   captures per-layer mass rows — use whichever gives per-layer
   sink/mounted/live partition; state which).
5. Lived persistent-frame receipts for the same probes (P2C Arm 1:
   `artifacts/lsr_p2c/lsr_p2c_g2_summary_d5efcb7969279860.json` and
   its arm1 receipts) — the comparison rows.

## Probes

The three refusers + two passing controls for contrast:
`sup_harbor_restatement`, `sup_praxis_fresh`, `sup_solace_fresh`
(refusers); `sup_reserve_meridian_docket`, `sup_reserve_tundra_ledger`
(controls, correct under both frames).

## Arms (each arm = fresh deposit-order build under the spec frame,
fixes ON, demand OFF, then ONE probe serve; fork/reuse the built
repository across arms where the arm does not change deposit)

- **A0 — spec frame, graft only (reproduce EB1 G2).** The baseline.
- **A1 — the crutch isolated: same node TEXT fed live, no graft
  mounted.** `feed(text, deposit=False)` of the node the lived run had
  in the live window (harbor: node 2 `harbor_c`; for each probe use
  exactly the ids EB1 recorded in the persistent frame's
  `live_graft_ids_after_install`), routing disabled for this arm.
  Prediction (lead): correct on all three refusers.
- **A2 — restatement as a MOUNT, not live.** Same node(s) as A1 but
  mounted as grafts alongside A0's mount (this is what recency-as-mount
  would do if the sup installer did not pin `kind="fact"`). Separates
  CONTENT (the restatement text helps) from POSITION (live band vs
  arena band). Prediction: correct on harbor (restatement is content);
  unknown on praxis/solace — report.
- **A3 — unquantized graft.** A0 with `storage_bits` = fp16/none so the
  mounted K/V is the capture-time payload. Prediction: no change
  (quantization is not the cause; INT8 was free on the composed E2E).
- **A4 — instruct-prior control.** A0 with the strict/forced-final
  wording the repo already uses for the GPT-OSS turn-50 needle
  (`docs/GPT_OSS_20B_APA_GRM_SYNTHESIS.md` protocol). Prediction:
  partial — if refusals flip here, the model READ the graft and the
  refusal is the instruct prior, not a retrieval failure.
- **A5 — mass partition on every arm.** Per-token mounted mass AND
  per-layer partition (sink / mounted band / live band / question) at
  the readout positions, for every arm and every probe. Prediction:
  A0 mounted mass on the refusers < the persistent-frame served-control
  masses (0.29–0.44); A1 shows the mass on the LIVE band; A2 moves it
  onto the mounted band.

Do NOT fold separators, retune thresholds, or change any production
path. If an arm needs a code seam that does not exist (e.g. mounting a
specific extra node id, or fp16 storage on this arena class), implement
it in the probe script only, or STOP and report it as a blocker.

## Gates (synchronous, FOREGROUND, in-call waits only; RED is a result)

Register before any gate run (`artifacts/grm_rs1/registration.json`):
arms, probe ids, the exact live/mount ids per probe taken from the EB1
receipts, predictions above. No numeric constants.

G1. `python3 -m pytest -q` sharded under 10 min per call — pre-existing
    set unchanged vs EB1's logs; new CPU tests for the probe script's
    pure parts (arm plan, id extraction from EB1 receipts, table
    assembly, partition arithmetic sums to one).
G2. **A0 reproduces EB1 G2** on the 5 probes (3 refusals + 2 correct;
    served text semantically equal). A probe that does not reproduce is
    excluded from the other arms and reported.
G3. **Arms A1–A5** on every reproduced probe. Table: probe × arm ×
    (served, correct, mounted-band mass at readout, live-band mass,
    sink mass, top contributing layer). Then a one-paragraph
    mechanism verdict per refuser, in the registered vocabulary:
    CONTENT (A2 rescues), POSITION (A1 rescues but A2 does not),
    QUANTIZATION (A3 rescues), PRIOR (A4 rescues; graft was read),
    UNRESOLVED (none rescue). Predictions are the lead's; report
    hits and misses.

GPU conventions: `/tmp/forge-gpu.lock` lease via the repo's
`gpu_lease`, single GPU, every run ≤ 10 min wall, ≥30 s gaps inside
the wrapper, operator has absolute right of way. Frozen run tree never
written.

## File boundary

Modify ONLY: `scripts/grm_rs1_*.py` (new), `tests/`,
`artifacts/grm_rs1/`, `logs/`. Production code (`core/`, `config/`)
READ-ONLY — a needed seam is implemented in the probe script or
reported as a blocker. Read-only: every `scripts/grm_det1_*.py`,
`scripts/lsr_*.py`, `scripts/grm_sc*.py`, `scripts/grm_eb1_*.py`
(import only), `docs/`, `orders/`, all receipts.

## Principles (binding)

Determinism. NO git (lead commits). NO subagents. **NO background
processes for waiting**: no `run_in_background`, no watcher/poll loops,
no "block until X exits" helpers; foreground only, wait in-call, every
Bash call under 10 minutes (use explicit `timeout`; the harness
auto-backgrounds long calls). **Never kill, signal, or interrupt any
process you did not start** (multiple projects share this machine; no
kill/pkill/killall/fuser -k/gpu-reset/systemctl; pgrep/ps hits are
off-limits; wait on the flock or report blocked). Acknowledge process
safety in the report. RED honesty. Verify your own claims against
artifacts before writing them.

## Done (verbatim in the final message)

1. pytest summary line(s) + pre-existing set status.
2. Registration block as written before gates (arms, ids, predictions).
3. G2 reproduction table; G3 full table; per-refuser mechanism verdict
   with the registered vocabulary; predictions hit/miss.
4. Files created/modified.
5. Ambiguities, deviations, residual risks, anything RED; process-
   safety acknowledgement.
