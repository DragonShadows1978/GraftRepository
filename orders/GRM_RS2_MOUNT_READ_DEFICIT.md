# GRM-RS2 — Why is mounted K/V read ~4× weaker than the same text live?

Lead-authored 2026-09-03 night from RS1's decisive negative
(`artifacts/grm_rs1/grm_rs1_results.json`): the exact node text fed
LIVE reads at live-band mass ~0.74 and answers; MOUNTED as a graft it
reads at mounted-band mass ~0.18 and refuses (A2b, all five probes).
Quantization exonerated (A3). The graft is read (A4 strict wording
rescues with the identical mount). Controls read their mounts at ~0.31;
refusers at 0.16–0.18. This order measures the candidate causes of the
gap. Measurement only. No production change. No decision needed.

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/GraftRepository` (branch
`lc1-wip`, HEAD e3f525d) — scripts, tests, artifacts, and BOUNDED GPU
runs AUTHORIZED. A registered order IS the permission: run first,
report after.

## Context (read fully, in order)

1. `orders/GRM_RS1_READ_STRENGTH_PROBE.md`, `artifacts/grm_rs1/`
   (registration, results, per-arm receipts), `scripts/grm_rs1_*.py`
   (reuse the build, the `LayerMassObserver`, the arm runner, the
   table assembly — extend, don't fork).
2. How grafts are captured: `core/graft_arena.py::deposit` (~505, the
   standalone harvest forward: fresh context [sink | text]) vs
   `deposit_from_cache` (~518, harvest from the LIVE cache — K/V
   computed with whatever preceded the text in the window), `feed`,
   and `scripts/grm_det1_3_gpu.py::_install_lived_nodes` (chronological
   `arena.feed` → the sup fixtures' grafts were captured WITH their
   predecessors in the live window). `docs/GRM_Methodology.md` §4
   (lossless mounting; pre-RoPE relocatable keys) — the claim under
   test is that a graft "is the text to the model".
3. Arena geometry: `ArenaCache.__init__` (`n_sink`, `arena_width`,
   `live_shift = n_sink + arena_width`, `extend_rope`), mount seating
   order in `_attempt`/injection, `core/gpt_oss20b_tc.py` attention
   layer types (`full_attention` vs sliding, window 128, sinks) and the
   harmony turn wrapper used by the live feed (`harmony_turn` in
   `grm_e2e_session.py`).
4. The mass instrument: RS1's `LayerMassObserver` over
   `core/grm_demand.py` — it captures FULL-attention layers only
   (sliding layers suppressed, by inheritance from the frozen DET1
   observer). RS2 needs the sliding layers too; extend the probe-script
   instrument (read-only on `core/`) to record per-layer mass for both
   layer types, and state exactly how the sliding-layer partition is
   computed and what "mounted band" means when the window cannot reach
   it.

## Probes

Same five as RS1: `sup_harbor_restatement`, `sup_praxis_fresh`,
`sup_solace_fresh` (refusers), `sup_reserve_meridian_docket`,
`sup_reserve_tundra_ledger` (controls). For solace, ALSO run every arm
with the solace FACT node (node 1, `solace_fact`) mounted explicitly
instead of graft 4, so the read-strength arms are not confounded by
RS1's wrong-graft finding — report both.

## Hypotheses and arms (spec frame, fixes ON, demand OFF; each arm =
same mount SET as RS1 A0, only the named variable changes)

- **B0 — baseline** = RS1 A0 (reproduce; 5/5 required).
- **H-CAPTURE — the graft's K/V carry the deposit-time context.**
  B1: re-capture the A0 node with `deposit(text)` (fresh context,
  [sink | text] only), mount that graft in place of the installed
  one. B1b: re-capture with the node's predecessors REMOVED but the
  question's harmony scaffold present (capture the text as it would be
  fed live: wrapped as a turn). Prediction (lead): B1 raises mounted
  mass on the refusers to ≥ 0.35 and flips ≥ 2/3 to correct; the
  installed grafts' keys were shaped by predecessors the query does
  not share.
- **H-SCAFFOLD — bare text vs chat-wrapped text.** B2: capture the node
  wrapped exactly as A1 fed it live (harmony user/assistant turn),
  mount that. Prediction: small additional gain over B1.
- **H-BAND-POSITION — where in the arena band the mount sits.** B3a:
  seat the A0 graft at the far end of the band (positions adjacent to
  `live_shift`, nearest the question); B3b: at the near end (adjacent
  to the sink). Prediction: modest effect; not the main cause.
- **H-SLIDING — sliding-window layers cannot see the band.** B4: with
  the extended instrument, per-layer mounted mass on sliding vs full
  layers for B0 and for RS1-style A1 (text live); report the gap on
  each layer type separately. Prediction: the gap exists on full
  layers too (RS1 measured 0.18 vs 0.74 on full layers alone), so
  sliding is a contributor, not the whole story; quantify the
  sliding-layer contribution.
- **B5 — the ceiling.** Same text fed live immediately before the
  question (RS1 A1) as the reference row in every table.

If an arm needs a seam that does not exist (seating at a chosen band
offset; capturing wrapped text as a graft; sliding-layer capture),
implement it in the probe script only, or STOP and report it as a
blocker with the site named.

## Gates (synchronous, FOREGROUND, in-call waits only; RED is a result)

Register before any gate run (`artifacts/grm_rs2/registration.json`):
arms, exact node ids and capture texts per probe (derived from
receipts, not typed), predictions above, and the sliding-layer
partition definition. No numeric constants.

G1. `python3 -m pytest -q` sharded under 10 min per call — pre-existing
    set unchanged vs RS1's logs; new CPU tests for the probe script's
    pure parts (arm plan, capture-text derivation, per-layer-type
    partition arithmetic sums to one, table assembly).
G2. **B0 reproduces RS1 A0** on the 5 probes (served text + mounted
    mass within float equality; any drift reported).
G3. **Arms B1–B5** on every reproduced probe (+ the solace-fact
    variant). Table: probe × arm × (served, correct, mounted mass
    full-layers, mounted mass sliding-layers, live mass, sink mass,
    top layer). Then ONE verdict per hypothesis in the registered
    vocabulary: SUPPORTED (the arm closes ≥ half the B0→B5 mass gap or
    flips ≥ 2/3 refusers), PARTIAL (measurable but < half), NOT
    DETECTED (under these arms; name what was not tried), with the
    lead's predictions scored hit/miss.

GPU conventions: `/tmp/forge-gpu.lock` lease via the repo's
`gpu_lease`, single GPU, every run ≤ 10 min wall, ≥30 s gaps inside
the wrapper, operator has absolute right of way. Frozen run tree never
written.

## File boundary

Modify ONLY: `scripts/grm_rs2_*.py` (new; may import from
`grm_rs1_*`), `tests/`, `artifacts/grm_rs2/`, `logs/`. Production code
(`core/`, `config/`) READ-ONLY — needed seams go in the probe script or
are reported as blockers. Read-only: every `scripts/grm_det1_*.py`,
`scripts/lsr_*.py`, `scripts/grm_sc*.py`, `scripts/grm_eb1_*.py`,
`scripts/grm_rs1_*.py` (import only), `docs/`, `orders/`, all receipts.

## Principles (binding)

Determinism. NO git (lead commits). NO subagents. **NO background
processes for waiting**: no `run_in_background`, no watcher/poll loops,
no "block until X exits" helpers; foreground only with explicit
`timeout`, wait in-call, every Bash call under 10 minutes. **Never
kill, signal, or interrupt any process you did not start** (multiple
projects share this machine; no kill/pkill/killall/fuser -k/gpu-reset/
systemctl; pgrep/ps hits are off-limits; wait on the flock or report
blocked). Acknowledge process safety in the report. RED honesty.
Verify your own claims against artifacts before writing them.

## Done (verbatim in the final message)

1. pytest summary line(s) + pre-existing set status.
2. Registration block as written before gates (arms, ids, capture
   texts, partition definition, predictions).
3. G2 reproduction table; G3 full table (incl. the solace-fact
   variant); per-hypothesis verdict in the registered vocabulary;
   predictions hit/miss.
4. Files created/modified; any seam implemented in the probe script
   that production lacks (named — these are the successor's spec).
5. Ambiguities, deviations, residual risks, anything RED; process-
   safety acknowledgement.
