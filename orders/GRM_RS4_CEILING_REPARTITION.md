# GRM-RS4 — Re-partition the live ceiling: how big is the residual really?

Lead-authored 2026-09-04 from RS3 (2712826). RS3's live-band ceiling
(C5: text fed live, no mount) reads "live mass 0.74". But under the
spec frame the live band holds the QUESTION's own tokens too, and in
the mounted arms the question's self-attention shows up as "live mass
~0.29" with nothing else in the band. So the honest comparison for a
mount at 0.41 is not 0.74 but 0.74 minus whatever the question spends
on itself. If that is ~0.29, the residual is ~0.04, not 0.33, and the
mechanism question changes shape. Measurement only. No production
change. CPU if per-key-row attention is persisted; one bounded GPU
re-run otherwise.

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/GraftRepository` (branch
`lc1-wip`, HEAD 2712826) — scripts, tests, artifacts, and BOUNDED GPU
runs AUTHORIZED. A registered order IS the permission.

## Context

1. `artifacts/grm_rs3/grm_rs3_results.json` (C0/C2/C3l/C5 rows, band
   masses), `scripts/grm_rs3_capture_seat_gpu.py` and RS2's
   `LayerTypeMassObserver` (band boundaries: sink `[0,n_sink)`, mount
   `[n_sink, n_sink+cur_mount_n)`, live `[n_sink+cur_mount_n, S)`,
   learned sink column). `core/grm_demand.py::_full_mass`.
2. RS3 registration: `n_sink=19`, `live_shift=115`, seat rule.

## Mission

1. Split the live band into **fed-text rows** and **question rows**
   (and, in generation, answer rows) at every readout position, for
   the RS3 arms C0, C2, C3l, C5 on the five probes + solace-fact.
   Derive row boundaries from the receipts (`S`, mount widths, the
   question's token count through the model tokenizer) — never typed.
2. Report per probe × arm: mass on fed text (C5 only), mass on the
   mount band (mounted arms), mass on question rows, mass on answer
   rows, sink, learned sink; per layer type; and the two comparable
   numbers side by side: **C5 text-mass vs C3l mount-mass**, and the
   residual = their difference.
3. If per-key-row attention is not persisted in the RS3 receipts,
   re-run exactly those arms with an extended observer that records
   the per-row split (bounded, leased; C0 must reproduce RS3 bit-equal
   first).

## Registered prediction (lead)

C5 question-row mass ≈ the mounted arms' live-band mass (0.29 ± 0.05);
C5 fed-text mass ≈ 0.45; residual C5-text minus C3l-mount ≤ 0.08. If
the residual is instead ≥ 0.20, the mechanism question stands as RS3
left it.

## Gates (synchronous, FOREGROUND, in-call waits only)

Register before any gate (`artifacts/grm_rs4/registration.json`,
immutable; amendments separate). G1: pytest sharded, pre-existing set
unchanged vs RS3, new CPU tests for the row-split arithmetic (bands sum
to one; boundaries derived). G2: if re-run, C0 bit-equal to RS3. G3:
the table + residual + prediction hit/miss.

GPU conventions as RS3. File boundary: `scripts/grm_rs4_*.py`,
`tests/`, `artifacts/grm_rs4/`, `logs/` only; `core/` read-only.
NO git, NO subagents, NO background waits (foreground with explicit
`timeout`, every call < 10 min), never kill a process you did not
start; process-safety acknowledgement in the report. RED honesty.

## Done

1. pytest lines + pre-existing set status. 2. Registration + sha.
3. G2 status; G3 table; the residual; prediction hit/miss.
4. Files. 5. Deviations/risks/RED; process safety.
