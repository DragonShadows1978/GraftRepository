# GRM Graft Storage Quantization Ledger

Execution record for the graft storage quantization work order. Immutable
plan: `docs/GRM_GRAFT_QUANT_PLAN.md`. Narrative continues in
`docs/GPT_OSS_20B_APA_GRM_SYNTHESIS.md`.

## 2026-07-08 (opening + P0 complete)

Action: Work order opened (David: "let's look at quantizing GQA grafts
at different levels and see where things break"). P0 inventory + fp16
baseline landed (Sonnet, flat), receipts committed before any transform.

Findings (evidence class: code inspection + gate runs on the current
engine, Project-Tensor main @ acc9994):
- TWO at-rest formats, both fp16 .npz, PRE-RoPE K/V. Sweep targets
  format 2 (GPT-OSS capture dirs, `layer_NNN.npz`, k/v (1,8,S,64) fp16,
  uncompressed; written stream_forward_smoke.py:454-457; mounted via
  load_graft_layer:207-216 → inject_kv). Format 1 = production node
  store (nodes/NNNN.npz zlib, graft_repository.py:3686-3759; dequant
  hook = unpack_node, graft_arena.py:2168-2173). REGISTERED LIMIT: the
  sweep validates format 2 only; a format-1 mount receipt is follow-up.
- Footprint at GPT-OSS geometry: 98,304 B/token → 201.3 MB at 4k tokens
  (disk-verified), ~4.8 GB at 96k. Plan multipliers apply to this term.
- Sweep battery (all consume pre-captured banks TODAY, no P1 reuse
  work): multifact + preference + supersession + addressing(fact_local),
  ~20-25 min/depth; bulk@4K optional anchor; exact_value (35 min) and
  repetition_drift excluded over budget; realtext_ppl out of scope
  (mounts no grafts). Frozen fp16 banks for the sweep saved under
  artifacts/grm_graft_quant/ (~193 MiB each, 3 banks).
- fp16 baseline (attention-mode STANDARD, held for the whole sweep —
  mode moves supersession's verdict): multifact 4/4 (+1.6/+4.9/+4.1/
  +4.4), supersession 4/4 (vault +0.125 — stale value in top-10),
  addressing 4/4 (+3.6..+5.6), preference 3/4 (color −0.25 = known
  control-hit furniture; track margins not verdicts). CLIFF NEEDLES:
  supersession/vault (+0.125) and preference/shade (+0.5) will feel
  noise long before the +4 facts.
- Risks registered: format-1 untested by sweep; --bulk-bits is an APA
  knob, storage bits must be named distinctly in the harness; docs
  corpus drifted (carry original capture receipts with quantized
  banks); NO noise band yet — one fp16 re-run required before the sweep
  so E1 has a denominator.
- Receipts: artifacts/grm_graft_quant/P0_FORMAT_AND_BATTERY.md + gate
  JSONs/logs/extract_margins.py.

Next action: P1 harness (quantize-at-rest, storage-bits naming, 16-bit
identity + RMSE receipts) + fp16 noise re-run, then the P2 sweep
(8/6/4/3/2), bounded spaced runs.

## 2026-07-08 (P1+P2 complete — WORK ORDER CLOSED)

Action: Harness built, sweep run, final receipts landed
(artifacts/grm_graft_quant/SWEEP_RESULTS.md + per-depth dirs +
SWEEP_CUMULATIVE.json). Depth-2 skipped per the plan's early-stop clause
— invoked by the chain's OWN guard (chain log: DEPTH3_EXIT:0 →
DEPTH3_CATASTROPHIC → CHAIN_DONE).

THE CURVE (group-32 symmetric, K+V uniform, dequant-on-mount, STANDARD
attention, noise band 0.0 — engine fully deterministic, fp16 re-run
margins bit-identical):
- 8 bits (1.88× disk incl. scale overhead): FREE — 16/16 cases green,
  deltas ≤ ±0.5. E1 CONFIRMED.
- 6 bits (2.46×): LAST GREEN DEPTH — battery = baseline. The headline
  operating point.
- 4 bits (3.56×): CLIFF SHOULDER — multifact FAIL (relay_marker, the
  STRONGEST fp16 margin +4.875, inverts to −1.125), preference 2/4,
  margins compressed system-wide. E2 shoulder.
- 3 bits (4.57×): FLOOR — all four gates fail; most answers absent from
  top-10. E2 CONFIRMED (cliff well above 2 bits).
- E3 PARTIAL — STRUCTURAL BREAK FLAGGED: at 4 bits, identical bank RMSE
  (K≈0.55) collapses the strongest margin while the weakest widens
  (vault +0.125→+0.75 monotonically to the shoulder, then −2.19).
  Amplitude noise does not explain the damage pattern; binding/rank
  structure does.

FINDING OF RECORD — QUANTIZATION RESURRECTS SUPERSEDED MEMORIES: at 4
bits, stale (deliberately overwritten) values begin climbing the
rankings; at 3 bits a stale value WINS (archive: superseded BLACK beats
current GRAY). For a memory system, storage quantization below the safe
floor does not merely forget — it un-supersedes. This failure mode, not
disk economics, is the sharpest reason the floor is a law: corrections
and retractions are the first casualties.

Anomalies (verbatim, not smoothed): preferred_color non-monotonic
(fail→pass@8→fail@6); several margins IMPROVE under quantization through
4 bits; vault widens monotonically before annihilation.

Ops: ~98 min GPU-active as 20 separated 287-307s runs (all within the
10-min bound), temps ≤57°C, no foreign process. Correction to the
2026-07-08 interim characterization: the depth-3→2 chain exit was BY
DESIGN (early-stop guard), not a process failure — though the agent
wake-notification after exit was genuinely lost (as it was twice
earlier; polling lesson recorded in lead memory).

Registered follow-up (P3, not started): packed on-disk format + dequant
hooks at the two P0-mapped sites (load_graft_layer, unpack_node), format-1
coverage, and the K/V-asymmetric sweep if the 4-6 bit gap ever matters.
Recommended production setting pending P3: INT8 at rest (conservative,
proven free) with INT6 available where disk pressure demands (green, one
depth of margin above the shoulder).

## 2026-07-08 (P3 opened — packed format; gates registered before implementation)

Action: David directed P3 execution (session goal). Scope as registered
at close: packed on-disk format + dequant hooks at the two P0-mapped
sites + format-1 coverage. Gates frozen here, before any code:

- Packed format: per-layer packed uint8 payload + per-group scales
  (group-32 symmetric, storage_bits ∈ {8,6}), explicit format-version
  field. Fail-closed: unknown version/bits → hard error, never a silent
  misread. Default behavior everywhere UNCHANGED (fp16); packed is
  opt-in.
- Round-trip gate: pack→unpack must be BIT-IDENTICAL to the sweep
  transform's dequantized output for the same bits (the P1 transform is
  the reference implementation; deterministic engine ⇒ exact match
  expected).
- Recall gate: battery re-run ONCE against real packed bytes at INT8
  (INT6 spot-check) — margins must equal the sweep's dequantized-dir
  results bit-for-bit (noise band 0.0 makes this a hard equality gate).
- Disk receipt: ACTUAL on-disk sizes both formats (format-1 is
  zlib-compressed — measured multipliers, not theoretical).
- Mount-speed receipt: packed vs fp16 load time (the NVMe page-in
  bonus), labeled instrument + state.
- Suites: router baseline + native runtime green.

Next action: P3 implementation (Sonnet, flat, no-delegation).

## 2026-07-08 (P3 complete — WORK ORDER FULLY CLOSED)

Action: Packed format shipped (Sonnet, flat), all registered gates green,
committed.

- core/graft_quant.py: shared pack/unpack core (P1 math refactored, not
  reimplemented; transform script now delegates — byte-identical vs the
  committed original, all depths). Fail-closed on unknown
  format_version/storage_bits. DEVIATION (flagged, resolved toward the
  stricter gate): scales stored fp32, not the plan's fp16 — fp16 scales
  broke the mandatory bit-identity round-trip (1-ULP drift, 10-22% of
  elements). Disk numbers below are the honest fp32-scale figures.
- Hooks: format-2 load_graft_layer transparent detect+dequant (+
  standalone pack tool, capture path untouched); format-1
  pack_node/unpack_node opt-in via storage_bits arg / GRM_GRAFT_STORAGE_BITS
  env. Defaults byte-identical everywhere.
- Gates: round-trip BIT-IDENTICAL (3 banks × {8,6,4,3}, exceeds
  registered scope); recall equality vs sweep BIT-IDENTICAL margins
  (multifact@8, multifact@6, supersession@6 incl. the +0.25 needle);
  suites 21/21 + 117/117 + 22 new format tests.
- Disk (MEASURED): format-1 zlib INT8 1.793×, INT6 2.326×; format-2
  uncompressed 1.777× (INT8; sub-8-bit packing not implemented, INT6
  gains only under compression).
- HONEST NEGATIVE — mount speed: packed INT8 mounts 3.76× SLOWER
  (43.4 vs 11.6 ms/layer, N=20 median, warm page cache — CPU dequant
  dominates; the predicted NVMe page-in win did not materialize
  locally). Policy implication registered: INT8 = archive tier;
  hot-mount tier stays fp16 pending GPU-side or vectorized dequant.
  The E2E receipt will price this seam at session cadence.

WORK ORDER FULLY CLOSED. Successor: E2E receipt (own plan/ledger) is the
first production consumer of the packed format.
