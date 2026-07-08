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
