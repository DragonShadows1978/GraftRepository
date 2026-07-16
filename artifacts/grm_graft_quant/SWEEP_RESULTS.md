# GRM Graft Storage Quantization — P1/P2 Sweep Results

Work order: `docs/GRM_GRAFT_QUANT_PLAN.md` (P1 harness + P2 sweep).
Ledger: `docs/GRM_GRAFT_QUANT_LEDGER.md`. P0 receipts:
`artifacts/grm_graft_quant/P0_FORMAT_AND_BATTERY.md`.
Repo: `grm-cuda-bridge-overhead` @ 1bfc23d. Date: 2026-07-08.
Engine: Project-Tensor main @ acc9994 (same as P0 baseline).
Host: MilleniumFalcon, RTX 4070 SUPER 12 GB. Attention mode
`standard` held for every run (P0 risk 3). Scope law: storage transform
of witnessed K/V banks — NOT APA, never synthesis.

Harness (new scripts only, product code untouched):
- `scripts/grm_graft_quant_transform.py` — quantize-at-rest: uniform
  symmetric per-group (group-32 along **head_dim**, the last/feature axis
  of the (1,8,S,64) shards — same axis convention as
  `tensor_cuda.quantization.quantize_affine_per_group`, which groups
  in_features, and `_kv_quant`, which reduces over D; grouping across
  tokens would mix unrelated values). Parameter is named `storage_bits`
  everywhere — never `bulk_bits` (that is an APA knob; P0 risk 5).
  Output = quantized-then-DEQUANTIZED fp16 graft dir (the zero-product-code
  hook P0 identified: gates mount any fp16 dir unchanged).
- `scripts/grm_graft_quant_sweep.py` — per-depth runner: transform all 3
  banks → run the P0 battery (`--skip-capture --graft-dir <quantized>`,
  addressing via `--source <fp16 multifact JSON> --graft-dir <quantized>`)
  → extract per-case margins (answer_logit − best other top-10 logit,
  identical formula to P0's `extract_margins.py`, extended to handle the
  addressing gate's list-shaped `runs`, which the P0 extractor cannot
  parse) → `sweep_<tag>/result.json` + `SWEEP_CUMULATIVE.json`. Spacing
  75 s between gate runs; `nvidia-smi` foreign-process check before each
  (wait ≤15 min in 60 s steps — never triggered, GPU was clear every time).

## Harness parity gate (bits=16 identity)

`--storage-bits 16` is bit-exact identity on all 3 banks (72 shards,
`np.array_equal` on every k/v pair): receipt in
`sweep_noise_fp16/banks/*/quant_receipt.json` (`bit_exact_at_16: true`,
RMSE 0.0 everywhere). Identity is against the existing fp16-at-rest path
(P0 note: capture already includes one bf16→fp16 round-trip).

## Noise band (E1 denominator)

One full fp16 re-run of the battery (unmodified banks via the identity
transform, tag `noise_fp16`) vs the P0 baseline: **all 16 mount margins
bit-identical, all verdicts identical (max |Δ| = 0.0)**. Control top
tokens/logits also bit-identical across all five batteries. The engine is
deterministic for this battery; run-to-run noise band = **0.0 logits**
(logit resolution 0.0625). Any margin movement in the sweep is therefore
signal attributable to the storage transform, not run noise.

Status-field note: `sweep_noise_fp16/result.json` records
`status: "gate_error"` — harness v1 marked any non-zero gate exit as an
error, and the preference gate exits rc=1 on its KNOWN fp16 3/4 fixed-fail
(`preferred_color` control-confound + case-strict scoring, P0 risk 2). The
status logic was corrected before the depth runs (crash = no parseable
margins JSON; a verdict-fail is data). The noise-run artifact was left
as-written to preserve receipts; its gate data is complete and valid.

## The curve — depth × gate → verdict + per-case mount margins

Margins: answer_logit − best other top-10 logit at the mount run
(positive = answer wins). "absent" = answer not in mount top-10 at all.
fp16 column = P0 baseline ≡ noise re-run (bit-identical).

| gate / item | fp16 | 8-bit | 6-bit | 4-bit | 3-bit |
|---|---:|---:|---:|---:|---:|
| **multifact** | **PASS 4/4** | **PASS 4/4** | **PASS 4/4** | **FAIL 3/4** | **FAIL 1/4** |
| vault_keyword | +1.625 | +1.75 | +2.125 | +2.25 | absent |
| relay_marker | +4.875 | +4.875 | +4.375 | **−1.125 (r1)** | absent |
| archive_color | +4.125 | +3.875 | +3.875 | +3.0 | +0.4375 |
| tool_metal | +4.375 | +4.375 | +4.0 | +2.75 | absent |
| **preference** | **FAIL 3/4** (known) | **PASS 4/4** | **FAIL 3/4** | **FAIL 2/4** | **FAIL 0/4** |
| preferred_color (furniture) | −0.25 | **+0.125** | −0.5 | −0.25 | −0.75 (r1) |
| preferred_signal | +4.625 | +4.625 | +4.25 | +2.5 | −0.625 (r4) |
| preferred_shade (needle) | +0.5 | +0.375 | +0.375 | **−1.0 (r1)** | −0.75 (r2) |
| preferred_metal | +2.375 | +2.875 | +1.875 | +2.0 | −0.625 (r1) |
| **supersession** | **PASS 4/4** | **PASS 4/4** | **PASS 4/4** | **PASS 4/4** | **FAIL 0/4** |
| vault_keyword (needle) | +0.125 | +0.125 | +0.25 | +0.75 | −2.1875 (r5) |
| relay_marker | +4.5 | +4.5 | +4.5 | +5.75 | absent |
| archive_color | +1.25 | +1.75 | +1.875 | +2.0 | **−1.625 (r1, STALE WINS)** |
| tool_metal | +2.125 | +2.125 | +1.75 | +3.0 | −1.875 (r9) |
| **addressing fact_local** | **PASS 4/4** | **PASS 4/4** | **PASS 4/4** | **PASS 4/4** | **FAIL 1/4** |
| vault_keyword | +3.625 | +3.5 | +4.25 | +0.625 | absent |
| relay_marker | +5.625 | +6.125 | +5.625 | +3.5 | absent |
| archive_color | +4.125 | +4.0 | +3.875 | +2.5 | +0.625 |
| tool_metal | +4.375 | +4.25 | +4.375 | +3.5 | absent |

**2-bit: SKIPPED** per the early-stop clause (all four gates fail at
3 bits — battery catastrophically broken; deeper depth uninformative).
The chained depth-3→depth-2 invocation exited after depth 3 BY DESIGN:
its built-in guard parsed `sweep_depth3/result.json`, found all four gate
classifications `fail`, printed `DEPTH3_CATASTROPHIC: all gates fail —
skipping depth2 per early-stop rule`, and ended cleanly (depth-3 exit 0,
`CHAIN_DONE` in the chain log). Not a watcher/chain crash.

**Last depth where the battery stays green: 6 bits** (multifact /
supersession / addressing PASS, preference 3/4 = exactly the fp16
baseline verdict with the same furniture item). 8-bit is green and
strictly ≥ baseline (16/16 mount hits — one better than fp16).
**Cliff shoulder: 4 bits** (first substantive losses). **Cliff floor:
3 bits** (2/16 mount hits, answers falling out of top-10 entirely).

### Anomalies (verbatim, not smoothed)

- `preferred_color` is non-monotonic across the whole curve:
  −0.25 FAIL (fp16) → **+0.125 PASS (8)** → −0.5 FAIL (6) → −0.25 FAIL (4)
  → −0.75 FAIL (3). The fp16 "fixed-fail furniture" item PASSED at INT8 —
  quantization noise nudged the case-strict "blue" vs "BLUE" battle across
  the line. Margin IMPROVING at lower bits, then regressing.
- Margins IMPROVE under heavier quantization on several items through
  4 bits: supersession vault +0.125→+0.75 (monotone WIDENING to the
  shoulder), supersession relay +4.5→+5.75 at 4, supersession tool
  +2.125→+3.0 at 4, multifact vault +1.625→+2.25 monotone rising.
- At 4 bits, within one bank at identical RMSE: relay_marker (the
  STRONGEST fp16 multifact margin, +4.875) collapsed to −1.125 while
  vault_keyword (the weakest, +1.625) rose to +2.25. Collapse is
  item-selective, not margin-ordered — P0's "thinnest margins die first"
  held for preference/shade but inverted for multifact.
- Stale-value resurrection (the supersession axis): at 4 bits stale
  values climb the mount top-10 (vault old RED logit 15.8→19.25 at
  ranks 2-3, relay old STONE at rank 1) while current still wins; at
  3 bits `archive_color`'s stale **BLACK outright beats current GRAY**
  (rank 0 vs margin −1.625) — quantization noise resurrected a
  superseded memory, the qualitatively legible failure P0 predicted.

## Needle trajectory

| needle (fp16 margin) | fp16 | 8-bit | 6-bit | 4-bit | 3-bit |
|---|---:|---:|---:|---:|---:|
| supersession/vault (+0.125) | +0.125 | +0.125 | +0.25 | +0.75 | −2.1875 (r5) |
| preference/shade (+0.5) | +0.5 | +0.375 | +0.375 | **−1.0 LOST (r1)** | −0.75 (r2) |

Shade behaved as forecast (first substantive casualty, dead at 4).
Vault did the opposite: monotone widening 0.125→0.75 through the
shoulder, then annihilated at the floor (answer at rank 5, −2.19). The
"thinnest margin = early-warning needle" heuristic is only half-true
under this transform.

## Reconstruction RMSE per depth (multifact bank; other banks within ±0.001)

| bits | K RMSE mean (max layer) | V RMSE mean (max) | K max-abs err | V max-abs err |
|---:|---:|---:|---:|---:|
| 16 | 0.0 (bit-exact) | 0.0 | 0.0 | 0.0 |
| 8 | 0.0390 (0.0958) | 0.0142 (0.0262) | 0.53 | 0.23 |
| 6 | 0.1532 (0.3683) | 0.0581 (0.1073) | 2.14 | 0.95 |
| 4 | 0.5457 (1.0427) | 0.2575 (0.4753) | 8.42 | 4.38 |
| 3 | 0.9678 (1.9975) | 0.6009 (1.1107) | 20.12 | 10.25 |
| 2* | 1.7255 | 1.7236 | 51.75 | 30.0 |

*2-bit row from the harness-validation transform of the supersession bank
(measured, then discarded; never swept — gates were not run at 2 bits).
K error > V error at every depth except 2, where V catches up (V's
distribution is flatter; at 2 bits both saturate). Per-layer receipts:
`sweep_depth*/banks/*/quant_receipt.json`.

## E1 / E2 / E3 verdicts (plan's registered expectations)

- **E1 (INT8 free): CONFIRMED, with a sharpening.** The noise band is
  exactly 0.0 (deterministic engine), so "within noise" is unattainable
  literally — margins DID move at INT8 (max |Δ| 0.5, mean |Δ| 0.16,
  two-sided: 5 items improved, 6 eroded, 5 unchanged). But every
  substantive verdict held, no rank changed adversely, and the battery
  scored 16/16 (better than fp16's 15/16). INT8 is free at the verdict
  level and ±0.5-logit-bounded at the margin level.
- **E2 (a cliff exists ≥2 bits): CONFIRMED.** Shoulder at 4 bits
  (multifact FAIL 3/4 — relay lost; preference 2/4 — shade lost), floor
  at 3 bits (1/4, 0/4, 0/4, 1/4; most answers absent from top-10; one
  stale value wins). The cliff sits between 6 and 3 with visible
  structure at 4: the plan's K/V-asymmetry follow-up (P3) has a
  well-localized target.
- **E3 (RMSE correlates with margin erosion): PARTIAL — structural break
  flagged.** In aggregate, deeper = worse, as expected. At the item
  level the correlation breaks exactly the way the plan told us to watch
  for: at 4 bits, K-RMSE ≈ 0.55 (≈ 4.4× the 6-bit value) yet HALF the
  battery's margins moved UP, and the collapse hit the strongest-margin
  item (relay +4.375→−1.125) while sparing the weakest. Margin movement
  under this transform is a rank/binding phenomenon (specific K/V
  directions breaking specific attention bindings — and at 3 bits,
  redistributing weight toward OTHER planted bindings, hence the stale
  win), not proportional amplitude noise. Flagged, not papered over.

## Disk multipliers

**What this sweep measured: recall only, via quantized-then-DEQUANTIZED
fp16 dirs — no packed bytes were written to disk; on-disk sizes of the
sweep banks are unchanged fp16 (~193 MiB each).** True packed-file
emission is P3 / format-1 work (per P0: the dequantized-dir approach is
deliberately zero-product-code). Multipliers below are therefore
theoretical; two variants given because the plan's headline numbers
(2/2.7/4/5.3/8×) count raw code bits only, while the group-32 scheme
carries one fp16 scale per 32 values (+0.0625 B/val):

| bits | raw multiplier (plan) | with group-32 fp16 scales | 4k-token graft (201.3 MB fp16) |
|---:|---:|---:|---:|
| 8 | 2.00× | 1.88× | 107 MB |
| 6 | 2.67× | 2.46× | 82 MB |
| 4 | 4.00× | 3.56× | 57 MB |
| 3 | 5.33× | 4.57× | 44 MB |
| 2 | 8.00× | 6.40× | 31 MB |

Headline pairing: **INT8 → 1.9× disk, verdict-free. INT6 → 2.5× disk,
battery still green vs baseline. INT4 → 3.6× disk, first recall
casualties. INT3 → 4.6× disk, memory effectively destroyed (and
occasionally falsified — stale wins).**

## Runtime / host receipts

- Five batteries (fp16-noise, 8, 6, 4, 3): gate-subprocess wall
  5,886 s ≈ **98 min GPU-active total**, as 20 separated runs of
  287–307 s each (every run inside the ≤10-min bound), 75 s idle between
  runs, ~2–20 min between batteries. Runner walls: 1202/1404/1399/1403/
  1407 s (include spacing + CPU-only transforms ~10 s/bank).
- Power law respected: no sustained multi-hour draw; longest continuous
  GPU stretch ≈ 5 min.
- GPU state before every battery: idle, 277–289 MiB desktop residue,
  43–53 °C, zero foreign compute processes (the ≤15-min wait loop never
  engaged). Controls bit-identical across all batteries (determinism
  receipt, doubles as engine-state invariance check).
- Transform CPU cost: ~10 s/bank/depth (NumPy, no GPU).

## Receipt paths

- Per-depth: `artifacts/grm_graft_quant/sweep_{noise_fp16,depth8,depth6,depth4,depth3}/`
  — `result.json` (cumulative per-depth), `{multifact,preference,supersession}_gate.json`
  + run dirs + logs, `addressing_fact_local_gate.json`,
  `banks/*/quant_receipt.json` (per-layer RMSE), `transform_*.log`.
- Cross-depth index: `artifacts/grm_graft_quant/SWEEP_CUMULATIVE.json`.
- Harness: `scripts/grm_graft_quant_transform.py`,
  `scripts/grm_graft_quant_sweep.py`.
- fp16 origin: P0 artifacts (unchanged), frozen banks
  `artifacts/grm_graft_quant/{multifact,preference,supersession}_4k_fp16_gate/graft/`.

## Registered limits (carried forward)

- Format 2 only (P0 risk 1): a format-1 (`nodes/NNNN.npz` /
  `unpack_node`) mount receipt is still follow-up work; cliff transfers
  byte-wise but the mount path differs.
- Margins over verdicts for `preferred_color` (P0 risk 2) — its verdict
  flapping across depths (fail/pass/fail) is case-scoring noise, not
  recall signal.
- 2-bit transform validated (RMSE receipt above) but never gated;
  the 2-bit row of the curve is formally SKIPPED, not failed.
- Uniform K=V depth only; K8V4/K4V8 asymmetry is P3, now aimed at the
  4-bit shoulder where K-vs-V structure is most likely to show.
