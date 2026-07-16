# P0 — Graft Storage Format Receipt + Gate Battery Inventory + fp16 Baseline

Work order: `docs/GRM_GRAFT_QUANT_PLAN.md` (P0 a/b/c). Scope law: storage
quantization of at-rest witnessed K/V banks — NOT APA, never synthesis.
Repo: `/mnt/ForgeRealm/GraftRepository` @ branch `grm-cuda-bridge-overhead`
(tip f6286c8). Date: 2026-07-08 (early AM EDT).

---

## (a) Graft storage format receipt

TWO distinct at-rest formats exist in this repo. The GPT-OSS gate battery
uses format 2; the turn-based memory system uses format 1. Both store
fp16 numpy arrays in `.npz`; the quantize-at-rest transform (P1) must
target format 2 first (that is what the sweep battery mounts), with
format 1 as the production follow-on.

### Format 1 — GraftRepository per-node store (turn/document memory)

Store/save path:
- `core/graft_repository.py:3475-3476` `_payload_file_path(i)` →
  `<repo>/nodes/NNNN.npz` (one file per graft node).
- `core/graft_repository.py:3686-3759` `flush_now()` — the checkpoint:
  per dirty node `_atomic_savez_compressed(f, **g["host_payload"])`
  (line 3720; zlib-compressed npz, atomic tmp+rename+fsync via
  `_atomic_savez_compressed`, lines 3020-3037), then `index.npz`
  (routing keys, line 3730-3731), then `manifest.json` (dialect,
  node metadata, WAL lsn; lines 3740-3751). WAL at `<repo>/wal/000001.wal`
  (`_wal_path`, line 3039-3040) carries text/metadata records, NEVER K/V
  payload bytes (NODE_UPSERT records only `has_payload` flag,
  line 3604-3608).
- Payload dict content is dialect-owned (`pack_node`):
  - GQA (`core/graft_arena.py:2162-2166`): `{"k": (L,H,S,D) fp16,
    "v": (L,H,S,D) fp16}` — L layers stacked, H kv-heads, S tokens,
    D head_dim, PRE-RoPE.
  - MLA (`core/graft_arena.py:274-279`): `{"c": (L,S,256) fp16,
    "kpe": (L,S,32) fp16}`.
- Routing index: GQA stores one variable-length array per node
  (`rkey_NNNN`, `core/graft_arena.py:2175-2180`); MLA a stacked
  (N,256) fp32 matrix (`core/graft_arena.py:289-295`).

Load/mount path (disk → RAM → VRAM):
1. `load()` (`core/graft_repository.py:3765-3815`): manifest+index read;
   non-retired nodes get `host_payload` read from `nodes/NNNN.npz`
   (`_read_payload_file`, line 3478-3480: `np.load` → `_payload_to_ram`
   → contiguous numpy in RAM), then `arena.unpack_node(host_payload)`
   uploads to device if within `vram_budget` (line 3805-3811).
2. On-demand (descent/pager): `ArenaCache._ensure_h`
   (`core/graft_arena.py:1221-1234`) — a mount of a paged-out node calls
   `node_loader(i)` = `GraftRepository._load_node`
   (`core/graft_repository.py:2956-2968`): RAM-first (`host_payload`
   if present, else `_read_payload_file` from NVMe), then
   `arena.unpack_node` → device.
3. `unpack_node` GQA (`core/graft_arena.py:2168-2173`): per layer
   `tc.tensor(np.ascontiguousarray(k[li][None])).astype(COMPUTE_DTYPE)`
   — the fp16-numpy → device-tensor conversion. THE natural
   dequantize-on-mount hook for format 1: dequantize INSIDE
   `unpack_node` (or in `_read_payload_file` on the RAM copy) before
   `tc.tensor(...)`; attention math unchanged downstream.
4. `swap(picks)` (`core/graft_arena.py:1385-1420`) does cache surgery
   with the already-device-resident tensors (`_graft_block`,
   line 1236-1247: concat + re-RoPE of K at arena seats). No per-swap
   host→device upload — mounts reuse `g["h"]`.

### Format 2 — GPT-OSS capture-harness graft dirs (what the gates mount)

Store path:
- `scripts/gpt_oss20b_stream_forward_smoke.py:454-457`: captured K/V
  (bf16 compute) are explicitly cast `astype(np.float16)` then
  `np.savez(shard, k=k_np, v=v_np)` — UNCOMPRESSED npz, one file per
  layer: `<graft_dir>/layer_NNN.npz`, plus `<graft_dir>/manifest.json`
  (schema `gpt_oss_20b_prerope_graft_manifest_v1`, per-layer
  `host_bytes`/shapes/`token_count`; its `dtype: bfloat16` records the
  COMPUTE dtype — at-rest storage is fp16, verified float16 on disk).
  So the CURRENT baseline already includes one bf16→fp16→bf16 storage
  round-trip; P1's "quantize at 16 ≡ identity" check is identity
  against this existing fp16-at-rest path, not against raw bf16.
- Shard content: `k (1, 8, S, 64) float16`, `v (1, 8, S, 64) float16`,
  PRE-RoPE (verified on
  `artifacts/gpt_oss_20b/h6_multifact_4k_gate/graft/layer_000.npz`).

Mount path (disk → VRAM, per gate-run subprocess; no persistent RAM tier):
- `scripts/gpt_oss20b_stream_forward_smoke.py:207-216`
  `load_graft_layer(root, layer_idx, dtype)`: `np.load` →
  `np.ascontiguousarray(z["k"]/z["v"])` (RAM) → `tc.tensor(...)
  .astype(BlockTC.COMPUTE_DTYPE)` (bf16, VRAM upload) — called per layer
  inside the streamed layer loop (line 431-436), then
  `block.self_attn.inject_kv = (kg, vg, 1.0)`;
  `graft_seats = token_count`. Live tokens RoPE-shift after graft seats
  (`rope_len = ids + mounted_graft_tokens`, line 381).
- THE natural dequantize-on-mount hook for format 2 (and for the P1
  sweep): inside `load_graft_layer` between `np.load` and
  `tc.tensor(...)` — a packed shard (e.g. `k_q`,`k_scale`,`v_q`,
  `v_scale` keys) dequantizes to fp16 numpy there; zero change to
  attention math or the injection interface. Since P1 is
  harness-only work, an equivalent wrapper script that unpacks
  quantized shards into a temp fp16 graft dir also satisfies the
  contract without touching product code.

Reusable quantization primitives already in-tree:
- `core/mistral7b_tc.py:224-244` `_kv_quant`/`_kv_dequant` — symmetric
  uint8 KV-cache quant, per-(B,H,S)-vector scale, q=round(x/s)+128,
  s=max|x|/127; dequant returns COMPUTE_DTYPE.
- `tensor_cuda.quantization.quantize_affine_per_group` (group-32 house
  packing convention; used for weights at `core/mistral7b_tc.py:115`).

### Per-graft byte footprint at GPT-OSS geometry

Geometry: 8 kv-heads × head_dim 64 × 2 (K+V) = 1024 vals/token/layer
(`VALS_PER_TOK_LAYER`, `core/graft_arena.py:1857`), 24 layers, fp16
(2 B/val). bytes = ntok × 1024 × 2 × 24 (= `_node_bytes`,
`core/graft_repository.py:2582-2586`, per node across all layers):

| tokens | bytes (all 24 layers) | on disk |
|-------:|----------------------:|--------:|
| 128    | 6,291,456             | 6.3 MB  |
| 512    | 25,165,824            | 25.2 MB |
| 1,024  | 50,331,648            | 50.3 MB |
| 4,096  | 201,326,592           | 201.3 MB (observed: 24 × 8,389,098 B shards = 193 MiB `du`) |
| 16,384 | 805,306,368           | 805 MB  |
| 98,304 (96K) | 4,831,838,208   | ~4.8 GB |

Format 2 shards are uncompressed → disk = raw bytes + ~500 B npz header
per shard (observed 8,389,098 vs 8,388,608 raw). Format 1 nodes are
zlib-compressed (fp16 K/V compresses only marginally). Disk multipliers
from the plan (8/6/4/3/2 bits → 2/2.7/4/5.3/8×) apply to the raw term.

---

## (b) Gate battery inventory

All gates target GPT-OSS-20B (HF snapshot 6cee5e81, 24 layers, 8 kv-heads,
head_dim 64) and mount format-2 graft dirs by shelling into
`gpt_oss20b_stream_greedy_smoke.py` / `gpt_oss20b_stream_forward_smoke.py`
with `--mount-graft-dir`. "Pre-capture OK" = the script accepts
`--graft-dir <existing>` + `--skip-capture` (or never captures at all).
Runtimes: measured receipts (this session at fp16/standard where noted;
otherwise the 2026-07-07 apa_selective artifacts under
`artifacts/gpt_oss_20b/`).

| gate script | measures | pre-capture OK? | runtime | margin metric |
|---|---|---|---|---|
| `gpt_oss20b_multifact_graft_gate.py` | 4 planted one-token facts (vault/relay/archive/tool), forced-final recall, control-vs-mount | YES (`--graft-dir` + `--skip-capture`, mf:97,115) | 4K std/fp16 measured: 152s capture + 289s runs = **~7.4 min**; 16K capture alone 954s (OVER budget) | per-fact `answer_rank`, `answer_logit`, top-10 logits → answer-minus-best-other margin; `hit_count/fact_count` |
| `gpt_oss20b_preference_graft_gate.py` | 4 preference records ("user's preferred X"), same harness (imports multifact) | YES (same flags, pref:74,92) | 4K std/fp16 measured: 152s capture + 291s runs = **~7.4 min** | same as multifact |
| `gpt_oss20b_supersession_graft_gate.py` | old→corrected value pairs; asks CURRENT value (stale-vs-current binding) | YES (same flags, sup:80,98) | 4K std/fp16 measured: 151s capture + 292s runs = **~7.4 min** | same + stale-value check (old_answer in top-k) |
| `gpt_oss20b_exact_value_graft_gate.py` | multi-token exact values (asset code, access number, operator name), 12-step greedy decode per item | YES (`--graft-dir`+`--skip-capture`, ev:128,149) | 4K reference: 222s capture + **1864s runs ≈ 35 min — OVER the 10-min bound** (steps=12 × 3 items × control+mount) | `hit_count` (exact) + `generated_hit_count` (normalized-in-text); per-step rank/logit |
| `gpt_oss20b_multifact_addressing_gate.py` | addressing-policy probes (fact_local / metadata_card / conversational_*) against an EXISTING multifact graft | YES — reuse-ONLY (requires `--source` multifact artifact; never captures) | fact_local 4 probes: 249s (apa ref) / **290s std/fp16 measured ≈ 4.8 min**; conversational 8 probes 487s (ref) | same rank/logit per probe + `pass_mode` exact_top1 or generated_value |
| `gpt_oss20b_bulk_graft_gate.py` | single needle in bulk capture; gold-vs-4-decoys candidate logit scoring (no greedy) | YES (`--graft-dir`+`--skip-capture`, bulk:54,72) | 4K reference: ~220s capture + 2 forwards ≈ **~7 min**; DEFAULT target-tokens=131072 (the 96K/128K monster — hours; excluded per plan) | `gold_minus_best_decoy_logit` — cleanest single-number margin |
| `gpt_oss20b_repetition_drift_gate.py` | instruction-retention record recalled repeatedly (drift over `--repeats`×`--steps` decodes) | YES — reuse-ONLY (`--graft-dir` REQUIRED, rd:48) | not re-measured; multi-step decode ⇒ exact_value-class runtime (over budget at defaults) | per-repeat hit + drift classification |
| `gpt_oss20b_realtext_ppl_gate.py` | teacher-forced PPL windows, standard-vs-APA | N/A — mounts NO grafts (no `--graft-dir`) | small windows: minutes | weighted NLL/PPL |

OUT of the graft-quant battery: `realtext_ppl_gate` (no graft in the
loop — APA-wing instrument, wrong scope); `bulk_graft_gate` at its
131072-token default (the plan's excluded monster; the 96K fp16 anchor
stands from `h5_bulk_graft_96k_candidate_gate_rerun.json`);
`exact_value` and `repetition_drift` at their step counts (over the
10-min bound; margin-per-item is redundant with multifact for the
uniform sweep — bring back for the K/V-asymmetry refinement if the
cliff shows structure).

### Recommended sweep battery (per depth: quantize banks → mount → score)

1. `multifact_graft_gate --skip-capture --graft-dir <quantized>` — 4
   facts, ~290s scoring, margins per fact.
2. `preference_graft_gate --skip-capture --graft-dir <quantized>` — 4
   items incl. one thin-margin item (`preferred_shade` +0.5 at fp16 —
   an early-warning needle for the cliff) and one known-confounded item
   (`preferred_color`, control-hit + case miss at fp16 — treat as
   fixed-fail furniture, compare margins not verdicts).
3. `supersession_graft_gate --skip-capture --graft-dir <quantized>` —
   4 items, adds the stale-vs-current binding axis (does quantization
   resurrect superseded values?).
4. `multifact_addressing_gate --source <multifact artifact> --graft-dir
   <quantized> --variants fact_local` — zero capture, ~250s, the
   cheapest margin readout; ideal for densifying the bit-depth axis.
5. (optional anchor per depth) `bulk_graft_gate --target-tokens 4096
   --skip-capture --graft-dir <quantized>` — single
   gold-minus-best-decoy number.

Battery cost per depth ≈ 20-25 min of bounded minutes-class runs
(scoring only; capture happens once at fp16). Five depths ≈ 2h GPU
spread as separated bounded runs (power law respected).

P1 reuse-path work needed: NONE for the five recommended gates — all
consume pre-captured graft dirs today. The quantize-at-rest transform
itself (pack/unpack + round-trip receipt) is the only new machinery;
it can be a standalone script emitting a dequantized fp16 graft dir
(gates mount it unchanged), or a dequant hook inside
`load_graft_layer` (product-code change, one function).

---

## (c) fp16 baseline re-stamp — CURRENT engine

Engine receipt: gates hardcode
`sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")`
(`scripts/gpt_oss20b_stream_forward_smoke.py:23`) → Project-Tensor
main @ **acc9994** ("docs: program close — 3.2 and Phase 5 negative"),
working tree CLEAN;
`_tensor_cuda.cpython-312-x86_64-linux-gnu.so` mtime
2026-07-07 20:46:48 EDT (built 2 min before the acc9994 commit
timestamp 20:48:22 — post-2026-07-07-kernel-change build, as the plan
requires). Host: MilleniumFalcon, RTX 4070 SUPER 12 GB, driver
595.71.05 / CUDA 13.2. GPU idle before each run (278 MiB desktop
residue only). Runs: `--attention-mode standard` (the unmodified
non-APA path), fp16-at-rest grafts (the existing storage path),
4096-token captures, docs/ corpus (48 md files).

| gate | verdict | hits | per-item margins (answer_logit − best other in top-10) |
|---|---|---|---|
| multifact_4k | **PASS** | 4/4 | vault +1.625 (23.25), relay +4.875 (23.375), archive +4.125 (23.75), tool +4.375 (24.875) — all rank 0 |
| preference_4k | **FAIL 3/4** (known confound) | 3/4 | color MISS rank 1, −0.25 (control already "blue"; case-strict "blue"≠"BLUE"); signal +4.625; shade **+0.5** (thin); metal +2.375 |
| supersession_4k | **PASS** | 4/4 | vault **+0.125** (22.875; stale RED in top-10 — thinnest margin in the battery), relay +4.5 (26.75), archive +1.25 (23.5), tool +2.125 (23.5) — all rank 0, no stale-value wins |
| multifact_addressing fact_local | **PASS** | 4/4 | vault +3.625 (26.25), relay +5.625 (27.0), archive +4.125 (23.375), tool +4.375 (25.375) — all rank 0; strongest margins in the battery (fact-local addressing lifts vault from +1.625 to +3.625) |

Wall receipts: multifact 152.3s capture + 288.8s runs = 441.1s;
preference 151.8s + 290.5s = 442.3s; supersession 150.9s + 291.7s =
442.6s; addressing (no capture) 289.7s. All ≤10-min bound.
`exact_value` NOT run: prior receipt 222s capture + 1864s runs ≈ 35 min
— exceeds the bound; excluded from the sweep battery (flagged above).

Per-gate artifacts (this dir): `multifact_4k_fp16_gate.json` (+ run dir
`multifact_4k_fp16_gate/` incl. the fp16 graft capture),
`preference_4k_fp16_gate.json`, `supersession_4k_fp16_gate.json`,
`multifact_addressing_fact_local_fp16_gate.json`; logs `*_fp16.log`;
margin extractor `extract_margins.py`.

Baseline notes for the curve's y-origin:
- preference/`preferred_color` is a FIXED-FAIL at fp16 (control
  confound + case-strict scoring): in the sweep, compare its MARGIN
  (−0.25 at fp16), not its verdict.
- preference/`preferred_shade` (+0.5) is the thinnest passing margin in
  the battery — expected first casualty as depth drops; watch it.
- vs the 2026-07-07 apa_selective reference runs: multifact 4/4 = same;
  preference 3/4 same failing item = same; supersession IMPROVED
  (apa_selective 3/4 FAIL → standard/fp16 4/4 PASS). The sweep must
  hold attention-mode fixed at `standard` (this baseline), or the
  attention-mode axis confounds the bit-depth axis.
- supersession/`vault_keyword` (+0.125, stale RED at rank 1 in top-10)
  is the single best cliff needle: quantization noise of ~0.13 logits
  flips it, and a flip TO the stale value is a qualitatively legible
  failure (resurrected superseded memory).

CAPTURED fp16 BANKS FOR THE SWEEP (the "capture once" of the plan —
quantize THESE): `multifact_4k_fp16_gate/graft/`,
`preference_4k_fp16_gate/graft/`, `supersession_4k_fp16_gate/graft/`
(each 24 shards, 4096 tokens, ~193 MiB). The addressing gate reuses the
multifact bank. GPU state after the battery: idle (0% util, 279 MiB,
43C). Total GPU-active this session ≈ 27 min in four separated
minutes-class runs (power law respected).

## Risks / ambiguities flagged

1. TWO storage formats (above): the sweep exercises format 2 only.
   A cliff verdict transfers to format 1 byte-wise (same fp16 pre-RoPE
   K/V content) but format-1 mount goes through
   `unpack_node`/`vram_budget` paths the sweep never runs — P3 should
   say so explicitly rather than claim coverage.
2. Strict-scoring confound: preference/`preferred_color` fails at fp16
   for scoring reasons (control-hit + case). Sweep must track MARGINS
   per item, not just `classification` (plan already requires this;
   the baseline confirms why).
3. Attention-mode sensitivity: supersession verdict differs between
   apa_selective (3/4, 2026-07-07) and standard (4/4, this baseline).
   Fix `--attention-mode standard` for the whole sweep.
4. Corpus drift: docs/ is a LIVE directory (43 → 48 files since the
   reference captures). The fp16 banks captured this session are the
   frozen reference; re-captures would embed a different corpus.
5. `bulk_bits` CLI arg on all gates is the APA bulk path parameter —
   inert at `--attention-mode standard`; it is NOT a graft-storage
   bit-depth knob. Do not repurpose the flag name for P1 (collision
   confusion; cf. the GHOST_PRECISION decay_rate name-collision
   precedent).
6. Run-to-run noise not yet measured (single baseline stamp). E1
   ("INT8 within noise") needs a noise band — cheapest source: re-run
   ONE gate once more at fp16 before the sweep, or accept the
   2026-07-07-vs-today margin deltas as the band.
7. The 96K fp16 anchor stands:
   `artifacts/gpt_oss_20b/h5_bulk_graft_96k_candidate_gate_rerun.json`
   — PASS, mount gold−best_decoy = +4.0625 (control −1.80), capture
   5094s. NOT in the sweep loop (plan law).

Reuse caveat (receipts hygiene, not correctness): under
`--skip-capture --graft-dir <old>`, the gate still rewrites
`capture_prompt.txt`/`capture_ids.json` in the NEW run dir from the
CURRENT docs corpus (multifact main, lines 415-426 run
unconditionally). docs/ has drifted since the 2026-07-07 captures
(43 → 48 `*.md`), so those regenerated files and the `corpus.sha256`
receipt describe a prompt the mounted graft was NOT captured from.
The mounted bytes, questions, and answers are unaffected (prompts are
label-based). For sweep hygiene, carry the ORIGINAL capture receipt
alongside each quantized graft dir.
