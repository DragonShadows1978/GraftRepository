# ORDER NC17-P1 — Qwen3-1.7B tensor_cuda adapter + APA r0.15 battery (bf16)

YOUR WRITABLE TARGET is /mnt/ForgeRealm/GraftRepository — the adapter
work in `core/` and `adapters/` as described below, NEW test/harness
files under `tests/nc17/`, outputs under `logs/nc17/`, plus /tmp.
Everything else read-only. The engine at
/mnt/ForgeRealm/Project-Tensor/tensor_cuda is READ-ONLY — if the
adapter seems to need an engine change, STOP and report; engine work is
another seat's lane.

Read `docs/QWEN3_1P7B_NAMECHECKER_PLAN.md` (immutable) first; its
Measurement protocol governs. Prerequisite artifacts from P0 (must
exist before your gate/battery steps): `logs/nc17/p0_gt.npz`,
`logs/nc17/p0_ppl_tokens.npz`, sentinel `logs/nc17/P0_GT_READY`. Build
code freely before they exist; do not fabricate substitutes for them.

## Task — Phase P1

1. **Adapter**: Qwen3-1.7B support via the existing Qwen3 adapter
   family (`core/qwen3_tc.py`, built for Qwen3-4B — read it fully
   first). Expected to be config-driven: 28L, 16Q/8KV, head_dim 128,
   hidden 2048, inter 6144, vocab 151936, RoPE θ1e6, per-head qk-norm,
   NO sliding. **Known delta: tied embeddings** (`tie_word_embeddings:
   true` — the 4B is untied). Handle the tied head explicitly; record
   the resident-vs-host decision and its VRAM cost in the receipt.
   Weights: bf16 from the HF cache snapshot P0 downloaded.
2. **Parity gate** vs P0 GT (`tests/nc17/p1_parity.py`): margin
   protocol — top-1 agreement on the GT prompt set; flips acceptable
   ONLY at near-tie margins (report every flip with its GT margin);
   max|Δlogit| distribution reported. Name the exact GT file+hash used
   (matched-reference law).
3. **APA battery** at refine r0.15, bf16 weights:
   - APA-off control and APA-on runs at the same ladder rungs.
   - OOM/context ladder (one probe per subprocess, prefill + decode
     ceilings, binary search) — both modes.
   - Perplexity per the plan protocol on `p0_ppl_tokens.npz`
     tokenization, windows 2048(control)/4096/8192/16384/32768 as they
     fit — both modes. **APA engagement must be asserted per run**
     (log the engaged fraction / threshold stats; a run where APA
     never engaged is not an APA datapoint — the June law).
   - Peak fill per run: 1s nvidia-smi poller + engine allocator peak.
4. **Deliverables**: printed tables + `logs/nc17/p1_summary.json` —
   parity margins, ceilings (APA on/off), ppl×window (APA on/off vs
   P0 baseline), peak×ctx per the plan's memory table.

## Rails

- GPU: FOREGROUND blocking only, `flock -w 3600 /tmp/forge-gpu.lock`,
  `timeout 590` per run; setsid for any child inside a flocked shell;
  NEVER detach. Other seats share the flock — queueing is normal.
- NO git. NO subagents. No engine edits.
- OOM = measurement, not failure. RED honesty on real failures,
  verbatim errors. Evidence classes per plan.
- If a P0 artifact is missing when you reach the gate step, report
  BLOCKED with what exists on disk — do not improvise a GT.

## Done — final message must contain, verbatim

- Adapter file paths + the tied-head decision and its measured cost.
- Parity table: top-1 agreement, every flip with GT margin, max|Δ|.
- Ceiling table APA-on/off; ppl table APA-on/off/P0; peaks per rung.
- APA engagement stats per scored run.
- Exact created-file paths; deviations stated as deviations.
