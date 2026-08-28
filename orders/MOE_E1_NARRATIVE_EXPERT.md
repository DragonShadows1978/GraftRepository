# ORDER MOE-E1 — first bolt-on expert: NarrativeForge guides, asymmetric residual addon on frozen GPT-OSS-20B

## Grant (read this first)

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/GraftRepository` — edits, builds,
CPU and GPU runs AUTHORIZED. A registered order IS the permission.

READ-ONLY paths (read/import in place, never write):
- `/mnt/ForgeRealm/Project-Tensor` (tensor_cuda)
- `/home/vader/.cache/huggingface/hub/models--openai--gpt-oss-20b/...`
- `/home/vader/.cache/huggingface/hub/datasets--wikitext`
- `/mnt/Shared/01 - Narrative Project/01 - Narrative Research/Methodology/GUIDELINES/Active_Guides`
- `/mnt/Shared/01 - Narrative Project/01 - Narrative Research/Methodology/GUIDELINES/Active_J_Guides`
- `/mnt/Shared/01 - Narrative Project/01 - Narrative Research/Methodology/GUIDELINES/Active_O_Guides`
  (the expert corpus: 35 files, ~846 KB. These three folders ONLY —
  nothing else under /mnt/Shared, and NOTHING from any folder with
  "Consciousness", "RCFT", or "Thesis" in its path.)
- `artifacts/moe_rt2/` and `artifacts/moe_rt2_1/` (negative-pool
  captures + key machinery receipts; never overwrite)

FORBIDDEN: git (lead commits), subagents, network, modifying the
committed RT1/RT2/RT2.1 scripts (import their helpers freely). New code
in `scripts/gpt_oss20b_expert_e1.py` (siblings allowed with the same
prefix); new artifacts in `artifacts/moe_e1/`.

EXPECTED SANDBOX LIMIT: your worker likely has no `/dev/nvidia*`. Build
everything, validate all CPU-testable machinery on synthetic data, and
emit the blocked report + exact GPU resume commands (the proven RT1/RT2
pattern). CPU-only stages that depend on GPU captures obviously cannot
run; structure the script in modes so the lead can execute the whole
chain: `prepare` / `capture-keys` / `fit-key` / `capture-pairs` /
`train` / `eval-gates` / `analyze`.

## Context (one paragraph)

Receipts so far: RT1 — the frozen GPT-OSS-20B router geometry separates
domains (24/24 layers). RT2/RT2.1 — a closed-form shrinkage-Fisher-LDA
key over router-input hidden states addresses a domain with high
precision (best: L5, recall 0.853 @ generic FPR 0.000, code FPR 0.034,
frozen fit-side τ). E1 now builds the first REAL expert: a small
asymmetric low-rank residual module carrying David's NarrativeForge
writing-guide domain, attached to the frozen MoE via the registered ABI
(threshold-gated residual side-path; zero installs = bit-identical
base). Training is CONSOLIDATION, not fine-tuning: the teacher is the
frozen base WITH guide text in context; the student is the frozen base
alone; the adapter learns offline, from captured activation pairs, to
reproduce the teacher's hidden-state shift on domain text. The 20B only
ever runs inference. This is the first expert-viability receipt; gates
are behavioral.

## ABI (registered)

At install layer L\*: let h be the router-input hidden state (identical
tensor the native router reads). Expert: score s = h·k (k = Fisher-LDA
key); if s ≥ τ, add `B·silu(A·h)` to the layer's residual-stream output
(additive alongside the native MoE output; native routing untouched).
A: 2880→r, B: r→2880, fp16 storage. B is zero-initialized (a fresh
expert is a no-op). r = 64 primary; r ∈ {32, 128} may be tried but
selected on validation pairs only (fit-side). Tokens with s < τ MUST be
bit-identical to base — by construction; G0 verifies.

## Corpus + splits (registered; violation = G2/G4 RED)

- Deterministic file-level split of the 35 guide files by sha256 of
  filename: ~70% TRAIN files, ~30% HELDOUT files. Record the lists.
- Teacher prefixes: 2,048-token excerpts drawn ONLY from TRAIN files
  (deterministic seeded rotation per window; record which).
- Pair windows (512 tokens) for training: TRAIN files only.
  ≥48 train windows, ≥16 validation windows (disjoint).
- Behavioral eval windows: HELDOUT files only, ≥16 windows. HELDOUT
  text must never appear in any prefix, pair, or key-fit input.
- Key-fit windows: TRAIN files (even/odd FIT/EVAL discipline as RT2).
- Negative pools for the key: the EXISTING RT2 captures — generic,
  code, AND the GRM-domain captures (three negative corpora now).

## Stages

1. `capture-keys` (GPU): router-input h for narrative windows, all 24
   layers, RT2 chunking (≤590 s runs).
2. `fit-key` (CPU): K4 shrinkage Fisher LDA, narrative vs pooled
   {generic, code, grm} FIT negatives, frozen fit-side τ per the RT2.1
   policy. Install layer L\* = best qualifying layer.
3. `capture-pairs` (GPU): for each pair window, two forwards — teacher
   (prefix + window) and student (window alone) — capturing h and the
   layer-L\* block OUTPUT for the shared window tokens. G1 verifies
   token-id alignment. (Capture at the top-3 qualifying layers if L\*
   selection wants comparison; fit-side choice only.)
4. `train` (CPU or brief GPU): adapter learns Δ = out_teacher −
   out_student at L\* from h_student, MSE loss, early stopping on
   validation pairs. Tiny model — minutes, no sustained load.
5. `eval-gates` (GPU, bounded): behavioral ppl runs (below) with the
   expert wired into the real forward path.
6. `analyze` (CPU): tables, CIs (bootstrap over windows), report.

## Gates (registered; a failed gate is a RESULT — RED with receipts)

- G0 ABI identity: expert mounted, gate live — on tokens where the gate
  does not fire, final hidden states BIT-IDENTICAL to base. And with
  zero experts installed: bit-identical, byte-for-byte.
- G1 pair truth: teacher/student shared-window token ids identical;
  hidden states differ; alignment receipts for ≥2 windows.
- G2 address gate: the narrative K4 key must qualify at ≥1 layer under
  the RT2.1 rule extended with the third negative: EVAL recall ≥ 0.50,
  generic FPR ≤ 0.02, code FPR ≤ 0.05, grm FPR ≤ 0.05, frozen τ. No
  qualifying layer → RED and STOP after reporting (an expert without an
  address is not installable; that is a finding).
- G3 adapter signal: validation-pair MSE beats the zero-predictor
  (predicting Δ=0) by ≥ 10%. Below → RED (consolidation signal too weak
  at this scale — report, do not tune past it).
- G4 BEHAVIORAL PRIMARY: on HELDOUT windows compute ppl_base (no
  prefix, no expert), ppl_teacher (TRAIN-file prefix in context), and
  ppl_expert (expert mounted, gate live, no prefix). Precondition: the
  teacher gap (ppl_base − ppl_teacher) must be positive — if guides in
  context do not help ppl on held-out guide text, report that as the
  E1-premise finding and stop. Registered rule: E1 SUPPORTED iff
  ppl_expert recovers ≥ 25% of the teacher gap. Report recovery % with
  bootstrap 95% CI over windows.
- G5 non-interference: with expert mounted and gate live — wikitext ppl
  delta vs base ≤ 0.5%; realized fire rate ≤ 2% on generic, ≤ 5% on
  code. Receipts: the deltas and rates.

## Run constraints (standing house laws — mandatory)

Every GPU invocation: `flock -w 7200 /tmp/forge-gpu.lock`, ≤10 min
wall-clock, idle gaps between invocations, single GPU, no sustained
load (chunk everything; checkpoint between chunks). Adapter training
runs on CPU unless a single ≤10-min GPU run suffices. RED honesty; no
monitor-idling. Evidence class: behavioral inference measurement on one
model + one domain — no generality claims beyond it.

## Done

Final message MUST contain verbatim:
1. G0–G5 verdicts, one line each, GREEN/RED/NOT_MEASURED with key
   numbers (or the blocked report + full resume command sequence).
2. The E1 registered verdict sentence (or premise-finding sentence).
3. The install row: layer L\*, r, key metrics (recall/FPRs), τ.
4. The G4 row: ppl_base, ppl_teacher, ppl_expert, recovery % + CI.
5. ExpertPack manifest path (key, τ, A, B, L\*, r, provenance) —
   `artifacts/moe_e1/expertpack_narrative_v0/`.
6. Exact paths of every file created or modified; anything you could
   not do, stated plainly.
