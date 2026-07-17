# Graft Repository

**Routed, hierarchical, tokenless memory for frozen LLMs — effectively
unbounded conversation context at constant VRAM residency, zero training,
on consumer hardware.**

Documents and conversation turns are harvested ONCE as the model's own
attention K/V ("grafts"), stored off-context, routed per turn with the
model's own representations, and mounted into a fixed positional arena by
cache surgery. History never re-pays tokenization or prefill; the live
context stays permanently small; nothing is silently dropped.

Built and measured starting 2026-06-10 on MiniCPM3-4B (MLA latent
attention, INT4) over the
[tensor_cuda](https://github.com/DragonShadows1978/Project-Tensor) engine,
RTX 3070 8GB / RTX 4070 SUPER 12GB. Every claim below names its receipt;
plans and ledgers live in `docs/`, sealed gate receipts in `artifacts/`
(receipt-class files committed; bulk payloads stay local).

## WHAT WORKS (every row gated; docs/ + artifacts/ hold the receipts)

| Result | Number |
|---|---|
| Graft ≡ in-context (lossless mounting) | logit-level: top-1 identical, diff = bf16 noise floor |
| MLA latent graft size | 288 values/token/layer (~22× smaller than full K/V) |
| E1 router recall | routed top-3 **beats mount-all** (10/10 vs 4/10 Qwen3; latent-centroid router on MiniCPM3) |
| E2 digest fidelity, chained | step decay: ~0.89 once, then digests are **fixed points** |
| E4 conversation memory | 6/6 = full-transcript baseline at ~25% residency; amnesia control 0/6 |
| Persistent arena | 6/6 on one never-rebuilt cache; swap/evict = cache surgery |
| Consolidation (E4-C) | 6/6 through QC'd digest grafts; routing pool 14→8 |
| Shuttling | 1-mount arena + grounded trips = 3-mount arena (6/6) |
| CORPUS-100 | 20/20 against 100 near-duplicate docs; 50KB index; 1.3s/probe |
| Cross-session resume | fresh process, **7/7** from disk artifacts alone (26.4MB) |
| Ephemeral boat ("infinite context") | 42-turn history at ≤456 resident seats, flat, 8/8 recall incl. era-folded facts + anaphora |
| Decode speed (fast stack) | 675 → **21.6 ms/token** (31×), parity-gated |
| Deferred librarian | 42 turns: hot path **0.27s max, flat**; folds drain in idle(); recall 8/8 unchanged |
| Fidelity-gated folding | a fold keeping <70% of source FACTS aborts; sources stay resident — recall > compression |
| GQA arena (Qwen3-4B), unified dialect surface | MLA suite bit-identical; GQA arena 6/6, trips 6/6, E4-C 6/6 |
| VRAM paging | LRU write-back pager: 100 docs at 64MB budget, 20/20, +0.1s/probe |
| **Composed E2E receipt (GPT-OSS-20B)** | 34-turn live session through production chat()→step(): witnessed deposit→evict→route→mount→recall **7/7 exact across a process restart**; VRAM flat 10.8→11.0GB (docs/GRM_E2E_RECEIPT_LEDGER.md) |
| **CUDA route (MLA)** | 1M-node route **2.22 ms** (from 925.6 ms — 417×); ≤100k byte-exact vs python (docs/GRM_MLA_CUDA_ROUTE_*) |
| **GQA CUDA bridge** | route entry 0.19–0.97 ms = 1.26–1.44× direct (was 25–50×) |
| **Exact ragged GQA CUDA router** | 175/175 four-way parity (numpy/CPU/CUDA/bridge), 512 nodes p50 **1.59 ms**; disabled by default (docs/GRM_GQA_EXACT_RAGGED_CUDA_*) |
| **Graft storage quantization** | INT8 free (1.88× disk), INT6 last green (2.46×); packed on-disk format landed (docs/GRM_GRAFT_QUANT_LEDGER.md) |
| **Trinity NoPE grafts** | NoPE dissolves the arena position-hole law: recall at live_shift 789 = 2× GPT-OSS's salad depth, controls prove carriage (fdc478c) |
| **Supersession fix (L2 revision-aware mount resolution)** | stale readback 2/5 → **0/5**; multi-hop mounts lineage head only; 100% of effect attributed to L2 by resolve-only diagnostic; **flag, default OFF** (docs/GRM_SUPERSESSION_LEDGER.md) |
| **S4 grounding-hit importance signal** | vs teacher-forced counterfactual arbiter: median Spearman **0.756**, top-1 **87.5%** (bars 0.5/0.5) at ZERO extra forward passes (docs/GRM_S4_LEDGER.md) |
| **M11 fold-after-recovery guard** | crash-recovered sessions no longer brick the librarian; 11 regressions crash pre-fix (docs/GRM_BUG_QUEUE.md) |

## WHAT DOESN'T WORK (stated plainly; same receipt discipline)

- **S4-aware paging loses to LRU** (G2-S4, RED): recall tied 14/16
  but +65% page-ins (112 vs 68) at 4.6× overcommit — early-session
  zero-hit nodes include the just-deposited ones routing wants next;
  zero-hit-first spilling breaks recency locality. The S4 SIGNAL
  stands (G1 green above); as a paging POLICY it doesn't pay yet.
  `spill_policy="s4"` remains in-tree, flagged, default LRU.
  docs/GRM_S4_LEDGER.md.
- **Attention mass does not measure memory importance** (S1, RED):
  median Spearman 0.473 vs the counterfactual arbiter, bar 0.5 — a
  15-probe partial pass FLIPPED when the full 18-probe gate ran.
  Independently reproduces Jain & Wallace 2019 ("Attention is not
  Explanation") at KV-memory granularity. docs/GRM_IMPORTANCE_LEDGER.md.
- **4B self-report salience is dead** (S2, RED): rankable on 7/18
  probes; scores standing preferences 0.0 against a 2.0 bar under the
  frozen fact-worded rubric. The Generative-Agents-style importance
  channel, validated causally, fails at this scale.
- **Co-mount confusion** (open, pre-existing): routed right, mounted
  right, answered from the co-mounted sibling — corpus-100 class,
  fresh receipt in the supersession battery's fresh-fact control (1/2).
- **L1 route length-debias is undecidable on MLA** (open): the
  GQA-characterized max-pool length bias has NO MLA analogue (0
  inversions at baseline; battery identical with debias on). Its
  decisive test is the GQA dialect — registered successor.
- **Route latency at session scale** (open): ~756 ms at 37 nodes on the
  GPT-OSS python path (per-turn arena re-prep + lex rescore + ragged
  CUDA bank non-engagement). Registered seams 2/3/4 in
  docs/GRM_E2E_RECEIPT_LEDGER.md.
- **Lifecycle test suite RED since 2026-07-08** (M10, open): FakeArena
  test double lacks `_bump_cuda_gqa_epoch`; 91/101 fail — the bug
  queue's rule-2 gate was silently dead for a week.
- **`_ensure_h` silent fall-through** (open): an unbacked payload
  downgrades instead of raising a named error; the M11 guard keeps the
  fold path clear of it, other callers can still index None.
- **GQA re-gates pending** (open, paused): repository resume (6/7
  pre-early-stop-fix), descent 42-turn (5/8); cross-model migrate gate
  written, NEVER RUN. Qwen3 first-gen digests fail the 0.70 fidelity
  bar (gate holds; prompt tuning needed).
- **Era-depth at 4B** (refuted, default respects it): multi-digest eras
  strip or invent relations; era folding is fidelity-gated and eras
  route but are never read; descent expands them at the primary attempt.

## Measurement laws the gates enforce (learned the hard way)

- **First-run effect**: the first forward of a process differs ≤0.5
  logit from all subsequent runs (warm runs bit-identical) — every
  same-process A/B warms up before capturing side A.
- **Seating-epoch invariant**: per-seat attention telemetry is valid
  only within a stable seating epoch; cache surgery discards the
  accumulator (under-attribute, never misattribute).
- **Teacher-force any cache-equivalence comparison**; generation-based
  A/B is garbage past the first greedy divergence.
- **K/V-irreducibility** (four independent instances: SCRIBE, sub-floor
  storage quant, route cards, BABEL): compressed proxies of
  contextualized K/V don't degrade — they vanish. Keep exact payloads;
  engineer the scale.

## Layout

- `core/kv_graft.py` — harvest / inject / routing primitives (GQA + MLA)
- `core/graft_arena.py` — ArenaCache: persistent arena, 3-channel routing
  (latent centroid + lexical identifier keys + hierarchical descent),
  grounded trips with clean-room retry and rollback, consolidate(),
  S1 telemetry tap, supersession-aware mount resolution (flagged)
- `core/graft_repository.py` — GraftRepository: chat / add_turn /
  add_document API, auto-librarian (deferred mode, fold-recovery guard),
  disk persistence, dialect wall, cross-session resume, ephemeral mode,
  S2 salience pass (flagged), S4 grounding-hit ledger, spill policies
- `core/{minicpm3,mistral7b,qwen,qwen3,qwen35,gpt_oss20b,deepseek_v2_lite,gemma4}_tc.py`
  — model adapters
- `core/grm_cuda_router.py`, `core/grm_native.py`, `core/grm_runtime.py`,
  `cpp/` — CUDA/native route + runtime
- `tests/` — every gate above, self-contained harnesses
- `docs/` — plans (immutable) + ledgers (receipts) per program; design doc
- `artifacts/` — sealed gate receipts (≤1MB receipt-class committed;
  payloads machine-local)
- `orders/` — implementation work orders as dispatched (committed before
  dispatch)

## Quickstart

```python
import sys
sys.path.insert(0, "/path/to/Project-Tensor/tensor_cuda")  # engine
sys.path.insert(0, "/path/to/GraftRepository")

import tensor_cuda as tc
from core.minicpm3_tc import MiniCPM3_TC, _snap
from core.mistral7b_tc import QuantLinearTC, RMSNormTC
from core.graft_repository import GraftRepository
from tokenizers import Tokenizer

# fast stack (all parity-gated, default-off)
QuantLinearTC.FUSED_DECODE = True
RMSNormTC.USE_FUSED = True
tok = Tokenizer.from_file(f"{_snap()}/tokenizer.json")
model, _ = MiniCPM3_TC.from_pretrained()
tc.set_alloc_pooling(True)
for L in model.layers:
    L.self_attn.absorbed_decode = True

repo = GraftRepository(model,
                       encode=lambda t: tok.encode(t).ids,
                       decode=lambda i: tok.decode(i),
                       path="~/graft-repo",
                       ephemeral=True)          # constant residency
repo.add_document("RUNBOOK. The ingest replacement listens on port 7443.")
answer, info = repo.chat("What port does the ingest replacement use?")
```

All new capabilities are opt-in flags with byte-identical default paths:
`set_telemetry()` (S1 tap), `s2_salience_enabled`, supersession
`--resolve` semantics, `spill_policy="s4"`, ragged CUDA router.

## Dependencies

- the tensor_cuda engine (Project-Tensor repo) built for your GPU arch
- `numpy`, `tokenizers`
- MiniCPM3-4B weights in the HuggingFace cache (per-model adapters have
  their own weight expectations)

## License

Copyright (C) 2026 David Perry.

This repository is licensed under the GNU Affero General Public License
v3.0 — see [LICENSE](LICENSE). Any software derived from this code,
including software served over a network, must be released under the same
terms. **Commercial licensing outside the AGPL terms is available** —
contact `dave@ai-storyforge.com`.

The associated research papers are licensed CC BY 4.0 via their Zenodo
records.
