# Graft Repository

**Routed, hierarchical, tokenless memory for frozen LLMs — effectively
unbounded conversation context at constant VRAM residency, zero training,
on consumer hardware.**

Documents and conversation turns are harvested ONCE as the model's own
attention K/V ("grafts"), stored off-context, routed per turn with the
model's own representations, and mounted into a fixed positional arena by
cache surgery. History never re-pays tokenization or prefill; the live
context stays permanently small; nothing is silently dropped.

Built and measured 2026-06-10 on MiniCPM3-4B (MLA latent attention, INT4)
over the [tensor_cuda](https://github.com/DragonShadows1978/Project-Tensor)
engine, RTX 3070 8GB.

## The receipts (every claim gated; see docs/)

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
| Cross-session resume | fresh process, **7/7** from disk artifacts alone (26.4MB) — fidelity-gated digests cleared the offsite residual |
| Ephemeral boat ("infinite context") | 42-turn history at ≤456 resident seats, flat, 8/8 recall incl. era-folded facts + anaphora |
| Decode speed (fast stack) | 675 → **21.6 ms/token** (31×), parity-gated |
| Deferred librarian | 42 turns: hot path **0.27s max, flat** (inline spikes 3-9s); folds drain in idle(); recall 8/8 unchanged |
| Fidelity-gated folding | a fold keeping <70% of source FACTS aborts; sources stay resident (no_fold, persisted) — recall > compression |
| GQA arena (Qwen3-4B) | round-1 port 6/6 = MLA parity; full-key re-RoPE surgery, layer-0 \|q.k\| router, bounded residency |
| GQA port unified | dialect-surface ArenaCache: MLA suite preserved bit-identical; GQA arena 6/6, trips 6/6, E4-C recall 6/6; early-stop decoding (reasoning-leak style attractor killed, 0/6 -> 6/6) |

## Layout

- `core/kv_graft.py` — harvest / inject / routing primitives (GQA + MLA)
- `core/graft_arena.py` — ArenaCache: persistent arena, 3-channel routing
  (latent centroid + lexical identifier keys + hierarchical descent),
  grounded trips with clean-room retry and rollback, consolidate()
- `core/graft_repository.py` — GraftRepository: chat / add_turn /
  add_document API, auto-librarian, disk persistence, dialect wall,
  cross-session resume, ephemeral mode
- `core/{minicpm3,mistral7b,qwen,qwen3}_tc.py` — model adapters (graft
  hooks, absorbed MLA decode, fast-stack flags)
- `tests/` — every gate above, self-contained harnesses
- `docs/` — the design document (foundations table = the receipts),
  KV-Graft writeup, MiniCPM3 engine results

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

## Dependencies

- the tensor_cuda engine (Project-Tensor repo) built for your GPU arch
- `numpy`, `tokenizers`
- MiniCPM3-4B weights in the HuggingFace cache

## Known open items

- Era folding is DEFAULT-ON and fidelity-gated like every fold: an era
  whose chronicle can't cover its children's facts ABORTS and the digests
  stay directly routable. Exempting eras ("index nodes, never read") was
  tested and refuted — folding retires the children's routing surfaces
  and era expansion is budget-bound.
- Topical routing between sibling digests (no-identifier probes) was the
  one routing soft spot; the resume-gate instance cleared with
  fidelity-gated digests. fit() truncation stays expansion-ordered (naive
  score order favors digests over verbatim turns — refuted).
- Librarian: deferred mode keeps the hot path flat (folds drain between
  turns); true background-mission offload planned.
- GQA (Qwen3) port: UNIFIED — ArenaCache is a dialect surface (base class
  = MLA; GQAArenaCache overrides payload/router/RoPE/injection/persistence).
  Gated green: full MLA suite preserved bit-identical; GQA arena 6/6,
  starved-arena trips 6/6 (hybrid harvest-on-generate deposits work),
  E4-C recall 6/6 with BOTH folds fidelity-aborted (Qwen3 first-gen digest
  generation needs prompt tuning — the gate holds the line). Two refuted
  on the way: unit-normalized layer-0 routing (rankings collapse
  probe-independent; norm info is load-bearing) and full-ngen decoding on
  a reasoning-tuned model (post-answer leak in the live cache became a
  style attractor, trips 6/6 -> 0/6; EARLY-STOP decoding fixed it and cut
  probe latency 2-3x). Not yet re-gated after the early-stop fix: GQA
  repository resume (6/7 pre-fix), GQA descent 42-turn (5/8 pre-fix);
  cross-model migrate tool + gate written, NEVER RUN.
  (VRAM paging: DONE — LRU write-back pager, 100 docs at 64MB, 20/20.)
