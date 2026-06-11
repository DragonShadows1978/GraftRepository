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
| Cross-session resume | fresh process, 6/7 from disk artifacts alone (24.9MB) |
| Ephemeral boat ("infinite context") | 42-turn history at ≤456 resident seats, flat |
| Decode speed (fast stack) | 675 → **21.6 ms/token** (31×), parity-gated |

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

- Era-depth consolidation is DEFAULT-OFF but VIABLE with DESCENT (built):
  era texts are index nodes, never readers (lists strip relations, prose
  invents them — measured); the trips ladder expands eras to their child
  digests at the primary attempt, descends digests on grounding failure,
  reloads cold-storage children, and budget-fits every mount set to the
  arena. Era-folded 42-turn recall: 3/8 without descent → 6/8 with. The
  residual gap is FIRST-GEN digest quality (bare-bullet lists bind
  relations weakly).
- Topical routing between sibling digests (no-identifier probes) is the
  one routing soft spot.
- Librarian runs in-process; background-mission offload planned.
- VRAM paging for active nodes; GQA (Qwen3) arena port.
