# Graft Repository

> **GRM gives a frozen local model persistent, routed memory without replaying
> the full conversation into its context window.** It captures the model's own
> attention state once, stores that state as a graft, routes relevant grafts for
> each request, and mounts them into a bounded live cache. The model weights are
> unchanged, old history does not repay tokenization or prefill, and the active
> VRAM budget remains bounded by the configured arena.

Graft Repository is the research runtime for **Graft Repository Memory (GRM)**.
It stores model-native attention state outside the live context and restores
only the memories selected for the current request. The current implementation
runs on the [Project-Tensor](https://github.com/DragonShadows1978/Project-Tensor)
CUDA engine and uses model-specific cache dialects.

**The named models below are validation points, not a compatibility
whitelist.** TensorCUDA execution, APA evaluation, and GRM end-to-end
certification are separate claims.

## How it works

1. Keep the model frozen; GRM does not train or modify its weights.
2. Run a document or prior turn once and capture its native attention state.
3. Store that state and provenance as a persistent graft outside live context.
4. Score the current prompt against the repository and route relevant grafts.
5. Mount the selected grafts into fixed seats in a bounded live arena.
6. Generate from the frozen model, then deposit the new turn for later routing.

```text
document / prior turn
        │ harvest once
        ▼
model-native K/V graft ──► persistent repository
                                  │ route relevant memories
                                  ▼
current prompt ──────────► bounded live arena ──► frozen model response
```

The user-visible result is persistent recall without replaying old history as
prompt text. GRM can provide **effectively unbounded conversation history only
in the operational sense that active residency stays bounded; bounded active
residency does not mean zero host/disk growth, universal recall, or unlimited
addressable storage.** Retrieval quality, trained context geometry, repository
capacity, and storage policy remain real limits.

## Representative receipts

| Receipt | Result | Source |
|---|---|---|
| Lossless mounting | MiniCPM3 graft vs in-context: top-1 identical; max logit difference 0.41 | [Methodology](docs/GRM_Methodology.md#8-experiment-ledger) |
| Bounded active residency | 42 turns, 8/8 recall, at most 456 resident seats | [Methodology](docs/GRM_Methodology.md#8-experiment-ledger) |
| Persistence and restart | Fresh-process resume reached 7/7 from disk artifacts | [Architecture receipts](docs/GraftRepository_Memory_Architecture.md) |
| MLA lifecycle result | MiniCPM3 conversation recall 6/6 at about 25% residency; amnesia 0/6 | [Methodology](docs/GRM_Methodology.md#8-experiment-ledger) |
| GQA result | Qwen3 arena, starved trips, and consolidation each reached 6/6; later repository re-gates were not closed in that receipt | [Architecture receipts](docs/GraftRepository_Memory_Architecture.md) |
| GPT-OSS composed E2E | Deposit → evict → route → mount → recall, 7/7 exact across process restart | [E2E ledger](docs/GRM_E2E_RECEIPT_LEDGER.md) |
| Honest negative | S4-aware paging tied LRU recall at 14/16 but required 112 vs 68 page-ins | [S4 ledger](docs/GRM_S4_LEDGER.md) |

The [complete results index](docs/RESULTS_INDEX.md) preserves every result,
failure, correction, and open receipt formerly listed here.

## GRM certification matrix

These labels follow the repository-wide public vocabulary. In particular,
**GRM adapter** means dialect code exists; it does not imply the lifecycle gates
required for **GRM certified**.

| Label | Meaning |
|---|---|
| **Engine port** | Model or pipeline loads and executes through TensorCUDA. |
| **Parity-gated** | Engine output was compared against a registered reference under a stated tolerance. |
| **APA evaluated** | APA was actually engaged and measured on the named target. |
| **APA positive** | The registered quality/cost gate passed at a stated operating point. |
| **APA boundary** | The experiment is informative but exposes a cost, quality, or geometry limit. |
| **APA negative** | The registered gate failed or the mode was abandoned. |
| **GRM adapter** | Model-specific capture/restore dialect code exists. |
| **GRM certified** | Deposit, route, mount, recall, persistence/restart, and relevant controls passed. |
| **External receipt** | Result was collaborator-reported and is not a locally reproduced gate. |
| **Planned / unconfirmed** | Code or a plan exists, but the required evaluation has not closed. |

| Model / dialect | GRM status | Receipt and boundary |
|---|---|---|
| MiniCPM3-4B / MLA | **GRM certified** | Full foundational lifecycle, disk resume, controls, and regressions in [architecture receipts](docs/GraftRepository_Memory_Architecture.md) and [methodology](docs/GRM_Methodology.md) |
| Qwen3-4B / GQA | **GRM adapter** | Arena, route, mount, trips, and consolidation passed; the cited receipt leaves repository resume and descent re-gates open in [architecture receipts](docs/GraftRepository_Memory_Architecture.md) |
| Qwen3-1.7B / GQA | **GRM adapter** | Mount equivalence, state save/restore, E4 recall, and amnesia control passed; no process-restart lifecycle receipt in [name-checker ledger](docs/QWEN3_1P7B_NAMECHECKER_LEDGER.md) |
| Qwen3.5-9B / GQA + DeltaNet | **GRM adapter** | Prefix state restore passed; multi-graft arena composition remains open in [Qwen3.5 report](docs/QWEN35_APA_GRM_REPORT.md) |
| GPT-OSS-20B / GQA + sliding | **GRM adapter** | Production `chat()` → `step()` fresh-fact lifecycle reached 7/7 across restart, but supersession-under-competition and route-wall controls were RED in the [E2E ledger](docs/GRM_E2E_RECEIPT_LEDGER.md) |
| Gemma-4 12B / MQA + sliding | **GRM adapter** | Prefix mount/state evidence exists, but no full lifecycle receipt in [port ledger](docs/GEMMA4_PORT_LEDGER.md). APA is **APA negative** under the operative [MQA adjudication](https://github.com/DragonShadows1978/Project-Tensor/blob/main/docs/GEMMA4_MQA_ADJUDICATION.md): one shared KV head and at-most-1024-key sliding windows fail the statistics and economics axes. |
| DeepSeek-V2-Lite / MLA | **Planned / unconfirmed** | The [latent INT4 report](docs/DeepSeek-V2-Lite_APA_Latent_INT4.md) does not close the GRM lifecycle |
| Mistral-7B / GQA | **Planned / unconfirmed** | No in-repository documentation receipt closes adapter or lifecycle status |

## MiniCPM3 developer example

This example exercises the current native integration; it is not a
backend-neutral facade.

```python
import sys
sys.path.insert(0, "/path/to/Project-Tensor/tensor_cuda")
sys.path.insert(0, "/path/to/GraftRepository")

import tensor_cuda as tc
from core.minicpm3_tc import MiniCPM3_TC, _snap
from core.mistral7b_tc import QuantLinearTC, RMSNormTC
from core.graft_repository import GraftRepository
from tokenizers import Tokenizer

QuantLinearTC.FUSED_DECODE = True
RMSNormTC.USE_FUSED = True
tokenizer = Tokenizer.from_file(f"{_snap()}/tokenizer.json")
model, _ = MiniCPM3_TC.from_pretrained()
tc.set_alloc_pooling(True)
for layer in model.layers:
    layer.self_attn.absorbed_decode = True

memory = GraftRepository(
    model,
    encode=lambda text: tokenizer.encode(text).ids,
    decode=lambda ids: tokenizer.decode(ids),
    path="~/graft-repo",
    ephemeral=True,
)
memory.add_document("RUNBOOK. The ingest replacement listens on port 7443.")
answer, info = memory.chat("What port does the ingest replacement use?")
```

Other in-tree integrations include
[Qwen3](core/qwen3_tc.py), [Qwen3.5](core/qwen35_tc.py),
[GPT-OSS-20B](core/gpt_oss20b_tc.py), [Gemma-4](core/gemma4_tc.py), and
[DeepSeek-V2-Lite](core/deepseek_v2_lite_tc.py). Their certification status is
the matrix above, not the existence of these files.

## Current limitations

- GRM depends on the sibling Project-Tensor runtime, normally checked out at
  `/mnt/ForgeRealm/Project-Tensor`; build `tensor_cuda` for the target NVIDIA
  GPU before running this example.
- It is **not** a drop-in memory layer for arbitrary Hugging Face,
  llama.cpp, MLX, or vLLM models. Cache capture, routing geometry, position
  handling, and restore semantics are dialect-specific.
- Active VRAM can be bounded while repository payloads and metadata continue
  growing in host memory or on disk. Operators must set retention, paging, and
  durability policy.
- Recall is model- and routing-dependent. Fidelity gates can refuse a fold,
  co-mounted memories can interfere, and several advanced policies remain
  opt-in or experimental.
- Receipts are primarily from Linux, local consumer NVIDIA GPUs, and a
  research-grade setup. Broader hardware and OS coverage is unconfirmed.

## Documentation

- [Complete results, failures, corrections, and receipt map](docs/RESULTS_INDEX.md)
- [GRM methodology and experiment ledger](docs/GRM_Methodology.md)
- [Architecture record](docs/GraftRepository_Memory_Architecture.md)
- [GRM research paper draft](docs/GRM_PAPER_DRAFT.md)
- [Project-Tensor APA overview](https://github.com/DragonShadows1978/Project-Tensor/blob/main/docs/APA.md)
- [Project-Tensor status matrix](https://github.com/DragonShadows1978/Project-Tensor/blob/main/docs/SUPPORT_MATRIX.md)
- [License](LICENSE): GNU AGPL v3.0; research papers are CC BY 4.0 via their Zenodo records

Copyright (C) 2026 David Perry. Commercial licensing outside the AGPL terms is
available from `dave@ai-storyforge.com`.
