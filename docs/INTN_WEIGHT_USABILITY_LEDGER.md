# INT3 Weight Usability Ledger

This ledger records whether the Qwen3.5-9B INT3 PPL jump is worth using for
actual behavior, not just whether it fits in memory.

## 2026-07-06

Action: Opened the Qwen3.5-9B INT3 usability track.

Reason:
- The 9B INT3 smoke PPL was damaged but not collapsed.
- The important question is whether that damage matters for useful model
  behavior.

Implementation:
- Added `scripts/qwen35_intn_usability_gate.py`.
- The gate loads real Qwen3.5-9B weights through `Qwen35_TC`.
- It compares INT4 and INT3.
- It runs standard attention and APA r0.15.
- It asks greedy exact-extraction questions over planted facts.
- It writes JSON artifacts under `artifacts/intn_usability/`.

Pending:
- Run the gate after the broad 9B PPL sweep completes.
- Record exact hit counts and generated answers.

## 2026-07-06 17:16-17:21 EDT

Action: Ran the first Qwen3.5-9B INT4/INT3 usability gate.

Protocol:
- Real Qwen3.5-9B TensorCUDA model.
- Weight widths: INT4 and INT3.
- Attention settings: standard and APA r0.15.
- Prompt type: short planted-fact document.
- Decode: greedy.
- Score: exact expected-value containment after normalization, plus a simple
  degeneration check.

Command:
- `env PYTHONUNBUFFERED=1 PYTHONPATH=/mnt/ForgeRealm/GraftRepository:/mnt/ForgeRealm/Project-Tensor/tensor_cuda PYTHONPYCACHEPREFIX=/tmp/codex_pycache python3 scripts/qwen35_intn_usability_gate.py --model-dir /home/vader/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a --bits 4,3 --modes standard,apa_r0.15 --max-new 32`

Artifact:
- `artifacts/intn_usability/qwen35_intn_usability_20260706_171636.json`

Results:

| Bits | Setting | Hits | Failures | Load MiB | Max observed MiB |
| ---: | --- | ---: | ---: | ---: | ---: |
| 4 | standard | 5/5 | 0 | 4536 | 4584 |
| 4 | APA r0.15 | 5/5 | 0 | 4536 | 4616 |
| 3 | standard | 5/5 | 0 | 3622 | 3622 |
| 3 | APA r0.15 | 5/5 | 0 | 3622 | 3622 |

Exact answers:
- Vault code: `LUMEN-482`
- Backup pilot: `Mara Voss`
- Coolant marker: `amber` / `Amber`
- Checkpoint city: `Kenora`
- Numeric handshake: `73194`

Interpretation:
- INT4 passed the basic exact-extraction behavior gate.
- INT3 also passed the same gate in both standard and APA r0.15 modes.
- This does not prove INT3 is production-ready. It does show that the +23% PPL
  hit does not immediately destroy simple greedy fact extraction on Qwen3.5-9B.
- The next usability gate should be harder: longer documents, distractors,
  multi-fact questions, and at least one open-ended answer where exact matching
  alone is not enough.
