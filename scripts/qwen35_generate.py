"""Qwen3.5-9B chat/generate on tensor_cuda — the ready-to-work wrapper.

Greedy or temperature sampling with the hybrid cache (KV @ 8 attention
layers, DeltaNet conv+recurrent states @ 24 linear layers). Chat
template applied via the HF tokenizer (thinking disabled by default —
enable_thinking=False mirrors ollama's `think: false`).

  python3 scripts/qwen35_generate.py "prompt text" [max_new] [--raw]
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc                                    # noqa: E402
from core.qwen35_tc import Qwen35_TC, _snap                 # noqa: E402

PROMPT = sys.argv[1] if len(sys.argv) > 1 else "Why is the sky blue?"
MAX_NEW = int(sys.argv[2]) if len(sys.argv) > 2 else 128
RAW = "--raw" in sys.argv

from transformers import AutoTokenizer                      # noqa: E402
tok = AutoTokenizer.from_pretrained(_snap())

if RAW:
    ids = tok(PROMPT, return_tensors="np").input_ids.astype(np.int64)
else:
    text = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, return_tensors="np").input_ids.astype(np.int64)

tc.set_alloc_pooling(True)
m, info = Qwen35_TC.from_pretrained()
print(f"loaded: {info}", flush=True)

S = ids.shape[1]
t0 = time.perf_counter()
with tc.no_grad():
    logits, caches = m(ids, last_token_only=True)
tc.synchronize()
t1 = time.perf_counter()
print(f"prefill {S} tok in {t1 - t0:.2f}s", flush=True)

def _greedy_next(lg):
    """Pick the argmax token id for the last position of `lg` (1, L, vocab),
    L==1 for every call site here (prefill uses last_token_only=True; decode
    feeds one token at a time).

    Host argmax. A device-side argmax fast path (tc.argmax_last_axis) was
    tried here and REVERTED per the kernel-opt program's registered gate
    (KERNEL_OPT_IMPLEMENTATION_LEDGER 2026-07-07: interleaved retest −0.9%
    e2e — the per-token vocab-row copy is not a material cost at this
    workload). The op itself remains in tensor_cuda; graph-mode decode
    (parked branch kernel-opt-phase2-parked) uses it internally.
    """
    return int(lg.float().numpy()[0, -1].argmax())


out = []
nxt = _greedy_next(logits)
off = S

# Graph-mode decode (kernel-opt Phase 2, B3): gated behind TC_DECODE_GRAPH,
# defaulting OFF. Unset/"0" -> the branch below is skipped entirely and
# control flow through the eager per-token m(...) loop is BYTE-IDENTICAL to
# before this change (same call, same bookkeeping, same loop shape). "1"
# captures the 8 per-run DeltaNet CUDA graphs once (after prefill, seeded
# from prefill's own caches) and replays them every subsequent token,
# interspersed with eager attention-layer calls; see core/qwen35_tc.py
# (_Qwen35DecodeGraphs) for the actual mechanics -- this script only adds
# the env-var branch and the graphs-handle setup/threading.
USE_GRAPH = os.environ.get("TC_DECODE_GRAPH", "0") == "1"

if USE_GRAPH:
    with tc.no_grad():
        graphs = m.build_decode_graphs()
        kv_caches = graphs.seed_from_prefill_caches(caches)
        while len(out) < MAX_NEW:
            if nxt == m.config.eos_token_id:
                break
            out.append(nxt)
            lg, kv_caches = m.decode_step_graph(nxt, off, graphs, kv_caches)
            off += 1
            nxt = _greedy_next(lg)
else:
    with tc.no_grad():
        for _ in range(MAX_NEW):
            if nxt == m.config.eos_token_id:
                break
            out.append(nxt)
            lg, caches = m(np.array([[nxt]], np.int64), caches=caches,
                           position_offset=off)
            off += 1
            nxt = _greedy_next(lg)
tc.synchronize()
t2 = time.perf_counter()
n = len(out)
print(f"decode {n} tok in {t2 - t1:.2f}s = {n / max(t2 - t1, 1e-9):.1f} tok/s",
      flush=True)
print("---")
print(tok.decode(out, skip_special_tokens=True))
if "--print-ids" in sys.argv:
    # Deterministic raw-token-id readout for the graph-mode parity check
    # (kernel-opt Phase 2, C1) -- the decoded text above isn't precise
    # enough to prove token-for-token identity byte-for-byte.
    print("IDS:" + ",".join(str(t) for t in out))
print("DONE", flush=True)
