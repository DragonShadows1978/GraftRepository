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

out = []
nxt = int(logits.float().numpy()[0, -1].argmax())
off = S
with tc.no_grad():
    for _ in range(MAX_NEW):
        if nxt == m.config.eos_token_id:
            break
        out.append(nxt)
        lg, caches = m(np.array([[nxt]], np.int64), caches=caches,
                       position_offset=off)
        off += 1
        nxt = int(lg.float().numpy()[0, -1].argmax())
tc.synchronize()
t2 = time.perf_counter()
n = len(out)
print(f"decode {n} tok in {t2 - t1:.2f}s = {n / max(t2 - t1, 1e-9):.1f} tok/s",
      flush=True)
print("---")
print(tok.decode(out, skip_special_tokens=True))
print("DONE", flush=True)
