"""APA gate for Qwen3.5: apa_selective (bulk4 / refine 0.15, cuBLAS
blend path) vs the engine's own standard attention — engine-vs-engine,
both INT4, so disagreements must be near-tie flips under the STANDARD
run's logits. Plus a chat coherence smoke in APA mode.

  python3 tests/test_qwen35_apa.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc                                    # noqa: E402
from core.qwen35_tc import Qwen35_TC, _snap                 # noqa: E402

GT_DIR = "/mnt/ForgeRealm/qwen35_gt"

tc.set_alloc_pooling(True)
m, info = Qwen35_TC.from_pretrained()
print(f"loaded: {info}", flush=True)


def set_mode(mode):
    for L in m.layers:
        if L.is_attn:
            L.mixer.attention_mode = mode


overall = True
for pi in range(3):
    gt = np.load(os.path.join(GT_DIR, f"gt_{pi}.npz"))
    ids = gt["ids"]
    set_mode("standard")
    with tc.no_grad():
        lg_std, _ = m(ids)
    std = lg_std.float().numpy()[0]
    set_mode("apa_selective")
    with tc.no_grad():
        lg_apa, _ = m(ids)
    apa = lg_apa.float().numpy()[0]
    dis = np.nonzero(std.argmax(-1) != apa.argmax(-1))[0]
    costs = [float(std[t].max() - std[t][apa[t].argmax()]) for t in dis]
    worst = max(costs, default=0.0)
    ok = worst <= 3.0
    overall &= ok
    print(f"prompt {pi}: {len(dis)}/{ids.shape[1]} flips vs standard | "
          f"worst cost {worst:.3f} | {'OK' if ok else 'MISMATCH'}",
          flush=True)

# coherence smoke in APA mode
from transformers import AutoTokenizer                      # noqa: E402
tok = AutoTokenizer.from_pretrained(_snap())
text = tok.apply_chat_template(
    [{"role": "user", "content": "Why is the sky blue? One sentence."}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False)
ids = tok(text, return_tensors="np").input_ids.astype(np.int64)
set_mode("apa_selective")
t0 = time.perf_counter()
with tc.no_grad():
    lg, caches = m(ids, last_token_only=True)
    nxt = int(lg.float().numpy()[0, -1].argmax())
    off, out = ids.shape[1], []
    for _ in range(40):
        if nxt == m.config.eos_token_id:
            break
        out.append(nxt)
        lg, caches = m(np.array([[nxt]], np.int64), caches=caches,
                       position_offset=off)
        off += 1
        nxt = int(lg.float().numpy()[0, -1].argmax())
print(f"\nAPA-mode answer ({len(out)} tok, "
      f"{len(out) / (time.perf_counter() - t0):.1f} tok/s): "
      f"{tok.decode(out, skip_special_tokens=True)}", flush=True)
print(f"APA GATE: {'PASS' if overall else 'FAIL'}", flush=True)
print("DONE", flush=True)
