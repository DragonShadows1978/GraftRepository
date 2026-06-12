"""Margin diagnostic: are engine-vs-GT top-1 disagreements INT4
tie-flips (GT top-2 margin small at exactly the disagreeing positions)
or systematic (large-margin flips = math bug)? Plus a chat-template
coherence smoke (the ready-to-work check) on the same load.

  python3 tests/qwen35_margin_diag.py
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

for pi in range(3):
    gt = np.load(os.path.join(GT_DIR, f"gt_{pi}.npz"))
    ids = gt["ids"]
    with tc.no_grad():
        logits, _ = m(ids)
    eng = logits.float().numpy()[0]
    ref = gt["logits"]
    dis = np.nonzero(eng.argmax(-1) != ref.argmax(-1))[0]
    rows = []
    for t in dis:
        r = ref[t]
        top2 = np.partition(r, -2)[-2:]
        gt_margin = float(top2[1] - top2[0])           # GT top1 - top2 gap
        # margin between GT's top1 and the ENGINE's chosen token, under GT
        flip_cost = float(r.max() - r[eng[t].argmax()])
        rows.append((int(t), round(gt_margin, 3), round(flip_cost, 3)))
    worst = max((r[2] for r in rows), default=0.0)
    print(f"prompt {pi}: {len(dis)} disagreements "
          f"(pos, GT-top2-margin, GT-cost-of-engine-pick): {rows} "
          f"| worst flip cost {worst:.3f}", flush=True)

# ---- coherence smoke through the chat template
from transformers import AutoTokenizer                      # noqa: E402
tok = AutoTokenizer.from_pretrained(_snap())
for q in ("Why is the sky blue? Answer in one sentence.",
          "Write a haiku about a foundry."):
    text = tok.apply_chat_template([{"role": "user", "content": q}],
                                   tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    ids = tok(text, return_tensors="np").input_ids.astype(np.int64)
    t0 = time.perf_counter()
    with tc.no_grad():
        lg, caches = m(ids, last_token_only=True)
        nxt = int(lg.float().numpy()[0, -1].argmax())
        off, out = ids.shape[1], []
        for _ in range(60):
            if nxt == m.config.eos_token_id:
                break
            out.append(nxt)
            lg, caches = m(np.array([[nxt]], np.int64), caches=caches,
                           position_offset=off)
            off += 1
            nxt = int(lg.float().numpy()[0, -1].argmax())
    dt = time.perf_counter() - t0
    print(f"\nQ: {q}\nA: {tok.decode(out, skip_special_tokens=True)}\n"
          f"({len(out)} tok, {len(out) / dt:.1f} tok/s incl prefill)",
          flush=True)
print("DONE", flush=True)
