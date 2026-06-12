"""Long-context GT for the Gemma 4 port: one ~1200-token prompt, fp32
CPU, last-row logits only. This is the EXTERNAL adjudicator for the
sliding-window semantics (band mask + ring trim) — the short-prompt GT
never makes the window bind, and engine-internal consistency can't
catch a consistently-wrong window convention.

  python3 tests/gemma4_longctx_gt.py /mnt/ForgeRealm/gemma4_gt
"""
import os
import sys

import numpy as np
import torch

OUT = sys.argv[1] if len(sys.argv) > 1 else "/mnt/ForgeRealm/gemma4_gt"
SNAP = os.environ.get("GEMMA4_SNAP", "/mnt/ForgeRealm/models/gemma-4-12B-it")

from transformers import (  # noqa: E402
    AutoTokenizer,
    Gemma4UnifiedForConditionalGeneration,
)

tok = AutoTokenizer.from_pretrained(SNAP)

# non-periodic filler so attention has real structure across the window
parts = []
i = 0
while True:
    parts.append(
        f"Entry {i}: depot {chr(65 + i % 23)}{100 + (i * 37) % 900} logged "
        f"{(i * 13) % 97} crates of grade-{1 + i % 5} ore at hour "
        f"{(i * 7) % 24}. ")
    i += 1
    ids = tok("".join(parts), return_tensors="pt").input_ids
    if ids.shape[1] >= 1200:
        break
ids = ids[:, :1200]
print(f"prompt: {ids.shape[1]} tokens", flush=True)

model = Gemma4UnifiedForConditionalGeneration.from_pretrained(
    SNAP, dtype=torch.float32)
model.eval()
print("model loaded fp32 CPU", flush=True)

with torch.no_grad():
    out = model(input_ids=ids, logits_to_keep=1)
np.savez(os.path.join(OUT, "gt_long.npz"),
         ids=ids.numpy(),
         last_logits=out.logits[0, -1].float().numpy())
print("gt_long saved; GT COMPLETE", flush=True)
