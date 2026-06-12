"""bf16 GT discriminator for the Gemma 4 port.

If engine-vs-fp32-GT distance is precision accumulation (bf16 compute +
INT4 weights over 48 layers + tied softcapped head) rather than a port
bug, then GT re-dumped in the model's TRAINED dtype (bf16) should sit
much closer to the engine: margins collapse, layer cosines rise. A port
bug would keep the distance regardless of GT dtype.

  python3 tests/gemma4_gt_bf16.py /mnt/ForgeRealm/gemma4_gt
"""
import os
import sys

import numpy as np
import torch

OUT = sys.argv[1] if len(sys.argv) > 1 else "/mnt/ForgeRealm/gemma4_gt"
SNAP = "/mnt/ForgeRealm/models/gemma-4-12B-it"

from transformers import (  # noqa: E402
    AutoTokenizer,
    Gemma4UnifiedForConditionalGeneration,
)

tok = AutoTokenizer.from_pretrained(SNAP)
model = Gemma4UnifiedForConditionalGeneration.from_pretrained(
    SNAP, dtype=torch.bfloat16)
model.eval()
print("model loaded bf16 CPU", flush=True)

for pi, prompt in enumerate(["The capital of France is"]):
    ids = tok(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True)
    np.savez(
        os.path.join(OUT, f"gt_bf16_{pi}.npz"),
        ids=ids.numpy(),
        logits=out.logits[0].float().numpy(),
        hidden=np.stack([h[0].float().numpy() for h in out.hidden_states]),
    )
    print(f"prompt {pi} saved (bf16)", flush=True)
print("GT COMPLETE", flush=True)
