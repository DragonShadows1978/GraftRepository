"""Ground-truth vectors for the Gemma 4 12B Unified tensor_cuda port.

PyTorch fp32 CPU reference (the GT law: adjudicate against PyTorch,
never an in-repo run). Per prompt: token ids, teacher-forced logits
(S, V) fp32, per-layer hidden states fp32, greedy 16-token
continuation, and layer-0 (sliding) / layer-5 (global K=V) cache
tensors for bisection.

  python3 tests/gemma4_gt.py /mnt/ForgeRealm/gemma4_gt
"""
import os
import sys

import numpy as np
import torch

OUT = sys.argv[1] if len(sys.argv) > 1 else "/mnt/ForgeRealm/gemma4_gt"
SNAP = os.environ.get("GEMMA4_SNAP", "/mnt/ForgeRealm/models/gemma-4-12B-it")

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    if n",
    "SERVICE RECORD. The dome motor (unit VORMEK-4821) received its "
    "overhaul; technician logged 73 hours runtime and signed the sheet. "
    "The unit code mentioned above is",
    # The -it model is hard template-bound: raw text greedy-decodes into
    # degenerate loops (pure near-tie territory). Templated prompt gives
    # the margin gate one coherent, high-margin anchor.
    {"role": "user", "content": "What is the capital of France? "
     "Answer in one word."},
]

os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0)

from transformers import (  # noqa: E402
    AutoTokenizer,
    Gemma4UnifiedForConditionalGeneration,
)

tok = AutoTokenizer.from_pretrained(SNAP)
print("tokenizer loaded", flush=True)
model = Gemma4UnifiedForConditionalGeneration.from_pretrained(
    SNAP, dtype=torch.float32)
model.eval()
print("model loaded fp32 CPU", flush=True)

for pi, prompt in enumerate(PROMPTS):
    if isinstance(prompt, dict):
        ids = tok.apply_chat_template(
            [prompt], add_generation_prompt=True, return_tensors="pt",
            return_dict=True)["input_ids"]
    else:
        ids = tok(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True,
                    use_cache=True)
        gen = model.generate(ids, max_new_tokens=16, do_sample=False)
    cache = out.past_key_values
    hiddens = [h[0].float().numpy() for h in out.hidden_states]
    print(f"prompt {pi}: ids {ids.shape} hiddens {len(hiddens)} "
          f"x {hiddens[0].shape} logits {out.logits.shape}", flush=True)
    l0 = cache.layers[0]   # sliding: 8 kv heads x 256
    l5 = cache.layers[5]   # global: 1 kv head x 512, K=V shared proj
    np.savez(
        os.path.join(OUT, f"gt_{pi}.npz"),
        ids=ids.numpy(),
        logits=out.logits[0].float().numpy(),
        hidden=np.stack(hiddens),
        greedy=gen[0].numpy(),
        l0_k=l0.keys.float().numpy(),
        l0_v=l0.values.float().numpy(),
        l5_k=l5.keys.float().numpy(),
        l5_v=l5.values.float().numpy(),
    )
    print(f"prompt {pi} saved; greedy tail: "
          f"{tok.decode(gen[0][ids.shape[1]:])!r}", flush=True)

print("GT COMPLETE", flush=True)
