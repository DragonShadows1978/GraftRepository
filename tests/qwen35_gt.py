"""Ground-truth vectors for the Qwen3.5-9B tensor_cuda port.

PyTorch fp32 CPU reference (the GT law: adjudicate against PyTorch,
never an in-repo run). Per prompt: token ids, teacher-forced logits
(S, V) fp32, all 33 hidden states (embeddings + 32 layers) fp32,
greedy 16-token continuation, and the layer-0/layer-3 cache states
(DeltaNet conv+recurrent, attention K/V) for bisection.

  python3 tests/qwen35_gt.py /mnt/ForgeRealm/qwen35_gt
"""
import os
import sys

import numpy as np
import torch

OUT = sys.argv[1] if len(sys.argv) > 1 else "/mnt/ForgeRealm/qwen35_gt"
SNAP = ("/home/vader/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/"
        "snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a")

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    if n",
    "SERVICE RECORD. The dome motor (unit VORMEK-4821) received its "
    "overhaul; technician logged 73 hours runtime and signed the sheet. "
    "The unit code mentioned above is",
]

os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0)

from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration  # noqa: E402

tok = AutoTokenizer.from_pretrained(SNAP)
print("tokenizer loaded", flush=True)
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    SNAP, dtype=torch.float32)
model.eval()
print("model loaded fp32 CPU", flush=True)

for pi, prompt in enumerate(PROMPTS):
    ids = tok(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True,
                    use_cache=True)
        gen = model.generate(ids, max_new_tokens=16, do_sample=False)
    cache = out.past_key_values
    # layer 0 = DeltaNet (conv + recurrent states); layer 3 = attention K/V
    l0 = cache.layers[0]
    l3 = cache.layers[3]
    np.savez(
        os.path.join(OUT, f"gt_{pi}.npz"),
        ids=ids.numpy(),
        logits=out.logits[0].float().numpy(),
        hidden=np.stack([h[0].float().numpy() for h in out.hidden_states]),
        greedy=gen[0].numpy(),
        l0_conv=l0.conv_states.float().numpy(),
        l0_rec=l0.recurrent_states.float().numpy(),
        l3_k=l3.keys.float().numpy(),
        l3_v=l3.values.float().numpy(),
    )
    print(f"GT {pi}: S={ids.shape[1]} top1={out.logits[0,-1].argmax().item()}"
          f" greedy={tok.decode(gen[0][ids.shape[1]:])!r}", flush=True)

print("GT DONE", flush=True)
