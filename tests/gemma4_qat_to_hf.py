"""Convert the Gemma 4 QAT q4_0 GGUF to an fp32 HF checkpoint so
transformers can serve as the GT oracle for the SAME weights the
engine runs. fp32 because the q4_0 grid (s fp16 x (q-8)) does not
round-trip through bf16/fp16 exactly. Config/tokenizer files are
copied from the bf16 release dir (same arch/vocab).

  python3 tests/gemma4_qat_to_hf.py /mnt/ForgeRealm/models/gemma4-qat-hf
"""
import os
import shutil
import sys

import numpy as np
import torch
from gguf import GGUFReader
from gguf.quants import dequantize
from safetensors.torch import save_file

OUT = sys.argv[1] if len(sys.argv) > 1 else \
    "/mnt/ForgeRealm/models/gemma4-qat-hf"
SRC = "/mnt/ForgeRealm/models/gemma-4-12B-it"
GGUF = ("/mnt/ForgeRealm/models/gemma-4-12B-it-qat/"
        "gemma-4-12b-it-qat-q4_0.gguf")

NORM_MAP = {
    "attn_norm": "input_layernorm",
    "post_attention_norm": "post_attention_layernorm",
    "ffn_norm": "pre_feedforward_layernorm",
    "post_ffw_norm": "post_feedforward_layernorm",
}
ATTN_MAP = {"attn_q": "q_proj", "attn_k": "k_proj", "attn_v": "v_proj",
            "attn_output": "o_proj", "attn_q_norm": "q_norm",
            "attn_k_norm": "k_norm"}
FFN_MAP = {"ffn_gate": "gate_proj", "ffn_up": "up_proj",
           "ffn_down": "down_proj"}

P = "model.language_model"


def hf_name(name):
    if name == "token_embd.weight":
        return f"{P}.embed_tokens.weight"
    if name == "output_norm.weight":
        return f"{P}.norm.weight"
    _, li, rest = name.split(".", 2)
    rest = rest.rsplit(".weight", 1)[0]
    if rest == "layer_output_scale":
        return f"{P}.layers.{li}.layer_scalar"
    if rest in NORM_MAP:
        return f"{P}.layers.{li}.{NORM_MAP[rest]}.weight"
    if rest in ATTN_MAP:
        return f"{P}.layers.{li}.self_attn.{ATTN_MAP[rest]}.weight"
    if rest in FFN_MAP:
        return f"{P}.layers.{li}.mlp.{FFN_MAP[rest]}.weight"
    raise KeyError(f"unmapped tensor {name}")


def layer_of(name):
    return int(name.split(".")[1]) if name.startswith("blk.") else -1


# Two shards (fp32 total ~48GB; a single save_file dict OOMs 62GB RAM):
# shard 1 = embeddings + blk.0..23, shard 2 = blk.24..47 + final norm.
r = GGUFReader(GGUF)
os.makedirs(OUT, exist_ok=True)
for f in ("config.json", "generation_config.json", "tokenizer.json",
          "tokenizer_config.json", "chat_template.jinja",
          "processor_config.json"):
    shutil.copy2(os.path.join(SRC, f), os.path.join(OUT, f))

index = {"metadata": {"total_size": 0}, "weight_map": {}}
for si, keep in ((1, lambda li: li < 24), (2, lambda li: li >= 24)):
    sd = {}
    for t in r.tensors:
        if t.name == "rope_freqs.weight":
            continue
        li = layer_of(t.name)
        if li == -1:
            in_shard = (si == 1) if "embd" in t.name else (si == 2)
        else:
            in_shard = keep(li)
        if not in_shard:
            continue
        data = dequantize(t.data, t.tensor_type).astype(np.float32) \
            if int(t.tensor_type) != 0 else np.asarray(t.data, np.float32)
        sd[hf_name(t.name)] = torch.from_numpy(np.ascontiguousarray(data))
    fname = f"model-0000{si}-of-00002.safetensors"
    save_file(sd, os.path.join(OUT, fname), metadata={"format": "pt"})
    for k, v in sd.items():
        index["weight_map"][k] = fname
        index["metadata"]["total_size"] += v.numel() * 4
    print(f"shard {si}: {len(sd)} tensors saved", flush=True)
    del sd

import json                                                # noqa: E402
with open(os.path.join(OUT, "model.safetensors.index.json"), "w") as fh:
    json.dump(index, fh)
print(f"index written; total "
      f"{index['metadata']['total_size'] / 1e9:.1f} GB", flush=True)
print("DONE", flush=True)
