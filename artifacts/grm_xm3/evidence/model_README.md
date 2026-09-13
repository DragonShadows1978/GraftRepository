---
license: other
language:
- en
- es
- fr
- de
- it
- pt
- ru
- ar
- hi
- ko
- zh
library_name: transformers
base_model:
- arcee-ai/Trinity-Nano-Base
license_link: LICENSE
license_name: openmdw-1.1
---
<div align="center">
  <picture>
    <img
      src="https://cdn-uploads.huggingface.co/production/uploads/6435718aaaef013d1aec3b8b/i-v1KyAMOW_mgVGeic9WJ.png"
      alt="Arcee Trinity Mini"
      style="max-width: 100%; height: auto;"
    >
  </picture>
</div>

# Trinity Nano Preview

Trinity Nano Preview is a preview of Arcee AI's 6B MoE model with 1B active parameters. It is the small-sized model in our new Trinity family, a series of open-weight models for enterprise and tinkerers alike.

This is a chat tuned model, with a delightful personality and charm we think users will love. We note that this model is pushing the limits of sparsity in small language models with only 800M non-embedding parameters active per token, and as such **may be unstable** in certain use cases, especially in this preview.

This is an *experimental* release, it's fun to talk to but will not be hosted anywhere, so download it and try it out yourself!

***

Trinity Nano Preview is trained on 10T tokens gathered and curated through a key partnership with [Datology](https://www.datologyai.com/), building upon the excellent dataset we used on [AFM-4.5B](https://huggingface.co/arcee-ai/AFM-4.5B) with additional math and code.

Training was performed on a cluster of 512 H200 GPUs powered by [Prime Intellect](https://www.primeintellect.ai/) using HSDP parallelism.

More details, including key architecture decisions, can be found on our blog [here](https://www.arcee.ai/blog/the-trinity-manifesto)

***

## Model Details

* **Model Architecture:** AfmoeForCausalLM
* **Parameters:** 6B, 1B active
* **Experts:** 128 total, 8 active, 1 shared
* **Context length:** 128k
* **Training Tokens:** 10T
* **License:** [OpenMDW-1.1](https://huggingface.co/arcee-ai/Trinity-Mini#license)

***

<div align="center">
  <picture>
      <img src="https://cdn-uploads.huggingface.co/production/uploads/6435718aaaef013d1aec3b8b/sSVjGNHfrJKmQ6w8I18ek.png" style="background-color:ghostwhite;padding:5px;" width="17%" alt="Powered by Datology">
  </picture>
</div>

### Running our model

- [Transformers](https://huggingface.co/arcee-ai/Trinity-Mini#transformers)
- [VLLM](https://huggingface.co/arcee-ai/Trinity-Mini#vllm)
- [llama.cpp](https://huggingface.co/arcee-ai/Trinity-Mini#llamacpp)
- [LM Studio](https://huggingface.co/arcee-ai/Trinity-Mini#lm-studio)

## Transformers

Use the `main` transformers branch

```
git clone https://github.com/huggingface/transformers.git
cd transformers

# pip
pip install '.[torch]'

# uv
uv pip install '.[torch]'
```

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "arcee-ai/Trinity-Nano-Preview"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

messages = [
    {"role": "user", "content": "Who are you?"},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.5,
    top_k=50,
    top_p=0.95
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

If using a released transformers, simply pass "trust_remote_code=True":

```python
model_id = "arcee-ai/Trinity-Nano-Preview"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
```

## VLLM

Supported in VLLM release 0.11.1

```
# pip
pip install "vllm>=0.11.1"
```

Serving the model with suggested settings:

```
vllm serve arcee-train/Trinity-Nano-Preview \
  --dtype bfloat16 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_r1 \
  --tool-call-parser hermes
```

## llama.cpp

Supported in llama.cpp release b7061

Download the latest [llama.cpp release](https://github.com/ggml-org/llama.cpp/releases)

```
llama-server -hf arcee-ai/Trinity-Nano-Preview-GGUF:q4_k_m
```

## LM Studio

Supported in latest LM Studio runtime

Update to latest available, then verify your runtime by:

1. Click "Power User" at the bottom left
2. Click the green "Developer" icon at the top left
3. Select "LM Runtimes" at the top
4. Refresh the list of runtimes and verify that the latest is installed

Then, go to Model Search and search for `arcee-ai/Trinity-Nano-Preview-GGUF`, download your prefered size, and load it up in the chat


## License

Trinity-Nano-Preview is released under the OpenMDW-1.1 license.