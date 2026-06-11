"""Qwen2.5-1.5B fp16 (unquantized) on tensor_cuda. Mid-size point (1.5B) between
TinyLlama-1.1B and Mistral-7B for the APA-cost-vs-model-size curve.

Differs from Llama/Mistral: q/k/v projections have BIAS; embeddings are TIED
(lm_head reuses embed_tokens.weight); 1536 hidden, 28 layers, 12 Q / 2 KV heads,
head_dim 128, 8960 FFN, 152K vocab, rope_theta 1e6.
"""
import os
import numpy as np

from core.mistral7b_tc import Mistral7B_TC, LinearTC, BlockTC, tc


class QwenConfig:
    vocab_size = 151936
    hidden_dim = 1536
    intermediate_dim = 8960
    num_layers = 28
    num_heads = 12
    num_kv_heads = 2
    head_dim = 128            # 1536/12
    max_position_embeddings = 32768
    rope_theta = 1000000.0
    rms_norm_eps = 1e-6

    @property
    def num_heads_per_kv(self):
        return self.num_heads // self.num_kv_heads


_SNAP = ("/home/vader/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B/"
         "snapshots/8faed761d45a263340a0528343f099c05c9a4323")


class _TiedLMHead:
    """lm_head that reuses the embedding weight (vocab, hidden): y = x @ W^T."""
    def __init__(self, embed_weight):
        self.w = embed_weight  # tc tensor (vocab, hidden) fp16

    def __call__(self, x):
        # trans_b reads (vocab, hidden) row-major via cuBLAS OP_T — the old
        # .transpose() materialized a 188M-element permute EVERY forward
        # (38.6ms/step on MiniCPM3, 10% of decode).
        return tc.matmul(x, self.w, trans_b=True)

    def vram_bytes(self):
        return 0  # shares the embedding's memory


class Qwen_TC(Mistral7B_TC):
    def load_weights_fp16(self, model_dir=None):
        import torch
        from safetensors.torch import load_file
        path = os.path.join(model_dir or _SNAP, "model.safetensors")
        W = {k: v for k, v in load_file(path).items()}

        def g(name):
            return W.pop(name).to(torch.float32).numpy()

        def gb(name):
            return g(name) if name in W else None

        emb = g("model.embed_tokens.weight").astype(np.float32)
        self.embed_tokens.weight = tc.tensor(np.ascontiguousarray(emb)).astype("bfloat16")

        for i in range(self.config.num_layers):
            p = f"model.layers.{i}"
            L = self.layers[i]
            L.input_layernorm.weight = tc.tensor(
                np.ascontiguousarray(g(f"{p}.input_layernorm.weight")), dtype="float32")
            L.post_attention_layernorm.weight = tc.tensor(
                np.ascontiguousarray(g(f"{p}.post_attention_layernorm.weight")), dtype="float32")
            L.self_attn.q_proj = LinearTC(g(f"{p}.self_attn.q_proj.weight"),
                                          gb(f"{p}.self_attn.q_proj.bias"))
            L.self_attn.k_proj = LinearTC(g(f"{p}.self_attn.k_proj.weight"),
                                          gb(f"{p}.self_attn.k_proj.bias"))
            L.self_attn.v_proj = LinearTC(g(f"{p}.self_attn.v_proj.weight"),
                                          gb(f"{p}.self_attn.v_proj.bias"))
            L.self_attn.o_proj = LinearTC(g(f"{p}.self_attn.o_proj.weight"))
            L.mlp.gate_proj = LinearTC(g(f"{p}.mlp.gate_proj.weight"))
            L.mlp.up_proj = LinearTC(g(f"{p}.mlp.up_proj.weight"))
            L.mlp.down_proj = LinearTC(g(f"{p}.mlp.down_proj.weight"))

        self.norm.weight = tc.tensor(
            np.ascontiguousarray(g("model.norm.weight")), dtype="float32")
        # tied: lm_head reuses the embedding weight
        self.lm_head = _TiedLMHead(self.embed_tokens.weight)
        W.clear()
        return {"loaded": "fp16", "framework": "tensor_cuda fp16 (unquantized, Qwen)"}

    @classmethod
    def from_pretrained(cls, model_dir=None, attention_mode="standard", config=None):
        if config is None:
            config = QwenConfig()
        # Qwen2.5 activations overflow fp16 mid-network — run the whole forward in
        # bf16 (fp32 exponent range). Set BEFORE building/loading so all linears
        # and the residual stream use bf16.
        BlockTC.COMPUTE_DTYPE = "bfloat16"
        LinearTC.DTYPE = "bfloat16"
        with tc.no_grad():
            m = cls(config, attention_mode)
            info = m.load_weights_fp16(model_dir)
        # Qwen's keys are outlier-heavy (unit-norm top-dim variance ~9x mean), so
        # 2-bit TurboQuant bulk destroys them (score corr 0.03 -> ppl 555). 8-bit
        # bulk recovers it (corr 0.85 -> ppl 8.13 == standard). Outlier-key models
        # need a higher bulk bit-width matched to their key distribution.
        for L in m.layers:
            L.self_attn.bulk_bits = 8
        return m, info
