"""Qwen3-4B fp16/bf16 (unquantized) on tensor_cuda. The model Prometheus Mind used
— larger than Qwen2.5-1.5B, with more coherence headroom for the KV-graft work.

Differs from Qwen2.5: NO q/k/v bias (attention_bias=false); adds PER-HEAD QK-norm
(RMSNorm over head_dim=128, applied AFTER the head reshape — unlike OLMoE which
norms over the full hidden_dim BEFORE reshape). 36 layers, 2560 hidden,
32 Q / 8 KV heads (GQA), head_dim 128 (note 32*128=4096 != 2560, so q_proj is
2560->4096), 9728 FFN, 152K vocab tied, rope_theta 1e6, bf16-native.
"""
import os
import numpy as np

from core.mistral7b_tc import (Mistral7B_TC, LinearTC, QuantLinearTC, RMSNormTC,
                               BlockTC, F, tc)
from core.qwen_tc import _TiedLMHead


class Qwen3Config:
    vocab_size = 151936
    hidden_dim = 2560
    intermediate_dim = 9728
    num_layers = 36
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128                    # 32*128 = 4096 != hidden 2560 (non-square q_proj)
    max_position_embeddings = 40960
    rope_theta = 1000000.0
    rms_norm_eps = 1e-6

    @property
    def num_heads_per_kv(self):
        return self.num_heads // self.num_kv_heads


_SNAP_GLOB = "/home/vader/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/*"


def _snap():
    import glob
    hits = sorted(glob.glob(_SNAP_GLOB))
    if not hits:
        raise FileNotFoundError("Qwen3-4B snapshot not found in HF cache")
    return hits[-1]


def _per_head_qk_norm(att, x, n_heads, head_dim, B, L):
    """Apply Qwen3 per-head RMSNorm over head_dim. x is (B, L, n_heads*head_dim);
    norm each head's 128-vec independently with the (head_dim,) weight, in fp32."""
    xr = x.reshape([B, L, n_heads, head_dim])
    xf = xr.float()
    ms = (xf * xf).mean([-1], True)
    normed = xf * (ms + att._qk_eps).pow(-0.5) * att._qk_w  # weight broadcasts over head_dim
    cdt = BlockTC.COMPUTE_DTYPE
    normed = normed.half() if cdt == "float16" else normed.astype(cdt)
    return normed.reshape([B, L, n_heads * head_dim])


class Qwen3_TC(Mistral7B_TC):
    def load_weights_fp16(self, model_dir=None):
        import torch, glob
        from safetensors.torch import load_file
        d = model_dir or _snap()
        W = {}
        for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
            W.update(load_file(f))

        def g(name):
            return W.pop(name).to(torch.float32).numpy()

        emb = g("model.embed_tokens.weight").astype(np.float32)
        self.embed_tokens.weight = tc.tensor(np.ascontiguousarray(emb)).astype("bfloat16")

        cfg = self.config
        for i in range(cfg.num_layers):
            p = f"model.layers.{i}"
            L = self.layers[i]
            L.input_layernorm.weight = tc.tensor(
                np.ascontiguousarray(g(f"{p}.input_layernorm.weight")), dtype="float32")
            L.post_attention_layernorm.weight = tc.tensor(
                np.ascontiguousarray(g(f"{p}.post_attention_layernorm.weight")), dtype="float32")
            # INT4-quantized projections + FFN (bf16 4B weights are ~8GB — too big
            # for the 8GB card unquantized; INT4 fits like Mistral-7B). NO bias.
            # Optional KV-head mean-pooling (GQA-style 8 -> pool_kv heads): the
            # reverse of MHA->GQA conversion, UNHEALED (no fine-tune). Adjacent
            # head groups pool together so the q->kv grouping stays ordered.
            def _pool(w):
                pk = getattr(self, "_pool_kv", None)
                if not pk:
                    return w
                hd, nk = cfg.head_dim, 8           # source: 8 KV heads
                grp = nk // pk
                return w.reshape(nk, hd, -1).reshape(pk, grp, hd, -1).mean(axis=1).reshape(pk * hd, -1)
            L.self_attn.q_proj = QuantLinearTC(np.ascontiguousarray(g(f"{p}.self_attn.q_proj.weight")))
            L.self_attn.k_proj = QuantLinearTC(np.ascontiguousarray(_pool(g(f"{p}.self_attn.k_proj.weight"))))
            L.self_attn.v_proj = QuantLinearTC(np.ascontiguousarray(_pool(g(f"{p}.self_attn.v_proj.weight"))))
            L.self_attn.o_proj = QuantLinearTC(np.ascontiguousarray(g(f"{p}.self_attn.o_proj.weight")))
            # per-head qk-norm weights (head_dim,) — kept fp32, tiny
            L.self_attn._qk_w = tc.tensor(
                np.ascontiguousarray(g(f"{p}.self_attn.q_norm.weight")), dtype="float32")
            L.self_attn._k_qk_w = tc.tensor(
                np.ascontiguousarray(g(f"{p}.self_attn.k_norm.weight")), dtype="float32")
            L.self_attn._qk_eps = cfg.rms_norm_eps
            L.mlp.gate_proj = QuantLinearTC(np.ascontiguousarray(g(f"{p}.mlp.gate_proj.weight")))
            L.mlp.up_proj = QuantLinearTC(np.ascontiguousarray(g(f"{p}.mlp.up_proj.weight")))
            L.mlp.down_proj = QuantLinearTC(np.ascontiguousarray(g(f"{p}.mlp.down_proj.weight")))
            import gc; gc.collect()

        self.norm.weight = tc.tensor(
            np.ascontiguousarray(g("model.norm.weight")), dtype="float32")
        self.lm_head = _TiedLMHead(self.embed_tokens.weight)
        W.clear()
        return {"loaded": "bf16", "framework": "tensor_cuda bf16 (unquantized, Qwen3-4B)"}

    @classmethod
    def from_pretrained(cls, model_dir=None, attention_mode="standard", config=None,
                        pool_kv=None):
        if config is None:
            config = Qwen3Config()
        if pool_kv:
            # fewer KV heads at load (unhealed mean-pool); group size grows.
            config.num_kv_heads = pool_kv
        BlockTC.COMPUTE_DTYPE = "bfloat16"
        LinearTC.DTYPE = "bfloat16"
        with tc.no_grad():
            m = cls(config, attention_mode)
            m._pool_kv = pool_kv
            # wire the per-head qk-norm hook onto each attention module: it replaces
            # the OLMoE-style full-hidden q_norm/k_norm with per-head head_dim norm.
            for L in m.layers:
                att = L.self_attn
                att.q_norm = None    # disable the base (hidden-dim) qk-norm path
                att.k_norm = None
                att._use_qwen3_qknorm = True
            info = m.load_weights_fp16(model_dir)
        for L in m.layers:
            L.self_attn.bulk_bits = 8   # Qwen-family outlier keys (same as 2.5)
        return m, info
