"""Qwen3-1.7B bf16 (unquantized) on tensor_cuda — NC17 name-checker track (P1).

Config variant of core/qwen3_tc.py (Qwen3-4B) with ONE structural delta:
TIED EMBEDDINGS (`tie_word_embeddings: true`; the 4B config is untied). Handled
explicitly below via the resident `_TiedLMHead` — see the tied-head note.

Deltas vs Qwen3-4B (all config-driven, none structural except the tied head):
  4B : 36L, 32Q/8KV, hidden 2560, inter 9728, untied lm_head
  1.7B: 28L, 16Q/8KV, hidden 2048, inter 6144, tied lm_head
Same family: head_dim 128, vocab 151936, RoPE theta 1e6, per-head qk-norm,
attention_bias=false (NO q/k/v bias), rms_eps 1e-6, bf16-native, NO sliding.

WEIGHTS: bf16 full precision (LinearTC, DTYPE=bfloat16) — this is the P1
"APA full-precision" stage. NOT the INT4 QuantLinearTC path the 4B adapter's
load_weights_fp16 uses; P2 is the INT4 stage. Embedding + lm_head kept bf16.

TIED-HEAD DECISION (recorded in the receipt / this docstring):
  On disk the HF snapshot materializes BOTH `model.embed_tokens.weight` and
  `lm_head.weight` as separate tensors, but they are bit-identical (verified:
  torch.equal == True, max|diff| == 0.0). Because they are tied we keep ONE
  resident copy — the embedding — and route the LM head through `_TiedLMHead`,
  which does y = x @ W_embed^T via cuBLAS OP_T (trans_b) with NO materialized
  transpose and NO second weight tensor. Resident-vs-host: RESIDENT (the shared
  embedding is already resident for the input gather). Measured VRAM COST of the
  tied head: 0 bytes (`_TiedLMHead.vram_bytes() == 0`). Loading lm_head.weight
  as a separate resident tensor would have cost vocab*hidden*2 =
  151936*2048*2 = 622,329,856 B ~= 593.5 MiB of otherwise-redundant VRAM.
"""
import os
import glob
import numpy as np

from core.mistral7b_tc import (Mistral7B_TC, LinearTC, BlockTC, F, tc)
from core.qwen_tc import _TiedLMHead


class Qwen3_1p7bConfig:
    vocab_size = 151936
    hidden_dim = 2048
    intermediate_dim = 6144
    num_layers = 28
    num_heads = 16
    num_kv_heads = 8
    head_dim = 128                    # 16*128 = 2048 == hidden (square q_proj here)
    max_position_embeddings = 40960
    rope_theta = 1000000.0
    rms_norm_eps = 1e-6

    @property
    def num_heads_per_kv(self):
        return self.num_heads // self.num_kv_heads


_SNAP_GLOB = "/home/vader/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/*"


def _snap():
    hits = sorted(glob.glob(_SNAP_GLOB))
    if not hits:
        raise FileNotFoundError("Qwen3-1.7B snapshot not found in HF cache")
    return hits[-1]


class Qwen3_1p7b_TC(Mistral7B_TC):
    def load_weights_fp16(self, model_dir=None):
        """Load bf16 weights (full precision). LinearTC.DTYPE must be bf16
        (set in from_pretrained before construction)."""
        import torch
        from safetensors.torch import load_file
        d = model_dir or _snap()
        W = {}
        for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
            W.update(load_file(f))

        def g(name):
            return W.pop(name).to(torch.float32).numpy()

        cfg = self.config

        # Tied-head handling: the embedding is the ONE resident copy; the
        # lm_head reuses it. We assert the on-disk tie (bit-identical) so a
        # future untied checkpoint cannot silently degrade correctness.
        emb_t = W.pop("model.embed_tokens.weight")
        if "lm_head.weight" in W:
            lm_t = W.pop("lm_head.weight")
            if not torch.equal(emb_t, lm_t):
                md = (emb_t.float() - lm_t.float()).abs().max().item()
                raise RuntimeError(
                    f"Qwen3-1.7B expected TIED embeddings but lm_head.weight "
                    f"differs from embed_tokens.weight (max|diff|={md}); tied "
                    f"head assumption violated — do not use _TiedLMHead.")
            del lm_t
        emb = emb_t.to(torch.float32).numpy().astype(np.float32)
        del emb_t
        self.embed_tokens.weight = tc.tensor(
            np.ascontiguousarray(emb)).astype("bfloat16")

        for i in range(cfg.num_layers):
            p = f"model.layers.{i}"
            L = self.layers[i]
            L.input_layernorm.weight = tc.tensor(
                np.ascontiguousarray(g(f"{p}.input_layernorm.weight")), dtype="float32")
            L.post_attention_layernorm.weight = tc.tensor(
                np.ascontiguousarray(g(f"{p}.post_attention_layernorm.weight")), dtype="float32")
            # bf16 full-precision projections (LinearTC), NO bias (Qwen3 family:
            # attention_bias=false — unlike the Qwen2.5 adapter which had qkv bias).
            L.self_attn.q_proj = LinearTC(g(f"{p}.self_attn.q_proj.weight"))
            L.self_attn.k_proj = LinearTC(g(f"{p}.self_attn.k_proj.weight"))
            L.self_attn.v_proj = LinearTC(g(f"{p}.self_attn.v_proj.weight"))
            L.self_attn.o_proj = LinearTC(g(f"{p}.self_attn.o_proj.weight"))
            # per-head qk-norm weights (head_dim,) — kept fp32, tiny
            L.self_attn._qk_w = tc.tensor(
                np.ascontiguousarray(g(f"{p}.self_attn.q_norm.weight")), dtype="float32")
            L.self_attn._k_qk_w = tc.tensor(
                np.ascontiguousarray(g(f"{p}.self_attn.k_norm.weight")), dtype="float32")
            L.self_attn._qk_eps = cfg.rms_norm_eps
            L.mlp.gate_proj = LinearTC(g(f"{p}.mlp.gate_proj.weight"))
            L.mlp.up_proj = LinearTC(g(f"{p}.mlp.up_proj.weight"))
            L.mlp.down_proj = LinearTC(g(f"{p}.mlp.down_proj.weight"))
            import gc; gc.collect()

        self.norm.weight = tc.tensor(
            np.ascontiguousarray(g("model.norm.weight")), dtype="float32")
        # TIED head: reuse the (bf16) embedding weight; 0 extra VRAM.
        self.lm_head = _TiedLMHead(self.embed_tokens.weight)
        W.clear()
        return {"loaded": "bf16",
                "framework": "tensor_cuda bf16 (unquantized, Qwen3-1.7B)",
                "tied_head": True,
                "tied_head_vram_bytes": self.lm_head.vram_bytes()}

    def measure_vram_bf16(self):
        """bf16 resident VRAM accounting (measure_vram in the base assumes
        QuantLinearTC projections; here projections are bf16 LinearTC)."""
        embed = self.embed_tokens.weight.numpy().nbytes
        proj = 0
        norm = 0
        for layer in self.layers:
            for pj in (layer.self_attn.q_proj, layer.self_attn.k_proj,
                       layer.self_attn.v_proj, layer.self_attn.o_proj,
                       layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj):
                proj += pj.vram_bytes()
            norm += layer.input_layernorm.weight.numpy().nbytes
            norm += layer.post_attention_layernorm.weight.numpy().nbytes
            norm += layer.self_attn._qk_w.numpy().nbytes
            norm += layer.self_attn._k_qk_w.numpy().nbytes
        norm += self.norm.weight.numpy().nbytes
        lm = self.lm_head.vram_bytes()  # tied -> 0
        total = embed + proj + norm + lm
        return {
            "embedding_mb": embed / 1024**2,
            "projections_mb": proj / 1024**2,
            "norms_mb": norm / 1024**2,
            "lm_head_mb": lm / 1024**2,
            "lm_head_tied": True,
            "total_mb": total / 1024**2,
        }

    @classmethod
    def from_pretrained(cls, model_dir=None, attention_mode="standard", config=None):
        if config is None:
            config = Qwen3_1p7bConfig()
        # Qwen3 activations use bf16 (fp32 exponent range). Set BEFORE building so
        # linears, residual stream, RoPE, and KV cache all use bf16.
        BlockTC.COMPUTE_DTYPE = "bfloat16"
        LinearTC.DTYPE = "bfloat16"
        with tc.no_grad():
            m = cls(config, attention_mode)
            # wire the per-head qk-norm hook onto each attention module (Qwen3
            # per-head head_dim norm; disable the base hidden-dim q_norm/k_norm).
            for L in m.layers:
                att = L.self_attn
                att.q_norm = None
                att.k_norm = None
                att._use_qwen3_qknorm = True
            info = m.load_weights_fp16(model_dir)
        # Qwen-family outlier keys: 2-bit bulk destroys them; 8-bit bulk recovers
        # (qwen_tc / qwen3_tc precedent). Applies to the apa_selective path's
        # key quantization; the pure `apa` path uses the engine's default
        # bulk_bits=2 z-score branch (see mistral7b_tc GQAAttentionTC).
        for L in m.layers:
            L.self_attn.bulk_bits = 8
        return m, info
