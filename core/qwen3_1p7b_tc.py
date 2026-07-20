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

from core.mistral7b_tc import (Mistral7B_TC, LinearTC, QuantLinearTC, GROUP_SIZE,
                               BlockTC, F, tc)
from core.qwen_tc import _TiedLMHead


class QuantLinearINT6TC:
    """P3 INT6 group-quantized linear — SYMMETRIC per-group, engine-native.

    Mirrors the INT4 QuantLinearTC surface (packed uint8 + fp16 scales + a
    'zeros' tensor + group_size), but the INT6 kernel's native contract is
    SYMMETRIC: codes dequantize as (q - 32) * scale with an EMPTY fp16 zeros
    tensor selecting that grid (z = -32*s implicitly). This is the ONLY INT6
    variant per the order — no asymmetric INT6 (the INT4 path is asymmetric
    affine; the difference is recorded in the receipt, not "fixed").

    The INT6 ops (tc.int6_linear_fused / int6_linear / int6_dequant) exist ONLY
    in the Project-Tensor fork build (branch int6-weights). Quantization uses the
    fork's quantize_symmetric_per_group (imported at construction) so pack layout
    and the symmetric grid match the CUDA kernel exactly. Forward uses the fused
    packed path (GEMV at M==1 decode, tile at larger M) — no full fp16 weight
    transient, matching the fused INT4 decode rationale.
    """

    def __init__(self, weight_fp32: np.ndarray, group_size: int = GROUP_SIZE):
        from tensor_cuda.quantization import quantize_symmetric_per_group
        self.out_features, self.in_features = weight_fp32.shape
        self.group_size = group_size
        self.bits = 6
        q = quantize_symmetric_per_group(weight_fp32, 6, group_size)
        # empty fp16 zeros tensor selects the kernel's symmetric grid
        self._zeros_np = np.empty((0,), dtype=np.float16)
        self.packed = tc.tensor(q.packed, dtype="uint8")
        self.scales = tc.tensor(q.scales, dtype="float16")
        self.zeros = tc.tensor(self._zeros_np, dtype="float16")
        self._vram = q.packed.nbytes + q.scales.nbytes  # empty zeros = 0 bytes

    def __call__(self, x):
        return tc.int6_linear_fused(x, self.packed, self.scales, self.zeros,
                                    self.group_size)

    def vram_bytes(self) -> int:
        return self._vram


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

    def load_weights_int4(self, model_dir=None, group_size=GROUP_SIZE):
        """P2: INT4-quantize the projection + FFN matrices from the bf16 HF
        snapshot with the HOUSE stack (QuantLinearTC INT4 == the same machinery
        core/qwen3_tc.py's 4B adapter uses). Self-quantized from bf16 — NO GGUF
        import.

        HOUSE quant config (recorded in the receipt via int4_quant_config()):
          bits=4, group_size=128 (default), per-group AFFINE (asymmetric) with
          scale=(max-min)/15, zero-point=min, packed nibbles (uint8 N x K/2),
          scales/zeros fp16. This is _quantize_int4 in core/mistral7b_tc.py —
          byte-identical to QuantizedLinear; the exact scheme the 4B port loads.

        Tensors kept HIGHER precision (NOT INT4):
          - embed_tokens.weight: bf16 resident (also serves the tied head)
          - lm_head: TIED to the bf16 embedding (P1 resident decision; 0 extra
            VRAM). NOT quantized — matches P1's resident/host head decision.
          - all RMSNorm weights (input/post_attention/final norm): fp32
          - per-head qk-norm weights (q_norm/k_norm): fp32
        Quantized (INT4): every layer's q/k/v/o_proj and gate/up/down_proj.
        """
        import torch
        from safetensors.torch import load_file
        d = model_dir or _snap()
        W = {}
        for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
            W.update(load_file(f))

        def g(name):
            return np.ascontiguousarray(W.pop(name).to(torch.float32).numpy())

        cfg = self.config

        # Tied-head handling identical to the bf16 path: ONE resident bf16 copy
        # (the embedding); assert the on-disk tie so an untied checkpoint fails
        # loud rather than silently degrading.
        emb_t = W.pop("model.embed_tokens.weight")
        if "lm_head.weight" in W:
            lm_t = W.pop("lm_head.weight")
            if not torch.equal(emb_t, lm_t):
                md = (emb_t.float() - lm_t.float()).abs().max().item()
                raise RuntimeError(
                    f"Qwen3-1.7B expected TIED embeddings but lm_head.weight "
                    f"differs (max|diff|={md}); tied-head assumption violated.")
            del lm_t
        emb = emb_t.to(torch.float32).numpy().astype(np.float32)
        del emb_t
        self.embed_tokens.weight = tc.tensor(
            np.ascontiguousarray(emb)).astype("bfloat16")

        self._int4_group_size = group_size
        for i in range(cfg.num_layers):
            p = f"model.layers.{i}"
            L = self.layers[i]
            L.input_layernorm.weight = tc.tensor(
                g(f"{p}.input_layernorm.weight"), dtype="float32")
            L.post_attention_layernorm.weight = tc.tensor(
                g(f"{p}.post_attention_layernorm.weight"), dtype="float32")
            # INT4-quantized projections (QuantLinearTC, bits=4, group_size). NO
            # bias (Qwen3 attention_bias=false).
            L.self_attn.q_proj = QuantLinearTC(g(f"{p}.self_attn.q_proj.weight"),
                                               group_size=group_size, bits=4)
            L.self_attn.k_proj = QuantLinearTC(g(f"{p}.self_attn.k_proj.weight"),
                                               group_size=group_size, bits=4)
            L.self_attn.v_proj = QuantLinearTC(g(f"{p}.self_attn.v_proj.weight"),
                                               group_size=group_size, bits=4)
            L.self_attn.o_proj = QuantLinearTC(g(f"{p}.self_attn.o_proj.weight"),
                                               group_size=group_size, bits=4)
            # per-head qk-norm weights (head_dim,) — kept fp32, tiny
            L.self_attn._qk_w = tc.tensor(
                g(f"{p}.self_attn.q_norm.weight"), dtype="float32")
            L.self_attn._k_qk_w = tc.tensor(
                g(f"{p}.self_attn.k_norm.weight"), dtype="float32")
            L.self_attn._qk_eps = cfg.rms_norm_eps
            L.mlp.gate_proj = QuantLinearTC(g(f"{p}.mlp.gate_proj.weight"),
                                            group_size=group_size, bits=4)
            L.mlp.up_proj = QuantLinearTC(g(f"{p}.mlp.up_proj.weight"),
                                          group_size=group_size, bits=4)
            L.mlp.down_proj = QuantLinearTC(g(f"{p}.mlp.down_proj.weight"),
                                            group_size=group_size, bits=4)
            import gc; gc.collect()

        self.norm.weight = tc.tensor(g("model.norm.weight"), dtype="float32")
        # TIED head: reuse the bf16 embedding; 0 extra VRAM (P1 decision).
        self.lm_head = _TiedLMHead(self.embed_tokens.weight)
        W.clear()
        return {"loaded": "int4",
                "framework": "tensor_cuda INT4 (self-quantized, Qwen3-1.7B)",
                "bits": 4, "group_size": group_size,
                "quant_scheme": "affine per-group (asymmetric), zero=min, "
                                "scale=(max-min)/15, packed nibbles",
                "tied_head": True,
                "tied_head_vram_bytes": self.lm_head.vram_bytes(),
                "exempt_tensors": ["embed_tokens(bf16)", "lm_head(tied bf16)",
                                   "all_rmsnorm(fp32)", "qk_norm(fp32)"]}

    def load_weights_int6(self, model_dir=None, group_size=GROUP_SIZE):
        """P3: INT6-quantize the projection + FFN matrices from the bf16 HF
        snapshot with the FORK engine's SYMMETRIC per-group quantizer
        (quantize_symmetric_per_group, bits=6). Self-quantized from bf16.

        INT6 quant contract (recorded in the receipt):
          bits=6, group_size=128, per-group SYMMETRIC — codes dequantize as
          (q - 32) * scale, EMPTY fp16 zeros tensor (z = -32*s implicit). 4 codes
          packed per 3 bytes. This DIFFERS from the P2 INT4 path (asymmetric
          affine, zero=group-min, nibble-packed): the difference is the INT6
          kernel's native contract, matched to Sol's gates, NOT a regression.

        Tensors kept HIGHER precision (NOT INT6), identical to the INT4 path:
          embed_tokens (bf16, serves tied head), lm_head (tied bf16, 0 VRAM),
          all RMSNorm weights (fp32), per-head qk-norm (fp32).
        Quantized (INT6): every layer's q/k/v/o_proj and gate/up/down_proj.
        """
        import torch
        from safetensors.torch import load_file
        d = model_dir or _snap()
        W = {}
        for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
            W.update(load_file(f))

        def g(name):
            return np.ascontiguousarray(W.pop(name).to(torch.float32).numpy())

        cfg = self.config

        # Tied-head handling identical to the bf16/INT4 paths: ONE resident bf16
        # copy (the embedding); assert the on-disk tie so an untied checkpoint
        # fails loud rather than silently degrading.
        emb_t = W.pop("model.embed_tokens.weight")
        if "lm_head.weight" in W:
            lm_t = W.pop("lm_head.weight")
            if not torch.equal(emb_t, lm_t):
                md = (emb_t.float() - lm_t.float()).abs().max().item()
                raise RuntimeError(
                    f"Qwen3-1.7B expected TIED embeddings but lm_head.weight "
                    f"differs (max|diff|={md}); tied-head assumption violated.")
            del lm_t
        emb = emb_t.to(torch.float32).numpy().astype(np.float32)
        del emb_t
        self.embed_tokens.weight = tc.tensor(
            np.ascontiguousarray(emb)).astype("bfloat16")

        self._int6_group_size = group_size
        for i in range(cfg.num_layers):
            p = f"model.layers.{i}"
            L = self.layers[i]
            L.input_layernorm.weight = tc.tensor(
                g(f"{p}.input_layernorm.weight"), dtype="float32")
            L.post_attention_layernorm.weight = tc.tensor(
                g(f"{p}.post_attention_layernorm.weight"), dtype="float32")
            # INT6-quantized projections (QuantLinearINT6TC, symmetric g128). NO
            # bias (Qwen3 attention_bias=false).
            L.self_attn.q_proj = QuantLinearINT6TC(g(f"{p}.self_attn.q_proj.weight"),
                                                   group_size=group_size)
            L.self_attn.k_proj = QuantLinearINT6TC(g(f"{p}.self_attn.k_proj.weight"),
                                                   group_size=group_size)
            L.self_attn.v_proj = QuantLinearINT6TC(g(f"{p}.self_attn.v_proj.weight"),
                                                   group_size=group_size)
            L.self_attn.o_proj = QuantLinearINT6TC(g(f"{p}.self_attn.o_proj.weight"),
                                                   group_size=group_size)
            L.self_attn._qk_w = tc.tensor(
                g(f"{p}.self_attn.q_norm.weight"), dtype="float32")
            L.self_attn._k_qk_w = tc.tensor(
                g(f"{p}.self_attn.k_norm.weight"), dtype="float32")
            L.self_attn._qk_eps = cfg.rms_norm_eps
            L.mlp.gate_proj = QuantLinearINT6TC(g(f"{p}.mlp.gate_proj.weight"),
                                                group_size=group_size)
            L.mlp.up_proj = QuantLinearINT6TC(g(f"{p}.mlp.up_proj.weight"),
                                              group_size=group_size)
            L.mlp.down_proj = QuantLinearINT6TC(g(f"{p}.mlp.down_proj.weight"),
                                                group_size=group_size)
            import gc; gc.collect()

        self.norm.weight = tc.tensor(g("model.norm.weight"), dtype="float32")
        # TIED head: reuse the bf16 embedding; 0 extra VRAM (P1 decision).
        self.lm_head = _TiedLMHead(self.embed_tokens.weight)
        W.clear()
        return {"loaded": "int6",
                "framework": "tensor_cuda INT6 (self-quantized, Qwen3-1.7B, "
                             "fork int6-weights)",
                "bits": 6, "group_size": group_size,
                "quant_scheme": "symmetric per-group, code=(q-32), z=-32*s "
                                "(empty zeros), 4-codes-per-3-bytes",
                "symmetry": "symmetric (NB: INT4 P2 path is asymmetric affine)",
                "tied_head": True,
                "tied_head_vram_bytes": self.lm_head.vram_bytes(),
                "exempt_tensors": ["embed_tokens(bf16)", "lm_head(tied bf16)",
                                   "all_rmsnorm(fp32)", "qk_norm(fp32)"]}

    def measure_vram_int6(self):
        """INT6 resident VRAM accounting. Projections are QuantLinearINT6TC
        (packed INT6 4-per-3-byte + fp16 scales, empty zeros); embedding bf16;
        norms fp32; head tied (0)."""
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
            "projections_int6_mb": proj / 1024**2,
            "norms_mb": norm / 1024**2,
            "lm_head_mb": lm / 1024**2,
            "lm_head_tied": True,
            "group_size": getattr(self, "_int6_group_size", GROUP_SIZE),
            "total_mb": total / 1024**2,
        }

    def measure_vram_int4(self):
        """INT4 resident VRAM accounting. Projections are QuantLinearTC (packed
        INT4 + fp16 scales/zeros); embedding bf16; norms fp32; head tied (0)."""
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
            "projections_int4_mb": proj / 1024**2,
            "norms_mb": norm / 1024**2,
            "lm_head_mb": lm / 1024**2,
            "lm_head_tied": True,
            "group_size": getattr(self, "_int4_group_size", GROUP_SIZE),
            "total_mb": total / 1024**2,
        }

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
    def from_pretrained(cls, model_dir=None, attention_mode="standard", config=None,
                        int4=False, int6=False, group_size=GROUP_SIZE):
        """P1 default: bf16 (int4=int6=False). P2: int4=True self-quantizes the
        projection/FFN matrices to INT4 (house QuantLinearTC path, asymmetric).
        P3: int6=True self-quantizes to INT6 (fork QuantLinearINT6TC path,
        SYMMETRIC — needs the fork engine build on PYTHONPATH for tc.int6_*).
        The residual stream / RoPE / KV cache / embedding / head stay bf16 in
        every case; only the projection weights change format."""
        if int4 and int6:
            raise ValueError("int4 and int6 are mutually exclusive")
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
            if int6:
                info = m.load_weights_int6(model_dir, group_size=group_size)
            elif int4:
                info = m.load_weights_int4(model_dir, group_size=group_size)
            else:
                info = m.load_weights_fp16(model_dir)
        # Qwen-family outlier keys: 2-bit bulk destroys them; 8-bit bulk recovers
        # (qwen_tc / qwen3_tc precedent). Applies to the apa_selective path's
        # key quantization; the pure `apa` path uses the engine's default
        # bulk_bits=2 z-score branch (see mistral7b_tc GQAAttentionTC).
        for L in m.layers:
            L.self_attn.bulk_bits = 8
        return m, info
