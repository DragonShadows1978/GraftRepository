import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensor_cuda as tc  # noqa: E402
from tensor_cuda.quantization import dequantize_affine_per_group  # noqa: E402

from core.gpt_oss20b_tc import (  # noqa: E402
    BiasedQuantLinearTC,
    FP4_VALUES,
    GptOss20B_TC,
    GptOss20BConfig,
    GptOssAttentionTC,
    gpt_oss_expert_activation_np,
    gpt_oss_expert_activation_tc,
    gpt_oss_grm_dialect_kwargs,
    mxfp4_dequantize_blocks_np,
    resolve_gpt_oss_attention_mode,
    sink_apa_blend_attention_tc,
    sink_attention_tc,
    sliding_sink_attention_tc,
)
from core.mistral7b_tc import BlockTC  # noqa: E402
from core.mistral7b_tc import F  # noqa: E402
from core.mistral7b_tc import QuantLinearTC  # noqa: E402


GPT_OSS_CONFIG = Path(
    "/home/vader/.cache/huggingface/hub/models--openai--gpt-oss-20b/"
    "snapshots/6cee5e81ee83917806bbde320786a8fb61efebee/config.json"
)


def _require_tensor_cuda():
    try:
        tc.tensor(np.zeros((1,), dtype=np.float32))
    except RuntimeError as exc:
        pytest.skip(f"TensorCUDA device unavailable: {exc}")


def test_gpt_oss_config_parses_pinned_metadata():
    if not GPT_OSS_CONFIG.exists():
        pytest.skip("pinned GPT-OSS metadata is not present in HF cache")
    cfg = GptOss20BConfig.from_model_dir(GPT_OSS_CONFIG.parent)

    assert cfg.repository == "openai/gpt-oss-20b"
    assert cfg.revision == "6cee5e81ee83917806bbde320786a8fb61efebee"
    assert cfg.hidden_dim == 2880
    assert cfg.num_layers == 24
    assert cfg.num_heads == 64
    assert cfg.num_kv_heads == 8
    assert cfg.head_dim == 64
    assert cfg.num_heads_per_kv == 8
    assert cfg.num_local_experts == 32
    assert cfg.num_experts_per_tok == 4
    assert cfg.full_attention_indices() == list(range(1, 24, 2))
    assert cfg.sliding_attention_indices() == list(range(0, 24, 2))
    assert cfg.rope_scaling["rope_type"] == "yarn"
    assert cfg.group_size == 64


def test_resolve_gpt_oss_attention_mode_scopes_apa_to_full_layers():
    cfg = GptOss20BConfig(
        num_layers=2,
        layer_types=("sliding_attention", "full_attention"),
    )

    sliding = resolve_gpt_oss_attention_mode(
        cfg, 0, "apa_selective", apa_layer_scope="full"
    )
    full = resolve_gpt_oss_attention_mode(
        cfg, 1, "apa_selective", apa_layer_scope="full"
    )
    all_scope = resolve_gpt_oss_attention_mode(
        cfg, 0, "apa_selective", apa_layer_scope="all"
    )
    standard = resolve_gpt_oss_attention_mode(
        cfg, 1, "standard", apa_layer_scope="all"
    )

    assert sliding["effective_attention_mode"] == "standard"
    assert sliding["apa_active"] is False
    assert sliding["apa_skip_reason"] == "sliding_window_bounded"
    assert full["effective_attention_mode"] == "apa_selective"
    assert full["apa_active"] is True
    assert full["apa_skip_reason"] is None
    assert all_scope["effective_attention_mode"] == "apa_selective"
    assert all_scope["apa_active"] is True
    assert standard["effective_attention_mode"] == "standard"
    assert standard["apa_skip_reason"] == "requested_standard"


def test_gpt_oss_grm_dialect_uses_native_gqa_remount_profile():
    cfg = GptOss20BConfig(
        hidden_dim=2880,
        num_layers=24,
        num_heads=64,
        num_kv_heads=8,
        head_dim=64,
        layer_types=tuple(
            "sliding_attention" if i % 2 == 0 else "full_attention"
            for i in range(24)
        ),
    )

    got = gpt_oss_grm_dialect_kwargs(cfg)

    assert got["model_type"] == "GptOss20B_TC"
    assert got["payload_kind"] == "gqa"
    assert got["position_law"] == "rope_full_yarn"
    assert got["state_kind"] == "kv"
    assert got["graftability"] == "seat_remountable"
    assert got["remountable"] is True
    assert got["composition"] == "multi_mount"
    assert got["route_layer"] == 1
    assert got["num_kv_heads"] == 8
    assert got["head_dim"] == 64
    assert got["vals_per_tok_layer"] == 1024


class _FixedProjection:
    def __init__(self, array):
        self.array = np.ascontiguousarray(array, dtype=np.float32)

    def __call__(self, _x):
        return tc.tensor(self.array, dtype="float32")


class _IdentityProjection:
    def __call__(self, x):
        return x


def test_gpt_oss_attention_captures_prerope_and_seats_graft_prefix():
    _require_tensor_cuda()
    cfg = GptOss20BConfig(
        hidden_dim=4,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        head_dim=2,
        layer_types=("full_attention",),
    )
    att = GptOssAttentionTC(cfg, 0)

    q_np = np.asarray([[[0.20, -0.10, 0.05, 0.30], [0.01, 0.10, -0.20, 0.05]]])
    k_np = np.asarray([[[0.25, 0.75], [-0.50, 0.10]]])
    v_np = np.asarray([[[0.30, -0.20], [0.40, 0.10]]])
    kg_np = np.asarray([[[[0.60, -0.40]]]], dtype=np.float32)
    vg_np = np.asarray([[[[0.70, 0.80]]]], dtype=np.float32)

    att.q_proj = _FixedProjection(q_np)
    att.k_proj = _FixedProjection(k_np)
    att.v_proj = _FixedProjection(v_np)
    att.o_proj = _IdentityProjection()
    att.sinks = tc.tensor(np.zeros((cfg.num_heads,), dtype=np.float32))
    att._capture = True
    att._capture_q = True
    att.inject_kv = (tc.tensor(kg_np), tc.tensor(vg_np), 1.0)
    att.graft_seats = int(kg_np.shape[2])

    angles = np.arange(6, dtype=np.float32)[:, None] * np.asarray(
        [[0.25, 0.25]], dtype=np.float32
    )
    cos = tc.tensor(np.cos(angles).astype(np.float32))
    sin = tc.tensor(np.sin(angles).astype(np.float32))
    x = tc.tensor(np.zeros((1, 2, cfg.hidden_dim), dtype=np.float32))

    _, kv = att(x, cos, sin, position_offset=0, kv_cache=None)

    np.testing.assert_allclose(att._captured[0], k_np.reshape(1, 2, 1, 2).transpose(0, 2, 1, 3))
    np.testing.assert_allclose(att._captured[1], v_np.reshape(1, 2, 1, 2).transpose(0, 2, 1, 3))
    np.testing.assert_allclose(att._captured_q, q_np.reshape(1, 2, 2, 2).transpose(0, 2, 1, 3))

    kg_expected = F.apply_rotary(
        tc.tensor(kg_np), cos.slice(0, 0, 1), sin.slice(0, 0, 1)
    ).numpy()
    live_expected = F.apply_rotary(
        tc.tensor(k_np.reshape(1, 2, 1, 2).transpose(0, 2, 1, 3)),
        cos.slice(0, 1, 2),
        sin.slice(0, 1, 2),
    ).numpy()

    got_k = kv[0].numpy()
    got_v = kv[1].numpy()
    np.testing.assert_allclose(got_k[:, :, :1], kg_expected, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(got_k[:, :, 1:], live_expected, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(got_v[:, :, :1], vg_np, rtol=2e-5, atol=2e-5)


def test_gpt_oss_attention_does_not_reinject_graft_on_cached_decode():
    _require_tensor_cuda()
    cfg = GptOss20BConfig(
        hidden_dim=4,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        head_dim=2,
        layer_types=("full_attention",),
    )
    att = GptOssAttentionTC(cfg, 0)
    q_np = np.asarray([[[0.20, -0.10, 0.05, 0.30]]], dtype=np.float32)
    k_np = np.asarray([[[0.25, 0.75]]], dtype=np.float32)
    v_np = np.asarray([[[0.30, -0.20]]], dtype=np.float32)
    kg_np = np.asarray([[[[0.60, -0.40]]]], dtype=np.float32)
    vg_np = np.asarray([[[[0.70, 0.80]]]], dtype=np.float32)
    cache_k = tc.tensor(np.asarray([[[[0.11, 0.12], [0.21, 0.22]]]], dtype=np.float32))
    cache_v = tc.tensor(np.asarray([[[[0.31, 0.32], [0.41, 0.42]]]], dtype=np.float32))

    att.q_proj = _FixedProjection(q_np)
    att.k_proj = _FixedProjection(k_np)
    att.v_proj = _FixedProjection(v_np)
    att.o_proj = _IdentityProjection()
    att.sinks = tc.tensor(np.zeros((cfg.num_heads,), dtype=np.float32))
    att.inject_kv = (tc.tensor(kg_np), tc.tensor(vg_np), 1.0)
    att.graft_seats = int(kg_np.shape[2])

    angles = np.arange(6, dtype=np.float32)[:, None] * np.asarray(
        [[0.25, 0.25]], dtype=np.float32
    )
    cos = tc.tensor(np.cos(angles).astype(np.float32))
    sin = tc.tensor(np.sin(angles).astype(np.float32))
    x = tc.tensor(np.zeros((1, 1, cfg.hidden_dim), dtype=np.float32))

    _, kv = att(x, cos, sin, position_offset=1, kv_cache=(cache_k, cache_v))

    assert kv[0].shape[2] == 3
    np.testing.assert_allclose(kv[0].slice(2, 0, 2).numpy(), cache_k.numpy())
    np.testing.assert_allclose(kv[1].slice(2, 0, 2).numpy(), cache_v.numpy())


def _mxfp4_reference(blocks, scales):
    b = np.asarray(blocks, dtype=np.uint8)
    s = np.asarray(scales, dtype=np.uint8).astype(np.int32) - 127
    vals = np.empty(b.shape[:-1] + (b.shape[-1] * 2,), dtype=np.float32)
    vals[..., 0::2] = FP4_VALUES[b & 0x0F]
    vals[..., 1::2] = FP4_VALUES[b >> 4]
    vals = np.ldexp(vals, s[..., None])
    e, o, g, expanded = vals.shape
    return vals.reshape(e, o, g * expanded).swapaxes(1, 2)


def test_mxfp4_dequantize_blocks_matches_reference_layout():
    rng = np.random.default_rng(20260706)
    blocks = rng.integers(0, 256, size=(2, 3, 4, 5), dtype=np.uint8)
    scales = rng.integers(123, 132, size=(2, 3, 4), dtype=np.uint8)

    got = mxfp4_dequantize_blocks_np(blocks, scales)
    ref = _mxfp4_reference(blocks, scales)

    assert got.shape == (2, 40, 3)
    np.testing.assert_array_equal(got, ref)


def test_gpt_oss_expert_activation_np_matches_formula():
    gate_up = np.asarray(
        [[[-10.0, -9.0, -1.0, 0.0, 0.5, 2.0, 9.0, 9.0]]],
        dtype=np.float32,
    )
    got = gpt_oss_expert_activation_np(gate_up)

    gate = np.minimum(gate_up[..., 0::2], 7.0)
    up = np.clip(gate_up[..., 1::2], -7.0, 7.0)
    ref = (up + 1.0) * gate * (1.0 / (1.0 + np.exp(-(gate * 1.702))))

    np.testing.assert_allclose(got, ref, rtol=0.0, atol=1e-7)


def test_gpt_oss_expert_activation_tc_matches_numpy():
    rng = np.random.default_rng(7)
    gate_up = rng.normal(size=(2, 3, 12)).astype(np.float32)

    got = gpt_oss_expert_activation_tc(tc.tensor(gate_up, dtype="float32")).numpy()
    ref = gpt_oss_expert_activation_np(gate_up)

    np.testing.assert_allclose(got, ref, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("bits", [4, 3])
def test_biased_quant_linear_matches_dequant_reference(bits):
    rng = np.random.default_rng(300 + bits)
    old_dtype = BlockTC.COMPUTE_DTYPE
    BlockTC.COMPUTE_DTYPE = "float16"
    try:
        n, k = 40, 256
        x_np = rng.standard_normal((4, k), dtype=np.float32) * 0.2
        w_np = rng.standard_normal((n, k), dtype=np.float32) * 0.05
        b_np = rng.standard_normal(n, dtype=np.float32) * 0.01

        layer = BiasedQuantLinearTC(w_np, b_np, group_size=64, bits=bits)
        y = layer(tc.tensor(x_np, dtype="float32")).numpy().astype(np.float32)

        w = dequantize_affine_per_group(
            layer.packed.numpy(),
            layer.scales.numpy(),
            layer.zeros.numpy(),
            layer.bits,
            layer.in_features,
            layer.group_size,
        )
        ref = x_np.astype(np.float32) @ w.T.astype(np.float32) + b_np
    finally:
        BlockTC.COMPUTE_DTYPE = old_dtype

    np.testing.assert_allclose(y, ref, rtol=5e-4, atol=5e-4)


def test_sink_attention_matches_numpy_reference():
    rng = np.random.default_rng(42)
    q = rng.normal(size=(1, 4, 2, 3)).astype(np.float32)
    k = rng.normal(size=(1, 2, 5, 3)).astype(np.float32)
    v = rng.normal(size=(1, 2, 5, 3)).astype(np.float32)
    sinks = rng.normal(size=(4,)).astype(np.float32)
    scale = 0.57735026919

    got = sink_attention_tc(
        tc.tensor(q, dtype="float32"),
        tc.tensor(k, dtype="float32"),
        tc.tensor(v, dtype="float32"),
        tc.tensor(sinks, dtype="float32"),
        scale=scale,
        num_heads_per_kv=2,
    ).numpy()

    k_rep = np.repeat(k, 2, axis=1)
    v_rep = np.repeat(v, 2, axis=1)
    scores = np.einsum("bhld,bhsd->bhls", q, k_rep) * scale
    combined = np.concatenate(
        [scores, np.broadcast_to(sinks[None, :, None, None], (1, 4, 2, 1))],
        axis=-1,
    )
    shifted = combined - combined.max(axis=-1, keepdims=True)
    probs = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)
    ref = np.einsum("bhls,bhsd->bhld", probs[..., :-1], v_rep)

    np.testing.assert_allclose(got, ref, rtol=2e-5, atol=2e-5)


def test_sliding_sink_attention_matches_full_mask_reference():
    rng = np.random.default_rng(20260707)
    B, H, KVH, L, S, D, VD = 1, 4, 2, 6, 11, 3, 5
    q = rng.normal(size=(B, H, L, D)).astype(np.float32)
    k = rng.normal(size=(B, KVH, S, D)).astype(np.float32)
    v = rng.normal(size=(B, KVH, S, VD)).astype(np.float32)
    sinks = rng.normal(size=(H,)).astype(np.float32)
    scale = D ** -0.5
    window = 4

    got = sliding_sink_attention_tc(
        tc.tensor(q, dtype="float32"),
        tc.tensor(k, dtype="float32"),
        tc.tensor(v, dtype="float32"),
        tc.tensor(sinks, dtype="float32"),
        scale=scale,
        sliding_window=window,
        num_heads_per_kv=H // KVH,
        attn_block=2,
    ).numpy()

    k_rep = np.repeat(k, H // KVH, axis=1)
    v_rep = np.repeat(v, H // KVH, axis=1)
    scores = np.einsum("bhld,bhsd->bhls", q, k_rep) * scale
    q_abs = np.arange(S - L, S, dtype=np.int64)[:, None]
    k_abs = np.arange(S, dtype=np.int64)[None, :]
    allowed = (k_abs <= q_abs) & (k_abs > (q_abs - window))
    scores = scores + np.where(allowed, 0.0, -1.0e4).astype(np.float32)[None, None]
    sink_logits = np.broadcast_to(sinks[None, :, None, None], (B, H, L, 1))
    combined = np.concatenate([scores, sink_logits], axis=-1)
    weights = np.exp(combined - combined.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    ref = np.einsum("bhls,bhsd->bhld", weights[..., :-1], v_rep)

    np.testing.assert_allclose(got, ref, rtol=2e-5, atol=2e-5)


def _sink_apa_ref(q, k, kq, v, sinks, mask, scale, zthr):
    B, H, L, D = q.shape
    KVH, S, VD = k.shape[1], k.shape[2], v.shape[3]
    group = H // KVH
    out = np.zeros((B, H, L, VD), dtype=np.float32)
    for b in range(B):
        for h in range(H):
            kh = h // group
            for i in range(L):
                bulk = (kq[b, kh] @ q[b, h, i]) * scale + mask[0, 0, i]
                rank = (k[b, kh] @ q[b, h, i]) * scale + mask[0, 0, i]
                valid = bulk > -5e3
                a = np.abs(bulk[valid])
                thr = a.mean() + zthr * np.sqrt(max(a.var(), 0.0))
                score = np.where(valid & (np.abs(bulk) >= thr), rank, bulk)
                combined = np.concatenate([score, [sinks[h]]])
                weights = np.exp(combined - combined.max())
                weights /= weights.sum()
                out[b, h, i] = weights[:-1] @ v[b, kh]
    return out


def test_sink_apa_blend_attention_matches_numpy_reference():
    rng = np.random.default_rng(20260706)
    B, H, KVH, L, S, D, VD = 1, 4, 2, 3, 7, 5, 3
    q = (rng.standard_normal((B, H, L, D)) * 0.2).astype(np.float32)
    k = (rng.standard_normal((B, KVH, S, D)) * 0.2).astype(np.float32)
    kq = (k + rng.standard_normal((B, KVH, S, D)) * 0.03).astype(np.float32)
    v = (rng.standard_normal((B, KVH, S, VD)) * 0.2).astype(np.float32)
    sinks = rng.standard_normal(H).astype(np.float32) * 0.2
    mask = np.zeros((1, 1, L, S), dtype=np.float32)
    mask[:, :, 0, 4:] = -1e4
    mask[:, :, 1, 5:] = -1e4
    scale = D ** -0.5
    zthr = 0.25

    got = sink_apa_blend_attention_tc(
        tc.tensor(q),
        tc.tensor(k),
        tc.tensor(kq),
        tc.tensor(v),
        tc.tensor(sinks),
        scale=scale,
        zthr=zthr,
        attention_mask=tc.tensor(mask),
        num_heads_per_kv=H // KVH,
        attn_block=2,
    ).numpy()
    ref = _sink_apa_ref(q, k, kq, v, sinks, mask, scale, zthr)

    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, rtol=3e-5, atol=3e-5)


# --------------------------------------------------------------------------
# GptOss20B_TC: the full-model class (docs/GRM_E2E_RECEIPT_LEDGER.md P1
# prerequisite). Tests here use a tiny synthetic config/weights (no real
# snapshot) so they run fast and do not require the HF cache; the real
# 24-layer weight parity against stream_forward_smoke.py is a separate,
# GPU-bounded manual receipt (not a unit test — real weights are ~13GB).
# --------------------------------------------------------------------------

def _tiny_gpt_oss_config(**overrides):
    base = dict(
        vocab_size=32,
        hidden_dim=32,
        intermediate_dim=32,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=4,
        num_local_experts=2,
        num_experts_per_tok=1,
        sliding_window=4,
        layer_types=("full_attention", "sliding_attention"),
        group_size=4,
        initial_context_length=64,
    )
    base.update(overrides)
    return GptOss20BConfig(**base)


class _FakeEmbed:
    """Deterministic host row-gather standing in for GptOssRowEmbedding —
    avoids needing a real safetensors snapshot for these tiny-model tests."""

    def __init__(self, vocab, hidden, seed=0):
        rng = np.random.default_rng(seed)
        self.table = (rng.standard_normal((vocab, hidden)) * 0.05).astype(np.float32)

    def __call__(self, ids_np):
        ids_np = np.asarray(ids_np)
        rows = self.table[ids_np.reshape(-1)]
        rows = rows.reshape(ids_np.shape + (rows.shape[-1],))
        dt = BlockTC.COMPUTE_DTYPE
        return tc.tensor(np.ascontiguousarray(rows), dtype="float32").astype(dt)


class _LMHeadWrap:
    def __init__(self, lin):
        self.lin = lin

    def __call__(self, x):
        return self.lin(x)


def _rand_biased_linear(rng, out_f, in_f, group_size):
    w = (rng.standard_normal((out_f, in_f)) * 0.05).astype(np.float32)
    b = (rng.standard_normal((out_f,)) * 0.01).astype(np.float32)
    return BiasedQuantLinearTC(w, b, group_size=group_size)


def _build_tiny_model(cfg, seed=0):
    """A fully synthetic GptOss20B_TC: random attention/MoE/lm_head weights,
    dequant expert mode (no mxfp4_linear_expert kernel dependency at these
    tiny, non-block-aligned shapes)."""
    rng = np.random.default_rng(seed)
    m = GptOss20B_TC(cfg)
    m.embed_tokens = _FakeEmbed(cfg.vocab_size, cfg.hidden_dim, seed=seed)

    groups = cfg.hidden_dim // 32 if cfg.hidden_dim >= 32 else 1
    assert cfg.hidden_dim % 32 == 0, "tiny config must keep hidden_dim a multiple of 32"
    gate_up_out = cfg.intermediate_dim * 2
    down_groups = max(1, cfg.intermediate_dim // 32)

    for layer in m.layers:
        att = layer.self_attn
        att.q_proj = _rand_biased_linear(rng, cfg.num_heads * cfg.head_dim, cfg.hidden_dim, cfg.group_size)
        att.k_proj = _rand_biased_linear(rng, cfg.num_kv_heads * cfg.head_dim, cfg.hidden_dim, cfg.group_size)
        att.v_proj = _rand_biased_linear(rng, cfg.num_kv_heads * cfg.head_dim, cfg.hidden_dim, cfg.group_size)
        att.o_proj = _rand_biased_linear(rng, cfg.hidden_dim, cfg.num_heads * cfg.head_dim, cfg.group_size)
        att.sinks = tc.tensor(rng.standard_normal((cfg.num_heads,)).astype(np.float32))

        mlp = layer.mlp
        mlp.expert_mode = "dequant"
        mlp.router_weight = tc.tensor(
            rng.standard_normal((cfg.num_local_experts, cfg.hidden_dim)).astype(np.float32),
            dtype="float32",
        )
        mlp.router_bias = tc.tensor(
            rng.standard_normal((cfg.num_local_experts,)).astype(np.float32), dtype="float32"
        )
        mlp.gate_up_blocks = rng.integers(
            0, 256, size=(cfg.num_local_experts, gate_up_out, groups, 16), dtype=np.uint8
        )
        mlp.gate_up_scales = rng.integers(
            120, 135, size=(cfg.num_local_experts, gate_up_out, groups), dtype=np.uint8
        )
        mlp.gate_up_bias = (rng.standard_normal((cfg.num_local_experts, gate_up_out)) * 0.01).astype(
            np.float32
        )
        mlp.down_blocks = rng.integers(
            0, 256, size=(cfg.num_local_experts, cfg.hidden_dim, down_groups, 16), dtype=np.uint8
        )
        mlp.down_scales = rng.integers(
            120, 135, size=(cfg.num_local_experts, cfg.hidden_dim, down_groups), dtype=np.uint8
        )
        mlp.down_bias = (rng.standard_normal((cfg.num_local_experts, cfg.hidden_dim)) * 0.01).astype(
            np.float32
        )

    m.norm.weight = tc.tensor(np.ones((cfg.hidden_dim,), dtype=np.float32), dtype="float32")
    m.lm_head = _LMHeadWrap(_rand_biased_linear(rng, cfg.vocab_size, cfg.hidden_dim, cfg.group_size))
    return m


def test_gpt_oss_20b_tc_exposes_the_arena_contract_surface():
    """Static contract check (ladder step 1): every attribute/method
    ArenaCache/GQAArenaCache and kv_graft touch on the model object."""
    _require_tensor_cuda()
    old_dtype = BlockTC.COMPUTE_DTYPE
    BlockTC.COMPUTE_DTYPE = "bfloat16"
    try:
        cfg = _tiny_gpt_oss_config()
        m = _build_tiny_model(cfg)

        assert len(m.layers) == cfg.num_layers
        assert m.config is cfg
        assert m.config.num_kv_heads == cfg.num_kv_heads
        assert m.config.head_dim == cfg.head_dim
        assert m.rope_cos is not None and m.rope_sin is not None
        assert callable(m.extend_rope)

        from core import kv_graft

        for layer in m.layers:
            att = kv_graft._attention_module(layer)
            assert att is not None
            assert hasattr(att, "inject_kv")
            assert hasattr(att, "graft_seats")
            att.inject_kv = None
            att.graft_seats = 0
    finally:
        BlockTC.COMPUTE_DTYPE = old_dtype


def test_gpt_oss_20b_tc_extend_rope_grows_and_is_idempotent_below_len():
    _require_tensor_cuda()
    old_dtype = BlockTC.COMPUTE_DTYPE
    BlockTC.COMPUTE_DTYPE = "bfloat16"
    try:
        cfg = _tiny_gpt_oss_config(initial_context_length=16)
        m = GptOss20B_TC(cfg)
        assert m.rope_cos.shape[0] == 16
        assert m.rope_sin.shape[0] == 16

        m.extend_rope(8)          # below current length: no-op
        assert m.rope_cos.shape[0] == 16

        m.extend_rope(100)        # grow
        assert m.rope_cos.shape[0] == 100
        assert m.rope_sin.shape[0] == 100
        assert m.rope_cos.shape[1] == cfg.head_dim
    finally:
        BlockTC.COMPUTE_DTYPE = old_dtype


def test_gpt_oss_20b_tc_incremental_decode_matches_full_refeed():
    """Ladder step 3 (unit-test scale): greedy decode via KV-cache reuse
    must match a full-refeed forward, token for token, on a synthetic
    model — the same property validated at full scale against the real
    24-layer snapshot in the parity receipt (see the E2E ledger)."""
    _require_tensor_cuda()
    old_dtype = BlockTC.COMPUTE_DTYPE
    old_fused = QuantLinearTC.FUSED_DECODE if hasattr(QuantLinearTC, "FUSED_DECODE") else None
    BlockTC.COMPUTE_DTYPE = "bfloat16"
    try:
        cfg = _tiny_gpt_oss_config()
        m = _build_tiny_model(cfg, seed=7)
        ngen = 4
        prompt = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)

        with tc.no_grad():
            logits, caches = m(prompt, last_token_only=True)
            inc_out = [int(logits.float().numpy().reshape(-1, cfg.vocab_size)[-1].argmax())]
            pos = prompt.shape[1]
            for _ in range(ngen - 1):
                logits, caches = m(
                    np.array([[inc_out[-1]]], dtype=np.int64),
                    kv_caches=caches, position_offset=pos, last_token_only=True,
                )
                pos += 1
                inc_out.append(int(logits.float().numpy().reshape(-1, cfg.vocab_size)[-1].argmax()))

        refeed_out = []
        cur = prompt
        with tc.no_grad():
            for _ in range(ngen):
                logits, _ = m(cur, last_token_only=True)
                nxt = int(logits.float().numpy().reshape(-1, cfg.vocab_size)[-1].argmax())
                refeed_out.append(nxt)
                cur = np.concatenate([cur, np.array([[nxt]], dtype=np.int64)], axis=1)

        assert inc_out == refeed_out
    finally:
        BlockTC.COMPUTE_DTYPE = old_dtype
        if old_fused is not None:
            QuantLinearTC.FUSED_DECODE = old_fused


def test_gpt_oss_20b_tc_caches_alias_matches_kv_caches_and_rejects_both():
    """__call__ accepts kv_caches= (the ArenaCache/Mistral7B_TC/Qwen3_TC
    convention) and a caches= alias (the Qwen35_TC/Gemma4_TC convention),
    must produce the same result either way, and must reject both at once."""
    _require_tensor_cuda()
    old_dtype = BlockTC.COMPUTE_DTYPE
    BlockTC.COMPUTE_DTYPE = "bfloat16"
    try:
        cfg = _tiny_gpt_oss_config()
        m = _build_tiny_model(cfg, seed=3)
        prompt = np.array([[1, 2, 3]], dtype=np.int64)

        with tc.no_grad():
            logits_a, caches_a = m(prompt, last_token_only=True)
            logits_b, caches_b = m(np.array([[4]], dtype=np.int64), kv_caches=caches_a,
                                    position_offset=3, last_token_only=True)

            logits_a2, caches_a2 = m(prompt, last_token_only=True)
            logits_c, caches_c = m(np.array([[4]], dtype=np.int64), caches=caches_a2,
                                    position_offset=3, last_token_only=True)

        np.testing.assert_array_equal(
            logits_b.float().numpy(), logits_c.float().numpy()
        )

        with pytest.raises(ValueError):
            m(np.array([[4]], dtype=np.int64), kv_caches=caches_b, caches=caches_c)
    finally:
        BlockTC.COMPUTE_DTYPE = old_dtype


def test_gpt_oss_20b_tc_dialect_kwargs_key_off_type_name():
    """DialectDescriptor.from_model (core/graft_repository.py) uses
    type(model).__name__ as model_type -- confirm the real class produces
    the same string the dialect metadata already hard-codes (previously
    only a bare string per the ledger; now the actual class name)."""
    assert GptOss20B_TC.__name__ == "GptOss20B_TC"
    cfg = _tiny_gpt_oss_config()
    got = gpt_oss_grm_dialect_kwargs(cfg)
    assert got["model_type"] == GptOss20B_TC.__name__
