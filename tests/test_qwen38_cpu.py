import json
import tempfile
from types import SimpleNamespace

import numpy as np

from core.qwen38_tc import (DEFAULT_MODEL_DIR, INT3PackCache, Qwen38Config,
                            PackedRowChunkedQuantLinearTC,
                            compute_qwen38_memory_budget, gate_int3_pack)


def test_qwen38_vector_pack_is_engine_bit_exact():
    assert gate_int3_pack()["max_byte_diff"] == 0


def test_qwen38_real_config_ratios_and_gate():
    cfg = Qwen38Config.from_model_dir(DEFAULT_MODEL_DIR)
    with open(f"{DEFAULT_MODEL_DIR}/config.json") as fh:
        checkpoint_text_config = json.load(fh)["text_config"]
    assert (cfg.hidden_dim, cfg.num_layers, cfg.intermediate_dim) == (5120, 64, 17408)
    assert cfg.n_v_heads // cfg.n_k_heads == 3
    assert cfg.num_heads // cfg.num_kv_heads == 6
    assert checkpoint_text_config["output_gate_type"] == "swish"
    assert cfg.output_gate_type == "sigmoid"
    assert cfg.attention_layer_indices() == list(range(3, 64, 4))


def test_qwen38_computed_budget_is_below_order_ceiling():
    budget = compute_qwen38_memory_budget(DEFAULT_MODEL_DIR)
    assert budget["total_vram_batch1_seq128"] < int(11.5 * 2**30)
    assert budget["host_bf16_embedding_ram"] == 248320 * 5120 * 2


def test_qwen38_lm_head_default_is_exactly_eight_packed_row_chunks():
    ranges = PackedRowChunkedQuantLinearTC.row_ranges(248320)
    assert len(ranges) == 8
    assert ranges[0] == (0, 31040)
    assert ranges[-1] == (217280, 248320)


def test_qwen38_last_token_is_sliced_before_lm_head():
    from core.qwen38_tc import Qwen38_TC

    class FakeTensor:
        def __init__(self, shape):
            self.shape = tuple(shape)

        def astype(self, _dtype):
            return self

        def half(self):
            return self

        def slice(self, dim, start, length):
            shape = list(self.shape)
            shape[dim] = length
            return FakeTensor(shape)

    seen = []
    model = Qwen38_TC.__new__(Qwen38_TC)
    model.config = SimpleNamespace()
    model.layers = []
    model._rope_len = 4096
    model.embed_tokens = lambda ids: FakeTensor((*ids.shape, 5120))
    model.norm = lambda h: h
    model.lm_head = lambda h: seen.append(h.shape) or FakeTensor(
        (*h.shape[:-1], 248320))
    logits, caches = model(np.zeros((1, 37), dtype=np.int64),
                           last_token_only=True)
    assert seen == [(1, 1, 5120)]
    assert logits.shape == (1, 1, 248320)
    assert caches == []


def test_qwen38_lm_head_invokes_two_stage_once_per_packed_chunk(monkeypatch):
    import core.qwen38_tc as q38

    calls = []

    class FakeTC:
        @staticmethod
        def intn_linear(x, packed, scales, zeros, bits, in_features, group_size):
            calls.append((x, packed, scales, zeros, bits, in_features, group_size))
            return f"logits-{packed}"

        @staticmethod
        def cat(parts, dim):
            return tuple(parts), dim

    monkeypatch.setattr(q38, "tc", FakeTC)
    head = PackedRowChunkedQuantLinearTC.__new__(PackedRowChunkedQuantLinearTC)
    head.chunks = [(f"p{i}", f"s{i}", f"z{i}") for i in range(8)]
    head.bits = 3
    head.in_features = 5120
    head.group_size = 128
    got = head("hidden")
    assert len(calls) == 8
    assert all(call[0] == "hidden" and call[4:] == (3, 5120, 128)
               for call in calls)
    assert got == (tuple(f"logits-p{i}" for i in range(8)), -1)


def test_qwen38_cache_quantizer_matches_engine_reference():
    import torch
    from tensor_cuda.quantization import quantize_affine_per_group

    rng = np.random.default_rng(38)
    weight = rng.standard_normal((17, 256)).astype(np.float32)
    name = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
    with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
        cache = INT3PackCache(cache_dir=tmp, row_chunk=5)
        entry = cache._quantize_tensor(torch.from_numpy(weight), name)
        packed = np.load(f"{tmp}/{entry['packed']}")
        scales = np.load(f"{tmp}/{entry['scales']}")
        zeros = np.load(f"{tmp}/{entry['zeros']}")
    expected = quantize_affine_per_group(weight, 3, 128)
    np.testing.assert_array_equal(packed, expected.packed)
    np.testing.assert_array_equal(scales, expected.scales)
    np.testing.assert_array_equal(zeros, expected.zeros)
