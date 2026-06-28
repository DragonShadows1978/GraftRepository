import inspect

from core.deepseek_v2_lite_tc import (DeepSeekMLAArenaCache, DeepSeekMLATC,
                                      DeepSeekV2LiteConfig)


def test_deepseek_arena_uses_deepseek_mla_payload_width():
    cfg = DeepSeekV2LiteConfig

    assert DeepSeekMLAArenaCache.PAYLOAD == (("c", 1), ("kpe", 2))
    assert DeepSeekMLAArenaCache.VALS_PER_TOK_LAYER == (
        cfg.kv_lora_rank + cfg.qk_rope_head_dim)


def test_deepseek_attention_exposes_grm_hook_contract():
    src = inspect.getsource(DeepSeekMLATC.__call__)

    assert "_capture_q" in src
    assert "_captured_q" in src
    assert "_capture" in src
    assert "_captured" in src
    assert "inject_kv" in src
    assert "graft_seats" in src
    assert "live_shift" in src
    assert "_deepseek_rope_prep(kpe_g)" in src


def test_deepseek_smoke_gate_can_enable_native_runtime():
    from tests import deepseek_grm_smoke_gate

    src = inspect.getsource(deepseek_grm_smoke_gate.run_repo_gate)
    assert "native_lib_path=args.native_lib" in src
    assert 'st["native"]["host_payload_tensors"] >= 2' in src
    assert 'rst["native"]["route_entries"] >= 1' in src
