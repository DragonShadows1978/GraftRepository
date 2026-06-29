import inspect

from core.deepseek_v2_lite_tc import (DeepSeekMLAArenaCache, DeepSeekMLATC,
                                      DeepSeekV2LiteConfig)
from core.graft_arena import ArenaCache


def test_deepseek_arena_uses_deepseek_mla_payload_width():
    cfg = DeepSeekV2LiteConfig

    assert DeepSeekMLAArenaCache.PAYLOAD == (("c", 1), ("kpe", 2))
    assert DeepSeekMLAArenaCache.VALS_PER_TOK_LAYER == (
        cfg.kv_lora_rank + cfg.qk_rope_head_dim)


def test_deepseek_librarian_uses_text_scaffolded_prompts():
    assert ArenaCache.TEXT_SCAFFOLD_CONSOLIDATION is False
    assert ArenaCache.ALLOW_HIGH_COVERAGE_LIST_DIGESTS is False
    assert ArenaCache.ENABLE_ERA_FOLDING is True
    assert DeepSeekMLAArenaCache.TEXT_SCAFFOLD_CONSOLIDATION is True
    assert DeepSeekMLAArenaCache.CONSOLIDATE_NGEN > ArenaCache.CONSOLIDATE_NGEN
    assert DeepSeekMLAArenaCache.ALLOW_HIGH_COVERAGE_LIST_DIGESTS is True
    assert DeepSeekMLAArenaCache.ENABLE_ERA_FOLDING is False

    arena = DeepSeekMLAArenaCache.__new__(DeepSeekMLAArenaCache)
    prompt = arena._consolidation_prompts(
        deep=False,
        source_texts=("User: Alpha code A11-2200.\nAssistant: Recorded.",))[0]

    assert "Source excerpts:" in prompt
    assert "use every source below" in prompt
    assert "[source 1]" in prompt
    assert "A11-2200" in prompt
    assert prompt.count("User:") == 1
    assert prompt.count("Assistant:") == 1
    assert "Assistant: Recorded" not in prompt
    assert prompt.rsplit("Assistant:", 1)[1].strip()


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
