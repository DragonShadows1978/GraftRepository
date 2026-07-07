import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from gpt_oss20b_context_ladder import (  # noqa: E402
    classify_run,
    gpu_memory_mib,
    max_observed_memory_mib,
    parse_lengths,
    parse_setting,
    summarize_ladder,
)


def test_parse_lengths_rejects_empty_and_short_values():
    assert parse_lengths("128, 256") == [128, 256]
    with pytest.raises(ValueError):
        parse_lengths("")
    with pytest.raises(ValueError):
        parse_lengths("1")


def test_classify_run_detects_pass_oom_and_failure():
    assert classify_run({"status": "ok"}, 0, "") == "pass"
    assert classify_run({"status": "error", "error": "CUDA out of memory"}, 1, "") == "oom"
    assert classify_run(None, 1, "RuntimeError: cudaMalloc failed: out of memory") == "oom"
    assert classify_run({"status": "error", "error": "shape mismatch"}, 1, "") == "fail"


def test_summarize_ladder_records_first_boundaries():
    runs = [
        {"setting": "standard", "target_tokens": 128, "classification": "pass"},
        {"setting": "standard", "target_tokens": 256, "classification": "pass"},
        {"setting": "standard", "target_tokens": 512, "classification": "oom"},
        {"setting": "apa_r0.15", "target_tokens": 128, "classification": "pass"},
    ]

    summary = summarize_ladder(runs)

    assert summary["standard"]["max_pass_tokens"] == 256
    assert summary["standard"]["first_oom"] == 512
    assert summary["standard"]["first_failure"] == 512
    assert summary["apa_r0.15"]["max_pass_tokens"] == 128
    assert summary["apa_r0.15"]["first_failure"] is None


def test_max_observed_memory_reads_layer_peaks():
    artifact = {
        "gpu_before": "NVIDIA GeForce RTX 4070 SUPER, 274, 12282, 36",
        "gpu_after": "NVIDIA GeForce RTX 4070 SUPER, 500, 12282, 8",
        "layers": [
            {"gpu_after_layer": "NVIDIA GeForce RTX 4070 SUPER, 800, 12282, 40"},
            {"gpu_after_layer": "NVIDIA GeForce RTX 4070 SUPER, 700, 12282, 30"},
        ],
    }

    assert max_observed_memory_mib(artifact) == 800


def test_gpu_memory_mib_returns_int_or_none():
    value = gpu_memory_mib()
    assert value is None or isinstance(value, int)


def test_parse_setting_standard_and_apa():
    assert parse_setting("standard")["attention_mode"] == "standard"
    apa = parse_setting("apa_r0.15")
    assert apa["attention_mode"] == "apa_selective"
    assert apa["refine_percentile"] == 0.15
    with pytest.raises(ValueError):
        parse_setting("bad")
