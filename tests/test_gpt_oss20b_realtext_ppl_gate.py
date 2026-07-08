import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from gpt_oss20b_realtext_ppl_gate import (  # noqa: E402
    max_observed_memory_mib,
    parse_gpu_memory_mib,
    parse_setting,
    summarize_setting,
)


def test_parse_gpu_memory_mib_from_nvidia_smi_row():
    row = "NVIDIA GeForce RTX 4070 SUPER, 798, 12282, 8"

    assert parse_gpu_memory_mib(row) == 798
    assert parse_gpu_memory_mib(None) is None
    assert parse_gpu_memory_mib("bad row") is None


def test_max_observed_memory_mib_scans_artifact_layers():
    artifact = {
        "gpu_before": "NVIDIA GeForce RTX 4070 SUPER, 274, 12282, 36",
        "gpu_before_lm_head": "NVIDIA GeForce RTX 4070 SUPER, 500, 12282, 36",
        "gpu_after": "NVIDIA GeForce RTX 4070 SUPER, 798, 12282, 8",
        "layers": [
            {"gpu_after_layer": "NVIDIA GeForce RTX 4070 SUPER, 905, 12282, 44"},
            {"gpu_after_layer": "NVIDIA GeForce RTX 4070 SUPER, 880, 12282, 42"},
        ],
    }

    assert max_observed_memory_mib(artifact) == 905


def test_summarize_setting_uses_weighted_nll():
    runs = [
        {
            "status": "ok",
            "ppl": {"token_count": 2, "mean_nll": 1.0},
            "max_observed_memory_mib": 700,
        },
        {
            "status": "ok",
            "ppl": {"token_count": 6, "mean_nll": 2.0},
            "max_observed_memory_mib": 900,
        },
    ]

    summary = summarize_setting(runs)

    assert summary["status"] == "ok"
    assert summary["scored_tokens"] == 8
    assert summary["mean_nll"] == pytest.approx(1.75)
    assert summary["ppl"] == pytest.approx(math.exp(1.75))
    assert summary["max_observed_memory_mib"] == 900


def test_parse_setting_accepts_standard_and_apa_refine():
    assert parse_setting("standard") == {
        "name": "standard",
        "attention_mode": "standard",
        "refine_percentile": None,
    }
    assert parse_setting("apa_r0.15") == {
        "name": "apa_r0.15",
        "attention_mode": "apa_selective",
        "refine_percentile": 0.15,
    }
    with pytest.raises(ValueError):
        parse_setting("bulk")
