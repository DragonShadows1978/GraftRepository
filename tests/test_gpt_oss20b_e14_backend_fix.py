import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import gpt_oss20b_e14_backend_fix as e14  # noqa: E402
import gpt_oss20b_expert_e1 as e1  # noqa: E402


class Attention:
    def __init__(self, sliding_window):
        self.sliding_window = sliding_window
        self.attention_mode = "unset"
        self.refine_percentile = -1.0
        self.bulk_bits = -1
        self.last_attention_backend = ""


class Mlp:
    route_detail = "unset"
    empty_cache_interval = -1


class Block:
    def __init__(self, sliding_window):
        self.self_attn = Attention(sliding_window)
        self.mlp = Mlp()


def test_e14_backend_matches_proven_full_scope_registration():
    full = Block(None)
    sliding = Block(128)

    e1.configure_block(full, e1.BACKEND_E14_APA)
    e1.configure_block(sliding, e1.BACKEND_E14_APA)

    assert full.self_attn.attention_mode == "apa_selective"
    assert sliding.self_attn.attention_mode == "standard"
    assert full.self_attn.refine_percentile == 0.15
    assert full.self_attn.bulk_bits == 8
    assert e1.backend_receipt_contract(e1.BACKEND_E14_APA)[
        "full_layer_observed_backend"
    ] == "apa_selective_sink_fused"


def test_historical_backend_and_paths_remain_default(monkeypatch):
    monkeypatch.setattr(sys, "argv", [str(e1.SCRIPT_PATH), "analyze"])
    args = e1.parse_args()

    assert args.backend == e1.BACKEND_HISTORICAL
    assert args.teacher_rule == e1.TEACHER_RULE_ORIGINAL
    assert e1.pair_root(e1.DEFAULT_OUTPUT_DIR, e1.ADDRESS_RULE_E11) == (
        e1.DEFAULT_OUTPUT_DIR / "pairs_e11"
    )
    assert e1.eval_root(e1.DEFAULT_OUTPUT_DIR, e1.ADDRESS_RULE_E11) == (
        e1.DEFAULT_OUTPUT_DIR / "eval_e11"
    )


def test_e14_selection_rule_and_gpu_laws_are_frozen():
    winner = e14.choose_teacher(
        {"p0": 0.2, "p1": 0.2, "p2": 0.1, "p3": 0.0, "p4": -0.1}
    )
    assert winner == ("p0", 0.2)
    assert e14.choose_teacher(
        {"p0": 0.0, "p1": -0.1, "p2": -0.2, "p3": -0.3, "p4": -0.4}
    )[0] is None

    for shell in (e14.resweep_commands(), e14.chain_commands()):
        assert "flock -w 7200 /tmp/forge-gpu.lock" in shell
        assert "timeout --signal=TERM --kill-after=5s 590s" in shell
        assert "CUDA_VISIBLE_DEVICES=0" in shell
        assert "sleep 30" in shell
    assert e14.resweep_commands().index("# FG0 RUNS FIRST") < (
        e14.resweep_commands().index("# Re-run C0")
    )
