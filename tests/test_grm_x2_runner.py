"""CPU checks of the GPU contract; no model imports or GPU calls.

Prior art: house EB1/RS3 (2026) receipt/invariant checks; ours: X2 runner
scope, no-vacuity, stop-rail, and prompt-boundary tests.
"""
import ast
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import grm_x2_common as common
from scripts import grm_x2_gpu as gpu


def test_direct_worker_pins_production_live_shift_before_opening_turn():
    from types import SimpleNamespace
    layers = [SimpleNamespace(self_attn=SimpleNamespace(live_shift=None)),
              SimpleNamespace(self_attn=SimpleNamespace(live_shift=17))]
    observed = []
    def begin():
        observed.extend(layer.self_attn.live_shift for layer in layers)
        return [3, 4]
    arena = SimpleNamespace(m=SimpleNamespace(layers=layers), live_shift=99, eb1_begin_turn=begin)
    assert gpu.open_turn(arena) == [3, 4]
    assert observed == [99, 99]


def test_dry_run_enumerates_all_48_paired_cells_without_gpu_imports():
    before = set(sys.modules)
    data = gpu.plan()
    assert len(data["rows"]) == 48
    assert len({r["question_id"] for r in data["rows"]}) == 24
    assert len({r["cell"] for r in data["rows"]}) == 6
    for case in {r["question_id"] for r in data["rows"]}:
        rows = [r for r in data["rows"] if r["question_id"] == case]
        assert {r["arm"] for r in rows} == {"A", "B"}
        assert len({r["question_sha256"] for r in rows}) == 1
        assert not any(r["prompt_metadata"] for r in rows)
    assert not any(m.startswith(("tensor_cuda", "torch", "transformers")) for m in set(sys.modules) - before)


def test_create_only_receipt_cannot_overwrite(tmp_path):
    path = tmp_path / "receipt.json"
    common.create_json(path, {"first": True})
    with pytest.raises(FileExistsError):
        common.create_json(path, {"first": False})
    assert common.read(path) == {"first": True}


def test_receipt_fingerprint_changes_with_scoped_code(monkeypatch):
    baseline, _ = common.fingerprint()
    real = common.sha_file
    monkeypatch.setattr(common, "sha_file", lambda p: "a" * 64 if str(p).endswith("grm_x2_witnesses.py") else real(p))
    changed, _ = common.fingerprint()
    assert changed != baseline


def test_cpu_red_rail_blocks_gpu_before_any_preflight(monkeypatch):
    real_read = gpu.read
    monkeypatch.setattr(gpu, "read", lambda p: {"status": "RED_KILL", "gpu_allowed": False} if p == gpu.CPU_RECEIPT else real_read(p))
    with pytest.raises(RuntimeError, match="RED kill"):
        gpu.ready()


def test_missing_gpu_data_cannot_pass_P3(monkeypatch, tmp_path):
    # Summary uses the real frozen fixture IDs but an empty receipt root.
    real_read = gpu.read
    monkeypatch.setattr(gpu, "OUT", tmp_path)
    monkeypatch.setattr(gpu, "read", lambda p: real_read(common.OUT / "fixtures/live.json") if str(p).endswith("fixtures/live.json") else real_read(p))
    result = gpu.summary()
    assert result["P3_pass"] is None and result["correct_answers_B_rejected"] is None
    assert len(result["missing"]) == 24 and result["status"] == "BLOCKED_OR_INCOMPLETE"


def test_runner_has_no_process_termination_calls_or_background_launch():
    tree = ast.parse((ROOT / "scripts/grm_x2_gpu.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            assert name not in {"kill", "killpg", "terminate", "system", "alarm"}
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert node.value not in {"pkill", "killall", "gpu-reset", "systemctl"}
    assert "--wait" in (ROOT / "scripts/grm_x2_gpu.py").read_text()
    assert "setitimer" in (ROOT / "scripts/grm_x2_gpu.py").read_text()


def test_P3_counts_correct_raw_answers_already_rejected_by_A(monkeypatch, tmp_path):
    fixture = common.read(common.OUT / "fixtures/live.json")
    def score(exact=False, abstained=False):
        return dict(exact_answer=exact, false_answer=False, abstained=abstained,
                    abstention_correct=False, abstention_incorrect=abstained,
                    manual_review=False, unsupported_lucky_truth=False)
    rows = []
    for i, case in enumerate(fixture):
        correct = case["mount_class"] == "correct"
        rows.append({"question_id": case["id"], "mount_class": case["mount_class"],
                     "scores": {"raw": score(exact=correct), "A": score(exact=correct and i >= 2, abstained=i < 2 or not correct),
                                "B": score(exact=correct and i >= 2, abstained=i < 2 or not correct)},
                     "correct_answer_rejected_by_B": False, "raw_correct_answer_rejected_by_B": i < 2})
    common.create_json(tmp_path / "gpu/fake/cell.receipt.json", {"status": "COMPLETE", "rows": rows, "worker_wall_s": 0})
    real_read = gpu.read
    monkeypatch.setattr(gpu, "OUT", tmp_path)
    monkeypatch.setattr(gpu, "fingerprint", lambda: ("fake", {}))
    monkeypatch.setattr(gpu, "read", lambda p: fixture if str(p).endswith("fixtures/live.json") else real_read(p))
    result = gpu.summary()
    assert result["correct_answers_B_rejected"] == 0
    assert result["raw_correct_answers_B_rejected"] == 2
    assert result["P3_pass"] is False
