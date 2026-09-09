"""CPU campaign/negative-scoring gates; house evidence discipline (2026).
Synthetic metrics below are unit-test operands, never model observations.
"""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts import grm_x1_campaign as c
from scripts.grm_x1_gpu import clone_node, duplicate_children
from scripts.grm_x1_register import create


def test_frozen_fixture_hashes_queries_and_gpu_matrix():
    assert c.verify_fixtures()["queries"] == 24
    matrix = c.dry_run()
    assert [row["turns"] for row in matrix["cells"]] == [72] * 6 + [48] * 3
    assert matrix["primary_worker_max_s"] <= 1800
    assert matrix["all_worker_max_s"] <= 2700
    pairs = [(q, row["multiplicity"], arm, cond) for row in matrix["cells"]
             for q in row["query_ids"] for arm in row["arms"] for cond in row["conditions"]]
    assert len(pairs) == len(set(pairs)) == 576


@pytest.mark.parametrize("answer,exact,false,abstain", [
    ("Cobalt-3-Comet", True, False, False), ('"COBALT‑3‑COMET".', True, False, False),
    ("The value is Cobalt-3-Comet", False, True, False),
    ("I don't know; Amber-1-Atlas", False, True, False),
    ("Cobalt-3-Comet or Amber-1-Atlas", False, True, False),
    ("I don't know.", False, False, True), ("", False, True, False),
    ("wrong", False, True, False)])
def test_strict_scoring_cannot_hide_false_answers(answer, exact, false, abstain):
    q = {"expected": "cobalt-3-comet", "rejected": ["amber-1-atlas"]}
    row = c.score(answer, q, "present", {"abstained": True, "x1_page_fault": True})
    assert (row["exact_answer"], row["false_answer"], row["abstained"]) == (exact, false, abstain)


def test_absent_correct_value_is_still_an_unsupported_answer():
    q = {"expected": "Cobalt-3-Comet", "rejected": []}
    assert c.score(q["expected"], q, "absent", {})["false_answer"] is True


def synthetic_rows():
    from core.grm_x1_addresses import ABSTENTION
    rows = []
    for q in c.queries():
        for m in (1, 10, 100):
            for arm in ("A", "B", "C"):
                for condition in ("present", "absent"):
                    answer, info = q["expected"], {}
                    if arm == "A" and m == 100:
                        answer = "deliberately wrong"
                    if arm == "C" and condition == "absent":
                        answer, info = ABSTENTION, {"x1_page_fault": True}
                    rows.append({"query_id": q["query_id"], "family_id": q["family_id"], "kind": q["kind"],
                        "arm": arm, "condition": condition, "multiplicity": m,
                        "wall_s": 1., "address_coverage": True, "generation_calls": 0 if info else 1,
                        **c.score(answer, q, condition, info)})
    return rows


def test_positive_fork_requires_all_cells_no_duplicates_and_coverage():
    rows = synthetic_rows()
    result = c.aggregate(rows)
    assert result["oracle_complete"] and result["oracle_positive"]
    assert not c.aggregate(rows[:-1])["oracle_positive"]
    assert not c.aggregate(rows + [rows[0]])["oracle_positive"]
    next(r for r in rows if r["arm"] == "B" and r["condition"] == "present")["address_coverage"] = False
    assert not c.aggregate(rows)["oracle_positive"]


def test_null_is_not_positive_or_a_zero_false_rate():
    assert c.metric([])["false_rate"] is None
    assert c.aggregate([])["predictions"] is None
    rows = synthetic_rows()
    for r in rows:
        if r["arm"] == "A":
            r["false_answer"] = False
    assert c.aggregate(rows)["predictions"]["P3"] is None


def test_natural_is_locked_and_resume_stops_on_failure():
    with patch.object(c, "summary", return_value={"failed": False, "oracle_positive": False, "cell_status": {}}):
        with pytest.raises(ValueError, match="locked"):
            c.next_cell("natural_m1")
    with patch.object(c, "summary", return_value={"failed": True}):
        with pytest.raises(ValueError, match="stop"):
            c.next_cell()


def test_receipts_are_create_only(tmp_path):
    path = tmp_path / "receipt"
    create(path, b"first")
    with pytest.raises(FileExistsError):
        create(path, b"second")
    assert path.read_bytes() == b"first"


@pytest.mark.parametrize("multiplicity", [1, 10, 100])
def test_duplicate_stress_uses_real_family_identity_and_shared_native_payload(multiplicity):
    h = object()
    nodes = [{"sources": [1, 2], "metadata": {"width_guard_parent": True}},
             {"h": h, "cent": [1], "metadata": {"width_guard_child": True}, "sources": [0]},
             {"h": h, "cent": [2], "metadata": {"width_guard_child": True}, "sources": [0]}]
    copies = duplicate_children(nodes, 0, multiplicity)
    assert len(copies) == multiplicity
    assert len(nodes[0]["sources"]) == 2 + multiplicity
    for i in copies:
        assert nodes[i]["h"] is h
        assert nodes[i]["node_id"] == i
        assert nodes[i]["metadata"]["width_guard_child"]
        assert nodes[i]["metadata"]["culled_from"] == 0
        assert nodes[i]["metadata"] is not nodes[1]["metadata"]
    with pytest.raises(ValueError):
        duplicate_children([{"metadata": {}}], 0, 1)


def test_controller_has_foreground_bounded_flock_and_one_cooldown(tmp_path):
    observed = []
    with patch.object(c, "fingerprint", return_value="f" * 64), patch.object(c, "next_cell", return_value={"cell": "oracle_m1_s0"}), \
         patch.object(c.subprocess, "run", side_effect=lambda args, **kw: (observed.append((args, kw)) or type("Result", (), {"returncode": 1})())), \
         patch.object(c.time, "sleep") as sleep, patch.object(c, "emit_receipt"):
        result = c.run_controller()
    args, kwargs = observed[0]
    assert args[:5] == ["flock", "--wait", "250", "--no-fork", "/tmp/forge-gpu.lock"]
    assert "timeout" not in kwargs
    assert result["status"] == "RED"
    sleep.assert_called_once_with(30)


def test_flock_descriptor_survives_exec_and_busy_lock_yields(tmp_path):
    # Real foreground process protocol on a private CPU-only lock; no GPU,
    # no background child and no process is signalled or killed.
    import fcntl
    import subprocess
    import sys
    lock = tmp_path / "cpu-only.lock"
    code = "import pathlib,sys; p=pathlib.Path(sys.argv[1]); assert any(x.resolve()==p for x in pathlib.Path('/proc/self/fd').iterdir())"
    result = subprocess.run(["flock", "--wait", "0.1", "--no-fork", str(lock), sys.executable, "-c", code, str(lock)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    with lock.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        busy = subprocess.run(["flock", "--wait", "0.1", "--no-fork", str(lock), sys.executable, "-c", "raise AssertionError('must not launch')"], capture_output=True, text=True)
        assert busy.returncode == 1 and "AssertionError" not in busy.stderr
