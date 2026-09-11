"""CPU campaign/negative-scoring gates; house evidence discipline (2026).
Synthetic metrics below are unit-test operands, never model observations.
"""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts import grm_x1_campaign as c
from scripts.grm_x1_gpu import clone_node, duplicate_children, payload_digest
from scripts.grm_x1_register import create


@pytest.mark.campaign_receipt(registration='artifacts/grm_x1/registration.json (frozen source_shas)')
def test_frozen_fixture_hashes_queries():
    assert c.verify_fixtures()["queries"] == 24


def test_frozen_fixture_hashes_queries_and_gpu_matrix(continuation_tree):
    matrix = c.dry_run()
    assert [row["turns"] for row in matrix["cells"]] == [72] * 6 + [48] * 3
    if (c.OUT / "continuation_03.json").exists():
        assert matrix["budget"]["total_s"] == 5400
        assert matrix["unit_count"] == 144
    else:
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


def payload_repo(tmp_path, node, pack_node=None):
    # House GraftRepository (2026): exercise the real host/durable accessor
    # without constructing a model, native store or CUDA arena.
    from types import SimpleNamespace
    from core.graft_repository import GraftRepository
    repo = GraftRepository.__new__(GraftRepository)
    repo.path = str(tmp_path)
    repo.arena = SimpleNamespace(grafts=[node], pack_node=pack_node)
    return repo


def test_payload_digest_h_none_host_backing_and_cold_file_match(tmp_path):
    import numpy as np
    payload = {"0_k": np.arange(6, dtype=np.int8).reshape(2, 3),
               "0_ks": np.array([0.5], dtype=np.float32)}
    host = {"h": None, "host_payload": payload}
    expected = payload_digest(payload_repo(tmp_path, host), 0)
    assert host["h"] is None
    assert host["host_payload"] is payload
    (tmp_path / "nodes").mkdir()
    np.savez(tmp_path / "nodes/0000.npz", **payload)
    cold = {"h": None, "host_payload": None, "durable": True}
    assert payload_digest(payload_repo(tmp_path, cold), 0) == expected
    assert cold["h"] is None  # no device rehydration
    snapshot = clone_node(cold)
    assert snapshot["host_payload"] is cold["host_payload"]
    # A fresh arm has no capture NPZ file; its clone must carry usable backing.
    assert payload_digest(payload_repo(tmp_path / "fresh_arm", snapshot), 0) == expected
    resident_h = object()
    calls = []
    def pack(h):
        calls.append(h)
        return payload
    resident = {"h": resident_h, "host_payload": None}
    assert payload_digest(payload_repo(tmp_path, resident, pack), 0) == expected
    assert calls == [resident_h]


def test_payload_digest_packed_content_identity(tmp_path):
    import numpy as np
    payload = {"0_k": np.arange(6, dtype=np.int8).reshape(2, 3),
               "0_ks": np.array([0.5], dtype=np.float32)}
    def digest(backing):
        return payload_digest(payload_repo(tmp_path, {"h": None, "host_payload": backing}), 0)
    expected = digest(payload)
    assert digest(dict(reversed(list(payload.items())))) == expected
    assert digest({**payload, "0_k": np.asfortranarray(payload["0_k"])}) == expected
    assert digest({**payload, "0_k": payload["0_k"] + 1}) != expected
    assert digest({**payload, "0_ks": payload["0_ks"] * 2}) != expected
    assert digest({**payload, "0_k": payload["0_k"].reshape(3, 2)}) != expected
    assert digest({**payload, "0_k": payload["0_k"].view(np.uint8)}) != expected
    assert digest({"1_k": payload["0_k"], "0_ks": payload["0_ks"]}) != expected


@pytest.mark.parametrize("durable", [False, True])
def test_payload_digest_missing_backing_remains_red(tmp_path, durable):
    node = {"h": None, "host_payload": None, "durable": durable,
            "payload_pending": True}
    repo = payload_repo(tmp_path, node)
    with pytest.raises(FileNotFoundError if durable else RuntimeError):
        payload_digest(repo, 0)


def test_payload_digest_empty_backing_remains_red(tmp_path):
    repo = payload_repo(tmp_path, {"h": None, "host_payload": {}})
    with pytest.raises(RuntimeError, match="no packed payload"):
        payload_digest(repo, 0)


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


def test_controller_has_foreground_bounded_flock_and_one_cooldown(tmp_path, continuation_tree):
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


@pytest.fixture
def continuation_tree(tmp_path, monkeypatch):
    # Prior art: house X1 (2026) isolated CPU fixtures and synthetic paired
    # rows. New: amendment-2 state transitions; these are never E2E evidence.
    import shutil
    root = c.ROOT
    sources = c.read(c.OUT / "continuation_02.json")["sources"].copy()
    out = tmp_path / "artifacts/grm_x1"
    shutil.copytree(c.OUT, out, ignore=shutil.ignore_patterns("sessions", "r2", "r3", "r4", "continuation_03*", "continuation_04*"))
    shutil.copyfile(out / "lead_commands_r2.txt", out / "lead_commands.txt")
    order = tmp_path / "orders/GRM_X1_AMENDMENT_2.md"
    order.parent.mkdir()
    shutil.copyfile(root / "orders/GRM_X1_AMENDMENT_2.md", order)
    monkeypatch.setattr(c, "ROOT", tmp_path)
    monkeypatch.setattr(c, "OUT", out)
    monkeypatch.setattr(c, "FIX", out / "fixtures")
    monkeypatch.setattr(c, "source_manifest", lambda: sources.copy())
    # The five gates target continuation identity/state; the full baseline
    # separately checks real fixtures, and live preflight checks dependencies.
    monkeypatch.setattr(c, "verify_fixtures", lambda: {"status": "PASS"})
    monkeypatch.setattr(c, "dependency_inventory", lambda: c.read(out / "handoff_manifest.json")["dependencies"])
    return out, sources


def rewrite_test_continuation(out, value, *, refresh_checksum=True):
    # Deliberate forgery in a private temporary tree only.
    path = out / "continuation_02.json"
    path.write_bytes(c.raw_json(value))
    if refresh_checksum:
        (out / "continuation_02.sha256").write_text(c.sha(path) + "  continuation_02.json\n")


def simulated_cell(out, cell, status, rows=None, *, claim_only=False):
    fp = c.fingerprint()
    create(out / "claims/r2" / f"{cell}_{fp}.json", c.raw_json(
        {"cell": cell, "fingerprint": fp, "reserved_gpu_s": 285}))
    if not claim_only:
        c.emit_receipt("gpu_" + cell, {"cell": cell, "fingerprint": fp,
            "registration_sha256": c.sha(out / "registration.json"),
            "status": status, "rows": rows or [], "error": "synthetic CPU operand" if status == "RED" else None},
            directory=out / "receipts/r2")


def test_continuation_accepted(continuation_tree):
    out, _ = continuation_tree
    assert c.fingerprint() == c.sha(out / "continuation_02.json")
    assert c.next_cell()["cell"] == "oracle_m1_s0"
    state = c.summary()
    assert state["active_epoch"] == "r2" and not state["failed"]
    assert state["campaign_verdict"] == "PENDING" and state["rows"] == 0
    assert state["epochs"]["r1"]["cell_status"] == {"oracle_m1_s0": "RED"}
    assert state["epochs"]["r2"]["cell_status"] == {}
    assert len(state["historical_red"]) == 1
    assert state["retry_eligible"] == ["oracle_m1_s0"]
    assert (state["reserved_gpu_s"], state["reservation_cap_s"]) == (285, 2850)
    matrix = c.dry_run()
    assert [row["cell"] for row in matrix["cells"]] == [row["cell"] for row in c.read(c.FIX / "cells.json")]
    assert [row["retry"] for row in matrix["cells"]] == [True] + [False] * 8
    assert matrix["campaign_reservation_max_s"] == 2850
    before = (out / "continuation_02.json").read_bytes()
    with pytest.raises(FileExistsError, match="immutable"):
        c.seal_continuation()
    assert (out / "continuation_02.json").read_bytes() == before


@pytest.mark.parametrize("attack", ["checksum", "order", "payload_handoff", "registration", "r1_handoff", "budget", "retry_limit", "retry_cell", "receipt_directory"])
def test_forged_continuation_refused(continuation_tree, attack):
    out, _ = continuation_tree
    value = c.read(out / "continuation_02.json")
    if attack in {"order", "payload_handoff", "registration", "r1_handoff"}:
        key = {"order": "orders/GRM_X1_AMENDMENT_2.md",
               "payload_handoff": "artifacts/grm_x1/payload_amendment_01_handoff.json",
               "registration": "artifacts/grm_x1/registration.json",
               "r1_handoff": "artifacts/grm_x1/handoff_manifest.json"}[attack]
        value["bindings"][key] = "0" * 64
    elif attack == "budget":
        value["budget"]["total_worker_s"] = 3135
    elif attack == "retry_limit":
        value["retry_limit"] = 2
    elif attack == "retry_cell":
        value["retry_cells"].append("oracle_m1_s1")
    else:
        value["receipt_directory"] = "artifacts/grm_x1/receipts"
    rewrite_test_continuation(out, value, refresh_checksum=attack != "checksum")
    with pytest.raises(ValueError, match="continuation"):
        c.fingerprint()


def test_stale_continuation_refused(continuation_tree):
    _, sources = continuation_tree
    sources["scripts/grm_x1_campaign.py"] = "0" * 64
    with pytest.raises(ValueError, match="stale continuation: source SHA mismatch"):
        c.fingerprint()


@pytest.mark.parametrize("outcome", ["RED", "INCOMPLETE_CLAIM", "COMPLETE"])
def test_second_retry_refused(continuation_tree, outcome):
    out, _ = continuation_tree
    assert c.next_cell("oracle_m1_s0")["cell"] == "oracle_m1_s0"
    simulated_cell(out, "oracle_m1_s0", outcome, claim_only=outcome == "INCOMPLETE_CLAIM")
    with pytest.raises(ValueError, match="stop|already consumed"):
        c.next_cell("oracle_m1_s0")
    assert c.summary()["retry_eligible"] == []
    if outcome != "COMPLETE":
        assert c.summary()["campaign_verdict"] == "RED"
        with pytest.raises(ValueError, match="registered stop"):
            c.next_cell("oracle_m1_s1")
        with pytest.raises(ValueError, match="registered stop"):
            c.next_cell()


def test_r1_receipt_untouched(continuation_tree):
    out, _ = continuation_tree
    historical = c.validated_continuation()["historical_files"]
    before = {path: (c.ROOT / path).read_bytes() for path in historical}
    simulated_cell(out, "oracle_m1_s0", "RED")
    assert c.summary()["historical_red"][0]["error"]["message"] == "'NoneType' object is not iterable"
    for path, original in before.items():
        assert (c.ROOT / path).read_bytes() == original
        assert c.sha(c.ROOT / path) == historical[path]


def test_original_handoff_still_accepted_without_continuation(continuation_tree):
    out, sources = continuation_tree
    (out / "continuation_02.json").unlink()  # private CPU fixture only
    sources.update(c.read(out / "handoff_manifest.json")["sources"])
    assert c.fingerprint() == c.sha(out / "handoff_manifest.json")
    with pytest.raises(ValueError, match="registered stop"):
        c.next_cell()


def test_lead_commands_order_cpu_simulation(continuation_tree):
    import shlex
    out, _ = continuation_tree
    commands = [shlex.split(line)[2:] for line in (out / "lead_commands.txt").read_text().splitlines()
                if line.startswith("bash scripts/grm_x1_lead_gpu.sh ")]
    cells = c.read(c.FIX / "cells.json")
    assert commands == [["preflight"], *[["run", cell["cell"]] for cell in cells], ["summary"]]
    oracle = synthetic_rows()
    natural = [{**row, "arm": "N"} for row in oracle if row["arm"] == "C"]
    def cpu_worker(args, **kwargs):
        cell_id = args[args.index("_worker") + 1]
        cell = next(item for item in cells if item["cell"] == cell_id)
        rows = [row for row in oracle + natural if row["query_id"] in cell["query_ids"]
                and row["multiplicity"] == cell["multiplicity"] and row["arm"] in cell["arms"]]
        assert len(rows) == len(cell["query_ids"]) * len(cell["conditions"]) * len(cell["arms"])
        simulated_cell(out, cell_id, "COMPLETE", rows)
        return type("Result", (), {"returncode": 0})()
    with patch.object(c.subprocess, "run", side_effect=cpu_worker), patch.object(c.time, "sleep"):
        assert c.preflight()["fingerprint"] == c.sha(out / "continuation_02.json")
        for command in commands[1:-1]:
            assert c.run_controller(command[1])["status"] == "RETURNED"
    result = c.summary()
    assert result["rows"] == 576 and result["oracle_positive"]
    assert not result["failed"] and len(result["historical_red"]) == 1
    assert result["reserved_gpu_s"] == 2850
    assert result["epochs"]["r1"]["reserved_gpu_s"] == 285
    assert result["epochs"]["r2"]["reserved_gpu_s"] == 2565
    assert c.next_cell() is None
