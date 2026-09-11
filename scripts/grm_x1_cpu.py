#!/usr/bin/env python3
"""CPU-only author baseline and registered mutation checks.

Prior art: HOUSE_RULES (2026), mutation/invariant verification; mutation
testing antecedents DeMillo/Lipton/Sayward (1978), unverified — lead to check
"Hints on Test Data Selection". Borrow deliberate defect injection; new:
the frozen X1 fault list. Mutants live in /tmp, production source untouched.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import grm_x1_campaign as c
from scripts.grm_x1_register import raw_json, sha

MUTANTS = {
    "off_defaults_on": ('env.get(ENV_NAME, "")', 'env.get(ENV_NAME, "1")'),
    "accept_stale_version": ('if history and version <= history[-1].version:', 'if history and version == history[-1].version:'),
    "partial_page_set_accepted": ('sorted(set(row.page_ids) - set(self.available_pages()))', '[]'),
    "physical_mount_guard_removed": ('if page_fault and tuple(mounted) != row.page_ids:', 'if False:'),
    "relation_identity_collapsed": ('normalize(self.relation)', '"any_relation"'),
}
BASELINE = ["tests/test_grm_x1_reduced.py", "tests/test_grm_x1_units.py", "tests/test_grm_x1_addresses.py", "tests/test_grm_x1_campaign.py",
            "tests/test_grm_rt1_split_child_routing.py", "tests/test_grm_rs3_capture_pin_seat.py"]


def environment():
    return {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(ROOT) + ":/mnt/ForgeRealm/Project-Tensor/tensor_cuda"}


def pytest_run(tests, *, extra_env=None):
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests]
    started = time.monotonic()
    run = subprocess.run(cmd, cwd=ROOT, env={**environment(), **(extra_env or {})},
                         text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return {"command": cmd, "exit_code": run.returncode, "wall_s": time.monotonic() - started, "output": run.stdout}


def tokenizer_gate():
    # Compile only the existing prompt constants/function; avoid loading the
    # GPU-serving module. Actual tokenizer is local-files-only AutoTokenizer.
    from transformers import AutoTokenizer
    tree = ast.parse((ROOT / "scripts/grm_e2e_session.py").read_text())
    selected = [n for n in tree.body if (isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id in ("SYSTEM_PREFIX", "ASSISTANT_FINAL") for t in n.targets))
        or (isinstance(n, ast.FunctionDef) and n.name == "harmony_turn")]
    scope = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])), "census_harmony_contract", "exec"), scope)
    tokenizer = AutoTokenizer.from_pretrained(c.registration()["frame"]["model_dir"], local_files_only=True)
    rows = []
    for family in c.read(c.FIX / "families.json").values():
        for node in family["nodes"]:
            user, assistant = node["text"].split("\nAssistant: ", 1)
            text = scope["harmony_turn"](user.removeprefix("User: "), assistant)
            count = len(tokenizer.encode(text, add_special_tokens=False))
            is_target, is_decoy = node["node_id"] == family["target_node"], node["node_id"] == family["decoy_node"]
            if is_target and count > 96:
                raise ValueError(f"unseatable full current-version source: {family['family_id']} {count}")
            if is_decoy and count <= 96:
                raise ValueError("decoy must require real width-split children")
            rows.append({"family_id": family["family_id"], "node_id": node["node_id"],
                         "target": is_target, "decoy": is_decoy, "tokens": count,
                         "capture_text_sha256": hashlib.sha256(text.encode()).hexdigest()})
    return {"status": "PASS", "evidence_class": "unit test: actual local tokenizer and original Harmony scaffold, no GPU", "rows": rows}


def baseline():
    result = pytest_run(BASELINE)
    # Preserve the result even if a later CPU lane fails.
    test_path = c.emit_receipt("cpu_pytest", {"evidence_class": "unit test", "source_shas": c.source_manifest(), **result})
    if result["exit_code"]:
        print(result["output"])
        raise SystemExit(f"RED CPU baseline: {test_path}")
    fixture_result = c.verify_fixtures()
    tokens = tokenizer_gate()
    dry = c.dry_run()
    syntax = subprocess.run(["bash", "-n", "scripts/grm_x1_lead_gpu.sh"], cwd=ROOT, capture_output=True, text=True)
    if syntax.returncode:
        raise RuntimeError(syntax.stderr)
    payload = {"status": "PASS", "evidence_class": "unit test", "pytest_receipt": str(test_path.relative_to(ROOT)),
               "pytest_sha256": sha(test_path), "fixtures": fixture_result, "tokenizer": tokens,
               "dry_run": dry, "shell_syntax": "PASS", "source_shas": c.source_manifest()}
    path = c.emit_receipt("cpu_baseline", payload)
    print(result["output"])
    print(json.dumps({"receipt": str(path), "status": "PASS"}))


def register_mutants():
    payload = {"schema": "grm.x1.cpu-mutations.v1", "evidence_class": "reasoning: registration before mutation gate",
               "registration_sha256": sha(c.OUT / "registration.json"), "mutants": MUTANTS,
               "minimum_kill_fraction": .80, "rule": "Run only after passing baseline. Exclude syntax/import errors from denominator, require all 5 runnable. Original module never edited."}
    from scripts.grm_x1_register import create
    create(c.OUT / "cpu_mutation_registration.json", raw_json(payload))
    print(json.dumps(payload, indent=2))


def mutation_gate():
    registration = c.read(c.OUT / "cpu_mutation_registration.json")
    if registration["mutants"] != {k: list(v) for k, v in MUTANTS.items()}:
        raise ValueError("mutation definition drift")
    baselines = sorted((c.OUT / "receipts").glob("cpu_baseline_*.json"))
    if not any(c.read(p)["status"] == "PASS" and c.read(p)["source_shas"] == c.source_manifest() for p in baselines):
        raise ValueError("mutation requires a passing baseline on these exact sources")
    path = ROOT / "core/grm_x1_addresses.py"
    original, original_sha = path.read_text(), sha(path)
    rows = []
    with tempfile.TemporaryDirectory(prefix="grm_x1_mutants_") as directory:
        for name, (before, after) in MUTANTS.items():
            if original.count(before) != 1:
                raise ValueError(f"mutant {name} patch does not identify exactly one site")
            mutant = Path(directory) / f"{name}.py"
            mutant.write_text(original.replace(before, after))
            compile(mutant.read_text(), str(mutant), "exec")
            result = pytest_run(["tests/test_grm_x1_addresses.py"], extra_env={"GRM_X1_TEST_MODULE": str(mutant)})
            runnable = result["exit_code"] in (0, 1) and "ERROR collecting" not in result["output"]
            rows.append({"name": name, "runnable": runnable, "killed": runnable and result["exit_code"] == 1,
                         "mutant_sha256": sha(mutant), **result})
    count = sum(r["runnable"] for r in rows)
    killed = sum(r["killed"] for r in rows)
    passed = count == len(MUTANTS) and killed / count >= .80 and sha(path) == original_sha
    receipt = c.emit_receipt("cpu_mutations", {"status": "PASS" if passed else "RED",
        "evidence_class": "unit test: author mutation gate", "runnable": count, "killed": killed,
        "kill_fraction": killed / count if count else None, "minimum": .80,
        "original_sha256_before": original_sha, "original_sha256_after": sha(path),
        "registration_sha256": sha(c.OUT / "cpu_mutation_registration.json"), "rows": rows})
    print(json.dumps({"receipt": str(receipt), "runnable": count, "killed": killed, "passed": passed}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("baseline", "register-mutants", "mutations"))
    command = parser.parse_args().command
    {"baseline": baseline, "register-mutants": register_mutants, "mutations": mutation_gate}[command]()
