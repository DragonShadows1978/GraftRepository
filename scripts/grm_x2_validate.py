#!/usr/bin/env python3
"""One create-only CPU verification run; never starts a GPU worker.

Prior art: house §4/§8 (2026), baseline then mutation checks and receipts.
Ours: X2-scoped orchestration, no new testing algorithm.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.grm_x2_common import OUT, contract, fingerprint, create_json, sha_file, read
from scripts.grm_x2_gpu import plan


def foreground(command, log_name):
    path = ROOT / "logs" / log_name
    path.parent.mkdir(exist_ok=True)
    started = time.monotonic()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
    with path.open("x") as f:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        code = proc.wait()
    output = path.read_text()
    print(output, flush=True)
    return {"command": command, "exit_code": code, "wall_s": time.monotonic() - started,
            "log": str(path.relative_to(ROOT)), "sha256": sha_file(path)}


def main():
    contract()
    if (OUT / "validation.json").exists():
        raise RuntimeError("validation receipt is immutable; use an amendment for any successor")
    fp, records = fingerprint()
    tests = foreground([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                        "tests/test_grm_x2_witnesses.py", "tests/test_grm_x2_runner.py",
                        "tests/test_grm_sc1_1_grounding_glyphs.py", "tests/test_grm_eb1_ephemeral_frame.py"], "grm_x2_unit_gate.log")
    shell = foreground(["bash", "-n", "scripts/grm_x2_lead_gpu.sh"], "grm_x2_shell_gate.log")
    create_json(OUT / "dry_run.json", plan())
    cpu = foreground([sys.executable, "scripts/grm_x2_cpu.py"], "grm_x2_cpu_gate.log") if tests["exit_code"] == 0 else None
    mutations = foreground([sys.executable, "scripts/grm_x2_mutations.py"], "grm_x2_mutation_gate.log") if cpu and cpu["exit_code"] == 0 else None
    status = "PASS" if all(r is not None and r["exit_code"] == 0 for r in (tests, shell, cpu, mutations)) else "RED"
    final_fp, _ = fingerprint()
    if final_fp != fp:
        raise RuntimeError("code drift during verification")
    receipt = {"evidence_class": "unit test / CPU suite and frozen falsifier; NO GPU", "status": status,
               "fingerprint": fp, "fingerprint_inputs": records, "tests": tests, "shell": shell, "cpu": cpu, "mutations": mutations,
               "dry_run": {"path": "artifacts/grm_x2/dry_run.json", "sha256": sha_file(OUT / "dry_run.json"), "questions": 24, "arms": 48, "cells": 6},
               "gpu_status": "BLOCKED_NO_GPU_IN_SEAT", "blind_verification": "NOT RUN; lead dispatch required"}
    create_json(OUT / "validation.json", receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k not in {"fingerprint_inputs", "tests", "cpu", "mutations"}}, indent=2))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
