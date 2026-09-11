#!/usr/bin/env python3
"""Create-only validation for metric addendum 001; no GPU work.

Prior art: house immutable amendments/receipt discipline (2026), reused.
Ours: second fingerprint receipt; no algorithm or gate threshold change.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.grm_x2_common import OUT, contract, fingerprint, create_json, sha_file, read
from scripts.grm_x2_validate import foreground
from scripts.grm_x2_cpu import run
from scripts.grm_x2_gpu import plan


def main(revision="001"):
    contract()
    if revision not in {"001", "002"}:
        raise ValueError("unregistered revision")
    amendment = OUT / "amendments" / ("001_metric_audit.json" if revision == "001" else "002_turn_geometry.json")
    if (OUT / f"validation_{revision}.json").exists():
        raise RuntimeError("addendum receipt exists and is immutable")
    fp, records = fingerprint()
    tests = foreground([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                        "tests/test_grm_x2_witnesses.py", "tests/test_grm_x2_runner.py",
                        "tests/test_grm_sc1_1_grounding_glyphs.py", "tests/test_grm_eb1_ephemeral_frame.py"], f"grm_x2_unit_gate_{revision}.log")
    result = run()
    create_json(OUT / f"cpu_gate_{revision}.json", result)
    mutation = read(OUT / "mutation_gate.json")
    mutation_valid = mutation["status"] == "PASS" and mutation["production_sha256_unchanged"] == sha_file(ROOT / "core/grm_x2_witnesses.py")
    shell = foreground(["bash", "-n", "scripts/grm_x2_lead_gpu.sh"], f"grm_x2_shell_gate_{revision}.log")
    create_json(OUT / f"dry_run_{revision}.json", plan())
    status = "PASS" if tests["exit_code"] == 0 and shell["exit_code"] == 0 and result["status"] == "PASS" and mutation_valid else "RED"
    assert fingerprint()[0] == fp
    receipt = {"evidence_class": "unit test / CPU addendum audit; NO GPU", "status": status, "fingerprint": fp,
               "fingerprint_inputs": records, "amendment_sha256": sha_file(amendment), "tests": tests, "shell": shell,
               "cpu": {"path": f"artifacts/grm_x2/cpu_gate_{revision}.json", "sha256": sha_file(OUT / f"cpu_gate_{revision}.json")},
               "mutation_reused": {"path": "artifacts/grm_x2/mutation_gate.json", "sha256": sha_file(OUT / "mutation_gate.json"), "unchanged_module_verified": mutation_valid},
               "dry_run": {"path": f"artifacts/grm_x2/dry_run_{revision}.json", "sha256": sha_file(OUT / f"dry_run_{revision}.json")},
               "gpu_status": "BLOCKED_NO_GPU_IN_SEAT", "blind_verification": "NOT RUN; lead dispatch required"}
    create_json(OUT / f"validation_{revision}.json", receipt)
    print(json.dumps({"status": status, "fingerprint": fp, "matrices": result["confusion_matrices"], "mutation_unchanged": mutation_valid}, indent=2))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "001"))
