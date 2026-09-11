"""C5 offline cells, immutable receipts, and bounded CPU gates.

Prior art: GRM DET/SC/WC sha-bound cells (2026), imported comparator and
production grounding. Ours: four-arm table, missing-data reporting and gates.
Mutation testing lead: DeMillo, Lipton and Sayward (1978), unverified — lead
to check; borrowed deliberate defects, ours five registered mutations.
"""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.grm_c5_register import OUT, digest, record, write_new
from scripts.grm_c5_rules import ARMS, TextArena, binding_candidates, grounding_verdict
from scripts.grm_det1_common import contains_value


def registration():
    path = OUT / "registration.json"
    expected = (OUT / "registration.sha256").read_text().split()[0]
    if digest(path) != expected:
        raise RuntimeError("registration SHA-256 mismatch")
    reg = json.loads(path.read_text())
    for item in [reg["fixtures"], reg["protected_files"], reg["order"], *reg["inputs"]]:
        if digest(item["path"]) != item["sha256"]:
            raise RuntimeError(f"registered input changed: {item['path']}")
    return reg


def loaded_sources():
    # Prior art: GRM file_record receipts (2026). Include every loaded local
    # Python module, plus this executable even when invoked as __main__.
    paths = {Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path and Path(path).suffix == ".py":
            path = Path(path).resolve()
            if path.is_relative_to(ROOT) and path.is_file():
                paths.add(path)
    return [record(path) for path in sorted(paths)]


def receipt(kind, data):
    unresolved = []
    for name, module in tuple(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if path and Path(path).suffix == ".py" and not Path(path).is_file():
            unresolved.append({"module": name, "declared_file": str(path)})
    return {"schema": "grm.c5.receipt.v1", "cell": kind,
            "registration": record(OUT / "registration.json"),
            "executed_sources": loaded_sources(),
            "unresolvable_module_file_metadata": unresolved,
            "python": sys.version, "executable": sys.executable,
            "evidence_class": "CPU finite validation", "gpu_seconds": 0,
            **data}


def load_fixtures():
    registration()
    return json.loads((OUT / "fixtures.json").read_text())


def score_fixture(row, arm):
    # Prior art: DET1 contains_value (GRM, 2026) imported unchanged. It is
    # correctness scoring only; grounding never receives the expected value.
    expected_hit = any(contains_value(row["answer"], v) for v in row["expected"])
    if row["texts"] is None:
        return {"grounded": None, "contributors": [], "expected_value_hit": expected_hit,
                "status": "BLOCKED", "reason": row["blocked_reason"]}
    grounded, contributors = grounding_verdict(
        row["answer"], row["texts"], row["question"], rule=arm)
    return {"grounded": grounded, "contributors": sorted(contributors),
            "expected_value_hit": expected_hit, "status": "SCORED"}


def offline(output):
    started = time.monotonic()
    fixtures = load_fixtures()
    rows = []
    for fixture in fixtures:
        arms = {arm: score_fixture(fixture, arm) for arm in ARMS}
        rows.append({"id": fixture["id"], "class": fixture["class"],
                     "kind": fixture.get("kind"), "evidence": fixture["evidence"],
                     "arms": arms})
    indexed = {row["id"]: row for row in rows}
    controls = [r for r in rows if r["class"] in ("S", "N")]
    new_false = {a: [r["id"] for r in controls if r["arms"][a]["grounded"]
                    and not r["arms"]["0"]["grounded"]] for a in ARMS}
    inherited = [r["id"] for r in controls if r["arms"]["0"]["grounded"]]
    regressions = {a: [r["id"] for r in rows if r["class"] == "original"
                      and r["arms"]["0"]["grounded"] is True
                      and r["arms"][a]["grounded"] is False] for a in ARMS}
    recovered = {}
    for arm, target in (("S", "SC1:t30"), ("N", "EB1:t33")):
        row = indexed[target]["arms"]
        recovered[arm] = (row["0"]["grounded"] is False
                          and row[arm]["grounded"] is True
                          and row[arm]["expected_value_hit"])
    w_hit = any(r["id"] in new_false["W"] and r["kind"] in
                ("scattered_word", "alias_collision") for r in controls)
    blocked = [r["id"] for r in rows if r["arms"]["0"]["grounded"] is None]
    gates = {"S_recovery": recovered["S"], "N_recovery": recovered["N"],
             "S_zero_new_false_acceptances": not new_false["S"],
             "N_zero_new_false_acceptances": not new_false["N"],
             "S_no_regressions": not regressions["S"],
             "N_no_regressions": not regressions["N"],
             "W_negative_control_prediction": w_hit,
             "full_grounding_input_coverage": not blocked}
    audits = []
    for f in fixtures:
        if f.get("turn") != 33:
            continue
        texts = [f["source_node_text"]]
        audits.append({"fixture": f["id"], "source_node_id": f["source_node_id"],
                       "source_excluded_live": f["source_node_id"] in f["excluded_live_ids"],
                       "recorded_mounts": f["mounted_ids"], "infer_calls": f["infer_calls"],
                       "abstain_reason": f["abstain_reason"],
                       "source_binds_if_eligible_0": bool(binding_candidates(f["question"], texts)),
                       "source_binds_if_eligible_N": bool(binding_candidates(f["question"], texts, rule="N")),
                       "scope": "candidate predicate only; no eligibility change, mount or generation"})
    result = receipt("offline", {
        "rows": rows, "gates": gates, "science_status": "GREEN" if all(gates.values()) else "RED",
        "new_false_acceptances": new_false, "inherited_false_acceptances": inherited,
        "regressions": regressions, "blocked_fixtures": blocked, "t33_binding_audit": audits,
        "fixture_count": len(rows), "cell_count": 4 * len(rows),
        "scored_cells": 4 * (len(rows)-len(blocked)), "blocked_cells": 4 * len(blocked),
        "wall_seconds": time.monotonic()-started,
        "gpu_cells": [], "gpu_decision": "No missing served texts. RT1 lacks mount text, not answers. Regeneration condition is not met.",
        "claim_scope": "A passing rule is a narrower grounding rule for a named class, not a prose-grounding certificate.",
    })
    write_new(output, result)
    table = ["| Fixture | 0 | S | N | W |", "|---|---:|---:|---:|---:|"]
    for row in rows:
        cells = ["BLOCKED" if row["arms"][a]["grounded"] is None
                 else "accept" if row["arms"][a]["grounded"] else "reject" for a in ARMS]
        table.append("| " + " | ".join([row["id"], *cells]) + " |")
    with output.with_suffix(".md").open("x") as stream:
        stream.write("Grounding verdicts only; value correctness is separate in JSON.\n\n" + "\n".join(table) + "\n")
    print(json.dumps({k: result[k] for k in ("science_status", "gates", "new_false_acceptances", "scored_cells", "blocked_cells")}))
    return 0 if all(gates.values()) else 2


def dry_run(output):
    reg = registration()
    result = receipt("dry-run", {"cells": reg["offline_cells"] + [
        {"id": "cpu-gate", "device": "CPU", "wall_estimate_s": 90},
        {"id": "mutation-gate", "device": "CPU", "wall_estimate_s": 20},
        {"id": "integrity", "device": "CPU", "wall_estimate_s": 5}],
        "gpu_cells": [], "gpu_hours_estimate": 0,
        "offline_process_import_overhead_s": 5,
        "total_wall_estimate_s": len(reg["offline_cells"])*0.05+120})
    write_new(output, result)
    print(f"Enumerated {len(result['cells'])} cells in {output}; GPU estimate 0 h")
    return 0


def cpu_gate(output):
    reg = registration()
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "",
           "PYTHONPATH": str(ROOT), "GRM_C5_PYTEST_RECEIPT": str(output)}
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
               "-p", "scripts.grm_c5_pytest_receipt", *reg["cpu_gate"]["tests"]]
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
    with output.with_suffix(".log").open("x") as stream:
        stream.write(result.stdout + result.stderr)
    if not output.exists():
        write_new(output, receipt("cpu-gate-driver-failure", {
            "exitstatus": result.returncode, "tests_collected": 0,
            "collection_count_note": "unknown: pytest hook did not emit receipt",
            "command": command, "log": record(output.with_suffix(".log")),
            "passed": False}))
    print(result.stdout[-4500:] + result.stderr[-2000:])
    return result.returncode


def mutation_checks(module):
    # Prior art: registered contrast sets and default pin, GRM C5 (2026).
    # These fixed assertions test consequences, not source-string identity.
    failures = []
    if inspect.signature(module.grounding_verdict).parameters["rule"].default != "0":
        failures.append("default_pin")
    q = "What is the current atlas tone value?"
    source = ["The current atlas tone value is Cobalt-1-India."]
    if not module.grounding_verdict("Cobalt 1 India", source, q, rule="S")[0]:
        failures.append("S_recovery")
    for row in load_fixtures():
        if row["class"] in ("S", "N"):
            base = grounding_verdict(row["answer"], row["texts"], row["question"])[0]
            for a in ("S", "N"):
                if not base and module.grounding_verdict(row["answer"], row["texts"], row["question"], rule=a)[0]:
                    failures.append(row["id"]+":"+a)
    scattered = ["The current atlas tone value is Amber-1-Echo. Cobalt visited. India was next."]
    if not module.grounding_verdict("Cobalt 1 India", scattered, q, rule="W")[0]:
        failures.append("W_prediction")
    if not module.grounding_verdict("The Polaris mark value is Marble-4-Juliet.",
                                    ["The current polaris mark value is Marble-4-Juliet."],
                                    "What is the current polaris mark value?", rule="N")[0]:
        failures.append("N_bound_name")
    return failures


def mutations(output, baseline):
    registration()
    cpu = json.loads(baseline.read_text())
    if cpu["exitstatus"] != 0 or cpu["tests_collected"] == 0:
        raise RuntimeError("Mutation gate requires a passing nonempty CPU baseline")
    import scripts.grm_c5_rules as original
    assert not mutation_checks(original), "Unmutated mutation-check baseline failed"
    source = (ROOT / "scripts/grm_c5_rules.py").read_text()
    edits = {
        "default_S": ('DEFAULT_RULE = "0"', 'DEFAULT_RULE = "S"'),
        "accept_all": ('return bool(contributors), contributors', 'return True, {0}'),
        "S_uses_N": ('separator_fold=rule == "S"', 'separator_fold=False'),
        "W_uses_baseline": ('if rule == "W":', 'if rule == "W":\n        return baseline'),
        "N_uses_baseline": ('contributors = _span_support(', 'if rule == "N":\n        return baseline\n    contributors = _span_support('),
    }
    rows = []
    with tempfile.TemporaryDirectory(prefix="grm_c5_mutants_") as directory:
        for name, (old, new) in edits.items():
            assert source.count(old) == 1
            path = Path(directory) / (name+".py")
            path.write_text(source.replace(old, new))
            spec = importlib.util.spec_from_file_location("c5_mutant_"+name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            failures = mutation_checks(module)
            rows.append({"mutant": name, "executed_copy": record(path),
                         "edit": {"old": old, "new": new},
                         "killed": bool(failures), "failed_checks": failures, "error_mutant": False})
    fraction = sum(r["killed"] for r in rows)/len(rows)
    result = receipt("mutation-gate", {"baseline": record(baseline), "rows": rows,
                                       "kill_fraction": fraction, "passed": fraction >= 0.8,
                                       "source_unchanged": digest(ROOT / "scripts/grm_c5_rules.py") == __import__("hashlib").sha256(source.encode()).hexdigest()})
    write_new(output, result)
    print(json.dumps({"kill_fraction": fraction, "passed": result["passed"]}))
    return 0 if result["passed"] else 1


def integrity(output):
    registration()
    protected = json.loads((OUT / "protected_files.json").read_text())
    changed = [r["path"] for r in protected if digest(r["path"]) != r["sha256"]]
    write_new(output, receipt("integrity", {"protected_count": len(protected),
                                           "changed": changed, "passed": not changed}))
    print(json.dumps({"protected_count": len(protected), "changed": changed}))
    return bool(changed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    for flag in ("dry-run", "offline", "cpu-gate", "mutations", "integrity"):
        group.add_argument("--"+flag, action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Immutable receipt already exists: {args.output}")
    if args.dry_run:
        return dry_run(args.output)
    if args.offline:
        return offline(args.output)
    if args.cpu_gate:
        return cpu_gate(args.output)
    if args.mutations:
        return mutations(args.output, args.baseline)
    return integrity(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
