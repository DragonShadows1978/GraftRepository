#!/usr/bin/env python3
"""GRM-C3 CPU evaluator and foreground bounded trace runner.

Prior art: local DET1/SC2/SC1.2 (GraftRepository contributors, 2026) supplies
planting, observation, answer matching, first crossing and minimum reduction.
This is a new experiment adapter, not a new attention detector. External
priority is unverified — lead to check: attention-based uncertainty detection,
selective prediction, Fisher 1935 blocked experiments, NIST SHA-256 2015.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.grm_c3_register import digest
from scripts import grm_sc2_calibration as sc2
from scripts import grm_sc1_2_session as sc12

OUT = ROOT / "artifacts/grm_c3"
REG = OUT / "registration.json"
PINS = OUT / "execution_manifest.json"
NATIVE = Path("/mnt/ForgeRealm/GraftRepository/cpp/build/libgrm_runtime.so")
TENSOR = Path("/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
MODEL = Path("/home/vader/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee")
ENV = {"GRM_DEMAND_NGH": "0", "GRM_DEMAND_EARLY_ABORT": "0",
       "GRM_PERSISTENT_BOAT": "0", "GRM_CAPTURE_PIN": "off",
       "GRM_SEAT_NEAR_LIVE": "0", "GRM_ADM_DECISIVE": "1",
       "GRM_LSR_FIXES": "1", "GRM_RT1_RULE": "1", "CUDA_VISIBLE_DEVICES": "0"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def record(path):
    # Prior art: NIST SHA-256 (2015); chunked streaming avoids loading model
    # shards into RAM. No new hashing or content-addressing algorithm.
    path = Path(path).absolute()
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return {"path": str(path), "sha256": sha.hexdigest(), "bytes": path.stat().st_size}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def registration():
    require(record(REG)["sha256"] == (OUT / "registration.sha256").read_text().split()[0],
            "registration SHA drift")
    reg = json.loads(REG.read_text())
    require(record(ROOT / reg["order"])["sha256"] == reg["order_sha256"], "order drift")
    return reg


def pin_sources(path):
    # Prior art: immutable manifests from DET1 (2026); local full source-tree
    # pins cover transitive project imports, runtime libraries and model bytes.
    paths = {REG, ROOT / registration()["order"], ROOT / "config/grm_demand_registered.json",
             Path(__file__), ROOT / "scripts/grm_c3_register.py", NATIVE}
    for folder in (ROOT / "scripts", ROOT / "core", TENSOR):
        paths.update(folder.rglob("*.py"))
        paths.update(folder.rglob("*.so"))
    paths.update((ROOT / "tests").glob("test_grm_c3*.py"))
    paths.update(p for p in MODEL.iterdir() if p.is_file())
    write(path, {"schema": "grm.c3.execution_manifest.v1",
                 "registration": record(REG), "files": [record(p) for p in sorted(paths)],
                 "python": sys.executable, "python_version": sys.version,
                 "environment": ENV, "evidence_class": "source and model byte inventory"})


def check_pins(path, *, include_model=True):
    manifest = json.loads(Path(path).read_text())
    require(manifest["registration"] == record(REG), "manifest registration mismatch")
    for item in manifest["files"]:
        if not include_model and str(item["path"]).startswith(str(MODEL) + "/"):
            continue
        require(record(item["path"]) == item, f"execution input drift: {item['path']}")
    return record(path)


def fixture_gate(reg):
    # Prior art: DET1 planted-fixture guards (2026). New negative controls
    # require oracle absence from BOTH repository and prompt, never withhold
    # a still-deposited target and call it unavailable knowledge.
    cases = reg["cases"]
    require(len(cases) == 48, "expected 48 cases")
    require(Counter(c["class"] for c in cases) == Counter(reg["class_counts"]), "class balance")
    require(len({c["id"] for c in cases}) == 48, "duplicate case id")
    require(len({c["question"] for c in cases}) == 48, "duplicate question")
    require(digest(cases) == reg["case_set_sha256"], "case set SHA drift")
    # Search all existing fixture JSON plus prior grm scripts, not only SC2 fit.
    corpus_paths = sorted(set((ROOT / "tests/fixtures").rglob("*.json")) |
                          set((ROOT / "scripts").glob("grm_*.py")))
    corpus_paths = [p for p in corpus_paths if not p.name.startswith("grm_c3")]
    corpus = "\n".join(p.read_text(errors="replace") for p in corpus_paths)
    for c in cases:
        require(digest({k: v for k, v in c.items() if k != "sha256"}) == c["sha256"], "case SHA drift")
        require(c["id"] not in corpus and c["question"] not in corpus and c["oracle_fact"] not in corpus,
                f"fixture not new: {c['id']}")
        require(c["id"] not in {"e2e_t09_cypher_bridge", "e2e_t16_lyra_dock"}, "SC2 fit fixture")
        require(all(n["text"] not in corpus for n in c["nodes"]), "old node text")
        values = c["expected_values"]
        require(not any(v.casefold() in c["question"].casefold() for v in values), "answer in prompt")
        if c["class"] == "unavailable":
            require(c["nodes"] == [] and c["mount_node_ids"] == [], "unavailable has deposits")
        else:
            require(len(c["nodes"]) == 1 and c["mount_node_ids"] == [c["nodes"][0]["node_id"]], "wrong planned mounts")
            node = c["nodes"][0]
            require(c["entity"] in node["text"], "node not topically matched")
            present = any(v.casefold() in node["text"].casefold() for v in values)
            require(present == (c["class"] in {"correct", "weak"}), "target presence mismatch")
            require(node["role"] == ("decoy" if c["class"] == "decoy" else "target"), "node role mismatch")
        if c["class"] == "weak":
            require(c["weakness_target"] in {"separator_number", "proper_name"}, "missing weakness target")
    assigned = [cid for cell in reg["cells"] for cid in cell["case_ids"]]
    require(Counter(assigned) == Counter(c["id"] for c in cases), "cell coverage")
    require(sum(c["worker_cap_seconds"] for c in reg["cells"]) <= reg["gpu_budget_seconds"], "GPU budget")
    require(all(c["estimate_seconds"] <= c["worker_cap_seconds"] <= 285 and c["outer_cap_seconds"] <= 590
                for c in reg["cells"]), "non-fit planned cell")
    return {"passed": True, "cases": 48, "class_counts": dict(Counter(c["class"] for c in cases)),
            "novelty_inputs": [record(p) for p in corpus_paths],
            "weakness": "targeted, empirical weakness and correct reading require GPU evidence"}


def decisions(masses, mounted_tokens, rules):
    # Prior art: DET1/SC2 strict any-token crossing, imported unchanged.
    # The empty-band baseline is the Scout (2026) structural null check;
    # 1e-8 is this registration's tolerance, not a fitted envelope.
    require(isinstance(mounted_tokens, int) and not isinstance(mounted_tokens, bool) and mounted_tokens >= 0, "bad mounted token count")
    require(bool(masses), "empty trace is not zero mass")
    require(all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and 0 <= v <= 1 for v in masses), "invalid mass")
    require(mounted_tokens != 0 or all(v == 0 for v in masses), "empty band has nonzero mounted mass")
    out = {}
    for name, rule in rules.items():
        if name == "empty_band":
            fire = mounted_tokens == 0
            out[name] = {"fire": fire, "first_index": 0 if fire else None,
                         "timing": "pre-generation structural decision"}
        else:
            idx = sc2._first_below(masses, rule["threshold"])
            out[name] = {"fire": idx is not None, "first_index": idx}
    return out


def metrics(rows, reg):
    # Prior art: classical confusion matrix and selective prediction;
    # external lead: Chow (1970) optimum recognition error/reject tradeoff.
    # Imported matching supplies OFF correctness; at-risk is NOT measured loss.
    cases = {c["id"]: c for c in reg["cases"]}
    require(Counter(r["id"] for r in rows) == Counter(cases.keys()), "missing/duplicate/extra cases")
    per_case = []
    counts = {name: dict(TP=0, FP=0, TN=0, FN=0, correct_answers_at_risk=0) for name in reg["rules"]}
    qualification_failures = []
    for row in rows:
        c = cases[row["id"]]
        require(type(row["answer_correct"]) is bool, "correctness must be boolean")
        require(type(row["qualification_pass"]) is bool, "qualification must be boolean")
        masses = [t["ngh"]["mounted_mass"] for t in row["signals"]["tokens"]]
        verdicts = decisions(masses, row["mounted_token_count"], reg["rules"])
        require(sc12.min_mounted_mass(row) == min(masses), "SC1.2 minimum parity")
        miss = c["class"] in {"decoy", "unavailable"}
        if not row["qualification_pass"]:
            qualification_failures.append(row["id"])
        for name, d in verdicts.items():
            key = ("TP" if d["fire"] else "FN") if miss else ("FP" if d["fire"] else "TN")
            counts[name][key] += 1
            counts[name]["correct_answers_at_risk"] += int(d["fire"] and row["answer_correct"])
        per_case.append({"id": c["id"], "class": c["class"], "answer": row["answer"],
                         "answer_correct": row["answer_correct"], "min_mass": min(masses),
                         "mounted_tokens": row["mounted_token_count"], "decisions": verdicts})
    for name, value in counts.items():
        value["miss_detection"] = value["TP"] / 24
        value["false_fire_rate"] = value["FP"] / 24
        value["detection_bar_pass"] = value["TP"] >= 22 and value["FP"] <= 1
        value["conservative_offline_eligible"] = value["detection_bar_pass"] and value["correct_answers_at_risk"] == 0 and not qualification_failures
        value["lost_correct_answers"] = None
        value["full_demand_flip_bar"] = "UNMEASURED: demand-ON preservation"
    separating = [r["id"] for r in per_case if len({d["fire"] for d in r["decisions"].values()}) > 1]
    weak_avoidable = [r["id"] for r in per_case if r["class"] == "weak" and r["answer_correct"]
                      and r["decisions"]["carried"]["fire"] and any(not d["fire"] for n, d in r["decisions"].items() if n != "carried")]
    decoy_misses = [r["id"] for r in per_case if r["class"] == "decoy" and r["mounted_tokens"] > 0 and not r["decisions"]["near_zero"]["fire"]]
    eligible = [n for n, v in counts.items() if v["conservative_offline_eligible"]]
    return {"evidence_class": "offline CPU evaluation of detector-OFF GPU traces; finite 48-case controlled-mount experiment",
            "rules": counts, "per_case": per_case, "separating_cases": separating,
            "qualification_failures": qualification_failures,
            "prediction_evidence": {"near_zero_nonempty_decoy_misses": decoy_misses,
                                    "weak_avoidable_carried_fires": weak_avoidable,
                                    "near_zero_detects_all_empty": all(r["decisions"]["near_zero"]["fire"] for r in per_case if r["mounted_tokens"] == 0)},
            "recommendation": (f"Offline candidates: {eligible}. Demand flip remains uncertified without demand-ON preservation evidence." if eligible else "No rule meets the conservative offline bar. Do not flip demand."),
            "threshold_caveat": reg["threshold_caveat"]}


def observe_attempt(arena, case, picks, observer_type, ngen):
    # Prior art: DET1 DetectorObserver + production ArenaCache._attempt (2026).
    # Explicit reset implements EB1 for this controlled-mount invocation.
    arena.reset_live_cache()
    with observer_type(arena, set(picks), ngen, active_detectors=frozenset({"D-NGH"})) as observer:
        answer, _info = arena._attempt(case["question"], picks, ngen, False,
                                      arena.stop_sequences or (), defer_memory=True)
    return answer, observer.finish()


def case_worker(case_id, pins):
    reg = registration()
    c = next(c for c in reg["cases"] if c["id"] == case_id)
    pin_record = check_pins(pins, include_model=False)
    require(os.environ.get("GRM_C3_LEASE_PARENT") == str(os.getppid()), "case worker requires its owning leased parent")
    os.environ.update(ENV)
    from scripts import grm_det1_2_gpu as loader
    from scripts import grm_det1_5_workers as workers
    from scripts.grm_det1_3_gpu import _install_lived_nodes
    from scripts.grm_det1_common import DetectorObserver
    dest = OUT / "runs" / case_id
    dest.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    model = tokenizer = repo = None
    try:
        # Temporary imported-loader binding; no library/source/config edits.
        with patch.object(loader, "NATIVE_LIB", NATIVE):
            e2e, model, tokenizer, repo, info = loader._load_model_repo(dest, {"resolved_flags": {"adm_decisive": True}})
        arena = repo.arena
        require(arena.ephemeral, "EB1 frame not active")
        node_map, token_ledgers = _install_lived_nodes(repo, e2e, c)
        picks = [node_map[n] for n in c["mount_node_ids"]]
        require(sum(arena.grafts[i]["ntok"] for i in picks) <= arena.width, "registered node exceeds arena; NON_FIT")
        answer, signals = observe_attempt(arena, c, picks, DetectorObserver, reg["trace"]["ngen"])
        correct = workers.campaign._answer_correct(answer, c)
        mounted = list(arena.cur_mounts)
        require(mounted == picks, "actual mounts differ from registered intervention")
        require(len(arena.grafts) == len(c["nodes"]), "unexpected repository content")
        tokens = signals["tokens"]
        require(0 < len(tokens) <= reg["trace"]["ngen"], "no answer token trace")
        require(all(t["full_attention_layers"] == observer_layers(model) for t in tokens), "incomplete full layer capture")
        decisions([t["ngh"]["mounted_mass"] for t in tokens], int(arena.cur_mount_n), reg["rules"])
        row = {"id": case_id, "class": c["class"], "case_sha256": c["sha256"],
               "registration": record(REG), "execution_manifest": pin_record,
               "answer": answer, "answer_correct": correct, "signals": signals,
               "mounted_ids": mounted, "mounted_node_ids": c["mount_node_ids"],
               "mounted_token_count": int(arena.cur_mount_n), "repository_node_ids": list(node_map),
               "token_ledgers": token_ledgers,
               "qualification_pass": bool(correct or c["class"] in {"decoy", "unavailable"}),
               "weakness_empirically_confirmed": None,
               "detector_on": False, "model": reg["model"], "model_info": info,
               "environment": ENV, "pid": os.getpid(), "python": sys.version,
               "elapsed_seconds": time.monotonic() - started,
               "threshold_caveat": reg["threshold_caveat"]}
        write(dest / "trace.json", row)
    finally:
        workers._close_model(repo, model, tokenizer)


def observer_layers(model):
    return sum(l.self_attn.layer_type == "full_attention" for l in model.layers)


def run_cell(cell_id, pins):
    # Prior art: DET1 leased workers and POSIX flock; Python subprocess
    # timeout kills/waits only the child this invocation starts. New wrapper
    # shares a 225s TOTAL deadline across four fresh case processes. Never
    # clear/unlink a lock; fail immediately if occupied. No background work.
    reg = registration()
    cell = next(c for c in reg["cells"] if c["id"] == cell_id)
    pin_record = check_pins(pins)
    require((OUT / "cpu_gate.json").is_file(), "CPU gate receipt missing")
    gate = json.loads((OUT / "cpu_gate.json").read_text())
    require(gate["passed"] and gate["execution_manifest"] == pin_record, "CPU gate not passed for these sources")
    prior = reg["cells"][:reg["cells"].index(cell)]
    for p in prior:
        receipt = json.loads((OUT / "cells" / f"{p['id']}.json").read_text())
        require(receipt["status"] == "COMPLETE" and receipt["execution_manifest"] == pin_record, "predecessor not complete for these sources")
    claim = OUT / "cells" / f"{cell_id}.started.json"
    require(not claim.exists(), "cell already attempted; no retry permitted")
    # Foreground cooldown belongs outside the lease. It never holds the GPU.
    time.sleep(cell["cooldown_seconds"])
    with Path("/tmp/forge-gpu.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        start = time.monotonic()
        write(claim, {"cell": cell, "registration": record(REG), "execution_manifest": pin_record,
                      "pid": os.getpid(), "started_unix": time.time()})
        result = {"cell": cell, "registration": record(REG), "execution_manifest": pin_record,
                  "status": "COMPLETE", "traces": [], "threshold_caveat": reg["threshold_caveat"]}
        try:
            for case_id in cell["case_ids"]:
                remaining = cell["worker_cap_seconds"] - (time.monotonic() - start)
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(case_id, cell["worker_cap_seconds"])
                log = OUT / "cells" / f"{case_id}.log"
                env = {**os.environ, **ENV, "GRM_C3_LEASE_PARENT": str(os.getpid()), "PYTHONDONTWRITEBYTECODE": "1"}
                with log.open("x") as stream:
                    proc = subprocess.run([sys.executable, str(Path(__file__)), "--case-worker", case_id,
                                           "--pins", str(pins)], cwd=ROOT, env=env,
                                          stdout=stream, stderr=subprocess.STDOUT, timeout=remaining)
                require(proc.returncode == 0, f"case child exit {proc.returncode}; see {log}")
                result["traces"].append(record(OUT / "runs" / case_id / "trace.json"))
        except subprocess.TimeoutExpired as exc:
            result.update(status="NON_FIT", error=str(exc), retry_permitted=False)
        except Exception as exc:
            result.update(status="RED", error=f"{type(exc).__name__}: {exc}", retry_permitted=False)
        finally:
            result["lease_elapsed_seconds"] = time.monotonic() - start
            fcntl.flock(lock, fcntl.LOCK_UN)
    # Revalidate source bytes after execution outside the compute lease.
    try:
        check_pins(pins, include_model=False)
    except Exception as exc:
        result.update(status="RED", error=str(exc))
    write(OUT / "cells" / f"{cell_id}.json", result)
    require(result["status"] == "COMPLETE", result.get("error", "cell failed"))


def evaluate(pins):
    reg = registration()
    pin_record = check_pins(pins, include_model=False)
    rows, receipts = [], []
    for cell in reg["cells"]:
        path = OUT / "cells" / f"{cell['id']}.json"
        payload = json.loads(path.read_text())
        require(payload["status"] == "COMPLETE", "incomplete cell")
        require(payload["cell"] == cell and payload["registration"] == record(REG) and payload["execution_manifest"] == pin_record, "cell binding mismatch")
        require(payload["lease_elapsed_seconds"] <= cell["worker_cap_seconds"], "cell exceeded rail")
        cell_rows = []
        for rec in payload["traces"]:
            require(record(rec["path"]) == rec, "trace fingerprint mismatch")
            row = json.loads(Path(rec["path"]).read_text())
            c = next(c for c in reg["cases"] if c["id"] == row["id"])
            require(row["registration"] == record(REG) and row["case_sha256"] == c["sha256"] and row["execution_manifest"] == pin_record, "trace binding mismatch")
            require(row["detector_on"] is False and row["class"] == c["class"], "trace arm/class mismatch")
            require(row["mounted_node_ids"] == c["mount_node_ids"] and row["repository_node_ids"] == [n["node_id"] for n in c["nodes"]], "trace mount/repository mismatch")
            cell_rows.append(row)
        require(Counter(r["id"] for r in cell_rows) == Counter(cell["case_ids"]), "cell case mismatch")
        rows.extend(cell_rows)
        receipts.append(record(path))
    result = metrics(rows, reg)
    result.update(registration=record(REG), execution_manifest=pin_record, cell_receipts=receipts,
                  prior_art=reg["prior_art"])
    write(OUT / "evaluation.json", result)
    report = ["# GRM-C3 measured result", "", result["recommendation"], "", result["evidence_class"], "",
              "| Rule | TP | FP | TN | FN | Correct OFF answers at risk |", "|---|---:|---:|---:|---:|---:|"]
    for name, v in result["rules"].items():
        report.append(f"| {name} | {v['TP']} | {v['FP']} | {v['TN']} | {v['FN']} | {v['correct_answers_at_risk']} |")
    report += ["", "Separating cases: " + ", ".join(result["separating_cases"]),
               "", "Qualification failures: " + repr(result["qualification_failures"]),
               "", "No demand-ON answers were generated; lost correct answers remain unmeasured.",
               "", "## Prior art", "", reg["prior_art"], "", "## RED and scope",
               "", "Finite controlled mounts, not natural routing. Weakness is targeted, not certified. See evaluation.json for all traces and predictions."]
    with (OUT / "measured_report.md").open("x") as stream:
        stream.write("\n".join(report) + "\n")


def cpu_gate(pins):
    pin_record = check_pins(pins, include_model=False)
    validity = fixture_gate(registration())
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_grm_c3.py"],
                          cwd=ROOT, capture_output=True, text=True, timeout=120,
                          env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"})
    write(OUT / "cpu_gate.json", {"passed": proc.returncode == 0, "fixture_validity": validity,
                                 "pytest_returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr,
                                 "registration": record(REG), "execution_manifest": pin_record,
                                 "evidence_class": "author CPU unit-test baseline; blind lead verification pending"})
    print(proc.stdout)
    require(proc.returncode == 0, proc.stderr or "pytest gate RED")


def main():
    parser = argparse.ArgumentParser(__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pin", action="store_true")
    group.add_argument("--cpu-gate", action="store_true")
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--run-cell")
    group.add_argument("--case-worker", help=argparse.SUPPRESS)
    group.add_argument("--evaluate", action="store_true")
    parser.add_argument("--pins", type=Path, default=PINS)
    args = parser.parse_args()
    if args.pin:
        pin_sources(args.pins)
    elif args.cpu_gate:
        cpu_gate(args.pins)
    elif args.dry_run:
        reg = registration()
        print(json.dumps({"gpu_execution": False, "registration": record(REG),
                          "cells": reg["cells"], "offline_arms": list(reg["rules"]),
                          "estimated_gpu_seconds": sum(c["estimate_seconds"] for c in reg["cells"]),
                          "maximum_gpu_seconds": sum(c["worker_cap_seconds"] for c in reg["cells"]),
                          "estimated_cooldown_seconds": 360, "timing_evidence": "reasoning; unmeasured"}, indent=2))
    elif args.case_worker:
        case_worker(args.case_worker, args.pins)
    elif args.run_cell:
        run_cell(args.run_cell, args.pins)
    else:
        evaluate(args.pins)


if __name__ == "__main__":
    main()
