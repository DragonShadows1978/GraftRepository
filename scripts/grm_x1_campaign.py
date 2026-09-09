#!/usr/bin/env python3
"""X1 CPU inventory, metrics, and create-only GPU campaign controller.

Prior art: Fisher (1935), blocked/paired experiments (unverified — lead to
check, The Design of Experiments); house EB1/SUP/RT1/RS3/RS4 (2026).
Borrow frozen fixtures, equal-state arms and native runtime. New: X1 cells,
fault conditions, conservative scoring and fingerprinted handoff receipts.
No inference about model accuracy is produced by --dry-run or unit tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.grm_x1_register import create, raw_json, sha

OUT = ROOT / "artifacts/grm_x1"
FIX = OUT / "fixtures"
TC_ROOT = Path("/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
SEED = 20260908
CONTINUATION_BINDINGS = {
    "orders/GRM_X1_AMENDMENT_2.md": "f8d29fba246fbbc6f30495d1b430c86ec2004fb5022bb4b443f8d38e3dc2445c",
    "artifacts/grm_x1/payload_amendment_01_handoff.json": "e208379019aa2dc59091def442b3cb9eae5923af4e8aeecc20f605cef22cae4b",
    "artifacts/grm_x1/registration.json": "defb5014f6ee66113139c1ca3fe902f9bdc51fb81d137cbc0a47e5d06f8f3e57",
    "artifacts/grm_x1/handoff_manifest.json": "bf923a9ccb4f06a8d2ae643cb2727de5f7c116304febf429f71a2da7e9087121",
    "artifacts/grm_x1/continuation_02_registration.json": "26dc26ddae0e505421c192a0b15ed89ad8d83b7589bc4e518054fcb3a726b293",
}
CONTINUATION_BUDGET = {"original_worker_s": 2565, "extra_retry_worker_s": 285,
    "total_worker_s": 2850, "gpu_hours": 2850 / 3600, "common_cap_s": 2700,
    "authority": "Lead amendment 2 explicitly authorizes overage."}


def read(path):
    return json.loads(Path(path).read_text())


def registration():
    reg = read(OUT / "registration.json")
    expected = (OUT / "registration.sha256").read_text().split()[0]
    if sha(OUT / "registration.json") != expected:
        raise ValueError("registration SHA mismatch")
    return reg


def queries():
    amendment = read(OUT / "fixture_amendment_01.json")
    if amendment["registration_sha256"] != sha(OUT / "registration.json"):
        raise ValueError("fixture amendment registration binding drift")
    if amendment["original_queries_sha256"] != sha(FIX / "queries.json"):
        raise ValueError("fixture amendment predecessor drift")
    path = ROOT / amendment["active_queries_path"]
    if sha(path) != amendment["active_queries_sha256"]:
        raise ValueError("amended query fixture drift")
    return read(path)


def verify_fixtures():
    reg = registration()
    for path, digest in {**reg["source_shas"], **reg["fixture_shas"]}.items():
        if sha(ROOT / path) != digest:
            raise ValueError(f"frozen source/fixture drift: {path}")
    if sha(FIX / "SHA256SUMS") != reg["fixture_manifest_sha"]:
        raise ValueError("fixture manifest drift")
    active_queries, families, cells = queries(), read(FIX / "families.json"), read(FIX / "cells.json")
    if len(active_queries) != 24 or len({q["question"] for q in active_queries}) != 24:
        raise ValueError("expected exactly 24 unique held-out wordings")
    if [sum(q["kind"] == k for q in active_queries) for k in ("alias", "correction")] != [12, 12]:
        raise ValueError("alias/correction imbalance")
    for q in active_queries:
        f = families[q["family_id"]]
        target = next(n for n in f["nodes"] if n["node_id"] == f["target_node"])
        if q["oracle_address"] != target["address"] or q["expected"] not in target["text"].casefold():
            raise ValueError("target/oracle metadata mismatch")
        if any(v.casefold() in q["question"].casefold() for v in [q["expected"], *q["rejected"]]):
            raise ValueError("answer value leaked into query")
        if q["kind"] == "correction" and not target["supersedes"]:
            raise ValueError("correction query lacks a real correction")
    if len(cells) != 9 or sum(c["worker_cap_s"] for c in cells) != 2565:
        raise ValueError("cell/budget drift")
    return {"status": "PASS", "evidence_class": "unit test: fixture integrity",
            "queries": len(active_queries), "alias": 12, "correction": 12,
            "families": len(families), "cells": len(cells), "source_shas": reg["source_shas"],
            "fixture_shas": reg["fixture_shas"], "query_amendment": read(OUT / "fixture_amendment_01.json")}


def dry_run():
    if (OUT / "continuation_03.json").exists():
        from scripts import grm_x1_units
        return grm_x1_units.dry_run()
    verify_fixtures()
    continuation = validated_continuation() if (OUT / "continuation_02.json").exists() else None
    return {"evidence_class": "reasoning: enumeration, zero GPU work",
            "registration_sha256": sha(OUT / "registration.json"),
            "fingerprint": sha(OUT / "continuation_02.json") if continuation else None,
            "cells": [{**c, "turns": len(c["query_ids"]) * len(c["arms"]) * len(c["conditions"]),
                       "command": f"bash scripts/grm_x1_lead_gpu.sh run {c['cell']}",
                       "retry": bool(continuation and c["cell"] in continuation["retry_cells"]),
                       "conditional": c["phase"] == "natural"} for c in read(FIX / "cells.json")],
            "primary_worker_max_s": 1710, "conditional_worker_max_s": 855,
            "all_worker_max_s": 2565, "cooldown_per_invocation_s": 30,
            "campaign_reservation_max_s": 2850 if continuation else 2565,
            "historical_reserved_s": 285 if continuation else 0,
            "wall_no_contention_max_estimate_s": 2835,
            "wall_with_all_lock_waits_max_estimate_s": 5085,
            "wall_per_call_max_estimate_s": 565, "outer_rail_s": 590,
            "wall_note": "estimates from registered caps, not measured; native call overrun remains RED"}


def normalize_answer(text):
    # Prior art: Unicode UAX15; house DET1 glyph-normalized value scoring.
    # Borrow dash/case normalization; X1 strict full-answer equality is new
    # experimental scoring, not a semantic answer-equivalence claim.
    text = unicodedata.normalize("NFKC", str(text)).casefold().strip()
    text = re.sub(r"[‐‑‒–—−]", "-", text)
    text = re.sub(r"\s*-\s*", "-", text)
    return " ".join(text.strip("\"'` .").split())


REFUSALS = {"i don't know", "i do not know", "i cannot answer", "not in memory",
            "i don't have that information", "i do not have that information"}


def score(answer, query, condition, info):
    normalized = normalize_answer(answer)
    expected = normalize_answer(query["expected"])
    rejected = [normalize_answer(v) for v in query["rejected"]]
    # Structural flags are trusted only for the exact production refusal or
    # X1 fixed refusal. A wrong answer containing "don't know" cannot escape.
    from core.grm_x1_addresses import ABSTENTION
    abstain = (str(answer) == ABSTENTION and info.get("x1_page_fault") is True) or (
        info.get("abstained") is True and str(answer).startswith("Not in memory: no stored record matches "))
    abstain = bool(abstain or normalized in REFUSALS)
    exact = condition == "present" and normalized == expected and not abstain
    contains = lambda value: re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", normalized) is not None
    return {"exact_answer": exact, "false_answer": not abstain and not exact,
            "abstained": abstain, "correct_abstention": abstain and condition == "absent",
            "value_match_diagnostic": contains(expected) and not any(contains(v) for v in rejected),
            "structural_demand_fired": bool(info.get("x1_structural_demand_fired", False)),
            "demand_fired": bool(info.get("demand_fired", False))}


def metric(rows):
    if not rows:
        return {"n": 0, "exact_rate": None, "false_rate": None, "abstention_correctness": None, "mean_wall_s": None}
    n = len(rows)
    abstentions = sum(r["abstained"] for r in rows)
    return {"n": n, "exact_rate": sum(r["exact_answer"] for r in rows) / n,
            "false_rate": sum(r["false_answer"] for r in rows) / n,
            "abstentions": abstentions,
            "abstention_correctness": sum(r["correct_abstention"] for r in rows) / abstentions if abstentions else None,
            "abstention_rate": abstentions / n,
            "structural_demand_fires": sum(r["structural_demand_fired"] for r in rows),
            "demand_fires": sum(r["demand_fired"] for r in rows),
            "mean_wall_s": sum(r["wall_s"] for r in rows) / n}


def aggregate(rows, cells=None):
    # Prior art: house X1 r1 (2026) thresholds reused verbatim. Amendment 4
    # (lead, 2026) changes required population only, not row scoring or gates.
    multiplicities = sorted({cell['multiplicity'] for cell in cells}) if cells is not None else (1, 10, 100)
    table = {f"{arm}/{m}/{cond}": metric([r for r in rows if r["arm"] == arm
                and r["multiplicity"] == m and (cond == "all" or r["condition"] == cond)])
             for arm in ("A", "B", "C", "N") for m in (1, 10, 100) for cond in ("present", "absent", "all")}
    required = {(q["query_id"], m, arm, cond) for q in queries()
                for m in (1, 10, 100) for arm in ("A", "B", "C") for cond in ("present", "absent")}
    if cells is not None:
        required = {(qid, cell['multiplicity'], arm, cond) for cell in cells if cell['phase'] == 'oracle'
                    for qid in cell['query_ids'] for arm in cell['arms'] for cond in cell['conditions']}
    observed = [(r["query_id"], r["multiplicity"], r["arm"], r["condition"]) for r in rows if r["arm"] != "N"]
    complete = len(observed) == len(required) and set(observed) == required
    result = {"evidence_class": "E2E session receipt" if rows else "reasoning: GPU blocked/no observations",
              "oracle_complete": complete, "oracle_positive": False, "table": table,
              "predictions": None, "kill": None, "rows": len(rows)}
    result["source_strata"] = {f"{source}/{arm}/{m}/{cond}": metric([r for r in rows
        if r["family_id"].startswith(source + "_") and r["arm"] == arm
        and r["multiplicity"] == m and r["condition"] == cond]) for source in ("sup", "census")
        for arm in ("A", "B", "C", "N") for m in (1, 10, 100) for cond in ("present", "absent")}
    result["query_strata"] = {f"{kind}/{arm}/{m}/{cond}": metric([r for r in rows
        if r["kind"] == kind and r["arm"] == arm and r["multiplicity"] == m and r["condition"] == cond])
        for kind in ("alias", "correction") for arm in ("A", "B", "C", "N")
        for m in (1, 10, 100) for cond in ("present", "absent")}
    if not complete:
        return result
    exact = lambda a, m: table[f"{a}/{m}/present"]["exact_rate"]
    b = [exact("B", m) for m in multiplicities]
    coverage = all(r.get("address_coverage") is True for r in rows
                   if r["arm"] in ("B", "C") and r["condition"] == "present")
    a_false = table["A/100/all"]["false_rate"]
    ratios = {f"{a}/{m}": table[f"{a}/{m}/all"]["mean_wall_s"] / table[f"A/{m}/all"]["mean_wall_s"]
              for a in ("B", "C") for m in multiplicities}
    c_faults = all(table[f"C/{m}/absent"]["false_rate"] == 0 for m in multiplicities)
    c_retains = all(exact("C", m) >= exact("B", m) - .05 for m in multiplicities)
    result["oracle_positive"] = bool(min(b) >= .80 and max(b) - min(b) <= .05
        and exact("B", 100) - exact("A", 100) >= .20 and coverage and c_faults and c_retains)
    result["kill"] = {"ranking_dependence_with_correct_address": not coverage,
                      "B_accuracy_spread_over_5_points": max(b) - min(b) > .05}
    result["predictions"] = {"P1": exact("A", 1) - exact("A", 100) >= .20,
        "P2": max(b) - min(b) <= .05,
        "P3": table["C/100/all"]["false_rate"] <= .5 * a_false if a_false > 0 else None,
        "P3_null_reason": "no A false answers; half-reduction is unidentifiable" if a_false == 0 else None,
        "P4": all(.90 <= r <= 1.10 for r in ratios.values()),
        "S1": exact("A", 1) - exact("A", 100) < .20, "S2": max(b) - min(b) <= .05,
        "S3": all(r["correct_abstention"] and r.get("generation_calls") == 0 for r in rows if r["arm"] == "C" and r["condition"] == "absent"),
        "S4": all(r <= 1.10 for r in ratios.values()), "S5": None}
    result["wall_ratios"] = ratios
    natural = [r for r in rows if r["arm"] == "N"]
    required_n = {(q["query_id"], m, cond) for q in queries() for m in (1, 10, 100) for cond in ("present", "absent")}
    if cells is not None:
        required_n = {(qid, cell['multiplicity'], cond) for cell in cells if cell['phase'] == 'natural'
                      for qid in cell['query_ids'] for cond in cell['conditions']}
    observed_n = [(r["query_id"], r["multiplicity"], r["condition"]) for r in natural]
    if len(observed_n) == len(required_n) and set(observed_n) == required_n:
        result["predictions"]["S5"] = all(exact("N", m) >= exact("C", m) - .05 for m in multiplicities)
        result["kill"]["natural_erases_benefit"] = (not result["predictions"]["S5"] or exact("N", 100) - exact("A", 100) < .20)
    return result


def source_manifest():
    # SHA-256 content addressing is established (NIST FIPS 180-4, 2015;
    # unverified — lead to check). Borrow hashing/exclusive creation, not a
    # claim of tamper-proof storage or trust against a malicious operator.
    paths = sorted((ROOT / "core").glob("*.py"))
    paths += sorted((ROOT / "scripts").glob("grm_x1_*"))
    paths += sorted((ROOT / "tests").glob("test_grm_x1_*"))
    paths += [ROOT / "scripts/grm_e2e_session.py", ROOT / "scripts/grm_probe_ladder.py",
              OUT / "fixture_amendment_01.json", OUT / "cpu_mutation_registration.json"]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths if p.is_file()}


def dependency_inventory():
    frame = registration()["frame"]
    model = Path(frame["model_dir"])
    paths = [Path(frame["native_lib"]), TC_ROOT / "tensor_cuda/__init__.py"]
    paths += [model / n for n in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json")]
    engines = sorted((TC_ROOT / "tensor_cuda").glob("*_tensor_cuda*.so"))
    if not engines:
        engines = sorted((TC_ROOT / "build").glob("**/*tensor_cuda*.so"))
    paths += engines
    missing = [str(p) for p in paths if not p.is_file()]
    if not engines:
        missing.append("TensorCUDA extension .so")
    weights = []
    index = model / "model.safetensors.index.json"
    if index.is_file():
        for name in sorted(set(read(index)["weight_map"].values())):
            p = model / name
            if not p.is_file():
                missing.append(str(p))
            else:
                stat = p.stat()
                weights.append({"path": str(p), "resolved": str(p.resolve()), "bytes": stat.st_size,
                                "mtime_ns": stat.st_mtime_ns, "identity": "path/stat only; weights not content-hashed"})
    return {"file_shas": {str(p): sha(p) for p in paths if p.is_file()},
            "weight_inventory": weights, "missing": missing}


def seal():
    verify_fixtures()
    payload = {"schema": "grm.x1.handoff-manifest.v1", "registration_sha256": sha(OUT / "registration.json"),
               "sources": source_manifest(), "dependencies": dependency_inventory()}
    create(OUT / "handoff_manifest.json", raw_json(payload))
    return {"handoff_sha256": sha(OUT / "handoff_manifest.json"), "missing": payload["dependencies"]["missing"]}


def fingerprint():
    if (OUT / "continuation_03.json").exists():
        from scripts import grm_x1_units
        return grm_x1_units.fingerprint()
    verify_fixtures()
    frozen = read(OUT / "handoff_manifest.json")
    # Prefer the explicit continuation; never silently fall back on a forged
    # or stale continuation even if somebody restores the original sources.
    continuation = validated_continuation() if (OUT / "continuation_02.json").exists() else None
    if frozen["registration_sha256"] != sha(OUT / "registration.json") or (
            continuation is None and frozen["sources"] != source_manifest()):
        raise ValueError("handoff source drift; do not start a new campaign silently")
    current = dependency_inventory()
    if current != frozen["dependencies"] or current["missing"]:
        raise ValueError("lead dependencies missing or drifted from handoff")
    return sha(OUT / ("continuation_02.json" if continuation else "handoff_manifest.json"))


def continuation_payload():
    # Prior art: house X1 (2026) exclusive claims and content-addressed receipts;
    # NIST FIPS 180-4 (2015), SHA-256 (unverified — lead to check these terms).
    # Borrow integrity hashes, not signatures. New: this lead-authorized epoch
    # partition, fixed 285-second retry, and preservation of the exact r1 RED.
    for path, digest in CONTINUATION_BINDINGS.items():
        if sha(ROOT / path) != digest:
            raise ValueError(f"continuation authority binding mismatch: {path}")
    reg = read(OUT / "continuation_02_registration.json")
    prior = read(OUT / "payload_amendment_01_handoff.json")
    sources = source_manifest()
    allowed = {"scripts/grm_x1_campaign.py", "tests/test_grm_x1_campaign.py"}
    if set(sources) != set(prior["sources"]) or any(
            value != prior["sources"][path] for path, value in sources.items() if path not in allowed):
        raise ValueError("continuation changes sources outside amendment 2 scope")
    historical = {str(p.relative_to(ROOT)): sha(p)
                  for p in [*sorted((OUT / "claims").glob("*.json")),
                            *sorted((OUT / "receipts").glob("gpu_*.json"))]}
    if historical != reg["historical_files"]:
        raise ValueError("r1 evidence changed since continuation registration")
    return {"schema": "grm.x1.continuation.v2", "bindings": CONTINUATION_BINDINGS,
            "sources": sources, "historical_files": historical,
            "r1_fingerprint": sha(OUT / "handoff_manifest.json"),
            "retry_cells": ["oracle_m1_s0"], "retry_limit": 1,
            "budget": CONTINUATION_BUDGET,
            "receipt_directory": "artifacts/grm_x1/receipts/r2",
            "claim_directory": "artifacts/grm_x1/claims/r2",
            "receipt_naming": "gpu_{cell}_{receipt_content_sha256}.json; fingerprint = continuation file SHA-256",
            "evidence_class": "reasoning: lead-authorized registration before CPU gates; no GPU execution"}


def seal_continuation():
    verify_fixtures()
    payload = continuation_payload()
    path = OUT / "continuation_02.json"
    checksum = OUT / "continuation_02.sha256"
    if path.exists() or checksum.exists():
        raise FileExistsError("continuation already sealed; immutable")
    create(path, raw_json(payload))
    create(checksum, f"{sha(path)}  continuation_02.json\n".encode())
    return {"continuation": str(path), "sha256": sha(path)}


def validated_continuation():
    path = OUT / "continuation_02.json"
    if sha(path) != (OUT / "continuation_02.sha256").read_text().split()[0]:
        raise ValueError("continuation SHA mismatch")
    payload = read(path)
    expected = continuation_payload()
    if payload.get("sources") != expected["sources"]:
        raise ValueError("stale continuation: source SHA mismatch")
    if payload != expected:
        raise ValueError("forged continuation: binding or policy mismatch")
    return payload


def emit_receipt(stem, payload, directory=None):
    directory = directory if directory is not None else OUT / "receipts"
    data = raw_json(payload)
    path = directory / f"{stem}_{hashlib.sha256(data).hexdigest()}.json"
    create(path, data)
    return path


def campaign_epoch(directory, fp):
    # Prior art: house X1 (2026), create-only reservations and receipt hashes.
    # New: independently validate each epoch so historical RED cannot enter
    # the amended verdict, and an incomplete retry still consumes its slot.
    cells = {cell["cell"] for cell in read(FIX / "cells.json")}
    claims, receipts, statuses, rows = {}, {}, {}, []
    for path in sorted((OUT / "claims" / directory).glob("*.json")):
        payload = read(path)
        cell = payload["cell"]
        if (cell not in cells or payload["fingerprint"] != fp
                or path.name != f"{cell}_{fp}.json" or payload["reserved_gpu_s"] != 285):
            raise ValueError(f"invalid claim identity/reservation: {path}")
        if cell in claims:
            raise ValueError("duplicate cell claim")
        claims[cell] = {"path": str(path.relative_to(ROOT)), "sha256": sha(path), **payload}
        statuses[cell] = "INCOMPLETE_CLAIM"
    for path in sorted((OUT / "receipts" / directory).glob("gpu_*.json")):
        payload = read(path)
        cell = payload["cell"]
        digest = hashlib.sha256(raw_json(payload)).hexdigest()
        if (cell not in cells or path.name != f"gpu_{cell}_{digest}.json"
                or payload["fingerprint"] != fp
                or payload["registration_sha256"] != sha(OUT / "registration.json")
                or payload["status"] not in ("COMPLETE", "RED")):
            raise ValueError(f"invalid receipt digest/identity: {path}")
        if cell in receipts:
            raise ValueError("duplicate cell receipt")
        if cell not in claims:
            raise ValueError("receipt without reservation")
        receipts[cell] = {"path": str(path.relative_to(ROOT)), "sha256": sha(path), **payload}
        statuses[cell] = payload["status"]
        if payload["status"] == "COMPLETE":
            rows.extend(payload["rows"])
    return {"fingerprint": fp, "cell_status": statuses, "claims": claims,
            "receipts": receipts, "reserved_gpu_s": len(claims) * 285, "rows": rows,
            "failed": any(value != "COMPLETE" for value in statuses.values())}


def summary():
    if (OUT / "continuation_03.json").exists():
        from scripts import grm_x1_units
        return grm_x1_units.state()
    continuation = validated_continuation() if (OUT / "continuation_02.json").exists() else None
    r1 = campaign_epoch("", sha(OUT / "handoff_manifest.json"))
    r2 = campaign_epoch("r2", sha(OUT / "continuation_02.json")) if continuation else None
    if not continuation and (any((OUT / "claims/r2").glob("*.json"))
                             or any((OUT / "receipts/r2").glob("gpu_*.json"))):
        raise ValueError("r2 evidence without continuation")
    retry = []
    if continuation:
        # This order authorizes only the frozen r1 RED, never r1 COMPLETE,
        # unknown fingerprints, duplicate receipts, or an incomplete r1 claim.
        if set(r1["cell_status"]) != set(continuation["retry_cells"]) or any(
                value != "RED" for value in r1["cell_status"].values()):
            raise ValueError("continuation requires the sole registered r1 RED receipt")
        retry = [cell for cell in continuation["retry_cells"] if cell not in r2["cell_status"]]
    active = r2 if continuation else r1
    reserved = r1["reserved_gpu_s"] + (r2["reserved_gpu_s"] if r2 else 0)
    cap = 2850 if continuation else 2565
    if reserved > cap:
        raise ValueError("campaign reservation exceeds registered budget")
    result = aggregate(active["rows"])
    # Report the existing registered forks; no new accuracy threshold.
    verdict = "PENDING"
    if active["failed"]:
        verdict = "RED"
    elif result["oracle_complete"]:
        if any(result["kill"].values()):
            verdict = "KILLED"
        elif not result["oracle_positive"]:
            verdict = "ORACLE_NOT_POSITIVE"
        elif result["predictions"]["S5"] is None:
            verdict = "ORACLE_POSITIVE_NATURAL_PENDING"
        else:
            verdict = "COMPLETE_REGISTERED_BENEFIT_RETAINED"
    result.update({"active_epoch": "r2" if continuation else "r1",
                   "fingerprint": active["fingerprint"], "cell_status": active["cell_status"],
                   "reserved_gpu_s": reserved, "reservation_cap_s": cap,
                   "epochs": {"r1": r1, "r2": r2}, "retry_eligible": retry,
                   "historical_red": [receipt for receipt in r1["receipts"].values()
                                      if receipt["status"] == "RED"] if continuation else [],
                   "gpu_status": "BLOCKED_NO_GPU_EXECUTED" if not active["cell_status"] else "RECEIPTS_PRESENT",
                   "campaign_verdict": verdict,
                   "failed": active["failed"]})
    return result


def next_cell(requested=None):
    state = summary()
    if state["failed"]:
        raise ValueError("failed/incomplete cell: registered stop; no automatic retry")
    cells = read(FIX / "cells.json")
    if requested is not None and requested not in {c["cell"] for c in cells}:
        raise ValueError("unregistered cell")
    for cell in cells:
        if requested is not None and requested != cell["cell"]:
            continue
        if cell["cell"] in state["cell_status"]:
            if requested:
                raise ValueError("cell already consumed; receipts are create-only")
            continue
        if cell["phase"] == "natural" and not state["oracle_positive"]:
            if requested:
                raise ValueError("natural arm locked: oracle arm is not positive")
            continue
        return cell
    return None


def preflight():
    return {"evidence_class": "unit test: read-only CPU preflight", "fingerprint": fingerprint(),
            "dependencies": dependency_inventory(), "gpu_probe": "not performed; lead-only"}


def run_controller(requested=None):
    if (OUT / "continuation_03.json").exists():
        from scripts import grm_x1_units
        return grm_x1_units.controller(requested)
    fp = fingerprint()
    cell = next_cell(requested)
    if cell is None:
        return {"status": "NO_ELIGIBLE_CELL", "summary": summary()}
    # Foreground flock, bounded wait; worker self-timeout never signals/kills
    # another process. No subprocess timeout parameter (it would kill).
    command = ["flock", "--wait", "250", "--no-fork", "/tmp/forge-gpu.lock", sys.executable,
               str(Path(__file__).resolve()), "_worker", cell["cell"], "--fingerprint", fp]
    started = time.monotonic()
    result = subprocess.run(command, cwd=ROOT, check=False)
    time.sleep(30)  # Foreground cooldown; one cell per invocation, not a job loop.
    payload = {"status": "RETURNED" if result.returncode == 0 else "RED",
               "cell": cell["cell"], "returncode": result.returncode,
               "outer_wall_s": time.monotonic() - started, "outer_cap_s": 590,
               "fingerprint": fp, "evidence_class": "process receipt", "command": command}
    payload["outer_within_rail"] = payload["outer_wall_s"] <= 590
    emit_receipt("controller", payload, directory=OUT / "receipts" / ("r2" if (OUT / "continuation_02.json").exists() else ""))
    return payload


def worker(cell_id, fp):
    # flock --no-fork carries its open lock descriptor through exec. Refuse
    # direct _worker invocation without that inherited descriptor.
    import fcntl
    inherited = []
    for entry in Path("/proc/self/fd").iterdir():
        try:
            if entry.resolve() == Path("/tmp/forge-gpu.lock"):
                inherited.append(int(entry.name))
        except FileNotFoundError:
            continue
    if not inherited:
        raise RuntimeError("GPU worker requires inherited /tmp/forge-gpu.lock lease")
    fcntl.flock(inherited[0], fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Registered own-process exception timer. It is deliberately not a
    # timeout/kill wrapper. A non-returning native call remains a RED residual.
    def expired(_sig, _frame):
        raise TimeoutError("GRM-X1 worker reached its 280 s work rail (285 s outer worker allowance)")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, 280)
    started = time.monotonic()
    if fingerprint() != fp:
        raise ValueError("worker fingerprint mismatch")
    cell = next_cell(cell_id)
    claim = {"cell": cell_id, "fingerprint": fp, "reserved_gpu_s": 285,
             "pid": os.getpid(), "started_unix": time.time(), "evidence_class": "process reservation"}
    epoch = "r2" if (OUT / "continuation_02.json").exists() else ""
    create(OUT / "claims" / epoch / f"{cell_id}_{fp}.json", raw_json(claim))
    rows = []
    status, error = "COMPLETE", None
    try:
        # Card owner has right of way even if it is not using the advisory lock.
        probe = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                               text=True, capture_output=True, check=False)
        if probe.returncode or probe.stdout.strip():
            raise RuntimeError(f"GPU busy/unavailable; yield to operator: {probe.stdout.strip()} {probe.stderr.strip()}")
        from scripts.grm_x1_gpu import run_cell
        run_cell(cell, rows, OUT / "sessions" / f"{cell_id}_{fp}")
        expected = len(cell["query_ids"]) * len(cell["conditions"]) * len(cell["arms"])
        if len(rows) != expected:
            raise RuntimeError(f"incomplete cell: expected {expected} rows, got {len(rows)}")
    except Exception as exc:
        import traceback
        status, error = "RED", {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    elapsed = time.monotonic() - started
    if elapsed > 285:
        status = "RED"
        error = {"type": "WorkerOverrun", "message": f"{elapsed} s > 285 s", "previous_error": error}
    payload = {"schema": "grm.x1.gpu-cell.v1", "evidence_class": "E2E session receipt",
               "cell": cell_id, "fingerprint": fp, "status": status, "error": error,
               "worker_wall_s": elapsed, "rows": rows, "registration_sha256": sha(OUT / "registration.json")}
    path = emit_receipt("gpu_" + cell_id, payload, directory=OUT / "receipts" / epoch)
    print(json.dumps({"receipt": str(path), "status": status, "worker_wall_s": elapsed}), flush=True)
    return 0 if status == "COMPLETE" else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", default="list",
                        choices=("list", "check", "preflight", "seal", "seal-continuation", "seal-units", "seal-reduced", "summary", "run", "resume", "_worker"))
    parser.add_argument("cell", nargs="?")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fingerprint")
    parser.add_argument("--unit")
    parser.add_argument("--attempt", type=int, choices=(1, 2), default=1)
    parser.add_argument("--retry-unit")
    args = parser.parse_args()
    if args.command == "_worker":
        if (OUT / "continuation_03.json").exists():
            from scripts import grm_x1_units
            return grm_x1_units.worker(args.cell, args.fingerprint, args.unit, args.attempt)
        return worker(args.cell, args.fingerprint)
    if args.dry_run or args.command == "list":
        result = dry_run()
    elif args.command == "check":
        result = verify_fixtures()
    elif args.command == "preflight":
        result = preflight()
    elif args.command == "seal":
        result = seal()
    elif args.command == "seal-continuation":
        result = seal_continuation()
    elif args.command == "seal-units":
        from scripts import grm_x1_units
        result = grm_x1_units.seal()
    elif args.command == "seal-reduced":
        from scripts.grm_x1_reduced import seal as seal_reduced
        result = seal_reduced()
    elif args.command == "summary":
        result = summary()
    else:
        if args.command == "run" and args.cell is None:
            parser.error("run requires CELL")
        if args.retry_unit:
            from scripts import grm_x1_units
            result = grm_x1_units.controller(args.cell, args.retry_unit)
        else:
            result = run_controller(args.cell if args.command == "run" else None)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if result.get("status") == "RED" or str(result.get("status", "")).startswith("STOP_") else 0


if __name__ == "__main__":
    raise SystemExit(main())
