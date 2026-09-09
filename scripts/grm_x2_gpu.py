#!/usr/bin/env python3
"""Leased GRM-X2 live driver; GPU imports only inside the leased worker.

Prior art: EB1/RS3/RS4/RT1 (house, 2026), reused production deposit, frame,
fit and attempt; paired same-output treatment follows controlled comparisons
(Fisher 1935, unverified — lead to check: Design of Experiments 1935).
POSIX flock / util-linux (year unverified — lead to check: flock util-linux
manual) provides the existing house lease. Ours: six frozen four-question
cells, bounded cooperative worker and create-only fingerprinted receipts.
"""
import argparse
import contextlib
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.grm_x2_common import OUT, read, contract, fingerprint, create_json, cell_rows, sha_file, score_answer
from core.grm_x2_witnesses import FLAG, META, ABSTENTION, install, gate_answer, sha

MODEL_DIR = Path("/home/vader/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee")
NATIVE_LIB = Path("/mnt/ForgeRealm/GraftRepository/cpp/build/libgrm_runtime.so")
FLAGS = {"GRM_PERSISTENT_BOAT": "0", "GRM_CAPTURE_PIN": "live", "GRM_SEAT_NEAR_LIVE": "1",
         "GRM_RT1_RULE": "1", "GRM_LSR_FIXES": "1", "GRM_DEMAND_NGH": "0",
         "GRM_SUP_RESOLVE": "1", "GRM_ADM_DECISIVE": "1", FLAG: "1",
         "TC_WEIGHT_BITS": "4", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
# Addenda 001/002 clarify metrics and pin the production turn-open geometry
# before any GPU data. Extractor, fixtures and thresholds remain unchanged.
CPU_RECEIPT = OUT / "cpu_gate_002.json"
VALIDATION_RECEIPT = OUT / "validation_002.json"


class BudgetExpired(RuntimeError):
    pass


def alarm_handler(_signum, _frame):
    # A self-timer raises an ordinary Python exception. No process is killed,
    # and no signal is sent to another process. Native driver hangs may delay
    # Python signal delivery: this is explicitly a residual, not a hard claim.
    raise BudgetExpired("registered worker wall deadline reached")


def open_turn(arena):
    # Prior art: ArenaCache.step (house EB1/RS3, 2026) sets each attention
    # layer's live_shift BEFORE eb1_begin_turn. Direct _attempt callers must
    # carry that same prelude; capture_pin restores the previous layer shift.
    for layer in arena.m.layers:
        layer.self_attn.live_shift = arena.live_shift
    return arena.eb1_begin_turn()


def ready():
    reg = contract()
    cpu = read(CPU_RECEIPT)
    fp, records = fingerprint()
    if cpu["status"] != "PASS" or not cpu["gpu_allowed"]:
        raise RuntimeError("CPU falsifier hit registered RED kill; GPU prohibited")
    if cpu["fingerprint"] != fp:
        raise RuntimeError("code/fixture drift since frozen CPU gate; amendment and new evidence required")
    checks = read(VALIDATION_RECEIPT)
    if checks["fingerprint"] != fp or checks["status"] != "PASS":
        raise RuntimeError("matching CPU validation receipt missing or RED")
    return reg, fp, records


def plan():
    reg = contract()
    rows = []
    for cell in reg["gpu"]["cells"]:
        for case in cell_rows(cell):
            for arm in ("A", "B"):
                rows.append({"cell": cell, "question_id": case["id"], "mount_class": case["mount_class"], "arm": arm,
                             "question": case["question"], "question_sha256": sha(case["question"]),
                             "source_versions": [s["source_version"] for s in case["sources"]],
                             "prompt_metadata": False})
    return {"evidence_class": "unit test / dry-run enumeration; NOT E2E", "cells": reg["gpu"]["cells"],
            "model_id": reg["gpu"]["model_id"], "model_reasoning_effort": "low (existing Harmony sink)",
            "rows": rows, "n_questions": 24, "n_paired_scores": 48, "bounds": reg["gpu"],
            "worker_wall_max_s": 285, "cooldown_s": 30, "lease_wait_max_s": 240,
            "nominal_outer_bound_s": 555, "outer_limit_s": 590,
            "gpu_reserved_s": 1710, "gpu_run_status": "BLOCKED_NO_GPU_IN_SEAT"}


def preflight():
    # Read-only, once under the lease. Any operator GPU process wins.
    def query(fields):
        proc = subprocess.Popen(["nvidia-smi", fields, "--format=csv,noheader"], text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate()
        if proc.returncode:
            raise RuntimeError(f"nvidia-smi failed {proc.returncode}: {stderr}")
        return stdout.strip()
    processes = query("--query-compute-apps=pid,process_name,used_gpu_memory")
    if processes:
        raise RuntimeError("operator/other compute process present; no GPU work started: " + processes)
    inventory = query("--query-gpu=name,memory.total")
    if "4070 SUPER" not in inventory:
        raise RuntimeError("registered RTX 4070 SUPER unavailable: " + inventory)
    model_dir = Path(os.environ.get("GRM_X2_MODEL_DIR", str(MODEL_DIR)))
    native = Path(os.environ.get("GRM_X2_NATIVE_LIB", str(NATIVE_LIB)))
    required = [model_dir / p for p in ("config.json", "tokenizer.json", "model.safetensors.index.json")] + [native]
    for p in required:
        if not p.is_file():
            raise FileNotFoundError(f"live prerequisite missing: {p}")
    index = read(model_dir / "model.safetensors.index.json")
    shards = sorted(set(index["weight_map"].values()))
    weights = []
    for name in shards:
        p = model_dir / name
        if not p.is_file() or p.stat().st_size == 0:
            raise FileNotFoundError(f"model shard missing/empty: {p}")
        weights.append({"name": name, "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns})
    return {"gpu": inventory, "compute_processes": [], "model_dir": str(model_dir), "native_lib": str(native),
            "input_sha256": {str(p): sha_file(p) for p in required}, "weight_inventory": weights,
            "weight_sha256": "not read in bounded preflight; snapshot id + index SHA + shard sizes/mtimes only"}


def worker(cell, directory, prereq):
    # The timer covers imports, model load, four fresh repos and teardown.
    started = time.monotonic()
    old = signal.signal(signal.SIGALRM, alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, 285)
    repo = model = tokenizer = None
    rows = []
    try:
        os.environ.update(FLAGS)
        from scripts import grm_e2e_session as e2e
        from core.gpt_oss20b_tc import GptOss20B_TC, gpt_oss_grm_dialect_kwargs
        from core.graft_repository import GraftRepository
        from core.grm_admission import decisive_admission_profile, rt1_enabled
        from core import kv_graft
        from transformers import AutoTokenizer
        model, model_info = GptOss20B_TC.from_pretrained(Path(prereq["model_dir"]))
        tokenizer = AutoTokenizer.from_pretrained(prereq["model_dir"], local_files_only=True)
        dialect = gpt_oss_grm_dialect_kwargs(model.config)
        for case in cell_rows(cell):
            if time.monotonic() - started >= 270:
                raise BudgetExpired("insufficient remaining worker time for next question")
            # Fresh repository and empty boat per question; weights shared
            # only within the cell, never across another GPU process.
            encoded = []
            def encode(text):
                encoded.append(text)
                return tokenizer.encode(text, add_special_tokens=False)
            decode = lambda ids: tokenizer.decode(ids, clean_up_tokenization_spaces=False)
            repo = GraftRepository(model, encode, decode, str(directory / case["id"] / "repository"), autosave=False,
                                   arena_cls=e2e.GptOssGQAArenaCache, native_lib_path=prereq["native_lib"], native_auto=False,
                                   arena_width=96, route_layer=int(dialect["route_layer"]), topk=3, live_turns=2, max_live=4096,
                                   ephemeral=True, sink_text=e2e.HARMONY_SINK, prompt_template=e2e.harmony_turn,
                                   stop_sequences=e2e.HARMONY_STOPS, storage_bits=8, revision_resolution=True, decisive_admission=True)
            arena = repo.arena
            install(arena)
            ids, deposits = [], []
            for source in case["sources"]:
                idx = arena.deposit(source["text"], capture_pin="live")
                native_id = repo._native_sync_node(idx)
                graft = arena.grafts[idx]
                ids.append(idx)
                if graft["metadata"][META]["source_version"] != source["source_version"]:
                    raise RuntimeError("deposit text differs from frozen source")
                deposits.append({"graft_id": idx, "native_id": native_id, "fixture_source_id": source["source_id"],
                                 "metadata": graft["metadata"][META], "ntok": graft["ntok"], "capture_pin": graft.get("capture_pin")})
            recency = open_turn(arena)
            assert not arena.live_segs and arena.caches is None and arena.ephemeral and arena.width == 96
            assert rt1_enabled(arena)
            # Registered mount intervention. The production RT1-enabled
            # router is observed, not permitted to replace the assigned mount.
            route = decisive_admission_profile(arena, case["question"], exclude=set(), route_limit=3)
            fitted = e2e._budget_fit_mounts(arena, ids)
            if fitted != ids:
                raise RuntimeError(f"registered mount does not fit width 96: {case['id']} requested={ids} fitted={fitted}")
            arena._rs3_seat_explicit = True
            before = len(encoded)
            raw, info = arena._attempt(case["question"], fitted, 64, False, arena.stop_sequences or (), defer_memory=True)
            physical = [int(i) for i in arena.cur_mounts]
            if set(physical) != set(fitted):
                raise RuntimeError("physical mount intervention changed")
            grounded, contributors = arena._grounding_attribution(raw, physical, case["question"])
            arena._grounding_receipt(raw, physical, case["question"], info)
            a = raw if grounded else ABSTENTION
            b, b_info = gate_answer(raw, case["question"], [arena.grafts[i] for i in physical], {"grounded": grounded})
            # No source-text or witness serialization is passed to the prompt
            # formatter; record exact strings/ids and all encode calls during
            # generation so the lead can independently inspect the boundary.
            prompt = e2e.harmony_turn(case["question"], None)
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            if arena._format_step_prompt(case["question"]) != prompt:
                raise RuntimeError("production prompt changed from frozen question-only Harmony form")
            if any(META in t or '"witness_id"' in t or '"source_version"' in t for t in encoded[before:]):
                raise RuntimeError("witness metadata leaked into encoding")
            scores = {"raw": score_answer(raw, case), "A": score_answer(a, case, structural=not grounded),
                      "B": score_answer(b, case, structural=bool(b_info.get("abstained")))}
            row = {"evidence_class": "E2E session receipt / fresh EB1 repository, controlled mounts, paired post-generation guards",
                   "cell": cell, "question_id": case["id"], "mount_class": case["mount_class"], "question": case["question"],
                   "raw_answer": raw, "A": a, "B": b, "grounded_A": grounded, "contributors_A": sorted(contributors),
                   "witness": b_info[META], "scores": scores,
                   "correct_answer_rejected_by_B": scores["A"]["exact_answer"] and not scores["B"]["exact_answer"],
                   "raw_correct_answer_rejected_by_B": scores["raw"]["exact_answer"] and not scores["B"]["exact_answer"],
                   "prompt": prompt, "prompt_ids": prompt_ids, "prompt_sha256": sha(prompt),
                   "generation_encode_calls": encoded[before:], "witnesses_in_prompt": False,
                   "requested_mount_ids": ids, "physical_mount_ids": physical, "deposits": deposits,
                   "frame_ephemeral": arena.ephemeral, "recency_ids_at_open": list(recency),
                   "live_shift": arena.live_shift, "layer_live_shifts": [layer.self_attn.live_shift for layer in model.layers],
                   "live_segments_after": [list(x) for x in arena.live_segs], "arena_width": arena.width,
                   "seating": getattr(arena, "_rs3_last_seating", None), "rt1_enabled": rt1_enabled(arena),
                   "route_ranking_diagnostic": route.get("ranking"), "fit_split_children_exercised": False,
                   "grounding_normalized": info.get("grounding_normalized"), "model_id": "openai/gpt-oss-20b",
                   "model_reasoning_effort": "low", "worker_elapsed_s": time.monotonic() - started}
            create_json(directory / f"{case['id']}.json", row)
            rows.append(row)
            kv_graft.clear_injection(model)
            repo.close()
            repo = None
            del arena
            gc.collect()
        return rows
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
        if repo is not None:
            repo.close()
        # Normal teardown only. No process termination or service operations.
        del repo, model, tokenizer
        gc.collect()


def leased(cell):
    reg, fp, records = ready()
    if os.environ.get("GRM_X2_LEASED") != "1":
        raise RuntimeError("worker must be invoked by the flock wrapper")
    if cell not in reg["gpu"]["cells"]:
        raise ValueError("unknown registered cell")
    base = OUT / "gpu" / fp
    base.mkdir(parents=True, exist_ok=True)
    claim = base / f"{cell}.claim.json"
    if claim.exists():
        raise RuntimeError("cell already reserved; no overwrites/retries")
    reserved = len(list((OUT / "gpu").glob("*/*.claim.json"))) * 285
    if reserved + 285 > reg["gpu"]["budget_seconds"]:
        raise RuntimeError("registered total GPU budget exhausted")
    prereq = preflight()
    create_json(claim, {"cell": cell, "fingerprint": fp, "reserved_s": 285, "start_utc_ns": time.time_ns(),
                        "evidence_class": "reservation; not E2E success", "input_records": records, "prerequisites": prereq})
    directory = base / cell
    directory.mkdir(exist_ok=False)
    started = time.monotonic()
    result = {"cell": cell, "fingerprint": fp, "evidence_class": "E2E session receipt", "status": "RED", "rows": []}
    try:
        result["rows"] = worker(cell, directory, prereq)
        result["status"] = "COMPLETE" if len(result["rows"]) == 4 else "RED_INCOMPLETE"
    except BaseException as exc:
        result.update(status="RED_WORKER", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    finally:
        result["worker_wall_s"] = time.monotonic() - started
        if result["worker_wall_s"] > 285:
            result["status"] = "RED_WALL_OVERRUN"
        result["reserved_gpu_s"] = 285
        result["partial_receipts"] = sorted(str(p.relative_to(ROOT)) for p in directory.glob("*.json"))
        create_json(base / f"{cell}.receipt.json", result)
        # Foreground, under lease. Always cool down, including failed work.
        time.sleep(30)
    print(json.dumps({k: v for k, v in result.items() if k not in {"rows", "traceback"}}, indent=2))
    return 0 if result["status"] == "COMPLETE" else 2


def summary():
    reg = contract()
    fp, _ = fingerprint()
    base = OUT / "gpu" / fp
    receipts = [read(p) for p in sorted(base.glob("*.receipt.json"))]
    rows = [row for receipt in receipts if receipt["status"] == "COMPLETE" for row in receipt["rows"]]
    wanted = {r["id"] for r in read(OUT / "fixtures/live.json")}
    observed = [r["question_id"] for r in rows]
    complete = len(observed) == len(set(observed)) == 24 and set(observed) == wanted
    manual = sum(r["scores"][a]["manual_review"] for r in rows for a in ("raw", "A", "B"))
    table = {}
    for category in ("correct", "decoy", "unavailable"):
        subset = [r for r in rows if r["mount_class"] == category]
        table[category] = {"n": len(subset), **{arm: {key: sum(r["scores"][arm][key] for r in subset)
                         for key in ("exact_answer", "false_answer", "abstained", "abstention_correct", "abstention_incorrect", "manual_review", "unsupported_lucky_truth")}
                         for arm in ("raw", "A", "B")}}
    rejects = sum(r["correct_answer_rejected_by_B"] for r in rows if r["mount_class"] == "correct")
    raw_rejects = sum(r["raw_correct_answer_rejected_by_B"] for r in rows if r["mount_class"] == "correct")
    p3 = None
    if complete and not manual:
        p3 = table["decoy"]["B"]["false_answer"] * 2 <= table["decoy"]["A"]["false_answer"] and raw_rejects <= 1
    return {"evidence_class": "E2E session receipts only; missing GPU data never scores zero", "fingerprint": fp,
            "status": "COMPLETE" if complete and not manual else "BLOCKED_OR_INCOMPLETE",
            "receipt_count": len(receipts), "completed_questions": len(rows), "missing": sorted(wanted - set(observed)),
            "score_table": table, "correct_answers_B_rejected": rejects if rows else None,
            "raw_correct_answers_B_rejected": raw_rejects if rows else None,
            "P3_pass": p3, "decoy_A_zero_floor": bool(complete and table["decoy"]["A"]["false_answer"] == 0),
            "budget_reserved_s": len(list((OUT / "gpu").glob("*/*.claim.json"))) * 285,
            "wall_s_completed_and_failed": sum(r["worker_wall_s"] for r in receipts),
            "limitations": "controlled fresh mounts, not full 34-turn replay; split children not exercised; manual-review rows block P3"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("list", "run", "resume", "summary", "_leased"), default="list")
    parser.add_argument("cell", nargs="?")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run or args.action == "list":
        print(json.dumps(plan(), indent=2))
        return 0
    if args.action == "summary":
        print(json.dumps(summary(), indent=2))
        return 0
    if args.action == "_leased":
        return leased(args.cell)
    reg, fp, _ = ready()
    if args.action == "resume":
        # One cell per invocation keeps every foreground call <10 minutes.
        args.cell = next((cell for cell in reg["gpu"]["cells"] if not (OUT / "gpu" / fp / f"{cell}.claim.json").exists()), None)
        if args.cell is None:
            print(json.dumps(summary(), indent=2))
            return 0
    if args.cell not in reg["gpu"]["cells"]:
        raise ValueError("run requires registered CELL")
    if any(read(p)["status"] != "COMPLETE" for p in (OUT / "gpu").glob("*/*.receipt.json")):
        raise RuntimeError("prior GPU failure is RED; stop, lead amendment required")
    # A claim without completion can mean an interrupted process. Never wait
    # on or kill it; the lead inspects it. The flock already protects the GPU.
    for p in (OUT / "gpu").glob("*/*.claim.json"):
        if not p.with_name(p.name.replace(".claim.json", ".receipt.json")).exists():
            raise RuntimeError("incomplete claimed cell; lead inspection required")
    env = dict(os.environ, GRM_X2_LEASED="1", PYTHONDONTWRITEBYTECODE="1")
    command = ["flock", "--wait", "240", "--conflict-exit-code", "75", "/tmp/forge-gpu.lock",
               sys.executable, str(Path(__file__).resolve()), "_leased", args.cell]
    # No timeout-kill subprocess wrapper. Worker handles its own Python
    # deadline; the declared bound is conditional on responsive native calls.
    os.chdir(ROOT)
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
