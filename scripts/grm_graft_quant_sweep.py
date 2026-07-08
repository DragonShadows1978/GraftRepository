#!/usr/bin/env python3
"""P1/P2 sweep runner for the GRM graft storage-quantization work order.

Per docs/GRM_GRAFT_QUANT_PLAN.md + docs/GRM_GRAFT_QUANT_LEDGER.md +
artifacts/grm_graft_quant/P0_FORMAT_AND_BATTERY.md:

For a given `--storage-bits` depth, this script:
  1. Quantizes-then-dequantizes each of the three frozen fp16 graft banks
     (multifact / preference / supersession) via
     grm_graft_quant_transform.py, writing a new fp16 graft dir + a
     quant_receipt.json (RMSE/max-abs per layer).
  2. Runs the P0-recommended sweep battery against the quantized banks,
     ALWAYS with --attention-mode standard (P0 risk 3), --skip-capture
     (banks are pre-captured, never re-witnessed):
       - multifact_graft_gate.py    --graft-dir <quantized multifact bank>
       - preference_graft_gate.py   --graft-dir <quantized preference bank>
       - supersession_graft_gate.py --graft-dir <quantized supersession bank>
       - multifact_addressing_gate.py --source <fp16 multifact JSON>
             --graft-dir <quantized multifact bank> --variants fact_local
  3. Extracts per-item control/mount margins (answer_logit minus the best
     other top-10 logit) via the extract_margins() helper below (handles
     both dict-shaped `runs` [multifact/preference/supersession] and
     list-shaped `runs` [addressing] — the existing
     artifacts/grm_graft_quant/extract_margins.py only handles the dict
     shape and would raise on addressing gate JSON; this harness's
     extractor is a corrected superset, kept local to avoid touching that
     file).
  4. Writes one cumulative JSON (per depth) under
     artifacts/grm_graft_quant/sweep_depth_<bits>.json and appends a row
     to artifacts/grm_graft_quant/SWEEP_CUMULATIVE.json.

Each gate run is a single bounded subprocess (~5-8 min at this battery per
P0's measured receipts); the caller is responsible for spacing depths per
the power law (this script does not sleep or loop over depths itself —
call it once per depth so GPU idle time between runs is visible to the
operator/ledger).

Usage:
    grm_graft_quant_sweep.py --storage-bits 8 \
        --tag depth8 [--skip-transform] [--only-transform]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
QUANT_DIR = REPO_ROOT / "artifacts" / "grm_graft_quant"
PY = sys.executable

BANKS = {
    "multifact": QUANT_DIR / "multifact_4k_fp16_gate" / "graft",
    "preference": QUANT_DIR / "preference_4k_fp16_gate" / "graft",
    "supersession": QUANT_DIR / "supersession_4k_fp16_gate" / "graft",
}
MULTIFACT_SOURCE_JSON = QUANT_DIR / "multifact_4k_fp16_gate.json"

NEEDLES = {
    ("supersession", "vault_keyword"): "vault (supersession)",
    ("preference", "preferred_shade"): "shade (preference)",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--storage-bits", type=int, required=True, choices=(16, 8, 6, 4, 3, 2))
    p.add_argument("--tag", default=None, help="label for this run, default depth<bits>")
    p.add_argument("--skip-transform", action="store_true",
                    help="reuse an already-quantized bank dir (must exist)")
    p.add_argument("--only-transform", action="store_true",
                    help="run the quantize-at-rest transform only, skip gate battery")
    p.add_argument("--group-size", type=int, default=32)
    p.add_argument("--skip-addressing", action="store_true")
    p.add_argument("--spacing-seconds", type=int, default=75,
                    help="idle seconds between gate runs (power law: bounded, "
                         "separated minutes-class runs, ~1-2 min apart)")
    return p.parse_args()


def gpu_busy_with_other_work() -> list[str]:
    """Return a list of foreign GPU compute processes (empty = clear to run)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
    except Exception:
        return []
    return [line for line in out.splitlines() if line.strip()]


def wait_for_gpu(max_wait_s: int = 900, step_s: int = 60) -> tuple[bool, list[str]]:
    """Before each gate run: if another compute process holds the GPU, wait up
    to max_wait_s in step_s increments. Returns (clear, last_process_list)."""
    waited = 0
    procs = gpu_busy_with_other_work()
    while procs and waited < max_wait_s:
        print(f"gpu busy ({procs}); waiting {step_s}s ({waited}/{max_wait_s})", flush=True)
        time.sleep(step_s)
        waited += step_s
        procs = gpu_busy_with_other_work()
    return (not procs, procs)


def run(cmd: list[str], log_path: Path) -> dict[str, Any]:
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"$ {' '.join(cmd)}\n\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}\n",
        encoding="utf-8",
    )
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "wall_seconds": wall,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
        "log": str(log_path),
    }


def quantize_bank(name: str, src: Path, dst: Path, bits: int, group_size: int, log_dir: Path) -> dict[str, Any]:
    cmd = [
        PY, str(REPO_ROOT / "scripts" / "grm_graft_quant_transform.py"),
        "--src", str(src),
        "--dst", str(dst),
        "--storage-bits", str(bits),
        "--group-size", str(group_size),
        "--force",
    ]
    result = run(cmd, log_dir / f"transform_{name}.log")
    receipt = None
    receipt_path = dst / "quant_receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    return {"bank": name, "run": result, "receipt": receipt}


def margin_from_summary(summary: dict[str, Any] | None) -> float | None:
    if not summary:
        return None
    alogit = summary.get("answer_logit")
    arank = summary.get("answer_rank")
    tks = summary.get("top_tokens") or []
    if alogit is None or not tks:
        return None
    others = [t.get("logit") for t in tks if t.get("rank") != arank and t.get("logit") is not None]
    if not others:
        return None
    return float(alogit) - max(float(x) for x in others)


def extract_margins(gate_json_path: Path) -> dict[str, Any]:
    """Handles both dict-shaped runs (multifact/preference/supersession:
    runs["facts"|"items"]) and list-shaped runs (addressing: runs is a
    bare list of probe rows). See module docstring for why this differs
    from artifacts/grm_graft_quant/extract_margins.py."""
    d = json.loads(gate_json_path.read_text(encoding="utf-8"))
    runs = d.get("runs")
    rows_src: list[dict[str, Any]]
    if isinstance(runs, dict):
        key = "facts" if "facts" in runs else "items"
        rows_src = runs.get(key, [])
    elif isinstance(runs, list):
        rows_src = runs
    else:
        rows_src = []

    rows = []
    for row in rows_src:
        control_summary = (row.get("control") or {}).get("summary")
        mount_summary = (row.get("mount") or {}).get("summary")
        rows.append({
            "id": row.get("id"),
            "variant": row.get("variant"),
            "answer": row.get("answer"),
            "control_hit": (control_summary or {}).get("hit"),
            "control_margin": margin_from_summary(control_summary),
            "mount_hit": (mount_summary or {}).get("hit"),
            "mount_margin": margin_from_summary(mount_summary),
            "mount_answer_rank": (mount_summary or {}).get("answer_rank"),
        })
    return {
        "artifact": str(gate_json_path),
        "classification": d.get("classification"),
        "hit_count": d.get("hit_count"),
        "count": d.get("fact_count") or d.get("item_count") or len(rows_src),
        "rows": rows,
    }


def main() -> int:
    args = parse_args()
    bits = int(args.storage_bits)
    tag = args.tag or f"depth{bits}"
    run_dir = QUANT_DIR / f"sweep_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    quant_bank_root = run_dir / "banks"

    t_start = time.time()
    result: dict[str, Any] = {
        "schema": "grm_graft_quant_sweep_depth_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "storage_bits": bits,
        "tag": tag,
        "group_size": args.group_size,
        "attention_mode": "standard",
        "transform": {},
        "gates": {},
    }

    # --- Step 1: quantize-at-rest each bank ---
    quantized_dirs = {}
    for name, src in BANKS.items():
        dst = quant_bank_root / name
        if args.skip_transform:
            if not (dst / "manifest.json").exists():
                raise SystemExit(f"--skip-transform but {dst} has no manifest.json")
            result["transform"][name] = {"skipped": True, "dst": str(dst)}
        else:
            t_res = quantize_bank(name, src, dst, bits, args.group_size, run_dir)
            result["transform"][name] = t_res
            if t_res["run"]["returncode"] != 0:
                result["status"] = f"transform_error:{name}"
                (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
                print(json.dumps({"status": result["status"], "artifact": str(run_dir / 'result.json')}))
                return 1
        quantized_dirs[name] = dst

    if args.only_transform:
        result["status"] = "transform_only"
        result["wall_seconds"] = time.time() - t_start
        out_path = run_dir / "result.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({"status": "transform_only", "artifact": str(out_path)}, indent=2))
        return 0

    # --- Step 2: gate battery against the quantized banks ---
    common_tail = ["--attention-mode", "standard", "--skip-capture"]

    gate_specs = [
        ("multifact", REPO_ROOT / "scripts" / "gpt_oss20b_multifact_graft_gate.py",
         ["--graft-dir", str(quantized_dirs["multifact"])]),
        ("preference", REPO_ROOT / "scripts" / "gpt_oss20b_preference_graft_gate.py",
         ["--graft-dir", str(quantized_dirs["preference"])]),
        ("supersession", REPO_ROOT / "scripts" / "gpt_oss20b_supersession_graft_gate.py",
         ["--graft-dir", str(quantized_dirs["supersession"])]),
    ]

    first_gate = True
    for gate_name, script, extra in gate_specs:
        if not first_gate and args.spacing_seconds > 0:
            time.sleep(args.spacing_seconds)
        first_gate = False
        clear, procs = wait_for_gpu()
        if not clear:
            result["status"] = "gpu_busy_timeout"
            result["gpu_foreign_processes"] = procs
            (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(json.dumps({"status": result["status"], "procs": procs}))
            return 1
        out_json = run_dir / f"{gate_name}_gate.json"
        cmd = [PY, str(script), "--output", str(out_json)] + extra + common_tail
        gate_result = run(cmd, run_dir / f"{gate_name}_gate.log")
        margins = extract_margins(out_json) if out_json.exists() else None
        result["gates"][gate_name] = {"run": gate_result, "margins": margins}

    if not args.skip_addressing:
        if args.spacing_seconds > 0:
            time.sleep(args.spacing_seconds)
        clear, procs = wait_for_gpu()
        if not clear:
            result["status"] = "gpu_busy_timeout"
            result["gpu_foreign_processes"] = procs
            (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(json.dumps({"status": result["status"], "procs": procs}))
            return 1
        addressing_json = run_dir / "addressing_fact_local_gate.json"
        cmd = [
            PY, str(REPO_ROOT / "scripts" / "gpt_oss20b_multifact_addressing_gate.py"),
            "--source", str(MULTIFACT_SOURCE_JSON),
            "--graft-dir", str(quantized_dirs["multifact"]),
            "--variants", "fact_local",
            "--output", str(addressing_json),
            "--attention-mode", "standard",
        ]
        gate_result = run(cmd, run_dir / "addressing_gate.log")
        margins = extract_margins(addressing_json) if addressing_json.exists() else None
        result["gates"]["addressing_fact_local"] = {"run": gate_result, "margins": margins}

    # --- Needle trajectory extract ---
    needles = {}
    for (bank, item_id), label in NEEDLES.items():
        gate_margins = (result["gates"].get(bank) or {}).get("margins")
        if not gate_margins:
            continue
        for row in gate_margins["rows"]:
            if row["id"] == item_id:
                needles[label] = {
                    "bank": bank, "id": item_id,
                    "mount_margin": row["mount_margin"],
                    "mount_hit": row["mount_hit"],
                }
    result["needles"] = needles

    result["wall_seconds"] = time.time() - t_start
    # A gate exits non-zero when its classification is "fail" — at low bit
    # depths that IS the finding, not a harness error. Only a gate that
    # produced no parseable margins JSON counts as a crash.
    any_crash = any(
        g["run"]["returncode"] != 0
        and (g.get("margins") is None
             or (g["margins"] or {}).get("classification") is None)
        for g in result["gates"].values()
    )
    result["status"] = "crash" if any_crash else "ok"

    out_path = run_dir / "result.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Append to cumulative index
    cumulative_path = QUANT_DIR / "SWEEP_CUMULATIVE.json"
    cumulative = []
    if cumulative_path.exists():
        cumulative = json.loads(cumulative_path.read_text(encoding="utf-8"))
    cumulative = [c for c in cumulative if c.get("storage_bits") != bits or c.get("tag") != tag]
    cumulative.append({
        "storage_bits": bits,
        "tag": tag,
        "status": result["status"],
        "wall_seconds": result["wall_seconds"],
        "result_path": str(out_path),
        "needles": needles,
        "gate_classification": {k: (v["margins"] or {}).get("classification") if v.get("margins") else None
                                  for k, v in result["gates"].items()},
    })
    cumulative_path.write_text(json.dumps(cumulative, indent=2), encoding="utf-8")

    print(json.dumps({
        "status": result["status"],
        "storage_bits": bits,
        "artifact": str(out_path),
        "needles": needles,
        "wall_seconds": result["wall_seconds"],
    }, indent=2))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
