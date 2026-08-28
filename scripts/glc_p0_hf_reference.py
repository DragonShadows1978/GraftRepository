#!/usr/bin/env python3
"""GLC P0.a: offline HuggingFace reference scoring for sealed ramp cells.

The installed Transformers build understands the checkpoint's MXFP4 format,
but this host does not have the optional ``kernels`` package.  Evidence runs
therefore make the fallback explicit: BF16 dequantization with Accelerate
CPU/disk offload, eager sink-aware attention, and one independently bounded
process per ramp cell.  This script never tokenizes or reconstructs a corpus.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import resource
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np


sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import glc_p0_common as common


HF_OUTPUT = common.OUTPUT_ROOT / "hf_reference"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("preflight", "self-test", "score-cell", "analyze")
    )
    parser.add_argument("--length", type=int, choices=common.RAMP_LENGTHS)
    parser.add_argument("--attempt", type=int, default=0)
    parser.add_argument("--model-dir", type=Path, default=common.MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=common.OUTPUT_ROOT)
    parser.add_argument("--offload-dir", type=Path, default=Path("/tmp/glc_p0_hf_offload"))
    parser.add_argument("--cpu-memory", default="42GiB")
    parser.add_argument("--threads", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    parser.add_argument("--logit-chunk", type=int, default=32)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def receipt_path(length: int, attempt: int) -> Path:
    return HF_OUTPUT / f"ramp_{int(length)}_attempt{int(attempt):02d}.json"


def max_rss_mib() -> float:
    # Linux reports KiB; retain the platform fact in every receipt.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _timeout_handler(signum, frame):  # type: ignore[no-untyped-def]
    del signum, frame
    raise TimeoutError("received SIGTERM from the bounded CPU wrapper")


def nll_from_logits(logits: Any, targets: np.ndarray, chunk: int) -> dict[str, Any]:
    """Score causal targets without materializing a second full FP32 logit copy."""

    import torch

    if logits.ndim != 3 or logits.shape[0] != 1:
        raise RuntimeError(f"unexpected HF logits shape {tuple(logits.shape)}")
    target_flat = np.asarray(targets, dtype=np.int64).reshape(-1)
    if int(logits.shape[1]) != target_flat.size:
        raise RuntimeError(
            f"HF logit/target alignment {int(logits.shape[1])} != {target_flat.size}"
        )
    if chunk <= 0:
        raise ValueError("--logit-chunk must be positive")
    nll_parts: list[np.ndarray] = []
    for start in range(0, target_flat.size, int(chunk)):
        stop = min(start + int(chunk), target_flat.size)
        rows = logits[0, start:stop].to(torch.float32)
        local_targets = torch.as_tensor(
            target_flat[start:stop], dtype=torch.long, device=rows.device
        )
        selected = rows.gather(1, local_targets[:, None]).squeeze(1)
        values = torch.logsumexp(rows, dim=-1) - selected
        nll_parts.append(values.detach().cpu().numpy().astype(np.float64, copy=False))
        del rows, local_targets, selected, values
    nlls = np.concatenate(nll_parts)
    mean_nll = float(nlls.mean(dtype=np.float64))
    return {
        "nll_sum": float(nlls.sum(dtype=np.float64)),
        "mean_nll": mean_nll,
        "ppl": float(math.exp(mean_nll)),
        "token_count": int(nlls.size),
        "target_ids_sha256": common.sha256_array(target_flat),
    }


def preflight(args: argparse.Namespace) -> int:
    common.ensure_output_root(args.output_dir)
    common.ensure_model_dir(args.model_dir)
    manifest, arrays = common.validate_sealed_controls()
    cells: list[dict[str, Any]] = []
    for length in common.RAMP_LENGTHS:
        long_arm = common.ramp_arm(arrays["wikitext_0_2560"], length, "long")
        ref_arm = common.ramp_arm(arrays["wikitext_0_2560"], length, "reference")
        if not np.array_equal(long_arm["targets"], ref_arm["targets"]):
            raise RuntimeError(f"sealed ramp {length} targets differ")
        cells.append(
            {
                "length": length,
                "target_ids_sha256": long_arm["target_ids_sha256"],
                "long_input_tokens": int(long_arm["inputs"].shape[1]),
                "reference_input_tokens": int(ref_arm["inputs"].shape[1]),
            }
        )
    facts = common.runtime_facts()
    route = (
        "native_mxfp4_possible"
        if facts.get("kernels_package_available") and facts.get("torch_cuda_available")
        else "cpu_bf16_dequant_accelerate_offload_required"
    )
    print(
        json.dumps(
            {
                "status": "ready",
                "model_revision": manifest["model_revision"],
                "control_arrays_sha256": common.sha256_file(common.CONTROL_ARRAYS),
                "cells": cells,
                "load_route": route,
                "runtime": facts,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def self_test(args: argparse.Namespace) -> int:
    common.ensure_output_root(args.output_dir)
    _manifest, arrays = common.validate_sealed_controls()
    checks: list[dict[str, Any]] = []
    target_hashes: dict[str, str] = {}
    for length in common.RAMP_LENGTHS:
        long_arm = common.ramp_arm(arrays["wikitext_0_2560"], length, "long")
        ref_arm = common.ramp_arm(arrays["wikitext_0_2560"], length, "reference")
        same = bool(np.array_equal(long_arm["targets"], ref_arm["targets"]))
        checks.append({"name": f"same_targets_{length}", "passed": same})
        target_hashes[str(length)] = long_arm["target_ids_sha256"]

    import torch

    logits = torch.tensor(
        [[[1.0, 2.0, -1.0], [0.0, 0.5, 1.5]]], dtype=torch.bfloat16
    )
    targets = np.asarray([1, 2], dtype=np.int64)
    score = nll_from_logits(logits, targets, chunk=1)
    work = logits.float().numpy()[0].astype(np.float64)
    expected = []
    for row, target in zip(work, targets):
        maximum = float(row.max())
        expected.append(maximum + math.log(float(np.exp(row - maximum).sum())) - row[target])
    expected_mean = float(np.mean(expected))
    checks.append(
        {
            "name": "chunked_nll",
            "passed": bool(abs(score["mean_nll"] - expected_mean) <= 1.0e-7),
            "observed": score["mean_nll"],
            "expected": expected_mean,
        }
    )
    checks.extend(
        [
            {
                "name": "hg1_boundary_green",
                "passed": bool((4.05 - 4.00) <= common.HG1_DELTA_NATS_MAX + 1.0e-12),
            },
            {
                "name": "hg1_above_boundary_red",
                "passed": bool((4.0501 - 4.00) > common.HG1_DELTA_NATS_MAX),
            },
        ]
    )
    core = {
        "schema": "glc_p0a_self_test_v1",
        "status": "pass" if all(row["passed"] for row in checks) else "fail",
        "checks": checks,
        "target_ids_sha256": target_hashes,
        "control_arrays_sha256": common.sha256_file(common.CONTROL_ARRAYS),
        "script_sha256": common.sha256_file(SCRIPT_PATH),
    }
    digest = common.canonical_json_sha256(core)[:16]
    path = common.OUTPUT_ROOT / "self_test" / f"p0a_{digest}.json"
    if not path.exists():
        common.write_new_json(path, {**core, "created_at": common.now_iso()})
    print(json.dumps({"status": core["status"], "receipt": str(path)}), flush=True)
    return 0 if core["status"] == "pass" else 1


def _arm_metadata(data: dict[str, Any]) -> dict[str, Any]:
    return {
        key: data[key]
        for key in (
            "length",
            "view",
            "source_start",
            "source_stop_exclusive",
            "full_sequence_ids_sha256",
            "input_ids_sha256",
            "predictor_ids_sha256",
            "target_ids_sha256",
            "port_receipt",
            "port_mean_nll",
            "port_ppl",
        )
    } | {
        "input_tokens": int(data["inputs"].shape[1]),
        "scored_target_tokens": int(data["targets"].size),
    }


def score_cell(args: argparse.Namespace) -> int:
    common.ensure_output_root(args.output_dir)
    model_dir = common.ensure_model_dir(args.model_dir)
    if args.length is None:
        raise ValueError("score-cell requires --length")
    if args.attempt < 0:
        raise ValueError("--attempt must be nonnegative")
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    path = receipt_path(args.length, args.attempt)
    if path.exists():
        prior = common.read_json(path)
        if prior.get("status") == "complete":
            print(json.dumps({"status": "already_complete", "receipt": str(path)}))
            return 0
        raise FileExistsError(f"append-only failed attempt exists; choose --attempt: {path}")

    _manifest, arrays = common.validate_sealed_controls()
    wiki = arrays["wikitext_0_2560"]
    arms = {
        view: common.ramp_arm(wiki, args.length, view)
        for view in ("reference", "long")
    }
    if not np.array_equal(arms["reference"]["targets"], arms["long"]["targets"]):
        raise RuntimeError("long/reference target identity failed before HF load")

    receipt: dict[str, Any] = {
        "schema": "glc_p0a_hf_reference_cell_v1",
        "created_at": common.now_iso(),
        "status": "starting",
        "length": int(args.length),
        "attempt": int(args.attempt),
        "argv": sys.argv,
        "plan": str(common.PLAN_PATH),
        "model_dir": str(model_dir),
        "model_revision": common.MODEL_REVISION,
        "control_arrays": str(common.CONTROL_ARRAYS),
        "control_arrays_sha256": common.sha256_file(common.CONTROL_ARRAYS),
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
        "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE"),
        "hg1_delta_nats_max": common.HG1_DELTA_NATS_MAX,
        "arms": {},
        "runtime_preload": common.runtime_facts(),
    }
    started = time.perf_counter()
    signal.signal(signal.SIGTERM, _timeout_handler)
    try:
        import torch

        torch.set_num_threads(args.threads)
        load_started = time.perf_counter()
        model, load_info = common.load_hf_reference_model(
            model_dir=model_dir,
            cpu_memory=args.cpu_memory,
            offload_dir=args.offload_dir,
            for_causal_lm=True,
        )
        receipt["load"] = {
            **load_info,
            "wall_seconds": float(time.perf_counter() - load_started),
            "max_rss_mib_after_load": max_rss_mib(),
        }

        for view in ("reference", "long"):
            data = arms[view]
            arm_started = time.perf_counter()
            input_ids = torch.from_numpy(data["inputs"])
            with torch.inference_mode():
                outputs = model(
                    input_ids=input_ids,
                    use_cache=False,
                    logits_to_keep=common.N_TARGETS,
                    return_dict=True,
                )
            logits = outputs.logits
            score = nll_from_logits(logits, data["targets"], args.logit_chunk)
            if score["target_ids_sha256"] != data["target_ids_sha256"]:
                raise RuntimeError("HF score target hash changed")
            arm_receipt = {
                **_arm_metadata(data),
                "score": score,
                "logits_shape": [int(value) for value in logits.shape],
                "logits_dtype": str(logits.dtype),
                "wall_seconds": float(time.perf_counter() - arm_started),
                "max_rss_mib": max_rss_mib(),
            }
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                arm_receipt["cuda_max_memory_allocated_bytes"] = int(
                    torch.cuda.max_memory_allocated()
                )
            receipt["arms"][view] = arm_receipt
            del input_ids, outputs, logits
            gc.collect()

        long_score = receipt["arms"]["long"]["score"]
        ref_score = receipt["arms"]["reference"]["score"]
        delta = float(long_score["mean_nll"] - ref_score["mean_nll"])
        healthy = bool(delta <= common.HG1_DELTA_NATS_MAX)
        receipt["hg1"] = {
            "delta_mean_nll_nats": delta,
            "threshold_nats": common.HG1_DELTA_NATS_MAX,
            "healthy": healthy,
            "verdict": "healthy" if healthy else "degraded",
        }
        receipt["status"] = "complete"
        receipt["wall_seconds_total"] = float(time.perf_counter() - started)
        receipt["max_rss_mib"] = max_rss_mib()
        common.write_new_json(path, receipt)
        print(json.dumps({"status": "complete", "receipt": str(path), "hg1": receipt["hg1"]}))
        return 0
    except BaseException as exc:
        receipt["status"] = "error"
        receipt["error_type"] = type(exc).__name__
        receipt["error"] = str(exc)
        receipt["traceback"] = traceback.format_exc()
        receipt["wall_seconds_total"] = float(time.perf_counter() - started)
        receipt["max_rss_mib"] = max_rss_mib()
        common.write_new_json(path, receipt)
        raise


def _analysis_markdown(core: dict[str, Any]) -> str:
    lines = [
        "# GLC P0.a HuggingFace Reference Ramp",
        "",
        "Evidence class: HuggingFace inference measurement on the pinned model and sealed E1.3b tokens.",
        "",
        f"HG1 status: **{core['hg1_status']}**.",
        "",
        "| Length | HF long NLL | HF short NLL | delta nats | HF verdict | target SHA-256 | cell wall s |",
        "|---:|---:|---:|---:|---|---|---:|",
    ]
    for row in core["cells"]:
        if row["status"] != "complete":
            lines.append(f"| {row['length']} | n/a | n/a | n/a | NOT_MEASURED | n/a | n/a |")
            continue
        lines.append(
            f"| {row['length']} | {row['long_mean_nll']:.9f} | "
            f"{row['reference_mean_nll']:.9f} | {row['delta_mean_nll_nats']:.9f} | "
            f"{row['verdict']} | `{row['target_ids_sha256']}` | {row['wall_seconds_total']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Registered healthy threshold: long mean NLL <= short mean NLL + {common.HG1_DELTA_NATS_MAX:.2f} nats per cell.",
            "",
            "A degraded cell is retained as a RED result; it is not filtered or retried away.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(args: argparse.Namespace) -> int:
    common.ensure_output_root(args.output_dir)
    _manifest, _arrays = common.validate_sealed_controls()
    cells: list[dict[str, Any]] = []
    complete_count = 0
    any_degraded = False
    for length in common.RAMP_LENGTHS:
        found = common.latest_complete_receipt(HF_OUTPUT, f"ramp_{length}")
        if found is None:
            cells.append({"length": int(length), "status": "missing"})
            continue
        path, receipt = found
        long_arm = receipt["arms"]["long"]
        ref_arm = receipt["arms"]["reference"]
        if long_arm["target_ids_sha256"] != ref_arm["target_ids_sha256"]:
            raise RuntimeError(f"HF receipt {path} lost same-target identity")
        delta = float(receipt["hg1"]["delta_mean_nll_nats"])
        verdict = "healthy" if delta <= common.HG1_DELTA_NATS_MAX else "degraded"
        complete_count += 1
        any_degraded |= verdict == "degraded"
        cells.append(
            {
                "length": int(length),
                "status": "complete",
                "receipt": str(path),
                "receipt_sha256": common.sha256_file(path),
                "long_mean_nll": float(long_arm["score"]["mean_nll"]),
                "long_ppl": float(long_arm["score"]["ppl"]),
                "reference_mean_nll": float(ref_arm["score"]["mean_nll"]),
                "reference_ppl": float(ref_arm["score"]["ppl"]),
                "delta_mean_nll_nats": delta,
                "verdict": verdict,
                "target_ids_sha256": long_arm["target_ids_sha256"],
                "wall_seconds_total": float(receipt["wall_seconds_total"]),
                "long_wall_seconds": float(long_arm["wall_seconds"]),
                "reference_wall_seconds": float(ref_arm["wall_seconds"]),
            }
        )
    if complete_count < len(common.RAMP_LENGTHS):
        hg1_status = "NOT_MEASURED"
    elif any_degraded:
        hg1_status = "FAIL_DEGRADED"
    else:
        hg1_status = "PASS_HEALTHY"
    core = {
        "schema": "glc_p0a_hf_reference_analysis_v1",
        "status": "complete" if complete_count == len(common.RAMP_LENGTHS) else "incomplete",
        "hg1_status": hg1_status,
        "complete_cells": complete_count,
        "registered_cells": len(common.RAMP_LENGTHS),
        "hg1_delta_nats_max": common.HG1_DELTA_NATS_MAX,
        "model_revision": common.MODEL_REVISION,
        "control_arrays_sha256": common.sha256_file(common.CONTROL_ARRAYS),
        "cells": cells,
        "script_sha256": common.sha256_file(SCRIPT_PATH),
    }
    digest = common.canonical_json_sha256(core)[:16]
    json_path = common.OUTPUT_ROOT / "analysis" / f"p0a_{digest}.json"
    report_path = common.OUTPUT_ROOT / f"GLC_P0A_HF_REPORT_{digest}.md"
    if not json_path.exists():
        common.write_new_json(json_path, {**core, "created_at": common.now_iso()})
    common.write_once_text(report_path, _analysis_markdown(core))
    print(
        json.dumps(
            {
                "status": core["status"],
                "hg1_status": hg1_status,
                "analysis": str(json_path),
                "report": str(report_path),
            }
        ),
        flush=True,
    )
    if args.require_complete and core["status"] != "complete":
        return 2
    return 0


def main() -> int:
    args = parse_args()
    if args.mode == "preflight":
        return preflight(args)
    if args.mode == "self-test":
        return self_test(args)
    if args.mode == "score-cell":
        return score_cell(args)
    if args.mode == "analyze":
        return analyze(args)
    raise AssertionError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())

