#!/usr/bin/env python3
"""P3 receipt (d): load_graft_layer wall time, packed-int8 vs fp16, N=20
reps median, labeled. CPU+host-RAM+device-upload timing only (no model
forward pass) — isolates the mount hook's own cost (np.load + optional
dequant + tc.tensor upload), matching what P0/P2 already measured as
"host_bytes" mount cost.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, str(REPO_ROOT))

from scripts.gpt_oss20b_stream_forward_smoke import load_graft_layer  # noqa: E402
from core.mistral7b_tc import BlockTC  # noqa: E402

QUANT_DIR = REPO_ROOT / "artifacts" / "grm_graft_quant"
N_REPS = 20
LAYER_IDX = 0  # single-layer shard timing (mount cost is per-layer, called in a loop by the harness)


def time_reps(root: Path, n: int) -> list[float]:
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        k, v, tokens, nbytes = load_graft_layer(root, LAYER_IDX, BlockTC.COMPUTE_DTYPE)
        times.append(time.perf_counter() - t0)
        del k, v
    return times


def summarize(label: str, times: list[float]) -> dict:
    return {
        "label": label,
        "n": len(times),
        "median_s": statistics.median(times),
        "mean_s": statistics.mean(times),
        "min_s": min(times),
        "max_s": max(times),
        "stdev_s": statistics.stdev(times) if len(times) > 1 else 0.0,
    }


def main() -> int:
    fp16_root = QUANT_DIR / "multifact_4k_fp16_gate" / "graft"
    packed_root = QUANT_DIR / "P3" / "packed_banks" / "multifact_8bit"

    fp16_times = time_reps(fp16_root, N_REPS)
    packed_times = time_reps(packed_root, N_REPS)

    report = {
        "schema": "grm_graft_quant_p3_mount_speed_receipt_v1",
        "layer_idx": LAYER_IDX,
        "n_reps": N_REPS,
        "fp16": summarize("fp16 (default path)", fp16_times),
        "packed_int8": summarize("packed int8 (format-2 hook)", packed_times),
    }
    report["packed_vs_fp16_median_ratio"] = (
        report["packed_int8"]["median_s"] / report["fp16"]["median_s"]
    )
    out_path = QUANT_DIR / "P3" / "mount_speed_receipt.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
