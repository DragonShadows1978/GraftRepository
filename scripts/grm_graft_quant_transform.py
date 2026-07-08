#!/usr/bin/env python3
"""Quantize-at-rest transform for format-2 GPT-OSS graft capture dirs.

Reads a captured fp16 graft dir (`layer_NNN.npz` shards, k/v shape
(1, 8, S, 64) float16, PRE-RoPE — see
artifacts/grm_graft_quant/P0_FORMAT_AND_BATTERY.md section (a)) and emits a
NEW graft dir of the same shape/dtype/manifest schema, where each shard has
been round-tripped through uniform symmetric per-group quantization at
`storage_bits` bits and dequantized back to fp16.

This is the "zero-product-code" hook P0 identified: gate scripts
(gpt_oss20b_stream_forward_smoke.load_graft_layer) mount ANY fp16 graft dir
unchanged, so a standalone script that writes a dequantized-fp16 dir is
invisible to them. No product code (core/, scripts/gpt_oss20b_*) is
modified.

Naming: the bit-depth parameter is called `storage_bits` everywhere in this
harness, NEVER `bulk_bits` — `--bulk-bits` on the gate scripts is an
unrelated APA bulk-path CLI arg (P0 risk 5); reusing that name would
collide with an orthogonal knob (cf. GHOST_PRECISION's decay_rate
precedent).

Grouping axis: tensor_cuda.quantization.quantize_affine_per_group groups
along the LAST axis of a rank-2 (out_features, in_features) matrix — i.e.
the feature/reduction axis, not any batch/sequence axis (see
Project-Tensor tensor_cuda/tensor_cuda/quantization/affine.py:142-175:
`grouped = w.reshape(out_features, groups, group_size)` reshapes
in_features into (groups, group_size), scale/zero computed per group along
that axis). The KV shards here are (1, 8, S, 64): the axis analogous to
"in_features" (a fixed-width per-vector feature axis, independent of how
many tokens/heads exist) is `head_dim` (D=64), matching how `_kv_quant` in
core/mistral7b_tc.py also reduces over the last (D) axis for its
per-vector scale. The plan text says "group-32" explicitly, so this
harness groups head_dim (64) into 2 groups of 32 along the LAST axis,
one (scale, zero) pair per (kv_head, token, group) — i.e. grouping
reshapes (1, 8, S, 64) -> (1, 8, S, 2, 32) and reduces over the trailing
32. This keeps every group's dynamic range local to 32 contiguous
channels of one token's one head, consistent with both the house
per-group convention (last/feature axis) and the plan's stated group
size, rather than grouping across tokens (which would mix unrelated
values and is not how quantize_affine_per_group's axis choice works).

Quantization is uniform SYMMETRIC (signed, zero-point-free) per group,
matching the plan's "uniform symmetric per-group quantization" line and
_kv_quant's symmetric-scale convention (scale = max|x|/qmax, q=round(x/s)
clamped to a signed range), NOT the affine (min/max + zero) variant also
present in affine.py (dequantize_affine_per_group) — the plan's own words
pick symmetric explicitly, and it also gives a bit-exact 16-bit identity
path trivially (see --storage-bits 16 special-case below).

Usage:
    grm_graft_quant_transform.py --src <fp16 graft dir> --dst <new graft dir> \
        --storage-bits {16,8,6,4,3,2}

Emits <dst>/layer_NNN.npz (k, v keys, fp16, same shape) + <dst>/manifest.json
(copied from src, layer_type/shapes preserved) + <dst>/quant_receipt.json
(per-layer RMSE/max-abs for k and v, plus a whole-bank summary and, for
bits=16, a bit-exact-identity flag).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.graft_quant import (  # noqa: E402
    GROUP_SIZE,
    quantize_dequantize_symmetric_group32 as _core_quantize_dequantize_symmetric_group32,
    rmse_maxabs,
)

HEAD_DIM = 64  # GPT-OSS-20B geometry (P0 receipt)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, required=True, help="fp16 graft dir to quantize")
    p.add_argument("--dst", type=Path, required=True, help="output dequantized-fp16 graft dir")
    p.add_argument(
        "--storage-bits",
        type=int,
        required=True,
        choices=(16, 8, 6, 4, 3, 2),
        help="quantization bit depth for the storage transform (NOT bulk_bits/APA)",
    )
    p.add_argument("--group-size", type=int, default=GROUP_SIZE)
    p.add_argument("--force", action="store_true", help="overwrite an existing --dst")
    return p.parse_args()


def qmax_signed(bits: int) -> int:
    """Symmetric signed grid max code: bits=16 uses fp16 passthrough (no
    integer grid — see quantize_dequantize_symmetric special-case), else
    2^(bits-1) - 1 (e.g. bits=8 -> 127, matching _kv_quant's scale=max|x|/127).

    Delegates to core.graft_quant.qmax_signed (P3 refactor: this script is
    no longer the math's only home, core/graft_quant.py is — see that
    module's docstring). Kept as a thin wrapper so this script's own name
    stays importable/stable for anything that already depends on it."""
    from core.graft_quant import qmax_signed as _qmax_signed
    return _qmax_signed(bits)


def quantize_dequantize_symmetric_group32(
    x: np.ndarray, bits: int, group_size: int
) -> np.ndarray:
    """Uniform symmetric per-group quantize-then-dequantize, round-trip only
    (no packed bytes emitted — this harness stores the DEQUANTIZED fp16
    result, per the plan's "capture once, quantize many" / zero-kernel-work
    design). x: (1, H, S, D) float16. Groups D into (D // group_size) groups
    along the LAST axis, one scale per (h, s, group).

    bits=16 is an exact identity path: fp16 has no lossy quantization step
    to apply at 16 bits under this scheme (the scale grid at 16-bit signed
    symmetric already exceeds fp16's own precision), so we return x
    unchanged bit-for-bit — this IS the harness's own parity gate (P0/P1
    require bits=16 == identity, verified by the caller via bit-compare).

    P3 refactor: this is now a thin wrapper around
    `core.graft_quant.quantize_dequantize_symmetric_group32` (the reusable,
    product-code home of this math — see that module's docstring). The
    math is UNCHANGED (same reshape, same qmax, same clip range, same
    dtype casts) — this function exists only so existing callers/receipts
    referencing this script's own name keep working byte-identically.
    """
    return _core_quantize_dequantize_symmetric_group32(x, bits, group_size)


def main() -> int:
    args = parse_args()
    src = args.src.expanduser().resolve()
    dst = args.dst.expanduser().resolve()
    bits = int(args.storage_bits)
    group_size = int(args.group_size)

    if not src.exists():
        raise SystemExit(f"--src does not exist: {src}")
    manifest_path = src / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"missing manifest.json under --src: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if dst.exists() and any(dst.iterdir()) and not args.force:
        raise SystemExit(f"--dst exists and is non-empty (use --force): {dst}")
    dst.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    per_layer: list[dict[str, Any]] = []
    bit_exact = True
    new_layers_manifest = []

    for layer_entry in manifest["layers"]:
        layer_idx = int(layer_entry["layer"])
        src_shard = src / f"layer_{layer_idx:03d}.npz"
        if not src_shard.exists():
            raise SystemExit(f"missing shard {src_shard}")
        with np.load(src_shard) as z:
            k = np.ascontiguousarray(z["k"])
            v = np.ascontiguousarray(z["v"])
        if k.dtype != np.float16 or v.dtype != np.float16:
            raise SystemExit(
                f"layer {layer_idx}: expected float16 k/v, got {k.dtype}/{v.dtype}"
            )
        if k.shape[-1] != HEAD_DIM:
            raise SystemExit(
                f"layer {layer_idx}: expected head_dim {HEAD_DIM}, got {k.shape[-1]}"
            )

        k_dq = quantize_dequantize_symmetric_group32(k, bits, group_size)
        v_dq = quantize_dequantize_symmetric_group32(v, bits, group_size)

        if bits == 16:
            layer_bit_exact = np.array_equal(k, k_dq) and np.array_equal(v, v_dq)
            bit_exact = bit_exact and layer_bit_exact
        else:
            layer_bit_exact = None

        k_rmse, k_maxabs = rmse_maxabs(k, k_dq)
        v_rmse, v_maxabs = rmse_maxabs(v, v_dq)

        dst_shard = dst / f"layer_{layer_idx:03d}.npz"
        np.savez(dst_shard, k=k_dq, v=v_dq)

        new_entry = dict(layer_entry)
        new_entry["path"] = str(dst_shard)
        new_entry["host_bytes"] = int(k_dq.nbytes + v_dq.nbytes)
        new_layers_manifest.append(new_entry)

        per_layer.append(
            {
                "layer": layer_idx,
                "layer_type": layer_entry.get("layer_type"),
                "k_rmse": k_rmse,
                "k_maxabs": k_maxabs,
                "v_rmse": v_rmse,
                "v_maxabs": v_maxabs,
                "bit_exact": layer_bit_exact,
            }
        )

    new_manifest = dict(manifest)
    new_manifest["layers"] = new_layers_manifest
    new_manifest["storage_bits"] = bits
    new_manifest["storage_group_size"] = group_size
    new_manifest["quant_source_dir"] = str(src)
    new_manifest["quant_transform"] = (
        "uniform_symmetric_per_group_group32_headdim_axis"
        if bits != 16
        else "identity_fp16"
    )
    (dst / "manifest.json").write_text(json.dumps(new_manifest, indent=2), encoding="utf-8")

    k_rmses = [row["k_rmse"] for row in per_layer]
    v_rmses = [row["v_rmse"] for row in per_layer]
    k_maxabses = [row["k_maxabs"] for row in per_layer]
    v_maxabses = [row["v_maxabs"] for row in per_layer]

    receipt = {
        "schema": "grm_graft_quant_transform_receipt_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "src": str(src),
        "dst": str(dst),
        "storage_bits": bits,
        "storage_group_size": group_size,
        "grouping_axis": "head_dim (last axis of (1,H,S,D)), group_size=%d" % group_size,
        "quant_kind": "uniform_symmetric_per_group" if bits != 16 else "identity_fp16",
        "bit_exact_at_16": bit_exact if bits == 16 else None,
        "num_layers": len(per_layer),
        "wall_seconds": time.time() - t0,
        "summary": {
            "k_rmse_mean": float(np.mean(k_rmses)),
            "k_rmse_max": float(np.max(k_rmses)),
            "v_rmse_mean": float(np.mean(v_rmses)),
            "v_rmse_max": float(np.max(v_rmses)),
            "k_maxabs_max": float(np.max(k_maxabses)),
            "v_maxabs_max": float(np.max(v_maxabses)),
        },
        "per_layer": per_layer,
    }
    (dst / "quant_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")

    print(json.dumps({"dst": str(dst), "storage_bits": bits, "summary": receipt["summary"],
                       "bit_exact_at_16": receipt["bit_exact_at_16"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
