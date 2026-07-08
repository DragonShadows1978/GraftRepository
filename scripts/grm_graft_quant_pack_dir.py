#!/usr/bin/env python3
"""Standalone pack tool: quantize a captured fp16 format-2 graft dir
(`layer_NNN.npz` shards, {"k","v"} fp16, see
artifacts/grm_graft_quant/P0_FORMAT_AND_BATTERY.md section (a)) into a
PACKED graft dir at `storage_bits` bits, using the shared packed-format
core (`core/graft_quant.py`: `pack_kv_arrays` + `save_packed_npz`).

This is the P3 "opt-in flag or standalone pack tool" capture-side hook
(docs/GRM_GRAFT_QUANT_LEDGER.md P3 scope item 2): the capture path in
`gpt_oss20b_stream_forward_smoke.py` (line ~454-457) is left UNTOUCHED —
it always captures fp16, exactly as before this work order. To get a packed
graft dir, capture normally, then run this tool on the capture dir. The
mount side (`load_graft_layer`, same script) transparently detects and
dequantizes packed shards, so a packed dir mounts via the existing
`--mount-graft-dir` flag with zero further changes.

Output shards keep the same `layer_NNN.npz` naming and the same
`manifest.json` (with `storage_bits`/`format_version`/`quant_source_dir`
fields added, mirroring `grm_graft_quant_transform.py`'s manifest
convention) so `load_graft_manifest`'s `token_count` lookup still works
unchanged.

Usage:
    grm_graft_quant_pack_dir.py --src <fp16 graft dir> --dst <packed graft dir> \
        --storage-bits {8,6,4,3,2} [--group-size 32] [--force]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.graft_quant import (  # noqa: E402
    FORMAT_VERSION,
    GROUP_SIZE,
    SUPPORTED_BITS,
    save_packed_npz,
)

HEAD_DIM = 64  # GPT-OSS-20B geometry (P0 receipt)
PACKABLE_BITS = tuple(b for b in SUPPORTED_BITS if b != 16)  # 16 has no packed grid


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, required=True, help="fp16 graft dir to pack")
    p.add_argument("--dst", type=Path, required=True, help="output PACKED graft dir")
    p.add_argument("--storage-bits", type=int, required=True, choices=PACKABLE_BITS)
    p.add_argument("--group-size", type=int, default=GROUP_SIZE)
    p.add_argument("--force", action="store_true", help="overwrite an existing --dst")
    return p.parse_args()


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

        dst_shard = dst / f"layer_{layer_idx:03d}.npz"
        save_packed_npz(dst_shard, {"k": k, "v": v}, bits, group_size, compressed=False)

        new_entry = dict(layer_entry)
        new_entry["path"] = str(dst_shard)
        new_entry["host_bytes"] = int(dst_shard.stat().st_size)
        new_layers_manifest.append(new_entry)

    new_manifest = dict(manifest)
    new_manifest["layers"] = new_layers_manifest
    new_manifest["storage_bits"] = bits
    new_manifest["storage_group_size"] = group_size
    new_manifest["format_version"] = FORMAT_VERSION
    new_manifest["quant_source_dir"] = str(src)
    new_manifest["quant_transform"] = "packed_symmetric_group32_headdim_axis"
    new_manifest["created_at"] = datetime.now().isoformat(timespec="seconds")
    (dst / "manifest.json").write_text(json.dumps(new_manifest, indent=2), encoding="utf-8")

    result = {
        "dst": str(dst),
        "storage_bits": bits,
        "format_version": FORMAT_VERSION,
        "num_layers": len(new_layers_manifest),
        "wall_seconds": time.time() - t0,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
