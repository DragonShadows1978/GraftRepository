#!/usr/bin/env python3
"""P3 receipt (a): pack->unpack bit-identity vs the P1 transform's
dequantized output, across all 3 frozen fp16 banks x {8, 6} (registered
gate minimum; extended here to include 4 and 3 for full-curve coverage
since the check is cheap and CPU-only).

For each bank/bits: quantize-dequantize via
`core.graft_quant.quantize_dequantize_symmetric_group32` (the P1 reference
math, reproduced bit-for-bit — see core/graft_quant.py docstring) per layer,
then pack via `pack_kv_arrays` + unpack via `unpack_kv_arrays` (the actual
packed-format code path used by the format-1/format-2 hooks), and assert
`np.array_equal` between the two. This is a CPU-only, no-GPU, bounded check.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core.graft_quant import (  # noqa: E402
    pack_kv_arrays,
    quantize_dequantize_symmetric_group32,
    unpack_kv_arrays,
)

QUANT_DIR = REPO_ROOT / "artifacts" / "grm_graft_quant"
BANKS = {
    "multifact": QUANT_DIR / "multifact_4k_fp16_gate" / "graft",
    "preference": QUANT_DIR / "preference_4k_fp16_gate" / "graft",
    "supersession": QUANT_DIR / "supersession_4k_fp16_gate" / "graft",
}
BITS_TO_CHECK = (8, 6, 4, 3)  # registered minimum is {8,6}; 4/3 added for full curve


def main() -> int:
    t0 = time.time()
    report = {
        "schema": "grm_graft_quant_p3_roundtrip_receipt_v1",
        "banks": {},
        "all_bit_identical": True,
    }
    for bank_name, bank_dir in BANKS.items():
        manifest = json.loads((bank_dir / "manifest.json").read_text(encoding="utf-8"))
        bank_report = {}
        for bits in BITS_TO_CHECK:
            n_layers = 0
            all_ok = True
            first_fail = None
            for layer_entry in manifest["layers"]:
                layer_idx = int(layer_entry["layer"])
                shard = bank_dir / f"layer_{layer_idx:03d}.npz"
                with np.load(shard) as z:
                    k = np.ascontiguousarray(z["k"])
                    v = np.ascontiguousarray(z["v"])
                ref_k = quantize_dequantize_symmetric_group32(k, bits, 32)
                ref_v = quantize_dequantize_symmetric_group32(v, bits, 32)
                packed = pack_kv_arrays({"k": k, "v": v}, bits, 32)
                unpacked = unpack_kv_arrays(packed, ["k", "v"])
                ok_k = np.array_equal(ref_k, unpacked["k"])
                ok_v = np.array_equal(ref_v, unpacked["v"])
                if not (ok_k and ok_v):
                    all_ok = False
                    if first_fail is None:
                        first_fail = {"layer": layer_idx, "k_ok": ok_k, "v_ok": ok_v}
                n_layers += 1
            bank_report[bits] = {
                "num_layers": n_layers,
                "bit_exact": all_ok,
                "first_fail": first_fail,
            }
            if not all_ok:
                report["all_bit_identical"] = False
        report["banks"][bank_name] = bank_report
    report["wall_seconds"] = time.time() - t0

    out_path = QUANT_DIR / "P3" / "roundtrip_receipt.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["all_bit_identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
