#!/usr/bin/env python3
"""NC17-P2: self-quantize Qwen3-1.7B projections to INT4 from the bf16 HF
snapshot and SAVE the packed artifact to weights_nc17/qwen3_1p7b_int4/ so the
on-disk size is a receipt (the battery re-quantizes at load; this is the
persisted artifact + config record, not a load path).

Writes:
  weights_nc17/qwen3_1p7b_int4/int4_weights.npz  (packed/scales/zeros per proj)
  weights_nc17/qwen3_1p7b_int4/quant_config.json (bits/group/scheme/exempt/sizes)
Uses the HOUSE _quantize_int4 (byte-identical to QuantLinearTC INT4).
"""
import glob
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from core.mistral7b_tc import _quantize_int4, GROUP_SIZE  # noqa: E402
from core.qwen3_1p7b_tc import _snap, Qwen3_1p7bConfig  # noqa: E402

OUTDIR = REPO / "weights_nc17" / "qwen3_1p7b_int4"
OUTDIR.mkdir(parents=True, exist_ok=True)

PROJ_NAMES = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
              "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def main():
    import torch
    from safetensors.torch import load_file
    d = _snap()
    W = {}
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        W.update(load_file(f))
    cfg = Qwen3_1p7bConfig()

    arrays = {}
    orig_bf16_bytes = 0
    packed_bytes = 0
    scales_bytes = 0
    zeros_bytes = 0
    n_quantized = 0
    for i in range(cfg.num_layers):
        for pn in PROJ_NAMES:
            key = f"model.layers.{i}.{pn}.weight"
            w = np.ascontiguousarray(W[key].to(torch.float32).numpy())
            orig_bf16_bytes += w.shape[0] * w.shape[1] * 2  # bf16 reference size
            packed, scales, zeros = _quantize_int4(w, GROUP_SIZE)
            tag = f"L{i}.{pn}"
            arrays[f"{tag}.packed"] = packed
            arrays[f"{tag}.scales"] = scales
            arrays[f"{tag}.zeros"] = zeros
            packed_bytes += packed.nbytes
            scales_bytes += scales.nbytes
            zeros_bytes += zeros.nbytes
            n_quantized += 1
    npz_path = OUTDIR / "int4_weights.npz"
    np.savez(npz_path, **arrays)
    npz_size = npz_path.stat().st_size

    h = hashlib.sha256()
    with open(npz_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)

    int4_total = packed_bytes + scales_bytes + zeros_bytes
    cfg_out = {
        "bits": 4,
        "group_size": GROUP_SIZE,
        "quant_scheme": "affine per-group (asymmetric); zero=group min; "
                        "scale=(max-min)/15; nibble-packed uint8 (N,K/2); "
                        "scales/zeros fp16 — house _quantize_int4",
        "symmetry": "asymmetric",
        "n_quantized_matrices": n_quantized,
        "quantized_tensors": "q/k/v/o_proj + gate/up/down_proj, all 28 layers",
        "exempt_tensors": {
            "embed_tokens": "bf16 (also serves tied head)",
            "lm_head": "TIED to bf16 embedding (P1 resident decision), 0 extra VRAM",
            "rmsnorm_all": "fp32 (input/post_attention/final)",
            "qk_norm": "fp32 (per-head q_norm/k_norm)",
        },
        "sizes_bytes": {
            "orig_bf16_proj_bytes": int(orig_bf16_bytes),
            "int4_packed_bytes": int(packed_bytes),
            "int4_scales_bytes": int(scales_bytes),
            "int4_zeros_bytes": int(zeros_bytes),
            "int4_proj_total_bytes": int(int4_total),
            "npz_on_disk_bytes": int(npz_size),
        },
        "sizes_mib": {
            "orig_bf16_proj_mib": orig_bf16_bytes / 1024**2,
            "int4_proj_total_mib": int4_total / 1024**2,
            "npz_on_disk_mib": npz_size / 1024**2,
            "compression_ratio_proj": orig_bf16_bytes / int4_total,
        },
        "artifact_path": str(npz_path),
        "artifact_sha256": h.hexdigest(),
        "source_snapshot": d,
    }
    (OUTDIR / "quant_config.json").write_text(json.dumps(cfg_out, indent=2))
    print(json.dumps(cfg_out, indent=2), flush=True)
    print(f"[p2q] wrote {npz_path} ({npz_size/1024**2:.1f} MiB) + quant_config.json",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
