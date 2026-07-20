#!/usr/bin/env python3
"""NC17-P3: self-quantize Qwen3-1.7B projections to INT6 SYMMETRIC g128 from the
bf16 HF snapshot with the FORK engine's quantize_symmetric_per_group, and SAVE
the packed artifact to weights_nc17/qwen3_1p7b_int6/ so the on-disk size is a
receipt (the battery re-quantizes at load; this is the persisted artifact +
config record).

REQUIRES the fork engine on PYTHONPATH (tc.int6_* / quantize_symmetric_per_group
live ONLY in Project-Tensor-int6). The INT6 grid is SYMMETRIC (code=(q-32),
z=-32*s, empty zeros) — this DIFFERS from P2's asymmetric INT4 affine grid; the
difference is recorded in quant_config.json, not "fixed".

Writes:
  weights_nc17/qwen3_1p7b_int6/int6_weights.npz  (packed/scales per proj)
  weights_nc17/qwen3_1p7b_int6/quant_config.json (bits/group/scheme/exempt/sizes)
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
# Import tensor_cuda (fork, via PYTHONPATH) BEFORE core.* — core.mistral7b_tc
# hardcodes sys.path.insert(0, canonical) at import time, but Python caches the
# already-loaded fork module in sys.modules so the canonical insert is inert.
import tensor_cuda as tc  # noqa: E402  (asserts the fork build is importable)
from tensor_cuda.quantization import quantize_symmetric_per_group  # noqa: E402
from core.qwen3_1p7b_tc import _snap, Qwen3_1p7bConfig  # noqa: E402

GROUP_SIZE = 128
OUTDIR = REPO / "weights_nc17" / "qwen3_1p7b_int6"
OUTDIR.mkdir(parents=True, exist_ok=True)

PROJ_NAMES = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
              "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def main():
    print(f"[p3q] engine tc: {tc.__file__}", flush=True)
    assert "Project-Tensor-int6" in tc.__file__, (
        f"REFUSING: tc is not the fork int6 build: {tc.__file__}")
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
    n_quantized = 0
    for i in range(cfg.num_layers):
        for pn in PROJ_NAMES:
            key = f"model.layers.{i}.{pn}.weight"
            w = np.ascontiguousarray(W[key].to(torch.float32).numpy())
            orig_bf16_bytes += w.shape[0] * w.shape[1] * 2  # bf16 reference size
            q = quantize_symmetric_per_group(w, 6, GROUP_SIZE)
            tag = f"L{i}.{pn}"
            arrays[f"{tag}.packed"] = q.packed
            arrays[f"{tag}.scales"] = q.scales
            packed_bytes += q.packed.nbytes
            scales_bytes += q.scales.nbytes
            n_quantized += 1
    npz_path = OUTDIR / "int6_weights.npz"
    np.savez(npz_path, **arrays)
    npz_size = npz_path.stat().st_size

    h = hashlib.sha256()
    with open(npz_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)

    # zeros are EMPTY (symmetric grid) -> 0 resident bytes
    int6_total = packed_bytes + scales_bytes
    cfg_out = {
        "bits": 6,
        "group_size": GROUP_SIZE,
        "quant_scheme": "symmetric per-group; code=(q-32); z=-32*scale (implicit); "
                        "scale=max(max(w)/31, -min(w)/32); EMPTY fp16 zeros "
                        "selects the kernel symmetric grid; 4 codes packed per 3 "
                        "bytes — fork quantize_symmetric_per_group",
        "symmetry": "symmetric",
        "engine_build": tc.__file__,
        "engine_note": "INT6 ops exist ONLY in Project-Tensor fork branch "
                       "int6-weights; canonical tensor_cuda has NO int6 surface",
        "vs_p2_int4_scheme": "P2 INT4 is ASYMMETRIC affine (zero=group-min, "
                             "scale=(max-min)/15, nibble-packed); P3 INT6 is "
                             "SYMMETRIC (kernel-native). Difference recorded, "
                             "NOT an asymmetric INT6 variant (per order scope).",
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
            "int6_packed_bytes": int(packed_bytes),
            "int6_scales_bytes": int(scales_bytes),
            "int6_zeros_bytes": 0,
            "int6_proj_total_bytes": int(int6_total),
            "npz_on_disk_bytes": int(npz_size),
        },
        "sizes_mib": {
            "orig_bf16_proj_mib": orig_bf16_bytes / 1024**2,
            "int6_proj_total_mib": int6_total / 1024**2,
            "npz_on_disk_mib": npz_size / 1024**2,
            "compression_ratio_proj": orig_bf16_bytes / int6_total,
        },
        "artifact_path": str(npz_path),
        "artifact_sha256": h.hexdigest(),
        "source_snapshot": d,
    }
    (OUTDIR / "quant_config.json").write_text(json.dumps(cfg_out, indent=2))
    print(json.dumps(cfg_out, indent=2), flush=True)
    print(f"[p3q] wrote {npz_path} ({npz_size/1024**2:.1f} MiB) + quant_config.json",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
