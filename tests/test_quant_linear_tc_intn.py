import os
import sys

import numpy as np
import pytest

sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensor_cuda as tc  # noqa: E402
from tensor_cuda.quantization import dequantize_affine_per_group  # noqa: E402

from core.mistral7b_tc import QuantLinearTC  # noqa: E402


GROUP = 64


def _reference_output(layer, x_np):
    packed = layer.packed.numpy()
    scales = layer.scales.numpy()
    zeros = layer.zeros.numpy()
    w = dequantize_affine_per_group(
        packed, scales, zeros, layer.bits, layer.in_features, layer.group_size
    )
    return x_np.astype(np.float32) @ w.T.astype(np.float32)


@pytest.mark.parametrize("bits", [4, 3, 2])
def test_quant_linear_tc_bits_match_dequant_reference(bits):
    rng = np.random.default_rng(100 + bits)
    n, k = 48, 256
    x_np = rng.standard_normal((5, k), dtype=np.float32) * 0.25
    w_np = rng.standard_normal((n, k), dtype=np.float32) * 0.075

    layer = QuantLinearTC(w_np, group_size=GROUP, bits=bits)
    y = layer(tc.tensor(x_np, dtype="float32")).numpy().astype(np.float32)
    y_ref = _reference_output(layer, x_np)

    np.testing.assert_allclose(y, y_ref, rtol=3e-4, atol=3e-4)


@pytest.mark.parametrize("bits", [4, 3, 2])
def test_quant_linear_tc_fused_decode_matches_two_stage(bits):
    rng = np.random.default_rng(200 + bits)
    n, k = 64, 256
    x_np = rng.standard_normal((1, k), dtype=np.float32) * 0.25
    w_np = rng.standard_normal((n, k), dtype=np.float32) * 0.075
    layer = QuantLinearTC(w_np, group_size=GROUP, bits=bits)
    x = tc.tensor(x_np, dtype="float32")

    old_use = QuantLinearTC.USE_FUSED
    old_decode = QuantLinearTC.FUSED_DECODE
    old_mmax = QuantLinearTC.FUSED_M_MAX
    try:
        QuantLinearTC.USE_FUSED = False
        QuantLinearTC.FUSED_DECODE = False
        y_two = layer(x).numpy().astype(np.float32)

        QuantLinearTC.FUSED_DECODE = True
        QuantLinearTC.FUSED_M_MAX = 8
        y_fused = layer(x).numpy().astype(np.float32)
    finally:
        QuantLinearTC.USE_FUSED = old_use
        QuantLinearTC.FUSED_DECODE = old_decode
        QuantLinearTC.FUSED_M_MAX = old_mmax

    np.testing.assert_allclose(y_fused, y_two, rtol=3e-4, atol=3e-4)
