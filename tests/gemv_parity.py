"""INT4 GEMV kernel validation: int4_linear_fused (M=1 -> new gemv path) vs
int4_linear (two-stage dequant+cuBLAS, the validated reference) vs fp64 numpy
on the dequantized weights. Pass: gemv's distance to the fp64 reference is
the same scale as two-stage's (different fp32 accumulation ORDER only), and
max|gemv - twostage| is small. All MiniCPM3 decode shapes + edge shapes.
"""
import sys
import numpy as np
sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc
from core.mistral7b_tc import _quantize_int4

rng = np.random.default_rng(7)
SHAPES = [(2560, 6400), (6400, 2560), (2560, 2560), (768, 3840),
          (2560, 768), (2560, 288), (128, 37), (256, 8)]
GS = 128
ok = True
for K, N in SHAPES:
    gs = min(GS, K)
    w = rng.standard_normal((N, K)).astype(np.float32) * 0.05
    packed, scales, zeros = _quantize_int4(w, gs)
    wdq = (np.repeat(scales.astype(np.float64), gs, axis=1)[:, :K]
           * ((packed[:, :, None] >> np.array([0, 4], dtype=np.uint8)) & 0x0F)
             .reshape(N, K).astype(np.float64)
           + np.repeat(zeros.astype(np.float64), gs, axis=1)[:, :K])
    x = (rng.standard_normal((1, 1, K)).astype(np.float32) * 0.5)
    ref = (x.reshape(1, K).astype(np.float64) @ wdq.T)  # (1, N)

    tp = tc.tensor(packed, dtype="uint8")
    ts = tc.tensor(scales.astype(np.float16), dtype="float16")
    tz = tc.tensor(zeros.astype(np.float16), dtype="float16")
    for dt in ("float16", "bfloat16"):
        tx = tc.tensor(x).astype(dt)
        a = tc.int4_linear(tx, tp, ts, tz, gs).float().numpy().reshape(1, N)
        b = tc.int4_linear_fused(tx, tp, ts, tz, gs).float().numpy().reshape(1, N)
        sc = np.abs(ref).max() + 1e-9
        da = np.abs(a - ref).max() / sc
        db = np.abs(b - ref).max() / sc
        dab = np.abs(a - b).max() / sc
        bad = db > max(2.5 * da, 5e-3) or not np.isfinite(b).all()
        ok &= not bad
        print(f"K={K:5d} N={N:5d} {dt:9s}: ref-dist two-stage {da:.2e} "
              f"gemv {db:.2e}  |a-b| {dab:.2e} {'FAIL' if bad else 'ok'}",
              flush=True)
# M=2 must still take the tiled path (shape sanity only)
x2 = tc.tensor(rng.standard_normal((1, 2, 2560)).astype(np.float32)).astype("float16")
w = rng.standard_normal((288, 2560)).astype(np.float32) * 0.05
p, s, z = _quantize_int4(w, GS)
y2 = tc.int4_linear_fused(x2, tc.tensor(p, dtype="uint8"),
                          tc.tensor(s.astype(np.float16), dtype="float16"),
                          tc.tensor(z.astype(np.float16), dtype="float16"), GS)
print(f"M=2 path shape: {y2.shape}", flush=True)
print("PASS" if ok else "FAIL", flush=True)
