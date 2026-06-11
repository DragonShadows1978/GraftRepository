"""Phase-1/2 ARM training runner (ArmL floor or ArmS student). Self-contained,
resumable: picks up its checkpoint if present.

  python3 scribe/run_arm_l.py <mint_root> [epochs] [ArmL|ArmS]

Warm phase only (the floor). Logs per-layer error head/mid/tail (G1 raw
data) and per-domain loss; checkpoints every 200 steps and at end to
<mint_root>/arm_l.ckpt.
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc                                   # noqa: E402
from core.minicpm3_tc import MiniCPM3_TC, _snap            # noqa: E402
from scribe.mint import Minter                             # noqa: E402
from scribe.student import ArmL, ArmS, StudentConfig             # noqa: E402
from scribe.train import WarmTrainer                       # noqa: E402
from tokenizers import Tokenizer as HFTok                  # noqa: E402

ROOT = sys.argv[1]
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
ARM = sys.argv[3] if len(sys.argv) > 3 else "ArmL"
CKPT = os.path.join(ROOT, f"{ARM.lower()}.ckpt")

tok = HFTok.from_file(os.path.join(_snap(), "tokenizer.json"))
m, info = MiniCPM3_TC.from_pretrained()
m.extend_rope(4096)
print(f"loaded: {info}", flush=True)

minter = Minter(m, lambda t: tok.encode(t).ids, ROOT)
tc.set_alloc_pooling(True)
np.random.seed(0)
st = (ArmL if ARM == "ArmL" else ArmS)(StudentConfig())
# ArmS at window 1024 OOMs (quadratic attention activations, full graph
# retained — fit probe was at S=224; measured 2026-06-11). 512 fits.
# Warm-phase cost: chunk tails beyond 512 unseen by ArmS (logged).
tr = WarmTrainer(st, m, minter, lr=3e-4,
                 window=512 if ARM == "ArmS" else 1024)
print(f"train rows: {len(tr.rows)}", flush=True)
if os.path.exists(CKPT):
    meta = tr.load(CKPT)
    print(f"resumed at cursor {tr.cursor}", flush=True)

total = EPOCHS * len(tr.rows)
ema = None
while tr.cursor < total:
    loss = tr.step()
    ema = loss if ema is None else 0.98 * ema + 0.02 * loss
    if tr.cursor % 25 == 0:
        le = tr.layer_err
        print(f"step {tr.cursor}/{total} loss {loss:.4f} ema {ema:.4f} | "
              f"layer err L0 {le[0]:.3f} L31 {le[31]:.3f} L61 {le[61]:.3f}",
              flush=True)
    if tr.cursor % 200 == 0:
        tr.save(CKPT)
tr.save(CKPT)
print("\nG1 RAW — per-layer normalized error profile (EMA):", flush=True)
print("  " + " ".join(f"{e:.3f}" for e in tr.layer_err), flush=True)
print(f"{ARM} TRAINED: {tr.cursor} steps -> {CKPT}", flush=True)
print("DONE", flush=True)
