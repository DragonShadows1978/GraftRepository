"""L-func fine-tune runner: resume the warm ARM-S checkpoint, interleave
KL-through-the-reader steps with Huber anchors, checkpoint regularly.

  python3 scribe/run_func.py /mnt/ForgeRealm/scribe_mint_v1 [steps]
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc                                   # noqa: E402
from tensor_cuda.optim import AdamW                        # noqa: E402
from core.minicpm3_tc import MiniCPM3_TC, _snap            # noqa: E402
from scribe.mint import Minter                             # noqa: E402
from scribe.student import ArmS, StudentConfig             # noqa: E402
from scribe.functional import FuncTrainer                  # noqa: E402
from scribe import checkpoint as ckpt                      # noqa: E402
from tokenizers import Tokenizer as HFTok                  # noqa: E402

ROOT = sys.argv[1]
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 1200
WARM = os.path.join(ROOT, "arms.ckpt")
FUNC = os.path.join(ROOT, "arms_func.ckpt")

tok = HFTok.from_file(os.path.join(_snap(), "tokenizer.json"))
m, info = MiniCPM3_TC.from_pretrained()
m.extend_rope(4096)
tc.set_alloc_pooling(True)
print(f"loaded: {info}", flush=True)

minter = Minter(m, lambda t: tok.encode(t).ids, ROOT)
np.random.seed(0)
st = ArmS(StudentConfig())
opt = AdamW(st.parameters(), lr=1e-4)
tr = FuncTrainer(st, m, minter, opt, lambda t: tok.encode(t).ids)
src = FUNC if os.path.exists(FUNC) else WARM
opt2, meta = ckpt.load(src, st, AdamW, {"lr": 1e-4})
tr.opt = opt2
if src == FUNC:
    tr.cursor = meta["data_index"]
print(f"resumed from {src} (cursor {tr.cursor}); "
      f"{len(tr.rows)} short-doc rows", flush=True)

ema = {"kl": None, "anchor": None}
while tr.cursor < STEPS:
    kind, loss = tr.step()
    ema[kind] = loss if ema[kind] is None else 0.95 * ema[kind] + 0.05 * loss
    if tr.cursor % 50 == 0:
        print(f"step {tr.cursor}/{STEPS} | KL ema "
              f"{ema['kl'] if ema['kl'] is not None else -1:.4f} | anchor ema "
              f"{ema['anchor'] if ema['anchor'] is not None else -1:.4f}",
              flush=True)
    if tr.cursor % 200 == 0:
        ckpt.save(FUNC, st, tr.opt, data_index=tr.cursor, refine=1.0,
                  extra={"phase": "functional"})
ckpt.save(FUNC, st, tr.opt, data_index=tr.cursor, refine=1.0,
          extra={"phase": "functional"})
print(f"L-FUNC TRAINED: {tr.cursor} steps -> {FUNC}", flush=True)
print("DONE", flush=True)
