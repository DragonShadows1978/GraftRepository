#!/usr/bin/env python3
"""NC17-P2 smoke: load the Qwen3-1.7B INT4 adapter (standard attn), run the GT
prompt-0 prefill, print top-1 token + argmax logit vs GT, and report resident
VRAM (INT4 accounting). Fast sanity before the full parity gate / battery."""
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from core.qwen3_1p7b_tc import Qwen3_1p7b_TC  # noqa: E402
import tensor_cuda as tc  # noqa: E402

m, info = Qwen3_1p7b_TC.from_pretrained(attention_mode="standard", int4=True)
print("[smoke] load info:", info, flush=True)
vram = m.measure_vram_int4()
print("[smoke] vram:", vram, flush=True)

gt = np.load(REPO / "logs" / "nc17" / "p0_gt.npz", allow_pickle=False)
ids = gt["prompt_ids_0"].astype(np.int64)[None, :]
logits, _ = m(ids, last_token_only=True)
lg = logits.numpy()[0, -1].astype(np.float32)
top1 = int(lg.argmax())
gt_final = gt["final_logits_0"].astype(np.float32)
gt_top1 = int(gt_final.argmax())
print(f"[smoke] prompt0 tc-int4 top1={top1} logit={lg[top1]:.4f} | GT top1={gt_top1} "
      f"logit={gt_final[gt_top1]:.4f} | match={top1==gt_top1}", flush=True)
dl = (lg - gt_final)
print(f"[smoke] max|dlogit|={np.abs(dl).max():.4f} mean|dlogit|={np.abs(dl).mean():.4f}",
      flush=True)
print("[smoke] OK", flush=True)
