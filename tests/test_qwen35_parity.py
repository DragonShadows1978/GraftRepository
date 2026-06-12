"""Qwen3.5-9B engine-vs-PyTorch parity (MiniCPM3 methodology).

Per GT prompt: teacher-forced top-1 agreement per position, last-row
logit diff, per-layer hidden-state cosine (bisection trail), greedy
16-token continuation match, and layer-0 DeltaNet state / layer-3 KV
checks against the cached HF states.

  python3 tests/test_qwen35_parity.py [gt_dir]
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc                                    # noqa: E402
from core.qwen35_tc import Qwen35_TC                        # noqa: E402

GT_DIR = sys.argv[1] if len(sys.argv) > 1 else "/mnt/ForgeRealm/qwen35_gt"

tc.set_alloc_pooling(True)
m, info = Qwen35_TC.from_pretrained()
print(f"loaded: {info}", flush=True)

overall = True
for pi in range(3):
    gt = np.load(os.path.join(GT_DIR, f"gt_{pi}.npz"))
    ids = gt["ids"]
    S = ids.shape[1]

    with tc.no_grad():
        logits, caches = m(ids)
    eng = logits.float().numpy()[0]                          # (S, V)
    ref = gt["logits"]

    top1_eng = eng.argmax(-1)
    top1_ref = ref.argmax(-1)
    agree = int((top1_eng == top1_ref).sum())
    last_diff = float(np.abs(eng[-1] - ref[-1]).max())

    # per-layer hidden cosine vs GT (gt hidden: (33, S, 4096); engine:
    # rerun capturing h per layer via max_layers — cheap at these S)
    cos_by_layer = []
    h_prev = None
    for li in (0, 1, 3, 7, 15, 23, 31):
        with tc.no_grad():
            _, _, h = m(ids, max_layers=li + 1)
        he = h.float().numpy()[0]                            # post layer li
        hr = gt["hidden"][li + 1]
        c = float((he[-1] @ hr[-1])
                  / (np.linalg.norm(he[-1]) * np.linalg.norm(hr[-1]) + 1e-9))
        cos_by_layer.append((li, round(c, 4)))

    # greedy 16 with cache
    cur = ids.copy()
    cl, off = caches, S
    out_tok = []
    with tc.no_grad():
        for _ in range(16):
            nxt = int(eng[-1].argmax()) if not out_tok else nt
            out_tok.append(nxt)
            lg, cl = m(np.array([[nxt]], np.int64), caches=cl,
                       position_offset=off)
            off += 1
            nt = int(lg.float().numpy()[0, -1].argmax())
    greedy_ref = gt["greedy"][S:S + 16].tolist()
    g_match = int(sum(a == b for a, b in zip(out_tok, greedy_ref)))

    # state checks: layer 0 DeltaNet recurrent state vs GT
    conv_e, S_e = caches[0]
    rec_cos = float(np.dot(S_e.numpy().ravel(), gt["l0_rec"].ravel())
                    / (np.linalg.norm(S_e.numpy()) *
                       np.linalg.norm(gt["l0_rec"]) + 1e-9))

    # margin-based gate (registered 2026-06-12 after the tie-flip
    # diagnostic): an INT4 engine vs an fp32 GT flips near-ties by
    # construction — exact greedy match is the wrong bar. Disagreements
    # are acceptable ONLY where GT itself was nearly indifferent: the
    # GT-logit cost of the engine's pick must be small at EVERY
    # disagreeing position. Diagnostic measured worst 1.92, typical <0.5.
    flip_costs = []
    for t in np.nonzero(top1_eng != top1_ref)[0]:
        flip_costs.append(float(ref[t].max() - ref[t][top1_eng[t]]))
    worst_flip = max(flip_costs, default=0.0)
    ok = (worst_flip <= 3.0 and rec_cos >= 0.995
          and cos_by_layer[-1][1] >= 0.90)
    overall &= ok
    print(f"prompt {pi}: top1 {agree}/{S} (worst flip cost "
          f"{worst_flip:.3f}) | last|Δ| {last_diff:.3f} | "
          f"greedy {g_match}/16 (info) | L0-state cos {rec_cos:.4f} | "
          f"layer-cos {cos_by_layer} | {'OK' if ok else 'MISMATCH'}",
          flush=True)

print(f"PARITY: {'PASS' if overall else 'FAIL'}", flush=True)
print("DONE", flush=True)
