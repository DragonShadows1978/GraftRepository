"""Gemma 4 12B engine-vs-PyTorch parity (margin-gate methodology).

Per GT prompt: teacher-forced top-1 agreement with per-flip GT-logit
cost, last-row logit diff, per-layer hidden cosine (bisection trail),
greedy-16 continuation (informational), and L0 (sliding) / L5 (global
K=V) cache cosine vs the HF cached states.

Margin gate (registered on the Qwen3.5 port): an INT4 engine vs an fp32
GT flips near-ties by construction — exact greedy is the wrong bar.
Disagreements pass ONLY where GT itself was nearly indifferent.

  python3 tests/test_gemma4_parity.py [gt_dir]
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc                                    # noqa: E402
from core.gemma4_tc import Gemma4_TC                        # noqa: E402

GT_DIR = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("GEMMA4_GT", "/mnt/ForgeRealm/gemma4_gt_qat")

tc.set_alloc_pooling(True)
m, info = Gemma4_TC.from_pretrained()
print(f"loaded: {info}", flush=True)


def _cos(a, b):
    a, b = a.ravel(), b.ravel()
    return float(np.dot(a, b) /
                 (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


overall = True
for pi in range(4):
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

    # per-layer hidden cosine. GT hidden semantics (adjudicated from
    # checkpoint math, see ledger): hidden[0] = scaled embeds,
    # hidden[i+1] = layer-i output for i<=46, and hidden[48] is
    # POST-FINAL-NORM — so the last probe must compare the engine's
    # final-normed output, not raw layer 47 (massive-activation dims
    # make the pre-vs-post-norm cosine collapse by construction).
    cos_by_layer = []
    for li in (0, 1, 5, 11, 23, 35, 46):
        with tc.no_grad():
            _, _, h = m(ids, max_layers=li + 1)
        he = h.float().numpy()[0]
        hr = gt["hidden"][li + 1]
        cos_by_layer.append((li, round(_cos(he[-1], hr[-1]), 4)))
    with tc.no_grad():
        _, _, h = m(ids, max_layers=48)
        from core.gemma4_tc import _cast
        hn = _cast(m.norm(h)).float().numpy()[0]
    cos_by_layer.append(("fn", round(_cos(hn[-1], gt["hidden"][48][-1]), 4)))

    # greedy continuation with cache (informational; margin is the gate)
    # list() copy: the model CONSUMES the entries of the list passed in
    # (memory contract); the original list keeps the cache tuples alive
    # for the cache checks below.
    greedy_ref = gt["greedy"][S:].tolist()
    n_gen = len(greedy_ref)
    cur_caches, off = list(caches), S
    out_tok = []
    nt = int(eng[-1].argmax())
    with tc.no_grad():
        for _ in range(n_gen):
            out_tok.append(nt)
            lg, cur_caches = m(np.array([[nt]], np.int64),
                               caches=cur_caches, position_offset=off)
            off += 1
            nt = int(lg.float().numpy()[0, -1].argmax())
    g_match = int(sum(a == b for a, b in zip(out_tok, greedy_ref)))

    # cache checks vs HF cached states (post norm+rope, like ours)
    k0, v0 = caches[0]
    k5, v5 = caches[5]
    l0_cos = min(_cos(k0.float().numpy(), gt["l0_k"]),
                 _cos(v0.float().numpy(), gt["l0_v"]))
    l5_cos = min(_cos(k5.float().numpy(), gt["l5_k"]),
                 _cos(v5.float().numpy(), gt["l5_v"]))

    flip_costs = []
    for t in np.nonzero(top1_eng != top1_ref)[0]:
        flip_costs.append(float(ref[t].max() - ref[t][top1_eng[t]]))
    worst_flip = max(flip_costs, default=0.0)
    ok = (worst_flip <= 3.0 and l0_cos >= 0.99 and l5_cos >= 0.99
          and cos_by_layer[-1][1] >= 0.90)
    overall &= ok
    print(f"prompt {pi}: top1 {agree}/{S} (worst flip cost "
          f"{worst_flip:.3f}) | last|d| {last_diff:.3f} | "
          f"greedy {g_match}/{n_gen} (info) | L0kv {l0_cos:.4f} "
          f"L5kv {l5_cos:.4f} | layer-cos {cos_by_layer} | "
          f"{'OK' if ok else 'MISMATCH'}", flush=True)

print(f"PARITY: {'PASS' if overall else 'FAIL'}", flush=True)
print("DONE", flush=True)
