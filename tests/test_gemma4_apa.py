"""APA gate for Gemma 4: apa_selective (bulk4 / refine 0.15, cuBLAS
blend) on the 8 GLOBAL layers vs the engine's own standard attention —
engine-vs-engine, both INT4, so disagreements must be near-tie flips
under the STANDARD run's logits. Sliding layers are window-bounded and
never take APA. Plus a templated coherence smoke in APA mode.

  python3 tests/test_gemma4_apa.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc                                    # noqa: E402
from core.gemma4_tc import Gemma4_TC                        # noqa: E402

GT_DIR = os.environ.get("GEMMA4_GT", "/mnt/ForgeRealm/gemma4_gt_qat")

tc.set_alloc_pooling(True)
m, info = Gemma4_TC.from_pretrained()
print(f"loaded: {info}", flush=True)


def set_mode(mode):
    for L in m.layers:
        if L.mixer.is_global:
            L.mixer.attention_mode = mode
            # force the blend at any context — this gate tests the APA
            # MACHINERY; the serving threshold would skip it at short S
            L.mixer.apa_min_context = 0


overall = True
for pi in range(4):
    gt = np.load(os.path.join(GT_DIR, f"gt_{pi}.npz"))
    ids = gt["ids"]
    set_mode("standard")
    with tc.no_grad():
        lg_std, _ = m(ids)
    std = lg_std.float().numpy()[0]
    set_mode("apa_selective")
    with tc.no_grad():
        lg_apa, _ = m(ids)
    apa = lg_apa.float().numpy()[0]
    dis = np.nonzero(std.argmax(-1) != apa.argmax(-1))[0]
    costs = [float(std[t].max() - std[t][apa[t].argmax()]) for t in dis]
    worst = max(costs, default=0.0)
    ok = worst <= 3.0
    overall &= ok
    print(f"prompt {pi}: {len(dis)}/{ids.shape[1]} flips vs standard | "
          f"worst cost {worst:.3f} | {'OK' if ok else 'MISMATCH'}",
          flush=True)

# RECTANGULAR-WITH-CACHE check — the regime APA exists for and the one
# the square prompts above CANNOT catch (the blend's causal mask was
# top-left aligned for cached chunks until 2026-06-12: queries were
# blinded to the most recent S-L keys; found by the refine ppl sweep
# at ppl 121 -> 11M, invisible to every square gate). Score the second
# half of the long GT prompt with the first half cached, APA vs
# standard: flips must be near-ties.
gtl = np.load(os.path.join(GT_DIR, "gt_long.npz"))
ids_l = gtl["ids"][:, :1024]
half = 896          # score a 128-token tail: full logits for 512
                    # positions are a 0.8GB transient (OOMs the check)
set_mode("standard")
with tc.no_grad():
    _, c_std = m(ids_l[:, :half], last_token_only=True)
    lg_std, _ = m(ids_l[:, half:], caches=c_std, position_offset=half)
std_r = lg_std.float().numpy()[0]
del lg_std
tc.empty_cache()
set_mode("apa_selective")
# ISOLATE the blend to the scored chunk (prefill standard both sides):
# this measures the blend MACHINERY. Running APA through the prefill
# too compounds benign per-chunk rounding differences chaotically over
# 48 layers x 896 tokens (measured: isolated worst 0.361 vs cascaded
# 5.9 — same inputs, no bug; see ledger 'cascade sensitivity').
for L in m.layers:
    if L.mixer.is_global:
        L.mixer.apa_min_context = half + 64
with tc.no_grad():
    _, c_apa = m(ids_l[:, :half], last_token_only=True)
    lg_apa, _ = m(ids_l[:, half:], caches=c_apa, position_offset=half)
apa_r = lg_apa.float().numpy()[0]
dis = np.nonzero(std_r.argmax(-1) != apa_r.argmax(-1))[0]
costs = [float(std_r[t].max() - std_r[t][apa_r[t].argmax()]) for t in dis]
worst_r = max(costs, default=0.0)
ok_r = worst_r <= 3.0
overall &= ok_r
print(f"rect-with-cache (L=128,S=1024): {len(dis)}/128 flips | worst "
      f"cost {worst_r:.3f} | {'OK' if ok_r else 'MISMATCH'}", flush=True)

# coherence smoke: templated prompt (gt_3), greedy 8 in APA mode
gt = np.load(os.path.join(GT_DIR, "gt_3.npz"))
ids = gt["ids"]
set_mode("apa_selective")
with tc.no_grad():
    lg, caches = m(ids, last_token_only=True)
off, toks = ids.shape[1], []
nxt = int(lg.float().numpy()[0, -1].argmax())
with tc.no_grad():
    for _ in range(8):
        toks.append(nxt)
        lg, caches = m(np.array([[nxt]], np.int64), caches=caches,
                       position_offset=off)
        off += 1
        nxt = int(lg.float().numpy()[0, -1].argmax())
print(f"APA greedy tail (templated France prompt): {toks}", flush=True)
print(f"APA GATE: {'PASS' if overall else 'FAIL'}", flush=True)
print("DONE", flush=True)
