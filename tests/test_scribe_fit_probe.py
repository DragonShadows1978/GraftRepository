"""SCRIBE Phase-0 fit + throughput probe (protocol §6 Phase 0).

Measures, with the INT4 target CO-RESIDENT:
  1. VRAM: baseline (target loaded) -> + student + optimizer -> peak
     during a warm-phase training step (target prefill no_grad + student
     fwd/bwd/step). Ground truth = nvidia-smi.
  2. Throughput: warm-phase steps/s and student-forward tok/s vs target
     prefill tok/s (the §5 amortization ratio C_student/C_target).
Decides: co-resident functional loss vs alternating-swap mode; sets
ARM-S dials. Wall-clock is calendar only — never shrinks the protocol.
"""
import os, sys, subprocess, time
import numpy as np
sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc
from tensor_cuda.optim import AdamW
from core.minicpm3_tc import MiniCPM3_TC, _snap
from core import kv_graft
from scribe.student import ArmL, ArmS, StudentConfig, target_embeddings
from scribe.losses import huber_per_layer
from tokenizers import Tokenizer as HFTok


def vram_mb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                          "--format=csv,noheader,nounits"],
                         capture_output=True, text=True)
    return int(out.stdout.strip().splitlines()[0])


tok = HFTok.from_file(os.path.join(_snap(), "tokenizer.json"))
m, info = MiniCPM3_TC.from_pretrained()
m.extend_rope(4096)
tc.synchronize()
base = vram_mb()
print(f"loaded target: {info} | VRAM {base} MB", flush=True)

TEXT = ("The survey vessel charted the southern shelf for nine days. "
        "Depth soundings were taken every two hundred meters along the "
        "transect lines, and the sediment cores were labeled by station. "
        ) * 6
ids = tok.encode(TEXT).ids[:256]
S = len(ids)

# ground-truth latents for one doc (the warm-phase target)
h = kv_graft.harvest_kv_mla(m, ids)
truth_np = np.concatenate(
    [np.concatenate([h[li]["c"][0], h[li]["kpe"][0, 0]], axis=-1)[None]
     for li in range(len(m.layers))])              # (62, S, 288)
truth = tc.tensor(truth_np[None].astype(np.float32))
emb = target_embeddings(m, ids)

for name, Student in (("ARM-L", ArmL), ("ARM-S", ArmS)):
    np.random.seed(0)
    st = Student(StudentConfig())
    n_par = sum(int(np.prod(p.shape)) for p in st.parameters())
    opt = AdamW(st.parameters(), lr=3e-4)
    tc.synchronize()
    with_student = vram_mb()

    # warm-phase step: student fwd/bwd/step (target idle but resident)
    times = []
    peak = with_student
    for it in range(6):
        t0 = time.perf_counter()
        pred = st(emb)
        loss, per_layer = huber_per_layer(pred, truth)
        opt.zero_grad()
        loss.backward()
        opt.step()
        tc.synchronize()
        times.append(time.perf_counter() - t0)
        peak = max(peak, vram_mb())
    step_s = float(np.median(times[1:]))

    # functional-shape step: TARGET PREFILL (no_grad) + student step
    t0 = time.perf_counter()
    with tc.no_grad():
        _ = kv_graft.harvest_kv_mla(m, ids)
    tc.synchronize()
    teach_s = time.perf_counter() - t0
    peak = max(peak, vram_mb())

    # student pure-forward (the MINTING cost — the product's price)
    with tc.no_grad():
        t0 = time.perf_counter()
        _ = st(emb)
        tc.synchronize()
        fwd_s = time.perf_counter() - t0
    ratio = (fwd_s / S) / (teach_s / S)
    print(f"\n{name}: params {n_par/1e6:.1f}M | VRAM target-only {base} MB "
          f"-> +student {with_student} MB -> peak {peak} MB (8192 cap)",
          flush=True)
    print(f"  warm step {step_s*1000:.0f} ms @ S={S} | student fwd "
          f"{fwd_s/S*1000:.2f} ms/tok | target prefill {teach_s/S*1000:.2f} "
          f"ms/tok | C_student/C_target = {ratio:.3f}", flush=True)
    print(f"  loss[0]={None if not times else float(loss.numpy()):.4f} "
          f"per-layer err head/tail: {per_layer[0]:.3f}/{per_layer[-1]:.3f}",
          flush=True)
    del st, opt, pred, loss
    tc.empty_cache()

print(f"\nFIT VERDICT: co-resident {'OK' if peak < 7600 else 'TIGHT/SWAP'} "
      f"(peak {peak} MB vs 8GB card)", flush=True)
print("DONE", flush=True)
