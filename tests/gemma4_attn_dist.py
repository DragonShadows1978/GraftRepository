"""Attention-distribution study — the OTHER side of the equation
(Architect, 2026-06-12): the K/V program quantifies key storage; this
quantifies what softmax DOES with it on Gemma 4.

Per layer (sliding and global), at the captured decode query positions:
post-softmax entropy, effective key count, and the fraction of
probability mass captured by the top 5% / 10% / 15% of keys — the
direct empirical test of APA's refine_percentile (refine covers the
top-|score| keys; mass coverage predicts how much the bulk-4bit
approximation can matter at r0.05 / r0.10 / r0.15).

2026-07-20 (IME-E1b): interface repair + E1 control-arm duty.
  * `core/gemma4_tc.py` now returns a `KVRing` from the mixer instead of
    a plain (k, v) tuple, so `new_kv[0]` raised
    `TypeError: 'KVRing' object is not subscriptable`. Repaired to read
    K via the ring's own `ordered()` accessor, which returns the VALID
    rows (`min(count, cap)`) in logical oldest-first order and unwraps a
    wrapped sliding ring. This is bound-honoring by construction: the
    serving path scores the full-cap buffer and additively masks the
    unwritten rows with -1e4; `ordered()` returns exactly the rows that
    survive that mask, so the recomputed distribution matches serving.
  * Runs as the DENSE CONTROL for docs/HYBRID_IME_E1E2_PLAN.md E1 on the
    shared workload (tests/ime_e1e2_prompts.py), emitting the same
    summary columns as tests/qwen35_attn_dist.py. Gemma's 40 sliding and
    8 global layers are reported as SEPARATE groups — different regimes,
    never pooled.

  flock -w 3600 /tmp/forge-gpu.lock python3 tests/gemma4_attn_dist.py

Evidence class: instrument measurement (distribution characterization)
only. No model-quality claims, no speed claims.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc                                      # noqa: E402
from tensor_cuda import functional as F                       # noqa: E402
from core.gemma4_tc import (Gemma4_TC, Gemma4Config,          # noqa: E402
                            KVRing, _head_rmsnorm)
from tests.ime_e1e2_prompts import build_prompts              # noqa: E402

DECODE_STEPS = 64
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

t_load = time.perf_counter()
tc.set_alloc_pooling(True)
m, info = Gemma4_TC.from_pretrained()
print(f"loaded: {info}", flush=True)
print(f"load wall-clock: {time.perf_counter() - t_load:.1f}s", flush=True)

# Tokenizer: Gemma's own. The shared workload module builds prompts from
# fixed on-disk files; passing the Gemma tokenizer keeps the SOURCE TEXT
# identical to the qwen35 run (token counts differ by tokenizer, which is
# unavoidable across architectures and is reported per prompt).
from transformers import AutoTokenizer                         # noqa: E402
_GEMMA_SNAP = os.environ.get(
    "GEMMA4_SNAP", "/mnt/ForgeRealm/models/gemma-4-12B-it")
tok = AutoTokenizer.from_pretrained(_GEMMA_SNAP)
prompts = build_prompts(tok)
for name, ids in prompts:
    print(f"prompt {name}: S={ids.shape[1]}", flush=True)

cfg = m.config
N_LAYERS = len(m.layers)
SLIDING = [i for i in range(N_LAYERS) if not Gemma4Config.is_global(i)]
GLOBAL = [i for i in range(N_LAYERS) if Gemma4Config.is_global(i)]
print(f"layers: {N_LAYERS} total = {len(SLIDING)} sliding + "
      f"{len(GLOBAL)} global (global at i%6==5)", flush=True)
for i, L in enumerate(m.layers):
    L.mixer._li = i

H = cfg.num_heads

AttnT = type(m.layers[0].mixer)
orig_call = AttnT.__call__

_probs = {}
_capturing = {"on": False}


def instrument_call(self, x, cos, sin, position_offset=0, kv_cache=None):
    """Read-only capture wrapper. The serving call runs FIRST and its
    output is returned UNCHANGED — this wrapper never perturbs attention.
    At each decode step it recomputes the post-qk-norm + post-RoPE query
    exactly as the serving path builds it, reads the post-RoPE cached K
    back out of the returned KVRing, and recomputes q.K^T in numpy.

    Gemma folds the QK scale into q_norm_w, so the serving matmul uses
    alpha=1.0 (core/gemma4_tc.py:588) — scores are raw q.K^T here too.
    """
    out, new_kv = orig_call(self, x, cos, sin, position_offset, kv_cache)
    B, L, _ = x.shape
    li = getattr(self, "_li", None)
    if not _capturing["on"] or L != 1 or li not in _probs:
        return out, new_kv
    cfg_l = self.cfg
    KV, D = self.kv_heads, self.head_dim
    # ---- q: mirrors the serving path (core/gemma4_tc.py:496-527) ----
    if self.qkv_proj is not None:
        q_lin = self.qkv_proj(x).slice(2, 0, H * D)
    else:
        q_lin = self.q_proj(x)
    q = _head_rmsnorm(q_lin, self.q_norm_w, cfg_l.rms_norm_eps, B, 1, H, D)
    q = q.reshape([B, 1, H, D]).transpose(1, 2)               # (B,H,1,D)
    if hasattr(tc, "rope_apply") and not tc.is_grad_enabled():
        q = tc.rope_apply(q, cos, sin, position_offset)
    else:
        q = F.apply_rotary(q, cos.slice(0, position_offset, 1),
                           sin.slice(0, position_offset, 1))
    qn = q.float().numpy()[0, :, 0, :]                        # (H, D)
    # ---- K: THE REPAIR. new_kv is a KVRing, not a tuple. ordered()
    # returns (k, v) valid rows only, oldest-first, wrapped ring unrolled.
    assert isinstance(new_kv, KVRing), type(new_kv)
    k_ord, _v_ord = new_kv.ordered()
    kn = k_ord.float().numpy()[0]                             # (KV, S, D)
    S = kn.shape[1]
    assert S == min(new_kv.count, new_kv.cap), (S, new_kv.count, new_kv.cap)
    group = H // KV                       # 2 (sliding, 8 kv) / 16 (global)
    sc = np.empty((H, S), np.float32)
    for h in range(H):
        sc[h] = (qn[h][None, :] * kn[h // group]).sum(-1)
    e = np.exp(sc - sc.max(-1, keepdims=True))
    _probs[li].append(e / e.sum(-1, keepdims=True))           # (H, S)
    return out, new_kv


def run_decode(ids, capture):
    """Prefill + DECODE_STEPS greedy steps. Returns final-step logits."""
    with tc.no_grad():
        lg, caches = m(ids, last_token_only=True)
    nxt = int(lg.float().numpy()[0, -1].argmax())
    off = ids.shape[1]
    last_lg = None
    _capturing["on"] = capture
    with tc.no_grad():
        for _ in range(DECODE_STEPS):
            lg, caches = m(np.array([[nxt]], np.int64), caches=caches,
                           position_offset=off)
            off += 1
            row = lg.float().numpy()[0, -1]
            last_lg = row
            nxt = int(row.argmax())
    _capturing["on"] = False
    del caches
    tc.empty_cache()
    return last_lg


# ---- parity assertion (mandatory, plan "Evidence class + scope laws") --
AttnT.__call__ = instrument_call
p0_name, p0_ids = prompts[0]
lg_uninstr = run_decode(p0_ids, capture=False)      # wrapper on, capture off
for li in range(N_LAYERS):
    _probs[li] = []
lg_instr = run_decode(p0_ids, capture=True)
parity_max = float(np.abs(lg_instr - lg_uninstr).max())
PARITY = "PASS" if parity_max == 0.0 else "FAIL"
parity_line = (f"PARITY (prompt '{p0_name}', instrumented vs uninstrumented "
               f"final-step logits): max|Δ|={parity_max:.3e} -> {PARITY}")
print(parity_line, flush=True)

summary = {"model": "gemma4-12b", "info": {k: str(v) for k, v in info.items()},
           "role": "E1 dense control arm",
           "decode_steps": DECODE_STEPS,
           "n_layers": N_LAYERS, "sliding_layers": SLIDING,
           "global_layers": GLOBAL, "num_q_heads": H,
           "sliding_kv_heads": cfg.num_kv_heads,
           "sliding_head_dim": cfg.head_dim,
           "sliding_window": cfg.sliding_window,
           "global_kv_heads": cfg.num_global_kv_heads,
           "global_head_dim": cfg.global_head_dim,
           "parity_max_abs_delta": parity_max, "parity": PARITY,
           "prompts": {}}

raw = {}
npz_out = {}
prompt_wall = {}
for pidx, (name, ids) in enumerate(prompts):
    if pidx == 0:
        prompt_wall[name] = float("nan")     # reuses the parity capture
    else:
        for li in range(N_LAYERS):
            _probs[li] = []
        t_p = time.perf_counter()
        run_decode(ids, capture=True)
        prompt_wall[name] = time.perf_counter() - t_p
    per_layer = {}
    for li in range(N_LAYERS):
        steps = _probs[li]
        ent = np.empty((len(steps), H), np.float32)
        eff = np.empty((len(steps), H), np.float32)
        top = np.empty((len(steps), H, 3), np.float32)
        skeys = np.empty(len(steps), np.int32)
        for si, p in enumerate(steps):                        # p: (H, S)
            S = p.shape[1]
            skeys[si] = S
            ent[si] = (-p * np.log(p + 1e-12)).sum(-1)
            eff[si] = np.exp(ent[si])
            srt = np.sort(p, -1)[:, ::-1]
            cum = srt.cumsum(-1)
            for j, f in enumerate((0.05, 0.10, 0.15)):
                top[si, :, j] = cum[:, max(1, int(S * f)) - 1]
        per_layer[li] = (ent, eff, top, skeys)
        npz_out[f"{name}_L{li}_entropy"] = ent
        npz_out[f"{name}_L{li}_effkeys"] = eff
        npz_out[f"{name}_L{li}_topmass"] = top
        npz_out[f"{name}_L{li}_nkeys"] = skeys
    raw[name] = per_layer
    summary["prompts"][name] = {"prefill_S": int(ids.shape[1]), "layers": {}}
    for li in range(N_LAYERS):
        ent, eff, top, skeys = per_layer[li]
        summary["prompts"][name]["layers"][str(li)] = {
            "kind": "global" if Gemma4Config.is_global(li) else "sliding",
            "mean_n_keys": float(skeys.mean()),
            "mean_entropy": float(ent.mean()),
            "mean_eff_keys": float(eff.mean()),
            "mean_top5": float(top[..., 0].mean()),
            "mean_top10": float(top[..., 1].mean()),
            "mean_top15": float(top[..., 2].mean()),
        }

AttnT.__call__ = orig_call

# ---- printed tables: SLIDING and GLOBAL reported separately ------------
HDR = (f"{'layer':>7s} {'n_keys':>7s} {'entropy':>8s} {'eff_keys':>8s} "
       f"{'top5%':>7s} {'top10%':>7s} {'top15%':>7s}")


def _agg(li):
    ents, effs, t5, t10, t15, nk = [], [], [], [], [], []
    for name, _ in prompts:
        ent, eff, top, skeys = raw[name][li]
        ents.append(ent.mean()); effs.append(eff.mean())
        t5.append(top[..., 0].mean()); t10.append(top[..., 1].mean())
        t15.append(top[..., 2].mean()); nk.append(skeys.mean())
    return (float(np.mean(ents)), float(np.mean(effs)), float(np.mean(t5)),
            float(np.mean(t10)), float(np.mean(t15)), float(np.mean(nk)))


agg = {li: _agg(li) for li in range(N_LAYERS)}
print("\n=== gemma4-12b attention distribution — E1 DENSE CONTROL "
      "(mean over Q-heads x last-64 decode steps x 3 prompts) ===",
      flush=True)
for gname, group in (("SLIDING (40 layers, window "
                      f"{cfg.sliding_window}, GQA {H}q/{cfg.num_kv_heads}kv "
                      f"hd{cfg.head_dim})", SLIDING),
                     ("GLOBAL (8 layers, full context, MQA "
                      f"{H}q/{cfg.num_global_kv_heads}kv "
                      f"hd{cfg.global_head_dim})", GLOBAL)):
    print(f"\n--- {gname} ---", flush=True)
    print(HDR, flush=True)
    for li in group:
        r = agg[li]
        print(f"  L{li:2d}   {r[5]:7.0f} {r[0]:8.3f} {r[1]:8.1f} "
              f"{r[2]:7.3f} {r[3]:7.3f} {r[4]:7.3f}", flush=True)
    a = np.array([agg[li] for li in group])
    print(f"  {'GROUP':>5s}  {a[:, 5].mean():7.0f} {a[:, 0].mean():8.3f} "
          f"{a[:, 1].mean():8.1f} {a[:, 2].mean():7.3f} "
          f"{a[:, 3].mean():7.3f} {a[:, 4].mean():7.3f}", flush=True)

summary["aggregate_over_prompts"] = {
    str(li): {"kind": "global" if Gemma4Config.is_global(li) else "sliding",
              "mean_entropy": agg[li][0], "mean_eff_keys": agg[li][1],
              "mean_top5": agg[li][2], "mean_top10": agg[li][3],
              "mean_top15": agg[li][4], "mean_n_keys": agg[li][5]}
    for li in range(N_LAYERS)}
summary["group_aggregate"] = {}
for gkey, group in (("sliding", SLIDING), ("global", GLOBAL)):
    a = np.array([agg[li] for li in group])
    summary["group_aggregate"][gkey] = {
        "n_layers": len(group), "mean_entropy": float(a[:, 0].mean()),
        "mean_eff_keys": float(a[:, 1].mean()),
        "mean_top5": float(a[:, 2].mean()),
        "mean_top10": float(a[:, 3].mean()),
        "mean_top15": float(a[:, 4].mean()),
        "mean_n_keys": float(a[:, 5].mean())}

for name, _ in prompts:
    print(f"\n--- gemma4, prompt '{name}' (mean over heads x 64 steps) ---",
          flush=True)
    for gkey, group in (("sliding", SLIDING), ("global", GLOBAL)):
        rows = [summary["prompts"][name]["layers"][str(li)] for li in group]
        print(f"  {gkey:8s} entropy={np.mean([r['mean_entropy'] for r in rows]):7.3f} "
              f"eff_keys={np.mean([r['mean_eff_keys'] for r in rows]):7.1f} "
              f"top5={np.mean([r['mean_top5'] for r in rows]):6.3f} "
              f"top10={np.mean([r['mean_top10'] for r in rows]):6.3f} "
              f"top15={np.mean([r['mean_top15'] for r in rows]):6.3f}",
              flush=True)

npz_path = os.path.join(LOG_DIR, "ime_e1_gemma.npz")
json_path = os.path.join(LOG_DIR, "ime_e1_gemma_summary.json")
np.savez(npz_path, **npz_out)
summary["prompt_wall_seconds"] = prompt_wall
with open(json_path, "w") as fh:
    json.dump(summary, fh, indent=2)
print(f"\nwrote {npz_path}", flush=True)
print(f"wrote {json_path}", flush=True)
print(f"per-prompt decode wall-clock: "
      f"{ {k: round(v, 1) for k, v in prompt_wall.items()} }", flush=True)
print(parity_line, flush=True)
print("DONE", flush=True)
