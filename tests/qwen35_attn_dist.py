"""E1 attention-distribution instrument for the Qwen3.5-9B hybrid port
(plan docs/HYBRID_IME_E1E2_PLAN.md, E1). Ports tests/gemma4_attn_dist.py
to the hybrid: measures, per GQA layer x Q-head x the last-64 decode query
positions x prompt, post-softmax entropy and the fraction of attention
mass in the top 5% / 10% / 15% of keys by score.

Capture pattern (Gemma instrument): wrap Qwen35AttentionTC.__call__; at
each decode step (L==1) recompute the post-qk-norm + post-RoPE query q
from the layer input, read the FULL post-RoPE cached K back from the call's
returned new_kv, and recompute scores q.K^T in numpy. The serving path's
own output is returned UNCHANGED -- the wrapper only reads, it does not
perturb attention. (The built-in _capture hooks stash PRE-RoPE, PRE-cat
tensors, which are the wrong capture point for this measurement.)

Mandatory in-script parity assertion: final-step logits of the
instrumented run == the uninstrumented run on prompt 1 (composed path vs
composed path, apa_selective mode both).

  flock -w 3600 /tmp/forge-gpu.lock python3 tests/qwen35_attn_dist.py

Evidence class: instrument measurement (distribution characterization)
only. No model-quality claims, no speed claims.
"""
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc                                       # noqa: E402
from tensor_cuda import functional as F                        # noqa: E402
from core.qwen35_tc import Qwen35_TC, _per_head_rmsnorm, _snap  # noqa: E402
from tests.ime_e1e2_prompts import build_prompts               # noqa: E402

DECODE_STEPS = 64
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

t_load = time.perf_counter()
tc.set_alloc_pooling(True)
m, info = Qwen35_TC.from_pretrained()
print(f"loaded: {info}", flush=True)
print(f"load wall-clock: {time.perf_counter() - t_load:.1f}s", flush=True)

from transformers import AutoTokenizer                          # noqa: E402
tok = AutoTokenizer.from_pretrained(_snap())
prompts = build_prompts(tok)
for name, ids in prompts:
    print(f"prompt {name}: S={ids.shape[1]}", flush=True)

ATTN_LAYERS = m.config.attention_layer_indices()
print(f"GQA layers under study: {ATTN_LAYERS}", flush=True)
for i, L in enumerate(m.layers):
    L.mixer._li = i

cfg = m.config
H, KV, D, R = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, \
    cfg.partial_rotary_dim
GROUP = H // KV
SCALE = D ** -0.5

AttnT = type(m.layers[ATTN_LAYERS[0]].mixer)
orig_call = AttnT.__call__

# capture buffer: {layer_idx: [ per-step (H, S) prob arrays ]}
_probs = {}
_capturing = {"on": False}


def instrument_call(self, x, cos, sin, position_offset=0, kv_cache=None):
    out, new_kv = orig_call(self, x, cos, sin, position_offset, kv_cache)
    B, L, _ = x.shape
    li = getattr(self, "_li", None)
    if not _capturing["on"] or L != 1 or li not in _probs:
        return out, new_kv
    # recompute post-qk-norm + post-RoPE query q (read-only; matches the
    # serving path's q construction exactly, lines 318-354 of qwen35_tc).
    qg = self.q_proj(x).reshape([B, L, H, 2 * D])
    q = qg.slice(3, 0, D).reshape([B, L, H * D])
    q = _per_head_rmsnorm(q, self.q_norm_w, self.cfg.rms_norm_eps, B, L, H, D)
    q = q.reshape([B, L, H, D]).transpose(1, 2)                # (B,H,1,D)
    shift = getattr(self, "live_shift", None)
    if shift is None:
        shift = getattr(self, "graft_seats", 0)
    cseg = cos.slice(0, position_offset + shift, L)
    sseg = sin.slice(0, position_offset + shift, L)
    q = tc.cat([F.apply_rotary(q.slice(3, 0, R), cseg, sseg),
                q.slice(3, R, D - R)], dim=3)
    qn = q.float().numpy()[0, :, 0, :]                          # (H, D)
    kn = new_kv[0].float().numpy()[0]                           # (KV, S, D)
    S = kn.shape[1]
    # scores per Q-head against its KV group; softmax over all S cached keys
    # (decode: the single live query attends causally over the whole cache).
    sc = np.empty((H, S), np.float32)
    for h in range(H):
        kvh = h // GROUP
        sc[h] = (qn[h][None, :] * kn[kvh]).sum(-1) * SCALE
    e = np.exp(sc - sc.max(-1, keepdims=True))
    p = e / e.sum(-1, keepdims=True)                           # (H, S)
    _probs[li].append(p)
    return out, new_kv


def run_decode(ids, capture):
    """Prefill + DECODE_STEPS greedy steps. If capture, stash per-step
    per-head probs for the GQA layers. Returns final-step logits (V,)."""
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
    return last_lg


# ---- parity assertion (mandatory): prompt 1, instrumented vs not -------
AttnT.__call__ = instrument_call
p0_name, p0_ids = prompts[0]

# uninstrumented: wrapper active but capture OFF (returns orig output
# verbatim) -- this is the composed serving path.
t0 = time.perf_counter()
lg_uninstr = run_decode(p0_ids, capture=False)
# instrumented: capture ON for the GQA layers.
for li in ATTN_LAYERS:
    _probs[li] = []
lg_instr = run_decode(p0_ids, capture=True)
parity_max = float(np.abs(lg_instr - lg_uninstr).max())
PARITY = "PASS" if parity_max == 0.0 else "FAIL"
parity_line = (f"PARITY (prompt '{p0_name}', instrumented vs uninstrumented "
               f"final-step logits): max|Δ|={parity_max:.3e} -> {PARITY}")
print(parity_line, flush=True)

# reset capture buffers, run all three prompts instrumented for the tables
raw = {}   # name -> (n_layers, H, steps, [ent, S, top5, top10, top15]) later
summary = {"model": "qwen35", "info_repository": info.get("repository"),
           "decode_steps": DECODE_STEPS, "attn_layers": ATTN_LAYERS,
           "num_q_heads": H, "num_kv_heads": KV, "head_dim": D,
           "parity_max_abs_delta": parity_max, "parity": PARITY,
           "prompts": {}}

# per-prompt capture. Store raw per (layer, head, step): entropy + top-k mass.
# prompt 1's instrumented capture above is reused (saves one GPU pass).
npz_out = {}
prompt_wall = {}
for pidx, (name, ids) in enumerate(prompts):
    if pidx == 0:
        # _probs already holds prompt 0's instrumented capture from parity.
        prompt_wall[name] = float("nan")
    else:
        for li in ATTN_LAYERS:
            _probs[li] = []
        t_p = time.perf_counter()
        run_decode(ids, capture=True)
        prompt_wall[name] = time.perf_counter() - t_p
    # arrays per layer: probs list of (H, S_step) with S growing per step.
    per_layer = {}
    for li in ATTN_LAYERS:
        steps = _probs[li]                    # list length DECODE_STEPS
        ent = np.empty((len(steps), H), np.float32)
        eff = np.empty((len(steps), H), np.float32)
        top = np.empty((len(steps), H, 3), np.float32)
        for si, p in enumerate(steps):        # p: (H, S)
            S = p.shape[1]
            ent[si] = (-p * np.log(p + 1e-12)).sum(-1)
            eff[si] = np.exp(ent[si])
            srt = np.sort(p, -1)[:, ::-1]
            cum = srt.cumsum(-1)
            for j, f in enumerate((0.05, 0.10, 0.15)):
                top[si, :, j] = cum[:, max(1, int(S * f)) - 1]
        per_layer[li] = (ent, eff, top)
        npz_out[f"{name}_L{li}_entropy"] = ent
        npz_out[f"{name}_L{li}_effkeys"] = eff
        npz_out[f"{name}_L{li}_topmass"] = top
    raw[name] = per_layer

    # per-prompt per-layer summary (mean over heads and the 64 steps)
    summary["prompts"][name] = {"prefill_S": int(ids.shape[1]),
                                "layers": {}}
    for li in ATTN_LAYERS:
        ent, eff, top = per_layer[li]
        summary["prompts"][name]["layers"][str(li)] = {
            "mean_entropy": float(ent.mean()),
            "mean_eff_keys": float(eff.mean()),
            "mean_top5": float(top[..., 0].mean()),
            "mean_top10": float(top[..., 1].mean()),
            "mean_top15": float(top[..., 2].mean()),
        }

AttnT.__call__ = orig_call

# ---- printed per-layer table (mean over heads, steps, and the 3 prompts) --
print("\n=== qwen35 GQA attention distribution "
      "(mean over Q-heads x last-64 decode steps x 3 prompts) ===",
      flush=True)
print(f"{'layer':>7s} {'entropy':>8s} {'eff_keys':>8s} "
      f"{'top5%':>7s} {'top10%':>7s} {'top15%':>7s}", flush=True)
agg = {}
for li in ATTN_LAYERS:
    ents, effs, t5, t10, t15 = [], [], [], [], []
    for name, _ in prompts:
        ent, eff, top = raw[name][li]
        ents.append(ent.mean()); effs.append(eff.mean())
        t5.append(top[..., 0].mean()); t10.append(top[..., 1].mean())
        t15.append(top[..., 2].mean())
    row = (float(np.mean(ents)), float(np.mean(effs)),
           float(np.mean(t5)), float(np.mean(t10)), float(np.mean(t15)))
    agg[li] = row
    print(f"  L{li:2d}   {row[0]:8.3f} {row[1]:8.1f} "
          f"{row[2]:7.3f} {row[3]:7.3f} {row[4]:7.3f}", flush=True)
summary["aggregate_over_prompts"] = {
    str(li): {"mean_entropy": agg[li][0], "mean_eff_keys": agg[li][1],
              "mean_top5": agg[li][2], "mean_top10": agg[li][3],
              "mean_top15": agg[li][4]} for li in ATTN_LAYERS}

# also per-prompt tables for the record
for name, _ in prompts:
    print(f"\n--- qwen35, prompt '{name}' (mean over heads x 64 steps) ---",
          flush=True)
    print(f"{'layer':>7s} {'entropy':>8s} {'eff_keys':>8s} "
          f"{'top5%':>7s} {'top10%':>7s} {'top15%':>7s}", flush=True)
    for li in ATTN_LAYERS:
        s = summary["prompts"][name]["layers"][str(li)]
        print(f"  L{li:2d}   {s['mean_entropy']:8.3f} "
              f"{s['mean_eff_keys']:8.1f} {s['mean_top5']:7.3f} "
              f"{s['mean_top10']:7.3f} {s['mean_top15']:7.3f}", flush=True)

npz_path = os.path.join(LOG_DIR, "ime_e1_qwen35.npz")
json_path = os.path.join(LOG_DIR, "ime_e1_summary.json")
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
