"""ORDER IME-E2 — Per-channel state-readout distribution in the GDN layers.

Measures whether the per-channel readout contributions of the Gated
DeltaNet state show geometric-bulk + heavy-tail structure (IME), against
the plan's two registered nulls (docs/HYBRID_IME_E1E2_PLAN.md, E2).

Capture (composed GDN path, forced): at each decode step t, per GDN layer
and head, with post-norm query q_t (post l2norm + 1/sqrt(Dk) scale) and
post-update state S (B,H,Dk,Dv):
    c_i = || q_t[i] * S[i, :] ||_2   for each key-channel i = 1..Dk

Nulls (from the SAME captured tensors, fixed seed):
  N1 alignment: permute S's key-channel rows relative to q within each head.
  N2 moment   : S replaced by Gaussian, matched per-head mean/var.

Registered thresholds (FIXED, plan E2):
  Heavy-tail confirmed iff top-10% channel share >= 2x its N1 share in
    >= 75% of (layer, head) cells, sustained across captured decode steps.
  Stable-tail (state-APA license) iff median consecutive-step Jaccard of
    the top-10% channel set >= 0.5.

Parity (mandatory): final-step logits of an instrumented composed-path run
== an uninstrumented composed-path run on prompt 1.

  DECODE_STEPS=64 python3 tests/qwen35_state_readout_dist.py [n_prompts]

Evidence class: instrument measurement only. No model-quality/speed claims.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc                                    # noqa: E402
from core.qwen35_tc import (Qwen35_TC, GatedDeltaNetTC,     # noqa: E402
                            _cast, _softplus)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGDIR = os.path.join(_REPO, "logs")
os.makedirs(LOGDIR, exist_ok=True)

DECODE_STEPS = int(os.environ.get("DECODE_STEPS", "64"))
N_PROMPTS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
SEED = 1234

SNAP = ("/home/vader/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/"
        "snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a")

# ------------------------------------------------------------------ capture
_CAP = {"enabled": False, "step": -1}
_C = {}            # (li, step) -> c (H, Dk) real
_QS = {}           # (li, step) -> (q_vec (H,Dk), S (H,Dk,Dv))  raw for nulls
_SHAPES = {}       # li -> (q_t_shape, S_shape)


def _composed_capture(self, x, state=None):
    """Composed GDN path (verbatim from core.qwen35_tc.GatedDeltaNetTC,
    fused branch skipped) + additive q/S capture. Only stashes when
    _CAP['enabled'] and L == 1 (decode step)."""
    cfg = self.cfg
    B, L, _ = x.shape
    H, Dk, Dv = cfg.n_v_heads, cfg.d_k, cfg.d_v
    K = cfg.n_k_heads * cfg.d_k

    qkv = self.in_proj_qkv(x).float()
    z = self.in_proj_z(x).float()
    a = self.in_proj_a(x)
    b = self.in_proj_b(x)

    if state is None:
        conv_prev = tc.tensor(np.zeros((B, 3, 2 * K + H * Dv), np.float32))
    else:
        conv_prev = state[0]
    x_cat = tc.cat([conv_prev, qkv], dim=1)
    conv = (x_cat.slice(1, 0, L) * self.conv_w[0]
            + x_cat.slice(1, 1, L) * self.conv_w[1]
            + x_cat.slice(1, 2, L) * self.conv_w[2]
            + x_cat.slice(1, 3, L) * self.conv_w[3]).silu()
    new_conv = x_cat.slice(1, L, 3)

    # NOTE: fused branch intentionally omitted -> COMPOSED path always.
    q = conv.slice(2, 0, K).reshape([B, L, cfg.n_k_heads, Dk])
    k = conv.slice(2, K, K).reshape([B, L, cfg.n_k_heads, Dk])
    v = conv.slice(2, 2 * K, H * Dv).reshape([B, L, H, Dv])

    rep = H // cfg.n_k_heads
    q = q.reshape([B, L, cfg.n_k_heads, 1, Dk]).expand(
        [B, L, cfg.n_k_heads, rep, Dk]).reshape([B, L, H, Dk])
    k = k.reshape([B, L, cfg.n_k_heads, 1, Dk]).expand(
        [B, L, cfg.n_k_heads, rep, Dk]).reshape([B, L, H, Dk])

    q = q * ((q * q).sum([-1], True) + 1e-6).pow(-0.5) * (Dk ** -0.5)
    k = k * ((k * k).sum([-1], True) + 1e-6).pow(-0.5)

    beta = b.sigmoid()
    g = self.neg_A * _softplus(a + self.dt_bias)

    S = (tc.tensor(np.zeros((B, H, Dk, Dv), np.float32))
         if state is None else state[1])
    outs = []
    do_cap = _CAP["enabled"] and L == 1
    li = getattr(self, "_li", None)
    for t in range(L):
        q_t = q.slice(1, t, 1).reshape([B, H, Dk, 1])
        k_t = k.slice(1, t, 1).reshape([B, H, Dk, 1])
        v_t = v.slice(1, t, 1).reshape([B, H, 1, Dv])
        g_t = g.slice(1, t, 1).reshape([B, H, 1, 1]).exp()
        beta_t = beta.slice(1, t, 1).reshape([B, H, 1, 1])
        S = S * g_t
        kv_mem = (S * k_t).sum([2], True)
        delta = (v_t - kv_mem) * beta_t
        S = S + k_t * delta
        outs.append((S * q_t).sum([2], True))
        if do_cap:
            q_np = q_t.float().numpy()[0, :, :, 0]         # (H, Dk)
            S_np = S.float().numpy()[0]                     # (H, Dk, Dv)
            contrib = q_np[:, :, None] * S_np               # (H, Dk, Dv)
            c = np.linalg.norm(contrib, axis=2)             # (H, Dk)
            step = _CAP["step"]
            _C[(li, step)] = c.astype(np.float32)
            _QS[(li, step)] = (q_np.astype(np.float32),
                               S_np.astype(np.float32))
            if li not in _SHAPES:
                _SHAPES[li] = (tuple(q_t.shape), tuple(S.shape))
    o = tc.cat(outs, dim=2)
    o = o.transpose(1, 2)

    of = o.reshape([B, L, H, Dv])
    ms = (of * of).mean([-1], True)
    of = of * (ms + cfg.rms_norm_eps).pow(-0.5) * self.norm_w
    of = of * z.reshape([B, L, H, Dv]).silu()
    out = self.out_proj(_cast(of.reshape([B, L, H * Dv])))
    return out, (new_conv, S)


def main():
    from transformers import AutoTokenizer
    import tests.ime_e1e2_prompts as P

    prompt_src = ("copied from E1 seat's file "
                  "/mnt/ForgeRealm/GraftRepository-wt-e1/tests/"
                  "ime_e1e2_prompts.py (identical prompt set)")

    tc.set_alloc_pooling(True)
    # FORCE COMPOSED PATH (order-critical): fused kernel internalizes q.
    GatedDeltaNetTC.USE_FUSED_STEP = False
    GatedDeltaNetTC.__call__ = _composed_capture

    tok = AutoTokenizer.from_pretrained(SNAP)
    m, info = Qwen35_TC.from_pretrained()
    for i, L in enumerate(m.layers):
        L.mixer._li = i
    gdn_layers = [i for i in range(m.config.num_layers)
                  if not m.config.is_attention(i)]
    print(f"loaded: {info}", flush=True)
    print(f"GDN layers ({len(gdn_layers)}): {gdn_layers}", flush=True)
    print(f"composed forced: USE_FUSED_STEP={GatedDeltaNetTC.USE_FUSED_STEP}"
          f" | decode steps={DECODE_STEPS} | prompts={N_PROMPTS}"
          f" | prompt_src={prompt_src}", flush=True)

    built = P.build_prompts(tok)[:N_PROMPTS]
    prompt_names = [n for n, _ in built]
    prompt_ids = [ids for _, ids in built]
    for pi, ids in enumerate(prompt_ids):
        print(f"  prompt {pi} ({prompt_names[pi]}): S={ids.shape[1]}",
              flush=True)

    t_wall0 = time.perf_counter()

    ids0 = prompt_ids[0]

    def run_decode(ids, capture, n_steps):
        _C.clear(); _QS.clear()
        _CAP["enabled"] = capture
        with tc.no_grad():
            lg, caches = m(ids, last_token_only=True)
            nxt = int(lg.float().numpy()[0, -1].argmax())
            off = ids.shape[1]
            final_lg = None
            for s in range(n_steps):
                _CAP["step"] = s
                lg, caches = m(np.array([[nxt]], np.int64), caches=caches,
                               position_offset=off)
                off += 1
                final_lg = lg.float().numpy()[0, -1]
                nxt = int(final_lg.argmax())
        return final_lg

    # -------- Parity (mandatory): prompt 0, instrumented composed vs
    # uninstrumented composed. Both composed (fused off); only _CAP toggles.
    par_steps = 2
    fl_instr = run_decode(ids0, True, par_steps)
    fl_plain = run_decode(ids0, False, par_steps)
    _CAP["enabled"] = False
    par_max = float(np.abs(fl_instr - fl_plain).max())
    par_bit = bool(np.array_equal(fl_instr, fl_plain))
    parity_line = (f"PARITY (composed instrumented vs composed uninstrumented,"
                   f" prompt 0, final-step logits): "
                   f"bit-identical={par_bit} max|delta|={par_max:.3e} -> "
                   f"{'PASS' if par_bit else 'FAIL'}")
    print(parity_line, flush=True)

    # -------- Full capture run per prompt.
    all_C = {}
    all_QS = {}
    for pi, ids in enumerate(prompt_ids):
        _C.clear(); _QS.clear()
        _CAP["enabled"] = True
        with tc.no_grad():
            lg, caches = m(ids, last_token_only=True)
            nxt = int(lg.float().numpy()[0, -1].argmax())
            off = ids.shape[1]
            for s in range(DECODE_STEPS):
                _CAP["step"] = s
                lg, caches = m(np.array([[nxt]], np.int64), caches=caches,
                               position_offset=off)
                off += 1
                nxt = int(lg.float().numpy()[0, -1].argmax())
        _CAP["enabled"] = False
        for (li, step), c in _C.items():
            all_C[(pi, li, step)] = c
        for (li, step), qs in _QS.items():
            all_QS[(pi, li, step)] = qs
        print(f"  captured prompt {pi}: "
              f"{sum(1 for k in all_C if k[0]==pi)} (layer,step) cells",
              flush=True)

    wall = time.perf_counter() - t_wall0
    print(f"CAPTURE wall-clock (this process, incl parity): {wall:.1f}s",
          flush=True)

    # ---------------------------------------------------------- statistics
    rng = np.random.default_rng(SEED)
    Dk = m.config.d_k
    top_k = max(1, int(round(0.10 * Dk)))

    def top_share(c_row):
        s = np.sort(c_row)[::-1]
        tot = s.sum() + 1e-30
        return float(s[:top_k].sum() / tot)

    def top_set(c_row):
        return set(np.argsort(c_row)[::-1][:top_k].tolist())

    steps_sorted = sorted({k[2] for k in all_C})
    per_cell = {}
    decay_real = np.zeros(Dk); decay_n1 = np.zeros(Dk); decay_n2 = np.zeros(Dk)
    decay_n = 0

    layers = sorted({k[1] for k in all_C})
    n_v_heads = m.config.n_v_heads
    for pi in range(len(prompt_ids)):
        for li in layers:
            for h in range(n_v_heads):
                real_shares, n1_shares, n2_shares = [], [], []
                top_sets_by_step = {}
                for step in steps_sorted:
                    key = (pi, li, step)
                    if key not in all_C:
                        continue
                    c = all_C[key][h]
                    q_np, S_np = all_QS[key]
                    qh = q_np[h]
                    Sh = S_np[h]
                    perm = rng.permutation(Dk)
                    Sh_n1 = Sh[perm]
                    c_n1 = np.linalg.norm(qh[:, None] * Sh_n1, axis=1)
                    mu = Sh.mean(); var = Sh.var()
                    Sh_n2 = rng.normal(mu, np.sqrt(var + 1e-30),
                                       size=Sh.shape).astype(np.float32)
                    c_n2 = np.linalg.norm(qh[:, None] * Sh_n2, axis=1)

                    real_shares.append(top_share(c))
                    n1_shares.append(top_share(c_n1))
                    n2_shares.append(top_share(c_n2))
                    top_sets_by_step[step] = top_set(c)

                    decay_real += np.sort(c)[::-1]
                    decay_n1 += np.sort(c_n1)[::-1]
                    decay_n2 += np.sort(c_n2)[::-1]
                    decay_n += 1
                if not real_shares:
                    continue
                jac = []
                sk = sorted(top_sets_by_step)
                for a_, b_ in zip(sk, sk[1:]):
                    A, Bs = top_sets_by_step[a_], top_sets_by_step[b_]
                    u = len(A | Bs)
                    jac.append(len(A & Bs) / u if u else 1.0)
                per_cell[(pi, li, h)] = {
                    "real_share": float(np.mean(real_shares)),
                    "n1_share": float(np.mean(n1_shares)),
                    "n2_share": float(np.mean(n2_shares)),
                    "median_jaccard": float(np.median(jac)) if jac else float("nan"),
                    "n_steps": len(real_shares),
                }
    decay_real /= max(1, decay_n)
    decay_n1 /= max(1, decay_n)
    decay_n2 /= max(1, decay_n)

    cells = list(per_cell.values())
    n_cells = len(cells)
    ge2x = sum(1 for c in cells if c["real_share"] >= 2.0 * c["n1_share"])
    frac_ge2x = ge2x / n_cells if n_cells else 0.0
    heavy_tail_pass = frac_ge2x >= 0.75

    med_jac = float(np.median([c["median_jaccard"] for c in cells
                               if not np.isnan(c["median_jaccard"])])) \
        if cells else float("nan")
    stable_tail_pass = (not np.isnan(med_jac)) and med_jac >= 0.5

    per_layer = {}
    for li in layers:
        rs = [c["real_share"] for (pi, l, h), c in per_cell.items() if l == li]
        n1 = [c["n1_share"] for (pi, l, h), c in per_cell.items() if l == li]
        n2 = [c["n2_share"] for (pi, l, h), c in per_cell.items() if l == li]
        jc = [c["median_jaccard"] for (pi, l, h), c in per_cell.items()
              if l == li and not np.isnan(c["median_jaccard"])]
        cell_ge2x = sum(1 for (pi, l, h), c in per_cell.items()
                        if l == li and c["real_share"] >= 2.0 * c["n1_share"])
        cell_tot = sum(1 for (pi, l, h) in per_cell if l == li)
        per_layer[li] = {
            "real": float(np.mean(rs)), "n1": float(np.mean(n1)),
            "n2": float(np.mean(n2)),
            "jaccard": float(np.median(jc)) if jc else float("nan"),
            "frac_ge2x": cell_ge2x / cell_tot if cell_tot else 0.0,
        }

    # ------------------------------------------------------------- outputs
    print("", flush=True)
    print(f"{'layer':>6s} {'real10%':>8s} {'N1_10%':>8s} {'N2_10%':>8s} "
          f"{'r/N1':>6s} {'cells>=2xN1':>11s} {'medJac':>7s}", flush=True)
    for li in layers:
        d = per_layer[li]
        ratio = d["real"] / (d["n1"] + 1e-30)
        print(f"  L{li:2d}  {d['real']:8.3f} {d['n1']:8.3f} {d['n2']:8.3f} "
              f"{ratio:6.2f} {d['frac_ge2x']*100:9.1f}%  {d['jaccard']:7.3f}",
              flush=True)
    print("", flush=True)
    print(f"THRESHOLD 1 (Heavy-tail): real>=2x N1 in {frac_ge2x*100:.1f}% of "
          f"{n_cells} (layer,head) cells (need >=75.0%) -> "
          f"{'PASS' if heavy_tail_pass else 'FAIL'}", flush=True)
    print(f"THRESHOLD 2 (Stable-tail / state-APA license): median "
          f"consecutive-step Jaccard = {med_jac:.4f} (need >=0.5) -> "
          f"{'PASS' if stable_tail_pass else 'FAIL'}", flush=True)

    npz_path = os.path.join(LOGDIR, "ime_e2_qwen35.npz")
    keys = sorted(all_C.keys())
    c_stack = np.stack([all_C[k] for k in keys]) if keys else np.zeros((0,))
    key_arr = np.array(keys, dtype=np.int64)
    np.savez_compressed(
        npz_path,
        c_vectors=c_stack,                 # (n_cells, H, Dk)
        c_keys=key_arr,                    # (n_cells, 3) = (pi, li, step)
        decay_real=decay_real, decay_n1=decay_n1, decay_n2=decay_n2,
        per_cell_real=np.array([c["real_share"] for c in cells]),
        per_cell_n1=np.array([c["n1_share"] for c in cells]),
        per_cell_n2=np.array([c["n2_share"] for c in cells]),
        per_cell_jac=np.array([c["median_jaccard"] for c in cells]),
        seed=np.array([SEED]), decode_steps=np.array([DECODE_STEPS]),
    )
    summary = {
        "evidence_class": "instrument measurement (distribution characterization)",
        "prompt_src": prompt_src,
        "prompts": {prompt_names[pi]: int(prompt_ids[pi].shape[1])
                    for pi in range(len(prompt_ids))},
        "decode_steps": DECODE_STEPS,
        "n_prompts": len(prompt_ids),
        "composed_forced": True,
        "seed": SEED,
        "q_S_shapes_per_layer": {str(li): {"q_t": list(_SHAPES[li][0]),
                                           "S": list(_SHAPES[li][1])}
                                 for li in sorted(_SHAPES)},
        "parity": {"bit_identical": par_bit, "max_abs_delta": par_max,
                   "line": parity_line},
        "threshold_heavy_tail": {
            "definition": "real top-10% share >= 2x N1 in >=75% of (layer,head) cells",
            "frac_cells_ge2x": frac_ge2x, "n_cells": n_cells,
            "threshold": 0.75, "verdict": "PASS" if heavy_tail_pass else "FAIL"},
        "threshold_stable_tail": {
            "definition": "median consecutive-step Jaccard of top-10% set >= 0.5",
            "median_jaccard": med_jac, "threshold": 0.5,
            "verdict": "PASS" if stable_tail_pass else "FAIL"},
        "per_layer": {str(li): per_layer[li] for li in layers},
        "wall_clock_s": wall,
    }
    json_path = os.path.join(LOGDIR, "ime_e2_summary.json")
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print("", flush=True)
    print("q/S shapes per GDN layer (q_t, S):", flush=True)
    for li in sorted(_SHAPES):
        print(f"  L{li:2d}: q_t={_SHAPES[li][0]} S={_SHAPES[li][1]}",
              flush=True)
    print(f"wrote: {npz_path}", flush=True)
    print(f"wrote: {json_path}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
