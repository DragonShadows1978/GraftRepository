#!/usr/bin/env python3
"""NC17-P4 GATE A — GQA graft-vs-in-context logit EQUIVALENCE on Qwen3-1.7B.

Port of the MLA G1/G2 gate (tests/test_graft_mla_gate.py) to the GQA dialect
on the 1.7B name-checker adapter. The graft hooks are the SHARED GQAAttentionTC
machinery (core/mistral7b_tc.py) inherited by core/qwen3_1p7b_tc.py — the same
_capture / inject_kv / graft_seats / live_shift surface the Qwen3-4B adapter
uses. No adapter graft code is added: the 4B "pattern" IS inheritance of that
attention module; this gate proves it fires identically on the 1.7B.

Engine: selected by run wrapper. int6=True + fork engine on PYTHONPATH is the
product-realistic config (label ENGINE=int6-fork). bf16 canonical engine is the
adjudication re-run (label ENGINE=bf16-canon) invoked ONLY if a gate FAILS on
INT6, to separate GRM-machinery failure from quant sensitivity.

GATE A.1 (equivalence, the falsifiable prediction): logits of
  [BRIEFING + PROMPT]  in-context
vs
  harvest(BRIEFING) mounted at scale 1.0 + PROMPT alone
compared at the SAME absolute prompt seats, teacher-forced (no generation).
BOTTOM-RIGHT MASK LAW: the prompt sits in the bottom-right causal block; with
the graft catted in front at seats 0..Sg-1 and the prompt RoPE-shifted by Sg,
the prompt's causal window is exactly [graft | prompt] — identical to the
in-context [briefing | prompt] block. MARGIN PROTOCOL: a top-1 flip is a
failure ONLY at a reference near-tie margin >= NEAR_TIE (0.25); flips inside
the near-tie band are engine numeric noise, reported but not counted RED.

GATE A.2 (cached teacher-forced stream): a fixed chunk stream fed through the
KV cache with the graft mounted at chunk-0 prefill vs one full prefill. PLAIN
(no graft) establishes THIS engine's own cache-vs-prefill noise floor; GRAFT
must match that floor. graft_seats persistence across cached decode is what is
under test (the multi-turn cache law).
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import tensor_cuda as tc  # noqa: E402  import BEFORE core.* (P3 import-order law)
from core.qwen3_1p7b_tc import Qwen3_1p7b_TC, _snap  # noqa: E402
from core import kv_graft  # noqa: E402
from tokenizers import Tokenizer as HFTok  # noqa: E402

NEAR_TIE = 0.25

ap = argparse.ArgumentParser()
ap.add_argument("--engine", choices=["int6", "bf16"], default="int6")
ap.add_argument("--out", default=str(REPO / "logs" / "nc17" / "p4_gate_a.json"))
args = ap.parse_args()
ELABEL = "int6-fork" if args.engine == "int6" else "bf16-canon"
print(f"[gateA] engine tc: {tc.__file__}", flush=True)
if args.engine == "int6":
    assert "Project-Tensor-int6" in tc.__file__, (
        f"REFUSING: --engine int6 but tc is not the fork build: {tc.__file__}")
    assert hasattr(tc, "int6_linear_fused"), "fork engine lacks int6_linear_fused"
else:
    assert "Project-Tensor-int6" not in tc.__file__, (
        f"REFUSING: --engine bf16 but tc IS the fork build: {tc.__file__}")

tok = HFTok.from_file(str(Path(_snap()) / "tokenizer.json"))
m, info = Qwen3_1p7b_TC.from_pretrained(attention_mode="standard",
                                        int6=(args.engine == "int6"))
print(f"[gateA] load: {info}", flush=True)
for L in m.layers:
    L.self_attn.quant_kv_cache = False   # graft surgery needs fp16 KV
m.extend_rope(4096)

BRIEFING = ("CLASSIFIED BRIEFING.\nProject codename: AZURE HERON.\n"
            "Access code: 73-4412.\nContact: Dr. Selena Vask.\nEND BRIEFING.\n")
PROMPT = ("User: What is the project codename?\nAssistant: The codename is")

b_ids = tok.encode(BRIEFING).ids
p_ids = tok.encode(PROMPT).ids
Sg = len(b_ids)
print(f"[gateA] ENGINE={ELABEL} briefing {Sg} tok, prompt {len(p_ids)} tok",
      flush=True)


def all_logits(idlist, caches=None, pos=0):
    lg, c = m(np.array([idlist], dtype=np.int64), kv_caches=caches,
              position_offset=pos, last_token_only=False)
    return lg.numpy()[0].astype(np.float32), c


# ---------------------------------------------------- A.1 equivalence
kv_graft.clear_injection(m)
ctx, _ = all_logits(b_ids + p_ids)
ctx_p = ctx[Sg:]                                    # in-context logits @ prompt seats

harv = kv_graft.harvest_kv(m, b_ids)                # pre-RoPE (k,v) per layer
sz = sum((h["k"].nbytes + h["v"].nbytes) for h in harv if h)
kv_graft.set_injection(m, harv, scale=1.0)          # scale 1.0 — the equivalence point
gra, _ = all_logits(p_ids)                          # prompt alone, graft mounted
kv_graft.clear_injection(m)

dl = gra - ctx_p
maxd = float(np.abs(dl).max())
meand = float(np.abs(dl).mean())
gt_top1 = ctx_p.argmax(-1)
gr_top1 = gra.argmax(-1)
flip_pos = np.where(gt_top1 != gr_top1)[0]
hard = []
neartie = []
for pos in flip_pos:
    row = ctx_p[pos]
    s = np.sort(row)[::-1]
    margin = float(s[0] - s[1])                     # GT top1 vs GT-2nd gap
    (hard if margin >= NEAR_TIE else neartie).append((int(pos), margin))
a1_pass = len(hard) == 0
print(f"[gateA] ENGINE={ELABEL} A1 graft-vs-incontext scale=1.0 "
      f"max|dlogit|={maxd:.4f} mean|dlogit|={meand:.5f} "
      f"top1 flips={len(flip_pos)} (HARD>={NEAR_TIE}: {len(hard)}, "
      f"near-tie: {len(neartie)}) graft={sz/1e6:.1f}MB -> "
      f"{'PASS' if a1_pass else 'FAIL'}", flush=True)
for pos, mg in hard:
    print(f"[gateA]   HARD flip @prompt+{pos} GTmargin={mg:.4f}", flush=True)

# ---------------------------------------------------- A.2 cached stream floor
CHUNKS = [tok.encode("User: Hello, are you ready?\nAssistant: Yes, ready.").ids,
          tok.encode("\nUser: Name a river in Europe.\nAssistant: The Rhine.").ids,
          tok.encode("\nUser: What is 17+25?\nAssistant: 42.").ids,
          tok.encode("\nUser: What is the access code?\nAssistant: 73-4412.").ids]


def run_stream(with_graft):
    kv_graft.clear_injection(m)
    if with_graft:
        kv_graft.set_injection(m, harv, scale=1.0)
    full = [t for c in CHUNKS for t in c]
    A, _ = all_logits(full)                          # one-prefill ground truth
    kv_graft.clear_injection(m)
    if with_graft:
        kv_graft.set_injection(m, harv, scale=1.0)
    caches, pos, idx, rows = None, 0, 0, []
    for ci, ch in enumerate(CHUNKS):
        Lg, caches = all_logits(ch, caches, pos)
        seg = A[idx:idx + len(ch)]
        dd = float(np.abs(Lg - seg).max())
        tt = bool((Lg.argmax(-1) == seg.argmax(-1)).all())
        rows.append({"chunk": ci, "max_dlogit": dd, "all_top1_same": tt})
        pos += len(ch); idx += len(ch)
    kv_graft.clear_injection(m)
    return rows


plain = run_stream(False)
graft = run_stream(True)
plain_max = max(r["max_dlogit"] for r in plain)
graft_max = max(r["max_dlogit"] for r in graft)
a2_pass = graft_max <= max(plain_max * 2.0, plain_max + 0.05)
print(f"[gateA] ENGINE={ELABEL} A2 cache-vs-prefill floor: "
      f"PLAIN max|d|={plain_max:.4f} GRAFT(Sg={Sg}) max|d|={graft_max:.4f} -> "
      f"{'PASS' if a2_pass else 'FAIL'}", flush=True)
for tag, rows in (("PLAIN", plain), ("GRAFT", graft)):
    for r in rows:
        print(f"[gateA]   {tag} chunk{r['chunk']}: max|d|={r['max_dlogit']:.4f} "
              f"top1={'SAME' if r['all_top1_same'] else 'DIFF'}", flush=True)

result = {
    "gate": "A_equivalence",
    "engine": ELABEL,
    "tc_file": tc.__file__,
    "load_info": {k: (v if isinstance(v, (int, float, bool, str, list)) else str(v))
                  for k, v in info.items()},
    "near_tie_margin": NEAR_TIE,
    "briefing_tokens": Sg,
    "prompt_tokens": len(p_ids),
    "graft_mb_host": sz / 1e6,
    "a1_equivalence": {
        "max_dlogit": maxd, "mean_dlogit": meand,
        "top1_flips": int(len(flip_pos)),
        "hard_flips": hard, "neartie_flips": neartie,
        "pass": bool(a1_pass),
    },
    "a2_cached_stream": {
        "plain_max_dlogit": plain_max, "graft_max_dlogit": graft_max,
        "plain": plain, "graft": graft, "pass": bool(a2_pass),
    },
    "pass": bool(a1_pass and a2_pass),
}
Path(args.out).write_text(json.dumps(result, indent=2))
print(f"[gateA] ENGINE={ELABEL} GATE A "
      f"{'PASS' if result['pass'] else 'FAIL'} -> {args.out}", flush=True)
print("[gateA] DONE", flush=True)
