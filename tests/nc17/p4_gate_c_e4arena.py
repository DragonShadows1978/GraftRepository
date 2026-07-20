#!/usr/bin/env python3
"""NC17-P4 GATE C — E4-class 20-turn conversation on Qwen3-1.7B (GQA arena).

Ports tests/test_graft_gqa_arena.py (Qwen3-4B) to the 1.7B name-checker
adapter and adds the full E4 measurement surface the order asks for: recall
table, residency per turn, amnesia control. ONE persistent arena cache; layer-0
|q.k| routed mounts (the GQA dialect router, route_layer=0); swaps + evictions
across the run.

SCRIPTED / PROBES are lifted verbatim from tests/test_graft_e4_conversation.py
(14 scripted turns, 6 planted facts in turns 1/3/4/6/7/8; 6 probes). "20-turn"
= 14 scripted + 6 probe turns on the same cache.

Three arms (protocol fixed in advance):
  A  BASELINE (upper bound): full transcript in-context at every probe. This is
     the recall ceiling and the residency WORST case (transcript grows).
  B  ARENA (the system): GQAArenaCache. Scripted turns feed through the live
     region + deposit as (k,v) turn-grafts; each probe routes the bare question
     via layer-0 |q.k|, swaps the arena to the top-k by cache surgery, evicts
     live turns outside the recency window. Recall table + per-turn residency.
  C0 AMNESIA (lower bound / control): last-2-turns live window, NO mounts —
     early facts (turns 1-8, pushed out by filler 9-14) MUST fail. Confirms the
     probes are not guessable and the arm-B hits come from the grafts.

PASS: B recall at parity with A (within one probe) AND B bounded residency far
below A's growing transcript. A recall MISS is a numbered row, not hidden.

Engine selected by wrapper (int6-fork product config / bf16-canon adjudication).
"""
import argparse
import ast
import json
import re
import sys
import time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import tensor_cuda as tc  # noqa: E402  import BEFORE core.* (P3 import-order law)
from core.qwen3_1p7b_tc import Qwen3_1p7b_TC, _snap  # noqa: E402
from core.graft_arena import GQAArenaCache  # noqa: E402
from core import kv_graft  # noqa: E402
from tokenizers import Tokenizer as HFTok  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--engine", choices=["int6", "bf16"], default="int6")
ap.add_argument("--out", default=str(REPO / "logs" / "nc17" / "p4_gate_c.json"))
args = ap.parse_args()
ELABEL = "int6-fork" if args.engine == "int6" else "bf16-canon"
print(f"[gateC] engine tc: {tc.__file__}", flush=True)
if args.engine == "int6":
    assert "Project-Tensor-int6" in tc.__file__, (
        f"REFUSING: --engine int6 but tc is not the fork build: {tc.__file__}")
    assert hasattr(tc, "int6_linear_fused"), "fork engine lacks int6_linear_fused"
else:
    assert "Project-Tensor-int6" not in tc.__file__, (
        f"REFUSING: --engine bf16 but tc IS the fork build: {tc.__file__}")

NGEN = 48
LIVE_W = 2

_src = (REPO / "tests" / "test_graft_e4_conversation.py").read_text()
SCRIPTED = ast.literal_eval(re.search(r"SCRIPTED = (\[.*?\n\])\n", _src, re.S).group(1))
PROBES = ast.literal_eval(re.search(r"PROBES = (\[.*?\n\])\n", _src, re.S).group(1))

tok = HFTok.from_file(str(Path(_snap()) / "tokenizer.json"))
m, info = Qwen3_1p7b_TC.from_pretrained(attention_mode="standard",
                                        int6=(args.engine == "int6"))
print(f"[gateC] load: {info}", flush=True)
for L in m.layers:
    L.self_attn.quant_kv_cache = False


def turn_text(u, a):
    return f"User: {u}\nAssistant: {a}\n"


def hit(text, accepts):
    t = text.lower()
    return any(a in t for a in accepts)


def gen_reprefill(prompt_text):
    """Greedy decode from a fresh prefill (arms A and C0)."""
    ids = tok.encode(prompt_text).ids
    lg, caches = m(np.array([ids], dtype=np.int64), last_token_only=True)
    row = lg.numpy()[0, -1].astype(np.float32)
    pos = len(ids)
    out = [int(row.argmax())]
    for _ in range(NGEN - 1):
        lg, caches = m(np.array([[out[-1]]], dtype=np.int64), kv_caches=caches,
                       position_offset=pos, last_token_only=True)
        pos += 1
        out.append(int(lg.numpy()[0, -1].argmax()))
    txt = tok.decode(out)
    for stop in ("\nUser:", "User:", "\n\n"):
        if stop in txt:
            txt = txt.split(stop)[0]
    return txt.strip(), len(ids)


# =================================================================== Arm A
print(f"\n[gateC] ENGINE={ELABEL} === Arm A: full-transcript baseline ===",
      flush=True)
m.extend_rope(4096)
transcript = "".join(turn_text(u, a) for u, a in SCRIPTED)
arm_a = []
hits_a = 0
for q, acc in PROBES:
    ans, ptoks = gen_reprefill(transcript + f"User: {q}\nAssistant:")
    ok = hit(ans, acc)
    hits_a += ok
    arm_a.append({"q": q, "ctx_tokens": ptoks, "hit": bool(ok), "ans": ans[:80]})
    print(f"[gateC]   A [{ptoks:4d} ctx] {'HIT ' if ok else 'MISS'} | {ans[:66]!r}",
          flush=True)
print(f"[gateC] ENGINE={ELABEL} Arm A: {hits_a}/{len(PROBES)}", flush=True)

# =================================================================== Arm C0
print(f"\n[gateC] ENGINE={ELABEL} === Arm C0: amnesia control "
      f"(last {LIVE_W} turns, no mounts) ===", flush=True)
kv_graft.clear_injection(m)
arm_c = []
hits_c = 0
window = []
for u, a in SCRIPTED:
    window.append(turn_text(u, a))
    window[:] = window[-LIVE_W:]
for q, acc in PROBES:
    prompt = "".join(window) + f"User: {q}\nAssistant:"
    ans, ptoks = gen_reprefill(prompt)
    ok = hit(ans, acc)
    hits_c += ok
    arm_c.append({"q": q, "live_tokens": ptoks, "hit": bool(ok), "ans": ans[:80]})
    print(f"[gateC]   C0 [{ptoks:4d} live] {'HIT ' if ok else 'MISS'} | {ans[:60]!r}",
          flush=True)
    window.append(turn_text(q, ans))
    window[:] = window[-LIVE_W:]
print(f"[gateC] ENGINE={ELABEL} Arm C0: {hits_c}/{len(PROBES)}", flush=True)

# =================================================================== Arm B
print(f"\n[gateC] ENGINE={ELABEL} === Arm B: GQA arena (routed, one cache) ===",
      flush=True)
kv_graft.clear_injection(m)
arena = GQAArenaCache(m,
                      encode=lambda t: tok.encode(t).ids,
                      decode=lambda ids: tok.decode(ids),
                      sink_text="<conversation>\n",
                      arena_width=256, route_layer=0, topk=3, live_turns=LIVE_W,
                      cache_deposits=False)
print(f"[gateC]   arena: sink={arena.n_sink} seats width={arena.width} "
      f"live_shift={arena.live_shift}", flush=True)

feed_res = []
for i, (u, a) in enumerate(SCRIPTED):
    t0 = time.perf_counter()
    arena.feed(f"User: {u}\nAssistant: {a}\n")
    S = arena._cache_len()
    feed_res.append({"turn": i + 1, "resident": int(S)})
    print(f"[gateC]   feed turn {i+1:2d} resident={S:3d} "
          f"({time.perf_counter()-t0:.2f}s)", flush=True)

arm_b = []
hits_b = 0
max_resident_b = 0
for q, acc in PROBES:
    t0 = time.perf_counter()
    ans, info_ = arena.step(q, ngen=NGEN, deposit=False)
    dt = time.perf_counter() - t0
    ok = hit(ans, acc)
    hits_b += ok
    res = int(info_["resident"])
    max_resident_b = max(max_resident_b, res)
    arm_b.append({"q": q, "resident": res, "live_tokens": int(info_["live_tokens"]),
                  "evicted": int(info_["evicted"]), "mounts": info_["mounts"],
                  "hit": bool(ok), "ans": ans[:80]})
    print(f"[gateC]   B [res={res:3d} live={info_['live_tokens']:3d} "
          f"evict={info_['evicted']:3d} {dt:4.1f}s] mounts={info_['mounts']} "
          f"{'HIT ' if ok else 'MISS'} | {ans[:52]!r}", flush=True)
print(f"[gateC] ENGINE={ELABEL} Arm B: {hits_b}/{len(PROBES)} "
      f"(max resident {max_resident_b} seats)", flush=True)

# ------------------------------------------------------------------ verdict
max_ctx_a = max(r["ctx_tokens"] for r in arm_a)
recall_ok = hits_b >= hits_a - 1
residency_ok = max_resident_b < max_ctx_a
control_ok = hits_c < hits_a
c_pass = recall_ok and residency_ok and control_ok
print(f"\n[gateC] ENGINE={ELABEL} E4: baseline {hits_a}/6 routed-arena {hits_b}/6 "
      f"amnesia {hits_c}/6 | resident_B_max={max_resident_b} vs ctx_A_max={max_ctx_a} "
      f"-> {'PASS' if c_pass else 'FAIL'}", flush=True)

result = {
    "gate": "C_e4_arena",
    "engine": ELABEL,
    "tc_file": tc.__file__,
    "n_scripted": len(SCRIPTED),
    "n_probes": len(PROBES),
    "recall": {"baseline_A": hits_a, "arena_B": hits_b, "amnesia_C0": hits_c,
               "total": len(PROBES)},
    "residency": {"ctx_A_max": int(max_ctx_a), "resident_B_max": int(max_resident_b),
                  "arena_width": arena.width, "n_sink": arena.n_sink},
    "arm_A": arm_a,
    "arm_B": arm_b,
    "arm_C0": arm_c,
    "feed_residency": feed_res,
    "checks": {"recall_parity_within_1": bool(recall_ok),
               "residency_bounded": bool(residency_ok),
               "amnesia_control_fails": bool(control_ok)},
    "pass": bool(c_pass),
}
Path(args.out).write_text(json.dumps(result, indent=2))
print(f"[gateC] ENGINE={ELABEL} GATE C {'PASS' if c_pass else 'FAIL'} -> {args.out}",
      flush=True)
print("[gateC] DONE", flush=True)
