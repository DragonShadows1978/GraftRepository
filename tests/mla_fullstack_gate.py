"""Full-stack decode gate: REFERENCE (all flags off, eager, two-stage INT4)
vs FAST (pool + absorbed + fused-GEMV INT4 + fused RMSNorm, no_grad).
G1: greedy token parity + teacher-forced logit diff on the same stream.
G2: arena conditions (live_shift + mounted graft).
Reference noise context: plain cache-vs-prefill floor on this model 0.34-1.28.
"""
import os, sys, numpy as np
sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensor_cuda as tc
from core.minicpm3_tc import MiniCPM3_TC, _snap
from core.mistral7b_tc import QuantLinearTC, RMSNormTC
from core import kv_graft
from tokenizers import Tokenizer as HFTok

tok = HFTok.from_file(os.path.join(_snap(), "tokenizer.json"))
m, info = MiniCPM3_TC.from_pretrained()
print(f"loaded: {info}", flush=True)
m.extend_rope(4096)

def set_flags(fast):
    QuantLinearTC.FUSED_DECODE = fast
    RMSNormTC.USE_FUSED = fast
    tc.set_alloc_pooling(fast)
    for L in m.layers:
        L.self_attn.absorbed_decode = fast

def last_logits(idlist, caches=None, pos=0):
    with tc.no_grad():
        lg, c = m(np.array([idlist], dtype=np.int64), kv_caches=caches,
                  position_offset=pos, last_token_only=True)
    return lg.numpy()[0, -1].astype(np.float32), c

PROMPT = ("The lighthouse keeper recorded the storm in his log: the glass "
          "had fallen fast since noon, and by dusk the swell was breaking "
          "clean over the mole. He wrote")

def greedy(fast, n=64):
    set_flags(fast)
    ids = tok.encode(PROMPT).ids
    row, caches = last_logits(ids)
    pos, out, rows = len(ids), [int(row.argmax())], [row]
    for _ in range(n - 1):
        row, caches = last_logits([out[-1]], caches, pos)
        pos += 1
        out.append(int(row.argmax()))
        rows.append(row)
    return out, rows

def forced(fast, stream):
    set_flags(fast)
    ids = tok.encode(PROMPT).ids
    row, caches = last_logits(ids)
    pos, rows = len(ids), [row]
    for t in stream[:-1]:
        row, caches = last_logits([t], caches, pos)
        pos += 1
        rows.append(row)
    return rows

ref_toks, ref_rows = greedy(False)
fast_toks, _ = greedy(True)
print(f"\nG1 greedy parity: {'IDENTICAL' if ref_toks == fast_toks else 'DIVERGED'}", flush=True)
print(f"  ref:  {tok.decode(ref_toks)[:66]!r}", flush=True)
print(f"  fast: {tok.decode(fast_toks)[:66]!r}", flush=True)
fr = forced(True, ref_toks)
diffs = [float(np.max(np.abs(a - b))) for a, b in zip(fr, ref_rows)]
flips = sum(int(a.argmax() != b.argmax()) for a, b in zip(fr, ref_rows))
print(f"G1 teacher-forced: max|logit diff| {max(diffs):.4f} "
      f"(mean {np.mean(diffs):.4f}), top-1 flips {flips}/{len(diffs)}", flush=True)

BRIEF = ("CLASSIFIED BRIEFING.\nProject codename: AZURE HERON.\n"
         "Access code: 73-4412.\nEND BRIEFING.\n")
set_flags(False)
harv = kv_graft.harvest_kv_mla(m, tok.encode(BRIEF).ids)
def arena_run(fast):
    set_flags(fast)
    kv_graft.clear_injection(m)
    kv_graft.set_injection_mla(m, harv)
    for L in m.layers:
        L.self_attn.live_shift = 262
    ids = tok.encode("User: What is the codename?\nAssistant:").ids
    row, caches = last_logits(ids)
    pos, out, rows = len(ids), [int(row.argmax())], [row]
    for _ in range(31):
        row, caches = last_logits([out[-1]], caches, pos)
        pos += 1
        out.append(int(row.argmax()))
        rows.append(row)
    kv_graft.clear_injection(m)
    for L in m.layers:
        L.self_attn.live_shift = None
    return out, rows
ga_t, ga_r = arena_run(False)
gb_t, gb_r = arena_run(True)
d2 = [float(np.max(np.abs(a - b))) for a, b in zip(ga_r, gb_r)]
print(f"\nG2 arena: tokens {'IDENTICAL' if ga_t == gb_t else 'DIVERGED'}, "
      f"max|logit diff| {max(d2):.4f}", flush=True)
print(f"  out: {tok.decode(gb_t)[:60]!r}", flush=True)
set_flags(False)
print("DONE", flush=True)
