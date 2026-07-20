#!/usr/bin/env python3
"""NC17-P4 GATE B — STATE save/restore bit-identical continuation (Qwen3-1.7B).

Session multiplexing is the product mechanism this gate proves: a live KV
session can be serialized to host, evicted from device, later restored, and its
continuation must be BIT-IDENTICAL to never having left the device — otherwise
"pause session, run another, resume" silently corrupts the paused session.

Protocol (teacher-forced + greedy, no randomness):
  1. Prefill PROMPT -> per-layer KV cache. Continue REF greedy-decode of N
     tokens straight through, recording each step's full logit row. This is
     the REFERENCE continuation.
  2. Re-prefill the SAME PROMPT to a fresh cache. SNAPSHOT that cache to host
     (numpy copies of every per-layer tensor) + the decode position + the last
     token. Tear the device cache down (drop refs, empty_cache).
  3. RESTORE: rebuild the device cache from the host snapshot (tc.tensor round
     trip) and continue greedy-decode N tokens from the restored position.
  4. Compare the restored continuation logits to the reference row-for-row.
     PASS = max|dlogit| == 0.0 AND every top-1 identical (bit-identical), for
     all N steps. A non-zero delta is a REPORTABLE failure with the number.

A graft is mounted through the whole run (graft_seats persists across the
save/restore boundary) so the gate also covers restoring a session that has a
graft resident — the multi-session product case.

Engine selected by wrapper (int6-fork product config / bf16-canon adjudication).
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

N_DECODE = 24

ap = argparse.ArgumentParser()
ap.add_argument("--engine", choices=["int6", "bf16"], default="int6")
ap.add_argument("--out", default=str(REPO / "logs" / "nc17" / "p4_gate_b.json"))
args = ap.parse_args()
ELABEL = "int6-fork" if args.engine == "int6" else "bf16-canon"
print(f"[gateB] engine tc: {tc.__file__}", flush=True)
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
print(f"[gateB] load: {info}", flush=True)
for L in m.layers:
    L.self_attn.quant_kv_cache = False
m.extend_rope(4096)

BRIEFING = ("SESSION NOTE.\nActive ticket: FORGE-2291.\nOwner: Priya "
            "Raghunathan.\nStatus: build server relocated to rack BX-44.\n")
PROMPT = ("User: Summarize the active ticket and who owns it.\nAssistant:")
b_ids = tok.encode(BRIEFING).ids
p_ids = tok.encode(PROMPT).ids
Sg = len(b_ids)

harv = kv_graft.harvest_kv(m, b_ids)


def prefill_and_first(idlist):
    kv_graft.set_injection(m, harv, scale=1.0)
    lg, caches = m(np.array([idlist], dtype=np.int64), kv_caches=None,
                   position_offset=0, last_token_only=True)
    return lg.numpy()[0, -1].astype(np.float32), caches


def decode_step(tok_id, caches, pos):
    lg, caches = m(np.array([[tok_id]], dtype=np.int64), kv_caches=caches,
                   position_offset=pos, last_token_only=True)
    return lg.numpy()[0, -1].astype(np.float32), caches


def snapshot(caches):
    """Deep host copy of every per-layer cache tensor tuple, tagged with each
    tensor's device dtype string. bf16 has NO numpy equivalent (t.numpy()
    returns fp32), so the dtype tag is load-bearing: restore must re-cast to
    the original device dtype or the KV cat hits a matmul dtype mismatch."""
    snap = []
    for cl in caches:
        snap.append(tuple((np.array(t.numpy(), copy=True), t.dtype) for t in cl))
    return snap


def restore(snap):
    """Rebuild device caches from the host snapshot, re-casting to the recorded
    device dtype (bf16 stays bf16) so the restored cache is dtype-identical to
    the one snapshotted."""
    out = []
    for cl in snap:
        out.append(tuple(tc.tensor(np.ascontiguousarray(a)).astype(dt)
                         for (a, dt) in cl))
    return out


# -------- (1) reference continuation, straight through
kv_graft.set_injection(m, harv, scale=1.0)
row, caches = prefill_and_first(p_ids)
pos = len(p_ids)
cur = int(row.argmax())
ref_rows = [row]
ref_ids = [cur]
for _ in range(N_DECODE - 1):
    row, caches = decode_step(cur, caches, pos)
    pos += 1
    cur = int(row.argmax())
    ref_rows.append(row)
    ref_ids.append(cur)
ref_rows = np.stack(ref_rows)
del caches
if hasattr(tc, "empty_cache"):
    tc.empty_cache()

# -------- (2) fresh prefill, snapshot, tear down
kv_graft.set_injection(m, harv, scale=1.0)
row2, caches2 = prefill_and_first(p_ids)
pos2 = len(p_ids)
first_tok = int(row2.argmax())
snap = snapshot(caches2)
snap_bytes = sum(a.nbytes for cl in snap for (a, _dt) in cl)
del caches2, row2
if hasattr(tc, "empty_cache"):
    tc.empty_cache()

# -------- (3) restore + continue
kv_graft.set_injection(m, harv, scale=1.0)   # graft_seats keeps the +Sg shift
caches3 = restore(snap)
cur = first_tok
row, caches3 = decode_step(cur, caches3, pos2)
pos2 += 1
res_rows = [row]
res_ids = [cur, int(row.argmax())]
cur = int(row.argmax())
for _ in range(N_DECODE - 2):
    row, caches3 = decode_step(cur, caches3, pos2)
    pos2 += 1
    cur = int(row.argmax())
    res_rows.append(row)
    res_ids.append(cur)
kv_graft.clear_injection(m)
res_rows = np.stack(res_rows)

# -------- (4) compare restored continuation to reference [1:]
ref_cmp = ref_rows[1:1 + res_rows.shape[0]]
n = min(ref_cmp.shape[0], res_rows.shape[0])
ref_cmp = ref_cmp[:n]; res_cmp = res_rows[:n]
diff = np.abs(ref_cmp - res_cmp)
maxd = float(diff.max())
meand = float(diff.mean())
top1_same = bool((ref_cmp.argmax(-1) == res_cmp.argmax(-1)).all())
bit_identical = (maxd == 0.0) and top1_same
seq_ref = ref_ids[1:1 + n]
seq_res = res_ids[1:1 + n]
seq_same = seq_ref == seq_res

print(f"[gateB] ENGINE={ELABEL} save/restore over {n} decode steps: "
      f"max|dlogit|={maxd:.6g} mean|dlogit|={meand:.6g} top1_all_same={top1_same} "
      f"token_seq_identical={seq_same} snapshot={snap_bytes/1e6:.1f}MB host -> "
      f"{'PASS (bit-identical)' if bit_identical else 'FAIL'}", flush=True)
print(f"[gateB]   ref  ids: {seq_ref}", flush=True)
print(f"[gateB]   rest ids: {seq_res}", flush=True)

result = {
    "gate": "B_state_save_restore",
    "engine": ELABEL,
    "tc_file": tc.__file__,
    "n_decode_compared": int(n),
    "snapshot_mb_host": snap_bytes / 1e6,
    "graft_seats": int(Sg),
    "max_dlogit": maxd,
    "mean_dlogit": meand,
    "top1_all_same": top1_same,
    "token_seq_identical": bool(seq_same),
    "ref_ids": seq_ref,
    "restored_ids": seq_res,
    "bit_identical": bool(bit_identical),
    "pass": bool(bit_identical),
}
Path(args.out).write_text(json.dumps(result, indent=2))
print(f"[gateB] ENGINE={ELABEL} GATE B "
      f"{'PASS' if result['pass'] else 'FAIL'} -> {args.out}", flush=True)
print("[gateB] DONE", flush=True)
