#!/usr/bin/env python3
"""NC17-P3y — the last open NC17 cell: can a Qwen3-1.7B INT6 session hold 32,768
tokens RESIDENT with APA r0.15 active, when the KV cache is grown INCREMENTALLY
(chunked prefill at a REDUCED chunk size) instead of one-shot?

Context: the single-shot / CHUNK=1024 chunked build OOMs; the p3x APA-decode
ceiling converged to last_solid=22528 / first_oom=23552 (true cudaMalloc wall,
receipts logs/nc17/p3_ceiling_apa_decode.json). Those OOMs are PREFILL/decode
TRANSIENTS — steady residency at 22528 was only 3713 MiB, and standard attention
holds 32768 at 4831 MiB steady. This probe tests whether shrinking the chunk
(starting 512, halving on transient OOM down to a floor of 64) lets the
incremental build reach 32768 and then run APA decode over the full KV.

Method (single process, ONE probe per Gemma law):
  1. Build cache to CTX tokens by chunked prefill at CHUNK (start 512).
     On CUDA-OOM during a chunk-forward, HALVE the chunk and RETRY from the
     last-good cache state (the failed forward does not mutate the persistent
     `caches` list — a new list is only bound on success). Floor = 64; below
     that we stop and report the highest KV length reached + the verbatim
     blocker.
  2. Per growth stage record: tokens resident, elapsed_s, smi_used_mb.
  3. At CTX: run --decode-steps (default 64) cached APA decode steps, one token
     each, engagement asserted (p1_apa_engage). Record steady smi, ms/token.
  4. Emit RESULT json to stdout AND write --out (logs/nc17/p3y_32k.json).

Rails asserted in-log: fork engine, int6=True, attention_mode=apa, refine=0.15,
bulk_bits=8 on every layer.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests" / "nc17"))
import tensor_cuda as tc  # noqa: E402
from core.qwen3_1p7b_tc import Qwen3_1p7b_TC  # noqa: E402
import p1_apa_engage as engage  # noqa: E402

_ENGINE = tc.__file__
_IS_FORK = "Project-Tensor-int6" in _ENGINE


def smi_used_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            timeout=10).decode().strip().splitlines()
        return float(out[0])
    except Exception:
        return None


def is_oom(msg):
    low = msg.lower()
    return any(s in low for s in ("out of memory", "oom", "cuda_error_out",
                                  "alloc", "cublas_status_alloc", "cuda error",
                                  "cudamalloc"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--chunk", type=int, default=512, help="initial chunk size")
    ap.add_argument("--chunk-floor", type=int, default=64)
    ap.add_argument("--refine", type=float, default=0.15)
    ap.add_argument("--decode-steps", type=int, default=64)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    rec = {
        "probe": "p3y_incremental_32k",
        "ctx_target": args.ctx,
        "chunk_start": args.chunk,
        "chunk_floor": args.chunk_floor,
        "refine_percentile": args.refine,
        "decode_steps": args.decode_steps,
        "engine_build": _ENGINE,
        "is_fork": _IS_FORK,
        "int6": True,
        "attention_mode": "apa",
        "bulk_bits_assert": None,
        "reached_ctx": 0,
        "status": "INIT",
        "growth_trace": [],
        "chunk_size_used_at_target": None,
        "steady_smi_mb": None,
        "poller_peak_mb": None,   # filled by driver from poll log
        "ms_per_token": None,
        "engagement": {"APA_ENGAGED": None},
        "error": "",
    }

    if not _IS_FORK:
        rec["status"] = "ERR"
        rec["error"] = f"tc is not the fork int6 build: {_ENGINE}"
        print("RESULT " + json.dumps(rec), flush=True)
        Path(args.out).write_text(json.dumps(rec, indent=2))
        return 0

    engage.install()
    engage.reset()
    t_all = time.time()

    try:
        print(f"[p3y] loading INT6 fork model attention_mode=apa refine={args.refine}",
              flush=True)
        m, info = Qwen3_1p7b_TC.from_pretrained(attention_mode="apa", int6=True)
        m.set_attention_mode("apa", refine_percentile=args.refine)
        # bulk_bits assertion (from_pretrained sets 8 on every layer)
        bb = sorted({int(L.self_attn.bulk_bits) for L in m.layers})
        rec["bulk_bits_assert"] = bb
        assert bb == [8], f"bulk_bits not uniformly 8: {bb}"
        print(f"[p3y] loaded {info.get('loaded')} bulk_bits={bb} "
              f"engine={_ENGINE}", flush=True)
        m.extend_rope(args.ctx + 8)

        rng = np.random.default_rng(0)
        L = args.ctx
        # Build the KV to L tokens by chunked prefill. The final decode phase
        # queries pos=L over the full L-length KV (p3 decode-cell semantics).
        chunk = args.chunk
        caches = None
        pos = 0
        remaining = L
        next_smi_at = 2048
        stage_i = 0

        while remaining > 0:
            c = min(chunk, remaining)
            blk = rng.integers(0, 151936, size=(1, c), dtype=np.int64)
            try:
                t0 = time.time()
                logits, new_caches = m(blk, kv_caches=caches, position_offset=pos,
                                       last_token_only=True)
                _ = logits.numpy()  # force materialize / surface OOM here
                caches = new_caches
                pos += c
                remaining -= c
                dt = time.time() - t0
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                if is_oom(msg) and chunk > args.chunk_floor:
                    new_chunk = max(args.chunk_floor, chunk // 2)
                    rec["growth_trace"].append({
                        "stage": stage_i, "event": "OOM_HALVE",
                        "tokens_resident": pos, "chunk_from": chunk,
                        "chunk_to": new_chunk,
                        "smi_used_mb": smi_used_mb(),
                        "error": msg[:300],
                    })
                    print(f"[p3y] OOM at pos={pos} chunk={chunk} -> halve to "
                          f"{new_chunk}; retry from last-good", flush=True)
                    stage_i += 1
                    chunk = new_chunk
                    continue
                elif is_oom(msg):
                    # OOM at the floor chunk -> highest reachable is pos.
                    rec["status"] = "OOM_AT_FLOOR"
                    rec["reached_ctx"] = pos
                    rec["chunk_size_used_at_target"] = chunk
                    rec["error"] = msg
                    rec["growth_trace"].append({
                        "stage": stage_i, "event": "OOM_AT_FLOOR",
                        "tokens_resident": pos, "chunk": chunk,
                        "smi_used_mb": smi_used_mb(), "error": msg[:400],
                    })
                    print(f"[p3y] OOM at floor chunk={chunk} pos={pos}; "
                          f"BLOCKER: {msg[:200]}", flush=True)
                    rec["engagement"] = engage.summary()
                    print("RESULT " + json.dumps(rec), flush=True)
                    Path(args.out).write_text(json.dumps(rec, indent=2))
                    return 0
                else:
                    raise

            if pos >= next_smi_at or remaining == 0:
                smi = smi_used_mb()
                rec["growth_trace"].append({
                    "stage": stage_i, "event": "GROW",
                    "tokens_resident": pos, "chunk": chunk,
                    "last_chunk_ms": round(dt * 1000, 2),
                    "smi_used_mb": smi,
                })
                print(f"[p3y] grow pos={pos}/{L} chunk={chunk} "
                      f"smi={smi} last_chunk_ms={dt*1000:.1f}", flush=True)
                stage_i += 1
                next_smi_at = pos + 4096

        rec["reached_ctx"] = pos
        rec["chunk_size_used_at_target"] = chunk
        smi_built = smi_used_mb()
        rec["steady_smi_after_build_mb"] = smi_built
        print(f"[p3y] BUILD COMPLETE pos={pos} chunk={chunk} smi={smi_built}",
              flush=True)

        # ---- APA decode: N steps over the full-L KV ----
        engage.reset()  # engagement measured on the decode phase specifically
        cur = rng.integers(0, 151936, size=(1, 1), dtype=np.int64)
        dstep_ms = []
        for s in range(args.decode_steps):
            t0 = time.time()
            slog, caches = m(cur, kv_caches=caches, position_offset=pos,
                             last_token_only=True)
            nxt = int(slog.numpy().reshape(-1).argmax())
            dstep_ms.append((time.time() - t0) * 1000.0)
            pos += 1
            cur = np.array([[nxt % 151936]], dtype=np.int64)

        steady = smi_used_mb()
        rec["steady_smi_mb"] = steady
        # exclude the first (warm) step from ms/token
        timed = dstep_ms[1:] if len(dstep_ms) > 1 else dstep_ms
        rec["ms_per_token"] = round(float(np.mean(timed)), 3)
        rec["ms_per_token_median"] = round(float(np.median(timed)), 3)
        rec["decode_first_step_ms"] = round(dstep_ms[0], 3)
        rec["engagement"] = engage.summary()
        rec["status"] = "OK"
        rec["total_elapsed_s"] = round(time.time() - t_all, 2)
        print(f"[p3y] DECODE OK steady_smi={steady} "
              f"ms/token={rec['ms_per_token']} "
              f"APA_ENGAGED={rec['engagement']['APA_ENGAGED']} "
              f"apa_calls={rec['engagement']['apa_calls']}", flush=True)

    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        rec["status"] = "OOM" if is_oom(msg) else "ERR"
        rec["error"] = msg
        rec["engagement"] = engage.summary()
        rec["total_elapsed_s"] = round(time.time() - t_all, 2)
        print(f"[p3y] {rec['status']}: {msg[:300]}", flush=True)

    print("RESULT " + json.dumps(rec), flush=True)
    Path(args.out).write_text(json.dumps(rec, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
