#!/usr/bin/env python3
"""NC17-P3 context-ceiling binary-search driver (FORK INT6 engine).

Runs INSIDE a single flock (the caller holds /tmp/forge-gpu.lock); launches ONE
p3_ladder_probe.py SUBPROCESS PER RUNG (fresh CUDA context each — Gemma law),
binary-searching the context length to the wall for a given (mode, kind). The
INT6 fork engine reaches the probes via the inherited PYTHONPATH (the driver is
launched through run_gpu_driver_int6.sh which exports it).

Each probe subprocess: setsid, `timeout` seconds, foreground blocking. status OK
-> raise floor; OOM/timeout -> lower ceiling; ERR (non-OOM) -> abort with the
verbatim error. Reports last-solid (highest OK) and first-OOM.

Args: --mode {standard,apa} --kind {prefill,decode} --lo L --hi H
      [--probe-timeout S] [--refine R] [--max-iters N]
Writes logs/nc17/p3_ceiling_<mode>_<kind>.json
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PROBE = REPO / "tests" / "nc17" / "p3_ladder_probe.py"


def run_probe(mode, kind, ctx, refine, timeout_s):
    cmd = ["setsid", "timeout", str(timeout_s), "python3", str(PROBE),
           "--mode", mode, "--kind", kind, "--ctx", str(ctx),
           "--refine", str(refine)]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout_s + 30)
        out = p.stdout
        rc = p.returncode
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") if isinstance(e.stdout, str) else ""
        rc = 124
    rec = None
    for line in out.splitlines():
        if line.startswith("RESULT "):
            try:
                rec = json.loads(line[len("RESULT "):])
            except Exception:
                pass
    if rec is None:
        rec = {"mode": mode, "kind": kind, "ctx": ctx,
               "status": "OOM" if rc in (124, 137, -9) else "ERR",
               "smi_used_mb_after": None, "elapsed_s": time.time() - t0,
               "engagement": {"APA_ENGAGED": None},
               "error": f"no RESULT line; rc={rc}; tail={out[-300:]!r}"}
    rec["_probe_rc"] = rc
    rec["_wall_s"] = time.time() - t0
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["standard", "apa"])
    ap.add_argument("--kind", required=True, choices=["prefill", "decode"])
    ap.add_argument("--lo", type=int, required=True)
    ap.add_argument("--hi", type=int, required=True)
    ap.add_argument("--probe-timeout", type=int, default=540)
    ap.add_argument("--refine", type=float, default=0.15)
    ap.add_argument("--max-iters", type=int, default=8)
    args = ap.parse_args()

    trace = []
    lo_rec = run_probe(args.mode, args.kind, args.lo, args.refine, args.probe_timeout)
    trace.append(lo_rec)
    print(f"[ladder] {args.mode}/{args.kind} lo={args.lo} -> {lo_rec['status']} "
          f"(smi={lo_rec.get('smi_used_mb_after')} eng={lo_rec.get('engine_build')})", flush=True)
    if lo_rec["status"] == "ERR":
        print(f"[ladder] ABORT: lo probe ERR: {lo_rec.get('error')}", flush=True)
        _write(args, trace, None, None); return 0
    if lo_rec["status"] != "OK":
        print("[ladder] lo already OOM — ceiling below lo", flush=True)
        _write(args, trace, None, args.lo); return 0

    last_solid = args.lo
    first_oom = None
    hi = args.hi

    hi_rec = run_probe(args.mode, args.kind, hi, args.refine, args.probe_timeout)
    trace.append(hi_rec)
    print(f"[ladder] {args.mode}/{args.kind} hi={hi} -> {hi_rec['status']} "
          f"(smi={hi_rec.get('smi_used_mb_after')})", flush=True)
    if hi_rec["status"] == "ERR":
        print(f"[ladder] ABORT: hi probe ERR: {hi_rec.get('error')}", flush=True)
        _write(args, trace, last_solid, None); return 0
    if hi_rec["status"] == "OK":
        last_solid = hi
        print(f"[ladder] hi solid; ceiling >= {hi}", flush=True)
        _write(args, trace, last_solid, None); return 0
    first_oom = hi

    it = 0
    while it < args.max_iters and (first_oom - last_solid) > max(256, last_solid // 20):
        mid = (last_solid + first_oom) // 2
        rec = run_probe(args.mode, args.kind, mid, args.refine, args.probe_timeout)
        trace.append(rec)
        print(f"[ladder] {args.mode}/{args.kind} mid={mid} -> {rec['status']} "
              f"(smi={rec.get('smi_used_mb_after')})", flush=True)
        if rec["status"] == "ERR":
            print(f"[ladder] ABORT: mid ERR: {rec.get('error')}", flush=True)
            break
        if rec["status"] == "OK":
            last_solid = mid
        else:
            first_oom = mid
        it += 1

    print(f"[ladder] {args.mode}/{args.kind} DONE last_solid={last_solid} "
          f"first_oom={first_oom}", flush=True)
    _write(args, trace, last_solid, first_oom)
    return 0


def _write(args, trace, last_solid, first_oom):
    OUT = REPO / "logs" / "nc17" / f"p3_ceiling_{args.mode}_{args.kind}.json"
    OUT.write_text(json.dumps({
        "mode": args.mode, "kind": args.kind,
        "engine_build": (trace[0].get("engine_build") if trace else None),
        "search_lo": args.lo, "search_hi": args.hi,
        "last_solid_ctx": last_solid, "first_oom_ctx": first_oom,
        "probe_timeout_s": args.probe_timeout, "refine_percentile": args.refine,
        "trace": trace,
    }, indent=2))
    print(f"[ladder] wrote {OUT}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
