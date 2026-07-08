# NOTE (2026-07-08): companion to the PARKED kernel-opt Phase 2 CUDA-graph
# decode work (Project-Tensor branch kernel-opt-phase2-parked, b7a1c6d).
# Inert unless that engine build is installed and TC_DECODE_GRAPH=1; the
# graph path was closed as a negative result at the registered gate
# (Project-Tensor docs/KERNEL_OPT_IMPLEMENTATION_LEDGER.md 2026-07-07).
"""Graph-mode decode parity receipt (kernel-opt Phase 2, C1).

Runs the fixed default prompt ("Why is the sky blue?") for 32-token greedy
generation TWICE, as two separate subprocess invocations of
scripts/qwen35_generate.py: once with TC_DECODE_GRAPH unset, once with
TC_DECODE_GRAPH=1. Parses the raw token-id sequence out of each run's
"IDS:..." line (added via --print-ids) and asserts they are IDENTICAL, not
just close. This is the only way to prove the env-var gating actually works
end-to-end from the real CLI entry point, rather than calling internal
functions directly.

Usage: python3 scripts/qwen35_graph_parity.py
Exit code 0 on PASS, 1 on FAIL/error.
"""
import hashlib
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(HERE, "qwen35_generate.py")
PROMPT = "Why is the sky blue?"
MAX_NEW = "32"


def run_once(env_extra):
    env = dict(os.environ)
    env.update(env_extra)
    proc = subprocess.run(
        [sys.executable, GEN, PROMPT, MAX_NEW, "--print-ids"],
        env=env, capture_output=True, text=True, timeout=1800)
    return proc


def parse_ids(stdout):
    for line in stdout.splitlines():
        if line.startswith("IDS:"):
            payload = line[len("IDS:"):].strip()
            if not payload:
                return []
            return [int(x) for x in payload.split(",")]
    return None


def main():
    print("=== run 1: TC_DECODE_GRAPH unset (eager) ===", flush=True)
    r1 = run_once({"TC_DECODE_GRAPH": ""})
    print(r1.stdout)
    if r1.returncode != 0:
        print("--- stderr (eager run) ---")
        print(r1.stderr)
        print("FAIL: eager run exited nonzero")
        return 1
    ids1 = parse_ids(r1.stdout)
    if ids1 is None:
        print("FAIL: eager run produced no IDS: line")
        return 1

    print("=== run 2: TC_DECODE_GRAPH=1 (graph-mode) ===", flush=True)
    r2 = run_once({"TC_DECODE_GRAPH": "1"})
    print(r2.stdout)
    if r2.returncode != 0:
        print("--- stderr (graph run) ---")
        print(r2.stderr)
        print("FAIL: graph run exited nonzero")
        return 1
    ids2 = parse_ids(r2.stdout)
    if ids2 is None:
        print("FAIL: graph run produced no IDS: line")
        return 1

    h1 = hashlib.sha256(",".join(map(str, ids1)).encode()).hexdigest()
    h2 = hashlib.sha256(",".join(map(str, ids2)).encode()).hexdigest()

    print("=== result ===")
    print(f"eager ids  ({len(ids1)}): {ids1}")
    print(f"graph ids  ({len(ids2)}): {ids2}")
    print(f"eager sha256: {h1}")
    print(f"graph sha256: {h2}")

    if ids1 == ids2:
        print("PASS: token id sequences are IDENTICAL")
        return 0
    else:
        print("FAIL: token id sequences DIFFER")
        return 1


if __name__ == "__main__":
    sys.exit(main())
