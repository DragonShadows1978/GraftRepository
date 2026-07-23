#!/usr/bin/env python3
"""Native 512-node revision-aware resolver parity/latency/restart gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
import tempfile
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.grm_native import NativeGraftStore  # noqa: E402


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-lib", required=True)
    parser.add_argument("--nodes", type=int, default=512)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument(
        "--out", type=Path,
        default=Path("artifacts/grm_revision_resolver/native_gate.json"))
    return parser.parse_args(argv)


def open_store(lib):
    return NativeGraftStore(
        lib, model_type="RevisionResolverGate", num_layers=2,
        hidden_dim=8, vals_per_tok_layer=4, route_layer=0,
        payload_kind="mla", latent_rank=4, rope_dim=0)


def percentile(values, q):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    at = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * q)))
    return ordered[at]


def vmrss_kib():
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except OSError:
        pass
    return None


def run(args):
    if args.nodes < 1 or args.queries < 1 or args.iterations < 1:
        raise ValueError("nodes, queries, and iterations must be positive")
    node_ids = []
    subjects = []
    with open_store(args.native_lib) as store:
        for i in range(args.nodes):
            subject = f"family{i:04d} slot"
            node_id = store.add_node(f"fact {i}", b"", ntok=1)
            store.set_metadata(node_id, {
                "kind": "fact", "active": True, "subject": subject,
                "predicate": "value", "value": f"V{i:04d}",
                "scope": "project",
            })
            node_ids.append(node_id)
            subjects.append(subject)

        query_count = min(args.queries, args.nodes)
        query_indices = [
            (i * args.nodes) // query_count for i in range(query_count)]
        parity = []
        for idx in query_indices:
            result = store.resolve_fact_query(
                [f"family{idx:04d}", "slot"], node_ids)
            parity.append(result.state == "exact" and
                          result.candidate_node_ids == (node_ids[idx],))

        for i in range(min(200, args.iterations)):
            idx = query_indices[i % query_count]
            store.resolve_fact_query([f"family{idx:04d}", "slot"], node_ids)
        rss_before = vmrss_kib()
        samples_ms = []
        for i in range(args.iterations):
            idx = query_indices[i % query_count]
            started = time.perf_counter_ns()
            result = store.resolve_fact_query(
                [f"family{idx:04d}", "slot"], node_ids)
            samples_ms.append((time.perf_counter_ns() - started) / 1e6)
            if result.state != "exact" or result.candidate_node_ids != (
                    node_ids[idx],):
                parity.append(False)
        rss_after = vmrss_kib()

        concurrency_errors = []
        stop = threading.Event()

        def mutate():
            try:
                for i in range(1000):
                    store.set_fact_identity(
                        node_ids[0],
                        subject=("family0000 slot" if i % 2 == 0
                                 else "alternate0000 slot"),
                        predicate="value", value="V0000", scope="project")
            except Exception as exc:  # pragma: no cover - receipt path
                concurrency_errors.append(f"writer: {exc}")
            finally:
                stop.set()

        writer = threading.Thread(target=mutate, daemon=True)
        writer.start()
        while not stop.is_set():
            try:
                result = store.resolve_fact_query(
                    ["family0000", "slot"], node_ids)
                if result.state not in ("exact", "no_match"):
                    concurrency_errors.append(
                        f"unexpected concurrent state {result.state}")
                    break
                if (result.state == "exact" and
                        result.candidate_node_ids != (node_ids[0],)):
                    concurrency_errors.append(
                        f"unexpected concurrent ids {result.candidate_node_ids}")
                    break
            except Exception as exc:  # pragma: no cover - receipt path
                concurrency_errors.append(f"reader: {exc}")
                break
        writer.join()
        store.set_fact_identity(
            node_ids[0], subject=subjects[0], predicate="value",
            value="V0000", scope="project")

        with tempfile.TemporaryDirectory(prefix="grm-resolver-") as td:
            checkpoint = Path(td)
            store.save_checkpoint(checkpoint)
            checkpoint_bytes = (checkpoint / "grm_store.bin").read_bytes()
            checkpoint_sha = hashlib.sha256(checkpoint_bytes).hexdigest()
            with open_store(args.native_lib) as restored:
                restored.load_checkpoint(checkpoint)
                restart = []
                for idx in query_indices:
                    result = restored.resolve_fact_query(
                        [f"family{idx:04d}", "slot"], node_ids)
                    restart.append(
                        result.state == "exact" and
                        result.candidate_node_ids == (node_ids[idx],))

    p50 = statistics.median(samples_ms)
    p95 = percentile(samples_ms, 0.95)
    passed = (all(parity) and all(restart) and not concurrency_errors
              and p95 <= 0.25)
    return {
        "schema": "grm_revision_resolver_native_gate_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "nodes": args.nodes,
        "queries": len(query_indices),
        "iterations": args.iterations,
        "parity_exact": sum(bool(v) for v in parity),
        "parity_total": len(parity),
        "restart_exact": sum(bool(v) for v in restart),
        "restart_total": len(restart),
        "checkpoint_sha256": checkpoint_sha,
        "latency_ms": {
            "p50": p50,
            "p95": p95,
            "max": max(samples_ms),
            "rail_p95": 0.25,
        },
        "rss_kib": {
            "after_warmup": rss_before,
            "after_measurement": rss_after,
            "delta": (None if rss_before is None or rss_after is None
                      else rss_after - rss_before),
        },
        "concurrency_errors": concurrency_errors,
        "passed": passed,
    }


def main(argv):
    args = parse_args(argv)
    receipt = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
