#!/usr/bin/env python3
"""Sweep native GRM GQA routing across real Qwen3.5 capture layers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import grm_gqa_router_benchmark as bench  # noqa: E402
from scripts import grm_router_baseline as baseline  # noqa: E402


def _default_node_counts(shards: int) -> list[int]:
    return [n for n in (128, 256, 512) if n <= shards] or [shards]


def _row_with_layer(row: dict, layer: int, capture_stats: dict) -> dict:
    out = dict(row)
    out["capture_layer"] = int(layer)
    out["capture_shards"] = int(capture_stats["shards"])
    out["capture_skipped_shape"] = int(capture_stats["skipped_shape"])
    return out


def write_markdown(result: dict, path: Path) -> None:
    lines = [
        "# GRM GQA Capture Layer Sweep",
        "",
        "Native GQA raw q.k route benchmark across real capture layers.",
        "",
        "## Shape",
        "",
        f"- preset: `{result['preset']}`",
        f"- capture_dir: `{result['capture_dir']}`",
        f"- capture_role: `{result['capture_role']}`",
        f"- query_tokens: {result['query_tokens']}",
        f"- topk: {result['topk']}",
        f"- openmp: {str(result['openmp']).lower()}",
        f"- native_only: {str(result['native_only']).lower()}",
        f"- runtime_flags: `{json.dumps(result.get('runtime_flags', {}), sort_keys=True)}`",
        "",
        "## Results",
        "",
        "| layer | nodes | shards | parity | native p50 ms | native p95 ms | "
        "parity reference | parity queries |",
        "| ---: | ---: | ---: | :---: | ---: | ---: | --- | ---: |",
    ]
    for row in result["results"]:
        parity = "n/a" if row["parity"] is None else str(row["parity"]).lower()
        lines.append(
            f"| {row['capture_layer']} | {row['nodes']} | "
            f"{row['capture_shards']} | {parity} | "
            f"{row['native_route_ms_p50']:.4f} | "
            f"{row['native_route_ms_p95']:.4f} | "
            f"{row['parity_reference'] or 'n/a'} | "
            f"{row['parity_queries']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=sorted(bench.PRESETS),
                   default="qwen35-2b-attn")
    p.add_argument("--capture-dir", type=Path, required=True)
    p.add_argument("--capture-role", default="source")
    p.add_argument("--layers", nargs="+", type=int,
                   default=[3, 7, 11, 15, 19, 23])
    p.add_argument("--node-counts", nargs="+", type=int)
    p.add_argument("--capture-limit", type=int, default=1200,
                   help="maximum capture shards to load per layer; 0 loads all")
    p.add_argument("--queries", type=int, default=6)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--query-tokens", type=int, default=4)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--route-layer", type=int, default=None)
    p.add_argument("--cxx", default="g++")
    p.add_argument("--openmp", action="store_true")
    p.set_defaults(native_only=True, parity_all_queries=True)
    p.add_argument("--native-only", action="store_true",
                   help="time native routing only; this is the default")
    p.add_argument("--include-python-timing", action="store_false",
                   dest="native_only",
                   help="include Python reference timing in the measured route loop")
    p.add_argument("--parity-all-queries", action="store_true",
                   help="check all generated queries; this is the default")
    p.add_argument("--parity-sample-only", action="store_false",
                   dest="parity_all_queries",
                   help="use --parity-sample-queries instead of all-query parity")
    p.add_argument("--parity-sample-queries", type=int, default=0)
    p.add_argument("--parity-reference", choices=("scalar", "batched"),
                   default="batched")
    p.add_argument("--use-lexical", action="store_true")
    p.add_argument("--out", type=Path,
                   default=Path("/tmp/grm_gqa_layer_sweep.json"))
    p.add_argument("--markdown-out", type=Path)
    p.add_argument("--progress", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.warmup >= args.queries:
        raise SystemExit("--warmup must be lower than --queries")
    preset = dict(bench.PRESETS[args.preset])
    parity_all_queries = (
        bool(args.parity_all_queries) or int(args.parity_sample_queries) <= 0
    )
    all_results: list[dict] = []
    actual_node_counts: set[int] = set()
    layer_stats: dict[str, dict] = {}

    with tempfile.TemporaryDirectory(prefix="grm_gqa_layer_sweep_") as td:
        lib = baseline.build_native_lib(Path(td), cxx=args.cxx, openmp=args.openmp)
        for layer in args.layers:
            routes, capture_stats = bench.load_capture_routes(
                args.capture_dir,
                role=args.capture_role,
                layer=layer,
                limit=args.capture_limit,
                token_limit=0,
                kv_heads=preset["kv_heads"],
                head_dim=preset["head_dim"],
            )
            layer_stats[str(layer)] = capture_stats
            queries = bench.make_capture_queries(
                routes,
                query_count=args.queries,
                query_heads=preset["query_heads"],
                query_tokens=args.query_tokens,
            )
            node_counts = args.node_counts or _default_node_counts(routes.shape[0])
            for node_count in node_counts:
                actual_node_counts.add(int(node_count))
                if node_count > routes.shape[0]:
                    raise SystemExit(
                        f"node count {node_count} exceeds layer {layer} "
                        f"captures {routes.shape[0]}")
                row = bench.benchmark_count(
                    routes=routes[:node_count],
                    parity_routes=None,
                    queries=queries,
                    lib_path=lib,
                    preset=preset,
                    route_layer=args.route_layer
                    if args.route_layer is not None else layer,
                    topk=args.topk,
                    warmup=args.warmup,
                    native_only=args.native_only,
                    use_lexical=args.use_lexical,
                    parity_all_queries=parity_all_queries,
                    parity_sample_queries=args.parity_sample_queries,
                    parity_reference_mode=args.parity_reference,
                )
                row = _row_with_layer(row, layer, capture_stats)
                all_results.append(row)
                if args.progress:
                    parity = "n/a" if row["parity"] is None else str(row["parity"]).lower()
                    print(
                        f"[grm_gqa_layer] layer={layer} nodes={node_count} "
                        f"native_p50_ms={row['native_route_ms_p50']:.4f} "
                        f"parity={parity}",
                        file=sys.stderr,
                        flush=True,
                    )

    out = {
        "schema": "grm-gqa-layer-sweep-v1",
        "repo": str(ROOT),
        "preset": args.preset,
        "capture_dir": str(args.capture_dir),
        "capture_role": args.capture_role,
        "capture_limit": int(args.capture_limit),
        "layers": [int(x) for x in args.layers],
        "node_counts": sorted(actual_node_counts),
        "query_tokens": int(args.query_tokens),
        "topk": int(args.topk),
        "queries": int(args.queries),
        "warmup": int(args.warmup),
        "openmp": bool(args.openmp),
        "native_only": bool(args.native_only),
        "parity_all_queries": bool(parity_all_queries),
        "parity_sample_queries": int(args.parity_sample_queries),
        "parity_reference": args.parity_reference,
        "lexical": bool(args.use_lexical),
        "runtime_flags": bench.gqa_runtime_flags(),
        "capture_stats": layer_stats,
        "results": all_results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(out, args.markdown_out)
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
