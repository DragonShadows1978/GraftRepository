import json
import subprocess

import numpy as np

from scripts import grm_router_baseline as baseline
from scripts import grm_gqa_router_benchmark as gqa_benchmark


def test_hashed_centroid_is_deterministic_and_normalized():
    a = baseline.hashed_centroid("Router facts route through attention state", 32)
    b = baseline.hashed_centroid("Router facts route through attention state", 32)

    assert a is not None
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(np.linalg.norm(a), 1.0, rtol=1e-6)


def test_python_route_scan_drops_non_finite_scores():
    routes = np.asarray([
        [np.nan, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ], dtype=np.float32)
    query = np.asarray([1.0, 0.0], dtype=np.float32)

    assert baseline.python_route_scan(routes, query, (), topk=2) == [1, 2]


def test_router_baseline_smoke_cli_writes_json_and_markdown(tmp_path):
    out = tmp_path / "baseline.json"
    md = tmp_path / "baseline.md"

    subprocess.run([
        "python3",
        "scripts/grm_router_baseline.py",
        "--smoke",
        "--out",
        str(out),
        "--markdown-out",
        str(md),
    ], cwd=baseline.ROOT, check=True)

    data = json.loads(out.read_text())
    assert data["schema"] == "grm-router-baseline-v1"
    assert data["harvest"]["centroid_bank_rows"] > 0
    assert [row["nodes"] for row in data["results"]] == [32, 96]
    assert all(row["parity"] for row in data["results"])
    assert "GRM Router Baseline" in md.read_text()


def test_router_baseline_smoke_cli_supports_int4_refine_all(tmp_path):
    out = tmp_path / "baseline_int4.json"

    subprocess.run([
        "python3",
        "scripts/grm_router_baseline.py",
        "--smoke",
        "--int4",
        "--refine-m",
        "128",
        "--out",
        str(out),
    ], cwd=baseline.ROOT, check=True)

    data = json.loads(out.read_text())
    assert [row["nodes"] for row in data["results"]] == [32, 96]
    assert all(row["parity"] for row in data["results"])


def test_router_baseline_smoke_cli_supports_native_fp32_parity(tmp_path):
    out = tmp_path / "baseline_native_fp32.json"

    subprocess.run([
        "python3",
        "scripts/grm_router_baseline.py",
        "--smoke",
        "--native-only",
        "--native-fp32-parity",
        "--int4",
        "--refine-m",
        "128",
        "--out",
        str(out),
    ], cwd=baseline.ROOT, check=True)

    data = json.loads(out.read_text())
    assert all(row["parity"] for row in data["results"])
    assert {row["parity_reference"] for row in data["results"]} == {
        "native_fp32"}


def test_router_baseline_smoke_cli_sweeps_refine_m(tmp_path):
    out = tmp_path / "baseline_sweep.json"

    subprocess.run([
        "python3",
        "scripts/grm_router_baseline.py",
        "--smoke",
        "--native-only",
        "--native-fp32-parity",
        "--int4",
        "--sweep-refine-m",
        "32",
        "128",
        "--out",
        str(out),
    ], cwd=baseline.ROOT, check=True)

    data = json.loads(out.read_text())
    assert [row["refine_m"] for row in data["results"]] == [32, 32, 128, 128]
    assert all(row["mode"] == "int4" for row in data["results"])


def test_gqa_router_benchmark_smoke_cli_writes_json_and_markdown(tmp_path):
    out = tmp_path / "gqa_benchmark.json"
    md = tmp_path / "gqa_benchmark.md"
    progress = tmp_path / "gqa_progress.jsonl"

    subprocess.run([
        "python3",
        "scripts/grm_gqa_router_benchmark.py",
        "--smoke",
        "--out",
        str(out),
        "--markdown-out",
        str(md),
        "--progress-out",
        str(progress),
    ], cwd=baseline.ROOT, check=True)

    data = json.loads(out.read_text())
    assert data["schema"] == "grm-gqa-router-benchmark-v1"
    assert data["shape"]["query_heads"] == 4
    assert "GRM_ROUTER_GQA_ROWBLOCK" in data["runtime_flags"]
    assert [row["nodes"] for row in data["results"]] == [32, 96]
    assert all(row["parity"] for row in data["results"])
    assert "GRM GQA Router Benchmark" in md.read_text()
    progress_rows = [
        json.loads(line) for line in progress.read_text().splitlines()]
    assert [row["result"]["nodes"] for row in progress_rows] == [32, 96]
    assert [row["completed"] for row in progress_rows] == [1, 2]
    assert all(row["schema"] == "grm-gqa-router-benchmark-progress-v1"
               for row in progress_rows)
    assert all(row["compact"]["enabled"] is False for row in progress_rows)
    assert all("GRM_ROUTER_GQA_ROWBLOCK" in row["runtime_flags"]
               for row in progress_rows)


def test_gqa_cuda_probe_help_cli_smoke():
    result = subprocess.run([
        "python3",
        "scripts/grm_gqa_cuda_probe.py",
        "--help",
    ], cwd=baseline.ROOT, check=True, capture_output=True, text=True)

    assert "CUDA/cuBLAS GQA raw q.k router" in result.stdout


def test_gqa_router_benchmark_smoke_cli_reads_capture_shards(tmp_path):
    captures = tmp_path / "captures"
    captures.mkdir()
    rng = np.random.default_rng(13579)
    for i in range(4):
        key = rng.normal(size=(1, 1, 4, 16)).astype(np.float16)
        np.savez_compressed(captures / f"source_doc{i:02d}_chunk000000.npz",
                            l3_k=key)

    out = tmp_path / "gqa_capture.json"
    subprocess.run([
        "python3",
        "scripts/grm_gqa_router_benchmark.py",
        "--smoke",
        "--capture-dir",
        str(captures),
        "--capture-layer",
        "3",
        "--capture-limit",
        "4",
        "--node-counts",
        "2",
        "4",
        "--queries",
        "3",
        "--warmup",
        "1",
        "--no-lexical",
        "--out",
        str(out),
    ], cwd=baseline.ROOT, check=True)

    data = json.loads(out.read_text())
    assert data["route_source"] == "capture"
    assert data["harvest"]["route"]["shards"] == 4
    assert [row["nodes"] for row in data["results"]] == [2, 4]
    assert all(row["parity"] for row in data["results"])
    assert all(not row["lexical"] for row in data["results"])


def test_gqa_layer_sweep_cli_records_runtime_flags(tmp_path):
    captures = tmp_path / "captures"
    captures.mkdir()
    rng = np.random.default_rng(86420)
    for i in range(3):
        l3 = rng.normal(size=(1, 2, 4, 256)).astype(np.float16)
        l7 = rng.normal(size=(1, 2, 4, 256)).astype(np.float16)
        np.savez_compressed(captures / f"source_doc{i:02d}_chunk000000.npz",
                            l3_k=l3, l7_k=l7)

    out = tmp_path / "gqa_layer_sweep.json"
    md = tmp_path / "gqa_layer_sweep.md"
    subprocess.run([
        "python3",
        "scripts/grm_gqa_layer_sweep.py",
        "--capture-dir",
        str(captures),
        "--layers",
        "3",
        "7",
        "--capture-limit",
        "3",
        "--node-counts",
        "2",
        "--queries",
        "3",
        "--warmup",
        "1",
        "--query-tokens",
        "2",
        "--topk",
        "2",
        "--out",
        str(out),
        "--markdown-out",
        str(md),
    ], cwd=baseline.ROOT, check=True)

    data = json.loads(out.read_text())
    assert data["schema"] == "grm-gqa-layer-sweep-v1"
    assert "GRM_ROUTER_GQA_ROWBLOCK" in data["runtime_flags"]
    assert [row["capture_layer"] for row in data["results"]] == [3, 7]
    assert all(row["nodes"] == 2 for row in data["results"])
    assert all(row["parity"] for row in data["results"])
    assert "runtime_flags" in md.read_text()


def test_gqa_batched_python_reference_matches_scalar_route_law():
    rng = np.random.default_rng(97531)
    routes = rng.normal(size=(24, 2, 2, 5, 16)).astype(np.float32)
    queries = rng.normal(size=(4, 4, 3, 16)).astype(np.float32)
    routes = gqa_benchmark._normalize_last_dim(routes)
    queries = gqa_benchmark._normalize_last_dim(queries)

    for i, query in enumerate(queries):
        lexical = baseline.lexical_keys_for_query(i, routes.shape[0])
        scalar = gqa_benchmark.python_gqa_route_scan(
            routes, query, lexical, topk=7)
        batched = gqa_benchmark.python_gqa_route_batched(
            routes, query, lexical, topk=7)
        assert batched == scalar


def test_gqa_batched_python_reference_preserves_capture_tie_order():
    rng = np.random.default_rng(112233)
    routes = rng.normal(size=(96, 1, 2, 32, 16)).astype(np.float16)
    routes = gqa_benchmark._normalize_last_dim(routes.astype(np.float32))
    queries = gqa_benchmark.make_capture_queries(
        routes,
        query_count=4,
        query_heads=4,
        query_tokens=4,
    )

    for i, query in enumerate(queries):
        scalar = gqa_benchmark.python_gqa_route_scan(
            routes, query, (), topk=12)
        batched = gqa_benchmark.python_gqa_route_batched(
            routes, query, (), topk=12)
        assert batched == scalar


def test_gqa_capture_compaction_selects_representative_tokens():
    routes = np.arange(2 * 1 * 1 * 6 * 3, dtype=np.float32).reshape(
        2, 1, 1, 6, 3)

    compact, stats = gqa_benchmark.compact_capture_routes(
        routes, token_count=3, mode="stride")

    assert compact.shape == (2, 1, 1, 3, 3)
    assert stats["enabled"] is True
    assert stats["indices"] == [0, 2, 5]
    np.testing.assert_array_equal(compact, routes[:, :, :, [0, 2, 5], :])


def test_gqa_router_benchmark_native_only_supports_sampled_parity(tmp_path):
    captures = tmp_path / "captures"
    captures.mkdir()
    rng = np.random.default_rng(24680)
    for i in range(6):
        key = rng.normal(size=(1, 1, 4, 16)).astype(np.float16)
        np.savez_compressed(captures / f"source_doc{i:02d}_chunk000000.npz",
                            l3_k=key)

    out = tmp_path / "gqa_capture_sampled_parity.json"
    subprocess.run([
        "python3",
        "scripts/grm_gqa_router_benchmark.py",
        "--smoke",
        "--capture-dir",
        str(captures),
        "--capture-layer",
        "3",
        "--capture-limit",
        "6",
        "--node-counts",
        "6",
        "--queries",
        "4",
        "--warmup",
        "1",
        "--native-only",
        "--parity-sample-queries",
        "2",
        "--no-lexical",
        "--out",
        str(out),
    ], cwd=baseline.ROOT, check=True)

    data = json.loads(out.read_text())
    row = data["results"][0]
    assert data["native_only"] is True
    assert row["parity"] is True
    assert row["parity_reference"] == "python_gqa_raw_sampled"
    assert row["parity_sampled"] is True
    assert row["parity_queries"] == 2
    assert row["python_route_ms_p50"] is None


def test_gqa_router_benchmark_compact_routes_can_check_full_parity(tmp_path):
    captures = tmp_path / "captures"
    captures.mkdir()
    rng = np.random.default_rng(31415)
    for i in range(6):
        base = rng.normal(size=(1, 1, 1, 16)).astype(np.float16)
        key = np.repeat(base, repeats=4, axis=2)
        np.savez_compressed(captures / f"source_doc{i:02d}_chunk000000.npz",
                            l3_k=key)

    out = tmp_path / "gqa_capture_compact_full_parity.json"
    subprocess.run([
        "python3",
        "scripts/grm_gqa_router_benchmark.py",
        "--smoke",
        "--capture-dir",
        str(captures),
        "--capture-layer",
        "3",
        "--capture-limit",
        "6",
        "--node-counts",
        "6",
        "--queries",
        "4",
        "--warmup",
        "1",
        "--native-only",
        "--parity-sample-queries",
        "2",
        "--parity-reference",
        "batched",
        "--compact-route-tokens",
        "1",
        "--compact-route-mode",
        "prefix",
        "--compact-parity-full",
        "--no-lexical",
        "--out",
        str(out),
    ], cwd=baseline.ROOT, check=True)

    data = json.loads(out.read_text())
    row = data["results"][0]
    assert data["compact"]["enabled"] is True
    assert data["compact"]["token_count"] == 1
    assert row["key_tokens"] == 1
    assert row["parity_key_tokens"] == 4
    assert row["parity"] is True
    assert row["parity_reference"] == "python_gqa_raw_batched_sampled"


def test_gqa_router_benchmark_native_only_supports_batched_sampled_parity(tmp_path):
    captures = tmp_path / "captures"
    captures.mkdir()
    rng = np.random.default_rng(86420)
    for i in range(6):
        key = rng.normal(size=(1, 1, 4, 16)).astype(np.float16)
        np.savez_compressed(captures / f"source_doc{i:02d}_chunk000000.npz",
                            l3_k=key)

    out = tmp_path / "gqa_capture_batched_sampled_parity.json"
    subprocess.run([
        "python3",
        "scripts/grm_gqa_router_benchmark.py",
        "--smoke",
        "--capture-dir",
        str(captures),
        "--capture-layer",
        "3",
        "--capture-limit",
        "6",
        "--node-counts",
        "6",
        "--queries",
        "4",
        "--warmup",
        "1",
        "--native-only",
        "--parity-sample-queries",
        "2",
        "--parity-reference",
        "batched",
        "--no-lexical",
        "--out",
        str(out),
    ], cwd=baseline.ROOT, check=True)

    data = json.loads(out.read_text())
    row = data["results"][0]
    assert row["parity"] is True
    assert row["parity_reference"] == "python_gqa_raw_batched_sampled"
    assert row["parity_sampled"] is True
    assert row["parity_queries"] == 2
    assert row["python_route_ms_p50"] is None


def test_gqa_router_benchmark_native_only_supports_exhaustive_parity(tmp_path):
    captures = tmp_path / "captures"
    captures.mkdir()
    rng = np.random.default_rng(97531)
    for i in range(6):
        key = rng.normal(size=(1, 1, 4, 16)).astype(np.float16)
        np.savez_compressed(captures / f"source_doc{i:02d}_chunk000000.npz",
                            l3_k=key)

    out = tmp_path / "gqa_capture_exhaustive_parity.json"
    subprocess.run([
        "python3",
        "scripts/grm_gqa_router_benchmark.py",
        "--smoke",
        "--capture-dir",
        str(captures),
        "--capture-layer",
        "3",
        "--capture-limit",
        "6",
        "--node-counts",
        "6",
        "--queries",
        "4",
        "--warmup",
        "1",
        "--native-only",
        "--parity-all-queries",
        "--parity-reference",
        "batched",
        "--no-lexical",
        "--out",
        str(out),
    ], cwd=baseline.ROOT, check=True)

    data = json.loads(out.read_text())
    row = data["results"][0]
    assert row["parity"] is True
    assert row["parity_reference"] == "python_gqa_raw_batched"
    assert row["parity_sampled"] is False
    assert row["parity_exhaustive"] is True
    assert row["parity_queries"] == 4
    assert row["python_route_ms_p50"] is None
