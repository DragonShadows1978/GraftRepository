import json
import subprocess

import numpy as np

from scripts import grm_router_baseline as baseline


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

    subprocess.run([
        "python3",
        "scripts/grm_gqa_router_benchmark.py",
        "--smoke",
        "--out",
        str(out),
        "--markdown-out",
        str(md),
    ], cwd=baseline.ROOT, check=True)

    data = json.loads(out.read_text())
    assert data["schema"] == "grm-gqa-router-benchmark-v1"
    assert data["shape"]["query_heads"] == 4
    assert [row["nodes"] for row in data["results"]] == [32, 96]
    assert all(row["parity"] for row in data["results"])
    assert "GRM GQA Router Benchmark" in md.read_text()


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
