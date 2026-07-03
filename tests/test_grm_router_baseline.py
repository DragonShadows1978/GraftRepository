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
