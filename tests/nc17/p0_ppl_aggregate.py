#!/usr/bin/env python3
"""Aggregate sliced ppl RESULT json records into one window ppl.

Usage: p0_ppl_aggregate.py <window> <slice_json_1> [<slice_json_2> ...]
Each slice json is a RESULT dict (sliced=true) with nll_sum + n_scored_tokens.
Sums them, computes ppl = exp(sum_nll / sum_scored), writes logs/nc17/ppl_<W>.json.
Verifies the slices tile the window grid with no gap/overlap.
"""
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LOGDIR = REPO_ROOT / "logs" / "nc17"


def main():
    W = int(sys.argv[1])
    slices = [json.loads(Path(p).read_text()) for p in sys.argv[2:]]
    slices.sort(key=lambda s: s["start_win"])

    # tiling check
    total = slices[0]["total_windows"]
    expect = 0
    for s in slices:
        assert s["window"] == W, f"window mismatch {s['window']} != {W}"
        assert s["start_win"] == expect, f"gap/overlap: expected start {expect}, got {s['start_win']}"
        expect = s["end_win"]
    assert expect == total, f"slices cover {expect} of {total} windows"

    nll = sum(s["nll_sum"] for s in slices)
    scored = sum(s["n_scored_tokens"] for s in slices)
    wall = sum(s["wall_s"] for s in slices)
    peak_alloc = max(s["max_mem_alloc_MiB"] for s in slices)
    ppl = math.exp(nll / scored)

    rec = {
        "window": W, "stride": slices[0]["stride"], "ppl": round(ppl, 6),
        "n_windows": sum(s["n_windows"] for s in slices),
        "n_scored_tokens": scored, "n_corpus_tokens": slices[0]["n_corpus_tokens"],
        "wall_s": round(wall, 2), "max_mem_alloc_MiB": peak_alloc,
        "sliced_into": len(slices),
    }
    (LOGDIR / f"ppl_{W}.json").write_text(json.dumps(rec))
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
