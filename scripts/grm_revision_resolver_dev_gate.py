#!/usr/bin/env python3
"""Deterministic supersession-under-competition development gate.

The corpus mirrors the composed-session failure shape without model work:
an inactive old fact, two higher-semantic derived/update turns carrying the
same label, and one authoritative active fact. Baseline semantic+query-lex
routing places the fact at rank 3 and outside a two-seat first mount.

After product implementation, ``--resolver`` installs the repository's
revision-aware callback and requires the active fact at rank 1 for all probes.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.graft_arena import ArenaCache  # noqa: E402
from core.graft_repository import GraftRepository  # noqa: E402


FAMILIES = (
    "orion pin", "lyra dock", "mira seal", "terra port",
    "nova key", "ember code", "atlas tone", "polaris mark",
    "cypher bridge", "lumen gate", "vega latch", "solace token",
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolver", action="store_true")
    parser.add_argument(
        "--out", type=Path,
        default=Path("artifacts/grm_revision_resolver/dev_gate.json"))
    return parser.parse_args(argv)


def _make_arena() -> tuple[ArenaCache, list[dict]]:
    arena = ArenaCache.__new__(ArenaCache)
    arena.native_store = None
    arena.last_route_backend = "python"
    arena.grafts = []
    cases = []

    for family_index, family in enumerate(FAMILIES):
        old_value = f"Old-{family_index:02d}-Alpha"
        new_value = f"New-{family_index:02d}-Tango"
        wrong_value = f"Wrong-{family_index:02d}-Sierra"
        old_id = len(arena.grafts)
        arena.grafts.append({
            "cent": np.asarray([0.99, 0.01], np.float32),
            "_baseline_score": 0.99,
            "rare": set(),
            "text": f"The current {family} value is {old_value}.",
            "kind": "fact",
            "retired": True,
            "metadata": {
                "kind": "fact", "active": False,
                "subject": family, "predicate": "value",
                "value": old_value, "scope": "project",
                "superseded_by": [old_id + 3],
            },
        })
        derived_id = len(arena.grafts)
        arena.grafts.append({
            "cent": np.asarray([1.0, 0.0], np.float32),
            "_baseline_score": 1.0,
            "rare": set(),
            "text": (
                f"Recall probe for {family}; prior answer was {wrong_value}."),
            "kind": "turn", "retired": False,
            "metadata": {"kind": "turn", "active": True},
        })
        update_id = len(arena.grafts)
        arena.grafts.append({
            "cent": np.asarray([0.95, 0.05], np.float32),
            "_baseline_score": 0.95,
            "rare": set(),
            "text": (
                f"Authoritative update: current {family} value is {new_value}."),
            "kind": "turn", "retired": False,
            "metadata": {"kind": "turn", "active": True},
        })
        current_id = len(arena.grafts)
        arena.grafts.append({
            "cent": np.asarray([0.86, 0.14], np.float32),
            "_baseline_score": 0.86,
            "rare": set(),
            "text": f"The current {family} value is {new_value}.",
            "kind": "fact", "retired": False,
            "metadata": {
                "kind": "fact", "active": True,
                "subject": family, "predicate": "value",
                "value": new_value, "scope": "project",
                "supersedes": [old_id],
            },
        })
        cases.append({
            "family": family,
            "old_id": old_id,
            "derived_id": derived_id,
            "update_id": update_id,
            "current_id": current_id,
            "query": (
                f"Recall probe. What is the current {family} value?"),
        })

    arena._probe_key = lambda _text: np.asarray([1.0, 0.0], np.float32)
    arena._vector_route_scores = lambda _p, cand: {
        idx: float(arena.grafts[idx]["_baseline_score"]) for idx in cand}
    arena._normalize_scores = lambda base: base
    return arena, cases


def run(resolver: bool) -> dict:
    arena, cases = _make_arena()
    repo = GraftRepository.__new__(GraftRepository)
    repo.arena = arena
    repo.native_store = None
    repo._native_node_ids = {}
    repo.revision_aware_resolver = bool(resolver)
    if resolver:
        callback = getattr(repo, "_resolve_route_family", None)
        if not callable(callback):
            raise RuntimeError("revision-aware resolver is not implemented")
        arena.route_resolver = callback

    rows = []
    rank1 = 0
    top3 = 0
    first_mount = 0
    stale_ranked = 0
    for case in cases:
        ranking = arena.route(case["query"], exclude=set(), limit=5)
        current_rank = ranking.index(case["current_id"]) + 1
        resolution = getattr(arena, "last_route_resolution", None)
        pinned = list((resolution or {}).get("pinned_node_ids", ()))
        mount_plan = pinned if (resolution or {}).get("state") == "exact" \
            else ranking[:2]
        rank1 += int(current_rank == 1)
        top3 += int(current_rank <= 3)
        first_mount += int(mount_plan == [case["current_id"]])
        stale_ranked += int(case["old_id"] in ranking)
        rows.append({
            **case,
            "ranking": ranking,
            "current_rank": current_rank,
            "first_mount": mount_plan,
            "resolution": resolution,
        })

    expected_rank1 = len(cases) if resolver else 0
    expected_first = len(cases) if resolver else 0
    passed = (
        rank1 == expected_rank1
        and first_mount == expected_first
        and top3 == len(cases)
        and stale_ranked == 0
    )
    return {
        "schema": "grm_revision_resolver_dev_gate_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "resolver" if resolver else "baseline",
        "families": len(cases),
        "current_rank1": rank1,
        "current_top3": top3,
        "current_first_mount": first_mount,
        "stale_ranked": stale_ranked,
        "passed_expected_mode": passed,
        "rows": rows,
    }


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    receipt = run(bool(args.resolver))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps({
        key: receipt[key] for key in (
            "mode", "families", "current_rank1", "current_top3",
            "current_first_mount", "stale_ranked", "passed_expected_mode")
    }, indent=2, sort_keys=True))
    return 0 if receipt["passed_expected_mode"] else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
