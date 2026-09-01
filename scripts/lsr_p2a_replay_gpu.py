#!/usr/bin/env python3
"""GRM-LSR-P2A G2/G3 — lived-probe replay with the fit-honesty fix ON.

Program: LSR (Lived-Serving Reliability), Phase 2 item 1+2.
Order:   orders/GRM_LSR_P2A_FIT_HONESTY_SHUTTLE.md

WHAT THIS IS.  A fork-from-snapshot replay of the supersession leg of the
DET1.11 lived-serving census
(``artifacts/grm_det1/run_20260831T160525Z_2/det1_11/census/``), re-served
with LSR-P2A Ruling 1 (plan-priority fit + SHUTTLE) and Ruling 2
(not-in-memory abstention) in force.

WHAT IT REUSES, NEVER REIMPLEMENTS.
  * session construction, dialect, flags   -> the lsr_p0_gpu runtime frame
  * fixture installation                   -> grm_det1_gpu._install_fixture_nodes
  * the serving path itself                -> grm_e2e_session._probe_ladder_chat
  * value comparison (G4)                  -> grm_det1_common.contains_value
    plus the DET1.5 negative guards (stale_values / wrong_fact_values), i.e.
    the unified semantic comparator with negative guards.  NEVER glyph
    equality.

GPU DISCIPLINE (house rules, enforced here).
  * self-lease on /tmp/forge-gpu.lock via scripts.grm_cmc1_gpu_arms.gpu_lease;
    the operator has absolute right of way;
  * ONE fixture per process, one lease per fixture, so the battery is several
    short leases rather than one long one;
  * <= 580 s per lease (MAX_LEASE_SECONDS 590 is the hard cap);
  * the caller inserts an inter-process gap.

SCOPE, STATED HONESTLY.  This module replays the NINE supersession-family
probes of the census (4 ADMISSION-PRUNE reserves, 2 NOT-YET-DEPOSITED, the
praxis control, and the two remaining fixture controls).  The ten
``e2e_t*`` probes of the census are turns inside one certified 34-turn
session, not standalone fixtures; replaying them means re-running that whole
session, which is a separate lease budget and a separate driver
(``scripts/grm_e2e_session.py`` in full-session mode).  It is deliberately
NOT faked here.  The per-probe table reports exactly the probes this module
served.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.grm_cmc1_mechanism import canonical_json_bytes, sha256_bytes  # noqa: E402
from scripts.grm_det1_common import contains_value, file_record  # noqa: E402

ARTIFACT_DIR = ROOT / "artifacts" / "lsr_p2a"
SCHEMA_PREFIX = "grm.lsr_p2a"
RUNTIME_FRAME = (
    ROOT / "artifacts" / "grm_det1" / "run_20260831T160525Z_2"
    / "runtime_frame_28b3196f8fb04a41.json"
)
CENSUS = (
    ROOT / "artifacts" / "grm_det1" / "run_20260831T160525Z_2" / "det1_11"
    / "census" / "lived_serving_census_98ef71e88dec17a1.json"
)
SUP_FIXTURES = ROOT / "tests" / "fixtures" / "supersession_battery"
LEASE_SECONDS = 580
LOCK_WAIT_SECONDS = 7200

# G3 held-out abstention probe: an identifier deposited in NO fixture, asked
# against the meridian (multi_hop_a_b_c) fixture.  Registered before the run;
# the bound control on the same fixture is the census's own reserve question.
G3_UNBOUND_QUESTION = "What is the current Zephyr manifest value?"
G3_UNBOUND_FIXTURE = "multi_hop_a_b_c"
G3_BOUND_QUESTION = "What is the current Meridian docket value?"


class P2AError(RuntimeError):
    """The replay could not be performed as registered."""


def _read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def census_rows() -> dict[str, dict[str, Any]]:
    """The DET1.11 census rows, keyed by probe id (the served-before column)."""
    payload = _read(CENSUS)
    return {str(row["probe_id"]): dict(row) for row in payload["rows"]}


def sup_probe_plan() -> list[dict[str, Any]]:
    """Every supersession-family probe of the census, with its fixture.

    Fixture probes come from the fixture files themselves; the four DET1.9
    cross-session reserves come from the frozen SUP_RESERVE_PROBES table.
    Both are read, never re-invented.
    """
    from scripts.grm_det1_5_workers import SUP_RESERVE_PROBES

    rows = census_rows()
    plan: list[dict[str, Any]] = []
    for path in sorted(SUP_FIXTURES.glob("*.json")):
        session_id = path.stem
        fixture = _read(path)
        for probe in fixture["probes"]:
            probe_id = f"sup_{probe['probe_id']}"
            if probe_id not in rows:
                continue
            plan.append({
                "probe_id": probe_id,
                "session_id": session_id,
                "question": str(probe["question"]),
                "expected_values": [
                    str(v) for v in probe.get("expected_values", ())],
                "rejected_values": [
                    *[str(v) for v in probe.get("stale_values", ())],
                    *[str(v) for v in probe.get("wrong_fact_values", ())],
                ],
                "role": str(rows[probe_id].get("role", "")),
            })
        reserve = SUP_RESERVE_PROBES.get(session_id)
        if reserve is not None:
            probe_id = f"sup_{reserve['probe_id']}"
            if probe_id in rows:
                plan.append({
                    "probe_id": probe_id,
                    "session_id": session_id,
                    "question": str(reserve["question"]),
                    "expected_values": [
                        str(v) for v in reserve.get("expected_values", ())],
                    "rejected_values": [
                        *[str(v) for v in reserve.get("stale_values", ())],
                        *[str(v) for v in reserve.get("wrong_fact_values", ())],
                    ],
                    "role": str(rows[probe_id].get("role", "")),
                })
    plan.sort(key=lambda row: row["probe_id"])
    return plan


def answer_verdict(
    answer: str,
    *,
    expected_values: Sequence[str],
    rejected_values: Sequence[str],
) -> dict[str, Any]:
    """G4: the DET1 unified semantic comparator with its negative guards.

    ``contains_value`` normalizes Markdown emphasis and U+2010/U+2011 hyphens
    (DET1.4 / DET1.10) and matches whole words; the rejected list is the
    negative guard, so an answer naming BOTH the expected value and a stale
    one does not count as correct.
    """
    hit = [v for v in expected_values if contains_value(answer, v)]
    bad = [v for v in rejected_values if contains_value(answer, v)]
    return {
        "expected_hits": hit,
        "rejected_hits": bad,
        "correct": bool(hit and not bad),
    }


def _load_model(model_dir: str):
    """Load the 20B once per lease.

    Two GPT-OSS-20B residencies do not fit on the 12 GB card (measured:
    cudaMalloc OOM on the second load), so the model is loaded ONCE and every
    per-probe repository is built over it.
    """
    from core.gpt_oss20b_tc import GptOss20B_TC
    from transformers import AutoTokenizer

    model, _info = GptOss20B_TC.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    return model, tokenizer


def _build_session(flags: Mapping[str, Any], model, tokenizer,
                   *, repo_dir: str | None = None):
    """Session construction, identical to lsr_p0_gpu._serve_and_witness."""
    from core.gpt_oss20b_tc import gpt_oss_grm_dialect_kwargs
    from core.graft_repository import GraftRepository
    from core.grm_admission import adm_decisive_enabled
    from core.grm_supersession import sup_resolve_enabled
    from scripts import grm_e2e_session as e2e

    dialect = gpt_oss_grm_dialect_kwargs(model.config)
    repo = GraftRepository(
        model,
        lambda text: tokenizer.encode(text, add_special_tokens=False),
        lambda ids: tokenizer.decode(ids, clean_up_tokenization_spaces=False),
        repo_dir or os.environ.get("LSR_P2A_REPO_DIR", "/tmp/lsr_p2a_repo"),
        autosave=False,
        arena_cls=e2e.GptOssGQAArenaCache,
        route_layer=int(dialect["route_layer"]),
        sink_text=e2e.HARMONY_SINK,
        prompt_template=e2e.harmony_turn,
        stop_sequences=e2e.HARMONY_STOPS,
        storage_bits=int(flags["graft_storage_bits"]),
        arena_width=int(flags["arena_width"]),
        topk=int(flags["topk"]),
        live_turns=int(flags["live_turns"]),
        max_live=int(flags["max_live"]),
        revision_resolution=sup_resolve_enabled(bool(flags["sup_resolve"])),
        decisive_admission=adm_decisive_enabled(bool(flags["adm_decisive"])),
    )
    return repo


_INFO_KEYS = (
    "fit_planned", "fit_seated", "fit_dropped_planned",
    "fit_dropped_filler", "fit_unseatable", "fit_shuttle",
    "fit_shuttle_trips", "served_without_plan_head",
    "abstained", "abstain_reason", "abstain_identifier_tokens",
    "admission_policy_branch", "admission_rank_plan",
    "mount_plan", "mount_fitted", "ranking_ids", "trip",
)


def _fit_fields(info: Mapping[str, Any]) -> dict[str, Any]:
    return {key: info[key] for key in _INFO_KEYS if key in info}


def serve_fixture(
    session_id: str,
    *,
    extra_probes: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Install one fixture and serve every census probe bound to it."""
    from scripts import grm_e2e_session as e2e
    from scripts.grm_det1_gpu import _install_fixture_nodes

    frame = _read(RUNTIME_FRAME)
    flags = frame["resolved_flags"]
    probes = [
        row for row in sup_probe_plan() if row["session_id"] == session_id]
    probes = [*probes, *[dict(row) for row in extra_probes]]
    if not probes:
        raise P2AError(f"no census probes bound to fixture {session_id}")

    fixture = _read(SUP_FIXTURES / f"{session_id}.json")
    served: list[dict[str, Any]] = []
    # FORK-FROM-SNAPSHOT, at the repository layer: a FRESH GraftRepository per
    # probe over ONE resident model. Clearing arena.grafts in place does NOT
    # reset the repository's native store / WAL / index, so a reused repo
    # installs the fixture on top of stale native rows and the arena the probe
    # sees is not the arena the lived run served (measured: every probe
    # refused, mounting two nodes where the lived receipts mount one).
    model, tokenizer = _load_model(frame["model"]["path"])
    for probe in probes:
        # A distinct on-disk repository per probe: reusing one directory
        # rehydrates the previous probe's WAL and defeats the fork.
        repo = _build_session(
            flags, model, tokenizer,
            repo_dir=str(Path(tempfile.mkdtemp(prefix="lsr_p2a_"))),
        )
        try:
            arena = repo.arena
            _install_fixture_nodes(repo, fixture)
            started = time.time_ns()
            answer, info = e2e._probe_ladder_chat(
                repo,
                str(probe["question"]),
                topk=int(flags["topk"]),
                ngen=int(flags["ngen"]),
                max_trips=int(flags["max_trips"]),
            )
            elapsed_ns = int(time.time_ns() - started)
            verdict = answer_verdict(
                str(answer),
                expected_values=probe.get("expected_values", ()),
                rejected_values=probe.get("rejected_values", ()),
            )
            served.append({
                "probe_id": str(probe["probe_id"]),
                "session_id": session_id,
                "question": str(probe["question"]),
                "expected_values": list(probe.get("expected_values", ())),
                "rejected_values": list(probe.get("rejected_values", ())),
                "role": str(probe.get("role", "")),
                "served_answer": str(answer),
                "verdict": verdict,
                "info": _fit_fields(info),
                "mounted_ids": [int(v) for v in arena.cur_mounts],
                "arena_cur_mount_n": int(arena.cur_mount_n),
                "elapsed_ns": elapsed_ns,
            })
        finally:
            try:
                repo.close()
            except Exception:  # noqa: BLE001 - teardown must not mask a result
                pass
    return {
        "schema": f"{SCHEMA_PREFIX}.fixture_replay.v1",
        "program": "LSR",
        "phase": "2A",
        "session_id": session_id,
        "arena_width": int(flags["arena_width"]),
        "probes": served,
        "declares_in_det_envelope": False,
        "runtime_frame": file_record(RUNTIME_FRAME),
        "census": file_record(CENSUS),
        "sources": {
            "lsr_p2a_replay_gpu": file_record(Path(__file__).resolve()),
            "grm_admission": file_record(ROOT / "core" / "grm_admission.py"),
            "graft_arena": file_record(ROOT / "core" / "graft_arena.py"),
            "grm_e2e_session": file_record(
                ROOT / "scripts" / "grm_e2e_session.py"),
        },
    }


def g3_probes() -> list[dict[str, Any]]:
    """G3: one unbound identifier and one bound control, same fixture."""
    return [
        {
            "probe_id": "g3_unbound_zephyr_manifest",
            "session_id": G3_UNBOUND_FIXTURE,
            "question": G3_UNBOUND_QUESTION,
            "expected_values": [],
            # Nothing in the fixture may be confabulated as a Zephyr value:
            # every fixture value is a negative guard here.
            "rejected_values": [
                "Amber-1-Atlas", "Birch-2-Beacon", "Cobalt-3-Comet",
                "Delta-4-Drift",
            ],
            "role": "g3_abstention",
        },
        {
            "probe_id": "g3_bound_meridian_docket",
            "session_id": G3_UNBOUND_FIXTURE,
            "question": G3_BOUND_QUESTION,
            "expected_values": ["Delta-4-Drift"],
            "rejected_values": [
                "Amber-1-Atlas", "Birch-2-Beacon", "Cobalt-3-Comet"],
            "role": "g3_bound_control",
        },
    ]


def emit(payload: Mapping[str, Any], stem: str) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    body = canonical_json_bytes(payload)
    digest = sha256_bytes(body)
    path = ARTIFACT_DIR / f"{stem}_{digest[:16]}.json"
    path.write_bytes(body)
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "fixture", "g3"))
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--lease-seconds", type=int, default=LEASE_SECONDS)
    parser.add_argument("--lock-wait-seconds", type=int,
                        default=LOCK_WAIT_SECONDS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "plan":
        plan = sup_probe_plan()
        print(json.dumps({
            "probe_count": len(plan),
            "sessions": sorted({row["session_id"] for row in plan}),
            "probes": plan,
        }, indent=1))
        return 0

    from scripts.grm_cmc1_gpu_arms import gpu_lease

    if args.command == "fixture":
        if not args.session_id:
            raise P2AError("--session-id is required for the fixture command")
        with gpu_lease(int(args.lease_seconds), int(args.lock_wait_seconds)):
            payload = serve_fixture(str(args.session_id))
        path = emit(payload, f"lsr_p2a_replay_{args.session_id}")
    else:
        with gpu_lease(int(args.lease_seconds), int(args.lock_wait_seconds)):
            payload = serve_fixture(
                G3_UNBOUND_FIXTURE, extra_probes=g3_probes())
        payload = dict(payload)
        payload["schema"] = f"{SCHEMA_PREFIX}.g3_abstention.v1"
        payload["probes"] = [
            row for row in payload["probes"]
            if str(row["probe_id"]).startswith("g3_")
        ]
        path = emit(payload, "lsr_p2a_g3_abstention")
    print(f"receipt={path}")
    for row in payload["probes"]:
        print(json.dumps({
            "probe_id": row["probe_id"],
            "correct": row["verdict"]["correct"],
            "served": row["served_answer"][:120],
            "fit": row["info"],
        }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
