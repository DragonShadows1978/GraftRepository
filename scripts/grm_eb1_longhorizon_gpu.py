#!/usr/bin/env python3
"""GRM-EB1 Part 3 — does recall survive DISTANCE under the spec frame?

Program: GRM, order ``orders/GRM_EB1_EPHEMERAL_BOAT_FRAME.md`` (Part 3, G5).

THE QUESTION.  David's spec says the chat can grow to ANY length because the
chat log is never kept in context — recall is a routed graft, not a live
window.  The certified census is 34 turns, which is not long enough to test
that claim: every probe there sits within a few turns of its fact.  This
module extends the SAME certified session past 100 turns and probes facts
deposited >= 30 turns earlier, so "recall vs turn distance" is measured
rather than assumed.

WHAT IT DOES NOT DO.  It does not invent a new fixture family, a new prompt
shape, or a new scoring rule.  The deposit turns are built by the driver's OWN
certified constructors (``fact_turn`` / ``supersede_turn`` / ``probe_turn``),
and the generator only supplies FRESH VALUES in the existing Word-N-Phonetic
code shape.  A new corpus would confound "distance hurts recall" with "this
corpus is harder", which is the one comparison this gate exists to make.

THE REGISTERED GENERATOR.  Deterministic, seeded, and stated here before any
run so the sequence is reproducible and cannot be tuned after seeing results:

  * families cycle through the census's own fact families, in census order;
  * value = ``f"{WORD}-{n}-{PHONETIC}"`` drawn by INDEX from the frozen word
    and NATO lists below — index arithmetic only, no RNG state to drift;
  * a family met again is SUPERSEDED with the new value, the way the census
    supersedes, but now at a distance;
  * a probe every ``PROBE_EVERY`` turns targets the eligible fact whose
    deposit is FURTHEST back, so measured distance grows across the run
    instead of hovering at the ``MIN_DISTANCE`` floor.

These are not gameplay/tuning constants in the house sense: they are the
EXPERIMENT'S REGISTERED SHAPE, fixed before the run and reported with it.

GPU DISCIPLINE.  One shard per process, one lease per shard, <= 580 s,
operator absolute right of way; the caller inserts the inter-run gap.  Shards
resume through the driver's own save/restore, exactly as ``lsr_p2c_e2e_gpu``
does — no new persistence mechanism is introduced.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.grm_frame import (  # noqa: E402
    ENV_NAME as PERSISTENT_BOAT_ENV,
    env_persistent_boat,
    ephemeral_frame_enabled,
)
from scripts.grm_cmc1_mechanism import (  # noqa: E402
    canonical_json_bytes, sha256_bytes,
)
from scripts.grm_det1_common import contains_value, file_record  # noqa: E402
from scripts.lsr_p2c_replay_gpu import RUNTIME_FRAME  # noqa: E402

ARTIFACT_DIR = ROOT / "artifacts" / "grm_eb1"
SCHEMA_PREFIX = "grm.eb1"
LEASE_SECONDS = 580
LOCK_WAIT_SECONDS = 7200

#: The registered shape of the extension.  Fixed BEFORE the run.
TARGET_TURNS = 104          # >= 100, a whole number of probe cycles
PROBE_EVERY = 10            # "probing every 10 turns", per the order
MIN_DISTANCE = 30           # "a fact deposited >= 30 turns earlier"
GENERATOR_SEED = 20260903   # deterministic; index arithmetic only

#: The census's own fact families, in census order.  These are the CHURN
#: families: the extension supersedes them on every pass, so they generate the
#: interference a long chat really produces.  Reused, not re-invented.
FAMILIES = (
    "orion pin", "lyra dock", "cypher bridge", "nova key", "mira seal",
    "terra port", "ember code", "atlas tone", "polaris mark",
)

#: ANCHOR families: deposited ONCE each, early in the extension, and NEVER
#: superseded.  They are what makes "recall vs turn distance" measurable at
#: all.
#:
#: WHY THEY EXIST (a defect found in this module's first plan, recorded rather
#: than quietly fixed).  The first version probed the churn families, whose
#: newest deposit is re-written every full cycle -- so no fact was ever 30
#: turns old, ``eligible`` was always empty, and the extension generated ZERO
#: probes.  Superseding a fact resets its distance to zero by definition, so a
#: churn-only script cannot answer the distance question no matter how long it
#: runs.  Anchors are held fixed precisely so distance is the variable.
#:
#: They use fresh family names in the census's own naming style, because
#: re-using a census family would let the extension's churn supersede the very
#: fact the probe is asking about.
ANCHOR_FAMILIES = (
    "beacon ward", "tessera lock", "verdant key", "obsidian tag",
    "lantern seal", "cascade mark", "granite pin", "meridian bolt",
)

#: Anchors are laid down in the first ``ANCHOR_STRIDE``-spaced deposit slots of
#: the extension, so the earliest is probed at the greatest distance.
ANCHOR_STRIDE = 2

#: The census's own value vocabulary shape: Word-N-Phonetic.
WORDS = (
    "Auric", "Nadir", "Vortex", "Quartz", "Silver", "Harbor", "Violet",
    "Cobalt", "Marble", "Kestrel", "Zenith", "Gold", "Amber", "Cinder",
    "Onyx", "Pewter", "Russet", "Saffron", "Teal", "Umber",
)
PHONETIC = (
    "Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf",
    "Hotel", "India", "Juliet", "Kilo", "Lima", "Mike", "November",
    "Oscar", "Papa", "Quebec", "Romeo", "Sierra", "Tango",
)

#: Shard turn boundaries.  Shard 1 reproduces the certified 34-turn session
#: exactly (it IS the census script); the rest extend it.
#:
#: The stops sit one PAST each probe turn on purpose.  ``install_stop_boundary``
#: halts BEFORE executing the turn it names, so a stop of exactly 70 would end
#: the shard without ever running the turn-70 probe -- measured on the first
#: lh-3 run, whose scorecard held only the ten census probes.
SHARDS = ("lh-1", "lh-2", "lh-3", "lh-4", "lh-5")
SHARD_STOPS = {"lh-1": 34, "lh-2": 52, "lh-3": 71, "lh-4": 91, "lh-5": 104}


class LongHorizonError(RuntimeError):
    """The Part 3 extension could not be performed as registered."""


#: The driver's ORIGINAL 34-turn script builder.
#:
#: ``run_shard`` rebinds ``e2e.build_full_script`` to ``build_long_script`` so
#: the driver builds the extended session.  ``build_long_script`` then needs
#: the ORIGINAL builder for its certified prefix, and reading the attribute at
#: that moment would hand it itself.  Pinning the function object here -- set
#: exactly once, the first time anything binds the driver -- gives it a handle
#: the rebinding cannot shadow.  (Measured: RecursionError on the first run.)
_CENSUS_BUILDER = None


def _bind_census_builder(e2e) -> None:
    """Pin the driver's original builder, once, before any rebinding."""
    global _CENSUS_BUILDER
    if _CENSUS_BUILDER is None:
        _CENSUS_BUILDER = e2e.build_full_script


def _read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def generated_value(ordinal: int) -> str:
    """The registered generator: index arithmetic, no RNG state.

    ``ordinal`` is the deposit's index in the extension (0-based).  The
    strides are chosen coprime with their list lengths so a value repeats only
    after the full cycle, and the seed shifts the whole sequence without
    changing its shape.
    """
    base = int(GENERATOR_SEED) + int(ordinal)
    word = WORDS[(base * 7) % len(WORDS)]
    digit = (base * 3) % 10
    phon = PHONETIC[(base * 11) % len(PHONETIC)]
    return f"{word}-{digit}-{phon}"


def build_long_script() -> list[dict[str, Any]]:
    """The certified 34-turn census script, then the registered extension.

    The extension's turns are built by the DRIVER'S OWN constructors, so the
    prompt shapes, the acceptance text and the scoring contract are the
    census's and not this module's.
    """
    from scripts import grm_e2e_session as e2e
    from scripts.grm_det1_5_workers import _install_polaris_probe

    # The census script, with DET1.9's registered turn-33 polaris probe
    # installed -- the same replacement ``lsr_p2c_e2e_gpu`` installs, so shard
    # 1 is the certified session and not a variant of it.
    #
    # ``_install_polaris_probe`` WRAPS whatever is bound at the time, so the
    # ORIGINAL census builder must be put back before it wraps -- otherwise it
    # wraps this function and the prefix builds itself forever.
    _bind_census_builder(e2e)
    current = e2e.build_full_script
    e2e.build_full_script = _CENSUS_BUILDER
    try:
        _install_polaris_probe(e2e)
        script = list(e2e.build_full_script())
    finally:
        e2e.build_full_script = current

    # ANCHORS: fact_id -> (deposit turn index, value), written once and never
    # superseded.  Only anchors are probe targets, because a superseded fact's
    # distance resets to zero and cannot answer the distance question.
    anchors: dict[str, tuple[int, str]] = {}
    # CHURN: the census families, re-superseded every pass, generating the
    # interference the anchors have to survive.  Never probed by this module.
    churn: dict[str, str] = {}
    for event in script:
        if event.get("kind") in ("fact", "supersede") and event.get("fact_id"):
            churn[str(event["fact_id"])] = str(event["value"])

    ordinal = 0
    anchors_laid = 0
    probed: set[str] = set()
    while len(script) < TARGET_TURNS:
        turn_index = len(script)
        if turn_index % PROBE_EVERY == 0:
            eligible = [
                (fid, t, v) for fid, (t, v) in anchors.items()
                if turn_index - t >= MIN_DISTANCE and fid not in probed
            ]
            if eligible:
                # Furthest-back first, so measured distance GROWS across the
                # run instead of hovering at the MIN_DISTANCE floor.
                eligible.sort(key=lambda row: (row[1], row[0]))
                fid, src, val = eligible[0]
                probed.add(fid)
                script.append(e2e.probe_turn(fid, val, source_turn=int(src)))
                continue
        # Lay the anchors down first, spaced, then churn for the rest.
        if (anchors_laid < len(ANCHOR_FAMILIES)
                and ordinal % ANCHOR_STRIDE == 0):
            family = ANCHOR_FAMILIES[anchors_laid]
            value = generated_value(ordinal)
            script.append(e2e.fact_turn(family, value))
            anchors[family] = (turn_index, value)
            anchors_laid += 1
        else:
            family = FAMILIES[ordinal % len(FAMILIES)]
            value = generated_value(ordinal)
            prior = churn.get(family)
            if prior is None:
                script.append(e2e.fact_turn(family, value))
            else:
                script.append(e2e.supersede_turn(family, prior, value))
            churn[family] = value
        ordinal += 1
    return script


def extension_plan() -> dict[str, Any]:
    """The registered sequence, computable WITHOUT a GPU."""
    script = build_long_script()
    probes = []
    for idx, event in enumerate(script):
        if event.get("kind") == "probe":
            src = int(event.get("source_turn", -1))
            probes.append({
                "turn": idx,
                "fact_id": event.get("fact_id"),
                "expected": event.get("expected"),
                "source_turn": src,
                "distance": idx - src,
                "in_certified_census": idx < 34,
            })
    return {
        "schema": f"{SCHEMA_PREFIX}.longhorizon_plan.v1",
        "total_turns": len(script),
        "certified_prefix_turns": 34,
        "registered_shape": {
            "TARGET_TURNS": TARGET_TURNS,
            "PROBE_EVERY": PROBE_EVERY,
            "MIN_DISTANCE": MIN_DISTANCE,
            "GENERATOR_SEED": GENERATOR_SEED,
            "families": list(FAMILIES),
        },
        "shards": [
            {"spec": s, "stop_after_turns": SHARD_STOPS[s],
             "resumes": (SHARDS[i - 1] if i else None)}
            for i, s in enumerate(SHARDS)
        ],
        "probes": probes,
        "extension_probes": [p for p in probes if not p["in_certified_census"]],
    }


def run_shard(spec: str, *, run_dir: Path) -> dict[str, Any]:
    """Run one shard of the extended session, resuming the previous one."""
    from scripts import grm_det1_e2e as det_e2e
    from scripts import grm_e2e_session as e2e

    if spec not in SHARD_STOPS:
        raise LongHorizonError(f"unknown shard {spec}")
    frame = _read(RUNTIME_FRAME)
    flags = frame["resolved_flags"]

    session_dir = Path(run_dir) / spec / "session"
    index = SHARDS.index(spec)
    resume = index > 0
    if session_dir.exists():
        shutil.rmtree(session_dir)
    session_dir.parent.mkdir(parents=True, exist_ok=True)
    if resume:
        prior = Path(run_dir) / SHARDS[index - 1] / "session"
        if not prior.is_dir():
            raise LongHorizonError(
                f"shard {spec} needs {SHARDS[index - 1]} first: {prior} missing")
        shutil.copytree(prior, session_dir, symlinks=True)

    stop_after = SHARD_STOPS[spec]
    _bind_census_builder(e2e)
    restore_script = e2e.build_full_script
    e2e.build_full_script = build_long_script
    det_e2e.install_stop_boundary(e2e, stop_after)
    argv = [
        "--mode", "full",
        "--session-dir", str(session_dir),
        "--model-dir", str(frame["model"]["path"]),
        "--native-lib", str(ROOT / frame["native_library"]["path"]),
        "--ngen", str(flags["ngen"]),
        "--max-trips", str(flags["max_trips"]),
        "--live-turns", str(flags["live_turns"]),
        "--arena-width", str(flags["arena_width"]),
        "--max-live", str(flags["max_live"]),
        "--topk", str(flags["topk"]),
        "--turn-pipeline", str(flags["turn_pipeline"]),
        "--restart-after", "999",
        "--skip-gpu-idle-check", "--probe-ladder", "--sup-resolve",
        "--adm-decisive" if flags["adm_decisive"] else "--no-adm-decisive",
    ]
    if resume:
        argv.append("--resume")
    started = time.monotonic()
    stopped = False
    try:
        e2e.main(argv)
    except det_e2e._DETStageStop:
        stopped = True
    finally:
        e2e.build_full_script = restore_script
    elapsed = time.monotonic() - started
    if not stopped:
        raise LongHorizonError(f"shard {spec} did not stop at turn {stop_after}")
    scorecard = session_dir / "probe_scorecard.json"
    if not scorecard.is_file():
        raise LongHorizonError(f"shard {spec} produced no probe scorecard")
    return {
        "spec": spec,
        "stop_after_turns": int(stop_after),
        "resumed": bool(resume),
        "session_dir": str(session_dir),
        "elapsed_seconds": float(elapsed),
        "scorecard": file_record(scorecard),
        "probe_rows": len(_read(scorecard).get("probes", [])),
    }


def _bucket(distance: int) -> str:
    """Reporting buckets only -- no threshold is gated on them."""
    for edge in (10, 20, 30, 40, 60, 80):
        if distance <= edge:
            return f"<={edge}"
    return ">80"


def score_run(run_dir: Path) -> dict[str, Any]:
    """Recall vs turn distance, read off the final shard's scorecard."""
    final = Path(run_dir) / SHARDS[-1] / "session" / "probe_scorecard.json"
    if not final.is_file():
        raise LongHorizonError(
            f"no final scorecard at {final}; run every shard first")
    rows = {int(r["turn"]): r for r in _read(final).get("probes", [])
            if "turn" in r}
    plan = extension_plan()
    table = []
    for probe in plan["probes"]:
        row = rows.get(int(probe["turn"]))
        answer = str((row or {}).get("answer", ""))
        table.append({
            **probe,
            "served_answer": answer,
            "correct": bool(
                probe["expected"]
                and contains_value(answer, str(probe["expected"]))),
            "turn_row_found": row is not None,
            "mounted_ids": (row or {}).get("mounted_ids"),
            "source_rank": (
                ((row or {}).get("route_ranking") or {}).get("source_rank")),
            "live_node_ids_before": (
                ((row or {}).get("eviction_check") or {}).get(
                    "live_node_ids_before")),
        })

    by_distance: dict[str, dict[str, int]] = {}
    for row in table:
        if not row["turn_row_found"]:
            continue
        cell = by_distance.setdefault(
            _bucket(int(row["distance"])), {"n": 0, "correct": 0})
        cell["n"] += 1
        cell["correct"] += 1 if row["correct"] else 0

    measured = [r for r in table if r["turn_row_found"]]
    return {
        "schema": f"{SCHEMA_PREFIX}.longhorizon_result.v1",
        "program": "GRM",
        "order": "orders/GRM_EB1_EPHEMERAL_BOAT_FRAME.md",
        "gate": "G5",
        "frame": {
            "ephemeral_declared": bool(ephemeral_frame_enabled()),
            "escape_env": os.environ.get(PERSISTENT_BOAT_ENV),
            "escape_active": bool(env_persistent_boat()),
        },
        "registered_shape": plan["registered_shape"],
        "total_turns": plan["total_turns"],
        "probes": table,
        "measured_count": len(measured),
        "correct_count": sum(1 for r in measured if r["correct"]),
        "recall_by_distance_bucket": by_distance,
        "runtime_frame": file_record(RUNTIME_FRAME),
        "sources": {
            "grm_eb1_longhorizon_gpu": file_record(Path(__file__).resolve()),
            "graft_arena": file_record(ROOT / "core" / "graft_arena.py"),
            "grm_frame": file_record(ROOT / "core" / "grm_frame.py"),
            "grm_e2e_session": file_record(
                ROOT / "scripts" / "grm_e2e_session.py"),
        },
    }


def emit(payload: Mapping[str, Any], stem: str) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    body = canonical_json_bytes(payload)
    digest = sha256_bytes(body)
    path = ARTIFACT_DIR / f"{stem}_{digest[:16]}.json"
    path.write_bytes(body)
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "shard", "score"))
    parser.add_argument("--spec", default=None, choices=SHARDS)
    parser.add_argument("--run-dir", type=Path,
                        default=ROOT / "artifacts" / "grm_eb1" / "g5_run")
    parser.add_argument("--lease-seconds", type=int, default=LEASE_SECONDS)
    parser.add_argument("--lock-wait-seconds", type=int,
                        default=LOCK_WAIT_SECONDS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "plan":
        print(json.dumps(extension_plan(), indent=1))
        return 0

    if args.command == "score":
        payload = score_run(args.run_dir)
        path = emit(payload, "grm_eb1_longhorizon")
        print(f"receipt={path}")
        print(f"measured={payload['measured_count']} "
              f"correct={payload['correct_count']}")
        print(f"by_distance={json.dumps(payload['recall_by_distance_bucket'])}")
        for row in payload["probes"]:
            print(json.dumps({
                "turn": row["turn"],
                "fact": row["fact_id"],
                "distance": row["distance"],
                "correct": row["correct"],
                "source_rank": row["source_rank"],
                "served": str(row["served_answer"])[:48],
            }))
        return 0

    if not args.spec:
        raise LongHorizonError("--spec is required for the shard command")

    from scripts.grm_cmc1_gpu_arms import gpu_lease

    with gpu_lease(int(args.lease_seconds), int(args.lock_wait_seconds)):
        out = run_shard(str(args.spec), run_dir=args.run_dir)
    print(json.dumps({
        "spec": out["spec"],
        "stop_after_turns": out["stop_after_turns"],
        "resumed": out["resumed"],
        "elapsed_seconds": round(out["elapsed_seconds"], 1),
        "probe_rows": out["probe_rows"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
