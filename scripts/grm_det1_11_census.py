#!/usr/bin/env python3
"""GRM lived-serving reliability census — receipts only, no cause analysis.

The DET1 race campaign collected a lived served control for every probe it
touched: the fourteen registered plant slots (two calibration, twelve
evaluation) and the five cross-session reserves DET1.9 added.  Across
campaign rounds r4-r7 a meaningful fraction of those controls came back
either as a refusal or as a confidently WRONG value, with the CORRECT single
graft mounted.  That is a distinct defect class from co-mount blending, and
this module is its evidence base.

What this module does: it replays the persisted plant-registration
observations, classifies each probe's lived served control, and freezes an
append-only content-addressed census receipt.

Receipts form a LINEAGE.  Campaign rounds keep collecting, so the lived
evidence legitimately moves between runs; a frozen receipt is never rewritten
to match.  Instead each round that finds drifted evidence authors a SUCCESSOR
naming its predecessor in ``supersedes``, and the campaign consumes the
newest member (``current_census``).  A round whose evidence is unchanged
returns the existing receipt untouched, so re-running the lead script does
not grow the chain.  ``census_lineage`` reconstructs the order from the
supersedes links rather than filenames — content addresses sort arbitrarily —
and fails closed on an orphan, a fork, a cycle, or two originals, because any
of those makes "which receipt is current" ambiguous.

What this module deliberately does NOT do, per the DET1.11 order: it does not
investigate why any probe served what it served.  No hypothesis, no
mechanism, no remediation.  The successor investigation owns that; this is
the foundation it will stand on.

Classification is a partition — every probe lands in exactly one class:

``LAWFUL``
    The served control matched an expected value under the registered
    comparator.  Sub-classed as ``exact`` or ``separator_variant`` so the
    DET1.10 rescue is visible rather than silently folded in.
``REFUSAL``
    The model declined (``_is_refusal`` markers).  It asserted no value.
``WRONG_VALUE``
    The model asserted a value, confidently, and it was not the expected
    one.  This is the new defect class.

The separator-artifact-rescued probes are reported as LAWFUL with subclass
``separator_variant``: under the pre-DET1.10 comparator they were counted as
lived failures, and the census says so explicitly, because the successor
investigation needs to know the rescue happened rather than rediscover it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.grm_det1_common import (  # noqa: E402
    DETError,
    contains_value,
    file_record,
    normalize_value_text,
    read_jsonl,
    utc_now,
    write_content_addressed,
)
from scripts.grm_det1_3_gpu import _is_refusal  # noqa: E402


FROZEN_RUN = ROOT / "artifacts/grm_det1/run_20260831T160525Z_2"
ORDER = ROOT / "orders/GRM_DET1_11_ACHIEVED_COUNT.md"
REGISTRATION = FROZEN_RUN / "registration_62cb6c09cbec211d.json"
SHARD_ROOT = (
    FROZEN_RUN / "det1_4/campaign/det1_7/plant_registration/shards"
)
CENSUS_DIR = FROZEN_RUN / "det1_11/census"

SCHEMA = "grm.det1_11.lived_serving_reliability_census.v1"
TITLE = "GRM lived-serving reliability census"
ORDER_ID = "GRM-DET1.11"
RUN_ID = "run_20260831T160525Z_2"
STATUS = "RECEIPTS_ONLY_CAUSE_NOT_INVESTIGATED"
UNPLANTABLE_REASON = "LIVED_SERVED_CONTROL_INCORRECT_OR_REFUSAL"

# The fields that constitute the census's EVIDENCE.  Two receipts agreeing on
# every one of these describe the same lived population, so the existing
# receipt is returned unchanged (idempotence).  A difference in any of them is
# genuine evidence drift and earns a successor.  created_utc and the file
# records are deliberately NOT evidence — they move on every derivation and
# would otherwise make idempotence impossible.
EVIDENCE_FIELDS = (
    "schema",
    "title",
    "order",
    "run_id",
    "status",
    "counts",
    "rows",
    "attempt_history",
    "markdown",
    "cause_investigated",
    "scope_note",
    "defect_class_note",
)

CLASS_LAWFUL = "LAWFUL"
CLASS_REFUSAL = "REFUSAL"
CLASS_WRONG_VALUE = "WRONG_VALUE"
CLASSES = (CLASS_LAWFUL, CLASS_REFUSAL, CLASS_WRONG_VALUE)

SUBCLASS_EXACT = "exact"
SUBCLASS_SEPARATOR = "separator_variant"

# The census speaks about a defect class; it does not explain it.  This
# sentence is the boundary, recorded in the receipt so a later reader cannot
# mistake the census for an analysis.
SCOPE_NOTE = (
    "Receipts only.  This census records WHAT each lived served control "
    "returned and how it classifies.  It does not investigate WHY, propose a "
    "mechanism, or recommend a remedy; the DET1.11 order reserves cause "
    "analysis for the successor investigation."
)

DEFECT_CLASS_NOTE = (
    "The WRONG_VALUE class is distinct from co-mount blending: each such "
    "probe served a confident value with its CORRECT single graft mounted, "
    "so no competing mount was present to blend with."
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DETError(message)


def _attempt_index(path: Path) -> int:
    name = path.parent.name
    _require(
        name.startswith("attempt_"),
        f"observation is not under an attempt directory: {path}",
    )
    return int(name.split("_", 1)[1])


def collect_latest_observations(
    shard_root: Path = SHARD_ROOT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (latest observation per probe, per-probe attempt history).

    A probe may be collected several times across campaign rounds.  The
    census reports the LATEST attempt as the probe's lived result, and keeps
    the attempt history so a status that changed between rounds — the DET1.10
    separator rescue is exactly such a case — stays visible.
    """

    shard_root = Path(shard_root).resolve()
    _require(shard_root.is_dir(), f"shard root is absent: {shard_root}")
    by_probe: dict[str, list[tuple[int, str, dict[str, Any]]]] = {}
    for path in sorted(shard_root.glob("*/attempt_*/observations.jsonl")):
        spec = path.parts[-3]
        attempt = _attempt_index(path)
        for row in read_jsonl(path):
            probe_id = str(row.get("effective_fixture_id") or "")
            _require(
                bool(probe_id),
                f"observation lacks an effective_fixture_id: {path}",
            )
            by_probe.setdefault(probe_id, []).append((attempt, spec, dict(row)))

    latest: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    for probe_id in sorted(by_probe):
        rows = sorted(by_probe[probe_id], key=lambda item: item[0])
        attempt, spec, row = rows[-1]
        row = dict(row)
        row["_census_spec"] = spec
        row["_census_attempt"] = attempt
        latest.append(row)
        history.append({
            "probe_id": probe_id,
            "attempts": [
                {
                    "attempt": item[0],
                    "spec": item[1],
                    "status": item[2].get("status"),
                    "served_answer_correct": item[2].get(
                        "served_answer_correct"),
                }
                for item in rows
            ],
            "status_changed_across_attempts": len({
                str(item[2].get("status")) for item in rows
            }) > 1,
        })
    return latest, history


def classify_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Classify one lived served control into the census partition."""

    effective = observation.get("effective_fixture") or {}
    expected = [str(value) for value in (effective.get("expected_values") or ())]
    _require(
        bool(expected),
        f"probe has no expected values: "
        f"{observation.get('effective_fixture_id')}",
    )
    answer = observation.get("served_answer")
    answer_text = "" if answer is None else str(answer)
    status = str(observation.get("status") or "")

    if status == "LAWFUL_LIVED_TARGET":
        # A lawful control matched under the registered comparator.  Report
        # whether the match needed DET1.10 separator normalization, because
        # under the pre-DET1.10 comparator this probe read as a failure.
        exact = any(
            normalize_value_text(value).casefold()
            in normalize_value_text(answer_text).casefold()
            for value in expected
        )
        return {
            "census_class": CLASS_LAWFUL,
            "census_subclass": SUBCLASS_EXACT if exact else SUBCLASS_SEPARATOR,
            "separator_artifact_rescued": not exact,
        }

    if _is_refusal(answer_text):
        return {
            "census_class": CLASS_REFUSAL,
            "census_subclass": "declined_to_assert_a_value",
            "separator_artifact_rescued": False,
        }

    # Not lawful, not a refusal: the model asserted a value and the value was
    # wrong.  Guard the partition — a "wrong" answer that actually contains an
    # expected value would mean the comparator and this census disagree, which
    # is a defect in one of them and must fail closed rather than be reported.
    _require(
        not any(contains_value(answer_text, value) for value in expected),
        f"census/comparator disagreement for "
        f"{observation.get('effective_fixture_id')}: answer classified "
        f"unlawful yet contains an expected value",
    )
    return {
        "census_class": CLASS_WRONG_VALUE,
        "census_subclass": "asserted_a_confident_incorrect_value",
        "separator_artifact_rescued": False,
    }


def _census_row(observation: Mapping[str, Any]) -> dict[str, Any]:
    effective = observation.get("effective_fixture") or {}
    classification = classify_observation(observation)
    mounted = [int(value) for value in (observation.get("mounted_ids") or ())]
    answer = observation.get("served_answer")
    return {
        "probe_id": str(observation.get("effective_fixture_id")),
        "slot_fixture_id": str(observation.get("fixture_id")),
        "spec": str(observation.get("_census_spec")),
        "attempt": int(observation.get("_census_attempt")),
        "role": (
            "reserve" if bool(observation.get("reserve_candidate"))
            else "registered_slot"
        ),
        "split": str(effective.get("split") or ""),
        "source_family": str(effective.get("source_family") or ""),
        "status": str(observation.get("status") or ""),
        "unplantable_reason": observation.get("unplantable_reason"),
        "served_answer_correct": observation.get("served_answer_correct"),
        "expected_values": [
            str(value) for value in (effective.get("expected_values") or ())
        ],
        "served_answer": None if answer is None else str(answer),
        "mounted_ids": mounted,
        "mount_count": len(mounted),
        "selected_target_id": observation.get("selected_target_id"),
        # Load-bearing for the successor: a single mount means no co-mount
        # blending was available as an explanation.
        "single_mount": len(mounted) == 1,
        **classification,
    }


def _counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    lawful = [r for r in rows if r["census_class"] == CLASS_LAWFUL]
    refusal = [r for r in rows if r["census_class"] == CLASS_REFUSAL]
    wrong = [r for r in rows if r["census_class"] == CLASS_WRONG_VALUE]
    rescued = [r for r in lawful if r["separator_artifact_rescued"]]
    unlawful = refusal + wrong
    wrong_single_mount = [r for r in wrong if r["single_mount"]]
    return {
        "probes_total": len(rows),
        "lawful": len(lawful),
        "lawful_exact": len(lawful) - len(rescued),
        "lawful_separator_artifact_rescued": len(rescued),
        "unlawful": len(unlawful),
        "unlawful_refusal": len(refusal),
        "unlawful_wrong_value": len(wrong),
        "wrong_value_with_correct_single_mount": len(wrong_single_mount),
        "registered_slots": len(
            [r for r in rows if r["role"] == "registered_slot"]),
        "reserves": len([r for r in rows if r["role"] == "reserve"]),
        "registered_slots_lawful": len([
            r for r in rows
            if r["role"] == "registered_slot"
            and r["census_class"] == CLASS_LAWFUL
        ]),
        "reserves_lawful": len([
            r for r in rows
            if r["role"] == "reserve" and r["census_class"] == CLASS_LAWFUL
        ]),
        "unlawful_rate": (
            round(len(unlawful) / len(rows), 4) if rows else 0.0
        ),
    }


def _markdown(
    rows: Sequence[Mapping[str, Any]], counts: Mapping[str, Any]
) -> str:
    lines = [
        f"# {TITLE}",
        "",
        f"Run `{RUN_ID}` — every lived-collected probe in the DET1 race "
        f"campaign: registered plant slots (calibration and evaluation) and "
        f"DET1.9 cross-session reserves alike.",
        "",
        SCOPE_NOTE,
        "",
        "## Counts",
        "",
        "| Class | Probes |",
        "|---|---:|",
        f"| Lawful | {counts['lawful']} |",
        f"| — exact match | {counts['lawful_exact']} |",
        f"| — separator-artifact rescued (DET1.10) | "
        f"{counts['lawful_separator_artifact_rescued']} |",
        f"| Unlawful ({UNPLANTABLE_REASON}) | {counts['unlawful']} |",
        f"| — refusal | {counts['unlawful_refusal']} |",
        f"| — wrong value | {counts['unlawful_wrong_value']} |",
        f"| **Total probes** | **{counts['probes_total']}** |",
        "",
        f"Unlawful rate: {counts['unlawful']}/{counts['probes_total']} "
        f"({100.0 * float(counts['unlawful_rate']):.1f}%). "
        f"Registered slots lawful: {counts['registered_slots_lawful']}/"
        f"{counts['registered_slots']}. "
        f"Reserves lawful: {counts['reserves_lawful']}/{counts['reserves']}.",
        "",
        f"Wrong values served with a correct SINGLE mount: "
        f"{counts['wrong_value_with_correct_single_mount']}. "
        f"{DEFECT_CLASS_NOTE}",
        "",
        "## Per-probe",
        "",
        "| Probe | Role | Split | Class | Mounts | Expected | Served answer |",
        "|---|---|---|---|---:|---|---|",
    ]
    for row in rows:
        served = row["served_answer"]
        served_text = "—" if served is None else served.replace("|", "\\|")
        expected = ", ".join(row["expected_values"]).replace("|", "\\|")
        subclass = row["census_subclass"]
        klass = row["census_class"]
        label = (
            f"{klass} ({subclass})"
            if klass == CLASS_LAWFUL and subclass == SUBCLASS_SEPARATOR
            else klass
        )
        lines.append(
            f"| `{row['probe_id']}` | {row['role']} | {row['split']} | "
            f"{label} | {row['mount_count']} | `{expected}` | {served_text} |"
        )
    return "\n".join(lines)


def build_census(
    shard_root: Path = SHARD_ROOT,
    *,
    record_root: Path | None = None,
    supersedes: Mapping[str, Any] | None = None,
    generation: int = 1,
) -> dict[str, Any]:
    """Derive the census payload without writing anything.

    ``supersedes`` names the immediate predecessor receipt (its file record)
    when this payload is a successor authored over drifted evidence;
    ``generation`` counts from 1 for the original.  Neither participates in
    the evidence comparison — a successor over identical evidence is exactly
    what must never be authored.
    """

    observations, history = collect_latest_observations(shard_root)
    _require(bool(observations), "census found no lived observations")
    rows = [_census_row(observation) for observation in observations]
    rows.sort(key=lambda row: (row["role"], row["split"], row["probe_id"]))
    counts = _counts(rows)
    _require(
        counts["lawful"] + counts["unlawful"] == counts["probes_total"],
        "census classes do not partition the probe population",
    )
    for row in rows:
        # Every unlawful row must carry the one reason the campaign records;
        # if a new reason string ever appears the census must not silently
        # absorb it into an existing class.
        if row["census_class"] != CLASS_LAWFUL:
            _require(
                row["unplantable_reason"] == UNPLANTABLE_REASON,
                f"unexpected unplantable reason for {row['probe_id']}: "
                f"{row['unplantable_reason']!r}",
            )
    return {
        "schema": SCHEMA,
        "title": TITLE,
        "order": ORDER_ID,
        "order_record": file_record(ORDER, root=record_root or ROOT),
        "run_id": RUN_ID,
        "status": STATUS,
        "scope_note": SCOPE_NOTE,
        "defect_class_note": DEFECT_CLASS_NOTE,
        "cause_investigated": False,
        "created_utc": utc_now(),
        "registration": file_record(REGISTRATION, root=record_root or ROOT),
        # Lineage.  The original census has supersedes=None and generation=1;
        # each successor names the receipt it replaces, so the chain can be
        # walked back to the original without reading any directory listing.
        "supersedes": None if supersedes is None else dict(supersedes),
        "generation": int(generation),
        "counts": counts,
        "rows": rows,
        "attempt_history": history,
        "markdown": _markdown(rows, counts),
    }


def _evidence_drift(
    stored: Mapping[str, Any], census: Mapping[str, Any]
) -> list[str]:
    """The evidence fields on which two census payloads disagree."""
    return [
        field for field in EVIDENCE_FIELDS
        if stored.get(field) != census.get(field)
    ]


def census_lineage(
    directory: Path = CENSUS_DIR,
) -> list[tuple[Path, dict[str, Any]]]:
    """Every census receipt in the directory, ordered original -> newest.

    Receipts are content-addressed, so filename order is meaningless; the
    chain is reconstructed from the ``supersedes`` links themselves.  A
    directory holding an unlinked receipt, a fork, or a cycle fails closed
    rather than letting the campaign guess which member is current.
    """

    directory = Path(directory)
    if not directory.is_dir():
        return []
    receipts: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(directory.glob("lived_serving_census_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        digest = file_record(path)["sha256"]
        receipts[digest] = (path, payload)
    if not receipts:
        return []

    # Index by predecessor: each receipt names the one it supersedes.
    successor_of: dict[str | None, list[str]] = {}
    for digest, (_path, payload) in receipts.items():
        parent = payload.get("supersedes")
        parent_sha = (
            None if parent is None else str((parent or {}).get("sha256", ""))
        )
        successor_of.setdefault(parent_sha, []).append(digest)

    roots = successor_of.get(None, [])
    _require(
        len(roots) == 1,
        f"census lineage must have exactly one original receipt "
        f"(supersedes=None), found {len(roots)}",
    )
    chain: list[tuple[Path, dict[str, Any]]] = []
    seen: set[str] = set()
    current: str | None = roots[0]
    while current is not None:
        _require(current not in seen, "census lineage contains a cycle")
        seen.add(current)
        _require(
            current in receipts,
            f"census lineage names a receipt that is not present: {current}",
        )
        chain.append(receipts[current])
        children = successor_of.get(current, [])
        _require(
            len(children) <= 1,
            f"census lineage forks at {current}: {len(children)} successors",
        )
        current = children[0] if children else None
    _require(
        len(chain) == len(receipts),
        f"census directory holds {len(receipts) - len(chain)} receipt(s) "
        f"outside the lineage chain",
    )
    for index, (path, payload) in enumerate(chain, start=1):
        # A receipt authored before the lineage fields existed carries no
        # generation.  It can only be the original — it has no supersedes
        # link, so the root check above already placed it at index 1 — and
        # is accepted as generation 1 rather than forcing the immutable
        # artifact to be rewritten.
        recorded = payload.get("generation")
        if recorded is None and index == 1:
            continue
        _require(
            isinstance(recorded, int) and not isinstance(recorded, bool)
            and recorded == index,
            f"census generation is out of order at {path.name}: "
            f"expected {index}, got {recorded!r}",
        )
    return chain


def current_census(
    directory: Path = CENSUS_DIR,
) -> tuple[Path, dict[str, Any]] | None:
    """The newest lineage member — the receipt the campaign consumes."""
    chain = census_lineage(directory)
    return chain[-1] if chain else None


def write_census(
    directory: Path = CENSUS_DIR,
    *,
    shard_root: Path = SHARD_ROOT,
    record_root: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Freeze the current census receipt, authoring a successor on drift.

    Three branches, and only three:

    * No receipt yet — author the original (``generation`` 1,
      ``supersedes`` None).
    * A receipt exists and its evidence is byte-identical to a fresh
      derivation — return it unchanged.  The payload carries ``created_utc``,
      so re-deriving always produces a different content address; comparing
      EVIDENCE rather than bytes is what keeps repeated writes idempotent.
    * A receipt exists and the evidence genuinely moved — author a SUCCESSOR
      naming it in ``supersedes``.  Prior receipts are never rewritten or
      removed; the campaign consumes the newest lineage member.

    A successor is authored only for real drift.  That guard is what stops
    the lineage from growing a link every time the lead script runs.
    """

    census = build_census(shard_root, record_root=record_root)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    chain = census_lineage(directory)
    if not chain:
        path = write_content_addressed(directory, "lived_serving_census", census)
        return path, census

    newest_path, newest = chain[-1]
    drifted = _evidence_drift(newest, census)
    if not drifted:
        return newest_path, newest

    # Real drift: fresh lived collections moved the evidence.  Author the
    # successor this module's error message has always promised.
    successor = build_census(
        shard_root,
        record_root=record_root,
        supersedes=file_record(newest_path),
        generation=int(newest.get("generation", len(chain))) + 1,
    )
    successor["superseded_evidence_fields"] = list(drifted)
    path = write_content_addressed(directory, "lived_serving_census", successor)
    _require(
        path != newest_path,
        "census successor collided with the receipt it supersedes",
    )
    return path, successor


def selftest() -> dict[str, Any]:
    """CPU-only contract checks over synthetic observations."""

    lawful = {
        "effective_fixture_id": "probe_lawful",
        "fixture_id": "probe_lawful",
        "status": "LAWFUL_LIVED_TARGET",
        "served_answer": "Auric-4-Alpha",
        "served_answer_correct": True,
        "mounted_ids": [3],
        "effective_fixture": {
            "split": "eval",
            "source_family": "certified_34_turn",
            "expected_values": ["Auric-4-Alpha"],
        },
    }
    rescued = {
        **lawful,
        "effective_fixture_id": "probe_rescued",
        "served_answer": "Cobalt 1 India",
        "effective_fixture": {
            **lawful["effective_fixture"],
            "expected_values": ["Cobalt-1-India"],
        },
    }
    refusal = {
        **lawful,
        "effective_fixture_id": "probe_refusal",
        "status": "UNPLANTABLE",
        "unplantable_reason": UNPLANTABLE_REASON,
        "served_answer": "I'm sorry, but I don't have that information.",
        "served_answer_correct": False,
    }
    wrong = {
        **refusal,
        "effective_fixture_id": "probe_wrong",
        "served_answer": "Birch-2-Beacon.",
    }
    got = {
        str(row["effective_fixture_id"]): classify_observation(row)
        for row in (lawful, rescued, refusal, wrong)
    }
    _require(
        got["probe_lawful"]["census_class"] == CLASS_LAWFUL
        and got["probe_lawful"]["census_subclass"] == SUBCLASS_EXACT,
        "exact lawful classification failed",
    )
    _require(
        got["probe_rescued"]["census_class"] == CLASS_LAWFUL
        and got["probe_rescued"]["separator_artifact_rescued"] is True,
        "separator-rescue classification failed",
    )
    _require(
        got["probe_refusal"]["census_class"] == CLASS_REFUSAL,
        "refusal classification failed",
    )
    _require(
        got["probe_wrong"]["census_class"] == CLASS_WRONG_VALUE,
        "wrong-value classification failed",
    )
    return {
        "schema": "grm.det1_11.census_selftest.v1",
        "status": "PASS",
        "classes": list(CLASSES),
        "cause_investigated": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=TITLE)
    parser.add_argument(
        "command", choices=("report", "write", "selftest", "lineage"))
    parser.add_argument("--shard-root", type=Path, default=SHARD_ROOT)
    parser.add_argument("--out-dir", type=Path, default=CENSUS_DIR)
    parser.add_argument("--markdown", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "selftest":
        print(json.dumps(selftest(), sort_keys=True))
        return 0
    if args.command == "lineage":
        chain = census_lineage(args.out_dir)
        print(json.dumps({
            "count": len(chain),
            "current": None if not chain else file_record(chain[-1][0]),
            "members": [
                {
                    "generation": payload.get("generation", 1),
                    "record": file_record(member),
                    "supersedes": payload.get("supersedes"),
                    "superseded_evidence_fields": payload.get(
                        "superseded_evidence_fields"),
                    "probes_total": (payload.get("counts") or {}).get(
                        "probes_total"),
                }
                for member, payload in chain
            ],
        }, indent=2, sort_keys=True))
        return 0
    if args.command == "report":
        # Read the frozen current member when a lineage exists, so the
        # report shows what the campaign consumes rather than a fresh
        # derivation that may not have been frozen yet.
        current = current_census(args.out_dir)
        census = current[1] if current else build_census(args.shard_root)
        if args.markdown:
            print(census["markdown"])
        else:
            print(json.dumps(census, indent=2, sort_keys=True))
        return 0
    path, census = write_census(args.out_dir, shard_root=args.shard_root)
    chain = census_lineage(args.out_dir)
    print(json.dumps({
        "census": file_record(path),
        "counts": census["counts"],
        "status": census["status"],
        # The campaign consumes the newest lineage member; prior receipts are
        # retained append-only and named here so the chain is auditable.
        "generation": census.get("generation"),
        "supersedes": census.get("supersedes"),
        "superseded_evidence_fields": census.get(
            "superseded_evidence_fields"),
        "lineage": [file_record(member) for member, _payload in chain],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
