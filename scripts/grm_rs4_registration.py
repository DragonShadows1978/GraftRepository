#!/usr/bin/env python3
"""GRM-RS4 — write the registration BEFORE any gate runs.

ORDER: ``orders/GRM_RS4_CEILING_REPARTITION.md``.

The order's Gates section says: "Register before any gate
(``artifacts/grm_rs4/registration.json``, immutable; amendments separate)."
This module writes that file and refuses to rewrite it.

WHAT RS4 REGISTERS, and why each number is DERIVED rather than typed.

RS3's live-band ceiling row (C5: the node text fed LIVE, nothing mounted) reads
``live_mass_full_layers = 0.7420`` on ``sup_harbor_restatement``, against
``0.4130`` of MOUNT-band mass for C3l on the same probe.  RS3 reported that
0.74-vs-0.41 pair as the ceiling and the residual as 0.33.  But the live band
is defined by ``core/grm_demand.py::_full_mass`` as
``[n_sink + cur_mount_n, S)`` — EVERY physical cache row that is not sink and
not mount.  In C5 that band holds three different things at once: the FED TEXT,
the QUESTION's own prompt rows, and the ANSWER rows generated so far.  In the
mounted arms (C2/C3l) it holds only the question and the answer, because
``_clear_boat`` empties the live cache before the serve and nothing is fed.

So RS3's "live mass 0.29" on a mounted arm is NOT residual attention on
something the mount failed to carry — it is, structurally, the question reading
ITSELF.  The comparable numbers are therefore:

    C5 FED-TEXT-row mass          vs    C3l MOUNT-band mass

and the residual is their difference, not 0.74 minus 0.41.  RS4 measures the
first of those two numbers, which RS3 did not persist.

THE ROW BOUNDARIES ARE DERIVED, never typed.  The chain, per serve:

  * ``S`` is ``key.shape[2]`` inside the observed attention call — the number
    of physical cache rows the kernel actually scored.  Read, not assumed.
  * ``n_sink`` and ``cur_mount_n`` come off the LIVE arena at observe time, the
    same two attributes ``_full_mass`` itself reads, so the sink and mount
    band edges are literally the production edges.
  * ``live0 = n_sink + cur_mount_n`` — ``_full_mass``'s own expression.
  * ``fed_ntok`` is the number of live rows the arm's own ``_feed_live``
    pushed, read off the feed receipt's ``live_segs_after_feed``.  Zero on
    every arm that feeds nothing.
  * ``prior_ntok`` is ``arena.pos`` at observer ENTRY minus ``fed_ntok``.
    ``_clear_boat`` sets ``pos = 0`` before every non-C0 arm, so it is zero
    there by that law; the C0 ladder does not clear, so its prior turns get
    their own sub-band instead of being folded into anything else.
  * ``question_ntok`` is ``len(arena.encode(arena._format_step_prompt(q)))``
    — the model's own tokenizer over ``_attempt``'s own prompt string.  This
    module computes the same number here, ahead of the run, purely so the
    registration can be checked against the receipts; the ARMS read it off the
    arena, never off this file.
  * ``answer_ntok`` is the remainder, ``S - live0 - prior - fed - question``.
    A remainder rather than a count, so the sub-bands are a PARTITION of the
    live band by construction and any drift shows up as a negative width
    instead of a silent mis-attribution.

THE DERIVATION WAS CHECKED AGAINST RS3's OWN RECEIPTS BEFORE IT WAS
REGISTERED, and the check is recorded below as
``derivation_crosscheck_against_rs3``: RS3's C5 receipt for
``sup_harbor_restatement`` carries ``live_tokens_after_feed = 211`` and a
sliding-layer ``S_min = 266``; ``266 - 19 - 211 = 36``, and this module's
independent tokenizer count of that probe's harmony prompt is 36.  The two
agree, so the row split is arithmetic over numbers RS3 already measured, not a
new assumption.

WHAT IS HARD-CODED, deliberately: receipt paths; the four arm names the order
names (C0, C2, C3l, C5); RS3's own pre-existing failing test set (so G1
compares against a frozen expectation); and the order's registered PREDICTION
numbers (0.29 +/- 0.05, 0.45, residual <= 0.08, the >= 0.20 branch), which are
the lead's numbers and must be carried verbatim rather than re-derived.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.grm_cmc1_mechanism import canonical_json_bytes  # noqa: E402
from scripts.grm_det1_common import file_record  # noqa: E402

ARTIFACT_DIR = ROOT / "artifacts" / "grm_rs4"
RS3_DIR = ROOT / "artifacts" / "grm_rs3"
RS3_REGISTRATION = RS3_DIR / "registration.json"
RS3_RESULTS = RS3_DIR / "grm_rs3_results.json"
RUNTIME_FRAME = (
    ROOT / "artifacts" / "grm_det1" / "run_20260831T160525Z_2"
    / "runtime_frame_28b3196f8fb04a41.json")
SUP_FIXTURES = ROOT / "tests" / "fixtures" / "supersession_battery"

#: The arms RS4 re-runs, in registration order.  A SUBSET of RS3's seven: the
#: order names exactly these four ("the RS3 arms C0, C2, C3l, C5").
ARMS = ("C0", "C2", "C3l", "C5")

#: The live band's sub-bands, in physical row order.  This tuple IS the
#: partition: the observer sums into these names and nothing else, and the
#: partition law checks that they add back to the live band's own total.
LIVE_SUBBANDS = ("prior_live_mass", "fed_text_mass", "question_mass",
                 "answer_mass")

#: The full band vocabulary a partition row must sum to 1.0 over.
BAND_NAMES = (("physical_sink_mass", "mounted_mass", "learned_sink_mass")
              + LIVE_SUBBANDS)


class RS4Error(RuntimeError):
    """Any RS4 registration or measurement violation."""


def _sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ======================================================================
# Geometry and question lengths — DERIVED, never typed
# ======================================================================
def derive_geometry() -> dict[str, Any]:
    """The band geometry RS4 measures in, recomputed rather than transcribed.

    Same chain RS3 registered (``arena_width`` off the frozen runtime frame,
    ``n_sink`` by encoding the frozen harmony sink through the model's own
    tokenizer, ``live_shift`` by ``ArenaCache.__init__``'s expression), and
    then CHECKED against RS3's registered geometry.  A drift in the sink text
    or the tokenizer fails this script instead of silently re-partitioning a
    band that no longer matches the one RS3 measured.
    """
    from transformers import AutoTokenizer

    from scripts import grm_det1_2_gpu as det1_2
    from scripts.grm_e2e_session import HARMONY_SINK

    frame = _read(RUNTIME_FRAME)
    arena_width = int(frame["resolved_flags"]["arena_width"])
    tokenizer = AutoTokenizer.from_pretrained(
        str(det1_2.MODEL_DIR), local_files_only=True)
    n_sink = len(tokenizer.encode(HARMONY_SINK, add_special_tokens=False))
    live_shift = n_sink + arena_width

    rs3 = _read(RS3_REGISTRATION)["geometry_REGISTERED_BEFORE_ANY_GATE"]
    for key, value in (("n_sink", n_sink), ("arena_width", arena_width),
                       ("live_shift", live_shift)):
        if int(rs3[key]) != int(value):
            raise RS4Error(
                f"RS4 derives {key}={value} but RS3 registered {rs3[key]}; the "
                "two orders would not be measuring the same band")
    return {
        "n_sink": int(n_sink),
        "n_sink_derived_by": (
            "len(tokenizer.encode(grm_e2e_session.HARMONY_SINK, "
            "add_special_tokens=False)) through the model's own tokenizer"),
        "harmony_sink_sha256": _sha256_text(HARMONY_SINK),
        "arena_width": int(arena_width),
        "arena_width_derived_by": (
            f"{RUNTIME_FRAME.name}::resolved_flags.arena_width — the same "
            "frozen frame the lived build reads"),
        "live_shift": int(live_shift),
        "live_shift_derived_by": (
            "ArenaCache.__init__'s own expression: n_sink + arena_width"),
        "agrees_with_rs3_registration": True,
        "band_layout": (
            "PHYSICAL cache rows: [0, n_sink) SINK; "
            "[n_sink, n_sink + cur_mount_n) MOUNT; "
            "[n_sink + cur_mount_n, S) LIVE — which RS4 splits further into "
            "prior_live | fed_text | question | answer, in that row order; "
            "plus the LEARNED sink column at index S, outside the cache"),
    }


def sessions_from_rs3() -> list[str]:
    """The fixtures in play, read off RS3's registration rather than typed."""
    probes = _read(RS3_REGISTRATION)["probes_REGISTERED_BEFORE_ANY_GATE"]
    return sorted({str(p["session_id"]) for p in probes.values()})


def derive_question_lengths() -> dict[str, Any]:
    """``len(encode(_format_step_prompt(question)))`` for every fixture probe.

    Computed HERE through the model's own tokenizer and the arena's own prompt
    template (``grm_e2e_session.harmony_turn``, which the lived build installs
    as ``prompt_template``), so the registration carries a number a receipt can
    be checked against.  THE ARMS DO NOT READ THIS.  Each arm recomputes the
    same count off the live arena (``arena.encode(arena._format_step_prompt(
    question))``) at serve time, and the results assembler compares the two —
    a mismatch is a reported failure, not a silent fallback.
    """
    from transformers import AutoTokenizer

    from scripts import grm_det1_2_gpu as det1_2
    from scripts.grm_e2e_session import harmony_turn
    from scripts.grm_rs1_read_strength_gpu import _probe_plan

    tokenizer = AutoTokenizer.from_pretrained(
        str(det1_2.MODEL_DIR), local_files_only=True)
    out: dict[str, Any] = {}
    for session_id in sessions_from_rs3():
        for probe in _probe_plan(session_id):
            question = str(probe["question"])
            prompt = harmony_turn(question, None)
            out[str(probe["probe_id"])] = {
                "session_id": str(session_id),
                "question": question,
                "prompt_template": (
                    "grm_e2e_session.harmony_turn(user_text, None) — the "
                    "callable the lived build installs as "
                    "ArenaCache.prompt_template, so this IS "
                    "_format_step_prompt's own output"),
                "prompt_sha256": _sha256_text(prompt),
                "question_ntok": int(len(
                    tokenizer.encode(prompt, add_special_tokens=False))),
                "question_ntok_derived_by": (
                    "len(tokenizer.encode(harmony_turn(question, None), "
                    "add_special_tokens=False)) — recomputed independently by "
                    "each arm off the live arena and cross-checked"),
            }
    return out


def rs3_rows_to_reproduce() -> dict[str, Any]:
    """RS3's C0 rows, carried so gate G2 has something to be bit-equal to.

    G2 says "if re-run, C0 bit-equal to RS3".  The rows it must match are
    RS3's own G3 table, filtered to C0 — served text, correctness, and every
    mass band RS3 reported.  Carried here at registration time so the gate is
    comparing against a frozen expectation rather than against whatever the
    RS3 artifact says when the gate happens to run.
    """
    table = _read(RS3_RESULTS)["G3_arms"]["table"]
    rows: dict[str, Any] = {}
    for row in table:
        if str(row["arm"]) != "C0":
            continue
        rows[f"{row['probe_id']}:{row['variant']}"] = {
            "probe_id": str(row["probe_id"]),
            "variant": str(row["variant"]),
            "served": str(row["served"]),
            "correct": bool(row["correct"]),
            "mounted_ids": [int(v) for v in row.get("mounted_ids", ())],
            "mounted_mass_full_layers": row["mounted_mass_full_layers"],
            "live_mass_full_layers": row["live_mass_full_layers"],
            "sink_mass_full_layers": row["sink_mass_full_layers"],
            "learned_sink_mass_full_layers": row[
                "learned_sink_mass_full_layers"],
            "mounted_mass_sliding_layers": row["mounted_mass_sliding_layers"],
            "live_mass_sliding_layers": row["live_mass_sliding_layers"],
            "answer_positions": row["answer_positions"],
        }
    if not rows:
        raise RS4Error(
            f"no C0 rows found in {RS3_RESULTS}; G2 would have nothing to "
            "reproduce against")
    return rows


def rs3_ceiling_pair() -> dict[str, Any]:
    """RS3's own 0.74-vs-0.41 framing, carried verbatim off its results.

    The whole point of RS4 is to restate this pair.  Reading it off RS3's
    artifact (rather than typing 0.74 and 0.41) means the restatement is
    anchored to the numbers RS3 actually published, and a reader can see both
    framings on one row.
    """
    table = _read(RS3_RESULTS)["G3_arms"]["table"]
    by_key = {(str(r["probe_id"]), str(r["variant"]), str(r["arm"])): r
              for r in table}
    pairs: dict[str, Any] = {}
    for (probe_id, variant, arm), row in sorted(by_key.items()):
        if arm != "C5":
            continue
        mounted = by_key.get((probe_id, variant, "C3l"))
        if mounted is None:
            continue
        pairs[f"{probe_id}:{variant}"] = {
            "probe_id": probe_id,
            "variant": variant,
            "rs3_C5_live_mass_full_layers": row["live_mass_full_layers"],
            "rs3_C5_mounted_mass_full_layers": row["mounted_mass_full_layers"],
            "rs3_C3l_mounted_mass_full_layers": mounted[
                "mounted_mass_full_layers"],
            "rs3_C3l_live_mass_full_layers": mounted["live_mass_full_layers"],
            "rs3_framing_residual": (
                float(row["live_mass_full_layers"])
                - float(mounted["mounted_mass_full_layers"])),
            "rs3_framing_note": (
                "RS3's own subtraction: the WHOLE live band of C5 minus the "
                "mount band of C3l. RS4 exists because the C5 live band also "
                "holds the question's and the answer's own rows."),
        }
    return pairs


# ======================================================================
# The registration payload
# ======================================================================
def build() -> dict[str, Any]:
    geometry = derive_geometry()
    questions = derive_question_lengths()
    implied = 266 - int(geometry["n_sink"]) - 211
    return {
        "schema": "grm.rs4.registration.v1",
        "program": "GRM",
        "phase": "RS4",
        "branch": "lc1-wip",
        "order": "orders/GRM_RS4_CEILING_REPARTITION.md",
        "question": (
            "RS3 read the live-band ceiling as 0.74 and the mount at 0.41, and "
            "called the residual 0.33. But the live band holds the QUESTION's "
            "own rows too. Split the live band into fed-text / question / "
            "answer rows and restate the residual as C5 FED-TEXT mass minus "
            "C3l MOUNT mass. Measurement only; no production change."),
        "immutability": (
            "This file is IMMUTABLE once written. scripts/grm_rs4_registration"
            ".py refuses to overwrite it. Any amendment goes in a SEPARATE "
            "file under artifacts/grm_rs4/ that cites this file's sha256."),
        "geometry_REGISTERED_BEFORE_ANY_GATE": geometry,
        "arm_order": list(ARMS),
        "arms_REGISTERED_BEFORE_ANY_GATE": {
            "C0": {
                "label": "both RS3 levers OFF — the reproduction row",
                "capture_pin": "off",
                "seat_near_live": False,
                "serving_path": "e2e._probe_ladder_chat (PRODUCTION ladder)",
                "live_band_holds": (
                    "prior conversation rows (the ladder does not clear the "
                    "boat between probes) + question + answer"),
                "requirement": (
                    "G2: C0 must reproduce RS3's C0 rows BIT-EQUAL — served "
                    "text, correctness and every mass band at float equality "
                    "— before any RS4 number counts."),
            },
            "C2": {
                "label": "capture OFF, seat ON",
                "capture_pin": "off",
                "seat_near_live": True,
                "serving_path": "ArenaCache._attempt over an explicit mount",
                "live_band_holds": (
                    "question + answer ONLY — _clear_boat empties the live "
                    "cache before the serve and nothing is fed"),
            },
            "C3l": {
                "label": "pin 'live' + seat ON (RS3's best mounted arm)",
                "capture_pin": "live",
                "seat_near_live": True,
                "serving_path": "ArenaCache._attempt over an explicit mount",
                "live_band_holds": "question + answer ONLY",
            },
            "C5": {
                "label": "the live ceiling — text fed live, nothing mounted",
                "capture_pin": "off",
                "seat_near_live": False,
                "serving_path": (
                    "_feed_live pushes the registered node turns through the "
                    "live cache, then ArenaCache._attempt with picks=[]"),
                "live_band_holds": "fed text + question + answer",
            },
        },
        "live_band_split_REGISTERED_BEFORE_ANY_GATE": {
            "sub_bands_in_physical_row_order": list(LIVE_SUBBANDS),
            "full_band_vocabulary": list(BAND_NAMES),
            "live0": "n_sink + cur_mount_n — _full_mass's own expression",
            "prior_live_rows": (
                "[live0, live0 + prior_ntok) where prior_ntok = arena.pos at "
                "the moment the arm's serve began MINUS fed_ntok. Zero on "
                "every arm whose serve is preceded by _clear_boat."),
            "fed_text_rows": (
                "[live0 + prior_ntok, live0 + prior_ntok + fed_ntok) where "
                "fed_ntok = sum of the ntok the arm's own _feed_live pushed, "
                "read off the feed receipt's live_segs_after_feed. Zero on "
                "every arm that feeds nothing (C0, C2, C3l)."),
            "question_rows": (
                "the next question_ntok rows, where question_ntok = "
                "len(arena.encode(arena._format_step_prompt(question))) — "
                "the model's own tokenizer over _attempt's own prompt string, "
                "recomputed by the arm and cross-checked against this file"),
            "answer_rows": (
                "the REMAINDER, [.., S). A remainder rather than a count, so "
                "the sub-bands partition the live band by construction; a "
                "negative width is a hard failure, not a rounding artifact."),
            "partition_law": (
                "physical_sink + mounted + prior_live + fed_text + question + "
                "answer + learned_sink sums to 1.0 within the same tolerance "
                "grm_demand._summarize_mass enforces (0.98..1.02), AND "
                "prior_live + fed_text + question + answer equals the "
                "live_mass the unmodified RS2/RS3 instrument reports for the "
                "same layer at the same answer position, to float equality."),
            "layer_types": (
                "reported for BOTH layer types the RS2 instrument partitions: "
                "full_attention and sliding_attention. The headline residual "
                "is taken on FULL layers, as every RS1/RS2/RS3 headline was."),
            "nothing_is_typed": (
                "S, n_sink, cur_mount_n and arena.pos are read off the live "
                "objects inside the observed call; question_ntok comes from "
                "the model tokenizer; fed_ntok comes from the feed receipt. "
                "No row index in this partition is a literal."),
        },
        "question_lengths_REGISTERED_BEFORE_ANY_GATE": questions,
        "rs3_c0_rows_RS4_MUST_REPRODUCE": rs3_rows_to_reproduce(),
        "rs3_ceiling_pair_AS_RS3_FRAMED_IT": rs3_ceiling_pair(),
        "derivation_crosscheck_against_rs3": {
            "why": (
                "The row split had to be shown to be arithmetic over numbers "
                "RS3 already measured BEFORE it was registered, or it would "
                "be a new assumption dressed as a derivation."),
            "probe_id": "sup_harbor_restatement",
            "rs3_receipt": (
                "artifacts/grm_rs3/"
                "grm_rs3_C5_correction_then_restatement_80736d57224a2e19.json"),
            "rs3_live_tokens_after_feed": 211,
            "rs3_sliding_S_min": 266,
            "n_sink": int(geometry["n_sink"]),
            "cur_mount_n_on_C5": 0,
            "implied_question_ntok": int(implied),
            "independently_tokenized_question_ntok": int(
                questions["sup_harbor_restatement"]["question_ntok"]),
            "agree": bool(
                implied
                == int(questions["sup_harbor_restatement"]["question_ntok"])),
        },
        "predictions_REGISTERED_BEFORE_ANY_GATE": {
            "source": "the order's 'Registered prediction (lead)' section",
            "P1_c5_question_row_mass_matches_mounted_live_band": {
                "claim": (
                    "C5 question-row mass ~= the mounted arms' live-band mass"),
                "centre": 0.29,
                "tolerance": 0.05,
                "measured_on": "FULL-attention layers, mean over refuser rows",
            },
            "P2_c5_fed_text_mass": {
                "claim": "C5 fed-text mass ~= 0.45",
                "centre": 0.45,
                "tolerance": 0.05,
                "tolerance_note": (
                    "the order states '~ 0.45' without a band; the same +/- "
                    "0.05 the order gives P1 is carried, and the raw number "
                    "is reported so a reader can apply any band."),
                "measured_on": "FULL-attention layers, mean over refuser rows",
            },
            "P3_residual": {
                "claim": (
                    "residual = C5 fed-text mass minus C3l mount mass <= 0.08"),
                "hit_if_at_most": 0.08,
                "mechanism_stands_if_at_least": 0.20,
                "between_note": (
                    "a residual strictly between 0.08 and 0.20 is neither the "
                    "hit nor the 'stands as RS3 left it' branch; it is "
                    "reported as INCONCLUSIVE against the registered "
                    "prediction rather than rounded into either."),
                "measured_on": "FULL-attention layers, mean over refuser rows",
            },
        },
        "gates_REGISTERED_BEFORE_ANY_GATE": {
            "G1": (
                "pytest sharded, pre-existing failing set unchanged vs RS3's "
                "FINAL shards, plus NEW CPU tests for the row-split "
                "arithmetic: bands sum to one, sub-bands sum to the "
                "unmodified live mass, boundaries derived not typed."),
            "G2": (
                "C0 bit-equal to RS3 on served text, correctness and every "
                "mass band, on every registered row."),
            "G3": (
                "the per probe x arm table, the residual, and the registered "
                "prediction hit/miss."),
        },
        "rs3_preexisting_failing_set": {
            "why": (
                "G1 compares against RS3's set, so RS3's set is registered "
                "here rather than re-read from a log after the fact."),
            "source_logs": ["logs/grm_rs3_pytest_FINAL_a.log .. _h.log"],
            "failed": [
                "tests/test_apamq_fc_ppl.py::"
                "test_summary_is_paired_and_contains_no_verdict",
                "tests/test_graft_quant_format.py::"
                "test_pack_node_save_load_npz_cycle_with_storage_bits",
                "tests/test_graft_quant_format.py::"
                "test_unpack_node_roundtrips_default_and_packed",
            ],
            "collection_errors": [
                "tests/test_gemma4_parity.py",
                "tests/test_gemma4_state.py",
                "tests/test_qwen35_apa.py",
                "tests/test_qwen35_parity.py",
            ],
            "collection_error_cause": (
                "CUDA out-of-memory at import/session setup — these four "
                "files build large non-GRM models and erred under RS3's own "
                "run. Recorded as a CONTENTION-SENSITIVE class: an RS4 run "
                "that reproduces them reproduces the set; an RS4 run in which "
                "they pass is reported as such and is not counted as a "
                "regression in either direction."),
        },
        "file_boundary": {
            "writable": ["scripts/grm_rs4_*.py", "tests/",
                         "artifacts/grm_rs4/", "logs/"],
            "read_only": ["core/"],
            "no_production_change": (
                "RS4 is MEASUREMENT ONLY. The extended observer lives in "
                "scripts/, wraps RS2's LayerTypeMassObserver from the outside, "
                "and changes no file under core/."),
        },
        "gpu_discipline": (
            "one arm per process under a self-lease on /tmp/forge-gpu.lock via "
            "grm_cmc1_gpu_arms.gpu_lease; single GPU; each run under 10 "
            "minutes; >= 30 s between GPU processes; the operator has ABSOLUTE "
            "right of way and no RS4 process ever signals, kills or interrupts "
            "a process it did not start."),
        "sources": _sources(),
    }


def _sources() -> dict[str, dict[str, Any]]:
    paths = (
        ROOT / "orders" / "GRM_RS4_CEILING_REPARTITION.md",
        RS3_REGISTRATION, RS3_RESULTS,
        RUNTIME_FRAME,
        ROOT / "core" / "graft_arena.py",
        ROOT / "core" / "grm_demand.py",
        ROOT / "core" / "grm_frame.py",
        ROOT / "core" / "gpt_oss20b_tc.py",
        ROOT / "scripts" / "grm_e2e_session.py",
        ROOT / "scripts" / "grm_rs1_read_strength_gpu.py",
        ROOT / "scripts" / "grm_rs2_mount_read_gpu.py",
        ROOT / "scripts" / "grm_rs3_capture_seat_gpu.py",
        SUP_FIXTURES / "correction_then_restatement.json",
        SUP_FIXTURES / "fresh_fact_controls.json",
        SUP_FIXTURES / "multi_hop_a_b_c.json",
    )
    out: dict[str, dict[str, Any]] = {}
    for path in paths:
        record = file_record(Path(path))
        out[record["path"]] = {
            "bytes": record["bytes"], "sha256": record["sha256"]}
    return out


def main(argv: list[str] | None = None) -> int:
    target = ARTIFACT_DIR / "registration.json"
    if target.exists():
        raise RS4Error(
            f"{target} already exists and the registration is IMMUTABLE; "
            "an amendment goes in a SEPARATE file citing its sha256")
    payload = build()
    cross = payload["derivation_crosscheck_against_rs3"]
    if not cross["agree"]:
        raise RS4Error(
            "the row-split derivation does not reproduce RS3's own C5 "
            f"receipt: implied question_ntok={cross['implied_question_ntok']} "
            f"but the tokenizer says "
            f"{cross['independently_tokenized_question_ntok']}")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    target.write_bytes(canonical_json_bytes(payload))
    record = file_record(target)
    print(f"registration={target}")
    print(json.dumps({
        "sha256": record["sha256"],
        "bytes": record["bytes"],
        "geometry": payload["geometry_REGISTERED_BEFORE_ANY_GATE"],
        "arms": payload["arm_order"],
        "sub_bands": list(LIVE_SUBBANDS),
        "crosscheck": cross,
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
