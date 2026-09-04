#!/usr/bin/env python3
"""GRM-RS3 — write the registration BEFORE any gate runs.

ORDER: ``orders/GRM_RS3_CAPTURE_PIN_SEAT_NEAR_LIVE.md``.

The order's Gates section says: "Register before any gate run
(``artifacts/grm_rs3/registration.json``): both pins' exact geometry (numbers
derived from the arena, not typed), the seating rule, arms, predictions.  **The
registration file is IMMUTABLE after it is written**: any amendment goes in a
SEPARATE file that cites the registration's sha256 (RS1's discipline, not
RS2's)."

EVERY GEOMETRY NUMBER IS DERIVED, never typed.  The chain is:

  * ``arena_width`` comes off the FROZEN runtime frame's ``resolved_flags`` —
    the same source the lived build reads, so a registration that disagreed
    with the build would be impossible rather than merely wrong;
  * ``n_sink`` is ``len(encode(HARMONY_SINK))``, computed HERE through the
    model's own tokenizer against the frozen sink string — not transcribed
    from RS2's diagnostic, so an edited sink or a re-tokenised vocabulary
    fails this script instead of silently steering a GPU arm;
  * ``live_shift`` is ``n_sink + arena_width``, which is
    ``ArenaCache.__init__``'s own expression, restated as a computation
    against the two derived numbers rather than as a literal;
  * the probes, their roles, their mount ids and their capture texts come from
    RS2's registration, which derived them from RS1's, which derived them from
    the EB1/P2C receipts.

THE PINS' GEOMETRY.  ``mount`` = ``n_sink``: the first ARENA seat, which is
where ``_attempt``'s bootstrap branch always lands a mount (the sink occupies
[0, n_sink) and the mount block follows immediately, RoPE'd at
``cos.slice(0, 0, graft_seats)``).  ``live`` = ``live_shift``: the first LIVE
seat, the geometry of text fed live immediately before a question, and the
geometry RS2's B1p harvested at.

THE SEATING RULE is registered as a computation too, not a number: the block is
filled TOP-DOWN so the plan head's LAST token sits at ``live_shift - 1``.

No probe id, no graft index, no threshold, no layer count and no arena width is
hard-coded.  What IS hard-coded: receipt paths, the registered VOCABULARY
strings, the ARM LABELS the order names, and the order's own PREDICTION
thresholds (0.50 mass, 2/3 flips, 0.10 movement, 0.35 residual) — which are the
order's registered numbers and must be carried verbatim, not re-derived.
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

from core.grm_frame import (  # noqa: E402
    CAPTURE_PIN_ENV, CAPTURE_PIN_LIVE, CAPTURE_PIN_MOUNT, CAPTURE_PIN_OFF,
    SEAT_NEAR_LIVE_ENV,
)
from scripts.grm_cmc1_mechanism import canonical_json_bytes  # noqa: E402
from scripts.grm_det1_common import file_record  # noqa: E402

ARTIFACT_DIR = ROOT / "artifacts" / "grm_rs3"
RS1_REGISTRATION = ROOT / "artifacts" / "grm_rs1" / "registration.json"
RS1_RESULTS = ROOT / "artifacts" / "grm_rs1" / "grm_rs1_results.json"
RS2_REGISTRATION = ROOT / "artifacts" / "grm_rs2" / "registration.json"
RS2_RESULTS = ROOT / "artifacts" / "grm_rs2" / "grm_rs2_results.json"
RS2_AMENDMENT = (ROOT / "artifacts" / "grm_rs2"
                 / "amendment_b1p_capture_position.json")
RUNTIME_FRAME = (
    ROOT / "artifacts" / "grm_det1" / "run_20260831T160525Z_2"
    / "runtime_frame_28b3196f8fb04a41.json")
SUP_FIXTURES = ROOT / "tests" / "fixtures" / "supersession_battery"

#: The order's G3 vocabulary, verbatim from its Gates section.
VOCABULARY = {
    "CLOSES": (
        ">= 0.50 mounted-band mass on the refusers, or >= 2/3 of the refusers "
        "flipped to correct with the correct value"),
    "MOVES": (
        "a measurable change in mounted-band mass, below both CLOSES lines"),
    "NOTHING": (
        "no measurable effect under these arms; the report names what was "
        "not tried"),
}

#: The arms the order names, in the order it names them.
ARM_ORDER = ("C0", "C1m", "C1l", "C2", "C3m", "C3l", "C5")
BASELINE_ARM = "C0"
CEILING_ARM = "C5"


class RS3Error(RuntimeError):
    """The registration could not be built from the receipts as ordered."""


def _read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


# ======================================================================
# Geometry — DERIVED, never typed
# ======================================================================
def derive_geometry() -> dict[str, Any]:
    """The arena's band geometry, computed rather than transcribed.

    ``arena_width`` off the frozen runtime frame; ``n_sink`` by encoding the
    frozen harmony sink with the model's own tokenizer; ``live_shift`` by
    ``ArenaCache.__init__``'s own expression over those two.  Nothing here is
    read back from an RS2 receipt, so a drift in the sink text or the
    tokenizer surfaces as a DIFFERENT registration rather than a silent
    mismatch between what is registered and what the arms run on.
    """
    from transformers import AutoTokenizer

    from scripts import grm_det1_2_gpu as det1_2
    from scripts.grm_e2e_session import HARMONY_SINK

    frame = _read(RUNTIME_FRAME)
    arena_width = int(frame["resolved_flags"]["arena_width"])
    tokenizer = AutoTokenizer.from_pretrained(
        str(det1_2.MODEL_DIR), local_files_only=True)
    sink_ids = tokenizer.encode(HARMONY_SINK, add_special_tokens=False)
    n_sink = len(sink_ids)
    live_shift = n_sink + arena_width
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
        "band_layout": (
            "seats [0, n_sink) SINK; [n_sink, n_sink + arena_width) ARENA "
            "(mounts occupy a PREFIX, the remainder is a positional hole); "
            "[live_shift, ...) LIVE"),
    }


def pin_registration(geometry: dict[str, Any]) -> dict[str, Any]:
    """Part 1's two pins, their geometry, and what OFF means."""
    n_sink = int(geometry["n_sink"])
    live_shift = int(geometry["live_shift"])
    return {
        "env": CAPTURE_PIN_ENV,
        "default": CAPTURE_PIN_OFF,
        "fail_direction": (
            "CLOSED TO OFF. Only the exact tokens 'mount' and 'live' select a "
            "pin; absent, empty, 'off', and ANY unknown token — INCLUDING a "
            "true-ish token like '1' — resolve OFF. The flag has two ON "
            "states with different geometries, so guessing which one an "
            "operator meant would be exactly the silent change the rule "
            "exists to prevent. An EXPLICIT unknown pin from a caller RAISES "
            "instead, because that is a caller bug, not an ambient typo."),
        "site": (
            "core/graft_arena.py::ArenaCache._capture_geometry, applied by "
            "ArenaCache.deposit around the harvest forward AND the routing "
            "key, and by ArenaCache.deposit_from_cache around its standalone "
            "_node_key fallback. Every harvest path in the stack reaches the "
            "model through those two calls, so pinning them pins all of "
            "them: feed's ephemeral branch, the P2C split children "
            "(_split_source_text_chunks), the consolidation note "
            "(_deposit_consolidation), the abstention deposit, and the "
            "repository installers."),
        "mechanism": (
            "GptOssAttentionTC.__call__ rotates QUERIES at "
            "cos.slice(0, position_offset + shift, L) with shift = "
            "self_attn.live_shift (falling back to graft_seats when None). "
            "The harvested KEYS are pre-RoPE and position-free — the "
            "relocatable-keys invariant is untouched — but the QUERIES decide "
            "each layer's attention output, which is the next layer's K/V "
            "input, so the capture-time query position propagates into every "
            "layer above 0. MEASURED (RS2 amendment): layer 0 identical, "
            "layers 1..23 all changed."),
        "pins": {
            CAPTURE_PIN_OFF: {
                "capture_shift": None,
                "geometry": (
                    "LEGACY, unpinned. live_shift is NOT TOUCHED — not even "
                    "written back to the value found — so the harvest runs on "
                    "whatever the caller's state left. On a virgin arena that "
                    "is None (the attention falls back to graft_seats == 0 "
                    "and the queries sit at [0, L)); after any served turn it "
                    "is live_shift. This dependence on whether a turn has "
                    "been served is the accident the pin removes."),
                "byte_identity": (
                    "the OFF path is the legacy path operand for operand; "
                    "test-pinned by "
                    "tests/test_grm_rs3_capture_pin_seat.py::"
                    "test_capture_pin_off_does_not_touch_live_shift and by "
                    "the C0 digest reproduction in gate G2"),
            },
            CAPTURE_PIN_MOUNT: {
                "capture_shift": n_sink,
                "derived_from": "arena.n_sink",
                "geometry": (
                    "queries at [n_sink, n_sink + L) — the geometry the graft "
                    "will be READ in when mounted at the band start, which is "
                    "where production's bootstrap branch always seats it. "
                    "'A graft is the text to the model' literally requires "
                    "the capture and the read to share a geometry, and with "
                    "the seating lever OFF this is that geometry."),
            },
            CAPTURE_PIN_LIVE: {
                "capture_shift": live_shift,
                "derived_from": "arena.live_shift (= n_sink + arena_width)",
                "geometry": (
                    "queries at [live_shift, live_shift + L) — the geometry "
                    "of text fed LIVE immediately before a question. This is "
                    "the geometry RS2's B1p harvested at, where the harbor "
                    "refuser went 0.184 -> 0.303 and flipped to correct."),
            },
        },
        "receipt_fields": ["capture_pin", "capture_shift",
                           "capture_shift_observed",
                           "capture_shift_derived_from"],
    }


def seating_registration(geometry: dict[str, Any]) -> dict[str, Any]:
    """Part 2's seating rule, registered as a COMPUTATION not a number."""
    n_sink = int(geometry["n_sink"])
    live_shift = int(geometry["live_shift"])
    return {
        "env": SEAT_NEAR_LIVE_ENV,
        "default": False,
        "fail_direction": (
            "CLOSED TO OFF. True only for an explicit true token; absent, "
            "false, and any unknown token resolve OFF."),
        "site": (
            "core/graft_arena.py::ArenaCache._rs3_seat_plan and "
            "_rs3_rotate_injection, called from _attempt's BOOTSTRAP branch. "
            "Under the spec (ephemeral) frame eb1_begin_turn sets "
            "self.caches = None at the start of EVERY turn, so the bootstrap "
            "branch IS the production seating path for every served turn."),
        "legacy_rule": (
            "the sink payload and the mount payloads are concatenated into "
            "ONE injection block that GptOssAttentionTC.__call__ RoPEs at "
            f"cos.slice(0, 0, graft_seats), so the mount lands at [{n_sink}, "
            f"{n_sink} + mount_ntok) and the PLAN HEAD (picks[0], the first "
            "seat grm_admission.plan_priority_fit fills) takes the band's "
            "SINK end — as far from the question as the band allows, with "
            "the filler between it and the live tokens"),
        "seat_near_live_rule": (
            "the band is filled from the TOP DOWN: the plan head is seated "
            "LAST in the block so its FINAL token sits at live_shift - 1 = "
            f"{live_shift - 1}, immediately below the first live token; the "
            "filler and the other mounts sit below it; the SINK is untouched "
            "at [0, n_sink), because moving it would be a second variable"),
        "seat_order_rule": (
            "seat_order = picks[1:] + picks[:1] — the plan head moved from "
            "the front of the block to the back, every other pick keeping "
            "its relative order"),
        "mount_pos0_rule": (
            "mount_pos0 = live_shift - mount_ntok (computed per attempt from "
            "the seated grafts' own ntok, never a constant)"),
        "seat_offset_plan_head_rule": (
            "seat_offset_plan_head = live_shift - plan_head_ntok"),
        "mechanism": (
            "PRE-ROTATION, composed with the layer's own. The layer will RoPE "
            "the injected block at [0, graft_seats) no matter what — that "
            "call site is NOT changed by this order. The mount occupies block "
            "rows [n_sink, n_sink + mount_ntok), so the layer rotates row "
            "n_sink + j by n_sink + j. To land it at mount_pos0 + j the "
            "payload is pre-rotated by delta = mount_pos0 - n_sink, and the "
            "layer's rotation composes with it. Only the ROPE-carrying "
            "payload key is touched; the VALUE payload passes through "
            "untouched and the PHYSICAL cache rows are unchanged (the mount "
            "stays a packed prefix immediately after the sink). This is the "
            "pre-RoPE relocatable-keys invariant (docs/GRM_Methodology.md "
            "section 4) doing exactly what it promises."),
        "constant_delta_law": (
            "THE PRE-ROTATION IS BY A CONSTANT DELTA PER ROW, not a span "
            "rotation. RoPE tables are indexed by ABSOLUTE position, so "
            "_rope_tensor(x, delta) rotates row j by delta + j, which "
            "composes with the layer's n_sink + j to n_sink + delta + 2j: a "
            "SHEAR that progressively de-phases the block's rows against one "
            "another rather than translating the block. MEASURED in pure "
            "numpy before any GPU run: constant-delta composition matches the "
            "direct rotation to 4.4e-16 in float64 and 9.8e-4 in the engine's "
            "fp16, while the span-style pre-rotation is off by 4.56 on the "
            "same input. Test-pinned both ways by "
            "tests/test_grm_rs3_capture_pin_seat.py::"
            "test_constant_delta_composition_is_the_relocation_and_span_"
            "style_is_not."),
        "fail_closed_when_unseatable": (
            "a block whose mount_ntok would place mount_pos0 below n_sink "
            "cannot be seated top-down without overrunning the sink. "
            "Production's own width law already forbids such a block (swap "
            "raises above self.width); this lever DECLINES to the legacy "
            "seating and records declined_reason rather than corrupting the "
            "sink."),
        "byte_identity": (
            "OFF returns the picks unchanged with delta 0, and "
            "_rs3_rotate_injection returns the CALLER'S OWN OBJECT (not a "
            "copy), so the legacy branch's injection block reaches "
            "_set_injection_host untouched; test-pinned by "
            "test_seat_plan_off_returns_the_picks_unchanged and "
            "test_rotate_injection_off_returns_the_callers_own_object"),
        "receipt_fields": ["seat_near_live", "seat_order", "seat_order_legacy",
                           "seat_offset_plan_head", "mount_pos0",
                           "delta_positions", "plan_head",
                           "plan_head_last_position", "seat_rule"],
        "sliding_window_note": (
            "with the plan head adjacent to the live band it is trivially "
            "inside the 128-row sliding window on sliding layers. The arms "
            "report sliding-layer mounted mass so RS2's ~15% "
            "sliding/full ratio has a comparison."),
    }


# ======================================================================
# Arms
# ======================================================================
def arms_registration() -> dict[str, Any]:
    """The seven arms the order names, each with what it varies."""
    return {
        "C0": {
            "label": "both levers OFF (must reproduce RS2 B0 bit-equal)",
            "capture_pin": CAPTURE_PIN_OFF,
            "seat_near_live": False,
            "varies": "nothing (this is the reproduction row)",
            "build": (
                "spec frame, fixes ON, demand OFF; the lived deposit-order "
                "build and the production probe ladder, unchanged — RS2's B0 "
                "path with both new flags at their defaults"),
            "requirement": (
                "C0 must reproduce RS2 B0 BIT-EQUAL before any other arm "
                "counts. Served text and mounted mass at float equality."),
        },
        "C1m": {
            "label": "capture pin 'mount', seat OFF",
            "capture_pin": CAPTURE_PIN_MOUNT,
            "seat_near_live": False,
            "varies": "the capture-time QUERY POSITION only",
            "build": (
                "the fixture grafts are RE-CAPTURED under the pin — same "
                "texts, through the same installer path with the flag ON — "
                "and the probe is served over those grafts"),
        },
        "C1l": {
            "label": "capture pin 'live', seat OFF",
            "capture_pin": CAPTURE_PIN_LIVE,
            "seat_near_live": False,
            "varies": "the capture-time QUERY POSITION only",
            "build": "as C1m, at the live geometry (RS2 B1p's geometry)",
        },
        "C2": {
            "label": "capture OFF, seat ON",
            "capture_pin": CAPTURE_PIN_OFF,
            "seat_near_live": True,
            "varies": "the mount block's BAND POSITION and ORDER only",
            "build": (
                "the INSTALLED grafts, unchanged, seated top-down through "
                "PRODUCTION's own injection path — not RS2's probe hack"),
        },
        "C3m": {
            "label": "pin 'mount' + seat ON",
            "capture_pin": CAPTURE_PIN_MOUNT,
            "seat_near_live": True,
            "varies": "both levers",
            "build": "C1m's re-capture, seated as C2 seats",
        },
        "C3l": {
            "label": "pin 'live' + seat ON",
            "capture_pin": CAPTURE_PIN_LIVE,
            "seat_near_live": True,
            "varies": "both levers",
            "build": "C1l's re-capture, seated as C2 seats",
        },
        "C5": {
            "label": "the live ceiling (RS2 B5) as the reference row",
            "capture_pin": CAPTURE_PIN_OFF,
            "seat_near_live": False,
            "varies": "the payload's BAND: live instead of mounted",
            "build": (
                "RS1's A1 / RS2's B5 seam, unchanged and imported: the "
                "registered live node text is pushed through the live cache "
                "and the question asked immediately after"),
        },
    }


def predictions_registration() -> list[dict[str, Any]]:
    """The lead's registered predictions, verbatim from the order's Part 3."""
    return [
        {
            "id": "P1",
            "source": "lead, stated in the order",
            "claim": (
                "C3l or C3m reaches mounted-band mass >= 0.50 on the refusers "
                "and flips >= 2/3 with correct values"),
            "decides": "whether the PAIR CLOSES",
            "mass_line": 0.50,
            "flip_line": "2/3",
        },
        {
            "id": "P2",
            "source": "lead, stated in the order",
            "claim": "C1 alone ~= RS2 B1p (harbor only)",
            "decides": "whether the capture pin ALONE closes",
        },
        {
            "id": "P3",
            "source": "lead, stated in the order",
            "claim": (
                "C2 alone moves mass >= 0.10 without breaking generation "
                "(the RS2 B3a movement, now with production seating)"),
            "decides": "whether the seating lever ALONE moves",
            "movement_line": 0.10,
        },
        {
            "id": "P4",
            "source": "lead, stated in the order",
            "claim": (
                "if C3 stays <= 0.35 on the refusers, the residual is "
                "sink/RoPE distance itself and this order reports THAT as its "
                "result"),
            "decides": "the order's fallback finding",
            "residual_line": 0.35,
        },
        {
            "id": "P5",
            "source": "lead, stated in the order (Part 4)",
            "claim": (
                "sup >= 7/9 (the three refusers recover, nothing correct "
                "becomes wrong); census 9/10 with 0 regressions (t33's "
                "admission miss is not this order's); the 4 distance-36-60 "
                "probes from EB1 G5 stay correct"),
            "decides": "Part 4, only if Part 3's best arm meets the gate",
        },
    ]


def part4_gate() -> dict[str, Any]:
    """The order's own trigger for running (or skipping) Part 4."""
    return {
        "rule": (
            "Part 4 runs ONLY if Part 3's best arm meets the mass >= 0.50 OR "
            "flips >= 2/3 line; otherwise SKIP and say so, naming the number "
            "that triggered the skip."),
        "mass_line": 0.50,
        "flip_line": "2/3 of the refusers, with the correct value",
        "measured_on": "the refusers, mounted-band mass on FULL layers",
    }


# ======================================================================
# Build
# ======================================================================
def build() -> dict[str, Any]:
    rs2_reg = _read(RS2_REGISTRATION)
    rs2_res = _read(RS2_RESULTS)
    geometry = derive_geometry()

    probes = rs2_reg["probes_REGISTERED_BEFORE_ANY_GATE"]
    refusers = list(rs2_reg["refusers"])
    controls = list(rs2_reg["controls"])

    # THE ROWS C0 MUST REPRODUCE, lifted from RS2's own G3 table so the
    # comparison is against RS2's receipts rather than a transcription.
    b0_rows: dict[str, Any] = {}
    b5_rows: dict[str, Any] = {}
    for row in rs2_res["G3_arms"]["table"]:
        key = f"{row['probe_id']}:{row.get('variant', 'registered')}"
        if str(row["arm"]) == "B0":
            b0_rows[key] = {
                "probe_id": str(row["probe_id"]),
                "variant": str(row.get("variant", "registered")),
                "served": str(row["served"]),
                "correct": bool(row["correct"]),
                "mounted_mass_full_layers": float(
                    row["mounted_mass_full_layers"]),
                "mounted_mass_sliding_layers": float(
                    row["mounted_mass_sliding_layers"]),
                "live_mass_full_layers": float(row["live_mass_full_layers"]),
                "sink_mass_full_layers": float(row["sink_mass_full_layers"]),
                "mounted_ids": [int(v) for v in row["mounted_ids"]],
            }
        elif str(row["arm"]) == "B5":
            b5_rows[key] = {
                "probe_id": str(row["probe_id"]),
                "variant": str(row.get("variant", "registered")),
                "served": str(row["served"]),
                "correct": bool(row["correct"]),
                "live_mass_full_layers": float(row["live_mass_full_layers"]),
                "live_mass_sliding_layers": float(
                    row["live_mass_sliding_layers"]),
            }
    if not b0_rows:
        raise RS3Error("no B0 rows found in the RS2 results table")
    if not b5_rows:
        raise RS3Error("no B5 rows found in the RS2 results table")

    return {
        "schema": "grm.rs3.registration.v1",
        "program": "GRM",
        "phase": "RS3",
        "order": "orders/GRM_RS3_CAPTURE_PIN_SEAT_NEAR_LIVE.md",
        "branch": "lc1-wip",
        "question": (
            "RS1 measured the gap (mounted ~0.18 vs live ~0.74) and RS2 found "
            "two levers that move it: the capture geometry is unpinned, and "
            "the mount's band position matters. RS3 builds BOTH as production "
            "seams behind flags (default OFF), measures them alone and "
            "together against the live-band ceiling, and — if the pair "
            "reproduces the ceiling — runs both lived batteries with the pair "
            "ON. THE FLIP STAYS DAVID'S."),
        "immutability": (
            "THIS FILE IS IMMUTABLE ONCE WRITTEN. It is written before any "
            "gate runs and is never edited afterwards. Any amendment goes in "
            "a SEPARATE file in this directory that cites this file's "
            "sha256 (RS1's discipline, which the order names explicitly)."),
        "geometry_REGISTERED_BEFORE_ANY_GATE": geometry,
        "capture_pin_REGISTERED_BEFORE_ANY_GATE": pin_registration(geometry),
        "seating_REGISTERED_BEFORE_ANY_GATE": seating_registration(geometry),
        "arms_REGISTERED_BEFORE_ANY_GATE": arms_registration(),
        "arm_order": list(ARM_ORDER),
        "baseline_arm": BASELINE_ARM,
        "ceiling_arm": CEILING_ARM,
        "predictions_REGISTERED_BEFORE_ANY_GATE": predictions_registration(),
        "vocabulary_REGISTERED_BEFORE_ANY_GATE": VOCABULARY,
        "verdict_rule_REGISTERED_BEFORE_ANY_GATE": (
            "ONE verdict per LEVER (capture pin; seating) and ONE for the "
            "PAIR, in the registered vocabulary. CLOSES if the arm reaches "
            ">= 0.50 mounted-band mass on the refusers (mean over the "
            "refusers, FULL-attention layers) OR flips >= 2/3 of the refusers "
            "to correct with the correct value. MOVES if the change in "
            "mounted-band mass is measurable but below both CLOSES lines. "
            "NOTHING if there is no measurable effect. A control that BREAKS "
            "(was correct at C0, wrong at the arm) is reported on the arm's "
            "row and disqualifies a CLOSES verdict for that arm."),
        "part4_gate_REGISTERED_BEFORE_ANY_GATE": part4_gate(),
        "probes_REGISTERED_BEFORE_ANY_GATE": probes,
        "refusers": refusers,
        "controls": controls,
        "solace_fact_variant_REGISTERED_BEFORE_ANY_GATE": rs2_reg[
            "solace_fact_variant_REGISTERED_BEFORE_ANY_GATE"],
        "rs2_B0_rows_C0_must_reproduce": b0_rows,
        "rs2_B5_ceiling_rows": b5_rows,
        "flags_default_off": {
            CAPTURE_PIN_ENV: CAPTURE_PIN_OFF,
            SEAT_NEAR_LIVE_ENV: False,
            "the_flip_is_davids": (
                "both flags ship DEFAULT OFF with test-pinned byte-identical "
                "OFF paths. This order measures and reports; turning either "
                "one on in production is David's decision, made against the "
                "Part 4 receipts."),
        },
        "gpu_discipline": {
            "lease": (
                "/tmp/forge-gpu.lock via scripts.grm_cmc1_gpu_arms.gpu_lease"),
            "single_gpu": True,
            "max_wall_per_run_s": "<= 600 (house rule), lease cap below it",
            "inter_run_gap_s": ">= 30, inside the wrapper",
            "operator_right_of_way": (
                "absolute; the harness WAITS on the flock and never signals, "
                "kills or interrupts any process it did not start"),
            "frozen_run_tree": "never written",
        },
        "file_boundary": {
            "modified": [
                "core/graft_arena.py", "core/grm_frame.py",
                "scripts/grm_rs3_*.py", "tests/", "artifacts/grm_rs3/",
                "logs/",
            ],
            "engine_untouched": (
                "core/gpt_oss20b_tc.py is NOT modified by this order. The "
                "seating lever composes with the engine's existing RoPE slice "
                "rather than changing it, so no attention-math change was "
                "needed and none was made."),
        },
        "sources": _sources(),
    }


def _sources() -> dict[str, dict[str, Any]]:
    paths = (
        ROOT / "orders" / "GRM_RS3_CAPTURE_PIN_SEAT_NEAR_LIVE.md",
        RS1_REGISTRATION, RS1_RESULTS,
        RS2_REGISTRATION, RS2_RESULTS, RS2_AMENDMENT,
        RUNTIME_FRAME,
        ROOT / "core" / "graft_arena.py",
        ROOT / "core" / "grm_frame.py",
        ROOT / "core" / "gpt_oss20b_tc.py",
        ROOT / "core" / "grm_demand.py",
        ROOT / "core" / "kv_graft.py",
        ROOT / "scripts" / "grm_det1_3_gpu.py",
        ROOT / "scripts" / "grm_e2e_session.py",
        ROOT / "scripts" / "grm_rs1_read_strength_gpu.py",
        ROOT / "scripts" / "grm_rs2_mount_read_gpu.py",
        ROOT / "tests" / "test_grm_rs3_capture_pin_seat.py",
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
        # IMMUTABLE. Re-running must never silently rewrite a registration a
        # gate has already been measured against.
        raise RS3Error(
            f"{target} already exists and the registration is IMMUTABLE; "
            "an amendment goes in a SEPARATE file citing its sha256")
    payload = build()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    target.write_bytes(canonical_json_bytes(payload))
    record = file_record(target)
    print(f"registration={target}")
    print(json.dumps({
        "sha256": record["sha256"],
        "bytes": record["bytes"],
        "geometry": payload["geometry_REGISTERED_BEFORE_ANY_GATE"],
        "arms": payload["arm_order"],
        "refusers": payload["refusers"],
        "controls": payload["controls"],
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
