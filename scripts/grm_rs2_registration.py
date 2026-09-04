#!/usr/bin/env python3
"""GRM-RS2 — write the registration BEFORE any gate runs.

ORDER: ``orders/GRM_RS2_MOUNT_READ_DEFICIT.md``.

The order's Gates section says: "Register before any gate run
(``artifacts/grm_rs2/registration.json``): arms, exact node ids and capture
texts per probe (derived from receipts, not typed), predictions above, and the
sliding-layer partition definition.  No numeric constants."

EVERY ID AND EVERY CAPTURE TEXT IS DERIVED, never typed.  The chain is:

  * probe ids, roles, ``a0_mounted_ids``, ``live_ids``/``live_node_ids`` and the
    fixture ``node_to_idx`` map come from ``artifacts/grm_rs1/registration.json``
    (RS1's own registration, itself derived from the EB1 and P2C receipts);
  * A0's served text and mounted mass come from ``artifacts/grm_rs1/
    grm_rs1_results.json`` (the G2/G3 tables) — these are the numbers B0 must
    reproduce and the numbers the B0->B5 gap is measured against;
  * the CAPTURE TEXT for every node is rebuilt from the fixture file through
    the SAME two frozen functions the lived installer used
    (``grm_det1_3_gpu._split_fixture_turn`` then ``grm_e2e_session.harmony_turn``),
    so the registered text is the text the installed graft was captured from
    rather than a transcription of it.  Its sha256 is registered next to it.

"No numeric constants" is honoured literally: this module hard-codes no probe
id, no graft index, no threshold, no layer count and no arena width.  It
hard-codes only RECEIPT PATHS, the registered VOCABULARY strings, and the ARM
LABELS the order itself names.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.grm_cmc1_mechanism import canonical_json_bytes  # noqa: E402
from scripts.grm_det1_common import file_record  # noqa: E402

ARTIFACT_DIR = ROOT / "artifacts" / "grm_rs2"
RS1_REGISTRATION = ROOT / "artifacts" / "grm_rs1" / "registration.json"
RS1_RESULTS = ROOT / "artifacts" / "grm_rs1" / "grm_rs1_results.json"
RS1_AMENDMENT = ROOT / "artifacts" / "grm_rs1" / "amendment_a2_width.json"
SUP_FIXTURES = ROOT / "tests" / "fixtures" / "supersession_battery"

#: The order's G3 vocabulary, verbatim from the order's Gates section.
VOCABULARY = {
    "SUPPORTED": (
        "the arm closes >= half the B0->B5 mounted-mass gap on the probe, or "
        "flips >= 2/3 of the refusers to correct"),
    "PARTIAL": "measurable but < half the B0->B5 mounted-mass gap",
    "NOT DETECTED": (
        "no measurable effect under these arms; the report names what was "
        "not tried"),
}

#: The reproduced baseline row and the ceiling row every table is read against.
BASELINE_ARM = "B0"
CEILING_ARM = "B5"


class RS2Error(RuntimeError):
    """The registration could not be built from the receipts as ordered."""


def _read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


# ======================================================================
# Capture-text derivation (pure — CPU tested)
# ======================================================================
def capture_text_for_node(node_text: str) -> str:
    """The text the LIVED installer captured a node's graft from.

    ``grm_det1_3_gpu._install_lived_nodes`` splits the fixture's
    ``"User: ...\\nAssistant: ..."`` string with ``_split_fixture_turn`` and
    wraps it with ``grm_e2e_session.harmony_turn`` before calling
    ``arena.feed(turn_text, deposit=True)``.  Both functions are imported from
    the read-only lived modules, so this returns the installer's own string
    rather than a re-derivation of it.
    """
    from scripts.grm_det1_3_gpu import _split_fixture_turn
    from scripts.grm_e2e_session import harmony_turn

    user, assistant = _split_fixture_turn(str(node_text))
    return str(harmony_turn(user, assistant))


def question_scaffold_text(node_text: str) -> str:
    """The node's user text wrapped as an UNANSWERED harmony turn.

    ``harmony_turn(user, None)`` is the exact prompt shape a QUESTION is fed in
    (``ArenaCache._format_step_prompt`` uses the same template), so this is
    "the node presented the way the query is presented".  B2 needs a capture
    text that differs from the installed one along the SCAFFOLD axis, and this
    is the only scaffold in the repository that does.
    """
    from scripts.grm_det1_3_gpu import _split_fixture_turn
    from scripts.grm_e2e_session import harmony_turn

    user, _assistant = _split_fixture_turn(str(node_text))
    return str(harmony_turn(user, None))


def fixture_nodes(session_id: str) -> dict[str, str]:
    """``node_id -> raw fixture text`` for one sup fixture."""
    payload = _read(SUP_FIXTURES / f"{session_id}.json")
    return {str(node["node_id"]): str(node["text"])
            for node in payload["nodes"]}


# ======================================================================
# The arms
# ======================================================================
def arms() -> dict[str, dict[str, Any]]:
    """The order's arms, with the seam each one needs named.

    The B1/B1b/B2 block carries a STRUCTURAL FACT read off the production
    source before any gate ran.  It is registered here so the arms' results are
    read against it rather than against an assumption:

      Under the SPEC (ephemeral) frame ``ArenaCache.feed(text, deposit=True)``
      takes its ``if self.ephemeral:`` branch and calls ``self.deposit(text)``.
      ``deposit`` harvests through ``kv_graft.harvest_kv(model, encode(text))``
      — a fresh forward over the TEXT ALONE, no sink, no predecessors, no
      injection — and the lived installer's text is already
      ``harmony_turn(user, assistant)``.

    So the A0/B0 grafts in this battery were ALREADY captured by exactly the
    call B1 asks for, from exactly the harmony-wrapped text B2 asks for.  The
    arms are still RUN — a registered arm is not skipped on a reading of the
    source — and each carries a PAYLOAD-IDENTITY check whose result is the
    finding.  B1b is the arm that actually varies the capture CONTEXT.
    """
    return {
        "B0": {
            "label": "baseline = RS1 A0 (reproduce; 5/5 required)",
            "build": (
                "spec frame, fixes ON, demand OFF; the lived deposit-order "
                "build and the production probe ladder, unchanged"),
            "varies": "nothing (this is the reproduction row)",
            "prediction": (
                "reproduces the RS1 A0 rows: served text and mounted mass "
                "within float equality"),
            "prediction_source":
                "RS1's own G2/G3 receipts (not a lead prediction)",
        },
        "B1": {
            "label": (
                "H-CAPTURE: re-capture the A0 node with deposit(text) (fresh "
                "context) and mount that graft in place of the installed one"),
            "build": (
                "B0's repository; arena.deposit(capture_text) appends a FRESH "
                "graft and the probe is served over that new id in place of "
                "the registered a0_mounted_ids"),
            "varies":
                "which graft object is mounted (fresh re-capture vs installed)",
            "structural_note": (
                "the installed graft was itself produced by deposit(text) via "
                "the ephemeral feed branch, so this arm is predicted to be a "
                "PAYLOAD-IDENTICAL re-capture; the receipt carries a sha256 of "
                "both payloads and the arm reports the comparison"),
            "prediction": (
                "lead: raises mounted mass on the refusers to >= 0.35 and "
                "flips >= 2/3 to correct"),
            "prediction_source": "lead, stated in the order",
        },
        "B1p": {
            "label": (
                "H-CAPTURE, capture POSITION varied: re-capture the A0 node "
                "with deposit(text) at the READ geometry — the query position "
                "a served turn leaves — instead of the lived installer's"),
            "build": (
                "B0's repository; every layer's self_attn.live_shift is pinned "
                "to arena.live_shift for the duration of the harvest forward "
                "only, arena.deposit(capture_text) appends a FRESH graft, the "
                "shift is restored, and the probe is served over the new id"),
            "varies": (
                "the capture-time QUERY POSITION: [live_shift, live_shift+L) "
                "instead of [0, L)"),
            "seam": (
                "PRODUCTION LACKS THIS, and the lack is the point. "
                "ArenaCache.deposit does not pin the capture-time query "
                "position at all: GptOssAttentionTC.__call__ reads "
                "self_attn.live_shift, the lived installer never sets it (so "
                "it is None and the fallback graft_seats == 0 applies), and "
                "any served turn leaves it at n_sink + arena_width. So the "
                "SAME deposit call produces a DIFFERENT graft depending on "
                "whether a turn has been served. This arm pins it explicitly "
                "and reports the value in force. Implemented in "
                "scripts/grm_rs2_mount_read_gpu.py"),
            "prediction": (
                "REGISTERED BEFORE THIS ARM RAN, after B1 on "
                "correction_then_restatement returned a NON-identical payload "
                "and the cause was isolated to live_shift on a separate "
                "diagnostic: if the capture/read position mismatch is the "
                "mechanism, capturing at the read geometry raises mounted mass "
                "and flips refusers"),
            "prediction_source": (
                "seat, derived from an isolated diagnostic — NOT a lead "
                "prediction, and scored separately from the lead's"),
        },
        "B1b": {
            "label": (
                "H-CAPTURE, context varied: capture with the node's "
                "predecessors REMOVED but the arena's harmony sink PRESENT in "
                "the harvest forward"),
            "build": (
                "B0's repository; a probe-script harvest runs the forward over "
                "encode(HARMONY_SINK) + encode(capture_text) and stores ONLY "
                "the capture_text rows, then mounts that graft in place of the "
                "registered a0_mounted_ids"),
            "varies": "the capture-time CONTEXT: [sink | text] instead of [text]",
            "seam": (
                "PRODUCTION LACKS THIS. ArenaCache.deposit harvests the text "
                "alone; no call harvests a text with a prefix and keeps only "
                "the text's rows. Implemented in "
                "scripts/grm_rs2_mount_read_gpu.py"),
            "prediction": (
                "lead (H-CAPTURE as written for B1): the installed grafts' "
                "keys were shaped by predecessors the query does not share"),
            "prediction_source": "lead, stated in the order",
        },
        "B2": {
            "label": (
                "H-SCAFFOLD: capture the node wrapped exactly as B5 fed it "
                "live (harmony user/assistant turn), mount that"),
            "build": (
                "B0's repository; arena.deposit(scaffold_text) appends a FRESH "
                "graft and the probe is served over it"),
            "varies": "the capture TEXT's scaffold",
            "structural_note": (
                "the installed graft's capture text ALREADY IS "
                "harmony_turn(user, assistant) — the same string B5 feeds live "
                "— so the order's literal B2 is byte-identical to B1. The arm "
                "is therefore run on the one scaffold in the repository that "
                "DOES differ: harmony_turn(user, None), the UNANSWERED "
                "question-shaped turn. Both texts and both sha256s are on the "
                "receipt."),
            "prediction": "lead: small additional gain over B1",
            "prediction_source": "lead, stated in the order",
        },
        "B3a": {
            "label": (
                "H-BAND-POSITION, far end: seat the B0 graft at the arena "
                "positions adjacent to live_shift (nearest the question)"),
            "build": (
                "B0's mount set, injected with the mount block RoPE'd at "
                "positions [live_shift - mount_ntok, live_shift) instead of "
                "[n_sink, n_sink + mount_ntok); physical cache rows unchanged"),
            "varies": "the mount's ROPE POSITIONS inside the arena band",
            "seam": (
                "PRODUCTION LACKS THIS. At bootstrap the sink and the mounts "
                "are concatenated into ONE injection block that "
                "GptOssAttentionTC.__call__ RoPEs at cos.slice(0, 0, "
                "graft_seats) — positions 0.. — so a mount always lands "
                "immediately after the sink. Implemented in "
                "scripts/grm_rs2_mount_read_gpu.py"),
            "prediction": "lead: modest effect; not the main cause",
            "prediction_source": "lead, stated in the order",
        },
        "B3b": {
            "label": (
                "H-BAND-POSITION, near end: seat the B0 graft at the arena "
                "positions adjacent to the sink"),
            "build": (
                "B0's mount set with the mount block RoPE'd at [n_sink, "
                "n_sink + mount_ntok) — the PRODUCTION seating — run through "
                "the SAME probe-script injection path as B3a so the two arms "
                "differ only in the offset"),
            "varies": "the mount's ROPE POSITIONS inside the arena band",
            "control_for": (
                "B3a. Running production's own offset through the arm's own "
                "injection path proves the path itself changes nothing, so a "
                "B3a effect is the offset and not the seam."),
            "prediction": "lead: modest effect; not the main cause",
            "prediction_source": "lead, stated in the order",
        },
        "B4": {
            "label": (
                "H-SLIDING: per-layer mounted mass on SLIDING vs FULL "
                "attention layers, for B0 and for B5"),
            "build": (
                "no serve of its own: the extended instrument records both "
                "layer types on EVERY arm, and B4 is the READ of the B0 and B5 "
                "rows split by layer type"),
            "varies": "nothing served; it is a measurement partition",
            "seam": (
                "PRODUCTION LACKS THIS. core.grm_demand.DemandObserver wraps "
                "sliding_sink_attention_tc solely to SUPPRESS it (its "
                "sliding_wrapper sets _suppress_sink), and RS1's "
                "LayerMassObserver inherits that suppression. The RS2 "
                "instrument records the sliding layers instead. Implemented in "
                "scripts/grm_rs2_mount_read_gpu.py"),
            "prediction": (
                "lead: the gap exists on full layers too (RS1 measured 0.18 vs "
                "0.74 on full layers alone), so sliding is a contributor, not "
                "the whole story; quantify the sliding-layer contribution"),
            "prediction_source": "lead, stated in the order",
        },
        "B5": {
            "label": (
                "the ceiling: the same text fed LIVE immediately before the "
                "question (RS1 A1), the reference row in every table"),
            "build": (
                "RS1's A1 seam, unchanged: the registered live node text is "
                "pushed through the live cache with arena.ephemeral pinned "
                "False for the FEED ONLY, deposit=False, and the serve runs "
                "with an EMPTY mount set through arena._attempt"),
            "varies": "the payload's BAND: live instead of mounted",
            "prediction": (
                "the ceiling row; RS1 measured live-band mass ~0.74 and "
                "answers on 4 of the 5 probes"),
            "prediction_source": "RS1's own A1 receipts",
        },
    }


def predictions() -> list[dict[str, Any]]:
    """The lead's predictions, carried verbatim, scored later hit/miss."""
    return [
        {
            "hypothesis": "H-CAPTURE",
            "arms": ["B1", "B1b"],
            "prediction": (
                "B1 raises mounted mass on the refusers to >= 0.35 and flips "
                ">= 2/3 to correct; the installed grafts' keys were shaped by "
                "predecessors the query does not share"),
            "source": "lead, stated in the order",
        },
        {
            "hypothesis": "H-CAPTURE",
            "arms": ["B1p"],
            "prediction": (
                "capturing at the READ geometry (the query position a served "
                "turn leaves) rather than the lived installer's position 0 "
                "raises mounted mass and flips refusers"),
            "source": (
                "SEAT, added by amendment_b1p_capture_position.json after B1's "
                "payload-identity prediction missed and the cause was isolated "
                "— NOT a lead prediction, and scored separately from the "
                "lead's"),
        },
        {
            "hypothesis": "H-SCAFFOLD",
            "arms": ["B2"],
            "prediction": "small additional gain over B1",
            "source": "lead, stated in the order",
        },
        {
            "hypothesis": "H-BAND-POSITION",
            "arms": ["B3a", "B3b"],
            "prediction": "modest effect; not the main cause",
            "source": "lead, stated in the order",
        },
        {
            "hypothesis": "H-SLIDING",
            "arms": ["B4"],
            "prediction": (
                "the gap exists on full layers too, so sliding is a "
                "contributor, not the whole story"),
            "source": "lead, stated in the order",
        },
    ]


# ======================================================================
# The sliding-layer partition definition
# ======================================================================
def partition_definition() -> dict[str, Any]:
    """EXACTLY how the sliding-layer partition is computed, and what "mounted
    band" means when the window cannot reach it.

    Registered before any gate because the order requires it, and because a
    partition defined after the numbers are seen is not a measurement.
    """
    return {
        "cache_row_layout": (
            "one KV cache row sequence per layer, identical across layer "
            "types: rows [0, n_sink) SINK; rows [n_sink, n_sink + "
            "cur_mount_n) MOUNTED BAND; rows [n_sink + cur_mount_n, S) LIVE "
            "(under this harness that carries the question prompt, plus, on "
            "B5, the fed node text). One extra softmax column past S is the "
            "per-head LEARNED SINK logit."),
        "full_layer_partition": (
            "UNCHANGED from RS1 and from production: "
            "core.grm_demand.DemandObserver._full_mass, operand for operand. "
            "The last query row's scores over all S key rows plus the appended "
            "sink logit, softmaxed, then summed into the four bands above and "
            "averaged over heads."),
        "sliding_layer_partition": (
            "THE SAME FOUR-BAND SUM over THE SAME cache-row boundaries, "
            "computed on the sliding layer's OWN attention call. The sliding "
            "kernel (core.gpt_oss20b_tc.sliding_sink_attention_tc) does not "
            "score all S rows: for the last query row it slices keys to "
            "[k0, k1) with k0 = max(0, q_abs - window + 1) and q_abs = S - 1, "
            "then applies the causal+window mask inside that slice. The RS2 "
            "instrument reproduces that same slice and mask for the LAST query "
            "row only, softmaxes over [scores | sink logit] exactly as the "
            "kernel does, and sums the resulting probabilities into the four "
            "bands using the SAME absolute row boundaries — rows outside "
            "[k0, k1) contribute exactly zero because the kernel never scores "
            "them."),
        "window_indexing_law": (
            "q_abs and k_abs in the sliding kernel are CACHE ROW INDICES, not "
            "RoPE positions. The window therefore reaches back "
            "`sliding_window` ROWS from the query row, across the "
            "sink/mount/live boundaries, and is unaffected by the arena's "
            "positional hole. Whether it reaches the mounted band is a "
            "MEASURED property of S, cur_mount_n and n_sink at each readout "
            "position; the receipt records k0, k1, S and the boundaries for "
            "every captured sliding row so the reach is auditable rather than "
            "assumed."),
        "mounted_band_when_unreachable": (
            "If the window's [k0, k1) does not intersect [n_sink, n_sink + "
            "cur_mount_n), the sliding layer's mounted_mass is EXACTLY ZERO "
            "and that zero is REPORTED AS ZERO — never as missing, never "
            "dropped from the mean. A zero meaning 'structurally cannot see "
            "the band' is recorded alongside a boolean `window_reaches_mount` "
            "and the intersection width, so a reader can tell a structural "
            "zero from a measured near-zero. The sliding partition still sums "
            "to one, because the mass the kernel would have put on unreachable "
            "rows is simply never in the softmax."),
        "partition_law": (
            "the four masses sum to one per layer and per token on BOTH layer "
            "types; the instrument raises if any captured row leaves "
            "[0.98, 1.02] — the same tolerance core.grm_demand enforces"),
        "layer_type_source": (
            "the model config's layer_types, read through "
            "GptOssAttentionTC.layer_type on each layer; no layer index is "
            "typed into the harness"),
        "full_layer_numbers_are_the_production_ones": (
            "the full-attention rows come from the wrapped production "
            "DemandObserver, unchanged, so B0's full-layer mounted mass is "
            "directly comparable to RS1's A0 number and to any production "
            "demand receipt"),
        "reported_statistic": (
            "per arm and per probe: the mean over the answer positions the "
            "race's flush-drop convention keeps, of the per-layer-type mean of "
            "each band"),
    }


# ======================================================================
# Probe registration
# ======================================================================
def probe_registration() -> dict[str, dict[str, Any]]:
    """Per-probe ids and capture texts, derived from RS1's receipts."""
    rs1_reg = _read(RS1_REGISTRATION)
    rs1_results = _read(RS1_RESULTS)
    rows = dict(rs1_reg["probes_REGISTERED_BEFORE_ANY_GATE"])

    a0_by_probe: dict[str, dict[str, Any]] = {}
    a1_by_probe: dict[str, dict[str, Any]] = {}
    for row in rs1_results["G3_arms"]["table"]:
        if str(row["arm"]) == "A0":
            a0_by_probe[str(row["probe_id"])] = dict(row)
        elif str(row["arm"]) == "A1":
            a1_by_probe[str(row["probe_id"])] = dict(row)

    out: dict[str, dict[str, Any]] = {}
    for probe_id, row in rows.items():
        session_id = str(row["session_id"])
        node_to_idx = {
            str(k): int(v) for k, v in row["fixture_node_to_idx"].items()}
        idx_to_node = {v: k for k, v in node_to_idx.items()}
        texts = fixture_nodes(session_id)

        a0_ids = [int(v) for v in row["a0_mounted_ids"]]
        # The node whose CAPTURE TEXT the capture arms re-capture.  An
        # a0_mounted_id that is not a fixture node (a fit-time split CHILD,
        # which exists only after a fit has run) has no fixture text to
        # re-capture from; that is recorded rather than papered over.
        capture_nodes: list[dict[str, Any]] = []
        for graft_id in a0_ids:
            node_id = idx_to_node.get(int(graft_id))
            if node_id is None:
                capture_nodes.append({
                    "graft_id": int(graft_id),
                    "node_id": None,
                    "is_fixture_node": False,
                    "note": (
                        "not a fixture node: this id is a FIT-TIME SPLIT CHILD "
                        "materialised by production's descent during the A0 "
                        "ladder, so it has no fixture capture text. The "
                        "capture arms re-capture from the graft's OWN stored "
                        "text (arena.grafts[id]['text']) and the receipt "
                        "records that substitution."),
                })
                continue
            capture_text = capture_text_for_node(texts[node_id])
            scaffold_text = question_scaffold_text(texts[node_id])
            capture_nodes.append({
                "graft_id": int(graft_id),
                "node_id": node_id,
                "is_fixture_node": True,
                "fixture_text": texts[node_id],
                "fixture_text_sha256": _sha256_text(texts[node_id]),
                "capture_text_B0_B1": capture_text,
                "capture_text_B0_B1_sha256": _sha256_text(capture_text),
                "capture_text_B2_scaffold": scaffold_text,
                "capture_text_B2_scaffold_sha256": _sha256_text(scaffold_text),
                "capture_texts_identical": bool(capture_text == scaffold_text),
            })

        live_nodes = []
        for node_id in [str(v) for v in row["live_node_ids"]]:
            live_text = capture_text_for_node(texts[node_id])
            live_nodes.append({
                "node_id": node_id,
                "graft_id": node_to_idx[node_id],
                "fed_text": live_text,
                "fed_text_sha256": _sha256_text(live_text),
            })

        out[probe_id] = {
            "probe_id": probe_id,
            "session_id": session_id,
            "role": str(row["role"]),
            "fixture_node_to_idx": node_to_idx,
            "a0_mounted_ids": a0_ids,
            "capture_nodes": capture_nodes,
            "live_ids": [int(v) for v in row["live_ids"]],
            "live_node_ids": [str(v) for v in row["live_node_ids"]],
            "live_nodes": live_nodes,
            "rs1_A0": {
                "served": str(a0_by_probe[probe_id]["served"]),
                "correct": bool(a0_by_probe[probe_id]["correct"]),
                "mounted_ids": [
                    int(v) for v in a0_by_probe[probe_id]["mounted_ids"]],
                "mounted_band_mass_full_layers": float(
                    a0_by_probe[probe_id]["mounted_band_mass"]),
                "live_band_mass": float(
                    a0_by_probe[probe_id]["live_band_mass"]),
            },
            "rs1_A1_ceiling": {
                "served": str(a1_by_probe[probe_id]["served"]),
                "correct": bool(a1_by_probe[probe_id]["correct"]),
                "mounted_band_mass_full_layers": float(
                    a1_by_probe[probe_id]["mounted_band_mass"]),
                "live_band_mass": float(
                    a1_by_probe[probe_id]["live_band_mass"]),
            },
        }
    return out


def solace_fact_variant() -> dict[str, Any]:
    """The order's extra solace run: mount the solace FACT node explicitly.

    "For solace, ALSO run every arm with the solace FACT node (node 1,
    ``solace_fact``) mounted explicitly instead of graft 4, so the
    read-strength arms are not confounded by RS1's wrong-graft finding —
    report both."

    The id is DERIVED from RS1's registered ``fixture_node_to_idx`` for the
    solace probe, not typed; the order's parenthetical "(node 1)" is checked
    against that map and the check is registered.
    """
    rs1_reg = _read(RS1_REGISTRATION)
    rows = rs1_reg["probes_REGISTERED_BEFORE_ANY_GATE"]
    probe_id = "sup_solace_fresh"
    if probe_id not in rows:
        raise RS2Error(
            f"{probe_id} is not in RS1's registration; the solace-fact "
            "variant cannot be derived")
    row = rows[probe_id]
    node_to_idx = {
        str(k): int(v) for k, v in row["fixture_node_to_idx"].items()}
    node_id = "solace_fact"
    if node_id not in node_to_idx:
        raise RS2Error(
            f"{node_id!r} is not a node of fixture {row['session_id']!r}")
    graft_id = node_to_idx[node_id]
    texts = fixture_nodes(str(row["session_id"]))
    capture_text = capture_text_for_node(texts[node_id])
    scaffold_text = question_scaffold_text(texts[node_id])
    return {
        "probe_id": probe_id,
        "session_id": str(row["session_id"]),
        "node_id": node_id,
        "graft_id": graft_id,
        "order_parenthetical": "node 1, solace_fact",
        "derived_graft_id_matches_order_parenthetical": bool(graft_id == 1),
        "replaces_a0_mounted_ids": [int(v) for v in row["a0_mounted_ids"]],
        "why": (
            "RS1's A0 mounted graft 4 on this probe — a FIT-TIME SPLIT CHILD "
            "of the long sable_competitor, not the solace fact. Every RS2 arm "
            "is therefore ALSO run with solace_fact mounted explicitly, so a "
            "read-strength number for this probe is not confounded by the "
            "wrong-graft finding."),
        "capture_text": capture_text,
        "capture_text_sha256": _sha256_text(capture_text),
        "scaffold_text": scaffold_text,
        "scaffold_text_sha256": _sha256_text(scaffold_text),
    }


def build() -> dict[str, Any]:
    probes = probe_registration()
    return {
        "schema": "grm.rs2.registration.v1",
        "program": "GRM",
        "phase": "RS2",
        "order": "orders/GRM_RS2_MOUNT_READ_DEFICIT.md",
        "branch": "lc1-wip",
        "question": (
            "RS1 established that the exact node text fed LIVE reads at "
            "live-band mass ~0.74 and answers, while the same text MOUNTED as "
            "a graft reads at mounted-band mass ~0.18 and refuses. "
            "Quantization was exonerated and the graft IS read (strict wording "
            "rescues with the identical mount). This registration fixes WHAT "
            "is measured, before anything is measured: the candidate causes of "
            "that gap."),
        "measurement_only": (
            "No production change. core/ and config/ are READ-ONLY for this "
            "order; every seam an arm needs lives in scripts/grm_rs2_*.py."),
        "arms_REGISTERED_BEFORE_ANY_GATE": arms(),
        "arm_order": ["B0", "B1", "B1p", "B1b", "B2", "B3a", "B3b", "B5"],
        "amended_by": {
            "path": "artifacts/grm_rs2/amendment_b1p_capture_position.json",
            "adds": ["arms_REGISTERED_BEFORE_ANY_GATE.B1p"],
            "why": (
                "B1's measured payload was NOT identical to the installed "
                "graft's, contrary to this registration's own prediction. The "
                "cause was isolated to ArenaCache.deposit not pinning the "
                "capture-time query position (self_attn.live_shift). B1p makes "
                "that position an explicit arm variable. ADDITION ONLY — "
                "nothing registered before the first gate is changed."),
        },
        "arm_B4_is_a_measurement_partition_not_a_serve": True,
        "baseline_arm": BASELINE_ARM,
        "ceiling_arm": CEILING_ARM,
        "probes_REGISTERED_BEFORE_ANY_GATE": probes,
        "refusers": sorted(
            pid for pid, row in probes.items() if row["role"] == "refuser"),
        "controls": sorted(
            pid for pid, row in probes.items() if row["role"] == "control"),
        "solace_fact_variant_REGISTERED_BEFORE_ANY_GATE": solace_fact_variant(),
        "sliding_partition_REGISTERED_BEFORE_ANY_GATE": partition_definition(),
        "predictions_REGISTERED_BEFORE_ANY_GATE": predictions(),
        "vocabulary_REGISTERED_BEFORE_ANY_GATE": VOCABULARY,
        "verdict_rule_REGISTERED_BEFORE_ANY_GATE": (
            "ONE verdict per hypothesis. SUPPORTED if the hypothesis's arm "
            "closes >= half the B0->B5 mounted-mass gap (mean over the "
            "reproduced probes, full-attention layers, the same statistic RS1 "
            "reported) OR flips >= 2/3 of the refusers from incorrect to "
            "correct; PARTIAL if the effect is measurable but closes < half "
            "the gap and flips < 2/3; NOT DETECTED if there is no measurable "
            "effect, in which case the report names what was not tried. "
            "'Correct' is the frozen DET1 semantic comparator "
            "(scripts/grm_det1_common.contains_value): an expected value "
            "present and no rejected value present."),
        "structural_finding_REGISTERED_BEFORE_ANY_GATE": {
            "what": (
                "Under the SPEC (ephemeral) frame the lived installer's "
                "arena.feed(text, deposit=True) takes ArenaCache.feed's "
                "`if self.ephemeral:` branch and calls self.deposit(text). "
                "ArenaCache.deposit harvests via kv_graft.harvest_kv over the "
                "TEXT ALONE — no sink, no predecessors, no injection — and the "
                "installer's text is already harmony_turn(user, assistant)."),
            "consequence": (
                "the order's B1 ('re-capture with deposit(text)') and B2 "
                "('capture the node wrapped as a harmony turn') both describe "
                "the capture the installed grafts ALREADY have. Both arms are "
                "still run, each carrying a payload-identity check; B1b varies "
                "the capture CONTEXT (a sink-prefixed harvest) and B2 varies "
                "the SCAFFOLD (the unanswered question-shaped turn), so the "
                "H-CAPTURE and H-SCAFFOLD axes are still probed rather than "
                "declared closed by a source reading."),
            "read_from": [
                "core/graft_arena.py::ArenaCache.feed",
                "core/graft_arena.py::ArenaCache.deposit",
                "core/kv_graft.py::harvest_kv",
                "scripts/grm_det1_3_gpu.py::_install_lived_nodes",
            ],
        },
        "gpu_discipline": {
            "lease":
                "/tmp/forge-gpu.lock via scripts.grm_cmc1_gpu_arms.gpu_lease",
            "single_gpu": True,
            "max_wall_per_run_s": "<= 600 (house rule), lease cap below it",
            "inter_run_gap_s": ">= 30, inside the wrapper",
            "operator_right_of_way": (
                "absolute; the harness WAITS on the flock and never signals, "
                "kills or interrupts any process"),
            "frozen_run_tree": "never written",
        },
        "sources": _sources(),
    }


def _sources(paths: Sequence[Path] | None = None) -> dict[str, dict[str, Any]]:
    """Hash inventory of every file this registration was derived from."""
    if paths is None:
        paths = (
            RS1_REGISTRATION,
            RS1_RESULTS,
            RS1_AMENDMENT,
            ROOT / "orders" / "GRM_RS2_MOUNT_READ_DEFICIT.md",
            ROOT / "core" / "graft_arena.py",
            ROOT / "core" / "grm_demand.py",
            ROOT / "core" / "gpt_oss20b_tc.py",
            ROOT / "core" / "kv_graft.py",
            ROOT / "scripts" / "grm_det1_3_gpu.py",
            ROOT / "scripts" / "grm_e2e_session.py",
            ROOT / "scripts" / "grm_rs1_read_strength_gpu.py",
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


def main(argv: Sequence[str] | None = None) -> int:
    payload = build()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    target = ARTIFACT_DIR / "registration.json"
    target.write_bytes(canonical_json_bytes(payload))
    print(f"registration={target}")
    print(json.dumps({
        "arms": payload["arm_order"],
        "probes": sorted(payload["probes_REGISTERED_BEFORE_ANY_GATE"]),
        "refusers": payload["refusers"],
        "controls": payload["controls"],
        "solace_fact_graft_id": payload[
            "solace_fact_variant_REGISTERED_BEFORE_ANY_GATE"]["graft_id"],
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
