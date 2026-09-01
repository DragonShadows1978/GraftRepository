#!/usr/bin/env python3
"""LSR Phase 0 — CPU core: window regions, value occurrences, adjudication.

Program: LSR (Lived-Serving Reliability), opened 2026-09-01.
Spec (IMMUTABLE): docs/LSR_LIVED_SERVING_PLAN.md

This module deliberately imports NO model and NO CUDA.  It owns:

  * the WINDOW LAYOUT derivation (physical cache seats -> named regions),
    read off the runtime's own seating law (core/graft_arena.py module
    docstring + ArenaCache.__init__/evict/_mount_seat_ranges), never guessed;
  * the offline fp32 witness reduction over the FULL window (the DET/CMC1.2
    instrument with its arena-only scope widened), including GPT-OSS's
    attention-sink denominator column and its sliding-window layers;
  * value-occurrence location by token span across every region;
  * the frozen Phase-0 adjudication vocabulary.

PROVENANCE POSTURE: LSR is a NEW program.  It declares NOTHING in any DET
envelope.  Its receipts are its own, content-addressed under
artifacts/lsr_p0/, and they carry file records of the DET/census inputs they
read purely as inbound references.

Reuse (imported, never copied):
  scripts.grm_cmc1_mechanism  softmax_fp32, canonical_json_bytes, sha256_*
  scripts.grm_det1_common     normalize_value_text, value_separator_regex,
                              file_record, contains_value
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.grm_cmc1_mechanism import (  # noqa: E402
    canonical_json_bytes,
    sha256_bytes,
)
from scripts.grm_det1_common import (  # noqa: E402
    contains_value,
    file_record,
    normalize_value_text,
    value_separator_regex,
)

ARTIFACT_DIR = ROOT / "artifacts" / "lsr_p0"
PLAN = ROOT / "docs" / "LSR_LIVED_SERVING_PLAN.md"
CENSUS = (
    ROOT / "artifacts" / "grm_det1" / "run_20260831T160525Z_2" / "det1_11"
    / "census" / "lived_serving_census_98ef71e88dec17a1.json"
)
SUP_FIXTURE_DIR = ROOT / "tests" / "fixtures" / "supersession_battery"
DET1_RUN = ROOT / "artifacts" / "grm_det1" / "run_20260831T160525Z_2"

SCHEMA_PREFIX = "grm.lsr_p0"

# ---------------------------------------------------------------- vocabulary
# FROZEN by docs/LSR_LIVED_SERVING_PLAN.md "Phase 0".  These three strings are
# the ONLY value-provenance verdicts this program may emit.  Anything the
# instrument cannot decide is NOT_MEASURED plus the lead script path -- never
# a fourth verdict and never a silent default.
VERDICT_RECENT_TURNS = "RECENT-TURNS"
VERDICT_ARENA = "ARENA"
VERDICT_ABSENT = "ABSENT"
VERDICT_NOT_MEASURED = "NOT_MEASURED"
REGISTERED_VERDICTS = (VERDICT_RECENT_TURNS, VERDICT_ARENA, VERDICT_ABSENT)

# Region names.  The plan names four: arena / recent / anchor / live.
# "anchor" is the runtime's SINK, "recent" is the live region's older
# segments, "live" is the live region's newest segment (the turn being
# served).  See derive_window_layout for the exact seat law.
REGION_ANCHOR = "anchor"
REGION_ARENA = "arena"
REGION_RECENT = "recent"
REGION_LIVE = "live"
REGIONS = (REGION_ANCHOR, REGION_ARENA, REGION_RECENT, REGION_LIVE)

# H-LSR-1 conviction rule, registered in the plan before any run:
#   "CONVICTED iff >=4/6 wrong-value probes show the served value present in
#    recent turns AND plurality readout mass on those tokens."
CONVICTION_NUMERATOR = 4
CONVICTION_DENOMINATOR = 6


class LSRError(RuntimeError):
    """An LSR Phase-0 invariant, receipt, or registered rule was violated."""


# ------------------------------------------------------------ window layout
def derive_window_layout(
    *,
    n_sink: int,
    arena_width: int,
    cur_mount_n: int,
    mount_seat_ranges: Mapping[int, Sequence[int]],
    live_segs: Sequence[Sequence[Any]],
    live_turns: int,
    prompt_ntok: int = 0,
) -> dict[str, Any]:
    """Map the runtime's physical cache seats onto the plan's four regions.

    DERIVATION (recorded here because the plan requires stating how the
    boundaries were obtained).  Every clause cites the runtime source that
    establishes it -- none of this is inferred from observed data:

    1. core/graft_arena.py module docstring, the seating plan:
           seats [0 .. n_sink)             SINK
           seats [n_sink .. n_sink+width)  ARENA
           seats [n_sink+width .. )        LIVE
       That is the POSITIONAL (RoPE) law.  `live_shift = n_sink +
       arena_width` is fixed for the cache's lifetime.

    2. The PHYSICAL cache tensor is NOT the positional layout.  Mounts are a
       packed PREFIX with no hole padding -- ArenaCache._mount_seat_ranges:
       "mounts are a packed PREFIX of the arena starting at n_sink (no hole
       padding in the physical cache tensor -- the hole is a RoPE-position-
       only concept)".  So physically the arena occupies
       [n_sink, n_sink + cur_mount_n) and live rows begin immediately after.
       Attention score COLUMNS index physical rows, so regions are physical.

    3. ArenaCache.evict() confirms the physical head:
           head = self.n_sink + self.cur_mount_n
       and compacts live rows against it.  Therefore live physical rows are
       contiguous from `live_start = n_sink + cur_mount_n`.

    4. ArenaCache.evict() also fixes WHICH turns survive:
           self.live_segs = self.live_segs[-self.live_turns:]
       so at most `live_turns` segments remain, in chronological order, each
       carrying its own token count.  scripts/grm_det1_5_gpu._live_token_ids
       reconstructs their ids in exactly this order.

    5. RECENT vs LIVE split: the surviving live SEGMENTS are prior turns --
       the RECENT-TURNS region H-LSR-1 accuses.  The probe's own prompt rows,
       appended by _attempt's prefill after those segments, are the LIVE
       region (the turn being served).  `prompt_ntok` supplies their count;
       with live_segs empty and prompt_ntok>0 the window is
       [anchor | arena | (empty recent) | live=prompt].

    Returns absolute physical [start, end) seat ranges keyed by region, plus
    the per-segment breakdown so a value hit can be attributed to a turn.
    """
    n_sink = int(n_sink)
    arena_width = int(arena_width)
    cur_mount_n = int(cur_mount_n)
    live_turns = int(live_turns)
    prompt_ntok = int(prompt_ntok)
    if n_sink < 0 or arena_width < 0 or cur_mount_n < 0 or prompt_ntok < 0:
        raise LSRError("window layout received a negative seat count")
    if cur_mount_n > arena_width:
        raise LSRError(
            f"mounted seats {cur_mount_n} exceed arena width {arena_width}")

    segs = [(seg[0], int(seg[1])) for seg in live_segs]
    if live_turns > 0 and len(segs) > live_turns:
        raise LSRError(
            f"{len(segs)} live segments survive a live_turns={live_turns} "
            "window; capture happened before evict()")

    arena_start = n_sink
    arena_end = n_sink + cur_mount_n
    seat = arena_end

    # RECENT vs LIVE among the resident segments.  When the layout is derived
    # AFTER the serve, arena.live_segs already ends with the probe's OWN turn
    # (core/graft_arena.py:2775 appends it post-generation), and that turn is
    # the LIVE region, not a recent turn.  It is recognizable because the
    # witness serves with deposit=False, so its graft_index is None
    # (graft_arena.py:2752 `gidx = None; if deposit: gidx = ...`).
    #
    # Only a TRAILING anonymous segment is the probe turn.  An anonymous
    # segment in any earlier position is a prior undeposited turn and stays
    # RECENT -- being undeposited does not make an older turn "live".
    probe_turn_index = (
        len(segs) - 1
        if segs and segs[-1][0] is None and not prompt_ntok
        else None
    )
    seg_rows: list[dict[str, Any]] = []
    for index, (graft_index, count) in enumerate(segs):
        is_probe_turn = index == probe_turn_index
        seg_rows.append({
            "segment_index": int(index),
            "graft_index": None if graft_index is None else int(graft_index),
            "ntok": int(count),
            "start": int(seat),
            "end": int(seat + count),
            "region": REGION_LIVE if is_probe_turn else REGION_RECENT,
            **({"is_probe_turn": True} if is_probe_turn else {}),
        })
        seat += int(count)
    if probe_turn_index is None:
        recent_end = seat
    else:
        recent_end = int(seg_rows[probe_turn_index]["start"])
    live_end = seat + prompt_ntok
    if prompt_ntok:
        # Pre-serve layout: the probe's prompt is not yet a live segment, so
        # it is supplied separately and forms the LIVE region.
        seg_rows.append({
            "segment_index": len(segs),
            "graft_index": None,
            "ntok": prompt_ntok,
            "start": int(seat),
            "end": int(live_end),
            "region": REGION_LIVE,
            "is_probe_prompt": True,
        })

    regions = {
        REGION_ANCHOR: (0, n_sink),
        REGION_ARENA: (arena_start, arena_end),
        REGION_RECENT: (arena_end, recent_end),
        REGION_LIVE: (recent_end, live_end),
    }
    for name, (start, end) in regions.items():
        if not (0 <= start <= end <= live_end):
            raise LSRError(f"region {name} span {(start, end)} is malformed")
    covered = sum(end - start for start, end in regions.values())
    if covered != live_end:
        raise LSRError(
            f"regions cover {covered} seats but the window holds {live_end}")

    return {
        "schema": f"{SCHEMA_PREFIX}.window_layout.v1",
        "derivation": (
            "physical cache seats; anchor=[0,n_sink); arena=packed mount "
            "prefix [n_sink,n_sink+cur_mount_n) per "
            "ArenaCache._mount_seat_ranges; live rows compacted against "
            "head=n_sink+cur_mount_n per ArenaCache.evict(); surviving "
            "live_segs (order per _live_token_ids) = RECENT; the probe's own "
            "prefill rows = LIVE"
        ),
        "derivation_sources": [
            "core/graft_arena.py::ArenaCache module docstring (seating plan)",
            "core/graft_arena.py::ArenaCache._mount_seat_ranges",
            "core/graft_arena.py::ArenaCache.evict",
            "core/graft_arena.py::ArenaCache._attempt (prefill of prompt_ids)",
            "scripts/grm_det1_5_gpu.py::_live_token_ids",
            "scripts/grm_det1_3_snapshot.py::positions.physical_sections",
        ],
        "positional_law": {
            "live_shift": int(n_sink + arena_width),
            "note": (
                "RoPE positions use live_shift; the unused arena remainder is "
                "a positional hole with no physical row (free_seats finding). "
                "Region spans below are PHYSICAL rows, which is what the "
                "attention score columns index."
            ),
            "positional_hole_size": int(arena_width - cur_mount_n),
        },
        "n_sink": n_sink,
        "arena_width": arena_width,
        "cur_mount_n": cur_mount_n,
        "live_turns": live_turns,
        "prompt_ntok": prompt_ntok,
        "window_tokens": int(live_end),
        "regions": {
            name: {"start": int(start), "end": int(end),
                   "ntok": int(end - start)}
            for name, (start, end) in regions.items()
        },
        "live_segments": seg_rows,
        "mount_seat_ranges": {
            str(int(key)): [int(value[0]), int(value[1])]
            for key, value in dict(mount_seat_ranges).items()
        },
    }


def layout_from_snapshot_state(
    state: Mapping[str, Any], *, live_turns: int = 2,
) -> dict[str, Any]:
    """Build the layout from a frozen DET1.3 snapshot's state block.

    The snapshot is read as an INBOUND REFERENCE only; nothing is written
    back into any DET envelope.  Fields consumed are the ones
    grm_det1_3_snapshot.capture_arena_snapshot records:
    arena.n_sink, arena.width, arena.cur_mount_n, arena.cur_mounts,
    live.segments, tokens.prompt_count.
    """
    def scalar(key: str, default: Any = None) -> Any:
        value = state.get(key, default)
        if isinstance(value, str):
            try:
                return json.loads(value)
            except (TypeError, ValueError):
                return value
        return value

    segments = scalar("live.segments", []) or []
    prompt_count = scalar("tokens.prompt_count", 0) or 0
    mounts = scalar("arena.cur_mounts", []) or []
    n_sink = int(scalar("arena.n_sink", 0) or 0)
    seat = n_sink
    ranges: dict[int, tuple[int, int]] = {}
    for value in mounts:
        ranges[int(value)] = (seat, seat)
    return derive_window_layout(
        n_sink=n_sink,
        arena_width=int(scalar("arena.width", 0) or 0),
        cur_mount_n=int(scalar("arena.cur_mount_n", 0) or 0),
        mount_seat_ranges=ranges,
        live_segs=[(row[0], row[1]) if isinstance(row, (list, tuple))
                   else (None, int(row)) for row in segments],
        live_turns=int(scalar("arena.live_turns", live_turns) or live_turns),
        prompt_ntok=int(prompt_count),
    )


def region_of_seat(layout: Mapping[str, Any], seat: int) -> str | None:
    """Name the region owning a physical seat, or None if outside the window."""
    seat = int(seat)
    for name in REGIONS:
        span = layout["regions"][name]
        if int(span["start"]) <= seat < int(span["end"]):
            return name
    return None


# ------------------------------------------------------- value localization
def locate_value_spans(
    *,
    layout: Mapping[str, Any],
    token_ids: Sequence[int],
    token_strings: Sequence[str],
    value: str,
    label: str,
) -> list[dict[str, Any]]:
    """Every token span across the window whose text carries `value`.

    Matching reuses the DET1.10 registered separator semantics
    (grm_det1_common.value_separator_regex + normalize_value_text) so a hit
    here means exactly what a hit means in the census: token payloads and
    token COUNT stay load-bearing, separator glyphs and Markdown emphasis do
    not.  Character offsets are walked back to token indices, and each hit is
    attributed to the region owning its FIRST token.
    """
    if len(token_ids) != len(token_strings):
        raise LSRError("token id/string ledgers differ in length")
    starts: list[int] = []
    cursor = 0
    for text in token_strings:
        starts.append(cursor)
        cursor += len(text)
    joined = "".join(token_strings)
    normalized = normalize_value_text(joined)
    # normalize_value_text may delete Markdown emphasis delimiters and so
    # shift character offsets.  Offsets are load-bearing here (they become
    # token indices), so only use the normalized text when it is
    # length-preserving; otherwise match case-insensitively on the raw text.
    haystack = normalized if len(normalized) == len(joined) else joined
    pattern = (
        rf"(?<![A-Za-z0-9_-]){value_separator_regex(normalize_value_text(value))}"
        rf"(?![A-Za-z0-9_-])"
    )
    hits: list[dict[str, Any]] = []
    for match in re.finditer(pattern, haystack, flags=re.IGNORECASE):
        start_char, end_char = match.start(), match.end()
        first = last = None
        for index, char_start in enumerate(starts):
            char_end = char_start + len(token_strings[index])
            if char_end <= start_char or char_start >= end_char:
                continue
            if first is None:
                first = index
            last = index
        if first is None or last is None:
            continue
        segment = None
        for row in layout["live_segments"]:
            if int(row["start"]) <= first < int(row["end"]):
                segment = int(row["segment_index"])
                break
        hits.append({
            "label": label,
            "value": str(value),
            "token_start": int(first),
            "token_end": int(last + 1),
            "token_ids": [int(v) for v in token_ids[first:last + 1]],
            "surface": joined[start_char:end_char],
            "region": region_of_seat(layout, first),
            "live_segment_index": segment,
        })
    return hits


# ------------------------------------------------------ full-window witness
def full_window_readout_mass(
    *,
    softmax_operands_by_layer: Mapping[int, np.ndarray],
    layout: Mapping[str, Any],
    sink_logits_by_layer: Mapping[int, np.ndarray] | None = None,
    allowed_masks_by_layer: Mapping[int, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Offline fp32 witness over the WHOLE window at answer-readout positions.

    This is the DET/CMC1.2 instrument with its ARENA-ONLY SCOPE WIDENED.  Two
    independent arena restrictions exist upstream and both are removed here:

      (a) scripts/grm_cmc1_gpu_arms.py::SDPAInterceptor._capture slices the
          captured score columns to the mounted arena --
              mount_start = min(start for start,_ in absolute_ranges.values())
              mount_end   = max(end   for _,end   in absolute_ranges.values())
              mounted_scores = operand_last.slice(3, mount_start, ...)
      (b) core/graft_arena.py::ArenaCache.s1_mass normalizes over "total
          non-live mass", seats [0, n_sink+cur_mount_n), excluding live.

    Phase 0 requires the complement: every column, live included.  The GPU
    capture must therefore pass the FULL score row and this function
    normalizes across all of it; a short row is rejected rather than padded.

    CMC1.2 law retained: the offline fp32 work starts at the ENGINE'S OWN
    scaled-QK score values after its compute-dtype (bf16) store, never at a
    re-derived idealized fp32 QK contraction.  Callers hand in exactly the
    tensor the engine passed to its softmax.

    GPT-OSS SINK COLUMN: core/gpt_oss20b_tc.py::sink_attention_tc appends a
    per-head sink logit as an extra column, softmaxes over the concatenation,
    then DROPS that column.  Its mass is real and must sit in the denominator
    or every region's share is inflated.  Pass `sink_logits_by_layer` (shape
    (H,)) and it is carried as an explicit attention_sink term outside the
    four regions.

    SLIDING-WINDOW LAYERS: core/gpt_oss20b_tc.py::sliding_sink_attention_tc
    masks keys outside a 128-token trailing window (24 layers alternate
    full/sliding).  Masked columns are supplied as an allowed-mask so they are
    excluded rather than counted as genuine near-zero mass, and per-layer
    region VISIBILITY is reported so a plurality claim is never built on a
    layer that could not see the region at all.

    Shapes: softmax_operands_by_layer[layer] is
    (readout_positions, heads, window_tokens); sink logits (heads,);
    allowed mask (readout_positions, window_tokens) or (window_tokens,), bool.
    """
    layers = sorted(int(key) for key in softmax_operands_by_layer)
    if not layers:
        raise LSRError("full-window witness received no layers")
    window_tokens = int(layout["window_tokens"])

    per_layer: list[dict[str, Any]] = []
    region_totals = {name: 0.0 for name in REGIONS}
    sink_total = 0.0
    weight_sum = 0.0
    readouts: int | None = None
    per_token_mass = np.zeros(window_tokens, dtype=np.float64)

    for layer in layers:
        scores = np.asarray(softmax_operands_by_layer[layer], dtype=np.float32)
        if scores.ndim != 3:
            raise LSRError(
                f"layer {layer}: expected (readouts,heads,window) scores, "
                f"got shape {scores.shape}")
        if int(scores.shape[2]) != window_tokens:
            raise LSRError(
                f"layer {layer}: score row covers {scores.shape[2]} columns "
                f"but the derived window holds {window_tokens}; the capture "
                "was not widened past the arena")
        if readouts is None:
            readouts = int(scores.shape[0])
        elif readouts != int(scores.shape[0]):
            raise LSRError("layers disagree on the answer-readout count")

        masked = scores
        allowed = None
        if allowed_masks_by_layer is not None and layer in allowed_masks_by_layer:
            allowed = np.asarray(allowed_masks_by_layer[layer], dtype=bool)
            if allowed.ndim == 1:
                allowed = np.broadcast_to(
                    allowed[None, :], (scores.shape[0], window_tokens))
            if allowed.shape != (scores.shape[0], window_tokens):
                raise LSRError(
                    f"layer {layer}: allowed-mask shape {allowed.shape} does "
                    "not match (readouts,window)")
            masked = np.where(
                allowed[:, None, :], scores,
                np.float32(-np.inf)).astype(np.float32)

        sink = None
        if sink_logits_by_layer is not None and layer in sink_logits_by_layer:
            sink = np.asarray(
                sink_logits_by_layer[layer], dtype=np.float32).reshape(-1)
            if sink.size != scores.shape[1]:
                raise LSRError(
                    f"layer {layer}: {sink.size} sink logits for "
                    f"{scores.shape[1]} heads")
            column = np.broadcast_to(
                sink[None, :, None],
                (masked.shape[0], masked.shape[1], 1)).astype(np.float32)
            combined = np.concatenate([masked, column], axis=-1)
        else:
            combined = masked

        finite = np.isfinite(combined)
        if not finite.any(axis=-1).all():
            raise LSRError(
                f"layer {layer}: a readout row has no attendable column")
        shifted = combined - np.max(
            np.where(finite, combined, -np.inf), axis=-1, keepdims=True)
        numer = np.where(finite, np.exp(shifted, dtype=np.float32), 0.0)
        denom = numer.sum(axis=-1, keepdims=True, dtype=np.float32)
        if np.any(denom <= 0):
            raise LSRError(f"layer {layer}: softmax denominator is invalid")
        weights = (numer / denom).astype(np.float32)

        if sink is not None:
            sink_mass = float(weights[..., -1].sum())
            weights = weights[..., :window_tokens]
        else:
            sink_mass = 0.0

        layer_tokens = weights.sum(axis=(0, 1), dtype=np.float64)
        per_token_mass += layer_tokens
        sink_total += sink_mass
        weight_sum += float(layer_tokens.sum()) + sink_mass

        layer_regions = {}
        for name in REGIONS:
            span = layout["regions"][name]
            mass = float(layer_tokens[int(span["start"]):int(span["end"])].sum())
            layer_regions[name] = mass
            region_totals[name] += mass

        visible = {}
        for name in REGIONS:
            span = layout["regions"][name]
            if int(span["ntok"]) == 0:
                visible[name] = False
            elif allowed is None:
                visible[name] = True
            else:
                visible[name] = bool(
                    allowed[:, int(span["start"]):int(span["end"])].any())
        per_layer.append({
            "layer": int(layer),
            "attention_sink_mass": sink_mass,
            "region_mass": layer_regions,
            "region_visible": visible,
            "masked": allowed is not None,
        })

    if readouts is None or weight_sum <= 0:
        raise LSRError("full-window witness accumulated no mass")

    split = {name: region_totals[name] / weight_sum for name in REGIONS}
    ranked = sorted(split.items(), key=lambda item: (-item[1], item[0]))
    return {
        "schema": f"{SCHEMA_PREFIX}.full_window_readout_mass.v1",
        "scope": "full_window_anchor_arena_recent_live",
        "scope_widening": (
            "DET/CMC1.2 witness with the arena-only column slice removed "
            "(grm_cmc1_gpu_arms._capture sliced to [mount_start,mount_end)) "
            "and without s1_mass's non-live-only denominator"
        ),
        "score_operand": "engine_softmax_input_compute_dtype_values",
        "softmax_recompute": "numpy_fp32_from_engine_softmax_operand",
        "attention_sink_handled": sink_logits_by_layer is not None,
        "sliding_window_masks_applied": allowed_masks_by_layer is not None,
        "layers": len(layers),
        "answer_readout_positions": int(readouts),
        "window_tokens": window_tokens,
        "region_mass_share": split,
        "attention_sink_mass_share": sink_total / weight_sum,
        "plurality_region": ranked[0][0],
        "plurality_share": ranked[0][1],
        "runner_up_region": ranked[1][0] if len(ranked) > 1 else None,
        "runner_up_share": ranked[1][1] if len(ranked) > 1 else None,
        "per_token_mass": [float(value) for value in per_token_mass],
        "per_layer": per_layer,
        "total_weight": float(weight_sum),
    }


def span_mass_share(
    mass: Mapping[str, Any], spans: Sequence[Mapping[str, Any]],
) -> float:
    """Share of total readout mass sitting on a set of token spans."""
    per_token = np.asarray(mass["per_token_mass"], dtype=np.float64)
    total = float(mass["total_weight"])
    if total <= 0:
        raise LSRError("cannot take a span share of zero total mass")
    seats: set[int] = set()
    for span in spans:
        seats.update(range(int(span["token_start"]), int(span["token_end"])))
    if not seats:
        return 0.0
    index = np.fromiter(sorted(seats), dtype=np.int64)
    if int(index.max()) >= per_token.size:
        raise LSRError("a value span lies outside the witnessed window")
    return float(per_token[index].sum() / total)


# ------------------------------------------------------------ adjudication
def probe_verdict(
    *,
    probe_id: str,
    served_value: str,
    expected_value: str,
    layout: Mapping[str, Any],
    mass: Mapping[str, Any],
    served_spans: Sequence[Mapping[str, Any]],
    expected_spans: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Per-probe value-provenance verdict, per the plan's registered rule.

    The plan's Phase-0 deliverable is "value-provenance verdict per probe
    (RECENT-TURNS / ARENA / ABSENT)".  Applied to the SERVED value -- the
    question the phase asks is where the wrong value physically came from:

      ABSENT       the served value occurs nowhere in the window
                   (H-LSR-3's prediction)
      RECENT-TURNS the served value occurs in the recent-turns region and
                   that region's occurrences carry the plurality of readout
                   mass among the regions where it occurs (H-LSR-1)
      ARENA        the served value occurs in the arena and the arena's
                   occurrences carry that plurality

    Occurrences in anchor/live alone support neither RECENT-TURNS nor ARENA;
    that case is reported NOT_MEASURED rather than forced, because the plan's
    vocabulary has no term for it and inventing a fourth verdict is
    forbidden.
    """
    regions_hit = {}
    for name in REGIONS:
        spans = [row for row in served_spans if row.get("region") == name]
        regions_hit[name] = {
            "occurrences": len(spans),
            "mass_share": span_mass_share(mass, spans) if spans else 0.0,
        }
    present_anywhere = any(row["occurrences"] for row in regions_hit.values())

    if not present_anywhere:
        verdict = VERDICT_ABSENT
        basis = "served value occurs in no region of the window"
    else:
        candidates = {
            REGION_RECENT: regions_hit[REGION_RECENT],
            REGION_ARENA: regions_hit[REGION_ARENA],
        }
        live = [name for name, row in candidates.items() if row["occurrences"]]
        if not live:
            verdict = VERDICT_NOT_MEASURED
            basis = (
                "served value occurs only outside arena and recent-turns "
                "(anchor/live); the registered vocabulary has no verdict for "
                "that provenance"
            )
        else:
            best = max(
                live, key=lambda name: (candidates[name]["mass_share"], name))
            verdict = (
                VERDICT_RECENT_TURNS if best == REGION_RECENT
                else VERDICT_ARENA
            )
            basis = (
                f"served value present in {best}; that region's occurrences "
                f"carry mass share {candidates[best]['mass_share']:.6f}"
            )

    recent_present = bool(regions_hit[REGION_RECENT]["occurrences"])
    plurality_on_recent_spans = False
    if recent_present:
        recent_spans = [
            row for row in served_spans if row.get("region") == REGION_RECENT
        ]
        recent_share = span_mass_share(mass, recent_spans)
        other = max(
            (regions_hit[name]["mass_share"]
             for name in REGIONS if name != REGION_RECENT),
            default=0.0,
        )
        plurality_on_recent_spans = bool(recent_share > other)

    return {
        "schema": f"{SCHEMA_PREFIX}.probe_verdict.v1",
        "probe_id": str(probe_id),
        "served_value": str(served_value),
        "expected_value": str(expected_value),
        "verdict": verdict,
        "verdict_basis": basis,
        "region_mass_share": dict(mass["region_mass_share"]),
        "attention_sink_mass_share": float(mass["attention_sink_mass_share"]),
        "plurality_region": str(mass["plurality_region"]),
        "served_value_by_region": regions_hit,
        "served_value_spans": [dict(row) for row in served_spans],
        "expected_value_spans": [dict(row) for row in expected_spans],
        "served_value_present_in_recent_turns": recent_present,
        "plurality_readout_mass_on_served_recent_spans": plurality_on_recent_spans,
        "h_lsr_1_leg_satisfied": bool(recent_present and plurality_on_recent_spans),
    }


def adjudicate_h_lsr_1(
    wrong_value_verdicts: Sequence[Mapping[str, Any]],
    control_verdicts: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """The plan's registered Phase-0 adjudication, applied verbatim.

    'H-LSR-1 CONVICTED iff >=4/6 wrong-value probes show the served value
    present in recent turns AND plurality readout mass on those tokens.'

    Both legs are per-probe and conjunctive; the count is over probes
    satisfying BOTH.  The denominator is the registered 6.  If fewer than 6
    wrong-value probes were actually measured, the rule cannot be evaluated
    and the result is NOT_MEASURED -- a short count is never silently treated
    as a failure to convict.
    """
    rows = [dict(row) for row in wrong_value_verdicts]
    measured = [row for row in rows if row.get("verdict") != VERDICT_NOT_MEASURED]
    satisfied = [row for row in rows if row.get("h_lsr_1_leg_satisfied")]
    if len(measured) < CONVICTION_DENOMINATOR:
        status = VERDICT_NOT_MEASURED
        convicted = None
    else:
        convicted = bool(len(satisfied) >= CONVICTION_NUMERATOR)
        status = "CONVICTED" if convicted else "NOT_CONVICTED"
    return {
        "schema": f"{SCHEMA_PREFIX}.h_lsr_1_adjudication.v1",
        "rule": (
            "H-LSR-1 CONVICTED iff >=4/6 wrong-value probes show the served "
            "value present in recent turns AND plurality readout mass on "
            "those tokens"
        ),
        "rule_source": "docs/LSR_LIVED_SERVING_PLAN.md#phase-0",
        "registered_before_any_run": True,
        "numerator_threshold": CONVICTION_NUMERATOR,
        "denominator": CONVICTION_DENOMINATOR,
        "wrong_value_probes_measured": len(measured),
        "wrong_value_probes_satisfying_both_legs": len(satisfied),
        "satisfying_probe_ids": [str(row["probe_id"]) for row in satisfied],
        "status": status,
        "h_lsr_1_convicted": convicted,
        "per_probe_verdicts": {
            str(row["probe_id"]): str(row["verdict"]) for row in rows
        },
        "control_verdicts": {
            str(row["probe_id"]): str(row["verdict"]) for row in control_verdicts
        },
        "control_note": (
            "Lawful same-session controls are reported, never counted in the "
            "registered fraction; the plan's denominator is the 6 wrong-value "
            "probes alone."
        ),
    }


# -------------------------------------------------------- probe enumeration
def enumerate_probes(census_path: Path = CENSUS) -> dict[str, Any]:
    """The 6 wrong-value probes + 2 lawful same-session controls.

    Wrong-value probes: census rows with census_class == 'WRONG_VALUE'.  The
    census counts 6 and also asserts wrong_value_with_correct_single_mount=6.

    Controls: the plan asks for '2 lawful same-session controls'.  Same-session
    means the control shares a supersession session_id with a wrong-value
    probe, so its window is built from the same fixture text.  Selection is
    deterministic and registered here rather than chosen at run time: LAWFUL
    rows from the supersession family whose session_id also hosts a
    wrong-value probe, ordered by probe_id, first two.
    """
    census = json.loads(Path(census_path).read_text(encoding="utf-8"))
    rows = list(census["rows"])
    wrong = sorted(
        (row for row in rows if row.get("census_class") == "WRONG_VALUE"),
        key=lambda row: str(row["probe_id"]),
    )
    if len(wrong) != int(census["counts"]["unlawful_wrong_value"]):
        raise LSRError(
            f"census lists {census['counts']['unlawful_wrong_value']} "
            f"wrong-value probes but {len(wrong)} rows carry the class")
    if len(wrong) != CONVICTION_DENOMINATOR:
        raise LSRError(
            f"the registered denominator is {CONVICTION_DENOMINATOR} but the "
            f"census carries {len(wrong)} wrong-value probes")

    sessions = probe_session_map()
    wrong_sessions = {
        sessions.get(str(row["probe_id"]), {}).get("session_id")
        for row in wrong
    }
    lawful = sorted(
        (row for row in rows
         if row.get("census_class") == "LAWFUL"
         and row.get("source_family") == "supersession_battery_on_gpt_oss"
         and sessions.get(str(row["probe_id"]), {}).get("session_id")
         in wrong_sessions),
        key=lambda row: str(row["probe_id"]),
    )
    controls = lawful[:2]
    if len(controls) != 2:
        raise LSRError(
            "the census does not carry two lawful same-session controls "
            f"(found {len(lawful)})")

    def project(row: Mapping[str, Any], role: str) -> dict[str, Any]:
        probe_id = str(row["probe_id"])
        binding = sessions.get(probe_id, {})
        return {
            "probe_id": probe_id,
            "lsr_role": role,
            "census_class": str(row["census_class"]),
            "source_family": str(row["source_family"]),
            "spec": str(row["spec"]),
            "split": str(row["split"]),
            "role": str(row["role"]),
            "served_answer": str(row["served_answer"]),
            "served_value": binding.get("served_value"),
            "served_answer_correct": bool(row["served_answer_correct"]),
            "expected_values": [str(v) for v in row["expected_values"]],
            "mounted_ids": [int(v) for v in row["mounted_ids"]],
            "mount_count": int(row["mount_count"]),
            "single_mount": bool(row["single_mount"]),
            "attempt": int(row["attempt"]),
            "session_id": binding.get("session_id"),
            "question": binding.get("question"),
        }

    return {
        "schema": f"{SCHEMA_PREFIX}.probe_enumeration.v1",
        "census": file_record(Path(census_path)),
        "census_generation": int(census.get("generation", 1)),
        "census_run_id": str(census["run_id"]),
        "wrong_value_probes": [project(row, "wrong_value") for row in wrong],
        "lawful_controls": [project(row, "lawful_control") for row in controls],
        "control_selection_rule": (
            "LAWFUL rows from source_family=supersession_battery_on_gpt_oss "
            "whose session_id also hosts a wrong-value probe, ordered by "
            "probe_id, first two"
        ),
    }


# The served VALUE extracted from each served ANSWER sentence.  The census
# stores prose ("The current Meridian seal is **Cobalt-3-Comet**."); the
# witness needs the bare value to locate its token span.  Extraction is
# recorded here explicitly rather than parsed heuristically at run time.
SERVED_VALUE_BINDINGS: dict[str, str] = {
    "sup_lumen_head": "Birch-2-Beacon",
    "sup_orion_current": "Auric-4-Alpha",
    "sup_reserve_falcon_registry": "42",
    "sup_reserve_juniper_pass": "0",
    "sup_reserve_meridian_docket": "Cobalt-3-Comet",
    "sup_reserve_tundra_ledger": "Tundra-42-Eagle-Blue-Sapphire",
    "sup_harbor_restatement": "Nacre-6-Blue",
    "sup_praxis_fresh": "Quartz-8-Jade",
    "sup_solace_fresh": "Raven-9-Ivory",
}

# The DET1.9 reserve probes are appended to an existing certified session
# (scripts/grm_det1_5_workers.SUP_RESERVE_PROBES); this maps each to the
# supersession fixture file whose nodes form its arena.
RESERVE_SESSION_BINDINGS: dict[str, str] = {
    "sup_reserve_juniper_pass": "correction_then_restatement",
    "sup_reserve_tundra_ledger": "fresh_fact_controls",
    "sup_reserve_meridian_docket": "multi_hop_a_b_c",
    "sup_reserve_falcon_registry": "short_correction_long_competitor",
}


def load_sup_sessions(directory: Path = SUP_FIXTURE_DIR) -> dict[str, Any]:
    """Load the four certified supersession fixtures by session_id."""
    sessions: dict[str, Any] = {}
    for path in sorted(Path(directory).glob("*.json")):
        fixture = json.loads(path.read_text(encoding="utf-8"))
        sessions[str(fixture["session_id"])] = {
            "fixture": fixture,
            "file": file_record(path),
        }
    if not sessions:
        raise LSRError(f"no supersession fixtures under {directory}")
    return sessions


def probe_session_map(directory: Path = SUP_FIXTURE_DIR) -> dict[str, Any]:
    """Bind every sup_* probe id to its session, question, and served value.

    Registered probes come from the fixture's own `probes` list (see
    scripts/grm_det1_register.py, which names them f"sup_{probe_id}").
    Reserve probes come from SUP_RESERVE_PROBES and are bound by
    RESERVE_SESSION_BINDINGS.
    """
    sessions = load_sup_sessions(directory)
    out: dict[str, Any] = {}
    for session_id, entry in sessions.items():
        for probe in entry["fixture"]["probes"]:
            probe_id = f"sup_{probe['probe_id']}"
            out[probe_id] = {
                "session_id": session_id,
                "question": str(probe["question"]),
                "expected_values": [str(v) for v in probe["expected_values"]],
                "served_value": SERVED_VALUE_BINDINGS.get(probe_id),
                "source": "fixture_probes",
                "fixture_file": entry["file"],
            }
    reserves = _reserve_probe_definitions()
    for probe_id, session_id in RESERVE_SESSION_BINDINGS.items():
        entry = sessions[session_id]
        reserve = reserves.get(session_id, {})
        out[probe_id] = {
            "session_id": session_id,
            "question": reserve.get("question"),
            "expected_values": [
                str(v) for v in (reserve.get("expected_values") or [])
            ] or None,
            "served_value": SERVED_VALUE_BINDINGS.get(probe_id),
            "source": "det1_9_reserve",
            "fixture_file": entry["file"],
        }
    return out


def _reserve_probe_definitions() -> dict[str, Any]:
    """The frozen DET1.9 reserve probe definitions, keyed by session_id.

    Imported from scripts/grm_det1_5_workers.SUP_RESERVE_PROBES rather than
    restated, so the question text the witness serves is byte-identical to
    the one the census served.  That module imports no model and no CUDA at
    file scope, so this is safe on the CPU path.
    """
    from scripts.grm_det1_5_workers import SUP_RESERVE_PROBES

    return {str(key): dict(value) for key, value in SUP_RESERVE_PROBES.items()}


def session_node_inventory(
    probe_id: str, *, directory: Path = SUP_FIXTURE_DIR,
) -> dict[str, Any]:
    """Where each session value lives, and which node the arena mounted.

    Node ORDER is the deposit index: scripts/grm_det1_gpu._install_fixture_nodes
    iterates `fixture["nodes"]` and calls `arena.deposit(node["text"])`, whose
    return is the graft index.  So node i == graft index i, which is what the
    snapshot's `arena.cur_mounts` refers to.
    """
    sessions = load_sup_sessions(directory)
    binding = probe_session_map(directory)[probe_id]
    fixture = sessions[binding["session_id"]]["fixture"]
    return {
        "probe_id": probe_id,
        "session_id": binding["session_id"],
        "fixture_file": binding["fixture_file"],
        "index_law": (
            "node order == deposit index == graft index; see "
            "scripts/grm_det1_gpu.py::_install_fixture_nodes"
        ),
        "nodes": [
            {
                "graft_index": index,
                "node_id": str(node["node_id"]),
                "role": str(node["role"]),
                "value": str(node["value"]),
                "ntok_text_chars": len(str(node["text"])),
            }
            for index, node in enumerate(fixture["nodes"])
        ],
    }


def value_presence_across_session(
    probe_id: str, value: str, *, directory: Path = SUP_FIXTURE_DIR,
) -> dict[str, Any]:
    """Which session nodes carry `value`, by the census's own matcher.

    This is a TEXT-level pre-check, not the witness.  It answers "could the
    value have come from the arena at all", using grm_det1_common.contains_value
    so a hit means what a census hit means.  It cannot substitute for the
    attention-mass leg and is never used to emit a verdict on its own.
    """
    sessions = load_sup_sessions(directory)
    binding = probe_session_map(directory)[probe_id]
    fixture = sessions[binding["session_id"]]["fixture"]
    rows = []
    for index, node in enumerate(fixture["nodes"]):
        rows.append({
            "graft_index": index,
            "node_id": str(node["node_id"]),
            "role": str(node["role"]),
            "value": str(node["value"]),
            "carries_value": bool(contains_value(str(node["text"]), value)),
        })
    return {
        "probe_id": probe_id,
        "session_id": binding["session_id"],
        "value": str(value),
        "matcher": "scripts.grm_det1_common.contains_value (DET1.10 semantics)",
        "nodes": rows,
        "present_in_any_node": any(row["carries_value"] for row in rows),
        "carrying_graft_indices": [
            row["graft_index"] for row in rows if row["carries_value"]
        ],
    }


# --------------------------------------------------------------- selftests
def _synthetic_window(
    *, n_sink: int, mount_n: int, recent_ntok: int, live_ntok: int,
) -> dict[str, Any]:
    segs = [(0, recent_ntok)] if recent_ntok else []
    return derive_window_layout(
        n_sink=n_sink,
        arena_width=96,
        cur_mount_n=mount_n,
        mount_seat_ranges={0: (n_sink, n_sink + mount_n)},
        live_segs=segs,
        live_turns=2,
        prompt_ntok=live_ntok,
    )


def _scores_for(
    layout: Mapping[str, Any], heavy: Sequence[int], *,
    layers: int = 3, heads: int = 4, readouts: int = 2, weight: float = 8.0,
) -> dict[int, np.ndarray]:
    window = int(layout["window_tokens"])
    out = {}
    for layer in range(layers):
        scores = np.full((readouts, heads, window), -4.0, dtype=np.float32)
        for seat in heavy:
            scores[:, :, int(seat)] = np.float32(weight)
        out[layer] = scores
    return out


def cpu_selftest() -> dict[str, Any]:
    """Synthetic-window selftests for every registered rule.

    No model, no CUDA, no captured artifact: each case builds a window from
    the runtime's seat law, plants mass where the case demands, and checks the
    instrument reaches the registered verdict.
    """
    cases: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: Any) -> None:
        cases.append({"case": name, "pass": bool(passed), "detail": detail})

    # --- layout ------------------------------------------------------------
    layout = _synthetic_window(n_sink=4, mount_n=10, recent_ntok=7, live_ntok=5)
    record(
        "layout_regions_partition_the_window",
        layout["regions"][REGION_ANCHOR] == {"start": 0, "end": 4, "ntok": 4}
        and layout["regions"][REGION_ARENA] == {"start": 4, "end": 14, "ntok": 10}
        and layout["regions"][REGION_RECENT] == {"start": 14, "end": 21, "ntok": 7}
        and layout["regions"][REGION_LIVE] == {"start": 21, "end": 26, "ntok": 5}
        and layout["window_tokens"] == 26,
        layout["regions"],
    )
    record(
        "layout_positional_law_is_separate_from_physical_rows",
        layout["positional_law"]["live_shift"] == 100
        and layout["positional_law"]["positional_hole_size"] == 86
        and layout["regions"][REGION_ARENA]["end"] == 14,
        layout["positional_law"],
    )
    empty_recent = _synthetic_window(
        n_sink=19, mount_n=63, recent_ntok=0, live_ntok=30)
    record(
        "layout_handles_an_empty_recent_region",
        empty_recent["regions"][REGION_RECENT]["ntok"] == 0
        and empty_recent["regions"][REGION_LIVE] == {
            "start": 82, "end": 112, "ntok": 30},
        empty_recent["regions"],
    )
    try:
        derive_window_layout(
            n_sink=4, arena_width=96, cur_mount_n=10,
            mount_seat_ranges={}, live_segs=[(0, 3), (1, 3), (2, 3)],
            live_turns=2)
        record("layout_rejects_unevicted_live_segments", False, "no raise")
    except LSRError as exc:
        record("layout_rejects_unevicted_live_segments", True, str(exc))

    # --- the REAL post-serve segment shape -------------------------------
    # After _attempt(deposit=False), arena.live_segs ends with the probe's own
    # turn carrying graft_index None (core/graft_arena.py:2752,2775).  That
    # trailing anonymous segment is the LIVE region; anything before it is
    # RECENT.  This is the shape the third shakedown actually produced.
    post = derive_window_layout(
        n_sink=19, arena_width=96, cur_mount_n=63,
        mount_seat_ranges={1: (19, 82)},
        live_segs=[(5, 30), (None, 37)], live_turns=2, prompt_ntok=0)
    record(
        "post_serve_trailing_anonymous_segment_is_the_LIVE_probe_turn",
        post["regions"][REGION_RECENT] == {"start": 82, "end": 112, "ntok": 30}
        and post["regions"][REGION_LIVE] == {"start": 112, "end": 149, "ntok": 37}
        and post["live_segments"][1].get("is_probe_turn") is True
        and post["live_segments"][0]["region"] == REGION_RECENT,
        {"regions": post["regions"],
         "segs": [(s["segment_index"], s["graft_index"], s["region"])
                  for s in post["live_segments"]]},
    )
    clean = derive_window_layout(
        n_sink=19, arena_width=96, cur_mount_n=63,
        mount_seat_ranges={1: (19, 82)},
        live_segs=[(None, 37)], live_turns=2, prompt_ntok=0)
    record(
        "clean_room_post_serve_leaves_recent_empty_and_live_the_probe_turn",
        clean["regions"][REGION_RECENT]["ntok"] == 0
        and clean["regions"][REGION_LIVE]["ntok"] == 37,
        clean["regions"],
    )
    earlier = derive_window_layout(
        n_sink=4, arena_width=96, cur_mount_n=6,
        mount_seat_ranges={0: (4, 10)},
        live_segs=[(None, 5), (7, 4)], live_turns=2, prompt_ntok=0)
    record(
        "an_earlier_anonymous_segment_stays_RECENT_not_the_probe_turn",
        earlier["regions"][REGION_RECENT]["ntok"] == 9
        and earlier["regions"][REGION_LIVE]["ntok"] == 0
        and all(s["region"] == REGION_RECENT
                for s in earlier["live_segments"]),
        [(s["segment_index"], s["graft_index"], s["region"])
         for s in earlier["live_segments"]],
    )
    pre = derive_window_layout(
        n_sink=19, arena_width=96, cur_mount_n=63,
        mount_seat_ranges={1: (19, 82)},
        live_segs=[], live_turns=2, prompt_ntok=37)
    record(
        "pre_serve_layout_still_takes_the_prompt_as_the_LIVE_region",
        pre["regions"][REGION_RECENT]["ntok"] == 0
        and pre["regions"][REGION_LIVE] == {"start": 82, "end": 119, "ntok": 37}
        and pre["live_segments"][0].get("is_probe_prompt") is True,
        pre["regions"],
    )

    # --- witness scope -----------------------------------------------------
    heavy_recent = list(range(14, 21))
    mass = full_window_readout_mass(
        softmax_operands_by_layer=_scores_for(layout, heavy_recent),
        layout=layout)
    record(
        "witness_finds_plurality_on_recent_when_mass_is_planted_there",
        mass["plurality_region"] == REGION_RECENT
        and mass["region_mass_share"][REGION_RECENT] > 0.9,
        {"plurality": mass["plurality_region"],
         "share": mass["region_mass_share"]},
    )
    record(
        "witness_region_shares_and_sink_sum_to_one",
        abs(sum(mass["region_mass_share"].values())
            + mass["attention_sink_mass_share"] - 1.0) < 1e-5,
        sum(mass["region_mass_share"].values()),
    )
    mass_arena = full_window_readout_mass(
        softmax_operands_by_layer=_scores_for(layout, list(range(4, 14))),
        layout=layout)
    record(
        "witness_finds_plurality_on_arena_when_mass_is_planted_there",
        mass_arena["plurality_region"] == REGION_ARENA,
        mass_arena["region_mass_share"],
    )
    try:
        narrow = {0: np.zeros(
            (2, 4, layout["regions"][REGION_ARENA]["ntok"]), dtype=np.float32)}
        full_window_readout_mass(softmax_operands_by_layer=narrow, layout=layout)
        record("witness_rejects_arena_only_capture", False, "no raise")
    except LSRError as exc:
        record("witness_rejects_arena_only_capture", True, str(exc))

    # --- attention sink ----------------------------------------------------
    sinks = {layer: np.full((4,), 8.0, dtype=np.float32) for layer in range(3)}
    mass_sink = full_window_readout_mass(
        softmax_operands_by_layer=_scores_for(layout, heavy_recent),
        layout=layout, sink_logits_by_layer=sinks)
    record(
        "attention_sink_column_absorbs_mass_and_is_reported_separately",
        mass_sink["attention_sink_mass_share"] > 0.0
        and mass_sink["region_mass_share"][REGION_RECENT]
        < mass["region_mass_share"][REGION_RECENT]
        and abs(sum(mass_sink["region_mass_share"].values())
                + mass_sink["attention_sink_mass_share"] - 1.0) < 1e-5,
        {"sink": mass_sink["attention_sink_mass_share"],
         "recent": mass_sink["region_mass_share"][REGION_RECENT]},
    )

    # --- sliding window ----------------------------------------------------
    allowed = np.zeros((2, layout["window_tokens"]), dtype=bool)
    allowed[:, 21:] = True
    masked = full_window_readout_mass(
        softmax_operands_by_layer=_scores_for(layout, heavy_recent),
        layout=layout,
        allowed_masks_by_layer={layer: allowed for layer in range(3)})
    record(
        "sliding_window_mask_excludes_hidden_regions_and_flags_visibility",
        masked["region_mass_share"][REGION_RECENT] == 0.0
        and masked["plurality_region"] == REGION_LIVE
        and all(row["region_visible"][REGION_RECENT] is False
                for row in masked["per_layer"]),
        masked["region_mass_share"],
    )

    # --- value localization ------------------------------------------------
    tokens = ["<s>", "conv", "ation", ">"]
    tokens += ["arena "] * 6 + ["Delta", "-4-", "Drift", " "]
    tokens += ["User", ": Cobalt", "-3-", "Comet", " noted", ".", " y"]
    tokens += ["Question", ":", " what", " is", " it"]
    tokens = (tokens + [" pad"] * 26)[:26]
    ids = list(range(len(tokens)))
    served_spans = locate_value_spans(
        layout=layout, token_ids=ids, token_strings=tokens,
        value="Cobalt-3-Comet", label="served")
    expected_spans = locate_value_spans(
        layout=layout, token_ids=ids, token_strings=tokens,
        value="Delta-4-Drift", label="expected")
    record(
        "value_spans_land_in_the_right_regions",
        len(served_spans) == 1 and served_spans[0]["region"] == REGION_RECENT
        and len(expected_spans) == 1
        and expected_spans[0]["region"] == REGION_ARENA,
        {"served": served_spans, "expected": expected_spans},
    )
    record(
        "separator_agnostic_but_token_count_load_bearing",
        bool(locate_value_spans(
            layout=layout, token_ids=ids, token_strings=tokens,
            value="Cobalt‑3‑Comet", label="x"))
        and not locate_value_spans(
            layout=layout, token_ids=ids, token_strings=tokens,
            value="Cobalt-2-Comet", label="x")
        and not locate_value_spans(
            layout=layout, token_ids=ids, token_strings=tokens,
            value="Cobalt Comet", label="x"),
        "u2011 matches; wrong token and missing token do not",
    )

    # --- per-probe verdicts -------------------------------------------------
    mass_on_served = full_window_readout_mass(
        softmax_operands_by_layer=_scores_for(layout, list(range(
            served_spans[0]["token_start"], served_spans[0]["token_end"]))),
        layout=layout)
    verdict_recent = probe_verdict(
        probe_id="synthetic_recent", served_value="Cobalt-3-Comet",
        expected_value="Delta-4-Drift", layout=layout, mass=mass_on_served,
        served_spans=served_spans, expected_spans=expected_spans)
    record(
        "verdict_RECENT_TURNS_when_served_value_sits_in_recent_with_mass",
        verdict_recent["verdict"] == VERDICT_RECENT_TURNS
        and verdict_recent["h_lsr_1_leg_satisfied"] is True,
        verdict_recent["verdict_basis"],
    )

    mass_on_arena = full_window_readout_mass(
        softmax_operands_by_layer=_scores_for(layout, list(range(
            expected_spans[0]["token_start"], expected_spans[0]["token_end"]))),
        layout=layout)
    verdict_arena = probe_verdict(
        probe_id="synthetic_arena", served_value="Delta-4-Drift",
        expected_value="Delta-4-Drift", layout=layout, mass=mass_on_arena,
        served_spans=expected_spans, expected_spans=expected_spans)
    record(
        "verdict_ARENA_when_served_value_sits_in_arena_with_mass",
        verdict_arena["verdict"] == VERDICT_ARENA
        and verdict_arena["h_lsr_1_leg_satisfied"] is False,
        verdict_arena["verdict_basis"],
    )

    verdict_absent = probe_verdict(
        probe_id="synthetic_absent", served_value="Nowhere-9-Zulu",
        expected_value="Delta-4-Drift", layout=layout, mass=mass_on_arena,
        served_spans=[], expected_spans=expected_spans)
    record(
        "verdict_ABSENT_when_served_value_is_nowhere_in_the_window",
        verdict_absent["verdict"] == VERDICT_ABSENT,
        verdict_absent["verdict_basis"],
    )

    anchor_tokens = list(tokens)
    anchor_tokens[1] = " Zulu-1-Anchor "
    anchor_spans = locate_value_spans(
        layout=layout, token_ids=ids, token_strings=anchor_tokens,
        value="Zulu-1-Anchor", label="served")
    verdict_unmeasured = probe_verdict(
        probe_id="synthetic_anchor_only", served_value="Zulu-1-Anchor",
        expected_value="Delta-4-Drift", layout=layout, mass=mass_on_arena,
        served_spans=anchor_spans, expected_spans=expected_spans)
    record(
        "anchor_only_provenance_reports_NOT_MEASURED_not_a_new_verdict",
        verdict_unmeasured["verdict"] == VERDICT_NOT_MEASURED
        and bool(anchor_spans)
        and anchor_spans[0]["region"] == REGION_ANCHOR,
        verdict_unmeasured["verdict_basis"],
    )

    # --- registered adjudication -------------------------------------------
    def leg(probe_id: str, satisfied: bool) -> dict[str, Any]:
        return {
            "probe_id": probe_id,
            "verdict": VERDICT_RECENT_TURNS if satisfied else VERDICT_ARENA,
            "h_lsr_1_leg_satisfied": satisfied,
        }

    four = adjudicate_h_lsr_1([leg(f"p{i}", i < 4) for i in range(6)])
    three = adjudicate_h_lsr_1([leg(f"p{i}", i < 3) for i in range(6)])
    short = adjudicate_h_lsr_1([leg(f"p{i}", True) for i in range(4)])
    record(
        "adjudication_convicts_at_exactly_four_of_six",
        four["status"] == "CONVICTED" and four["h_lsr_1_convicted"] is True,
        four["status"])
    record(
        "adjudication_does_not_convict_at_three_of_six",
        three["status"] == "NOT_CONVICTED"
        and three["h_lsr_1_convicted"] is False,
        three["status"])
    record(
        "short_measurement_is_NOT_MEASURED_never_a_failure_to_convict",
        short["status"] == VERDICT_NOT_MEASURED
        and short["h_lsr_1_convicted"] is None,
        short["status"])

    # --- probe enumeration --------------------------------------------------
    try:
        enumerated = enumerate_probes()
        ok = (
            len(enumerated["wrong_value_probes"]) == 6
            and len(enumerated["lawful_controls"]) == 2
            and all(row["single_mount"] and row["mount_count"] == 1
                    for row in enumerated["wrong_value_probes"])
            and all(row["served_value"] for row in
                    enumerated["wrong_value_probes"])
        )
        record("probe_enumeration_matches_the_census", ok, {
            "wrong": [row["probe_id"] for row in enumerated["wrong_value_probes"]],
            "controls": [row["probe_id"] for row in enumerated["lawful_controls"]],
        })
    except Exception as exc:  # noqa: BLE001 - selftest reports, never dies
        record("probe_enumeration_matches_the_census", False, repr(exc))

    # --- session inventory --------------------------------------------------
    try:
        inv = value_presence_across_session(
            "sup_reserve_meridian_docket", "Cobalt-3-Comet")
        record(
            "session_inventory_locates_a_served_value_in_its_session_nodes",
            inv["present_in_any_node"] and inv["carrying_graft_indices"] == [2],
            inv["carrying_graft_indices"])
    except Exception as exc:  # noqa: BLE001
        record(
            "session_inventory_locates_a_served_value_in_its_session_nodes",
            False, repr(exc))

    passed = sum(1 for row in cases if row["pass"])
    return {
        "schema": f"{SCHEMA_PREFIX}.cpu_selftest.v1",
        "cases": cases,
        "case_count": len(cases),
        "passed": passed,
        "failed": len(cases) - passed,
        "all_passed": passed == len(cases),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LSR Phase 0 CPU core")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--enumerate", action="store_true")
    parser.add_argument("--emit", action="store_true",
                        help="write a content-addressed receipt")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not (args.selftest or getattr(args, "enumerate")):
        args.selftest = True
    payload: dict[str, Any] = {
        "schema": f"{SCHEMA_PREFIX}.cpu_receipt.v1",
        "program": "LSR",
        "phase": "0",
        "plan": file_record(PLAN),
        "sources": {"lsr_p0_core": file_record(Path(__file__).resolve())},
        "declares_in_det_envelope": False,
        "provenance_note": (
            "LSR is a new program; DET/CMC inputs appear only as inbound file "
            "records.  Nothing here is declared in any DET envelope."
        ),
    }
    if getattr(args, "enumerate"):
        payload["probe_enumeration"] = enumerate_probes()
    if args.selftest:
        payload["cpu_selftest"] = cpu_selftest()
    blob = canonical_json_bytes(payload)
    if args.emit:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        path = ARTIFACT_DIR / f"lsr_p0_cpu_{sha256_bytes(blob)[:16]}.json"
        if path.exists():
            if path.read_bytes() != blob:
                raise LSRError(f"content-address collision: {path}")
        else:
            with path.open("xb") as handle:
                handle.write(blob)
        print(f"receipt {path}")
    else:
        print(blob.decode("utf-8"))
    selftest = payload.get("cpu_selftest")
    if selftest is not None and not selftest["all_passed"]:
        for row in selftest["cases"]:
            if not row["pass"]:
                print(f"FAIL {row['case']}: {row['detail']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
