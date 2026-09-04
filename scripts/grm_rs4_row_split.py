#!/usr/bin/env python3
"""GRM-RS4 — the live-band row split: pure arithmetic, CPU-testable.

ORDER: ``orders/GRM_RS4_CEILING_REPARTITION.md`` (mission 1, gate G1).
REGISTRATION: ``artifacts/grm_rs4/registration.json``
(``live_band_split_REGISTERED_BEFORE_ANY_GATE``).

WHY THIS IS A SEPARATE MODULE.  The order's G1 asks for "new CPU tests for the
row-split arithmetic (bands sum to one; boundaries derived)".  Everything in
here is a pure function over integers and an already-computed per-key-row
probability array, so the whole partition can be pinned by tests that touch no
GPU, no model and no arena.  The GPU module imports these functions rather than
re-implementing them, so the thing the tests pin IS the thing the arms run.

THE PARTITION.  ``core/grm_demand.py::_full_mass`` reads the softmaxed
probabilities for ONE query row over ``S`` physical key rows plus one learned
sink column at index ``S``, and sums them into four bands::

    [0, n_sink)                        physical sink
    [n_sink, n_sink + cur_mount_n)     mount
    [n_sink + cur_mount_n, S)          LIVE
    column S                           learned sink

RS4 leaves the first, second and fourth exactly as they are and cuts the third
into four sub-bands, in physical row order::

    prior_live   rows the live cache already held when this serve began,
                 minus whatever this arm's own feed contributed
    fed_text     the rows this arm's _feed_live pushed
    question     _attempt's own prompt rows
    answer       the remainder, [.., S)

Every width is DERIVED (see ``live_band_bounds``); ``answer`` is a remainder so
the four sub-bands partition the live band by construction.  A negative or
overrunning width is raised, never clamped: it would mean the derivation had
drifted from the cache the kernel actually scored, and a clamp would hide
exactly that.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.grm_rs4_registration import (  # noqa: E402
    BAND_NAMES, LIVE_SUBBANDS, RS4Error,
)

#: The band names ``_full_mass`` itself reports, unchanged.  RS4 adds sub-bands
#: beside them; it renames and re-scales nothing.
BASE_NAMES = ("physical_sink_mass", "mounted_mass", "live_mass",
              "learned_sink_mass")

#: The names an RS4 layer row carries: the four base bands plus the four
#: sub-bands.  ``live_mass`` is kept so every receipt can show that the
#: sub-bands add back to it.
ROW_NAMES = tuple(BASE_NAMES) + tuple(LIVE_SUBBANDS)

#: Float tolerance for "the sub-bands add back to the live band".  The
#: sub-band sum and the live-band sum are the SAME float additions in a
#: different association order, so the only admissible difference is the last
#: bit or two of a float32 accumulation.
SUBBAND_SUM_ATOL = 1e-6

#: The partition tolerance ``grm_demand._summarize_mass`` enforces, carried
#: rather than re-chosen so an RS4 row and a production demand row are judged
#: by one rule.
PARTITION_LO = 0.98
PARTITION_HI = 1.02


def live_band_bounds(
    *,
    S: int,
    n_sink: int,
    cur_mount_n: int,
    prior_ntok: int,
    fed_ntok: int,
    question_ntok: int,
) -> dict[str, Any]:
    """The four live sub-bands as absolute ``[lo, hi)`` physical row ranges.

    EVERY BOUNDARY IS COMPUTED FROM AN OBSERVED QUANTITY.  ``S`` is the key-row
    count the kernel was handed; ``n_sink`` and ``cur_mount_n`` are the arena's
    own attributes (the two ``_full_mass`` reads to place the mount band);
    ``prior_ntok`` and ``fed_ntok`` come off the arena's live-token counter and
    the arm's feed receipt; ``question_ntok`` comes from the model tokenizer
    over ``_attempt``'s own prompt string.  Nothing here is a literal row index.

    The ANSWER band is the remainder rather than a count, which is what makes
    the four sub-bands a partition of ``[live0, S)`` by construction: the only
    way they can fail to cover it is for one of the leading widths to overrun,
    and that raises.
    """
    S = int(S)
    n_sink = int(n_sink)
    cur_mount_n = int(cur_mount_n)
    prior_ntok = int(prior_ntok)
    fed_ntok = int(fed_ntok)
    question_ntok = int(question_ntok)

    for name, value in (("S", S), ("n_sink", n_sink),
                        ("cur_mount_n", cur_mount_n),
                        ("prior_ntok", prior_ntok), ("fed_ntok", fed_ntok),
                        ("question_ntok", question_ntok)):
        if value < 0:
            raise RS4Error(f"row-split input {name}={value} is negative")

    live0 = n_sink + cur_mount_n
    if live0 > S:
        raise RS4Error(
            f"live band starts at {live0} but the kernel scored only {S} key "
            "rows: n_sink + cur_mount_n overruns the cache")

    prior_hi = live0 + prior_ntok
    fed_hi = prior_hi + fed_ntok
    question_hi = fed_hi + question_ntok
    if question_hi > S:
        raise RS4Error(
            "the derived live sub-bands overrun the cache: "
            f"live0={live0} prior={prior_ntok} fed={fed_ntok} "
            f"question={question_ntok} ends at {question_hi} but S={S}. "
            "The derivation has drifted from the rows the kernel scored; "
            "this is a hard failure, not something to clamp.")

    return {
        "S": S,
        "n_sink": n_sink,
        "cur_mount_n": cur_mount_n,
        "live0": int(live0),
        "sink_band": [0, int(min(n_sink, S))],
        "mount_band": [int(min(n_sink, S)), int(min(live0, S))],
        "live_band": [int(live0), S],
        "prior_live_band": [int(live0), int(prior_hi)],
        "fed_text_band": [int(prior_hi), int(fed_hi)],
        "question_band": [int(fed_hi), int(question_hi)],
        "answer_band": [int(question_hi), S],
        "prior_live_rows": int(prior_ntok),
        "fed_text_rows": int(fed_ntok),
        "question_rows": int(question_ntok),
        "answer_rows": int(S - question_hi),
        "live_rows": int(S - live0),
        "bands_are_contiguous_and_cover_live": True,
    }


def split_full_row(values: Any, bounds: Mapping[str, Any]) -> dict[str, Any]:
    """One FULL-attention layer's bands, from its per-key-row probabilities.

    ``values`` has shape ``(heads, S + 1)`` — exactly the array
    ``_full_mass`` builds and then sums, with the learned sink at column ``S``.
    The per-head sums are averaged over heads, which is ``_full_mass``'s own
    reduction, so the three bands RS4 does not touch come out bit-identical to
    the production numbers and the four sub-bands are the same additions the
    production ``live`` sum performs, merely stopped at three interior points.
    """
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise RS4Error(
            f"expected a (heads, S+1) probability array, got shape "
            f"{values.shape}")
    S = int(bounds["S"])
    if int(values.shape[1]) != S + 1:
        raise RS4Error(
            f"probability array has {values.shape[1]} columns but the bounds "
            f"say S={S}, so the learned sink column would be at {S}")

    def band(name: str) -> float:
        lo, hi = (int(v) for v in bounds[name])
        if hi <= lo:
            return 0.0
        return float(np.mean(values[:, lo:hi].sum(axis=1)))

    row: dict[str, Any] = {
        "physical_sink_mass": band("sink_band"),
        "mounted_mass": band("mount_band"),
        "live_mass": band("live_band"),
        "learned_sink_mass": float(np.mean(values[:, S])),
        "prior_live_mass": band("prior_live_band"),
        "fed_text_mass": band("fed_text_band"),
        "question_mass": band("question_band"),
        "answer_mass": band("answer_band"),
    }
    row.update({f"{key}_rows": int(bounds[f"{key}_rows"])
                for key in ("prior_live", "fed_text", "question", "answer",
                            "live")})
    return row


def subbands_sum_to_live(row: Mapping[str, Any],
                         atol: float = SUBBAND_SUM_ATOL) -> bool:
    """Do the four sub-bands add back to the live band the base tool reports?

    This is the check that makes the split trustworthy: it says the sub-bands
    are a RE-PARTITION of the very number RS3 published, not a second, parallel
    computation that happens to look similar.  Pure — CPU tested.
    """
    live = row.get("live_mass")
    if live is None:
        return False
    total = sum(float(row.get(name) or 0.0) for name in LIVE_SUBBANDS)
    return bool(abs(float(live) - total) <= float(atol))


def partition_sums_to_one(row: Mapping[str, Any],
                          lo: float = PARTITION_LO,
                          hi: float = PARTITION_HI) -> bool:
    """Sink + mount + the four sub-bands + learned sink ~= 1.0.

    Uses ``BAND_NAMES`` — the registered vocabulary, which deliberately does
    NOT include ``live_mass``, because counting the live band AND its own four
    sub-bands would double the live mass.  Pure — CPU tested.
    """
    values = [row.get(name) for name in BAND_NAMES]
    if any(value is None for value in values):
        return False
    total = float(sum(float(v) for v in values))
    return bool(float(lo) <= total <= float(hi))


def mean_over_positions(
    steps: Sequence[Sequence[Mapping[str, Any]]],
    names: Sequence[str] = ROW_NAMES,
) -> dict[str, Any]:
    """Mean over answer positions of the per-layer mean, for one layer type.

    The SAME two-level reduction
    ``grm_rs2_mount_read_gpu._summarize_layer_type`` performs (mean over layers
    inside a step, then mean over steps), applied to the extended name set.
    Kept structurally identical so an RS4 headline and an RS3 headline are the
    same statistic over the same rows.  Pure — CPU tested.
    """
    kept = [list(step) for step in steps if step]
    if not kept:
        return {
            "layers_observed": 0,
            "answer_positions": 0,
            "mean_over_answer_positions": {name: None for name in names},
            "per_layer_mean": [],
        }
    width = len(kept[0])
    for step in kept:
        if len(step) != width:
            raise RS4Error(
                "layer capture count drifted between answer positions "
                f"({width} then {len(step)})")
    mean = {
        name: float(np.mean([
            float(np.mean([float(row[name]) for row in step]))
            for step in kept]))
        for name in names
    }
    per_layer: list[dict[str, Any]] = []
    for ordinal in range(width):
        entry: dict[str, Any] = {
            name: float(np.mean([float(step[ordinal][name]) for step in kept]))
            for name in names
        }
        entry["layer_ordinal"] = int(ordinal)
        per_layer.append(entry)
    return {
        "layers_observed": int(width),
        "answer_positions": len(kept),
        "mean_over_answer_positions": mean,
        "per_layer_mean": per_layer,
        "subbands_sum_to_live": bool(subbands_sum_to_live(mean)),
        "partition_sums_to_one": bool(partition_sums_to_one(mean)),
        "partition_sum": float(sum(
            float(mean[name]) for name in BAND_NAMES)),
    }
