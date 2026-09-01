#!/usr/bin/env python3
"""LSR Phase 0 — GPU leg: full-window witness at answer-readout positions.

Program: LSR (Lived-Serving Reliability).  Spec: docs/LSR_LIVED_SERVING_PLAN.md

INVOCATION (bare, from the repo root; the lead shell wraps this):
    PYTHONPATH=/mnt/ForgeRealm/GraftRepository:/mnt/ForgeRealm/Project-Tensor/tensor_cuda \
        python3 scripts/lsr_p0_gpu.py witness --probe <probe_id> \
            --lease-seconds 580 --lock-wait-seconds 7200

GPU DISCIPLINE (standing house rules, enforced here, not merely documented):
  * self-lease via scripts.grm_cmc1_gpu_arms.gpu_lease on /tmp/forge-gpu.lock;
    the operator has absolute right of way;
  * <= 580 s per lease (MAX_LEASE_SECONDS 590 is the hard cap);
  * one probe per process, one lease per probe, so a long battery is many
    short leases rather than one long one;
  * the caller inserts a 30 s inter-process gap (the lead shell does).

WHAT IT MEASURES.  For one probe it re-serves the lived probe as the DET1 race
did, taps the engine's own softmax operands at the answer-readout positions,
and runs the offline fp32 witness over the FULL window -- anchor + arena +
recent-turns + live -- not just the arena.

THE TAP (CMC1.2 / DET1.3 EXPORT pattern, adapted to GPT-OSS).
grm_cmc1_gpu_arms taps MiniCPM3's tc.causal_softmax.  GPT-OSS instead routes
through core.gpt_oss20b_tc.sink_attention_tc / sliding_sink_attention_tc,
which concatenate a per-head SINK LOGIT column, softmax over the
concatenation, then drop that column (core/gpt_oss20b_tc.py::sink_attention_tc).
This module wraps those two functions -- the same seam DetectorObserver
already uses -- and EXPORTS their operands to host:

    Q             the last query row of each forward (the readout position),
                  pulled to host with .float().numpy()
    K             the key bank the call was handed, likewise exported
    sinks         the per-head sink logits, so the denominator matches
    allowed mask  for sliding layers, which keys were attendable at all
    scale         the engine's own `scale=` keyword, never re-derived

Scores are then recomputed OFFLINE in numpy fp32 (SinkAttentionTap.stacked /
_recompute_row), which is precisely
grm_cmc1_mechanism.materialized_softmax's contraction.

WHY EXPORT RATHER THAN COMPUTE IN-ENGINE.  An earlier revision issued its own
`tc.matmul` inside the tap to materialize scores on device.  That touched
tensors whose lifetime and layout the engine owns and died with "CUDA error
at to_host: an illegal memory access was encountered".  Exporting operands
and doing the arithmetic offline removes that failure class outright.  It
also moots the G2-style equivalence question that the in-engine approach
would have raised: offline fp32 math over engine-exported operands IS the
established instrument in this lineage, not an approximation of it.

Only the LAST query row of each forward is kept: that is the answer-readout
position (core/graft_arena.py::_attempt -- prefill's last row predicts token 0,
then one decode forward per subsequent token).  The trailing cache-commit
forward is dropped exactly as scripts/grm_det1_common.DetectorObserver.finish
drops it, because its logits are unused and it is not an answer position.

PROVENANCE: LSR declares NOTHING in any DET envelope.  Receipts are
content-addressed under artifacts/lsr_p0/ with their own grm.lsr_p0.* schemas;
DET inputs appear only as inbound file records.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.grm_cmc1_mechanism import canonical_json_bytes, sha256_bytes  # noqa: E402
from scripts.grm_det1_common import file_record  # noqa: E402
from scripts.lsr_p0_core import (  # noqa: E402
    ARTIFACT_DIR,
    CENSUS,
    PLAN,
    SCHEMA_PREFIX,
    VERDICT_NOT_MEASURED,
    LSRError,
    adjudicate_h_lsr_1,
    derive_window_layout,
    enumerate_probes,
    full_window_readout_mass,
    locate_value_spans,
    probe_session_map,
    probe_verdict,
)

LEASE_SECONDS = 580
LOCK_WAIT_SECONDS = 7200
RUNTIME_FRAME = (
    ROOT / "artifacts" / "grm_det1" / "run_20260831T160525Z_2"
    / "runtime_frame_28b3196f8fb04a41.json"
)
SUP_FIXTURES = ROOT / "tests" / "fixtures" / "supersession_battery"


class SinkAttentionTap:
    """Capture GPT-OSS softmax operands at answer-readout positions.

    Wraps core.gpt_oss20b_tc.sink_attention_tc and sliding_sink_attention_tc
    -- the same seam scripts/grm_det1_common.DetectorObserver wraps -- so the
    engine's own arithmetic is untouched: the wrapper records its inputs and
    delegates to the original function for the actual compute.

    LAYER IDENTITY -- deliberately NOT a call ordinal.  grm_cmc1_gpu_arms
    derives the layer from `calls % n_layers`, which is sound for MiniCPM3
    because its SDPA site fires exactly once per layer.  It is NOT sound here:
    core/gpt_oss20b_tc.py::sliding_sink_attention_tc calls sink_attention_tc
    ONCE PER CHUNK internally (:419), so a shared counter sees several calls
    for one sliding layer and desynchronizes for every layer after it.

    Instead the tap binds each attention module's `sinks` tensor identity to
    its `layer_idx` up front (every GptOssAttentionTC owns its own sinks
    tensor, :668), and each intercepted call is attributed by looking up the
    sinks object it was handed.  Nested and chunked calls therefore land on
    the right layer, and a call whose sinks are unrecognized raises rather
    than being silently misfiled.

    SCALE: read from the engine's own `scale=` keyword (GptOssAttentionTC
    passes `scale=self.scaling`), never re-derived from head_dim, so a future
    change to `scaling` cannot silently desynchronize the witness.

    DEVICE DISCIPLINE: this tap performs NO device arithmetic.  It slices and
    exports operands with `.float().numpy()` and nothing else; every score is
    recomputed offline in numpy fp32.  Nothing device-resident is retained
    past the call that produced it.
    """

    def __init__(self, layers: Sequence[Any], *, head_dim: int):
        self.head_dim = int(head_dim)
        self.n_layers = len(layers)
        self.forward_count = 0
        # Exported host-side operands per layer, one entry per readout row:
        # {q, k, scale, key_offset, width}.  No device tensors are retained.
        self.operands: dict[int, list[dict[str, Any]]] = {}
        self.sinks: dict[int, np.ndarray] = {}
        self.allowed: dict[int, list[np.ndarray]] = {}
        self.layer_kind: dict[int, str] = {}
        self._pending: dict[int, tuple[int, bool]] = {}
        self._seen_this_forward: set[int] = set()
        self._by_sinks: dict[int, int] = {}
        for layer in layers:
            attn = getattr(layer, "self_attn", None)
            if attn is None or getattr(attn, "sinks", None) is None:
                raise LSRError(
                    "layer exposes no self_attn.sinks; the tap cannot bind "
                    "layer identity")
            self._by_sinks[id(attn.sinks)] = int(attn.layer_idx)
        if len(self._by_sinks) != self.n_layers:
            raise LSRError(
                f"{self.n_layers} layers share only {len(self._by_sinks)} "
                "distinct sinks tensors; layer identity would be ambiguous")
        self._module = None
        self._orig_full = None
        self._orig_sliding = None

    def __enter__(self) -> "SinkAttentionTap":
        from core import gpt_oss20b_tc as gpt

        self._module = gpt
        self._orig_full = gpt.sink_attention_tc
        self._orig_sliding = gpt.sliding_sink_attention_tc

        def wrapped_full(query, key, value, sinks, **kwargs):
            # Chunks of a sliding layer re-enter here.  `_pending` carries the
            # chunk's absolute key placement, set by wrapped_sliding just
            # before it delegates; a bare full-attention call has none.
            layer = self._layer_of(sinks)
            pending = self._pending.pop(layer, None)
            if pending is None:
                self._record(
                    query, key, sinks, kind="full", window=None,
                    scale=kwargs.get("scale"),
                    attention_mask=kwargs.get("attention_mask"))
            else:
                total_keys, last_chunk = pending
                # Only the chunk holding the final query row is an answer
                # readout; earlier chunks are skipped, not folded.
                if last_chunk:
                    offset = int(total_keys) - int(key.shape[2])
                    self._record(
                        query, key, sinks, kind="sliding",
                        window=None, scale=kwargs.get("scale"),
                        key_offset=max(0, offset), total_keys=int(total_keys),
                        attention_mask=kwargs.get("attention_mask"))
            return self._orig_full(query, key, value, sinks, **kwargs)

        def wrapped_sliding(query, key, value, sinks, *, sliding_window, **kwargs):
            # Delegate the actual compute; the inner per-chunk
            # sink_attention_tc calls carry the operands and wrapped_full
            # records them.  The chunk that contains the LAST query row is the
            # readout: sliding_sink_attention_tc walks i in range(0, L, blk),
            # so that is the final iteration.
            layer = self._layer_of(sinks)
            self.layer_kind[layer] = "sliding"
            total_keys = int(key.shape[2])
            length = int(query.shape[2])
            blk = max(1, int(kwargs.get("attn_block", 128)))
            starts = list(range(0, length, blk))
            seen = {"n": 0}
            orig_full = self._orig_full

            def chunk_probe(q_i, k_i, v_i, s_i, **kw):
                seen["n"] += 1
                self._pending[layer] = (
                    total_keys, seen["n"] == len(starts))
                return wrapped_full(q_i, k_i, v_i, s_i, **kw)

            gpt.sink_attention_tc = chunk_probe
            try:
                return self._orig_sliding(
                    query, key, value, sinks,
                    sliding_window=sliding_window, **kwargs)
            finally:
                gpt.sink_attention_tc = wrapped_full
                self._pending.pop(layer, None)
                del orig_full

        gpt.sink_attention_tc = wrapped_full
        gpt.sliding_sink_attention_tc = wrapped_sliding
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._module is not None:
            if self._orig_full is not None:
                self._module.sink_attention_tc = self._orig_full
            if self._orig_sliding is not None:
                self._module.sliding_sink_attention_tc = self._orig_sliding
        return False

    def _layer_of(self, sinks) -> int:
        layer = self._by_sinks.get(id(sinks))
        if layer is None:
            raise LSRError(
                "intercepted a sink attention call whose sinks tensor belongs "
                "to no known layer; layer attribution would be wrong")
        return int(layer)

    def _record(self, query, key, sinks, *, kind: str, window: int | None,
                scale: Any, key_offset: int = 0, total_keys: int | None = None,
                attention_mask=None) -> None:
        """Record the LAST query row's scores for one (possibly chunked) call.

        `key_offset`/`total_keys` place a sliding chunk's columns back into the
        full key axis.  A chunk that does not contain the final query row is
        skipped: only the last row is an answer-readout position, and for the
        chunked path that row lives in the final chunk alone.

        EXPORT-ONLY (CMC1.2 / DET1.3 lineage).  Nothing is computed on the
        device here.  The tap does exactly what
        grm_cmc1_gpu_arms.SDPAInterceptor._capture does -- slice the operand
        it wants and pull it to host with `.float().numpy()` -- and the score
        recompute happens offline in numpy fp32 (see `stacked`).  An earlier
        revision issued its own `tc.matmul` at this point to materialize
        scores in-engine; that touched device tensors whose lifetime and
        layout the engine owns and died with an illegal memory access at
        to_host.  Exporting operands and recomputing offline removes that
        entire failure class, and it is the established instrument rather
        than a workaround: grm_cmc1_mechanism.materialized_softmax is defined
        as fp32 numpy math over exported Q/K.
        """
        layer = self._layer_of(sinks)
        self.layer_kind.setdefault(layer, kind)
        if layer not in self._seen_this_forward:
            if not self._seen_this_forward:
                self.forward_count += 1
            self._seen_this_forward.add(layer)
            if len(self._seen_this_forward) == self.n_layers:
                self._seen_this_forward = set()

        width = int(total_keys if total_keys is not None else key.shape[2])
        # Q: the LAST query row only -- the answer-readout position.
        rows = int(query.shape[2])
        q_last = query.slice(2, rows - 1, 1)
        q_np = np.asarray(
            q_last.float().numpy(), dtype=np.float32)[0, :, 0, :]  # (Hq,D)
        # K: the whole bank this call was handed (a chunk's slice for the
        # sliding path), exported once per call.
        k_np = np.asarray(
            key.float().numpy(), dtype=np.float32)[0]               # (Hkv,S,D)

        allowed = np.zeros(width, dtype=bool)
        allowed[key_offset:key_offset + k_np.shape[1]] = True
        if attention_mask is not None:
            # sliding chunks carry an additive 0 / -1e4 mask; -1e4 entries are
            # keys the engine excluded, not keys with genuine tiny mass.
            additive = np.asarray(
                attention_mask.float().numpy(), dtype=np.float32)
            last_row = additive.reshape(
                additive.shape[-2], additive.shape[-1])[-1]
            keep = last_row > -1.0e3
            allowed[key_offset:key_offset + k_np.shape[1]] &= keep

        self.operands.setdefault(layer, []).append({
            "q": q_np,
            "k": k_np,
            "scale": (float(scale) if scale is not None
                      else float(self.head_dim ** -0.5)),
            "key_offset": int(key_offset),
            "width": int(width),
        })
        self.allowed.setdefault(layer, []).append(allowed)
        if layer not in self.sinks:
            self.sinks[layer] = np.asarray(
                sinks.float().numpy(), dtype=np.float32).reshape(-1)

    def finalize(self, ngen: int) -> None:
        """Drop the trailing cache-commit forward, per DetectorObserver.finish.

        ArenaCache._attempt runs one extra forward solely to commit the final
        predicted token to KV; its logits are unused and it is not an answer
        readout position.
        """
        if self.forward_count == int(ngen) + 1:
            for store in (self.operands, self.allowed):
                for layer in list(store):
                    if store[layer]:
                        store[layer].pop()

    def stacked(self) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray],
                               dict[int, np.ndarray]]:
        """Pad each layer's rows to the final key length and stack them.

        Successive decode forwards see a growing key axis (one more row each
        step).  Columns beyond a given readout's key length did not exist for
        it, so they are marked NOT allowed rather than filled with a fake
        score -- the witness then excludes them instead of counting them as
        genuine near-zero mass.
        """
        scores: dict[int, np.ndarray] = {}
        allowed: dict[int, np.ndarray] = {}
        for layer, records in self.operands.items():
            if not records:
                continue
            width = max(int(row["width"]) for row in records)
            rows, masks = [], []
            for index, row in enumerate(records):
                rows.append(self._recompute_row(row, width))
                mask = np.zeros(width, dtype=bool)
                have = int(self.allowed[layer][index].shape[0])
                mask[:have] = self.allowed[layer][index][:have]
                masks.append(mask)
            scores[layer] = np.stack(rows, axis=0)
            allowed[layer] = np.stack(masks, axis=0)
        return scores, dict(self.sinks), allowed

    def _recompute_row(self, record: Mapping[str, Any], width: int) -> np.ndarray:
        """Offline fp32 scaled-QK for one readout row, from exported operands.

        This is grm_cmc1_mechanism.materialized_softmax's contraction --
        `np.einsum("hqd,hsd->hqs", q, k) * scale` -- specialized to a single
        query row and carrying GPT-OSS's GQA expansion.  All numpy, all host
        memory: no device tensor is touched, which is the whole point of the
        pivot.

        GQA: the engine expands KV heads with core.mistral7b_tc._repeat_kv,
        "(B,KVH,L,D) -> (B,KVH*rep,L,D), each kv head repeated rep times".
        np.repeat(..., axis=0) reproduces that contiguous repetition exactly,
        so query head h pairs with the same KV head the engine used.
        """
        q = np.asarray(record["q"], dtype=np.float32)      # (Hq,D)
        k = np.asarray(record["k"], dtype=np.float32)      # (Hkv,S,D)
        if q.shape[-1] != k.shape[-1]:
            raise LSRError(
                f"exported Q/K head dims differ: {q.shape} vs {k.shape}")
        heads_q, heads_kv = int(q.shape[0]), int(k.shape[0])
        if heads_q % heads_kv:
            raise LSRError(
                f"{heads_q} query heads do not divide over {heads_kv} KV heads")
        rep = heads_q // heads_kv
        if rep > 1:
            k = np.repeat(k, rep, axis=0)
        row = np.einsum("hd,hsd->hs", q, k, dtype=np.float32)
        row *= np.float32(record["scale"])
        # Absent columns are EXCLUDED (-inf + a False allowed mask), never
        # counted as genuine near-zero mass.
        out = np.full((heads_q, width), -np.inf, dtype=np.float32)
        offset = int(record["key_offset"])
        out[:, offset:offset + row.shape[1]] = row
        return out


def tap_selftest() -> dict[str, Any]:
    """Exercise SinkAttentionTap against a fake engine, on CPU.

    Covers what the two live shakedowns caught and what they would have caught
    next: model/layer attribute paths, layer attribution under the NESTED
    chunked sliding path, engine-supplied scale, mask handling, GQA head
    pairing, and the offline recompute's agreement with the reference
    contraction.

    It monkeypatches core.gpt_oss20b_tc's two attention functions with numpy
    stand-ins, so no CUDA and no weights are needed.  Note that NO fake `tc`
    module is installed: after the export pivot the tap performs no device
    arithmetic at all, so if any `tc.*` call crept back in, these tests would
    fail on the real (CUDA-requiring) import rather than silently passing
    against a stand-in.  That absence is deliberate coverage.
    """
    cases: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: Any) -> None:
        cases.append({"case": name, "pass": bool(passed), "detail": detail})

    import types

    from core import gpt_oss20b_tc as gpt

    class FakeTensor:
        def __init__(self, array):
            self.a = np.asarray(array, dtype=np.float32)

        @property
        def shape(self):
            return self.a.shape

        @property
        def dtype(self):
            return "float32"

        def slice(self, dim, start, length):
            index = [slice(None)] * self.a.ndim
            index[dim] = slice(int(start), int(start) + int(length))
            return FakeTensor(self.a[tuple(index)])

        def reshape(self, shape):
            return FakeTensor(self.a.reshape([int(v) for v in shape]))

        def float(self):
            return self

        def numpy(self):
            return self.a

    heads, head_dim = 2, 4
    n_layers = 3
    sinks_by_layer = [FakeTensor(np.full((heads,), 0.5 * (i + 1)))
                      for i in range(n_layers)]

    layers = []
    for index in range(n_layers):
        attn = types.SimpleNamespace(
            layer_idx=index, sinks=sinks_by_layer[index])
        layers.append(types.SimpleNamespace(self_attn=attn))

    orig_full = gpt.sink_attention_tc
    orig_sliding = gpt.sliding_sink_attention_tc

    def stub_full(query, key, value, sinks, **kwargs):
        return FakeTensor(np.zeros_like(query.a))

    def stub_sliding(query, key, value, sinks, *, sliding_window, **kwargs):
        length = int(query.shape[2])
        blk = max(1, int(kwargs.get("attn_block", 128)))
        total = int(key.shape[2])
        for i in range(0, length, blk):
            end = min(i + blk, length)
            q_abs0, q_abs1 = total - length + i, total - length + end - 1
            k0 = max(0, q_abs0 - int(sliding_window) + 1)
            k1 = min(total, q_abs1 + 1)
            mask = np.zeros((1, 1, end - i, k1 - k0), dtype=np.float32)
            gpt.sink_attention_tc(
                query.slice(2, i, end - i), key.slice(2, k0, k1 - k0),
                value.slice(2, k0, k1 - k0), sinks,
                scale=kwargs.get("scale"),
                attention_mask=FakeTensor(mask))
        return FakeTensor(np.zeros_like(query.a))

    try:
        gpt.sink_attention_tc = stub_full
        gpt.sliding_sink_attention_tc = stub_sliding

        try:
            SinkAttentionTap([types.SimpleNamespace(self_attn=None)],
                             head_dim=head_dim)
            record("tap_rejects_a_layer_without_self_attn_sinks", False, "no raise")
        except LSRError as exc:
            record("tap_rejects_a_layer_without_self_attn_sinks", True, str(exc))

        shared = FakeTensor(np.ones((heads,)))
        try:
            SinkAttentionTap(
                [types.SimpleNamespace(
                    self_attn=types.SimpleNamespace(layer_idx=i, sinks=shared))
                 for i in range(2)], head_dim=head_dim)
            record("tap_rejects_ambiguous_shared_sinks", False, "no raise")
        except LSRError as exc:
            record("tap_rejects_ambiguous_shared_sinks", True, str(exc))

        # One forward: layer 0 full, layers 1-2 sliding+chunked.
        total_keys = 10
        tap = SinkAttentionTap(layers, head_dim=head_dim)
        with tap:
            q = FakeTensor(np.random.RandomState(0).randn(
                1, heads, 4, head_dim))
            k = FakeTensor(np.random.RandomState(1).randn(
                1, heads, total_keys, head_dim))
            v = FakeTensor(np.zeros((1, heads, total_keys, head_dim)))
            gpt.sink_attention_tc(
                q, k, v, sinks_by_layer[0], scale=0.25)
            for layer in (1, 2):
                gpt.sliding_sink_attention_tc(
                    q, k, v, sinks_by_layer[layer],
                    scale=0.25, sliding_window=6, attn_block=2)

        record(
            "tap_attributes_every_layer_exactly_once_under_nested_chunking",
            sorted(tap.operands) == [0, 1, 2]
            and all(len(rows) == 1 for rows in tap.operands.values()),
            {layer: len(rows) for layer, rows in sorted(tap.operands.items())},
        )
        record(
            "tap_exports_host_operands_and_retains_no_device_tensor",
            all(
                isinstance(row["q"], np.ndarray)
                and isinstance(row["k"], np.ndarray)
                and not isinstance(row["q"], FakeTensor)
                and not isinstance(row["k"], FakeTensor)
                for rows in tap.operands.values() for row in rows
            ),
            {"exported_kinds": sorted({
                type(row[key]).__name__
                for rows in tap.operands.values() for row in rows
                for key in ("q", "k")
            })},
        )
        record(
            "tap_labels_layer_kinds_from_the_real_dispatch",
            tap.layer_kind.get(0) == "full"
            and tap.layer_kind.get(1) == "sliding"
            and tap.layer_kind.get(2) == "sliding",
            dict(sorted(tap.layer_kind.items())),
        )
        widths = {layer: int(rows[0]["width"])
                  for layer, rows in tap.operands.items()}
        record(
            "sliding_chunks_are_placed_back_on_the_full_key_axis",
            all(width == total_keys for width in widths.values()),
            widths,
        )
        record(
            "sliding_layers_expose_only_their_window_as_allowed",
            int(tap.allowed[0][0].sum()) == total_keys
            and 0 < int(tap.allowed[1][0].sum()) < total_keys,
            {"full": int(tap.allowed[0][0].sum()),
             "sliding": int(tap.allowed[1][0].sum())},
        )
        record(
            "tap_counts_one_forward",
            tap.forward_count == 1, tap.forward_count)

        scores, sinks, allowed = tap.stacked()
        record(
            "stacked_yields_readout_by_head_by_key_and_per_layer_sinks",
            all(value.shape == (1, heads, total_keys) for value in scores.values())
            and sorted(sinks) == [0, 1, 2]
            and all(value.shape == (heads,) for value in sinks.values())
            and all(value.shape == (1, total_keys) for value in allowed.values()),
            {"scores": {k: v.shape for k, v in scores.items()}},
        )

        # The engine's scale must be the one used, not head_dim ** -0.5, and
        # the offline recompute must reproduce the reference contraction.
        expected = np.einsum(
            "hqd,hsd->hqs", q.a[0], k.a[0])[:, -1, :] * 0.25
        record(
            "tap_uses_the_engine_supplied_scale_not_a_re_derived_one",
            np.allclose(scores[0][0], expected, atol=1e-5),
            {"max_abs": float(np.abs(scores[0][0] - expected).max())},
        )

        # Reference cross-check against grm_cmc1_mechanism's own contraction:
        # the offline recompute IS materialized_softmax's pre-softmax math.
        from scripts.grm_cmc1_mechanism import softmax_fp32

        ref_scores = np.einsum(
            "hqd,hsd->hqs", q.a[0], k.a[0], dtype=np.float32) * np.float32(0.25)
        record(
            "offline_recompute_matches_the_cmc1_reference_contraction",
            np.allclose(
                softmax_fp32(scores[0][0][:, None, :])[:, 0, :],
                softmax_fp32(ref_scores)[:, -1, :], atol=1e-6),
            "softmax over recomputed row == softmax over reference row",
        )

        # GQA: 4 query heads over 2 KV heads must pair the way _repeat_kv does
        # ("each kv head repeated rep times", contiguously).
        gqa_q = np.random.RandomState(2).randn(4, head_dim).astype(np.float32)
        gqa_k = np.random.RandomState(3).randn(2, 5, head_dim).astype(np.float32)
        gqa_row = SinkAttentionTap.__dict__["_recompute_row"](
            tap, {"q": gqa_q, "k": gqa_k, "scale": 1.0,
                  "key_offset": 0, "width": 5}, 5)
        gqa_expected = np.einsum(
            "hd,hsd->hs", gqa_q, np.repeat(gqa_k, 2, axis=0), dtype=np.float32)
        record(
            "gqa_query_heads_pair_with_repeat_kv_expanded_key_heads",
            gqa_row.shape == (4, 5)
            and np.allclose(gqa_row, gqa_expected, atol=1e-6),
            {"shape": gqa_row.shape,
             "max_abs": float(np.abs(gqa_row - gqa_expected).max())},
        )
        try:
            SinkAttentionTap.__dict__["_recompute_row"](
                tap, {"q": np.zeros((3, head_dim), dtype=np.float32),
                      "k": np.zeros((2, 5, head_dim), dtype=np.float32),
                      "scale": 1.0, "key_offset": 0, "width": 5}, 5)
            record("recompute_rejects_indivisible_gqa_head_counts", False,
                   "no raise")
        except LSRError as exc:
            record("recompute_rejects_indivisible_gqa_head_counts", True,
                   str(exc))
    finally:
        gpt.sink_attention_tc = orig_full
        gpt.sliding_sink_attention_tc = orig_sliding

    passed = sum(1 for row in cases if row["pass"])
    return {
        "schema": f"{SCHEMA_PREFIX}.tap_selftest.v1",
        "cases": cases,
        "case_count": len(cases),
        "passed": passed,
        "failed": len(cases) - passed,
        "all_passed": passed == len(cases),
    }


def _window_tokens(arena, prompt_ids: Sequence[int], sink_text: str) -> dict[str, Any]:
    """Per-token surface text for the whole window, region by region.

    Anchor and live token ids are exact.  Arena and recent tokens are
    recovered by re-encoding each graft's own text -- the way
    scripts/grm_det1_5_gpu._live_token_ids recovers live ids.  If a re-encode
    length disagrees with the seated token count the window is reported
    unresolvable rather than guessed.
    """
    sink_ids = [int(v) for v in arena.encode(sink_text)]
    if len(sink_ids) != int(arena.n_sink):
        raise LSRError(
            f"sink re-encode is {len(sink_ids)} tokens but n_sink is "
            f"{arena.n_sink}")
    arena_ids: list[int] = []
    for index in arena.cur_mounts:
        graft = arena.grafts[int(index)]
        ids = [int(v) for v in arena.encode(str(graft.get("text", "")))]
        if len(ids) != int(graft["ntok"]):
            raise LSRError(
                f"graft {index} re-encode is {len(ids)} tokens but the arena "
                f"seats {graft['ntok']}")
        arena_ids.extend(ids)
    recent_ids: list[int] = []
    for graft_index, count in arena.live_segs:
        if graft_index is None:
            raise LSRError("anonymous live segment: token ledger unrecoverable")
        ids = [int(v) for v in arena.encode(
            str(arena.grafts[int(graft_index)].get("text", "")))]
        if len(ids) != int(count):
            raise LSRError("live segment re-encode length differs from cache")
        recent_ids.extend(ids)
    token_ids = sink_ids + arena_ids + recent_ids + [int(v) for v in prompt_ids]
    return {
        "token_ids": token_ids,
        "token_strings": [str(arena.decode([int(v)])) for v in token_ids],
        "counts": {
            "anchor": len(sink_ids),
            "arena": len(arena_ids),
            "recent": len(recent_ids),
            "live": len(prompt_ids),
        },
    }


def witness_probe(
    probe_id: str, *, lease_seconds: int, lock_wait_seconds: int,
) -> dict[str, Any]:
    """Serve one probe under a self-lease and witness the full window."""
    from scripts.grm_cmc1_gpu_arms import gpu_lease

    enumerated = enumerate_probes()
    rows = {
        str(row["probe_id"]): row
        for group in ("wrong_value_probes", "lawful_controls")
        for row in enumerated[group]
    }
    if probe_id not in rows:
        raise LSRError(f"{probe_id} is not an enumerated LSR Phase-0 probe")
    probe = rows[probe_id]
    binding = probe_session_map()[probe_id]
    served_value = probe.get("served_value")
    expected_value = (probe["expected_values"] or [None])[0]
    if not served_value:
        raise LSRError(f"{probe_id} has no registered served-value binding")

    started = time.time_ns()
    with gpu_lease(int(lease_seconds), int(lock_wait_seconds)):
        result = _serve_and_witness(
            probe_id=probe_id, binding=binding,
            served_value=str(served_value), expected_value=expected_value)
    result["lease"] = {
        "lease_seconds": int(lease_seconds),
        "lock_wait_seconds": int(lock_wait_seconds),
        "lock": "/tmp/forge-gpu.lock",
        "elapsed_ns": int(time.time_ns() - started),
    }
    return result


def _serve_and_witness(
    *, probe_id: str, binding: Mapping[str, Any],
    served_value: str, expected_value: str | None,
) -> dict[str, Any]:
    """Build the session, serve the probe with the tap installed, witness.

    The session is constructed exactly as the DET1 sup stage does
    (scripts/grm_det1_gpu.py::_sup_stage): the same GptOssGQAArenaCache, the
    same HARMONY sink/template/stops, the same dialect route_layer, and the
    census runtime frame's own resolved flags.  Mount selection goes through
    the registered production route + admission path
    (grm_det1_common.route_fixture_profile then _budget_fit_mounts), so the
    window under the witness is the window the census actually served.
    """
    from core.gpt_oss20b_tc import GptOss20B_TC, gpt_oss_grm_dialect_kwargs
    from core.graft_repository import GraftRepository
    from core.grm_admission import adm_decisive_enabled
    from core.grm_supersession import sup_resolve_enabled
    from scripts import grm_e2e_session as e2e
    from scripts.grm_det1_common import route_fixture_profile
    from scripts.grm_det1_gpu import _install_fixture_nodes
    from transformers import AutoTokenizer

    frame = json.loads(RUNTIME_FRAME.read_text(encoding="utf-8"))
    flags = frame["resolved_flags"]
    model_dir = frame["model"]["path"]

    model, _model_info = GptOss20B_TC.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    def decode(ids):
        return tokenizer.decode(ids, clean_up_tokenization_spaces=False)

    dialect = gpt_oss_grm_dialect_kwargs(model.config)
    repo = GraftRepository(
        model, encode, decode,
        os.environ.get("LSR_REPO_DIR", "/tmp/lsr_p0_repo"),
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
    try:
        fixture = json.loads(
            (SUP_FIXTURES / f"{binding['session_id']}.json").read_text(
                encoding="utf-8"))
        _install_fixture_nodes(repo, fixture)
        arena = repo.arena
        question = binding.get("question")
        if not question:
            raise LSRError(
                f"{probe_id} has no question binding; reserve probes carry "
                "theirs in scripts/grm_det1_5_workers.SUP_RESERVE_PROBES")

        ngen = int(flags["ngen"])
        topk = int(flags["topk"])
        max_trips = int(flags["max_trips"])
        live_excluded = {
            int(g) for g, _n in arena.live_segs if g is not None}
        profile = route_fixture_profile(
            arena, str(question), str(expected_value or ""),
            live_excluded=live_excluded,
            route_limit=max(topk, (max_trips + 1) * topk),
        )
        planned = [int(v) for v in profile["served_effective_admission_plan"]]
        picks = sorted(e2e._budget_fit_mounts(arena, planned))
        prompt_ids = arena.encode(arena._format_step_prompt(str(question)))

        # GptOss20B_TC exposes its dataclass as .config (core/gpt_oss20b_tc.py
        # :1218 `self.config = cfg or GptOss20BConfig()`), NOT .cfg -- the
        # class's own contract-surface docstring lists ".config
        # GptOss20BConfig".  Only the per-layer attention modules carry a bare
        # .head_dim (GptOssAttentionTC.__init__ :639), and the arena drives
        # the model object, not those modules, so the config path is the one
        # that holds here.
        tap = SinkAttentionTap(
            arena.m.layers, head_dim=int(model.config.head_dim))
        with tap:
            answer, _info = arena._attempt(
                str(question), picks, ngen, False,
                arena.stop_sequences or (), defer_memory=True)
        tap.finalize(ngen)

        layout = derive_window_layout(
            n_sink=int(arena.n_sink),
            arena_width=int(arena.width),
            cur_mount_n=int(arena.cur_mount_n),
            mount_seat_ranges=arena._mount_seat_ranges(),
            live_segs=list(arena.live_segs),
            live_turns=int(arena.live_turns),
            prompt_ntok=len(prompt_ids),
        )
        tokens = _window_tokens(arena, prompt_ids, e2e.HARMONY_SINK)
        scores, sinks, allowed = tap.stacked()
        mass = full_window_readout_mass(
            softmax_operands_by_layer=scores,
            layout=layout,
            sink_logits_by_layer=sinks,
            allowed_masks_by_layer=allowed,
        )
        served_spans = locate_value_spans(
            layout=layout, token_ids=tokens["token_ids"],
            token_strings=tokens["token_strings"],
            value=served_value, label="served")
        expected_spans = (
            locate_value_spans(
                layout=layout, token_ids=tokens["token_ids"],
                token_strings=tokens["token_strings"],
                value=expected_value, label="expected")
            if expected_value else []
        )
        verdict = probe_verdict(
            probe_id=probe_id, served_value=served_value,
            expected_value=expected_value or "",
            layout=layout, mass=mass,
            served_spans=served_spans, expected_spans=expected_spans)
        return {
            "schema": f"{SCHEMA_PREFIX}.probe_witness.v1",
            "program": "LSR",
            "phase": "0",
            "probe_id": probe_id,
            "session_id": str(binding["session_id"]),
            "question": str(question),
            "served_answer_this_run": str(answer),
            "mounted_ids": [int(v) for v in arena.cur_mounts],
            "route_profile": {
                "raw_ranking": [int(v) for v in profile["raw_ranking"]],
                "logical_router_rank1": profile["logical_router_rank1"],
                "production_admitted_rank1": profile["production_admitted_rank1"],
                "served_effective_admission_plan": planned,
                "target_contains_expected": profile["target_contains_expected"],
                "route_backend": profile["route_backend"],
            },
            "window_layout": layout,
            "window_token_counts": tokens["counts"],
            "layer_kinds": {str(k): v for k, v in sorted(tap.layer_kind.items())},
            "readout_mass": mass,
            "verdict": verdict,
            "declares_in_det_envelope": False,
            "plan": file_record(PLAN),
            "runtime_frame": file_record(RUNTIME_FRAME),
            "sources": {
                "lsr_p0_core": file_record(ROOT / "scripts" / "lsr_p0_core.py"),
                "lsr_p0_gpu": file_record(Path(__file__).resolve()),
            },
        }
    finally:
        try:
            repo.close()
        except Exception:  # noqa: BLE001 - teardown must not mask a result
            pass


def adjudicate(receipt_dir: Path = ARTIFACT_DIR) -> dict[str, Any]:
    """Apply the registered adjudication over whatever witnesses exist."""
    enumerated = enumerate_probes()
    wanted = {
        str(row["probe_id"]): str(row["lsr_role"])
        for group in ("wrong_value_probes", "lawful_controls")
        for row in enumerated[group]
    }
    found: dict[str, dict[str, Any]] = {}
    for path in sorted(Path(receipt_dir).glob("lsr_p0_witness_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        probe_id = str(payload.get("probe_id", ""))
        if probe_id in wanted:
            found[probe_id] = payload["verdict"]

    wrong = [found[pid] for pid, role in wanted.items()
             if role == "wrong_value" and pid in found]
    controls = [found[pid] for pid, role in wanted.items()
                if role == "lawful_control" and pid in found]
    result = adjudicate_h_lsr_1(wrong, controls)
    result["witnessed_probes"] = sorted(found)
    result["missing_probes"] = sorted(set(wanted) - set(found))
    result["plan"] = file_record(PLAN)
    result["census"] = file_record(CENSUS)
    result["declares_in_det_envelope"] = False
    return result


def _emit(payload: Mapping[str, Any], stem: str) -> Path:
    blob = canonical_json_bytes(payload)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT_DIR / f"{stem}_{sha256_bytes(blob)[:16]}.json"
    if path.exists():
        if path.read_bytes() != blob:
            raise LSRError(f"content-address collision: {path}")
        return path
    with path.open("xb") as handle:
        handle.write(blob)
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LSR Phase 0 GPU witness")
    sub = parser.add_subparsers(dest="command", required=True)
    witness = sub.add_parser("witness", help="witness one probe under a lease")
    witness.add_argument("--probe", required=True)
    witness.add_argument("--lease-seconds", type=int, default=LEASE_SECONDS)
    witness.add_argument("--lock-wait-seconds", type=int,
                         default=LOCK_WAIT_SECONDS)
    sub.add_parser("adjudicate", help="apply the registered adjudication")
    sub.add_parser("list-probes", help="print the enumerated probe ids")
    sub.add_parser("selftest", help="CPU tap selftest (no GPU, no lease)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "selftest":
        result = tap_selftest()
        for row in result["cases"]:
            print(f"{'PASS' if row['pass'] else 'FAIL'} {row['case']}")
            if not row["pass"]:
                print(f"     {row['detail']}")
        print(f"tap selftest: {result['passed']}/{result['case_count']}")
        return 0 if result["all_passed"] else 1
    if args.command == "list-probes":
        enumerated = enumerate_probes()
        for group in ("wrong_value_probes", "lawful_controls"):
            for row in enumerated[group]:
                print(f"{row['lsr_role']:14s} {row['probe_id']}")
        return 0
    if args.command == "witness":
        if int(args.lease_seconds) > LEASE_SECONDS:
            raise LSRError(
                f"lease {args.lease_seconds}s exceeds the standing "
                f"{LEASE_SECONDS}s cap")
        payload = witness_probe(
            str(args.probe),
            lease_seconds=int(args.lease_seconds),
            lock_wait_seconds=int(args.lock_wait_seconds))
        path = _emit(payload, f"lsr_p0_witness_{args.probe}")
        verdict = payload["verdict"]
        print(f"probe={payload['probe_id']} verdict={verdict['verdict']}")
        print(f"  region_mass_share={verdict['region_mass_share']}")
        print(f"  recent_present="
              f"{verdict['served_value_present_in_recent_turns']}")
        print(f"receipt {path}")
        return 0
    if args.command == "adjudicate":
        payload = adjudicate()
        path = _emit(payload, "lsr_p0_adjudication")
        print(f"status={payload['status']} "
              f"satisfied={payload['wrong_value_probes_satisfying_both_legs']}"
              f"/{payload['denominator']}")
        if payload["missing_probes"]:
            print(f"missing={payload['missing_probes']} -> "
                  f"{VERDICT_NOT_MEASURED}")
        print(f"receipt {path}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
