#!/usr/bin/env python3
"""Guarded live T1-T4 runner for the GRM-CMC1.1 supersession fixture.

The ``all`` orchestrator launches one fresh child process per stage.  Every
child holds a flock-protected single-GPU lease capped below 590 seconds; the
orchestrator leaves a 30-second gap between stages.  G0 is written before any
arm and every later child refuses to run unless that live receipt says the
supersession fresh control still reproduces 1/2.

All intervention code is local to this script.  The production runtime is not
edited: T1 calls the arena's existing final assembly choke point with explicit
orders, and T2-T4 temporarily wrap the standard SDPA function in-process.
With no arm selected, no wrapper or runtime flag exists.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any, Iterator

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from grm_cmc1_mechanism import (  # noqa: E402
    CMCError,
    adjudicate_position,
    canonical_json_bytes,
    materialized_softmax,
    sha256_file,
    softmax_fp32,
    summarize_attention_capture,
    write_content_addressed,
)


DEFAULT_ARTIFACT_DIR = ROOT / "artifacts" / "grm_cmc1"
FIXTURE_PATH = (
    ROOT / "tests" / "fixtures" / "supersession_battery"
    / "fresh_fact_controls.json"
)
HARNESS_PATH = ROOT / "tests" / "test_grm_supersession_battery.py"
ORDER_PATH = ROOT / "orders" / "GRM_CMC1_COMOUNT_MECHANISM.md"
AMENDMENT_PATH = ROOT / "orders" / "GRM_CMC1_1_PER_FIXTURE.md"
LOCK_PATH = Path("/tmp/forge-gpu.lock")
MAX_LEASE_SECONDS = 590
DEFAULT_LEASE_SECONDS = 580
GAP_SECONDS = 30
STAGES = ("g0", "g1", "t1", "t2", "t3", "t4")
T4_DELTAS = (0.0, 0.05, 0.10, 0.20, 0.50, 1.00)
G2_TOLERANCE = 2.0e-3


class LiveError(CMCError):
    pass


def _load_file_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise LiveError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _source_provenance() -> dict[str, Any]:
    paths = (
        ORDER_PATH,
        AMENDMENT_PATH,
        Path(__file__).resolve(),
        ROOT / "scripts" / "grm_cmc1_mechanism.py",
        ROOT / "core" / "graft_arena.py",
        ROOT / "core" / "minicpm3_tc.py",
        HARNESS_PATH,
        FIXTURE_PATH,
    )
    return {
        str(path.relative_to(ROOT)): {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in paths
    }


@contextmanager
def gpu_lease(seconds: int, wait_seconds: int) -> Iterator[None]:
    if seconds <= 0 or seconds > MAX_LEASE_SECONDS:
        raise LiveError(
            f"lease must be in 1..{MAX_LEASE_SECONDS}s, got {seconds}")
    handle = LOCK_PATH.open("a+")
    started_wait = time.monotonic()
    last_notice = -30
    while True:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            waited = int(time.monotonic() - started_wait)
            if waited >= int(wait_seconds):
                handle.close()
                raise LiveError(
                    f"GPU lock unavailable after {wait_seconds}s: {LOCK_PATH}")
            if waited - last_notice >= 30:
                print(f"waiting_for_gpu_lock_s={waited}", flush=True)
                last_notice = waited
            time.sleep(5)

    old_handler = signal.getsignal(signal.SIGALRM)

    def _expired(_signum, _frame):
        raise TimeoutError(f"GPU lease exceeded {seconds}s")

    signal.signal(signal.SIGALRM, _expired)
    signal.alarm(int(seconds))
    started = time.monotonic()
    try:
        print(f"gpu_lease_acquired={LOCK_PATH} cap_s={seconds}", flush=True)
        yield
    finally:
        elapsed = time.monotonic() - started
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
        print(f"gpu_lease_released_s={elapsed:.3f}", flush=True)


def _unique_stage(run_dir: Path, stem: str) -> tuple[Path, dict[str, Any]]:
    hits = sorted(run_dir.glob(f"{stem}_*.json"))
    if len(hits) != 1:
        raise LiveError(
            f"expected one {stem}_*.json under {run_dir}, found {len(hits)}")
    value = json.loads(hits[0].read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise LiveError(f"stage receipt must be an object: {hits[0]}")
    return hits[0], value


def _require_g0(run_dir: Path) -> dict[str, Any]:
    _path, g0 = _unique_stage(run_dir, "g0")
    if not (
        g0.get("fixture_status") == "REPRODUCES"
        and g0.get("fresh_control_correct") == 1
        and g0.get("fresh_control_total") == 2
    ):
        raise LiveError("CMC-G0 does not authorize supersession arms")
    return g0


def _normalize_answer_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "probe_id": row["probe_id"],
        "answer_text": row["answer_text"],
        "classification": row["classification"],
        "classification_match": row.get("classification_match"),
        "ranking_nodes": row.get("ranking_nodes"),
        "mounted_nodes": row.get("mounted_nodes"),
        "mounted_indices": row.get("mounted_indices"),
    }


class LiveFixture:
    """One model load and one exact fresh-control arena."""

    def __init__(self):
        sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
        from tokenizers import Tokenizer as HFTok
        from core.graft_arena import ArenaCache
        from core.minicpm3_tc import MiniCPM3_TC, _snap

        self.harness = _load_file_module(
            "grm_cmc1_live_supersession_harness", HARNESS_PATH)
        self.fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        self.harness.validate_fixture(self.fixture, str(FIXTURE_PATH))
        self.model, self.model_info = MiniCPM3_TC.from_pretrained()
        self.tokenizer = HFTok.from_file(str(Path(_snap()) / "tokenizer.json"))
        self.arena = ArenaCache(
            self.model,
            encode=lambda text: self.tokenizer.encode(text).ids,
            decode=lambda ids: self.tokenizer.decode(ids),
            sink_text="<conversation>\n",
            arena_width=256,
            route_layer=44,
            topk=3,
            live_turns=0,
            cache_deposits=False,
            length_debias=False,
            revision_resolution=False,
        )
        self.node_to_idx = self.harness._install_fixture_nodes(
            self.arena, self.fixture)
        self.idx_to_node = {
            int(idx): node_id for node_id, idx in self.node_to_idx.items()
        }
        self.nodes_by_id = {
            node["node_id"]: node for node in self.fixture["nodes"]
        }
        self.probes_by_id = {
            probe["probe_id"]: probe for probe in self.fixture["probes"]
        }
        self.original_h = [graft["h"] for graft in self.arena.grafts]
        self.baseline_order = [
            self.node_to_idx[node["node_id"]] for node in self.fixture["nodes"]
        ]

    def reset(self) -> None:
        from core import kv_graft

        kv_graft.clear_injection(self.model)
        self.arena.caches = None
        self.arena.pos = 0
        self.arena.live_segs = []
        self.arena.cur_mounts = []
        self.arena.cur_mount_n = 0
        for idx, h in enumerate(self.original_h):
            self.arena.grafts[idx]["h"] = h
        for layer in self.model.layers:
            attn = layer.self_attn
            for name in ("_captured_q", "_captured"):
                if hasattr(attn, name):
                    delattr(attn, name)

    def classify(self, answer: str, probe: dict[str, Any]) -> tuple[str, Any]:
        return self.harness._classify_answer(answer, probe)

    def _mounted(self, info: dict[str, Any]) -> tuple[list[int], list[str]]:
        indices = [int(value) - 1 for value in info.get("mounts", ())]
        return indices, [self.idx_to_node.get(i, f"graft:{i}") for i in indices]

    def run_default_fixture(self) -> list[dict[str, Any]]:
        """Exact current test-harness sequence for the two fresh probes."""
        self.reset()
        rows = []
        for probe in self.fixture["probes"]:
            ranking = list(self.arena.route(
                probe["question"], exclude=set(), limit=len(self.arena.grafts)))
            answer, info = self.arena.step(
                probe["question"], ngen=32, deposit=False, max_trips=0)
            mounted_indices, mounted_nodes = self._mounted(info)
            classification, matched = self.classify(answer, probe)
            rows.append({
                "probe_id": probe["probe_id"],
                "answer_text": answer,
                "classification": classification,
                "classification_match": matched,
                "ranking_nodes": [self.idx_to_node.get(i, f"graft:{i}")
                                  for i in ranking],
                "mounted_nodes": mounted_nodes,
                "mounted_indices": mounted_indices,
            })
        return rows

    def run_attempt(
        self, probe_id: str, order: list[int], *, ngen: int = 32,
        interceptor=None,
    ) -> dict[str, Any]:
        self.reset()
        probe = self.probes_by_id[probe_id]
        if interceptor is None:
            answer, info = self.arena._attempt(
                probe["question"], list(order), ngen, False,
                self.arena.stop_sequences)
        else:
            interceptor.prepare(self, probe, order)
            with interceptor:
                answer, info = self.arena._attempt(
                    probe["question"], list(order), ngen, False,
                    self.arena.stop_sequences)
            interceptor.finalize(ngen)
        classification, matched = self.classify(answer, probe)
        mounted_indices, mounted_nodes = self._mounted(info)
        return {
            "probe_id": probe_id,
            "answer_text": answer,
            "classification": classification,
            "classification_match": matched,
            "mounted_indices": mounted_indices,
            "mounted_nodes": mounted_nodes,
        }

    def mount_ranges(
        self, order: list[int], *, absolute: bool
    ) -> dict[str, tuple[int, int]]:
        seat = int(self.arena.n_sink) if absolute else 0
        ranges = {}
        for idx in order:
            start = seat
            seat += int(self.arena.grafts[idx]["ntok"])
            ranges[self.idx_to_node[idx]] = (start, seat)
        return ranges

    def winner_identity(self, matched: Any) -> str | None:
        if matched is None:
            return None
        folded = str(matched).casefold()
        for node in self.fixture["nodes"]:
            if str(node["value"]).casefold() == folded:
                return node["node_id"]
        return None

    def close(self) -> None:
        self.reset()
        del self.original_h
        del self.arena
        del self.model
        gc.collect()
        try:
            from core.mistral7b_tc import tc
            tc.empty_cache()
        except Exception:
            pass


class SDPAInterceptor:
    """Temporary wrapper around MiniCPM3's standard SDPA call."""

    def __init__(self, mode: str, *, percentile: float | None = None,
                 delta: float | None = None):
        self.mode = mode
        self.percentile = percentile
        self.delta = delta
        self.fixture: LiveFixture | None = None
        self.order: list[int] = []
        self.absolute_ranges: dict[str, tuple[int, int]] = {}
        self.local_ranges: dict[str, tuple[int, int]] = {}
        self.bonus_positions: list[int] = []
        self.identifier_receipt: dict[str, Any] = {}
        self.queries: dict[int, list[np.ndarray]] = {}
        self.keys: dict[int, np.ndarray] = {}
        self.clip_stats: dict[int, dict[str, Any]] = {}
        self.clipped_keys: dict[int, Any] = {}
        self.bonus_masks: dict[tuple[Any, ...], Any] = {}
        self.g2: dict[str, Any] | None = None
        self.calls = 0
        self.forward_count = 0
        self._module = None
        self._original = None

    def prepare(
        self, fixture: LiveFixture, probe: dict[str, Any], order: list[int]
    ) -> None:
        self.fixture = fixture
        self.order = list(order)
        self.absolute_ranges = fixture.mount_ranges(order, absolute=True)
        self.local_ranges = fixture.mount_ranges(order, absolute=False)
        self.queries = {}
        self.keys = {}
        self.clip_stats = {}
        self.clipped_keys = {}
        self.bonus_masks = {}
        self.g2 = None
        self.calls = 0
        self.forward_count = 0
        self.bonus_positions = []
        self.identifier_receipt = {}
        if self.mode == "bonus":
            positions, receipt = _identifier_bonus_positions(
                fixture, probe, order, self.absolute_ranges)
            if not positions:
                raise LiveError(
                    f"identifier channel found no exact mounted token for {probe['probe_id']}")
            self.bonus_positions = positions
            self.identifier_receipt = receipt

    def __enter__(self):
        from core import minicpm3_tc

        self._module = minicpm3_tc
        self._original = minicpm3_tc.F.scaled_dot_product_attention
        minicpm3_tc.F.scaled_dot_product_attention = self._wrapped
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._module is not None and self._original is not None:
            self._module.F.scaled_dot_product_attention = self._original
        return False

    def finalize(self, ngen: int) -> None:
        # No-stop generation commits one final predicted token to KV through
        # an extra forward whose logits are unused.  It is not an answer
        # readout position, so remove that final query from every layer.
        if self.mode == "capture" and self.forward_count == int(ngen) + 1:
            for layer in self.queries:
                if self.queries[layer]:
                    self.queries[layer].pop()

    def _wrapped(self, query, key, value, *args, **kwargs):
        if self.fixture is None or self._original is None:
            raise LiveError("SDPA interceptor is not prepared")
        n_layers = len(self.fixture.model.layers)
        layer = self.calls % n_layers
        if layer == 0:
            self.forward_count += 1
        self.calls += 1
        if self.mode == "capture":
            self._capture(layer, query, key)
            return self._original(query, key, value, *args, **kwargs)
        if self.mode == "clip":
            clipped = self._clip(layer, key)
            return self._original(query, clipped, value, *args, **kwargs)
        if self.mode == "bonus":
            return self._bonus(query, key, value, *args, **kwargs)
        raise LiveError(f"unknown SDPA interception mode {self.mode!r}")

    def _capture(self, layer: int, query, key) -> None:
        q_last = query.slice(2, query.shape[2] - 1, 1).float().numpy()
        q_np = np.asarray(q_last, dtype=np.float32)[0, :, 0, :]
        self.queries.setdefault(layer, []).append(q_np)
        mount_start = min(start for start, _ in self.absolute_ranges.values())
        mount_end = max(end for _, end in self.absolute_ranges.values())
        if layer not in self.keys:
            mounted = key.slice(2, mount_start, mount_end - mount_start)
            self.keys[layer] = np.asarray(
                mounted.float().numpy(), dtype=np.float32)[0]
        if self.g2 is None:
            from core.mistral7b_tc import tc

            scale = float(query.shape[-1] ** -0.5)
            direct_scores = tc.matmul(
                query.slice(2, query.shape[2] - 1, 1), key,
                alpha=scale, trans_b=True)
            direct_weights = tc.causal_softmax(direct_scores)
            direct = np.asarray(
                direct_weights.float().numpy(), dtype=np.float32)[0, :, 0, :]
            q_full = np.asarray(q_last, dtype=np.float32)[0]
            k_full = np.asarray(key.float().numpy(), dtype=np.float32)[0]
            offline = materialized_softmax(q_full, k_full)[:, 0, :]
            diff = np.abs(direct - offline)
            self.g2 = {
                "case": "live_praxis_short_standard_path_layer0_last_prompt_row",
                "layer": int(layer),
                "query_rows": 1,
                "key_count": int(key.shape[2]),
                "head_count": int(key.shape[1]),
                "direct": "tensor_cuda_materialized_qk_causal_softmax",
                "offline": "numpy_fp32_materialized_qk_softmax",
                "max_abs": float(diff.max()),
                "mean_abs": float(diff.mean()),
                "tolerance": G2_TOLERANCE,
                "pass": bool(float(diff.max()) <= G2_TOLERANCE),
            }

    def _clip(self, layer: int, key):
        from core.mistral7b_tc import tc

        if self.percentile is None:
            raise LiveError("clip interceptor has no percentile")
        sibling = "solace_fact"
        start, end = self.absolute_ranges[sibling]
        clipped_tensor = self.clipped_keys.get(layer)
        if clipped_tensor is None:
            sibling_tensor = key.slice(2, start, end - start)
            sibling_np = np.asarray(
                sibling_tensor.float().numpy(), dtype=np.float32)
            norms = np.linalg.norm(sibling_np, axis=-1)  # (B,H,S)
            thresholds = np.percentile(
                norms, float(self.percentile), axis=-1, keepdims=True)
            scales = np.minimum(
                1.0, thresholds / np.maximum(norms, np.float32(1.0e-12)))
            clipped_np = sibling_np * scales[..., None].astype(np.float32)
            self.clip_stats[layer] = {
                "layer": int(layer),
                "percentile": float(self.percentile),
                "clipped_key_vectors": int(np.count_nonzero(scales < 1.0)),
                "total_sibling_key_vectors": int(scales.size),
                "threshold_p50_across_heads": float(np.percentile(thresholds, 50)),
                "threshold_min": float(thresholds.min()),
                "threshold_max": float(thresholds.max()),
            }
            clipped_tensor = tc.tensor(
                np.ascontiguousarray(clipped_np)).astype(key.dtype)
            self.clipped_keys[layer] = clipped_tensor
        parts = []
        if start:
            parts.append(key.slice(2, 0, start))
        parts.append(clipped_tensor)
        if end < key.shape[2]:
            parts.append(key.slice(2, end, key.shape[2] - end))
        return parts[0] if len(parts) == 1 else tc.cat(parts, dim=2)

    def _bonus(self, query, key, value, *args, **kwargs):
        from core.mistral7b_tc import tc

        delta = float(self.delta or 0.0)
        scale = kwargs.get("scale")
        if scale is None:
            scale = float(query.shape[-1] ** -0.5)
        scores = tc.matmul(query, key, alpha=float(scale), trans_b=True)
        valid = [pos for pos in self.bonus_positions if pos < scores.shape[-1]]
        if not valid:
            raise LiveError("identifier bonus positions are outside attention span")
        mask_key = (
            *(int(dim) for dim in scores.shape), str(scores.dtype), float(delta),
            tuple(valid),
        )
        mask_tensor = self.bonus_masks.get(mask_key)
        if mask_tensor is None:
            mask = np.zeros(
                tuple(int(dim) for dim in scores.shape), dtype=np.float32)
            mask[:, :, -1, valid] = np.float32(delta)
            mask_tensor = tc.tensor(
                np.ascontiguousarray(mask)).astype(scores.dtype)
            self.bonus_masks[mask_key] = mask_tensor
        scores = scores + mask_tensor
        is_causal = bool(kwargs.get("is_causal", False))
        if len(args) >= 2:
            is_causal = bool(args[1])
        # MiniCPM's standard path is causal for prefill and non-causal only
        # for L=1 decode.  tensor_cuda's own SDPA uses causal_softmax in both
        # cases (L=1 bottom-right causal == ordinary full-row softmax), so
        # retain that exact kernel and change only the additive logits.
        if not is_causal and int(query.shape[2]) != 1:
            raise LiveError("identifier hook saw an unsupported non-causal prefill")
        weights = tc.causal_softmax(scores)
        return tc.matmul(weights, value)


def _word_spans(text: str, terms: set[str]) -> list[tuple[int, int, str]]:
    spans = []
    for match in re.finditer(r"[A-Za-z0-9][\w:.,\-]*", text):
        token = match.group(0).rstrip(".,:;").lower()
        if token in terms:
            spans.append((match.start(), match.start() + len(token), token))
    return spans


def _token_indices_for_terms(tokenizer, text: str, terms: set[str]):
    encoded = tokenizer.encode(text)
    spans = _word_spans(text, terms)
    indices = []
    for index, (start, end) in enumerate(encoded.offsets):
        if end <= start:
            continue
        if any(start < word_end and end > word_start
               for word_start, word_end, _ in spans):
            indices.append(index)
    return encoded, indices, spans


def _identifier_bonus_positions(
    fixture: LiveFixture, probe: dict[str, Any], order: list[int],
    ranges: dict[str, tuple[int, int]],
) -> tuple[list[int], dict[str, Any]]:
    # This is the outer router's current lexical identifier extraction:
    # rare identifiers union stopword-filtered content tokens.
    terms = set(fixture.arena._query_lex_tokens(probe["question"]))
    query_enc, query_indices, query_spans = _token_indices_for_terms(
        fixture.tokenizer, probe["question"], terms)
    query_ids = {int(query_enc.ids[index]) for index in query_indices}
    positions = []
    by_graft = {}
    for idx in order:
        node_id = fixture.idx_to_node[idx]
        text = fixture.nodes_by_id[node_id]["text"]
        encoded, token_indices, _spans = _token_indices_for_terms(
            fixture.tokenizer, text, terms)
        exact = [
            token_index for token_index in token_indices
            if int(encoded.ids[token_index]) in query_ids
        ]
        start, end = ranges[node_id]
        if len(encoded.ids) != end - start:
            raise LiveError(
                f"token/range mismatch for {node_id}: {len(encoded.ids)} vs {end-start}")
        absolute = [start + token_index for token_index in exact]
        positions.extend(absolute)
        by_graft[node_id] = {
            "matching_token_count": len(absolute),
            "matching_token_ids": [int(encoded.ids[i]) for i in exact],
            "absolute_positions": absolute,
        }
    receipt = {
        "outer_identifier_extractor": "ArenaCache._query_lex_tokens",
        "identifier_terms": sorted(terms),
        "query_identifier_spans": [list(span) for span in query_spans],
        "query_identifier_token_ids": sorted(query_ids),
        "matching_mounted_tokens": by_graft,
    }
    return sorted(set(positions)), receipt


def _stage_base(stage: str, live: LiveFixture) -> dict[str, Any]:
    return {
        "schema": f"grm.cmc1_1.{stage}.v1",
        "order": "GRM-CMC1.1",
        "fixture": "supersession_fresh_control",
        "model": "MiniCPM3",
        "model_info": str(live.model_info),
        "dialect": "mla",
        "runtime": {
            "arena_width": 256,
            "route_layer": 44,
            "topk": 3,
            "live_turns": 0,
            "ngen": 32,
            "attention_mode": "standard",
        },
        "provenance": _source_provenance(),
    }


def run_g0(live: LiveFixture) -> dict[str, Any]:
    rows = live.run_default_fixture()
    normalized = [_normalize_answer_row(row) for row in rows]
    correct = sum(row["classification"] == "correct" for row in rows)
    praxis = next(row for row in rows if row["probe_id"] == "praxis_fresh")
    reproduces = bool(
        correct == 1
        and praxis["classification"] == "wrong-fact"
        and str(praxis.get("classification_match", "")).casefold()
        == "raven-9-ivory"
        and "praxis_fact" in praxis["mounted_nodes"]
        and "solace_fact" in praxis["mounted_nodes"]
    )
    result = _stage_base("g0", live)
    result.update({
        "gate": "CMC-G0",
        "fixture_status": "REPRODUCES" if reproduces else "HEALED_EXCLUDED",
        "fresh_control_correct": int(correct),
        "fresh_control_total": len(rows),
        "arms_authorized": reproduces,
        "registered_baseline_payload_sha256": hashlib.sha256(
            canonical_json_bytes(normalized)).hexdigest(),
        "transcript": normalized,
    })
    return result


def run_g1(live: LiveFixture, g0: dict[str, Any]) -> dict[str, Any]:
    rows = [_normalize_answer_row(row) for row in live.run_default_fixture()]
    replay_hash = hashlib.sha256(canonical_json_bytes(rows)).hexdigest()
    baseline_hash = str(g0["registered_baseline_payload_sha256"])
    result = _stage_base("g1", live)
    result.update({
        "gate": "CMC-G1",
        "flags": {
            "sdpa_interceptor": False,
            "order_override": False,
            "selective_clip": False,
            "identifier_delta": 0.0,
            "payload_rehydrate": False,
        },
        "registered_baseline_payload_sha256": baseline_hash,
        "flags_off_replay_payload_sha256": replay_hash,
        "byte_identity": replay_hash == baseline_hash,
        "transcript": rows,
    })
    return result


def run_t1(live: LiveFixture) -> dict[str, Any]:
    rows = []
    for order_tuple in itertools.permutations(live.baseline_order):
        order = list(order_tuple)
        outcome = live.run_attempt("praxis_fresh", order)
        winner = live.winner_identity(outcome.get("classification_match"))
        winner_position = order.index(live.node_to_idx[winner]) + 1 if winner else None
        rows.append({
            "mount_order": [live.idx_to_node[idx] for idx in order],
            "winner_identity": winner,
            "winner_position": winner_position,
            **outcome,
        })
    result = _stage_base("t1", live)
    result.update({
        "arm": "T1_order_swap",
        "ordering_rule": "all_3_factorial_physical_mount_orders",
        "orderings": rows,
        "position_adjudication": adjudicate_position(rows),
    })
    return result


def run_t2(live: LiveFixture) -> dict[str, Any]:
    interceptor = SDPAInterceptor("capture")
    outcome = live.run_attempt(
        "praxis_fresh", list(live.baseline_order), interceptor=interceptor)
    if outcome["classification"] != "wrong-fact":
        raise LiveError("T2 baseline attempt did not retain the reproducing wrong read")
    queries = {
        layer: np.stack(values, axis=0)
        for layer, values in interceptor.queries.items()
    }
    attention = summarize_attention_capture(
        queries,
        interceptor.keys,
        live.mount_ranges(live.baseline_order, absolute=False),
        target="praxis_fact",
        sibling="solace_fact",
    )
    if interceptor.g2 is None:
        raise LiveError("T2 did not produce a CMC-G2 comparison")
    result = _stage_base("t2", live)
    result.update({
        "arm": "T2_attention_witness",
        "answer": outcome,
        "attention": attention,
        "CMC-G2": interceptor.g2,
    })
    return result


def _full_int8_rehydrate(live: LiveFixture) -> tuple[list[Any], dict[str, Any]]:
    from core.graft_quant import pack_kv_arrays, unpack_kv_arrays

    new_h = []
    metrics = {}
    for idx, original_h in enumerate(live.original_h):
        arrays = live.arena.pack_node(original_h)
        packed = pack_kv_arrays(arrays, 8)
        unpacked = unpack_kv_arrays(packed, list(arrays))
        new_h.append(live.arena.unpack_node(unpacked))
        node_id = live.idx_to_node[idx]
        metrics[node_id] = {}
        for name in arrays:
            diff = arrays[name].astype(np.float32) - unpacked[name].astype(np.float32)
            metrics[node_id][name] = {
                "shape": list(arrays[name].shape),
                "max_abs": float(np.max(np.abs(diff))),
                "mean_abs": float(np.mean(np.abs(diff))),
            }
    return new_h, metrics


def run_t3(live: LiveFixture) -> dict[str, Any]:
    rows = []
    for percentile, setting in ((99.0, "clip_p99"), (99.9, "clip_p99_9")):
        interceptor = SDPAInterceptor("clip", percentile=percentile)
        outcome = live.run_attempt(
            "praxis_fresh", list(live.baseline_order), interceptor=interceptor)
        rows.append({
            "case_id": "praxis_fresh",
            "setting": setting,
            "percentile": percentile,
            "target_graft_touched": False,
            "sibling_graft": "solace_fact",
            "clip_stats": list(interceptor.clip_stats.values()),
            **outcome,
        })

    live.reset()
    rehydrated_h, quant_metrics = _full_int8_rehydrate(live)
    try:
        for idx, h in enumerate(rehydrated_h):
            live.arena.grafts[idx]["h"] = h
        # reset() restores original h, so clear only cache state here.
        live.arena.caches = None
        live.arena.pos = 0
        live.arena.live_segs = []
        live.arena.cur_mounts = []
        live.arena.cur_mount_n = 0
        probe = live.probes_by_id["praxis_fresh"]
        answer, info = live.arena._attempt(
            probe["question"], list(live.baseline_order), 32, False,
            live.arena.stop_sequences)
        classification, matched = live.classify(answer, probe)
        mounted_indices, mounted_nodes = live._mounted(info)
        rows.append({
            "case_id": "praxis_fresh",
            "setting": "full_int8_rehydrate_all_mounts",
            "storage_bits": 8,
            "group_size": 32,
            "answer_text": answer,
            "classification": classification,
            "classification_match": matched,
            "mounted_indices": mounted_indices,
            "mounted_nodes": mounted_nodes,
            "quantization_diff": quant_metrics,
        })
    finally:
        for idx, h in enumerate(live.original_h):
            live.arena.grafts[idx]["h"] = h
        del rehydrated_h

    result = _stage_base("t3", live)
    result.update({
        "arm": "T3_selective_clipping",
        "percentile_method": "numpy_linear_per_layer_per_head_over_sibling_tokens",
        "repro_case_ids": ["praxis_fresh"],
        "grid": rows,
    })
    return result


def run_t4(live: LiveFixture) -> dict[str, Any]:
    grid = []
    masks = {}
    for delta in T4_DELTAS:
        outcomes = {}
        for probe_id in ("praxis_fresh", "solace_fresh"):
            interceptor = SDPAInterceptor("bonus", delta=delta)
            outcome = live.run_attempt(
                probe_id, list(live.baseline_order), interceptor=interceptor)
            outcomes[probe_id] = outcome
            masks.setdefault(probe_id, interceptor.identifier_receipt)
        grid.append({
            "delta": float(delta),
            "repro_probe": "praxis_fresh",
            "repro_answer_text": outcomes["praxis_fresh"]["answer_text"],
            "repro_classification": outcomes["praxis_fresh"]["classification"],
            "other_probe": "solace_fresh",
            "other_probe_answer_text": outcomes["solace_fresh"]["answer_text"],
            "other_probe_classification": outcomes["solace_fresh"]["classification"],
            "other_probe_regressed": (
                outcomes["solace_fresh"]["classification"] != "correct"),
        })
    result = _stage_base("t4", live)
    result.update({
        "arm": "T4_identifier_channel",
        "delta_grid": list(T4_DELTAS),
        "small_delta_ceiling": 0.5,
        "bonus_scope": (
            "last query row of every standard-path layer at answer readout; "
            "mounted arena token positions only"
        ),
        "identifier_masks": masks,
        "grid": grid,
    })
    return result


def _run_stage(stage: str, run_dir: Path) -> tuple[Path, dict[str, Any]]:
    if stage != "g0":
        g0 = _require_g0(run_dir)
    else:
        g0 = None
    live = LiveFixture()
    try:
        if stage == "g0":
            result = run_g0(live)
        elif stage == "g1":
            result = run_g1(live, g0)
        elif stage == "t1":
            result = run_t1(live)
        elif stage == "t2":
            result = run_t2(live)
        elif stage == "t3":
            result = run_t3(live)
        elif stage == "t4":
            result = run_t4(live)
        else:
            raise LiveError(f"unknown stage {stage!r}")
        path = write_content_addressed(run_dir, stage, result)
        return path, result
    finally:
        live.close()


def _new_run_dir(base: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = base.resolve() / f"live_arms_{stamp}_{os.getpid()}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def cpu_self_test() -> dict[str, Any]:
    """Exercise live-runner rails without importing a model or CUDA."""
    fixture_data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    if [probe["probe_id"] for probe in fixture_data["probes"]] != [
        "praxis_fresh", "solace_fresh"
    ]:
        raise LiveError("fresh-control probe registration changed")
    if tuple(T4_DELTAS) != (0.0, 0.05, 0.10, 0.20, 0.50, 1.00):
        raise LiveError("T4 delta grid drifted")

    class FakeEncoding:
        def __init__(self, text, vocab):
            self.ids = []
            self.offsets = []
            for match in re.finditer(r"[A-Za-z0-9][\w:.,\-]*", text):
                word = match.group(0).rstrip(".,:;").lower()
                self.ids.append(vocab.setdefault(word, len(vocab) + 1))
                self.offsets.append((match.start(), match.start() + len(word)))

    class FakeTokenizer:
        def __init__(self):
            self.vocab = {}

        def encode(self, text):
            return FakeEncoding(text, self.vocab)

    class FakeArena:
        @staticmethod
        def _query_lex_tokens(_text):
            return {"praxis", "dock"}

    class FakeFixture:
        pass

    fake = FakeFixture()
    fake.arena = FakeArena()
    fake.tokenizer = FakeTokenizer()
    fake.idx_to_node = {0: "target", 1: "sibling", 2: "competitor"}
    fake.nodes_by_id = {
        "target": {"text": "Praxis dock"},
        "sibling": {"text": "Solace key"},
        "competitor": {"text": "Praxis dock planning context"},
    }
    ranges = {"target": (10, 12), "sibling": (12, 14), "competitor": (14, 18)}
    positions, identifier = _identifier_bonus_positions(
        fake,
        {"probe_id": "praxis_fresh", "question": "Praxis dock"},
        [0, 1, 2],
        ranges,
    )
    if positions != [10, 11, 14, 15]:
        raise LiveError(f"identifier exact-token mask drifted: {positions}")

    # Exercise the actual frozen MiniCPM tokenizer without loading the model.
    from tokenizers import Tokenizer as HFTok

    tokenizer_hits = sorted(Path(
        "/home/vader/.cache/huggingface/hub/models--openbmb--MiniCPM3-4B/"
        "snapshots"
    ).glob("*/tokenizer.json"))
    if not tokenizer_hits:
        raise LiveError("MiniCPM3 tokenizer snapshot is missing")
    real_tokenizer = HFTok.from_file(str(tokenizer_hits[-1]))
    expected_terms = {
        "praxis_fresh": {"praxis", "dock"},
        "solace_fresh": {"solace", "key"},
    }
    actual_mask_counts = {}
    for probe in fixture_data["probes"]:
        terms = expected_terms[probe["probe_id"]]
        query_enc, query_indices, _ = _token_indices_for_terms(
            real_tokenizer, probe["question"], terms)
        query_ids = {int(query_enc.ids[index]) for index in query_indices}
        node_counts = {}
        for node in fixture_data["nodes"]:
            encoded, token_indices, _ = _token_indices_for_terms(
                real_tokenizer, node["text"], terms)
            node_counts[node["node_id"]] = sum(
                int(encoded.ids[index]) in query_ids for index in token_indices)
        actual_mask_counts[probe["probe_id"]] = node_counts
    if not (
        actual_mask_counts["praxis_fresh"]["praxis_fact"] > 0
        and actual_mask_counts["praxis_fresh"]["solace_fact"] == 0
        and actual_mask_counts["solace_fresh"]["solace_fact"] > 0
        and actual_mask_counts["solace_fresh"]["praxis_fact"] == 0
    ):
        raise LiveError(f"actual tokenizer identifier masks drifted: {actual_mask_counts}")
    try:
        with gpu_lease(MAX_LEASE_SECONDS + 1, 0):
            pass
    except LiveError:
        lease_guard = True
    else:
        lease_guard = False
    if not lease_guard:
        raise LiveError("GPU lease cap guard failed")
    return {
        "schema": "grm.cmc1_1.gpu_arms.cpu_selftest.v1",
        "status": "PASS",
        "fixture_registration": "PASS",
        "t4_delta_registration": list(T4_DELTAS),
        "identifier_exact_token_positions": positions,
        "identifier_terms": identifier["identifier_terms"],
        "actual_minicpm_tokenizer_mask_counts": actual_mask_counts,
        "lease_cap_guard": "PASS",
        "gpu_imported_or_used": False,
    }


def _validate_run_dir(path: Path, artifact_dir: Path) -> Path:
    root = artifact_dir.resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise LiveError(f"run dir must remain under {root}: {resolved}")
    if not resolved.is_dir():
        raise LiveError(f"run dir does not exist: {resolved}")
    return resolved


def _orchestrate(args: argparse.Namespace) -> int:
    run_dir = _new_run_dir(args.artifact_dir)
    print(f"run_dir={run_dir}", flush=True)
    for index, stage in enumerate(STAGES):
        if index:
            print(f"gpu_gap_s={GAP_SECONDS}", flush=True)
            time.sleep(GAP_SECONDS)
        cmd = [
            sys.executable, str(Path(__file__).resolve()),
            "--stage", stage,
            "--run-dir", str(run_dir),
            "--artifact-dir", str(args.artifact_dir),
            "--gpu", str(args.gpu),
            "--lease-seconds", str(args.lease_seconds),
            "--lock-wait-seconds", str(args.lock_wait_seconds),
        ]
        completed = subprocess.run(cmd, check=False)
        if completed.returncode:
            print(f"stage_failed={stage} returncode={completed.returncode}", flush=True)
            return completed.returncode
    adjudicate = [
        sys.executable, str(ROOT / "scripts" / "grm_cmc1_mechanism.py"),
        "adjudicate", "--run-dir", str(run_dir), "--append-report",
    ]
    return subprocess.run(adjudicate, check=False).returncode


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all",) + STAGES, default="all")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--lease-seconds", type=int, default=DEFAULT_LEASE_SECONDS)
    parser.add_argument("--lock-wait-seconds", type=int, default=7200)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        result = cpu_self_test()
        path = write_content_addressed(
            args.artifact_dir.resolve(), "gpu_arms_cpu_selftest", result)
        print(canonical_json_bytes(result).decode("utf-8"), end="")
        print(f"receipt={path}")
        return 0
    if args.stage == "all":
        if args.run_dir is not None:
            raise LiveError("--run-dir is child-stage-only; all creates a new run")
        return _orchestrate(args)
    if args.run_dir is None:
        raise LiveError("a direct child stage requires --run-dir")
    if int(args.lease_seconds) > MAX_LEASE_SECONDS:
        raise LiveError(f"lease cap exceeds {MAX_LEASE_SECONDS}s")
    gpu = str(args.gpu).strip()
    if not gpu or "," in gpu:
        raise LiveError("--gpu must name exactly one GPU index")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    run_dir = _validate_run_dir(args.run_dir, args.artifact_dir)
    if list(run_dir.glob(f"{args.stage}_*.json")):
        raise LiveError(f"append-only stage already exists: {args.stage}")
    with gpu_lease(int(args.lease_seconds), int(args.lock_wait_seconds)):
        path, result = _run_stage(args.stage, run_dir)
    print(canonical_json_bytes({
        "stage": args.stage,
        "receipt": str(path),
        "schema": result["schema"],
    }).decode("utf-8"), end="")
    if args.stage == "g0" and not result.get("arms_authorized"):
        return 3
    if args.stage == "g1" and not result.get("byte_identity"):
        return 4
    if args.stage == "t2" and not result.get("CMC-G2", {}).get("pass"):
        return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
