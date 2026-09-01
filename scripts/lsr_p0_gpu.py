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

THE TAP (CMC1.2 pattern, adapted to GPT-OSS).  grm_cmc1_gpu_arms taps
MiniCPM3's tc.causal_softmax.  GPT-OSS instead routes through
core.gpt_oss20b_tc.sink_attention_tc / sliding_sink_attention_tc, which
concatenate a per-head SINK LOGIT column, softmax over the concatenation, then
drop that column (core/gpt_oss20b_tc.py::sink_attention_tc).  This module
wraps those two functions -- the same seam DetectorObserver already uses --
and records:

    scores        the scaled-QK values in the engine's compute dtype (bf16),
                  exported value-preserving via .float().numpy(); the offline
                  fp32 work starts HERE, never at a re-derived idealized fp32
                  QK contraction (CMC1.2 law)
    sinks         the per-head sink logits, so the denominator matches
    allowed mask  for sliding layers, which keys were attendable at all

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

    For each call the LAST query row is kept.  Layer identity comes from the
    call ordinal modulo the layer count, the convention
    grm_cmc1_gpu_arms.SDPAInterceptor._wrapped already uses.
    """

    def __init__(self, n_layers: int, *, head_dim: int):
        self.n_layers = int(n_layers)
        self.head_dim = int(head_dim)
        self.calls = 0
        self.forward_count = 0
        self.scores: dict[int, list[np.ndarray]] = {}
        self.sinks: dict[int, np.ndarray] = {}
        self.allowed: dict[int, list[np.ndarray]] = {}
        self.layer_kind: dict[int, str] = {}
        self._module = None
        self._orig_full = None
        self._orig_sliding = None

    def __enter__(self) -> "SinkAttentionTap":
        from core import gpt_oss20b_tc as gpt

        self._module = gpt
        self._orig_full = gpt.sink_attention_tc
        self._orig_sliding = gpt.sliding_sink_attention_tc

        def wrapped_full(query, key, value, sinks, **kwargs):
            self._record(query, key, sinks, kind="full", window=None)
            return self._orig_full(query, key, value, sinks, **kwargs)

        def wrapped_sliding(query, key, value, sinks, *, sliding_window, **kwargs):
            self._record(query, key, sinks, kind="sliding",
                         window=int(sliding_window))
            return self._orig_sliding(
                query, key, value, sinks,
                sliding_window=sliding_window, **kwargs)

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

    def _record(self, query, key, sinks, *, kind: str, window: int | None) -> None:
        from core.mistral7b_tc import tc

        layer = self.calls % self.n_layers
        if layer == 0:
            self.forward_count += 1
        self.calls += 1
        self.layer_kind[layer] = kind

        # The engine's own scaled-QK materialization, in its compute dtype.
        # The scale reproduces GptOssAttentionTC.scaling (head_dim ** -0.5)
        # exactly; the matmul itself is the engine's kernel.
        scale = float(self.head_dim ** -0.5)
        scores = tc.matmul(query, key, alpha=scale, trans_b=True)
        rows = int(scores.shape[2])
        last = scores.slice(2, rows - 1, 1)
        array = np.asarray(last.float().numpy(), dtype=np.float32)[0, :, 0, :]
        self.scores.setdefault(layer, []).append(array)
        if layer not in self.sinks:
            self.sinks[layer] = np.asarray(
                sinks.float().numpy(), dtype=np.float32).reshape(-1)

        total = int(key.shape[2])
        if kind == "sliding" and window:
            q_abs = total - 1
            index = np.arange(total, dtype=np.int64)
            allowed = (index <= q_abs) & (index > (q_abs - int(window)))
        else:
            allowed = np.ones(total, dtype=bool)
        self.allowed.setdefault(layer, []).append(allowed)

    def finalize(self, ngen: int) -> None:
        """Drop the trailing cache-commit forward, per DetectorObserver.finish.

        ArenaCache._attempt runs one extra forward solely to commit the final
        predicted token to KV; its logits are unused and it is not an answer
        readout position.
        """
        if self.forward_count == int(ngen) + 1:
            for store in (self.scores, self.allowed):
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
        for layer, rows in self.scores.items():
            if not rows:
                continue
            width = max(int(row.shape[-1]) for row in rows)
            padded, masks = [], []
            for index, row in enumerate(rows):
                have = int(row.shape[-1])
                block = np.zeros((row.shape[0], width), dtype=np.float32)
                block[:, :have] = row
                padded.append(block)
                mask = np.zeros(width, dtype=bool)
                mask[:have] = self.allowed[layer][index][:have]
                masks.append(mask)
            scores[layer] = np.stack(padded, axis=0)
            allowed[layer] = np.stack(masks, axis=0)
        return scores, dict(self.sinks), allowed


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

        tap = SinkAttentionTap(
            len(arena.m.layers), head_dim=int(model.cfg.head_dim))
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
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
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
