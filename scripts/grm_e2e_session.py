#!/usr/bin/env python3
"""Composed GPT-OSS + GRM end-to-end session receipt driver.

P1 Leg 2 driver for docs/GRM_E2E_RECEIPT_PLAN.md. The full run is left for
the lead; use ``--mode smoke`` for the bounded 8-10 turn proof.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")

import tensor_cuda as tc  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from core.gpt_oss20b_tc import (  # noqa: E402
    GptOss20B_TC,
    gpt_oss_grm_dialect_kwargs,
)
from core.graft_arena import GQAArenaCache  # noqa: E402
from core import kv_graft  # noqa: E402
from core.graft_repository import GraftRepository  # noqa: E402
from core.grm_three_pass import (  # noqa: E402
    MemoryLedgerBuilder,
    Pass2ReadOnlyGuard,
    StagedWorkingSetResolver,
    TurnStepIOTracker,
    arena_state_sha256,
    build_route_receipt,
)
from core import paging_telemetry as _paging_telemetry  # noqa: E402
from core import grm_demand  # noqa: E402
from core.grm_frame import (  # noqa: E402
    ENV_NAME as PERSISTENT_BOAT_ENV,
    env_persistent_boat,
    ephemeral_frame_enabled,
)
from core.grm_supersession import (  # noqa: E402
    sup_resolve_cli_argv,
    sup_resolve_enabled,
)
from core.grm_admission import (  # noqa: E402
    admission_info_fields,
    adm_decisive_cli_argv,
    adm_decisive_enabled,
    chunk_trip_cap,
    decisive_admission_profile,
    fit_info_fields,
    identifier_serving_decision,
    mountable_budget,
    plan_priority_fit,
    shuttle_trip_cap,
    split_info_fields,
)
from scripts.grm_probe_ladder import (  # noqa: E402
    build_probe_ladder_attempts,
    identifier_tokens_from_parts,
    probe_ladder_cli_argv,
    probe_ladder_enabled,
    rank1_covers_identifiers,
)


SNAPSHOT = (
    "/home/vader/.cache/huggingface/hub/models--openai--gpt-oss-20b/"
    "snapshots/6cee5e81ee83917806bbde320786a8fb61efebee"
)
NATIVE_LIB = ROOT / "cpp" / "build" / "libgrm_runtime.so"
# Harmony SYS sink (Leg-1 / diagnosis-validated). n_sink derives from this;
# default ArenaCache sink "<conversation>\n" mis-seats GPT-OSS YARN.
HARMONY_SINK = (
    "<|start|>system<|message|>You are ChatGPT. Reasoning: low. "
    "Valid channel: final.<|end|>"
)
SYSTEM_PREFIX = (
    "<|start|>system<|message|>You are ChatGPT. Reasoning: low. "
    "Valid channel: final.<|end|><|start|>user<|message|>"
)
ASSISTANT_FINAL = "<|end|><|start|>assistant<|channel|>final<|message|>"
HARMONY_STOPS = (
    "<|return|>", "<|end|>", "<|start|>user", "<|start|>assistant")


class GptOssGQAArenaCache(GQAArenaCache):
    POSITION_LAW = "rope_full_yarn"
    STATE_KIND = "kv"
    GRAFTABILITY = "seat_remountable"
    REMOUNTABLE = True
    COMPOSITION = "multi_mount"


def harmony_turn(user_text: str, assistant_text: str | None) -> str:
    prompt = f"{SYSTEM_PREFIX}{user_text}{ASSISTANT_FINAL}"
    if assistant_text is None:
        return prompt
    return f"{prompt}{assistant_text}<|end|>"


def plant_acceptance(event: dict[str, Any]) -> str:
    """Scripted assistant acceptance for Leg-1 feed() plant turns.

    Value-bearing complete text — never free-generated. Embeds the fact
    phrase so the deposited K/V carries the payload (not a refusal).
    """
    phrase = event.get("fact_phrase")
    if not phrase:
        phrase = f"current {event['fact_id']} value is {event['value']}"
    return f"Understood — the {phrase}."


def fact_turn(fact_id: str, value: str, *, label: str | None = None) -> dict[str, Any]:
    label = label or fact_id
    phrase = f"current {fact_id} value is {value}"
    event = {
        "kind": "fact",
        "fact_id": fact_id,
        "value": value,
        "user": (
            f"Memory planting turn. fact {fact_id}. The {phrase}. "
            f"If asked later for {label}, answer {value} only."
        ),
        "fact_phrase": phrase,
    }
    event["assistant"] = plant_acceptance(event)
    return event


def supersede_turn(fact_id: str, old: str, new: str) -> dict[str, Any]:
    old_phrase = f"current {fact_id} value is {old}"
    new_phrase = f"current {fact_id} value is {new}"
    event = {
        "kind": "supersede",
        "fact_id": fact_id,
        "old_value": old,
        "value": new,
        "user": (
            f"Authoritative update turn. fact {fact_id}. The {new_phrase}. "
            f"This is the current value. If asked later, answer {new} only."
        ),
        "correction_command": f"correct memory: {old_phrase} => The {new_phrase}.",
        "fact_phrase": new_phrase,
    }
    event["assistant"] = plant_acceptance(event)
    return event


def filler_turn(tag: str) -> dict[str, Any]:
    return {
        "kind": "filler",
        "user": (
            f"Session continuity filler {tag}. Acknowledge briefly. "
            "Do not change any stored fact values."
        ),
    }


def probe_turn(
    fact_id: str,
    expected: str,
    *,
    source_turn: int,
    supersession: bool = False,
    old_value: str | None = None,
) -> dict[str, Any]:
    return {
        "kind": "probe",
        "fact_id": fact_id,
        "expected": expected,
        "accepts": [expected],
        "old_value": old_value,
        "source_turn": int(source_turn),
        "supersession": bool(supersession),
        "user": (
            f"Recall probe. What is the current {fact_id} value? "
            "Reply with only the value."
        ),
    }


def build_smoke_script() -> list[dict[str, Any]]:
    # Fork-A code-shaped values (exact map from
    # artifacts/grm_e2e/smoke_session_20260708_154313/run_config.json).
    return [
        fact_turn("orion pin", "Auric-4-Alpha"),
        filler_turn("smoke alpha"),
        supersede_turn("orion pin", "Auric-4-Alpha", "Kestrel-9-Tango"),
        fact_turn("cypher bridge", "Vortex-3-Sierra"),
        filler_turn("smoke beta"),
        probe_turn("cypher bridge", "Vortex-3-Sierra", source_turn=3),
        probe_turn(
            "orion pin", "Kestrel-9-Tango", source_turn=2,
            supersession=True, old_value="Auric-4-Alpha"),
        filler_turn("smoke gamma"),
        filler_turn("smoke delta"),
        filler_turn("smoke epsilon"),
    ]


def build_full_script() -> list[dict[str, Any]]:
    # Smoke values recovered from run_config; remaining facts re-derived
    # in the same Word-N-Phonetic code-shaped style.
    return [
        fact_turn("orion pin", "Auric-4-Alpha"),
        filler_turn("full alpha"),
        fact_turn("lyra dock", "Nadir-1-Delta"),
        filler_turn("full beta"),
        fact_turn("cypher bridge", "Vortex-3-Sierra"),
        probe_turn("orion pin", "Auric-4-Alpha", source_turn=0),
        supersede_turn("orion pin", "Auric-4-Alpha", "Kestrel-9-Tango"),
        fact_turn("nova key", "Quartz-5-Bravo"),
        filler_turn("full gamma"),
        probe_turn("cypher bridge", "Vortex-3-Sierra", source_turn=4),
        fact_turn("mira seal", "Silver-6-Charlie"),
        supersede_turn("lyra dock", "Nadir-1-Delta", "Zenith-2-Echo"),
        filler_turn("full delta"),
        probe_turn(
            "orion pin", "Kestrel-9-Tango", source_turn=6,
            supersession=True, old_value="Auric-4-Alpha"),
        fact_turn("terra port", "Harbor-8-Golf"),
        filler_turn("full epsilon"),
        probe_turn(
            "lyra dock", "Zenith-2-Echo", source_turn=11,
            supersession=True, old_value="Nadir-1-Delta"),
        supersede_turn("mira seal", "Silver-6-Charlie", "Gold-7-Foxtrot"),
        filler_turn("full zeta"),
        probe_turn("nova key", "Quartz-5-Bravo", source_turn=7),
        fact_turn("ember code", "Violet-2-Hotel"),
        filler_turn("full eta"),
        probe_turn(
            "mira seal", "Gold-7-Foxtrot", source_turn=17,
            supersession=True, old_value="Silver-6-Charlie"),
        filler_turn("full theta"),
        probe_turn("terra port", "Harbor-8-Golf", source_turn=14),
        filler_turn("full iota"),
        probe_turn("ember code", "Violet-2-Hotel", source_turn=20),
        fact_turn("atlas tone", "Cobalt-1-India"),
        filler_turn("full kappa"),
        filler_turn("full lambda"),
        probe_turn("atlas tone", "Cobalt-1-India", source_turn=27),
        filler_turn("full mu"),
        fact_turn("polaris mark", "Marble-4-Juliet"),
        filler_turn("full nu"),
    ]


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("smoke", "full"), default="full")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--session-dir", type=Path, default=None)
    p.add_argument("--model-dir", type=Path, default=Path(SNAPSHOT))
    p.add_argument("--native-lib", type=Path, default=NATIVE_LIB)
    p.add_argument("--ngen", type=int, default=None)
    p.add_argument("--max-trips", type=int, default=1)
    p.add_argument("--live-turns", type=int, default=None)
    # Leg-1 proven / diagnosis: width 384 → live_shift≈387 collapses GPT-OSS.
    p.add_argument("--arena-width", type=int, default=96)
    p.add_argument("--max-live", type=int, default=4096)
    # F-COLD (GRM3P-P4): bounded device-byte budget for saved graft tensors.
    # Least-recently-mounted saved nodes spill to cold storage above it via the
    # existing GraftRepository LRU pager (_page); step-1 prep pages them back in
    # via node_loader. None = unbounded (registered default, prior frames).
    p.add_argument("--vram-budget-mb", type=int, default=None,
                   help="graft-tensor device byte budget in MB (LRU spill "
                        "above it); None = unbounded (default)")
    # Fork A: production-realistic multi-mount (top-k 2-3, not argmax-only).
    # Arena.step already slices ranking into self.topk mounts; this flag is
    # the driver call-site width of that slice (default 3 for this receipt).
    p.add_argument("--topk", type=int, default=3,
                   help="route multi-mount count (arena.topk); Fork-A default 3")
    p.add_argument("--restart-after", type=int, default=None)
    p.add_argument("--skip-gpu-idle-check", action="store_true")
    p.add_argument(
        "--turn-pipeline",
        choices=("single", "three_pass"),
        default="single",
        help="turn scheduler (default single preserves the registered path)",
    )
    # GRM3P-LADDER-ON: probe ladder is DEFAULT-ON permanently. Escape to the
    # legacy multimount path with --no-probe-ladder or GRM_PROBE_LADDER=0.
    # BooleanOptionalAction: None = unset on CLI → fall through to env/default.
    p.add_argument(
        "--probe-ladder",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enforce Arena laws on Fork-A probe turns (default ON; "
             "escape with --no-probe-ladder or GRM_PROBE_LADDER=0)",
    )
    # GRM-SUP-L2-ON: M5-edge mount resolution is permanently DEFAULT ON.
    # The CLI is primarily for frozen experiment frames and restart locking;
    # operators can restore the legacy path with GRM_SUP_RESOLVE=0.
    p.add_argument(
        "--sup-resolve",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="resolve M5 supersession lineages at mount time (default ON; "
             "escape with --no-sup-resolve or GRM_SUP_RESOLVE=0)",
    )
    # GRM-ADM2: frozen A-DEC admission is permanently DEFAULT ON. The CLI
    # pins experiment/restart frames; GRM_ADM_DECISIVE=0 is the registered
    # byte-exact escape to fixed k=3 admission.
    p.add_argument(
        "--adm-decisive",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="use frozen A-DEC mount admission (default ON; escape with "
             "--no-adm-decisive or GRM_ADM_DECISIVE=0)",
    )
    return p.parse_args(argv)


def default_session_dir(mode: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT / "artifacts" / "grm_e2e" / f"{mode}_session_{stamp}"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def active_compute_pids() -> list[str]:
    proc = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=10,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def wait_for_idle_gpu(max_wait_s: int = 900) -> dict[str, Any]:
    waited = 0
    checks = []
    while True:
        pids = active_compute_pids()
        checks.append({"waited_s": waited, "pids": pids})
        if not pids:
            return {"idle": True, "waited_s": waited, "checks": checks}
        if waited >= max_wait_s:
            return {"idle": False, "waited_s": waited, "checks": checks}
        print(f"GPU busy with compute PIDs {pids}; waiting 60s", flush=True)
        time.sleep(60)
        waited += 60


def vram_snapshot() -> dict[str, Any]:
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=10,
    )
    raw = proc.stdout.strip()
    out: dict[str, Any] = {"raw": raw, "returncode": int(proc.returncode)}
    if proc.returncode == 0 and raw:
        rows = []
        for line in raw.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                rows.append({"used_mb": int(parts[0]), "total_mb": int(parts[1])})
        out["gpus"] = rows
    else:
        out["stderr"] = proc.stderr.strip()
    return out


class TurnTimers:
    def __init__(self) -> None:
        self.route_ms = 0.0
        self.route_calls = 0
        self.deposit_ms = 0.0
        self.deposit_calls = 0
        self.mount_ms = 0.0
        self.mount_calls = 0
        self.infer_ms = 0.0
        self.infer_calls = 0
        self.importance_ms = 0.0
        self.importance_calls = 0
        self.supersession_ms = 0.0
        self.supersession_calls = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "route_wall_ms": self.route_ms,
            "route_calls": self.route_calls,
            "deposit_wall_ms": self.deposit_ms,
            "deposit_calls": self.deposit_calls,
            "mount_wall_ms": self.mount_ms,
            "mount_calls": self.mount_calls,
            "infer_wall_ms": self.infer_ms,
            "infer_calls": self.infer_calls,
            "importance_bookkeeping_wall_ms": self.importance_ms,
            "importance_bookkeeping_calls": self.importance_calls,
            "supersession_wall_ms": self.supersession_ms,
            "supersession_calls": self.supersession_calls,
        }


def install_turn_timers(arena) -> tuple[TurnTimers, Any]:
    timers = TurnTimers()
    original = {
        "route": arena.route,
        "swap": arena.swap,
        "deposit": arena.deposit,
        "deposit_from_cache": arena.deposit_from_cache,
        "forward": arena._forward,
        "importance": arena._commit_s4_attempt,
    }

    def route_wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return original["route"](*args, **kwargs)
        finally:
            timers.route_ms += (time.perf_counter() - t0) * 1000.0
            timers.route_calls += 1

    def swap_wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return original["swap"](*args, **kwargs)
        finally:
            timers.mount_ms += (time.perf_counter() - t0) * 1000.0
            timers.mount_calls += 1

    def deposit_wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return original["deposit"](*args, **kwargs)
        finally:
            timers.deposit_ms += (time.perf_counter() - t0) * 1000.0
            timers.deposit_calls += 1

    def deposit_from_cache_wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return original["deposit_from_cache"](*args, **kwargs)
        finally:
            timers.deposit_ms += (time.perf_counter() - t0) * 1000.0
            timers.deposit_calls += 1

    def forward_wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return original["forward"](*args, **kwargs)
        finally:
            timers.infer_ms += (time.perf_counter() - t0) * 1000.0
            timers.infer_calls += 1

    def importance_wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return original["importance"](*args, **kwargs)
        finally:
            timers.importance_ms += (time.perf_counter() - t0) * 1000.0
            timers.importance_calls += 1

    arena.route = route_wrapper
    arena.swap = swap_wrapper
    arena.deposit = deposit_wrapper
    arena.deposit_from_cache = deposit_from_cache_wrapper
    arena._forward = forward_wrapper
    arena._commit_s4_attempt = importance_wrapper

    def restore() -> None:
        arena.route = original["route"]
        arena.swap = original["swap"]
        arena.deposit = original["deposit"]
        arena.deposit_from_cache = original["deposit_from_cache"]
        arena._forward = original["forward"]
        arena._commit_s4_attempt = original["importance"]

    return timers, restore


def contains_accept(answer: str, values: list[str]) -> bool:
    for value in values:
        pattern = r"(?<![A-Za-z0-9_-])" + re.escape(value) + r"(?![A-Za-z0-9_-])"
        if re.search(pattern, answer, flags=re.IGNORECASE):
            return True
    return False


def score_probe(answer: str, event: dict[str, Any]) -> dict[str, Any]:
    accepts = [str(x) for x in event.get("accepts", ())]
    old_value = event.get("old_value")
    hit = contains_accept(answer, accepts)
    stale_hit = contains_accept(answer, [str(old_value)]) if old_value else False
    return {
        "fact_id": event["fact_id"],
        "expected": event["expected"],
        "accepts": accepts,
        "old_value": old_value,
        "answer": answer,
        "pass": bool(hit and not stale_hit),
        "contains_expected": bool(hit),
        "contains_stale": bool(stale_hit),
    }


def collect_route_diagnostics(
    arena,
    probe_text: str,
    source_node_id: int | None,
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    """Driver-side top-K route ranking for probe turns (ids + scores).

    Uses the same arena.route() entry as step(), then recomputes Python-side
    scores for the returned nodes (native/CUDA path returns ids only).
    """
    import math

    live_idx = {g for g, _ in arena.live_segs if g is not None}
    n = len(getattr(arena, "grafts", ()) or ())
    limit = max(int(top_k), n) if n else int(top_k)
    ranking = list(arena.route(probe_text, exclude=live_idx, limit=limit) or [])
    backend = getattr(arena, "last_route_backend", None)

    scores: dict[int, float] = {}
    try:
        p = arena._probe_key(probe_text)
        qrare = arena._rare_tokens(probe_text)
        cand = list(ranking) if ranking else [
            i for i in range(n) if i not in live_idx]
        base = arena._vector_route_scores(p, cand)
        if base is None:
            base = {}
            for i in cand:
                sc = arena._cent_score(p, arena.grafts[i])
                if sc == sc:  # finite
                    base[i] = float(sc)
        base = arena._normalize_scores(base) or {}
        for i in cand:
            if i not in base:
                continue
            sc = float(base[i]) + float(arena._lex_bonus(qrare, arena.grafts[i]))
            if math.isfinite(sc):
                scores[int(i)] = sc
    except Exception as err:  # diagnostics must not abort the turn
        scores = {}
        score_err = repr(err)
    else:
        score_err = None

    top = []
    for rank, idx in enumerate(ranking[: int(top_k)], start=1):
        idx_i = int(idx)
        g = arena.grafts[idx_i] if 0 <= idx_i < n else {}
        text = str(g.get("text", "") or "")
        top.append({
            "rank": rank,
            "node_id": idx_i,
            "score": scores.get(idx_i),
            "text_prefix": text[:120],
            "contains_value_hint": bool(
                re.search(r"[A-Za-z]+-\d+-[A-Za-z]+", text)),
        })

    source_rank = None
    if source_node_id is not None:
        try:
            source_rank = ranking.index(int(source_node_id)) + 1
        except ValueError:
            source_rank = None

    return {
        "top5": top,
        "source_node_id": (
            int(source_node_id) if source_node_id is not None else None),
        "source_rank": source_rank,
        "ranking_len": len(ranking),
        "route_backend": backend,
        "score_error": score_err,
        "live_excluded": sorted(int(x) for x in live_idx),
        # Full rank order kept so max-pool length bias stays visible even
        # when the mount slice is wider than 1 (Fork A: do not hide bias).
        "ranking_ids": [int(x) for x in ranking],
    }


# ---------------------------------------------------------------------------
# GRM-LSR-P2C — the fixes flag.
#
# G2 Arm 0 (reproduction) needs the P2A+P2C fixes OFF so the replay can be
# shown lived-equivalent BEFORE Arm 1 counts as evidence.  The flag is
# DEFAULT ON: production keeps the fixes, and only a gate that has declared
# itself a reproduction arm turns them off.
# ---------------------------------------------------------------------------
LSR_FIXES_ENV = "GRM_LSR_FIXES"
_LSR_ENV_FALSE = frozenset(("0", "false", "no", "off"))


def lsr_fixes_enabled(explicit: bool | None = None) -> bool:
    """Resolve the P2A+P2C fix switch: explicit caller, then env, then ON.

    Fails CLOSED to ON for an unknown token, the same direction the A-DEC
    switch fails: an operator who mistypes the value keeps the fixes.
    """
    if explicit is not None:
        return bool(explicit)
    value = str(os.environ.get(LSR_FIXES_ENV, "")).strip().casefold()
    if value in _LSR_ENV_FALSE:
        return False
    return True


def _stamp_grounding_receipt(arena, ans, picks, user_text, info) -> None:
    """SC1.1: stamp the glyph receipt when the arena can produce one.

    The driver is called with FAKE arenas by several fixtures (a namespace
    carrying only the handful of attributes the ladder touches).  The receipt
    is a receipt, not a verdict -- an arena that cannot compute it simply
    does not get the two fields, and nothing about serving changes.  A real
    ``ArenaCache`` always can.
    """
    stamp = getattr(arena, "_grounding_receipt", None)
    if stamp is None:
        return
    stamp(ans, picks, user_text, info)


def _budget_fit_mounts(arena, picks: list[int]) -> list[int]:
    """Pack mounts into arena.width in the given order (rank order)."""
    budget = int(arena.width)
    out: list[int] = []
    used = 0
    for i in picks:
        n = int(arena.grafts[int(i)]["ntok"])
        if used + n <= budget:
            out.append(int(i))
            used += n
    return out


def _probe_identifier_tokens(arena, user_text: str) -> set[str]:
    """Identifier tokens for point-lookup / precise-first (reuse arena lex)."""
    return identifier_tokens_from_parts(
        arena._rare_tokens(user_text),
        arena._query_lex_tokens(user_text),
    )


def _probe_is_point_lookup(arena, user_text: str) -> bool:
    """Identifier-shaped probe ⇒ RECENCY LAW applies (exclude live seats)."""
    return bool(_probe_identifier_tokens(arena, user_text))


def _probe_rank1_covers_identifiers(
    arena, ranking: list[int], id_tokens: set[str]
) -> list[int] | None:
    """Precise-first: rank-1 alone when it covers all probe identifiers."""
    if not ranking or not id_tokens:
        return None
    g0 = arena.grafts[int(ranking[0])]
    if "rare" not in g0:
        g0["rare"] = arena._rare_tokens(g0.get("text", ""))
    if rank1_covers_identifiers(
        id_tokens,
        g0.get("rare") or (),
        arena._node_text_tokens(g0.get("text", "")),
    ):
        return [int(ranking[0])]
    return None


def recency_augmented_probe(
    probe_text: str,
    transcript: list[dict[str, Any]],
    *,
    live_turns: int,
) -> str:
    """Frozen exploratory construction: recent complete turns + bare query.

    This probe is routing-only and is never fed to user-visible inference.
    The main working-set probe remains the E4 bare message.
    """
    recent = transcript[-max(0, int(live_turns)):]
    lines = ["Routing recency context (exploratory only)."]
    for row in recent:
        lines.append(f"Recent user: {row.get('user', '')}")
        lines.append(f"Recent assistant: {row.get('assistant', '')}")
    lines.append(f"Current user: {probe_text}")
    return "\n".join(lines)


def _route_backend_pair(
    arena,
    probe_text: str,
    *,
    exclude: set[int],
    timers: TurnTimers,
    semantic_only: bool = False,
) -> dict[str, Any]:
    """Capture one Q probe, then rank its identical fp32 values twice."""
    eligible = [
        int(i) for i in arena._route_cand_base() if int(i) not in exclude
    ]
    if not eligible:
        return {
            "routed": False,
            "candidate_ids": [],
            "candidate_count": 0,
            "python_backend": None,
            "cuda_backend": None,
            "cuda_policy_backend": None,
            "semantic_only": bool(semantic_only),
            "python_ranking_ids": [],
            "cuda_ranking_ids": [],
            "ranking_ids_byte_equal": True,
            "probe_capture_wall_ms": 0.0,
            "pair_wall_ms": 0.0,
        }
    # The current exact CUDA top-k ABI is bounded at 16. The registered smoke
    # repository stays below it; fail closed if that frame ever stops doing so.
    if len(eligible) > 16:
        raise RuntimeError(
            "exact CUDA full-repository ranking requires <=16 eligible "
            f"candidates in this registered frame, found {len(eligible)}")

    pair_started = time.perf_counter()
    capture_started = time.perf_counter()
    probe_key = arena._probe_key(probe_text)
    capture_wall_ms = (time.perf_counter() - capture_started) * 1000.0
    # install_turn_timers wraps route(), while this deliberately shared probe
    # capture sits outside route(); attribute it to the route stage explicitly.
    timers.route_ms += capture_wall_ms

    python_ranking = list(arena.route(
        probe_text,
        exclude=exclude,
        limit=len(eligible),
        route_backend="python",
        query_lex=False if semantic_only else None,
        semantic_only=semantic_only,
        probe_key=probe_key,
    ) or [])
    python_backend = str(arena.last_route_backend)
    cuda_ranking = list(arena.route(
        probe_text,
        exclude=exclude,
        limit=len(eligible),
        route_backend="cuda",
        query_lex=False if semantic_only else None,
        semantic_only=semantic_only,
        probe_key=probe_key,
    ) or [])
    cuda_backend = str(arena.last_route_backend)
    cuda_policy_backend = getattr(
        arena, "last_route_policy_backend", None)
    pair_wall_ms = (time.perf_counter() - pair_started) * 1000.0
    return {
        "routed": True,
        "candidate_ids": eligible,
        "candidate_count": len(eligible),
        "python_backend": python_backend,
        "cuda_backend": cuda_backend,
        "cuda_policy_backend": cuda_policy_backend,
        "semantic_only": bool(semantic_only),
        "python_ranking_ids": [int(i) for i in python_ranking],
        "cuda_ranking_ids": [int(i) for i in cuda_ranking],
        "ranking_ids_byte_equal": (
            [int(i) for i in python_ranking]
            == [int(i) for i in cuda_ranking]),
        "probe_capture_wall_ms": capture_wall_ms,
        "pair_wall_ms": pair_wall_ms,
    }


def _ranking_rows(arena, ranking: list[int]) -> list[dict[str, Any]]:
    rows = []
    for rank, node_id in enumerate(ranking, start=1):
        graft = arena.grafts[int(node_id)]
        rows.append({
            "rank": int(rank),
            "node_id": int(node_id),
            "text_prefix": str(graft.get("text", "") or "")[:120],
        })
    return rows


def _source_rank(ranking: list[int], source_node_id: int | None) -> int | None:
    if source_node_id is None:
        return None
    try:
        return ranking.index(int(source_node_id)) + 1
    except ValueError:
        return None


def _prep_primary_mounts(
    arena,
    event: dict[str, Any],
    ranking: list[int],
    *,
    topk: int,
) -> list[int]:
    """Mirror the primary pass-2 mount plan without running inference."""
    if event["kind"] in ("fact", "supersede"):
        # Complete scripted turns historically do not route/mount. Staging is
        # still performed, but changing seats would change their harvested KV
        # and violate the registered equivalence gate.
        return []
    head = [int(i) for i in ranking[:max(1, int(topk))]]
    if event["kind"] == "probe":
        return sorted(_budget_fit_mounts(arena, head))

    qrare = arena._rare_tokens(event["user"])
    expanded = arena._descent_expand(
        sorted(head), ("era",), qrare=qrare)
    resolved = arena._resolve_revision_mounts(expanded)
    budget = int(arena.width)
    out = []
    used = 0
    for node_id in resolved:
        ntok = int(arena.grafts[int(node_id)]["ntok"])
        if used + ntok <= budget:
            out.append(int(node_id))
            used += ntok
    return sorted(out)


def _working_set_path(paths: dict[str, Path], turn_idx: int) -> Path:
    return paths["working_set"] / f"turn_{turn_idx:04d}" / "working_set.json"


def run_step1_prep(
    repo: GraftRepository,
    event: dict[str, Any],
    turn_idx: int,
    *,
    transcript: list[dict[str, Any]],
    source_node_id: int | None,
    paths: dict[str, Path],
    args: argparse.Namespace,
    timers: TurnTimers,
    io_tracker: TurnStepIOTracker,
) -> tuple[dict[str, Any], Path]:
    """Search, compare, page in, and prepare the primary working set."""
    arena = repo.arena
    started = time.perf_counter()
    probe_text = str(event["user"])
    live_excluded = {
        int(g) for g, _ntok in arena.live_segs if g is not None
    }
    page_ins_before = int(getattr(arena, "page_ins", 0))
    main_pair = _route_backend_pair(
        arena,
        probe_text,
        exclude=set(),
        timers=timers,
        semantic_only=event["kind"] in ("fact", "supersede"),
    )
    repository_ranking = list(main_pair["cuda_ranking_ids"])
    python_repository_ranking = list(main_pair["python_ranking_ids"])
    ranking = [
        int(i) for i in repository_ranking if int(i) not in live_excluded
    ]
    python_ranking = [
        int(i) for i in python_repository_ranking
        if int(i) not in live_excluded
    ]
    route_window = max(
        int(args.topk), (int(args.max_trips) + 1) * int(args.topk))
    staged_ids = [int(i) for i in ranking[:route_window]]

    # Device materialization is complete before any mount surgery.
    arena._ensure_h(staged_ids)
    prepared_mount_ids = _prep_primary_mounts(
        arena, event, ranking, topk=int(args.topk))
    mount_surgery_prepped = False
    mount_surgery_reason = None
    if event["kind"] in ("fact", "supersede"):
        mount_surgery_reason = "complete_turn_has_no_retrieval_mount"
    elif arena.caches is None:
        mount_surgery_reason = "cache_bootstrap_deferred_to_inference"
    else:
        arena.swap(prepared_mount_ids)
        mount_surgery_prepped = True

    enrichment = None
    if event["kind"] == "probe":
        augmented = recency_augmented_probe(
            probe_text, transcript, live_turns=int(args.live_turns))
        bare_exploratory_pair = _route_backend_pair(
            arena,
            probe_text,
            exclude=set(),
            timers=timers,
            semantic_only=True,
        )
        augmented_pair = _route_backend_pair(
            arena,
            augmented,
            exclude=set(),
            timers=timers,
            semantic_only=True,
        )
        bare_exploratory_ranking = [
            int(i) for i in bare_exploratory_pair["cuda_ranking_ids"]
            if int(i) not in live_excluded
        ]
        bare_exploratory_python_ranking = [
            int(i) for i in bare_exploratory_pair["python_ranking_ids"]
            if int(i) not in live_excluded
        ]
        augmented_ranking = [
            int(i) for i in augmented_pair["cuda_ranking_ids"]
            if int(i) not in live_excluded
        ]
        augmented_python_ranking = [
            int(i) for i in augmented_pair["python_ranking_ids"]
            if int(i) not in live_excluded
        ]
        enrichment = {
            "status": "exploratory_report_only",
            "construction": (
                "last live_turns complete transcript turns, labelled Recent "
                "user/assistant, followed by labelled Current user"),
            "comparison_lexical_policy": "semantic_only_exact_cuda",
            "bare_probe": probe_text,
            "augmented_probe": augmented,
            "bare_ranking_ids": bare_exploratory_ranking,
            "augmented_ranking_ids": augmented_ranking,
            "bare_source_rank": _source_rank(
                bare_exploratory_ranking, source_node_id),
            "augmented_source_rank": _source_rank(
                augmented_ranking, source_node_id),
            "bare_recall_at_3": bool(
                source_node_id is not None
                and int(source_node_id) in bare_exploratory_ranking[:3]),
            "augmented_recall_at_3": bool(
                source_node_id is not None
                and int(source_node_id) in augmented_ranking[:3]),
            "bare_python_ranking_ids": bare_exploratory_python_ranking,
            "augmented_python_ranking_ids": (
                augmented_python_ranking),
            "bare_cuda_python_byte_equal": (
                bare_exploratory_pair["ranking_ids_byte_equal"]),
            "augmented_cuda_python_byte_equal": (
                augmented_pair["ranking_ids_byte_equal"]),
        }

    prep_wall_ms = (time.perf_counter() - started) * 1000.0
    page_in_count = (
        int(getattr(arena, "page_ins", 0)) - page_ins_before)
    step1_io = io_tracker.receipt()["steps"]["1_prep"]
    source_rank = _source_rank(ranking, source_node_id)
    receipt = {
        "schema": "grm.three_pass.working_set.v1",
        "turn": int(turn_idx),
        "turn_pipeline": "three_pass",
        "step": 1,
        "event_kind": event["kind"],
        "probe_policy": "bare_message_e4",
        "probe_used": probe_text,
        "lexical_policy": (
            "semantic_only_exact_cuda"
            if main_pair["semantic_only"]
            else "existing_query_lex_exact_python_rescore"),
        "live_excluded_ids": sorted(live_excluded),
        "routed": bool(main_pair["routed"]),
        "route_backend": main_pair["cuda_backend"],
        "route_policy_backend": main_pair["cuda_policy_backend"],
        "python_backend": main_pair["python_backend"],
        "ranking_ids_byte_equal": main_pair["ranking_ids_byte_equal"],
        "repository_ranking_ids": repository_ranking,
        "python_repository_ranking_ids": python_repository_ranking,
        "ranking_ids": ranking,
        "python_ranking_ids": python_ranking,
        "ranking": _ranking_rows(arena, ranking),
        "repository_candidate_count": main_pair["candidate_count"],
        "staged_ids": staged_ids,
        "prepared_mount_ids": prepared_mount_ids,
        "mount_surgery_prepped": mount_surgery_prepped,
        "mount_surgery_reason": mount_surgery_reason,
        "page_in_count": int(page_in_count),
        "page_in_ids": [
            int(row["node_id"]) for row in step1_io["events"]
            if row.get("kind") == "graft_page_in"
            and row.get("success") is True
        ],
        "upload_count": int(step1_io["upload_count"]),
        "upload_events": [
            row for row in step1_io["events"]
            if row.get("kind") in (
                "graft_payload_upload", "cuda_route_bank_upload")
        ],
        "prep_route_wall_ms": float(main_pair["pair_wall_ms"]),
        "probe_capture_wall_ms": float(
            main_pair["probe_capture_wall_ms"]),
        "prep_wall_ms": float(prep_wall_ms),
        "source_node_id": (
            int(source_node_id) if source_node_id is not None else None),
        "source_rank": source_rank,
        "direct_route_recall_at_3": (
            bool(source_node_id is not None
                 and int(source_node_id) in ranking[:3])
            if event["kind"] == "probe" else None),
        "source_present_in_staged_set": (
            bool(source_node_id is not None
                 and int(source_node_id) in staged_ids)
            if event["kind"] == "probe" else None),
        "l1": {
            "route_l1_calls": 0,
            "payload_l1_resolutions": 0,
            "l2_miss_count": 0,
            "l2_misses": [],
        },
        "enrichment": enrichment,
    }
    receipt_path = _working_set_path(paths, turn_idx)
    write_json(receipt_path, receipt)
    return receipt, receipt_path


def _probe_finish_deposit(
    repo: GraftRepository,
    before: dict[str, Any],
    user_text: str,
    ans: str,
    info: dict[str, Any],
    *,
    defer_memory: bool,
) -> dict[str, Any]:
    """Repository deposit bookkeeping shared by legacy and ladder probe paths.

    GRM-EB1: this is the single funnel every probe-path return passes through,
    so it is where the frame receipt is stamped — ``frame_ephemeral``,
    ``frame_escape_active``, ``recency_mounted_ids``, ``recency_seats`` and
    the live-window fields.  Stamping it here rather than at each return means
    a new return site cannot silently ship a turn with no frame receipt.
    """
    info = dict(info or {})
    frame_info = getattr(repo.arena, "_eb1_frame_info", None)
    if callable(frame_info):
        info.update(frame_info())
    if not defer_memory:
        extracted = repo._extract_from_new_turns(
            before,
            context={
                "event": "chat",
                "user_text": user_text,
                "assistant_text": ans,
            },
        )
        if extracted:
            info["extraction"] = extracted
        repo.runtime._finish_turn_event(
            "chat", before, extraction=extracted, autosave=True)
    return info


def _probe_mount_snapshot(
    repo: GraftRepository,
    arena,
    *,
    live_idx: set,
    picks: list[int],
    planned: list[int],
    turn_idx: int | None,
) -> None:
    """Opt-in contam mount snapshot; never aborts the probe path."""
    tel = getattr(repo, "_paging_tel", None) or _paging_telemetry.get_telemetry()
    if not (tel.snapshot_enabled and turn_idx is not None):
        return
    try:
        snap_ids = sorted({int(i) for i in live_idx} | {int(i) for i in picks})
        tel.snapshot_nodes(
            arena,
            snap_ids,
            turn=int(turn_idx),
            label="probe_mount",
            extra={
                "live_ids": sorted(int(i) for i in live_idx),
                "mount_fitted": [int(i) for i in picks],
                "mount_plan": [int(i) for i in planned],
            },
        )
    except Exception as snap_err:
        if tel.enabled:
            tel.log(
                "mount_snapshot_error",
                -1,
                error=repr(snap_err),
                turn=int(turn_idx),
            )


def _route_observation(
    *,
    route_limit: int,
    excluded_live_ids,
    admission_profile: dict[str, Any] | None,
    trips: list[dict[str, Any]],
    serving_path: str,
) -> dict[str, Any]:
    """LSR-P2B observation hand-off (pure bookkeeping, no control flow).

    ``_probe_ladder_chat`` already knows the route limit, the live exclusions,
    the admission profile object, and every ladder trip it walked.  Phase 1 had
    to reconstruct all four from snapshot manifests; carrying them out through
    ``info`` under one private key is what closes that gap.  P2A owns the
    ladder/fit logic; this function only reads what the ladder produced.
    """
    return {
        "serving_path": str(serving_path),
        "route_limit": int(route_limit),
        "excluded_live_ids": sorted(int(value) for value in excluded_live_ids),
        "admission_profile": admission_profile,
        "trips": list(trips),
    }


def _probe_ladder_chat(
    repo: GraftRepository,
    user_text: str,
    *,
    topk: int,
    ngen: int,
    max_trips: int = 1,
    defer_memory: bool = False,
    turn_idx: int | None = None,
    lsr_fixes: bool | None = None,
    demand_ngh: bool | None = None,
    demand_early_abort: bool | None = None,
) -> tuple[str, dict[str, Any]]:
    """Fork-A probe path with production laws enforced (flag-on only).

    Reuses arena.route / _rare_tokens / _query_lex_tokens / _node_text_tokens /
    _attempt / _grounding_attribution. Does not reimplement scoring.

    ``lsr_fixes`` (default: the ``GRM_LSR_FIXES`` env switch, itself default
    ON) gates the LSR-P2A fit-honesty rulings and the LSR-P2C split/descent.
    Turning it OFF restores the pre-P2A packing for the G2 Arm-0
    reproduction arm; production never runs with it off.

    ``demand_ngh`` (GRM-SC1, default: the ``GRM_DEMAND_NGH`` env switch,
    itself DEFAULT OFF) enables the Stage C demand loop.  With it off this
    function's served outputs are byte-identical to P2C.
    """
    arena = repo.arena
    fixes_on = lsr_fixes_enabled(lsr_fixes)
    before = repo._snapshot_state()
    # GRM-EB1: this is a PRODUCTION SERVING PATH, so it opens its turn under
    # the same frame ``ArenaCache.step()`` does, through the same helper.
    #
    # Before EB1 only ``step()`` carried the ephemeral logic. Run against an
    # ephemeral arena, this path served a THIRD frame -- no live window
    # (nothing feeds one) and no recency mounts either (it never nominated
    # any) -- so the routing exclusion the persistent frame got for free from
    # its live window simply vanished. Measured on the G2 sup battery, that
    # cost three probes the persistent frame served correctly. The fix is to
    # share the frame, not to special-case the probe.
    rec = arena.eb1_begin_turn()
    live_idx = {g for g, _ in arena.live_segs if g is not None} | set(rec)
    want = max(int(topk), 1)
    admission_profile = None
    if getattr(arena, "decisive_admission", False):
        route_limit = max(want, (int(max_trips) + 1) * want)
        admission_profile = decisive_admission_profile(
            arena, user_text, exclude=live_idx, route_limit=route_limit)
        ranking = list(admission_profile["ranking"])
    else:
        # GRM_ADM_DECISIVE=0 retains the legacy bounded route call exactly.
        route_limit = max(want, (int(max_trips) + 1) * want)
        ranking = list(
            arena.route(user_text, exclude=live_idx, limit=route_limit) or [])
    # LSR-P2A Ruling 2: not-in-memory abstention, before any mount work.
    # Gated by the same fixes switch: the Arm-0 reproduction arm must serve
    # what the lived run served, and the lived run had no abstention rule.
    abstain = (
        identifier_serving_decision(arena, user_text, admission_profile,
                                    exclude=live_idx) if fixes_on else None)
    if abstain is not None and abstain.get("served_from"):
        # Prior art: core FIX-4 decision/reader (GRM contributors, 2026).
        # Share both with step; no second admission or recency policy here.
        ans, info = arena._serve_live_binding(
            user_text, abstain, rec, ngen=int(ngen), deposit=not defer_memory,
            defer_memory=defer_memory, stops=arena.stop_sequences)
        info.update(admission_info_fields(admission_profile))
        info.update(driver_probe_multimount=True, driver_probe_ladder=True,
                    driver_topk=int(want), point_lookup=True, precise_first=False,
                    mount_plan=[], mount_fitted=[], mount_dropped_for_width=[],
                    ranking_ids=[int(x) for x in ranking])
        info["_route_observation"] = _route_observation(
            route_limit=route_limit, excluded_live_ids=live_idx,
            admission_profile=admission_profile,
            trips=[{"ordinal": 0, "clean_room": False, "planned": [],
                    "mount_set": list(arena.cur_mounts),
                    "served_from": "recency_mount",
                    "served_from_node_ids": abstain["served_from_node_ids"]}],
            serving_path="grm_e2e_session._probe_ladder_chat:recency_mount")
        return ans, _probe_finish_deposit(
            repo, before, user_text, ans, info, defer_memory=defer_memory)
    if abstain is not None:
        ans = str(abstain["abstain_text"])
        info: dict[str, Any] = {
            "trip": 0,
            "driver_probe_multimount": True,
            "driver_probe_ladder": True,
            "driver_topk": int(want),
            "point_lookup": True,
            "precise_first": False,
            "mount_plan": [],
            "mount_fitted": [],
            "mount_dropped_for_width": [],
            "ranking_ids": [int(x) for x in ranking],
            "abstained": True,
            "abstain_reason": str(abstain["abstain_reason"]),
            "abstain_identifier_tokens": [
                str(t) for t in abstain["abstain_identifier_tokens"]],
        }
        info.update(admission_info_fields(admission_profile))
        if not defer_memory:
            # Ruling 2.2: the user turn is a lived turn and still deposits;
            # the abstention output is pinned kind="recall" so it can never
            # later be routed as a stored fact.
            gidx = arena.deposit(arena._format_step_turn(user_text, ans))
            arena.grafts[gidx]["kind"] = "recall"
            arena._bump_cuda_gqa_epoch()
            info["abstain_deposited_graft"] = int(gidx)
        # LSR-P2B: the abstention short-circuit returns BEFORE any mount work,
        # so it walks no rungs — an empty trip list is the honest record, and
        # the receipt still carries the route, the admission profile, and the
        # abstain_* fields via the generic pass-through.
        info["_route_observation"] = _route_observation(
            route_limit=route_limit,
            excluded_live_ids=live_idx,
            admission_profile=admission_profile,
            trips=[],
            serving_path="grm_e2e_session._probe_ladder_chat:abstention",
        )
        info = _probe_finish_deposit(
            repo, before, user_text, ans, info, defer_memory=defer_memory)
        return ans, info

    id_tokens = _probe_identifier_tokens(arena, user_text)
    point_lookup = bool(id_tokens)
    precise = _probe_rank1_covers_identifiers(arena, ranking, id_tokens)
    baseline_attempts = build_probe_ladder_attempts(
        ranking=ranking,
        topk=want,
        precise=precise,
        point_lookup=point_lookup,
        max_trips=int(max_trips),
    )
    # LSR-P2C Arm-0 switch: with the fixes OFF, the FIT stage reverts to the
    # pre-P2A rank-order packing (_budget_fit_mounts) and no split/descent
    # runs. Routing and the A-DEC plan are untouched either way, because Arm 0
    # must reproduce the LIVED ranking and the lived plan — only the fit-stage
    # rulings are what P2A/P2C changed.
    fit_profile = admission_profile if fixes_on else None
    rank_plan = (
        [int(value) for value in admission_profile["rank_plan"]]
        if fit_profile is not None else []
    )
    if admission_profile is None:
        attempts = baseline_attempts
    else:
        first = (
            [int(value) for value in admission_profile["rank_plan"]],
            bool(point_lookup),
        )
        attempts = [first]
        for candidate in [*baseline_attempts[1:], *baseline_attempts[:1]]:
            normalized = ([int(value) for value in candidate[0]], bool(candidate[1]))
            if normalized not in attempts:
                attempts.append(normalized)
        attempts = attempts[:max(1, int(max_trips) + 1)]

    # LSR-P2A Ruling 1.2: SHUTTLE. Plan members that cannot co-seat with the
    # rest of the plan get their OWN trip, in plan order, additive to
    # max_trips up to the registered hard cap len(rank_plan). This is
    # triggered by the FIT DROP, not by a grounding failure — the defect
    # anatomy (adjudication 31e5c894) is precisely that a wrong-but-grounded
    # trip 0 ended the turn while the planned node was still unseated.
    shuttle_trips: list[list[int]] = []
    plan_head_receipt: dict[str, Any] = {}

    # LSR-P2C Part 2: FIT-TIME DESCENT at the driver's fit site — the site
    # the lived probes actually went through. A plan member whose own ntok
    # exceeds the budget is UNSEATABLE: no fit and no plan-shuttle can seat
    # it. Split it and make its identifier-bearing children the plan head,
    # then shuttle across the chunks in document order if they do not co-fit.
    split_parent: int | None = None
    split_children: list[int] = []
    split_ephemeral: bool | None = None
    descended_head: list[int] = []
    chunk_trips: list[list[int]] = []
    chunk_owed: list[int] = []
    split_head_binds = False
    splitter = getattr(arena, "_split_unseatable", None)
    if rank_plan and callable(splitter):
        turn_budget = mountable_budget(arena)
        unseatable_now = [
            int(v) for v in rank_plan
            if int(arena.grafts[int(v)]["ntok"]) > turn_budget
        ]
        # One split per turn: the plan head is the member the turn is about,
        # and splitting every unseatable member would multiply chunk trips
        # past the registered cap.
        for member in unseatable_now[:1]:
            children, ephemeral = splitter(member, turn_budget)
            if not children:
                # Nothing could be split. P2A's explicit degrade stands and
                # the receipt still says so — honest, not silent.
                continue
            split_parent = int(member)
            split_children = [int(v) for v in children]
            split_ephemeral = bool(ephemeral)
            # The IDENTIFIER-BEARING child set. See
            # ArenaCache._identifier_bearing_children: the rare (code/number)
            # channel is EMPTY for the whole ADMISSION-PRUNE probe class, so
            # the frozen ADM1 lexical binding predicate is what discriminates
            # the chunks. Duck-typed so the CPU stub harness still works.
            qrare = arena._rare_tokens(user_text)
            picker = getattr(arena, "_identifier_bearing_children", None)
            if callable(picker):
                descended_head, head_binds = picker(
                    member, user_text, qrare, split_children,
                    with_binding_flag=True)
                descended_head = [int(v) for v in descended_head]
            else:
                descended_head = [
                    int(v) for v in arena._descent_source_children(
                        member, qrare=qrare)
                ] or list(split_children)
                head_binds = True
            # SUBSTITUTE IN PLACE, never re-rank. The children stand exactly
            # where their parent stood in the plan; promoting them to the head
            # would demote a SEATABLE, higher-ranked plan member behind the
            # chunks of an unseatable lower-ranked one.
            was_head = int(rank_plan[0]) == int(member) and bool(head_binds)
            # A split whose chunks bind NOTHING does not enter the plan at all
            # when the identifier binds some OTHER plan member: a plan member
            # is owed a SHUTTLE TRIP by Ruling 1.2, so keeping non-binding
            # chunks in the plan hands the turn to a competitor's filler when
            # the real answer node fails grounding (measured: sup_solace_fresh
            # produced fit_shuttle_trips [[0], [3], [4]] and trip 4 served the
            # competitor's Sable-0-Copper).
            other_binds = bool(
                {int(v) for v in admission_profile["identified_candidates"]}
                - {int(member)}
            ) if admission_profile is not None else False
            drop_chunks = bool(not head_binds and other_binds)
            substituted: list[int] = []
            trailing: list[int] = []
            seen_plan: set[int] = set()
            for value in rank_plan:
                if int(value) == int(member):
                    if drop_chunks:
                        continue
                    incoming = descended_head
                    sink = substituted if head_binds else trailing
                else:
                    incoming, sink = [int(value)], substituted
                for entry in incoming:
                    # DE-DUPLICATE: a previous turn's persisted child can
                    # already be a plan member in its own right, and
                    # substituting its parent would list it twice (measured:
                    # fit_planned = [4, 4] on sup_reserve_tundra_ledger).
                    if int(entry) not in seen_plan:
                        sink.append(int(entry))
                        seen_plan.add(int(entry))
            rank_plan = [*substituted, *trailing]
            head_fit = plan_priority_fit(
                plan=rank_plan,
                candidates=rank_plan,
                ntok={
                    int(i): int(arena.grafts[int(i)]["ntok"])
                    for i in rank_plan
                },
                budget=int(turn_budget),
            )
            head_picks = [int(v) for v in head_fit["fit_seated"]]
            if head_picks and was_head:
                head_rung = (sorted(head_picks), bool(point_lookup))
                attempts = [
                    head_rung,
                    *[a for a in attempts
                      if (sorted(a[0]), bool(a[1])) != head_rung],
                ]
            # Chunks the head rung could not co-seat are owed their OWN trip,
            # in DOCUMENT ORDER, additive and capped at chunk_trip_cap.
            # A split whose chunks bind NOTHING gets no trips: giving them
            # their own rungs lets a competitor's filler ground the turn while
            # the planned answer node goes unread (measured: sup_solace_fresh
            # served the competitor's Sable-0-Copper from chunk trip 3).
            split_head_binds = bool(head_binds)
            chunk_owed = (
                [int(v) for v in descended_head
                 if int(v) not in set(head_picks)
                 ][:chunk_trip_cap(descended_head)]
                if head_binds else []
            )
            for chunk in chunk_owed:
                rung = ([int(chunk)], bool(point_lookup))
                if rung not in attempts:
                    attempts.append(rung)
                    chunk_trips.append([int(chunk)])

    if rank_plan:
        resolved_plan = [
            int(v) for v in arena._resolve_revision_mounts(list(rank_plan))]
        head_receipt = plan_priority_fit(
            plan=rank_plan,
            candidates=resolved_plan,
            ntok={
                int(i): int(arena.grafts[int(i)]["ntok"])
                for i in resolved_plan
            },
            budget=int(arena.width),
        )
        plan_head_receipt = head_receipt
        seated_anywhere = {
            int(v)
            for planned_ids, _clean in attempts
            for v in _budget_fit_mounts(arena, planned_ids)
        }
        owed = [
            int(v) for v in head_receipt["fit_shuttle_pending"]
            if int(v) not in seated_anywhere
        ]
        for member in owed[:shuttle_trip_cap(rank_plan)]:
            rung = ([int(member)], bool(point_lookup))
            if rung not in attempts:
                attempts.append(rung)
                shuttle_trips.append([int(member)])

    for layer in arena.m.layers:
        layer.self_attn.live_shift = arena.live_shift
    stops = arena.stop_sequences or ()

    # Snapshot pre-attempt state for shuttling rollback (mirrors Arena.step).
    snap = (
        arena.caches,
        arena.pos,
        list(arena.live_segs),
        arena.cur_mounts,
        arena.cur_mount_n,
        len(arena.grafts),
    )
    best: tuple[str, dict[str, Any], tuple, list[int], list[int]] | None = None
    last_planned: list[int] = []
    last_picks: list[int] = []
    fit_receipts: dict[tuple[int, ...], dict[str, Any]] = {}

    def _fit_for(planned_ids: list[int]) -> dict[str, Any]:
        """Plan-priority fit for one rung (Ruling 1.1).

        Plan members come first in PLAN ORDER, then filler in rank order —
        `_budget_fit_mounts`' original packing law, now confined to filler.
        The legacy (`admission_profile is None`) path keeps the original
        rank-order packing byte-for-byte.
        """
        ids = [int(v) for v in planned_ids]
        if fit_profile is None:
            picks_ = _budget_fit_mounts(arena, ids)
            return {
                "fit_planned": ids,
                "fit_seated": list(picks_),
                "fit_dropped_planned": [],
                "fit_dropped_filler": [v for v in ids if v not in set(picks_)],
                "fit_unseatable": [],
                "fit_shuttle_pending": [],
            }
        receipt = plan_priority_fit(
            plan=rank_plan,
            candidates=ids,
            ntok={int(i): int(arena.grafts[int(i)]["ntok"]) for i in ids},
            # GRM-EB1: the recency mounts are co-seated at mount-set assembly
            # below, so their seats must come off this rung's budget exactly
            # as ``step()::fit_detail`` takes them off its own.  The charge is
            # zero for identifier queries (the point-lookup rule), and the
            # helper records which branch ran onto the turn's receipt.
            budget=arena.eb1_charge_recency(rec, id_tokens),
        )
        fit_receipts[tuple(sorted(receipt["fit_seated"]))] = receipt
        return receipt

    def _attach_fit(info: dict[str, Any], seated: list[int]) -> dict[str, Any]:
        """NEVER SILENT (Ruling 1.3): fit fields on every served turn.

        The receipt describes the PLAN's fate across the whole turn, not
        merely the served rung's own packing: a rung the ladder chose can
        omit the plan entirely (the baseline merge in this driver puts
        ``ranking[topk:2*topk]`` ahead of the widened top-k rung), and a
        receipt that reported only that rung would be silent about a plan
        member no rung ever seated — the exact silence Ruling 1.3 forbids.
        """
        if fit_profile is None:
            return info
        seated_set = {int(v) for v in seated}
        rung = fit_receipts.get(tuple(sorted(seated_set))) or {}
        # UNSEATABLE is a property of the plan member and the arena width,
        # not of whichever rung happened to serve.
        unseatable = [
            int(v) for v in plan_head_receipt.get("fit_unseatable", ())]
        seated_anywhere_now = seated_set | {
            int(v) for trip in shuttle_trips for v in trip} | {
            int(v) for trip in chunk_trips for v in trip}
        dropped_planned = [
            int(v) for v in rank_plan
            if int(v) in set(unseatable) and int(v) not in seated_anywhere_now
        ]
        receipt = {
            "fit_planned": list(rank_plan),
            "fit_seated": sorted(seated_set),
            "fit_dropped_planned": dropped_planned,
            "fit_dropped_filler": [
                int(v) for v in rung.get("fit_dropped_filler", ())],
            "fit_unseatable": unseatable,
        }
        # LSR-P2C: after a split-and-descend the turn served WITH the plan
        # head whenever the grounded trip seated any chunk of it — the answer
        # is composed from the grounded trips, so served_without_plan_head is
        # true ONLY when every chunk trip failed grounding.
        # The chunk set counts as "the plan head" ONLY when the split member
        # WAS the head and its chunks BOUND the identifier. A demoted,
        # non-binding chunk set is filler that happened to ground; calling
        # that "served with the plan head" is the unlabeled substitution
        # Ruling 1.4 forbids.
        head_served = bool(rank_plan) and int(rank_plan[0]) in seated_set
        if (split_parent is not None and not head_served
                and split_head_binds):
            head_served = bool(seated_set & set(descended_head))
        info.update(fit_info_fields(
            receipt,
            shuttle=bool(shuttle_trips),
            shuttle_trips=shuttle_trips,
            # Ruling 1.4: explicit degrade, never unlabeled substitution.
            served_without_plan_head=bool(rank_plan and not head_served),
        ))
        # LSR-P2C receipt fields, all fit_-prefixed so P2B persists them via
        # ROUTE_RECEIPT_INFO_PREFIXES with no edit to core/grm_three_pass.py.
        info.update(split_info_fields(
            split_parent=split_parent,
            split_children=split_children,
            split_ephemeral=split_ephemeral,
            descended_head=descended_head,
            chunk_trips=chunk_trips,
        ))
        return info

    # -----------------------------------------------------------------
    # GRM-SC1 Stage C: the DEMAND LOOP at the DRIVER site.
    #
    # The lived probes go through THIS function, not ``ArenaCache.step`` —
    # the same reason P2A/P2C had to land their rulings at both fit sites.
    # The mechanism is identical and comes from ``core.grm_demand``; only
    # the local variable names differ.
    #
    # FLAG OFF => ``_serve`` is ``arena._attempt`` by identity and no other
    # line below executes, so served outputs are byte-identical to P2C.
    demand_on = grm_demand.demand_enabled(demand_ngh)
    demand_state: dict[str, Any] = {"rows": [], "info": {}}
    if demand_on:
        support = grm_demand.arena_support(arena)
        if not support.get("demand_supported"):
            # LOUD refusal: an unsupported arena must never be reported as
            # "no demand detected".
            demand_on = False
            demand_state["info"] = grm_demand.unsupported_info(
                str(support.get("demand_unsupported_reason")))
    demand_threshold = (
        grm_demand.registered_threshold() if demand_on else None)

    # GRM-SC2: the probe path's mirror of the arena's early-abort seam.
    # Consulted ONLY when demand is on; with demand off nothing below runs and
    # this path is byte-identical to SC1.
    early_abort_on = (
        grm_demand.early_abort_enabled(demand_early_abort)
        if demand_on else False)
    demand_state["early_abort"] = bool(early_abort_on)

    def _serve(*args, **kwargs):
        """One generation attempt, observed when the demand flag is on.

        GRM-SC2: when early abort is on, an attempt whose D-NGH mass falls
        below the line stops generating at that token and comes back as a
        SUSPENSION handle rather than a finished answer.  The handle rides on
        ``info`` so ``_demand_trip`` can consume it, and it never reaches a
        receipt (it is popped there).
        """
        started = time.perf_counter()
        if not demand_on:
            return arena._attempt(*args, **kwargs)
        abort_now = bool(early_abort_on) and not demand_state.get(
            "suppress_early_abort")
        with grm_demand.DemandObserver(
            arena, int(ngen), float(demand_threshold),
            early_abort=abort_now,
        ) as observer:
            out = arena._attempt(*args, **kwargs)
        demand_state["rows"] = observer.finish()
        demand_state["wall_ms"] = (time.perf_counter() - started) * 1000.0
        if isinstance(out, dict) and out.get("grm_sc2_suspended_attempt"):
            rows = list(demand_state["rows"])
            index = int(out["abort_token_index"])
            # Rebuild the emitted ids from the observer rows, never from a
            # decoded string: decode->encode is not a BPE round trip, and the
            # resume has to continue the EXACT token sequence.
            out["out"] = [
                int(row["prediction_token_id"]) for row in rows[:index + 1]]
            out["tokens_generated_before_abort"] = len(out["out"])
            return "", {"_grm_sc2_suspended": out}
        return out

    def _demand_trip(ans, info, picks, trip, rows=None):
        """The ONE registered demand trip, at the point of serving.

        Returns ``(answer, info, picks)``.  Same four steps as
        ``ArenaCache.step``: decide on the attempt about to be served; on a
        fire roll back to ``snap`` and re-route on question + the model's own
        partial output (excluding what is already mounted or live); serve the
        trip's answer iff it grounds, else restore and serve the original;
        record a refire without acting on it (cap 1, registered).
        """
        if not demand_on:
            if demand_state["info"]:
                info.update(demand_state["info"])
            return ans, info, picks
        observed = list(demand_state["rows"] if rows is None else rows)
        # GRM-SC2: the suspended attempt this trip stands in for, if any.
        suspended = (info or {}).pop("_grm_sc2_suspended", None)
        attempt_wall_ms = demand_state.get("wall_ms")
        decision = grm_demand.decide(observed, float(demand_threshold))
        if suspended is not None and not decision["demand_fired"]:
            raise grm_demand.DemandError(
                "GRM-SC2: an attempt was suspended by the early abort but the "
                "decision over its rows says it never fired; the abort and "
                "the carried decision rule have diverged")
        if not decision["demand_fired"]:
            info.update(grm_demand.demand_info_fields(
                supported=True, decision=decision, served="original",
                threshold=float(demand_threshold)))
            return ans, info, picks

        original_state = (
            arena.caches, arena.pos, list(arena.live_segs),
            arena.cur_mounts, arena.cur_mount_n, list(arena.grafts),
        )
        prefix = grm_demand.demand_prefix_text(
            arena, observed, int(decision["demand_token_index"]))
        query = grm_demand.demand_query_text(user_text, prefix)
        (arena.caches, arena.pos, arena.live_segs, arena.cur_mounts,
         arena.cur_mount_n) = (
            snap[0], snap[1], list(snap[2]), snap[3], snap[4])
        del arena.grafts[snap[5]:]
        arena._bump_cuda_gqa_epoch()
        demand_exclude = set(live_idx) | {int(v) for v in picks}
        if getattr(arena, "decisive_admission", False):
            demand_profile = decisive_admission_profile(
                arena, query, exclude=demand_exclude, route_limit=route_limit)
            demand_ranking = [int(v) for v in demand_profile["ranking"]]
            demand_plan = [int(v) for v in demand_profile["rank_plan"]]
        else:
            demand_ranking = [
                int(v) for v in (arena.route(
                    query, exclude=demand_exclude, limit=route_limit) or [])]
            demand_plan = demand_ranking[:want]
        demand_picks = sorted(
            int(v) for v in _budget_fit_mounts(arena, demand_plan))
        base_fields = dict(
            supported=True, decision=decision,
            threshold=float(demand_threshold),
            query_text_sha256=grm_demand.demand_query_sha256(query),
            prefix_token_count=int(decision["demand_token_index"]),
            ranking=demand_ranking,
        )

        def _restore_original():
            (arena.caches, arena.pos, arena.live_segs, arena.cur_mounts,
             arena.cur_mount_n) = original_state[:5]
            arena.grafts[:] = original_state[5]
            arena._bump_cuda_gqa_epoch()

        def _resume_if_suspended(ans, info):
            """GRM-SC2: finish the suspended attempt so the turn has an answer.

            The arena state has already been restored to the aborted attempt's
            state by ``_restore_original`` -- the same verbatim restore SC1
            performs when a trip fails to ground -- so the cache the resume
            continues from is the one the abort left behind. That is what
            makes the finished text the ORIGINAL text rather than a
            regeneration of it.
            """
            if suspended is None:
                return ans, info, None
            started = time.perf_counter()
            r_ans, r_info = arena._attempt(
                user_text, list(suspended["picks"]), int(ngen),
                False if defer_memory else True, stops,
                defer_memory=defer_memory, suspended=suspended)
            resume_ms = (time.perf_counter() - started) * 1000.0
            merged = dict(r_info or {})
            for key, value in (info or {}).items():
                if key.startswith("_grm_sc2_"):
                    continue
                merged.setdefault(key, value)
            return r_ans, merged, resume_ms

        def _abort_fields(**extra):
            """The GRM-SC2 receipt block, or nothing when nothing aborted."""
            if suspended is None:
                return {}
            return grm_demand.demand_info_fields(
                supported=True, decision=decision,
                threshold=float(demand_threshold),
                early_abort=True,
                abort_token_index=int(suspended["abort_token_index"]),
                tokens_generated_before_abort=int(
                    suspended["tokens_generated_before_abort"]),
                wall_ms_attempt=attempt_wall_ms,
                **extra)

        if not demand_picks:
            # Nothing new to fetch. Honest: the receipt says the trip found
            # nothing rather than pretending the detector never fired.
            _restore_original()
            ans, info, resume_ms = _resume_if_suspended(ans, info)
            info.update(grm_demand.demand_info_fields(
                **base_fields, served="original", fetched=[],
                trip_taken=False))
            info.update(_abort_fields(
                served="original", trip_taken=False, tokens_saved=0,
                resumed_original=True, wall_ms_resume=resume_ms))
            return ans, info, picks

        # The demand trip runs UNDER the observer as well, but ONLY so a
        # second fire can be RECORDED. Cap 1 is registered: `demand_refired`
        # is a receipt field, never a branch.
        # GRM-SC2: the trip itself is NEVER early-aborted. Cap 1 is
        # registered -- a second fire is RECORDED and not acted on -- so
        # aborting the trip would throw away the only answer the turn has
        # left to serve and leave it with nothing to put out.
        trip_started = time.perf_counter()
        demand_state["suppress_early_abort"] = True
        try:
            if defer_memory:
                d_ans, d_info = _serve(
                    user_text, demand_picks, int(ngen), False, stops,
                    defer_memory=True)
            else:
                d_ans, d_info = _serve(
                    user_text, demand_picks, int(ngen), True, stops)
        finally:
            demand_state["suppress_early_abort"] = False
        trip_wall_ms = (time.perf_counter() - trip_started) * 1000.0
        refire = grm_demand.decide(
            list(demand_state["rows"]), float(demand_threshold))
        # SC1.1: the demand trip's own grounding receipt. Both branches below
        # apply ``fields``, so the receipt travels whether the trip serves or
        # is rejected.
        grounding_fields: dict[str, Any] = {}
        d_grounded, _dc = arena._grounding_attribution(
            d_ans, demand_picks, user_text)
        _stamp_grounding_receipt(
            arena, d_ans, demand_picks, user_text, grounding_fields)
        fields = grm_demand.demand_info_fields(
            **base_fields,
            served="demand_trip" if d_grounded else "original",
            fetched=[int(v) for v in demand_picks],
            refired=bool(refire["demand_fired"]),
            refire_token_index=refire["demand_token_index"],
            trip_taken=True,
            trip_grounded=bool(d_grounded),
        )
        fields.update(grounding_fields)
        if d_grounded:
            # The trip serves. The suspended attempt is ABANDONED, never
            # resumed: every token after the fire index is one this turn never
            # had to generate, and that is the whole saving.
            d_info = dict(d_info or {})
            d_info["trip"] = int(trip)
            d_info["demand_source_trip"] = int(trip)
            d_info.update(fields)
            d_info.update(_abort_fields(
                served="demand_trip", trip_taken=True, trip_grounded=True,
                resumed_original=False, wall_ms_trip=trip_wall_ms))
            if suspended is not None:
                d_info.update(grounding_fields)
            return d_ans, d_info, list(demand_picks)
        _restore_original()
        # GRM-SC2: under early abort the "original answer" does not exist yet
        # -- the attempt stopped at the fire token. Resume and finish it, so
        # what goes out is exactly what SC1 put out on this probe.
        ans, info, resume_ms = _resume_if_suspended(ans, info)
        info.update(fields)
        info.update(_abort_fields(
            served="original", trip_taken=True, trip_grounded=False,
            tokens_saved=0, resumed_original=True,
            wall_ms_trip=trip_wall_ms, wall_ms_resume=resume_ms))
        if suspended is not None:
            info.update(grounding_fields)
        return ans, info, picks

    # LSR-P2B: one row per ladder trip actually walked, recorded as it happens.
    trip_rows: list[dict[str, Any]] = []

    for trip, (planned, clean) in enumerate(attempts):
        last_planned = list(planned)
        if trip:
            (arena.caches, arena.pos, arena.live_segs, arena.cur_mounts,
             arena.cur_mount_n) = (
                snap[0], snap[1], list(snap[2]), snap[3], snap[4])
            had_appended = len(arena.grafts) > snap[5]
            del arena.grafts[snap[5]:]
            if not defer_memory or had_appended:
                arena._bump_cuda_gqa_epoch()
        if clean:
            # RECENCY LAW: point lookups exclude live/recency seats.
            arena.caches, arena.pos, arena.live_segs = None, 0, []
            arena.cur_mounts, arena.cur_mount_n = [], 0

        receipt = _fit_for(list(planned))
        picks = sorted(int(v) for v in receipt["fit_seated"])
        # GRM-EB1: co-seat the recency mounts, under EXACTLY ``step()``'s
        # ``use_rec`` rule (graft_arena.step: ``rec and not qrare and not
        # clean and picks != precise``).  Recency joins topical/anaphora
        # attempts only: an identifier lookup is a point read, and a
        # clean-room rung is the RECENCY LAW rung that exists precisely to
        # exclude these seats.  Restating the rule here rather than inventing
        # a probe-specific one is what keeps the two serving paths one frame.
        use_rec = bool(rec) and not id_tokens and not clean and (
            picks != (precise or []))
        if use_rec:
            picks = sorted(
                int(v) for v in arena._resolve_revision_mounts(
                    sorted(set(rec) | set(picks))))
        last_picks = list(picks)
        if not picks and planned:
            # Nothing fits — treat as empty attempt. A later shuttle rung
            # may still seat an owed plan member (Ruling 1.2).
            # LSR-P2B: the rung is still a rung the ladder walked, so it gets
            # its own row; "grounded: False" describes THIS rung, not the
            # turn — a later shuttle rung can still ground.
            trip_rows.append({
                "ordinal": int(trip),
                "clean_room": bool(clean),
                "planned": [int(v) for v in planned],
                "mount_set": [],
                "grounded": False,
                "shuttle": [int(v) for v in planned] in shuttle_trips,
            })
            continue
        if defer_memory:
            ans, info = _serve(
                user_text, picks, int(ngen), False, stops, defer_memory=True)
        else:
            ans, info = _serve(
                user_text, picks, int(ngen), True, stops)
        info = dict(info or {})
        info["trip"] = int(trip)
        if clean:
            info["clean_room"] = True
        # GRM-SC2: a SUSPENDED attempt has no text to ground. Grounding the
        # empty string would fail, and the rung would fall through to the
        # ungrounded-keep-first path carrying an answer that does not exist.
        # Route it straight into the demand trip -- which is where an
        # un-aborted fired attempt would have arrived anyway, just without
        # generating the rest of the wrong answer on the way.
        suspended_here = info.get("_grm_sc2_suspended") is not None
        if suspended_here:
            grounded, _contributors = True, ()
        else:
            grounded, _contributors = arena._grounding_attribution(
                ans, picks, user_text)
            _stamp_grounding_receipt(arena, ans, picks, user_text, info)
        trip_rows.append({
            "ordinal": int(trip),
            "clean_room": bool(clean),
            "planned": [int(v) for v in planned],
            "mount_set": [int(v) for v in picks],
            # GRM-SC2: a suspended rung has NO grounding verdict -- there is
            # no text to attribute. Recording ``True`` here would be a claim
            # the rung never made; ``None`` says what actually happened, and
            # ``early_aborted`` says why.
            "grounded": None if suspended_here else bool(grounded),
            "early_aborted": bool(suspended_here),
            # SC1.1: per-rung glyph receipt, so a rung that grounded ONLY
            # because of the projection is visible in the ladder trace.
            "grounding_normalized": bool(info.get("grounding_normalized")),
            "grounding_glyph_rescued": bool(
                info.get("grounding_glyph_rescued")),
            # LSR-P2A Ruling 1.2 rungs are additive to max_trips; naming them
            # keeps a reader from mistaking a shuttle rung for a normal trip.
            "shuttle": [int(v) for v in planned] in shuttle_trips,
        })
        state = (
            arena.caches, arena.pos, list(arena.live_segs),
            arena.cur_mounts, arena.cur_mount_n, list(arena.grafts),
        )
        if grounded:
            # SC1: the turn is about to serve. Take the one registered demand
            # trip FIRST, so the mount snapshot and every receipt field below
            # describe the mounts the SERVED answer was actually read from.
            ans, info, picks = _demand_trip(ans, info, list(picks), int(trip))
            _probe_mount_snapshot(
                repo, arena, live_idx=live_idx, picks=picks,
                planned=planned, turn_idx=turn_idx)
            info["driver_probe_multimount"] = True
            info["driver_probe_ladder"] = True
            info["driver_topk"] = int(want)
            info["point_lookup"] = bool(point_lookup)
            info["precise_first"] = bool(
                precise is not None and list(planned) == list(precise))
            info["mount_plan"] = list(planned)
            info["mount_fitted"] = list(picks)
            info["mount_dropped_for_width"] = [
                i for i in planned if i not in set(picks)]
            info["ranking_ids"] = [int(x) for x in ranking]
            if admission_profile is not None:
                info.update(admission_info_fields(admission_profile))
            # P2A's fit receipt lands in ``info`` FIRST so P2B's generic
            # ``fit_*`` pass-through picks it up on this same turn.
            info = _attach_fit(info, list(picks))
            info["_route_observation"] = _route_observation(
                route_limit=route_limit,
                excluded_live_ids=live_idx,
                admission_profile=admission_profile,
                trips=trip_rows,
                serving_path="grm_e2e_session._probe_ladder_chat",
            )
            info = _probe_finish_deposit(
                repo, before, user_text, ans, info, defer_memory=defer_memory)
            return ans, info
        if best is None:
            best = (ans, info, state, list(picks), list(planned),
                    list(demand_state["rows"]))

    # Nothing grounded — keep FIRST attempt (arena.step convention).
    if best is None:
        ans, info = "", {
            "trip": 0,
            "no_mount_fit": True,
            "driver_probe_multimount": True,
            "driver_probe_ladder": True,
            "driver_topk": int(want),
            "point_lookup": bool(point_lookup),
            "precise_first": False,
            "mount_plan": list(last_planned),
            "mount_fitted": list(last_picks),
            "mount_dropped_for_width": [
                i for i in last_planned if i not in set(last_picks)],
            "ranking_ids": [int(x) for x in ranking],
        }
        if defer_memory:
            # Empty attempt still needs a deferred shell for pass-3.
            ans, info_att = _serve(
                user_text, [], int(ngen), False, stops, defer_memory=True)
            info = dict(info_att or {})
            info.update({
                "trip": 0,
                "no_mount_fit": True,
                "driver_probe_multimount": True,
                "driver_probe_ladder": True,
                "driver_topk": int(want),
                "point_lookup": bool(point_lookup),
                "precise_first": False,
                "mount_plan": list(last_planned),
                "mount_fitted": [],
                "mount_dropped_for_width": list(last_planned),
                "ranking_ids": [int(x) for x in ranking],
            })
        else:
            ans, info_att = _serve(
                user_text, [], int(ngen), True, stops)
            info = dict(info_att or {})
            info.update({
                "trip": 0,
                "no_mount_fit": True,
                "driver_probe_multimount": True,
                "driver_probe_ladder": True,
                "driver_topk": int(want),
                "point_lookup": bool(point_lookup),
                "precise_first": False,
                "mount_plan": list(last_planned),
                "mount_fitted": [],
                "mount_dropped_for_width": list(last_planned),
                "ranking_ids": [int(x) for x in ranking],
            })
        # SC1: a no-mount turn is the purest demand case there is — the
        # detector sees zero mounted mass by construction. One trip, and
        # ``mount_fitted`` names whatever the served answer actually read.
        ans, info, demand_picks = _demand_trip(ans, info, [], 0)
        if demand_picks:
            info["mount_fitted"] = list(demand_picks)
        if admission_profile is not None:
            info.update(admission_info_fields(admission_profile))
        info = _attach_fit(info, list(demand_picks))
        info["_route_observation"] = _route_observation(
            route_limit=route_limit,
            excluded_live_ids=live_idx,
            admission_profile=admission_profile,
            trips=trip_rows,
            serving_path="grm_e2e_session._probe_ladder_chat",
        )
        info = _probe_finish_deposit(
            repo, before, user_text, ans, info, defer_memory=defer_memory)
        return ans, info

    ans, info, st, picks, planned, best_rows = best
    (arena.caches, arena.pos, arena.live_segs,
     arena.cur_mounts, arena.cur_mount_n) = st[0], st[1], st[2], st[3], st[4]
    arena.grafts[:] = st[5]
    if not defer_memory:
        arena._bump_cuda_gqa_epoch()
    info = dict(info or {})
    # SC1: the ungrounded-keep-first answer is still an answer about to be
    # SERVED, so the demand loop applies. Its detector rows are the ones
    # captured on THAT attempt (``best_rows``), not whichever later rung ran
    # last.
    ans, info, picks = _demand_trip(ans, info, list(picks), 0, rows=best_rows)
    info["driver_probe_multimount"] = True
    info["driver_probe_ladder"] = True
    info["driver_topk"] = int(want)
    info["point_lookup"] = bool(point_lookup)
    info["precise_first"] = bool(
        precise is not None and list(planned) == list(precise))
    info["mount_plan"] = list(planned)
    info["mount_fitted"] = list(picks)
    info["mount_dropped_for_width"] = [
        i for i in planned if i not in set(picks)]
    info["ranking_ids"] = [int(x) for x in ranking]
    info["ungrounded_kept_first"] = True
    if admission_profile is not None:
        info.update(admission_info_fields(admission_profile))
    info = _attach_fit(info, list(picks))
    info["_route_observation"] = _route_observation(
        route_limit=route_limit,
        excluded_live_ids=live_idx,
        admission_profile=admission_profile,
        trips=trip_rows,
        serving_path="grm_e2e_session._probe_ladder_chat",
    )
    _probe_mount_snapshot(
        repo, arena, live_idx=live_idx, picks=picks,
        planned=planned, turn_idx=turn_idx)
    info = _probe_finish_deposit(
        repo, before, user_text, ans, info, defer_memory=defer_memory)
    return ans, info


def probe_multimount_chat(
    repo: GraftRepository,
    user_text: str,
    *,
    topk: int,
    ngen: int,
    defer_memory: bool = False,
    turn_idx: int | None = None,
    probe_ladder: bool = True,
    max_trips: int = 1,
) -> tuple[str, dict[str, Any]]:
    """Probe path: mount route top-k via arena multi-mount, not argmax-only.

    Arena.step's precise-first policy collapses identifier probes to rank-1
    alone when rank-1 covers all probe rare tokens. Fork A wants the
    production multi-mount slice (top-k) seated so a correct memory at
    ranks 2-3 is present even when rank-1 is a length-biased distractor.
    Diagnostics still record full ranking / source_rank separately.

    When ``probe_ladder`` is True (default under GRM3P-LADDER-ON; CLI
    ``--probe-ladder`` / env truthy), enforce the registered production laws
    on this path: point-lookup clean context, precise-first, grounding + one
    retry. Escape ``--no-probe-ladder`` / ``GRM_PROBE_LADDER=0`` restores the
    legacy multimount path exactly.
    """
    if probe_ladder:
        return _probe_ladder_chat(
            repo,
            user_text,
            topk=topk,
            ngen=ngen,
            max_trips=max_trips,
            defer_memory=defer_memory,
            turn_idx=turn_idx,
        )

    arena = repo.arena
    before = repo._snapshot_state()
    live_idx = {g for g, _ in arena.live_segs if g is not None}
    want = max(int(topk), 1)
    admission_profile = None
    if getattr(arena, "decisive_admission", False):
        admission_profile = decisive_admission_profile(
            arena, user_text, exclude=live_idx, route_limit=want)
        ranking = list(admission_profile["ranking"])
        planned = [int(value) for value in admission_profile["rank_plan"]]
    else:
        ranking = list(
            arena.route(user_text, exclude=live_idx, limit=want) or [])
        planned = [int(x) for x in ranking[:want]]
    fitted = _budget_fit_mounts(arena, planned)
    # Seat in sorted order (matches arena._attempt / swap convention).
    picks = sorted(fitted)
    for layer in arena.m.layers:
        layer.self_attn.live_shift = arena.live_shift
    stops = arena.stop_sequences or ()
    if defer_memory:
        ans, info = arena._attempt(
            user_text, picks, int(ngen), False, stops, defer_memory=True)
    else:
        ans, info = arena._attempt(user_text, picks, int(ngen), True, stops)
    # GRM3P-DIAG-CONTAM dual mount-time snapshot (opt-in). Taken AFTER
    # _attempt so fitted nodes are already device-resident via the normal
    # mount path; no extra pre-probe _ensure_h (avoids LRU/clock side
    # effects on the registered probe). RECENCY = live-window ids at
    # probe start, plus fitted mounts.
    _probe_mount_snapshot(
        repo, arena, live_idx=live_idx, picks=picks,
        planned=planned, turn_idx=turn_idx)
    info = dict(info or {})
    info["trip"] = 0
    info["driver_probe_multimount"] = True
    info["driver_topk"] = int(want)
    info["mount_plan"] = planned
    info["mount_fitted"] = picks
    info["mount_dropped_for_width"] = [
        i for i in planned if i not in set(picks)]
    info["ranking_ids"] = [int(x) for x in ranking]
    if admission_profile is not None:
        info.update(admission_info_fields(admission_profile))
    info["_route_observation"] = _route_observation(
        route_limit=int(want),
        excluded_live_ids=live_idx,
        admission_profile=admission_profile,
        trips=[{
            "ordinal": 0,
            "clean_room": False,
            "planned": [int(v) for v in planned],
            "mount_set": [int(v) for v in picks],
            "grounded": None,
        }],
        serving_path="grm_e2e_session.probe_multimount_chat",
    )
    info = _probe_finish_deposit(
        repo, before, user_text, ans, info, defer_memory=defer_memory)
    return ans, info


def script_for_mode(mode: str) -> list[dict[str, Any]]:
    return build_smoke_script() if mode == "smoke" else build_full_script()


def repo_node_count(repo: GraftRepository) -> int:
    return len(repo.arena.grafts)


def live_node_ids(repo: GraftRepository) -> list[int]:
    return [int(g) for g, _ntok in repo.arena.live_segs if g is not None]


def load_model_and_repo(args: argparse.Namespace, session_dir: Path):
    model, model_info = GptOss20B_TC.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True)
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    decode = lambda ids: tokenizer.decode(ids, clean_up_tokenization_spaces=False)
    cfg = model.config
    dialect = gpt_oss_grm_dialect_kwargs(cfg)
    arena_kw = {
        "route_layer": int(dialect["route_layer"]),
        "arena_width": int(args.arena_width),
        "topk": int(args.topk),
        "live_turns": int(args.live_turns),
        "max_live": int(args.max_live),
        # GRM-EB1: the driver serves under the SPEC frame — the ephemeral
        # boat.  ``ephemeral_frame_enabled(None)`` resolves the registered
        # GRM_PERSISTENT_BOAT escape and fails CLOSED to ephemeral, so the
        # driver is explicit about the frame it is running rather than
        # inheriting a constructor default silently.
        "ephemeral": ephemeral_frame_enabled(),
        "sink_text": HARMONY_SINK,
        "prompt_template": harmony_turn,
        "stop_sequences": HARMONY_STOPS,
        "storage_bits": 8,
        "revision_resolution": sup_resolve_enabled(args.sup_resolve),
        "decisive_admission": adm_decisive_enabled(args.adm_decisive),
    }
    repo = GraftRepository(
        model,
        encode,
        decode,
        str(session_dir / "repository"),
        autosave=False,
        arena_cls=GptOssGQAArenaCache,
        native_lib_path=str(args.native_lib) if args.native_lib else None,
        native_auto=False,
        vram_budget_mb=(
            int(args.vram_budget_mb)
            if args.vram_budget_mb is not None else None),
        **arena_kw,
    )
    return model, tokenizer, repo, model_info


def refeed_live_window(repo: GraftRepository, transcript: list[dict[str, Any]],
                       live_turns: int) -> list[dict[str, Any]]:
    """ESCAPE-ONLY (GRM-EB1).  Re-establish a persistent-frame live window.

    This pushes the last ``live_turns`` transcript turns back through the live
    cache with ``deposit=False``.  That is a CHAT LOG IN CONTEXT, which the
    spec forbids, so ``main()`` calls it only under ``GRM_PERSISTENT_BOAT``,
    to reproduce frozen receipts taken before the spec frame existed.  Under
    the spec frame resume is repository state: the turns are already grafts
    and recency-as-mount pulls them by routing on the next step.
    """
    replayed = []
    for row in transcript[-int(live_turns):]:
        turn_text = harmony_turn(row["user"], row["assistant"])
        before = len(repo.arena.live_segs)
        repo.arena.feed(turn_text, deposit=False)
        if len(repo.arena.live_segs) > before and row.get("chat_node_id") is not None:
            _gidx, ntok = repo.arena.live_segs[-1]
            repo.arena.live_segs[-1] = (int(row["chat_node_id"]), ntok)
        replayed.append({
            "turn": int(row["turn"]),
            "chat_node_id": row.get("chat_node_id"),
            "ntok": int(repo.arena.live_segs[-1][1]) if repo.arena.live_segs else 0,
        })
    return replayed


def stage_paths(session_dir: Path) -> dict[str, Path]:
    return {
        "config": session_dir / "run_config.json",
        "transcript": session_dir / "transcript.jsonl",
        "instrumentation": session_dir / "instrumentation.jsonl",
        "scorecard": session_dir / "probe_scorecard.json",
        "summary": session_dir / "summary.json",
        "restart": session_dir / "restart.json",
        "stage_timing": session_dir / "stage_timing.json",
        "arena_state": session_dir / "arena_state.json",
        "memory_ledger": session_dir / "memory_ledger",
        "working_set": session_dir / "working_set",
    }


def plant_complete_turn(
    repo: GraftRepository,
    event: dict[str, Any],
) -> tuple[str, dict[str, Any], int | None]:
    """Leg-1 feed() COMPLETE plant: user + scripted acceptance, deposit/evict.

    No free generation. Deposit path is the same arena.feed(deposit=True)
    used by runtime.add_turn, but with the Harmony complete-turn text so the
    harvested K/V is value-bearing.
    """
    answer = event.get("assistant") or plant_acceptance(event)
    turn_text = harmony_turn(event["user"], answer)
    before = repo._snapshot_state()
    nodes_before = len(repo.arena.grafts)
    repo.arena.feed(turn_text, deposit=True)
    repo._set_new_node_provenance(before, "exchange_span")
    extracted = repo._extract_from_new_turns(
        before,
        context={
            "event": "plant_feed",
            "user_text": event["user"],
            "assistant_text": answer,
        },
    )
    result = repo.runtime._finish_turn_event(
        "add_turn", before, extraction=extracted, autosave=False)
    new_nodes = list(getattr(result, "new_nodes", ()) or ())
    chat_node_id = int(new_nodes[0]) if new_nodes else None
    # Prefer the live-seg graft if feed assigned one.
    if repo.arena.live_segs:
        gidx, _ntok = repo.arena.live_segs[-1]
        if gidx is not None:
            chat_node_id = int(gidx)
    deposited_text = ""
    if chat_node_id is not None and 0 <= chat_node_id < len(repo.arena.grafts):
        deposited_text = str(repo.arena.grafts[chat_node_id].get("text", "") or "")
    value = str(event.get("value") or "")
    phrase = str(event.get("fact_phrase") or "")
    info = {
        "plant_mode": "feed_complete",
        "mounts": list(repo.arena.cur_mounts),
        "live_tokens": int(sum(n for _g, n in repo.arena.live_segs)),
        "new_nodes": [int(x) for x in new_nodes],
        "nodes_before": int(nodes_before),
        "nodes_after": int(len(repo.arena.grafts)),
        "deposit_contains_value": bool(value and value in deposited_text),
        "deposit_contains_phrase": bool(phrase and phrase in deposited_text),
        "deposit_text_prefix": deposited_text[:200],
        "extraction": list(extracted or ()),
    }
    return answer, info, chat_node_id


def infer_complete_turn_deferred(
    repo: GraftRepository,
    event: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Step 2 for scripted complete turns: build KV, defer persistence.

    This is ``ArenaCache.feed`` split at its mutation boundary. The complete
    turn is forwarded exactly once here; step 3 deposits the already-built
    cache span, advances the persistent conversation ordinal, and performs
    repository bookkeeping.
    """
    arena = repo.arena
    answer = event.get("assistant") or plant_acceptance(event)
    turn_text = harmony_turn(event["user"], answer)
    for layer in arena.m.layers:
        layer.self_attn.live_shift = arena.live_shift
    ids = arena.encode(turn_text)
    if arena.caches is None:
        arena._set_injection_host(arena.sink_h)
    arena._forward(ids)
    kv_graft.clear_injection(arena.m)
    route_key_token = int(
        getattr(arena, "_deferred_route_key_token", 0)) + 1
    arena._deferred_route_key_token = route_key_token
    route_keys = getattr(arena, "_deferred_route_keys", None)
    if not isinstance(route_keys, dict):
        route_keys = {}
        arena._deferred_route_keys = route_keys
    route_keys[route_key_token] = arena._node_key(turn_text)
    arena.live_segs.append((None, len(ids)))
    evicted = arena.evict()
    info = {
        "plant_mode": "feed_complete",
        "mounts": list(arena.cur_mounts),
        "live_tokens": int(sum(n for _g, n in arena.live_segs)),
        "evicted": int(evicted),
        "_deferred_memory": {
            "turn_text": turn_text,
            "user_text": event["user"],
            "seg_cache_ntok": int(len(ids)),
            # feed() does not classify complete observed turns as derivative
            # recalls, even when old mounts remain seated.
            "picks": [],
            "deposited": False,
            "importance_committed": False,
            "advance_s4_turn": True,
            "route_key_prepared": True,
            "route_key_token": int(route_key_token),
        },
    }
    return answer, info


def _turn_route_receipt(
    repo: GraftRepository,
    event: dict[str, Any],
    info: dict[str, Any],
    answer: str,
    *,
    turn_idx: int,
    session_id: str,
    prep_receipt: dict[str, Any] | None,
    arena_before_sha256: str | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """LSR-P2B: assemble this turn's grm.route_receipt.v1 from serve state.

    Every turn gets one — including plants and fillers, which route nothing;
    those record an empty ranking rather than no receipt at all, so a session
    ledger has no holes for a later investigation to interpret.
    """
    observation = dict(info.get("_route_observation") or {})
    prep = prep_receipt or {}
    ranking_ids = observation.get("ranking_ids")
    if ranking_ids is None:
        ranking_ids = info.get("ranking_ids")
    if ranking_ids is None and prep.get("routed"):
        ranking_ids = prep.get("ranking_ids")
    ranking_scores = None
    excluded_live = observation.get("excluded_live_ids")
    if excluded_live is None:
        excluded_live = prep.get("live_excluded_ids") or []
    if prep.get("routed") and prep.get("ranking_ids") == list(
            int(v) for v in (ranking_ids or ())):
        # ``_ranking_rows`` already carries the score column for this exact
        # ranking; reuse it rather than re-scoring (no model, no GPU).
        ranking_scores = [row.get("score") for row in prep.get("ranking") or ()]
    route_limit = observation.get("route_limit")
    if route_limit is None:
        route_limit = max(
            int(args.topk), (int(args.max_trips) + 1) * int(args.topk))
    return build_route_receipt(
        session_id=session_id,
        turn_id=str(turn_idx),
        info=info,
        arena=repo.arena,
        request_text=event["user"],
        output_text=answer,
        arena_before_sha256=arena_before_sha256,
        repository_size=len(repo.arena.grafts),
        serving_path=str(observation.get(
            "serving_path", f"grm_e2e_session.run_turn:{event['kind']}")),
        route_limit=route_limit,
        candidate_count=prep.get("repository_candidate_count"),
        ranking_ids=ranking_ids,
        ranking_scores=ranking_scores,
        excluded_live_ids=excluded_live,
        excluded_recency_ids=observation.get("excluded_recency_ids"),
        admission_profile=observation.get("admission_profile"),
        trips=observation.get("trips"),
        turn_kind=event["kind"],
    )


def run_pass3_memory_management(
    repo: GraftRepository,
    event: dict[str, Any],
    answer: str,
    info: dict[str, Any],
    before: list[Any],
    *,
    turn_idx: int,
    session_id: str,
    timers: TurnTimers,
    route_receipt: dict[str, Any] | None = None,
) -> tuple[
    dict[str, Any], int | None, dict[str, Any] | None,
    dict[str, Any], dict[str, Any],
]:
    """Execute all persistent turn mutation after output and receipt it."""
    ledger = MemoryLedgerBuilder(
        repo,
        session_id=session_id,
        turn_id=str(turn_idx),
        request_text=event["user"],
        output_text=answer,
    )
    # LSR-P2B: sidecar attach, before finalize().  Mutation rows and the
    # completeness audit are unaffected by construction.
    ledger.attach_route_receipt(route_receipt)
    nodes_before = len(repo.arena.grafts)
    chat_node_id = None
    correction_result = None
    extracted: list[Any] = []

    if event["kind"] in ("fact", "supersede"):
        turn_text = harmony_turn(event["user"], answer)

        def deposit_plant() -> None:
            repo.arena.deposit_deferred_turn(info)
            if info["_deferred_memory"].get("advance_s4_turn"):
                repo.arena._s4_turn = (
                    int(getattr(repo.arena, "_s4_turn", 0)) + 1)
                info["_deferred_memory"]["advance_s4_turn"] = False
            repo._set_new_node_provenance(before, "exchange_span")

        ledger.record_operation(
            "deposit",
            reason_code="complete_turn_deposit",
            reason_detail="deposit scripted complete plant after output",
            source_text=turn_text,
            operation=deposit_plant,
        )

        result_box: dict[str, Any] = {}

        def finish_plant() -> None:
            extracted.extend(repo._extract_from_new_turns(
                before,
                context={
                    "event": "plant_feed",
                    "user_text": event["user"],
                    "assistant_text": answer,
                },
            ))
            result_box["result"] = repo.runtime._finish_turn_event(
                "add_turn", before, extraction=extracted, autosave=False)

        ledger.record_operation(
            "deposit",
            reason_code="turn_deposit_bookkeeping",
            reason_detail="provenance, extraction, lifecycle, and paging",
            source_text=turn_text,
            operation=finish_plant,
        )
        result = result_box["result"]
        new_nodes = list(getattr(result, "new_nodes", ()) or ())
        chat_node_id = int(new_nodes[0]) if new_nodes else None
        if repo.arena.live_segs:
            gidx, _ntok = repo.arena.live_segs[-1]
            if gidx is not None:
                chat_node_id = int(gidx)
        deposited_text = ""
        if (chat_node_id is not None
                and 0 <= chat_node_id < len(repo.arena.grafts)):
            deposited_text = str(
                repo.arena.grafts[chat_node_id].get("text", "") or "")
        value = str(event.get("value") or "")
        phrase = str(event.get("fact_phrase") or "")
        info.update({
            "plant_mode": "feed_complete",
            "mounts": list(repo.arena.cur_mounts),
            "live_tokens": int(sum(n for _g, n in repo.arena.live_segs)),
            "new_nodes": [int(x) for x in new_nodes],
            "nodes_before": int(nodes_before),
            "nodes_after": int(len(repo.arena.grafts)),
            "deposit_contains_value": bool(value and value in deposited_text),
            "deposit_contains_phrase": bool(phrase and phrase in deposited_text),
            "deposit_text_prefix": deposited_text[:200],
            "extraction": list(extracted or ()),
        })
    else:
        ledger.record_operation(
            "deposit",
            reason_code="generated_turn_deposit",
            reason_detail="deposit accepted pass-2 output from live cache",
            source_text=str(
                info.get("_deferred_memory", {}).get("turn_text", "")),
            operation=lambda: repo.arena.deposit_deferred_turn(info),
        )
        if info.get("_deferred_memory", {}).get("importance_bookkeeping") is not None:
            ledger.record_operation(
                "importance_bookkeeping",
                reason_code="accepted_route_grounding",
                reason_detail="commit deferred S4 route/mount/grounding counters",
                source_text=event["user"],
                operation=lambda: repo.arena.commit_deferred_importance(info),
            )

        result_box = {}

        def finish_generated() -> None:
            extracted.extend(repo._extract_from_new_turns(
                before,
                context={
                    "event": "chat",
                    "user_text": event["user"],
                    "assistant_text": answer,
                },
            ))
            if extracted:
                info["extraction"] = extracted
            result_box["result"] = repo.runtime._finish_turn_event(
                "chat", before, extraction=extracted, autosave=True)

        ledger.record_operation(
            "deposit",
            reason_code="turn_deposit_bookkeeping",
            reason_detail="extraction, lifecycle, librarian, and paging",
            source_text=event["user"],
            operation=finish_generated,
        )
        result = result_box["result"]
        new_nodes = list(getattr(result, "new_nodes", ()) or ())
        chat_node_id = int(new_nodes[0]) if new_nodes else None

    if event["kind"] == "supersede":
        started = time.perf_counter()
        try:
            correction_result = ledger.record_operation(
                "supersession",
                reason_code="authoritative_correction_command",
                reason_detail="retire stale fact and write current replacement",
                source_text=event["correction_command"],
                operation=lambda: repo.apply_memory_command(
                    event["correction_command"]),
            )
        finally:
            timers.supersession_ms += (
                time.perf_counter() - started) * 1000.0
            timers.supersession_calls += 1

    receipt, audit = ledger.finalize()
    return info, chat_node_id, correction_result, receipt, audit


def run_turn(
    repo: GraftRepository,
    event: dict[str, Any],
    turn_idx: int,
    *,
    paths: dict[str, Path],
    transcript: list[dict[str, Any]],
    turn_records: dict[int, dict[str, Any]],
    probe_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    resumed: bool,
) -> None:
    # Opt-in paging telemetry context (no-op when disabled).
    tel = getattr(repo, "_paging_tel", None) or _paging_telemetry.get_telemetry()
    tel.set_context(turn=int(turn_idx), step="turn")
    mounts_before = list(repo.arena.cur_mounts)
    nodes_before = repo_node_count(repo)
    live_before = live_node_ids(repo)
    timers, restore = install_turn_timers(repo.arena)
    started = time.perf_counter()
    answer = ""
    chat_node_id = None
    correction_result = None
    correction_wall_ms = 0.0
    plant_mode = None
    route_diag = None
    exc = None
    info: dict[str, Any] = {}
    pass2_wall_ms = None
    pass3_wall_ms = None
    pass2_visible_memory_overhead_ms = None
    pass2_arena_read_only = None
    pass2_arena_before_sha256 = None
    pass2_arena_after_sha256 = None
    memory_ledger_audit = None
    memory_ledger_path = None
    route_receipt = None
    prep_receipt = None
    working_set_path = None
    prep_wall_ms = None
    step_io_receipt = None
    resolver_receipt = None
    io_tracker = TurnStepIOTracker(repo.arena)
    io_tracker.__enter__()
    try:
        if args.turn_pipeline == "single":
            # Registered legacy path. Fact/supersede plants use feed()
            # COMPLETE; probe/filler keep production chat()→step().
            if event["kind"] in ("fact", "supersede"):
                answer, info, chat_node_id = plant_complete_turn(repo, event)
                plant_mode = info.get("plant_mode")
                if event["kind"] == "supersede":
                    t0 = time.perf_counter()
                    try:
                        correction_result = repo.apply_memory_command(
                            event["correction_command"])
                    finally:
                        correction_wall_ms = (
                            time.perf_counter() - t0) * 1000.0
                        timers.supersession_ms += correction_wall_ms
                        timers.supersession_calls += 1
            else:
                if event["kind"] == "probe":
                    source_turn = int(event["source_turn"])
                    source_record = turn_records.get(source_turn, {})
                    source_node = source_record.get(
                        "memory_node_id", source_record.get("chat_node_id"))
                    route_diag = collect_route_diagnostics(
                        repo.arena, event["user"], source_node, top_k=5)
                    # Fork A: widen probe mount from argmax to top-k at driver
                    # call site (arena multi-mount picks list), not product patch.
                    # Probe ladder (default ON) enforces production laws here;
                    # --no-probe-ladder / GRM_PROBE_LADDER=0 restores legacy.
                    answer, info = probe_multimount_chat(
                        repo,
                        event["user"],
                        topk=int(args.topk),
                        ngen=int(args.ngen),
                        turn_idx=int(turn_idx),
                        probe_ladder=probe_ladder_enabled(args),
                        max_trips=int(args.max_trips),
                    )
                    if route_diag is not None:
                        route_diag = dict(route_diag)
                        route_diag["mount_plan"] = info.get("mount_plan")
                        route_diag["mount_fitted"] = info.get("mount_fitted")
                        route_diag["mount_dropped_for_width"] = info.get(
                            "mount_dropped_for_width")
                else:
                    answer, info = repo.runtime.chat(
                        event["user"],
                        ngen=int(args.ngen),
                        max_trips=int(args.max_trips),
                    )
                last = repo.runtime.last_result
                new_nodes = list(getattr(last, "new_nodes", ()) or ())
                chat_node_id = int(new_nodes[0]) if new_nodes else None
            # LSR-P2B: the single pipeline has no per-turn memory ledger file,
            # so its route receipt rides the instrumentation row instead.
            route_receipt = _turn_route_receipt(
                repo,
                event,
                info,
                answer,
                turn_idx=turn_idx,
                session_id=paths["memory_ledger"].parent.name,
                prep_receipt=None,
                arena_before_sha256=None,
                args=args,
            )
        else:
            source_node = None
            if event["kind"] == "probe":
                source_turn = int(event["source_turn"])
                source_record = turn_records.get(source_turn, {})
                source_node = source_record.get(
                    "memory_node_id", source_record.get("chat_node_id"))

            io_tracker.set_step("1_prep")
            prep_receipt, working_set_path = run_step1_prep(
                repo,
                event,
                turn_idx,
                transcript=transcript,
                source_node_id=source_node,
                paths=paths,
                args=args,
                timers=timers,
                io_tracker=io_tracker,
            )
            prep_wall_ms = float(prep_receipt["prep_wall_ms"])
            if event["kind"] == "probe":
                route_diag = {
                    "top5": prep_receipt["ranking"][:5],
                    "source_node_id": prep_receipt["source_node_id"],
                    "source_rank": prep_receipt["source_rank"],
                    "ranking_len": len(prep_receipt["ranking_ids"]),
                    "route_backend": prep_receipt["route_backend"],
                    "score_error": None,
                    "live_excluded": prep_receipt["live_excluded_ids"],
                    "ranking_ids": prep_receipt["ranking_ids"],
                }

            before = repo._snapshot_state()
            guard = Pass2ReadOnlyGuard(repo)
            resolver_context = (
                StagedWorkingSetResolver(
                    repo.arena,
                    probe_text=prep_receipt["probe_used"],
                    ranking_ids=prep_receipt["ranking_ids"],
                    staged_ids=prep_receipt["staged_ids"],
                    route_backend=str(prep_receipt["route_backend"]),
                )
                if prep_receipt.get("routed") else nullcontext(None)
            )
            io_tracker.set_step("2_inference")
            pass2_started = time.perf_counter()
            with guard, resolver_context as resolver:
                if event["kind"] in ("fact", "supersede"):
                    answer, info = infer_complete_turn_deferred(repo, event)
                    plant_mode = info.get("plant_mode")
                elif event["kind"] == "probe":
                    answer, info = probe_multimount_chat(
                        repo,
                        event["user"],
                        topk=int(args.topk),
                        ngen=int(args.ngen),
                        defer_memory=True,
                        turn_idx=int(turn_idx),
                        probe_ladder=probe_ladder_enabled(args),
                        max_trips=int(args.max_trips),
                    )
                    if route_diag is not None:
                        route_diag = dict(route_diag)
                        route_diag["mount_plan"] = info.get("mount_plan")
                        route_diag["mount_fitted"] = info.get("mount_fitted")
                        route_diag["mount_dropped_for_width"] = info.get(
                            "mount_dropped_for_width")
                else:
                    answer, info = repo.arena.step(
                        event["user"],
                        ngen=int(args.ngen),
                        max_trips=int(args.max_trips),
                        deposit=False,
                        defer_memory=True,
                    )
            if resolver is not None:
                resolver_receipt = resolver.receipt()
                prep_receipt["l1"] = resolver_receipt
                write_json(working_set_path, prep_receipt)
            pass2_wall_ms = (
                time.perf_counter() - pass2_started) * 1000.0
            pass2_visible_memory_overhead_ms = guard.visible_overhead_ms
            pass2_arena_read_only = guard.read_only
            pass2_arena_before_sha256 = guard.before_sha256
            pass2_arena_after_sha256 = guard.after_sha256

            io_tracker.set_step("3_cleanup")
            pass3_started = time.perf_counter()
            # LSR-P2B: build the route receipt from serve-time state BEFORE
            # pass-3 mutates the arena, so repository size / mount geometry
            # are the ones the serve actually saw.  Written on every turn:
            # probe, plant, filler, abstained, and no_mount_fit alike.
            route_receipt = _turn_route_receipt(
                repo,
                event,
                info,
                answer,
                turn_idx=turn_idx,
                session_id=paths["memory_ledger"].parent.name,
                prep_receipt=prep_receipt,
                arena_before_sha256=pass2_arena_before_sha256,
                args=args,
            )
            (info, chat_node_id, correction_result, memory_receipt,
             memory_ledger_audit) = run_pass3_memory_management(
                repo,
                event,
                answer,
                info,
                before,
                turn_idx=turn_idx,
                session_id=paths["memory_ledger"].parent.name,
                timers=timers,
                route_receipt=route_receipt,
            )
            correction_wall_ms = timers.supersession_ms
            plant_mode = info.get("plant_mode", plant_mode)
            memory_ledger_path = paths["memory_ledger"] / f"turn_{turn_idx:04d}.json"
            write_json(memory_ledger_path, memory_receipt)
            pass3_wall_ms = (
                time.perf_counter() - pass3_started) * 1000.0
            if not memory_ledger_audit.get("complete"):
                raise RuntimeError(
                    "memory-ledger completeness audit failed: "
                    f"{memory_ledger_audit['missing_targets']}")
    except Exception as err:  # receipt scripts should persist the failure row
        info = info or {}
        exc = repr(err)
    finally:
        step_io_receipt = io_tracker.receipt()
        io_tracker.__exit__(None, None, None)
        if prep_receipt is not None and working_set_path is not None:
            prep_receipt["step_io"] = step_io_receipt
            write_json(working_set_path, prep_receipt)
        restore()

    wall_ms = (time.perf_counter() - started) * 1000.0
    mounts_after = list(repo.arena.cur_mounts)
    nodes_after = repo_node_count(repo)
    live_after = live_node_ids(repo)
    reported_route_backend = (
        prep_receipt.get("route_backend")
        if (args.turn_pipeline == "three_pass"
            and prep_receipt is not None
            and prep_receipt.get("routed"))
        else repo.arena.last_route_backend
    )
    transcript_row = {
        "turn": int(turn_idx),
        "kind": event["kind"],
        "fact_id": event.get("fact_id"),
        "user": event["user"],
        "assistant": answer,
        "chat_node_id": chat_node_id,
        "resumed": bool(resumed),
        "plant_mode": plant_mode,
    }
    transcript.append(transcript_row)
    append_jsonl(paths["transcript"], transcript_row)

    turn_records[int(turn_idx)] = {
        "chat_node_id": chat_node_id,
        "kind": event["kind"],
        "fact_id": event.get("fact_id"),
        "value": event.get("value") or event.get("expected"),
    }
    if correction_result and correction_result.get("node_id") is not None:
        turn_records[int(turn_idx)]["memory_node_id"] = int(correction_result["node_id"])

    eviction_check = None
    probe_score = None
    if event["kind"] == "probe":
        source_turn = int(event["source_turn"])
        source_record = turn_records.get(source_turn, {})
        source_node = source_record.get("memory_node_id", source_record.get("chat_node_id"))
        eviction_check = {
            "source_turn": source_turn,
            "live_turns": int(args.live_turns),
            "policy_evicted_before_probe": bool(source_turn < turn_idx - int(args.live_turns)),
            "source_node_id": source_node,
            "live_node_ids_before": live_before,
            "source_node_live_before": bool(source_node in live_before)
            if source_node is not None else None,
        }
        probe_score = score_probe(answer, event)
        # info["mounts"] from arena._attempt is 1-indexed; also keep the
        # 0-indexed seated set (cur_mounts / mount_fitted) for scorecards.
        mounted_ids = list(info.get("mount_fitted") or mounts_after)
        probe_score.update({
            "turn": int(turn_idx),
            "route_backend": reported_route_backend,
            "mounts": info.get("mounts", []),
            "mounted_ids": [int(x) for x in mounted_ids],
            "mount_plan": info.get("mount_plan"),
            "eviction_check": eviction_check,
            "resumed": bool(resumed),
            "route_ranking": route_diag,
            # GRM-EB1: the frame receipt reaches the SCORECARD, not only the
            # instrumentation row's full ``info`` blob.  The scorecard is what
            # every downstream scorer reads, so a frame receipt that stopped
            # at ``info`` would leave "which frame served this probe, and what
            # did recency cost it" unanswerable from the gate's own artifact.
            "frame": {
                key: info[key]
                for key in (
                    "frame_ephemeral",
                    "frame_escape_active",
                    "recency_mounted_ids",
                    "recency_seats",
                    "live_segments_inherited",
                    "live_segments_carried_into_turn",
                    "live_segments_after_turn",
                )
                if key in info
            },
        })
        if info.get("admission_policy") == "A-DEC":
            probe_score["admission"] = {
                key: info[key]
                for key in (
                    "admission_policy",
                    "admission_policy_branch",
                    "admission_rank_plan",
                    "admission_identifier_hit_count",
                    "admission_identified_candidates",
                    "admission_route_margin_1_2",
                    "admission_route_margin_evaluated",
                    "admission_margin_threshold",
                    "admission_rule_sha256",
                )
            }
        probe_rows.append(probe_score)

    row = {
        "schema": "grm_e2e_session_turn_v1",
        "turn": int(turn_idx),
        "kind": event["kind"],
        "fact_id": event.get("fact_id"),
        "resumed": bool(resumed),
        "route_backend": reported_route_backend,
        "plant_mode": plant_mode,
        "mounts_before": mounts_before,
        "mounts_after": mounts_after,
        "mounts_changed": mounts_before != mounts_after,
        "live_node_ids_before": live_before,
        "live_node_ids_after": live_after,
        "live_window_tokens": int(info.get(
            "live_tokens", sum(n for _g, n in repo.arena.live_segs))),
        "repo_node_count": int(nodes_after),
        "repo_node_count_before": int(nodes_before),
        "new_node_count": int(nodes_after - nodes_before),
        "turn_wall_ms": wall_ms,
        "correction_wall_ms": correction_wall_ms,
        "turn_pipeline": args.turn_pipeline,
        "pass2_wall_ms": pass2_wall_ms,
        "pass3_wall_ms": pass3_wall_ms,
        "pass2_visible_memory_overhead_ms": pass2_visible_memory_overhead_ms,
        "pass2_arena_read_only": pass2_arena_read_only,
        "pass2_arena_before_sha256": pass2_arena_before_sha256,
        "pass2_arena_after_sha256": pass2_arena_after_sha256,
        "prep_wall_ms": prep_wall_ms,
        "prep_calls": 1 if prep_receipt is not None else 0,
        "working_set_path": (
            str(working_set_path) if working_set_path is not None else None),
        "working_set": prep_receipt,
        "step_io": step_io_receipt,
        "vram": vram_snapshot(),
        "info": info,
        "error": exc,
        **timers.as_dict(),
    }
    if correction_result is not None:
        row["correction_result"] = correction_result
    if eviction_check is not None:
        row["eviction_check"] = eviction_check
    if route_diag is not None:
        row["route_ranking"] = route_diag
    if probe_score is not None:
        row["probe_score"] = probe_score
    if memory_ledger_audit is not None:
        row["memory_ledger_audit"] = memory_ledger_audit
        row["memory_ledger_path"] = str(memory_ledger_path)
    if route_receipt is not None:
        # Also on the row so a single-pipeline session (which writes no
        # per-turn ledger file) still persists the route decision.
        row["route_receipt"] = route_receipt
    append_jsonl(paths["instrumentation"], row)
    print(json.dumps({
        "turn": turn_idx,
        "kind": event["kind"],
        "backend": row["route_backend"],
        "plant_mode": plant_mode,
        "route_ms": round(row["route_wall_ms"], 3),
        "deposit_ms": round(row["deposit_wall_ms"], 3),
        "mount_ms": round(row["mount_wall_ms"], 3),
        "nodes": row["repo_node_count"],
        "probe_pass": probe_score.get("pass") if probe_score else None,
        "source_rank": (route_diag or {}).get("source_rank"),
        "mounted_ids": (probe_score or {}).get("mounted_ids"),
        "mount_plan": (probe_score or {}).get("mount_plan"),
        "error": exc,
    }), flush=True)
    if exc is not None:
        raise RuntimeError(f"turn {turn_idx} failed: {exc}")


def write_scorecard(paths: dict[str, Path], probe_rows: list[dict[str, Any]]) -> None:
    passed = sum(1 for row in probe_rows if row.get("pass"))
    write_json(paths["scorecard"], {
        "schema": "grm_e2e_probe_scorecard_v1",
        "passed": int(passed),
        "total": int(len(probe_rows)),
        "all_passed": bool(passed == len(probe_rows)),
        "probes": probe_rows,
    })


def write_stage_timing(
    paths: dict[str, Path],
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    stage_fields = (
        ("prep", "prep_wall_ms", "prep_calls"),
        ("route", "route_wall_ms", "route_calls"),
        ("mount", "mount_wall_ms", "mount_calls"),
        ("infer", "infer_wall_ms", "infer_calls"),
        ("deposit", "deposit_wall_ms", "deposit_calls"),
        ("supersession", "supersession_wall_ms", "supersession_calls"),
        ("importance_bookkeeping", "importance_bookkeeping_wall_ms",
         "importance_bookkeeping_calls"),
    )
    stages = {}
    for name, wall_field, calls_field in stage_fields:
        values = [float(row.get(wall_field, 0.0) or 0.0) for row in rows]
        stages[name] = {
            "wall_ms_total": sum(values),
            "wall_ms_mean_per_turn": sum(values) / len(values) if values else 0.0,
            "wall_ms_max_turn": max(values) if values else 0.0,
            "calls": sum(int(row.get(calls_field, 0) or 0) for row in rows),
        }
    turn_values = [float(row.get("turn_wall_ms", 0.0) or 0.0) for row in rows]
    pass2_values = [float(row.get("pass2_wall_ms", 0.0) or 0.0)
                    for row in rows if row.get("pass2_wall_ms") is not None]
    pass3_values = [float(row.get("pass3_wall_ms", 0.0) or 0.0)
                    for row in rows if row.get("pass3_wall_ms") is not None]
    payload = {
        "schema": "grm.three_pass.stage_timing.v1",
        "turn_pipeline": args.turn_pipeline,
        "frame": {
            "mode": args.mode,
            "turns": len(rows),
            "arena_width": int(args.arena_width),
            "topk": int(args.topk),
            "ngen": int(args.ngen),
            "max_trips": int(args.max_trips),
            "live_turns": int(args.live_turns),
            "storage_bits": 8,
            "probe_ladder": bool(probe_ladder_enabled(args)),
            "sup_resolve": bool(sup_resolve_enabled(args.sup_resolve)),
            "adm_decisive": bool(adm_decisive_enabled(args.adm_decisive)),
        },
        "stages": stages,
        "turn_wall_ms_total": sum(turn_values),
        "turn_wall_ms_mean": (
            sum(turn_values) / len(turn_values) if turn_values else 0.0),
        "pass2_wall_ms_total": sum(pass2_values) if pass2_values else None,
        "pass3_wall_ms_total": sum(pass3_values) if pass3_values else None,
        "pass2_visible_memory_overhead_ms_max": max(
            (float(row.get("pass2_visible_memory_overhead_ms", 0.0) or 0.0)
             for row in rows
             if row.get("pass2_visible_memory_overhead_ms") is not None),
            default=None,
        ),
        "pass2_all_arena_read_only": (
            all(row.get("pass2_arena_read_only") is True for row in rows)
            if args.turn_pipeline == "three_pass" else None),
        "turns": [{
            "turn": int(row["turn"]),
            "kind": row["kind"],
            "turn_wall_ms": float(row["turn_wall_ms"]),
            **{
                name: float(row.get(wall_field, 0.0) or 0.0)
                for name, wall_field, _calls_field in stage_fields
            },
            "pass2_wall_ms": row.get("pass2_wall_ms"),
            "pass3_wall_ms": row.get("pass3_wall_ms"),
        } for row in rows],
    }
    write_json(paths["stage_timing"], payload)
    return payload


def maybe_restart(args: argparse.Namespace, session_dir: Path, paths: dict[str, Path],
                  repo: GraftRepository, turn_idx: int) -> None:
    print("checkpoint_restart=flush_now", flush=True)
    t0 = time.perf_counter()
    repo.flush_now()
    flush_ms = (time.perf_counter() - t0) * 1000.0
    write_json(paths["restart"], {
        "schema": "grm_e2e_restart_v1",
        "after_turn": int(turn_idx),
        "flush_wall_ms": flush_ms,
        "vram_before_exec": vram_snapshot(),
        "mode": args.mode,
    })
    try:
        repo.close()
    finally:
        del repo
        gc.collect()
        if hasattr(tc, "empty_cache"):
            tc.empty_cache()
    argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--mode", args.mode,
        "--resume",
        "--session-dir", str(session_dir),
        "--model-dir", str(args.model_dir),
        "--native-lib", str(args.native_lib),
        "--ngen", str(args.ngen),
        "--max-trips", str(args.max_trips),
        "--live-turns", str(args.live_turns),
        "--arena-width", str(args.arena_width),
        "--max-live", str(args.max_live),
        "--topk", str(args.topk),
        "--turn-pipeline", args.turn_pipeline,
        "--restart-after", str(args.restart_after),
        "--skip-gpu-idle-check",
    ]
    if args.vram_budget_mb is not None:
        argv += ["--vram-budget-mb", str(args.vram_budget_mb)]
    # Always freeze the resolved ladder decision on re-exec (default-on means
    # an escape-off parent must re-assert --no-probe-ladder; env alone is not
    # enough when the parent used only the CLI escape).
    argv += probe_ladder_cli_argv(probe_ladder_enabled(args))
    # Freeze L2 independently too: a restart must not re-resolve against an
    # env change made after the parent process established its frame.
    argv += sup_resolve_cli_argv(sup_resolve_enabled(args.sup_resolve))
    # Freeze A-DEC independently for the same reason: the restart belongs to
    # the parent process's admission frame even if its environment changes.
    argv += adm_decisive_cli_argv(adm_decisive_enabled(args.adm_decisive))
    os.execvpe(sys.executable, argv, os.environ.copy())


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.ngen is None:
        args.ngen = 24 if args.mode == "smoke" else 32
    if args.live_turns is None:
        args.live_turns = 1 if args.mode == "smoke" else 2
    script = script_for_mode(args.mode)
    if args.restart_after is None:
        args.restart_after = 5 if args.mode == "smoke" else len(script) // 2
    session_dir = args.session_dir or default_session_dir(args.mode)
    session_dir = session_dir.expanduser().resolve()
    paths = stage_paths(session_dir)
    if not args.resume and session_dir.exists():
        raise SystemExit(f"session dir already exists: {session_dir}")
    session_dir.mkdir(parents=True, exist_ok=True)

    os.environ["GRM_GQA_CUDA_ROUTE"] = "1"
    os.environ["GRM_GRAFT_STORAGE_BITS"] = "8"
    # Contam diag: if the caller enabled telemetry via env with the literal
    # token "{session}", expand it to this session directory. Empty/unset
    # keeps the default path completely dark.
    for env_key in (
        "GRM_PAGING_TELEMETRY_PATH",
        "GRM_MOUNT_SNAPSHOT_DIR",
    ):
        raw = os.environ.get(env_key, "")
        if "{session}" in raw:
            os.environ[env_key] = raw.replace("{session}", str(session_dir))

    if not args.skip_gpu_idle_check:
        idle = wait_for_idle_gpu()
        if not idle["idle"]:
            write_json(paths["summary"], {
                "schema": "grm_e2e_session_summary_v1",
                "status": "blocked_gpu_busy",
                "gpu_idle_check": idle,
            })
            raise SystemExit("GPU stayed busy for 15 minutes")
    else:
        idle = {"idle": "skipped", "waited_s": 0, "checks": []}

    if not args.resume:
        write_json(paths["config"], {
            "schema": "grm_e2e_session_config_v1",
            "mode": args.mode,
            "session_dir": str(session_dir),
            "model_dir": str(args.model_dir),
            "native_lib": str(args.native_lib),
            "turns": len(script),
            "restart_after": int(args.restart_after),
            "live_turns": int(args.live_turns),
            "arena_width": int(args.arena_width),
            "topk": int(args.topk),
            "ngen": int(args.ngen),
            "max_trips": int(args.max_trips),
            "turn_pipeline": args.turn_pipeline,
            "probe_ladder": bool(probe_ladder_enabled(args)),
            "sup_resolve": bool(sup_resolve_enabled(args.sup_resolve)),
            "adm_decisive": bool(adm_decisive_enabled(args.adm_decisive)),
            "vram_budget_mb": (
                int(args.vram_budget_mb)
                if args.vram_budget_mb is not None else None),
            "template_decision": (
                "ArenaCache prompt_template + stop_sequences hook; "
                "probe/filler use real GRMRuntime.chat()/ArenaCache.step(); "
                "fact/supersede plants use arena.feed() COMPLETE turns "
                "(scripted acceptance; value-bearing deposit, no free-gen)."
            ),
            "env": {
                "GRM_GQA_CUDA_ROUTE": os.environ["GRM_GQA_CUDA_ROUTE"],
                "GRM_GRAFT_STORAGE_BITS": os.environ["GRM_GRAFT_STORAGE_BITS"],
                "GRM_PAGING_TELEMETRY_PATH": os.environ.get(
                    "GRM_PAGING_TELEMETRY_PATH", ""),
                "GRM_MOUNT_SNAPSHOT_DIR": os.environ.get(
                    "GRM_MOUNT_SNAPSHOT_DIR", ""),
                "GRM_PROBE_LADDER": os.environ.get("GRM_PROBE_LADDER", ""),
                "GRM_SUP_RESOLVE": os.environ.get("GRM_SUP_RESOLVE", ""),
                "GRM_ADM_DECISIVE": os.environ.get(
                    "GRM_ADM_DECISIVE", ""),
            },
            "gpu_idle_check": idle,
            "script": script,
        })

    transcript = read_jsonl(paths["transcript"])
    instrumentation = read_jsonl(paths["instrumentation"])
    probe_rows = []
    if paths["scorecard"].exists():
        probe_rows = json.loads(paths["scorecard"].read_text(encoding="utf-8")).get(
            "probes", [])
    completed = len(instrumentation)
    if args.resume:
        start_turn = completed
    else:
        start_turn = 0

    t_load = time.perf_counter()
    model, tokenizer, repo, model_info = load_model_and_repo(args, session_dir)
    load_ms = (time.perf_counter() - t_load) * 1000.0
    refeed = []
    if args.resume:
        # GRM-EB1: RESUME = REPOSITORY STATE, NOT TRANSCRIPT RE-FEED.
        #
        # The old resume path pushed the last ``live_turns`` transcript turns
        # back through the live cache with ``deposit=False`` — literally
        # re-establishing a chat log in the model's context on every restart,
        # which is the thing the spec forbids.  Under the spec frame the
        # repository already holds those turns as grafts; recency-as-mount
        # picks them up on the next step, from the repository, by routing.
        # The re-feed therefore runs ONLY under the registered escape, where
        # it is needed to reproduce frozen persistent-frame receipts.
        frame_ephemeral = bool(getattr(repo.arena, "ephemeral", False))
        if frame_ephemeral:
            refeed_skipped_reason = "spec_frame_resume_is_repository_state"
        else:
            refeed_skipped_reason = None
            refeed = refeed_live_window(repo, transcript, int(args.live_turns))
        restart_payload = json.loads(paths["restart"].read_text(encoding="utf-8"))
        restart_payload.update({
            "resume_load_wall_ms": load_ms,
            "refeed": refeed,
            "frame_ephemeral": frame_ephemeral,
            "frame_escape_active": env_persistent_boat(),
            "refeed_skipped_reason": refeed_skipped_reason,
            "vram_after_refeed": vram_snapshot(),
        })
        write_json(paths["restart"], restart_payload)

    turn_records: dict[int, dict[str, Any]] = {}
    for row in transcript:
        turn_records[int(row["turn"])] = {
            "chat_node_id": row.get("chat_node_id"),
            "kind": row.get("kind"),
            "fact_id": row.get("fact_id"),
        }
    for row in instrumentation:
        corr = row.get("correction_result") or {}
        if corr.get("node_id") is not None:
            turn_records.setdefault(int(row["turn"]), {})["memory_node_id"] = int(
                corr["node_id"])

    started = time.perf_counter()
    final_arena_state = None
    try:
        for turn_idx in range(start_turn, len(script)):
            run_turn(
                repo,
                script[turn_idx],
                turn_idx,
                paths=paths,
                transcript=transcript,
                turn_records=turn_records,
                probe_rows=probe_rows,
                args=args,
                resumed=bool(args.resume),
            )
            write_scorecard(paths, probe_rows)
            if (not args.resume) and turn_idx + 1 == int(args.restart_after):
                maybe_restart(args, session_dir, paths, repo, turn_idx)
    finally:
        try:
            repo.flush_now()
            final_arena_state = {
                "schema": "grm.three_pass.arena_state.v1",
                "turn_pipeline": args.turn_pipeline,
                "node_count": len(repo.arena.grafts),
                "canonical_sha256": arena_state_sha256(repo),
                "canonical_with_payload_sha256": arena_state_sha256(
                    repo, include_payload=True),
            }
            repo.close()
        except Exception:
            pass
        del model
        del tokenizer
        gc.collect()
        if hasattr(tc, "empty_cache"):
            tc.empty_cache()

    all_rows = read_jsonl(paths["instrumentation"])
    write_scorecard(paths, probe_rows)
    stage_timing = write_stage_timing(paths, all_rows, args)
    if final_arena_state is not None:
        write_json(paths["arena_state"], final_arena_state)
    summary = {
        "schema": "grm_e2e_session_summary_v1",
        "status": "ok" if all(row.get("pass") for row in probe_rows) else "probe_failures",
        "mode": args.mode,
        "turn_pipeline": args.turn_pipeline,
        "session_dir": str(session_dir),
        "turns_completed": len(all_rows),
        "turns_expected": len(script),
        "probes_total": len(probe_rows),
        "probes_passed": sum(1 for row in probe_rows if row.get("pass")),
        "supersession_probes": sum(1 for row in probe_rows if row.get("old_value")),
        "post_restart_probes": sum(1 for row in probe_rows if row.get("resumed")),
        "restart": json.loads(paths["restart"].read_text(encoding="utf-8"))
        if paths["restart"].exists() else None,
        "model_info": model_info,
        "load_wall_ms_last_process": load_ms,
        "run_wall_ms_this_process": (time.perf_counter() - started) * 1000.0,
        "sample_instrumentation": all_rows[-1] if all_rows else None,
        "stage_timing": stage_timing,
        "arena_state": final_arena_state,
    }
    write_json(paths["summary"], summary)
    print(f"summary={paths['summary']}", flush=True)
    print(json.dumps({
        "status": summary["status"],
        "turns": summary["turns_completed"],
        "probes": [summary["probes_passed"], summary["probes_total"]],
        "session_dir": str(session_dir),
    }), flush=True)
    return 0 if summary["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
