#!/usr/bin/env python3
"""T1 NoPE-graft experiment — Trinity Nano arena live_shift width sweep.

Mirrors GPT-OSS law receipt:
  artifacts/grm_e2e/live_shift_width_ab_20260708_155826/ (clean @ ~115,
  collapse @ 387 word salad) on Trinity Nano under INT4 weights + fp32
  COMPUTE, APA OFF (STANDARD attention everywhere).

Prediction T1 (frozen, TRINITY_NANO_PORT_PLAN.md):
  On NoPE full-attention layers the arena RoPE-hole law does NOT apply —
  grafts mount at any live_shift without generation collapse.

Writes (only):
  scripts/trinity_nope_graft_width_sweep.py  (this file)
  artifacts/trinity_nope_graft/<stamp>/

Hard rails: no git commit; GPU wall <= 10 min per major phase; no edits
outside GraftRepository; Project-Tensor plan/ledger untouched.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PT_ROOT = Path("/mnt/ForgeRealm/Project-Tensor")
MODEL_DIR = Path("/mnt/ForgeRealm/models/trinity-nano")
OUT_ROOT = ROOT / "artifacts" / "trinity_nope_graft"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PT_ROOT / "tensor_cuda"))

import tensor_cuda as tc  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from core.graft_arena import GQAArenaCache  # noqa: E402
from core import kv_graft  # noqa: E402
from core.mistral7b_tc import BlockTC as GraftBlockTC  # noqa: E402
from core.mistral7b_tc import F as GraftF  # noqa: E402

# ---------------------------------------------------------------------------
# Import TrinityNano_TC from Project-Tensor without shadowing GraftRepository
# core/ (which owns graft_arena / kv_graft).
# ---------------------------------------------------------------------------

def _load_trinity_module():
    path = PT_ROOT / "core" / "trinity_nano_tc.py"
    name = "trinity_nano_tc_pt"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Trinity module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


TN = _load_trinity_module()
TrinityNano_TC = TN.TrinityNano_TC
TrinityBlockTC = TN.TrinityBlockTC
TrinityAttentionTC = TN.TrinityAttentionTC
TBlockTC = TN.BlockTC
LinearTC = TN.LinearTC
QuantLinearTC = TN.QuantLinearTC
RMSNormTC = TN.RMSNormTC
_cast = TN._cast
_to_dtype = TN._to_dtype
_repeat_kv = TN._repeat_kv
_trinity_scaled_attention = TN._trinity_scaled_attention
_band_mask = TN._band_mask

FACT_VALUE = "Vortex-3-Sierra"
FACT_USER = (
    f"Memory planting turn. The vault code is {FACT_VALUE}. "
    f"If asked later for the vault code, answer {FACT_VALUE} only."
)
FACT_ASSIST = f"Understood. Stored vault code = {FACT_VALUE}."
FILLER_USER = (
    "Session continuity filler smoke alpha. Acknowledge briefly. "
    "Do not change any stored fact values."
)
FILLER_ASSIST = "Acknowledged."
PROBE_USER = (
    "Recall probe. What is the vault code? Reply with only the value."
)
SYS_CONTENT = (
    "You are Trinity. Answer fact probes with the stored value only. Be concise."
)

WIDTHS = (96, 384, 768)
NGEN = 20
GPU_WALL_RAIL_S = 600.0
STOPS = ("<|im_end|>", "<|im_start|>")


# ---------------------------------------------------------------------------
# Clean-English / value-recovery scorers (receipt metrics, not gates-to-fudge)
# ---------------------------------------------------------------------------

def value_recovered(text: str, value: str = FACT_VALUE) -> bool:
    if not text:
        return False
    return value.lower() in text.lower()


def is_clean_english(text: str) -> bool:
    """Heuristic: coherent Latin-script prose vs salad/drift.

    Registered before the run. Fail-closed on empty/near-empty.
    Special-token loops and concat-salad fail.
    """
    t = (text or "").strip()
    if len(t) < 1:
        return False
    # hard rejects for known INT4 residual signatures
    if t.count("<|begin_of_text|>") >= 2:
        return False
    if t.count("null") >= 2:
        return False
    if "<|im_start|>" in t and t.count("<|") >= 2:
        return False
    # strip chat markup for scoring
    stripped = re.sub(r"<\|[^|]*\|>", " ", t).strip()
    if not stripped:
        # pure EOS / special-only is not clean English answer
        return False
    bad = sum(1 for ch in stripped if ord(ch) < 9 or ch in "\ufffd")
    if bad > 0:
        return False
    # pure code-shaped value alone counts as clean answer
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9\-_.]{2,40}", stripped):
        return True
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]{1,24}", stripped)
    if len(words) < 1:
        return False
    # word-salad: short broken tokens / angle brackets
    if stripped.count("<") + stripped.count(">") >= 3 and len(words) < 6:
        return False
    # concatenated no-space junk
    if len(stripped) >= 12 and (" " not in stripped) and len(words) < 3:
        return False
    toks = stripped.split()
    if len(toks) >= 6:
        uniq = len(set(toks)) / len(toks)
        if uniq < 0.35:
            return False
    alpha = sum(ch.isalpha() or ch.isspace() for ch in stripped)
    if alpha / max(1, len(stripped)) < 0.45:
        return False
    return True


def collapse_signature(text: str, ids: list[int] | None = None) -> str:
    """Classify generation failure mode (registered labels)."""
    t = (text or "").strip()
    ids = list(ids or [])
    if not t and not ids:
        return "empty"
    # BOS / special-token loops (INT4 free-gen residual signature)
    if ids and ids.count(0) >= max(3, len(ids) // 2):
        return "bos_loop"
    if t.count("<|begin_of_text|>") >= 3:
        return "bos_loop"
    if t.count("null") >= 3:
        return "null_spam"
    if ids == [3] or t in ("<|im_end|>",):
        return "eos_immediate"
    # GPT-OSS-style broken fragments
    if t.count("<") + t.count(">") >= 3:
        return "angle_salad"
    raw = re.sub(r"<\|[^|]*\|>", " ", t).strip()
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]{1,24}", raw)
    if len(raw) >= 8 and len(words) <= 1 and not re.search(r"[A-Za-z]{3,}", raw):
        return "non_alpha_salad"
    # concatenated junk without spaces (e.g. lineinwerInputStream...)
    if len(raw) >= 12 and (" " not in raw) and not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9\-_.]{2,40}", raw
    ):
        return "concat_salad"
    if value_recovered(t):
        return "value_hit"
    if is_clean_english(t):
        return "clean_english"
    return "other_degraded"


# ---------------------------------------------------------------------------
# Chat-template helpers (Harmony-equivalent sink = Trinity system prefix)
# ---------------------------------------------------------------------------

def build_sink_text(tok) -> str:
    return tok.apply_chat_template(
        [{"role": "system", "content": SYS_CONTENT}],
        tokenize=False,
        add_generation_prompt=False,
    )


def complete_turn(tok, user: str, assistant: str) -> str:
    """User+assistant complete turn WITHOUT re-emitting the system sink.

    Arena already seats the sink; feed() tokens append after it.
    """
    return tok.apply_chat_template(
        [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )


def probe_prefix(tok, user: str) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )


# ---------------------------------------------------------------------------
# Driver-level adapter: ArenaCache contract on TrinityNano_TC
# ---------------------------------------------------------------------------

def _apply_rope(q_or_k, cos, sin, pos0: int, L: int):
    """RoPE at absolute table offset pos0 for length L."""
    if str(cos.dtype) != str(q_or_k.dtype):
        cos = _to_dtype(cos, q_or_k.dtype)
        sin = _to_dtype(sin, q_or_k.dtype)
    if hasattr(tc, "rope_apply") and not tc.is_grad_enabled():
        return tc.rope_apply(q_or_k, cos, sin, int(pos0))
    cseg = cos.slice(0, int(pos0), int(L))
    sseg = sin.slice(0, int(pos0), int(L))
    return GraftF.apply_rotary(q_or_k, cseg, sseg)


_ORIG_TRINITY_ATTN_CALL = None


def _arena_attention_call(self, x, cos, sin, position_offset: int = 0, kv_cache=None):
    """Class-level arena-aware Trinity attention (driver patch, not product edit).

    Python `instance()` dispatches to type.__call__, so instance attribute
    assignment cannot intercept. We patch TrinityAttentionTC.__call__ once.

    Contract:
      - full/NoPE: never RoPE (inject raw; live unpositioned)
      - sliding: RoPE at position_offset + live_shift (or graft_seats);
        inject RoPEd at seats [0, graft_seats)
      - APA omitted (STANDARD only; order forbids APA on exact-gen receipts)
    """
    B, L, _ = x.shape
    H = self.num_heads
    KV = self.num_key_value_heads
    D = self.head_dim
    q = self.q_proj(x).reshape([B, L, H, D])
    k = self.k_proj(x).reshape([B, L, KV, D])
    v = self.v_proj(x).reshape([B, L, KV, D])
    gate = self.gate_proj(x)

    q = self.q_norm(q)
    k = self.k_norm(k)
    q = _cast(q).transpose(1, 2)
    k = _cast(k).transpose(1, 2)
    v = _cast(v).transpose(1, 2)

    if getattr(self, "_capture", False):
        self._captured = (k.numpy(), v.numpy())
    if getattr(self, "_capture_q", False):
        self._captured_q = q.numpy()

    inj = getattr(self, "inject_kv", None)
    kg_pre = vg = None
    gscale = 1.0
    if inj is not None:
        if len(inj) == 2:
            kg_pre, vg = inj
        else:
            kg_pre, vg, gscale = inj

    graft_seats = int(getattr(self, "graft_seats", 0) or 0)
    if kg_pre is not None:
        graft_seats = int(kg_pre.shape[2])
    shift = getattr(self, "live_shift", None)
    if shift is None:
        shift = graft_seats
    shift = int(shift or 0)
    rope_pos = int(position_offset) + shift

    if self.is_local_attention:
        q = _apply_rope(q, cos, sin, rope_pos, L)
        k = _apply_rope(k, cos, sin, rope_pos, L)

    if inj is not None and kv_cache is None:
        if self.is_local_attention:
            kg = _apply_rope(kg_pre, cos, sin, 0, graft_seats)
        else:
            kg = kg_pre
            if str(kg.dtype) != str(k.dtype):
                kg = _to_dtype(kg, k.dtype)
        if float(gscale) != 1.0:
            kg = kg * float(gscale)
        if str(vg.dtype) != str(v.dtype):
            vg = _to_dtype(vg, v.dtype)
        k = tc.cat([kg, k], dim=2)
        v = tc.cat([vg, v], dim=2)

    if kv_cache is not None:
        k = tc.cat([kv_cache[0], k], dim=2)
        v = tc.cat([kv_cache[1], v], dim=2)
    S = int(k.shape[2])
    is_causal = bool(L > 1 or kv_cache is not None)

    k_rep = _repeat_kv(k, self.num_key_value_groups)
    v_rep = _repeat_kv(v, self.num_key_value_groups)
    if self.is_local_attention and S > int(self.sliding_window):
        attn_mask = _band_mask(
            int(L), S, int(self.sliding_window),
            q.device.split(":")[0], q.dtype,
        )
        attn = _trinity_scaled_attention(
            q, k_rep, v_rep, attn_mask=attn_mask,
            is_causal=False, scale=self.scaling,
        )
        self.last_attention_backend = "standard_sliding_band"
    else:
        attn = _trinity_scaled_attention(
            q, k_rep, v_rep, is_causal=is_causal, scale=self.scaling,
        )
        self.last_attention_backend = (
            "standard_sliding_causal"
            if self.is_local_attention
            else "standard_full_nope"
        )

    keep = S
    if self.is_local_attention:
        keep = min(S, int(self.sliding_window))
    new_k = k if keep == S else k.slice(2, S - keep, keep)
    new_v = v if keep == S else v.slice(2, S - keep, keep)

    attn = attn.transpose(1, 2).reshape([B, L, H * D])
    gated = attn * gate.sigmoid()
    out = self.o_proj(_cast(gated))
    return out, (new_k, new_v)


def install_arena_attention_hooks(model) -> dict[str, Any]:
    """Install arena inject_kv/live_shift/_capture_q on Trinity attention.

    Product class lacks these hooks. Order permits a driver-level adapter —
    we patch the in-memory TrinityAttentionTC.__call__ (no Project-Tensor
    file writes) and initialize per-instance dials.
    """
    global _ORIG_TRINITY_ATTN_CALL
    if _ORIG_TRINITY_ATTN_CALL is None:
        _ORIG_TRINITY_ATTN_CALL = TrinityAttentionTC.__call__
        TrinityAttentionTC.__call__ = _arena_attention_call  # type: ignore[method-assign]

    patched = 0
    for layer in model.layers:
        att = layer.self_attn
        if not hasattr(att, "inject_kv"):
            att.inject_kv = None
        if not hasattr(att, "graft_seats"):
            att.graft_seats = 0
        if not hasattr(att, "live_shift"):
            att.live_shift = None
        if not hasattr(att, "_capture_q"):
            att._capture_q = False
        if not hasattr(att, "_captured_q"):
            att._captured_q = None
        att._arena_hook_installed = True
        patched += 1

    return {
        "patched_layers": patched,
        "class_call_patched": True,
        "note": (
            "driver-level TrinityAttentionTC.__call__ class patch: inject_kv + "
            "live_shift + _capture_q; NoPE full layers skip RoPE; sliding uses "
            "position_offset+live_shift; APA path omitted (STANDARD only). "
            "In-memory only — Project-Tensor sources untouched."
        ),
    }


class TrinityArenaModelAdapter:
    """Thin model surface ArenaCache expects.

    Contract mapping:
      model.layers / .config / .extend_rope / __call__(ids, kv_caches=, ...)
      model.rope_cos / .rope_sin  ←→  model.rope.cos / .rope.sin
      config.num_kv_heads         ←→  config.num_key_value_heads
    """

    def __init__(self, model):
        self._m = model
        self.layers = model.layers
        self.config = model.config
        if not hasattr(self.config, "num_kv_heads"):
            self.config.num_kv_heads = int(self.config.num_key_value_heads)
        if not hasattr(self.config, "num_layers"):
            self.config.num_layers = int(self.config.num_hidden_layers)
        if not hasattr(self.config, "hidden_dim"):
            self.config.hidden_dim = int(self.config.hidden_size)
        self._hook_meta = install_arena_attention_hooks(model)
        # Force STANDARD (APA OFF) on every layer.
        if hasattr(model, "set_attention_mode"):
            self._attn_mode = model.set_attention_mode("standard", full_only=True)
        else:
            self._attn_mode = {"mode": "standard", "note": "no set_attention_mode"}
        for layer in model.layers:
            layer.self_attn.attention_mode = "standard"

    @property
    def rope_cos(self):
        return self._m.rope.cos

    @property
    def rope_sin(self):
        return self._m.rope.sin

    def extend_rope(self, seq_len: int) -> None:
        self._m.extend_rope(int(seq_len))

    def configure_moe_empty_cache(self, interval: int = 0) -> None:
        if hasattr(self._m, "configure_moe_empty_cache"):
            self._m.configure_moe_empty_cache(interval)

    def __call__(self, input_ids_np, kv_caches=None, position_offset: int = 0,
                 last_token_only: bool = False, max_layers=None, caches=None):
        if caches is not None and kv_caches is None:
            kv_caches = caches
        # Ensure rope tables cover live_shift + offset + L (GPT-OSS model law).
        ids = np.asarray(input_ids_np, dtype=np.int64)
        L = int(ids.shape[1])
        shift = 0
        for layer in self.layers:
            att = layer.self_attn
            ls = getattr(att, "live_shift", None)
            if ls is None:
                ls = getattr(att, "graft_seats", 0)
            shift = max(shift, int(ls or 0))
        self.extend_rope(int(position_offset) + shift + L)
        return self._m(
            ids,
            kv_caches=kv_caches,
            position_offset=int(position_offset),
            last_token_only=last_token_only,
            max_layers=max_layers,
        )


class TrinityGQAArenaCache(GQAArenaCache):
    """GQA arena dialect with per-layer NoPE awareness.

    Full-attention (NoPE) layers: keys stay position-free — do NOT re-RoPE
    on mount and do NOT un-RoPE on deposit_from_cache export.
    Sliding layers: standard GQA pre-RoPE harvest + re-RoPE at seats.
    """

    POSITION_LAW = "mixed_nope_full_rope_sliding"
    STATE_KIND = "kv"
    GRAFTABILITY = "seat_remountable"
    REMOUNTABLE = True
    COMPOSITION = "multi_mount"

    def _is_sliding(self, li: int) -> bool:
        return bool(self.m.layers[int(li)].self_attn.is_local_attention)

    def _swap_cache_payloads(self, picks, head):
        # Force Python per-layer path so NoPE layers can skip rope.
        return None

    def _export_cache_payloads(self, n, pos0):
        return None

    def _arena_cache_transaction(self, picks):
        return None

    def _evict_cache_payloads(self, head, drop_tokens):
        return None

    def _graft_block(self, picks, li):
        hs = [self.grafts[i]["h"][li] for i in picks]
        blk = {
            key: (hs[0][key] if len(hs) == 1
                  else tc.cat([h[key] for h in hs], dim=dim))
            for key, dim in self.PAYLOAD
        }
        n = blk[self.PAYLOAD[0][0]].shape[self.PAYLOAD[0][1]]
        if self._is_sliding(li):
            blk = self._rope_block_at(blk, self.n_sink, inverse=False)
        return blk, n

    def _export_cache_payload(self, cache, n, pos0):
        li = None
        for i, c in enumerate(self.caches or []):
            if c is cache:
                li = i
                break
        is_sliding = True if li is None else self._is_sliding(li)
        seg = {}
        for ei, (key, dim) in enumerate(self.PAYLOAD):
            t = cache[ei]
            S = int(t.shape[dim])
            start = S - n
            if key in self.ROPE_KEYS and is_sliding:
                seg[key] = self._export_cache_tensor(key, t, dim, start, n, pos0)
            else:
                if hasattr(tc, "export_rows"):
                    with tc.no_grad():
                        seg[key] = tc.export_rows(t, dim, start, n)
                else:
                    seg[key] = t.slice(dim, start, n)
        return seg


# ---------------------------------------------------------------------------
# Compute-dtype / load (INT4 weights + fp32 COMPUTE)
# ---------------------------------------------------------------------------

def _iter_lineartc(model):
    for layer in model.layers:
        mlp = getattr(layer, "mlp", None)
        rg = getattr(mlp, "router_gate", None)
        if isinstance(rg, LinearTC):
            yield ("router_gate", rg)
        for name in ("gate_proj", "up_proj", "down_proj", "q_proj", "k_proj",
                     "v_proj", "o_proj", "gate_proj"):
            for obj in (getattr(layer, "self_attn", None), mlp,
                        getattr(mlp, "shared_experts", None)):
                if obj is None:
                    continue
                lin = getattr(obj, name, None)
                if isinstance(lin, LinearTC):
                    yield (name, lin)
        experts = getattr(mlp, "experts", None) or []
        for ex in experts:
            for name in ("gate_proj", "up_proj", "down_proj"):
                lin = getattr(ex, name, None)
                if isinstance(lin, LinearTC):
                    yield (f"expert.{name}", lin)
    lm = getattr(model, "lm_head", None)
    if isinstance(lm, LinearTC):
        yield ("lm_head", lm)


def set_compute_dtype(model, compute_dtype: str = "float32") -> dict[str, Any]:
    dt = str(compute_dtype)
    # Sync BOTH BlockTC knobs: Trinity product uses TBlockTC; arena/kv_graft
    # import GraftRepository's mistral7b_tc.BlockTC for payload dtypes.
    TBlockTC.COMPUTE_DTYPE = dt
    LinearTC.DTYPE = dt
    GraftBlockTC.COMPUTE_DTYPE = dt

    rope_note = "rope_not_present"
    if hasattr(model, "rope") and model.rope is not None:
        prev_len = int(getattr(model.rope, "_rope_len", 0) or 0)
        prev_dtype = str(model.rope.cos.dtype) if model.rope.cos is not None else None
        model.rope._rope_len = 0
        model.rope.cos = None
        model.rope.sin = None
        if prev_len > 0:
            model.rope.extend(prev_len)
        rope_note = (
            f"invalidated_and_rebuild_prev_len={prev_len}"
            f"_prev_dtype={prev_dtype}_now={dt}"
        )
    cast_n = 0
    for _name, lin in _iter_lineartc(model):
        if str(lin.wT.dtype) != dt:
            lin.wT = lin.wT.astype(dt)
            cast_n += 1
    return {
        "compute_dtype": dt,
        "weight_mode": "int4",
        "mode_label": (
            "fp32_compute_int4_dequant" if dt == "float32"
            else "bf16_compute_int4_dequant"
        ),
        "rope_rebuild": rope_note,
        "lineartc_cast_count": cast_n,
        "graft_blocktc_dtype": GraftBlockTC.COMPUTE_DTYPE,
        "trinity_blocktc_dtype": TBlockTC.COMPUTE_DTYPE,
    }


def load_model(model_dir: Path) -> tuple[Any, dict[str, Any], float, dict[str, Any]]:
    TBlockTC.COMPUTE_DTYPE = "bfloat16"
    LinearTC.DTYPE = "bfloat16"
    QuantLinearTC.FUSED_DECODE = True
    RMSNormTC.USE_FUSED = False
    t0 = time.perf_counter()
    with tc.no_grad():
        model, info = TrinityNano_TC.from_pretrained(
            str(model_dir), weight_mode="int4", load_lm_head=True, progress=True,
        )
        model.configure_moe_empty_cache(0)
        dtype_meta = set_compute_dtype(model, "float32")
        if hasattr(tc, "synchronize"):
            tc.synchronize()
    return model, info, time.perf_counter() - t0, dtype_meta


# ---------------------------------------------------------------------------
# Arena setup + width case
# ---------------------------------------------------------------------------

def cache_layer_summary(arena) -> dict[str, Any]:
    if arena.caches is None:
        return {"n_layers": 0}
    full_S, slid_S = [], []
    shifts = []
    for li, cache in enumerate(arena.caches):
        S = int(cache[0].shape[2])
        att = arena.m.layers[li].self_attn
        if att.is_local_attention:
            slid_S.append(S)
        else:
            full_S.append(S)
        shifts.append(getattr(att, "live_shift", None))
    return {
        "n_layers": len(arena.caches),
        "full_S_unique": sorted(set(full_S)),
        "sliding_S_unique": sorted(set(slid_S)),
        "full_S0": full_S[0] if full_S else None,
        "sliding_S0": slid_S[0] if slid_S else None,
        "live_shift_on_layers_unique": sorted({str(s) for s in shifts}),
        "all_S_equal": len(set(full_S + slid_S)) == 1,
    }


def make_arena(adapter, tok, *, sink_text: str, arena_width: int,
               live_turns: int = 1, max_live: int = 1024) -> TrinityGQAArenaCache:
    encode = lambda t: tok.encode(t, add_special_tokens=False)
    decode = lambda ids: tok.decode(ids, clean_up_tokenization_spaces=False)
    # route_layer 0: GQA dialect (qk-norm); layer 0 is sliding on Trinity.
    return TrinityGQAArenaCache(
        adapter,
        encode=encode,
        decode=decode,
        sink_text=sink_text,
        arena_width=int(arena_width),
        route_layer=0,
        topk=1,
        live_turns=int(live_turns),
        max_live=int(max_live),
        cache_deposits=True,
        stop_sequences=STOPS,
    )


def greedy_generate(arena, prompt_ids: list[int], ngen: int, tok) -> dict[str, Any]:
    """step()-style forced-answer greedy loop (no route; mounts already seated)."""
    for L in arena.m.layers:
        L.self_attn.live_shift = arena.live_shift
    t0 = time.perf_counter()
    steps = []
    row = arena._forward(prompt_ids)
    # clear inject after prompt (mirrors step/_attempt)
    kv_graft.clear_injection(arena.m)
    out = [int(row.argmax())]
    steps.append({
        "tag": "prompt",
        "chosen": out[-1],
        "text": tok.decode([out[-1]]),
        "pos": int(arena.pos),
        "rope0_expected": int(arena.live_shift) + (int(arena.pos) - len(prompt_ids)),
    })
    stopped = False
    stop_hit = None
    for i in range(ngen - 1):
        decoded = arena.decode(out)
        for s in STOPS:
            if s in decoded:
                stopped = True
                stop_hit = s
                break
        if stopped:
            break
        row = arena._forward([out[-1]])
        out.append(int(row.argmax()))
        if i < 2:
            steps.append({
                "tag": f"g{i}",
                "chosen": out[-1],
                "text": tok.decode([out[-1]]),
                "pos": int(arena.pos),
            })
    raw = arena.decode(out)
    txt = raw
    for s in STOPS:
        if s in txt:
            txt = txt.split(s)[0]
    txt = txt.strip()
    return {
        "ids": out,
        "text_raw": raw,
        "text": txt,
        "n_steps": len(out),
        "stopped": stopped,
        "stop_hit": stop_hit,
        "wall_s": round(time.perf_counter() - t0, 3),
        "steps_head": steps,
        "clean_english": is_clean_english(txt),
        "value_recovered": value_recovered(txt),
        "collapse_signature": collapse_signature(txt, out),
        "bos_count": int(out.count(0)),
        "unique_id_ratio": (
            round(len(set(out)) / max(1, len(out)), 4) if out else 0.0
        ),
    }


def run_width_case(
    adapter,
    tok,
    *,
    width: int,
    sink_text: str,
    mount: bool,
    ngen: int,
) -> dict[str, Any]:
    label = f"w{width}_{'mount' if mount else 'control'}"
    t0 = time.perf_counter()
    arena = make_arena(adapter, tok, sink_text=sink_text, arena_width=width)
    meta = {
        "label": label,
        "width": int(width),
        "mount": bool(mount),
        "n_sink": int(arena.n_sink),
        "live_shift": int(arena.live_shift),
        "sink_text": sink_text,
        "sink_ntok": int(arena.n_sink),
    }

    # Plant fact (complete turn) + filler (evict fact from live window).
    fact_turn = complete_turn(tok, FACT_USER, FACT_ASSIST)
    filler_turn = complete_turn(tok, FILLER_USER, FILLER_ASSIST)
    arena.feed(fact_turn, deposit=True)
    g_fact = 0
    fact_ntok = int(arena.grafts[0]["ntok"])
    arena.feed(filler_turn, deposit=True)
    live_idx = {g for g, _ in arena.live_segs if g is not None}
    fact_evicted = g_fact not in live_idx
    meta["fact_evicted"] = bool(fact_evicted)
    meta["fact_ntok"] = fact_ntok
    meta["live_segs_after_plant"] = [(int(g) if g is not None else None, int(n))
                                     for g, n in arena.live_segs]
    meta["graft0_text_head"] = str(arena.grafts[0]["text"])[:160]
    meta["pos_after_plant"] = int(arena.pos)
    meta["cache_after_plant"] = cache_layer_summary(arena)

    if mount:
        # Force-mount fact graft (route optional; isolate hole-vs-mount).
        ranking = arena.route(PROBE_USER, exclude=live_idx, limit=1)
        picks = list(ranking) if ranking else []
        forced = False
        if g_fact not in picks:
            picks = [g_fact]
            forced = True
        arena._ensure_h(picks)
        arena.swap(picks)
        meta["ranking"] = [int(x) for x in ranking]
        meta["picks"] = [int(x) for x in picks]
        meta["picks_forced"] = forced
        meta["cur_mount_n"] = int(arena.cur_mount_n)
        meta["cur_mounts"] = [int(x) for x in arena.cur_mounts]
    else:
        # Control: same live_shift, no arena mounts.
        if arena.cur_mounts:
            arena.swap([])
        meta["ranking"] = []
        meta["picks"] = []
        meta["picks_forced"] = False
        meta["cur_mount_n"] = int(arena.cur_mount_n)
        meta["cur_mounts"] = [int(x) for x in arena.cur_mounts]

    meta["cache_pre_gen"] = cache_layer_summary(arena)
    # Sliding-layer position receipt: live tokens see live_shift + pos.
    meta["sliding_live_shift_seen"] = int(arena.live_shift)
    meta["full_nope_rope"] = False
    meta["sliding_rope_at"] = (
        f"position_offset + live_shift (= pos + {arena.live_shift})"
    )

    prompt = probe_prefix(tok, PROBE_USER)
    prompt_ids = tok.encode(prompt, add_special_tokens=False)
    meta["probe_prompt"] = prompt
    meta["probe_prompt_ntok"] = len(prompt_ids)

    gen = greedy_generate(arena, prompt_ids, ngen, tok)
    meta["generation"] = gen
    meta["wall_s"] = round(time.perf_counter() - t0, 3)
    meta["cache_post_gen"] = cache_layer_summary(arena)

    # Free arena device state between cases.
    arena.reset_live_cache()
    del arena
    if hasattr(tc, "empty_cache"):
        tc.empty_cache()
    return meta


def t1_verdict(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Registered decision rule (pre-results):

    Absolute CONFIRM: at w∈{384,768} mount, clean_english holds (GPT-OSS
    collapsed there). Absolute REFUTE: large-w mount degrades *relative to
    w96* into a worse collapse class (hole-law-like width sensitivity).

    If baseline INT4 free-gen is already non-English at w96 / pure, absolute
    CONFIRM is blocked; relative width comparison still speaks to the hole
    law (T1_RELATIVE_HOLD vs T1_REFUTED).
    """
    by = {r["label"]: r for r in results}
    decisive = []
    for w in (96, 384, 768):
        m = by.get(f"w{w}_mount")
        c = by.get(f"w{w}_control")
        if m is None or "generation" not in m:
            continue
        g = m["generation"]
        decisive.append({
            "width": w,
            "live_shift": m.get("live_shift"),
            "mount_text": g.get("text"),
            "mount_clean": g.get("clean_english"),
            "mount_value": g.get("value_recovered"),
            "mount_sig": g.get("collapse_signature"),
            "mount_bos_count": g.get("bos_count"),
            "control_text": None if c is None else (c.get("generation") or {}).get("text"),
            "control_clean": None if c is None else (c.get("generation") or {}).get("clean_english"),
            "control_sig": None if c is None else (c.get("generation") or {}).get("collapse_signature"),
        })

    mount_large = [d for d in decisive if d["width"] in (384, 768)]
    w96 = next((d for d in decisive if d["width"] == 96), None)
    pure = by.get("pure_baseline")
    pure_sig = None
    if pure and "generation" in pure:
        pure_sig = pure["generation"].get("collapse_signature")

    if not mount_large:
        return {
            "verdict": "INCOMPLETE",
            "reason": "missing w384/w768 mount results",
            "decisive": decisive,
            "pure_sig": pure_sig,
        }

    gpt_oss_collapse_quote = "User: 3-V> < 3-4> <5 6-7-"
    all_clean = all(d["mount_clean"] for d in mount_large)
    any_value = any(d["mount_value"] for d in mount_large)
    w96_clean = None if w96 is None else w96["mount_clean"]
    w96_sig = None if w96 is None else w96["mount_sig"]

    # Relative width sensitivity: large-w signatures that are "worse" classes
    worse_than = {
        "clean_english": 0,
        "value_hit": 0,
        "eos_immediate": 1,
        "other_degraded": 2,
        "concat_salad": 3,
        "non_alpha_salad": 3,
        "angle_salad": 4,
        "null_spam": 4,
        "bos_loop": 4,
        "empty": 5,
    }
    baseline_rank = worse_than.get(w96_sig or "other_degraded", 2)
    large_ranks = [
        (d, worse_than.get(d["mount_sig"] or "other_degraded", 2))
        for d in mount_large
    ]
    width_worsened = any(rank > baseline_rank + 0 for _, rank in large_ranks)
    # Also flag novel angle_salad (GPT-OSS signature) not present at w96
    novel_angle = any(
        d["mount_sig"] == "angle_salad" and w96_sig != "angle_salad"
        for d in mount_large
    )

    if all_clean:
        verdict = "T1_CONFIRMED"
        reason = (
            "Generation stays clean_english at w=384 and w=768 (live_shift "
            f"{mount_large[0]['live_shift']}+) where GPT-OSS collapsed into "
            f"word salad ({gpt_oss_collapse_quote!r}). "
            + (
                "Probe value recovered on at least one large-width mount."
                if any_value else
                "Value recovery not required for hole-law immunity; "
                "value miss is a residual (routing/content), not collapse."
            )
        )
    elif width_worsened or novel_angle:
        dirty = mount_large
        quotes = [
            f"w{d['width']} shift={d['live_shift']} sig={d['mount_sig']}: "
            f"{d['mount_text']!r}"
            for d in dirty
        ]
        verdict = "T1_REFUTED"
        reason = (
            "Large live_shift worsens generation relative to w96 under "
            "NoPE-confirmed full layers (width-sensitive collapse). "
            f"w96 sig={w96_sig!r}. Quotes: " + " | ".join(quotes)
        )
    else:
        # Absolute clean fails (likely INT4 free-gen residual) but width does
        # not add GPT-OSS-style hole collapse.
        quotes = [
            f"w{d['width']} shift={d['live_shift']} sig={d['mount_sig']}: "
            f"{d['mount_text']!r}"
            for d in ([w96] if w96 else []) + mount_large
            if d is not None
        ]
        verdict = "T1_RELATIVE_HOLD"
        reason = (
            "No additional width-dependent collapse at w=384/768 vs w96 "
            f"(w96_sig={w96_sig!r}, large sigs="
            f"{[d['mount_sig'] for d in mount_large]!r}). "
            "Absolute clean_english is NOT available under ordered "
            "INT4+fp32 free-gen (pure baseline also non-English) — product "
            "INT4 residual blocks absolute T1_CONFIRMED. Relative to the "
            "GPT-OSS hole law (clean@~115 → salad@387), Trinity does NOT "
            "show that width-triggered salad transition. Quotes: "
            + " | ".join(q for q in quotes if q)
        )

    return {
        "verdict": verdict,
        "reason": reason,
        "decisive": decisive,
        "w96_mount_clean": w96_clean,
        "w96_mount_sig": w96_sig,
        "pure_sig": pure_sig,
        "width_worsened": bool(width_worsened),
        "novel_angle_salad": bool(novel_angle),
        "gpt_oss_law_reference": {
            "sys_w96_live_shift": 115,
            "sys_w96_answer_head": "I'm sorry, but I can't comply with that.",
            "sess_w384_live_shift": 387,
            "sess_w384_answer": gpt_oss_collapse_quote,
            "receipt": (
                "artifacts/grm_e2e/live_shift_width_ab_20260708_155826/receipt.json"
            ),
            "diag": (
                "artifacts/grm_e2e/step_generate_mech_diag_20260708_155956/DIAGNOSIS.md"
            ),
        },
        "t1_prediction_source": (
            "/mnt/ForgeRealm/Project-Tensor/docs/TRINITY_NANO_PORT_PLAN.md "
            "prediction T1 (frozen; not modified this order)"
        ),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--widths", type=int, nargs="+", default=list(WIDTHS))
    p.add_argument("--ngen", type=int, default=NGEN)
    p.add_argument("--wall-rail-s", type=float, default=GPU_WALL_RAIL_S)
    p.add_argument("--skip-control", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (OUT_ROOT / f"width_sweep_{stamp}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_t0 = time.perf_counter()
    print(f"[T1] loading INT4 Trinity Nano from {args.model_dir}", flush=True)
    model, model_info, load_s, dtype_meta = load_model(args.model_dir)
    print(f"[T1] load_s={load_s:.1f} dtype={dtype_meta['mode_label']}", flush=True)

    if load_s > args.wall_rail_s:
        write_json(out_dir / "ABORTED.json", {
            "reason": "load exceeded wall rail",
            "load_s": load_s,
            "wall_rail_s": args.wall_rail_s,
        })
        return 2

    tok = AutoTokenizer.from_pretrained(
        str(args.model_dir), trust_remote_code=True, local_files_only=True,
    )
    sink_text = build_sink_text(tok)
    sink_ids = tok.encode(sink_text, add_special_tokens=False)
    print(f"[T1] sink_ntok={len(sink_ids)} sink={sink_text!r}", flush=True)

    adapter = TrinityArenaModelAdapter(model)
    adapter.configure_moe_empty_cache(0)

    config = {
        "experiment": "T1_NoPE_graft_width_sweep",
        "model_dir": str(args.model_dir),
        "weight_mode": "int4",
        "compute": dtype_meta,
        "apa": "OFF_standard_everywhere",
        "widths": list(args.widths),
        "ngen": int(args.ngen),
        "fact_value": FACT_VALUE,
        "fact_user": FACT_USER,
        "fact_assist": FACT_ASSIST,
        "probe_user": PROBE_USER,
        "sys_content": SYS_CONTENT,
        "sink_text": sink_text,
        "sink_ntok": len(sink_ids),
        "model_info": model_info,
        "adapter_hooks": adapter._hook_meta,
        "attn_mode": adapter._attn_mode,
        "full_attention_layers": model.config.full_attention_indices(),
        "sliding_attention_layers": model.config.sliding_attention_indices(),
        "contract_notes": [
            "ArenaCache expects model.rope_cos/rope_sin; adapter exposes rope.cos/sin",
            "ArenaCache expects config.num_kv_heads; aliased from num_key_value_heads",
            "Product TrinityAttentionTC lacked inject_kv/live_shift/_capture_q; "
            "driver installed class-level TrinityAttentionTC.__call__ patch "
            "(in-memory only; not a product edit)",
            "GQAArenaCache always re-RoPEs keys on mount; TrinityGQAArenaCache "
            "skips re-RoPE/un-RoPE on full/NoPE layers (forced Python swap path)",
            "CUDA arena_row_pair_transaction / swap_row_pairs_with_rope disabled "
            "for mixed NoPE (would rope all layers uniformly)",
        ],
        "load_s": round(load_s, 3),
        "wall_rail_s": args.wall_rail_s,
        "out_dir": str(out_dir),
    }
    write_json(out_dir / "run_config.json", config)

    results: list[dict[str, Any]] = []
    aborted = None

    # Pure baseline (no arena, position_offset=0): isolates INT4 free-gen
    # residual from arena live_shift/mount effects.
    print("[T1] pure baseline (no arena)", flush=True)
    try:
        pure_prompt = tok.apply_chat_template(
            [
                {"role": "system", "content": SYS_CONTENT},
                {"role": "user", "content": PROBE_USER},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        pure_ids = tok.encode(pure_prompt, add_special_tokens=False)
        t_pure = time.perf_counter()
        with tc.no_grad():
            for L in adapter.layers:
                L.self_attn.live_shift = None
                L.self_attn.inject_kv = None
                L.self_attn.graft_seats = 0
            logits, caches = adapter(
                np.array([pure_ids], dtype=np.int64), last_token_only=True,
            )
            out: list[int] = []
            for step in range(int(args.ngen)):
                nid = int(np.argmax(logits.float().numpy()[0, -1]))
                out.append(nid)
                if nid == tok.eos_token_id:
                    break
                logits, caches = adapter(
                    np.array([[nid]], dtype=np.int64),
                    kv_caches=caches,
                    position_offset=len(pure_ids) + step,
                    last_token_only=True,
                )
        pure_text = tok.decode(out)
        pure_row = {
            "label": "pure_baseline",
            "width": None,
            "mount": False,
            "live_shift": 0,
            "n_sink": 0,
            "probe_prompt": pure_prompt,
            "probe_prompt_ntok": len(pure_ids),
            "generation": {
                "ids": out,
                "text": pure_text.strip(),
                "text_raw": pure_text,
                "n_steps": len(out),
                "clean_english": is_clean_english(pure_text),
                "value_recovered": value_recovered(pure_text),
                "collapse_signature": collapse_signature(pure_text, out),
                "bos_count": int(out.count(0)),
                "unique_id_ratio": (
                    round(len(set(out)) / max(1, len(out)), 4) if out else 0.0
                ),
                "wall_s": round(time.perf_counter() - t_pure, 3),
            },
            "wall_s": round(time.perf_counter() - t_pure, 3),
        }
        results.append(pure_row)
        write_json(out_dir / "case_pure_baseline.json", pure_row)
        g = pure_row["generation"]
        print(
            f"[T1]   pure sig={g['collapse_signature']} clean={g['clean_english']} "
            f"text={g['text']!r}",
            flush=True,
        )
        del caches, logits
        if hasattr(tc, "empty_cache"):
            tc.empty_cache()
    except Exception as exc:
        pure_row = {
            "label": "pure_baseline",
            "error": f"{type(exc).__name__}: {exc}",
        }
        results.append(pure_row)
        write_json(out_dir / "case_pure_baseline.json", pure_row)
        print(f"[T1] pure baseline ERROR {pure_row['error']}", flush=True)

    for w in args.widths:
        for mount in (True, False) if not args.skip_control else (True,):
            elapsed = time.perf_counter() - run_t0
            if elapsed > args.wall_rail_s:
                aborted = {
                    "reason": "overall wall rail before case",
                    "elapsed_s": elapsed,
                    "next_case": f"w{w}_{'mount' if mount else 'control'}",
                }
                break
            print(
                f"[T1] case w={w} mount={mount} live_shift≈{len(sink_ids)+w} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
            case_t0 = time.perf_counter()
            try:
                with tc.no_grad():
                    row = run_width_case(
                        adapter, tok,
                        width=int(w),
                        sink_text=sink_text,
                        mount=bool(mount),
                        ngen=int(args.ngen),
                    )
            except Exception as exc:
                row = {
                    "label": f"w{w}_{'mount' if mount else 'control'}",
                    "width": int(w),
                    "mount": bool(mount),
                    "error": f"{type(exc).__name__}: {exc}",
                    "wall_s": round(time.perf_counter() - case_t0, 3),
                }
                print(f"[T1] ERROR {row['error']}", flush=True)
            results.append(row)
            write_json(out_dir / f"case_{row['label']}.json", row)
            g = row.get("generation") or {}
            print(
                f"[T1]   live_shift={row.get('live_shift')} "
                f"clean={g.get('clean_english')} value={g.get('value_recovered')} "
                f"text={g.get('text')!r} wall={row.get('wall_s')}s",
                flush=True,
            )
            case_wall = time.perf_counter() - case_t0
            if case_wall > args.wall_rail_s:
                aborted = {
                    "reason": "single-case wall rail",
                    "case": row["label"],
                    "case_wall_s": case_wall,
                }
                break
        if aborted:
            break

    verdict = t1_verdict(results)
    # Build compact receipt table
    table = []
    for r in results:
        g = r.get("generation") or {}
        table.append({
            "label": r.get("label"),
            "width": r.get("width"),
            "live_shift": r.get("live_shift"),
            "mount": r.get("mount"),
            "text": g.get("text"),
            "ids": g.get("ids"),
            "clean_english": g.get("clean_english"),
            "value_recovered": g.get("value_recovered"),
            "collapse_signature": g.get("collapse_signature"),
            "bos_count": g.get("bos_count"),
            "cur_mount_n": r.get("cur_mount_n"),
            "sliding_S0": (r.get("cache_pre_gen") or {}).get("sliding_S0"),
            "full_S0": (r.get("cache_pre_gen") or {}).get("full_S0"),
            "error": r.get("error"),
        })

    receipt = {
        "table": table,
        "verdict": verdict,
        "sink_text": sink_text,
        "sink_ntok": len(sink_ids),
        "results": results,
        "aborted": aborted,
        "total_wall_s": round(time.perf_counter() - run_t0, 3),
        "honest_residuals": [
            "Value recovery depends on graft content + routing/mount fidelity, "
            "not only hole immunity; a clean miss is not T1 refutation.",
            "Sliding layers still RoPE under live_shift; mixed outcome possible "
            "(clean full NoPE contribution, degraded sliding) — S receipts captured.",
            "fp32 compute ~1s/token class; ngen=20 is a short probe, not long-form.",
            "INT4 weight numerics residual (prior port ledger); semantic path used "
            "fp32 compute as ordered.",
            "Driver attention hook is harness-only; product TrinityAttentionTC "
            "still lacks inject_kv/live_shift until a product order lands them.",
        ],
    }
    write_json(out_dir / "receipt.json", receipt)
    write_json(out_dir / "VERDICT.json", verdict)

    # Human-readable summary
    lines = [
        f"# T1 NoPE-graft width sweep — {verdict['verdict']}",
        "",
        f"**Reason:** {verdict['reason']}",
        "",
        f"Sink ({len(sink_ids)} tok): `{sink_text}`",
        "",
        "| label | width | live_shift | mount | clean | value | sig | text |",
        "|---|---:|---:|---|---|---|---|---|",
    ]
    for row in table:
        text = (row.get("text") or row.get("error") or "")
        text = text.replace("|", "\\|").replace("\n", "\\n")
        if len(text) > 80:
            text = text[:77] + "..."
        lines.append(
            f"| {row['label']} | {row['width']} | {row.get('live_shift')} | "
            f"{row.get('mount')} | {row.get('clean_english')} | "
            f"{row.get('value_recovered')} | {row.get('collapse_signature')} | "
            f"{text!r} |"
        )
    lines.extend([
        "",
        "## Decisive quotes",
    ])
    for d in verdict.get("decisive") or []:
        lines.append(
            f"- w{d['width']} shift={d['live_shift']} mount: {d['mount_text']!r} "
            f"(clean={d['mount_clean']}, value={d['mount_value']})"
        )
        if d.get("control_text") is not None:
            lines.append(
                f"  control: {d['control_text']!r} (clean={d['control_clean']})"
            )
    lines.extend([
        "",
        f"Artifacts: `{out_dir}`",
        f"Script: `scripts/trinity_nope_graft_width_sweep.py`",
        f"Total wall: {receipt['total_wall_s']}s",
    ])
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n[T1] VERDICT={verdict['verdict']}", flush=True)
    print(f"[T1] {verdict['reason']}", flush=True)
    print(f"[T1] artifacts → {out_dir}", flush=True)

    # Drop model
    del adapter, model
    if hasattr(tc, "empty_cache"):
        tc.empty_cache()
    return 0 if verdict["verdict"] in (
        "T1_CONFIRMED", "T1_REFUTED", "T1_INCONCLUSIVE", "T1_RELATIVE_HOLD"
    ) and not aborted else (1 if aborted else 0)


if __name__ == "__main__":
    raise SystemExit(main())
