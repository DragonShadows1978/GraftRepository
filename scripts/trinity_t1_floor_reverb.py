#!/usr/bin/env python3
"""T1 floor re-verdict: template → compute mode → width 96/768 verdict.

Stages (pre-registered forks in
artifacts/trinity_nope_graft/T1_FLOOR_REVERDICT_IMPLEMENTATION_PLAN.md):

  Stage 1 — REAL chat template + INT4/fp32 one-arm (w96 mount) + pure floors
  Stage 2 — bf16 layer-stream free-gen floor (if Stage 1 still broken)
  Stage 3 — T1 verdict: w96 vs w768, mount+control, in clean-floor mode

Writes only: scripts/trinity_*, artifacts/trinity_nope_graft/
Rails: no commit; GPU ≤10 min/run; Project-Tensor read-only.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PT_ROOT = Path("/mnt/ForgeRealm/Project-Tensor")
MODEL_DIR = Path("/mnt/ForgeRealm/models/trinity-nano")
OUT_ROOT = ROOT / "artifacts" / "trinity_nope_graft"
GPU_WALL_RAIL_S = 600.0
NGEN_STAGE1 = 20
NGEN_STAGE2 = 16
# Stage 3 under stream: fewer tokens so 4 cases can each stay under rail.
NGEN_STAGE3_STREAM = 8
NGEN_STAGE3_INT4 = 20

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PT_ROOT / "tensor_cuda"))
sys.path.insert(0, str(ROOT / "scripts"))

import tensor_cuda as tc  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

# Reuse T1 width-sweep harness (adapter, arena, scorers, loaders).
import trinity_nope_graft_width_sweep as W  # noqa: E402

from core import kv_graft  # noqa: E402


# ---------------------------------------------------------------------------
# Strict clean-English (self-contained; fail-closed)
# ---------------------------------------------------------------------------

def is_clean_english(text: str) -> bool:
    """Fail-closed prose detector for free-gen floor.

    Width-sweep heuristic false-positived INT4 natural salad
    ('.\\n\\n5c1256...'). Evidence class: heuristic scorer (not LM judge).
    """
    t = (text or "").strip()
    if len(t) < 1:
        return False
    if t.count("<|begin_of_text|>") >= 2:
        return False
    if t.count("null") >= 2:
        return False
    if "<|im_start|>" in t and t.count("<|") >= 2:
        return False
    stripped = re.sub(r"<\|[^|]*\|>", " ", t).strip()
    stripped = re.sub(r"\s+", " ", stripped)
    if not stripped or len(stripped) < 6:
        return False
    if sum(1 for ch in stripped if ord(ch) < 9 or ch in "\ufffd") > 0:
        return False
    # pure value token
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9\-_.]{2,40}", stripped):
        return True
    letters = sum(ch.isalpha() for ch in stripped)
    digits = sum(ch.isdigit() for ch in stripped)
    if letters < 8:
        return False
    if digits >= 3 and digits / max(1, letters) > 0.25:
        if not re.search(r"[A-Za-z]{3,}-\d-[A-Za-z]{3,}", stripped):
            return False
    junk_toks = re.findall(r"[A-Za-z]*\d[A-Za-z0-9']{2,}", stripped)
    if len(junk_toks) >= 2:
        return False
    if stripped.count("<") + stripped.count(">") >= 3:
        return False
    words = re.findall(r"[A-Za-z]{2,24}", stripped)
    if len(words) < 2:
        return False
    if " " not in stripped and len(words) < 4:
        return False
    # Camel/concat salad: very long alpha tokens without spaces
    if any(len(w) >= 12 for w in words) and len(stripped.split()) <= 3:
        return False
    common = {
        "the", "a", "an", "is", "are", "was", "of", "in", "to", "and", "or",
        "for", "it", "that", "this", "with", "as", "on", "at", "by", "from",
        "paris", "france", "capital", "located", "north", "city", "largest",
        "understood", "stored", "vault", "code", "sorry", "cannot", "can",
        "not", "you", "we", "answer", "value", "yes", "no", "but", "i",
    }
    low = {w.lower() for w in words}
    if low & common and len(words) >= 3:
        return True
    if len(words) >= 4 and letters / max(1, len(stripped)) >= 0.7 and digits <= 2:
        return True
    return False


def collapse_signature(text: str, ids: list[int] | None = None) -> str:
    if is_clean_english(text):
        if W.value_recovered(text):
            return "value_hit"
        return "clean_english"
    return _collapse_fallback(text, ids)


def _collapse_fallback(text: str, ids: list[int] | None = None) -> str:
    t = (text or "").strip()
    ids = list(ids or [])
    if not t and not ids:
        return "empty"
    if ids and ids.count(0) >= max(3, len(ids) // 2):
        return "bos_loop"
    if t.count("<|begin_of_text|>") >= 3:
        return "bos_loop"
    if t.count("null") >= 3:
        return "null_spam"
    if ids == [3] or t in ("<|im_end|>",):
        return "eos_immediate"
    if t.count("<") + t.count(">") >= 3:
        return "angle_salad"
    raw = re.sub(r"<\|[^|]*\|>", " ", t).strip()
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]{1,24}", raw)
    if len(raw) >= 8 and len(words) <= 1 and not re.search(r"[A-Za-z]{3,}", raw):
        return "non_alpha_salad"
    if len(raw) >= 12 and (" " not in raw) and not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9\-_.]{2,40}", raw
    ):
        return "concat_salad"
    if W.value_recovered(t):
        return "value_hit"
    return "other_degraded"


# Patch width-sweep helpers so run_width_case / greedy_generate use strict scorers.
# Only patch if W still has its own originals (module name check).
if getattr(W.is_clean_english, "__module__", "") == "trinity_nope_graft_width_sweep":
    W.is_clean_english = is_clean_english  # type: ignore[assignment]
    W.collapse_signature = collapse_signature  # type: ignore[assignment]
else:
    # Already patched from a prior import — rebind to our current defs.
    W.is_clean_english = is_clean_english  # type: ignore[assignment]
    W.collapse_signature = collapse_signature  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def append_ledger(ledger_path: Path, entry: dict[str, Any]) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    if ledger_path.exists():
        data = json.loads(ledger_path.read_text(encoding="utf-8"))
    else:
        data = {"schema": "t1_floor_reverb_ledger_v1", "entries": []}
    entry = dict(entry)
    entry["ts"] = datetime.now().isoformat(timespec="seconds")
    data["entries"].append(entry)
    write_json(ledger_path, data)


def nvidia_smi_line() -> str:
    import subprocess
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=10,
        ).strip()
        return out
    except Exception as exc:  # noqa: BLE001
        return f"nvidia-smi_failed:{type(exc).__name__}:{exc}"


# ---------------------------------------------------------------------------
# Stage 1 — template extract + INT4 arm
# ---------------------------------------------------------------------------

def extract_template_receipt(tok, model_dir: Path) -> dict[str, Any]:
    """Ledger exact chat-template bytes + rendered probe bytes."""
    jinja_path = model_dir / "chat_template.jinja"
    jinja_text = jinja_path.read_text(encoding="utf-8") if jinja_path.exists() else None
    loaded = tok.chat_template
    loaded_s = loaded if isinstance(loaded, str) else None

    def sha(s: str | None) -> str | None:
        if s is None:
            return None
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    p1_hash = "c295e73aea820982584a1f874fa71c61b1f3e6856adc6ef1d7efe339b936f2ad"

    msgs = {
        "sink": [{"role": "system", "content": W.SYS_CONTENT}],
        "probe_user_add_gen": [{"role": "user", "content": W.PROBE_USER}],
        "full_sys_probe_add_gen": [
            {"role": "system", "content": W.SYS_CONTENT},
            {"role": "user", "content": W.PROBE_USER},
        ],
        "fact_complete": [
            {"role": "user", "content": W.FACT_USER},
            {"role": "assistant", "content": W.FACT_ASSIST},
        ],
        "filler_complete": [
            {"role": "user", "content": W.FILLER_USER},
            {"role": "assistant", "content": W.FILLER_ASSIST},
        ],
    }
    rendered: dict[str, Any] = {}
    for name, m in msgs.items():
        add_gen = name.endswith("add_gen")
        text = tok.apply_chat_template(m, tokenize=False, add_generation_prompt=add_gen)
        ids = tok.encode(text, add_special_tokens=False)
        raw = text.encode("utf-8")
        rendered[name] = {
            "text": text,
            "text_repr": repr(text),
            "utf8_hex": raw.hex(),
            "utf8_sha256": hashlib.sha256(raw).hexdigest(),
            "n_bytes": len(raw),
            "n_tokens": len(ids),
            "ids_head": ids[:32],
            "ids_tail": ids[-8:] if len(ids) > 8 else ids,
            "starts_with_bos_token_id": bool(ids and ids[0] == 0),
            "add_generation_prompt": add_gen,
        }

    # Prior-sweep sink text (byte-compare).
    prior_sink = (
        "<|im_start|>system\n"
        "You are Trinity. Answer fact probes with the stored value only. Be concise."
        "<|im_end|>\n"
    )
    prior_full = (
        prior_sink
        + "<|im_start|>user\n"
        + W.PROBE_USER
        + "<|im_end|>\n"
        + "<|im_start|>assistant\n"
    )
    real_sink = rendered["sink"]["text"]
    real_full = rendered["full_sys_probe_add_gen"]["text"]

    return {
        "model_dir": str(model_dir),
        "jinja_path": str(jinja_path) if jinja_path.exists() else None,
        "jinja_sha256": sha(jinja_text),
        "loaded_chat_template_sha256": sha(loaded_s),
        "p1_reference_hash": p1_hash,
        "hashes_match_p1": sha(loaded_s) == p1_hash and sha(jinja_text) == p1_hash,
        "jinja_equals_loaded": jinja_text == loaded_s,
        "bos_token": tok.bos_token,
        "bos_token_id": int(tok.bos_token_id) if tok.bos_token_id is not None else None,
        "eos_token": tok.eos_token,
        "eos_token_id": int(tok.eos_token_id) if tok.eos_token_id is not None else None,
        "add_bos_token": bool(getattr(tok, "add_bos_token", False)),
        "rendered": rendered,
        "prior_sweep_sink_byte_equal": prior_sink.encode("utf-8") == real_sink.encode("utf-8"),
        "prior_sweep_full_prompt_byte_equal": (
            prior_full.encode("utf-8") == real_full.encode("utf-8")
        ),
        "finding": (
            "Prior width-sweep sink/probe already match REAL apply_chat_template "
            "bytes (hash c295e73…); template-invention hypothesis is refutable if "
            "Stage-1 arm still bos_loops."
            if prior_sink == real_sink and prior_full == real_full
            else "Prior sweep text differs from current apply_chat_template render."
        ),
    }


def pure_generate_resident(
    adapter,
    tok,
    prompt_ids: list[int],
    ngen: int,
    *,
    label: str,
) -> dict[str, Any]:
    """Greedy free-gen with no arena (live_shift=None)."""
    t0 = time.perf_counter()
    with tc.no_grad():
        for L in adapter.layers:
            L.self_attn.live_shift = None
            L.self_attn.inject_kv = None
            L.self_attn.graft_seats = 0
        logits, caches = adapter(
            np.array([prompt_ids], dtype=np.int64), last_token_only=True,
        )
        out: list[int] = []
        for step in range(int(ngen)):
            nid = int(np.argmax(logits.float().numpy()[0, -1]))
            out.append(nid)
            if tok.eos_token_id is not None and nid == int(tok.eos_token_id):
                break
            if nid == 0 and out.count(0) >= 3:
                # early-stop hard BOS loop to save wall
                # still record remaining as bos for signature
                break
            logits, caches = adapter(
                np.array([[nid]], dtype=np.int64),
                kv_caches=caches,
                position_offset=len(prompt_ids) + step,
                last_token_only=True,
            )
    text = tok.decode(out, clean_up_tokenization_spaces=False)
    txt = text
    for s in W.STOPS:
        if s in txt:
            txt = txt.split(s)[0]
    txt = txt.strip()
    del caches, logits
    if hasattr(tc, "empty_cache"):
        tc.empty_cache()
    return {
        "label": label,
        "width": None,
        "mount": False,
        "live_shift": 0,
        "n_sink": 0,
        "probe_prompt_ntok": len(prompt_ids),
        "generation": {
            "ids": out,
            "text": txt,
            "text_raw": text,
            "n_steps": len(out),
            "clean_english": W.is_clean_english(txt),
            "value_recovered": W.value_recovered(txt),
            "collapse_signature": W.collapse_signature(txt, out),
            "bos_count": int(out.count(0)),
            "unique_id_ratio": (
                round(len(set(out)) / max(1, len(out)), 4) if out else 0.0
            ),
            "wall_s": round(time.perf_counter() - t0, 3),
        },
        "wall_s": round(time.perf_counter() - t0, 3),
    }


def stage1_int4(
    out_dir: Path,
    ledger_path: Path,
    model_dir: Path,
    *,
    wall_rail_s: float,
) -> dict[str, Any]:
    stage_dir = out_dir / "stage1_template_int4"
    stage_dir.mkdir(parents=True, exist_ok=True)
    run_t0 = time.perf_counter()

    tok = AutoTokenizer.from_pretrained(
        str(model_dir), trust_remote_code=True, local_files_only=True,
    )
    tmpl = extract_template_receipt(tok, model_dir)
    write_json(stage_dir / "template_receipt.json", tmpl)
    append_ledger(ledger_path, {
        "stage": 1,
        "event": "template_extracted",
        "hashes_match_p1": tmpl["hashes_match_p1"],
        "prior_byte_equal": tmpl["prior_sweep_full_prompt_byte_equal"],
        "jinja_sha256": tmpl["jinja_sha256"],
        "sink_utf8_sha256": tmpl["rendered"]["sink"]["utf8_sha256"],
        "full_prompt_utf8_sha256": tmpl["rendered"]["full_sys_probe_add_gen"]["utf8_sha256"],
        "finding": tmpl["finding"],
    })

    print(f"[S1] template hash={tmpl['loaded_chat_template_sha256']}", flush=True)
    print(f"[S1] match_p1={tmpl['hashes_match_p1']} prior_byte_eq={tmpl['prior_sweep_full_prompt_byte_equal']}", flush=True)
    print(f"[S1] sink bytes: {tmpl['rendered']['sink']['text_repr']}", flush=True)

    print(f"[S1] loading INT4 model from {model_dir}", flush=True)
    model, model_info, load_s, dtype_meta = W.load_model(model_dir)
    print(f"[S1] load_s={load_s:.1f} dtype={dtype_meta['mode_label']} gpu={nvidia_smi_line()}", flush=True)
    append_ledger(ledger_path, {
        "stage": 1,
        "event": "model_loaded",
        "load_s": round(load_s, 3),
        "dtype_meta": dtype_meta,
        "model_info_weight_mode": model_info.get("weight_mode"),
        "gpu": nvidia_smi_line(),
    })
    if load_s > wall_rail_s:
        payload = {"status": "ABORTED", "reason": "load exceeded wall", "load_s": load_s}
        write_json(stage_dir / "ABORTED.json", payload)
        return payload

    adapter = W.TrinityArenaModelAdapter(model)
    adapter.configure_moe_empty_cache(0)
    sink_text = tmpl["rendered"]["sink"]["text"]

    results: list[dict[str, Any]] = []

    # Pure chat baseline (real template).
    pure_prompt = tmpl["rendered"]["full_sys_probe_add_gen"]["text"]
    pure_ids = tok.encode(pure_prompt, add_special_tokens=False)
    print(f"[S1] pure_chat ntok={len(pure_ids)}", flush=True)
    pure_chat = pure_generate_resident(
        adapter, tok, pure_ids, NGEN_STAGE1, label="pure_chat_baseline",
    )
    pure_chat["probe_prompt"] = pure_prompt
    pure_chat["mode"] = "int4_fp32_resident"
    results.append(pure_chat)
    write_json(stage_dir / "case_pure_chat_baseline.json", pure_chat)
    g = pure_chat["generation"]
    print(
        f"[S1]   pure_chat sig={g['collapse_signature']} clean={g['clean_english']} "
        f"text={g['text']!r}",
        flush=True,
    )
    append_ledger(ledger_path, {
        "stage": 1,
        "event": "pure_chat",
        "sig": g["collapse_signature"],
        "clean": g["clean_english"],
        "text": g["text"],
        "ids": g["ids"],
        "wall_s": g["wall_s"],
    })

    # Pure natural positive control (T4 parity prompt).
    nat = "The capital of France is"
    nat_ids = tok.encode(nat, add_special_tokens=False)
    pure_nat = pure_generate_resident(
        adapter, tok, nat_ids, NGEN_STAGE1, label="pure_natural_baseline",
    )
    pure_nat["probe_prompt"] = nat
    pure_nat["mode"] = "int4_fp32_resident"
    results.append(pure_nat)
    write_json(stage_dir / "case_pure_natural_baseline.json", pure_nat)
    g = pure_nat["generation"]
    print(
        f"[S1]   pure_natural sig={g['collapse_signature']} clean={g['clean_english']} "
        f"text={g['text']!r}",
        flush=True,
    )
    append_ledger(ledger_path, {
        "stage": 1,
        "event": "pure_natural",
        "sig": g["collapse_signature"],
        "clean": g["clean_english"],
        "text": g["text"],
        "ids": g["ids"],
        "wall_s": g["wall_s"],
        "evidence_class": "end-to-end free-gen vs T4 parity natural prompt",
    })

    # ONE arm: w96 mounted
    elapsed = time.perf_counter() - run_t0
    if elapsed > wall_rail_s:
        payload = {"status": "ABORTED", "reason": "wall before w96 mount", "elapsed_s": elapsed}
        write_json(stage_dir / "ABORTED.json", payload)
        return payload

    print("[S1] w96_mount INT4+fp32", flush=True)
    with tc.no_grad():
        w96 = W.run_width_case(
            adapter, tok, width=96, sink_text=sink_text, mount=True, ngen=NGEN_STAGE1,
        )
    w96["mode"] = "int4_fp32_resident"
    results.append(w96)
    write_json(stage_dir / "case_w96_mount.json", w96)
    g = w96.get("generation") or {}
    print(
        f"[S1]   w96_mount shift={w96.get('live_shift')} sig={g.get('collapse_signature')} "
        f"clean={g.get('clean_english')} value={g.get('value_recovered')} "
        f"text={g.get('text')!r}",
        flush=True,
    )
    append_ledger(ledger_path, {
        "stage": 1,
        "event": "w96_mount",
        "live_shift": w96.get("live_shift"),
        "sig": g.get("collapse_signature"),
        "clean": g.get("clean_english"),
        "value": g.get("value_recovered"),
        "text": g.get("text"),
        "ids": g.get("ids"),
        "wall_s": w96.get("wall_s"),
    })

    pure_chat_clean = pure_chat["generation"]["clean_english"]
    pure_nat_clean = pure_nat["generation"]["clean_english"]
    w96_clean = bool(g.get("clean_english"))
    pure_chat_sig = pure_chat["generation"]["collapse_signature"]
    pure_nat_sig = pure_nat["generation"]["collapse_signature"]
    w96_sig = g.get("collapse_signature")

    # Floor decision (registered):
    # clean floor requires pure_chat clean OR (if chat inherently EOS-y) pure_nat
    # clean AND w96 mount clean. Absolute T1 needs clean generation on probe path.
    floor_clean = bool(pure_chat_clean or w96_clean)
    still_bos = (not floor_clean) and (
        pure_chat_sig == "bos_loop" or w96_sig == "bos_loop"
    )

    decision = {
        "stage": 1,
        "template_was_already_real": bool(tmpl["prior_sweep_full_prompt_byte_equal"]),
        "hashes_match_p1": bool(tmpl["hashes_match_p1"]),
        "pure_chat_clean": pure_chat_clean,
        "pure_chat_sig": pure_chat_sig,
        "pure_chat_text": pure_chat["generation"]["text"],
        "pure_natural_clean": pure_nat_clean,
        "pure_natural_sig": pure_nat_sig,
        "pure_natural_text": pure_nat["generation"]["text"],
        "w96_mount_clean": w96_clean,
        "w96_mount_sig": w96_sig,
        "w96_mount_text": g.get("text"),
        "w96_mount_value": g.get("value_recovered"),
        "floor_clean": floor_clean,
        "still_bos_or_degraded": not floor_clean,
        "fork": (
            "PROCEED_STAGE3_INT4"
            if floor_clean
            else "PROCEED_STAGE2_BF16_STREAM"
        ),
        "note": (
            "Template bytes match P1 + prior sweep; residual is not template invention."
            if tmpl["prior_sweep_full_prompt_byte_equal"] and not floor_clean
            else (
                "Floor cleaned under real template (unexpected if prior was real)."
                if floor_clean else "degraded"
            )
        ),
    }
    # Extra signal: natural clean under INT4 would refute universal free-gen break.
    if pure_nat_clean and not pure_chat_clean:
        decision["int4_natural_ok_chat_broken"] = True
        decision["note"] += (
            " INT4 free-gen OK on natural prompt; chat/probe path degraded "
            "(not universal INT4 free-gen death)."
        )
    elif not pure_nat_clean:
        decision["int4_natural_also_broken"] = True
        decision["note"] += (
            " INT4 free-gen also broken on natural T4-parity prompt "
            f"(sig={pure_nat_sig}); supports weight/compute residual over template."
        )

    write_json(stage_dir / "DECISION.json", decision)
    write_json(stage_dir / "results.json", {"results": results, "decision": decision})
    append_ledger(ledger_path, {"stage": 1, "event": "decision", **decision})

    # Free model before Stage 2.
    del adapter, model
    gc.collect()
    if hasattr(tc, "empty_cache"):
        tc.empty_cache()

    decision["total_wall_s"] = round(time.perf_counter() - run_t0, 3)
    decision["stage_dir"] = str(stage_dir)
    decision["gpu_after"] = nvidia_smi_line()
    write_json(stage_dir / "DECISION.json", decision)
    print(f"[S1] DECISION fork={decision['fork']} wall={decision['total_wall_s']}s", flush=True)
    return decision


# ---------------------------------------------------------------------------
# Stage 2 — bf16 layer-stream free-gen floor
# ---------------------------------------------------------------------------

def _import_stream_bits():
    """Load Trinity stream primitives from product (read-only import)."""
    # trinity_nano_tc already loaded via W.TN
    TN = W.TN
    return {
        "TrinityNanoConfig": TN.TrinityNanoConfig,
        "SafeTensorSource": TN.SafeTensorSource,
        "TrinityBlockTC": TN.TrinityBlockTC,
        "HostEmbedding": TN.HostEmbedding,
        "RMSNormTC": TN.RMSNormTC,
        "RoPECache": TN.RoPECache,
        "_linear_from_weight": TN._linear_from_weight,
        "_cast": TN._cast,
        "TBlockTC": TN.BlockTC,
        "LinearTC": TN.LinearTC,
        "QuantLinearTC": TN.QuantLinearTC,
        "RMSNormTC_cls": TN.RMSNormTC,
    }


class Bf16StreamRunner:
    """One-layer-at-a-time bf16 runner (parity precedent). Optional inject hooks."""

    def __init__(self, model_dir: Path, *, compute_dtype: str = "float32"):
        bits = _import_stream_bits()
        self.bits = bits
        self.model_dir = Path(model_dir)
        self.compute_dtype = compute_dtype
        bits["TBlockTC"].COMPUTE_DTYPE = compute_dtype
        bits["LinearTC"].DTYPE = "bfloat16"  # weight storage dtype on load
        bits["QuantLinearTC"].FUSED_DECODE = True
        bits["RMSNormTC_cls"].USE_FUSED = False
        W.GraftBlockTC.COMPUTE_DTYPE = compute_dtype

        self.cfg = bits["TrinityNanoConfig"].from_model_dir(self.model_dir)
        self.source = bits["SafeTensorSource"](self.model_dir)
        self.source.__enter__()
        self.embed_tokens = bits["HostEmbedding"]()
        emb = self.source.get_np("model.embed_tokens.weight")
        self.embed_tokens.weight = np.ascontiguousarray(emb.astype(np.float32, copy=False))
        del emb
        self.norm = bits["RMSNormTC"](self.cfg.hidden_size, self.cfg.rms_norm_eps)
        self.norm.weight = tc.tensor(self.source.get_np("model.norm.weight"), dtype="float32")
        # lm_head as bf16 LinearTC then cast compute if needed
        self.lm_head = bits["_linear_from_weight"](
            self.source.get_np("lm_head.weight"), "bf16"
        )
        if str(getattr(self.lm_head, "wT", self.lm_head).dtype) != compute_dtype:
            if hasattr(self.lm_head, "wT"):
                self.lm_head.wT = self.lm_head.wT.astype(compute_dtype)
        self.rope = bits["RoPECache"](self.cfg)
        self.rope.extend(min(4096, self.cfg.max_position_embeddings))
        # Per-layer dials (reapplied each layer load)
        self.live_shift: int | None = None
        self.inject_by_layer: dict[int, Any] | None = None
        self.graft_seats: int = 0
        # Class-level attention patch for live_shift / inject_kv dials.
        if W._ORIG_TRINITY_ATTN_CALL is None:
            W._ORIG_TRINITY_ATTN_CALL = W.TrinityAttentionTC.__call__
            W.TrinityAttentionTC.__call__ = W._arena_attention_call  # type: ignore

    def close(self) -> None:
        try:
            self.source.__exit__(None, None, None)
        except Exception:  # noqa: BLE001
            pass

    def _cast_linears(self, block) -> None:
        """Cast plain LinearTC weights to compute_dtype (bf16 load → fp32 compute)."""
        dt = self.compute_dtype
        bits = self.bits
        LinearTC = bits["LinearTC"]

        def _cast_lin(lin):
            if lin is None or not isinstance(lin, LinearTC):
                return
            if hasattr(lin, "wT") and str(lin.wT.dtype) != dt:
                lin.wT = lin.wT.astype(dt)

        att = block.self_attn
        for name in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"):
            _cast_lin(getattr(att, name, None))
        mlp = block.mlp
        if mlp is None:
            return
        _cast_lin(getattr(mlp, "router_gate", None))
        for name in ("gate_proj", "up_proj", "down_proj"):
            _cast_lin(getattr(mlp, name, None))
        shared = getattr(mlp, "shared_experts", None)
        if shared is not None:
            for name in ("gate_proj", "up_proj", "down_proj"):
                _cast_lin(getattr(shared, name, None))
        for ex in getattr(mlp, "experts", None) or []:
            for name in ("gate_proj", "up_proj", "down_proj"):
                _cast_lin(getattr(ex, name, None))

    def _prep_block(self, block, layer_idx: int):
        att = block.self_attn
        att.inject_kv = None
        att.graft_seats = 0
        att.live_shift = self.live_shift
        att.attention_mode = "standard"
        if self.inject_by_layer is not None and layer_idx in self.inject_by_layer:
            att.inject_kv = self.inject_by_layer[layer_idx]
            att.graft_seats = int(self.graft_seats)
        if block.moe_enabled:
            block.mlp.empty_cache_interval = 0
        self._cast_linears(block)

    def forward(
        self,
        input_ids_np: np.ndarray,
        *,
        kv_caches=None,
        position_offset: int = 0,
        last_token_only: bool = True,
        time_layers: bool = False,
    ):
        bits = self.bits
        input_ids_np = np.asarray(input_ids_np, dtype=np.int64)
        B, L = input_ids_np.shape
        shift = int(self.live_shift or 0)
        need = int(position_offset) + shift + L
        if getattr(self.rope, "_rope_len", 0) < need or self.rope.cos is None:
            # rebuild rope in compute dtype
            self.rope._rope_len = 0
            self.rope.cos = None
            self.rope.sin = None
            self.rope.extend(max(need, 64))
        else:
            self.rope.extend(need)

        h = self.embed_tokens(input_ids_np)
        if self.cfg.mup_enabled:
            h = h * (float(self.cfg.hidden_size) ** 0.5)
        new_caches = []
        layer_walls = []
        for layer_idx in range(self.cfg.num_hidden_layers):
            t0 = time.perf_counter()
            block = bits["TrinityBlockTC"].from_safetensors(
                self.cfg, self.source, layer_idx, weight_mode="bf16",
            )
            self._prep_block(block, layer_idx)
            cache = kv_caches[layer_idx] if kv_caches is not None else None
            h, kv, _route = block(
                h, self.rope.cos, self.rope.sin, int(position_offset), cache,
            )
            new_caches.append(kv)
            if kv_caches is not None:
                kv_caches[layer_idx] = None
            lw = time.perf_counter() - t0
            if time_layers:
                layer_walls.append(lw)
            del block, kv
            gc.collect()
            if hasattr(tc, "empty_cache"):
                tc.empty_cache()
        h = bits["_cast"](self.norm(h))
        if last_token_only and h.shape[1] > 1:
            h = h.slice(1, h.shape[1] - 1, 1)
        logits = self.lm_head(h)
        meta = {
            "layer_wall_sum_s": round(sum(layer_walls), 3) if layer_walls else None,
            "n_layers": self.cfg.num_hidden_layers,
        }
        return logits, new_caches, meta

    def greedy(
        self,
        prompt_ids: list[int],
        ngen: int,
        tok,
        *,
        position_offset: int = 0,
        live_shift: int | None = None,
        early_bos_stop: bool = True,
    ) -> dict[str, Any]:
        self.live_shift = live_shift
        t0 = time.perf_counter()
        step_walls = []
        with tc.no_grad():
            logits, caches, meta0 = self.forward(
                np.array([prompt_ids], dtype=np.int64),
                position_offset=int(position_offset),
                last_token_only=True,
                time_layers=True,
            )
            step_walls.append({"tag": "prompt", **meta0, "wall_s": meta0.get("layer_wall_sum_s")})
            out: list[int] = []
            for step in range(int(ngen)):
                st = time.perf_counter()
                nid = int(np.argmax(logits.float().numpy()[0, -1]))
                out.append(nid)
                if tok.eos_token_id is not None and nid == int(tok.eos_token_id):
                    step_walls.append({"tag": f"g{step}", "chosen": nid, "wall_s": round(time.perf_counter() - st, 3)})
                    break
                if early_bos_stop and out.count(0) >= 3:
                    step_walls.append({"tag": f"g{step}", "chosen": nid, "early_bos_stop": True})
                    break
                logits, caches, meta = self.forward(
                    np.array([[nid]], dtype=np.int64),
                    kv_caches=caches,
                    position_offset=int(position_offset) + len(prompt_ids) + step,
                    last_token_only=True,
                    time_layers=(step < 1),
                )
                step_walls.append({
                    "tag": f"g{step}",
                    "chosen": nid,
                    "wall_s": round(time.perf_counter() - st, 3),
                    "layer_wall_sum_s": meta.get("layer_wall_sum_s"),
                })
        text = tok.decode(out, clean_up_tokenization_spaces=False)
        txt = text
        for s in W.STOPS:
            if s in txt:
                txt = txt.split(s)[0]
        txt = txt.strip()
        return {
            "ids": out,
            "text": txt,
            "text_raw": text,
            "n_steps": len(out),
            "clean_english": W.is_clean_english(txt),
            "value_recovered": W.value_recovered(txt),
            "collapse_signature": W.collapse_signature(txt, out),
            "bos_count": int(out.count(0)),
            "unique_id_ratio": (
                round(len(set(out)) / max(1, len(out)), 4) if out else 0.0
            ),
            "wall_s": round(time.perf_counter() - t0, 3),
            "step_walls": step_walls,
            "live_shift": live_shift,
            "position_offset": position_offset,
        }


def project_stage2_wall(n_tokens_gen: int = NGEN_STAGE2, s_per_fwd: float = 33.0) -> dict[str, Any]:
    # pure: 1 prefill + n gen (gen may early-stop)
    pure_fwd = 1 + n_tokens_gen
    pure_proj = pure_fwd * s_per_fwd
    # mount arm rough: sink+fact+filler+probe prefills + n gen ≈ 4 + n
    mount_fwd = 4 + n_tokens_gen
    mount_proj = mount_fwd * s_per_fwd
    return {
        "s_per_forward_assumed": s_per_fwd,
        "ngen": n_tokens_gen,
        "pure_forwards": pure_fwd,
        "pure_proj_s": pure_proj,
        "pure_fits_600": pure_proj <= GPU_WALL_RAIL_S,
        "mount_forwards_rough": mount_fwd,
        "mount_proj_s": mount_proj,
        "mount_fits_600": mount_proj <= GPU_WALL_RAIL_S,
        "decision": (
            "launch pure natural + pure chat; mount only if pure_chat clean and mount_fits"
            if pure_proj <= GPU_WALL_RAIL_S
            else "ABORT: pure projection exceeds rail"
        ),
    }


def stage2_bf16_stream(
    out_dir: Path,
    ledger_path: Path,
    model_dir: Path,
    *,
    wall_rail_s: float,
) -> dict[str, Any]:
    stage_dir = out_dir / "stage2_bf16_stream"
    stage_dir.mkdir(parents=True, exist_ok=True)
    run_t0 = time.perf_counter()

    proj = project_stage2_wall(NGEN_STAGE2, 33.0)
    write_json(stage_dir / "projection.json", proj)
    append_ledger(ledger_path, {"stage": 2, "event": "projection", **proj})
    print(f"[S2] projection: {proj}", flush=True)
    if not proj["pure_fits_600"]:
        d = {
            "stage": 2,
            "fork": "STOP_PROJECTION",
            "floor_clean": False,
            "projection": proj,
        }
        write_json(stage_dir / "DECISION.json", d)
        return d

    tok = AutoTokenizer.from_pretrained(
        str(model_dir), trust_remote_code=True, local_files_only=True,
    )
    tmpl = extract_template_receipt(tok, model_dir)
    write_json(stage_dir / "template_receipt.json", tmpl)

    print("[S2] init bf16 stream runner compute=float32", flush=True)
    runner = Bf16StreamRunner(model_dir, compute_dtype="float32")
    append_ledger(ledger_path, {
        "stage": 2,
        "event": "runner_ready",
        "compute_dtype": "float32",
        "weight_mode": "bf16_layer_stream",
        "gpu": nvidia_smi_line(),
    })

    results = []

    # Timing pilot: 1 forward natural prefill only
    nat = "The capital of France is"
    nat_ids = tok.encode(nat, add_special_tokens=False)
    print(f"[S2] timing pilot prefill L={len(nat_ids)}", flush=True)
    t_pilot = time.perf_counter()
    with tc.no_grad():
        logits, caches, meta = runner.forward(
            np.array([nat_ids], dtype=np.int64),
            last_token_only=True,
            time_layers=True,
        )
    pilot_s = time.perf_counter() - t_pilot
    pilot = {
        "prefill_wall_s": round(pilot_s, 3),
        "layer_wall_sum_s": meta.get("layer_wall_sum_s"),
        "ntok": len(nat_ids),
        "gpu": nvidia_smi_line(),
    }
    write_json(stage_dir / "timing_pilot.json", pilot)
    append_ledger(ledger_path, {"stage": 2, "event": "timing_pilot", **pilot})
    print(f"[S2] pilot prefill_s={pilot_s:.1f} layer_sum={meta.get('layer_wall_sum_s')}", flush=True)
    del logits, caches
    gc.collect()
    if hasattr(tc, "empty_cache"):
        tc.empty_cache()

    # Re-project with measured s/fwd
    s_meas = max(float(pilot_s), 1.0)
    proj2 = project_stage2_wall(NGEN_STAGE2, s_meas)
    proj2["measured_from"] = "natural_prefill_L5"
    write_json(stage_dir / "projection_measured.json", proj2)
    append_ledger(ledger_path, {"stage": 2, "event": "projection_measured", **proj2})
    if not proj2["pure_fits_600"]:
        # reduce ngen to fit: ngen = floor((600 - s_meas)/s_meas)
        ngen_fit = max(4, int((wall_rail_s * 0.9 - s_meas) / s_meas))
        ngen_use = min(NGEN_STAGE2, ngen_fit)
        append_ledger(ledger_path, {
            "stage": 2,
            "event": "ngen_reduced_to_fit_rail",
            "ngen_use": ngen_use,
            "s_meas": s_meas,
        })
    else:
        ngen_use = NGEN_STAGE2

    # Pure natural
    print(f"[S2] pure_natural ngen={ngen_use}", flush=True)
    gen_nat = runner.greedy(nat_ids, ngen_use, tok, live_shift=None)
    row_nat = {
        "label": "pure_natural_bf16_stream",
        "mode": "bf16_stream_fp32_compute",
        "probe_prompt": nat,
        "probe_prompt_ntok": len(nat_ids),
        "generation": gen_nat,
        "wall_s": gen_nat["wall_s"],
    }
    results.append(row_nat)
    write_json(stage_dir / "case_pure_natural.json", row_nat)
    print(
        f"[S2]   natural sig={gen_nat['collapse_signature']} clean={gen_nat['clean_english']} "
        f"text={gen_nat['text']!r} wall={gen_nat['wall_s']}",
        flush=True,
    )
    append_ledger(ledger_path, {
        "stage": 2,
        "event": "pure_natural",
        "sig": gen_nat["collapse_signature"],
        "clean": gen_nat["clean_english"],
        "text": gen_nat["text"],
        "ids": gen_nat["ids"],
        "wall_s": gen_nat["wall_s"],
        "step_walls_head": (gen_nat.get("step_walls") or [])[:3],
    })

    elapsed = time.perf_counter() - run_t0
    # Pure chat if wall remains for another full pure
    pure_chat_budget = (1 + ngen_use) * s_meas
    do_chat = (elapsed + pure_chat_budget) <= wall_rail_s
    row_chat = None
    if do_chat:
        chat_prompt = tmpl["rendered"]["full_sys_probe_add_gen"]["text"]
        chat_ids = tok.encode(chat_prompt, add_special_tokens=False)
        print(f"[S2] pure_chat ntok={len(chat_ids)} ngen={ngen_use}", flush=True)
        gen_chat = runner.greedy(chat_ids, ngen_use, tok, live_shift=None)
        row_chat = {
            "label": "pure_chat_bf16_stream",
            "mode": "bf16_stream_fp32_compute",
            "probe_prompt": chat_prompt,
            "probe_prompt_ntok": len(chat_ids),
            "generation": gen_chat,
            "wall_s": gen_chat["wall_s"],
        }
        results.append(row_chat)
        write_json(stage_dir / "case_pure_chat.json", row_chat)
        print(
            f"[S2]   chat sig={gen_chat['collapse_signature']} clean={gen_chat['clean_english']} "
            f"text={gen_chat['text']!r} wall={gen_chat['wall_s']}",
            flush=True,
        )
        append_ledger(ledger_path, {
            "stage": 2,
            "event": "pure_chat",
            "sig": gen_chat["collapse_signature"],
            "clean": gen_chat["clean_english"],
            "text": gen_chat["text"],
            "ids": gen_chat["ids"],
            "wall_s": gen_chat["wall_s"],
        })
    else:
        append_ledger(ledger_path, {
            "stage": 2,
            "event": "pure_chat_skipped_wall",
            "elapsed_s": elapsed,
            "pure_chat_budget_s": pure_chat_budget,
        })

    nat_clean = bool(gen_nat["clean_english"])
    chat_clean = bool(row_chat["generation"]["clean_english"]) if row_chat else False
    chat_sig = row_chat["generation"]["collapse_signature"] if row_chat else None
    floor_clean = nat_clean or chat_clean

    # Mount arm only if chat floor clean and projection fits remaining wall
    mount_row = None
    elapsed = time.perf_counter() - run_t0
    mount_budget = (4 + ngen_use) * s_meas
    if chat_clean and (elapsed + mount_budget) <= wall_rail_s and proj2.get("mount_fits_600"):
        print("[S2] w96_mount under bf16 stream — building manual plant", flush=True)
        # Manual plant without full ArenaCache: not equivalent for graft seats;
        # if pure chat is clean, Stage 3 uses stream arena. Here we only need floor.
        append_ledger(ledger_path, {
            "stage": 2,
            "event": "mount_skipped_use_stage3",
            "reason": "floor established on pure; mount deferred to Stage 3 stream/resident path",
        })
    else:
        append_ledger(ledger_path, {
            "stage": 2,
            "event": "mount_not_run",
            "chat_clean": chat_clean,
            "elapsed_s": elapsed,
            "mount_budget_s": mount_budget,
        })

    if floor_clean:
        fork = "PROCEED_STAGE3_BF16_STREAM" if not chat_clean and nat_clean else (
            "PROCEED_STAGE3_BF16_STREAM" if chat_clean else "PROCEED_STAGE3_BF16_STREAM"
        )
        # If only natural clean, chat probe path still broken → floor for T1 chat probes NOT clean
        if chat_clean:
            fork = "PROCEED_STAGE3_BF16_STREAM"
            residual = "INT4 free-gen residual confirmed (bf16 stream chat clean)"
        elif nat_clean and row_chat is not None and not chat_clean:
            fork = "STOP_CHAT_PATH_BROKEN_EVEN_BF16"
            residual = (
                "bf16 natural clean but chat still degraded — not INT4-only; "
                "chat free-gen residual (template path or model chat behavior)"
            )
            floor_clean = False  # T1 uses chat probe path
        elif nat_clean and row_chat is None:
            fork = "PARTIAL_NATURAL_ONLY_CHAT_UNTESTED"
            residual = "natural clean; chat not run (wall) — chat floor unclassed"
            floor_clean = False
        else:
            residual = "unexpected"
    else:
        # both broken or natural broken
        if row_chat is not None and not chat_clean and not nat_clean:
            fork = "STOP_PURE_BROKEN_BF16"
            residual = (
                "pure free-gen broken under bf16 stream on natural AND chat — "
                "not explained by INT4 alone; not arena adapter (pure path)"
            )
        elif not nat_clean:
            fork = "STOP_NATURAL_BROKEN_BF16"
            residual = "natural free-gen broken under bf16 stream (unexpected vs T4 parity)"
        else:
            fork = "STOP_FLOOR_UNRESOLVED"
            residual = "floor unresolved"

    decision = {
        "stage": 2,
        "fork": fork,
        "floor_clean": floor_clean,
        "floor_mode": "bf16_stream_fp32_compute" if floor_clean else None,
        "pure_natural_clean": nat_clean,
        "pure_natural_sig": gen_nat["collapse_signature"],
        "pure_natural_text": gen_nat["text"],
        "pure_chat_clean": chat_clean if row_chat else None,
        "pure_chat_sig": chat_sig,
        "pure_chat_text": row_chat["generation"]["text"] if row_chat else None,
        "residual_read": residual,
        "ngen_use": ngen_use,
        "s_per_fwd_measured": s_meas,
        "total_wall_s": round(time.perf_counter() - run_t0, 3),
        "gpu_after": nvidia_smi_line(),
        "stage_dir": str(stage_dir),
    }
    write_json(stage_dir / "DECISION.json", decision)
    write_json(stage_dir / "results.json", {"results": results, "decision": decision})
    append_ledger(ledger_path, {"stage": 2, "event": "decision", **decision})
    print(f"[S2] DECISION fork={fork} floor_clean={floor_clean} wall={decision['total_wall_s']}s", flush=True)

    runner.close()
    del runner
    gc.collect()
    if hasattr(tc, "empty_cache"):
        tc.empty_cache()
    return decision


# ---------------------------------------------------------------------------
# Stage 3 — T1 verdict 96 vs 768
# ---------------------------------------------------------------------------

def stage3_verdict_int4(
    out_dir: Path,
    ledger_path: Path,
    model_dir: Path,
    *,
    wall_rail_s: float,
    widths: tuple[int, ...] = (96, 768),
) -> dict[str, Any]:
    stage_dir = out_dir / "stage3_verdict_int4"
    stage_dir.mkdir(parents=True, exist_ok=True)
    run_t0 = time.perf_counter()

    tok = AutoTokenizer.from_pretrained(
        str(model_dir), trust_remote_code=True, local_files_only=True,
    )
    sink_text = W.build_sink_text(tok)
    print(f"[S3-int4] loading model", flush=True)
    model, model_info, load_s, dtype_meta = W.load_model(model_dir)
    adapter = W.TrinityArenaModelAdapter(model)
    adapter.configure_moe_empty_cache(0)

    results = []
    # pure baseline for floor reconfirm
    pure_prompt = tok.apply_chat_template(
        [
            {"role": "system", "content": W.SYS_CONTENT},
            {"role": "user", "content": W.PROBE_USER},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    pure_ids = tok.encode(pure_prompt, add_special_tokens=False)
    pure = pure_generate_resident(
        adapter, tok, pure_ids, NGEN_STAGE3_INT4, label="pure_chat_baseline",
    )
    pure["probe_prompt"] = pure_prompt
    pure["mode"] = "int4_fp32_resident"
    results.append(pure)
    write_json(stage_dir / "case_pure_chat_baseline.json", pure)

    for w in widths:
        for mount in (True, False):
            elapsed = time.perf_counter() - run_t0
            if elapsed > wall_rail_s:
                append_ledger(ledger_path, {
                    "stage": 3, "event": "wall_abort", "elapsed_s": elapsed,
                    "next": f"w{w}_{'mount' if mount else 'control'}",
                })
                break
            print(f"[S3-int4] w={w} mount={mount}", flush=True)
            with tc.no_grad():
                row = W.run_width_case(
                    adapter, tok, width=int(w), sink_text=sink_text,
                    mount=bool(mount), ngen=NGEN_STAGE3_INT4,
                )
            row["mode"] = "int4_fp32_resident"
            results.append(row)
            write_json(stage_dir / f"case_{row['label']}.json", row)
            g = row.get("generation") or {}
            print(
                f"[S3]   shift={row.get('live_shift')} clean={g.get('clean_english')} "
                f"value={g.get('value_recovered')} sig={g.get('collapse_signature')} "
                f"text={g.get('text')!r}",
                flush=True,
            )
            append_ledger(ledger_path, {
                "stage": 3,
                "event": "case",
                "label": row["label"],
                "live_shift": row.get("live_shift"),
                "sig": g.get("collapse_signature"),
                "clean": g.get("clean_english"),
                "value": g.get("value_recovered"),
                "text": g.get("text"),
            })
        else:
            continue
        break

    verdict = t1_verdict_strict(results)
    write_json(stage_dir / "VERDICT.json", verdict)
    write_json(stage_dir / "results.json", {"results": results, "verdict": verdict})
    append_ledger(ledger_path, {"stage": 3, "event": "verdict", **verdict})

    del adapter, model
    gc.collect()
    if hasattr(tc, "empty_cache"):
        tc.empty_cache()
    verdict["total_wall_s"] = round(time.perf_counter() - run_t0, 3)
    verdict["stage_dir"] = str(stage_dir)
    write_json(stage_dir / "VERDICT.json", verdict)
    return verdict


def stage3_verdict_bf16_stream(
    out_dir: Path,
    ledger_path: Path,
    model_dir: Path,
    *,
    wall_rail_s: float,
    widths: tuple[int, ...] = (96, 768),
    ngen: int = NGEN_STAGE3_STREAM,
) -> dict[str, Any]:
    """Stream-mode width cases: NO full arena (VRAM). 

    Honest limitation: without resident KV for all layers simultaneously with
    graft swap, we cannot fully exercise arena mount under pure stream.
    We approximate T1 by:
      - pure chat free-gen floor (must stay clean)
      - synthetic live_shift position_offset = n_sink + width on pure chat gen
        (NoPE full layers ignore RoPE; sliding sees shifted positions — partial
        hole-law stress without actual graft content).
      - For value recovery, we require a resident path; if only stream is
        available, value recovery is UNTESTED and reported as such.
    """
    stage_dir = out_dir / "stage3_verdict_bf16_stream"
    stage_dir.mkdir(parents=True, exist_ok=True)
    run_t0 = time.perf_counter()

    tok = AutoTokenizer.from_pretrained(
        str(model_dir), trust_remote_code=True, local_files_only=True,
    )
    tmpl = extract_template_receipt(tok, model_dir)
    sink_ntok = tmpl["rendered"]["sink"]["n_tokens"]
    chat_prompt = tmpl["rendered"]["full_sys_probe_add_gen"]["text"]
    chat_ids = tok.encode(chat_prompt, add_special_tokens=False)

    runner = Bf16StreamRunner(model_dir, compute_dtype="float32")
    results = []

    # Floor reconfirm
    print(f"[S3-bf16] pure_chat floor ngen={ngen}", flush=True)
    gen = runner.greedy(chat_ids, ngen, tok, live_shift=None)
    pure = {
        "label": "pure_chat_baseline",
        "mode": "bf16_stream_fp32_compute",
        "probe_prompt": chat_prompt,
        "generation": gen,
        "live_shift": 0,
        "mount": False,
        "width": None,
        "wall_s": gen["wall_s"],
    }
    results.append(pure)
    write_json(stage_dir / "case_pure_chat_baseline.json", pure)
    append_ledger(ledger_path, {
        "stage": 3, "event": "pure_floor", "clean": gen["clean_english"],
        "sig": gen["collapse_signature"], "text": gen["text"],
    })

    if not gen["clean_english"]:
        decision = {
            "verdict": "T1_FLOOR_UNRESOLVED",
            "reason": "Stage3 bf16 stream pure chat floor not clean at start of verdict run",
            "pure_text": gen["text"],
            "pure_sig": gen["collapse_signature"],
        }
        write_json(stage_dir / "VERDICT.json", decision)
        runner.close()
        return decision

    for w in widths:
        for mount in (True, False):
            elapsed = time.perf_counter() - run_t0
            # each case ~ (1+ngen)*s; abort if remaining wall insufficient
            if elapsed > wall_rail_s * 0.95:
                append_ledger(ledger_path, {
                    "stage": 3, "event": "wall_abort", "elapsed_s": elapsed,
                })
                break
            # Hole-law stress is live_shift magnitude, not mount bit. Both arms
            # get the same shift; graft content is NOT seated under stream.
            live_shift = int(sink_ntok + w)
            print(
                f"[S3-bf16] w={w} mount={mount} live_shift={live_shift} "
                f"(stream live_shift stress; graft content NOT seated)",
                flush=True,
            )
            gen = runner.greedy(chat_ids, ngen, tok, live_shift=live_shift)
            gen["value_recovered"] = False
            row = {
                "label": f"w{w}_{'mount' if mount else 'control'}",
                "width": int(w),
                "mount": bool(mount),
                "live_shift": live_shift,
                "mode": "bf16_stream_fp32_compute_live_shift_stress",
                "graft_content_seated": False,
                "honest_limitation": (
                    "Stream path cannot seat arena graft KV; live_shift applied to "
                    "attention RoPE offset only. Value recovery UNCLASSABLE here."
                ),
                "probe_prompt": chat_prompt,
                "generation": gen,
                "wall_s": gen["wall_s"],
            }
            results.append(row)
            write_json(stage_dir / f"case_{row['label']}.json", row)
            print(
                f"[S3]   clean={gen['clean_english']} sig={gen['collapse_signature']} "
                f"text={gen['text']!r} wall={gen['wall_s']}",
                flush=True,
            )
            append_ledger(ledger_path, {
                "stage": 3,
                "event": "case",
                "label": row["label"],
                "live_shift": live_shift,
                "clean": gen["clean_english"],
                "sig": gen["collapse_signature"],
                "text": gen["text"],
                "graft_content_seated": False,
            })
        else:
            continue
        break

    verdict = t1_verdict_strict(results, stream_no_graft=True)
    verdict["total_wall_s"] = round(time.perf_counter() - run_t0, 3)
    verdict["stage_dir"] = str(stage_dir)
    write_json(stage_dir / "VERDICT.json", verdict)
    write_json(stage_dir / "results.json", {"results": results, "verdict": verdict})
    append_ledger(ledger_path, {"stage": 3, "event": "verdict", **verdict})
    runner.close()
    return verdict


def t1_verdict_strict(
    results: list[dict[str, Any]],
    *,
    stream_no_graft: bool = False,
) -> dict[str, Any]:
    """Strict T1 labels per plan.

    T1_CONFIRMED: clean floor both widths + value at mounted 768
    T1_CLEAN_VALUE_MISS: clean both widths, value miss both
    T1_RELATIVE_HOLD: no width-worsened collapse, absolute clean unavailable
    T1_REFUTED: large width worsens class vs w96
    """
    by = {r["label"]: r for r in results}
    pure = by.get("pure_chat_baseline") or by.get("pure_baseline")
    pure_clean = False
    pure_sig = None
    pure_text = None
    if pure and "generation" in pure:
        pure_clean = bool(pure["generation"].get("clean_english"))
        pure_sig = pure["generation"].get("collapse_signature")
        pure_text = pure["generation"].get("text")

    decisive = []
    for w in (96, 768):
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
            "control_text": None if c is None else (c.get("generation") or {}).get("text"),
            "control_clean": None if c is None else (c.get("generation") or {}).get("clean_english"),
            "control_sig": None if c is None else (c.get("generation") or {}).get("collapse_signature"),
            "graft_content_seated": m.get("graft_content_seated", True),
        })

    if len(decisive) < 2:
        return {
            "verdict": "INCOMPLETE",
            "reason": "need w96 and w768 mount results",
            "decisive": decisive,
            "pure_clean": pure_clean,
            "pure_sig": pure_sig,
            "pure_text": pure_text,
        }

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
    w96 = next(d for d in decisive if d["width"] == 96)
    w768 = next(d for d in decisive if d["width"] == 768)
    r96 = worse_than.get(w96["mount_sig"] or "other_degraded", 2)
    r768 = worse_than.get(w768["mount_sig"] or "other_degraded", 2)
    width_worsened = r768 > r96
    both_clean = bool(w96["mount_clean"] and w768["mount_clean"])
    value_768 = bool(w768["mount_value"])
    value_any = bool(w96["mount_value"] or w768["mount_value"])

    if stream_no_graft:
        # Cannot confirm value; clean-at-both-shifts is a PARTIAL on hole law only
        if both_clean and pure_clean:
            verdict = "T1_CLEAN_SHIFTS_VALUE_UNCLASSABLE"
            reason = (
                "Clean English free-gen at live_shift≈n_sink+{96,768} under bf16 "
                "stream (graft content NOT seated — value recovery unclassable). "
                "Hole-law width salad not observed. Not full T1_CONFIRMED."
            )
        elif width_worsened:
            verdict = "T1_REFUTED"
            reason = (
                f"Width worsens collapse w96 sig={w96['mount_sig']!r} → "
                f"w768 sig={w768['mount_sig']!r} under live_shift stress. "
                f"Quotes: {w96['mount_text']!r} | {w768['mount_text']!r}"
            )
        else:
            verdict = "T1_RELATIVE_HOLD"
            reason = (
                "No width-worsened collapse under stream live_shift stress; "
                f"absolute clean unavailable (w96_clean={w96['mount_clean']}, "
                f"w768_clean={w768['mount_clean']}, pure_clean={pure_clean}). "
                f"Quotes: {w96['mount_text']!r} | {w768['mount_text']!r}"
            )
    else:
        if both_clean and pure_clean and value_768:
            verdict = "T1_CONFIRMED"
            reason = (
                "Clean floor at w96 and w768; value recovered on mounted graft at "
                f"w768 (live_shift={w768['live_shift']}). GPT-OSS hole law does not "
                "apply on Trinity NoPE full layers under this probe."
            )
        elif both_clean and pure_clean and not value_any:
            verdict = "T1_CLEAN_VALUE_MISS"
            reason = (
                "Clean English at both widths under mount; value recovered at NEITHER "
                "width — residual is route/readout/content, not width-hole collapse."
            )
        elif both_clean and pure_clean and value_any and not value_768:
            verdict = "T1_CLEAN_VALUE_MISS"
            reason = (
                f"Clean both widths; value hit at w96 only (not 768). "
                "Not full T1_CONFIRMED; not hole-law refutation."
            )
        elif width_worsened:
            verdict = "T1_REFUTED"
            reason = (
                f"Large width worsens generation vs w96: "
                f"{w96['mount_sig']!r}@{w96['live_shift']} → "
                f"{w768['mount_sig']!r}@{w768['live_shift']}. "
                f"Quotes: {w96['mount_text']!r} | {w768['mount_text']!r}"
            )
        else:
            verdict = "T1_RELATIVE_HOLD"
            reason = (
                "No additional width-dependent collapse at w768 vs w96; absolute "
                f"clean_english unavailable (pure_clean={pure_clean}, "
                f"w96_clean={w96['mount_clean']}, w768_clean={w768['mount_clean']}). "
                f"Quotes: pure={pure_text!r} | w96={w96['mount_text']!r} | "
                f"w768={w768['mount_text']!r}"
            )

    return {
        "verdict": verdict,
        "reason": reason,
        "decisive": decisive,
        "pure_clean": pure_clean,
        "pure_sig": pure_sig,
        "pure_text": pure_text,
        "both_widths_clean": both_clean,
        "value_at_768_mount": value_768,
        "width_worsened": width_worsened,
        "stream_no_graft": stream_no_graft,
        "gpt_oss_law_reference": {
            "sys_w96_live_shift": 115,
            "sess_w384_live_shift": 387,
            "sess_w384_answer": "User: 3-V> < 3-4> <5 6-7-",
        },
        "t1_prediction_source": (
            "/mnt/ForgeRealm/Project-Tensor/docs/TRINITY_NANO_PORT_PLAN.md "
            "prediction T1 (frozen; not modified this order)"
        ),
    }


def write_summary(out_dir: Path, s1, s2, s3) -> None:
    lines = [
        "# T1 Floor Re-Verdict — SUMMARY",
        "",
        f"Out: `{out_dir}`",
        "",
        "## Stage decisions",
        f"- Stage1: fork={(s1 or {}).get('fork')} floor_clean={(s1 or {}).get('floor_clean')}",
        f"  pure_chat={(s1 or {}).get('pure_chat_sig')!r} text={(s1 or {}).get('pure_chat_text')!r}",
        f"  pure_natural={(s1 or {}).get('pure_natural_sig')!r} text={(s1 or {}).get('pure_natural_text')!r}",
        f"  w96_mount={(s1 or {}).get('w96_mount_sig')!r} text={(s1 or {}).get('w96_mount_text')!r}",
    ]
    if s2:
        lines.append(
            f"- Stage2: fork={s2.get('fork')} floor_clean={s2.get('floor_clean')} "
            f"residual={s2.get('residual_read')!r}"
        )
        lines.append(
            f"  natural={s2.get('pure_natural_sig')!r} text={s2.get('pure_natural_text')!r}"
        )
        lines.append(
            f"  chat={s2.get('pure_chat_sig')!r} text={s2.get('pure_chat_text')!r}"
        )
    if s3:
        lines.append(f"- Stage3 verdict: **{s3.get('verdict')}**")
        lines.append(f"  reason: {s3.get('reason')}")
        lines.append("")
        lines.append("| width | shift | mount_clean | value | sig | text |")
        lines.append("|---:|---:|---|---|---|---|")
        for d in s3.get("decisive") or []:
            t = (d.get("mount_text") or "")[:60].replace("|", "\\|")
            lines.append(
                f"| {d['width']} | {d.get('live_shift')} | {d.get('mount_clean')} | "
                f"{d.get('mount_value')} | {d.get('mount_sig')} | {t!r} |"
            )
    lines.append("")
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--wall-rail-s", type=float, default=GPU_WALL_RAIL_S)
    p.add_argument(
        "--only-stage",
        type=int,
        choices=(1, 2, 3),
        default=None,
        help="Run a single stage (for resume / wall budgeting)",
    )
    p.add_argument(
        "--stage3-mode",
        choices=("auto", "int4", "bf16_stream"),
        default="auto",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else (OUT_ROOT / f"floor_reverb_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = out_dir / "LEDGER.json"
    write_json(out_dir / "run_header.json", {
        "experiment": "T1_floor_reverb",
        "stamp": stamp,
        "model_dir": str(args.model_dir),
        "wall_rail_s": args.wall_rail_s,
        "plan": str(OUT_ROOT / "T1_FLOOR_REVERDICT_IMPLEMENTATION_PLAN.md"),
        "gpu_before": nvidia_smi_line(),
    })
    append_ledger(ledger_path, {"stage": 0, "event": "start", "out_dir": str(out_dir)})

    s1 = s2 = s3 = None

    if args.only_stage in (None, 1):
        s1 = stage1_int4(out_dir, ledger_path, args.model_dir, wall_rail_s=args.wall_rail_s)
        write_json(out_dir / "stage1_decision.json", s1)
        if args.only_stage == 1:
            write_summary(out_dir, s1, s2, s3)
            return 0
        if s1.get("fork") == "PROCEED_STAGE3_INT4":
            s3 = stage3_verdict_int4(
                out_dir, ledger_path, args.model_dir, wall_rail_s=args.wall_rail_s,
            )
            write_json(out_dir / "FINAL_VERDICT.json", s3)
            write_summary(out_dir, s1, s2, s3)
            return 0
        # else fall through to stage 2

    if args.only_stage in (None, 2) and (s1 is None or s1.get("fork") == "PROCEED_STAGE2_BF16_STREAM"):
        s2 = stage2_bf16_stream(
            out_dir, ledger_path, args.model_dir, wall_rail_s=args.wall_rail_s,
        )
        write_json(out_dir / "stage2_decision.json", s2)
        if args.only_stage == 2:
            write_summary(out_dir, s1, s2, s3)
            return 0
        if s2.get("fork") == "PROCEED_STAGE3_BF16_STREAM":
            s3 = stage3_verdict_bf16_stream(
                out_dir, ledger_path, args.model_dir, wall_rail_s=args.wall_rail_s,
            )
            write_json(out_dir / "FINAL_VERDICT.json", s3)
            write_summary(out_dir, s1, s2, s3)
            return 0
        if s2.get("fork") == "PROCEED_STAGE3_INT4":
            s3 = stage3_verdict_int4(
                out_dir, ledger_path, args.model_dir, wall_rail_s=args.wall_rail_s,
            )
            write_json(out_dir / "FINAL_VERDICT.json", s3)
            write_summary(out_dir, s1, s2, s3)
            return 0
        # STOP forks
        final = {
            "verdict": "T1_FLOOR_UNRESOLVED",
            "reason": s2.get("residual_read") or s2.get("fork"),
            "stage1": s1,
            "stage2": s2,
        }
        write_json(out_dir / "FINAL_VERDICT.json", final)
        write_summary(out_dir, s1, s2, s3)
        print(f"[FINAL] {final['verdict']}: {final['reason']}", flush=True)
        return 0

    if args.only_stage == 3:
        mode = args.stage3_mode
        if mode == "auto":
            # prefer int4 if stage1 decision on disk says clean
            d1 = out_dir / "stage1_decision.json"
            d2 = out_dir / "stage2_decision.json"
            if d1.exists() and json.loads(d1.read_text()).get("floor_clean"):
                mode = "int4"
            elif d2.exists() and json.loads(d2.read_text()).get("floor_clean"):
                mode = "bf16_stream"
            else:
                mode = "int4"
        if mode == "int4":
            s3 = stage3_verdict_int4(
                out_dir, ledger_path, args.model_dir, wall_rail_s=args.wall_rail_s,
            )
        else:
            s3 = stage3_verdict_bf16_stream(
                out_dir, ledger_path, args.model_dir, wall_rail_s=args.wall_rail_s,
            )
        write_json(out_dir / "FINAL_VERDICT.json", s3)
        write_summary(out_dir, s1, s2, s3)
        return 0

    write_summary(out_dir, s1, s2, s3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
