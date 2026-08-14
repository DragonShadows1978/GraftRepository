#!/usr/bin/env python3
"""APAMQ-FC engaged-scoring perplexity arms (G-C).

One GPU-bearing arm is run per process::

    python3 scripts/apamq_fc_ppl.py --arm standard --output artifacts/apamq_fc/standard.json
    python3 scripts/apamq_fc_ppl.py --arm apa_blend --output artifacts/apamq_fc/apa_blend.json
    python3 scripts/apamq_fc_ppl.py --arm apa_fused --output artifacts/apamq_fc/apa_fused.json
    TENSOR_CUDA_ROOT=/mnt/ForgeRealm/wt/apamq-fa/tensor_cuda \
      python3 scripts/apamq_fc_ppl.py --arm apa_int4 --output artifacts/apamq_fc/apa_int4.json
    python3 scripts/apamq_fc_ppl.py --arm apa_gemm --output artifacts/apamq_fc/apa_gemm.json
    python3 scripts/apamq_fc_ppl.py --arm summary --output artifacts/apamq_fc/summary.json \
      --legacy-receipts-dir /mnt/ForgeRealm/wt/apamq-fb/artifacts/apamq_fc

The scoring corpus is the locally cached public WikiText-2 raw-v1 test split.
It is read directly from its Arrow file: ``datasets.load_dataset`` attempts to
write a cache lock even in offline mode, which is incompatible with read-only
model/corpus mounts.  The full corpus and selected token stream are hash-bound
in every receipt.

Every scored target is produced by a true ``(1, 1)`` int64 teacher-forced
decode call after an 8192-token prefill.  The first decode input is token 8192
and its logits score token 8193, so a 2048-target document consumes 10241
tokens.  No chunked forward contributes an NLL.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback
from typing import Any, Callable, Iterable

import numpy as np


REPO = Path(__file__).resolve().parents[1]
CANONICAL_TC_ROOT = Path("/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
FA_TC_ROOT = Path("/mnt/ForgeRealm/wt/apamq-fa/tensor_cuda")
MODEL_DIR = Path("/mnt/ForgeRealm/models/gemma-4-12B-it")

LEGACY_MEASUREMENT_ARMS = ("standard", "apa_blend", "apa_fused", "apa_int4")
MEASUREMENT_ARMS = (*LEGACY_MEASUREMENT_ARMS, "apa_gemm")
ALL_ARMS = (*MEASUREMENT_ARMS, "summary", "cpu-check")
GLOBAL_LAYERS = tuple(range(5, 48, 6))
SEED = 20260813
PREFILL_TOKENS = 8192
SCORED_TOKENS_PER_DOCUMENT = 2048
TOKENS_PER_DOCUMENT = PREFILL_TOKENS + 1 + SCORED_TOKENS_PER_DOCUMENT
DOCUMENT_COUNT = 25
EXPECTED_SCORED_TOKENS = DOCUMENT_COUNT * SCORED_TOKENS_PER_DOCUMENT
EXPECTED_APA_DECODE_CALLS = EXPECTED_SCORED_TOKENS * len(GLOBAL_LAYERS)
REFINE_PERCENTILE = 0.15
BULK_BITS = 4
APA_MIN_CONTEXT = 2048
FAST_MAX_SEQ = 4096
RUNTIME_BUDGET_S = 45 * 60
SCHEMA = "apamq-fc-ppl-v1"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def resolve_tc_root(arm: str | None, environ: dict[str, str] | None = None) -> Path:
    env = os.environ if environ is None else environ
    default = FA_TC_ROOT if arm == "apa_int4" else CANONICAL_TC_ROOT
    return Path(env.get("TENSOR_CUDA_ROOT", str(default))).expanduser().resolve()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def canonical_hash(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")
    return sha256_bytes(data)


def current_code_identity() -> dict[str, str]:
    return {
        "harness_sha256": sha256_file(Path(__file__).resolve()),
        "gemma4_tc_sha256": sha256_file(REPO / "core" / "gemma4_tc.py"),
        "mistral7b_tc_sha256": sha256_file(REPO / "core" / "mistral7b_tc.py"),
    }


def discover_wikitext_arrow() -> Path:
    explicit = os.environ.get("APAMQ_FC_WIKITEXT_ARROW")
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"APAMQ_FC_WIKITEXT_ARROW not found: {path}")
        return path

    cache = Path(os.environ.get(
        "HF_DATASETS_CACHE", str(Path.home() / ".cache/huggingface/datasets")
    )).expanduser()
    matches = sorted(cache.glob(
        "wikitext/wikitext-2-raw-v1/**/wikitext-test.arrow"))
    if len(matches) != 1:
        raise FileNotFoundError(
            "expected exactly one cached WikiText-2 raw-v1 test Arrow file; "
            f"found {len(matches)} below {cache}. Set APAMQ_FC_WIKITEXT_ARROW."
        )
    return matches[0].resolve()


def load_corpus_tokens() -> tuple[np.ndarray, dict[str, Any]]:
    """Load and tokenize the fixed public corpus without a network/cache write."""
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from datasets import Dataset
    from transformers import AutoTokenizer

    arrow = discover_wikitext_arrow()
    dataset = Dataset.from_file(str(arrow))
    raw_text = "\n\n".join(dataset["text"])
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR), local_files_only=True)
    ids = np.asarray(tokenizer(
        raw_text, add_special_tokens=False)["input_ids"], dtype=np.int64)
    if ids.size < 200_000:
        raise AssertionError(
            f"public corpus has {ids.size} tokens; protocol requires >=200000")
    tokenizer_path = MODEL_DIR / "tokenizer.json"
    meta = {
        "name": "WikiText-2 raw-v1 test",
        "source": "wikitext/wikitext-2-raw-v1 split=test (cached Arrow)",
        "dataset_rows": int(len(dataset)),
        "join": "\\n\\n between dataset rows",
        "add_special_tokens": False,
        "raw_utf8_bytes": len(raw_text.encode("utf-8")),
        "raw_text_sha256": sha256_bytes(raw_text.encode("utf-8")),
        "full_token_count": int(ids.size),
        "full_token_ids_int64_sha256": sha256_bytes(
            np.ascontiguousarray(ids).tobytes()),
        "tokenizer_dir": str(MODEL_DIR),
        "tokenizer_json_sha256": sha256_file(tokenizer_path),
        "arrow_path": str(arrow),
        "arrow_sha256": sha256_file(arrow),
    }
    return ids, meta


def select_documents(token_ids: np.ndarray) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Select 25 seeded non-overlapping fixed token blocks."""
    ids = np.ascontiguousarray(token_ids, dtype=np.int64)
    candidate_count = int(ids.size // TOKENS_PER_DOCUMENT)
    if candidate_count < DOCUMENT_COUNT:
        raise AssertionError(
            f"only {candidate_count} full {TOKENS_PER_DOCUMENT}-token blocks; "
            f"need {DOCUMENT_COUNT}")
    rng = np.random.default_rng(SEED)
    selected = [int(x) for x in rng.permutation(candidate_count)[:DOCUMENT_COUNT]]
    documents = [np.ascontiguousarray(
        ids[i * TOKENS_PER_DOCUMENT:(i + 1) * TOKENS_PER_DOCUMENT]
    ) for i in selected]
    records = [{
        "document_index": order,
        "source_block_index": block,
        "source_token_start": block * TOKENS_PER_DOCUMENT,
        "source_token_end_exclusive": (block + 1) * TOKENS_PER_DOCUMENT,
        "token_count": TOKENS_PER_DOCUMENT,
        "token_ids_int64_sha256": sha256_bytes(doc.tobytes()),
    } for order, (block, doc) in enumerate(zip(selected, documents))]
    stream_hasher = hashlib.sha256()
    for doc in documents:
        stream_hasher.update(doc.tobytes())
    selection = {
        "seed": SEED,
        "selection": "first 25 entries of seeded permutation of aligned, "
                     "non-overlapping 10241-token blocks",
        "candidate_block_count": candidate_count,
        "document_count": DOCUMENT_COUNT,
        "tokens_per_document": TOKENS_PER_DOCUMENT,
        "prefill_tokens_per_document": PREFILL_TOKENS,
        "unscored_first_decode_inputs_per_document": 1,
        "scored_tokens_per_document": SCORED_TOKENS_PER_DOCUMENT,
        "selected_token_count": DOCUMENT_COUNT * TOKENS_PER_DOCUMENT,
        "scored_token_count": EXPECTED_SCORED_TOKENS,
        "selected_stream_int64_sha256": stream_hasher.hexdigest(),
        "documents": records,
    }
    return documents, selection


def protocol_payload(corpus: dict[str, Any], selection: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "corpus": corpus,
        "selection": selection,
        "teacher_forcing": {
            "prefill_shape": [1, PREFILL_TOKENS],
            "decode_shape": [1, 1],
            "input_dtype": "int64",
            "nll_logits_source": "L=1 decode output only",
            "target_at_step": "document[8193 + step] from input document[8192 + step]",
        },
        "model": "google/gemma-4-12B-it local QAT port",
        "raw_text_absolute_ppl_warning": (
            "instruction-tuned/template-bound model scored on raw text; only "
            "same-stream relative arm deltas are adjudication material"),
    }
    payload["fingerprint_sha256"] = canonical_hash(payload)
    return payload


def arm_config(arm: str) -> dict[str, Any]:
    if arm not in MEASUREMENT_ARMS:
        raise ValueError(arm)
    is_apa = arm != "standard"
    return {
        "attention_mode": "apa_selective" if is_apa else "standard",
        "refine_percentile": REFINE_PERCENTILE if is_apa else None,
        "bulk_bits": BULK_BITS if is_apa else None,
        "apa_min_context": APA_MIN_CONTEXT if is_apa else None,
        "fast_max_seq": FAST_MAX_SEQ if is_apa else None,
        "GEMMA4_APA_DECODE_FUSED": "0" if arm == "apa_blend" else "1",
        "GEMMA4_APA_INT4": "1" if arm == "apa_int4" else "0",
        "GEMMA4_APA_GEMM": "1" if arm == "apa_gemm" else "0",
        "GEMMA4_QUANT_V": "0",
        "GEMMA4_QUANT_KV4": "0",
        "engine_class": "FA worktree" if arm == "apa_int4" else "canonical",
    }


def configure_environment(arm: str) -> None:
    cfg = arm_config(arm)
    os.environ["GEMMA4_APA_DECODE_FUSED"] = cfg["GEMMA4_APA_DECODE_FUSED"]
    os.environ["GEMMA4_APA_INT4"] = cfg["GEMMA4_APA_INT4"]
    os.environ["GEMMA4_APA_GEMM"] = cfg["GEMMA4_APA_GEMM"]
    # Storage quantization is a different experiment. Pin it off before
    # importing core.gemma4_tc, whose KVRing class snapshots these env vars.
    os.environ["GEMMA4_QUANT_V"] = "0"
    os.environ["GEMMA4_QUANT_KV4"] = "0"


def configure_model(model: Any, arm: str) -> None:
    cfg = arm_config(arm)
    for layer in model.layers:
        if layer.mixer.is_global:
            layer.mixer.attention_mode = cfg["attention_mode"]
            if arm != "standard":
                layer.mixer.refine_percentile = REFINE_PERCENTILE
                layer.mixer.bulk_bits = BULK_BITS
                layer.mixer.apa_min_context = APA_MIN_CONTEXT
                layer.mixer.fast_max_seq = FAST_MAX_SEQ


def empty_branch_census() -> dict[str, dict[str, int]]:
    return {
        "prefill": {"blend": 0, "fused": 0, "int4": 0, "gemm": 0},
        "decode": {"blend": 0, "fused": 0, "int4": 0, "gemm": 0},
    }


def install_instrumentation(
        tc: Any, gemma: Any, mistral: Any, runtime: dict[str, Any]
        ) -> Callable[[], None]:
    """Count the exact adapter branches and every global L=1 call."""
    orig_attn = gemma.Gemma4AttentionTC.__call__
    orig_blend = mistral._cublas_blend_attention
    orig_fused = tc.apa_selective_attention
    orig_int4 = getattr(tc, "apa_selective_attention_int4", None)
    orig_gemm = getattr(tc, "apa_gemm_selective_attention", None)

    def attention_call(self: Any, x: Any, cos: Any, sin: Any,
                       position_offset: int = 0, kv_cache: Any = None) -> Any:
        if self.is_global and runtime["phase"] in ("prefill", "decode"):
            length = int(x.shape[1])
            runtime["global_attention_calls"][runtime["phase"]] += 1
            if runtime["phase"] == "decode":
                runtime["decode_attention_lengths"][str(length)] = (
                    runtime["decode_attention_lengths"].get(str(length), 0) + 1)
                if length != 1:
                    raise AssertionError(f"scored global attention received L={length}")
        return orig_attn(self, x, cos, sin, position_offset, kv_cache)

    def blend_call(q: Any, k: Any, kq: Any, v: Any, group: int,
                   scale: float, zthr: float, causal: bool, blk: int) -> Any:
        phase = runtime["phase"]
        if phase in runtime["branch_census"]:
            runtime["branch_census"][phase]["blend"] += 1
            if phase == "decode" and int(q.shape[2]) != 1:
                raise AssertionError(f"blend scoring call L={q.shape[2]}")
        return orig_blend(q, k, kq, v, group, scale, zthr, causal, blk)

    def fused_call(q: Any, k: Any, kq: Any, v: Any, scale: float,
                   zthr: float, causal: bool) -> Any:
        phase = runtime["phase"]
        if phase in runtime["branch_census"]:
            runtime["branch_census"][phase]["fused"] += 1
            if phase == "decode" and int(q.shape[2]) != 1:
                raise AssertionError(f"fused scoring call L={q.shape[2]}")
        return orig_fused(q, k, kq, v, scale, zthr, causal)

    def int4_call(q: Any, k: Any, v: Any, scale: float,
                  zthr: float, causal: bool) -> Any:
        phase = runtime["phase"]
        if phase in runtime["branch_census"]:
            runtime["branch_census"][phase]["int4"] += 1
            if phase == "decode" and int(q.shape[2]) != 1:
                raise AssertionError(f"INT4 scoring call L={q.shape[2]}")
        if orig_int4 is None:
            raise AssertionError("INT4 wrapper installed without engine entry")
        return orig_int4(q, k, v, scale, zthr, causal)

    def gemm_call(q: Any, k: Any, v: Any, scale: float,
                  zthr: float, causal: bool, *, Lq: int = 0,
                  row0: int = 0, window: int = 0,
                  k_codes: Any = None, k_scales: Any = None) -> Any:
        phase = runtime["phase"]
        if phase in runtime["branch_census"]:
            runtime["branch_census"][phase]["gemm"] += 1
            if phase == "decode" and int(q.shape[2]) != 1:
                raise AssertionError(f"GEMM scoring call L={q.shape[2]}")
        if orig_gemm is None:
            raise AssertionError("GEMM wrapper installed without engine entry")
        return orig_gemm(
            q, k, v, scale, zthr, causal, Lq=Lq, row0=row0, window=window,
            k_codes=k_codes, k_scales=k_scales)

    gemma.Gemma4AttentionTC.__call__ = attention_call
    mistral._cublas_blend_attention = blend_call
    tc.apa_selective_attention = fused_call
    if orig_int4 is not None:
        tc.apa_selective_attention_int4 = int4_call
    if orig_gemm is not None:
        tc.apa_gemm_selective_attention = gemm_call

    def restore() -> None:
        gemma.Gemma4AttentionTC.__call__ = orig_attn
        mistral._cublas_blend_attention = orig_blend
        tc.apa_selective_attention = orig_fused
        if orig_int4 is not None:
            tc.apa_selective_attention_int4 = orig_int4
        if orig_gemm is not None:
            tc.apa_gemm_selective_attention = orig_gemm

    return restore


def ring_receipt(caches: list[Any], gemma: Any, arm: str,
                 document_index: int) -> dict[str, Any]:
    rows = []
    expect_kq = arm in ("apa_blend", "apa_fused")
    for layer_index in GLOBAL_LAYERS:
        cache = caches[layer_index]
        if not isinstance(cache, gemma.KVRing):
            raise AssertionError(
                f"doc {document_index} layer {layer_index}: expected KVRing, "
                f"got {type(cache)}")
        count = int(cache.count)
        if count != PREFILL_TOKENS + SCORED_TOKENS_PER_DOCUMENT:
            raise AssertionError(
                f"doc {document_index} layer {layer_index}: ring count {count}")
        kb_dtype = str(cache.kb.dtype)
        vb_dtype = str(cache.vb.dtype)
        if kb_dtype != "bfloat16" or vb_dtype != "bfloat16":
            raise AssertionError(
                f"doc {document_index} layer {layer_index}: raw ring dtypes "
                f"{kb_dtype}/{vb_dtype}, expected bfloat16/bfloat16")
        if expect_kq:
            if cache.kqb is None or str(cache.kqb.dtype) != "bfloat16":
                raise AssertionError(
                    f"doc {document_index} layer {layer_index}: expected bf16 kqb")
            if int(cache.kq_count) != count:
                raise AssertionError((layer_index, cache.kq_count, count))
            kq = {"presence": "present", "dtype": str(cache.kqb.dtype),
                  "kq_count": int(cache.kq_count)}
        else:
            if cache.kqb is not None or int(cache.kq_count) != 0:
                raise AssertionError(
                    f"doc {document_index} layer {layer_index}: forbidden kqb ring")
            kq = {"presence": "absent", "dtype": None, "kq_count": 0}
        rows.append({
            "layer": layer_index,
            "is_kv_ring": True,
            "count": count,
            "capacity": int(cache.cap),
            "raw_k_dtype": kb_dtype,
            "raw_v_dtype": vb_dtype,
            "derived_kq_ring": kq,
        })
    return {
        "document_index": document_index,
        "assertion": "PASS",
        "expected_derived_kq": "bf16-present" if expect_kq else "absent",
        "layers": rows,
    }


def nll_from_logits(logits: Any, target: int) -> float:
    row = np.asarray(logits.float().numpy()[0, -1], dtype=np.float64)
    maximum = float(row.max())
    return float(maximum + np.log(np.exp(row - maximum).sum()) - row[target])


def expected_decode_branch(arm: str) -> str | None:
    return {
        "standard": None,
        "apa_blend": "blend",
        "apa_fused": "fused",
        "apa_int4": "int4",
        "apa_gemm": "gemm",
    }[arm]


def prefill_census_valid(arm: str, census: dict[str, int]) -> bool:
    if arm == "standard":
        return sum(census.values()) == 0
    if arm in ("apa_blend", "apa_fused"):
        return (census.get("blend", 0) > 0 and census.get("fused", 0) > 0
                and census.get("int4", 0) == 0
                and census.get("gemm", 0) == 0)
    if arm == "apa_int4":
        return (census.get("blend", 0) > 0 and census.get("int4", 0) > 0
                and census.get("fused", 0) == 0
                and census.get("gemm", 0) == 0)
    return (census.get("gemm", 0) > 0 and census.get("blend", 0) == 0
            and census.get("fused", 0) == 0
            and census.get("int4", 0) == 0)


def engagement_receipt(runtime: dict[str, Any], arm: str,
                       ring_documents: list[dict[str, Any]]) -> dict[str, Any]:
    expected = expected_decode_branch(arm)
    census = runtime["branch_census"]["decode"]
    actual_active = sum(census.values())
    global_calls = runtime["global_attention_calls"]["decode"]
    if global_calls != EXPECTED_APA_DECODE_CALLS:
        raise AssertionError(
            f"global L=1 calls {global_calls} != {EXPECTED_APA_DECODE_CALLS}")
    if runtime["decode_attention_lengths"] != {"1": EXPECTED_APA_DECODE_CALLS}:
        raise AssertionError(runtime["decode_attention_lengths"])
    if runtime["teacher_forced_model_calls"] != EXPECTED_SCORED_TOKENS:
        raise AssertionError(runtime["teacher_forced_model_calls"])
    if len(ring_documents) != DOCUMENT_COUNT:
        raise AssertionError(f"ring receipts {len(ring_documents)} != {DOCUMENT_COUNT}")
    if expected is None:
        valid = actual_active == 0
    else:
        valid = (census[expected] == EXPECTED_APA_DECODE_CALLS
                 and actual_active == EXPECTED_APA_DECODE_CALLS)
    if not valid:
        raise AssertionError(
            f"decode branch census for {arm}: {census}, expected={expected}")
    prefill = runtime["branch_census"]["prefill"]
    if not prefill_census_valid(arm, prefill):
        raise AssertionError(f"prefill branch census for {arm}: {prefill}")
    return {
        "assertion": "PASS",
        "APA_ENGAGED": arm != "standard",
        "engagement_definition": (
            "every global-layer scoring call reached the configured APA "
            "operator with 0 < refine_percentile < 1 and context > apa_min_context"),
        "expected_decode_branch": expected or "standard_non_APA",
        "apa_active_decode_calls": actual_active,
        "expected_apa_active_decode_calls": (
            0 if arm == "standard" else EXPECTED_APA_DECODE_CALLS),
        "global_l1_decode_calls": global_calls,
        "decode_attention_length_census": runtime["decode_attention_lengths"],
        "branch_census": runtime["branch_census"],
        "prefill_mixed_dispatch_assertion": "PASS",
        "global_prefill_attention_calls": runtime["global_attention_calls"]["prefill"],
        "teacher_forced_model_calls": runtime["teacher_forced_model_calls"],
        "teacher_forced_input_assertion": {
            "shape": [1, 1], "dtype": "int64",
            "calls": runtime["teacher_forced_model_calls"],
            "assertion": "PASS",
        },
        "ring_assertions": {
            "documents_checked": len(ring_documents),
            "all_pass": all(x["assertion"] == "PASS" for x in ring_documents),
            "per_document": ring_documents,
        },
    }


def run_measurement(arm: str, output: Path) -> int:
    started_at = utc_now()
    wall_started = time.perf_counter()
    tc_root = resolve_tc_root(arm)
    if arm == "apa_int4" and tc_root != FA_TC_ROOT.resolve():
        raise AssertionError(
            f"apa_int4 must use FA worktree {FA_TC_ROOT.resolve()}, got {tc_root}")
    if arm != "apa_int4" and tc_root != CANONICAL_TC_ROOT.resolve():
        raise AssertionError(
            f"{arm} must use canonical engine {CANONICAL_TC_ROOT.resolve()}, "
            f"got {tc_root}")

    tokens, corpus = load_corpus_tokens()
    documents, selection = select_documents(tokens)
    protocol = protocol_payload(corpus, selection)
    del tokens

    base_receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "evidence_class": "model perplexity with engaged scoring",
        "arm": arm,
        "status": "starting",
        "started_at_utc": started_at,
        "gpu_measurement": True,
        "runtime_budget": {
            "seconds": RUNTIME_BUDGET_S,
            "document_count_fixed_at_minimum_valid": DOCUMENT_COUNT,
        },
        "config": arm_config(arm),
        "engine": {"requested_root": str(tc_root)},
        "protocol": protocol,
        "documents": [],
    }
    atomic_json(output, base_receipt)

    configure_environment(arm)
    sys.path.insert(0, str(tc_root))
    sys.path.insert(0, str(REPO))
    import tensor_cuda as tc
    import core.gemma4_tc as gemma
    import core.mistral7b_tc as mistral

    imported_engine = Path(tc.__file__).resolve()
    if not imported_engine.is_relative_to(tc_root):
        raise AssertionError(
            f"tensor_cuda imported from {imported_engine}, not {tc_root}")
    if not hasattr(tc, "apa_selective_attention"):
        raise RuntimeError("engine lacks apa_selective_attention")
    if arm == "apa_int4" and not hasattr(tc, "apa_selective_attention_int4"):
        raise RuntimeError("FA engine lacks apa_selective_attention_int4")
    if arm == "apa_gemm" and not hasattr(tc, "apa_gemm_selective_attention"):
        raise RuntimeError("canonical engine lacks apa_gemm_selective_attention")
    base_receipt["engine"].update({
        "imported_module": str(imported_engine),
        "python_api_sha256": sha256_file(imported_engine),
        "compiled_module": str(Path(tc._C.__file__).resolve()),
        "compiled_module_sha256": sha256_file(Path(tc._C.__file__).resolve()),
        "root_assertion": "PASS",
        "has_fused": hasattr(tc, "apa_selective_attention"),
        "has_int4": hasattr(tc, "apa_selective_attention_int4"),
        "has_gemm": hasattr(tc, "apa_gemm_selective_attention"),
    })
    base_receipt["code_identity"] = current_code_identity()

    tc.set_alloc_pooling(True)
    model, info = gemma.Gemma4_TC.from_pretrained()
    configure_model(model, arm)
    base_receipt["model_info"] = {str(k): str(v) for k, v in info.items()}

    runtime: dict[str, Any] = {
        "phase": "idle",
        "branch_census": empty_branch_census(),
        "global_attention_calls": {"prefill": 0, "decode": 0},
        "decode_attention_lengths": {},
        "teacher_forced_model_calls": 0,
    }
    restore = install_instrumentation(tc, gemma, mistral, runtime)
    total_nll = 0.0
    total_decode_wall_s = 0.0
    ring_documents: list[dict[str, Any]] = []
    try:
        for doc_index, ids in enumerate(documents):
            doc_started = time.perf_counter()
            with tc.no_grad():
                runtime["phase"] = "prefill"
                _, caches = model(
                    ids[None, :PREFILL_TOKENS], last_token_only=True)
                tc.synchronize()

                runtime["phase"] = "decode"
                decode_started = time.perf_counter()
                doc_nll = 0.0
                for step in range(SCORED_TOKENS_PER_DOCUMENT):
                    input_ids = np.asarray(
                        [[ids[PREFILL_TOKENS + step]]], dtype=np.int64)
                    if input_ids.shape != (1, 1) or input_ids.dtype != np.int64:
                        raise AssertionError((input_ids.shape, input_ids.dtype))
                    runtime["teacher_forced_model_calls"] += 1
                    logits, caches = model(
                        input_ids, caches=caches,
                        position_offset=PREFILL_TOKENS + step,
                        last_token_only=True)
                    target = int(ids[PREFILL_TOKENS + step + 1])
                    doc_nll += nll_from_logits(logits, target)
                tc.synchronize()
                decode_wall_s = time.perf_counter() - decode_started

            runtime["phase"] = "idle"
            rings = ring_receipt(caches, gemma, arm, doc_index)
            ring_documents.append(rings)
            record = {
                **selection["documents"][doc_index],
                "scored_tokens": SCORED_TOKENS_PER_DOCUMENT,
                "nll_sum": doc_nll,
                "mean_nll": doc_nll / SCORED_TOKENS_PER_DOCUMENT,
                "ppl": math.exp(doc_nll / SCORED_TOKENS_PER_DOCUMENT),
                "decode_wall_s": decode_wall_s,
                "decode_ms_per_token": (
                    1000.0 * decode_wall_s / SCORED_TOKENS_PER_DOCUMENT),
                "document_wall_s": time.perf_counter() - doc_started,
                "ring_assertion": "PASS",
            }
            base_receipt["documents"].append(record)
            total_nll += doc_nll
            total_decode_wall_s += decode_wall_s
            elapsed = time.perf_counter() - wall_started
            base_receipt.update({
                "status": "running",
                "completed_documents": doc_index + 1,
                "elapsed_s": elapsed,
            })
            atomic_json(output, base_receipt)
            print(
                f"[{arm}] document {doc_index + 1}/{DOCUMENT_COUNT}: "
                f"mean_nll={record['mean_nll']:.6f} "
                f"decode={record['decode_ms_per_token']:.3f} ms/tok "
                f"elapsed={elapsed:.1f}s",
                flush=True)
            del caches
            tc.empty_cache()
    finally:
        runtime["phase"] = "idle"
        restore()
        os.environ.pop("GEMMA4_APA_DECODE_FUSED", None)
        os.environ.pop("GEMMA4_APA_INT4", None)
        os.environ.pop("GEMMA4_APA_GEMM", None)
        os.environ.pop("GEMMA4_QUANT_V", None)
        os.environ.pop("GEMMA4_QUANT_KV4", None)

    engagement = engagement_receipt(runtime, arm, ring_documents)
    mean_nll = total_nll / EXPECTED_SCORED_TOKENS
    elapsed_s = time.perf_counter() - wall_started
    receipt = {
        **base_receipt,
        "status": "complete",
        "completed_at_utc": utc_now(),
        "completed_documents": DOCUMENT_COUNT,
        "scored_tokens": EXPECTED_SCORED_TOKENS,
        "mean_nll": mean_nll,
        "ppl": math.exp(mean_nll),
        "decode_wall_s": total_decode_wall_s,
        "decode_ms_per_token": 1000.0 * total_decode_wall_s / EXPECTED_SCORED_TOKENS,
        "decode_timing_scope": (
            "teacher-forced L=1 model calls plus required logit device-to-host "
            "transfer and host NLL; prefills excluded; synchronized per document"),
        "elapsed_s": elapsed_s,
        "runtime_budget_exceeded": elapsed_s > RUNTIME_BUDGET_S,
        "engagement": engagement,
        "comparisons": (
            "computed by --arm summary from all five paired receipts"),
    }
    atomic_json(output, receipt)
    print(
        f"[{arm}] COMPLETE docs={DOCUMENT_COUNT} scored={EXPECTED_SCORED_TOKENS} "
        f"mean_nll={mean_nll:.6f} ppl={receipt['ppl']:.6f} "
        f"decode={receipt['decode_ms_per_token']:.3f} ms/tok",
        flush=True)
    print(
        f"[{arm}] engagement APA={engagement['APA_ENGAGED']} "
        f"active_calls={engagement['apa_active_decode_calls']} "
        f"branches={engagement['branch_census']['decode']} ring=PASS",
        flush=True)
    print(f"[{arm}] wrote {output}", flush=True)
    return 0


def sample_std_sem(values: Iterable[float]) -> tuple[float, float]:
    xs = [float(x) for x in values]
    if len(xs) < 2:
        return 0.0, 0.0
    std = statistics.stdev(xs)
    return std, std / math.sqrt(len(xs))


def paired_comparison(arm: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    arm_docs = arm["documents"]
    ref_docs = reference["documents"]
    if len(arm_docs) != len(ref_docs):
        raise AssertionError("paired receipts have different document counts")
    nll_deltas = []
    relative_ppl_deltas = []
    for left, right in zip(arm_docs, ref_docs):
        identity = ("source_block_index", "token_ids_int64_sha256", "scored_tokens")
        if any(left[key] != right[key] for key in identity):
            raise AssertionError(
                f"paired document identity mismatch: {left.get('document_index')}")
        delta = float(left["mean_nll"] - right["mean_nll"])
        nll_deltas.append(delta)
        relative_ppl_deltas.append(100.0 * math.expm1(delta))
    nll_std, nll_sem = sample_std_sem(nll_deltas)
    rel_std, rel_sem = sample_std_sem(relative_ppl_deltas)
    aggregate_delta = float(arm["mean_nll"] - reference["mean_nll"])
    return {
        "reference_arm": reference["arm"],
        "aggregate_mean_nll_delta": aggregate_delta,
        "aggregate_relative_ppl_delta_percent": 100.0 * math.expm1(aggregate_delta),
        "paired_document_count": len(nll_deltas),
        "paired_mean_nll_delta": statistics.mean(nll_deltas),
        "paired_nll_delta_sample_std": nll_std,
        "paired_nll_delta_sem": nll_sem,
        "paired_relative_ppl_delta_percent_mean": statistics.mean(relative_ppl_deltas),
        "paired_relative_ppl_delta_percent_sample_std": rel_std,
        "paired_relative_ppl_delta_percent_sem": rel_sem,
        "std_convention": "sample standard deviation (ddof=1); sem=std/sqrt(n)",
    }


def read_receipts(directory: Path,
                  legacy_directory: Path | None = None
                  ) -> dict[str, dict[str, Any]]:
    receipts = {}
    for arm in MEASUREMENT_ARMS:
        path = directory / f"{arm}.json"
        if (not path.is_file() and arm in LEGACY_MEASUREMENT_ARMS
                and legacy_directory is not None):
            path = legacy_directory / f"{arm}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing {arm} receipt: {path}")
        receipt = json.loads(path.read_text())
        if receipt.get("schema") != SCHEMA or receipt.get("arm") != arm:
            raise AssertionError(f"wrong receipt/schema in {path}")
        if receipt.get("status") != "complete":
            raise AssertionError(f"incomplete {arm} receipt: {receipt.get('status')}")
        if receipt.get("scored_tokens") != EXPECTED_SCORED_TOKENS:
            raise AssertionError(f"wrong scored-token count in {path}")
        engagement = receipt.get("engagement", {})
        expected_branch = expected_decode_branch(arm)
        census = engagement.get("branch_census", {}).get("decode", {})
        prefill_census = engagement.get("branch_census", {}).get("prefill", {})
        if engagement.get("assertion") != "PASS":
            raise AssertionError(f"engagement assertion missing/failed in {path}")
        if engagement.get("global_l1_decode_calls") != EXPECTED_APA_DECODE_CALLS:
            raise AssertionError(f"wrong L=1 call count in {path}")
        if engagement.get("teacher_forced_model_calls") != EXPECTED_SCORED_TOKENS:
            raise AssertionError(f"wrong teacher-forced call count in {path}")
        if not engagement.get("ring_assertions", {}).get("all_pass"):
            raise AssertionError(f"ring assertion missing/failed in {path}")
        if (engagement.get("prefill_mixed_dispatch_assertion") != "PASS"
                or not prefill_census_valid(arm, prefill_census)):
            raise AssertionError(f"prefill dispatch assertion failed in {path}")
        if expected_branch is None:
            if engagement.get("APA_ENGAGED") is not False or sum(census.values()) != 0:
                raise AssertionError(f"standard arm reached APA branch in {path}")
        else:
            if engagement.get("APA_ENGAGED") is not True:
                raise AssertionError(f"APA engagement false in {path}")
            if (census.get(expected_branch) != EXPECTED_APA_DECODE_CALLS
                    or sum(census.values()) != EXPECTED_APA_DECODE_CALLS):
                raise AssertionError(f"wrong APA branch census in {path}: {census}")
        receipts[arm] = receipt
    fingerprints = {r["protocol"]["fingerprint_sha256"] for r in receipts.values()}
    if len(fingerprints) != 1:
        raise AssertionError(f"protocol fingerprints differ: {fingerprints}")
    # The four registered receipts predate the additive apa_gemm arm. Preserve
    # their cohort identity check while requiring the new receipt to match the
    # current additive harness/core. Protocol identity remains the cross-arm
    # comparability gate above.
    legacy_code_identities = {
        canonical_hash(receipts[arm].get("code_identity"))
        for arm in LEGACY_MEASUREMENT_ARMS
    }
    if len(legacy_code_identities) != 1:
        raise AssertionError("legacy quartet harness/core identities differ")
    if receipts["apa_gemm"].get("code_identity") != current_code_identity():
        raise AssertionError(
            "apa_gemm receipt does not match current harness/core sources")
    return receipts


def build_summary(receipts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    standard = receipts["standard"]
    blend = receipts["apa_blend"]
    rows = []
    for arm in MEASUREMENT_ARMS:
        receipt = receipts[arm]
        rows.append({
            "arm": arm,
            "mean_nll": receipt["mean_nll"],
            "ppl": receipt["ppl"],
            "vs_standard": paired_comparison(receipt, standard),
            "vs_apa_blend": paired_comparison(receipt, blend),
            "decode_ms_per_token": receipt["decode_ms_per_token"],
            "apa_active_decode_calls": receipt["engagement"]["apa_active_decode_calls"],
            "decode_branch_census": receipt["engagement"]["branch_census"]["decode"],
            "ring_assertion": receipt["engagement"]["ring_assertions"]["all_pass"],
        })
    return {
        "schema": SCHEMA,
        "evidence_class": "G-C paired receipt reduction",
        "arm": "summary",
        "status": "complete",
        "created_at_utc": utc_now(),
        "adjudication": "not_performed",
        "verdicts": None,
        "protocol_fingerprint_sha256": standard["protocol"]["fingerprint_sha256"],
        "code_identity_policy": (
            "legacy quartet shares its registered identity; apa_gemm matches "
            "the current additive harness/core"),
        "document_count": DOCUMENT_COUNT,
        "scored_tokens_per_arm": EXPECTED_SCORED_TOKENS,
        "rows": rows,
    }


def print_summary(summary: dict[str, Any]) -> None:
    print("G-C TABLE (measurement only; no verdicts)", flush=True)
    print(
        "arm        mean_nll        ppl   dPPL/std%  paired_std  paired_sem "
        " dPPL/blend% paired_std  paired_sem   ms/tok  APA calls  blend/fused/int4/gemm",
        flush=True)
    for row in summary["rows"]:
        std = row["vs_standard"]
        blend = row["vs_apa_blend"]
        census = row["decode_branch_census"]
        print(
            f"{row['arm']:<10s} {row['mean_nll']:10.6f} {row['ppl']:10.4f} "
            f"{std['aggregate_relative_ppl_delta_percent']:10.4f} "
            f"{std['paired_relative_ppl_delta_percent_sample_std']:10.4f} "
            f"{std['paired_relative_ppl_delta_percent_sem']:10.4f} "
            f"{blend['aggregate_relative_ppl_delta_percent']:11.4f} "
            f"{blend['paired_relative_ppl_delta_percent_sample_std']:10.4f} "
            f"{blend['paired_relative_ppl_delta_percent_sem']:10.4f} "
            f"{row['decode_ms_per_token']:8.3f} "
            f"{row['apa_active_decode_calls']:10d} "
            f"{census.get('blend', 0)}/{census.get('fused', 0)}/"
            f"{census.get('int4', 0)}/{census.get('gemm', 0)}",
            flush=True)


def run_summary(output: Path, legacy_receipts_dir: Path | None = None) -> int:
    receipts = read_receipts(output.parent, legacy_receipts_dir)
    summary = build_summary(receipts)
    atomic_json(output, summary)
    print_summary(summary)
    print(f"[summary] wrote {output}", flush=True)
    return 0


def cpu_check(output: Path) -> int:
    tokens, corpus = load_corpus_tokens()
    documents, selection = select_documents(tokens)
    protocol = protocol_payload(corpus, selection)
    assert len(documents) == DOCUMENT_COUNT
    assert sum(len(x) for x in documents) == DOCUMENT_COUNT * TOKENS_PER_DOCUMENT
    assert selection["scored_token_count"] == EXPECTED_SCORED_TOKENS
    assert resolve_tc_root("standard", {}) == CANONICAL_TC_ROOT.resolve()
    assert resolve_tc_root("apa_blend", {}) == CANONICAL_TC_ROOT.resolve()
    assert resolve_tc_root("apa_fused", {}) == CANONICAL_TC_ROOT.resolve()
    assert resolve_tc_root("apa_int4", {}) == FA_TC_ROOT.resolve()
    assert resolve_tc_root("apa_gemm", {}) == CANONICAL_TC_ROOT.resolve()
    assert arm_config("apa_blend")["GEMMA4_APA_DECODE_FUSED"] == "0"
    assert arm_config("apa_fused")["GEMMA4_APA_DECODE_FUSED"] == "1"
    assert arm_config("apa_int4")["GEMMA4_APA_INT4"] == "1"
    assert arm_config("apa_gemm")["GEMMA4_APA_GEMM"] == "1"
    assert all(arm_config(arm)["GEMMA4_APA_GEMM"] == "0"
               for arm in LEGACY_MEASUREMENT_ARMS)
    assert EXPECTED_APA_DECODE_CALLS == 409_600
    assert expected_decode_branch("apa_gemm") == "gemm"
    assert prefill_census_valid(
        "apa_gemm", {"blend": 0, "fused": 0, "int4": 0, "gemm": 1})
    assert not prefill_census_valid(
        "apa_gemm", {"blend": 1, "fused": 0, "int4": 0, "gemm": 1})
    assert all(arm_config(arm)[name] == "0"
               for arm in MEASUREMENT_ARMS
               for name in ("GEMMA4_QUANT_V", "GEMMA4_QUANT_KV4"))
    canonical_init = CANONICAL_TC_ROOT / "tensor_cuda" / "__init__.py"
    fa_init = FA_TC_ROOT / "tensor_cuda" / "__init__.py"
    assert canonical_init.is_file() and fa_init.is_file()
    assert "def apa_selective_attention(" in canonical_init.read_text()
    assert "def apa_gemm_selective_attention(" in canonical_init.read_text()
    assert "def apa_selective_attention_int4(" in fa_init.read_text()
    assert (MODEL_DIR / "model.safetensors").is_file()
    receipt = {
        "schema": SCHEMA,
        "arm": "cpu-check",
        "status": "complete",
        "gpu_measurement": False,
        "checks": {
            "offline_cached_public_corpus": "PASS",
            "token_count_at_least_200k": "PASS",
            "seeded_nonoverlapping_25_document_selection": "PASS",
            "8192_prefill_plus_2048_L1_target_shape": "PASS",
            "engine_root_matrix": "PASS",
            "engine_entry_static_capabilities": "PASS",
            "arm_environment_matrix": "PASS",
            "gemm_branch_and_409600_call_contract": "PASS",
        },
        "protocol": protocol,
        "expected_global_l1_decode_calls_per_apa_arm": EXPECTED_APA_DECODE_CALLS,
    }
    atomic_json(output, receipt)
    print("CPU OFFLINE CORPUS LOAD: PASS", flush=True)
    print(
        f"CORPUS: {corpus['name']} rows={corpus['dataset_rows']} "
        f"tokens={corpus['full_token_count']} "
        f"raw_sha256={corpus['raw_text_sha256']}", flush=True)
    print(
        f"CPU DOCUMENT SELECTION: PASS docs={DOCUMENT_COUNT} "
        f"tokens/doc={TOKENS_PER_DOCUMENT} selected={selection['selected_token_count']} "
        f"scored={selection['scored_token_count']}", flush=True)
    print("CPU TRUE-L1 PROTOCOL SHAPE: PASS prefill=8192 decode=(1,1) int64", flush=True)
    print("CPU ENGINE ROOT MATRIX: PASS canonical=4 arms FA=apa_int4", flush=True)
    print("CPU ENGINE ENTRY STATIC CHECK: PASS canonical=fused+gemm FA=int4",
          flush=True)
    print("CPU ARM ENVIRONMENT MATRIX: PASS", flush=True)
    print("CPU GEMM CENSUS CONTRACT: PASS expected_decode_calls=409600",
          flush=True)
    print(f"CPU PROTOCOL FINGERPRINT: {protocol['fingerprint_sha256']}", flush=True)
    print(f"[cpu-check] wrote {output}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=ALL_ARMS)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--legacy-receipts-dir", type=Path,
        help=("summary-only read fallback for the registered standard/"
              "apa_blend/apa_fused/apa_int4 receipts"))
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    try:
        if args.arm == "cpu-check":
            return cpu_check(output)
        if args.arm == "summary":
            legacy_dir = (args.legacy_receipts_dir.expanduser().resolve()
                          if args.legacy_receipts_dir is not None else None)
            return run_summary(output, legacy_dir)
        if args.legacy_receipts_dir is not None:
            raise ValueError("--legacy-receipts-dir is valid only for summary")
        return run_measurement(args.arm, output)
    except Exception as exc:
        prior: dict[str, Any] = {}
        if output.is_file():
            try:
                loaded = json.loads(output.read_text())
                if isinstance(loaded, dict):
                    prior = loaded
            except Exception:
                pass
        failure = {
            **prior,
            "schema": SCHEMA,
            "arm": args.arm,
            "status": "error",
            "gpu_measurement": args.arm in MEASUREMENT_ARMS,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "created_at_utc": utc_now(),
        }
        atomic_json(output, failure)
        print(f"[{args.arm}] ERROR: {failure['error']}", file=sys.stderr, flush=True)
        print(f"[{args.arm}] wrote RED receipt {output}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
