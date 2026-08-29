#!/usr/bin/env python3
"""GLC P0.d: native-Harmony versus raw-format likelihood discriminator.

The evidence-bearing Harmony cells reuse the P0.a HuggingFace BF16 CPU
loader and logit scorer.  The raw cells do not rerun the approximately
seven-hour reference measurement: they verify the registered attempt01
receipts by file SHA-256 and by rebuilding every sealed arm/token hash.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import glc_p0_common as common
import glc_p0_hf_reference as hf_reference


REPO_ROOT = SCRIPT_PATH.parents[1]
ADDENDUM_PATH = REPO_ROOT / "docs" / "GLC_LONG_CONTEXT_ADDENDUM_1.md"
ORDER_PATH = REPO_ROOT / "orders" / "GLC_P0D_FORMAT_DISCRIMINATOR.md"
OUTPUT_ROOT = REPO_ROOT / "artifacts" / "glc_p0d"
HARMONY_OUTPUT = OUTPUT_ROOT / "harmony"
RAW_OUTPUT = OUTPUT_ROOT / "raw"
P0A_HF_OUTPUT = common.OUTPUT_ROOT / "hf_reference"
CHAT_TEMPLATE_PATH = common.MODEL_DIR / "chat_template.jinja"
TOKENIZER_CONFIG_PATH = common.MODEL_DIR / "tokenizer_config.json"

LENGTHS = (2048, 2560)
FORMATS = ("harmony", "raw")
HARMONY_SUPPORTED_MAX_NATS = 0.30
HARMONY_REFUTED_MIN_NATS = 1.0
RAW_REQUIRED_MIN_NATS = 2.0
REGISTERED_SENTENCE = (
    "SUPPORTED: harmony deltas ≤ 0.30 nats at both lengths (raw ≥ 2.0 "
    "reproduced/verified). REFUTED: harmony deltas ≥ 1.0 at either length. "
    "MIXED: between."
)
SCAFFOLD_DESIGN = (
    "Each arm begins with the exact tokenizer-rendered Harmony system turn and "
    "open user-message header. The sealed WikiText IDs are concatenated directly "
    "after `<|message|>` without decoding or re-tokenizing them. No role, channel, "
    "end, or other scaffold token is inserted between the WikiText history and "
    "the 511 targets: every target directly continues its true preceding WikiText "
    "token inside the same open user document block. The one-token `<|end|>` "
    "closure that a complete rendered user turn would place after the document is "
    "deferred beyond the targets and omitted from the scoring prefix, so "
    "`logits_to_keep=511` selects exactly the 511 target-predicting positions."
)

EXPECTED_CHAT_TEMPLATE_SHA256 = (
    "a4c9919cbbd4acdd51ccffe22da049264b1b73e59055fa58811a99efbd7c8146"
)
EXPECTED_TOKENIZER_CONFIG_SHA256 = (
    "9279e942392b742d633c7adbb89ebe002c98399db8926a7af5125c726f404070"
)
EXPECTED_RAW_RECEIPT_SHA256 = {
    2048: "4267c35d558b3ddb8604704b80edffa8891e6c85e7e2440fc1ded84414c147cb",
    2560: "2d0fc5288b0363fac457072113db67d3ee736c64c37f69ee4dbb26c7ca52e85f",
}


@dataclass(frozen=True)
class Scaffold:
    prefix_ids: np.ndarray
    suffix_ids: np.ndarray
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("preflight", "self-test", "score-cell", "analyze")
    )
    parser.add_argument("--format", choices=FORMATS)
    parser.add_argument("--length", type=int, choices=LENGTHS)
    parser.add_argument("--attempt", type=int, default=0)
    parser.add_argument("--model-dir", type=Path, default=common.MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--offload-dir", type=Path, default=Path("/tmp/glc_p0d_hf_offload")
    )
    parser.add_argument("--cpu-memory", default="30GiB")
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--logit-chunk", type=int, default=32)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def ensure_output_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved != OUTPUT_ROOT.resolve():
        raise ValueError(f"GLC P0.d artifacts are restricted to {OUTPUT_ROOT.resolve()}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def receipt_path(cell_format: str, length: int, attempt: int) -> Path:
    directory = HARMONY_OUTPUT if cell_format == "harmony" else RAW_OUTPUT
    return directory / f"{cell_format}_{int(length)}_attempt{int(attempt):02d}.json"


def _content_addressed_json(directory: Path, stem: str, core: dict[str, Any]) -> Path:
    digest = common.canonical_json_sha256(core)[:16]
    path = directory / f"{stem}_{digest}.json"
    if path.exists():
        prior = common.read_json(path)
        if any(prior.get(key) != value for key, value in core.items()):
            raise RuntimeError(f"content-addressed artifact collision: {path}")
    else:
        common.write_new_json(path, {**core, "created_at": common.now_iso()})
    return path


def _as_1d_ids(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        value = value["input_ids"]
    elif hasattr(value, "keys") and "input_ids" in value:
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    work = np.asarray(value, dtype=np.int64)
    if work.ndim == 2 and work.shape[0] == 1:
        work = work[0]
    if work.ndim != 1:
        raise RuntimeError(f"template token IDs are not rank one: {work.shape}")
    return np.ascontiguousarray(work, dtype=np.int64)


def _template_ids(tokenizer: Any, messages: list[dict[str, str]]) -> np.ndarray:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
    )
    return _as_1d_ids(rendered)


def load_tokenizer(model_dir: Path) -> Any:
    model_dir = common.ensure_model_dir(model_dir)
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise RuntimeError("HF_HUB_OFFLINE=1 is mandatory")
    if os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise RuntimeError("TRANSFORMERS_OFFLINE=1 is mandatory")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    loaded = getattr(tokenizer, "chat_template", None)
    if not isinstance(loaded, str) or not loaded:
        raise RuntimeError("pinned tokenizer has no native chat template")
    source = CHAT_TEMPLATE_PATH.read_text(encoding="utf-8")
    if loaded != source:
        raise RuntimeError("loaded tokenizer chat template differs from snapshot source")
    if common.sha256_bytes(loaded.encode("utf-8")) != EXPECTED_CHAT_TEMPLATE_SHA256:
        raise RuntimeError("registered chat template SHA-256 drifted")
    if common.sha256_file(TOKENIZER_CONFIG_PATH) != EXPECTED_TOKENIZER_CONFIG_SHA256:
        raise RuntimeError("registered tokenizer config SHA-256 drifted")
    return tokenizer


def build_scaffold(tokenizer: Any) -> Scaffold:
    messages = [{"role": "user", "content": ""}]
    rendered_a = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    rendered_b = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    ids_a = _template_ids(tokenizer, messages)
    ids_b = _template_ids(tokenizer, messages)
    if rendered_a != rendered_b or not np.array_equal(ids_a, ids_b):
        raise RuntimeError("native chat template rendering is not deterministic")
    if not isinstance(rendered_a, str):
        raise RuntimeError("native chat template did not render text")

    closing_text = "<|end|>"
    expected_tail = "<|start|>user<|message|><|end|>"
    if not rendered_a.endswith(expected_tail):
        raise RuntimeError("empty user turn has an unexpected Harmony tail")
    prefix_text = rendered_a[: -len(closing_text)]
    prefix_ids = _as_1d_ids(tokenizer.encode(prefix_text, add_special_tokens=False))
    suffix_ids = _as_1d_ids(tokenizer.encode(closing_text, add_special_tokens=False))
    if suffix_ids.size != 1:
        raise RuntimeError("Harmony user-turn closure is not exactly one token")
    if not np.array_equal(ids_a, np.concatenate((prefix_ids, suffix_ids))):
        raise RuntimeError("open-user scaffold does not partition the native render")

    metadata = {
        "construction": "native_apply_chat_template_empty_user_turn",
        "messages": messages,
        "add_generation_prompt": False,
        "document_role": "user",
        "tokenizer_class": type(tokenizer).__name__,
        "chat_template_path": str(CHAT_TEMPLATE_PATH),
        "chat_template_sha256": common.sha256_file(CHAT_TEMPLATE_PATH),
        "tokenizer_config_path": str(TOKENIZER_CONFIG_PATH),
        "tokenizer_config_sha256": common.sha256_file(TOKENIZER_CONFIG_PATH),
        "rendered_empty_turn": rendered_a,
        "rendered_empty_turn_utf8_sha256": common.sha256_bytes(
            rendered_a.encode("utf-8")
        ),
        "rendered_empty_turn_token_count": int(ids_a.size),
        "inserted_prefix_text": prefix_text,
        "inserted_prefix_text_excerpt": prefix_text,
        "inserted_prefix_utf8_sha256": common.sha256_bytes(
            prefix_text.encode("utf-8")
        ),
        "inserted_prefix_ids_sha256": common.sha256_array(prefix_ids),
        "inserted_prefix_token_count": int(prefix_ids.size),
        "inserted_prefix_token_ids": prefix_ids.tolist(),
        "deferred_closing_suffix_text": closing_text,
        "deferred_closing_suffix_ids_sha256": common.sha256_array(suffix_ids),
        "deferred_closing_suffix_token_count": int(suffix_ids.size),
        "deferred_closing_suffix_token_ids": suffix_ids.tolist(),
        "scoring_sequence": "inserted_prefix + sealed_wikitext_context + sealed_targets",
        "design": SCAFFOLD_DESIGN,
    }
    return Scaffold(prefix_ids=prefix_ids, suffix_ids=suffix_ids, metadata=metadata)


def _arm_json(arm: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in arm.items() if key not in {"inputs", "targets"}}


def harmony_arm(
    wikitext: np.ndarray, length: int, view: str, scaffold: Scaffold
) -> dict[str, Any]:
    raw = common.ramp_arm(wikitext, length, view)
    document_ids = np.ascontiguousarray(raw["full_sequence"], dtype=np.int64)
    wrapped_full = np.ascontiguousarray(
        np.concatenate((scaffold.prefix_ids, document_ids)), dtype=np.int64
    )
    inputs = np.ascontiguousarray(wrapped_full[:-1][None, :], dtype=np.int64)
    targets = np.ascontiguousarray(raw["targets"], dtype=np.int64)
    predictors = np.ascontiguousarray(inputs[0, -common.N_TARGETS :], dtype=np.int64)
    if not np.array_equal(predictors, raw["predictors"]):
        raise RuntimeError(f"Harmony {length}/{view} predictor adjacency changed")
    if common.sha256_array(targets) != raw["target_ids_sha256"]:
        raise RuntimeError(f"Harmony {length}/{view} target SHA changed")
    target_start = int(wrapped_full.size - common.N_TARGETS)
    logit_start = int(inputs.shape[1] - common.N_TARGETS)
    if target_start != logit_start + 1:
        raise AssertionError("wrapped causal target/logit offset is not one")
    if inputs.shape[1] < common.N_TARGETS:
        raise AssertionError("wrapped input is shorter than the scoring window")

    complete_template = np.ascontiguousarray(
        np.concatenate((wrapped_full, scaffold.suffix_ids)), dtype=np.int64
    )
    metadata = {
        "format": "harmony",
        "length": int(length),
        "view": view,
        "source_start": int(raw["source_start"]),
        "source_stop_exclusive": int(raw["source_stop_exclusive"]),
        "wikitext_span_tokens": int(document_ids.size),
        "wikitext_full_sequence_ids_sha256": raw["full_sequence_ids_sha256"],
        "raw_input_ids_sha256": raw["input_ids_sha256"],
        "predictor_ids_sha256": common.sha256_array(predictors),
        "target_ids_sha256": common.sha256_array(targets),
        "wrapped_open_sequence_ids_sha256": common.sha256_array(wrapped_full),
        "complete_template_sequence_ids_sha256": common.sha256_array(
            complete_template
        ),
        "input_ids_sha256": common.sha256_array(inputs),
        "model_input_tokens": int(inputs.shape[1]),
        "scored_target_tokens": int(targets.size),
        "logits_to_keep": common.N_TARGETS,
        "wrapped_target_positions": {
            "start": target_start,
            "stop_exclusive": int(wrapped_full.size),
        },
        "model_logit_input_positions": {
            "start": logit_start,
            "stop_exclusive": int(inputs.shape[1]),
        },
        "scaffold_prefix_token_count": int(scaffold.prefix_ids.size),
        "deferred_suffix_token_count": int(scaffold.suffix_ids.size),
        "closure_in_model_input": False,
        "port_receipt": raw["port_receipt"],
        "raw_port_mean_nll": raw["port_mean_nll"],
        "raw_port_ppl": raw["port_ppl"],
    }
    return {**metadata, "inputs": inputs, "targets": targets}


def verify_template_boundary(
    tokenizer: Any, scaffold: Scaffold, document_ids: np.ndarray
) -> dict[str, Any]:
    document_ids = np.ascontiguousarray(document_ids, dtype=np.int64)
    document_text = tokenizer.decode(
        document_ids.tolist(),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    retokenized = _as_1d_ids(tokenizer.encode(document_text, add_special_tokens=False))
    if not np.array_equal(retokenized, document_ids):
        raise RuntimeError("sealed document IDs do not decode/re-tokenize identically")
    messages = [{"role": "user", "content": document_text}]
    native_ids = _template_ids(tokenizer, messages)
    expected = np.ascontiguousarray(
        np.concatenate((scaffold.prefix_ids, document_ids, scaffold.suffix_ids)),
        dtype=np.int64,
    )
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    if not np.array_equal(native_ids, expected):
        raise RuntimeError("native template tokenization changes the sealed document IDs")
    if rendered != (
        scaffold.metadata["inserted_prefix_text"]
        + document_text
        + scaffold.metadata["deferred_closing_suffix_text"]
    ):
        raise RuntimeError("native rendered document does not match scaffold partition")
    return {
        "document_ids_sha256": common.sha256_array(document_ids),
        "document_token_count": int(document_ids.size),
        "decoded_utf8_sha256": common.sha256_bytes(document_text.encode("utf-8")),
        "native_rendered_ids_sha256": common.sha256_array(native_ids),
        "native_rendered_token_count": int(native_ids.size),
        "exact_prefix_document_suffix_partition": True,
    }


def _check(
    checks: list[dict[str, Any]], name: str, condition: bool, detail: Any = None
) -> None:
    row: dict[str, Any] = {"name": name, "passed": bool(condition)}
    if detail is not None:
        row["detail"] = detail
    checks.append(row)
    if not condition:
        raise RuntimeError(f"raw receipt verification failed: {name}: {detail}")


def verify_raw_source(
    length: int, wikitext: np.ndarray
) -> dict[str, Any]:
    source_path = P0A_HF_OUTPUT / f"ramp_{int(length)}_attempt01.json"
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source_sha = common.sha256_file(source_path)
    source = common.read_json(source_path)
    checks: list[dict[str, Any]] = []
    _check(
        checks,
        "source_receipt_file_sha256",
        source_sha == EXPECTED_RAW_RECEIPT_SHA256[length],
        source_sha,
    )
    _check(checks, "source_status_complete", source.get("status") == "complete")
    _check(
        checks,
        "source_schema",
        source.get("schema") == "glc_p0a_hf_reference_cell_v1",
        source.get("schema"),
    )
    _check(checks, "source_length", source.get("length") == int(length))
    _check(
        checks,
        "source_model_revision",
        source.get("model_revision") == common.MODEL_REVISION,
    )
    _check(
        checks,
        "source_controls_sha256",
        source.get("control_arrays_sha256") == common.EXPECTED_CONTROL_ARRAYS_SHA256,
    )
    _check(
        checks,
        "source_cpu_bf16_route",
        source.get("load", {}).get("route")
        == "hf_transformers_cpu_bf16_dequant_accelerate_offload",
    )

    arms: dict[str, Any] = {}
    target_hash: str | None = None
    for view in ("reference", "long"):
        sealed = common.ramp_arm(wikitext, length, view)
        observed = source.get("arms", {}).get(view, {})
        for key in (
            "length",
            "view",
            "source_start",
            "source_stop_exclusive",
            "full_sequence_ids_sha256",
            "input_ids_sha256",
            "predictor_ids_sha256",
            "target_ids_sha256",
        ):
            expected = sealed[key]
            _check(
                checks,
                f"{view}_{key}",
                observed.get(key) == expected,
                {"observed": observed.get(key), "expected": expected},
            )
        _check(
            checks,
            f"{view}_input_token_count",
            observed.get("input_tokens") == int(sealed["inputs"].shape[1]),
        )
        score = observed.get("score", {})
        _check(
            checks,
            f"{view}_score_target_sha256",
            score.get("target_ids_sha256") == sealed["target_ids_sha256"],
        )
        _check(
            checks,
            f"{view}_score_token_count",
            score.get("token_count") == common.N_TARGETS,
        )
        mean_nll = float(score["mean_nll"])
        nll_sum = float(score["nll_sum"])
        ppl = float(score["ppl"])
        _check(
            checks,
            f"{view}_score_mean_from_sum",
            abs(mean_nll - nll_sum / common.N_TARGETS) <= 1.0e-12,
        )
        _check(
            checks,
            f"{view}_score_ppl_from_mean",
            abs(ppl - math.exp(mean_nll)) <= max(1.0e-12, abs(ppl) * 1.0e-12),
        )
        if target_hash is None:
            target_hash = sealed["target_ids_sha256"]
        _check(
            checks,
            f"{view}_shared_target_sha256",
            sealed["target_ids_sha256"] == target_hash,
        )
        arms[view] = {
            "length": int(length),
            "view": view,
            "source_start": int(sealed["source_start"]),
            "source_stop_exclusive": int(sealed["source_stop_exclusive"]),
            "full_sequence_ids_sha256": sealed["full_sequence_ids_sha256"],
            "input_ids_sha256": sealed["input_ids_sha256"],
            "predictor_ids_sha256": sealed["predictor_ids_sha256"],
            "target_ids_sha256": sealed["target_ids_sha256"],
            "input_tokens": int(sealed["inputs"].shape[1]),
            "scored_target_tokens": common.N_TARGETS,
            "score": {
                "nll_sum": nll_sum,
                "mean_nll": mean_nll,
                "ppl": ppl,
                "token_count": int(score["token_count"]),
                "target_ids_sha256": score["target_ids_sha256"],
            },
        }

    delta = float(
        arms["long"]["score"]["mean_nll"]
        - arms["reference"]["score"]["mean_nll"]
    )
    registered_delta = float(source.get("hg1", {}).get("delta_mean_nll_nats"))
    _check(
        checks,
        "source_delta_recomputed",
        abs(delta - registered_delta) <= 1.0e-12,
        {"recomputed": delta, "registered": registered_delta},
    )
    return {
        "method": "verify_attempt01_by_pinned_file_and_rebuilt_arm_sha256",
        "recomputed_model_forward": False,
        "source_receipt": str(source_path.resolve()),
        "source_receipt_sha256": source_sha,
        "source_receipt_expected_sha256": EXPECTED_RAW_RECEIPT_SHA256[length],
        "target_ids_sha256": target_hash,
        "arms": arms,
        "delta_mean_nll_nats": delta,
        "checks": checks,
        "all_checks_passed": all(row["passed"] for row in checks),
    }


def preflight(args: argparse.Namespace) -> int:
    ensure_output_root(args.output_dir)
    common.ensure_model_dir(args.model_dir)
    manifest, arrays = common.validate_sealed_controls()
    tokenizer = load_tokenizer(args.model_dir)
    scaffold = build_scaffold(tokenizer)
    wiki = arrays["wikitext_0_2560"]
    cells: list[dict[str, Any]] = []
    raw_sources: dict[str, Any] = {}
    for length in LENGTHS:
        raw = verify_raw_source(length, wiki)
        raw_sources[str(length)] = {
            key: raw[key]
            for key in (
                "method",
                "recomputed_model_forward",
                "source_receipt",
                "source_receipt_sha256",
                "target_ids_sha256",
                "delta_mean_nll_nats",
                "all_checks_passed",
            )
        }
        for view in ("reference", "long"):
            arm = harmony_arm(wiki, length, view, scaffold)
            boundary = verify_template_boundary(
                tokenizer,
                scaffold,
                wiki[int(arm["source_start"]) : int(arm["source_stop_exclusive"])],
            )
            if arm["target_ids_sha256"] != raw["target_ids_sha256"]:
                raise RuntimeError(f"{length}/{view} target SHA differs from raw receipt")
            cells.append(
                {
                    "format": "harmony",
                    "length": int(length),
                    "view": view,
                    "wikitext_span_tokens": arm["wikitext_span_tokens"],
                    "model_input_tokens": arm["model_input_tokens"],
                    "scored_target_tokens": arm["scored_target_tokens"],
                    "target_ids_sha256": arm["target_ids_sha256"],
                    "input_ids_sha256": arm["input_ids_sha256"],
                    "template_boundary": boundary,
                }
            )
    core = {
        "schema": "glc_p0d_preflight_v1",
        "status": "ready",
        "model_revision": manifest["model_revision"],
        "control_arrays_sha256": common.sha256_file(common.CONTROL_ARRAYS),
        "plan": str(common.PLAN_PATH),
        "addendum": str(ADDENDUM_PATH),
        "order": str(ORDER_PATH),
        "scaffold": scaffold.metadata,
        "scaffold_design": SCAFFOLD_DESIGN,
        "raw_mode": "verify_attempt01_by_sha_no_recompute",
        "raw_sources": raw_sources,
        "cells": cells,
        "runtime": common.runtime_facts(),
        "script_sha256": common.sha256_file(SCRIPT_PATH),
    }
    path = _content_addressed_json(OUTPUT_ROOT / "preflight", "p0d", core)
    print(
        json.dumps(
            {
                "status": "ready",
                "receipt": str(path),
                "scaffold_prefix_tokens": int(scaffold.prefix_ids.size),
                "deferred_suffix_tokens": int(scaffold.suffix_ids.size),
            }
        ),
        flush=True,
    )
    return 0


def _synthetic_alignment() -> dict[str, Any]:
    prefix = np.asarray([900, 901, 902], dtype=np.int64)
    document = np.asarray([10, 11, 12, 13, 14, 15], dtype=np.int64)
    n_targets = 3
    wrapped_full = np.concatenate((prefix, document))
    inputs = wrapped_full[:-1]
    targets = document[-n_targets:]
    predictors = inputs[-n_targets:]
    expected_pairs = [[12, 13], [13, 14], [14, 15]]
    observed_pairs = np.stack((predictors, targets), axis=1).tolist()
    target_start = int(wrapped_full.size - n_targets)
    logit_start = int(inputs.size - n_targets)
    passed = bool(
        observed_pairs == expected_pairs
        and target_start == logit_start + 1
        and inputs.size == wrapped_full.size - 1
    )
    return {
        "passed": passed,
        "prefix": prefix.tolist(),
        "document": document.tolist(),
        "inputs": inputs.tolist(),
        "targets": targets.tolist(),
        "predictor_target_pairs": observed_pairs,
        "target_positions": [target_start, int(wrapped_full.size)],
        "logit_input_positions": [logit_start, int(inputs.size)],
    }


def self_test(args: argparse.Namespace) -> int:
    ensure_output_root(args.output_dir)
    _manifest, arrays = common.validate_sealed_controls()
    tokenizer = load_tokenizer(args.model_dir)
    scaffold_a = build_scaffold(tokenizer)
    scaffold_b = build_scaffold(tokenizer)
    checks: list[dict[str, Any]] = [
        {
            "name": "template_rendering_determinism_text",
            "passed": scaffold_a.metadata["rendered_empty_turn"]
            == scaffold_b.metadata["rendered_empty_turn"],
        },
        {
            "name": "template_rendering_determinism_ids",
            "passed": bool(np.array_equal(scaffold_a.prefix_ids, scaffold_b.prefix_ids)),
        },
        {
            "name": "native_template_registered_sha256",
            "passed": scaffold_a.metadata["chat_template_sha256"]
            == EXPECTED_CHAT_TEMPLATE_SHA256,
        },
        {
            "name": "one_token_deferred_closure",
            "passed": int(scaffold_a.suffix_ids.size) == 1,
        },
    ]
    wiki = arrays["wikitext_0_2560"]
    target_hashes: dict[str, str] = {}
    boundary_checks: dict[str, Any] = {}
    for length in LENGTHS:
        raw = verify_raw_source(length, wiki)
        long_arm = harmony_arm(wiki, length, "long", scaffold_a)
        ref_arm = harmony_arm(wiki, length, "reference", scaffold_a)
        target_hashes[str(length)] = long_arm["target_ids_sha256"]
        checks.extend(
            [
                {
                    "name": f"target_sha_long_reference_{length}",
                    "passed": long_arm["target_ids_sha256"]
                    == ref_arm["target_ids_sha256"],
                },
                {
                    "name": f"target_sha_raw_receipt_{length}",
                    "passed": long_arm["target_ids_sha256"]
                    == raw["target_ids_sha256"],
                },
                {
                    "name": f"predictor_sha_long_{length}",
                    "passed": long_arm["predictor_ids_sha256"]
                    == common.ramp_arm(wiki, length, "long")[
                        "predictor_ids_sha256"
                    ],
                },
                {
                    "name": f"predictor_sha_reference_{length}",
                    "passed": ref_arm["predictor_ids_sha256"]
                    == common.ramp_arm(wiki, length, "reference")[
                        "predictor_ids_sha256"
                    ],
                },
            ]
        )
        for view, arm in (("long", long_arm), ("reference", ref_arm)):
            document = wiki[
                int(arm["source_start"]) : int(arm["source_stop_exclusive"])
            ]
            boundary = verify_template_boundary(tokenizer, scaffold_a, document)
            boundary_checks[f"{length}_{view}"] = boundary
            checks.append(
                {
                    "name": f"native_template_exact_partition_{length}_{view}",
                    "passed": boundary["exact_prefix_document_suffix_partition"],
                }
            )
    alignment = _synthetic_alignment()
    checks.append(
        {
            "name": "synthetic_causal_alignment_math",
            "passed": alignment["passed"],
            "detail": alignment,
        }
    )
    status = "pass" if all(row["passed"] for row in checks) else "fail"
    core = {
        "schema": "glc_p0d_self_test_v1",
        "status": status,
        "checks": checks,
        "boundary_checks": boundary_checks,
        "target_ids_sha256": target_hashes,
        "scaffold": scaffold_a.metadata,
        "control_arrays_sha256": common.sha256_file(common.CONTROL_ARRAYS),
        "script_sha256": common.sha256_file(SCRIPT_PATH),
    }
    path = _content_addressed_json(OUTPUT_ROOT / "self_test", "p0d", core)
    print(json.dumps({"status": status, "receipt": str(path)}), flush=True)
    return 0 if status == "pass" else 1


def _already_complete(path: Path) -> bool:
    if not path.exists():
        return False
    prior = common.read_json(path)
    if prior.get("status") == "complete":
        print(json.dumps({"status": "already_complete", "receipt": str(path)}))
        return True
    raise FileExistsError(f"append-only failed attempt exists; choose --attempt: {path}")


def score_raw_cell(args: argparse.Namespace, path: Path) -> int:
    started = time.perf_counter()
    receipt: dict[str, Any] = {
        "schema": "glc_p0d_raw_verification_cell_v1",
        "created_at": common.now_iso(),
        "status": "starting",
        "format": "raw",
        "length": int(args.length),
        "attempt": int(args.attempt),
        "argv": sys.argv,
        "model_revision": common.MODEL_REVISION,
        "control_arrays": str(common.CONTROL_ARRAYS),
        "control_arrays_sha256": common.sha256_file(common.CONTROL_ARRAYS),
        "raw_recompute": False,
        "runtime": common.runtime_facts(),
        "script_sha256": common.sha256_file(SCRIPT_PATH),
    }
    try:
        _manifest, arrays = common.validate_sealed_controls()
        verification = verify_raw_source(int(args.length), arrays["wikitext_0_2560"])
        receipt.update(
            {
                "status": "complete",
                "verification": {
                    key: verification[key]
                    for key in (
                        "method",
                        "recomputed_model_forward",
                        "source_receipt",
                        "source_receipt_sha256",
                        "source_receipt_expected_sha256",
                        "checks",
                        "all_checks_passed",
                    )
                },
                "target_ids_sha256": verification["target_ids_sha256"],
                "arms": verification["arms"],
                "delta_mean_nll_nats": verification["delta_mean_nll_nats"],
                "wall_seconds_total": float(time.perf_counter() - started),
            }
        )
        common.write_new_json(path, receipt)
        print(
            json.dumps(
                {
                    "status": "complete",
                    "receipt": str(path),
                    "delta_mean_nll_nats": receipt["delta_mean_nll_nats"],
                    "raw_recompute": False,
                }
            ),
            flush=True,
        )
        return 0
    except BaseException as exc:
        receipt.update(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "wall_seconds_total": float(time.perf_counter() - started),
            }
        )
        common.write_new_json(path, receipt)
        raise


def score_harmony_cell(args: argparse.Namespace, path: Path) -> int:
    model_dir = common.ensure_model_dir(args.model_dir)
    _manifest, arrays = common.validate_sealed_controls()
    tokenizer = load_tokenizer(model_dir)
    scaffold = build_scaffold(tokenizer)
    wiki = arrays["wikitext_0_2560"]
    arms = {
        view: harmony_arm(wiki, int(args.length), view, scaffold)
        for view in ("reference", "long")
    }
    raw_source = verify_raw_source(int(args.length), wiki)
    for view in ("reference", "long"):
        if arms[view]["target_ids_sha256"] != raw_source["target_ids_sha256"]:
            raise RuntimeError(f"Harmony {args.length}/{view} target SHA != raw receipt")

    receipt: dict[str, Any] = {
        "schema": "glc_p0d_harmony_hf_cell_v1",
        "created_at": common.now_iso(),
        "status": "starting",
        "format": "harmony",
        "length": int(args.length),
        "attempt": int(args.attempt),
        "argv": sys.argv,
        "plan": str(common.PLAN_PATH),
        "addendum": str(ADDENDUM_PATH),
        "order": str(ORDER_PATH),
        "model_dir": str(model_dir),
        "model_revision": common.MODEL_REVISION,
        "control_arrays": str(common.CONTROL_ARRAYS),
        "control_arrays_sha256": common.sha256_file(common.CONTROL_ARRAYS),
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
        "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE"),
        "scaffold": scaffold.metadata,
        "scaffold_design": SCAFFOLD_DESIGN,
        "raw_source_verification": {
            key: raw_source[key]
            for key in (
                "method",
                "recomputed_model_forward",
                "source_receipt",
                "source_receipt_sha256",
                "target_ids_sha256",
                "delta_mean_nll_nats",
                "all_checks_passed",
            )
        },
        "arms": {view: _arm_json(arm) for view, arm in arms.items()},
        "runtime_preload": common.runtime_facts(),
        "script_sha256": common.sha256_file(SCRIPT_PATH),
    }
    started = time.perf_counter()
    signal.signal(signal.SIGTERM, hf_reference._timeout_handler)
    try:
        import torch

        if torch.cuda.is_available():
            raise RuntimeError(
                "GLC P0.d Harmony reference is CPU-only; hide CUDA devices"
            )
        torch.set_num_threads(int(args.threads))
        load_started = time.perf_counter()
        model, load_info = hf_reference.common.load_hf_reference_model(
            model_dir=model_dir,
            cpu_memory=args.cpu_memory,
            offload_dir=args.offload_dir,
            for_causal_lm=True,
        )
        receipt["load"] = {
            **load_info,
            "wall_seconds": float(time.perf_counter() - load_started),
            "max_rss_mib_after_load": hf_reference.max_rss_mib(),
        }

        for view in ("reference", "long"):
            data = arms[view]
            arm_started = time.perf_counter()
            input_ids = torch.from_numpy(data["inputs"])
            with torch.inference_mode():
                outputs = model(
                    input_ids=input_ids,
                    use_cache=False,
                    logits_to_keep=common.N_TARGETS,
                    return_dict=True,
                )
            logits = outputs.logits
            score = hf_reference.nll_from_logits(
                logits, data["targets"], int(args.logit_chunk)
            )
            if score["target_ids_sha256"] != data["target_ids_sha256"]:
                raise RuntimeError("HF Harmony scorer changed the target SHA")
            receipt["arms"][view].update(
                {
                    "score": score,
                    "logits_shape": [int(value) for value in logits.shape],
                    "logits_dtype": str(logits.dtype),
                    "wall_seconds": float(time.perf_counter() - arm_started),
                    "max_rss_mib": hf_reference.max_rss_mib(),
                }
            )
            del input_ids, outputs, logits
            gc.collect()

        delta = float(
            receipt["arms"]["long"]["score"]["mean_nll"]
            - receipt["arms"]["reference"]["score"]["mean_nll"]
        )
        receipt.update(
            {
                "status": "complete",
                "delta_mean_nll_nats": delta,
                "cell_band": (
                    "supported_band"
                    if delta <= HARMONY_SUPPORTED_MAX_NATS
                    else "refuted_band"
                    if delta >= HARMONY_REFUTED_MIN_NATS
                    else "mixed_band"
                ),
                "wall_seconds_total": float(time.perf_counter() - started),
                "max_rss_mib": hf_reference.max_rss_mib(),
            }
        )
        common.write_new_json(path, receipt)
        print(
            json.dumps(
                {
                    "status": "complete",
                    "receipt": str(path),
                    "delta_mean_nll_nats": delta,
                    "cell_band": receipt["cell_band"],
                }
            ),
            flush=True,
        )
        return 0
    except BaseException as exc:
        receipt.update(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "wall_seconds_total": float(time.perf_counter() - started),
                "max_rss_mib": hf_reference.max_rss_mib(),
            }
        )
        common.write_new_json(path, receipt)
        raise


def score_cell(args: argparse.Namespace) -> int:
    ensure_output_root(args.output_dir)
    if args.format is None or args.length is None:
        raise ValueError("score-cell requires --format and --length")
    if args.attempt < 0:
        raise ValueError("--attempt must be nonnegative")
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    if args.logit_chunk <= 0:
        raise ValueError("--logit-chunk must be positive")
    path = receipt_path(args.format, args.length, args.attempt)
    if _already_complete(path):
        return 0
    if args.format == "raw":
        return score_raw_cell(args, path)
    return score_harmony_cell(args, path)


def _analysis_cell(
    cell_format: str, length: int, wikitext: np.ndarray
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    directory = HARMONY_OUTPUT if cell_format == "harmony" else RAW_OUTPUT
    found = common.latest_complete_receipt(directory, f"{cell_format}_{length}")
    if found is None:
        return (
            {"format": cell_format, "length": int(length), "status": "missing"},
            None,
        )
    path, receipt = found
    if receipt.get("format") != cell_format or receipt.get("length") != int(length):
        raise RuntimeError(f"cell identity mismatch in {path}")
    if receipt.get("model_revision") != common.MODEL_REVISION:
        raise RuntimeError(f"model revision mismatch in {path}")
    long_arm = receipt.get("arms", {}).get("long", {})
    ref_arm = receipt.get("arms", {}).get("reference", {})
    expected = common.ramp_arm(wikitext, length, "long")["target_ids_sha256"]
    for view, arm in (("long", long_arm), ("reference", ref_arm)):
        if arm.get("target_ids_sha256") != expected:
            raise RuntimeError(f"{path} {view} target SHA differs from sealed target")
        score = arm.get("score", {})
        if score.get("target_ids_sha256") != expected:
            raise RuntimeError(f"{path} {view} score target SHA differs")
        if score.get("token_count") != common.N_TARGETS:
            raise RuntimeError(f"{path} {view} score count differs")
    delta = float(long_arm["score"]["mean_nll"] - ref_arm["score"]["mean_nll"])
    if abs(delta - float(receipt["delta_mean_nll_nats"])) > 1.0e-12:
        raise RuntimeError(f"{path} delta does not recompute")
    row = {
        "format": cell_format,
        "length": int(length),
        "status": "complete",
        "receipt": str(path),
        "receipt_sha256": common.sha256_file(path),
        "long_mean_nll": float(long_arm["score"]["mean_nll"]),
        "long_ppl": float(long_arm["score"]["ppl"]),
        "reference_mean_nll": float(ref_arm["score"]["mean_nll"]),
        "reference_ppl": float(ref_arm["score"]["ppl"]),
        "delta_mean_nll_nats": delta,
        "target_ids_sha256": expected,
        "raw_recomputed": bool(receipt.get("raw_recompute", True))
        if cell_format == "raw"
        else None,
    }
    scaffold = receipt.get("scaffold") if cell_format == "harmony" else None
    return row, scaffold


def _latest_passing_self_test() -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    current_script_sha = common.sha256_file(SCRIPT_PATH)
    for path in sorted((OUTPUT_ROOT / "self_test").glob("p0d_*.json")):
        payload = common.read_json(path)
        if (
            payload.get("status") == "pass"
            and payload.get("script_sha256") == current_script_sha
        ):
            candidates.append(
                {
                    "receipt": str(path),
                    "receipt_sha256": common.sha256_file(path),
                    "schema": payload.get("schema"),
                    "script_sha256": current_script_sha,
                }
            )
    return candidates[-1] if candidates else None


def _analysis_markdown(core: dict[str, Any]) -> str:
    lines = [
        "# GLC P0.d Harmony-Format Discriminator",
        "",
        "Evidence class: HuggingFace BF16 CPU inference on the pinned snapshot, "
        "plus SHA-verified reuse of the registered raw P0.a attempt01 receipts.",
        "",
        f"Discriminator status: **{core['discriminator_status']}**.",
        "",
        "Registered reading (verbatim):",
        "",
        REGISTERED_SENTENCE,
        "",
        "| Format | WikiText length | long NLL | short NLL | delta nats | target SHA-256 | receipt |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for row in core["cells"]:
        if row["status"] != "complete":
            lines.append(
                f"| {row['format']} | {row['length']} | n/a | n/a | n/a | n/a | NOT_MEASURED |"
            )
            continue
        lines.append(
            f"| {row['format']} | {row['length']} | {row['long_mean_nll']:.9f} | "
            f"{row['reference_mean_nll']:.9f} | {row['delta_mean_nll_nats']:.9f} | "
            f"`{row['target_ids_sha256']}` | `{row['receipt']}` |"
        )
    lines.extend(
        [
            "",
            "## Scaffold design",
            "",
            SCAFFOLD_DESIGN,
            "",
            "The WikiText lengths label the sealed document spans. The model input "
            "also contains the recorded Harmony prefix; the deferred closure is not "
            "part of the model input.",
            "",
        ]
    )
    for item in core["scaffolds"]:
        scaffold = item["scaffold"]
        lengths = ", ".join(str(value) for value in item["lengths"])
        lines.extend(
            [
                f"### Render used for length(s) {lengths}",
                "",
                f"- Chat template SHA-256: `{scaffold['chat_template_sha256']}`",
                f"- Inserted prefix tokens: {scaffold['inserted_prefix_token_count']}",
                f"- Deferred closing suffix tokens: {scaffold['deferred_closing_suffix_token_count']}",
                f"- Complete empty-user render tokens: {scaffold['rendered_empty_turn_token_count']}",
                "- Rendered inserted scaffold (exact excerpt; this is the complete inserted prefix):",
                "",
                "```text",
                scaffold["inserted_prefix_text_excerpt"],
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Raw arms",
            "",
            "Raw cells use verify-by-SHA, not recomputation. Each P0.a attempt01 "
            "file hash is pinned, and every long/reference sequence, input, predictor, "
            "target, score-target, token-count, NLL-sum, perplexity, and delta contract "
            "is checked against the sealed controls.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(args: argparse.Namespace) -> int:
    ensure_output_root(args.output_dir)
    _manifest, arrays = common.validate_sealed_controls()
    wiki = arrays["wikitext_0_2560"]
    cells: list[dict[str, Any]] = []
    scaffold_rows: list[tuple[int, dict[str, Any]]] = []
    for cell_format in FORMATS:
        for length in LENGTHS:
            row, scaffold = _analysis_cell(cell_format, length, wiki)
            cells.append(row)
            if scaffold is not None:
                scaffold_rows.append((length, scaffold))

    if not scaffold_rows:
        fallback = build_scaffold(load_tokenizer(args.model_dir)).metadata
        scaffold_rows = [(length, fallback) for length in LENGTHS]
    grouped: dict[str, dict[str, Any]] = {}
    for length, scaffold in scaffold_rows:
        key = scaffold["inserted_prefix_ids_sha256"]
        if key not in grouped:
            grouped[key] = {"lengths": [], "scaffold": scaffold}
        grouped[key]["lengths"].append(int(length))
    scaffolds = list(grouped.values())

    complete = all(row["status"] == "complete" for row in cells)
    if not complete:
        discriminator_status = "NOT_MEASURED"
    else:
        harmony_deltas = [
            row["delta_mean_nll_nats"]
            for row in cells
            if row["format"] == "harmony"
        ]
        raw_deltas = [
            row["delta_mean_nll_nats"] for row in cells if row["format"] == "raw"
        ]
        if any(value >= HARMONY_REFUTED_MIN_NATS for value in harmony_deltas):
            discriminator_status = "REFUTED"
        elif all(
            value <= HARMONY_SUPPORTED_MAX_NATS for value in harmony_deltas
        ) and all(value >= RAW_REQUIRED_MIN_NATS for value in raw_deltas):
            discriminator_status = "SUPPORTED"
        else:
            discriminator_status = "MIXED"

    passing_self_test = _latest_passing_self_test()
    core = {
        "schema": "glc_p0d_analysis_v1",
        "status": "complete" if complete else "incomplete",
        "discriminator_status": discriminator_status,
        "registered_sentence": REGISTERED_SENTENCE,
        "complete_cells": sum(row["status"] == "complete" for row in cells),
        "registered_cells": 4,
        "cells": cells,
        "scaffolds": scaffolds,
        "scaffold_design": SCAFFOLD_DESIGN,
        "raw_mode": "verify_attempt01_by_sha_no_recompute",
        "self_test": passing_self_test,
        "model_revision": common.MODEL_REVISION,
        "control_arrays_sha256": common.sha256_file(common.CONTROL_ARRAYS),
        "plan": str(common.PLAN_PATH),
        "addendum": str(ADDENDUM_PATH),
        "order": str(ORDER_PATH),
        "script_sha256": common.sha256_file(SCRIPT_PATH),
    }
    digest = common.canonical_json_sha256(core)[:16]
    json_path = OUTPUT_ROOT / "analysis" / f"p0d_{digest}.json"
    report_path = OUTPUT_ROOT / f"GLC_P0D_REPORT_{digest}.md"
    if not json_path.exists():
        common.write_new_json(json_path, {**core, "created_at": common.now_iso()})
    common.write_once_text(report_path, _analysis_markdown(core))
    print(
        json.dumps(
            {
                "status": core["status"],
                "discriminator_status": discriminator_status,
                "analysis": str(json_path),
                "report": str(report_path),
                "registered_sentence": REGISTERED_SENTENCE,
            }
        ),
        flush=True,
    )
    if args.require_complete and not complete:
        return 2
    return 0


def main() -> int:
    args = parse_args()
    if args.mode == "preflight":
        return preflight(args)
    if args.mode == "self-test":
        return self_test(args)
    if args.mode == "score-cell":
        return score_cell(args)
    if args.mode == "analyze":
        return analyze(args)
    raise AssertionError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
