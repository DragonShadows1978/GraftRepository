"""CPU-only tests for the APAMQ-FC receipt protocol."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "apamq_fc_ppl", REPO / "scripts" / "apamq_fc_ppl.py")
assert SPEC is not None and SPEC.loader is not None
fc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fc)


def test_seeded_document_selection_is_exact_and_repeatable() -> None:
    token_count = fc.TOKENS_PER_DOCUMENT * 28 + 7
    tokens = np.arange(token_count, dtype=np.int64)
    left, left_meta = fc.select_documents(tokens)
    right, right_meta = fc.select_documents(tokens.copy())
    assert len(left) == fc.DOCUMENT_COUNT == 25
    assert left_meta == right_meta
    assert left_meta["selected_token_count"] == 25 * 10241
    assert left_meta["scored_token_count"] == 51200
    assert len({r["source_block_index"] for r in left_meta["documents"]}) == 25
    assert all(np.array_equal(a, b) for a, b in zip(left, right))


def test_engine_root_and_arm_matrix() -> None:
    assert fc.resolve_tc_root("standard", {}) == fc.CANONICAL_TC_ROOT.resolve()
    assert fc.resolve_tc_root("apa_fused", {}) == fc.CANONICAL_TC_ROOT.resolve()
    assert fc.resolve_tc_root("apa_int4", {}) == fc.FA_TC_ROOT.resolve()
    assert fc.expected_decode_branch("standard") is None
    assert fc.expected_decode_branch("apa_blend") == "blend"
    assert fc.expected_decode_branch("apa_fused") == "fused"
    assert fc.expected_decode_branch("apa_int4") == "int4"
    assert fc.arm_config("apa_int4")["GEMMA4_QUANT_KV4"] == "0"


class _Tensor:
    def __init__(self, dtype: str = "bfloat16") -> None:
        self.dtype = dtype


class _KVRing:
    def __init__(self, *, with_kq: bool) -> None:
        self.count = fc.PREFILL_TOKENS + fc.SCORED_TOKENS_PER_DOCUMENT
        self.cap = 16384
        self.kb = _Tensor()
        self.vb = _Tensor()
        self.kqb = _Tensor() if with_kq else None
        self.kq_count = self.count if with_kq else 0


class _Gemma:
    KVRing = _KVRing


def _fake_caches(*, with_kq: bool) -> list[object]:
    caches: list[object] = [object() for _ in range(48)]
    for layer in fc.GLOBAL_LAYERS:
        caches[layer] = _KVRing(with_kq=with_kq)
    return caches


def test_ring_receipt_distinguishes_bf16_kq_and_int4_no_ring() -> None:
    blend = fc.ring_receipt(_fake_caches(with_kq=True), _Gemma, "apa_blend", 0)
    int4 = fc.ring_receipt(_fake_caches(with_kq=False), _Gemma, "apa_int4", 0)
    assert blend["expected_derived_kq"] == "bf16-present"
    assert int4["expected_derived_kq"] == "absent"
    with pytest.raises(AssertionError, match="forbidden kqb"):
        fc.ring_receipt(_fake_caches(with_kq=True), _Gemma, "apa_int4", 0)


def test_engagement_requires_exact_active_branch_census() -> None:
    runtime = {
        "branch_census": fc.empty_branch_census(),
        "global_attention_calls": {"prefill": 100,
                                   "decode": fc.EXPECTED_APA_DECODE_CALLS},
        "decode_attention_lengths": {"1": fc.EXPECTED_APA_DECODE_CALLS},
        "teacher_forced_model_calls": fc.EXPECTED_SCORED_TOKENS,
    }
    runtime["branch_census"]["prefill"]["blend"] = 8
    runtime["branch_census"]["prefill"]["fused"] = 8
    runtime["branch_census"]["decode"]["fused"] = fc.EXPECTED_APA_DECODE_CALLS
    receipt = fc.engagement_receipt(
        runtime, "apa_fused", [{"assertion": "PASS"}] * fc.DOCUMENT_COUNT)
    assert receipt["APA_ENGAGED"] is True
    runtime["branch_census"]["decode"]["blend"] = 1
    with pytest.raises(AssertionError, match="decode branch census"):
        fc.engagement_receipt(
            runtime, "apa_fused", [{"assertion": "PASS"}] * fc.DOCUMENT_COUNT)


def _receipt(arm: str, offset: float = 0.0) -> dict:
    branch = fc.expected_decode_branch(arm)
    census = {"blend": 0, "fused": 0, "int4": 0}
    if branch is not None:
        census[branch] = fc.EXPECTED_APA_DECODE_CALLS
    prefill = {"blend": 0, "fused": 0, "int4": 0}
    if arm in ("apa_blend", "apa_fused"):
        prefill.update({"blend": 8, "fused": 8})
    elif arm == "apa_int4":
        prefill.update({"blend": 8, "int4": 8})
    documents = []
    for index in range(fc.DOCUMENT_COUNT):
        mean_nll = 5.0 + index * 0.001 + offset
        documents.append({
            "document_index": index,
            "source_block_index": index,
            "token_ids_int64_sha256": f"hash-{index}",
            "scored_tokens": fc.SCORED_TOKENS_PER_DOCUMENT,
            "mean_nll": mean_nll,
        })
    mean_nll = sum(x["mean_nll"] for x in documents) / len(documents)
    return {
        "schema": fc.SCHEMA,
        "arm": arm,
        "status": "complete",
        "scored_tokens": fc.EXPECTED_SCORED_TOKENS,
        "mean_nll": mean_nll,
        "ppl": math.exp(mean_nll),
        "decode_ms_per_token": 12.5,
        "protocol": {"fingerprint_sha256": "same-protocol"},
        "code_identity": fc.current_code_identity(),
        "documents": documents,
        "engagement": {
            "assertion": "PASS",
            "APA_ENGAGED": arm != "standard",
            "apa_active_decode_calls": sum(census.values()),
            "global_l1_decode_calls": fc.EXPECTED_APA_DECODE_CALLS,
            "teacher_forced_model_calls": fc.EXPECTED_SCORED_TOKENS,
            "branch_census": {"prefill": prefill, "decode": census},
            "prefill_mixed_dispatch_assertion": "PASS",
            "ring_assertions": {"all_pass": True},
        },
    }


def test_summary_is_paired_and_contains_no_verdict(tmp_path: Path) -> None:
    offsets = {"standard": 0.0, "apa_blend": 0.02,
               "apa_fused": 0.01, "apa_int4": 0.03}
    for arm, offset in offsets.items():
        (tmp_path / f"{arm}.json").write_text(json.dumps(_receipt(arm, offset)))
    receipts = fc.read_receipts(tmp_path)
    summary = fc.build_summary(receipts)
    assert summary["adjudication"] == "not_performed"
    assert summary["verdicts"] is None
    fused = next(row for row in summary["rows"] if row["arm"] == "apa_fused")
    assert fused["vs_standard"]["paired_document_count"] == 25
    assert fused["vs_standard"]["aggregate_relative_ppl_delta_percent"] == pytest.approx(
        100.0 * math.expm1(0.01))
    assert fused["vs_apa_blend"]["aggregate_relative_ppl_delta_percent"] == pytest.approx(
        100.0 * math.expm1(-0.01))
    assert fused["vs_standard"]["paired_nll_delta_sample_std"] == pytest.approx(0.0)


def test_summary_rejects_unengaged_apa_receipt(tmp_path: Path) -> None:
    for arm in fc.MEASUREMENT_ARMS:
        receipt = _receipt(arm)
        if arm == "apa_fused":
            receipt["engagement"]["APA_ENGAGED"] = False
        (tmp_path / f"{arm}.json").write_text(json.dumps(receipt))
    with pytest.raises(AssertionError, match="APA engagement false"):
        fc.read_receipts(tmp_path)
