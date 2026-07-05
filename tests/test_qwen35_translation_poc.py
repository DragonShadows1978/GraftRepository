import json
import inspect
from pathlib import Path
import subprocess
from types import SimpleNamespace
import sys

import numpy as np
import pytest

import core.qwen35_translation_poc as poc
from core.qwen35_translation_poc import (
    SourceValidationError,
    analyze_binding_eval,
    build_corpus_plan,
    choose_next_capture_role,
    evaluate_capture_identity,
    evaluate_translator,
    fit_ridge_translator,
    inspect_pipeline_status,
    make_binding_probe_set,
    refresh_capture_manifest,
    run_pipeline_next,
    validate_unquantized_source,
    validate_weight_pair,
    write_binding_probe_set,
    write_capture_shard,
    write_weight_manifest,
    _candidate_scores_from_logprobs,
    _translate_harvested_capture,
)


LOCAL_QWEN35_9B = Path(
    "/home/vader/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/"
    "snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
)
LOCAL_QWEN35_2B = Path(
    "/home/vader/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/"
    "snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc"
)


class TinyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(ch) % 97 for ch in text if not ch.isspace()]


def _make_fake_qwen35_dir(tmp_path, name, *, tokenizer="same", config_extra=None):
    root = tmp_path / name
    root.mkdir()
    config = {
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 1024,
            "num_hidden_layers": 4,
            "full_attention_interval": 2,
            "head_dim": 128,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "vocab_size": 32000,
        },
    }
    if config_extra:
        config.update(config_extra)
    (root / "config.json").write_text(json.dumps(config))
    (root / "tokenizer.json").write_text(
        json.dumps({"version": "1.0", "payload": tokenizer})
    )
    (root / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "QwenTokenizer"})
    )
    (root / "model.safetensors").write_bytes(b"fake-safetensors-shard")
    return root


def test_validate_unquantized_source_accepts_hf_safetensors_dir(tmp_path):
    model_dir = _make_fake_qwen35_dir(tmp_path, "qwen35-source")

    manifest = validate_unquantized_source(model_dir, role="source")

    assert manifest.role == "source"
    assert manifest.repository is None
    assert manifest.revision is None
    assert manifest.model_type == "qwen3_5"
    assert manifest.text_model_type == "qwen3_5_text"
    assert manifest.shard_count == 1
    assert manifest.total_safetensor_bytes == len(b"fake-safetensors-shard")
    assert manifest.text_shape["hidden_size"] == 1024
    assert manifest.text_shape["num_key_value_heads"] == 2
    assert manifest.text_shape["attention_layer_indices"] == [1, 3]
    assert len(manifest.tokenizer_sha256) == 64


def test_validate_unquantized_source_rejects_gguf_file(tmp_path):
    gguf = tmp_path / "Qwen3.5-9B.Q8_0.gguf"
    gguf.write_bytes(b"not-for-this-poc")

    with pytest.raises(SourceValidationError, match="GGUF"):
        validate_unquantized_source(gguf, role="source")


def test_validate_unquantized_source_rejects_quantization_config(tmp_path):
    model_dir = _make_fake_qwen35_dir(
        tmp_path,
        "qwen35-int4",
        config_extra={"quantization_config": {"bits": 4}},
    )

    with pytest.raises(SourceValidationError, match="quantization_config"):
        validate_unquantized_source(model_dir, role="source")


def test_validate_weight_pair_aborts_on_tokenizer_mismatch(tmp_path):
    source = _make_fake_qwen35_dir(tmp_path, "source", tokenizer="source")
    target = _make_fake_qwen35_dir(tmp_path, "target", tokenizer="target")

    with pytest.raises(SourceValidationError, match="tokenizer.json hashes differ"):
        validate_weight_pair(source, target)


def test_write_weight_manifest_records_pair_identity(tmp_path):
    source = _make_fake_qwen35_dir(tmp_path, "source", tokenizer="shared")
    target = _make_fake_qwen35_dir(tmp_path, "target", tokenizer="shared")
    out = tmp_path / "weights_manifest.json"

    manifest = write_weight_manifest(source, target, out)
    saved = json.loads(out.read_text())

    assert manifest["schema"] == "qwen35_graft_translation_weights_v1"
    assert saved["schema"] == manifest["schema"]
    assert saved["tokenizer_sha256"] == manifest["tokenizer_sha256"]
    assert saved["source"]["role"] == "source"
    assert saved["target"]["role"] == "target"
    assert saved["source"]["shard_count"] == 1
    assert saved["target"]["shard_count"] == 1
    assert saved["tokenizer_config_sha256_match"] is True


def test_write_capture_shard_records_layer_payloads(tmp_path):
    captured = [None] * 4
    queries = [None] * 4
    captured[3] = {
        "k": np.arange(1 * 2 * 5 * 4, dtype=np.float32).reshape(1, 2, 5, 4),
        "v": np.ones((1, 2, 5, 4), dtype=np.float32),
    }
    queries[3] = np.zeros((1, 8, 5, 4), dtype=np.float32)

    manifest = write_capture_shard(
        tmp_path,
        role="target",
        doc_id="doc-a",
        chunk_id=7,
        token_ids=[10, 11, 12, 13, 14],
        captured=captured,
        queries=queries,
        position_offset=128,
    )

    assert manifest["schema"] == "qwen35_graft_translation_capture_shard_v1"
    assert manifest["role"] == "target"
    assert manifest["doc_id"] == "doc-a"
    assert manifest["chunk_id"] == 7
    assert manifest["token_count"] == 5
    assert manifest["position_offset"] == 128
    assert manifest["has_queries"] is True
    assert manifest["layers"] == [{
        "layer": 3,
        "k_shape": [1, 2, 5, 4],
        "v_shape": [1, 2, 5, 4],
        "q_shape": [1, 8, 5, 4],
        "dtype": "float16",
    }]

    sidecar = Path(manifest["npz"]).with_suffix(".json")
    assert json.loads(sidecar.read_text()) == manifest
    data = np.load(manifest["npz"])
    assert data["token_ids"].tolist() == [10, 11, 12, 13, 14]
    assert data["position_offset"].tolist() == [128]
    assert data["layer_indices"].tolist() == [3]
    assert data["l3_k"].shape == (1, 2, 5, 4)
    assert data["l3_v"].shape == (1, 2, 5, 4)
    assert data["l3_q"].shape == (1, 8, 5, 4)


def test_build_corpus_plan_chunks_and_splits_by_document(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("alpha beta gamma")
    (corpus / "b.txt").write_text("delta epsilon zeta")

    plan = build_corpus_plan(
        [corpus],
        TinyTokenizer(),
        chunk_tokens=5,
        heldout_fraction=0.0,
        seed="unit",
        source_label="unit-corpus",
    )

    assert plan["schema"] == "qwen35_graft_translation_corpus_plan_v1"
    assert plan["source_label"] == "unit-corpus"
    assert plan["split_granularity"] == "document"
    assert plan["totals"]["documents"] == 2
    assert plan["totals"]["chunks"] >= 2
    assert plan["totals"]["heldout_tokens"] == 0
    assert plan["totals"]["train_tokens"] == plan["totals"]["tokens"]
    assert {d["split"] for d in plan["documents"]} == {"train"}
    assert {c["split"] for c in plan["chunks"]} == {"train"}
    assert all(c["token_count"] <= 5 for c in plan["chunks"])
    assert all(c["token_ids"] for c in plan["chunks"])


def test_refresh_capture_manifest_reports_expected_completion(tmp_path):
    plan = build_corpus_plan(
        [tmp_path],
        TinyTokenizer(),
        chunk_tokens=4,
        heldout_fraction=0.0,
    )
    # No docs exist yet, so build a one-chunk synthetic plan directly.
    plan["documents"] = [{
        "doc_id": "doc1",
        "doc_index": 0,
        "label": "doc1",
        "source_path": "doc1.txt",
        "text_sha256": "00",
        "split": "train",
        "token_count": 4,
    }]
    plan["chunks"] = [{
        "chunk_id": 0,
        "doc_id": "doc1",
        "doc_index": 0,
        "split": "train",
        "source_path": "doc1.txt",
        "label": "doc1",
        "token_start": 0,
        "token_end": 4,
        "token_count": 4,
        "token_ids": [1, 2, 3, 4],
    }]
    plan["totals"] = {
        "documents": 1,
        "chunks": 1,
        "tokens": 4,
        "train_tokens": 4,
        "heldout_tokens": 0,
    }
    plan_path = tmp_path / "corpus_plan.json"
    plan_path.write_text(json.dumps(plan))
    out_dir = tmp_path / "captures"
    captured = [None] * 4
    captured[3] = {
        "k": np.zeros((1, 2, 4, 8), dtype=np.float32),
        "v": np.zeros((1, 2, 4, 8), dtype=np.float32),
    }

    write_capture_shard(
        out_dir,
        role="source",
        doc_id="doc1",
        chunk_id=0,
        token_ids=[1, 2, 3, 4],
        captured=captured,
        metadata={"split": "train"},
    )
    manifest = refresh_capture_manifest(out_dir, plan_path)

    assert manifest["schema"] == "qwen35_graft_translation_capture_manifest_v1"
    assert manifest["roles"]["source"]["shards"] == 1
    assert manifest["roles"]["source"]["splits"]["train"]["tokens"] == 4
    assert manifest["expected"]["source"]["completed_chunks"] == 1
    assert manifest["expected"]["source"]["remaining_chunks"] == 0
    assert manifest["expected"]["source"]["next_missing_chunk"] is None
    assert manifest["expected"]["source"]["complete"] is True
    assert manifest["expected"]["target"]["completed_chunks"] == 0
    assert manifest["expected"]["target"]["remaining_chunks"] == 1
    assert manifest["expected"]["target"]["next_missing_chunk"] == 0
    assert manifest["expected"]["target"]["complete"] is False
    assert manifest["paired"]["shards"] == 0
    assert manifest["paired"]["source_only_chunks"] == 1

    queries = [None] * 4
    queries[3] = np.zeros((1, 4, 4, 8), dtype=np.float32)
    write_capture_shard(
        out_dir,
        role="target",
        doc_id="doc1",
        chunk_id=0,
        token_ids=[1, 2, 3, 4],
        captured=captured,
        queries=queries,
        metadata={"split": "train"},
    )
    manifest = refresh_capture_manifest(out_dir, plan_path)

    assert manifest["paired"]["shards"] == 1
    assert manifest["paired"]["tokens"] == 4
    assert manifest["paired"]["same_split_shards"] == 1
    assert manifest["paired"]["same_split_tokens"] == 4
    assert manifest["paired"]["splits"]["train"]["tokens"] == 4
    assert manifest["paired"]["source_only_chunks"] == 0
    assert manifest["paired"]["target_only_chunks"] == 0


def test_choose_next_capture_role_prefers_source_then_target_then_done():
    assert choose_next_capture_role({
        "expected": {
            "source": {"complete": False},
            "target": {"complete": False},
        }
    }) == "source"
    assert choose_next_capture_role({
        "expected": {
            "source": {"complete": True},
            "target": {"complete": False},
        }
    }) == "target"
    assert choose_next_capture_role({
        "expected": {
            "source": {"complete": True},
            "target": {"complete": True},
        }
    }) is None


def _write_pipeline_plan_and_shards(tmp_path, *, split="train"):
    root = tmp_path / "poc"
    capture_dir = root / "captures"
    plan = {
        "schema": "qwen35_graft_translation_corpus_plan_v1",
        "documents": [{
            "doc_id": "doc-pipe",
            "doc_index": 0,
            "label": "doc-pipe",
            "source_path": "doc-pipe.txt",
            "text_sha256": "00",
            "split": split,
            "token_count": 96,
        }],
        "chunks": [{
            "chunk_id": 0,
            "doc_id": "doc-pipe",
            "doc_index": 0,
            "split": split,
            "source_path": "doc-pipe.txt",
            "label": "doc-pipe",
            "token_start": 0,
            "token_end": 96,
            "token_count": 96,
            "token_ids": list(range(96)),
        }],
        "totals": {
            "documents": 1,
            "chunks": 1,
            "tokens": 96,
            "train_tokens": 96 if split == "train" else 0,
            "heldout_tokens": 96 if split == "heldout" else 0,
        },
    }
    plan_path = root / "corpus_plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan))
    captured = [None] * 4
    captured[3] = {
        "k": np.zeros((1, 1, 96, 2), dtype=np.float32),
        "v": np.ones((1, 1, 96, 2), dtype=np.float32),
    }
    queries = [None] * 4
    queries[3] = np.zeros((1, 1, 96, 2), dtype=np.float32)
    for role in ("source", "target"):
        write_capture_shard(
            capture_dir,
            role=role,
            doc_id="doc-pipe",
            chunk_id=0,
            token_ids=list(range(96)),
            captured=captured,
            queries=queries if role == "target" else None,
            metadata={"split": split},
            compress=False,
        )
    return root, plan_path, capture_dir


def test_pipeline_next_runs_capture_before_post_capture_work(tmp_path,
                                                             monkeypatch):
    root = tmp_path / "poc"
    capture_dir = root / "captures"
    plan = {
        "schema": "qwen35_graft_translation_corpus_plan_v1",
        "documents": [],
        "chunks": [{
            "chunk_id": 0,
            "doc_id": "doc-pipe",
            "doc_index": 0,
            "split": "train",
            "source_path": "doc-pipe.txt",
            "label": "doc-pipe",
            "token_start": 0,
            "token_end": 4,
            "token_count": 4,
            "token_ids": [1, 2, 3, 4],
        }],
        "totals": {"documents": 1, "chunks": 1, "tokens": 4},
    }
    plan_path = root / "corpus_plan.json"
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text(json.dumps(plan))

    def fake_capture_next(*args, **kwargs):
        return {
            "status": "ok",
            "selected_role": "source",
            "processed_chunks": 1,
            "skipped_existing": 0,
            "capture_manifest": {
                "out_dir": str(capture_dir),
                "expected": {
                    "source": {"complete": False},
                    "target": {"complete": False},
                },
                "roles": {},
            },
        }

    monkeypatch.setattr(poc, "run_capture_next", fake_capture_next)

    result = run_pipeline_next(
        root=root,
        plan=plan_path,
        source_model_dir="/source",
        target_model_dir="/target",
        skip_live_g0=True,
        skip_binding_eval=True,
    )

    assert result["stage"] == "capture-source"
    assert result["processed_chunks"] == 1
    status = json.loads((root / "pipeline_status.json").read_text())
    assert status["stage"] == "capture-source"
    history_lines = (root / "pipeline_history.jsonl").read_text().splitlines()
    assert len(history_lines) == 1
    history = json.loads(history_lines[0])
    assert history["schema"] == "qwen35_graft_translation_pipeline_history_v1"
    assert history["stage"] == "capture-source"
    assert history["processed_chunks"] == 1
    assert history["capture_expected"]["source"]["complete"] is False
    assert "roles" not in history


def test_pipeline_next_runs_g0_identity_after_capture_completion(tmp_path,
                                                                 monkeypatch):
    root, plan_path, _ = _write_pipeline_plan_and_shards(
        tmp_path, split="heldout")

    def fake_g0(capture_dir, out_path, *, split, topk):
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "schema": "qwen35_graft_translation_g0_capture_identity_v1",
            "target_shards": 1,
            "layers": [],
        }))
        return {"target_shards": 1, "layers": []}

    monkeypatch.setattr(poc, "evaluate_capture_identity", fake_g0)

    result = run_pipeline_next(
        root=root,
        plan=plan_path,
        source_model_dir="/source",
        target_model_dir="/target",
        skip_live_g0=True,
        skip_binding_eval=True,
    )

    assert result["stage"] == "g0-capture-identity"
    assert (root / "gates" / "g0_capture_identity_metrics.json").is_file()


def test_pipeline_next_fits_translator_after_g0_when_live_g0_skipped(
        tmp_path, monkeypatch):
    root, plan_path, _ = _write_pipeline_plan_and_shards(tmp_path)
    gates = root / "gates"
    gates.mkdir(parents=True)
    (gates / "g0_capture_identity_metrics.json").write_text(json.dumps({
        "schema": "qwen35_graft_translation_g0_capture_identity_v1",
    }))

    def fake_fit(capture_dir, out_dir, *, ridge_lambda, split,
                 control="normal", kinds="both"):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema": "qwen35_graft_translation_translator_v1",
            "control": control,
            "kinds": ["k", "v"] if kinds == "both" else [kinds],
            "paired_shards": 1,
            "artifacts": [],
        }
        (out / "translator_manifest.json").write_text(json.dumps(manifest))
        return {"translator_manifest": manifest, "fit_metrics": {}}

    monkeypatch.setattr(poc, "_run_pipeline_fit", fake_fit)

    result = run_pipeline_next(
        root=root,
        plan=plan_path,
        source_model_dir="/source",
        target_model_dir="/target",
        skip_live_g0=True,
        skip_binding_eval=True,
    )

    assert result["stage"] == "fit-translator"
    assert (root / "translator" / "translator_manifest.json").is_file()


def test_pipeline_next_reports_complete_when_optional_heavy_gates_skipped(
        tmp_path):
    root, plan_path, _ = _write_pipeline_plan_and_shards(tmp_path)
    gates = root / "gates"
    gates.mkdir(parents=True)
    (gates / "g0_capture_identity_metrics.json").write_text(json.dumps({
        "schema": "qwen35_graft_translation_g0_capture_identity_v1",
    }))
    for rel, control, kinds in (
        ("translator", "normal", ["k", "v"]),
        ("translator_wrong_layer", "wrong-layer", ["k", "v"]),
        ("translator_shuffled_docs", "shuffled-docs", ["k", "v"]),
        ("translator_k_only", "normal", ["k"]),
        ("translator_v_only", "normal", ["v"]),
    ):
        d = root / rel
        d.mkdir(parents=True)
        (d / "translator_manifest.json").write_text(json.dumps({
            "schema": "qwen35_graft_translation_translator_v1",
            "control": control,
            "kinds": kinds,
        }))
    (root / "translator" / "eval_metrics.json").write_text(json.dumps({
        "schema": "qwen35_graft_translation_eval_metrics_v1",
    }))
    write_binding_probe_set(
        gates / "binding_probes.json",
        count=4,
        seed="unit",
    )

    result = run_pipeline_next(
        root=root,
        plan=plan_path,
        source_model_dir="/source",
        target_model_dir="/target",
        skip_live_g0=True,
        skip_binding_eval=True,
    )

    assert result["status"] == "complete"
    assert result["stage"] == "complete"
    assert result["skipped"] == ["g0-logit-smoke", "eval-binding-probes"]


def test_pipeline_status_reports_next_stage_without_running_models(
        tmp_path, monkeypatch):
    root, plan_path, _ = _write_pipeline_plan_and_shards(
        tmp_path, split="heldout")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("pipeline-status must not run model work")

    monkeypatch.setattr(poc, "run_capture_next", fail_if_called)
    monkeypatch.setattr(poc, "evaluate_capture_identity", fail_if_called)
    monkeypatch.setattr(poc, "run_g0_logit_identity_smoke", fail_if_called)
    monkeypatch.setattr(poc, "_run_pipeline_fit", fail_if_called)
    monkeypatch.setattr(poc, "evaluate_translator", fail_if_called)
    monkeypatch.setattr(poc, "evaluate_binding_probes", fail_if_called)

    result = inspect_pipeline_status(
        root=root,
        plan=plan_path,
        skip_live_g0=True,
        skip_binding_eval=True,
    )

    assert result["status"] == "pending"
    assert result["stage"] == "g0-capture-identity"
    assert result["capture_manifest"]["expected"]["source"]["complete"] is True
    assert result["capture_manifest"]["expected"]["target"]["complete"] is True
    assert result["capture_manifest"]["paired"]["same_split_shards"] == 1
    assert result["artifacts"]["g0_capture_identity"]["ready"] is False


def test_pipeline_status_writes_complete_status_when_requested(tmp_path):
    root, plan_path, _ = _write_pipeline_plan_and_shards(tmp_path)
    gates = root / "gates"
    gates.mkdir(parents=True)
    (gates / "g0_capture_identity_metrics.json").write_text(json.dumps({
        "schema": "qwen35_graft_translation_g0_capture_identity_v1",
    }))
    for rel, control, kinds in (
        ("translator", "normal", ["k", "v"]),
        ("translator_wrong_layer", "wrong-layer", ["k", "v"]),
        ("translator_shuffled_docs", "shuffled-docs", ["k", "v"]),
        ("translator_k_only", "normal", ["k"]),
        ("translator_v_only", "normal", ["v"]),
    ):
        d = root / rel
        d.mkdir(parents=True)
        (d / "translator_manifest.json").write_text(json.dumps({
            "schema": "qwen35_graft_translation_translator_v1",
            "control": control,
            "kinds": kinds,
        }))
    (root / "translator" / "eval_metrics.json").write_text(json.dumps({
        "schema": "qwen35_graft_translation_eval_metrics_v1",
    }))
    write_binding_probe_set(
        gates / "binding_probes.json",
        count=4,
        seed="unit",
    )

    result = inspect_pipeline_status(
        root=root,
        plan=plan_path,
        skip_live_g0=True,
        skip_binding_eval=True,
        write_status=True,
    )

    saved = json.loads((root / "pipeline_status.json").read_text())
    assert result["status"] == "complete"
    assert saved["stage"] == "complete"
    assert saved["skipped"] == ["g0-logit-smoke", "eval-binding-probes"]


def test_translation_pipeline_shell_wrapper_is_safe_and_locked():
    script = Path("scripts/qwen35_translation_pipeline.sh")
    result = subprocess.run(
        ["bash", "-n", str(script)],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert script.stat().st_mode & 0o111

    src = script.read_text()
    assert "flock -n \"$LOCK\"" in src
    assert "pipeline-next" in src
    assert "pipeline-status" in src
    assert "pipeline_history.jsonl" in src
    assert "SOURCE_MAX_CHUNKS" in src
    assert "TARGET_MAX_CHUNKS" in src
    assert "sudo" not in src


def test_fit_ridge_translator_writes_artifacts_from_paired_shards(tmp_path):
    out_dir = tmp_path / "captures"
    x = np.array([
        [[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]]]
    ], dtype=np.float32).reshape(1, 1, 4, 2)
    y = np.concatenate([2.0 * x, -1.0 * x], axis=1)
    captured_source = [None] * 12
    captured_target = [None] * 12
    queries_target = [None] * 12
    captured_source[3] = {"k": x, "v": x + 1.0}
    captured_target[7] = {"k": y, "v": np.concatenate(
        [2.0 * (x + 1.0), -1.0 * (x + 1.0)], axis=1)}
    captured_target[11] = {"k": -y, "v": np.concatenate(
        [-2.0 * (x + 1.0), 1.0 * (x + 1.0)], axis=1)}
    queries_target[7] = y

    write_capture_shard(
        out_dir,
        role="source",
        doc_id="doc-fit",
        chunk_id=0,
        token_ids=[1, 2, 3, 4],
        captured=captured_source,
        metadata={"split": "train"},
        compress=False,
    )
    write_capture_shard(
        out_dir,
        role="target",
        doc_id="doc-fit",
        chunk_id=0,
        token_ids=[1, 2, 3, 4],
        captured=captured_target,
        queries=queries_target,
        metadata={"split": "train"},
        compress=False,
    )

    result = fit_ridge_translator(
        out_dir,
        tmp_path / "translator",
        ridge_lambda=1e-8,
        split="train",
    )

    manifest = result["translator_manifest"]
    metrics = result["fit_metrics"]
    assert manifest["schema"] == "qwen35_graft_translation_translator_v1"
    assert manifest["layer_alignment"] == [
        {"source_layer": 3, "target_layer": 7}]
    assert len(manifest["artifacts"]) == 2
    assert Path(manifest["fit_metrics"]).is_file()
    assert (tmp_path / "translator" / "translator_manifest.json").is_file()
    assert all(Path(a["path"]).is_file() for a in manifest["artifacts"])
    assert manifest["control"] == "normal"
    assert manifest["kinds"] == ["k", "v"]
    assert len(metrics["layers"]) == 2
    assert all(row["r2"] > 0.99 for row in metrics["layers"])

    wrong = fit_ridge_translator(
        out_dir,
        tmp_path / "translator_wrong_layer",
        ridge_lambda=1e-8,
        split="train",
        control="wrong-layer",
        kinds="k",
    )
    assert wrong["translator_manifest"]["control"] == "wrong-layer"
    assert wrong["translator_manifest"]["kinds"] == ["k"]
    assert wrong["translator_manifest"]["layer_alignment"] == [
        {"source_layer": 3, "target_layer": 11}]
    assert len(wrong["translator_manifest"]["artifacts"]) == 1
    assert wrong["translator_manifest"]["artifacts"][0]["kind"] == "k"

    v_only = fit_ridge_translator(
        out_dir,
        tmp_path / "translator_v_only",
        ridge_lambda=1e-8,
        split="train",
        kinds="v",
    )
    assert v_only["translator_manifest"]["kinds"] == ["v"]
    assert len(v_only["translator_manifest"]["artifacts"]) == 1
    assert v_only["translator_manifest"]["artifacts"][0]["kind"] == "v"

    with pytest.raises(ValueError, match="requires at least two pairs"):
        fit_ridge_translator(
            out_dir,
            tmp_path / "translator_shuffled",
            ridge_lambda=1e-8,
            split="train",
            control="shuffled-docs",
        )

    eval_metrics = evaluate_translator(
        out_dir,
        tmp_path / "translator",
        tmp_path / "translator" / "eval_metrics.json",
        split="train",
        topk=2,
    )
    assert eval_metrics["schema"] == "qwen35_graft_translation_eval_metrics_v1"
    assert Path(tmp_path / "translator" / "eval_metrics.json").is_file()
    progress = json.loads(
        Path(tmp_path / "translator" / "eval_metrics_progress.json").read_text()
    )
    assert progress["schema"] == "qwen35_graft_translation_eval_progress_v1"
    assert progress["status"] == "complete"
    assert progress["completed_shards"] == progress["paired_shards"]
    assert eval_metrics["layers"][0]["key_recall_at_2"] > 0.99
    assert eval_metrics["layers"][0]["wrong_layer"] == 11
    assert eval_metrics["layers"][0]["wrong_layer_key_recall_at_2"] is not None
    assert eval_metrics["layers"][0]["value_output_cosine"] > 0.99
    assert eval_metrics["layers"][0]["v_only_value_output_cosine"] > 0.99
    assert (
        eval_metrics["layers"][0]
        ["translated_attention_value_output_cosine"] > 0.99
    )
    assert (
        eval_metrics["layers"][0]
        ["k_only_native_value_output_mse"] < 1e-6
    )
    assert (
        eval_metrics["layers"][0]
        ["wrong_layer_value_output_mse"] >
        eval_metrics["layers"][0]["value_output_mse"]
    )

    identity = evaluate_capture_identity(
        out_dir,
        tmp_path / "translator" / "g0_capture_identity_metrics.json",
        split="train",
        topk=2,
    )
    assert identity["schema"] == (
        "qwen35_graft_translation_g0_capture_identity_v1")
    assert identity["evaluation_mode"] == "structural_exact_identity"
    assert identity["layers"][0]["identity_key_recall_at_2"] == 1.0
    assert identity["layers"][0]["identity_value_output_mse"] == 0.0
    assert identity["layers"][0]["identity_value_output_cosine"] > 0.99
    assert identity["layers"][0]["non_finite_tensors"] == 0


def test_topk_recall_matches_set_intersection_reference():
    rng = np.random.default_rng(1234)
    scores_a = rng.normal(size=(3, 4, 9)).astype(np.float32)
    scores_b = rng.normal(size=(3, 4, 9)).astype(np.float32)

    got = poc._topk_recall(scores_a, scores_b, 4)

    ia = np.argpartition(scores_a, -4, axis=-1)[..., -4:]
    ib = np.argpartition(scores_b, -4, axis=-1)[..., -4:]
    hits = 0
    rows = 0
    for a, b in zip(ia.reshape(-1, 4), ib.reshape(-1, 4)):
        hits += len(set(a.tolist()) & set(b.tolist()))
        rows += 1
    expected = hits / float(rows * 4)

    assert got == expected


def test_qwen35_attention_exposes_grm_injection_contract():
    sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
    from core.qwen35_tc import Qwen35AttentionTC

    src = inspect.getsource(Qwen35AttentionTC.__call__)

    assert "_capture_q" in src
    assert "_captured_q" in src
    assert "_capture" in src
    assert "_captured" in src
    assert "inject_kv" in src
    assert "graft_seats" in src
    assert "live_shift" in src
    assert "kg_pre.slice(3, 0, R)" in src
    assert "kv_cache is None" in src


def test_fit_ridge_translator_shuffled_docs_control_writes_artifact(tmp_path):
    out_dir = tmp_path / "captures"
    for chunk_id, offset in enumerate((0.0, 10.0)):
        x = np.array([
            [[[[1.0 + offset, 0.0],
               [0.0, 1.0 + offset],
               [1.0 + offset, 1.0],
               [2.0 + offset, 1.0]]]]
        ], dtype=np.float32).reshape(1, 1, 4, 2)
        y = np.concatenate([3.0 * x, -0.5 * x], axis=1)
        captured_source = [None] * 8
        captured_target = [None] * 8
        captured_source[3] = {"k": x, "v": x}
        captured_target[7] = {"k": y, "v": y}
        write_capture_shard(
            out_dir,
            role="source",
            doc_id=f"doc-shuffle-{chunk_id}",
            chunk_id=chunk_id,
            token_ids=[1, 2, 3, 4],
            captured=captured_source,
            metadata={"split": "train"},
            compress=False,
        )
        write_capture_shard(
            out_dir,
            role="target",
            doc_id=f"doc-shuffle-{chunk_id}",
            chunk_id=chunk_id,
            token_ids=[1, 2, 3, 4],
            captured=captured_target,
            metadata={"split": "train"},
            compress=False,
        )

    result = fit_ridge_translator(
        out_dir,
        tmp_path / "translator_shuffled",
        ridge_lambda=1e-8,
        split="train",
        control="shuffled-docs",
        kinds="k",
    )

    manifest = result["translator_manifest"]
    assert manifest["control"] == "shuffled-docs"
    assert manifest["kinds"] == ["k"]
    assert manifest["paired_shards"] == 2
    assert len(manifest["artifacts"]) == 1
    assert Path(manifest["artifacts"][0]["path"]).is_file()


def test_make_binding_probe_set_is_deterministic_and_serializable(tmp_path):
    probes_a = make_binding_probe_set(count=4, seed="unit")
    probes_b = make_binding_probe_set(count=4, seed="unit")
    out = tmp_path / "binding_probes.json"
    saved = write_binding_probe_set(out, count=4, seed="unit")

    assert probes_a == probes_b == saved
    assert out.is_file()
    loaded = json.loads(out.read_text())
    assert loaded["schema"] == "qwen35_graft_translation_binding_probes_v1"
    assert loaded["count"] == 4
    assert len(loaded["probes"]) == 4
    first = loaded["probes"][0]
    assert first["gold"].startswith(" ")
    assert len(first["decoys"]) == 3
    assert first["gold"] not in first["decoys"]
    assert first["entity"] in first["fact"]
    assert first["entity"] in first["question"]


def test_candidate_scores_compute_gold_minus_best_decoy():
    row = _candidate_scores_from_logprobs(
        {
            " GOLD": -1.0,
            " A": -4.0,
            " B": -2.0,
            " C": -3.0,
        },
        " GOLD",
        [" A", " B", " C"],
    )

    assert row["success"] is True
    assert row["best_decoy_score"] == -2.0
    assert row["gold_minus_best_decoy"] == 1.0


def test_analyze_binding_eval_summarizes_floor_and_misses(tmp_path):
    probes = {
        "schema": "qwen35_graft_translation_binding_probes_v1",
        "seed": "unit",
        "count": 2,
        "probes": [
            {
                "id": "bind-000",
                "entity": "Aster",
                "fact": "Project Aster has code AS-1111.",
                "question": "What code belongs to Project Aster?\nAnswer:",
                "gold": " AS-1111",
                "decoys": [" BE-2222", " CI-3333", " DO-4444"],
            },
            {
                "id": "bind-001",
                "entity": "Beryl",
                "fact": "Project Beryl has code BE-5555.",
                "question": "What code belongs to Project Beryl?\nAnswer:",
                "gold": " BE-5555",
                "decoys": [" AS-6666", " CI-7777", " DO-8888"],
            },
        ],
    }
    probes_path = tmp_path / "binding_probes.json"
    probes_path.write_text(json.dumps(probes))
    rows = []
    for probe_id, amnesia_margin, translated_margin in (
            ("bind-000", 0.25, 1.5),
            ("bind-001", 2.0, -0.5),
    ):
        for mode, margin in (
                ("amnesia", amnesia_margin),
                ("translated", translated_margin)):
            rows.append({
                "probe_id": probe_id,
                "mode": mode,
                "gold_score": margin,
                "best_decoy_score": 0.0,
                "gold_minus_best_decoy": margin,
                "success": margin > 0,
                "candidate_scores": {
                    "gold": margin,
                    "decoys": [-1.0, 0.0, -2.0],
                },
            })
    binding_eval = {
        "schema": "qwen35_graft_translation_binding_eval_v1",
        "probes_path": str(probes_path),
        "source_model": None,
        "target_model": None,
        "translator_dir": None,
        "modes": ["amnesia", "translated"],
        "probe_count": 2,
        "summaries": [],
        "rows": rows,
    }
    binding_path = tmp_path / "binding_eval_metrics.json"
    binding_path.write_text(json.dumps(binding_eval))

    analysis = analyze_binding_eval(
        binding_path,
        tmp_path / "binding_eval_analysis.json",
    )

    assert analysis["schema"] == "qwen35_graft_translation_binding_analysis_v1"
    assert analysis["probe_count"] == 2
    assert analysis["translated_misses"] == ["bind-001"]
    assert analysis["amnesia_successes"] == ["bind-000", "bind-001"]
    assert analysis["translated_beats_amnesia"] == ["bind-000"]
    assert analysis["amnesia_beats_translated"] == ["bind-001"]
    assert analysis["per_probe"][0]["modes"]["translated"]["best_decoy"] == (
        " CI-3333")
    assert analysis["per_probe"][0]["lengths"]["gold"]["chars"] == 8


def test_translate_harvested_capture_applies_artifacts_to_target_shape(tmp_path):
    source_capture = [None] * 4
    source_capture[3] = {
        "k": np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32),
        "v": np.array([[[[5.0, 6.0], [7.0, 8.0]]]], dtype=np.float32),
    }
    translator_dir = tmp_path / "translator"
    translator_dir.mkdir()
    artifacts = []
    for kind, add in (("k", 10.0), ("v", 20.0)):
        weight = np.array([
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
        ], dtype=np.float32)
        bias = np.array([add, add, -add, -add], dtype=np.float32)
        path = translator_dir / f"translator_l3_to_l7_{kind}.npz"
        np.savez(path, weight=weight, bias=bias,
                 source_layer=np.array([3], np.int64),
                 target_layer=np.array([7], np.int64),
                 kind=np.array([kind]))
        artifacts.append({
            "source_layer": 3,
            "target_layer": 7,
            "kind": kind,
            "path": str(path),
            "input_dim": 2,
            "output_dim": 4,
            "train_tokens": 2,
        })
    (translator_dir / "translator_manifest.json").write_text(json.dumps({
        "schema": "qwen35_graft_translation_translator_v1",
        "artifacts": artifacts,
    }))

    target_cfg = SimpleNamespace(num_layers=8, num_kv_heads=2, head_dim=2)
    translated = _translate_harvested_capture(
        source_capture, translator_dir, target_cfg)

    assert translated[7]["k"].shape == (1, 2, 2, 2)
    assert translated[7]["v"].shape == (1, 2, 2, 2)
    np.testing.assert_allclose(
        translated[7]["k"][0, :, 0, :],
        np.array([[11.0, 12.0], [-9.0, -8.0]], dtype=np.float32),
    )


def test_qwen35_forward_extends_rope_for_live_graft_shift():
    sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
    from core.qwen35_tc import Qwen35_TC

    src = inspect.getsource(Qwen35_TC.__call__)

    assert "live_shift = getattr(att, \"live_shift\", None)" in src
    assert "live_shift = getattr(att, \"graft_seats\", 0)" in src
    assert "self.extend_rope(position_offset + shift + L)" in src


def test_kv_graft_harvest_kv_and_queries_uses_one_forward():
    sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
    from core import kv_graft

    class FakeAttention:
        pass

    class FakeLayer:
        def __init__(self):
            self.self_attn = FakeAttention()

    class FakeModel:
        def __init__(self):
            self.layers = [FakeLayer()]
            self.calls = 0

        def __call__(self, ids, last_token_only=False):
            self.calls += 1
            att = self.layers[0].self_attn
            if getattr(att, "_capture", False):
                att._captured = (
                    np.zeros((1, 2, 3, 4), dtype=np.float32),
                    np.ones((1, 2, 3, 4), dtype=np.float32),
                )
            if getattr(att, "_capture_q", False):
                att._captured_q = np.zeros((1, 8, 3, 4), dtype=np.float32)
            return None, []

    model = FakeModel()
    captured, queries = kv_graft.harvest_kv_and_queries(
        model, [1, 2, 3], layer_filter={0})

    assert model.calls == 1
    assert captured[0]["k"].shape == (1, 2, 3, 4)
    assert captured[0]["v"].shape == (1, 2, 3, 4)
    assert queries[0].shape == (1, 8, 3, 4)
    att = model.layers[0].self_attn
    assert att._capture is False
    assert att._capture_q is False


def test_local_qwen35_9b_safetensors_source_when_available():
    if not LOCAL_QWEN35_9B.is_dir():
        pytest.skip("local Qwen3.5-9B safetensors cache is not present")

    manifest = validate_unquantized_source(LOCAL_QWEN35_9B, role="target")

    assert manifest.repository == "Qwen/Qwen3.5-9B"
    assert manifest.revision == "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    assert manifest.shard_count == 4
    assert manifest.text_shape["hidden_size"] == 4096
    assert manifest.text_shape["num_key_value_heads"] == 4
    assert manifest.text_shape["num_hidden_layers"] == 32
    assert manifest.text_shape["attention_layer_indices"] == [
        3, 7, 11, 15, 19, 23, 27, 31]


def test_qwen35_config_loads_real_2b_and_9b_shapes_when_available():
    if not LOCAL_QWEN35_2B.is_dir() or not LOCAL_QWEN35_9B.is_dir():
        pytest.skip("local Qwen3.5 2B/9B safetensors caches are not present")

    sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
    from core.qwen35_tc import Qwen35Config

    cfg_2b = Qwen35Config.from_model_dir(LOCAL_QWEN35_2B)
    cfg_9b = Qwen35Config.from_model_dir(LOCAL_QWEN35_9B)

    assert cfg_2b.repository == "Qwen/Qwen3.5-2B"
    assert cfg_2b.revision == "15852e8c16360a2fea060d615a32b45270f8a8fc"
    assert cfg_2b.hidden_dim == 2048
    assert cfg_2b.num_layers == 24
    assert cfg_2b.num_heads == 8
    assert cfg_2b.num_kv_heads == 2
    assert cfg_2b.attention_layer_indices() == [3, 7, 11, 15, 19, 23]
    assert cfg_2b.tie_word_embeddings is True

    assert cfg_9b.repository == "Qwen/Qwen3.5-9B"
    assert cfg_9b.revision == "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    assert cfg_9b.hidden_dim == 4096
    assert cfg_9b.num_layers == 32
    assert cfg_9b.num_heads == 16
    assert cfg_9b.num_kv_heads == 4
    assert cfg_9b.attention_layer_indices() == [3, 7, 11, 15, 19, 23, 27, 31]
    assert cfg_9b.tie_word_embeddings is False
