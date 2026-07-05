"""Phase-0 utilities for the Qwen3.5 graft-translation PoC.

The PoC starts from unquantized HF safetensors on both sides. These helpers
make that source-weight law executable before any GPU harvest or fitting work
runs.
"""
from dataclasses import asdict, dataclass
import argparse
import gc
import hashlib
import json
import numpy as np
from pathlib import Path
import sys
from datetime import datetime, timezone


class SourceValidationError(ValueError):
    """Raised when a model source cannot be used for this PoC."""


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load_json(path):
    with open(path) as fh:
        return json.load(fh)


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _text_config(config):
    return dict(config.get("text_config") or config)


def _hf_snapshot_identity(root):
    parts = root.parts
    if len(parts) < 3 or parts[-2] != "snapshots":
        return None, None
    repo_dir = parts[-3]
    repo_id = repo_dir
    if repo_dir.startswith("models--"):
        repo_id = repo_dir[len("models--"):].replace("--", "/")
    return repo_id, root.name


def _attention_layer_indices(text):
    layer_types = text.get("layer_types")
    if layer_types:
        return [i for i, kind in enumerate(layer_types)
                if kind == "full_attention"]
    interval = text.get("full_attention_interval")
    layers = text.get("num_hidden_layers")
    if not interval or not layers:
        return []
    return [i for i in range(int(layers)) if i % int(interval) == interval - 1]


@dataclass(frozen=True)
class SafetensorShard:
    path: str
    bytes: int


@dataclass(frozen=True)
class Qwen35SourceManifest:
    role: str
    path: str
    repository: str | None
    revision: str | None
    model_type: str
    text_model_type: str
    tokenizer_sha256: str
    tokenizer_config_sha256: str | None
    config_sha256: str
    shard_count: int
    total_safetensor_bytes: int
    shards: tuple[SafetensorShard, ...]
    text_shape: dict

    def to_json(self):
        out = asdict(self)
        out["shards"] = [asdict(s) for s in self.shards]
        return out


@dataclass(frozen=True)
class CaptureLayerRecord:
    layer: int
    k_shape: tuple[int, ...]
    v_shape: tuple[int, ...]
    q_shape: tuple[int, ...] | None
    dtype: str

    def to_json(self):
        return {
            "layer": self.layer,
            "k_shape": list(self.k_shape),
            "v_shape": list(self.v_shape),
            "q_shape": list(self.q_shape) if self.q_shape is not None else None,
            "dtype": self.dtype,
        }


def _shape(arr):
    return tuple(int(x) for x in arr.shape)


def _safe_id(value):
    text = str(value)
    out = []
    for ch in text:
        out.append(ch if ch.isalnum() or ch in ("-", "_") else "_")
    return "".join(out).strip("_") or "item"


def _as_int_list(values):
    return [int(x) for x in values]


def _encode_text(tokenizer, text):
    if hasattr(tokenizer, "encode"):
        try:
            return list(tokenizer.encode(text, add_special_tokens=False))
        except TypeError:
            return list(tokenizer.encode(text))
    encoded = tokenizer(text)
    ids = getattr(encoded, "input_ids", encoded["input_ids"])
    return list(ids)


def _logsumexp_np(row):
    row = np.asarray(row, dtype=np.float64)
    m = float(np.max(row))
    return m + float(np.log(np.exp(row - m).sum()))


def _iter_text_documents(paths, extensions=(".txt", ".md", ".jsonl")):
    exts = {e if e.startswith(".") else f".{e}" for e in extensions}
    files = []
    for raw in paths:
        path = Path(raw).expanduser()
        if path.is_dir():
            files.extend(sorted(p for p in path.rglob("*")
                                if p.is_file() and p.suffix in exts))
        elif path.is_file():
            files.append(path)
    for path in sorted(set(files)):
        if path.suffix == ".jsonl":
            with open(path) as fh:
                for line_no, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        text = str(obj.get("text", ""))
                    except json.JSONDecodeError:
                        text = line
                    if text.strip():
                        yield f"{path.name}:{line_no}", str(path), text
            continue
        with open(path, errors="replace") as fh:
            text = fh.read()
        if text.strip():
            yield path.name, str(path), text


def _split_for_doc(doc_id, heldout_fraction, seed):
    frac = max(0.0, min(1.0, float(heldout_fraction)))
    if frac <= 0.0:
        return "train"
    if frac >= 1.0:
        return "heldout"
    h = hashlib.sha256(f"{seed}:{doc_id}".encode("utf-8")).digest()
    score = int.from_bytes(h[:8], "big") / float(1 << 64)
    return "heldout" if score < frac else "train"


def build_corpus_plan(corpus_paths, tokenizer, *, chunk_tokens=512,
                      heldout_fraction=0.1, seed="qwen35-translation-poc",
                      max_total_tokens=0, min_doc_tokens=1,
                      max_docs=0, source_label=None, command_line=None):
    """Build a document-level train/held-out token plan.

    Token ids are stored in the manifest intentionally: source and target
    capture must consume byte-identical token sequences, and resumable cron
    runs should not depend on retokenizing mutable files.
    """
    chunk_tokens = int(chunk_tokens)
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    max_total_tokens = int(max_total_tokens or 0)
    min_doc_tokens = int(min_doc_tokens or 0)
    max_docs = int(max_docs or 0)

    docs = []
    chunks = []
    totals = {"train": 0, "heldout": 0}
    total_tokens = 0
    doc_count = 0
    for label, path, text in _iter_text_documents(corpus_paths):
        if max_docs and doc_count >= max_docs:
            break
        ids = [int(x) for x in _encode_text(tokenizer, text)]
        if len(ids) < min_doc_tokens:
            continue
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        doc_id = hashlib.sha256(
            f"{path}:{label}:{text_hash}".encode("utf-8")).hexdigest()[:16]
        split = _split_for_doc(doc_id, heldout_fraction, seed)
        doc_index = len(docs)
        docs.append({
            "doc_id": doc_id,
            "doc_index": doc_index,
            "label": label,
            "source_path": path,
            "text_sha256": text_hash,
            "split": split,
            "token_count": len(ids),
        })
        doc_count += 1
        for start in range(0, len(ids), chunk_tokens):
            part = ids[start:start + chunk_tokens]
            if not part:
                continue
            if max_total_tokens and total_tokens >= max_total_tokens:
                break
            if max_total_tokens and total_tokens + len(part) > max_total_tokens:
                part = part[:max_total_tokens - total_tokens]
            chunk = {
                "chunk_id": len(chunks),
                "doc_id": doc_id,
                "doc_index": doc_index,
                "split": split,
                "source_path": path,
                "label": label,
                "token_start": start,
                "token_end": start + len(part),
                "token_count": len(part),
                "token_ids": part,
            }
            chunks.append(chunk)
            totals[split] += len(part)
            total_tokens += len(part)
            if max_total_tokens and total_tokens >= max_total_tokens:
                break
        if max_total_tokens and total_tokens >= max_total_tokens:
            break

    return {
        "schema": "qwen35_graft_translation_corpus_plan_v1",
        "source_label": source_label,
        "corpus_paths": [str(Path(p).expanduser()) for p in corpus_paths],
        "chunk_tokens": chunk_tokens,
        "heldout_fraction": float(heldout_fraction),
        "seed": str(seed),
        "max_total_tokens": max_total_tokens,
        "min_doc_tokens": min_doc_tokens,
        "document_split_law": "sha256(seed:doc_id) < heldout_fraction",
        "split_granularity": "document",
        "command_line": command_line,
        "documents": docs,
        "chunks": chunks,
        "totals": {
            "documents": len(docs),
            "chunks": len(chunks),
            "tokens": total_tokens,
            "train_tokens": totals["train"],
            "heldout_tokens": totals["heldout"],
        },
    }


def write_corpus_plan(corpus_paths, tokenizer, out_path, **kwargs):
    plan = build_corpus_plan(corpus_paths, tokenizer, **kwargs)
    out = Path(out_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(plan, fh, indent=2)
        fh.write("\n")
    return plan


def load_hf_tokenizer(model_dir):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(Path(model_dir).expanduser()))


def validate_unquantized_source(path, *, role):
    """Validate one model source and return a manifest.

    The validator intentionally does not open tensor contents. Phase 0 only
    proves that the source is an unquantized HF-style safetensors directory and
    records enough identity for later GPU work to be reproducible.
    """
    root = Path(path).expanduser().resolve()
    if not root.exists():
        raise SourceValidationError(f"{role} source does not exist: {root}")
    if root.is_file():
        if root.suffix.lower() == ".gguf":
            raise SourceValidationError(
                f"{role} source is GGUF, not unquantized safetensors: {root}")
        raise SourceValidationError(
            f"{role} source must be a model directory, got file: {root}")
    if list(root.glob("*.gguf")):
        raise SourceValidationError(
            f"{role} source directory contains GGUF files; use safetensors")

    config_path = root / "config.json"
    tokenizer_path = root / "tokenizer.json"
    if not config_path.is_file():
        raise SourceValidationError(f"{role} source missing config.json")
    if not tokenizer_path.is_file():
        raise SourceValidationError(f"{role} source missing tokenizer.json")

    shards = tuple(
        SafetensorShard(str(p), p.stat().st_size)
        for p in sorted(root.glob("*.safetensors"))
        if p.is_file()
    )
    if not shards:
        raise SourceValidationError(
            f"{role} source has no .safetensors shards")

    config = _load_json(config_path)
    if config.get("quantization_config") is not None:
        raise SourceValidationError(
            f"{role} source declares quantization_config")
    text = _text_config(config)
    if text.get("quantization_config") is not None:
        raise SourceValidationError(
            f"{role} text_config declares quantization_config")
    model_type = str(config.get("model_type", ""))
    text_model_type = str(text.get("model_type", ""))
    if model_type and not model_type.startswith("qwen3_5"):
        raise SourceValidationError(
            f"{role} model_type is not qwen3_5*: {model_type!r}")
    if text_model_type and not text_model_type.startswith("qwen3_5"):
        raise SourceValidationError(
            f"{role} text model_type is not qwen3_5*: {text_model_type!r}")

    tokenizer_config = root / "tokenizer_config.json"
    shape_keys = (
        "hidden_size", "num_hidden_layers", "full_attention_interval",
        "head_dim", "num_attention_heads", "num_key_value_heads", "dtype",
        "intermediate_size", "linear_conv_kernel_dim",
        "linear_key_head_dim", "linear_value_head_dim",
        "linear_num_key_heads", "linear_num_value_heads",
        "max_position_embeddings", "vocab_size")
    text_shape = {k: text.get(k) for k in shape_keys if k in text}
    text_shape["attention_layer_indices"] = _attention_layer_indices(text)
    repo_id, revision = _hf_snapshot_identity(root)
    return Qwen35SourceManifest(
        role=str(role),
        path=str(root),
        repository=repo_id,
        revision=revision,
        model_type=model_type,
        text_model_type=text_model_type,
        tokenizer_sha256=_sha256(tokenizer_path),
        tokenizer_config_sha256=(
            _sha256(tokenizer_config) if tokenizer_config.is_file() else None),
        config_sha256=_sha256(config_path),
        shard_count=len(shards),
        total_safetensor_bytes=sum(s.bytes for s in shards),
        shards=shards,
        text_shape=text_shape)


def validate_weight_pair(source_path, target_path):
    source = validate_unquantized_source(source_path, role="source")
    target = validate_unquantized_source(target_path, role="target")
    if source.tokenizer_sha256 != target.tokenizer_sha256:
        raise SourceValidationError(
            "source and target tokenizer.json hashes differ; per-token "
            "alignment is invalid")
    return {
        "schema": "qwen35_graft_translation_weights_v1",
        "source": source.to_json(),
        "target": target.to_json(),
        "tokenizer_sha256": source.tokenizer_sha256,
        "tokenizer_config_sha256_match": (
            source.tokenizer_config_sha256 == target.tokenizer_config_sha256),
        "source_weight_law": (
            "BF16/FP16 safetensors only; local tensor_cuda INT4 quantization "
            "happens after validation"),
    }


def write_weight_manifest(source_path, target_path, out_path):
    manifest = validate_weight_pair(source_path, target_path)
    out = Path(out_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    return manifest


def write_capture_shard(out_dir, *, role, doc_id, chunk_id, token_ids,
                        captured, queries=None, position_offset=0,
                        storage_dtype=np.float16, metadata=None,
                        compress=True):
    """Write one Qwen3.5 translation-corpus capture shard.

    `captured` is the list returned by `kv_graft.harvest_kv`; `queries` is the
    optional list returned by `kv_graft.capture_queries` for the target model.
    Arrays are expected to be position-neutral pre-RoPE K/V/Q in shape
    `(B, heads, tokens, head_dim)`.
    """
    out_root = Path(out_dir).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)
    role = str(role)
    if role not in ("source", "target"):
        raise ValueError("role must be 'source' or 'target'")
    ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
    arrays = {
        "token_ids": ids,
        "position_offset": np.array([int(position_offset)], dtype=np.int64),
    }
    layer_records = []
    layer_indices = []
    for layer, cap in enumerate(captured):
        if cap is None:
            continue
        if "k" not in cap or "v" not in cap:
            raise ValueError(f"capture layer {layer} missing k/v arrays")
        q = None if queries is None else queries[layer]
        k = np.asarray(cap["k"], dtype=storage_dtype)
        v = np.asarray(cap["v"], dtype=storage_dtype)
        arrays[f"l{layer}_k"] = np.ascontiguousarray(k)
        arrays[f"l{layer}_v"] = np.ascontiguousarray(v)
        q_shape = None
        if q is not None:
            q_arr = np.asarray(q, dtype=storage_dtype)
            arrays[f"l{layer}_q"] = np.ascontiguousarray(q_arr)
            q_shape = _shape(q_arr)
        layer_indices.append(layer)
        layer_records.append(CaptureLayerRecord(
            layer=layer,
            k_shape=_shape(k),
            v_shape=_shape(v),
            q_shape=q_shape,
            dtype=np.dtype(storage_dtype).name,
        ))
    if not layer_records:
        raise ValueError("captured contains no attention-layer payloads")
    arrays["layer_indices"] = np.asarray(layer_indices, dtype=np.int64)

    stem = _capture_shard_stem(role, doc_id, chunk_id)
    npz_path = out_root / f"{stem}.npz"
    json_path = out_root / f"{stem}.json"
    if compress:
        np.savez_compressed(npz_path, **arrays)
    else:
        np.savez(npz_path, **arrays)
    manifest = {
        "schema": "qwen35_graft_translation_capture_shard_v1",
        "role": role,
        "doc_id": str(doc_id),
        "chunk_id": int(chunk_id),
        "token_count": int(ids.size),
        "position_offset": int(position_offset),
        "npz": str(npz_path),
        "json": str(json_path),
        "compressed": bool(compress),
        "has_queries": any(r.q_shape is not None for r in layer_records),
        "layers": [r.to_json() for r in layer_records],
    }
    if metadata:
        manifest["metadata"] = dict(metadata)
    with open(json_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    return manifest


def _capture_shard_stem(role, doc_id, chunk_id):
    return f"{role}_doc{_safe_id(doc_id)}_chunk{int(chunk_id):06d}"


def _capture_shard_paths(out_dir, role, doc_id, chunk_id):
    root = Path(out_dir).expanduser()
    stem = _capture_shard_stem(role, doc_id, chunk_id)
    return root / f"{stem}.npz", root / f"{stem}.json"


def _capture_shard_done(out_dir, role, chunk):
    npz_path, json_path = _capture_shard_paths(
        out_dir, role, chunk["doc_id"], chunk["chunk_id"])
    if not npz_path.is_file() or not json_path.is_file():
        return False
    try:
        meta = _load_json(json_path)
    except Exception:
        return False
    return (
        meta.get("schema") == "qwen35_graft_translation_capture_shard_v1"
        and meta.get("role") == role
        and meta.get("doc_id") == chunk["doc_id"]
        and int(meta.get("chunk_id", -1)) == int(chunk["chunk_id"])
        and int(meta.get("token_count", -1)) == int(chunk["token_count"])
    )


def _select_layers(model, layers):
    attention_layers = list(model.config.attention_layer_indices())
    if layers == "first":
        return {attention_layers[0]}
    if layers == "all":
        return set(attention_layers)
    return {int(x.strip()) for x in str(layers).split(",") if x.strip()}


def _paired_capture_summary(by_role):
    source = by_role.get("source", {})
    target = by_role.get("target", {})
    source_keys = set(source)
    target_keys = set(target)
    summary = {
        "shards": 0,
        "tokens": 0,
        "same_split_shards": 0,
        "same_split_tokens": 0,
        "source_only_chunks": len(source_keys - target_keys),
        "target_only_chunks": len(target_keys - source_keys),
        "token_count_mismatch": 0,
        "split_mismatch": 0,
        "splits": {},
    }
    for key in sorted(source_keys & target_keys):
        sm = source[key]
        tm = target[key]
        s_tokens = int(sm.get("token_count", 0))
        t_tokens = int(tm.get("token_count", 0))
        tokens = min(s_tokens, t_tokens)
        if s_tokens != t_tokens:
            summary["token_count_mismatch"] += 1
        s_split = sm.get("metadata", {}).get("split", "unknown")
        t_split = tm.get("metadata", {}).get("split", "unknown")
        split = s_split if s_split == t_split else "split_mismatch"
        if s_split != t_split:
            summary["split_mismatch"] += 1
        else:
            summary["same_split_shards"] += 1
            summary["same_split_tokens"] += tokens
        rec = summary["splits"].setdefault(split, {"shards": 0, "tokens": 0})
        rec["shards"] += 1
        rec["tokens"] += tokens
        summary["shards"] += 1
        summary["tokens"] += tokens
    return summary


def run_capture_smoke(model_dir, *, role, out_dir, doc_id="smoke",
                      chunk_id=0, token_ids=(1, 2, 3, 4),
                      layers="first"):
    sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
    from core.qwen35_tc import Qwen35_TC
    from core import kv_graft

    token_ids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
    model, info = Qwen35_TC.from_pretrained(model_dir)
    layer_filter = _select_layers(model, layers)
    if role == "target":
        captured, queries = kv_graft.harvest_kv_and_queries(
            model, token_ids, layer_filter=layer_filter)
    else:
        captured = kv_graft.harvest_kv(
            model, token_ids, layer_filter=layer_filter)
        queries = None
    manifest = write_capture_shard(
        out_dir,
        role=role,
        doc_id=doc_id,
        chunk_id=chunk_id,
        token_ids=token_ids,
        captured=captured,
        queries=queries,
    )
    return {"model": info, "capture": manifest}


def refresh_capture_manifest(out_dir, plan_path=None):
    root = Path(out_dir).expanduser()
    sidecars = []
    for path in sorted(root.glob("*.json")):
        try:
            meta = _load_json(path)
        except Exception:
            continue
        if meta.get("schema") == "qwen35_graft_translation_capture_shard_v1":
            sidecars.append(meta)

    by_role = {}
    by_key = {}
    for meta in sidecars:
        role = meta["role"]
        key = (str(meta["doc_id"]), int(meta["chunk_id"]))
        by_key.setdefault(role, {})[key] = meta
        split = meta.get("metadata", {}).get("split", "unknown")
        rec = by_role.setdefault(role, {
            "shards": 0,
            "tokens": 0,
            "splits": {},
            "layers": {},
        })
        rec["shards"] += 1
        rec["tokens"] += int(meta.get("token_count", 0))
        sp = rec["splits"].setdefault(split, {"shards": 0, "tokens": 0})
        sp["shards"] += 1
        sp["tokens"] += int(meta.get("token_count", 0))
        for layer in meta.get("layers", []):
            rec["layers"][str(layer["layer"])] = layer

    expected = {}
    plan = None
    if plan_path:
        plan = _load_json(Path(plan_path).expanduser())
        expected_chunks = len(plan.get("chunks", []))
        for role in ("source", "target"):
            missing = [
                int(chunk["chunk_id"]) for chunk in plan.get("chunks", [])
                if not _capture_shard_done(root, role, chunk)
            ]
            completed = expected_chunks - len(missing)
            expected[role] = {
                "expected_chunks": expected_chunks,
                "completed_chunks": completed,
                "remaining_chunks": len(missing),
                "next_missing_chunk": missing[0] if missing else None,
                "complete": completed == expected_chunks,
            }

    manifest = {
        "schema": "qwen35_graft_translation_capture_manifest_v1",
        "out_dir": str(root),
        "plan_path": str(Path(plan_path).expanduser()) if plan_path else None,
        "corpus_plan_sha256": (
            _sha256(Path(plan_path).expanduser()) if plan_path else None),
        "corpus_totals": plan.get("totals") if plan else None,
        "roles": by_role,
        "paired": _paired_capture_summary(by_key),
        "expected": expected,
    }
    out = root / "capture_manifest.json"
    root.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    return manifest


def _load_capture_sidecars(capture_dir):
    by_role = {"source": {}, "target": {}}
    for path in sorted(Path(capture_dir).expanduser().glob("*.json")):
        try:
            meta = _load_json(path)
        except Exception:
            continue
        if meta.get("schema") != "qwen35_graft_translation_capture_shard_v1":
            continue
        key = (str(meta["doc_id"]), int(meta["chunk_id"]))
        by_role.setdefault(meta["role"], {})[key] = meta
    return by_role


def fractional_attention_layer_map(source_layers, target_layers):
    source_layers = list(source_layers)
    target_layers = list(target_layers)
    if not source_layers or not target_layers:
        raise ValueError("source and target attention layer lists are required")
    if len(source_layers) == 1:
        return {source_layers[0]: target_layers[0]}
    out = {}
    for i, src in enumerate(source_layers):
        j = int(round(i * (len(target_layers) - 1) / (len(source_layers) - 1)))
        out[src] = target_layers[j]
    return out


def shifted_target_layer_map(layer_map, target_layers):
    """Map each source layer to the next target attention layer as a control."""
    target_layers = sorted(int(x) for x in target_layers)
    if len(target_layers) < 2:
        raise ValueError("wrong-layer control requires at least two target layers")
    out = {}
    for src, tgt in sorted(layer_map.items()):
        idx = target_layers.index(int(tgt))
        out[int(src)] = target_layers[(idx + 1) % len(target_layers)]
    return out


def _fit_kinds(kinds):
    if isinstance(kinds, str):
        raw = [x.strip().lower() for x in kinds.split(",") if x.strip()]
    else:
        raw = [str(x).strip().lower() for x in kinds]
    if not raw or raw == ["both"]:
        return ("k", "v")
    if set(raw) - {"k", "v"}:
        raise ValueError("kinds must be 'both', 'k', 'v', or 'k,v'")
    return tuple(k for k in ("k", "v") if k in raw)


def _capture_layers(meta):
    return [int(layer["layer"]) for layer in meta.get("layers", [])]


def _flatten_heads_tokens(arr):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 4 or arr.shape[0] != 1:
        raise ValueError(f"expected capture array shape (1, heads, tokens, dim), got {arr.shape}")
    return np.ascontiguousarray(arr[0].transpose(1, 0, 2).reshape(arr.shape[2], -1))


def _resolve_ridge_backend(backend):
    backend = str(backend or "cpu").lower()
    aliases = {"cuda": "cupy", "gpu": "cupy", "numpy": "cpu"}
    backend = aliases.get(backend, backend)
    if backend == "cpu":
        return "cpu", None
    if backend not in ("auto", "cupy"):
        raise ValueError(
            "backend must be 'cpu', 'auto', 'cupy', 'cuda', or 'gpu'")
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() <= 0:
            raise RuntimeError("no CUDA devices visible to CuPy")
        cp.cuda.Device(0).use()
        return "cupy", cp
    except Exception:
        if backend == "auto":
            return "cpu", None
        raise


def _new_ridge_accum(input_dim, output_dim, *, backend, cp=None):
    if backend == "cupy":
        return {
            "a": cp.zeros((input_dim, input_dim), dtype=cp.float64),
            "b": cp.zeros((input_dim, output_dim), dtype=cp.float64),
            "n": 0,
            "y_sum": cp.zeros(output_dim, dtype=cp.float64),
            "y_sq_sum": 0.0,
        }
    return {
        "a": np.zeros((input_dim, input_dim), np.float64),
        "b": np.zeros((input_dim, output_dim), np.float64),
        "n": 0,
        "y_sum": np.zeros(output_dim, np.float64),
        "y_sq_sum": 0.0,
    }


def _accumulate_ridge_stats(rec, x1, y, *, backend, cp=None):
    if backend == "cupy":
        x1g = cp.asarray(x1, dtype=cp.float64)
        yg = cp.asarray(y, dtype=cp.float64)
        rec["a"] += x1g.T @ x1g
        rec["b"] += x1g.T @ yg
        rec["n"] += x1.shape[0]
        rec["y_sum"] += yg.sum(axis=0)
        rec["y_sq_sum"] += float(cp.sum(yg * yg).get())
        return
    x1d = x1.astype(np.float64)
    yd = y.astype(np.float64)
    rec["a"] += x1d.T @ x1d
    rec["b"] += x1d.T @ yd
    rec["n"] += x1.shape[0]
    rec["y_sum"] += yd.sum(axis=0)
    rec["y_sq_sum"] += float((yd ** 2).sum())


def _solve_ridge(accum, ridge_lambda, *, backend="cpu", cp=None):
    dim = accum["a"].shape[0]
    if backend == "cupy":
        reg = cp.eye(dim, dtype=cp.float64) * float(ridge_lambda)
        reg[-1, -1] = 0.0
        lhs = accum["a"] + reg
        try:
            return cp.asnumpy(cp.linalg.solve(lhs, accum["b"]))
        except Exception:
            lhs = cp.asnumpy(lhs)
            rhs = cp.asnumpy(accum["b"])
            try:
                return np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                return np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    reg = np.eye(dim, dtype=np.float64) * float(ridge_lambda)
    reg[-1, -1] = 0.0
    lhs = accum["a"] + reg
    try:
        return np.linalg.solve(lhs, accum["b"])
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(lhs, accum["b"], rcond=None)[0]


def _ridge_y_sum(rec, *, backend, cp=None):
    if backend == "cupy":
        return cp.asnumpy(rec["y_sum"])
    return rec["y_sum"]


def _accumulate_prediction_metrics(m, x, y, weight, bias, *, backend, cp=None):
    if backend == "cupy":
        xg = cp.asarray(x, dtype=cp.float32)
        yg = cp.asarray(y, dtype=cp.float32)
        wg = cp.asarray(weight, dtype=cp.float32)
        bg = cp.asarray(bias, dtype=cp.float32)
        pred = xg @ wg + bg
        diff = pred - yg
        m["sse"] += float(cp.sum(diff.astype(cp.float64) ** 2).get())
        denom = (
            cp.linalg.norm(pred, axis=1) *
            cp.linalg.norm(yg, axis=1) + 1e-12)
        m["cosine_sum"] += float(
            cp.sum(cp.sum(pred * yg, axis=1) / denom).get())
        m["cosine_count"] += int(y.shape[0])
        return
    pred = x @ weight + bias
    diff = pred - y
    m["sse"] += float((diff.astype(np.float64) ** 2).sum())
    denom = (np.linalg.norm(pred, axis=1) *
             np.linalg.norm(y, axis=1) + 1e-12)
    m["cosine_sum"] += float(((pred * y).sum(axis=1) / denom).sum())
    m["cosine_count"] += int(y.shape[0])


def _accumulate_ridge_translator(capture_dir, *, split="train",
                                 control="normal", kinds=("k", "v"),
                                 backend="cpu"):
    requested_backend = str(backend or "cpu").lower()
    backend, cp = _resolve_ridge_backend(requested_backend)
    control = str(control)
    if control not in ("normal", "wrong-layer", "shuffled-docs"):
        raise ValueError(
            "control must be 'normal', 'wrong-layer', or 'shuffled-docs'")
    kinds = _fit_kinds(kinds)
    capture_dir = Path(capture_dir).expanduser()
    sidecars = _load_capture_sidecars(capture_dir)
    source = sidecars.get("source", {})
    target = sidecars.get("target", {})
    pairs = []
    for key in sorted(set(source) & set(target)):
        sm, tm = source[key], target[key]
        if split != "all":
            s_split = sm.get("metadata", {}).get("split")
            t_split = tm.get("metadata", {}).get("split")
            if s_split != split or t_split != split:
                continue
        pairs.append((key, sm, tm))
    if not pairs:
        raise ValueError(f"no paired source/target capture shards for split={split!r}")
    if control == "shuffled-docs":
        if len(pairs) < 2:
            raise ValueError("shuffled-docs control requires at least two pairs")
        shifted_targets = [tm for _, _, tm in pairs[1:]] + [pairs[0][2]]
        pairs = [(key, sm, shifted_targets[i])
                 for i, (key, sm, _) in enumerate(pairs)]

    source_layers = sorted({l for _, sm, _ in pairs for l in _capture_layers(sm)})
    target_layers = sorted({l for _, _, tm in pairs for l in _capture_layers(tm)})
    layer_map = fractional_attention_layer_map(source_layers, target_layers)
    if control == "wrong-layer":
        layer_map = shifted_target_layer_map(layer_map, target_layers)

    accum = {}
    for _, sm, tm in pairs:
        with np.load(sm["npz"]) as sz, np.load(tm["npz"]) as tz:
            if (control != "shuffled-docs" and
                    sz["token_ids"].tolist() != tz["token_ids"].tolist()):
                raise ValueError(
                    f"token mismatch for paired shard "
                    f"{sm['doc_id']}:{sm['chunk_id']}")
            for src_layer, tgt_layer in layer_map.items():
                for kind in kinds:
                    skey = f"l{src_layer}_{kind}"
                    tkey = f"l{tgt_layer}_{kind}"
                    if skey not in sz.files or tkey not in tz.files:
                        continue
                    x = _flatten_heads_tokens(sz[skey])
                    y = _flatten_heads_tokens(tz[tkey])
                    if x.shape[0] != y.shape[0]:
                        if control != "shuffled-docs":
                            raise ValueError(
                                f"token count mismatch for {skey}->{tkey}: "
                                f"{x.shape} vs {y.shape}")
                        n = min(x.shape[0], y.shape[0])
                        x = x[:n]
                        y = y[:n]
                    x1 = np.concatenate(
                        [x, np.ones((x.shape[0], 1), dtype=np.float32)],
                        axis=1)
                    key = (src_layer, tgt_layer, kind)
                    rec = accum.get(key)
                    if rec is None:
                        rec = _new_ridge_accum(
                            x1.shape[1],
                            y.shape[1],
                            backend=backend,
                            cp=cp,
                        )
                        accum[key] = rec
                    _accumulate_ridge_stats(
                        rec, x1, y, backend=backend, cp=cp)

    if not accum:
        raise ValueError("no overlapping source/target layer tensors were found")
    return {
        "capture_dir": capture_dir,
        "split": split,
        "control": control,
        "kinds": kinds,
        "requested_backend": requested_backend,
        "backend": backend,
        "cp": cp,
        "pairs": pairs,
        "layer_map": layer_map,
        "accum": accum,
    }


def _write_ridge_translator_solutions(state, specs, *, compute_fit_metrics=True):
    capture_dir = state["capture_dir"]
    split = state["split"]
    control = state["control"]
    kinds = state["kinds"]
    requested_backend = state["requested_backend"]
    backend = state["backend"]
    cp = state["cp"]
    pairs = state["pairs"]
    layer_map = state["layer_map"]
    accum = state["accum"]

    work = []
    for ridge_lambda, out_dir in specs:
        out_dir = Path(out_dir).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        solved = {}
        artifacts = []
        metrics = {}
        for key, rec in sorted(accum.items()):
            w_aug = _solve_ridge(
                rec, ridge_lambda, backend=backend, cp=cp)
            src_layer, tgt_layer, kind = key
            weight = w_aug[:-1].astype(np.float32)
            bias = w_aug[-1].astype(np.float32)
            name = f"translator_l{src_layer}_to_l{tgt_layer}_{kind}.npz"
            path = out_dir / name
            np.savez(path, weight=weight, bias=bias,
                     source_layer=np.array([src_layer], np.int64),
                     target_layer=np.array([tgt_layer], np.int64),
                     kind=np.array([kind]))
            solved[key] = {
                "weight": weight,
                "bias": bias,
                "path": str(path),
            }
            artifacts.append({
                "source_layer": src_layer,
                "target_layer": tgt_layer,
                "kind": kind,
                "path": str(path),
                "input_dim": int(weight.shape[0]),
                "output_dim": int(weight.shape[1]),
                "train_tokens": int(rec["n"]),
            })
            if compute_fit_metrics:
                y_sum = _ridge_y_sum(rec, backend=backend, cp=cp)
                metrics[key] = {
                    "sse": 0.0,
                    "cosine_sum": 0.0,
                    "cosine_count": 0,
                    "n": int(rec["n"]),
                    "sst": float(
                        rec["y_sq_sum"] - (y_sum @ y_sum) / max(rec["n"], 1)),
                }
        work.append({
            "ridge_lambda": float(ridge_lambda),
            "out_dir": out_dir,
            "solved": solved,
            "artifacts": artifacts,
            "metrics": metrics,
        })

    if compute_fit_metrics:
        for _, sm, tm in pairs:
            with np.load(sm["npz"]) as sz, np.load(tm["npz"]) as tz:
                for src_layer, tgt_layer in layer_map.items():
                    for kind in kinds:
                        key = (src_layer, tgt_layer, kind)
                        skey = f"l{src_layer}_{kind}"
                        tkey = f"l{tgt_layer}_{kind}"
                        if skey not in sz.files or tkey not in tz.files:
                            continue
                        x = _flatten_heads_tokens(sz[skey])
                        y = _flatten_heads_tokens(tz[tkey])
                        if x.shape[0] != y.shape[0]:
                            n = min(x.shape[0], y.shape[0])
                            x = x[:n]
                            y = y[:n]
                        for item in work:
                            if key not in item["solved"]:
                                continue
                            m = item["metrics"][key]
                            solved = item["solved"][key]
                            _accumulate_prediction_metrics(
                                m,
                                x,
                                y,
                                solved["weight"],
                                solved["bias"],
                                backend=backend,
                                cp=cp,
                            )

    results = []
    for item in work:
        metric_rows = []
        if compute_fit_metrics:
            for key, m in sorted(item["metrics"].items()):
                src_layer, tgt_layer, kind = key
                n_vals = max(m["n"], 1)
                row = {
                    "source_layer": src_layer,
                    "target_layer": tgt_layer,
                    "kind": kind,
                    "train_tokens": int(m["n"]),
                    "mse": float(m["sse"] / n_vals),
                    "r2": float(1.0 - m["sse"] / max(m["sst"], 1e-12)),
                    "mean_row_cosine": float(
                        m["cosine_sum"] / max(m["cosine_count"], 1)),
                }
                metric_rows.append(row)
        else:
            for key, rec in sorted(accum.items()):
                if key not in item["solved"]:
                    continue
                src_layer, tgt_layer, kind = key
                metric_rows.append({
                    "source_layer": src_layer,
                    "target_layer": tgt_layer,
                    "kind": kind,
                    "train_tokens": int(rec["n"]),
                    "mse": None,
                    "r2": None,
                    "mean_row_cosine": None,
                })

        out_dir = item["out_dir"]
        ridge_lambda = item["ridge_lambda"]
        fit_metrics = {
            "schema": "qwen35_graft_translation_fit_metrics_v1",
            "capture_dir": str(capture_dir),
            "ridge_lambda": float(ridge_lambda),
            "split": split,
            "control": control,
            "kinds": list(kinds),
            "backend_requested": requested_backend,
            "compute_backend": backend,
            "fit_metrics_computed": bool(compute_fit_metrics),
            "paired_shards": len(pairs),
            "layers": metric_rows,
        }
        translator_manifest = {
            "schema": "qwen35_graft_translation_translator_v1",
            "capture_dir": str(capture_dir),
            "ridge_lambda": float(ridge_lambda),
            "split": split,
            "control": control,
            "kinds": list(kinds),
            "backend_requested": requested_backend,
            "compute_backend": backend,
            "fit_metrics_computed": bool(compute_fit_metrics),
            "paired_shards": len(pairs),
            "layer_alignment": [
                {"source_layer": int(k), "target_layer": int(v)}
                for k, v in sorted(layer_map.items())
            ],
            "artifacts": item["artifacts"],
            "fit_metrics": str(out_dir / "fit_metrics.json"),
        }
        with open(out_dir / "fit_metrics.json", "w") as fh:
            json.dump(fit_metrics, fh, indent=2)
            fh.write("\n")
        with open(out_dir / "translator_manifest.json", "w") as fh:
            json.dump(translator_manifest, fh, indent=2)
            fh.write("\n")
        results.append({
            "translator_manifest": translator_manifest,
            "fit_metrics": fit_metrics,
        })
    return results


def fit_ridge_translator(capture_dir, out_dir, *, ridge_lambda=1e-4,
                         split="train", control="normal", kinds=("k", "v"),
                         backend="cpu", compute_fit_metrics=True):
    """Fit full-width per-layer ridge maps from source K/V into target K/V."""
    state = _accumulate_ridge_translator(
        capture_dir,
        split=split,
        control=control,
        kinds=kinds,
        backend=backend,
    )
    return _write_ridge_translator_solutions(
        state,
        [(float(ridge_lambda), Path(out_dir).expanduser())],
        compute_fit_metrics=compute_fit_metrics,
    )[0]


def _ridge_lambda_slug(value):
    text = str(value).strip().lower()
    text = text.replace("+", "")
    text = text.replace("e-0", "e-").replace("e0", "e")
    return text.replace(".", "p")


def fit_ridge_translator_sweep(capture_dir, out_root, *, ridge_lambdas,
                               out_prefix="translator_ridge", split="train",
                               control="normal", kinds=("k", "v"),
                               backend="cpu", compute_fit_metrics=True):
    labels = [str(x) for x in ridge_lambdas]
    lambdas = [float(x) for x in labels]
    if not lambdas:
        raise ValueError("ridge_lambdas must not be empty")
    out_root = Path(out_root).expanduser()
    state = _accumulate_ridge_translator(
        capture_dir,
        split=split,
        control=control,
        kinds=kinds,
        backend=backend,
    )
    specs = [
        (lam, out_root / f"{out_prefix}_{_ridge_lambda_slug(label)}")
        for lam, label in zip(lambdas, labels)
    ]
    results = _write_ridge_translator_solutions(
        state,
        specs,
        compute_fit_metrics=compute_fit_metrics,
    )
    return {
        "schema": "qwen35_graft_translation_ridge_sweep_v1",
        "capture_dir": str(Path(capture_dir).expanduser()),
        "out_root": str(out_root),
        "out_prefix": out_prefix,
        "ridge_lambdas": lambdas,
        "split": state["split"],
        "control": state["control"],
        "kinds": list(state["kinds"]),
        "backend_requested": state["requested_backend"],
        "compute_backend": state["backend"],
        "fit_metrics_computed": bool(compute_fit_metrics),
        "results": results,
    }


def _load_translator(translator_dir):
    root = Path(translator_dir).expanduser()
    manifest = _load_json(root / "translator_manifest.json")
    artifacts = {}
    for art in manifest.get("artifacts", []):
        z = np.load(art["path"])
        key = (int(art["source_layer"]), int(art["target_layer"]), art["kind"])
        artifacts[key] = {
            "weight": z["weight"].astype(np.float32),
            "bias": z["bias"].astype(np.float32),
            "artifact": art,
        }
    return manifest, artifacts


def _parse_layer_pair_spec(spec):
    pairs = set()
    for raw in str(spec or "").split(","):
        raw = raw.strip()
        if not raw:
            continue
        if "->" in raw:
            left, right = raw.split("->", 1)
        elif ":" in raw:
            left, right = raw.split(":", 1)
        else:
            raise ValueError(
                "layer pairs must use 'source:target' or 'source->target'")
        pairs.add((int(left.strip()), int(right.strip())))
    return pairs


def filter_translator_layers(translator_dir, out_dir, *, policy_name,
                             keep_pairs=None, drop_pairs=None):
    """Write a translator manifest filtered to selected source/target pairs."""
    keep = _parse_layer_pair_spec(keep_pairs)
    drop = _parse_layer_pair_spec(drop_pairs)
    if bool(keep) == bool(drop):
        raise ValueError("pass exactly one of keep_pairs or drop_pairs")

    translator_dir = Path(translator_dir).expanduser()
    out_dir = Path(out_dir).expanduser()
    manifest = _load_json(translator_dir / "translator_manifest.json")
    artifacts = list(manifest.get("artifacts", []))
    if not artifacts:
        raise ValueError("translator manifest has no artifacts")

    all_pairs = sorted({
        (int(art["source_layer"]), int(art["target_layer"]))
        for art in artifacts
    })
    selected = keep if keep else set(all_pairs) - drop
    unknown = sorted(selected - set(all_pairs))
    if unknown:
        raise ValueError(f"unknown translator layer pairs: {unknown!r}")

    kept_artifacts = [
        dict(art)
        for art in artifacts
        if (int(art["source_layer"]), int(art["target_layer"])) in selected
    ]
    kept_pairs = sorted({
        (int(art["source_layer"]), int(art["target_layer"]))
        for art in kept_artifacts
    })
    if not kept_pairs:
        raise ValueError("layer policy would keep no translator artifacts")
    kinds_by_pair = {}
    for art in kept_artifacts:
        key = (int(art["source_layer"]), int(art["target_layer"]))
        kinds_by_pair.setdefault(key, set()).add(str(art["kind"]))
    incomplete = sorted(
        pair for pair, kinds in kinds_by_pair.items()
        if not {"k", "v"}.issubset(kinds)
    )
    if incomplete:
        raise ValueError(
            f"layer policy produced incomplete K/V pairs: {incomplete!r}")

    out_manifest = dict(manifest)
    out_manifest["parent_translator_dir"] = str(translator_dir)
    out_manifest["layer_policy_name"] = str(policy_name)
    out_manifest["layer_policy_generated_utc"] = _utc_now()
    out_manifest["layer_policy"] = {
        "mode": "keep" if keep else "drop",
        "requested_pairs": [
            {"source_layer": int(src), "target_layer": int(tgt)}
            for src, tgt in sorted(keep or drop)
        ],
        "kept_pairs": [
            {"source_layer": int(src), "target_layer": int(tgt)}
            for src, tgt in kept_pairs
        ],
        "dropped_pairs": [
            {"source_layer": int(src), "target_layer": int(tgt)}
            for src, tgt in sorted(set(all_pairs) - set(kept_pairs))
        ],
    }
    out_manifest["layer_alignment"] = [
        {"source_layer": int(src), "target_layer": int(tgt)}
        for src, tgt in kept_pairs
    ]
    out_manifest["artifacts"] = kept_artifacts
    out_manifest["artifact_count"] = len(kept_artifacts)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "translator_manifest.json"
    with open(out_path, "w") as fh:
        json.dump(out_manifest, fh, indent=2)
        fh.write("\n")
    return out_manifest


def _unflatten_like(flat, template):
    template = np.asarray(template)
    _, heads, tokens, dim = template.shape
    return np.ascontiguousarray(
        flat.reshape(tokens, heads, dim).transpose(1, 0, 2)[None])


def _repeat_heads(arr, heads):
    rep = heads // arr.shape[0]
    return np.repeat(arr, rep, axis=0)


def _softmax_np(scores):
    s = scores - scores.max(axis=-1, keepdims=True)
    e = np.exp(s)
    return e / (e.sum(axis=-1, keepdims=True) + 1e-12)


def _attention_scores_np(q, kv):
    kv_heads = _repeat_heads(kv, q.shape[0])
    return np.matmul(q, np.swapaxes(kv_heads, -1, -2)) / np.sqrt(q.shape[-1])


def _attention_output_np(weights, v, heads):
    return np.matmul(weights, _repeat_heads(v, heads))


def _topk_recall(scores_a, scores_b, k):
    k = min(int(k), scores_a.shape[-1], scores_b.shape[-1])
    if k <= 0:
        return 0.0
    ia = np.argpartition(scores_a, -k, axis=-1)[..., -k:]
    ib = np.argpartition(scores_b, -k, axis=-1)[..., -k:]
    a_rows = ia.reshape(-1, k)
    b_rows = ib.reshape(-1, k)
    hits = (a_rows[:, :, None] == b_rows[:, None, :]).any(axis=2).sum()
    rows = a_rows.shape[0]
    return float(hits) / float(max(rows * k, 1))


def _mean_or_none(values):
    return None if not values else float(np.mean(values))


def _accumulate_output_fidelity(rec, prefix, pred, native):
    diff = pred - native
    rec[f"{prefix}_sse"] = rec.get(f"{prefix}_sse", 0.0) + float(
        (diff.astype(np.float64) ** 2).sum())
    rec[f"{prefix}_count"] = rec.get(f"{prefix}_count", 0) + int(diff.size)
    denom = (np.linalg.norm(pred, axis=-1) *
             np.linalg.norm(native, axis=-1) + 1e-12)
    rec[f"{prefix}_cos_sum"] = rec.get(f"{prefix}_cos_sum", 0.0) + float(
        ((pred * native).sum(axis=-1) / denom).sum())
    rec[f"{prefix}_cos_count"] = (
        rec.get(f"{prefix}_cos_count", 0) +
        int(native.shape[0] * native.shape[1])
    )


def _finish_output_fidelity(row, rec, prefix, *, alias=None):
    name = alias or prefix
    count = rec.get(f"{prefix}_count", 0)
    cos_count = rec.get(f"{prefix}_cos_count", 0)
    row[f"{name}_mse"] = (
        None if count == 0 else float(rec.get(f"{prefix}_sse", 0.0) / count)
    )
    row[f"{name}_cosine"] = (
        None if cos_count == 0
        else float(rec.get(f"{prefix}_cos_sum", 0.0) / cos_count)
    )


def evaluate_translator(capture_dir, translator_dir, out_path, *,
                        split="heldout", topk=16, max_pairs=0):
    """Evaluate G1/G2 plus cheap negative controls on paired shards."""
    capture_dir = Path(capture_dir).expanduser()
    out = Path(out_path).expanduser()
    progress_out = out.with_name(f"{out.stem}_progress.json")
    manifest, artifacts = _load_translator(translator_dir)
    sidecars = _load_capture_sidecars(capture_dir)
    pairs = []
    for key in sorted(set(sidecars.get("source", {})) &
                      set(sidecars.get("target", {}))):
        sm = sidecars["source"][key]
        tm = sidecars["target"][key]
        if split != "all":
            if (sm.get("metadata", {}).get("split") != split or
                    tm.get("metadata", {}).get("split") != split):
                continue
        pairs.append((key, sm, tm))
    max_pairs = int(max_pairs or 0)
    if max_pairs > 0:
        pairs = pairs[:max_pairs]
    if not pairs:
        raise ValueError(f"no paired source/target capture shards for split={split!r}")

    def write_progress(done, *, status="running"):
        progress_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "qwen35_graft_translation_eval_progress_v1",
            "status": status,
            "updated_utc": _utc_now(),
            "capture_dir": str(capture_dir),
            "translator_dir": str(Path(translator_dir).expanduser()),
            "out": str(out),
            "split": split,
            "topk": int(topk),
            "max_pairs": max_pairs,
            "paired_shards": len(pairs),
            "completed_shards": int(done),
            "remaining_shards": int(max(len(pairs) - done, 0)),
        }
        tmp = progress_out.with_suffix(f"{progress_out.suffix}.tmp")
        with open(tmp, "w") as fh:
            json.dump(payload, fh, indent=2)
            fh.write("\n")
        tmp.replace(progress_out)

    write_progress(0)
    metric_acc = {}
    for pair_idx, (_, sm, tm) in enumerate(pairs, start=1):
        sz = np.load(sm["npz"])
        tz = np.load(tm["npz"])
        if sz["token_ids"].tolist() != tz["token_ids"].tolist():
            raise ValueError(
                f"token mismatch for paired shard {sm['doc_id']}:{sm['chunk_id']}")
        for src_layer, tgt_layer, kind in sorted(artifacts):
            if kind != "k":
                continue
            skey = f"l{src_layer}_k"
            tkey = f"l{tgt_layer}_k"
            qkey = f"l{tgt_layer}_q"
            v_art_key = (src_layer, tgt_layer, "v")
            tvkey = f"l{tgt_layer}_v"
            svkey = f"l{src_layer}_v"
            if (skey not in sz.files or tkey not in tz.files or
                    qkey not in tz.files):
                continue
            target_layers = [
                layer for layer in _capture_layers(tm)
                if f"l{layer}_k" in tz.files and f"l{layer}_v" in tz.files
            ]
            wrong_layer = next(
                (layer for layer in target_layers if layer != tgt_layer),
                None)
            src_k = _flatten_heads_tokens(sz[skey])
            k_art = artifacts[(src_layer, tgt_layer, "k")]
            pred_k = src_k @ k_art["weight"] + k_art["bias"]
            pred_k = _unflatten_like(pred_k, tz[tkey]).astype(np.float32)[0]
            native_k = np.asarray(tz[tkey], dtype=np.float32)[0]
            q = np.asarray(tz[qkey], dtype=np.float32)[0]
            h = q.shape[0]
            native_scores = _attention_scores_np(q, native_k)
            pred_scores = _attention_scores_np(q, pred_k)
            shuffled_scores = _attention_scores_np(q, pred_k[:, ::-1, :])
            native_weights = _softmax_np(native_scores)
            pred_weights = _softmax_np(pred_scores)

            rec = metric_acc.setdefault((src_layer, tgt_layer), {
                "chunks": 0,
                "tokens": 0,
                "key_recall": [],
                "shuffled_key_recall": [],
                "wrong_layer_key_recall": [],
                "wrong_layer": wrong_layer,
            })
            rec["chunks"] += 1
            rec["tokens"] += int(q.shape[1])
            rec["key_recall"].append(_topk_recall(
                native_scores, pred_scores, topk))
            rec["shuffled_key_recall"].append(_topk_recall(
                native_scores, shuffled_scores, topk))
            if wrong_layer is not None:
                wrong_k = np.asarray(tz[f"l{wrong_layer}_k"],
                                     dtype=np.float32)[0]
                wrong_scores = _attention_scores_np(q, wrong_k)
                rec["wrong_layer_key_recall"].append(_topk_recall(
                    native_scores, wrong_scores, topk))

            if (v_art_key in artifacts and svkey in sz.files and
                    tvkey in tz.files):
                src_v = _flatten_heads_tokens(sz[svkey])
                v_art = artifacts[v_art_key]
                pred_v = src_v @ v_art["weight"] + v_art["bias"]
                pred_v = _unflatten_like(
                    pred_v, tz[tvkey]).astype(np.float32)[0]
                native_v = np.asarray(tz[tvkey], dtype=np.float32)[0]
                native_out = _attention_output_np(native_weights, native_v, h)
                v_only_out = _attention_output_np(native_weights, pred_v, h)
                k_only_out = _attention_output_np(pred_weights, native_v, h)
                kv_out = _attention_output_np(pred_weights, pred_v, h)
                _accumulate_output_fidelity(
                    rec, "v_only_value_output", v_only_out, native_out)
                _accumulate_output_fidelity(
                    rec, "k_only_native_value_output",
                    k_only_out, native_out)
                _accumulate_output_fidelity(
                    rec, "translated_attention_value_output",
                    kv_out, native_out)
                if wrong_layer is not None:
                    wrong_v = np.asarray(tz[f"l{wrong_layer}_v"],
                                         dtype=np.float32)[0]
                    wrong_v_out = _attention_output_np(
                        native_weights, wrong_v, h)
                    _accumulate_output_fidelity(
                        rec, "wrong_layer_value_output",
                        wrong_v_out, native_out)
        if pair_idx == len(pairs) or pair_idx % 25 == 0:
            write_progress(pair_idx)

    rows = []
    for (src_layer, tgt_layer), rec in sorted(metric_acc.items()):
        row = {
            "source_layer": src_layer,
            "target_layer": tgt_layer,
            "wrong_layer": rec.get("wrong_layer"),
            "chunks": rec["chunks"],
            "tokens": rec["tokens"],
            f"key_recall_at_{int(topk)}": _mean_or_none(rec["key_recall"]),
            f"shuffled_key_recall_at_{int(topk)}": _mean_or_none(
                rec["shuffled_key_recall"]),
            f"wrong_layer_key_recall_at_{int(topk)}": _mean_or_none(
                rec["wrong_layer_key_recall"]),
        }
        _finish_output_fidelity(
            row, rec, "v_only_value_output", alias="value_output")
        _finish_output_fidelity(row, rec, "v_only_value_output")
        _finish_output_fidelity(row, rec, "k_only_native_value_output")
        _finish_output_fidelity(row, rec, "translated_attention_value_output")
        _finish_output_fidelity(row, rec, "wrong_layer_value_output")
        rows.append(row)

    eval_metrics = {
        "schema": "qwen35_graft_translation_eval_metrics_v1",
        "capture_dir": str(capture_dir),
        "translator_manifest": manifest,
        "split": split,
        "topk": int(topk),
        "max_pairs": max_pairs,
        "paired_shards": len(pairs),
        "layers": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(eval_metrics, fh, indent=2)
        fh.write("\n")
    write_progress(len(pairs), status="complete")
    return eval_metrics


def evaluate_translator_sweep(capture_dir, translator_dirs, *, out_name=None,
                              out_paths=None, progress_out=None,
                              split="heldout", topk=16, max_pairs=0):
    """Evaluate several translators while sharing one paired-capture pass."""
    capture_dir = Path(capture_dir).expanduser()
    translator_dirs = [Path(p).expanduser() for p in translator_dirs]
    if not translator_dirs:
        raise ValueError("translator_dirs must not be empty")
    if out_paths is not None and out_name is not None:
        raise ValueError("pass either out_name or out_paths, not both")
    if out_paths is None:
        out_name = out_name or "eval_metrics.json"
        out_paths = [td / out_name for td in translator_dirs]
    out_paths = [Path(p).expanduser() for p in out_paths]
    if len(out_paths) != len(translator_dirs):
        raise ValueError("out_paths must match translator_dirs length")

    loaded = []
    for translator_dir, out_path in zip(translator_dirs, out_paths):
        manifest, artifacts = _load_translator(translator_dir)
        loaded.append({
            "translator_dir": translator_dir,
            "out": out_path,
            "manifest": manifest,
            "artifacts": artifacts,
            "metric_acc": {},
        })

    sidecars = _load_capture_sidecars(capture_dir)
    pairs = []
    for key in sorted(set(sidecars.get("source", {})) &
                      set(sidecars.get("target", {}))):
        sm = sidecars["source"][key]
        tm = sidecars["target"][key]
        if split != "all":
            if (sm.get("metadata", {}).get("split") != split or
                    tm.get("metadata", {}).get("split") != split):
                continue
        pairs.append((key, sm, tm))
    max_pairs = int(max_pairs or 0)
    if max_pairs > 0:
        pairs = pairs[:max_pairs]
    if not pairs:
        raise ValueError(f"no paired source/target capture shards for split={split!r}")

    eval_keys = sorted({
        key
        for item in loaded
        for key in item["artifacts"]
        if key[2] == "k"
    })
    if not eval_keys:
        raise ValueError("no key translators found in translator_dirs")

    if progress_out is None:
        progress_out = out_paths[0].with_name("eval_translator_sweep_progress.json")
    progress_out = Path(progress_out).expanduser()

    def write_progress(done, *, status="running"):
        progress_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "qwen35_graft_translation_eval_sweep_progress_v1",
            "status": status,
            "updated_utc": _utc_now(),
            "capture_dir": str(capture_dir),
            "translator_dirs": [str(item["translator_dir"]) for item in loaded],
            "out_paths": [str(item["out"]) for item in loaded],
            "split": split,
            "topk": int(topk),
            "max_pairs": max_pairs,
            "paired_shards": len(pairs),
            "completed_shards": int(done),
            "remaining_shards": int(max(len(pairs) - done, 0)),
        }
        tmp = progress_out.with_suffix(f"{progress_out.suffix}.tmp")
        with open(tmp, "w") as fh:
            json.dump(payload, fh, indent=2)
            fh.write("\n")
        tmp.replace(progress_out)

    write_progress(0)
    for pair_idx, (_, sm, tm) in enumerate(pairs, start=1):
        with np.load(sm["npz"]) as sz, np.load(tm["npz"]) as tz:
            if sz["token_ids"].tolist() != tz["token_ids"].tolist():
                raise ValueError(
                    "token mismatch for paired shard "
                    f"{sm['doc_id']}:{sm['chunk_id']}")
            target_layers = [
                layer for layer in _capture_layers(tm)
                if f"l{layer}_k" in tz.files and f"l{layer}_v" in tz.files
            ]
            for src_layer, tgt_layer, kind in eval_keys:
                skey = f"l{src_layer}_k"
                tkey = f"l{tgt_layer}_k"
                qkey = f"l{tgt_layer}_q"
                svkey = f"l{src_layer}_v"
                tvkey = f"l{tgt_layer}_v"
                if (skey not in sz.files or tkey not in tz.files or
                        qkey not in tz.files):
                    continue
                wrong_layer = next(
                    (layer for layer in target_layers if layer != tgt_layer),
                    None)
                src_k = _flatten_heads_tokens(sz[skey])
                native_k = np.asarray(tz[tkey], dtype=np.float32)[0]
                q = np.asarray(tz[qkey], dtype=np.float32)[0]
                h = q.shape[0]
                native_scores = _attention_scores_np(q, native_k)
                native_weights = _softmax_np(native_scores)
                wrong_scores = None
                wrong_v_out = None
                if wrong_layer is not None:
                    wrong_k = np.asarray(tz[f"l{wrong_layer}_k"],
                                         dtype=np.float32)[0]
                    wrong_scores = _attention_scores_np(q, wrong_k)

                native_v = None
                native_out = None
                if tvkey in tz.files:
                    native_v = np.asarray(tz[tvkey], dtype=np.float32)[0]
                    native_out = _attention_output_np(
                        native_weights, native_v, h)
                    if wrong_layer is not None:
                        wrong_v = np.asarray(tz[f"l{wrong_layer}_v"],
                                             dtype=np.float32)[0]
                        wrong_v_out = _attention_output_np(
                            native_weights, wrong_v, h)

                for item in loaded:
                    artifacts = item["artifacts"]
                    key = (src_layer, tgt_layer, kind)
                    if key not in artifacts:
                        continue
                    k_art = artifacts[key]
                    pred_k = src_k @ k_art["weight"] + k_art["bias"]
                    pred_k = _unflatten_like(
                        pred_k, tz[tkey]).astype(np.float32)[0]
                    pred_scores = _attention_scores_np(q, pred_k)
                    shuffled_scores = _attention_scores_np(
                        q, pred_k[:, ::-1, :])
                    pred_weights = _softmax_np(pred_scores)

                    rec = item["metric_acc"].setdefault(
                        (src_layer, tgt_layer), {
                            "chunks": 0,
                            "tokens": 0,
                            "key_recall": [],
                            "shuffled_key_recall": [],
                            "wrong_layer_key_recall": [],
                            "wrong_layer": wrong_layer,
                        })
                    rec["chunks"] += 1
                    rec["tokens"] += int(q.shape[1])
                    rec["key_recall"].append(_topk_recall(
                        native_scores, pred_scores, topk))
                    rec["shuffled_key_recall"].append(_topk_recall(
                        native_scores, shuffled_scores, topk))
                    if wrong_scores is not None:
                        rec["wrong_layer_key_recall"].append(_topk_recall(
                            native_scores, wrong_scores, topk))

                    v_art_key = (src_layer, tgt_layer, "v")
                    if (v_art_key in artifacts and svkey in sz.files and
                            native_v is not None and native_out is not None):
                        src_v = _flatten_heads_tokens(sz[svkey])
                        v_art = artifacts[v_art_key]
                        pred_v = src_v @ v_art["weight"] + v_art["bias"]
                        pred_v = _unflatten_like(
                            pred_v, tz[tvkey]).astype(np.float32)[0]
                        v_only_out = _attention_output_np(
                            native_weights, pred_v, h)
                        k_only_out = _attention_output_np(
                            pred_weights, native_v, h)
                        kv_out = _attention_output_np(
                            pred_weights, pred_v, h)
                        _accumulate_output_fidelity(
                            rec, "v_only_value_output",
                            v_only_out, native_out)
                        _accumulate_output_fidelity(
                            rec, "k_only_native_value_output",
                            k_only_out, native_out)
                        _accumulate_output_fidelity(
                            rec, "translated_attention_value_output",
                            kv_out, native_out)
                        if wrong_v_out is not None:
                            _accumulate_output_fidelity(
                                rec, "wrong_layer_value_output",
                                wrong_v_out, native_out)
        if pair_idx == len(pairs) or pair_idx % 25 == 0:
            write_progress(pair_idx)

    results = []
    for item in loaded:
        rows = []
        for (src_layer, tgt_layer), rec in sorted(item["metric_acc"].items()):
            row = {
                "source_layer": src_layer,
                "target_layer": tgt_layer,
                "wrong_layer": rec.get("wrong_layer"),
                "chunks": rec["chunks"],
                "tokens": rec["tokens"],
                f"key_recall_at_{int(topk)}": _mean_or_none(
                    rec["key_recall"]),
                f"shuffled_key_recall_at_{int(topk)}": _mean_or_none(
                    rec["shuffled_key_recall"]),
                f"wrong_layer_key_recall_at_{int(topk)}": _mean_or_none(
                    rec["wrong_layer_key_recall"]),
            }
            _finish_output_fidelity(
                row, rec, "v_only_value_output", alias="value_output")
            _finish_output_fidelity(row, rec, "v_only_value_output")
            _finish_output_fidelity(row, rec, "k_only_native_value_output")
            _finish_output_fidelity(
                row, rec, "translated_attention_value_output")
            _finish_output_fidelity(row, rec, "wrong_layer_value_output")
            rows.append(row)

        eval_metrics = {
            "schema": "qwen35_graft_translation_eval_metrics_v1",
            "capture_dir": str(capture_dir),
            "translator_manifest": item["manifest"],
            "split": split,
            "topk": int(topk),
            "max_pairs": max_pairs,
            "paired_shards": len(pairs),
            "layers": rows,
            "sweep_progress": str(progress_out),
        }
        item["out"].parent.mkdir(parents=True, exist_ok=True)
        with open(item["out"], "w") as fh:
            json.dump(eval_metrics, fh, indent=2)
            fh.write("\n")
        results.append({
            "translator_dir": str(item["translator_dir"]),
            "out": str(item["out"]),
            "eval_metrics": eval_metrics,
        })

    write_progress(len(pairs), status="complete")
    return {
        "schema": "qwen35_graft_translation_eval_sweep_v1",
        "capture_dir": str(capture_dir),
        "split": split,
        "topk": int(topk),
        "max_pairs": max_pairs,
        "paired_shards": len(pairs),
        "progress": str(progress_out),
        "results": results,
    }


def evaluate_capture_identity(capture_dir, out_path, *, split="heldout",
                              topk=16):
    """Evaluate the target-side capture identity floor for G0 attention gates."""
    capture_dir = Path(capture_dir).expanduser()
    sidecars = _load_capture_sidecars(capture_dir).get("target", {})
    shards = []
    for key, meta in sorted(sidecars.items()):
        if split != "all" and meta.get("metadata", {}).get("split") != split:
            continue
        shards.append((key, meta))
    if not shards:
        raise ValueError(f"no target capture shards for split={split!r}")

    metric_acc = {}
    for _, meta in shards:
        z = np.load(meta["npz"])
        for tgt_layer in _capture_layers(meta):
            kkey = f"l{tgt_layer}_k"
            vkey = f"l{tgt_layer}_v"
            qkey = f"l{tgt_layer}_q"
            if kkey not in z.files or vkey not in z.files or qkey not in z.files:
                continue
            native_k = np.asarray(z[kkey])
            native_v = np.asarray(z[vkey])
            q = np.asarray(z[qkey])
            if native_k.ndim != 4 or native_v.ndim != 4 or q.ndim != 4:
                raise ValueError(
                    f"bad target capture rank for layer {tgt_layer}: "
                    f"k={native_k.shape}, v={native_v.shape}, q={q.shape}")
            if native_k.shape != native_v.shape:
                raise ValueError(
                    f"target K/V shape mismatch for layer {tgt_layer}: "
                    f"{native_k.shape} vs {native_v.shape}")
            if native_k.shape[0] != 1 or q.shape[0] != 1:
                raise ValueError(
                    f"expected batch-1 target capture for layer {tgt_layer}: "
                    f"k={native_k.shape}, q={q.shape}")
            if native_k.shape[2] != q.shape[2] or native_k.shape[3] != q.shape[3]:
                raise ValueError(
                    f"target K/Q token or dim mismatch for layer {tgt_layer}: "
                    f"k={native_k.shape}, q={q.shape}")
            if q.shape[1] % native_k.shape[1] != 0:
                raise ValueError(
                    f"target query heads must be a multiple of KV heads for "
                    f"layer {tgt_layer}: k={native_k.shape}, q={q.shape}")
            rec = metric_acc.setdefault(tgt_layer, {
                "chunks": 0,
                "tokens": 0,
                "key_recall": [],
                "non_finite_tensors": 0,
            })
            rec["chunks"] += 1
            rec["tokens"] += int(q.shape[2])
            rec["key_recall"].append(1.0)
            if (not np.isfinite(native_k).all() or
                    not np.isfinite(native_v).all() or
                    not np.isfinite(q).all()):
                rec["non_finite_tensors"] += 1
            output_values = int(q.shape[1] * q.shape[2] * q.shape[3])
            output_rows = int(q.shape[1] * q.shape[2])
            rec["identity_value_output_sse"] = 0.0
            rec["identity_value_output_count"] = (
                rec.get("identity_value_output_count", 0) + output_values)
            rec["identity_value_output_cos_sum"] = (
                rec.get("identity_value_output_cos_sum", 0.0) + output_rows)
            rec["identity_value_output_cos_count"] = (
                rec.get("identity_value_output_cos_count", 0) + output_rows)

    rows = []
    for tgt_layer, rec in sorted(metric_acc.items()):
        row = {
            "target_layer": tgt_layer,
            "chunks": rec["chunks"],
            "tokens": rec["tokens"],
            "non_finite_tensors": rec.get("non_finite_tensors", 0),
            f"identity_key_recall_at_{int(topk)}": _mean_or_none(
                rec["key_recall"]),
        }
        _finish_output_fidelity(row, rec, "identity_value_output")
        rows.append(row)

    metrics = {
        "schema": "qwen35_graft_translation_g0_capture_identity_v1",
        "capture_dir": str(capture_dir),
        "split": split,
        "topk": int(topk),
        "evaluation_mode": "structural_exact_identity",
        "target_shards": len(shards),
        "layers": rows,
    }
    out = Path(out_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(metrics, fh, indent=2)
        fh.write("\n")
    return metrics


_BINDING_NAMES = (
    "Aster", "Beryl", "Cinder", "Dorian", "Elara", "Fenn", "Galen", "Helio",
    "Ivara", "Juno", "Kestrel", "Lumen", "Mira", "Nadir", "Orion", "Pax",
    "Quill", "Riven", "Sable", "Talon", "Umbra", "Vega", "Wren", "Xyra",
    "Yarrow", "Zephyr", "Arden", "Briar", "Corin", "Dax", "Ember", "Flint",
)


def make_binding_probe_set(count=32, *, seed="qwen35-binding-v1"):
    """Create synthetic planted-fact probes for the G3 binding gate."""
    rng = np.random.default_rng(
        int.from_bytes(hashlib.sha256(str(seed).encode()).digest()[:8], "big"))
    probes = []
    used_codes = set()
    for i in range(int(count)):
        name = _BINDING_NAMES[i % len(_BINDING_NAMES)]
        suffix = int(rng.integers(1000, 9999))
        gold = f"{name[:2].upper()}-{suffix}"
        while gold in used_codes:
            suffix = int(rng.integers(1000, 9999))
            gold = f"{name[:2].upper()}-{suffix}"
        used_codes.add(gold)
        decoys = []
        while len(decoys) < 3:
            dname = _BINDING_NAMES[(i + len(decoys) + 7) % len(_BINDING_NAMES)]
            decoy = f"{dname[:2].upper()}-{int(rng.integers(1000, 9999))}"
            if decoy != gold and decoy not in decoys:
                decoys.append(decoy)
        probes.append({
            "id": f"bind-{i:03d}",
            "entity": name,
            "fact": (
                f"ARCHIVE FACT: Project {name} has access code {gold}. "
                f"This code belongs only to Project {name}."
            ),
            "question": (
                f"Question: What access code belongs to Project {name}?\n"
                "Answer:"
            ),
            "gold": f" {gold}",
            "decoys": [f" {d}" for d in decoys],
        })
    return {
        "schema": "qwen35_graft_translation_binding_probes_v1",
        "seed": str(seed),
        "count": int(count),
        "probes": probes,
    }


_BINDING_V2_QUERY_TEMPLATES = (
    "Lookup record {handle}. Stored value:\nAnswer:",
    "For handle {handle}, return the stored value.\nAnswer:",
)


def _binding_v2_code(rng):
    alphabet = np.array(list("abcdefghjkmnpqrstuvwxyz23456789"))
    parts = []
    for _ in range(3):
        chars = rng.choice(alphabet, size=3, replace=True)
        parts.append("".join(str(c) for c in chars))
    return "-".join(parts)


def make_binding_probe_set_v2(count=32, *, seed="qwen35-binding-v2",
                              templates=2):
    """Create opaque, flattened planted-fact probes for a harder G3 floor."""
    rng = np.random.default_rng(
        int.from_bytes(hashlib.sha256(str(seed).encode()).digest()[:8], "big"))
    templates = int(templates)
    if templates <= 0 or templates > len(_BINDING_V2_QUERY_TEMPLATES):
        raise ValueError(
            f"templates must be between 1 and "
            f"{len(_BINDING_V2_QUERY_TEMPLATES)}")
    probes = []
    used_codes = set()
    for binding_idx in range(int(count)):
        handle = f"h{binding_idx:02d}-{_binding_v2_code(rng)}"
        gold = _binding_v2_code(rng)
        while gold in used_codes:
            gold = _binding_v2_code(rng)
        used_codes.add(gold)
        decoys = []
        while len(decoys) < 3:
            decoy = _binding_v2_code(rng)
            if decoy != gold and decoy not in used_codes:
                decoys.append(decoy)
                used_codes.add(decoy)
        fact = (
            f"OPAQUE BINDING: handle {handle} stores value {gold}. "
            f"Only handle {handle} stores this value."
        )
        for template_idx, template in enumerate(
                _BINDING_V2_QUERY_TEMPLATES[:templates]):
            probes.append({
                "id": f"bind-v2-{binding_idx:03d}-q{template_idx}",
                "binding_id": f"bind-v2-{binding_idx:03d}",
                "query_template": template_idx,
                "handle": handle,
                "entity": handle,
                "fact": fact,
                "question": template.format(handle=handle),
                "gold": f" {gold}",
                "decoys": [f" {d}" for d in decoys],
                "surface_class": "opaque-code-3x3",
            })
    return {
        "schema": "qwen35_graft_translation_binding_probes_v2",
        "seed": str(seed),
        "binding_count": int(count),
        "templates_per_binding": templates,
        "count": len(probes),
        "flattened": True,
        "surface_class": "opaque-code-3x3",
        "probes": probes,
    }


def write_binding_probe_set(out_path, *, count=32, seed="qwen35-binding-v1"):
    probes = make_binding_probe_set(count=count, seed=seed)
    out = Path(out_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(probes, fh, indent=2)
        fh.write("\n")
    return probes


def write_binding_probe_set_v2(out_path, *, count=32,
                               seed="qwen35-binding-v2", templates=2):
    probes = make_binding_probe_set_v2(
        count=count,
        seed=seed,
        templates=templates,
    )
    out = Path(out_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(probes, fh, indent=2)
        fh.write("\n")
    return probes


def _candidate_scores_from_logprobs(candidate_logprobs, gold, decoys):
    gold_score = float(candidate_logprobs[gold])
    decoy_scores = [float(candidate_logprobs[d]) for d in decoys]
    best_decoy = max(decoy_scores) if decoy_scores else float("-inf")
    return {
        "gold_score": gold_score,
        "best_decoy_score": float(best_decoy),
        "gold_minus_best_decoy": float(gold_score - best_decoy),
        "success": bool(gold_score > best_decoy),
        "candidate_scores": {
            "gold": gold_score,
            "decoys": decoy_scores,
        },
    }


def _score_candidate_logprob(model, tokenizer, prompt, candidate,
                             graft=None, layers=None):
    sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
    import tensor_cuda as tc
    from core import kv_graft

    prompt_ids = _as_int_list(_encode_text(tokenizer, prompt))
    cand_ids = _as_int_list(_encode_text(tokenizer, candidate))
    if not prompt_ids:
        raise ValueError("binding probe prompt tokenized to zero tokens")
    if not cand_ids:
        raise ValueError("binding candidate tokenized to zero tokens")
    input_ids = prompt_ids + cand_ids[:-1]
    if graft is not None:
        kv_graft.set_injection(model, graft, layers=layers)
    try:
        with tc.no_grad():
            logits, _ = model(np.asarray([input_ids], dtype=np.int64),
                              last_token_only=False)
        arr = logits.float().numpy()[0]
    finally:
        if graft is not None:
            kv_graft.clear_injection(model)

    start = len(prompt_ids) - 1
    score = 0.0
    for offset, tok in enumerate(cand_ids):
        row = arr[start + offset]
        score += float(row[tok]) - _logsumexp_np(row)
    return score


def _score_probe_candidates(model, tokenizer, probe, *, graft=None,
                            layers=None, fact_in_context=False):
    prompt = probe["question"]
    if fact_in_context:
        prompt = f"{probe['fact']}\n\n{prompt}"
    candidates = [probe["gold"]] + list(probe.get("decoys", []))
    scores = {
        candidate: _score_candidate_logprob(
            model, tokenizer, prompt, candidate, graft=graft, layers=layers)
        for candidate in candidates
    }
    return _candidate_scores_from_logprobs(
        scores, probe["gold"], list(probe.get("decoys", [])))


def _translate_harvested_capture(source_capture, translator_dir, target_cfg):
    _, artifacts = _load_translator(translator_dir)
    out = [None] * int(target_cfg.num_layers)
    grouped = {}
    for src_layer, tgt_layer, kind in artifacts:
        grouped.setdefault((src_layer, tgt_layer), {})[kind] = artifacts[
            (src_layer, tgt_layer, kind)]
    for (src_layer, tgt_layer), kinds in sorted(grouped.items()):
        src = source_capture[src_layer]
        if src is None:
            continue
        rec = {}
        for kind, art in kinds.items():
            if kind not in src:
                continue
            flat = _flatten_heads_tokens(src[kind])
            pred = flat @ art["weight"] + art["bias"]
            heads = int(target_cfg.num_kv_heads)
            dim = int(target_cfg.head_dim)
            if pred.shape[1] != heads * dim:
                raise ValueError(
                    f"translator output width {pred.shape[1]} does not match "
                    f"target KV width {heads * dim}")
            rec[kind] = np.ascontiguousarray(
                pred.reshape(flat.shape[0], heads, dim)
                .transpose(1, 0, 2)[None].astype(np.float32))
        if "k" in rec and "v" in rec:
            out[int(tgt_layer)] = rec
    return out


def _binding_summary(rows, mode):
    selected = [r for r in rows if r["mode"] == mode]
    margins = [r["gold_minus_best_decoy"] for r in selected]
    successes = sum(1 for r in selected if r["success"])
    threshold = 14 if len(selected) >= 32 else None
    return {
        "mode": mode,
        "probes": len(selected),
        "positive_margins": successes,
        "mean_margin": float(np.mean(margins)) if margins else None,
        "min_margin": float(np.min(margins)) if margins else None,
        "passes_r1_g3_threshold": (
            None if threshold is None else successes >= threshold),
    }


def evaluate_binding_probes(probes_path, out_path, *, source_model_dir=None,
                            target_model_dir=None, translator_dir=None,
                            modes=("amnesia", "source-native",
                                   "target-native", "translated"),
                            max_probes=0, layers="all"):
    """Run the G3 gold-vs-decoy binding probe ladder."""
    sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
    from core.qwen35_tc import Qwen35_TC
    from core import kv_graft

    spec = _load_json(Path(probes_path).expanduser())
    probes = list(spec.get("probes", []))
    if max_probes:
        probes = probes[:int(max_probes)]
    modes = tuple(x.strip() for x in modes if x.strip())
    rows = []
    source_captures = {}
    tokenizer = None

    needs_source = any(m in modes for m in (
        "source-native", "source-context", "translated"))
    if needs_source:
        if not source_model_dir:
            raise ValueError("source_model_dir is required for source modes")
        tokenizer = load_hf_tokenizer(source_model_dir)
        source_model, source_info = Qwen35_TC.from_pretrained(source_model_dir)
        source_layers = _select_layers(source_model, layers)
        try:
            for probe in probes:
                fact_ids = _as_int_list(_encode_text(tokenizer, probe["fact"]))
                source_cap = kv_graft.harvest_kv(
                    source_model, fact_ids, layer_filter=source_layers)
                source_captures[probe["id"]] = source_cap
                if "source-native" in modes:
                    scored = _score_probe_candidates(
                        source_model, tokenizer, probe, graft=source_cap,
                        layers=source_layers)
                    rows.append({
                        "probe_id": probe["id"],
                        "mode": "source-native",
                        **scored,
                    })
                if "source-context" in modes:
                    scored = _score_probe_candidates(
                        source_model, tokenizer, probe, fact_in_context=True)
                    rows.append({
                        "probe_id": probe["id"],
                        "mode": "source-context",
                        **scored,
                    })
        finally:
            del source_model
            gc.collect()
    else:
        source_info = None

    needs_target = any(m in modes for m in (
        "amnesia", "target-native", "translated"))
    if needs_target:
        if not target_model_dir:
            raise ValueError("target_model_dir is required for target modes")
        tokenizer = tokenizer or load_hf_tokenizer(target_model_dir)
        target_model, target_info = Qwen35_TC.from_pretrained(target_model_dir)
        target_layers = _select_layers(target_model, layers)
        try:
            for probe in probes:
                if "amnesia" in modes:
                    scored = _score_probe_candidates(
                        target_model, tokenizer, probe)
                    rows.append({
                        "probe_id": probe["id"],
                        "mode": "amnesia",
                        **scored,
                    })
                if "target-native" in modes:
                    fact_ids = _as_int_list(
                        _encode_text(tokenizer, probe["fact"]))
                    target_cap = kv_graft.harvest_kv(
                        target_model, fact_ids, layer_filter=target_layers)
                    scored = _score_probe_candidates(
                        target_model, tokenizer, probe, graft=target_cap,
                        layers=target_layers)
                    rows.append({
                        "probe_id": probe["id"],
                        "mode": "target-native",
                        **scored,
                    })
                if "translated" in modes:
                    if translator_dir is None:
                        raise ValueError(
                            "translator_dir is required for translated mode")
                    source_cap = source_captures.get(probe["id"])
                    if source_cap is None:
                        raise ValueError(
                            "translated mode requires source captures")
                    graft = _translate_harvested_capture(
                        source_cap, translator_dir, target_model.config)
                    scored = _score_probe_candidates(
                        target_model, tokenizer, probe, graft=graft,
                        layers=target_layers)
                    rows.append({
                        "probe_id": probe["id"],
                        "mode": "translated",
                        **scored,
                    })
        finally:
            del target_model
            gc.collect()
    else:
        target_info = None

    summaries = [_binding_summary(rows, mode) for mode in modes]
    metrics = {
        "schema": "qwen35_graft_translation_binding_eval_v1",
        "probes_path": str(Path(probes_path).expanduser()),
        "source_model": source_info,
        "target_model": target_info,
        "translator_dir": (
            str(Path(translator_dir).expanduser()) if translator_dir else None),
        "modes": list(modes),
        "probe_count": len(probes),
        "summaries": summaries,
        "rows": rows,
    }
    out = Path(out_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(metrics, fh, indent=2)
        fh.write("\n")
    return metrics


def _binding_probe_lengths(probe, tokenizer=None):
    fields = {
        "fact": probe.get("fact", ""),
        "question": probe.get("question", ""),
        "gold": probe.get("gold", ""),
    }
    for idx, decoy in enumerate(probe.get("decoys", [])):
        fields[f"decoy_{idx}"] = decoy
    lengths = {
        name: {"chars": len(str(text))}
        for name, text in fields.items()
    }
    if tokenizer is not None:
        for name, text in fields.items():
            lengths[name]["tokens"] = len(_as_int_list(
                _encode_text(tokenizer, str(text))))
    return lengths


def analyze_binding_eval(binding_eval_path, out_path, *, probes_path=None,
                         tokenizer_dir=None):
    """Join binding eval rows back to probes and summarize floor/miss patterns."""
    binding_eval_path = Path(binding_eval_path).expanduser()
    metrics = _load_json(binding_eval_path)
    if metrics.get("schema") != "qwen35_graft_translation_binding_eval_v1":
        raise ValueError(
            f"expected qwen35_graft_translation_binding_eval_v1, got "
            f"{metrics.get('schema')!r}")
    probes_path = Path(
        probes_path or metrics.get("probes_path") or "").expanduser()
    if not str(probes_path):
        raise ValueError("probes_path is required")
    probe_spec = _load_json(probes_path)
    probes = {p["id"]: p for p in probe_spec.get("probes", [])}
    rows_by_probe = {}
    for row in metrics.get("rows", []):
        rows_by_probe.setdefault(row["probe_id"], {})[row["mode"]] = row

    tokenizer = None
    tokenizer_source = None
    if tokenizer_dir is None:
        source_model = metrics.get("source_model") or {}
        tokenizer_dir = source_model.get("model_dir")
    if tokenizer_dir:
        try:
            tokenizer = load_hf_tokenizer(tokenizer_dir)
            tokenizer_source = str(Path(tokenizer_dir).expanduser())
        except Exception as exc:  # pragma: no cover - defensive artifact note.
            tokenizer_source = f"unavailable: {exc}"

    per_probe = []
    translated_misses = []
    amnesia_successes = []
    translated_beats_amnesia = []
    amnesia_beats_translated = []
    mode_success = {}
    for probe_id in sorted(probes):
        probe = probes[probe_id]
        modes = rows_by_probe.get(probe_id, {})
        mode_records = {}
        for mode, row in sorted(modes.items()):
            decoy_scores = row.get("candidate_scores", {}).get("decoys", [])
            best_idx = None
            best_decoy = None
            if decoy_scores:
                best_idx = int(np.argmax(np.asarray(decoy_scores)))
                decoys = list(probe.get("decoys", []))
                if best_idx < len(decoys):
                    best_decoy = decoys[best_idx]
            mode_records[mode] = {
                "gold_score": row["gold_score"],
                "best_decoy_score": row["best_decoy_score"],
                "gold_minus_best_decoy": row["gold_minus_best_decoy"],
                "success": bool(row["success"]),
                "best_decoy_index": best_idx,
                "best_decoy": best_decoy,
            }
            rec = mode_success.setdefault(mode, {
                "probes": 0,
                "positive_margins": 0,
                "mean_margin_values": [],
                "min_margin": None,
            })
            rec["probes"] += 1
            rec["positive_margins"] += int(bool(row["success"]))
            margin = float(row["gold_minus_best_decoy"])
            rec["mean_margin_values"].append(margin)
            rec["min_margin"] = (
                margin if rec["min_margin"] is None
                else min(rec["min_margin"], margin))

        amnesia = mode_records.get("amnesia")
        translated = mode_records.get("translated")
        if translated is not None and not translated["success"]:
            translated_misses.append(probe_id)
        if amnesia is not None and amnesia["success"]:
            amnesia_successes.append(probe_id)
        if amnesia is not None and translated is not None:
            if (translated["gold_minus_best_decoy"] >
                    amnesia["gold_minus_best_decoy"]):
                translated_beats_amnesia.append(probe_id)
            elif (amnesia["gold_minus_best_decoy"] >
                  translated["gold_minus_best_decoy"]):
                amnesia_beats_translated.append(probe_id)

        per_probe.append({
            "probe_id": probe_id,
            "entity": probe.get("entity"),
            "fact": probe.get("fact"),
            "question": probe.get("question"),
            "gold": probe.get("gold"),
            "decoys": list(probe.get("decoys", [])),
            "lengths": _binding_probe_lengths(probe, tokenizer),
            "modes": mode_records,
        })

    mode_summary = []
    for mode, rec in sorted(mode_success.items()):
        margins = rec.pop("mean_margin_values")
        mode_summary.append({
            "mode": mode,
            "probes": rec["probes"],
            "positive_margins": rec["positive_margins"],
            "mean_margin": float(np.mean(margins)) if margins else None,
            "min_margin": rec["min_margin"],
        })

    analysis = {
        "schema": "qwen35_graft_translation_binding_analysis_v1",
        "binding_eval_path": str(binding_eval_path),
        "binding_eval_sha256": _sha256(binding_eval_path),
        "probes_path": str(probes_path),
        "probes_sha256": _sha256(probes_path),
        "tokenizer_source": tokenizer_source,
        "probe_count": len(probes),
        "modes": list(metrics.get("modes", [])),
        "mode_summary": mode_summary,
        "translated_misses": translated_misses,
        "amnesia_successes": amnesia_successes,
        "translated_beats_amnesia": translated_beats_amnesia,
        "amnesia_beats_translated": amnesia_beats_translated,
        "per_probe": per_probe,
    }
    out = Path(out_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(analysis, fh, indent=2)
        fh.write("\n")
    return analysis


def _tensor_to_numpy(t):
    try:
        return t.float().numpy()
    except AttributeError:
        return t.numpy()


def run_g0_logit_identity_smoke(model_dir, *, prefix_token_ids,
                                probe_token_ids, layers="all", out_path=None):
    """Run a live 9B->9B attention-graft identity smoke against logits.

    The prefix is harvested into attention K/V and also run once to obtain the
    non-attention recurrent cache. Probe tokens are then decoded with attention
    layers served from the graft and DeltaNet layers served from the prefix
    recurrent state. This is the live counterpart to the capture-only G0 floor.
    """
    sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
    import tensor_cuda as tc
    from core.qwen35_tc import Qwen35_TC
    from core import kv_graft

    prefix = np.asarray(prefix_token_ids, dtype=np.int64).reshape(-1)
    probe = np.asarray(probe_token_ids, dtype=np.int64).reshape(-1)
    if prefix.size == 0 or probe.size == 0:
        raise ValueError("prefix_token_ids and probe_token_ids must be non-empty")

    model, info = Qwen35_TC.from_pretrained(model_dir)
    layer_filter = _select_layers(model, layers)
    try:
        with tc.no_grad():
            full_ids = np.concatenate([prefix, probe])[None, :]
            full_logits, _ = model(full_ids, last_token_only=False)
            ref = _tensor_to_numpy(full_logits)[:,
                                                prefix.size:prefix.size + probe.size,
                                                :]
            captured = kv_graft.harvest_kv(
                model, prefix, layer_filter=layer_filter)
            _, prefix_caches = model(prefix[None, :], last_token_only=True)
            graft_caches = [
                None if model.config.is_attention(i) else cache
                for i, cache in enumerate(prefix_caches)
            ]
            kv_graft.set_injection(model, captured, layers=layer_filter)
            graft_logits = []
            caches = graft_caches
            for step, tok in enumerate(probe.tolist()):
                logits, caches = model(
                    np.asarray([[tok]], dtype=np.int64),
                    caches=caches,
                    position_offset=step,
                    last_token_only=True,
                )
                graft_logits.append(_tensor_to_numpy(logits))
            graft = np.concatenate(graft_logits, axis=1)
    finally:
        kv_graft.clear_injection(model)
        del model
        gc.collect()

    diff = graft.astype(np.float64) - ref.astype(np.float64)
    ref_top = np.argmax(ref, axis=-1)
    graft_top = np.argmax(graft, axis=-1)
    flips = int((ref_top != graft_top).sum())
    total = int(ref_top.size)
    result = {
        "schema": "qwen35_graft_translation_g0_logit_identity_smoke_v1",
        "model": info,
        "prefix_tokens": int(prefix.size),
        "probe_tokens": int(probe.size),
        "layers": sorted(int(x) for x in layer_filter),
        "max_abs_delta": float(np.max(np.abs(diff))),
        "mean_abs_delta": float(np.mean(np.abs(diff))),
        "top1_flips": flips,
        "top1_flip_rate": float(flips / max(total, 1)),
        "thresholds": {
            "max_abs_delta": 2e-3,
            "top1_flip_rate": 0.001,
        },
        "passes_r1_g0_threshold": (
            float(np.max(np.abs(diff))) <= 2e-3 and
            float(flips / max(total, 1)) <= 0.001
        ),
    }
    if out_path:
        out = Path(out_path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            json.dump(result, fh, indent=2)
            fh.write("\n")
    return result


def run_capture_corpus(plan_path, model_dir, *, role, out_dir, layers="all",
                       split="all", max_chunks=0, resume=True,
                       compress=True):
    sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
    from core.qwen35_tc import Qwen35_TC
    from core import kv_graft

    if role not in ("source", "target"):
        raise ValueError("role must be 'source' or 'target'")
    plan_path = Path(plan_path).expanduser()
    plan = _load_json(plan_path)
    model, info = Qwen35_TC.from_pretrained(model_dir)
    layer_filter = _select_layers(model, layers)
    processed = []
    skipped = 0
    max_chunks = int(max_chunks or 0)

    try:
        for chunk in plan.get("chunks", []):
            if split != "all" and chunk.get("split") != split:
                continue
            if resume and _capture_shard_done(out_dir, role, chunk):
                skipped += 1
                continue
            token_ids = np.asarray(chunk["token_ids"], dtype=np.int64)
            if role == "target":
                captured, queries = kv_graft.harvest_kv_and_queries(
                    model, token_ids, layer_filter=layer_filter)
            else:
                captured = kv_graft.harvest_kv(
                    model, token_ids, layer_filter=layer_filter)
                queries = None
            shard = write_capture_shard(
                out_dir,
                role=role,
                doc_id=chunk["doc_id"],
                chunk_id=chunk["chunk_id"],
                token_ids=token_ids,
                captured=captured,
                queries=queries,
                position_offset=chunk.get("token_start", 0),
                metadata={
                    "split": chunk.get("split"),
                    "doc_index": chunk.get("doc_index"),
                    "source_path": chunk.get("source_path"),
                    "label": chunk.get("label"),
                    "token_start": chunk.get("token_start"),
                    "token_end": chunk.get("token_end"),
                    "corpus_plan": str(plan_path),
                    "corpus_plan_sha256": _sha256(plan_path),
                    "layers_request": layers,
                },
                compress=compress,
            )
            processed.append(shard)
            del captured, queries
            gc.collect()
            if max_chunks and len(processed) >= max_chunks:
                break
    finally:
        del model
        gc.collect()

    cap_manifest = refresh_capture_manifest(out_dir, plan_path)
    return {
        "status": "ok",
        "role": role,
        "model": info,
        "processed_chunks": len(processed),
        "skipped_existing": skipped,
        "capture_manifest": cap_manifest,
        "processed": processed,
    }


def choose_next_capture_role(capture_manifest):
    expected = capture_manifest.get("expected", {})
    if not expected.get("source", {}).get("complete", False):
        return "source"
    if not expected.get("target", {}).get("complete", False):
        return "target"
    return None


def run_capture_next(plan_path, *, source_model_dir, target_model_dir, out_dir,
                     layers="all", split="all", source_max_chunks=64,
                     target_max_chunks=16, resume=True, compress=True):
    """Run the next needed corpus-capture role for cron/Claude loops."""
    before = refresh_capture_manifest(out_dir, plan_path)
    role = choose_next_capture_role(before)
    if role is None:
        return {
            "status": "complete",
            "selected_role": None,
            "processed_chunks": 0,
            "skipped_existing": 0,
            "capture_manifest": before,
        }
    model_dir = source_model_dir if role == "source" else target_model_dir
    max_chunks = source_max_chunks if role == "source" else target_max_chunks
    result = run_capture_corpus(
        plan_path,
        model_dir,
        role=role,
        out_dir=out_dir,
        layers=layers,
        split=split,
        max_chunks=max_chunks,
        resume=resume,
        compress=compress,
    )
    result["selected_role"] = role
    result["status"] = "ok"
    result["capture_manifest_before"] = before
    return result


def _artifact_file_ready(path):
    path = Path(path).expanduser()
    return path.is_file() and path.stat().st_size > 0


def _json_artifact_ready(path, schema=None):
    path = Path(path).expanduser()
    if not _artifact_file_ready(path):
        return False
    if schema is None:
        return True
    try:
        data = _load_json(path)
    except Exception:
        return False
    return data.get("schema") == schema


def _translator_manifest_ready(out_dir, *, control="normal", kinds="both"):
    path = Path(out_dir).expanduser() / "translator_manifest.json"
    if not _artifact_file_ready(path):
        return False
    try:
        data = _load_json(path)
    except Exception:
        return False
    return (
        data.get("schema") == "qwen35_graft_translation_translator_v1"
        and data.get("control") == control
        and data.get("kinds") == list(_fit_kinds(kinds))
    )


def _capture_progress_for_status(manifest):
    return {
        "out_dir": manifest.get("out_dir"),
        "expected": manifest.get("expected", {}),
        "roles": manifest.get("roles", {}),
        "paired": manifest.get("paired", {}),
    }


def _write_pipeline_status(status_path, payload):
    if not status_path:
        return payload
    out = Path(status_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    return payload


def _compact_pipeline_history_record(payload):
    rec = {
        "schema": "qwen35_graft_translation_pipeline_history_v1",
        "timestamp_utc": payload.get("timestamp_utc"),
        "status": payload.get("status"),
        "stage": payload.get("stage"),
    }
    for key in (
            "selected_role", "processed_chunks", "skipped_existing",
            "target_shards", "paired_shards", "artifacts", "layers",
            "probe_count", "count", "seed", "skipped", "reason"):
        if key in payload:
            rec[key] = payload[key]
    if "out" in payload:
        rec["out"] = payload["out"]
        out = Path(str(payload["out"])).expanduser()
        if out.is_file():
            rec["out_sha256"] = _sha256(out)
    if "span" in payload:
        rec["span"] = payload["span"]
    if "summaries" in payload:
        rec["summaries"] = payload["summaries"]
    if "max_abs_delta" in payload:
        rec["max_abs_delta"] = payload["max_abs_delta"]
    if "top1_flip_rate" in payload:
        rec["top1_flip_rate"] = payload["top1_flip_rate"]
    if "passes_r1_g0_threshold" in payload:
        rec["passes_r1_g0_threshold"] = payload[
            "passes_r1_g0_threshold"]
    capture = payload.get("capture_manifest", {})
    expected = capture.get("expected") if isinstance(capture, dict) else None
    if expected:
        rec["capture_expected"] = expected
    paired = capture.get("paired") if isinstance(capture, dict) else None
    if paired:
        rec["capture_paired"] = paired
    return rec


def _append_pipeline_history(history_path, payload):
    if not history_path:
        return
    out = Path(history_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as fh:
        json.dump(_compact_pipeline_history_record(payload), fh,
                  separators=(",", ":"))
        fh.write("\n")


def _first_probe_span_from_plan(plan_path, *, prefix_tokens=64,
                                probe_tokens=8):
    plan = _load_json(Path(plan_path).expanduser())
    needed = int(prefix_tokens) + int(probe_tokens)
    for preferred_split in ("heldout", "train", "all"):
        for chunk in plan.get("chunks", []):
            if preferred_split != "all" and chunk.get("split") != preferred_split:
                continue
            token_ids = [int(x) for x in chunk.get("token_ids", [])]
            if len(token_ids) >= needed:
                return {
                    "doc_id": chunk.get("doc_id"),
                    "chunk_id": int(chunk.get("chunk_id", -1)),
                    "split": chunk.get("split"),
                    "prefix_token_ids": token_ids[:int(prefix_tokens)],
                    "probe_token_ids": token_ids[
                        int(prefix_tokens):int(prefix_tokens) + int(probe_tokens)],
                }
    raise ValueError(
        f"no corpus chunk has enough tokens for a {prefix_tokens}+"
        f"{probe_tokens} live G0 span")


def _pipeline_result(status_path, *, stage, status="ok", history_path=None,
                     **payload):
    result = {
        "schema": "qwen35_graft_translation_pipeline_status_v1",
        "timestamp_utc": _utc_now(),
        "status": status,
        "stage": stage,
        **payload,
    }
    _write_pipeline_status(status_path, result)
    _append_pipeline_history(history_path, result)
    return result


def _run_pipeline_fit(capture_dir, out_dir, *, ridge_lambda, split,
                      control="normal", kinds="both",
                      compute_fit_metrics=True):
    return fit_ridge_translator(
        capture_dir,
        out_dir,
        ridge_lambda=ridge_lambda,
        split=split,
        control=control,
        kinds=kinds,
        compute_fit_metrics=compute_fit_metrics,
    )


def _pipeline_control_specs(root):
    return (
        ("fit-control-wrong-layer", root / "translator_wrong_layer",
         {"control": "wrong-layer", "kinds": "both"}),
        ("fit-control-shuffled-docs", root / "translator_shuffled_docs",
         {"control": "shuffled-docs", "kinds": "both"}),
        ("fit-control-k-only", root / "translator_k_only",
         {"control": "normal", "kinds": "k"}),
        ("fit-control-v-only", root / "translator_v_only",
         {"control": "normal", "kinds": "v"}),
    )


def inspect_pipeline_status(*, root, plan=None, capture_dir=None,
                            translator_dir=None, gates_dir=None,
                            binding_probes=None, status_out=None,
                            skip_live_g0=False, skip_binding_eval=False,
                            write_status=False):
    """Inspect the next PoC pipeline stage without loading models."""
    root = Path(root).expanduser()
    plan = Path(plan).expanduser() if plan else root / "corpus_plan.json"
    capture_dir = Path(capture_dir).expanduser() if capture_dir else root / "captures"
    translator_dir = (
        Path(translator_dir).expanduser() if translator_dir else root / "translator")
    gates_dir = Path(gates_dir).expanduser() if gates_dir else root / "gates"
    binding_probes = (
        Path(binding_probes).expanduser()
        if binding_probes else gates_dir / "binding_probes.json")
    status_out = (
        Path(status_out).expanduser() if status_out
        else root / "pipeline_status.json")

    if not plan.is_file():
        return _pipeline_result(
            status_out if write_status else None,
            stage="missing-corpus-plan",
            status="blocked",
            root=str(root),
            plan=str(plan),
            reason="corpus_plan.json is missing",
        )

    capture_manifest = refresh_capture_manifest(capture_dir, plan)
    capture_status = _capture_progress_for_status(capture_manifest)
    artifacts = {
        "g0_capture_identity": {
            "path": str(gates_dir / "g0_capture_identity_metrics.json"),
            "ready": _json_artifact_ready(
                gates_dir / "g0_capture_identity_metrics.json",
                "qwen35_graft_translation_g0_capture_identity_v1"),
        },
        "g0_logit_smoke": {
            "path": str(gates_dir / "g0_logit_identity_smoke.json"),
            "ready": _json_artifact_ready(
                gates_dir / "g0_logit_identity_smoke.json",
                "qwen35_graft_translation_g0_logit_identity_smoke_v1"),
            "skipped": bool(skip_live_g0),
        },
        "translator": {
            "path": str(translator_dir / "translator_manifest.json"),
            "ready": _translator_manifest_ready(
                translator_dir, control="normal", kinds="both"),
        },
        "eval_translator": {
            "path": str(translator_dir / "eval_metrics.json"),
            "ready": _json_artifact_ready(
                translator_dir / "eval_metrics.json",
                "qwen35_graft_translation_eval_metrics_v1"),
        },
        "binding_probes": {
            "path": str(binding_probes),
            "ready": _json_artifact_ready(
                binding_probes,
                "qwen35_graft_translation_binding_probes_v1"),
        },
        "binding_eval": {
            "path": str(gates_dir / "binding_eval_metrics.json"),
            "ready": _json_artifact_ready(
                gates_dir / "binding_eval_metrics.json",
                "qwen35_graft_translation_binding_eval_v1"),
            "skipped": bool(skip_binding_eval),
        },
    }
    for stage, out_dir, opts in _pipeline_control_specs(root):
        artifacts[stage] = {
            "path": str(out_dir / "translator_manifest.json"),
            "ready": _translator_manifest_ready(
                out_dir, control=opts["control"], kinds=opts["kinds"]),
            "expected_control": opts["control"],
            "expected_kinds": list(_fit_kinds(opts["kinds"])),
        }

    role = choose_next_capture_role(capture_manifest)
    if role is not None:
        return _pipeline_result(
            status_out if write_status else None,
            stage=f"capture-{role}",
            status="pending",
            root=str(root),
            plan=str(plan),
            capture_manifest=capture_status,
            artifacts=artifacts,
        )
    if not artifacts["g0_capture_identity"]["ready"]:
        stage = "g0-capture-identity"
    elif not skip_live_g0 and not artifacts["g0_logit_smoke"]["ready"]:
        stage = "g0-logit-smoke"
    elif not artifacts["translator"]["ready"]:
        stage = "fit-translator"
    else:
        stage = None
        for control_stage, _, _ in _pipeline_control_specs(root):
            if not artifacts[control_stage]["ready"]:
                stage = control_stage
                break
        if stage is None and not artifacts["eval_translator"]["ready"]:
            stage = "eval-translator"
        elif stage is None and not artifacts["binding_probes"]["ready"]:
            stage = "make-binding-probes"
        elif (stage is None and not skip_binding_eval
              and not artifacts["binding_eval"]["ready"]):
            stage = "eval-binding-probes"

    if stage is None:
        skipped = []
        if skip_live_g0:
            skipped.append("g0-logit-smoke")
        if skip_binding_eval:
            skipped.append("eval-binding-probes")
        return _pipeline_result(
            status_out if write_status else None,
            stage="complete",
            status="complete",
            root=str(root),
            plan=str(plan),
            skipped=skipped,
            capture_manifest=capture_status,
            artifacts=artifacts,
        )
    return _pipeline_result(
        status_out if write_status else None,
        stage=stage,
        status="pending",
        root=str(root),
        plan=str(plan),
        capture_manifest=capture_status,
        artifacts=artifacts,
    )


def run_pipeline_next(*, root, plan, source_model_dir, target_model_dir,
                      capture_dir=None, translator_dir=None, gates_dir=None,
                      binding_probes=None, status_out=None, history_out=None,
                      layers="all", source_max_chunks=64,
                      target_max_chunks=16, ridge_lambda=1e-4, topk=16,
                      binding_max_probes=32, binding_modes=(
                          "amnesia", "source-native", "target-native",
                          "translated"),
                      live_g0_prefix_tokens=64, live_g0_probe_tokens=8,
                      skip_live_g0=False, skip_binding_eval=False,
                      skip_fit_metrics=False, resume=True, compress=True):
    """Run one bounded missing step of the full Qwen3.5 translation PoC.

    This is the cron/Claude handoff entry point. It never runs overlapping
    stages inside one invocation: the caller can invoke it repeatedly until the
    returned status is `complete`.
    """
    root = Path(root).expanduser()
    plan = Path(plan).expanduser()
    capture_dir = Path(capture_dir).expanduser() if capture_dir else root / "captures"
    translator_dir = (
        Path(translator_dir).expanduser() if translator_dir else root / "translator")
    gates_dir = Path(gates_dir).expanduser() if gates_dir else root / "gates"
    binding_probes = (
        Path(binding_probes).expanduser()
        if binding_probes else gates_dir / "binding_probes.json")
    status_out = (
        Path(status_out).expanduser() if status_out
        else root / "pipeline_status.json")
    history_out = (
        Path(history_out).expanduser() if history_out
        else root / "pipeline_history.jsonl")

    def emit(*, stage, status="ok", **payload):
        return _pipeline_result(
            status_out,
            stage=stage,
            status=status,
            history_path=history_out,
            **payload,
        )

    capture_manifest = refresh_capture_manifest(capture_dir, plan)
    role = choose_next_capture_role(capture_manifest)
    if role is not None:
        result = run_capture_next(
            plan,
            source_model_dir=source_model_dir,
            target_model_dir=target_model_dir,
            out_dir=capture_dir,
            layers=layers,
            source_max_chunks=source_max_chunks,
            target_max_chunks=target_max_chunks,
            resume=resume,
            compress=compress,
        )
        return emit(
            stage=f"capture-{role}",
            selected_role=result["selected_role"],
            processed_chunks=result["processed_chunks"],
            skipped_existing=result["skipped_existing"],
            capture_manifest=_capture_progress_for_status(
                result["capture_manifest"]),
        )

    capture_status = _capture_progress_for_status(capture_manifest)
    g0_capture = gates_dir / "g0_capture_identity_metrics.json"
    if not _json_artifact_ready(
            g0_capture, "qwen35_graft_translation_g0_capture_identity_v1"):
        metrics = evaluate_capture_identity(
            capture_dir,
            g0_capture,
            split="heldout",
            topk=topk,
        )
        return emit(
            stage="g0-capture-identity",
            out=str(g0_capture),
            target_shards=metrics["target_shards"],
            capture_manifest=capture_status,
        )

    g0_logit = gates_dir / "g0_logit_identity_smoke.json"
    if (not skip_live_g0 and not _json_artifact_ready(
            g0_logit, "qwen35_graft_translation_g0_logit_identity_smoke_v1")):
        span = _first_probe_span_from_plan(
            plan,
            prefix_tokens=live_g0_prefix_tokens,
            probe_tokens=live_g0_probe_tokens,
        )
        metrics = run_g0_logit_identity_smoke(
            target_model_dir,
            prefix_token_ids=span["prefix_token_ids"],
            probe_token_ids=span["probe_token_ids"],
            layers=layers,
            out_path=g0_logit,
        )
        return emit(
            stage="g0-logit-smoke",
            out=str(g0_logit),
            span={k: v for k, v in span.items()
                  if k not in ("prefix_token_ids", "probe_token_ids")},
            max_abs_delta=metrics["max_abs_delta"],
            top1_flip_rate=metrics["top1_flip_rate"],
            passes_r1_g0_threshold=metrics["passes_r1_g0_threshold"],
            capture_manifest=capture_status,
        )

    if not _translator_manifest_ready(
            translator_dir, control="normal", kinds="both"):
        result = _run_pipeline_fit(
            capture_dir,
            translator_dir,
            ridge_lambda=ridge_lambda,
            split="train",
            compute_fit_metrics=not skip_fit_metrics,
        )
        return emit(
            stage="fit-translator",
            out=str(translator_dir),
            paired_shards=result["translator_manifest"]["paired_shards"],
            artifacts=len(result["translator_manifest"]["artifacts"]),
            fit_metrics_computed=(
                result["translator_manifest"].get("fit_metrics_computed")),
            capture_manifest=capture_status,
        )

    for stage, out_dir, opts in _pipeline_control_specs(root):
        if _translator_manifest_ready(
                out_dir, control=opts["control"], kinds=opts["kinds"]):
            continue
        result = _run_pipeline_fit(
            capture_dir,
            out_dir,
            ridge_lambda=ridge_lambda,
            split="train",
            control=opts["control"],
            kinds=opts["kinds"],
            compute_fit_metrics=not skip_fit_metrics,
        )
        return emit(
            stage=stage,
            out=str(out_dir),
            paired_shards=result["translator_manifest"]["paired_shards"],
            artifacts=len(result["translator_manifest"]["artifacts"]),
            fit_metrics_computed=(
                result["translator_manifest"].get("fit_metrics_computed")),
            capture_manifest=capture_status,
        )

    eval_path = translator_dir / "eval_metrics.json"
    if not _json_artifact_ready(
            eval_path, "qwen35_graft_translation_eval_metrics_v1"):
        metrics = evaluate_translator(
            capture_dir,
            translator_dir,
            eval_path,
            split="heldout",
            topk=topk,
        )
        return emit(
            stage="eval-translator",
            out=str(eval_path),
            paired_shards=metrics["paired_shards"],
            layers=len(metrics["layers"]),
            capture_manifest=capture_status,
        )

    if not _json_artifact_ready(
            binding_probes,
            "qwen35_graft_translation_binding_probes_v1"):
        probes = write_binding_probe_set(
            binding_probes,
            count=32,
            seed="qwen35-binding-v1",
        )
        return emit(
            stage="make-binding-probes",
            out=str(binding_probes),
            count=probes["count"],
            seed=probes["seed"],
            sha256=_sha256(binding_probes),
            capture_manifest=capture_status,
        )

    binding_eval = gates_dir / "binding_eval_metrics.json"
    if not skip_binding_eval and not _json_artifact_ready(
            binding_eval, "qwen35_graft_translation_binding_eval_v1"):
        metrics = evaluate_binding_probes(
            binding_probes,
            binding_eval,
            source_model_dir=source_model_dir,
            target_model_dir=target_model_dir,
            translator_dir=translator_dir,
            modes=binding_modes,
            max_probes=binding_max_probes,
            layers=layers,
        )
        return emit(
            stage="eval-binding-probes",
            out=str(binding_eval),
            probe_count=metrics["probe_count"],
            summaries=metrics["summaries"],
            capture_manifest=capture_status,
        )

    skipped = []
    if skip_live_g0:
        skipped.append("g0-logit-smoke")
    if skip_binding_eval:
        skipped.append("eval-binding-probes")
    return emit(
        stage="complete",
        status="complete",
        skipped=skipped,
        capture_manifest=capture_status,
    )


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Qwen3.5 graft translation PoC utilities")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser(
        "validate-weights",
        help="validate safetensors sources and write weights_manifest.json")
    v.add_argument("--source", required=True, help="Qwen3.5 source model dir")
    v.add_argument("--target", required=True, help="Qwen3.5 target model dir")
    v.add_argument("--out", required=True, help="output weights_manifest.json")
    c = sub.add_parser(
        "capture-smoke",
        help="load one model and write a tiny real K/V capture shard")
    c.add_argument("--model-dir", required=True, help="Qwen3.5 model dir")
    c.add_argument("--role", required=True, choices=("source", "target"))
    c.add_argument("--out-dir", required=True, help="output shard directory")
    c.add_argument("--doc-id", default="smoke")
    c.add_argument("--chunk-id", type=int, default=0)
    c.add_argument("--token-ids", default="1,2,3,4",
                   help="comma-separated token ids for the smoke chunk")
    c.add_argument("--layers", default="first",
                   help="'first', 'all', or comma-separated layer ids")
    pc = sub.add_parser(
        "plan-corpus",
        help="tokenize text files and write a document-split corpus plan")
    pc.add_argument("--model-dir", required=True, help="tokenizer source dir")
    pc.add_argument("--corpus", required=True, action="append",
                    help="text/jsonl file or directory; repeatable")
    pc.add_argument("--out", required=True, help="output corpus_plan.json")
    pc.add_argument("--chunk-tokens", type=int, default=512)
    pc.add_argument("--heldout-fraction", type=float, default=0.1)
    pc.add_argument("--seed", default="qwen35-translation-poc")
    pc.add_argument("--max-total-tokens", type=int, default=0,
                    help="0 means use all discovered tokens")
    pc.add_argument("--min-doc-tokens", type=int, default=1)
    pc.add_argument("--max-docs", type=int, default=0,
                    help="0 means no document cap")
    pc.add_argument("--source-label", default=None)
    cc = sub.add_parser(
        "capture-corpus",
        help="capture planned corpus chunks for one role, resumably")
    cc.add_argument("--plan", required=True, help="corpus_plan.json")
    cc.add_argument("--model-dir", required=True, help="Qwen3.5 model dir")
    cc.add_argument("--role", required=True, choices=("source", "target"))
    cc.add_argument("--out-dir", required=True, help="capture shard directory")
    cc.add_argument("--layers", default="all",
                    help="'all', 'first', or comma-separated layer ids")
    cc.add_argument("--split", default="all",
                    choices=("all", "train", "heldout"))
    cc.add_argument("--max-chunks", type=int, default=0,
                    help="0 means run until the selected split is complete")
    cc.add_argument("--no-resume", action="store_true",
                    help="overwrite/recompute shards instead of skipping them")
    cc.add_argument("--no-compress", action="store_true",
                    help="write uncompressed npz shards for speed")
    cn = sub.add_parser(
        "capture-next",
        help="run the next needed source/target capture batch for cron")
    cn.add_argument("--plan", required=True, help="corpus_plan.json")
    cn.add_argument("--source-model-dir", required=True,
                    help="Qwen3.5 source model dir")
    cn.add_argument("--target-model-dir", required=True,
                    help="Qwen3.5 target model dir")
    cn.add_argument("--out-dir", required=True, help="capture shard directory")
    cn.add_argument("--layers", default="all",
                    help="'all', 'first', or comma-separated layer ids")
    cn.add_argument("--split", default="all",
                    choices=("all", "train", "heldout"))
    cn.add_argument("--source-max-chunks", type=int, default=64)
    cn.add_argument("--target-max-chunks", type=int, default=16)
    cn.add_argument("--no-resume", action="store_true",
                    help="overwrite/recompute shards instead of skipping them")
    cn.add_argument("--no-compress", action="store_true",
                    help="write uncompressed npz shards for speed")
    cs = sub.add_parser(
        "capture-status",
        help="refresh and print capture_manifest.json completion status")
    cs.add_argument("--plan", required=True, help="corpus_plan.json")
    cs.add_argument("--out-dir", required=True, help="capture shard directory")
    ft = sub.add_parser(
        "fit-translator",
        help="fit per-layer ridge K/V translators from paired capture shards")
    ft.add_argument("--capture-dir", required=True,
                    help="directory containing source/target capture shards")
    ft.add_argument("--out-dir", required=True,
                    help="directory for translator artifacts")
    ft.add_argument("--ridge-lambda", type=float, default=1e-4)
    ft.add_argument("--split", default="train",
                    choices=("train", "heldout", "all"))
    ft.add_argument("--control", default="normal",
                    choices=("normal", "wrong-layer", "shuffled-docs"),
                    help="fit the normal translator or a negative control")
    ft.add_argument("--kinds", default="both",
                    help="'both', 'k', 'v', or 'k,v'")
    ft.add_argument("--backend", default="cpu",
                    choices=("cpu", "auto", "cupy", "cuda", "gpu", "numpy"),
                    help="ridge math backend; default preserves CPU behavior")
    ft.add_argument("--skip-fit-metrics", action="store_true",
                    help="write translator after solve without the train "
                         "fit-metrics rescan")
    fs = sub.add_parser(
        "fit-translator-sweep",
        help="fit several ridge translators from one shared accumulation pass")
    fs.add_argument("--capture-dir", required=True,
                    help="directory containing source/target capture shards")
    fs.add_argument("--out-root", required=True,
                    help="root directory for per-lambda translator dirs")
    fs.add_argument("--out-prefix", default="translator_ridge")
    fs.add_argument("--ridge-lambdas", required=True,
                    help="comma-separated ridge lambdas")
    fs.add_argument("--split", default="train",
                    choices=("train", "heldout", "all"))
    fs.add_argument("--control", default="normal",
                    choices=("normal", "wrong-layer", "shuffled-docs"),
                    help="fit the normal translator or a negative control")
    fs.add_argument("--kinds", default="both",
                    help="'both', 'k', 'v', or 'k,v'")
    fs.add_argument("--backend", default="cpu",
                    choices=("cpu", "auto", "cupy", "cuda", "gpu", "numpy"),
                    help="ridge math backend; default preserves CPU behavior")
    fs.add_argument("--skip-fit-metrics", action="store_true",
                    help="write translators after solve without the train "
                         "fit-metrics rescan")
    fl = sub.add_parser(
        "filter-translator-layers",
        help="write a translator manifest with selected layer pairs")
    fl.add_argument("--translator-dir", required=True,
                    help="source translator directory")
    fl.add_argument("--out-dir", required=True,
                    help="output filtered translator directory")
    fl.add_argument("--policy-name", required=True)
    fl.add_argument("--keep-pairs", default=None,
                    help="comma-separated source:target pairs to keep")
    fl.add_argument("--drop-pairs", default=None,
                    help="comma-separated source:target pairs to drop")
    ev = sub.add_parser(
        "eval-translator",
        help="evaluate fitted translators on paired held-out capture shards")
    ev.add_argument("--capture-dir", required=True)
    ev.add_argument("--translator-dir", required=True)
    ev.add_argument("--out", required=True, help="output eval_metrics.json")
    ev.add_argument("--split", default="heldout",
                    choices=("train", "heldout", "all"))
    ev.add_argument("--topk", type=int, default=16)
    ev.add_argument("--max-pairs", type=int, default=0,
                    help="optional bounded diagnostic shard count")
    es = sub.add_parser(
        "eval-translator-sweep",
        help="evaluate several fitted translators in one paired-capture pass")
    es.add_argument("--capture-dir", required=True)
    es.add_argument("--translator-dirs", required=True,
                    help="comma-separated translator directories")
    es.add_argument("--out-name", default="eval_metrics.json",
                    help="file name written inside each translator dir")
    es.add_argument("--progress-out", default=None,
                    help="optional shared progress JSON path")
    es.add_argument("--split", default="heldout",
                    choices=("train", "heldout", "all"))
    es.add_argument("--topk", type=int, default=16)
    es.add_argument("--max-pairs", type=int, default=0,
                    help="optional bounded diagnostic shard count")
    gi = sub.add_parser(
        "eval-g0-capture-identity",
        help="evaluate target capture identity floor for G0 attention gates")
    gi.add_argument("--capture-dir", required=True)
    gi.add_argument("--out", required=True,
                    help="output g0_capture_identity_metrics.json")
    gi.add_argument("--split", default="heldout",
                    choices=("train", "heldout", "all"))
    gi.add_argument("--topk", type=int, default=16)
    gl = sub.add_parser(
        "g0-logit-smoke",
        help="live capture/reinject logit identity smoke on one model")
    gl.add_argument("--model-dir", required=True)
    gl.add_argument("--prefix-token-ids", required=True,
                    help="comma-separated prefix token ids to harvest")
    gl.add_argument("--probe-token-ids", required=True,
                    help="comma-separated probe token ids to score")
    gl.add_argument("--layers", default="all",
                    help="'all', 'first', or comma-separated layer ids")
    gl.add_argument("--out", default=None,
                    help="optional output g0_logit_identity_smoke.json")
    bp = sub.add_parser(
        "make-binding-probes",
        help="write synthetic gold-vs-decoy probes for the G3 binding gate")
    bp.add_argument("--out", required=True,
                    help="output binding_probes.json")
    bp.add_argument("--count", type=int, default=32)
    bp.add_argument("--seed", default="qwen35-binding-v1")
    bp.add_argument("--version", default="v1", choices=("v1", "v2"))
    bp.add_argument("--templates", type=int, default=2,
                    help="V2 query templates per binding")
    be = sub.add_parser(
        "eval-binding-probes",
        help="evaluate G3 binding probes with native and translated grafts")
    be.add_argument("--probes", required=True, help="binding_probes.json")
    be.add_argument("--out", required=True,
                    help="output binding_eval_metrics.json")
    be.add_argument("--source-model-dir", default=None)
    be.add_argument("--target-model-dir", default=None)
    be.add_argument("--translator-dir", default=None)
    be.add_argument("--modes",
                    default="amnesia,source-native,target-native,translated",
                    help="comma-separated modes: amnesia,source-native,"
                         "source-context,target-native,translated")
    be.add_argument("--max-probes", type=int, default=0)
    be.add_argument("--layers", default="all",
                    help="'all', 'first', or comma-separated layer ids")
    ba = sub.add_parser(
        "analyze-binding-eval",
        help="summarize binding eval misses, floor behavior, and best decoys")
    ba.add_argument("--binding-eval", required=True,
                    help="binding_eval_metrics.json")
    ba.add_argument("--out", required=True,
                    help="output binding analysis json")
    ba.add_argument("--probes", default=None,
                    help="defaults to probes_path recorded in binding eval")
    ba.add_argument("--tokenizer-dir", default=None,
                    help="optional tokenizer/model dir for token lengths")
    pn = sub.add_parser(
        "pipeline-next",
        help="run one next missing PoC stage for cron/Claude orchestration")
    pn.add_argument("--root", required=True,
                    help="PoC artifact root")
    pn.add_argument("--plan", default=None,
                    help="corpus_plan.json; defaults to ROOT/corpus_plan.json")
    pn.add_argument("--source-model-dir", required=True,
                    help="Qwen3.5 source model dir")
    pn.add_argument("--target-model-dir", required=True,
                    help="Qwen3.5 target model dir")
    pn.add_argument("--capture-dir", default=None,
                    help="defaults to ROOT/captures")
    pn.add_argument("--translator-dir", default=None,
                    help="defaults to ROOT/translator")
    pn.add_argument("--gates-dir", default=None,
                    help="defaults to ROOT/gates")
    pn.add_argument("--binding-probes", default=None,
                    help="defaults to GATES/binding_probes.json")
    pn.add_argument("--status-out", default=None,
                    help="defaults to ROOT/pipeline_status.json")
    pn.add_argument("--history-out", default=None,
                    help="defaults to ROOT/pipeline_history.jsonl")
    pn.add_argument("--layers", default="all",
                    help="'all', 'first', or comma-separated layer ids")
    pn.add_argument("--source-max-chunks", type=int, default=64)
    pn.add_argument("--target-max-chunks", type=int, default=16)
    pn.add_argument("--ridge-lambda", type=float, default=1e-4)
    pn.add_argument("--topk", type=int, default=16)
    pn.add_argument("--binding-max-probes", type=int, default=32)
    pn.add_argument("--binding-modes",
                    default="amnesia,source-native,target-native,translated",
                    help="comma-separated eval-binding-probes modes")
    pn.add_argument("--live-g0-prefix-tokens", type=int, default=64)
    pn.add_argument("--live-g0-probe-tokens", type=int, default=8)
    pn.add_argument("--skip-live-g0", action="store_true",
                    help="do not run the live G0 GPU smoke")
    pn.add_argument("--skip-binding-eval", action="store_true",
                    help="stop after binding probe generation")
    pn.add_argument("--skip-fit-metrics", action="store_true",
                    help="write translators without the train fit-metrics "
                         "rescan")
    pn.add_argument("--no-resume", action="store_true",
                    help="overwrite/recompute capture shards")
    pn.add_argument("--no-compress", action="store_true",
                    help="write uncompressed capture shards")
    ps = sub.add_parser(
        "pipeline-status",
        help="inspect the next pipeline stage without loading models")
    ps.add_argument("--root", required=True,
                    help="PoC artifact root")
    ps.add_argument("--plan", default=None,
                    help="corpus_plan.json; defaults to ROOT/corpus_plan.json")
    ps.add_argument("--capture-dir", default=None,
                    help="defaults to ROOT/captures")
    ps.add_argument("--translator-dir", default=None,
                    help="defaults to ROOT/translator")
    ps.add_argument("--gates-dir", default=None,
                    help="defaults to ROOT/gates")
    ps.add_argument("--binding-probes", default=None,
                    help="defaults to GATES/binding_probes.json")
    ps.add_argument("--status-out", default=None,
                    help="defaults to ROOT/pipeline_status.json")
    ps.add_argument("--skip-live-g0", action="store_true",
                    help="report completion without requiring live G0")
    ps.add_argument("--skip-binding-eval", action="store_true",
                    help="report completion without requiring G3 eval")
    ps.add_argument("--write-status", action="store_true",
                    help="write the inspected status to --status-out")
    args = ap.parse_args(argv)
    if args.cmd == "validate-weights":
        manifest = write_weight_manifest(args.source, args.target, args.out)
        print(json.dumps({
            "status": "ok",
            "out": str(Path(args.out).expanduser()),
            "source_shards": manifest["source"]["shard_count"],
            "target_shards": manifest["target"]["shard_count"],
            "tokenizer_sha256": manifest["tokenizer_sha256"],
        }, indent=2))
        return 0
    if args.cmd == "capture-smoke":
        token_ids = [int(x.strip()) for x in args.token_ids.split(",")
                     if x.strip()]
        result = run_capture_smoke(
            args.model_dir,
            role=args.role,
            out_dir=args.out_dir,
            doc_id=args.doc_id,
            chunk_id=args.chunk_id,
            token_ids=token_ids,
            layers=args.layers,
        )
        print(json.dumps({
            "status": "ok",
            "model": result["model"],
            "capture": result["capture"],
        }, indent=2))
        return 0
    if args.cmd == "plan-corpus":
        tokenizer = load_hf_tokenizer(args.model_dir)
        plan = write_corpus_plan(
            args.corpus,
            tokenizer,
            args.out,
            chunk_tokens=args.chunk_tokens,
            heldout_fraction=args.heldout_fraction,
            seed=args.seed,
            max_total_tokens=args.max_total_tokens,
            min_doc_tokens=args.min_doc_tokens,
            max_docs=args.max_docs,
            source_label=args.source_label,
            command_line=" ".join(sys.argv),
        )
        print(json.dumps({
            "status": "ok",
            "out": str(Path(args.out).expanduser()),
            "totals": plan["totals"],
        }, indent=2))
        return 0
    if args.cmd == "capture-corpus":
        result = run_capture_corpus(
            args.plan,
            args.model_dir,
            role=args.role,
            out_dir=args.out_dir,
            layers=args.layers,
            split=args.split,
            max_chunks=args.max_chunks,
            resume=not args.no_resume,
            compress=not args.no_compress,
        )
        print(json.dumps({
            "status": "ok",
            "role": result["role"],
            "processed_chunks": result["processed_chunks"],
            "skipped_existing": result["skipped_existing"],
            "capture_manifest": {
                "out_dir": result["capture_manifest"]["out_dir"],
                "expected": result["capture_manifest"]["expected"],
                "roles": result["capture_manifest"]["roles"],
                "paired": result["capture_manifest"].get("paired", {}),
            },
        }, indent=2))
        return 0
    if args.cmd == "capture-next":
        result = run_capture_next(
            args.plan,
            source_model_dir=args.source_model_dir,
            target_model_dir=args.target_model_dir,
            out_dir=args.out_dir,
            layers=args.layers,
            split=args.split,
            source_max_chunks=args.source_max_chunks,
            target_max_chunks=args.target_max_chunks,
            resume=not args.no_resume,
            compress=not args.no_compress,
        )
        print(json.dumps({
            "status": result["status"],
            "selected_role": result["selected_role"],
            "processed_chunks": result["processed_chunks"],
            "skipped_existing": result["skipped_existing"],
            "capture_manifest": {
                "out_dir": result["capture_manifest"]["out_dir"],
                "expected": result["capture_manifest"]["expected"],
                "roles": result["capture_manifest"]["roles"],
                "paired": result["capture_manifest"].get("paired", {}),
            },
        }, indent=2))
        return 0
    if args.cmd == "capture-status":
        manifest = refresh_capture_manifest(args.out_dir, args.plan)
        print(json.dumps({
            "status": "ok",
            "capture_manifest": str(
                Path(args.out_dir).expanduser() / "capture_manifest.json"),
            "expected": manifest["expected"],
            "roles": manifest["roles"],
            "paired": manifest.get("paired", {}),
        }, indent=2))
        return 0
    if args.cmd == "fit-translator":
        result = fit_ridge_translator(
            args.capture_dir,
            args.out_dir,
            ridge_lambda=args.ridge_lambda,
            split=args.split,
            control=args.control,
            kinds=args.kinds,
            backend=args.backend,
            compute_fit_metrics=not args.skip_fit_metrics,
        )
        print(json.dumps({
            "status": "ok",
            "translator_manifest": result["translator_manifest"],
            "fit_metrics": result["fit_metrics"],
        }, indent=2))
        return 0
    if args.cmd == "fit-translator-sweep":
        lambdas = [x.strip() for x in args.ridge_lambdas.split(",")
                   if x.strip()]
        result = fit_ridge_translator_sweep(
            args.capture_dir,
            args.out_root,
            ridge_lambdas=lambdas,
            out_prefix=args.out_prefix,
            split=args.split,
            control=args.control,
            kinds=args.kinds,
            backend=args.backend,
            compute_fit_metrics=not args.skip_fit_metrics,
        )
        print(json.dumps({
            "status": "ok",
            "schema": result["schema"],
            "out_root": result["out_root"],
            "ridge_lambdas": result["ridge_lambdas"],
            "compute_backend": result["compute_backend"],
            "fit_metrics_computed": result["fit_metrics_computed"],
            "results": [
                item["translator_manifest"]
                for item in result["results"]
            ],
        }, indent=2))
        return 0
    if args.cmd == "filter-translator-layers":
        manifest = filter_translator_layers(
            args.translator_dir,
            args.out_dir,
            policy_name=args.policy_name,
            keep_pairs=args.keep_pairs,
            drop_pairs=args.drop_pairs,
        )
        print(json.dumps({
            "status": "ok",
            "out": str(
                Path(args.out_dir).expanduser() / "translator_manifest.json"),
            "policy_name": manifest["layer_policy_name"],
            "artifact_count": manifest["artifact_count"],
            "kept_pairs": manifest["layer_policy"]["kept_pairs"],
            "dropped_pairs": manifest["layer_policy"]["dropped_pairs"],
        }, indent=2))
        return 0
    if args.cmd == "eval-translator":
        metrics = evaluate_translator(
            args.capture_dir,
            args.translator_dir,
            args.out,
            split=args.split,
            topk=args.topk,
            max_pairs=args.max_pairs,
        )
        print(json.dumps({
            "status": "ok",
            "out": str(Path(args.out).expanduser()),
            "max_pairs": metrics["max_pairs"],
            "paired_shards": metrics["paired_shards"],
            "layers": metrics["layers"],
        }, indent=2))
        return 0
    if args.cmd == "eval-translator-sweep":
        translator_dirs = [
            x.strip() for x in args.translator_dirs.split(",") if x.strip()
        ]
        result = evaluate_translator_sweep(
            args.capture_dir,
            translator_dirs,
            out_name=args.out_name,
            progress_out=args.progress_out,
            split=args.split,
            topk=args.topk,
            max_pairs=args.max_pairs,
        )
        print(json.dumps({
            "status": "ok",
            "schema": result["schema"],
            "capture_dir": result["capture_dir"],
            "split": result["split"],
            "topk": result["topk"],
            "max_pairs": result["max_pairs"],
            "paired_shards": result["paired_shards"],
            "progress": result["progress"],
            "results": [
                {
                    "translator_dir": item["translator_dir"],
                    "out": item["out"],
                    "layers": len(item["eval_metrics"]["layers"]),
                }
                for item in result["results"]
            ],
        }, indent=2))
        return 0
    if args.cmd == "eval-g0-capture-identity":
        metrics = evaluate_capture_identity(
            args.capture_dir,
            args.out,
            split=args.split,
            topk=args.topk,
        )
        print(json.dumps({
            "status": "ok",
            "out": str(Path(args.out).expanduser()),
            "target_shards": metrics["target_shards"],
            "layers": metrics["layers"],
        }, indent=2))
        return 0
    if args.cmd == "g0-logit-smoke":
        prefix_ids = [
            int(x.strip()) for x in args.prefix_token_ids.split(",")
            if x.strip()
        ]
        probe_ids = [
            int(x.strip()) for x in args.probe_token_ids.split(",")
            if x.strip()
        ]
        result = run_g0_logit_identity_smoke(
            args.model_dir,
            prefix_token_ids=prefix_ids,
            probe_token_ids=probe_ids,
            layers=args.layers,
            out_path=args.out,
        )
        print(json.dumps({
            "status": "ok",
            "out": str(Path(args.out).expanduser()) if args.out else None,
            "max_abs_delta": result["max_abs_delta"],
            "top1_flips": result["top1_flips"],
            "top1_flip_rate": result["top1_flip_rate"],
            "passes_r1_g0_threshold": result["passes_r1_g0_threshold"],
        }, indent=2))
        return 0
    if args.cmd == "make-binding-probes":
        if args.version == "v2":
            seed = (
                args.seed if args.seed != "qwen35-binding-v1"
                else "qwen35-binding-v2")
            probes = write_binding_probe_set_v2(
                args.out,
                count=args.count,
                seed=seed,
                templates=args.templates,
            )
        else:
            probes = write_binding_probe_set(
                args.out,
                count=args.count,
                seed=args.seed,
            )
        print(json.dumps({
            "status": "ok",
            "out": str(Path(args.out).expanduser()),
            "schema": probes["schema"],
            "count": probes["count"],
            "seed": probes["seed"],
        }, indent=2))
        return 0
    if args.cmd == "eval-binding-probes":
        modes = [x.strip() for x in args.modes.split(",") if x.strip()]
        metrics = evaluate_binding_probes(
            args.probes,
            args.out,
            source_model_dir=args.source_model_dir,
            target_model_dir=args.target_model_dir,
            translator_dir=args.translator_dir,
            modes=modes,
            max_probes=args.max_probes,
            layers=args.layers,
        )
        print(json.dumps({
            "status": "ok",
            "out": str(Path(args.out).expanduser()),
            "probe_count": metrics["probe_count"],
            "summaries": metrics["summaries"],
        }, indent=2))
        return 0
    if args.cmd == "analyze-binding-eval":
        analysis = analyze_binding_eval(
            args.binding_eval,
            args.out,
            probes_path=args.probes,
            tokenizer_dir=args.tokenizer_dir,
        )
        print(json.dumps({
            "status": "ok",
            "out": str(Path(args.out).expanduser()),
            "probe_count": analysis["probe_count"],
            "translated_misses": len(analysis["translated_misses"]),
            "amnesia_successes": len(analysis["amnesia_successes"]),
            "translated_beats_amnesia": len(
                analysis["translated_beats_amnesia"]),
            "amnesia_beats_translated": len(
                analysis["amnesia_beats_translated"]),
        }, indent=2))
        return 0
    if args.cmd == "pipeline-next":
        root = Path(args.root).expanduser()
        plan = Path(args.plan).expanduser() if args.plan else (
            root / "corpus_plan.json")
        modes = [x.strip() for x in args.binding_modes.split(",")
                 if x.strip()]
        result = run_pipeline_next(
            root=root,
            plan=plan,
            source_model_dir=args.source_model_dir,
            target_model_dir=args.target_model_dir,
            capture_dir=args.capture_dir,
            translator_dir=args.translator_dir,
            gates_dir=args.gates_dir,
            binding_probes=args.binding_probes,
            status_out=args.status_out,
            history_out=args.history_out,
            layers=args.layers,
            source_max_chunks=args.source_max_chunks,
            target_max_chunks=args.target_max_chunks,
            ridge_lambda=args.ridge_lambda,
            topk=args.topk,
            binding_max_probes=args.binding_max_probes,
            binding_modes=modes,
            live_g0_prefix_tokens=args.live_g0_prefix_tokens,
            live_g0_probe_tokens=args.live_g0_probe_tokens,
            skip_live_g0=args.skip_live_g0,
            skip_binding_eval=args.skip_binding_eval,
            skip_fit_metrics=args.skip_fit_metrics,
            resume=not args.no_resume,
            compress=not args.no_compress,
        )
        print(json.dumps(result, indent=2))
        return 0
    if args.cmd == "pipeline-status":
        result = inspect_pipeline_status(
            root=args.root,
            plan=args.plan,
            capture_dir=args.capture_dir,
            translator_dir=args.translator_dir,
            gates_dir=args.gates_dir,
            binding_probes=args.binding_probes,
            status_out=args.status_out,
            skip_live_g0=args.skip_live_g0,
            skip_binding_eval=args.skip_binding_eval,
            write_status=args.write_status,
        )
        print(json.dumps(result, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
