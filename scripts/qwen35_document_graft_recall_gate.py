#!/usr/bin/env python3
"""End-to-end Qwen3.5 2B->9B document graft recall gate.

This is intentionally different from the binding logprob gates: it captures a
real text document through Qwen3.5-2B, translates the harvested attention graft
into Qwen3.5-9B space, withholds the document text from the prompt, and asks
Qwen3.5-9B to answer greedily from the graft.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, "/mnt/ForgeRealm/Project-Tensor/tensor_cuda")
sys.path.insert(0, str(REPO))

import tensor_cuda as tc  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from core import kv_graft  # noqa: E402
from core.qwen35_tc import Qwen35_TC  # noqa: E402
from core.qwen35_translation_poc import (  # noqa: E402
    _select_layers,
    _sha256,
    _translate_harvested_capture,
)


DEFAULT_SOURCE = (
    "/home/vader/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/"
    "snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc"
)
DEFAULT_TARGET = (
    "/home/vader/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/"
    "snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
)
DEFAULT_TRANSLATOR = (
    "/mnt/ForgeRealm/qwen35_graft_translation_poc/"
    "translator_r47_residual_micro_sweep/translator_r47_micro_s0p6875_kv"
)
DEFAULT_OUT = (
    "/mnt/ForgeRealm/qwen35_graft_translation_poc/gates/"
    "document_recall_r49_translated_s0p6875.json"
)

DEFAULT_DOCUMENT = """\
Internal transfer note R49-needle.

The vault handle celadon-orbit stores retrieval code n7q-41-vesta.
The depot handle iron-kestrel stores retrieval code m4x-88-lumen.
The archive handle violet-sundial stores retrieval code t9p-03-cairn.
The relay handle amber-quartz stores retrieval code b6h-72-nadir.

These four handle-to-code bindings are private to this note. Do not infer them
from handle spelling; retrieve the exact code written beside the handle.
"""

DEFAULT_QUESTIONS = [
    {
        "id": "celadon-orbit",
        "question": (
            "The document is not in this prompt. Using the mounted graft only, "
            "what retrieval code is stored for vault handle celadon-orbit? "
            "Answer with exactly the code."
        ),
        "answer": "n7q-41-vesta",
    },
    {
        "id": "iron-kestrel",
        "question": (
            "The document is not in this prompt. Using the mounted graft only, "
            "what retrieval code is stored for depot handle iron-kestrel? "
            "Answer with exactly the code."
        ),
        "answer": "m4x-88-lumen",
    },
    {
        "id": "violet-sundial",
        "question": (
            "The document is not in this prompt. Using the mounted graft only, "
            "what retrieval code is stored for archive handle violet-sundial? "
            "Answer with exactly the code."
        ),
        "answer": "t9p-03-cairn",
    },
    {
        "id": "amber-quartz",
        "question": (
            "The document is not in this prompt. Using the mounted graft only, "
            "what retrieval code is stored for relay handle amber-quartz? "
            "Answer with exactly the code."
        ),
        "answer": "b6h-72-nadir",
    },
]


def utc_now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def encode_plain(tokenizer, text):
    return np.asarray(
        tokenizer(text, add_special_tokens=False).input_ids,
        dtype=np.int64,
    )


def encode_chat(tokenizer, prompt):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tokenizer(text, return_tensors="np").input_ids.astype(np.int64)


def normalize(text):
    keep = []
    for ch in text.lower():
        keep.append(ch if ch.isalnum() else " ")
    return " ".join("".join(keep).split())


def answer_hit(generated, expected):
    return normalize(expected) in normalize(generated)


def greedy_generate(model, tokenizer, prompt, *, graft=None, layers=None,
                    max_new_tokens=32):
    ids = encode_chat(tokenizer, prompt)
    mounted = graft is not None
    start = time.perf_counter()
    out = []
    with tc.no_grad():
        if mounted:
            kv_graft.set_injection(model, graft, layers=layers)
        logits, caches = model(ids, last_token_only=True)
        if mounted:
            kv_graft.clear_injection(model, free_seats=False)
        nxt = int(logits.float().numpy()[0, -1].argmax())
        off = int(ids.shape[1])
        for _ in range(int(max_new_tokens)):
            if nxt == model.config.eos_token_id:
                break
            out.append(nxt)
            logits, caches = model(
                np.asarray([[nxt]], dtype=np.int64),
                caches=caches,
                position_offset=off,
                last_token_only=True,
            )
            off += 1
            nxt = int(logits.float().numpy()[0, -1].argmax())
    if mounted:
        kv_graft.clear_injection(model, free_seats=True)
    tc.synchronize()
    elapsed = time.perf_counter() - start
    text = tokenizer.decode(out, skip_special_tokens=True)
    return {
        "prompt_tokens": int(ids.shape[1]),
        "generated_tokens": len(out),
        "elapsed_sec": elapsed,
        "text": text,
    }


def run_gate(args):
    source_model_dir = Path(args.source_model_dir).expanduser()
    target_model_dir = Path(args.target_model_dir).expanduser()
    translator_dir = Path(args.translator_dir).expanduser()
    document = DEFAULT_DOCUMENT
    if args.document:
        document = Path(args.document).expanduser().read_text()
    questions = DEFAULT_QUESTIONS
    if args.questions:
        questions = json.loads(Path(args.questions).expanduser().read_text())

    source_tokenizer = AutoTokenizer.from_pretrained(source_model_dir)
    target_tokenizer = AutoTokenizer.from_pretrained(target_model_dir)
    doc_ids_source = encode_plain(source_tokenizer, document)
    doc_ids_target = encode_plain(target_tokenizer, document)

    print("loading source 2B", flush=True)
    source_model, source_info = Qwen35_TC.from_pretrained(str(source_model_dir))
    source_layers = _select_layers(source_model, args.layers)
    try:
        print(f"harvesting source document: {doc_ids_source.size} tokens",
              flush=True)
        source_capture = kv_graft.harvest_kv(
            source_model,
            doc_ids_source,
            layer_filter=source_layers,
        )
    finally:
        del source_model
        gc.collect()

    print("loading target 9B", flush=True)
    target_model, target_info = Qwen35_TC.from_pretrained(str(target_model_dir))
    target_layers = _select_layers(target_model, args.layers)
    try:
        print("translating source graft into target space", flush=True)
        translated_graft = _translate_harvested_capture(
            source_capture,
            translator_dir,
            target_model.config,
        )
        translated_layers = [
            idx for idx, rec in enumerate(translated_graft) if rec is not None
        ]

        print("harvesting target-native control graft", flush=True)
        target_native_graft = kv_graft.harvest_kv(
            target_model,
            doc_ids_target,
            layer_filter=target_layers,
        )

        rows = []
        translated_layer_set = set(int(x) for x in translated_layers)
        modes = [
            "amnesia",
            "translated",
            "target-native-matched-layers",
            "target-native",
            "target-context",
        ]
        for qidx, q in enumerate(questions, start=1):
            print(f"question {qidx}/{len(questions)}: {q['id']}", flush=True)
            prompts = {
                "amnesia": q["question"],
                "translated": q["question"],
                "target-native-matched-layers": q["question"],
                "target-native": q["question"],
                "target-context": (
                    f"Document:\n{document}\n\nQuestion:\n{q['question']}"
                ),
            }
            grafts = {
                "amnesia": None,
                "translated": translated_graft,
                "target-native-matched-layers": target_native_graft,
                "target-native": target_native_graft,
                "target-context": None,
            }
            mode_layers = {
                "amnesia": target_layers,
                "translated": target_layers,
                "target-native-matched-layers": translated_layer_set,
                "target-native": target_layers,
                "target-context": target_layers,
            }
            for mode in modes:
                gen = greedy_generate(
                    target_model,
                    target_tokenizer,
                    prompts[mode],
                    graft=grafts[mode],
                    layers=mode_layers[mode],
                    max_new_tokens=args.max_new_tokens,
                )
                rows.append({
                    "question_id": q["id"],
                    "mode": mode,
                    "expected": q["answer"],
                    "success": answer_hit(gen["text"], q["answer"]),
                    "generated": gen["text"],
                    "prompt_tokens": gen["prompt_tokens"],
                    "generated_tokens": gen["generated_tokens"],
                    "elapsed_sec": gen["elapsed_sec"],
                })
    finally:
        kv_graft.clear_injection(target_model, free_seats=True)
        del target_model
        gc.collect()

    summaries = []
    for mode in sorted({row["mode"] for row in rows}):
        selected = [row for row in rows if row["mode"] == mode]
        summaries.append({
            "mode": mode,
            "questions": len(selected),
            "correct": sum(1 for row in selected if row["success"]),
            "accuracy": (
                sum(1 for row in selected if row["success"]) /
                max(len(selected), 1)
            ),
        })

    out = {
        "schema": "qwen35_graft_translation_document_recall_gate_v1",
        "generated_utc": utc_now(),
        "source_model": source_info,
        "target_model": target_info,
        "translator_dir": str(translator_dir),
        "translator_manifest_sha256": _sha256(
            translator_dir / "translator_manifest.json"),
        "layers": args.layers,
        "target_layers": sorted(int(x) for x in target_layers),
        "translated_layers": translated_layers,
        "document_sha256": hashlib.sha256(
            document.encode("utf-8")).hexdigest(),
        "document_tokens_source": int(doc_ids_source.size),
        "document_tokens_target": int(doc_ids_target.size),
        "question_count": len(questions),
        "max_new_tokens": int(args.max_new_tokens),
        "summaries": summaries,
        "rows": rows,
    }
    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
        fh.write("\n")
    return out_path, out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model-dir", default=DEFAULT_SOURCE)
    parser.add_argument("--target-model-dir", default=DEFAULT_TARGET)
    parser.add_argument("--translator-dir", default=DEFAULT_TRANSLATOR)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--document", default=None,
                        help="optional text document path")
    parser.add_argument("--questions", default=None,
                        help="optional JSON list of question specs")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()
    out_path, result = run_gate(args)
    print(json.dumps({
        "status": "ok",
        "out": str(out_path),
        "summaries": result["summaries"],
    }, indent=2))


if __name__ == "__main__":
    main()
