#!/usr/bin/env python3
"""NC17-P0 step 1: download Qwen/Qwen3-1.7B (bf16 safetensors) to the HF cache.

Records the resolved revision hash and on-disk size. No GPU. No git.
Writes logs/nc17/p0_download.json.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "Qwen/Qwen3-1.7B"
REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "logs" / "nc17" / "p0_download.json"


def dir_size_bytes(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path, followlinks=False):
        for f in files:
            fp = os.path.join(root, f)
            try:
                # count real bytes of the target the symlink points to
                total += os.path.getsize(os.path.realpath(fp))
            except OSError:
                pass
    return total


def main() -> int:
    # bf16 safetensors: exclude gguf/onnx variants if any; grab config+weights+tokenizer
    local_dir = snapshot_download(
        repo_id=REPO_ID,
        allow_patterns=[
            "*.safetensors",
            "*.json",
            "*.txt",
            "tokenizer*",
            "*.model",
            "merges.txt",
            "vocab.json",
        ],
    )
    # Resolve revision hash from the cache snapshot symlink target
    snap = Path(local_dir)
    revision = snap.name  # snapshots/<commit-hash>
    # Confirm via git-style: the parent structure is .../snapshots/<hash>
    parent = snap.parent
    revision_hash = revision if parent.name == "snapshots" else revision

    size_bytes = dir_size_bytes(local_dir)

    # List safetensors files
    st_files = sorted(str(p.name) for p in snap.glob("*.safetensors"))

    info = {
        "repo_id": REPO_ID,
        "revision_hash": revision_hash,
        "local_dir": str(local_dir),
        "size_bytes": size_bytes,
        "size_gib": round(size_bytes / (1024**3), 4),
        "safetensors_files": st_files,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
