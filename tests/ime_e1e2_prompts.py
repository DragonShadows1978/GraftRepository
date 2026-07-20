"""Shared workload for the Hybrid IME E1/E2 experiments (plan
docs/HYBRID_IME_E1E2_PLAN.md, "Shared workload"). Both experiments import
this module so they run on the IDENTICAL prompt set.

Three prompts, each truncated to ~2048 tokens of PREFILL under the
Qwen3.5 tokenizer, then 64 greedy decode steps in the experiment scripts:
  1. natural prose  -- this repo's README.md
  2. code           -- a real source file from this repo (core/mistral7b_tc.py)
  3. synthetic retrieval -- 40 planted key->value facts, queried at the end.

Determinism: fixed on-disk source files, fixed seed for the synthetic
generator, fixed target token budget. `build_prompts(tok)` returns a list
of (name, ids_np) with ids_np shaped (1, S), S ~= TARGET_TOKENS.

  from tests.ime_e1e2_prompts import build_prompts, TARGET_TOKENS
"""
import os
import random

import numpy as np

TARGET_TOKENS = 2048
SEED = 20260720

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROSE_FILE = os.path.join(_REPO, "README.md")
CODE_FILE = os.path.join(_REPO, "core", "mistral7b_tc.py")

# 40 synthetic facts: (unit code, attribute, value). Deterministic content;
# codes are distinctive tokens so retrieval is unambiguous.
_UNITS = [
    "VORMEK-4821", "KALDReN-7133", "THALOS-2907", "BRIMWAY-5510",
    "OCTANE-6644", "PELVAR-3388", "SUNDER-1902", "GALVEX-7756",
    "MORWEN-4013", "TESSRA-8829", "DRUVEN-2264", "HOLLIX-9471",
    "CANVER-3805", "PRYSMA-6120", "ZEPHOR-4497", "MARLOW-7302",
    "QUENDA-1548", "VESPER-8863", "ORRINE-2276", "FALCOR-9910",
    "GRENDL-3054", "SILVAN-6687", "TORVIK-4429", "WISPEN-7741",
    "CADMUS-1183", "NEVARA-8506", "OLBRIK-2938", "PYRENE-6015",
    "DELMAR-4472", "SORVEX-9127", "HAVLIN-3690", "MIRDAX-7854",
    "CORVUS-1406", "ELDRIN-8231", "TANVIR-2069", "GOSHEN-6773",
    "RALKEN-4188", "USHARA-9542", "BENTOR-3317", "WYVARN-7960",
]
_ATTRS = [
    ("runtime hours logged", lambda r: f"{r.randint(11, 998)} hours"),
    ("bay assignment", lambda r: f"bay {r.randint(1, 47)}"),
    ("signed off by technician", lambda r: f"tech #{r.randint(100, 899)}"),
    ("coolant grade", lambda r: f"grade {r.choice('ABCDEF')}"),
]


def _facts():
    r = random.Random(SEED)
    rows = []
    for i, unit in enumerate(_UNITS):
        attr, gen = _ATTRS[i % len(_ATTRS)]
        rows.append((unit, attr, gen(r)))
    return rows


_FILLER = (
    "The inspection was carried out on schedule and the crew confirmed "
    "all readings within tolerance before the sheet was filed")


def _synthetic_text():
    rows = _facts()
    lines = ["SERVICE RECORD LOG. The following maintenance facts are on "
             "file; each unit code is unique. Read carefully; you will be "
             "queried on a specific entry at the end.\n"]
    for i, (unit, attr, val) in enumerate(rows):
        # each entry = the key->value fact plus deterministic narration so
        # the 40 facts spread across ~2048 tokens with queries at the tail.
        lines.append(
            f"Entry {i + 1}. Unit {unit}: {attr} is {val}. {_FILLER}.")
    # queries at the very end (retrieval-heavy tail the hybrid routes to GQA)
    q0, _, _ = rows[0]
    lines.append(
        "\nEND OF LOG. Query: recalling the entries above, the runtime "
        f"hours logged for unit {q0} was")
    return "\n".join(lines)


def _truncate_ids(tok, text, n):
    ids = tok(text, return_tensors="np").input_ids.astype(np.int64)
    if ids.shape[1] > n:
        ids = ids[:, :n]
    return ids


def build_prompts(tok):
    """Return [(name, ids_np(1,S)), ...] for prose/code/synthetic, each
    ~TARGET_TOKENS prefill tokens. `tok` is the Qwen3.5 tokenizer."""
    with open(PROSE_FILE, encoding="utf-8") as fh:
        prose = fh.read()
    with open(CODE_FILE, encoding="utf-8") as fh:
        code = fh.read()
    synth = _synthetic_text()
    out = [
        ("prose", _truncate_ids(tok, prose, TARGET_TOKENS)),
        ("code", _truncate_ids(tok, code, TARGET_TOKENS)),
        ("synthetic", _truncate_ids(tok, synth, TARGET_TOKENS)),
    ]
    return out


if __name__ == "__main__":
    import glob
    from transformers import AutoTokenizer
    snap = sorted(glob.glob(
        "/home/vader/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/"
        "snapshots/*"))[-1]
    tk = AutoTokenizer.from_pretrained(snap)
    for name, ids in build_prompts(tk):
        print(f"{name:12s} S={ids.shape[1]}")
