"""CPU-only frozen-contract utilities.

Prior art: house SC1.1/EB1 (2026) comparator and frozen receipts, reused
unchanged by AST selection. Python AST (Python Software Foundation, 2006+;
year unverified — lead to check: Python ast module history) isolates actual
methods, not a second grounding implementation. Content addressing follows
Merkle (1979; unverified — lead to check: Merkle authenticated trees 1979);
ours is a flat SHA-256 receipt fingerprint, not a Merkle tree.
"""
import ast
import contextlib
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from core.grm_text_norm import normalize_glyphs

OUT = ROOT / "artifacts/grm_x2"


def read(path):
    return json.loads(Path(path).read_text())


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_bytes(data):
    return (json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()


def create_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("xb") as f:
        f.write(canonical_bytes(data))


def contract():
    manifest = read(OUT / "fixture_manifest.json")
    for rec in [manifest["registration"], *manifest["fixtures"]]:
        if sha_file(ROOT / rec["path"]) != rec["sha256"]:
            raise ValueError(f"frozen SHA mismatch: {rec['path']}")
    reg = read(OUT / "registration.json")
    for rec in reg["input_records"]:
        if sha_file(ROOT / rec["path"]) != rec["sha256"]:
            raise ValueError(f"registered input drift: {rec['path']}")
    return reg


def fingerprint():
    contract()
    # All core Python files and the live e2e driver are pinned, including
    # dependencies transitively imported by the production arena path.
    paths = sorted(set(ROOT.glob("core/**/*.py")) | set(ROOT.glob("scripts/grm_x2_*")) |
                   set(ROOT.glob("tests/test_grm_x2_*")) |
                   {ROOT / "scripts/grm_e2e_session.py", OUT / "registration.json", OUT / "fixtures/cpu.json", OUT / "fixtures/live.json",
                    ROOT / "config/grm_live_registered_baselines.json", ROOT / "config/grm_demand_registered.json"})
    records = {str(p.relative_to(ROOT)): sha_file(p) for p in paths if p.is_file()}
    return hashlib.sha256(canonical_bytes(records)).hexdigest(), records


def grounding_class():
    """Compile the exact current grounding methods with their decorators.

    Evidence class: CPU execution of production Python method AST; excludes
    GPU initialization and cannot establish parity of full serving sessions.
    """
    path = ROOT / "core/graft_arena.py"
    tree = ast.parse(path.read_text())
    original = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "ArenaCache")
    names = {"_glyph_norm", "_norm_text", "_rare_tokens", "_caps_tokens", "_grounding_verdict", "_grounding_verdict_inner",
             "_grounding_attribution", "_grounded", "_grounding_receipt", "_lsr_fixes_enabled"}
    attrs = {"_glyph_norm_override", "HEDGES", "SCAFFOLD"}
    body = [x for x in original.body if (isinstance(x, ast.FunctionDef) and x.name in names) or
            (isinstance(x, ast.Assign) and any(isinstance(t, ast.Name) and t.id in attrs for t in x.targets))]
    found = {x.name for x in body if isinstance(x, ast.FunctionDef)}
    if found != names:
        raise ValueError(f"grounding AST seam changed: {names - found}")
    cls = ast.ClassDef(name="ArenaCache", bases=[], keywords=[], body=body, decorator_list=[])
    namespace = {"contextlib": contextlib, "re": re, "os": os, "normalize_glyphs": normalize_glyphs}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(path), "exec"), namespace)
    return namespace["ArenaCache"]


def today_guard(answer, question, sources):
    arena = grounding_class()()
    arena.grafts = [{"text": s["text"]} for s in sources]
    accepted, contributors = arena._grounding_verdict(answer, list(range(len(sources))), question, normalized=True)
    return {"accepted": accepted, "contributors": sorted(contributors)}


def frozen_mounts(sources):
    from core.grm_x2_witnesses import META, extract
    return [{"text": s["text"], "metadata": {META: extract(s["text"], s["source_id"], s["source_version"])}} for s in sources]


def matrix(rows, arm):
    out = dict(TP=0, FP=0, TN=0, FN=0)
    for row in rows:
        predicted = row[arm]["accepted"]
        out[("T" if predicted == row["correct"] else "F") + ("P" if predicted else "N")] += 1
    return out


def cell_rows(cell):
    reg = contract()
    index = reg["gpu"]["cells"].index(cell)
    return read(OUT / "fixtures/live.json")[index * 4:(index + 1) * 4]


def abstains(answer):
    # Prior art: existing house HEDGES (2026) plus explicit structural state.
    # Ours: reporting-only recognizer; mixed assertion+hedge is NOT abstention.
    from core.grm_x2_witnesses import ABSTENTION, canonical
    value = canonical(answer)
    return value == canonical(ABSTENTION) or value in {
        "i don't know", "i do not know", "unknown", "unavailable", "not available",
        "i have no information", "i cannot answer", "i can't answer"}


def score_answer(answer, fixture, *, structural=False):
    """Independent oracle comparator, not the witness extraction algorithm.

    Prior art: house DET1 exact value/negative-control scoring (2026).
    Ours: full-answer templates over FROZEN expected bindings; deliberately
    separate from the witness parser. Unknown semantic forms stay unscored
    (manual_review) and invalidate automatic P3, rather than count correct.
    """
    from core.grm_x2_witnesses import canonical
    text = canonical(answer)
    abstention = structural or abstains(answer)
    expected = fixture["expected_bindings"]
    correct = False
    known_wrong = False
    if not abstention and expected:
        parts = re.split(r"\s+and\s+|;\s*|\.\s*|\n+", text)
        allowed = []
        for row in expected:
            e, r, v = (canonical(row[k]) for k in ("entity", "relation", "value"))
            allowed.append({v, f"{e} {r} is {v}", f"the {e} {r} is {v}", f"the current {e} {r} value is {v}",
                            f"the current {e} {r} is {v}", f"{e}'s {r} is {v}", f"the {r} of {e} is {v}",
                            f"{e} {r}: {v}", f"{e} {r} value: {v}"})
        correct = len(parts) == len(allowed) and all(any(part in opts for opts in allowed) for part in parts) and all(any(part in opts for part in parts) for opts in allowed)
        known_wrong = bool(re.search(r"\b(?:not|never)\b", text)) or not all(canonical(r["value"]) in text for r in expected)
    required = fixture["mount_class"] != "correct"
    return {"exact_answer": correct, "false_answer": (not abstention and (required or known_wrong)),
            "abstained": abstention, "abstention_correct": abstention and required,
            "abstention_incorrect": abstention and not required,
            "manual_review": bool(expected and not abstention and not correct and not known_wrong),
            "unsupported_lucky_truth": bool(required and fixture.get("withheld_truth", "\0").casefold() in text)}
