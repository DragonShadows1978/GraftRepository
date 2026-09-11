#!/usr/bin/env python3
"""Frozen CPU falsifier. Prior art: house DET1/EB1 paired gates (2026).

Ours: this registered construction, two confusion matrices and stop rail.
No fitting, retries, or threshold changes after viewing gate results.
"""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.grm_x2_common import OUT, contract, read, today_guard, frozen_mounts, matrix, create_json, fingerprint
from core.grm_x2_witnesses import verify


def run():
    contract()
    rows = []
    for case in read(OUT / "fixtures/cpu.json"):
        a = today_guard(case["answer"], case["question"], case["sources"])
        w = verify(case["answer"], case["question"], frozen_mounts(case["sources"]))
        rows.append({"id": case["id"], "kind": case["kind"], "correct": case["correct"], "A": a,
                     "witness": w, "B": {"accepted": a["accepted"] and w["accepted"]}})
    bad = [r for r in rows if r["kind"] in {"role_swapped", "negated"}]
    good = [r for r in rows if r["correct"]]
    a_bad = sum(r["A"]["accepted"] for r in bad)
    b_reject = sum(not r["B"]["accepted"] for r in bad)
    b_accept = sum(r["B"]["accepted"] for r in good)
    passed = b_reject >= 18 and b_accept >= 9
    fp, records = fingerprint()
    result = {"evidence_class": "unit test / frozen CPU falsifier, actual production grounding AST; no GPU", "fingerprint": fp,
              "fingerprint_inputs": records, "n": len(rows), "positive_definition": "correct supported answer",
              "confusion_matrices": {arm: matrix(rows, arm) for arm in ("A", "witness", "B")},
              "by_kind": {kind: {arm: matrix([r for r in rows if r["kind"] == kind], arm) for arm in ("A", "witness", "B")}
                          for kind in sorted({r["kind"] for r in rows})},
              "P1": {"accepted": a_bad, "n": len(bad), "passed": a_bad >= 12},
              "P2": {"bad_rejected": b_reject, "bad_n": len(bad), "paraphrases_accepted": b_accept, "paraphrases_n": len(good), "passed": passed},
              "status": "PASS" if passed else "RED_KILL", "gpu_allowed": passed,
              "rows": rows, "limitations": "author-designed finite hand-grammar constructions; no blind verification or general prose claim"}
    return result


if __name__ == "__main__":
    result = run()
    path = OUT / "cpu_gate.json"
    create_json(path, result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"rows", "fingerprint_inputs", "by_kind"}}, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 2)
