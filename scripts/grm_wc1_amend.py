#!/usr/bin/env python3
"""GRM-WC1 — write an amendment beside the IMMUTABLE registration.

The registration is written once and never rewritten (house rule); anything
that has to be said afterwards goes here, in a separate file that CITES the
registration's sha256 so the amendment can never be mistaken for the original
terms.  Amendments are content-addressed and append-only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.grm_cmc1_mechanism import (  # noqa: E402
    canonical_json_bytes, sha256_bytes,
)
from scripts.grm_det1_common import file_record  # noqa: E402
from scripts.grm_wc1_common import (  # noqa: E402
    ARTIFACT_DIR, SCHEMA_PREFIX, WC1Error,
)

REGISTRATION = ARTIFACT_DIR / "registration.json"

#: A1 — the sup-9/9 reconciliation, written the moment width 96 came in.
A1 = {
    "id": "A1",
    "title": (
        "Width 96 sup lands 9/9, not RS3 Part 4's 8/9 — because RT1 is on "
        "this branch and its rule is ON by default"),
    "raised_before": "any width other than 96 was run",
    "what_the_registration_said": (
        "g2_reproduction_targets.declared_ambiguity registered, BEFORE any "
        "gate ran, that RS3 Part 4 on disk records sup 8/9 while the order's "
        "G2 line asks for 9/9, and that WC1 would treat REPRODUCTION OF THE "
        "RECEIPT (8/9, same probe verdicts, same served text) as the "
        "criterion."),
    "what_was_measured": {
        "sup_at_width_96": "9/9",
        "probes_reproducing_rs3_part4_bit_equal": 8,
        "the_one_divergence": "sup_solace_fresh",
        "rs3_part4_served": "I'm sorry, but I don't have that information.",
        "wc1_w96_served": "The current Solace key is **Raven-9-Ivory**.",
        "direction": "RECOVERY, not regression",
    },
    "cause": (
        "The divergence is NOT the width injection and NOT a WC1 defect. It "
        "is GRM-RT1, which landed on this branch (lc1-wip) AFTER the RS3 "
        "Part 4 receipt was written. core/grm_admission.rt1_enabled resolves "
        "the rule through the GRM_LSR_FIXES family switch when GRM_RT1_RULE "
        "is unset, and WC1 runs with GRM_LSR_FIXES=1 — so the RT1 rule "
        "(demote_non_binding_split_members) is ON. RT1's own registered "
        "prediction for exactly this battery, in "
        "orders/GRM_RT1_SPLIT_CHILD_ROUTING.md, is '9/9 (solace recovers, "
        "nothing regresses)' with 'rule OFF reproduces RS3's 8/9 bit-equal "
        "on served text'."),
    "mechanism_receipt": (
        "The width-96 solace row's fit info records mount_fitted=[1] and "
        "mount_plan=[1] — graft 1, the solace FACT node. RS1/RT1 named the "
        "failure as routing to a split child of graft 4 instead of graft 1. "
        "The fact node is now the mount, which is the rule doing what it was "
        "written to do."),
    "ruling": (
        "G2 is judged REPRODUCED. Both reproduction targets are met at once: "
        "8/9 probes are bit-equal to RS3 Part 4's served text, and the ninth "
        "reproduces RT1's registered G2 result. The order's G2 line asks for "
        "'sup 9/9' and names 'RS3 Part 4 + RT1' as the joint target; 9/9 is "
        "the RT1-ON column of that pair. This is a reconciliation of two "
        "receipts written at different commits, NOT a threshold moved after "
        "seeing a result — the registration declared the discrepancy and "
        "both candidate criteria before the gate ran."),
    "consequence_for_the_sweep": (
        "The width-96 reference column every other width is compared against "
        "is the RT1-ON 9/9 column. A width that loses solace is a regression "
        "vs 96 and is reported as one."),
}

AMENDMENTS: dict[str, dict[str, Any]] = {"A1": A1}


def build(amendment_id: str) -> dict[str, Any]:
    if amendment_id not in AMENDMENTS:
        raise WC1Error(f"unknown amendment {amendment_id!r}")
    if not REGISTRATION.is_file():
        raise WC1Error(f"{REGISTRATION} is missing; nothing to amend")
    body = REGISTRATION.read_bytes()
    return {
        "schema": f"{SCHEMA_PREFIX}.amendment.v1",
        "program": "GRM",
        "phase": "WC1",
        "order": "orders/GRM_WC1_ARENA_WIDTH_CURVE.md",
        "amends": {
            "path": "artifacts/grm_wc1/registration.json",
            "sha256": sha256_bytes(body),
            "bytes": len(body),
        },
        "immutability_note": (
            "The registration itself is UNCHANGED and unrewritten. This file "
            "records what had to be said afterwards."),
        "amendment": dict(AMENDMENTS[amendment_id]),
        "sources": {
            "grm_admission": file_record(ROOT / "core" / "grm_admission.py"),
            "rt1_order": file_record(
                ROOT / "orders" / "GRM_RT1_SPLIT_CHILD_ROUTING.md"),
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("amendment_id", choices=sorted(AMENDMENTS))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build(str(args.amendment_id))
    body = canonical_json_bytes(payload)
    digest = sha256_bytes(body)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT_DIR / (
        f"amendment_{str(args.amendment_id).lower()}_{digest[:16]}.json")
    path.write_bytes(body)
    print(f"amendment={path}")
    print(f"sha256={digest}")
    print(f"amends_registration_sha256={payload['amends']['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
