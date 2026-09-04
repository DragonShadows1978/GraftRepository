#!/usr/bin/env python3
"""GRM-RS1 — the A2 amendment, written AFTER the fact that forced it.

ORDER: ``orders/GRM_RS1_READ_STRENGTH_PROBE.md``.

WHY AN AMENDMENT AND NOT AN EDIT.  ``artifacts/grm_rs1/registration.json`` was
written before any gate ran; that is the whole point of a registration and it
is not touched here.  What follows is a change of METHOD forced by a fact the
registration could not have known — one measured on the GPU, on the first A2
run, and recorded with its receipt — so it is written as a separate amendment
that CITES the registration rather than replacing it.

THE FACT.  The order defines A2 as "the same node(s) as A1 but mounted as
grafts ALONGSIDE A0's mount".  At the lived ``arena_width`` of 96 the battery's
fixture nodes are 52-66 tokens each and its competitor nodes 150-159, so:

  * NO TWO fixture nodes co-seat (52 + 56 = 108 > 96 is the cheapest pair);
  * a competitor node does not seat AT ALL (150 > 96).

The literal union is therefore unseatable, and production's own packer
(``grm_e2e_session._budget_fit_mounts``, rank order) resolves it by keeping
A0's mount and dropping the restatement.  Measured, with a receipt: the harbor
A2 run requested {1, 2, 3}, seated [1], and dropped [2, 3] — a byte-copy of A0,
which answers nothing.

An EARLIER run of that same arm, before the width law was applied, is also
recorded below: ``arena._attempt``'s bootstrap branch has no width check
(nothing reaches it unfitted in production, because the ladder has already
fitted), so it injected all three nodes, put 266 seats into a 96-seat arena,
and served an EMPTY string.  That receipt is a harness artifact, is named as
one here, and no verdict rests on it.

THE AMENDMENT.  A2 splits into two sub-arms, both run and both reported:

  A2a  THE LITERAL ARM.  A0's mount UNION the registered live ids, packed by
       production's own fit.  Kept because "the union does not fit" is itself
       a result, and because dropping the literal arm would leave only an arm
       the order did not write.
  A2b  THE ARM THE ORDER'S SENTENCE ASKS FOR.  The registered live-id node(s)
       mounted ALONE, in place of A0's mount.  This is A1's payload moved from
       the LIVE band to the MOUNT band with nothing else changed — which is
       exactly the CONTENT-vs-POSITION separation A2 exists to draw.

THE VERDICT RULE IS UNCHANGED.  "A2 rescues" is read as EITHER sub-arm
rescuing: a mount of that node text that serves correctly is content doing the
work, however the mount set was assembled.  ``mechanism_verdict`` routes A2
through ``a2_rescued`` so the split cannot move a verdict on its own, and
``tests/test_grm_rs1_read_strength.py`` pins that.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.grm_cmc1_mechanism import canonical_json_bytes  # noqa: E402
from scripts.grm_det1_common import file_record  # noqa: E402
from scripts import grm_rs1_read_strength_gpu as rs1  # noqa: E402

ARTIFACT_DIR = ROOT / "artifacts" / "grm_rs1"
REGISTRATION = ARTIFACT_DIR / "registration.json"

#: The pre-width-law harness artifact, kept so the amendment can point at it.
OVERFLOW_RECEIPT = (
    ARTIFACT_DIR
    / "grm_rs1_A2_correction_then_restatement_8da1a443b0af8857.json")
#: The width-law run that showed the literal union collapsing onto A0.
FITTED_RECEIPT = (
    ARTIFACT_DIR
    / "grm_rs1_A2_correction_then_restatement_bbf2e324971214bd.json")

#: The two A4 runs that ABSTAINED before reaching the stage A4 is about.
#: Both are kept on disk as the evidence for the A4 amendment below.
A4_ABSTAINED_RECEIPTS = (
    (ARTIFACT_DIR
     / "grm_rs1_A4_correction_then_restatement_563616b020bd9a74.json",
     "the sweep's literal wording, turn label included: the numeral 50 "
     "became the query's only identifier token"),
    (ARTIFACT_DIR
     / "grm_rs1_A4_correction_then_restatement_4264608fff8f6818.json",
     "the turn label dropped: the instruction's own content words "
     "(return/exactly/stored/sentence/punctuation) became the identifier "
     "tokens instead"),
)


def _fit_records(path: Path) -> dict[str, dict[str, Any]]:
    """Per-probe fit records, read off a receipt rather than retyped."""
    out: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return out
    payload = json.loads(path.read_text(encoding="utf-8"))
    for probe in payload["probes"]:
        info = probe.get("info") or {}
        if "rs1_fit_requested" not in info:
            continue
        out[str(probe["probe_id"])] = {
            "requested": info["rs1_fit_requested"],
            "seated": info["rs1_fit_seated"],
            "dropped": info["rs1_fit_dropped"],
            "arena_width": info["rs1_arena_width"],
            "cur_mount_n_at_serve": info["rs1_cur_mount_n_at_serve"],
        }
    return out


def build() -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    if OVERFLOW_RECEIPT.is_file():
        payload = json.loads(OVERFLOW_RECEIPT.read_text(encoding="utf-8"))
        row = next(
            (p for p in payload["probes"]
             if p["probe_id"] == "sup_harbor_restatement"), None)
        evidence["pre_width_law_overflow"] = {
            "receipt": file_record(OVERFLOW_RECEIPT),
            "picks_requested": (row or {}).get("picks_requested"),
            "mounted_ids": (row or {}).get("mounted_ids"),
            "served_answer": (row or {}).get("served_answer"),
            "resident_seats": ((row or {}).get("info") or {}).get("resident"),
            "classification": (
                "HARNESS ARTIFACT, not read-strength evidence. "
                "ArenaCache._attempt's bootstrap branch has no width check "
                "because production never reaches it unfitted; this arm did, "
                "and the arena overflowed. No verdict rests on this row."),
        }
    if FITTED_RECEIPT.is_file():
        evidence["width_law_applied"] = {
            "receipt": file_record(FITTED_RECEIPT),
            "fit": _fit_records(FITTED_RECEIPT),
            "reading": (
                "with production's own packer applied, the literal A2 union "
                "collapses onto A0's mount: the restatement is DROPPED and "
                "the arm becomes a copy of the baseline"),
        }
    a4_rows: list[dict[str, Any]] = []
    for path, why in A4_ABSTAINED_RECEIPTS:
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = next(
            (p for p in payload["probes"]
             if p["probe_id"] == "sup_harbor_restatement"), None)
        info = (row or {}).get("info") or {}
        a4_rows.append({
            "receipt": file_record(path),
            "wording": why,
            "question_asked": (row or {}).get("question_asked"),
            "abstained": info.get("abstained"),
            "abstain_reason": info.get("abstain_reason"),
            "abstain_identifier_tokens": info.get(
                "abstain_identifier_tokens"),
            "served_answer": (row or {}).get("served_answer"),
            "mounted_ids": (row or {}).get("mounted_ids"),
            "mass_answer_positions": (
                ((row or {}).get("mass") or {}).get("answer_positions")),
        })

    return {
        "schema": "grm.rs1.amendment.v1",
        "program": "GRM",
        "phase": "RS1",
        "order": "orders/GRM_RS1_READ_STRENGTH_PROBE.md",
        "amends": file_record(REGISTRATION),
        "amends_field": [
            "arms_REGISTERED_BEFORE_ANY_GATE.A2",
            "arms_REGISTERED_BEFORE_ANY_GATE.A4",
        ],
        "A4_amendment": {
            "cause": (
                "MEASURED ON THE CARD, twice: the registered strict wording "
                "put through the WHOLE probe path abstains before any mount "
                "work. _probe_identifier_tokens reads the added instruction "
                "words as IDENTIFIERS, nothing in the repository binds them, "
                "and P2A's identifier_unbound rule fires. Both runs served "
                "'Not in memory: ...' with an EMPTY mount and ZERO captured "
                "mass rows — which measures the ABSTENTION RULE, not the "
                "instruct prior."),
            "why_it_matters": (
                "the order's A4 asks whether 'the model READ the graft', "
                "which requires the graft to be MOUNTED; an arm that "
                "abstains before mounting cannot answer that question"),
            "amended_method": (
                "A4 changes ONLY the stage it is about: routing, admission "
                "and fit run on the ORIGINAL question, giving A0's mount set; "
                "GENERATION then runs on the strict wording over exactly that "
                "mount. The comparison against A0 is one variable — the words "
                "the model was asked to answer with."),
            "evidence": a4_rows,
        },
        "registration_unmodified": (
            "The registration file is NOT edited. It was written before any "
            "gate ran and stays as written; this amendment cites it."),
        "cause": (
            "MEASURED ON THE CARD, mid-run: at the lived arena_width the "
            "battery's nodes cannot co-seat, so the order's literal A2 mount "
            "set is unseatable and production's fit reduces it to A0's "
            "mount."),
        "amendment": {
            "A2a": {
                "label": "the LITERAL arm: A0's mount UNION the registered "
                         "live ids, packed by "
                         "grm_e2e_session._budget_fit_mounts",
                "kept_because": "'the union does not fit' is itself a result, "
                                "and the receipt names what the fit dropped",
            },
            "A2b": {
                "label": "the registered live-id node(s) mounted ALONE, in "
                         "place of A0's mount",
                "kept_because": "this is A1's payload moved from the LIVE "
                                "band to the MOUNT band with nothing else "
                                "changed — the CONTENT-vs-POSITION contrast "
                                "the order's A2 sentence describes",
            },
        },
        "verdict_rule_unchanged": (
            "'A2 rescues' means EITHER sub-arm serving correctly. "
            "mechanism_verdict routes A2 through a2_rescued, and "
            "tests/test_grm_rs1_read_strength.py pins that the split cannot "
            "move a verdict on its own."),
        "arms_after_amendment": list(rs1.ARMS),
        "evidence": evidence,
        "sources": {
            "grm_rs1_read_strength_gpu": file_record(
                ROOT / "scripts" / "grm_rs1_read_strength_gpu.py"),
            "grm_rs1_amendment": file_record(Path(__file__).resolve()),
            "budget_fit_mounts_site": file_record(
                ROOT / "scripts" / "grm_e2e_session.py"),
        },
    }


def main() -> int:
    payload = build()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT_DIR / "amendment_a2_width.json"
    path.write_bytes(canonical_json_bytes(payload))
    print(f"amendment={path}")
    print(json.dumps(payload["amendment"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
