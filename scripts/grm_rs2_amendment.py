#!/usr/bin/env python3
"""GRM-RS2 — the mid-run amendment that adds arm B1p.

ORDER: ``orders/GRM_RS2_MOUNT_READ_DEFICIT.md``.
AMENDS: ``artifacts/grm_rs2/registration.json``, field
``arms_REGISTERED_BEFORE_ANY_GATE.B1p`` (an ADDITION; no registered arm, probe
id, capture text, prediction, partition definition or verdict rule is changed).

WHY AN AMENDMENT AND NOT A QUIET EDIT.  The registration was written before any
gate ran and its bytes were hashed.  B1p did not exist then, because the fact it
tests had not been measured then.  Adding an arm after a gate has run is exactly
the kind of change that has to leave a trail, so this module writes the trail:
the pre-amendment registration's sha256, the receipts the amendment rests on,
the measured cause, and the new arm's prediction — REGISTERED BEFORE THE ARM RAN
and marked as the SEAT's, not the lead's, so it is scored separately.

THE CAUSE, measured and isolated (three diagnostics, on the card):

  1. B1 on ``correction_then_restatement`` re-captured the installed node's own
     text through ``arena.deposit`` and got a payload with a DIFFERENT sha256 —
     layer 0 identical, layers 1..23 all changed — and the harbor probe FLIPPED
     from refusal (mounted mass 0.184) to correct (0.303).  The registration had
     predicted a bit-identical re-capture, so this was a registered prediction
     MISSING, which is a result and is reported as one.

  2. A determinism diagnostic showed ``deposit(text)`` IS deterministic and that
     re-depositing the installed node's text on a freshly installed repository
     reproduces the installed payload BIT-FOR-BIT.  So the divergence was not in
     ``deposit`` and not in the storage quantization round trip (which produces
     a third, different digest — checked, and it matches neither).

  3. A state diagnostic narrowed it to ONE attribute.  ``GptOssAttentionTC
     .__call__`` rotates its QUERIES at ``cos.slice(0, position_offset + shift,
     L)`` where ``shift`` is ``self_attn.live_shift``, falling back to
     ``graft_seats`` when that is ``None``.  ``grm_det1_3_gpu
     ._install_lived_nodes`` never sets ``live_shift``, so every installed graft
     was harvested with its queries at positions [0, L).  Any served turn leaves
     ``live_shift = n_sink + arena_width`` on every layer, so an identical
     ``deposit`` call afterwards harvests with its queries at
     [live_shift, live_shift + L) and yields a different graft.  Pinning the
     attribute reproduced both digests on demand.

WHAT THAT MAKES B1p.  The mismatch it names is not a harness artifact — it is a
property of ``ArenaCache.deposit``, which does not pin the capture-time query
position at all.  A graft is therefore captured with its queries at position 0
and later READ by a query sitting past ``live_shift``.  B1p captures at the READ
geometry instead, making the position an explicit, named, restored-after arm
variable rather than a side effect of whether a turn happened to have been
served.  B1 is kept, unchanged, pinned to the LIVED geometry, so the pair is a
clean two-level contrast on one variable.

NO PRODUCTION CHANGE.  ``core/`` and ``config/`` remain read-only; the pin lives
in ``scripts/grm_rs2_mount_read_gpu.py`` and is restored in a ``finally``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.grm_cmc1_mechanism import canonical_json_bytes  # noqa: E402
from scripts.grm_det1_common import file_record  # noqa: E402

ARTIFACT_DIR = ROOT / "artifacts" / "grm_rs2"

#: The registration bytes as they stood BEFORE this amendment, hashed before the
#: amendment was applied.  Quoted here so the trail survives the rewrite.
PRE_AMENDMENT_REGISTRATION_SHA256 = (
    "05638479bb2eedea2661483c9d4a8dac02d577b26b0d01fb4694e51798920c0a")
PRE_AMENDMENT_REGISTRATION_BYTES = 33834

#: The three digests the diagnostics produced, quoted so the amendment's own
#: claims can be checked against the numbers rather than against its prose.
INSTALLED_PAYLOAD_SHA256 = (
    "e8192df3d8c88898e8f628e247fb05e9694ae19634405a166bf3b113761a0a2f")
READ_GEOMETRY_PAYLOAD_SHA256 = (
    "ce417319b514d0635347b8d4450ffc8b775f6904f8a0dc42432e6be0900a0130")
STORAGE_ROUNDTRIP_PAYLOAD_SHA256 = (
    "f433f595252bf9da7b0f2a19918d2ab51fa1b351c1316c3592c1f7bfe43e7865")


def _latest(pattern: str) -> Path:
    matches = sorted(ARTIFACT_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no RS2 receipt matches {pattern}")
    return matches[-1]


def _probe_row(payload: dict[str, Any], probe_id: str) -> dict[str, Any]:
    return next(
        row for row in payload["probes"]
        if row["probe_id"] == probe_id and row["registered_probe"])


def _mounted_full(row: dict[str, Any]) -> float:
    return float(
        row["mass"]["by_layer_type"]["full_attention"][
            "mean_over_answer_positions"]["mounted_mass"])


def build() -> dict[str, Any]:
    b0_receipt = _latest("grm_rs2_B0_correction_then_restatement_*.json")
    b1_receipt = _latest("grm_rs2_B1_correction_then_restatement_*.json")
    b0 = json.loads(b0_receipt.read_text(encoding="utf-8"))
    b1 = json.loads(b1_receipt.read_text(encoding="utf-8"))
    b0_row = _probe_row(b0, "sup_harbor_restatement")
    b1_row = _probe_row(b1, "sup_harbor_restatement")
    node = b1_row["info"]["rs2_capture"]["per_node"][0]

    return {
        "schema": "grm.rs2.amendment.v1",
        "program": "GRM",
        "phase": "RS2",
        "order": "orders/GRM_RS2_MOUNT_READ_DEFICIT.md",
        "amends": {
            "path": "artifacts/grm_rs2/registration.json",
            "sha256_before_amendment": PRE_AMENDMENT_REGISTRATION_SHA256,
            "bytes_before_amendment": PRE_AMENDMENT_REGISTRATION_BYTES,
        },
        "amends_field": ["arms_REGISTERED_BEFORE_ANY_GATE.B1p"],
        "amendment_kind": (
            "ADDITION ONLY. No registered arm, probe id, capture text, "
            "prediction, sliding-partition definition or verdict rule is "
            "changed or removed by this amendment."),
        "arms_after_amendment": [
            "B0", "B1", "B1p", "B1b", "B2", "B3a", "B3b", "B5"],
        "cause": (
            "MEASURED ON THE CARD, mid-run: ArenaCache.deposit does not pin the "
            "capture-time QUERY POSITION. GptOssAttentionTC.__call__ rotates "
            "queries at cos.slice(0, position_offset + shift, L) where shift is "
            "self_attn.live_shift (falling back to graft_seats when None). The "
            "lived installer never sets live_shift, so every installed graft "
            "was harvested with queries at positions [0, L); any served turn "
            "leaves live_shift = n_sink + arena_width, so the SAME deposit call "
            "afterwards harvests at [live_shift, live_shift + L) and yields a "
            "different graft."),
        "evidence": {
            "B1_flipped_the_harbor_refuser": {
                "receipt": file_record(b1_receipt),
                "baseline_receipt": file_record(b0_receipt),
                "probe_id": "sup_harbor_restatement",
                "B0_correct": bool(b0_row["correct"]),
                "B0_served": str(b0_row["served_answer"]),
                "B0_mounted_mass_full_layers": _mounted_full(b0_row),
                "B1_correct": bool(b1_row["correct"]),
                "B1_served": str(b1_row["served_answer"]),
                "B1_mounted_mass_full_layers": _mounted_full(b1_row),
                "installed_payload_sha256": str(
                    node["installed_payload"]["sha256"]),
                "recaptured_payload_sha256": str(
                    node["recaptured_payload"]["sha256"]),
                "payload_identical": bool(node["payload_identical"]),
                "ntok_identical": bool(node["ntok_identical"]),
                "registered_prediction_that_missed": (
                    "the registration predicted a PAYLOAD-IDENTICAL "
                    "re-capture, because the installed graft was itself "
                    "produced by deposit(text) through feed's ephemeral "
                    "branch. That prediction MISSED, and the miss is what "
                    "opened this amendment."),
            },
            "determinism_diagnostic": {
                "what": (
                    "deposit(text) run twice on a virgin arena, and once more "
                    "after the lived install"),
                "result": (
                    f"all three produced the SAME payload sha256 "
                    f"{INSTALLED_PAYLOAD_SHA256}, which is also the INSTALLED "
                    "harbor_b payload. deposit is deterministic and the "
                    "installed graft is a bit-for-bit deposit of its own text."),
                "rules_out": (
                    "non-determinism in deposit, and the P3 storage "
                    "quantization round trip — pack_node/unpack_node on either "
                    f"payload gives a THIRD digest "
                    f"{STORAGE_ROUNDTRIP_PAYLOAD_SHA256} that matches neither "
                    "observed graft."),
            },
            "live_shift_diagnostic": {
                "what": (
                    "the identical deposit(text) call repeated with "
                    "self_attn.live_shift pinned to each of three values"),
                "result": {
                    "live_shift_None": (
                        f"{INSTALLED_PAYLOAD_SHA256} — the installed payload, "
                        "bit-for-bit"),
                    "live_shift_0": (
                        f"{INSTALLED_PAYLOAD_SHA256} — the installed payload, "
                        "bit-for-bit"),
                    "live_shift_115": (
                        f"{READ_GEOMETRY_PAYLOAD_SHA256} — the B1 payload, "
                        "bit-for-bit"),
                },
                "layer_signature": (
                    "layer 0 identical, layers 1..23 all changed — exactly what "
                    "a QUERY-POSITION change predicts as it propagates through "
                    "the attention stack, and NOT what a payload quantization "
                    "would produce (which would move layer 0 too)"),
                "live_shift_observed_after_install": None,
                "live_shift_after_any_served_turn": 115,
                "n_sink": 19,
                "arena_width": 96,
                "read_from": [
                    "core/gpt_oss20b_tc.py::GptOssAttentionTC.__call__",
                    "core/graft_arena.py::ArenaCache.deposit",
                    "core/graft_arena.py::ArenaCache._harvest",
                    "scripts/grm_det1_3_gpu.py::_install_lived_nodes",
                ],
            },
        },
        "new_arm": {
            "B1p": {
                "label": (
                    "H-CAPTURE, capture POSITION varied: deposit(text) at the "
                    "READ geometry instead of the lived installer's"),
                "kept_because": (
                    "the position mismatch is a property of production's own "
                    "deposit, not of this harness, and it sits squarely on the "
                    "order's H-CAPTURE axis — 'the graft's K/V carry the "
                    "deposit-time context'. Position IS deposit-time context."),
                "B1_unchanged": (
                    "B1 is kept exactly as registered and is pinned to the "
                    "LIVED capture geometry, so B1/B1p is a clean two-level "
                    "contrast on ONE variable."),
                "prediction_is_the_seats_not_the_leads": True,
            },
        },
        "no_production_change": (
            "core/ and config/ remain READ-ONLY. The capture-position pin lives "
            "in scripts/grm_rs2_mount_read_gpu.py, is applied only around the "
            "harvest forward, and is restored in a finally block."),
        "sources": {
            record["path"]: {
                "bytes": record["bytes"], "sha256": record["sha256"]}
            for record in (
                file_record(b0_receipt),
                file_record(b1_receipt),
                file_record(ROOT / "core" / "graft_arena.py"),
                file_record(ROOT / "core" / "gpt_oss20b_tc.py"),
                file_record(ROOT / "scripts" / "grm_det1_3_gpu.py"),
            )
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    payload = build()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    target = ARTIFACT_DIR / "amendment_b1p_capture_position.json"
    target.write_bytes(canonical_json_bytes(payload))
    print(f"amendment={target}")
    print(json.dumps({
        "amends_field": payload["amends_field"],
        "arms_after_amendment": payload["arms_after_amendment"],
        "sha256_before_amendment": payload["amends"]["sha256_before_amendment"],
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
