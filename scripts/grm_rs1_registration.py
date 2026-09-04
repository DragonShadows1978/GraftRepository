#!/usr/bin/env python3
"""GRM-RS1 — write the registration BEFORE any gate runs.

ORDER: ``orders/GRM_RS1_READ_STRENGTH_PROBE.md``.

The order's Gates section says: "Register before any gate run
(``artifacts/grm_rs1/registration.json``): arms, probe ids, the exact
live/mount ids per probe taken from the EB1 receipts, predictions above.
No numeric constants."

EVERY ID IN THE REGISTRATION IS READ OFF A RECEIPT, never typed.  The two
frames' receipts are both consulted and both named:

  * the EPHEMERAL-frame (EB1 G2) arm-1 receipts, cited by
    ``artifacts/grm_eb1/grm_eb1_summary.json`` — these give ``mounted_ids``
    per probe under the spec frame (what A0 must reproduce) and
    ``live_graft_ids_after_install`` == [] (the spec frame's empty boat);
  * the PERSISTENT-frame arm-1 receipts, cited by
    ``artifacts/lsr_p2c/lsr_p2c_g2_summary_d5efcb7969279860.json`` — these
    give the ``live_graft_ids_after_install`` the lived run actually carried,
    which is the id set arms A1/A2 feed/mount.

"No numeric constants" is honoured literally: this module hard-codes no probe
id, no graft index, and no threshold.  It hard-codes only RECEIPT PATHS and
the registered VOCABULARY strings, and derives everything else.
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

ARTIFACT_DIR = ROOT / "artifacts" / "grm_rs1"
EB1_SUMMARY = ROOT / "artifacts" / "grm_eb1" / "grm_eb1_summary.json"
P2C_G2_SUMMARY = (
    ROOT / "artifacts" / "lsr_p2c"
    / "lsr_p2c_g2_summary_d5efcb7969279860.json")

#: The order's Probes section, by ROLE.  Named, not numbered: the order names
#: three refusers and two controls, and which graft ids those correspond to is
#: read from the receipts below.
REFUSERS = (
    "sup_harbor_restatement", "sup_praxis_fresh", "sup_solace_fresh")
CONTROLS = ("sup_reserve_meridian_docket", "sup_reserve_tundra_ledger")

#: The order's G3 vocabulary.  A mechanism verdict must be one of these.
VOCABULARY = {
    "CONTENT": "A2 rescues (the restatement TEXT is what helps)",
    "POSITION": "A1 rescues but A2 does not (the LIVE BAND is what helps)",
    "QUANTIZATION": "A3 rescues (the INT8 storage payload was the cause)",
    "PRIOR": "A4 rescues; the graft was read and the refusal is the "
             "instruct prior, not a retrieval failure",
    "UNRESOLVED": "no arm rescues",
}


class RS1Error(RuntimeError):
    """The registration could not be built from the receipts as ordered."""


def _read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def eb1_rows() -> dict[str, dict[str, Any]]:
    """EB1 G2 rows keyed by probe id — the SPEC-FRAME baseline A0 reproduces."""
    payload = _read(EB1_SUMMARY)
    rows = payload["G2_sup_battery_spec_frame"]["rows"]
    return {str(row["probe_id"]): dict(row) for row in rows}


def eb1_receipt_paths() -> dict[str, Path]:
    """The four EB1-cited spec-frame arm-1 receipts, by session id."""
    payload = _read(EB1_SUMMARY)
    receipts = payload["G2_sup_battery_spec_frame"]["receipts"]
    return {
        str(session): ROOT / str(record["path"])
        for session, record in receipts.items()
    }


def persistent_receipt_paths() -> dict[str, Path]:
    """The four P2C-G2-cited PERSISTENT-frame arm-1 receipts, by session id."""
    payload = _read(P2C_G2_SUMMARY)
    receipts = payload["arm1"]["receipts"]
    return {
        str(session): ROOT / str(record["path"])
        for session, record in receipts.items()
    }


def _assert_frame(receipt: dict[str, Any], *, ephemeral: bool,
                  path: Path) -> None:
    """The frame is READ off the receipt, never assumed from the filename."""
    frame = receipt.get("frame")
    observed = (
        None if not isinstance(frame, dict) else bool(frame.get("ephemeral")))
    if ephemeral and observed is not True:
        raise RS1Error(
            f"{path.name}: expected an EPHEMERAL-frame receipt, "
            f"frame.ephemeral={observed!r}")
    if not ephemeral and observed is True:
        raise RS1Error(
            f"{path.name}: expected a PERSISTENT-frame receipt, "
            f"frame.ephemeral={observed!r}")


def probe_registration() -> dict[str, dict[str, Any]]:
    """Per-probe ids, taken from the receipts of BOTH frames.

    For each probe the registration records:

    ``a0_mounted_ids``   the ids EB1's spec-frame run mounted (the A0 target)
    ``live_ids``         ``live_graft_ids_after_install`` from the
                         PERSISTENT-frame receipt for that probe's session —
                         the live window the lived run carried, which is the
                         id set A1 feeds live and A2 mounts
    ``live_node_ids``    those same ids resolved back to fixture node names
                         through the receipt's ``fixture_node_to_idx``
    """
    eb1 = eb1_rows()
    persistent = persistent_receipt_paths()
    ephemeral = eb1_receipt_paths()
    wanted = [*REFUSERS, *CONTROLS]
    missing = [probe for probe in wanted if probe not in eb1]
    if missing:
        raise RS1Error(f"probes absent from the EB1 G2 rows: {missing}")

    persistent_cache: dict[str, dict[str, Any]] = {}
    for session, path in persistent.items():
        receipt = _read(path)
        _assert_frame(receipt, ephemeral=False, path=path)
        persistent_cache[session] = receipt
    for session, path in ephemeral.items():
        receipt = _read(path)
        _assert_frame(receipt, ephemeral=True, path=path)
        if receipt.get("live_graft_ids_after_install"):
            raise RS1Error(
                f"{path.name}: the spec frame's live window after install is "
                "expected EMPTY; receipt says "
                f"{receipt['live_graft_ids_after_install']!r}")

    out: dict[str, dict[str, Any]] = {}
    for probe_id in wanted:
        row = eb1[probe_id]
        session = str(row["session_id"])
        receipt = persistent_cache.get(session)
        if receipt is None:
            raise RS1Error(
                f"{probe_id}: no persistent-frame receipt for {session}")
        live_ids = [int(v) for v in receipt["live_graft_ids_after_install"]]
        node_to_idx = {
            str(k): int(v) for k, v in receipt["fixture_node_to_idx"].items()}
        idx_to_node = {v: k for k, v in node_to_idx.items()}
        out[probe_id] = {
            "probe_id": probe_id,
            "session_id": session,
            "role": "refuser" if probe_id in REFUSERS else "control",
            "a0_mounted_ids": [int(v) for v in row["mounted_ids"]],
            "a0_served_answer": str(row["served_answer"]),
            "a0_correct": bool(row["correct"]),
            "persistent_frame_answer": str(row["p2c_arm1_answer"]),
            "persistent_frame_correct": bool(row["p2c_arm1_correct"]),
            "live_ids": live_ids,
            "live_node_ids": [idx_to_node[i] for i in live_ids],
            "fixture_node_to_idx": node_to_idx,
            "eb1_receipt": file_record(ephemeral[session]),
            "persistent_receipt": file_record(persistent[session]),
        }
    return out


def build() -> dict[str, Any]:
    probes = probe_registration()
    return {
        "schema": "grm.rs1.registration.v1",
        "program": "GRM",
        "phase": "RS1",
        "order": "orders/GRM_RS1_READ_STRENGTH_PROBE.md",
        "question": (
            "Under the EB1 spec (ephemeral-boat) frame three sup probes "
            "REFUSE with the correct node mounted. The live window, not the "
            "mounted graft, had been carrying those reads. This registration "
            "fixes WHAT is measured before anything is measured: what the "
            "live window contributed that the graft alone does not."),
        "measurement_only": (
            "No production change. core/ and config/ are READ-ONLY for this "
            "order; every seam an arm needs lives in scripts/grm_rs1_*.py."),
        "arms_REGISTERED_BEFORE_ANY_GATE": {
            "A0": {
                "label": "spec frame, graft only (reproduce EB1 G2)",
                "build": "fresh deposit-order build under the spec frame, "
                         "GRM_LSR_FIXES on, demand OFF, then ONE probe serve",
                "prediction": "reproduces the EB1 G2 rows: the three refusers "
                              "refuse, the two controls answer correctly",
                "prediction_source": "the EB1 G2 receipt itself (this is the "
                                     "baseline, not a lead prediction)",
                "gate": "G2 — a probe that does not reproduce is EXCLUDED "
                        "from A1-A5 and reported",
            },
            "A1": {
                "label": "the crutch isolated: same node TEXT fed live, "
                         "no graft mounted",
                "build": "A0's repository; the probe's registered live_ids "
                         "are pushed through the LIVE cache as turn text and "
                         "the serve runs with an EMPTY mount set; routing "
                         "disabled for this arm",
                "prediction": "correct on all three refusers",
                "prediction_source": "lead, stated in the order",
            },
            "A2": {
                "label": "restatement as a MOUNT, not live",
                "build": "A0's mount set UNION the probe's registered "
                         "live_ids, mounted as grafts; no live window",
                "separates": "CONTENT (the restatement text helps) from "
                             "POSITION (live band vs arena band)",
                "prediction": "correct on harbor (restatement is content); "
                              "UNKNOWN on praxis/solace — report",
                "prediction_source": "lead, stated in the order",
            },
            "A3": {
                "label": "unquantized graft",
                "build": "A0 with the arena's storage_bits unset (the fp16 "
                         "node payload) so the mounted K/V is the "
                         "capture-time payload",
                "prediction": "no change (quantization is not the cause; "
                              "INT8 was free on the composed E2E)",
                "prediction_source": "lead, stated in the order",
            },
            "A4": {
                "label": "instruct-prior control",
                "build": "A0 with the repo's registered strict/forced-final "
                         "wording for the GPT-OSS turn-50 needle "
                         "(docs/GPT_OSS_20B_APA_GRM_SYNTHESIS.md protocol; "
                         "scripts/gpt_oss20b_turn_prompt_sweep.py "
                         "'turn50_strict'), identifier tokens preserved",
                "prediction": "partial — if refusals flip here, the model "
                              "READ the graft and the refusal is the instruct "
                              "prior, not a retrieval failure",
                "prediction_source": "lead, stated in the order",
            },
            "A5": {
                "label": "mass partition on every arm",
                "build": "per-token mounted mass AND per-layer partition "
                         "(sink / mounted band / live band / question) at the "
                         "readout positions, for every arm and every probe",
                "prediction": "A0 mounted mass on the refusers < the "
                              "persistent-frame served-control masses "
                              "(0.29-0.44); A1 shows the mass on the LIVE "
                              "band; A2 moves it onto the mounted band",
                "prediction_source": "lead, stated in the order",
                "note": "the 0.29-0.44 figure is the LEAD'S PREDICTION as "
                        "written in the order, carried verbatim; it is not a "
                        "threshold this order applies and nothing branches "
                        "on it",
            },
        },
        "instrument_REGISTERED_BEFORE_ANY_GATE": {
            "chosen": "core.grm_demand.DemandObserver, subclassed in "
                      "scripts/grm_rs1_read_strength_gpu.py to ALSO retain "
                      "the per-layer rows it already computes",
            "why": "grm_demand is the PRODUCTION D-NGH observer and its "
                   "_full_mass is the frozen race computation operand for "
                   "operand (its own docstring says so, and SC1 G2 pins the "
                   "equality on real GPU numbers). The frozen "
                   "grm_det1_common.DetectorObserver computes the same rows; "
                   "the production one is chosen so the reported partition "
                   "is the one production would report.",
            "partition": [
                "physical_sink_mass", "mounted_mass", "live_mass",
                "learned_sink_mass"],
            "partition_law": "the four masses sum to one per layer and per "
                             "token; grm_demand raises if the mean partition "
                             "leaves [0.98, 1.02]",
            "band_definition": "physical sink = cache rows [0, n_sink); "
                               "mounted band = [n_sink, n_sink+cur_mount_n); "
                               "live band = the remaining real cache rows "
                               "(under this harness that band carries the "
                               "QUESTION prompt, and under A1 the fed node "
                               "text as well); learned sink = the appended "
                               "per-head sink logit",
            "question_vs_live": "the harness cannot split the live band into "
                                "question-vs-fed-text from the D-NGH rows "
                                "alone, so the receipt reports the live band "
                                "as ONE number plus the arena's own live_segs "
                                "ledger next to it, rather than inventing a "
                                "split",
            "demand_flag": "GRM_DEMAND_NGH stays OFF for every arm. The "
                           "observer is attached by the RS1 harness for "
                           "MEASUREMENT only; it never gates, never fires a "
                           "demand trip, and never changes the served text.",
        },
        "vocabulary_REGISTERED_BEFORE_ANY_GATE": VOCABULARY,
        "verdict_rule_REGISTERED_BEFORE_ANY_GATE": (
            "Per refuser, in order: QUANTIZATION if A3 rescues; else PRIOR if "
            "A4 rescues; else CONTENT if A2 rescues; else POSITION if A1 "
            "rescues and A2 does not; else UNRESOLVED. 'Rescues' means the "
            "arm's served answer is CORRECT under the frozen DET1 semantic "
            "comparator (expected value present, no rejected value present) "
            "where A0's was not."),
        "probes_REGISTERED_BEFORE_ANY_GATE": probes,
        "refusers": list(REFUSERS),
        "controls": list(CONTROLS),
        "gpu_discipline": {
            "lease": "/tmp/forge-gpu.lock via "
                     "scripts.grm_cmc1_gpu_arms.gpu_lease",
            "single_gpu": True,
            "max_wall_per_run_s": "<= 600 (house rule), lease cap below it",
            "inter_run_gap_s": ">= 30, inside the wrapper",
            "operator_right_of_way": "absolute; the harness WAITS on the "
                                     "flock and never signals another "
                                     "process",
        },
        "sources": {
            "eb1_summary": file_record(EB1_SUMMARY),
            "p2c_g2_summary": file_record(P2C_G2_SUMMARY),
            "grm_frame": file_record(ROOT / "core" / "grm_frame.py"),
            "grm_demand": file_record(ROOT / "core" / "grm_demand.py"),
            "graft_arena": file_record(ROOT / "core" / "graft_arena.py"),
            "grm_e2e_session": file_record(
                ROOT / "scripts" / "grm_e2e_session.py"),
            "lsr_p2c_replay_gpu": file_record(
                ROOT / "scripts" / "lsr_p2c_replay_gpu.py"),
            "grm_det1_3_gpu": file_record(
                ROOT / "scripts" / "grm_det1_3_gpu.py"),
            "registration_builder": file_record(Path(__file__).resolve()),
        },
    }


def main() -> int:
    payload = build()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT_DIR / "registration.json"
    path.write_bytes(canonical_json_bytes(payload))
    print(f"registration={path}")
    for probe_id, row in payload["probes_REGISTERED_BEFORE_ANY_GATE"].items():
        print(json.dumps({
            "probe_id": probe_id,
            "role": row["role"],
            "session_id": row["session_id"],
            "a0_mounted_ids": row["a0_mounted_ids"],
            "live_ids": row["live_ids"],
            "live_node_ids": row["live_node_ids"],
            "a0_correct": row["a0_correct"],
        }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
