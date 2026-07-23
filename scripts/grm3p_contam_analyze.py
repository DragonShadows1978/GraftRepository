#!/usr/bin/env python3
"""GRM3P-DIAG-CONTAM analysis: P1 correlation, P2 node-3 lifecycle, P3 diffs.

Reads live paging_events.jsonl + mount_snapshots produced by the contam
runner. Does not rewrite predictions. Emits a JSON receipt and prints the
verbatim tables required by the order.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np


VORTEX = "Vortex-3-Sierra"
PROBE_TURNS = (5, 9, 13, 16, 19, 22, 24, 26, 30)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_scorecard(session: Path) -> dict[str, Any]:
    return json.loads((session / "probe_scorecard.json").read_text(encoding="utf-8"))


def load_manifest_nodes(session: Path) -> list[dict[str, Any]]:
    man = json.loads(
        (session / "repository" / "manifest.json").read_text(encoding="utf-8")
    )
    return list(man.get("nodes") or [])


def vortex_node_ids(nodes: list[dict[str, Any]]) -> list[int]:
    out = []
    for i, n in enumerate(nodes):
        text = n.get("text") if isinstance(n, dict) else str(n)
        if VORTEX in str(text):
            out.append(int(i))
    return out


def ever_washed(events: list[dict[str, Any]], node_id: int, before_turn: int) -> dict[str, Any]:
    """A node is washed before a turn if it was pack-evicted and later page-in
    rehydrated at least once with event.turn < before_turn.

    Operational reading of pack->evict->rehydrate from live telemetry:
      - upload(source=device_pack) or host snapshot during spill prep
      - evict
      - page_in(success=True)  [rehydrate]
    We also accept (evict + later page_in) as wash even if pack was implicit
    in the same _page() cycle (pack is logged as upload before evict).
    """
    packed_turns = []
    evict_turns = []
    pagein_turns = []
    for e in events:
        if int(e.get("node_id", -1)) != int(node_id):
            continue
        t = e.get("turn")
        if t is None:
            continue
        t = int(t)
        if t >= int(before_turn):
            continue
        kind = e.get("kind")
        if kind == "upload" and e.get("packed"):
            packed_turns.append(t)
        elif kind == "evict":
            evict_turns.append(t)
        elif kind == "page_in" and e.get("success") is True:
            pagein_turns.append(t)
    # Wash requires eviction and a subsequent (or same-turn after) page-in,
    # with packing either observed or implied by the spill path.
    washed = False
    first_wash_turn = None
    evidence = []
    for et in evict_turns:
        later_pi = [p for p in pagein_turns if p >= et]
        if later_pi:
            washed = True
            first_wash_turn = et if first_wash_turn is None else min(first_wash_turn, et)
            evidence.append(f"evict@t{et}+page_in@t{later_pi[0]}")
    if not washed and packed_turns and evict_turns:
        evidence.append(
            f"packed@t{packed_turns[0]}+evict@t{evict_turns[0]} but no page_in before probe"
        )
    if not washed and not evict_turns:
        evidence.append("never_evicted")
    return {
        "washed": washed,
        "first_wash_turn": first_wash_turn,
        "packed_turns": packed_turns,
        "evict_turns": evict_turns,
        "pagein_turns": pagein_turns,
        "evidence": ";".join(evidence) if evidence else "none",
    }


def lifecycle_timeline(events: list[dict[str, Any]], node_id: int) -> list[dict[str, Any]]:
    rows = []
    for e in events:
        if int(e.get("node_id", -1)) != int(node_id):
            continue
        rows.append({
            "turn": e.get("turn"),
            "step": e.get("step"),
            "kind": e.get("kind"),
            "source": e.get("source"),
            "success": e.get("success"),
            "packed": e.get("packed"),
            "reason": e.get("reason"),
            "rehydrate": e.get("rehydrate"),
        })
    rows.sort(key=lambda r: (r["turn"] is None, r["turn"] if r["turn"] is not None else -1,
                             str(r["kind"])))
    return rows


def find_snapshot(session: Path, turn: int, node_id: int) -> Optional[Path]:
    snap_root = session / "mount_snapshots"
    if not snap_root.exists():
        return None
    # Prefer probe_mount label.
    candidates = sorted(snap_root.glob(f"turn_{turn:04d}_*/node_{node_id:04d}_device.npz"))
    if not candidates:
        candidates = sorted(snap_root.glob(f"turn_{turn:04d}_*/node_{node_id:04d}.json"))
        return candidates[0] if candidates else None
    return candidates[0]


def load_device_arrays(path: Path) -> Optional[dict[str, np.ndarray]]:
    if path is None or not path.exists() or path.suffix != ".npz":
        return None
    z = np.load(path)
    return {"k": z["k"].astype(np.float32), "v": z["v"].astype(np.float32)}


def load_snapshot_meta(session: Path, turn: int, node_id: int) -> Optional[dict[str, Any]]:
    snap_root = session / "mount_snapshots"
    if not snap_root.exists():
        return None
    metas = sorted(snap_root.glob(f"turn_{turn:04d}_*/node_{node_id:04d}.json"))
    if not metas:
        return None
    return json.loads(metas[0].read_text(encoding="utf-8"))


def per_layer_delta(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    if a.shape != b.shape:
        return {
            "shape_a": list(a.shape),
            "shape_b": list(b.shape),
            "error": "shape_mismatch",
        }
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    per = []
    for li in range(d.shape[0]):
        layer = d[li]
        per.append({
            "layer": int(li),
            "max_abs": float(layer.max()),
            "mean_abs": float(layer.mean()),
        })
    return {
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
        "per_layer_max_mean": [
            {"layer": p["layer"], "max_abs": p["max_abs"], "mean_abs": p["mean_abs"]}
            for p in per
        ],
        "layers_nonzero": int(sum(1 for p in per if p["max_abs"] > 0)),
    }


def key_norm_stats(arr: np.ndarray) -> dict[str, float]:
    # arr (L,H,S,D) -> norms over last dims per head
    norms = np.linalg.norm(arr.reshape(arr.shape[0], arr.shape[1], -1), axis=2)
    return {
        "mean": float(norms.mean()),
        "max": float(norms.max()),
        "min": float(norms.min()),
        "std": float(norms.std()),
    }


def analyze_leg(session: Path) -> dict[str, Any]:
    sc = load_scorecard(session)
    events = read_jsonl(session / "paging_events.jsonl")
    nodes = load_manifest_nodes(session)
    vortex_ids = vortex_node_ids(nodes)
    probes = []
    for p in sc.get("probes", []):
        turn = int(p["turn"])
        # Contaminator candidates present at the probe: live window + mounts.
        live = list((p.get("eviction_check") or {}).get("live_node_ids_before") or [])
        mounted = list(p.get("mounted_ids") or [])
        plan = list(p.get("mount_plan") or [])
        present = sorted(set(int(x) for x in (live + mounted + plan)))
        vortex_present = [i for i in present if i in vortex_ids]
        # Also any vortex id that appears in ranking top5 text.
        ranking = p.get("route_ranking") or {}
        for row in ranking.get("top5") or []:
            nid = row.get("node_id")
            if nid is not None and int(nid) in vortex_ids and int(nid) not in vortex_present:
                vortex_present.append(int(nid))
        # Primary contaminator: vortex node in present set; else first vortex.
        contam = vortex_present[0] if vortex_present else (vortex_ids[0] if vortex_ids else None)
        wash = (
            ever_washed(events, contam, turn)
            if contam is not None else
            {"washed": None, "evidence": "no_vortex_node"}
        )
        probes.append({
            "turn": turn,
            "fact_id": p.get("fact_id"),
            "pass": bool(p.get("pass")),
            "answer": p.get("answer"),
            "source_node_id": (ranking.get("source_node_id")
                               if ranking else None),
            "source_rank": ranking.get("source_rank") if ranking else None,
            "live": live,
            "mounted_ids": mounted,
            "mount_plan": plan,
            "vortex_node_ids_global": vortex_ids,
            "vortex_in_present": vortex_present,
            "contaminator_node_id": contam,
            "contaminator_washed": wash.get("washed"),
            "wash_evidence": wash.get("evidence"),
            "wash_detail": wash,
            "p1_agree": (
                None if wash.get("washed") is None
                else (bool(p.get("pass")) == bool(wash.get("washed")))
            ),
        })
    n3_timeline = lifecycle_timeline(events, 3)
    # Also cypher node (usually 4 in full script) for honesty.
    cypher_ids = [i for i, n in enumerate(nodes)
                  if "cypher bridge" in str(n.get("text", "")).lower()]
    cypher_timelines = {
        str(i): lifecycle_timeline(events, i) for i in cypher_ids
    }
    return {
        "session": str(session),
        "passed": sc.get("passed"),
        "total": sc.get("total", len(sc.get("probes", []))),
        "all_passed": sc.get("all_passed"),
        "n_paging_events": len(events),
        "vortex_node_ids": vortex_ids,
        "cypher_node_ids": cypher_ids,
        "probes": probes,
        "node3_timeline": n3_timeline,
        "cypher_timelines": cypher_timelines,
        "p1_agree_count": sum(1 for r in probes if r["p1_agree"] is True),
        "p1_disagree_count": sum(1 for r in probes if r["p1_agree"] is False),
        "p1_unknown_count": sum(1 for r in probes if r["p1_agree"] is None),
    }


def p3_compare(legs: dict[str, Path], node_id: int = 3, turn: int = 5) -> dict[str, Any]:
    arrays = {}
    metas = {}
    for name, session in legs.items():
        path = find_snapshot(session, turn, node_id)
        arrays[name] = load_device_arrays(path) if path is not None else None
        metas[name] = load_snapshot_meta(session, turn, node_id)
    pairs = []
    names = list(legs.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            aa, bb = arrays.get(a), arrays.get(b)
            if aa is None or bb is None:
                pairs.append({
                    "pair": f"{a}_vs_{b}",
                    "status": "missing_arrays",
                    "a_present": aa is not None,
                    "b_present": bb is not None,
                })
                continue
            pairs.append({
                "pair": f"{a}_vs_{b}",
                "status": "ok",
                "k_delta": per_layer_delta(aa["k"], bb["k"]),
                "v_delta": per_layer_delta(aa["v"], bb["v"]),
                "k_norm_a": key_norm_stats(aa["k"]),
                "k_norm_b": key_norm_stats(bb["k"]),
                "device_sha_a": {
                    "k": hashlib.sha256(aa["k"].astype(np.float16).tobytes()).hexdigest(),
                    "v": hashlib.sha256(aa["v"].astype(np.float16).tobytes()).hexdigest(),
                },
                "device_sha_b": {
                    "k": hashlib.sha256(bb["k"].astype(np.float16).tobytes()).hexdigest(),
                    "v": hashlib.sha256(bb["v"].astype(np.float16).tobytes()).hexdigest(),
                },
            })
    # Packed host control from snapshot meta.
    host = {}
    for name, meta in metas.items():
        if meta is None:
            host[name] = None
        else:
            host[name] = meta.get("host_payload_control")
    host_equal = {}
    hnames = [n for n, h in host.items() if h and h.get("aggregate_sha256")]
    for i in range(len(hnames)):
        for j in range(i + 1, len(hnames)):
            a, b = hnames[i], hnames[j]
            host_equal[f"{a}_vs_{b}"] = (
                host[a]["aggregate_sha256"] == host[b]["aggregate_sha256"]
            )
    return {
        "node_id": node_id,
        "turn": turn,
        "device_present": {n: arrays[n] is not None for n in legs},
        "host_control": {
            n: (None if host[n] is None else {
                "aggregate_sha256": host[n].get("aggregate_sha256"),
                "storage_bits": host[n].get("storage_bits"),
                "format_version": host[n].get("format_version"),
            })
            for n in legs
        },
        "host_equal": host_equal,
        "pairs": pairs,
        "metas_device_stats": {
            n: (None if metas[n] is None else metas[n].get("device_stats"))
            for n in legs
        },
    }


def scorecard_line(leg: dict[str, Any], tag: str) -> str:
    probes = leg["probes"]
    fails = [p for p in probes if not p["pass"]]
    if not fails:
        turns = ", ".join(str(p["turn"]) for p in probes)
        return (
            f"{tag}: {leg['passed']}/{leg['total']}; "
            f"PASS turns {turns}."
        )
    parts = [f"{tag}: {leg['passed']}/{leg['total']}"]
    for p in fails:
        ans = (p.get("answer") or "").replace("\n", " ")
        if len(ans) > 80:
            ans = ans[:77] + "..."
        parts.append(f"FAIL turn {p['turn']} {p.get('fact_id')} -> {ans!r}")
    passes = [str(p["turn"]) for p in probes if p["pass"]]
    parts.append(f"PASS turns {', '.join(passes)}")
    return "; ".join(parts) + "."


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--legs", nargs="+", required=True, type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--node3", type=int, default=3)
    ap.add_argument("--p3-turn", type=int, default=5)
    args = ap.parse_args(argv)

    legs_map: dict[str, Path] = {}
    analyses: dict[str, Any] = {}
    for path in args.legs:
        path = path.resolve()
        name = path.name
        # Normalize tags
        if "08mb" in name or "8mb" in name:
            tag = "8MB"
        elif "16mb" in name:
            tag = "16MB"
        elif "unbounded" in name:
            tag = "unbounded"
        else:
            tag = name
        legs_map[tag] = path
        analyses[tag] = analyze_leg(path)

    # P3 for node 3 and for actual vortex/cypher node if different.
    p3_node3 = p3_compare(legs_map, node_id=int(args.node3), turn=int(args.p3_turn))
    # Prefer full-script cypher node id from first leg.
    first = next(iter(analyses.values()))
    cypher_ids = first.get("cypher_node_ids") or []
    p3_cypher = None
    if cypher_ids and int(cypher_ids[0]) != int(args.node3):
        p3_cypher = p3_compare(
            legs_map, node_id=int(cypher_ids[0]), turn=int(args.p3_turn)
        )

    # Aggregate P1 over all legs
    p1_rows = []
    for tag, leg in analyses.items():
        for p in leg["probes"]:
            p1_rows.append({
                "leg": tag,
                **{k: p[k] for k in (
                    "turn", "fact_id", "pass", "contaminator_node_id",
                    "contaminator_washed", "wash_evidence", "p1_agree",
                    "vortex_in_present", "live", "mounted_ids",
                )},
            })

    agree = sum(1 for r in p1_rows if r["p1_agree"] is True)
    disagree = sum(1 for r in p1_rows if r["p1_agree"] is False)
    unknown = sum(1 for r in p1_rows if r["p1_agree"] is None)

    # P2 checks
    p2 = {}
    for tag, leg in analyses.items():
        tl = leg["node3_timeline"]
        ev = [e for e in tl if e["kind"] == "evict" and e.get("turn") is not None
              and 3 <= int(e["turn"]) < 5]
        pi = [e for e in tl if e["kind"] == "page_in" and e.get("success")
              and e.get("turn") is not None and 3 <= int(e["turn"]) <= 5]
        p2[tag] = {
            "evicted_between_t3_t5": bool(ev),
            "pagein_between_t3_t5": bool(pi),
            "evict_events": ev,
            "pagein_events": pi,
            "full_timeline": tl,
            "washed_before_t5": ever_washed(
                read_jsonl(legs_map[tag] / "paging_events.jsonl"), 3, 5
            ),
        }

    # Frozen prediction verdicts (not rewritten)
    # P1: pass <=> contaminator washed
    p1_verdict = (
        "SUPPORTED" if disagree == 0 and agree > 0 and unknown == 0
        else ("PARTIAL" if agree > disagree else "FALSIFIED")
    )
    # P2: 8MB node3 washed t3-t5; 16MB not
    p2_8 = p2.get("8MB", {})
    p2_16 = p2.get("16MB", {})
    p2_ok_8 = bool(p2_8.get("evicted_between_t3_t5") and (
        p2_8.get("pagein_between_t3_t5")
        or (p2_8.get("washed_before_t5") or {}).get("washed")
    ))
    p2_ok_16 = not bool(
        (p2_16.get("washed_before_t5") or {}).get("washed")
    ) and not bool(p2_16.get("evicted_between_t3_t5") and p2_16.get("pagein_between_t3_t5"))
    if "8MB" in p2 and "16MB" in p2:
        if p2_ok_8 and p2_ok_16:
            p2_verdict = "SUPPORTED"
        elif p2_ok_8 or p2_ok_16:
            p2_verdict = "PARTIAL"
        else:
            p2_verdict = "FALSIFIED"
    else:
        p2_verdict = "UNKNOWN"

    # P3: device differs 8 vs 16; host packs equal
    p3_verdict = "UNKNOWN"
    if p3_node3["pairs"]:
        pair_8_16 = next(
            (p for p in p3_node3["pairs"] if "8MB" in p["pair"] and "16MB" in p["pair"]),
            None,
        )
        host_eq = p3_node3.get("host_equal", {}).get("8MB_vs_16MB")
        if pair_8_16 and pair_8_16.get("status") == "ok":
            kmax = pair_8_16["k_delta"]["max_abs"]
            vmax = pair_8_16["v_delta"]["max_abs"]
            device_differs = (kmax > 0) or (vmax > 0)
            if device_differs and host_eq is True:
                p3_verdict = "SUPPORTED"
            elif device_differs and host_eq is False:
                p3_verdict = "PARTIAL_device_diff_host_also_differs"
            elif not device_differs and host_eq is True:
                p3_verdict = "FALSIFIED_device_identical"
            else:
                p3_verdict = "FALSIFIED_or_incomplete"
        elif pair_8_16 and pair_8_16.get("status") == "missing_arrays":
            p3_verdict = "INSUFFICIENT_ARTIFACTS"

    receipt = {
        "schema": "grm3p.diag.contam.analysis.v1",
        "scorecard_lines": {
            tag: scorecard_line(leg, tag) for tag, leg in analyses.items()
        },
        "legs": analyses,
        "p1_rows": p1_rows,
        "p1_summary": {
            "agree": agree,
            "disagree": disagree,
            "unknown": unknown,
            "total": len(p1_rows),
            "verdict": p1_verdict,
        },
        "p2_node3": p2,
        "p2_verdict": p2_verdict,
        "p3_node3": p3_node3,
        "p3_cypher": p3_cypher,
        "p3_verdict": p3_verdict,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # ---- pretty print for report paste ----
    print("=== SCORECARDS ===")
    for tag, line in receipt["scorecard_lines"].items():
        print(line)

    print("\n=== P1 CONTAMINATOR-WASH CORRELATION ===")
    print(f"{'leg':<10} {'turn':>4} {'pass':>5} {'contam':>6} {'washed':>6} {'agree':>5}  evidence")
    for r in p1_rows:
        print(
            f"{r['leg']:<10} {r['turn']:>4} {str(r['pass']):>5} "
            f"{str(r['contaminator_node_id']):>6} {str(r['contaminator_washed']):>6} "
            f"{str(r['p1_agree']):>5}  {r['wash_evidence']}"
        )
    print(
        f"P1 summary: agree={agree} disagree={disagree} unknown={unknown} "
        f"total={len(p1_rows)} verdict={p1_verdict}"
    )

    print("\n=== P2 NODE-3 LIFECYCLE ===")
    for tag, info in p2.items():
        print(f"-- {tag} --")
        print(
            f"  evicted_between_t3_t5={info['evicted_between_t3_t5']} "
            f"pagein_between_t3_t5={info['pagein_between_t3_t5']} "
            f"washed_before_t5={info['washed_before_t5']}"
        )
        for e in info["full_timeline"]:
            print(f"  t={e['turn']} {e['kind']} source={e.get('source')} "
                  f"packed={e.get('packed')} success={e.get('success')}")
    print(f"P2 verdict={p2_verdict}")

    print("\n=== P3 PAYLOAD DIFF (node 3 @ t5) ===")
    print(json.dumps({
        "device_present": p3_node3["device_present"],
        "host_control": p3_node3["host_control"],
        "host_equal": p3_node3["host_equal"],
        "pairs": [
            {
                "pair": p["pair"],
                "status": p["status"],
                **(
                    {
                        "k_max": p["k_delta"]["max_abs"],
                        "k_mean": p["k_delta"]["mean_abs"],
                        "v_max": p["v_delta"]["max_abs"],
                        "v_mean": p["v_delta"]["mean_abs"],
                        "k_norm_a": p["k_norm_a"],
                        "k_norm_b": p["k_norm_b"],
                    }
                    if p.get("status") == "ok" else {}
                ),
            }
            for p in p3_node3["pairs"]
        ],
    }, indent=2))
    if p3_cypher is not None:
        print("\n=== P3 PAYLOAD DIFF (cypher node @ t5) ===")
        print(json.dumps({
            "node": p3_cypher["node_id"],
            "device_present": p3_cypher["device_present"],
            "host_equal": p3_cypher["host_equal"],
            "pairs": [
                {
                    "pair": p["pair"],
                    "status": p["status"],
                    **(
                        {
                            "k_max": p["k_delta"]["max_abs"],
                            "k_mean": p["k_delta"]["mean_abs"],
                            "v_max": p["v_delta"]["max_abs"],
                            "v_mean": p["v_delta"]["mean_abs"],
                        }
                        if p.get("status") == "ok" else {}
                    ),
                }
                for p in p3_cypher["pairs"]
            ],
        }, indent=2))
    print(f"P3 verdict={p3_verdict}")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
