#!/usr/bin/env python3
"""NC17-P2 summary assembler (CPU-only). Reads the per-run P2 JSON artifacts under
logs/nc17/ plus P0/P1 summaries, and emits logs/nc17/p2_summary.json + prints the
required tables: quant config, INT4 parity (vs GT + vs P1-bf16 same-engine),
ceilings (APA on/off) beside P1, ppl x window (APA on/off) beside P0/P1, peaks."""
import glob
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LOG = REPO / "logs" / "nc17"
WEIGHTS = REPO / "weights_nc17" / "qwen3_1p7b_int4"


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def poller_peak(name):
    p = LOG / name
    if not p.exists():
        return None
    mx = 0
    for line in p.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            try:
                mx = max(mx, float(parts[1]))
            except Exception:
                pass
    return mx or None


def main():
    summary = {"stage": "NC17-P2", "model": "Qwen3-1.7B tc-INT4 (self-quantized)"}

    # --- quant config + artifact ---
    qc = load(WEIGHTS / "quant_config.json")
    summary["quant_config"] = qc

    # --- INT4 resident vram (from smoke) ---
    summary["resident_vram_note"] = (
        "measure_vram_int4: embedding 1187.0 MiB bf16 + INT4 projections 714.0 MiB "
        "+ norms 0.47 MiB + tied head 0 = 1901.47 MiB resident (vs P1 tc-bf16 "
        "3875.5 MiB). INT4 saves ~1974 MiB of projection weights.")

    # --- parity (vs GT + vs P1-bf16) + adjudication ---
    parity = load(LOG / "p2_parity.json")
    verdict = load(LOG / "p2_verdict.json")
    if parity:
        summary["parity"] = {
            "int4_vs_gt": parity["int4_parity_vs_gt"],
            "int4_vs_p1bf16_same_engine": parity["int4_vs_p1bf16_same_engine"],
            "hard_flip_attribution": parity["hard_flip_attribution"],
        }
    if verdict:
        summary["adjudication"] = {
            "method": verdict["method"],
            "n_int4_only_hard_candidates": verdict["n_int4_only_hard_candidates"],
            "n_drift_explained": verdict["n_drift_explained"],
            "n_real_quant_flips": verdict["n_real_quant_flips"],
            "n_control": verdict["n_control"], "n_control_ok": verdict["n_control_ok"],
            "headline": verdict["headline"],
        }

    # --- ppl x window x mode (P2 INT4) ---
    ppl = {}
    for f in sorted(glob.glob(str(LOG / "p2_ppl_*_w*.json"))):
        d = load(f)
        if not d:
            continue
        m = re.search(r"p2_ppl_(\w+)_w(\d+)\.json", f)
        mode, W = m.group(1), int(m.group(2))
        ppl.setdefault(str(W), {})[mode] = {
            "status": d.get("status", "OK"),
            "ppl": d["ppl"], "mean_nll": d["mean_nll"],
            "scored_tokens": d["scored_tokens"], "n_windows": d["n_windows"],
            "engagement": d.get("engagement"),
            "poller_peak_mb": poller_peak(f"poll_p2_ppl_{mode}_{W}.log"),
        }
    summary["ppl_int4"] = ppl

    # --- P0 + P1 ppl for the beside-comparison ---
    p1 = load(LOG / "p1_summary.json")
    summary["p1_bf16_ppl"] = (p1 or {}).get("ppl", {})
    summary["p0_baseline_ppl"] = (p1 or {}).get("p0_baseline_ppl", {})
    summary["ppl_coverage_note"] = (
        "tc-INT4 P2 ppl scored the same first N windows within the 590s cap as P1 "
        "tc-bf16 (identical GT tokenization/stride) — so INT4-vs-P1 is a MATCHED "
        "delta. P0 HF-bf16 scored the full corpus (299,077 tok); the tc-vs-P0 gap "
        "stays coverage-confounded. APA-on vs APA-off within INT4 is also matched.")

    # --- ceilings (P2 INT4) beside P1 ---
    ceil = {}
    for f in sorted(glob.glob(str(LOG / "p2_ceiling_*.json"))):
        d = load(f)
        if not d:
            continue
        key = f"{d['mode']}_{d['kind']}"
        peaks = [t.get("smi_used_mb_after") for t in d.get("trace", [])
                 if t.get("status") == "OK" and t.get("smi_used_mb_after")]
        ceil[key] = {
            "last_solid_ctx": d["last_solid_ctx"],
            "first_oom_ctx": d["first_oom_ctx"],
            "max_ok_smi_mb": max(peaks) if peaks else None,
            "trace": [{"ctx": t["ctx"], "status": t["status"],
                       "smi_used_mb": t.get("smi_used_mb_after"),
                       "apa_engaged": (t.get("engagement") or {}).get("APA_ENGAGED")}
                      for t in d.get("trace", [])],
        }
    summary["ceilings_int4"] = ceil
    summary["p1_bf16_ceilings"] = (p1 or {}).get("ceilings", {})

    (LOG / "p2_summary.json").write_text(json.dumps(summary, indent=2))

    # ---------------- printed tables ----------------
    print("\n================ NC17-P2 SUMMARY (Qwen3-1.7B tc-INT4) ================")
    if qc:
        s = qc["sizes_mib"]
        print("\n[QUANT CONFIG]")
        print(f"  bits={qc['bits']} group_size={qc['group_size']} symmetry={qc['symmetry']}")
        print(f"  scheme: {qc['quant_scheme']}")
        print(f"  quantized: {qc['quantized_tensors']} ({qc['n_quantized_matrices']} matrices)")
        print(f"  exempt (higher precision): {json.dumps(qc['exempt_tensors'])}")
        print(f"  artifact: {qc['artifact_path']}")
        print(f"    sha256: {qc['artifact_sha256']}")
        print(f"    on-disk npz: {s['npz_on_disk_mib']:.1f} MiB | INT4 proj {s['int4_proj_total_mib']:.1f} MiB "
              f"(vs bf16 proj {s['orig_bf16_proj_mib']:.1f} MiB, {s['compression_ratio_proj']:.2f}x)")
        print(f"  RESIDENT: 1901.47 MiB total (emb 1187.0 bf16 + INT4 proj 714.0 + norms 0.47 + tied head 0)")
        print(f"            vs P1 tc-bf16 resident 3875.5 MiB")

    if "parity" in summary:
        pg = summary["parity"]["int4_vs_gt"]
        pq = summary["parity"]["int4_vs_p1bf16_same_engine"]
        at = summary["parity"]["hard_flip_attribution"]
        print("\n[PARITY] tc-INT4 cached-chain vs P0 GT (same protocol/table as P1)")
        print(f"  top-1 agreement: {pg['top1_agreement']:.4f} over {pg['n_positions']} positions")
        print(f"    (P1 tc-bf16 cached-chain was 0.9538 over 715; RED-as-registered/drift)")
        print(f"  flips: {pg['n_flips']} ({pg['n_near_tie_flips']} near-tie, {pg['n_hard_flips']} HARD) "
              f"| VERDICT {pg['verdict']}")
        print(f"  INT4-vs-GT |dlogit|: max={pg['max_abs_dlogit']:.3f} mean={pg['mean_abs_dlogit']:.3f} "
              f"p99={pg['p99_abs_dlogit']:.3f}")
        print("\n[QUANT NOISE] INT4-vs-P1bf16 SAME-ENGINE cached-chain (clean isolation)")
        fv = pq["full_vector_absdelta"]; pp = pq["per_position_maxabs_delta"]
        print(f"  full-vector |delta|: mean={fv['mean']:.4f} p50={fv['p50']:.4f} "
              f"p99={fv['p99']:.3f} p99.9={fv['p999']:.3f} max={fv['max']:.3f}")
        print(f"  per-position max|delta|: mean={pp['mean']:.3f} p99={pp['p99']:.3f} max={pp['max']:.3f}")
        print(f"\n[ATTRIBUTION] HARD flips vs GT: {at['n_hard_int4_vs_gt']} total | "
              f"{at['n_hard_shared_with_bf16']} shared-with-bf16 (drift) | "
              f"{at['n_hard_int4_only']} INT4-only")
    if "adjudication" in summary:
        a = summary["adjudication"]
        print(f"\n[ADJUDICATION] fresh-refeed of {a['n_int4_only_hard_candidates']} INT4-only HARD decode flips:")
        print(f"  drift-explained {a['n_drift_explained']} | REAL quant-noise {a['n_real_quant_flips']} "
              f"| control {a['n_control_ok']}/{a['n_control']}")
        print(f"  => {a['headline']}")

    print("\n[PPL] window x mode  (INT4 P2 | P1 bf16 | P0 HF-bf16 full-corpus)")
    p1ppl = summary["p1_bf16_ppl"]; p0ppl = summary["p0_baseline_ppl"]
    allW = sorted(set(list(ppl) + list(p1ppl) + list(p0ppl)), key=int)
    print(f"  {'window':>7} | {'int4-std':>9} | {'int4-apa':>9} | {'eng':>5} | "
          f"{'bf16-std':>9} | {'bf16-apa':>9} | {'P0 HF':>9} | {'i4apa_pk':>8}")

    def cell(x):
        if not x:
            return "-"
        if x.get("status", "OK") != "OK":
            return x.get("status")
        v = x.get("ppl")
        return f"{v:.3f}" if (v is not None and v == v) else "nan"

    for W in allW:
        row = ppl.get(W, {})
        s = row.get("standard", {}); a = row.get("apa", {})
        eng = (a.get("engagement") or {}).get("APA_ENGAGED")
        ap = a.get("poller_peak_mb")
        b = p1ppl.get(W, {})
        bs = cell(b.get("standard")); ba = cell(b.get("apa"))
        p0c = p0ppl.get(W, {})
        p0s = "OOM" if p0c.get("oom") else (f"{p0c['ppl']:.3f}" if p0c.get("ppl") else "-")
        print(f"  {W:>7} | {cell(s):>9} | {cell(a):>9} | {str(eng):>5} | "
              f"{bs:>9} | {ba:>9} | {p0s:>9} | {(f'{ap:.0f}' if ap else '-'):>8}")
    print("  NOTE: INT4-vs-bf16 is matched (same windows/tokens); tc-vs-P0 coverage-confounded.")

    print("\n[CEILINGS] last-solid / first-OOM (ctx tokens)  INT4 | P1 bf16")
    p1c = summary["p1_bf16_ceilings"]
    for k in sorted(set(list(ceil) + list(p1c))):
        c = ceil.get(k, {})
        b = p1c.get(k, {})
        print(f"  {k:>18}: INT4 last_solid={c.get('last_solid_ctx')} "
              f"first_oom={c.get('first_oom_ctx')} max_ok_smi={c.get('max_ok_smi_mb')} "
              f"| bf16 last_solid={b.get('last_solid_ctx')} first_oom={b.get('first_oom_ctx')}")

    print(f"\n[wrote] {LOG / 'p2_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
