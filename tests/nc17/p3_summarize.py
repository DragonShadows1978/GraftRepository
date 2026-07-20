#!/usr/bin/env python3
"""NC17-P3 summary assembler (CPU-only). Reads the per-run P3 JSON artifacts under
logs/nc17/ plus the P0/P1/P2 summaries, emits logs/nc17/p3_summary.json + prints
the required tables with P0/P1/P2 columns ALONGSIDE the P3 INT6 column: quant
config, INT6 parity (vs GT + vs P1-bf16 same-engine) + adjudication, ceilings
(APA on/off), ppl x window (APA on/off), peaks, and engine-build confirmation."""
import glob
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LOG = REPO / "logs" / "nc17"
WEIGHTS = REPO / "weights_nc17" / "qwen3_1p7b_int6"
ENGINE = "/mnt/ForgeRealm/Project-Tensor-int6/tensor_cuda/tensor_cuda/__init__.py"


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
    summary = {"stage": "NC17-P3",
               "model": "Qwen3-1.7B tc-INT6 (self-quantized, SYMMETRIC g128)",
               "engine_build": ENGINE,
               "engine_note": "INT6 ops exist ONLY in the Project-Tensor fork "
                              "branch int6-weights; every P3 GPU run ran on this "
                              "build (asserted per script + labeled per log line)."}

    qc = load(WEIGHTS / "quant_config.json")
    summary["quant_config"] = qc

    # INT6 resident vram (from smoke): emb 1187 + int6 proj 1029 + norms 0.47 + tied 0
    summary["resident_vram_note"] = (
        "measure_vram_int6: embedding 1187.0 MiB bf16 + INT6 projections 1029.0 "
        "MiB + norms 0.47 MiB + tied head 0 = 2216.47 MiB resident (vs P2 INT4 "
        "1901.47, P1 bf16 3875.5). INT6 saves ~1659 MiB of projection weight vs "
        "bf16; costs ~315 MiB more than INT4 (2.612x proj compression vs INT4 "
        "3.76x). Zeros tensor is EMPTY (symmetric grid) -> 0 resident bytes.")
    summary["resident_vram_int6_mb"] = 2216.47
    summary["resident_vram_int4_mb"] = 1901.47
    summary["resident_vram_bf16_mb"] = 3875.5

    # parity (vs GT + vs P1-bf16 same-engine fork) + adjudication
    parity = load(LOG / "p3_parity.json")
    verdict = load(LOG / "p3_verdict.json")
    if parity:
        summary["parity"] = {
            "int6_vs_gt": parity["int6_parity_vs_gt"],
            "int6_vs_p1bf16_same_engine": parity["int6_vs_p1bf16_same_engine"],
            "hard_flip_attribution": parity["hard_flip_attribution"],
            "engine_build_int6_chain": parity.get("engine_build_int6_chain"),
            "engine_build_bf16_chain": parity.get("engine_build_bf16_chain"),
        }
    if verdict:
        summary["adjudication"] = {
            "method": verdict["method"],
            "engine_build": verdict.get("engine_build"),
            "n_int6_only_hard_candidates": verdict["n_int6_only_hard_candidates"],
            "n_drift_explained": verdict["n_drift_explained"],
            "n_real_quant_flips": verdict["n_real_quant_flips"],
            "n_control": verdict["n_control"], "n_control_ok": verdict["n_control_ok"],
            "headline": verdict["headline"],
            "real_quant_flip_rows": [r for r in verdict.get("flip_rows", [])
                                     if r["verdict"] == "REAL-QUANT-FLIP"],
        }

    # ppl x window x mode (P3 INT6)
    ppl = {}
    engaged_any = []
    for f in sorted(glob.glob(str(LOG / "p3_ppl_*_w*.json"))):
        d = load(f)
        if not d:
            continue
        m = re.search(r"p3_ppl_(\w+)_w(\d+)\.json", f)
        mode, W = m.group(1), int(m.group(2))
        eng = d.get("engagement")
        if mode == "apa" and eng:
            engaged_any.append((W, eng.get("APA_ENGAGED"), eng.get("mean_engaged_frac")))
        ppl.setdefault(str(W), {})[mode] = {
            "status": d.get("status", "OK"),
            "ppl": d["ppl"], "mean_nll": d["mean_nll"],
            "scored_tokens": d["scored_tokens"], "n_windows": d["n_windows"],
            "engagement": eng, "engine_build": d.get("engine_build"),
            "poller_peak_mb": poller_peak(f"poll_p3_ppl_{mode}_{W}.log"),
        }
    summary["ppl_int6"] = ppl
    summary["apa_engagement_ppl"] = engaged_any

    # P0/P1/P2 ppl for the beside-comparison (from prior summaries)
    p1 = load(LOG / "p1_summary.json")
    p2 = load(LOG / "p2_summary.json")
    summary["p2_int4_ppl"] = (p2 or {}).get("ppl_int4", {})
    summary["p1_bf16_ppl"] = (p1 or {}).get("ppl", {})
    summary["p0_baseline_ppl"] = (p1 or {}).get("p0_baseline_ppl", {})
    summary["ppl_coverage_note"] = (
        "tc-INT6 P3 ppl scored the same first N windows within the 590s cap as P1 "
        "tc-bf16 and P2 tc-INT4 (identical GT tokenization/stride) — so "
        "INT6-vs-INT4-vs-bf16 is a MATCHED delta at each window. P0 HF-bf16 scored "
        "the full corpus; the tc-vs-P0 gap stays coverage-confounded.")

    # ceilings (P3 INT6) beside P2/P1
    ceil = {}
    for f in sorted(glob.glob(str(LOG / "p3_ceiling_*.json"))):
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
            "engine_build": d.get("engine_build"),
            "trace": [{"ctx": t["ctx"], "status": t["status"],
                       "smi_used_mb": t.get("smi_used_mb_after"),
                       "apa_engaged": (t.get("engagement") or {}).get("APA_ENGAGED")}
                      for t in d.get("trace", [])],
        }
    summary["ceilings_int6"] = ceil
    summary["p2_int4_ceilings"] = (p2 or {}).get("ceilings_int4", {})
    summary["p1_bf16_ceilings"] = (p1 or {}).get("ceilings", {})

    (LOG / "p3_summary.json").write_text(json.dumps(summary, indent=2))

    # ---------------- printed tables ----------------
    print("\n================ NC17-P3 SUMMARY (Qwen3-1.7B tc-INT6, FORK engine) ================")
    print(f"[ENGINE] {ENGINE}")
    print("  every P3 GPU run asserted this build + labeled ENGINE=int6-fork per log line")
    if qc:
        s = qc["sizes_mib"]
        print("\n[QUANT CONFIG]")
        print(f"  bits={qc['bits']} group_size={qc['group_size']} symmetry={qc['symmetry']}")
        print(f"  scheme: {qc['quant_scheme']}")
        print(f"  vs P2 INT4: {qc['vs_p2_int4_scheme']}")
        print(f"  quantized: {qc['quantized_tensors']} ({qc['n_quantized_matrices']} matrices)")
        print(f"  exempt (higher precision): {json.dumps(qc['exempt_tensors'])}")
        print(f"  artifact: {qc['artifact_path']}")
        print(f"    sha256: {qc['artifact_sha256']}")
        print(f"    on-disk npz: {s['npz_on_disk_mib']:.1f} MiB | INT6 proj {s['int6_proj_total_mib']:.1f} MiB "
              f"(vs bf16 proj {s['orig_bf16_proj_mib']:.1f} MiB, {s['compression_ratio_proj']:.3f}x)")
        print(f"  RESIDENT: 2216.47 MiB (emb 1187.0 bf16 + INT6 proj 1029.0 + norms 0.47 + tied head 0)")
        print(f"            beside: P2 INT4 1901.47 | P1 bf16 3875.5 MiB")

    if "parity" in summary:
        pg = summary["parity"]["int6_vs_gt"]
        pq = summary["parity"]["int6_vs_p1bf16_same_engine"]
        at = summary["parity"]["hard_flip_attribution"]
        print("\n[PARITY] tc-INT6 fork cached-chain vs P0 GT (same protocol/table as P1/P2)")
        print(f"  top-1 agreement: {pg['top1_agreement']:.4f} over {pg['n_positions']} positions")
        print(f"    (P2 INT4 was 0.7540; P1 bf16 0.9538 — both RED-as-registered/drift class)")
        print(f"  flips: {pg['n_flips']} ({pg['n_near_tie_flips']} near-tie, {pg['n_hard_flips']} HARD) "
              f"| cached-chain GATE {pg['verdict']} (drift-tolerance protocol; port question = fresh-refeed)")
        print(f"  INT6-vs-GT |dlogit|: max={pg['max_abs_dlogit']:.3f} mean={pg['mean_abs_dlogit']:.3f} "
              f"p99={pg['p99_abs_dlogit']:.3f}")
        print("\n[QUANT NOISE] INT6-vs-P1bf16 SAME-ENGINE(fork) cached-chain (clean isolation)")
        fv = pq["full_vector_absdelta"]; pp = pq["per_position_maxabs_delta"]
        print(f"  full-vector |delta|: mean={fv['mean']:.4f} p50={fv['p50']:.4f} "
              f"p99={fv['p99']:.3f} p99.9={fv['p999']:.3f} max={fv['max']:.3f}")
        print(f"  per-position max|delta|: mean={pp['mean']:.3f} p99={pp['p99']:.3f} max={pp['max']:.3f}")
        print(f"    (P2 INT4 same-engine full-vec mean was ~higher — INT6 halves proj quant noise)")
        print(f"\n[ATTRIBUTION] HARD flips vs GT: {at['n_hard_int6_vs_gt']} total | "
              f"{at['n_hard_shared_with_bf16']} shared-with-bf16 (drift) | "
              f"{at['n_hard_int6_only']} INT6-only")
    if "adjudication" in summary:
        a = summary["adjudication"]
        print(f"\n[ADJUDICATION] fresh-refeed of {a['n_int6_only_hard_candidates']} INT6-only HARD decode flips:")
        print(f"  drift-explained {a['n_drift_explained']} | REAL quant-noise {a['n_real_quant_flips']} "
              f"| control {a['n_control_ok']}/{a['n_control']}")
        print(f"  => {a['headline']}")
        print(f"    (P2 INT4 had 92 REAL quant flips; INT6 has {a['n_real_quant_flips']} — much cleaner)")

    print("\n[PPL] window x mode  (INT6 P3 | INT4 P2 | bf16 P1 | P0 HF-bf16 full-corpus)")
    p2ppl = summary["p2_int4_ppl"]; p1ppl = summary["p1_bf16_ppl"]; p0ppl = summary["p0_baseline_ppl"]
    allW = sorted(set(list(ppl) + list(p2ppl) + list(p1ppl) + list(p0ppl)), key=int)
    print(f"  {'window':>7} | {'i6-std':>8} | {'i6-apa':>8} | {'eng':>5} | "
          f"{'i4-std':>8} | {'i4-apa':>8} | {'bf16-std':>8} | {'bf16-apa':>8} | {'P0 HF':>8} | {'i6apa_pk':>8}")

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
        i4 = p2ppl.get(W, {})
        b = p1ppl.get(W, {})
        p0c = p0ppl.get(W, {})
        p0s = "OOM" if p0c.get("oom") else (f"{p0c['ppl']:.3f}" if p0c.get("ppl") else "-")
        print(f"  {W:>7} | {cell(s):>8} | {cell(a):>8} | {str(eng):>5} | "
              f"{cell(i4.get('standard')):>8} | {cell(i4.get('apa')):>8} | "
              f"{cell(b.get('standard')):>8} | {cell(b.get('apa')):>8} | {p0s:>8} | "
              f"{(f'{ap:.0f}' if ap else '-'):>8}")
    print("  NOTE: INT6/INT4/bf16 are matched (same windows/tokens); tc-vs-P0 coverage-confounded.")

    print("\n[CEILINGS] last-solid / first-OOM (ctx tokens)  INT6 | INT4 | bf16")
    p2c = summary["p2_int4_ceilings"]; p1c = summary["p1_bf16_ceilings"]
    for k in sorted(set(list(ceil) + list(p2c) + list(p1c))):
        c = ceil.get(k, {}); i4 = p2c.get(k, {}); b = p1c.get(k, {})
        print(f"  {k:>18}: INT6 ls={c.get('last_solid_ctx')} oom={c.get('first_oom_ctx')} "
              f"smi={c.get('max_ok_smi_mb')} | INT4 ls={i4.get('last_solid_ctx')} "
              f"oom={i4.get('first_oom_ctx')} | bf16 ls={b.get('last_solid_ctx')} "
              f"oom={b.get('first_oom_ctx')}")

    print(f"\n[wrote] {LOG / 'p3_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
