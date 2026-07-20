#!/usr/bin/env python3
"""NC17-P1 summary assembler (CPU-only). Reads the per-run JSON artifacts under
logs/nc17/ and emits logs/nc17/p1_summary.json + prints the required tables:
parity margins, ceilings (APA on/off), ppl x window (APA on/off vs P0), peaks."""
import glob
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LOG = REPO / "logs" / "nc17"


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
    summary = {"stage": "NC17-P1", "model": "Qwen3-1.7B tc-bf16"}

    # --- parity ---
    parity = load(LOG / "p1_parity.json")
    if parity:
        summary["parity"] = {
            "gt_file": parity["gt_file"], "gt_sha256": parity["gt_sha256"],
            "top1_agreement": parity["top1_agreement"],
            "n_positions": parity["n_positions"],
            "n_flips": parity["n_flips"],
            "n_near_tie_flips": parity["n_near_tie_flips"],
            "n_hard_flips": parity["n_hard_flips"],
            "max_abs_dlogit": parity["max_abs_dlogit"],
            "mean_abs_dlogit": parity["mean_abs_dlogit"],
            "p99_abs_dlogit": parity["p99_abs_dlogit"],
            "flips": parity["flips"],
        }

    # --- ppl x window x mode ---
    ppl = {}
    for f in sorted(glob.glob(str(LOG / "p1_ppl_*_w*.json"))):
        d = load(f)
        if not d:
            continue
        m = re.search(r"p1_ppl_(\w+)_w(\d+)\.json", f)
        mode, W = m.group(1), int(m.group(2))
        ppl.setdefault(str(W), {})[mode] = {
            "status": d.get("status", "OK"),
            "ppl": d["ppl"], "mean_nll": d["mean_nll"],
            "scored_tokens": d["scored_tokens"], "n_windows": d["n_windows"],
            "engagement": d.get("engagement"),
            "poller_peak_mb": poller_peak(f"poll_ppl_{mode}_{W}.log"),
        }
    summary["ppl"] = ppl

    # --- ceilings ---
    ceil = {}
    for f in sorted(glob.glob(str(LOG / "p1_ceiling_*.json"))):
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
    summary["ceilings"] = ceil

    (LOG / "p1_summary.json").write_text(json.dumps(summary, indent=2))

    # ---- printed tables ----
    print("\n================ NC17-P1 SUMMARY (Qwen3-1.7B tc-bf16) ================")
    if "parity" in summary:
        p = summary["parity"]
        print("\n[PARITY] vs P0 GT")
        print(f"  GT sha256: {p['gt_sha256']}")
        print(f"  top-1 agreement: {p['top1_agreement']:.4f} over {p['n_positions']} positions")
        print(f"  flips: {p['n_flips']} ({p['n_near_tie_flips']} near-tie, "
              f"{p['n_hard_flips']} hard)")
        print(f"  max|dlogit|={p['max_abs_dlogit']:.3f} mean={p['mean_abs_dlogit']:.3f} "
              f"p99={p['p99_abs_dlogit']:.3f}")

    print("\n[PPL] window x mode  (P0 HF-bf16 baseline ppl: see p0 logs; these are tc)")
    print(f"  {'window':>7} | {'standard':>12} | {'apa(r0.15)':>12} | {'apa_engaged':>11} | {'std_peak':>8} | {'apa_peak':>8}")
    for W in sorted(ppl, key=int):
        row = ppl[W]
        s = row.get("standard", {})
        a = row.get("apa", {})
        eng = (a.get("engagement") or {}).get("APA_ENGAGED")
        sp = s.get("poller_peak_mb"); ap = a.get("poller_peak_mb")

        def cell(x):
            if x.get("status", "OK") != "OK":
                return x.get("status")
            v = x.get("ppl")
            return f"{v:.3f}" if v == v else "nan"  # nan check
        print(f"  {W:>7} | {cell(s):>12} | {cell(a):>12} | {str(eng):>11} | "
              f"{(f'{sp:.0f}' if sp else '-'):>8} | {(f'{ap:.0f}' if ap else '-'):>8}")

    print("\n[CEILINGS] last-solid / first-OOM (ctx tokens)")
    for k in sorted(ceil):
        c = ceil[k]
        print(f"  {k:>18}: last_solid={c['last_solid_ctx']} "
              f"first_oom={c['first_oom_ctx']} max_ok_smi_mb={c['max_ok_smi_mb']}")

    print(f"\n[wrote] {LOG / 'p1_summary.json'}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
