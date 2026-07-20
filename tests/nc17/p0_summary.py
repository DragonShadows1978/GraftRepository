#!/usr/bin/env python3
"""NC17-P0 step 5: aggregate deliverable tables -> logs/nc17/p0_summary.json.

Reads the download record, both OOM ladder jsonls, and the per-window ppl
records (logs/nc17/ppl_*.json) and prints + writes:
  (a) ceiling table (prefill/decode last-solid & first-OOM, poller peaks)
  (b) ppl x window (with wall-clock + poller peak)
  (c) peak-fill x context per the plan's memory-accounting table (HF-bf16 row)

No GPU. Pure aggregation of receipts already on disk.
"""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LOGDIR = REPO_ROOT / "logs" / "nc17"

# Model config facts (from the plan header; the config on disk confirms these).
CONFIG = {
    "layers": 28, "n_q_heads": 16, "n_kv_heads": 8, "head_dim": 128,
    "hidden": None, "vocab": 151936, "dtype": "bf16",
}
# KV cache bytes per token: 2 (K+V) * n_kv_heads * head_dim * 2 bytes(bf16) * layers
KV_BYTES_PER_TOKEN = 2 * CONFIG["n_kv_heads"] * CONFIG["head_dim"] * 2 * CONFIG["layers"]


def load_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main():
    dl = json.loads((LOGDIR / "p0_download.json").read_text())
    pre = load_jsonl(LOGDIR / "ladder_prefill.jsonl")
    dec = load_jsonl(LOGDIR / "ladder_decode.jsonl")

    def wall(records):
        solid = [r for r in records if r.get("solid")]
        oom = [r for r in records if r.get("oom")]
        last_solid = max((r["S"] for r in solid), default=None)
        first_oom = min((r["S"] for r in oom), default=None)
        ls = next((r for r in records if r["S"] == last_solid), None)
        fo = next((r for r in records if r["S"] == first_oom), None)
        return {
            "last_solid_S": last_solid,
            "last_solid_poller_peak_MiB": ls.get("poller_peak_MiB") if ls else None,
            "last_solid_max_alloc_MiB": ls.get("max_mem_alloc_MiB") if ls else None,
            "first_oom_S": first_oom,
            "first_oom_poller_peak_MiB": fo.get("poller_peak_MiB") if fo else None,
            "first_oom_tried_alloc": (fo.get("error") if fo else None),
        }

    ceilings = {"prefill": wall(pre), "decode": wall(dec)}

    # ppl records
    ppl_rows = []
    for pf in sorted(LOGDIR.glob("ppl_*.json")):
        r = json.loads(pf.read_text())
        ppl_rows.append(r)
    ppl_rows.sort(key=lambda r: r["window"])

    # (c) memory-accounting table (HF-bf16 row) at each ppl window rung + the
    # ladder last-solid rungs. weights resident measured from download size on
    # disk; KV computed; peak fill = poller peak.
    weights_MiB = round(dl["size_bytes"] / (1024**2), 1)

    def kv_mib(ctx):
        return round(KV_BYTES_PER_TOKEN * ctx / (1024**2), 1)

    mem_table = []
    # ppl windows
    for r in ppl_rows:
        if r.get("oom"):
            continue
        ctx = r["window"]
        mem_table.append({
            "stage": "HF-bf16 (ppl-score)", "ctx": ctx,
            "weights_resident_MiB": weights_MiB,
            "KV_at_ctx_MiB": kv_mib(ctx),
            "framework_peak_alloc_MiB": r.get("max_mem_alloc_MiB"),
            "poller_peak_MiB": r.get("poller_peak_MiB"),
        })
    # ladder last-solid rungs
    for mode, w in (("prefill", ceilings["prefill"]), ("decode", ceilings["decode"])):
        if w["last_solid_S"]:
            mem_table.append({
                "stage": f"HF-bf16 ({mode} last-solid)", "ctx": w["last_solid_S"],
                "weights_resident_MiB": weights_MiB,
                "KV_at_ctx_MiB": kv_mib(w["last_solid_S"]) if mode == "decode" else "n/a (prefill, no cache)",
                "framework_peak_alloc_MiB": w["last_solid_max_alloc_MiB"],
                "poller_peak_MiB": w["last_solid_poller_peak_MiB"],
            })

    summary = {
        "order": "NC17-P0",
        "model": {
            "repo_id": dl["repo_id"],
            "revision_hash": dl["revision_hash"],
            "download_size_bytes": dl["size_bytes"],
            "download_size_gib": dl["size_gib"],
            "safetensors_files": dl["safetensors_files"],
        },
        "config_facts": CONFIG,
        "kv_bytes_per_token": KV_BYTES_PER_TOKEN,
        "card_total_MiB": 12282,
        "a_ceilings": ceilings,
        "b_ppl_by_window": [
            {k: r.get(k) for k in ("window", "stride", "ppl", "n_windows",
                                   "n_scored_tokens", "wall_s",
                                   "max_mem_alloc_MiB", "poller_peak_MiB", "oom")}
            for r in ppl_rows
        ],
        "c_memory_accounting_HF_bf16": mem_table,
    }
    (LOGDIR / "p0_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
