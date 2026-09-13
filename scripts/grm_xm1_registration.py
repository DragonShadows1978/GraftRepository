#!/usr/bin/env python3
"""XM1 immutable, CPU-only registration builder.

Prior art: GRM contributors (2026), RS4 registration and LT1 immutable cell
receipts. Reused: frozen expectations, repo-relative hashes, pre-gate rails.
Ours: adapter inventory and cross-model cell identities; no new algorithm.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/grm_xm1'
MODELS = ('gpt-oss', 'qwen35', 'minicpm3', 'trinity', 'olmoe', 'gemma4')
ARMS = ('C3l', 'C5')
ADAPTERS = {
 'gpt-oss': {'path': 'core/gpt_oss20b_tc.py', 'kind': 'gqa_sink', 'status': 'READY', 'estimate_s': 150},
 'qwen35': {'path': 'core/qwen35_tc.py', 'kind': 'partial_rope', 'status': 'READY', 'estimate_s': 120},
 'minicpm3': {'path': 'core/minicpm3_tc.py', 'kind': 'mla', 'status': 'READY', 'estimate_s': 120},
 'trinity': {'path': 'scripts/trinity_nope_graft_width_sweep.py', 'kind': 'nope_mixed', 'status': 'READY', 'estimate_s': 150},
 'olmoe': {'path': 'scripts/olmoe_e2_experiment.py', 'kind': 'router_inputs_only', 'status': 'NO_GRAFT_PATH', 'estimate_s': 0},
 'gemma4': {'path': 'core/gemma4_tc.py', 'kind': 'ring_cache_only', 'status': 'NO_GRAFT_PATH', 'estimate_s': 0},
}
PREDICTION = 'parity holds on every model that has a GRM adapter'
PREDICTION_CONTEXT = 'Lead\'s registered prediction: **parity holds on every model that has\na GRM adapter** — the read gap is a seating/geometry effect, not a model\nproperty.'
FALSIFIER = 'Falsifier: any model where fed-minus-mount mass on the same probe\nexceeds 0.10 with seat-near-live ON, or where the served answer flips from\ncorrect (fed) to refusal (mounted) on ≥ 2 of the panel.'
ENV = {'GRM_LSR_FIXES': '1', 'GRM_DEMAND_NGH': '0',
       'GRM_FOLD_RETAIN_SOURCES': '0', 'GRM_FOLD_ALIAS_GUARD': '0',
       'GRM_ROUTE_SOLE_BINDER_INSURANCE': '0', 'GRM_ALIAS_FOLD_MERGE': '0',
       'GRM_ADMISSION_RULE': 'all_tokens_bind', 'GRM_PERSISTENT_BOAT': '0',
       'GRM_CAPTURE_PIN': 'off', 'GRM_SEAT_NEAR_LIVE': '0',
       'GRM_X1_ADDRESSES': '0'}

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def create(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        f.write('\n')

def build():
    rs3 = read(ROOT / 'artifacts/grm_rs3/registration.json')
    table = read(ROOT / 'artifacts/grm_rs4/grm_rs4_results.json')['table']
    probes = {}
    references = {}
    for row in table:
        if row['arm'] not in ARMS or not row['registered_probe'] or row['variant'] != 'registered':
            continue
        pid = row['probe_id']
        receipt_path = ROOT / 'artifacts/grm_rs4' / row['receipt']
        raw = next(r for r in read(receipt_path)['probes'] if r['probe_id'] == pid and r['variant'] == 'registered')
        p = probes.setdefault(pid, {'session_id': row['session_id'], 'role': rs3['probes_REGISTERED_BEFORE_ANY_GATE'][pid]['role'],
                                   'question': raw['question'], 'expected_values': raw['expected_values'], 'rejected_values': raw['rejected_values']})
        if row['arm'] == 'C3l':
            p['capture_texts'] = [n['capture_text'] for n in raw['info']['rs4_capture']['per_node']]
        else:
            p['feed_texts'] = [n['fed_text'] for n in rs3['probes_REGISTERED_BEFORE_ANY_GATE'][pid]['live_nodes']]
        references[f"{row['arm']}:{pid}"] = row
    probes = {p: probes[p] for p in ('sup_harbor_restatement', 'sup_praxis_fresh', 'sup_solace_fresh', 'sup_reserve_tundra_ledger', 'sup_reserve_meridian_docket')}
    pins = set(['scripts/grm_rs4_ceiling_gpu.py', 'scripts/grm_rs4_row_split.py', 'scripts/grm_rs4_registration.py',
                'scripts/grm_rs4_results.py', 'scripts/grm_rs3_capture_seat_gpu.py', 'scripts/grm_rs1_read_strength_gpu.py',
                'artifacts/grm_rs4/registration.json', 'artifacts/grm_rs3/registration.json', 'artifacts/grm_rs4/grm_rs4_results.json'])
    pins.update(str(p.relative_to(ROOT)) for p in (ROOT/'core').glob('*.py'))
    pins.update(a['path'] for a in ADAPTERS.values())
    pins.update(str(p.relative_to(ROOT)) for p in (ROOT/'artifacts/grm_rs4').glob('grm_rs4_C[35]*.json'))
    pins.update(str(p.relative_to(ROOT)) for p in (ROOT/'tests/fixtures/supersession_battery').glob('*.json'))
    pins.add('artifacts/grm_det1/run_20260831T160525Z_2/runtime_frame_28b3196f8fb04a41.json')
    return dict(schema='grm.xm1.registration.v1', immutable=True, model_order=list(MODELS), arms=list(ARMS),
                adapters=ADAPTERS, probes=probes, prediction_verbatim=PREDICTION_CONTEXT, falsifier_verbatim=FALSIFIER,
                environment=ENV, pins={p: sha(ROOT/p) for p in sorted(pins)}, rs4_references=references,
                cells=[dict(model=m, arm=a, probe=p, estimate_s=ADAPTERS[m]['estimate_s']) for m in MODELS for p in probes for a in ARMS],
                gpu_budget_s_per_model=1800, worker_cap_s=285, outer_cap_s=590,
                non_fit_candidates=['trinity: INT4 weights with float32 compute; transient MoE/attention memory unmeasured',
                                    'gemma4: no graft path; if enabled later, 12B plus mixed-width ring caches may not fit'],
                protocol='GPT-OSS delegates RS4 whole-session replay, then selects one registered probe; other adapters replay the recorded arm source texts with native chat templates and native graft capture.',
                deviations_registered=['RS4 C3l capture texts and C5 fed texts differ on this historical panel; not a same-token causal test. Preserve and expose both text sets; do not claim to resolve the stronger same-token question.',
                  'Gemma-4 has no _capture/inject_kv/graft_seats/live_shift path in the inspected adapter; blocked without a product order.',
                  'Non-GPT explicit native capture/serve uses no physical system sink, lossless host payload, and a band wide enough for the recorded text. These are not production-ladder parity claims.',
                  'Qwen3.5 graft transfers attention KV only, not DeltaNet recurrent state.',
                  'CPU doubles validate harness protocol and mass accounting, never GPU numeric parity. GPT-OSS reference replay remains GPU-blocked.'],
                gates=['Registration and separate implementation pin amendment before any gate.',
                  'Every model/arm/probe dispatch writes a create-only receipt, unavailable adapters explicitly NO_GRAFT_PATH.',
                  'Supported native adapter doubles traverse capture, seat, feed, generation, observation and scoring with only loader replaced.',
                  'GPT-OSS RS4 reference masses and served answers float-equal; any mismatch is STOP and blocks all other GPU models.',
                  'All physical band widths sum to S on every observed row; RS4 partition probabilities close at its existing tolerance.',
                  'Dry-run every non-comment line of lead_commands.txt with --dry-run appended, no GPU or lock.',
                  'Final command: python3 -m pytest -q --basetemp artifacts/grm_xm1/tmp/pytest tests/test_grm_xm1*.py tests/test_grm_rs4*.py'],
                prior_art='GRM contributors (2026), RS3/RS4 observer/capture/seat and LT1 cell receipts reused. RoPE: Su et al. (2021); MLA: DeepSeek-AI (2024), unverified — lead to check: RoFormer arXiv 2104.09864; DeepSeek-V2 arXiv 2405.04434. No new attention algorithm.')

if __name__ == '__main__':
    create(OUT/'registration.json', build())
    print(sha(OUT/'registration.json'))
