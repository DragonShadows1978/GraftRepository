#!/usr/bin/env python3
"""CPU-only historical receipt audit (no model load).

Prior art: GRM RS4 row partitions and XM1 native_text (contributors, 2026),
Arcee AI Trinity template (local snapshot; upstream unverified — lead to check
arcee-ai/Trinity-Nano-Preview chat_template.jinja). Ours: receipt diagnosis.
"""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import grm_xm1_parity as xm


def audit(tokenizer):
    base = xm.read(xm.REG)
    rows = []
    bands = ('sink_band', 'mount_band', 'prior_live_band', 'fed_text_band', 'question_band', 'answer_band')
    for path in sorted((ROOT / 'artifacts/grm_xm1/amendment_1_run/cells/gpu/trinity').glob('*.json')):
        r = xm.read(path)['result']
        arm, probe = path.stem.split('__')
        expected = base['probes'][probe]['expected_values']
        q = tokenizer.apply_chat_template([{'role':'user','content':base['probes'][probe]['question']}],
                                          tokenize=False, add_generation_prompt=True)
        rendered = [xm.native_text(tokenizer, text) for text in r['source_texts']]
        ids = [i for text in rendered for i in tokenizer.encode(text, add_special_tokens=False)]
        raw = tokenizer.decode(r['generated_token_ids'], skip_special_tokens=False)
        observed = [row for step in r['observed_rows'] for row in step]
        errors = []
        max_sum_error = max_live_error = 0.0
        for i, row in enumerate(observed):
            b = row['bounds']
            if sum(b[k][1] - b[k][0] for k in bands) != b['S']:
                errors.append([i,'width'])
            if not xm.partition_sums_to_one(row) or not xm.subbands_sum_to_live(row):
                errors.append([i,'probability'])
            total = row['mounted_mass'] + row['physical_sink_mass'] + row['learned_sink_mass'] + sum(row[k] for k in ('prior_live_mass','fed_text_mass','question_mass','answer_mass'))
            max_sum_error = max(max_sum_error, abs(total - 1))
            max_live_error = max(max_live_error, abs(sum(row[k] for k in ('prior_live_mass','fed_text_mass','question_mass','answer_mass')) - row['live_mass']))
        rows.append(dict(receipt=str(path.relative_to(ROOT)), arm=arm, probe=probe,
            served=r['served'], raw_generated_text=raw, generated_token_ids=r['generated_token_ids'],
            value_span=r['value_span'], expected_values=expected,
            expected_in_source={e:xm.normalize_value(e) in xm.normalize_value('\n'.join(r['source_texts'])) for e in expected},
            source_tokens=len(ids), question_tokens=len(r['question_token_ids']),
            decode_steps=len(r['generated_token_ids']), seating=r['seating'],
            source_ids_match=ids==r['source_token_ids'], rendered_match=rendered==r['rendered_texts'],
            question_ids_match=tokenizer.encode(q,add_special_tokens=False)==r['question_token_ids'],
            question_text=q, skip_special_decode_matches=tokenizer.decode(r['generated_token_ids'],skip_special_tokens=True)==r['served'],
            layer_rows=len(observed), S_min=min(x['bounds']['S'] for x in observed),
            S_max=max(x['bounds']['S'] for x in observed), band_errors=errors,
            max_partition_error=max_sum_error, max_subband_error=max_live_error,
            full_mass=r['per_layer']['full_attention']['mean_over_answer_positions']['mounted_mass' if arm=='C3l' else 'fed_text_mass']))
    return dict(evidence_class='CPU historical receipt/tokenizer audit, no model execution',
                cells=rows, total_layer_rows=sum(r['layer_rows'] for r in rows),
                template_source='artifacts/grm_xm3/evidence/chat_template.jinja',
                upstream='https://huggingface.co/arcee-ai/Trinity-Nano-Preview/blob/main/chat_template.jinja',
                upstream_status='unverified — lead to check; web open rejected', gpu_executed=False)


if __name__ == '__main__':
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('/mnt/ForgeRealm/models/trinity-nano', local_files_only=True, trust_remote_code=True)
    result = audit(tok)
    xm.create(ROOT / 'artifacts/grm_xm3/historical_audit_amendment_1.json', result)
    print(json.dumps({'cells':len(result['cells']),'layer_rows':result['total_layer_rows'],
                      'band_errors':sum(len(r['band_errors']) for r in result['cells']),
                      'template_ids_match':all(r['source_ids_match'] and r['question_ids_match'] and r['rendered_match'] for r in result['cells'])}))
