#!/usr/bin/env python3
"""Receipt-only audit. Prior art: GRM RS4 (2026) recorded layer means and
capture-node provenance; Qwen Team (2026) tokenizer/template. Ours: joins,
source-membership audit and token scaffold counts, no new attention method.
Upstream unverified — lead to check: Qwen3.5-9B tokenizer chat_template.
"""
import json
import re
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.grm_xm1_registration import read,sha,create

def build():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(ROOT/'artifacts/grm_xm2/sources'),local_files_only=True)
    reg=read(ROOT/'artifacts/grm_xm1/registration.json')
    rows=[]
    identifiers={'sup_harbor_restatement':'Harbor token','sup_praxis_fresh':'Praxis dock',
      'sup_solace_fresh':'Solace key','sup_reserve_tundra_ledger':'Tundra ledger',
      'sup_reserve_meridian_docket':'Meridian docket'}
    for pid,p in reg['probes'].items():
        paths={a:ROOT/f'artifacts/grm_xm1/amendment_1_run/cells/gpu/qwen35/{a}__{pid}.json' for a in ('C3l','C5')}
        result={a:read(path)['result'] for a,path in paths.items()}
        m,f=result['C3l'],result['C5']
        reference=reg['rs4_references']['C3l:'+pid]
        histpath=ROOT/'artifacts/grm_rs4'/reference['receipt']
        hist=next(x for x in read(histpath)['probes'] if x['probe_id']==pid and x['variant']=='registered')
        nodes=hist['info']['rs4_capture']['per_node']
        layers=[dict(layer_index=m['observed_rows'][0][i]['layer_index'],mounted=a['mounted_mass'],fed=b['fed_text_mass'],
                     fed_minus_mount=b['fed_text_mass']-a['mounted_mass'])
                for i,(a,b) in enumerate(zip(m['per_layer']['full_attention']['per_layer_mean'],f['per_layer']['full_attention']['per_layer_mean']))]
        tokenized={}
        for arm,r in result.items():
            actual=[i for t in r['rendered_texts'] for i in tok.encode(t,add_special_tokens=False)]
            if actual!=r['source_token_ids']: raise ValueError('tokenizer differs from receipt '+pid+arm)
            text=''.join(r['rendered_texts']); name=identifiers[pid]
            tokenized[arm]=dict(ntok=len(actual),n_turns=len(r['rendered_texts']),
                source_ids_match=True,identifier_mentions=text.count(name),
                expected_mentions={v:text.casefold().count(v.casefold()) for v in p['expected_values']},
                think_pairs=text.count('<think>'),
                marker_ntok=sum(t in {tok.encode(m,add_special_tokens=False)[0] for m in ('<|im_start|>','<|im_end|>','<think>','</think>')} for t in actual),
                token_ids=actual,token_pieces=tok.convert_ids_to_tokens(actual),
                rendered_texts=r['rendered_texts'])
        name_ids=tok.encode(' '+identifiers[pid],add_special_tokens=False)
        matched=re.search(re.escape(p['expected_values'][0]), ''.join(m['rendered_texts']+f['rendered_texts']), re.I)
        value_text=matched.group(0) if matched else p['expected_values'][0]
        value_ids=tok.encode(' '+value_text,add_special_tokens=False)
        rows.append(dict(probe=pid,receipt_pins={str(path.relative_to(ROOT)):sha(path) for path in paths.values()},
            layers=layers,means=dict(mounted=sum(x['mounted'] for x in layers)/8,
              fed=sum(x['fed'] for x in layers)/8,gap=sum(x['fed_minus_mount'] for x in layers)/8),
            qwen_geometry=dict(**m['seating'],capture_pin='live',capture_shift=m['seating']['live_shift'],
              geometry_evidence='capture shift inferred from pinned amendment-1 native_cell; seating measured in receipt',
              node_identity='synthetic Qwen payload from RS4 capture_texts; no native Qwen repository node ID'),
            historical_nodes=[dict(node_id=n['node_id'],installed_graft_id=n['installed_graft_id'],
              fresh_graft_id=n['fresh_graft_id'],fresh_native_node_id=n['fresh_native_node_id'],
              source=n['capture_text_source'],capture_receipt=n['capture_receipt'],
              installed_ntok=n['installed_payload']['ntok'],recaptured_ntok=n['recaptured_payload']['ntok']) for n in nodes],
            historical_receipt=str(histpath.relative_to(ROOT)),tokenizer=tokenized,
            identifier=dict(text=identifiers[pid],ids=name_ids,pieces=tok.convert_ids_to_tokens(name_ids)),
            value=dict(text=value_text,ids=value_ids,pieces=tok.convert_ids_to_tokens(value_ids)),
            old_final_token_count=0 if not any(x==248069 for x in m['generated_token_ids']) else None,
            old_served={a:r['served'] for a,r in result.items()},
            old_correct={a:r['value_span']['correct'] for a,r in result.items()}))
    return dict(evidence_class='copied GPU receipt audit + local CPU tokenizer replay; no model run',
        tokenizer_revision='c202236235762e1c871ad0ccb60c8ee5ba337b9a',rows=rows,
        caveat='Historical native source contrasts differ. Praxis C5 lacks expected fact; Solace C3l mounts Tundra child. Not a same-fact parity test.')

if __name__=='__main__':
    create(ROOT/'artifacts/grm_xm2/diagnosis.json',build())
