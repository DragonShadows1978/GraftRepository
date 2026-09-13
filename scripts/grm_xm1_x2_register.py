#!/usr/bin/env python3
"""Create-only XM2 registration, run before gates.
Prior art: GRM XM1/LT1 (2026) immutable hashes and factorial RS3 arms (2026).
Taken: source pins, 2x2 capture/seating controls, fixed falsifiers. Ours:
Qwen final-query window and historical source-confound registration.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.grm_xm1_registration import read,sha,create
OUT=ROOT/'artifacts/grm_xm2'

def build():
    parent=read(ROOT/'artifacts/grm_xm1/registration.json')
    cells=[]
    for probe in parent['probes']:
        for arm in ('C3l','C5'):
            cells.append(dict(id=arm+'__'+probe,model='qwen35',arm=arm,probe=probe,
              capture_pin='live' if arm=='C3l' else None,seat_near_live=arm=='C3l',reservation_s=110))
    for probe in ('sup_praxis_fresh','sup_solace_fresh'):
        for pin,near in (('off',False),('live',False),('off',True)):
            cells.append(dict(id=f'C3l_pin_{pin}_seat_{int(near)}__{probe}',
              model='qwen35',arm='C3l',probe=probe,capture_pin=pin,seat_near_live=near,reservation_s=110))
    hypotheses=[
      dict(id='H1',hypothesis='Thinking-window contamination accounts for the registered fresh-probe mass deficits.',
           falsifier='With nonempty EOS-complete final windows, either fresh C5-minus-C3l mean still >0.10; incomplete windows are RED/unevaluable, not parity.',
           evidence_before_run='Both original windows contain only thinking; causal effect unmeasured.'),
      dict(id='H2',hypothesis='A different capture pin on fresh versus comparator nodes explains the observed split.',
           falsifier='Pinned native code and per-node receipts show the same live pin for all five.',
           evidence_before_run='Falsified as a differential-pin attribution: Qwen inferred live shifts 279/280/292; historical RS4 recaptures all live/115.',
           lever_prediction='At fixed seating, off versus live recapture changes fresh mounted mean by >=0.05 on at least one probe.',
           lever_falsifier='Both fresh probe changes <0.05 at both seating settings with complete final windows.'),
      dict(id='H3',hypothesis='Moving the mounted head adjacent to live improves fresh-probe attention.',
           falsifier='For each capture pin, seat-on minus seat-off <0.05 on both fresh probes; empty/incomplete windows unevaluable.',
           evidence_before_run='Existing Qwen receipts are already seat-on; no seat-off treatment observed.'),
      dict(id='H4',hypothesis='The historical panel compares unequal fact/scaffold content, confounding a same-fact parity interpretation.',
           falsifier='Local tokenizer/source replay finds Quartz-8-Jade in Praxis C5 and Raven-9-Ivory in Solace C3l, with matched source sequences.',
           evidence_before_run='Source audit confirms missing facts in those arms; causal contribution NOT measured.',
           successor='A same-text, same-fact, matched-length experiment would test causal attribution; not in the 16 registered cells.'),
      dict(id='H5',hypothesis='Identifier fragmentation alone separates the failing fresh probes from comparators.',
           falsifier='A failing identifier is one name token like Harbor, or successful comparators are split into multiple name tokens.',
           evidence_before_run='Falsified as a sole separator: Praxis is one name token; Solace/Tundra/Meridian are split. Other lexical/scaffold effects remain untested.')]
    pins={
      'scripts/grm_xm1_parity.py','scripts/grm_xm1_cpu.py',
      'tests/test_grm_xm1_amendment_1.py','tests/test_grm_xm2_final.py',
      'artifacts/grm_xm2/IMPLEMENTATION_PLAN.md','artifacts/grm_xm2/diagnosis.json',
      'artifacts/grm_xm2/recorded_trace_cpu.json','artifacts/grm_xm2/lead_commands.txt'}
    pins.update(str(p.relative_to(ROOT)) for p in (ROOT/'scripts').glob('grm_xm1_x2_*.py'))
    pins.add('scripts/grm_xm1_qwen_final.py')
    pins.update(str(p.relative_to(ROOT)) for p in (OUT/'sources').glob('*'))
    pins.update(str(p.relative_to(ROOT)) for model in ('qwen35','gpt-oss') for p in (ROOT/f'artifacts/grm_xm1/amendment_1_run/cells/gpu/{model}').glob('*.json'))
    return dict(schema='grm.xm2.registration.v1',immutable=True,
      xm1_registration_sha256=sha(ROOT/'artifacts/grm_xm1/registration.json'),
      xm1_amendment_sha256=sha(ROOT/'artifacts/grm_xm1/registration_amendment_1.json'),
      model_id='Qwen/Qwen3.5-9B',model_revision='c202236235762e1c871ad0ccb60c8ee5ba337b9a',
      seat_model='gpt-6-astra',seat_effort='high',gpu_budget_s=1800,total_reserved_s=1760,
      flag={'GRM_QWEN35_FINAL_CHANNEL':'1'},max_generated_tokens=32,
      answer_window='Final content input-query tokens only: generated index i -> observer index i+1; omit prefill, thinking, delimiters, EOS, leading separator whitespace.',
      completion_rail='Require nonempty EOS-complete final answer per cell; NO_FINAL or TRUNCATED_FINAL is RED. Never extend token or time cap after seeing results.',
      cells=cells,hypotheses=hypotheses,
      geometry='width=max(capture native ntok,feed native ntok)+96; capture live=width/off=0 (cleared fresh native state); near seat=[width-ntok,width), off seat=[0,ntok); live queries start at width. No physical sink. No DeltaNet state transfer.',
      geometry_by_probe={r['probe']:r['qwen_geometry'] for r in read(OUT/'diagnosis.json')['rows']},
      execution='Register only in this seat. Worker leases itself fail-fast with wait=0; no outer flock. 110 s each, charge full reservation before load; 16 unique cells; no retry. 30 s minimum gap checked without waiting. Resume exact binding only.',
      gates=['Pins and original GPT-OSS reference barrier pass before dispatch/dry-run.',
        'CPU recorded trace final-window selection, OFF serialized identity on 10 cells, 4 RS3 geometries, flags/boundaries, source diagnosis.',
        'All 16 lead commands accept --dry-run, no GPU import or lease.',
        'Final command verbatim: python3 -m pytest -q --basetemp artifacts/grm_xm2/tmp tests/test_grm_xm1*.py tests/test_grm_xm2*.py'],
      pins={p:sha(ROOT/p) for p in sorted(pins)},
      prior_art='Qwen Team (2026), cached chat_template.jinja revision c202236235762e1c871ad0ccb60c8ee5ba337b9a; enable_thinking=False empty block reused. Upstream unverified — lead to check: Qwen3.5-9B chat template thinking mode. GRM contributors (2026), RS3 capture/seat, RS4 row means, XM1/LT1 binding and CMC1 lease reused. Ours: final input-query indices and source audit; no new attention algorithm.',
      deviations=['Source sets retained despite missing Praxis C5 fact and wrong-fact Solace C3l mount; not a same-fact parity claim.',
        'Final-query window intentionally changes predictor-row convention in legacy XM1; raw observations retained.',
        'Controlled capture off=0 approximates fresh native legacy state; not arbitrary residual production-arena live_shift.',
        '32-token direct-final cap retained for budget. EOS incomplete is RED, no automatic cap increase.',
        'CPU doubles are author baseline only, not blind verification, Qwen quality or GPU numeric parity.'])

if __name__=='__main__':
    create(OUT/'registration.json',build())
    print(sha(OUT/'registration.json'))
