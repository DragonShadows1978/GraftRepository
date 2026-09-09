"""C7 author CPU baseline, not blind review.
Prior art: local C2 input guards and checkpoint tests (GRM, 2026). C7 adds
independent hand answers and missing/forged evidence cases. No prior art known
to me for this exact fixture; adversarial testing is established practice.
"""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from scripts import grm_c7_common as c
from scripts import grm_c7_run as run


def test_fixture_integrity_and_probe_placement():
    r = c.verify()
    f = c.read(c.FIX)
    assert len(f['turns']) == 300
    assert [t['turn'] for t in f['turns']] == list(range(1,301))
    assert len(f['entities']) == len(set(f['entities'])) >= 40
    assert len(f['aliases']) >= 12 and len(f['corrections']) >= 12
    assert sum(x['earlier_correction'] for x in f['corrections']) >= 2
    assert len(f['probes']) >= 60
    assert len({p['turn'] for p in f['probes']}) == len(f['probes'])
    for p in f['probes']:
        assert p['turn']-p['source_turn'] == p['distance']
        assert max(p['source_turns']) == p['source_turn']
        assert p['oracle_source_texts'] == [f['turns'][t-1]['user'] for t in p['source_turns']]
        assert all(f['turns'][t-1]['kind'] != 'probe' for t in p['source_turns'])
        assert f['turns'][p['turn']-1]['probe_id'] == p['id']
        if p['answerable']:
            assert all(value in '\n'.join(p['oracle_source_texts']) for value in p['expected'].split('|'))
            assert p['expected'] not in p['question']
        else:
            assert 'inspection password' not in '\n'.join(p['oracle_source_texts'])
    for distance in c.DISTANCES:
        for cls in c.CLASSES:
            ps = [p for p in f['probes'] if p['distance']==distance and p['class']==cls]
            assert len(ps)==3 and sum(p['answerable'] for p in ps)==2
    assert f['restart_after_turns'] == [100,200]
    assert f['pressure_after_turns'] == [60,140,240]
    assert r['acceptance']['coverage'] == .70


@pytest.mark.parametrize('answer,required,expected,errors', [
    ('Onyx-911', True, 'Onyx-911', (0,0,0,0)),
    ('**Onyx\u2011911**.', True, 'Onyx-911', (0,0,0,0)),
    ('Onyx\u2011911.', True, 'Onyx-911', (0,0,0,0)),
    ('Flint-511', True, 'Onyx-911', (1,1,0,0)),
    ('Flint-511 or Onyx-911', True, 'Onyx-911', (1,1,0,0)),
    ('Not Onyx-911', True, 'Onyx-911', (1,1,0,0)),
    ('UNKNOWN', True, 'Onyx-911', (1,0,1,0)),
    ("I don't have that information.", True, 'Onyx-911', (1,0,1,0)),
    ('UNKNOWN', False, 'UNKNOWN', (0,0,0,0)),
    ('Onyx-911', False, 'UNKNOWN', (1,0,0,1)),
    ('Reed-611 | Cedar-1011', True, 'Reed-611|Cedar-1011', (0,0,0,0)),
    ('Cedar-1011|Reed-611', True, 'Reed-611|Cedar-1011', (1,1,0,0)),
])
def test_oracle_scorer_hand_cases(answer, required, expected, errors):
    got = c.score(answer, {'answerable': required, 'expected': expected})
    assert tuple(got[k] for k in ('exact_answer_error','wrong_value_error','abstention_error','unsupported_answer_error')) == errors


def test_cell_schedule_cost_and_defaults():
    r = c.verify()
    for arm in ('A','B'):
        cells = [x for x in r['cells'] if x['arm']==arm]
        assert [i for cell in cells for i in range(cell['start'],cell['stop']+1)] == list(range(1,301))
        assert all(x['estimate_seconds'] <= 280 < 285 and x['outer_seconds']==590 for x in cells)
        assert {100,200} <= {x['stop'] for x in cells}
        assert abs(sum(x['estimate_seconds'] for x in cells)-r['arms'][arm]['estimate_seconds']) < 1e-5
        assert all(x['depends']==(cells[i-1]['id'] if i else None) for i,x in enumerate(cells))
    assert r['arms']['A']['estimate_seconds'] <= 7200
    assert r['arms']['A']['flags']['arena_width']==96
    assert r['arms']['A']['flags']['capture_pin']=='live'
    assert r['arms']['B']['flags']['arena_width']==256
    assert r['arms']['B']['flags']['capture_pin']=='off'
    assert not r['arms']['B']['flags']['seat_near_live']
    assert all(not x['flags']['demand_ngh'] for x in r['arms'].values())
    cell = r['cells'][0]
    run.reserve_check(r,cell,6920,0)
    with pytest.raises(ValueError,match='BUDGET_RAIL'):
        run.reserve_check(r,cell,6920.001,0)
    with pytest.raises(ValueError,match='BUDGET_RAIL'):
        run.reserve_check(r,cell,6650,280)
    with pytest.raises(ValueError,match='NON_FIT'):
        run.reserve_check(r,next(x for x in r['cells'] if x['arm']=='B'),0,0)


def test_synthetic_cell_boundary_new_process_and_corruption(tmp_path):
    r = c.verify()
    next_turn = r['cells'][0]['stop']+1
    state = {'next_turn': next_turn, 'process_id': 'synthetic-writer',
             'writer_pid': os.getpid(), 'turn_records': {'1': {'chat_node_id': 0}},
             'capture': {'capture_pin':'live','capture_shift':115},
             'scores': {'c7_fresh_0_d005': {'exact_answer_error':0,'abstention_error':0}},
             'cold_nodes':[0], 'fold_events':[{'accepted':False,'best_cov':.69}]}
    c.create(tmp_path/'repository/manifest.json', {'nodes':[{'ntok':37,'sources':[],
                  'metadata': {'revision':2}, 'capture': state['capture']}]})
    (tmp_path/'repository/0000.synthetic').write_bytes(b'CPU synthetic payload; not model K/V')
    c.checkpoint(tmp_path,state,run.binding('A'))
    # New executable process reads identical next-turn, scores, provenance,
    # fold and cold state. No transcript refeed, model import or CUDA execution.
    child = subprocess.run([sys.executable,str(c.ROOT/'scripts/grm_c7_run.py'),
        '--synthetic-resume',str(tmp_path),'--next-turn',str(next_turn)],
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,check=True)
    got = json.loads(child.stdout)
    assert got['reader_pid'] != state['writer_pid']
    assert got['state'] == state
    with pytest.raises(ValueError,match='CHECKPOINT'):
        c.validate_checkpoint(tmp_path,next_turn+1,run.binding('A'))
    with pytest.raises(ValueError,match='CHECKPOINT'):
        c.validate_checkpoint(tmp_path,next_turn,run.binding('B'))
    (tmp_path/'repository/0000.synthetic').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='CHECKPOINT'):
        c.validate_checkpoint(tmp_path,next_turn,run.binding('A'))


def test_folded_receipt_requires_parent_without_raw_and_correct_lineage():
    nodes = [{'retired':True}, {'retired':True},
             {'kind':'digest','sources':[0,1],'retired':True},
             {'kind':'era','sources':[2]}]
    assert c.folded_evidence(nodes,[3],[0,1])
    assert not c.folded_evidence(nodes,[3,0],[0,1])
    assert not c.folded_evidence(nodes,[0,1],[0,1])
    assert not c.folded_evidence(nodes,[3],[])
    assert not c.folded_evidence(nodes,[3],[4])
    nodes[0]['retired'] = False
    assert not c.folded_evidence(nodes,[3],[0,1])


def test_missing_rows_are_unknown_and_duplicate_probe_rejected():
    f = c.read(c.FIX)
    tab = c.table([],f['probes'])
    assert len(tab)==20 and all(not x['complete'] and x['memory']['exact_answer_error_rate'] is None for x in tab)
    p = f['probes'][0]
    row = {'probe_id':p['id'],'memory':{'answer':p['expected']},'oracle':{'answer':p['expected']}}
    with pytest.raises(ValueError,match='DUPLICATE'):
        c.table([row,row],f['probes'])
    # Prior art: C7 campaign-state guards (GRM, 2026); FIX-4 imports
    # old receipts under explicit quarantine instead of claiming an unrun arm.
    from scripts.grm_c7_fix4 import enabled
    assert run.summary('A')['status'] == ('NOT_MEASURED' if enabled() else 'INCOMPLETE')


def test_oracle_uses_exact_source_and_cannot_deposit():
    from types import SimpleNamespace
    class Arena:
        caches='saved'; pos=11; live_segs=[]; cur_mounts=[4]; cur_mount_n=31
        stop_sequences=[]
        # Prior art: EB1 layer/template contract (GRM, 2026). Complete this
        # existing double's interface for the r2 oracle; keep all old pins.
        live_shift=115
        m=SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace(live_shift=None))])
        def _format_step_prompt(self, text):
            return 'wrapped: ' + text
        def reset_live_cache(self):
            self.caches=None; self.pos=0; self.cur_mounts=[]; self.cur_mount_n=0; self.live_segs=[]
        def encode(self,text):
            return list(text.encode())
        def _attempt(self,prompt,picks,ngen,deposit,stops,defer_memory):
            assert picks==[] and not deposit and defer_memory
            assert '[turn 2] Old code Flint-511.' in prompt
            assert '[turn 17] Correction: Onyx-911.' in prompt
            return 'Onyx-911',{}
    p = {'source_turns':[2,17], 'oracle_source_texts':['Old code Flint-511.','Correction: Onyx-911.'],
         'question':'Current code?', 'expected':'Onyx-911','answerable':True}
    a=Arena()
    value=run.oracle(a,p,32)
    assert value['score']['exact_correct'] and value['mounted_ids']==[]
    assert a.caches=='saved' and a.cur_mounts==[4]
    assert value['prompt_token_ids'] == a.encode('wrapped: ' + value['live_prompt'])
    assert a.m.layers[0].self_attn.live_shift is None


def test_residency_counts_token_rows_and_actual_recency_only():
    class Arena:
        cur_mounts=[0,1]; cur_mount_n=217; width=96; live_segs=[]
        grafts=[{'ntok':120},{'ntok':97},{'ntok':900}]
        _eb1_recency_mounted_ids=[1,2]
    row=run.seats(Arena())
    assert row['summed_token_seats']==217
    assert row['actual_recency_token_seats']==97
    assert row['summed_token_seats'] <= 192+row['actual_recency_token_seats']
