"""Prior art: LT1/C7 SHA-bound immutable amendments (GRM, 2026).
Borrow before archives and chain checks; core scope now authorized by FIX6.
No new registration algorithm; no prior art known for exact manifest composition.
"""
from pathlib import Path
import shutil
from scripts import grm_lt1 as lt

def main():
    out=lt.OUT/'amendment2';out.mkdir(exist_ok=True)
    assert not lt.AMEND2.exists()
    original=lt.read(lt.REG);a=lt.read(lt.AMEND);inputs=dict(original['immutable_inputs'])
    inputs.update({n:c['after_sha256'] for n,c in a['overrides'].items()});inputs.update(a['new_inputs'])
    overrides={}
    for name,digest in inputs.items():
        if lt.sha(lt.ROOT/name)!=digest:
            before=lt.ROOT/'artifacts/grm_scout_fix6/before'/name
            assert lt.sha(before)==digest,name
            overrides[name]=dict(before_sha256=digest,after_sha256=lt.sha(lt.ROOT/name),before_archive=str(before.relative_to(lt.ROOT)))
    names=['orders/GRM_SCOUT_FIX_6_ADMISSION_RULE.md','core/grm_three_pass.py','tests/test_grm_lsr_p2b_route_receipt.py',
           'tests/test_grm_scout_fix6.py','tests/test_grm_lt1_fix6.py','tests/test_grm_lt1_controller.py',
           'scripts/grm_scout_fix6_replay.py','scripts/grm_lt1_register_amendment2.py',
           'artifacts/grm_scout_fix6/registration.json','artifacts/grm_scout_fix6/replay_amendment.json',
           'artifacts/grm_scout_fix6/c2_replay_cells.json','artifacts/grm_scout_fix6/c2_unresolved.json']
    lt.create(lt.AMEND2,dict(schema='grm.lt1.amendment2.v1',registration_sha256=lt.sha(lt.REG),previous_amendment_sha256=lt.sha(lt.AMEND),
        order='orders/GRM_SCOUT_FIX_6_ADMISSION_RULE.md',overrides=overrides,new_inputs={n:lt.sha(lt.ROOT/n) for n in names if n not in inputs},
        admission_rule='margin_first',arms={k:dict(v,admission_rule='margin_first') for k,v in original['arms'].items()},
        status='REGISTERED_PENDING_CPU_GATES_FIX4_PREREQUISITE_RED',budget_gpu_seconds=10800,projection_gpu_seconds=original['projection']['total_gpu_seconds'],cells=original['cells'],
        fixture_manifest_sha256=lt.sha(lt.ROOT/'fixtures/lt1/manifest.json'),fixture_change='none',acceptance=original['acceptance'],predictions=original['predictions'],
        run_namespace=str(lt.RUN.relative_to(lt.ROOT)),free_space_minimum_bytes=20000000000,
        gate_policy='FIX6 core, C2 OFF132 + ON112/20 unresolved +31 registered known changes, LT1 CPU 70 profiles and both 200-turn fake worker arms, controller contracts. FIX4 fixtures missing implementation in this checkout: report RED and do not claim FIX1..4 pass.',
        gpu_policy='No GPU in dispatched seat. Lead reviews FIX4 prerequisite RED before using executable lead_commands. Worker pins flag after C2 environment scrub; all resumed bindings include amendment2 SHA and rule.',
        prior_art=__doc__,agent_model='GPT-6; exact deployment identifier not exposed',agent_effort='high requested; actual setting not independently exposed'))
    with lt.AMEND2.with_suffix('.sha256').open('x') as f:f.write(lt.sha(lt.AMEND2)+'  registration_amendment.json\n')
    print(lt.sha(lt.AMEND2))
if __name__=='__main__':main()
