"""Create-once amendment; original LT1 registration stays byte-identical.
Prior art: C7/C2 SHA amendments (GRM contributors, 2026), borrowed unchanged;
new input scope and lead-authorized 3h cap, no new registration algorithm.
"""
from scripts import grm_lt1 as lt


def main():
    original=lt.read(lt.REG);out=lt.OUT/'amendment1';overrides={}
    for name,digest in original['immutable_inputs'].items():
        if lt.sha(lt.ROOT/name)!=digest:
            before=out/'before'/name
            if not before.exists() or lt.sha(before)!=digest:raise ValueError('UNARCHIVED_CHANGE '+name)
            if name.startswith('core/'):raise ValueError('CORE_CHANGE_FORBIDDEN')
            overrides[name]=dict(before_sha256=digest,after_sha256=lt.sha(lt.ROOT/name),before_archive=str(before.relative_to(lt.ROOT)))
    names=['scripts/grm_lt1_admission.py','scripts/grm_lt1_offline.py','scripts/grm_lt1_worker.py',
        'scripts/grm_lt1_worker_cpu.py','scripts/grm_lt1_register_amendment1.py','tests/test_grm_lt1_amendment1.py',
        'orders/GRM_LT1_AMENDMENT_1.md','artifacts/grm_lt1/amendment1/PLAN.md']
    lt.create(lt.AMEND,dict(schema='grm.lt1.amendment1.v1',registration_sha256=lt.sha(lt.REG),
        order='orders/GRM_LT1_AMENDMENT_1.md',overrides=overrides,new_inputs={n:lt.sha(lt.ROOT/n) for n in names},
        admission_rule=0,status='RUNNABLE_RULE0_PENDING_CPU_GATES',budget_gpu_seconds=10800,
        projection_gpu_seconds=original['projection']['total_gpu_seconds'],cells=original['cells'],
        acceptance=original['acceptance'],predictions=original['predictions'],
        repeat_policy='If David changes admission, separately register new rule/core SHA and repeat all cells in a fresh namespace; never mix rules.',
        cpu_gate_policy='Author infrastructure gates must pass. Known Rule0 admission refusals remain scored quality RED, explicitly authorized measurement.',
        gates='tests/test_grm_lt1_amendment1.py; offline Rule0 real-core parity; original r1 quality failure remains archived, not weakened',
        fixture_manifest_sha256=lt.sha(lt.ROOT/'fixtures/lt1/manifest.json'),fixture_change='Only 60 unscored fact-turn assistant acknowledgments; all other fixture fields identical',
        deadline_contract='280s cooperative worker alarm, 285s measured lease rail, 590s outer with 30s cooldown; no kill. Native blocking can defer signal delivery; hard wall ceiling not guaranteed. Hold lease until child exits and mark overruns RED.',
        agent_model='GPT-6 (exact deployment model ID not exposed)',agent_effort='high requested; inference setting not independently exposed',
        prior_art='GRM C7/C2/EB1/A-DEC/RT1 (contributors, 2026); checkpoint/recovery concept ARIES (Mohan et al.,1992), lexical retrieval IIR (Manning et al.,2008), unverified — lead to check titles/authors. No prior art known to me for exact counterfactual composition.'))
    lt.AMEND.with_suffix('.sha256').write_text(lt.sha(lt.AMEND)+'  registration_amendment.json\n')
    print(lt.sha(lt.AMEND))
if __name__=='__main__':main()
