"""Create-once LT1 registration before gates.
Prior art: GRM C7 costed cells and immutable SHA registration (2026), reuse
historical EB1 proxy, 1.25 safety multiplier and reload/IO allowance; new
200-turn paired schedule. No new scheduling or cryptographic algorithm.
"""
from pathlib import Path
from scripts.grm_lt1 import ROOT,OUT,FIX,REG,create,read,sha


def main():
    if REG.exists(): raise ValueError('IMMUTABLE_REGISTRATION_EXISTS')
    f=read(FIX); timing=read(OUT/'timing_source.json')
    base=timing['mean_turn_seconds']*1.25
    oracle=timing['max_probe_seconds']*1.25
    correction=timing['max_correction_seconds']*1.25
    cells=[]
    for arm in ('A','B'):
        previous=None;start=1
        while start<=200:
            stop=min(start+7,70 if start<=70 else 140 if start<=140 else 200)
            cost=60
            for e in f['turns'][start-1:stop]:
                cost+=correction if e['kind']=='correction' else base
                if e['kind']=='probe':cost+=oracle
                # Recap 160 output-token budget vs 32-token point answer:
                # charge five oracle equivalents in addition to turn proxy.
                if e['kind']=='recap':cost+=5*oracle
            # Two read-only sentinel probes before and after each registered
            # restart, one fresh and one correction, same IDs/current values.
            sentinel_calls=2*int(stop in (70,140))+2*int(start in (71,141))
            cost+=sentinel_calls*oracle
            cid=f'{arm}-{start:03d}-{stop:03d}'
            cells.append(dict(id=cid,arm=arm,start=start,stop=stop,depends=previous,
                estimate_seconds=cost,worker_seconds=280,lease_seconds=285,
                outer_seconds=590,cooldown_seconds=30,restart_sentinel_calls=sentinel_calls))
            previous=cid;start=stop+1
    total=sum(c['estimate_seconds'] for c in cells)
    status='NON_FIT' if total>9000 or any(c['estimate_seconds']>280 for c in cells) else 'FIT_ESTIMATE'
    profile=read(ROOT/'config/grm_eb1_profile_registered.json')
    inputs={}
    paths=[p for d in ('core','scripts','config') for p in (ROOT/d).rglob('*') if p.is_file() and p.suffix in ('.py','.json')]
    paths += [ROOT/'tests/test_grm_lt1.py',ROOT/'orders/GRM_LT1_LONG_TURN_POC.md',
        ROOT/'AGENTS.md',FIX,ROOT/'fixtures/lt1/manifest.json',ROOT/'fixtures/lt1/manifest.sha256',
        ROOT/'fixtures/lt1/turn_plan.md',OUT/'lead_commands.txt',OUT/'timing_source.json']
    for p in paths: inputs[str(p.relative_to(ROOT))]=sha(p)
    projection=dict(total_gpu_seconds=total,total_gpu_hours=total/3600,
        per_arm_seconds={a:sum(c['estimate_seconds'] for c in cells if c['arm']==a) for a in ('A','B')},
        maximum_cell_seconds=max(c['estimate_seconds'] for c in cells),
        cooldown_wall_seconds=30*len(cells),cooldown_charged_as_gpu=False,
        budget_seconds=9000,base_turn_seconds=base,correction_turn_seconds=correction,
        oracle_seconds=oracle,reload_io_seconds_per_cell=60,
        evidence_class='reasoning projection from historical EB1 E2E timing; not measured LT1 or B runtime',
        uncertainty='B uses A proxy; natural prose, growing persistence, folding and 160-token recap unmeasured. No trimming or optimistic replacement with short early C7 cells.',
        history_source_sha256=sha(OUT/'timing_source.json'))
    reg=dict(schema='grm.lt1.registration.v1',immutable=True,status=status,
        order='orders/GRM_LT1_LONG_TURN_POC.md',immutable_inputs=inputs,
        fixture_manifest_sha256=sha(ROOT/'fixtures/lt1/manifest.json'),
        cells=cells,projection=projection,model=profile['model'],model_frame=profile['frame'],
        agent_model='GPT-6 (exact deployment model ID not exposed)',
        agent_effort='high requested; inference setting not independently exposed',
        arms={a:dict(flags=dict(profile['default_flags'],**(profile['flag_overrides'] if a=='A' else {})),status=status) for a in ('A','B')},
        turn_mix=read(ROOT/'fixtures/lt1/manifest.json')['turn_mix'],
        predictions=dict(A='Fresh 80-100% at each distance; corrected 50-80%; aliases 0-60%. I do not predict the lead prior will hold for every small stratum. Metadata retention expected; answer retention uncertain; residency <=2*width+recency expected.',
            B='Fresh 50-80%, corrected 20-60%, with worse aggregate recall at 100/150 than 10/25. No assertion that restart alone necessarily erases corrected values; predict at least one corrected-answer failure across restart.',
            oracle='Fresh and corrected >=80% each pooled; literal unknown is scored as abstention without repair.',
            recap='A 3/5, B 2/5; exploratory point predictions, not evidence.'),
        acceptance=dict(complete_turns_per_arm=200,complete_point_recalls_per_arm=35,
            A_fresh_and_corrected_rate_min_each_distance=.80,
            alias='report separately, no post-hoc threshold',
            recap='report 0..5 by current value-span for the five frozen decisions',
            residency='all phases summed mounted tokens <=2*arena_width+actual recency-mounted token seats',
            restart='new PID at 71 and 141; checkpoint payload and full node metadata hashes retained; fresh and corrected sentinel scores unchanged before/after',
            comparison='matched probe IDs and identical frozen user/assistant replay; A/B interleaved by cell; report oracle on both arms',
            small_n='each distance: 3 fresh, 2 correction, 2 alias; 80% means all 3 fresh and both corrected must pass',
            empty='NOT_RUN or INCOMPLETE, never PASS'),
        fixture_protocol=f['protocol'],
        restart_sentinels=[next(p['id'] for p in f['probes'] if p['class']==c) for c in ('fresh','correction')],
        restart_caveat='All 26 cells/arm require fresh process reloads; turns 70 and 140 are additional designated scored boundaries, not the only process restarts.',
        cpu_gates=['test_registration_and_frozen_fixture','test_recency_gate_rejects_recent_source',
            'test_instruction_collision_and_missing_oracle_are_rejected','test_value_span_and_disjoint_error_categories',
            'test_fake_attention_matches_actual_constructor','test_visible_reader_requires_sources_and_current_revision',
            'test_full_fixture_fake_mounts_answers_and_restarts[A]','test_full_fixture_fake_mounts_answers_and_restarts[B]',
            'test_c7_checkpoint_roundtrip_and_corruption','test_empty_summary_and_budget_fail_closed'],
        mutations=dict(only_after_passing_baseline=True,min_kill_rate=.80,
            planned=['source_age_9','recency_nominee_source','uppercase_unknown','missing_oracle_source','unordered_span']),
        stop_rule='Any RED: preserve evidence, no retry or threshold weakening. NON_FIT: no GPU, no arm subset, no shortened script; successor requires separately authorized registration. CPU preparation/gates remain authorized.',
        process_safety='No GPU in sandbox; no git command, no subagents, no background waits, no killing. Foreground CPU subprocesses only. C7 subprocess timeout kill is not reused under LT1 never-kill rule.',
        launch_status='BLOCKED at registered NON_FIT rail if projected over budget; prospective cell list retained. No executable GPU worker authorized under NON_FIT.',
        prior_art='GRM contributors (2026): C7/C2/EB1 cells, SHA bindings, actual serving ladder, a3 oracle and attention-surface fake. LT1 adds invented prose, plan-derived recency gate and C5 specified ordered span scorer. Unicode NFKC UAX15 (Unicode Consortium, 2001), unverified — lead to check: Unicode UAX15 normalization history. No prior art known to me for this exact dialogue composition.')
    create(REG,reg)
    with REG.with_suffix('.sha256').open('x') as stream:stream.write(sha(REG)+'  registration.json\n')
    with (OUT/'cell_list.tsv').open('x') as stream:
        stream.write('id\tdepends\testimate_seconds\tworker_seconds\tlease_seconds\touter_seconds\tcooldown_seconds\n')
        for c in cells:stream.write('\t'.join(str(c[k]) for k in ['id','depends','estimate_seconds','worker_seconds','lease_seconds','outer_seconds','cooldown_seconds'])+'\n')
    print(dict(status=status,sha256=sha(REG),projection=projection))
if __name__=='__main__':main()
