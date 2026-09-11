"""Create-once source-state amendment, before CPU gates.
Prior art: GRM contributors, LT1/C7 immutable SHA-chain registration (2026),
verified local source. Reuse hashes and explicit scopes; no new algorithm.
"""
from scripts import grm_lt1 as lt
from scripts.grm_lt1_amendment3 import ORDER, SOURCE_CHANGES, HARNESS, NEW_INPUTS, PROTOCOL, PREVIOUS_SHA


def main():
    assert not lt.AMEND3.exists()
    r = lt.read(lt.REG)
    inputs = dict(r['immutable_inputs'])
    for path in (lt.AMEND, lt.AMEND2):
        a = lt.read(path)
        assert lt.sha(path) == path.with_suffix('.sha256').read_text().split()[0]
        inputs.update({n:c['after_sha256'] for n,c in a['overrides'].items()})
        inputs.update(a['new_inputs'])
    overrides = {}
    cores = {n:dict(before_sha256=d, after_sha256=lt.sha(lt.ROOT/n))
             for n,d in inputs.items() if n.startswith('core/')}
    for n, old in inputs.items():
        new = lt.sha(lt.ROOT/n)
        if old == new:
            continue
        assert n in SOURCE_CHANGES or n in HARNESS, n
        c = dict(before_sha256=old, after_sha256=new)
        if n in SOURCE_CHANGES:
            assert (old, new) == SOURCE_CHANGES[n], n
        else:
            archive = 'artifacts/grm_lt1/amendment3/before/' + n
            assert lt.sha(lt.ROOT/archive) == old, n
            c['before_archive'] = archive
        overrides[n] = c
    assert set(overrides) == set(SOURCE_CHANGES) | HARNESS
    lt.create(lt.AMEND3, dict(schema='grm.lt1.amendment3.v1',
        registration_sha256=lt.sha(lt.REG), previous_amendment_sha256=lt.sha(lt.AMEND2),
        supersedes_amendment_sha256=PREVIOUS_SHA, order=ORDER, order_sha256=lt.sha(lt.ROOT/ORDER),
        protocol={k:a[k] for k in PROTOCOL}, core_shas=cores, overrides=overrides,
        new_inputs={n:lt.sha(lt.ROOT/n) for n in sorted(NEW_INPUTS-set(inputs))},
        status='REGISTERED_BEFORE_CPU_GATES', prior_art=__doc__))
    with lt.AMEND3.with_suffix('.sha256').open('x') as f:
        f.write(lt.sha(lt.AMEND3)+'  registration_amendment.json\n')
    lt.verify()
    print(lt.AMEND3, lt.sha(lt.AMEND3))


if __name__ == '__main__':
    main()
