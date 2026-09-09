"""LT1 amendment 3 source-state binding.

Prior art: GRM contributors, C7/LT1 SHA-chain amendments (2026), verified
in local grm_c7_common.py and grm_lt1.py. Reuse explicit scope, old-hash
chain checks, immutable files and checkpoint bindings. New: FIX4/5 source
pins and CPU receipt binding; no new retrieval algorithm.
"""
from scripts.grm_c7_common import read, sha

ORDER = 'orders/GRM_LT1_AMENDMENT_3.md'
# Lead-authorized source snapshot, read from the cherry-picked working files.
# These anchors reject a rehashed amendment that substitutes another source.
SOURCE_CHANGES = {
    'core/graft_arena.py': ('f871a9294c6fca8723122cbd992469538a250d17f23f6272bacdaf918c11ed7b', '83a2d4a2bc0ed2c8e0fc06f937f59d94f48ca5a70a65a96c33b1ba6eaa2e6aed'),
    'core/grm_admission.py': ('5ee1bd977eda936c85d5d2aef578fc43398e6510188901e98ea87b53d96d4a2d', 'd1afc26a68b4b1c8236fc01c7db62065ab49173a89d7f2853989eedfadb48fe0'),
    'core/grm_three_pass.py': ('2b68eb1f7ed22b7989a615578be6023a3363f541e6fe3a92cbacc4354048fb84', '312f4bc0e4922449b9231ea2d9c3e6fff3a116aef6d0a48712f530d9c419742f'),
    'scripts/grm_e2e_session.py': ('74434dbb750e8e7635afa2f55709ae0533a3562f5b63aa54fc1f81d664bfd450', 'f2e8f07027b735b9b0e6ee8749e1c25aa8e7883ca782fa03efb5395e34367506'),
}
HARNESS = {'scripts/grm_lt1.py', 'artifacts/grm_lt1/lead_commands.txt', 'tests/test_grm_lt1_fix6.py'}
NEW_INPUTS = {ORDER, 'scripts/grm_lt1_amendment3.py', 'scripts/grm_lt1_register_amendment3.py',
              'scripts/grm_lt1_amendment3_cpu.py', 'tests/test_grm_lt1_amendment3.py',
              'tests/test_grm_scout_fix4.py', 'tests/test_grm_scout_fix5.py',
              'artifacts/grm_lt1/amendment3/EXECUTION_REGISTRATION.md'}
NEW_INPUTS |= {'artifacts/grm_c7/r2/fix4_attempt_1/cells/A-109-116/folds/115-636a4f13086f4af6bdbd6dbf0b5d1614/start.json', 'artifacts/grm_c7/r2/fix4_attempt_1/cells/A-086-093/folds/091-b0a742d7fa0f498ea4ab9d653be59f4c/start.json', 'artifacts/grm_c7/r2/fix4_attempt_1/cells/A-048-055/folds/055-1b44742dab454a6685c75b4e4a614895/start.json', 'artifacts/grm_scout_fix4/nonrecency_before.json', 'artifacts/grm_c7/r2/fix4_attempt_1/cells/A-101-108/folds/107-037ea1021147420cae7505b99d7bc897/start.json', 'artifacts/grm_c7/r2/fix4_attempt_1/cells/A-109-116/folds/111-3ea748aa13dd40cfb399e9c3b3b7ce82/start.json', 'artifacts/grm_scout_fix5/registration.json', 'artifacts/grm_c7/r2/fix4_attempt_1/cells/A-094-100/folds/099-71a7a9d8b7324a65934f918d78828724/start.json', 'artifacts/grm_c7/r2/fix4_attempt_1/cells/A-094-100/folds/095-3df9be1aee50442c8a0fe437490940c1/start.json', 'artifacts/grm_c7/r2/fix4_attempt_1/cells/A-024-031/folds/026-681f4264c9fa4c2c9a7096d30106a8a0/start.json', 'artifacts/grm_scout_fix5/zero_fact_before.json', 'artifacts/grm_c7/r2/cells/A-001-008/folds/008-533d6c5a5b0c4e568b5d9f7410ce2587/start.json', 'artifacts/grm_lt1/amendment3/imported_cpu_inputs.json', 'artifacts/grm_c7/r2/cells/A-017-023/folds/017-ac4a09b2490441678f6a5e9289942d5d/start.json', 'artifacts/grm_c7/r2/fix4_attempt_1/cells/A-071-077/folds/074-542293a8731d432da4e42a6af5120f3c/start.json', 'artifacts/grm_c7/r2/fix4_attempt_1/cells/A-101-108/folds/103-2d04027c210443289c63a9655b90470e/start.json', 'artifacts/grm_c7/r2/fix4_attempt_1/cells/A-078-085/folds/085-d15b5a88d2d94e56a2915cde4a8b0dfa/start.json'}
FIRST_AMENDMENT = 'artifacts/grm_lt1/amendment3/registration_amendment.json'
FIRST_SHA = 'a904002e21b3af0a7ae674c2d78acc61060a2e3ae9aa9503c778809195cc7eb9'
NEW_INPUTS |= {FIRST_AMENDMENT, 'artifacts/grm_c7/fixture.json', 'artifacts/grm_lt1/amendment3/CORRECTION_REGISTRATION.md'}
PREVIOUS_REVISION = 'artifacts/grm_lt1/amendment3/registration_amendment_r2.json'
PREVIOUS_SHA = '34b0cabfa353e7c3192e2c7ec3a7d813f8da2bad5fa7dfa7d1de30c0c7936e50'
NEW_INPUTS |= {PREVIOUS_REVISION, 'artifacts/grm_lt1/amendment3/CORRECTION_REGISTRATION_R3.md', 'artifacts/grm_lt1/amendment3/r2/fix4_structural_difference.json'}
PROTOCOL = ('arms', 'admission_rule', 'budget_gpu_seconds', 'projection_gpu_seconds',
            'cells', 'fixture_manifest_sha256', 'acceptance', 'predictions',
            'run_namespace', 'free_space_minimum_bytes')


def apply(lt, inputs, previous):
    path = lt.AMEND3
    if not path.is_file() or not path.with_suffix('.sha256').is_file():
        raise ValueError('AMENDMENT3_REQUIRED')
    if sha(path) != path.with_suffix('.sha256').read_text().split()[0]:
        raise ValueError('AMENDMENT3_SHA_MISMATCH')
    a = read(path)
    if a.get('schema') != 'grm.lt1.amendment3.v1':
        raise ValueError('AMENDMENT3_SCHEMA_MISMATCH')
    if (a.get('registration_sha256') != sha(lt.REG)
            or a.get('previous_amendment_sha256') != sha(lt.AMEND2)
            or a.get('supersedes_amendment_sha256') != PREVIOUS_SHA
            or sha(lt.ROOT/PREVIOUS_REVISION) != PREVIOUS_SHA
            or sha(lt.ROOT/FIRST_AMENDMENT) != FIRST_SHA
            or a.get('order') != ORDER or a.get('order_sha256') != sha(lt.ROOT/ORDER)):
        raise ValueError('AMENDMENT3_CHAIN_MISMATCH')
    first = read(lt.ROOT/FIRST_AMENDMENT)
    if a.get('protocol') != {k: previous[k] for k in PROTOCOL}:
        raise ValueError('AMENDMENT3_PROTOCOL_MISMATCH')
    cores = {n for n in inputs if n.startswith('core/')}
    if set(a.get('core_shas', {})) != cores:
        raise ValueError('AMENDMENT3_CORE_SCOPE_MISMATCH')
    if a['core_shas'] != first['core_shas']:
        raise ValueError('AMENDMENT3_CORE_PIN_MISMATCH')
    for n, change in a['core_shas'].items():
        expected = SOURCE_CHANGES.get(n, (inputs[n], inputs[n]))
        if change != dict(before_sha256=inputs[n], after_sha256=expected[1]) or inputs[n] != expected[0]:
            raise ValueError('AMENDMENT3_CORE_PIN_MISMATCH: ' + n)
    if set(a.get('overrides', {})) != set(SOURCE_CHANGES) | HARNESS:
        raise ValueError('AMENDMENT3_OVERRIDE_SCOPE_MISMATCH')
    for n, c in a['overrides'].items():
        if c.get('before_sha256') != inputs[n]:
            raise ValueError('AMENDMENT3_BEFORE_MISMATCH: ' + n)
        if n in SOURCE_CHANGES:
            if c != dict(zip(('before_sha256', 'after_sha256'), SOURCE_CHANGES[n])):
                raise ValueError('AMENDMENT3_SOURCE_PIN_MISMATCH: ' + n)
        else:
            archive = 'artifacts/grm_lt1/amendment3/before/' + n
            if c.get('before_archive') != archive or sha(lt.ROOT/archive) != inputs[n]:
                raise ValueError('AMENDMENT3_ARCHIVE_MISMATCH: ' + n)
        inputs[n] = c['after_sha256']
    if set(a.get('new_inputs', {})) != NEW_INPUTS - set(inputs):
        raise ValueError('AMENDMENT3_NEW_INPUT_SCOPE_MISMATCH')
    inputs.update(a['new_inputs'])
    return a
