#!/usr/bin/env python3
"""GRM-D1 follow-up: LT1.1 amendment 1 — core rebind + the A / A+ arms.

The base LT1.1 registration (sha `02b44d02…`) is frozen: it was written before
A1 merged, truthfully recorded `GRM_ALIAS_FOLD_MERGE` as absent-at-that-time,
and pinned no core SHAs at all. This amendment is a SEPARATE, sha-chained
document that does exactly two things:

  1. REBIND the core inputs to this tree. The LT1 run's binding
     (`artifacts/grm_lt1/amendment3/registration_amendment_r3.json`) pins 26
     core files; five have moved since. Each is recorded before -> after with
     an attribution, because they did not all move for the same reason:
       * `graft_repository.py`, `grm_runtime.py` — A1 (alias fold hooks);
       * `graft_arena.py`, `grm_admission.py`, `grm_text_norm.py` — moved
         BEFORE A1, on `grm-merge`, and are not A1's doing.
     Saying "A1 changed two files" is true and also insufficient: five files
     differ from the run whose numbers LT1.1 is compared against, and a
     registration that pins only two would silently accept the other three.

  2. REGISTER TWO ARM-A VARIANTS off the same frozen conversation and the same
     26-cell schedule, separately resumable so the lead can run either alone:
       * `A`  — alias flag OFF, exactly as the base registration reads;
       * `A+` — `GRM_ALIAS_FOLD_MERGE=1` pinned AFTER `environment(flags)`
                (the R1 `pin_rule` idiom) and restored afterwards.

Nothing in `core/` is touched. The base registration is not rewritten.

Prior art:
  * The SHA-chained amendment shape (parent sha, before/after per input,
    `previous_amendment_sha256`) is LT1's own amendment 2/3/4 contract
    (`scripts/grm_lt1.py:verify`, GRM contributors 2026) and C7 r3's
    `prepare_fixture`. Taken unchanged.
  * The pin-after-strip-then-read-back contract is R1's `pin_rule`
    (`scripts/grm_r1_replay.py:242`, GRM contributors 2026). Taken: the rule
    that an arm variable is pinned after `environment(flags)` clears every
    ambient `GRM_*` and is then asserted, so an arm cannot silently run the
    default. Mine: applying it to a second variable and requiring restoration.
  * The two-arm one-state contrast (`off`/`on` over one recorded checkpoint)
    is R1's campaign shape, reused as A/A+.
  * Mine: the drift attribution split (A1 vs pre-A1) and the per-arm
    prediction record.
  * No prior art known to me for this exact composition.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / 'artifacts/grm_d1/lt1_1'
BASE_REGISTRATION = OUT / 'registration.json'
BASE_SHA = '02b44d02e3fbb82ca9809c4a5e08597cf2d195e5a6fe4df8e4e81f3f760d80e8'
ORDER = ROOT / 'orders/GRM_D1_LT1_RESIDUAL_DIAGNOSIS.md'
LT1_BINDING = ROOT / 'artifacts/grm_lt1/amendment3/registration_amendment_r3.json'

ALIAS_FLAG = 'GRM_ALIAS_FOLD_MERGE'
RULE_ENV = 'GRM_ADMISSION_RULE'

#: Per-arm GPU ceiling. Two arms of the base 1.70 GPU-h budget.
BUDGET_GPU_SECONDS_PER_ARM = 6120

#: Which merge each drifted core file came from. A1 touches exactly the two
#: files that reference `alias_fold`; the rest moved earlier on grm-merge.
A1_FILES = ('core/graft_repository.py', 'core/grm_runtime.py')

#: The follow-up instruction this amendment answers, recorded verbatim so the
#: document is self-describing even though the follow-up arrived as a message
#: rather than as a file in `orders/`.
FOLLOW_UP = (
    'Lead follow-up to GRM-D1 (2026-09-10): flip the absent-flag test to a '
    'present-flag test; LT1.1 amendment 1 rebinding core input shas to this '
    'tree and registering two arm-A variants (A: alias flag OFF; A+: '
    'GRM_ALIAS_FOLD_MERGE=1 pinned after environment(flags), R1 pin idiom, '
    'restored after), budget 2 x 1.70 GPU-h, separately resumable; the A+ CPU '
    'alias check; refreshed lead_commands.txt with both arms.')


def sha_path(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(obj):
    return (json.dumps(obj, sort_keys=True, indent=2) + '\n').encode()


def core_rebind():
    """before -> after for every core input that moved since the LT1 run."""
    pinned = {name: change['after_sha256']
              for name, change in json.loads(
                  LT1_BINDING.read_text())['core_shas'].items()}
    rebind, unchanged = {}, []
    for name, before in sorted(pinned.items()):
        path = ROOT / name
        if not path.exists():
            raise ValueError('LT11_A1_CORE_INPUT_MISSING: ' + name)
        after = sha_path(path)
        if after == before:
            unchanged.append(name)
            continue
        rebind[name] = dict(
            before_sha256=before, after_sha256=after,
            attribution=('A1 (alias fold-merge hooks)' if name in A1_FILES
                         else 'pre-A1 drift on grm-merge, not attributable '
                              'to A1'))
    # A1 also ADDS a file the LT1 binding never knew about.
    new = {}
    alias_module = ROOT / 'core/grm_alias_fold.py'
    if alias_module.exists():
        new['core/grm_alias_fold.py'] = dict(
            sha256=sha_path(alias_module),
            attribution='A1 (new module; did not exist at the LT1 run)')
    return rebind, unchanged, new


def arms():
    """A and A+ — identical but for the one pinned variable."""
    base = json.loads(BASE_REGISTRATION.read_text())
    flags = base['arms']['A']['flags']
    common = dict(
        flags=flags,
        admission_rule='margin_first',
        cells=base['cells'],
        fixture=base['fixture'],
        resumable=True,
        budget_gpu_seconds=BUDGET_GPU_SECONDS_PER_ARM,
        budget_gpu_hours=round(BUDGET_GPU_SECONDS_PER_ARM / 3600.0, 2),
    )
    return {
        'A': dict(common,
                  arm='A',
                  out_dir='artifacts/grm_d1/lt1_1/run_A',
                  alias_fold_merge=False,
                  env_after_environment_flags={RULE_ENV: 'margin_first'},
                  pin_note='No alias pin. `environment(flags)` strips every '
                           'ambient GRM_*, then only the admission rule is '
                           'pinned, so the alias flag is absent and '
                           '`alias_fold_enabled()` resolves OFF.',
                  predictions=dict(
                      corrections='5/5',
                      aliases='0/5 wrong rows fixed (recall_7_* stay wrong)',
                      fresh='6/7',
                      recap='unpredicted memory-side; oracle 5/5 gated on CPU')),
        'A+': dict(common,
                   arm='A+',
                   out_dir='artifacts/grm_d1/lt1_1/run_Aplus',
                   alias_fold_merge=True,
                   env_after_environment_flags={RULE_ENV: 'margin_first',
                                                ALIAS_FLAG: '1'},
                   pin_note='R1 `pin_rule` idiom (scripts/grm_r1_replay.py:242): '
                            'pin AFTER `environment(flags)` has stripped every '
                            'ambient GRM_*, READ BACK through '
                            '`core.grm_alias_fold.alias_fold_enabled()` and '
                            'assert, then RESTORE the prior environment when '
                            'the arm finishes. The flag is the ONLY difference '
                            'between A and A+.',
                   predictions=dict(
                       corrections='5/5',
                       aliases='4/5 of the wrong rows fixed; the lead proposed '
                               '>= 3/5 and that bar is also met',
                       fresh='6/7',
                       recap='>= 3/5 memory-side',
                       prediction_note=(
                           'The lead proposed aliases >= 3/5. I register 4/5 '
                           'and state why: the D1 CPU check '
                           '(artifacts/grm_d1/alias_cpu_aplus.json) shows the '
                           'fold firing, the digest naming BOTH names, being '
                           'ADMITTED and MOUNTED, and serving the right value '
                           'for 10/10 alias probes at every distance, with a '
                           '78-token digest inside the 96 width. The one '
                           'reservation that keeps me off 5/5 is that the CPU '
                           'reader is a regex double: the GPU reader must '
                           'still choose the digest value over the surrounding '
                           'prose, and LT1 already showed one long-distance '
                           'ladder drop (recall_3_150) that no lineage change '
                           'addresses. 4/5 prices exactly one such drop.'))),
    }


def amendment():
    if sha_path(BASE_REGISTRATION) != BASE_SHA:
        raise ValueError('LT11_BASE_REGISTRATION_SHA_MISMATCH')
    rebind, unchanged, new = core_rebind()
    base = json.loads(BASE_REGISTRATION.read_text())
    return dict(
        schema='grm.lt1_1.amendment1.v1',
        amendment=1,
        registration='artifacts/grm_d1/lt1_1/registration.json',
        registration_sha256=BASE_SHA,
        order='orders/GRM_D1_LT1_RESIDUAL_DIAGNOSIS.md',
        order_sha256=sha_path(ORDER),
        follow_up_instruction=FOLLOW_UP,
        fixture=base['fixture'],
        purpose=('Rebind the LT1.1 core inputs to this tree after A1/P1/R1 '
                 'merged, and register two separately-resumable arm-A '
                 'variants (alias flag OFF and ON).'),
        core_rebind=dict(
            source_binding='artifacts/grm_lt1/amendment3/'
                           'registration_amendment_r3.json',
            source_binding_sha256=sha_path(LT1_BINDING),
            changed=rebind,
            unchanged_count=len(unchanged),
            new_inputs=new,
            attribution_note=(
                'A1 touches exactly the two files that reference '
                '`alias_fold` (%s) and adds `core/grm_alias_fold.py`. The '
                'other changed files moved on grm-merge BEFORE A1 and are '
                'rebound here for the same reason: LT1.1 is compared against '
                'the LT1 run, and every core input that differs from that run '
                'must be named, whatever merged it.'
                % ', '.join(A1_FILES))),
        arms=arms(),
        budget_gpu_seconds_per_arm=BUDGET_GPU_SECONDS_PER_ARM,
        budget_gpu_seconds_total=2 * BUDGET_GPU_SECONDS_PER_ARM,
        budget_gpu_hours_total=round(2 * BUDGET_GPU_SECONDS_PER_ARM / 3600.0, 2),
        lead_choice=('Both arms are independently resumable. The lead may run '
                     'A+ alone (1.70 GPU-h) and read A off the existing LT1 '
                     'arm-A receipts, or run both (3.40 GPU-h) for a '
                     'same-tree contrast. Running A+ alone is the cheaper '
                     'test of A1; running both is the only way to attribute a '
                     'change to the flag rather than to the core rebind.'),
        cpu_evidence=dict(
            path='artifacts/grm_d1/alias_cpu_aplus.json',
            claim='On the CPU fake session with the flag pinned, the alias '
                  'fold fires for both LT1 alias entities, the digest carries '
                  'both names under the FIX-8 projection, is admitted by '
                  'routing, is mounted, and serves the correct value.',
            evidence_class='CPU fake session (lineage/routing/admission only; '
                           'NOT a language-model quality measurement)'),
        named_flags=[dict(
            name=ALIAS_FLAG,
            present_on_tree=True,
            reader='core/grm_alias_fold.py:alias_fold_enabled '
                   '(ENV_NAME = %r); hooks in core/graft_repository.py and '
                   'core/grm_runtime.py' % ALIAS_FLAG,
            default='OFF; an unknown token fails CLOSED to OFF',
            supersedes=('the base registration `present_on_tree: false`, '
                        'which was true when that document was written and '
                        'became false when A1 merged'),
            pinned_in_arm='A+ only')],
        process_safety=dict(
            gpu='single lease via flock --wait; operator has absolute right '
                'of way; no process this run did not start is ever signalled',
            arms='A and A+ never run concurrently on the same GPU',
            git='lead commits; the seat never runs git'),
        prior_art=[
            'SHA-chained amendment contract: scripts/grm_lt1.py:verify and C7 '
            'r3 prepare_fixture (GRM contributors, 2026)',
            'Pin-after-strip-then-read-back: R1 pin_rule, '
            'scripts/grm_r1_replay.py:242 (GRM contributors, 2026)',
            'Two-arm one-state contrast: R1 off/on campaign shape',
            'Alias fold-merge mechanism under test: core/grm_alias_fold.py '
            '(A1, GRM contributors 2026); its own prior art (RAPTOR 2401.18059; '
            'Press et al. 2210.03350; Gupta & Mumick 1995) is annotated there '
            'and is not re-derived here',
            'Drift attribution split and per-arm prediction record: no prior '
            'art known to me',
        ],
    )


def lead_commands(amend_sha):
    a = amendment()
    lines = [
        '# LT1.1 — arm A and arm A+ (amendment 1)',
        '# base registration sha256: %s' % BASE_SHA,
        '# amendment 1   sha256: %s' % amend_sha,
        '# fixture       sha256: %s' % a['fixture']['sha256'],
        '#',
        '# %d cells per arm. Budget %d s (%.2f GPU-h) PER ARM; %.2f GPU-h for both.'
        % (len(a['arms']['A']['cells']), BUDGET_GPU_SECONDS_PER_ARM,
           BUDGET_GPU_SECONDS_PER_ARM / 3600.0,
           a['budget_gpu_hours_total']),
        '# The arms are separately resumable: run either alone.',
        '#',
        '# CPU gates first (no GPU, no lease):',
        'cd %s' % ROOT,
        'python3 -m pytest -q tests/test_grm_d1*.py tests/test_grm_a1_alias_fold.py \\',
        '  tests/test_grm_scout_fix4.py tests/test_grm_admission.py',
        'python3 scripts/grm_d1_recap.py          # oracle 5/5 PASS',
        'python3 scripts/grm_d1_alias_cpu.py      # A+ alias fold/admit/mount',
        '',
        '# ---------------- arm A (alias fold OFF, as base-registered) -------',
        '# environment(flags) strips every ambient GRM_*; pin the rule only.',
        'env -u %s \\' % ALIAS_FLAG,
        '  %s=margin_first GRM_PROFILE=eb1_c2 \\' % RULE_ENV,
        '  flock --wait 7200 /var/lock/grm_gpu.lock \\',
        '  python3 scripts/grm_lt1.py --run --arm A \\',
        '    --registration artifacts/grm_d1/lt1_1/registration.json \\',
        '    --amendment artifacts/grm_d1/lt1_1/amendment1.json \\',
        '    --fixture artifacts/grm_d1/lt1_1/dialogue.json \\',
        '    --out %s' % a['arms']['A']['out_dir'],
        '',
        '# ---------------- arm A+ (GRM_ALIAS_FOLD_MERGE=1 pinned) -----------',
        '# R1 pin idiom: the flag is set AFTER the ambient strip, read back by',
        '# core.grm_alias_fold.alias_fold_enabled(), and restored on exit.',
        '# `env` scopes the pin to this command; the parent shell keeps its',
        '# environment, which is the restoration.',
        'env %s=1 \\' % ALIAS_FLAG,
        '  %s=margin_first GRM_PROFILE=eb1_c2 \\' % RULE_ENV,
        '  flock --wait 7200 /var/lock/grm_gpu.lock \\',
        '  python3 scripts/grm_lt1.py --run --arm A \\',
        '    --registration artifacts/grm_d1/lt1_1/registration.json \\',
        '    --amendment artifacts/grm_d1/lt1_1/amendment1.json \\',
        '    --fixture artifacts/grm_d1/lt1_1/dialogue.json \\',
        '    --out %s' % a['arms']['A+']['out_dir'],
        '',
        '# Readback the harness must print for A+ (and must NOT for A):',
        '#   alias_fold_enabled() -> True',
        '#   admission_rule()     -> margin_first',
        '',
        '# Registered predictions (before any run):',
    ]
    for name in ('A', 'A+'):
        arm = a['arms'][name]
        lines.append('#   %-3s %s' % (name, json.dumps(arm['predictions'],
                                                       sort_keys=True)))
    return '\n'.join(lines) + '\n'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    payload = canonical(amendment())
    (OUT / 'amendment1.json').write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    (OUT / 'amendment1.sha256').write_text('%s  amendment1.json\n' % digest)
    (OUT / 'lead_commands.txt').write_text(lead_commands(digest))

    a = amendment()
    print('amendment1 sha256 %s' % digest)
    print('parent registration %s' % BASE_SHA)
    print('core rebind: %d changed, %d unchanged, %d new'
          % (len(a['core_rebind']['changed']),
             a['core_rebind']['unchanged_count'],
             len(a['core_rebind']['new_inputs'])))
    for name, change in sorted(a['core_rebind']['changed'].items()):
        print('  %-32s %s -> %s   [%s]'
              % (name, change['before_sha256'][:12],
                 change['after_sha256'][:12], change['attribution']))
    print('arms: %s   %.2f GPU-h each, %.2f total'
          % (', '.join(sorted(a['arms'])),
             a['arms']['A']['budget_gpu_hours'], a['budget_gpu_hours_total']))


if __name__ == '__main__':
    main()
