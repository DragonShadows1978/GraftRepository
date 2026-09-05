"""Write the WC1 final evidence report and a complete SHA-256 inventory."""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.grm_wc1_gpu import check_registration
from scripts.grm_wc1_sweep import OUT, read, record, write


def main():
    check_registration()
    gate = read(OUT / 'g2.json')
    table = read(OUT / 'sweep_table.json')
    coverage = read(OUT / 'pytest_coverage.json')
    summaries = read(OUT / 'pytest_summary.json')
    sources = read(OUT / 'source_verification.json')
    if gate['status'] != 'BLOCKED_EXECUTION' or any(r['correct'] is not None for r in table):
        raise RuntimeError('This blocked-run report template no longer matches the evidence')
    if not sources['all_registered_sources_unchanged']:
        raise RuntimeError('Read-only source verification failed')

    all_failures = []
    for name in ('cpu', 'units', 'native_legacy'):
        log = ROOT / f'logs/grm_wc1_pytest_{name}.log'
        all_failures += [line for line in log.read_text().splitlines() if line.startswith(('FAILED ', 'ERROR '))]
    guard_fixed = ('test_wipe_workspace_removes_existing_directory_and_contents',
                   'test_wipe_workspace_leaves_sibling_paths_untouched',
                   'test_wipe_then_recreate_produces_a_fresh_empty_directory')
    effective = [line for line in all_failures if not any(k in line for k in guard_fixed)]
    write(OUT / 'g1.json', {
        'status': 'RED', 'preexisting_set_unchanged': False,
        'coverage': coverage, 'shards': summaries,
        'effective_failure_entries': effective,
        'superseded_guard_failures': list(guard_fixed),
        'reason': 'No CUDA device is available in this seat; receipt-dependent tests reference missing gitignored artifacts. APAMQ summary failure is also present in RS4 logs. No production fix attempted.',
        'comparison_limit': 'RS4 logs themselves include OOM, pickle errors, timeouts, and rechecks. This run cannot certify an unchanged failure set.'})

    created = sorted(ROOT.glob('scripts/grm_wc1_*.py')) + [ROOT / 'tests/test_grm_wc1_sweep.py', ROOT / 'tests/grm_wc1_guard.py']
    key_paths = [OUT / name for name in (
        'registration.json', 'amendment_a1_harness.json', 'amendment_a2_execution_environment.json',
        'g1.json', 'g2.json', 'sweep_table.json', 'predictions.json', 'batteries.json',
        'source_verification.json', 'pytest_coverage.json', 'pytest_summary.json',
        'rs4_preexisting_failures.json', 'seat_identity.json',
        'width_96/runtime_frame.json', 'width_96/sup/correction_then_restatement_error.json',
        'native_build/libgrm_runtime.so')]
    records = [record(p) for p in created + key_paths]
    registration_raw = (OUT / 'registration.json').read_text()
    lines = [
        '# GRM-WC1 — wc1-astra final report', '',
        '**BLOCKED at G2 before any probe. G1 is RED; G3 was not run.**', '',
        'The leased width-96 attempt reached GPT-OSS model construction and failed with '
        '`cudaMalloc failed: no CUDA-capable device is detected`. '
        '`nvidia-smi` separately exited 9 because this execution environment could not communicate '
        'with the NVIDIA driver. These observations describe this seat; they do not establish '
        'host-wide driver state. No probe answer, split count, residency mean, wall-time mean, '
        'or width-curve measurement was obtained.', '',
        '1. pytest summary line(s) + pre-existing set status.', '',
        f'All {coverage["preexisting_files"]} pre-existing test files were submitted to pytest '
        f'across three foreground leased shards. Missing files: {coverage["missing_files"]}.', '',
    ]
    selected = ('grm_wc1_pytest_cpu.log', 'grm_wc1_pytest_units.log',
                'grm_wc1_pytest_native_legacy.log', 'grm_wc1_pytest_guard_recheck.log',
                'grm_wc1_pytest_new_final.log')
    for name in selected:
        summary = next(r for r in summaries if Path(r['record']['path']).name == name)
        lines.append(f'- `{name}`: ' + '; '.join(f'`{s}`' for s in summary['summary_lines']))
    lines += ['',
        '**Pre-existing set status: not unchanged; G1 RED.** Missing worktree copies of frozen '
        'DET1/RS3/SC1/SC2 receipts produce missing-file and receipt-validation failures. '
        'CUDA-dependent tests fail at device initialization. The APAMQ summary failure also '
        'appears in RS4. Three initial `wipe_workspace` failures came from the new guard resolving '
        'dir_fd-relative paths incorrectly; the guard was corrected and all 64 tests in its recheck '
        'passed. The final new WC1 suite has 11 passing tests. The native/legacy shard’s 32 errors '
        'all occurred during CUDA initialization at collection. Full failure entries are in `g1.json`; '
        'no pass total is inflated by adding overlapping rechecks.', '',
        '2. Registration block as written before gates + its sha256; any amendment file.', '',
        f'Registration SHA-256: `{record(OUT / "registration.json")["sha256"]}`. '
        'Created exclusively before G1 and verified unchanged after execution. '
        'The following is the original registration, including its pre-run RT1 reference gap:', '',
        '```json', registration_raw.rstrip(), '```', '',
        'Amendments (both cite the immutable registration SHA-256):', '',
    ]
    for path in sorted(OUT.glob('amendment_*.json')):
        lines.append(f'- `{path.name}` — `{record(path)["sha256"]}`')
    lines += ['',
        '3. G2 reproduction table; G3 sweep table; predictions hit/miss; the reading of the curve.', '',
        '| G2 battery at width 96 | Required | Measured | Status |',
        '|---|---:|---|---|']
    for row in gate['rows']:
        lines.append(f'| {row["battery"]} | {row["required"]} | No probes served | {row["status"]} |')
    lines += ['',
        'The supersession attempt waited on the existing flock, acquired the lease, and released '
        'it normally after 5.044 seconds. Its first model allocation failed. '
        'Correctness and semantic reproduction are unmeasured, not 0/9. '
        'The order requires G2 before every other width, so no G3 workload was started.', '',
        '| Width | Sup | Census | Long-horizon |', '|---:|---|---|---|']
    by_key = {(r['width'], r['battery']): r for r in table}
    for width in sorted({r['width'] for r in table}):
        lines.append('| ' + str(width) + ' | ' + ' | '.join(by_key[(width, b)]['status'] for b in ('sup', 'census', 'longhorizon')) + ' |')
    lines += ['',
        'Every correct-count, regression list, split-parent/child count, mean resident seats, '
        'mean wall ms, and mounted-mass cell is explicitly null in `sweep_table.json`. '
        'No timing value from failed model loading is substituted for per-turn timing.', '',
        '| Registered prediction | Hit/miss |', '|---|---|',
        '| 128 and 192 match or beat 96 across the three batteries | NOT TESTED |',
        '| 64 loses at least two sup probes | NOT TESTED |',
        '| 256 increases wall ms and degrades a census probe | NOT TESTED |',
        '| Long-horizon 4/4 at every width >= 96 | NOT TESTED |', '',
        '**Reading of the curve:** No curve was measured because width 96 could not initialize '
        'the model. This run provides no evidence for an optimum, a 64-seat splitting penalty, '
        'or a gradual versus cliff-like YaRN failure. The existing RS3 Part 4 receipt establishes '
        '8/9 sup, 9/10 census, and 4/4 long-horizon for that earlier run; it cannot be relabeled '
        'as this run or as RT1’s requested 9/9 baseline. The four RT1 rule-on receipt files named '
        'in `scripts/grm_rt1_g2_table.py` and the main `artifacts/grm_rt1/` directory are absent. '
        'CUDA access and those reference receipts remain necessary for completing the experiment.', '',
        '4. Files created; artifact paths with sha256.', '',
        'New scripts implement immutable registration, the baseline-first plan, guarded leased '
        'execution, constructor-time width substitution, imported fixture/shard execution, '
        'split/residency accounting, semantic comparison, paired regressions, and evidence '
        'reporting. New tests exercise the pure plan, table, predictions, regression and guard logic. '
        'The native library was built in `artifacts/grm_wc1/native_build` from worktree C++ sources.', '',
        '| File | SHA-256 |', '|---|---|']
    for r in records:
        lines.append(f'| `{Path(r["path"]).relative_to(ROOT)}` | `{r["sha256"]}` |')
    lines += ['',
        'The complete artifact, build-output, script, test and task-log inventory is '
        '`sha256_manifest.json`. Its own digest is in `sha256_manifest.sha256`. '
        'The manifest includes this report and excludes itself and its digest file. '
        'The live dispatcher log is excluded because it continues to grow; its model/effort '
        'header is preserved in the hashed `seat_identity.json` artifact.', '',
        '5. Ambiguities, deviations, residual risks, anything RED; process-safety acknowledgement; seat model and effort used.', '',
        '- G1 RED; G2 BLOCKED_EXECUTION; G3 NOT_RUN. No probe or empirical width comparison was completed.',
        '- RT1 references are missing; the requested 9/9 sup target remains unchanged. No reference was fabricated.',
        '- Dispatch named fork 29c8ef8 while the order template named 29d860b. The dispatched worktree was used; no git command was run.',
        '- The loader hard-codes width 96. A measurement-only constructor wrapper changes that keyword before construction; all other loader arguments and the existing deposit sequence are delegated unchanged. Its live GPU behavior remains unverified.',
        '- Native output was placed under the authorized WC1 artifact directory. The legacy native tests read a hard-coded main-checkout C++ source; its SHA-256 was verified identical to the worktree source, and test binaries went to temporary paths.',
        '- The optional mass observer was not attached; mounted mass remains null. There was no observer-induced gate or timing claim.',
        '- Safety constants and seeds were inherited. “No numeric constants beyond the grid” was interpreted as no new experimental tuning constants, not removal of the order’s gate counts or timing bounds.',
        '- No production files were modified. All registered read-only source hashes still match. Writes were confined to new WC1 scripts/tests, WC1 artifacts, task logs, and ordinary temporary test paths.',
        '- Process safety: no git, no subagents, no background waiter, no other process inspected or signaled, no lock clearing. All leased calls were foreground with explicit timeout below ten minutes and the inherited 30-second gap. The operator’s right of way was respected through gpu_lease.',
        '- Seat model: **gpt-6-astra**. Reasoning effort: **xhigh**. Verified from the dispatch log header.', '',
    ]
    (OUT / 'REPORT.md').write_text('\n'.join(lines))
    files = [p for p in OUT.rglob('*') if p.is_file()
             and p.name not in ('sha256_manifest.json', 'sha256_manifest.sha256')]
    files += created + list((ROOT / 'logs').glob('grm_wc1_*.log'))
    files = sorted(set(files))
    manifest = write(OUT / 'sha256_manifest.json', {'files': [record(p) for p in files]})
    (OUT / 'sha256_manifest.sha256').write_text(manifest['sha256'] + '  sha256_manifest.json\n')
    print(json.dumps({'report': record(OUT / 'REPORT.md'), 'manifest': manifest}, indent=2))


if __name__ == '__main__':
    main()
