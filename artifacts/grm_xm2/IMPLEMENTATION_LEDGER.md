# GRM-XM1 X2 append-only ledger

## 2026-09-12 — intake and pre-run diagnosis
Read HOUSE_RULES and local order. No GPU, git, subagents, process signals,
background work, or writes outside grm-xm2. Quick memory lookup found only
unrelated GRM scope taxonomy; no historical result used in diagnosis.
Read copied GPU JSON, immutable registration, RS4 per-node capture receipts,
RS3 order, and cached Qwen template. Snapshot legacy worker and chat template
under sources/ before edits. Original plan frozen before any gates.

Receipt evidence: Praxis C5 excludes its expected Quartz-8-Jade value;
Solace C3l mounts Tundra split-child text, excluding Raven-9-Ivory. All native
captures use width= max(native capture/feed ntok)+96; pin live and seat near
live on all five. Historical RS4 recaptures also all live/115. Differential
fresh-versus-superseded capture-pin explanation is unsupported here.
No rerun performed. Original thinking-window falsifier remains RED.

Prior art: Qwen Team (2026), cached Qwen3.5-9B chat template revision
c202236235762e1c871ad0ccb60c8ee5ba337b9a; enable_thinking=False empty block
reused. Upstream unverified — lead to check: Qwen3.5-9B chat_template
thinking mode. GRM contributors (2026) RS3 constant-delta seat/capture and
RS4 row partition/mean reused; ours is source audit and final-query window.

## Audit correction before gates
Initial audit counted normalized expected values case-sensitively against mixed-case source text. Preserved initial artifact; corrected casefold membership and actual-case value tokenization in diagnosis.json. No model run or decision threshold changed. Structural markers counted explicitly rather than relying on tokenizer all_special_ids.

## First CPU gate — RED, then exact-pin fixture restoration
Command: python3 -m pytest -q --basetemp artifacts/grm_xm2/tmp/preflight tests/test_grm_xm2_final.py tests/test_grm_xm1_amendment_1.py tests/test_grm_xm1_parity.py tests/test_grm_xm1_results.py
Result: 5 failed, 94 passed, 2 warnings in 8.24s.
All five failures: FileNotFoundError: [Errno 2] No such file or directory:
/mnt/ForgeRealm/wt/grm-xm2/artifacts/grm_det1/run_20260831T160525Z_2/runtime_frame_28b3196f8fb04a41.json
The fork omitted one original XM1 pin. Read canonical copy, verified SHA-256
against unchanged XM1 registration, copied exact bytes into this worktree.
No original pins changed; no writes to canonical or grm-xm1. Decode/window,
10 OFF byte-identity cells and four RS3 geometry CPU doubles already passed.

## CPU recheck and delivery preparation
Restored missing original pinned fixture only; no source changes after
registration. Affected recheck command: python3 -m pytest -q --basetemp
artifacts/grm_xm2/tmp/recheck tests/test_grm_xm2_final.py::test_registration_matrix_budget_and_dry_run tests/test_grm_xm1_amendment_1.py tests/test_grm_xm1_parity.py::test_registration_and_pins_are_immutable_and_relative tests/test_grm_xm1_parity.py::test_lead_commands_all_accept_appended_dry_run
Result: 9 passed in 17.27s. This includes all sixteen XM2 and sixty original
XM1 CLI dry-runs (no model or lease). First gate's passing tests prove all ten
OFF serialized CPU cells, all ten original thinking traces select no final,
synthetic-continuation/native CPU input-query alignment and four RS3 geometries.
No GPU gate or blind verification ran. Registration SHA:
7d537ab25ba485094b0235c9b1407d75b542dcbba21ebdd587a6534d16d73c3d.

Runtime risk found during report audit: original first cold Harbor C3l elapsed
129.70 s; other original cells 64–69 s. New per-cell reservation remains 110 s,
total 1760 s, immutable. Cold fit is unproven and may be RED; no automatic
extension/rerun authorized by registration. This is registered-only, not a
claim of completed GPU cells. No process was signaled/killed; no GPU queried.

Prior art in final report and code: Qwen Team 2026 native empty-thinking
prefix/delimiters, GRM 2026 RS3/RS4/XM1/LT1/CMC1 helpers reused; ours is final
input-query indexing and receipt source/token audit. External leads unverified.
Report written before final exact-command full suite. Only foreground parent
receipt writes and scratch cleanup will follow that validation subprocess.

## Final artifact audit before last pytest
Revalidated all XM2 and inherited source pins and original GPT-OSS barrier.
Sixteen cells, 1760 s reserved; no XM2 GPU run directory exists.
Final foreground parent uses Python subprocess (Python Software Foundation),
pathlib/hashlib and shutil for stdout receipt and owned scratch cleanup;
no novel orchestration algorithm. This housekeeping follows the final exact
pytest subprocess without launching another command.

Final pytest receipt: **99 passed, 2 warnings in 22.76s**; exit code 0.
Raw output: `artifacts/grm_xm2/pytest_LAST.log`; structured receipt:
`artifacts/grm_xm2/pytest_LAST.json`. Owned basetemp removed: True. No XM2 GPU run directory: True.


## 2026-09-13 — amendment 1 intake and source refutation (before gates)
Authority: immutable orders/GRM_XM1_X2_AMENDMENT_1.md, gpt-6-astra / high.
Read HOUSE_RULES; no git, subagents, GPU, signals, or background work.
Evidence-before receipt: amendment_1/evidence_before.json. Same gpu_loader
AST SHA in XM1 and XM2: 4ea4108468919e5594e4c846197b085accd67b851d45ed8ad534b4fc2561928f.
Qwen adapter and QuantLinear source bytes match the read-only XM1 worktree.
The current adapter uses whole-matrix INT4 QuantLinearTC for lm_head; the
host matrix is the embedding. No chunked/host lm_head exists in the cited
current adapter/drivers. Thus a different loader path is refuted by source;
the allocation failure root cause remains unresolved and not claimed fixed.
Verbatim failure preserved: QuantLinearTC INT4 init failed tensor=lm_head.weight shape=(248320,4096).
The old Harbor receipt records elapsed_s=115.31824087025598; the second
Praxis attempt has no cell receipt. Both 110 s reservations remain charged.

New code imports gpu_loader directly from XM1; does not duplicate or alter
its body or adapter. CPU regression invokes XM1 execute's default loader
and X2 run, replays the actual adapter load_weights/from_pretrained methods
with tiny host arrays and fake device/I/O primitives, and compares identity,
lm_head branch and config. This is author CPU evidence only, no GPU fit.

Historical cold Harbor C3l elapsed_s=129.69799864804372; nine other XM1
Qwen receipts range 64.00301110185683 to 68.54434893094003 s. Asked optional
budget preference while implementing independent work; no response before
registration. Preserve the authorized 1800 s budget: cold cap ceil(elapsed)=
130 s; warm cap ceil((68.54434893094003+20)/10)*10=90 s. New reservations
1480 s plus prior 220 s =1700 s. A later cold process is not guaranteed to
fit a warm cap; it must stop RED. No threshold applied retroactively: old
registration, receipts, and source snapshots retained, new amendment and
new attempt namespace only. No global 130 s cap falsely squeezed into 1800.

Prior art unchanged: Qwen Team (2026), cached chat template at revision
c202236235762e1c871ad0ccb60c8ee5ba337b9a; external lead unverified — lead to
check Qwen3.5-9B chat template thinking mode. GRM contributors (2026),
XM1/LT1 immutable overlays, CPU doubles, RS3/RS4 and CMC1 lease/accounting
reused. Python AST source replay/unittest.mock reused. Ours: shared-loader
proof and explicit receipt-derived time-cap policy; no new load algorithm.


### Preparation correction and procedural RED
The first create-only registration command failed AssertionError because the
lead concurrently added a third 110 s attempt (Meridian C5), increasing old
reservations to 330 s. No amendment file was created. My following preflight
command still ran: this violated registration-before-gates and is procedural
RED. Result: 4 failed, 1 passed, 2 errors in 0.53s. Exact root errors:
FileNotFoundError: artifacts/grm_xm2/registration_amendment_1.json; and
FileNotFoundError: artifacts/grm_xm2/tmp/preflight (parent tmp not created).
The lone passing test compared loader/source identity. No CUDA or lease ran.
No thresholds were successfully registered or subsequently relaxed. Future
registration and gate commands are separate tool calls with checked exits.

Concurrent lead receipts: Praxis C3l TRUNCATED_FINAL, elapsed_s=
86.97211571317166; Meridian C5 PASS, elapsed_s=76.55421134503558.
Evidence captured in amendment_1/evidence_concurrent.json; originals retained.
Read-only process listing showed no matching XM1/X2 worker; no process signal.
Reconcile the unchanged 1800 s budget before registration: Harbor C3l 130 s;
Meridian C5 80 s (ceil its newer 76.55421134503558 s to 10 s); other 14 cells
90 s. New reservations 1470 s + prior 330 s =1800 s. This is an explicit
pre-run allocation from historical receipts, not a future-fit guarantee.
The first 220 s accounting statement above is superseded by this live audit.


### Successful amendment registration and CPU preflight
Create-only registration succeeded separately, exit 0, SHA-256
c300374de576251c6f622ef2371f4ede0ae3fac58c4ea8465e26d8547d3fc5ff.
Original 7d537ab2 registration unchanged; amendment pins all changed code,
original snapshots, source comparisons, prior attempts and receipt evidence.
Targeted registered gate: 7 passed in 3.98s, exit 0. Exact command/stdout:
amendment_1/cpu_preflight.json and amendment_1/cpu_preflight.log.
Includes both worker entry points, actual adapter load-method CPU replay,
loader identity/config/INT4 head comparison, historical source identity,
immutable original pins, cap evidence, drift and mode rejection, all 16
XM2 CLI dry-runs. No GPU or blind validation. Implementation unchanged
since successful registration. Final report and updated synthesis written;
final exact full pytest command will be last, followed only by in-process
receipt/manifest updates and cleanup of owned artifacts/grm_xm2/tmp.


Final pytest receipt: **sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute**; exit code 0.
Raw output: artifacts/grm_xm2/amendment_1/pytest_LAST.log; structured receipt:
artifacts/grm_xm2/amendment_1/pytest_LAST.json. Owned basetemp removed: True.
No subsequent shell/test command. GPU failure remains not claimed fixed; procedural RED retained.
