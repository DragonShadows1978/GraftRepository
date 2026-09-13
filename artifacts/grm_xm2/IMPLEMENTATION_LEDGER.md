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
