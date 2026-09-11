# GRM-RD1 amendment 1 ledger

## 2026-09-09 — registration before gates

Lead order `orders/GRM_RD1_AMENDMENT_1.md` read together with original order, AGENTS.md and /mnt/Shared/HOUSE_RULES.md. Original r1 registration/script/artifacts and both order files remain immutable. New harness and tests only; no core edits.

Registration `artifacts/grm_rd1/amendment_1/amendment.json`, SHA-256 `72f7ba59d705e2107627321557ab047de762cb149d29143380bcacfcc3634593`. Command: `PYTHONDONTWRITEBYTECODE=1 python scripts/grm_rd1_replay.py register`, exit 0. Written before any amendment gate. 138 scored probe requests = (30 memory + 16 oracle) x A0/A2/A4. 48 checkpoint/arm batches, maximum 7 requests per batch. Projection = r1 11793.813152259705 / 520 = 22.680409907679 s/request; x138 = 3129.896567259705 s = 0.869415713128 GPU-h, FIT_ESTIMATE at 3600 s. A1 NOT_RUN: 1043.298855753 s; A3 NOT_RUN: 2086.597711506 s on the reduced cohort. These are planning estimates, not measured timings. Prefix generation and repeated model loads are charged inside the cap; no inference run in this seat.

Source inspection: actual checkpoint state is state.json. Preceding tree, worker digest, manifest and original C7 flags are retained. Prefix non-probe turns use production run_turn, original forced folds and pressure. Earlier probes are deferred/non-deposit/ephemeral; their durable turn_records are advanced without extra probe generation. The historical no-split/deferred/frame contract is checked. This skips transient routing, cache and pager history and therefore needs the lead's A0 numerical gate; it is not claimed bitwise equivalent on CPU evidence alone. No post-cell checkpoint is substituted. Each recorded mounted payload is hashed; treatment payloads must equal A0.

A0 compares raw production _attempt UTF-8 text to r2 served text and writes any difference before stopping. Historical ladder may normalize text or try several mounts; no normalization waiver is registered. All 46 A0 rows must pass before A2; A4 follows complete A2. Score rule: +8 exact of fixed 30, at most 2 wrong, complete arm and valid A0. CPU fake model derives answers from actual restored token payloads/live source, never expected-answer tape.

Registered gates and five mutation sites/0.80 threshold are in amendment.json. Tests authored before registration. No tests executed yet. Lead commands are executable and pinned before gates.

Prior art: local C7/FIX4/EB1/C2 and diagnostic doubles (GRM contributors, 2026), verified from source. Taken: immutable checkpoint/hash/state contracts, production _attempt and ephemeral lifecycle, original folds/pressure, frozen scorer, foreground leases, owner/reservation accounting, fake numerical seams. Ours: fixed-residency replay/control flow and contrast table. No prior art known to me for this exact composition. No external literature or novelty claim. Code-site annotations present.

Safety: no GPU, git, subagents, service changes or process kills. Tool-level yielding on a broad read-only rg /mnt/Shared search was unintentional; no background shell command was launched. Future searches are bounded to specific paths. No workspace history findings used from the quick memory search.

## 2026-09-09 — CPU baseline gate

`CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_grm_rd1_replay.py -k 'not registered_mutations' > artifacts/grm_rd1/amendment_1/cpu_baseline_1.log 2>&1` — exit 0, 13 passed, 1 deselected, 8.71 s. Two SWIG deprecation warnings preserved in the log. Evidence class: author CPU control-flow baseline. Fake checkpoint payloads loaded through real GraftRepository; production ArenaCache._attempt generated fixture answers from restored tokens/live sources. Prefix gate created and mounted a node absent at the preceding checkpoint. Historical 16 checkpoint trees validated. No GPT-OSS numerical result or blind audit claimed. Original broad read-only search finished, exit 0; no process remains from it.

## 2026-09-09 — mutations, dry-run and blocked launch

After the passing baseline: `CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -s tests/test_grm_rd1_replay.py -k registered_mutations > artifacts/grm_rd1/amendment_1/cpu_mutations_1.log 2>&1` — exit 0, 1 passed / 13 deselected, 5.76 s. All five registered in-memory mutants killed: erase_mount_guard, force_A0_equal, force_A4_low, erase_prompt_suffix_change, skip_prefix_advance. Kill rate 1.0 >= 0.80. Mutated function copies only; production and harness source never rewritten. Author mutation evidence, not independent verification.

`PYTHONDONTWRITEBYTECODE=1 python scripts/grm_rd1_replay.py dry-run --output artifacts/grm_rd1/amendment_1/dry_run.json` — exit 0, FIT_ESTIMATE / BLOCKED_NO_GPU_IN_SANDBOX, 138 requests / 48 batches / 3129.896567 s / cap 3600 s.

`PYTHONDONTWRITEBYTECODE=1 python scripts/grm_rd1_replay.py run-batch --batch A0_A-024-031 --output artifacts/grm_rd1/amendment_1/blocked_report.json` — expected exit 2; actual error `ValueError: BLOCKED_NO_GPU_IN_SANDBOX`. Gate refused before model loading, GPU lease or campaign directory creation. No GPU attempt started.

`PYTHONDONTWRITEBYTECODE=1 python scripts/grm_rd1_replay.py summary --output artifacts/grm_rd1/amendment_1/summary_not_run.json` — exit 0, NOT_MEASURED, 0/138 observed. All 40 arm/class/side table combinations retained with null numerical results. A1/A3 NOT_RUN. `bash -n artifacts/grm_rd1/amendment_1/lead_commands.txt` — exit 0. Script executable bit checked at final integrity. Source model directory exists; native library is present and SHA-pinned in the 211 amendment input hashes.

## 2026-09-09 — deliverable synthesis

Created CPU receipt, NOT_RUN table, amendment report and checksums. Numerical A0 reproduction, internal reasoning behavior, treatment quality and reader improvement remain RED / NOT_MEASURED; not claimed fixed. No independent blind audit. Exact-r2 numerical equality is mandatory and cannot be inferred from these CPU gates. The worker omits transient history of non-deposit probes and compares raw _attempt output with historical served output; both limitations are explicit, with no tolerance waiver. All prefix work and model loading are metered within foreground leases; the projected budget does not prove actual fit. A Python lease alarm cannot guarantee native-call hard preemption: any overrun is RED and charged without clamping.

Safety: all executed inference used fake CPU numerical boundaries with CUDA_VISIBLE_DEVICES empty. No GPU model loading/allocation, git, subagents, service mutation, background shell jobs, process kills or core edits. Foreground tool calls occasionally yielded to the interface; both yielded commands finished normally. The broad Shared search was a discovery inefficiency, not a background wait command. Agent is Codex/GPT-6 family; exact serving model identifier is unavailable in session metadata; requested effort high. Model under test is openai/gpt-oss-20b revision 6cee5e81ee83917806bbde320786a8fb61efebee, recorded r2 resident_packed_mxfp4/tensor_cuda GptOss20B_TC. A0/A2 low, A4 medium via live Harmony system text; sink/source K/V unchanged.
