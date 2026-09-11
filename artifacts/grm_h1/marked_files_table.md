| Test module | Tests | Node ids | Class | Registration | Why it is a receipt |
|---|---:|---:|---|---|---|
| `tests/test_grm_a1_gpu_contrast.py` **(new)** | 16 | 16 | sha-bound | `artifacts/grm_a1/gpu_contrast_registration.json + gpu_contrast_amendment_1..3.json` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| `tests/test_grm_c2_amendment.py` | 12 | 18 | artifact-bound | `artifacts/grm_c2/registration.json + orders/GRM_C2_AMENDMENT_1.md` | reads gitignored campaign artifacts under artifacts/ that this tree does not carry complete |
| `tests/test_grm_c2_budget_a5.py` | 15 | 28 | artifact-bound | `artifacts/grm_c2/a5/ + orders/GRM_C2_AMENDMENT_5.md` | reads gitignored campaign artifacts under artifacts/ that this tree does not carry complete |
| `tests/test_grm_c2_epoch3.py` | 7 | 12 | sha-bound | `artifacts/grm_c2/epochs/scout-fix-2/ + orders/GRM_SCOUT_FIX_2.md` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| `tests/test_grm_c2_profile.py` | 1 | 1 | artifact-bound | `artifacts/grm_c2/checkpoints/profile/` | reads gitignored campaign artifacts under artifacts/ that this tree does not carry complete |
| `tests/test_grm_c2_score_a4.py` | 8 | 8 | sha-bound | `artifacts/grm_c2/a4_staging/ + artifacts/grm_c2/epochs/scout-fix-2/` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| `tests/test_grm_c7.py` | 4 | 4 | sha-bound | `artifacts/grm_c7/registration.json` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| `tests/test_grm_c7_amendment7.py` | 3 | 3 | sha-bound | `artifacts/grm_c7/amendment_7.json + orders/GRM_C7_AMENDMENT_7.md` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| `tests/test_grm_c7_lead_1.py` | 1 | 1 | sha-bound | `artifacts/grm_c7/amendment_lead_1.json` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| `tests/test_grm_c7_lead_2_diagnosis.py` | 6 | 6 | artifact-bound | `artifacts/grm_c7/diagnosis_lead_2/registration.json` | reads gitignored campaign artifacts under artifacts/ that this tree does not carry complete |
| `tests/test_grm_c7_r2_launch.py` | 1 | 4 | sha-bound | `artifacts/grm_c7/r2/ (C7 r2 registration)` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| `tests/test_grm_c7_r2_registration.py` | 2 | 12 | sha-bound | `artifacts/grm_c7/r2/ (C7 r2 registration)` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| `tests/test_grm_c7_r3.py` | 5 | 5 | sha-bound | `artifacts/grm_c7/r3/registration.json` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| `tests/test_grm_d1_amendment3.py` **(new)** | 1 | 1 | sha-bound | `artifacts/grm_d1/lt1_1/registration.json (LT1.1 chain, core pins rebound at D1 amendment 4)` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| `tests/test_grm_lt1.py` | 3 | 4 | sha-bound | `artifacts/grm_lt1/registration.json` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| `tests/test_grm_lt1_1_preflight.py` **(new)** | 8 | 11 | sha-bound | `artifacts/grm_d1/lt1_1/registration.json + artifacts/grm_lt1/amendment4/resume_registration.json` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| `tests/test_grm_lt1_amendment1.py` | 3 | 4 | sha-bound | `artifacts/grm_lt1/amendment1/registration_amendment.json` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| `tests/test_grm_lt1_amendment3.py` | 3 | 3 | sha-bound | `artifacts/grm_lt1/amendment3/registration_amendment.json` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| `tests/test_grm_lt1_amendment4.py` | 9 | 14 | artifact-bound | `artifacts/grm_lt1/amendment2/run_margin_first/cells/*/checkpoint` | reads gitignored campaign artifacts under artifacts/ that this tree does not carry complete |
| `tests/test_grm_lt1_controller.py` | 2 | 2 | sha-bound | `artifacts/grm_lt1/registration.json` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| `tests/test_grm_lt1_fix6.py` | 3 | 4 | sha-bound | `artifacts/grm_lt1/ (FIX-6 amendment)` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| `tests/test_grm_r1_amendment1.py` **(re-marked)** | 11 | 11 | sha-bound | `artifacts/grm_r1/amendment_1.json` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| `tests/test_grm_r1_amendment2.py` **(new)** | 6 | 6 | sha-bound | `artifacts/grm_r1/amendment_2.json` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| `tests/test_grm_r1_amendment3.py` **(new)** | 10 | 10 | sha-bound | `artifacts/grm_r1/amendment_3.json` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| `tests/test_grm_r1_amendment4.py` **(new)** | 8 | 8 | sha-bound | `artifacts/grm_r1/amendment_4.json` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| `tests/test_grm_r1_cpu_gate.py` | 1 | 1 | sha-bound | `artifacts/grm_r1/registration.json` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| `tests/test_grm_r1_replay.py` | 15 | 15 | sha-bound | `artifacts/grm_r1/registration.json + orders/GRM_R1_MARGIN_FIRST_REPLAY.md` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| `tests/test_grm_scout_fix4_continuation.py` | 3 | 10 | sha-bound | `artifacts/grm_scout_fix4/registration.json` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| `tests/test_grm_scout_fix5_runner.py` | 4 | 16 | artifact-bound | `artifacts/grm_scout_fix5/registration.json + artifacts/grm_c7/r2/cells/*/checkpoint` | reads gitignored campaign artifacts under artifacts/ that this tree does not carry complete |
| `tests/test_grm_scout_fix8_replay.py` | 5 | 5 | sha-bound | `artifacts/grm_scout_fix8/registration.json` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| `tests/test_grm_scout_fix8_resume.py` | 2 | 3 | sha-bound | `artifacts/grm_scout_fix8/resume_amendment_1/registration.json` | asserts INPUT_SHA_MISMATCH-class binding against core/scripts shas frozen at registration |
| **31 modules** | **178** | **246** | | | |
