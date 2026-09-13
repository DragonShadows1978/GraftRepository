# GRM-XM1 X2 immutable execution plan — 2026-09-12

Authority: lead order in this session, Qwen3.5-9B fresh-fact diagnosis.
Seat requested: gpt-6-astra, high. Writable worktree grm-xm2 only.
Read HOUSE_RULES first. No git, subagents, GPU, background waits, process
signals or edits to canonical/grm-xm1. Calls <10 minutes. CPU scratch under
artifacts/grm_xm2/tmp; remove owned scratch. Repo-relative source pins.

1. Diagnose copied ten Qwen receipts before remeasurement: layer profiles,
   mounted source identity, capture and seat geometry, tokenizer/scaffolds.
2. Add GRM_QWEN35_FINAL_CHANNEL, default ON for XM1 Qwen cells; OFF
   delegates unchanged legacy function. Use cached Qwen chat template's
   enable_thinking=False prefix; select final-answer input-query tokens only.
   Keep raw generation/observations. CPU replay recorded trace and adversarial
   boundary cases; demonstrate OFF exact serialized result equality.
3. Freeze a separate registration plus implementation hashes before gates.
   Five C3l/C5 pairs; praxis/solace 2x2 off/live capture and off/on seat,
   reusing live/on C3l baseline: 16 unique cells, <=1800 s total GPU lease
   reservations. Register only; worker leases itself, no outer flock.
4. Dry-run commands with source/barrier/budget checks and no GPU import.
   Author CPU tests; no blind verification authorized. Preserve immutable
   XM1 registration/amendment and copied GPU evidence.
5. Append ledger and write report (claim classes, prior art, RED). Final
   command verbatim: python3 -m pytest -q --basetemp artifacts/grm_xm2/tmp tests/test_grm_xm1*.py tests/test_grm_xm2*.py
   A foreground parent captures final stdout and cleans owned scratch after
   child exits; record result without a subsequent shell/test command.

Prior art: Qwen Team (2026) Qwen3.5-9B cached chat_template.jinja, revision
c202236235762e1c871ad0ccb60c8ee5ba337b9a. Upstream unverified — lead to check:
Qwen3.5-9B enable_thinking chat_template. Taken: empty think block for direct
final output. Ours: final input-query row selection and replay receipts.
GRM contributors (2026), RS3 capture/live seating, RS4 mean_over_positions,
XM1 immutable receipts; reused, no novel attention algorithm.
