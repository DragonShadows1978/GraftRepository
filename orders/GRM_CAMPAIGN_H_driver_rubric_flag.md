# CAMPAIGN-H (micro): --rubric flag on the g1g2 driver

YOUR WRITABLE TARGET: /mnt/ForgeRealm/GraftRepository/tests/test_grm_importance_g1g2.py ONLY.
No git, no subagents, no GPU. RED honesty.

Add --rubric {v1,v2} (default v1) to the CLI; pass rubric=<value> into
the GraftRepository construction in _run_gpu_driver (the class already
accepts it — see core/graft_repository.py rubric= parameter, CAMPAIGN-F).
Record the rubric in each convo artifact's top level as "rubric". Keep
artifact schema otherwise unchanged. One CPU test: flag parse + rubric
lands in artifact dict (mock the GPU path as existing tests do).

Verify: py_compile + python3 -m pytest tests/test_grm_importance_g1g2.py -q (paste line).
Done message: diff summary, pytest line, "no git, no GPU, no files outside grant".
