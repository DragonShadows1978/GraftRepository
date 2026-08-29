# ORDER MOE-E2-F1 — fix round: OLMoE GPU load path fails fused-expert conversion

## Grant

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/GraftRepository`. Same
read-onlys/forbiddens as `orders/MOE_E2_OLMOE_TESTBED.md`. Modify
`scripts/olmoe_e2_experiment.py` and regenerate the resume scripts.
CPU self-runs authorized (they work — bring-up proved it).

## The defect (lead log excerpt, deterministic 3/3 invocations)

GPU-path load (`logs/moe_e2_gpu_lead.log`) fails in transformers 5.12
weight auto-conversion for the fused OLMoE expert layout:

```
OlmoeForCausalLM LOAD REPORT ...
model.layers.11.mlp.experts.gate_up_proj | MISSING    |
model.layers.11.mlp.experts.gate_up_proj | CONVERSION |
RuntimeError: We encountered some issues during automatic conversion of
the weights. For details look at the `CONVERSION` entries...
(raised from modeling_utils._finalize_model_loading / loading_report.py:273)
```

Your CPU bring-up load of the SAME snapshot succeeded (bringup.json
GREEN), so the divergence is in the GPU/dispatch loading configuration
(device_map/max_memory path), not the checkpoint.

## Work

1. Reproduce the conversion failure CPU-side if possible by mimicking
   the GPU-path load config (device_map with max_memory caps etc.);
   identify the exact triggering difference.
2. Fix the harness load path so GPU runs load correctly and
   DETERMINISTICALLY with all weights present. Acceptable strategies
   (your judgment, receipts required): load-then-dispatch (full CPU
   load, then explicit module placement under an 11GiB VRAM budget);
   pinning a working conversion/loading configuration; or any
   equivalent that keeps bf16 and hook/injection fidelity. FORBIDDEN:
   quantized loads; editing site-packages; network.
3. Verification you run yourself (CPU): a load-report assertion that
   every checkpoint tensor is present and consumed (no MISSING /
   CONVERSION entries) wired into the harness as a hard gate before
   any forward; plus your existing synthetic contracts re-run.
4. Regenerate `GPU_E2_RESUME_COMMANDS.sh` (+ fallback script)
   unchanged in sequence, using the fixed load path.

## Done

Final message verbatim: root cause (file:line / config key), the fix,
proof the load-report gate passes on a CPU mimic (or honest statement
it only manifests on GPU + what the lead should expect), files
modified, anything you could not do.
