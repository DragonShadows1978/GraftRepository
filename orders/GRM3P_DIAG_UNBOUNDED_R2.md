# ORDER GRM3P-DIAG r2 — resume the unbounded-recall mechanism hunt

Your order is orders/GRM3P_DIAG_UNBOUNDED.md, unchanged, plus the
pre-registration your predecessor already froze in
docs/GRM3P_DIAG_UNBOUNDED_REPORT.md (H-REHYDRATE + C2-TOPK2 / C2-LIVE1 /
C3-EARLY-REHYDRATE / CONFIRM decision rules). Do not re-register; append
results to that report.

ENVIRONMENT CORRECTION (lead-verified on host): the GPU is HEALTHY —
nvidia-smi clean, all /dev/nvidia* nodes present continuously since
Jul 19. Your predecessor's "missing device nodes" observation was its
own sandbox namespace, not the host. The GRM3P-P0/P1 and P2 seats hit
the identical "no CUDA-capable device" sandbox red and proceeded by
requesting escalated/authorized GPU execution for their GPU commands —
use that same path for every GPU run. Do NOT run sudo; do NOT touch
/dev or host device configuration; do NOT re-diagnose host GPU health.
If escalated GPU execution is unavailable to you, STOP and report that
verbatim.

Resume at ARM B (reproduce F-FULL single 7/9), then the registered
decision tree: C1 --topk 1, then C2/C3 arms per the frozen rules, then
CONFIRM. Same honesty rails, same Done requirements as the base order.
