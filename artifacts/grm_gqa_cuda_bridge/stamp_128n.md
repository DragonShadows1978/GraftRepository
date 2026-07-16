# GRM GQA CUDA Bridge Smoke

Runtime bridge validation for `GRM_GQA_CUDA_ROUTE=1`.

## Result

- parity: true
- backend: `cuda`
- nodes: 128
- queries: 3
- topk: 5
- capture: `/mnt/ForgeRealm/qwen35_graft_translation_poc/captures`
- layer/key: 3 / `l3_k`
- key shape: [128, 2, 256, 256]
- query shape: [3, 8, 4, 256]
- first bridge wall ms: 1990.0110
- min bridge wall ms: 0.5626
- last direct CUDA route wall ms: 0.2529
- last direct CUDA device ms/query: 0.2253
