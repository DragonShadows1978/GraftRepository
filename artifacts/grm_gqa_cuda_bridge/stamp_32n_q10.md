# GRM GQA CUDA Bridge Smoke

Runtime bridge validation for `GRM_GQA_CUDA_ROUTE=1`.

## Result

- parity: true
- backend: `cuda`
- nodes: 32
- queries: 10
- topk: 5
- capture: `/mnt/ForgeRealm/qwen35_graft_translation_poc/captures`
- layer/key: 3 / `l3_k`
- key shape: [32, 2, 256, 256]
- query shape: [10, 8, 4, 256]
- first bridge wall ms: 1746.4897
- min bridge wall ms: 0.3521
- last direct CUDA route wall ms: 0.1498
- last direct CUDA device ms/query: 0.1219
