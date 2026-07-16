# GRM GQA CUDA Bridge Smoke

Runtime bridge validation for `GRM_GQA_CUDA_ROUTE=1`.

## Result

- parity: true
- backend: `cuda`
- nodes: 512
- queries: 3
- topk: 5
- capture: `/mnt/ForgeRealm/qwen35_graft_translation_poc/captures`
- layer/key: 3 / `l3_k`
- key shape: [512, 2, 256, 256]
- query shape: [3, 8, 4, 256]
- first bridge wall ms: 2111.0436
- min bridge wall ms: 38.9607
- last direct CUDA route wall ms: 0.7702
- last direct CUDA device ms/query: 0.7404
